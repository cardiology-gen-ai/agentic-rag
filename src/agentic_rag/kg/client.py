"""Read-only Neo4j client used by the knowledge-graph retriever."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase, Query, READ_ACCESS


_FORBIDDEN_CYPHER_RE = re.compile(
    r"\b("
    r"CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|"
    r"FOREACH|CALL|LOAD\s+CSV|GRANT|DENY|REVOKE"
    r")\b",
    flags=re.IGNORECASE,
)

_STRING_LITERAL_RE = re.compile(
    r"""'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*" """.strip(),
    flags=re.DOTALL,
)

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*?$", flags=re.MULTILINE)


@dataclass(frozen=True)
class Neo4jKGConfig:
    """Connection settings for a Neo4j database."""

    uri: str
    username: str
    password: str
    database: str | None = None
    query_timeout_seconds: float = 30.0

    @classmethod
    def from_env(
        cls,
        env_path: str | Path | None = None,
    ) -> "Neo4jKGConfig":
        """Load Neo4j settings from environment variables."""

        if env_path is None:
            load_dotenv()
        else:
            load_dotenv(dotenv_path=Path(env_path), override=False)

        uri = (os.getenv("NEO4J_URI") or "").strip()
        username = (
            os.getenv("NEO4J_USERNAME")
            or os.getenv("NEO4J_USER")
            or ""
        ).strip()
        password = (os.getenv("NEO4J_PASSWORD") or "").strip()

        database_raw = os.getenv("NEO4J_DATABASE")
        database = (
            database_raw.strip()
            if database_raw and database_raw.strip()
            else None
        )

        timeout_raw = (
            os.getenv("NEO4J_QUERY_TIMEOUT_SECONDS")
            or "30"
        ).strip()

        missing = [
            name
            for name, value in (
                ("NEO4J_URI", uri),
                ("NEO4J_USERNAME", username),
                ("NEO4J_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required Neo4j environment variables: "
                + ", ".join(missing)
            )

        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise ValueError(
                "NEO4J_QUERY_TIMEOUT_SECONDS must be numeric"
            ) from exc

        if timeout <= 0:
            raise ValueError(
                "NEO4J_QUERY_TIMEOUT_SECONDS must be greater than zero"
            )

        return cls(
            uri=uri,
            username=username,
            password=password,
            database=database,
            query_timeout_seconds=timeout,
        )


def _mask_literals_and_comments(cypher: str) -> str:
    """Remove comments and string values before safety validation."""

    masked = _BLOCK_COMMENT_RE.sub(" ", cypher)
    masked = _LINE_COMMENT_RE.sub(" ", masked)
    masked = _STRING_LITERAL_RE.sub("''", masked)
    return masked


def validate_read_only_cypher(cypher: str) -> str:
    """Validate that a Cypher statement is suitable for read-only execution."""

    normalized = str(cypher).strip()
    if not normalized:
        raise ValueError("Cypher query must not be empty")

    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()

    if ";" in normalized:
        raise ValueError("Multiple Cypher statements are not allowed")

    masked = _mask_literals_and_comments(normalized)

    forbidden = _FORBIDDEN_CYPHER_RE.search(masked)
    if forbidden:
        raise ValueError(
            "Write or procedure operation is not allowed in KG retrieval: "
            f"{forbidden.group(0)}"
        )

    upper = masked.lstrip().upper()

    if upper.startswith("SHOW "):
        return normalized

    if not re.search(r"\bRETURN\b", masked, flags=re.IGNORECASE):
        raise ValueError(
            "Read-only retrieval queries must contain a RETURN clause"
        )

    return normalized


class Neo4jKGClient:
    """Small read-only wrapper around the official Neo4j driver."""

    def __init__(
        self,
        config: Neo4jKGConfig,
        *,
        verify_connectivity: bool = True,
    ) -> None:
        self.config = config
        self._driver: Driver = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )

        if verify_connectivity:
            self.verify_connectivity()

    @classmethod
    def from_env(
        cls,
        env_path: str | Path | None = None,
        *,
        verify_connectivity: bool = True,
    ) -> "Neo4jKGClient":
        config = Neo4jKGConfig.from_env(env_path=env_path)
        return cls(
            config=config,
            verify_connectivity=verify_connectivity,
        )

    def verify_connectivity(self) -> dict[str, str]:
        """Verify the Aura connection and return basic server information."""

        self._driver.verify_connectivity()
        server_info = self._driver.get_server_info()

        return {
            "address": str(server_info.address),
            "agent": str(server_info.agent),
            "database": self.config.database or "<home database>",
        }

    def run_read(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        """Execute one validated read-only Cypher query."""

        validated_cypher = validate_read_only_cypher(cypher)

        timeout = (
            self.config.query_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if timeout <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        query = Query(
            validated_cypher,
            timeout=timeout,
        )

        session_kwargs: dict[str, Any] = {
            "default_access_mode": READ_ACCESS,
        }
        if self.config.database:
            session_kwargs["database"] = self.config.database

        with self._driver.session(**session_kwargs) as session:
            result = session.run(
                query,
                dict(parameters or {}),
            )
            return [record.data() for record in result]

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jKGClient":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

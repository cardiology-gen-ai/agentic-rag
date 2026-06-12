"""Structured LLM router for parameterized KG retrieval."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agentic_rag.agent import output
from agentic_rag.agent.services.llm_service import LLMService


KG_ROUTER_NODE_NAME = "kg_retrieval_router"
KG_MENTIONS_ROUTER_NODE_NAME = "kg_mentions_router"


class KGStructuredRouter:
    """
    Convert a natural-language question into a validated KG retrieval plan.

    The router uses the repository's existing LLMService, prompt registry,
    structured-output support, and retry behavior. It does not execute Cypher
    and does not call the KG retrieval tools directly.
    """

    def __init__(
        self,
        llm_service: LLMService,
        router_config: RunnableConfig | None = None,
        *,
        node_name: str = KG_ROUTER_NODE_NAME,
    ) -> None:
        self.llm_service = llm_service
        self.router_config = router_config
        self.node_name = self._validate_node_name(node_name)

        try:
            self._node = self.llm_service.build_node(
                self.node_name,
                structured_output=True,
                output_schema=output.KGRetrievalPlan,
            )
        except StopIteration as exc:
            raise RuntimeError(
                f"LLM node {self.node_name!r} is not registered in the "
                "configured nodes YAML"
            ) from exc

    def route(
        self,
        question: str,
        *,
        config: RunnableConfig | None = None,
    ) -> output.KGRetrievalPlan:
        """Return a validated retrieval plan for one natural-language question."""

        normalized_question = self._validate_question(question)
        invocation_config = (
            config
            if config is not None
            else self.router_config
        )

        result = self._node.invoke(
            {"question": normalized_question},
            config=invocation_config,
            with_retry=True,
        )

        if isinstance(result, output.KGRetrievalPlan):
            return result

        try:
            return output.KGRetrievalPlan.model_validate(result)
        except Exception as exc:
            raise RuntimeError(
                "The KG retrieval router returned an invalid structured result"
            ) from exc

    def route_dict(
        self,
        question: str,
        *,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Return the validated plan as a JSON-serializable dictionary."""

        return self.route(
            question,
            config=config,
        ).model_dump(mode="json")

    @staticmethod
    def _validate_question(question: str) -> str:
        normalized = str(question).strip()
        if not normalized:
            raise ValueError("question must be a non-empty string")
        return normalized

    @staticmethod
    def _validate_node_name(node_name: str) -> str:
        normalized = str(node_name).strip()
        if not normalized:
            raise ValueError("node_name must be a non-empty string")
        return normalized


def route_kg_question(
    llm_service: LLMService,
    question: str,
    *,
    router_config: RunnableConfig | None = None,
) -> output.KGRetrievalPlan:
    """Convenience wrapper for one-off KG routing."""

    router = KGStructuredRouter(
        llm_service=llm_service,
        router_config=router_config,
    )
    return router.route(question)

class KGMentionsRouter:
    """Extract a minimal validated concept plan for MENTIONS retrieval."""

    def __init__(
        self,
        llm_service: LLMService,
        router_config: RunnableConfig | None = None,
        *,
        node_name: str = KG_MENTIONS_ROUTER_NODE_NAME,
    ) -> None:
        self.llm_service = llm_service
        self.router_config = router_config
        self.node_name = KGStructuredRouter._validate_node_name(node_name)

        try:
            self._node = self.llm_service.build_node(
                self.node_name,
                structured_output=True,
                output_schema=output.KGMentionsPlan,
            )
        except StopIteration as exc:
            raise RuntimeError(
                f"LLM node {self.node_name!r} is not registered in the "
                "configured nodes YAML"
            ) from exc

    def route(
        self,
        question: str,
        *,
        config: RunnableConfig | None = None,
    ) -> output.KGMentionsPlan:
        normalized_question = KGStructuredRouter._validate_question(question)
        invocation_config = (
            config if config is not None else self.router_config
        )
        result = self._node.invoke(
            {"question": normalized_question},
            config=invocation_config,
            with_retry=True,
        )
        if isinstance(result, output.KGMentionsPlan):
            return result
        try:
            return output.KGMentionsPlan.model_validate(result)
        except Exception as exc:
            raise RuntimeError(
                "The KG MENTIONS router returned an invalid structured result"
            ) from exc

    def route_dict(
        self,
        question: str,
        *,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        return self.route(
            question,
            config=config,
        ).model_dump(mode="json")


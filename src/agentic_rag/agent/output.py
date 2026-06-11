from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DetectedLanguage(BaseModel):
    """Detect the language of the input text."""

    language: Literal["it", "en"] = Field(
        description=(
            "The detected language: 'it' for Italian, 'en' for English."
        )
    )

    @staticmethod
    def format_instruction() -> str:
        return (
            "Return ONLY a valid JSON object with exactly one key "
            "'language' whose value is either 'it' or 'en'."
        )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "The relevance score: 'yes' if document is relevant, "
            "'no' if not relevant."
        )
    )

    @staticmethod
    def format_instruction() -> str:
        return (
            "Return ONLY a valid JSON object with exactly one key "
            "'binary_score' whose value is either 'yes' or 'no'."
        )


class GradeDocumentsBatch(BaseModel):
    """Relevance grades for a batch of retrieved documents."""

    grades: List[GradeDocuments] = Field(
        description=(
            "List of grades, one per document, in the same order as "
            "the input documents."
        )
    )

    @staticmethod
    def format_instruction() -> str:
        return (
            "Return ONLY a valid JSON object with exactly one key "
            "'grades' containing a list of objects whose "
            "'binary_score' value is either 'yes' or 'no'."
        )


class DocumentRequest(BaseModel):
    """Whether the user's question requests or refers to a document."""

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "Return 'yes' if the user is asking for or referring to "
            "a document, 'no' otherwise."
        )
    )

    @staticmethod
    def format_instruction() -> str:
        return (
            "Return ONLY a valid JSON object with exactly one key "
            "'binary_score' whose value is either 'yes' or 'no'."
        )


class GradeGrounding(BaseModel):
    """Binary score for grounding check on generation."""

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "The grounding score: 'yes' if generation is grounded "
            "in facts, 'no' if not grounded."
        )
    )

    @staticmethod
    def format_instruction() -> str:
        return (
            "Return ONLY a valid JSON object with exactly one key "
            "'binary_score' whose value is either 'yes' or 'no'."
        )


class GradeAnswer(BaseModel):
    """Binary score for answer addressing question."""

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "The answer score: 'yes' if answer addresses the "
            "question, 'no' if it does not."
        )
    )

    @staticmethod
    def format_instruction() -> str:
        return (
            "Return ONLY a valid JSON object with exactly one key "
            "'binary_score' whose value is either 'yes' or 'no'."
        )


class RouteQuery(BaseModel):
    """Route a user query to the most relevant branch."""

    branch: Literal[
        "conversational",
        "document_request",
        "document_based",
    ] = Field(
        description=(
            "Given a human question, choose the most appropriate branch:\n"
            "- 'conversational': general, casual, social, greetings, "
            "small talk, personal opinions, or gratitude.\n"
            "- 'document_request': the user asks where to find a "
            "specific piece of information in documents.\n"
            "- 'document_based': the user wants a retrieval- or "
            "RAG-based answer from documents."
        )
    )

    @staticmethod
    def format_instruction() -> str:
        return (
            "Return ONLY a valid JSON object with exactly one key "
            "'branch' whose value is either 'conversational', "
            "'document_based', or 'document_request'."
        )


class MultipleQueries(BaseModel):
    """List of generated queries."""

    queries: List[str] = Field(
        description="List of generated queries."
    )


class QueryAmbiguity(BaseModel):
    """Whether the query is ambiguous with respect to the knowledge base."""

    status: Literal["ambiguous", "clear"] = Field(
        description=(
            "Whether the user query is ambiguous or clear with "
            "respect to the knowledge base available to the RAG agent."
        )
    )
    reason: Optional[str] = Field(
        description=(
            "Reason why the query was classified as ambiguous or clear."
        )
    )


class KGToolCall(BaseModel):
    """One deterministic knowledge-graph retrieval call."""

    model_config = ConfigDict(extra="forbid")

    tool: Literal[
        "search_sections_by_concepts",
        "search_sections_by_title",
    ] = Field(
        description=(
            "The deterministic KG retrieval tool to execute. Use "
            "'search_sections_by_concepts' for clinical entities and "
            "'search_sections_by_title' for section intent or document "
            "structure."
        )
    )
    terms: List[str] = Field(
        min_length=1,
        max_length=5,
        description=(
            "One to five concise retrieval terms. Terms must preserve "
            "clinically meaningful qualifiers and must not contain "
            "section numbers or document identifiers."
        ),
    )
    require_all: bool = Field(
        default=False,
        description=(
            "When true, all terms must match the same Section. Use "
            "false when relevant evidence may be distributed across "
            "multiple Sections."
        ),
    )

    @field_validator("terms")
    @classmethod
    def normalize_terms(cls, values: List[str]) -> List[str]:
        normalized: List[str] = []
        seen: set[str] = set()

        for value in values:
            term = str(value).strip()
            if not term:
                continue

            key = term.casefold()
            if key in seen:
                continue

            seen.add(key)
            normalized.append(term)

        if not normalized:
            raise ValueError(
                "At least one non-empty retrieval term is required"
            )

        if len(normalized) > 5:
            raise ValueError(
                "A KG tool call may contain at most five unique terms"
            )

        return normalized


class KGRetrievalPlan(BaseModel):
    """Structured plan for deterministic knowledge-graph retrieval."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal[
        "definition",
        "diagnosis",
        "management",
        "treatment",
        "risk_stratification",
        "monitoring",
        "comparison",
        "other",
    ] = Field(
        description=(
            "The main information need expressed by the question."
        )
    )
    expected_scope: Literal[
        "single_section",
        "multiple_sections",
        "cross_document",
    ] = Field(
        description=(
            "The expected evidence scope. This describes whether the "
            "answer is likely to require one Section, several Sections, "
            "or evidence from multiple documents. It must not identify "
            "specific gold documents."
        )
    )
    calls: List[KGToolCall] = Field(
        min_length=1,
        max_length=2,
        description=(
            "One or two deterministic KG retrieval calls. Use both "
            "concept and title search when the question combines a "
            "clinical entity with a distinct intent."
        ),
    )

    @staticmethod
    def format_instruction() -> str:
        return (
            "Return ONLY a valid JSON object with the keys 'intent', "
            "'expected_scope', and 'calls'. 'calls' must contain one "
            "or two objects, each with 'tool', 'terms', and "
            "'require_all'. Do not generate Cypher, document IDs, "
            "section IDs, explanations, or an answer to the question."
        )

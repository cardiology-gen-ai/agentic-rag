from typing import Literal, List, Optional

from pydantic import BaseModel, Field

class DetectedLanguage(BaseModel):
    """Detect the language of the input text."""
    language: Literal["it", "en"] = Field(description="The detected language: 'it' for Italian, 'en' for English.") #: :class:`typing.Literal`\[`it`, `en`\] : The detected language, 'it' for Italian, 'en' for English.

    @staticmethod
    def format_instruction() -> str:
        return "Return ONLY a valid JSON object with exactly one key 'language' whose value is either 'it' or 'en'."


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(description="The relevance score: 'yes' if document is relevant, 'no' if not relevant.") #: :class:`typing.Literal`\[`yes`, `no`\] : The relevance score, 'yes' if document is relevant, 'no' if not relevant.

    @staticmethod
    def format_instruction() -> str:
        return "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."


class GradeDocumentsBatch(BaseModel):
    """Relevance grades for a batch of retrieved documents."""
    grades: List[GradeDocuments] = Field(
        description="List of grades, one per document, in the same order as the input documents."
    )

    @staticmethod
    def format_instruction() -> str:
        return "Return ONLY a valid JSON object with exactly one key 'grades' which contain a list whose values is either 'yes' or 'no'."


class DocumentRequest(BaseModel):
    """Binary score to assess whether the user's question implies a request for a document."""
    binary_score: Literal["yes", "no"] = Field(description="Return 'yes' if the user is asking for or referring to a document, 'no' otherwise.") #: :class:`typing.Literal`\[`yes`, `no`\] : Return 'yes' if the user is asking for or referring to a document, 'no' otherwise.

    @staticmethod
    def format_instruction() -> str:
        return "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."


class GradeGrounding(BaseModel):
    """Binary score for grounding check on generation."""
    binary_score: Literal["yes", "no"] = Field(description="The grounding score: 'yes' if generation is grounded in facts, 'no' if not grounded.") #: :class:`typing.Literal`\[`yes`, `no`\] : The grounding score, 'yes' if generation is grounded in facts, 'no' if not grounded.

    @staticmethod
    def format_instruction() -> str:
        return "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."


class GradeAnswer(BaseModel):
    """Binary score for answer addressing question."""
    binary_score: Literal["yes", "no"] = Field(description="The answer score: 'yes' if answer addresses the question, 'no' if it doesn't.") #: :class:`typing.Literal`\[`yes`, `no`\] : The answer score, 'yes' if answer addresses the question, 'no' if it doesn't.

    @staticmethod
    def format_instruction() -> str:
        return "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."


class RouteQuery(BaseModel):
    """Route a user query to the most relevant branch."""
    branch: Literal["conversational", "document_request", "document_based"] = Field(
        description=("Given a human question, choose the most appropriate branch:\n "
                     "- 'conversational': general, casual, social, greetings, small talk, personal opinions, gratitude.\n"
                     "- 'document_request': user asks where to find a specific piece of information in documents (e.g., which file contains X).\n"
                     "- 'document_based': user wants a retrieval or RAG-based answer from documents.")
    ) #: :class:`typing.Literal`\[`document_based`, `conversational`, `other`\] :  Given a human question choose to route it to conversational branch or a document retrieving mechanism.

    @staticmethod
    def format_instruction() -> str:
        return "Return ONLY a valid JSON object with exactly one key 'branch' whose value is either 'conversational', 'document_based' or 'document_request'"


class MultipleQueries(BaseModel):
    """List of generated queries"""
    queries: List[str] = Field(description="List of generated queries")


class QueryAmbiguity(BaseModel):
    """Whether the user query is ambiguous w.r.t. the knowledge base available to the RAG agent."""
    status: Literal["ambiguous", "clear"] = Field(
        description="Whether the user query is ambiguous or clear w.r.t. the knowledge base available to the RAG agent."
    )
    reason: Optional[str] = Field(description="Reason why the status is selected as ambiguous or clear.")
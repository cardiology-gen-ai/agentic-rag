from typing import Literal

from pydantic import BaseModel, Field

class DetectLanguage(BaseModel):
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
    branch: Literal["document_based", "conversational", "other"] = Field(
        description="Given a human question choose to route it to conversational branch "
                    "or a document retrieving mechanism."
    ) #: :class:`typing.Literal`\[`document_based`, `conversational`, `other`\] :  Given a human question choose to route it to conversational branch or a document retrieving mechanism.

    @staticmethod
    def format_instruction() -> str:
        return "Return ONLY a valid JSON object with exactly one key 'branch' whose value is either 'conversational' or 'document_based'."
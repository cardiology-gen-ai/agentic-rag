from typing import Literal

from pydantic import BaseModel, Field

class DetectLanguage(BaseModel):
    """Detect the language of the input text."""
    language: Literal["it", "en"] = Field(description="The detected language: 'it' for Italian, 'en' for English.")

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(description="The relevance score: 'yes' if document is relevant, 'no' if not relevant")


class DocumentRequest(BaseModel):
    """Binary score to assess whether the user's question implies a request for a document."""
    binary_score: Literal["yes", "no"] = Field(description="Return 'yes' if the user is asking for or referring to a document, 'no' otherwise.")


class GradeGrounding(BaseModel):
    """Binary score for grounding check on generation."""
    binary_score: Literal["yes", "no"] = Field(description="The grounding score: 'yes' if generation is grounded in facts, 'no' if not grounded")


class GradeAnswer(BaseModel):
    """Binary score for answer addressing question."""
    binary_score: Literal["yes", "no"] = Field(description="The answer score: 'yes' if answer addresses the question, 'no' if it doesn't")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant branch."""
    branch: Literal["document_based", "conversational", "other"] = Field(
        description="Given a human question choose to route it to conversational branch "
                    "or a document retrieving mechanism."
    )
#!/usr/bin/env python3
from src.utils.base_state import BaseState

class RagState(BaseState):
    transform_query_count: int
    generation_count: int
    contextual_question: str
    response: str
    generation: str
    documents: list[str]
    examples: list[str]
    document_request: str
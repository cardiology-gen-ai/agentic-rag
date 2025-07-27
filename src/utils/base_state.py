#!/usr/bin/env python3
from typing import Optional, List, Any, Dict, Annotated
from langchain_core.messages import AnyMessage # type: ignore
from langgraph.graph.message import add_messages # type: ignore
from pydantic import BaseModel # type: ignore

class BaseState(BaseModel):
    """
    This class represents the short-term memory of the agentic-rag.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    question: str

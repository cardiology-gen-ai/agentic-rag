#!/usr/bin/env python3
from typing import Optional, List, Any, Dict, Annotated
from langchain_core.messages import AnyMessage # type: ignore
from langgraph.graph.message import add_messages # type: ignore
from pydantic import BaseModel # type: ignore

class State(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    response: Optional[str]=""
    transform_query_count: Optional[int]=0
    generation_count: Optional[int]=0
    documents: Optional[List[str]]=[]

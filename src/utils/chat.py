#!/usr/bin/env python3
from pydantic import BaseModel # type: ignore
from typing import Literal, Any
from datetime import datetime

class MessageSchema(BaseModel):
    id: str | None = None
    datetime: datetime
    role: Literal["user", "assistant", "admin"]
    content: str
    metadata: Any | None = None

class ConversationSchema(BaseModel):
    id: str
    agentId: str
    question: MessageSchema
    history: list[MessageSchema]

class ChatSchema(BaseModel):
    user: str
    conversation: ConversationSchema
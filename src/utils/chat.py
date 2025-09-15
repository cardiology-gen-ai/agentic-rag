from datetime import datetime
from typing import Literal, Any, List, Dict

from pydantic import BaseModel


class MessageSchema(BaseModel):
    id: str | None = None
    datetime: datetime
    role: Literal["user", "assistant", "admin"]
    content: str
    metadata: Any | None = None


class ConversationRequest(BaseModel):
    id: str
    chatbotId: str
    question: MessageSchema
    history: List[MessageSchema]


class ChatRequest(BaseModel):
    user: str
    user_id: str | None = None
    conversation: ConversationRequest


class ChatResponse(BaseModel):
    role: Literal["user", "assistant", "admin"]
    content: str
    metadata: Dict
    is_faulted: bool = False

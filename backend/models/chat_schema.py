from pydantic import BaseModel, Field
from typing import Literal


class MessageItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    case_id: str
    message: str
    history: list[MessageItem] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    intent: str | None = None
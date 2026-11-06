from pydantic import BaseModel, Field

SESSION_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_PATTERN, examples=["gustavo"])
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]
    tools_used: list[str]
    iterations: int


class HealthResponse(BaseModel):
    status: str
    model: str
    indexed_chunks: int


class DocumentResponse(BaseModel):
    filename: str
    chunks_indexed: int


class ErrorResponse(BaseModel):
    detail: str

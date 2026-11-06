import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app import config, factory
from app.schemas.models import (
    ChatRequest,
    ChatResponse,
    DocumentResponse,
    HealthResponse,
)
from app.services.llm import LLMError, LLMRateLimitError, LLMTimeoutError
from app.services.rag import ALLOWED

log = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 1_000_000


def create_app(agent=None, index=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is None:
            app.state.agent, app.state.index = factory.build()
        else:
            app.state.agent, app.state.index = agent, index
        yield

    app = FastAPI(title="ResearchPilot", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(LLMRateLimitError)
    async def _rate(_: Request, e: LLMRateLimitError):
        return JSONResponse({"detail": "Muitas requisições ao modelo. Tente novamente em instantes."}, 429)

    @app.exception_handler(LLMTimeoutError)
    async def _timeout(_: Request, e: LLMTimeoutError):
        return JSONResponse({"detail": "O modelo demorou demais para responder."}, 504)

    @app.exception_handler(LLMError)
    async def _llm(_: Request, e: LLMError):
        log.error("erro do LLM: %s", e)
        return JSONResponse({"detail": "Falha ao consultar o modelo."}, 502)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        idx = request.app.state.index
        return HealthResponse(
            status="ok", model=config.MODEL, indexed_chunks=len(idx.entries) if idx else 0
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request):
        r = request.app.state.agent.chat(body.session_id, body.message)
        return ChatResponse(
            session_id=body.session_id,
            answer=r.answer,
            sources=r.sources,
            tools_used=r.tools_used,
            iterations=r.iterations,
        )

    @app.post("/documents", response_model=DocumentResponse, status_code=201)
    def upload_document(file: UploadFile, request: Request):
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in ALLOWED:
            raise HTTPException(415, "Use arquivos .txt ou .md")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Arquivo maior que 1 MB")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(422, "O arquivo precisa estar em UTF-8")
        if not text.strip():
            raise HTTPException(422, "Arquivo vazio")
        idx = request.app.state.index
        idx.docs_dir.mkdir(parents=True, exist_ok=True)
        (idx.docs_dir / name).write_text(text, encoding="utf-8")
        idx.sync()
        chunks = sum(1 for e in idx.entries if e["source"] == name)
        log.info("documento %s indexado (%d trechos)", name, chunks)
        return DocumentResponse(filename=name, chunks_indexed=chunks)

    return app


app = create_app()

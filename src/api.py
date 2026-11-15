import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from groq import APIStatusError, RateLimitError
from pydantic import BaseModel, Field, field_validator

from src.embedding import build_vector_stores, get_compute_device
from src.generation import build_answer_chain, generate_answer
from src.retrieval import (
    build_reranker,
    build_router,
    route_and_retrieve,
)


logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question must not be blank.")
        return question


class SourceReference(BaseModel):
    source: str
    page_start: int
    page_end: int
    page_label: str
    reranker_score: float


class ChatResponse(BaseModel):
    answer: str
    session_id: UUID
    collections: list[str]
    sources: list[SourceReference]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for logger_name in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    app.state.router = build_router()
    app.state.answer_chain = build_answer_chain()
    if get_compute_device() == "cpu":
        with ThreadPoolExecutor(max_workers=2) as pool:
            stores_future = pool.submit(build_vector_stores)
            reranker_future = pool.submit(build_reranker)
            app.state.vector_stores = stores_future.result()
            app.state.reranker = reranker_future.result()
    else:
        app.state.vector_stores = build_vector_stores()
        app.state.reranker = build_reranker()
    app.state.sessions: dict[UUID, list[BaseMessage]] = {}
    app.state.sessions_lock = Lock()
    yield


app = FastAPI(
    title="Chatbot RAG API",
    description="Routes questions, retrieves and reranks Qdrant chunks, then generates cited answers.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    request_started = perf_counter()
    sessions: dict[UUID, list[BaseMessage]] = request.app.state.sessions
    session_lock: Lock = request.app.state.sessions_lock

    if payload.session_id is None:
        session_id = uuid4()
        history: list[BaseMessage] = []
    else:
        session_id = payload.session_id
        with session_lock:
            if session_id not in sessions:
                raise HTTPException(
                    status_code=404,
                    detail="Session not found. Start a new chat without session_id.",
                )
            history = list(sessions[session_id])

    try:
        collections, ranked_documents = route_and_retrieve(
            payload.question,
            request.app.state.router,
            request.app.state.vector_stores,
            request.app.state.reranker,
        )
        documents = [document for document, _ in ranked_documents]
        generation_started = perf_counter()
        answer = generate_answer(
            payload.question,
            documents,
            chat_history=history[-8:],
            chain=request.app.state.answer_chain,
        )
        logger.info(
            "Answer generation took %.2f seconds.",
            perf_counter() - generation_started,
        )
    except RateLimitError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Groq rejected the request due to its rate limit. "
                "Wait briefly and try again."
            ),
        ) from error
    except APIStatusError as error:
        if error.status_code == 401:
            detail = "Groq rejected the API key. Check GROQ_API_KEY in .env."
        elif error.status_code == 403:
            detail = "Groq blocked this request. Check the key and account limits."
        else:
            detail = (
                f"Groq request failed with HTTP {error.status_code}. "
                "Check model availability and account limits."
            )
        raise HTTPException(status_code=502, detail=detail) from error

    sources = []
    seen_sources = set()
    for document, score in ranked_documents:
        metadata = document.metadata
        source_key = (
            metadata["source"],
            metadata["page_start"],
            metadata["page_end"],
        )
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        sources.append(
            SourceReference(
                source=metadata["source"],
                page_start=metadata["page_start"],
                page_end=metadata["page_end"],
                page_label=metadata["page_label"],
                reranker_score=score,
            )
        )

    with session_lock:
        sessions[session_id] = (
            history + [HumanMessage(content=payload.question), AIMessage(content=answer)]
        )[-8:]

    logger.info(
        "Chat request completed in %.2f seconds.",
        perf_counter() - request_started,
    )
    return ChatResponse(
        answer=answer,
        session_id=session_id,
        collections=collections,
        sources=sources,
    )

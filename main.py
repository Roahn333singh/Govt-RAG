import asyncio
import json
import os
import time
import uuid
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

# FIX MACOS PROXY HANG BUG
urllib.request.getproxies = lambda: {}

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from agent.guards import (
    INJECTION_REFUSAL,
    MAX_MESSAGE_CHARS,
    PROMPT_VERSION,
    REFUSAL_TEXT,
    answer_echoes_markers,
    cited_sources,
    detect_answer_language,
    filter_output,
    is_injection,
    validate_message_length,
)
from agent.cache_policy import (
    CACHE_SIM_THRESHOLD,
    doc_version,
    entry_fresh,
    is_premise_correction,
    questions_match,
)
from agent.security import (
    SESSION_TOKEN_TTL_SECONDS,
    admin_token_ok,
    issue_session_token,
    real_client_ip,
    require_client,
)
from vectorDB import cache_admin, search

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "90"))
# Per-instance cap on concurrent graph runs (LLM + rerank); others queue.
MAX_INFLIGHT_LLM = int(os.getenv("MAX_INFLIGHT_LLM", "6"))
_llm_slots = asyncio.Semaphore(MAX_INFLIGHT_LLM)
BUSY_DETAIL = "Service busy, please retry"

_REFUSAL_MARKERS = (
    "I am an AI assistant specialized",
    "does not contain any information",
    "does not contain information about",
    "no information about",
    "does not specify",
    "do not explicitly list",
    "कोई जानकारी उपलब्ध नहीं",
    "जानकारी उपलब्ध नहीं है",
    "I can only answer questions about UP Government",
    "not in the documents",
    "provided documents do not",
)


CHAT_RATE_MINUTE = os.getenv("CHAT_RATE_MINUTE", "15/minute")
CHAT_RATE_DAY = os.getenv("CHAT_RATE_DAY", "100/day")

_RATE_LIMIT_STORAGE = os.getenv("RATE_LIMIT_STORAGE_URI") or "memory://"
if _RATE_LIMIT_STORAGE == "memory://":
    print("⚠️  RATE_LIMIT_STORAGE_URI not set — rate limits are per-instance only")

# swallow_errors + in-memory fallback: a Redis outage degrades to
# per-instance limits instead of failing every request.
limiter = Limiter(
    key_func=real_client_ip,
    storage_uri=_RATE_LIMIT_STORAGE,
    swallow_errors=True,
    in_memory_fallback_enabled=_RATE_LIMIT_STORAGE != "memory://",
)


def get_compile_all():
    from agent.lang_graph import compile_all

    return compile_all


def _configure_thread_pool() -> None:
    # Default pool is min(32, cpus + 4) = 6 on Cloud Run's 2 vCPU, which would
    # serialise concurrent embedding / Qdrant calls behind each other.
    from concurrent.futures import ThreadPoolExecutor

    workers = int(os.getenv("IO_THREADS", "32"))
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="io")
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_thread_pool()
    await asyncio.to_thread(search.init_services)
    # Warm-compile the LangGraph once
    get_compile_all()()
    yield


_DOCS_ENABLED = os.getenv("ENABLE_DOCS") == "1"
app = FastAPI(
    lifespan=lifespan,
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url="/redoc" if _DOCS_ENABLED else None,
    openapi_url="/openapi.json" if _DOCS_ENABLED else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_ALLOWED_ORIGINS = [
    o.strip()
    for o in (
        os.getenv("ALLOWED_ORIGINS")
        or os.getenv("CORS_ORIGINS")
        or "https://manual-ai-rag-nszha5soya-ey.a.run.app"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Session-Token"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.tailwindcss.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )
    return response


class ChatRequest(BaseModel):
    message: str | None = None
    text: str | None = None
    question: str | None = None
    prompt: str | None = None
    chatInput: str | None = None
    session_id: str | None = None
    sessionId: str | None = None


def generate() -> str:
    return str(uuid.uuid4())


def _pick_first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if not isinstance(value, str):
            continue
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


def resolve_message(message: str | None, chat_data: ChatRequest | None) -> str:
    resolved_message = _pick_first_non_empty(
        message,
        chat_data.message if chat_data else None,
        chat_data.text if chat_data else None,
        chat_data.question if chat_data else None,
        chat_data.prompt if chat_data else None,
        chat_data.chatInput if chat_data else None,
    )
    if not resolved_message:
        raise HTTPException(status_code=422, detail="Message is required.")
    try:
        validate_message_length(resolved_message)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    return resolved_message


def resolve_session_id(
    generated_session_id: str, session_id: str | None, chat_data: ChatRequest | None
) -> str:
    return _pick_first_non_empty(
        session_id,
        chat_data.session_id if chat_data else None,
        chat_data.sessionId if chat_data else None,
        generated_session_id,
    )


def _is_cacheable_answer(
    answer: str, sources: list | None, question: str | None = None
) -> bool:
    if not answer:
        return False
    if any(marker in answer for marker in _REFUSAL_MARKERS):
        return False
    if filter_output(answer, question) is None:
        return False
    if question and answer_echoes_markers(question, answer):
        return False
    # Never cache answers to soft-instruction / marker-bearing questions
    if question and is_injection(question):
        return False
    if question and is_premise_correction(question, answer):
        return False
    if not sources:
        return False
    return True


def _query_cache_sync(query_vector: list, user_lang: str):
    return search.qdrant_client.query_points(
        collection_name=search.CACHE_COLLECTION_NAME,
        query=query_vector,
        limit=1,
        query_filter=Filter(
            must=[FieldCondition(key="language", match=MatchValue(value=user_lang))]
        ),
    ).points


async def _lookup_cache(message: str, user_lang: str, query_vector: list):
    cache_results = await asyncio.to_thread(_query_cache_sync, query_vector, user_lang)
    if not cache_results:
        return None
    hit = cache_results[0]
    if hit.score < CACHE_SIM_THRESHOLD:
        print(f"❌ CACHE MISS. Closest {hit.score:.2f} (needs > {CACHE_SIM_THRESHOLD})")
        return None
    payload = hit.payload or {}
    cached_q = payload.get("question") or ""
    if not questions_match(message, cached_q):
        print(f"❌ CACHE MISS. Wording/numbers differ at score {hit.score:.2f}")
        return None
    if not entry_fresh(payload):
        print("❌ CACHE MISS. Stale prompt/doc version or TTL expired")
        return None
    print(f"✅ CACHE HIT! Match Confidence: {hit.score:.2f}")
    return payload


async def _write_cache(
    message: str,
    answer: str,
    user_lang: str,
    query_vector: list,
    sources: list,
):
    await asyncio.to_thread(
        search.qdrant_client.upsert,
        collection_name=search.CACHE_COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=query_vector,
                payload={
                    "question": message,
                    "answer": answer,
                    "language": user_lang,
                    "sources": sources,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "prompt_version": PROMPT_VERSION,
                    "doc_version": doc_version(),
                },
            )
        ],
    )


def _busy_response(session_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": "5"},
        content={
            "detail": BUSY_DETAIL,
            "answer": "The desk is busy. Please retry your question in a few seconds.",
            "sources": [],
            "session_id": session_id,
            "cached": False,
        },
    )


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _text_chunks(text: str, words_per_chunk: int = 6):
    words = (text or "").split(" ")
    total = len(words)
    for i in range(0, total, words_per_chunk):
        piece = " ".join(words[i : i + words_per_chunk])
        if i + words_per_chunk < total:
            piece += " "
        if piece:
            yield piece


def _chat_response(
    answer: str,
    sources: list | None,
    session_id: str,
    cached: bool,
) -> dict:
    return {
        "answer": answer,
        "sources": cited_sources(answer, sources),
        "session_id": session_id,
        "cached": cached,
    }


@app.get("/health")
async def health_chk():
    return {"status": "ok"}


@app.get("/ready")
async def ready_chk():
    if not search.services_ready():
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        await asyncio.to_thread(
            search.qdrant_client.get_collection, search.COLLECTION_NAME
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {e}") from e
    return {"status": "ready"}


@app.get("/ui/session", include_in_schema=False)
@limiter.limit("20/minute")
async def ui_session(request: Request):
    """Short-lived chat token for the same-origin web UI."""
    if request.headers.get("sec-fetch-site") != "same-origin":
        raise HTTPException(status_code=403, detail="Session tokens are only issued to the web UI.")
    return {
        "token": issue_session_token(real_client_ip(request)),
        "expires_in": SESSION_TOKEN_TTL_SECONDS,
    }


class CachePurgeRequest(BaseModel):
    contains: str | list[str] | None = None
    all: bool = False
    dry_run: bool = False


@app.post("/admin/cache/purge", include_in_schema=False)
@limiter.limit("10/minute")
async def admin_cache_purge(request: Request, body: CachePurgeRequest):
    if not admin_token_ok(request.headers.get("x-admin-token")):
        raise HTTPException(status_code=401, detail="Invalid admin token.")
    needles = [body.contains] if isinstance(body.contains, str) else body.contains
    needles = [n for n in (needles or []) if n and n.strip()]
    if not body.all and not needles:
        raise HTTPException(status_code=422, detail='Send {"contains": "..."} or {"all": true}.')
    result = await asyncio.to_thread(
        cache_admin.purge,
        search.qdrant_client,
        needles=needles,
        wipe=body.all,
        dry_run=body.dry_run,
        collection=search.CACHE_COLLECTION_NAME,
    )
    print(f"admin cache purge: all={body.all} needles={needles} -> {result['deleted']} deleted")
    return result


_UI_DIR = os.path.join(os.path.dirname(__file__), "govt_manual_ui")
if os.path.isdir(_UI_DIR):
    app.mount("/ui", StaticFiles(directory=_UI_DIR, html=True), name="ui")


@app.post("/chat")
@limiter.limit(CHAT_RATE_MINUTE)
@limiter.limit(CHAT_RATE_DAY)
async def users_chat(
    request: Request,
    generated_session_id: Annotated[str, Depends(generate)],
    _auth: Annotated[None, Depends(require_client)],
    message: str | None = None,
    session_id: str | None = None,
    chat_data: ChatRequest | None = Body(default=None),
):
    message = resolve_message(message, chat_data)
    session_id = resolve_session_id(generated_session_id, session_id, chat_data)
    user_lang = detect_answer_language(message)

    if is_injection(message):
        return _chat_response(INJECTION_REFUSAL, [], session_id, cached=False)

    try:
        async def _run_chat():
            query_vector = await asyncio.to_thread(
                search.dense_embedder.embed_query, message
            )

            cached = await _lookup_cache(message, user_lang, query_vector)
            if cached:
                raw = cached.get("answer") or ""
                if answer_echoes_markers(message, raw) or filter_output(raw, message) is None:
                    return _chat_response(INJECTION_REFUSAL, [], session_id, cached=False)
                answer = filter_output(raw, message) or REFUSAL_TEXT
                return _chat_response(
                    answer,
                    cached.get("sources") or [],
                    session_id,
                    cached=True,
                )

            print("Asking processor to think...")
            graph = get_compile_all()()
            config = {"configurable": {"thread_id": session_id}}
            async with _llm_slots:
                result = await graph.ainvoke(
                    {
                        "messages": [{"role": "user", "content": message}],
                        "session_id": session_id,
                        "answer_language": user_lang,
                        "query_vector": query_vector,
                        "sources": [],
                    },
                    config,
                )
            final_answer = result["messages"][-1].content
            sources = result.get("sources") or []

            filtered = filter_output(final_answer, message)
            if filtered is None:
                final_answer = INJECTION_REFUSAL
                sources = []
            else:
                final_answer = filtered
            sources = cited_sources(final_answer, sources)

            if _is_cacheable_answer(final_answer, sources, message):
                await _write_cache(
                    message, final_answer, user_lang, query_vector, sources
                )

            return _chat_response(final_answer, sources, session_id, cached=False)

        return await asyncio.wait_for(_run_chat(), timeout=REQUEST_TIMEOUT_SECONDS)

    except asyncio.TimeoutError:
        print(f"/chat timeout after {REQUEST_TIMEOUT_SECONDS}s")
        return _busy_response(session_id)
    except Exception as e:
        print(f"/chat error: {type(e).__name__}: {e}")
        return _busy_response(session_id)


@app.post("/chat/stream")
@limiter.limit(CHAT_RATE_MINUTE)
@limiter.limit(CHAT_RATE_DAY)
async def users_chat_stream(
    request: Request,
    generated_session_id: Annotated[str, Depends(generate)],
    _auth: Annotated[None, Depends(require_client)],
    message: str | None = None,
    session_id: str | None = None,
    chat_data: ChatRequest | None = Body(default=None),
):
    message = resolve_message(message, chat_data)
    session_id = resolve_session_id(generated_session_id, session_id, chat_data)
    user_lang = detect_answer_language(message)

    if is_injection(message):

        async def refusal_stream():
            yield _sse({"type": "token", "text": INJECTION_REFUSAL})
            yield _sse({"type": "done", "sources": [], "session_id": session_id, "cached": False})

        return StreamingResponse(refusal_stream(), media_type="text/event-stream")

    try:
        query_vector = await asyncio.to_thread(
            search.dense_embedder.embed_query, message
        )
        cached = await _lookup_cache(message, user_lang, query_vector)
        if cached:
            raw = cached.get("answer") or ""
            if answer_echoes_markers(message, raw) or filter_output(raw, message) is None:
                answer, sources = INJECTION_REFUSAL, []
            else:
                answer = filter_output(raw, message) or REFUSAL_TEXT
                sources = cited_sources(answer, cached.get("sources") or [])

            async def cached_stream():
                for piece in _text_chunks(answer):
                    yield _sse({"type": "token", "text": piece})
                yield _sse(
                    {
                        "type": "done",
                        "sources": sources,
                        "session_id": session_id,
                        "cached": True,
                    }
                )

            return StreamingResponse(cached_stream(), media_type="text/event-stream")

        graph = get_compile_all()()
        config = {"configurable": {"thread_id": session_id}}

        async def generate_and_save_stream():
            final_answer = ""
            sources: list = []
            try:
                async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                    async with _llm_slots:
                        async for event in graph.astream_events(
                            {
                                "messages": [{"role": "user", "content": message}],
                                "session_id": session_id,
                                "answer_language": user_lang,
                                "query_vector": query_vector,
                                "sources": [],
                            },
                            config,
                            version="v2",
                        ):
                            kind = event.get("event")
                            if kind == "on_chat_model_stream":
                                meta = event.get("metadata") or {}
                                node = meta.get("langgraph_node")
                                if node and node != "generator":
                                    continue
                                chunk = event.get("data", {}).get("chunk")
                                content = getattr(chunk, "content", None) if chunk else None
                                if isinstance(content, list):
                                    content = "".join(
                                        part.get("text", "")
                                        for part in content
                                        if isinstance(part, dict)
                                    )
                                if content:
                                    yield _sse({"type": "token", "text": content})
                                    final_answer += content
                            elif kind == "on_chain_end":
                                output = (event.get("data") or {}).get("output") or {}
                                if isinstance(output, dict) and output.get("sources"):
                                    sources = output["sources"]

                filtered = filter_output(final_answer, message)
                if filtered is None:
                    final_answer = INJECTION_REFUSAL
                    sources = []
                    yield _sse({"type": "replace", "text": INJECTION_REFUSAL})
                else:
                    final_answer = filtered
                    sources = cited_sources(final_answer, sources)
                    if _is_cacheable_answer(final_answer, sources, message):
                        await _write_cache(
                            message, final_answer, user_lang, query_vector, sources
                        )
                yield _sse(
                    {
                        "type": "done",
                        "sources": sources,
                        "session_id": session_id,
                        "cached": False,
                    }
                )
            except TimeoutError:
                print(f"/chat/stream timeout after {REQUEST_TIMEOUT_SECONDS}s")
                yield _sse({"type": "token", "text": "The desk is busy. Please retry your question in a few seconds."})
                yield _sse({"type": "done", "sources": [], "session_id": session_id, "cached": False})
            except Exception as e:
                print(f"Streaming error: {type(e).__name__}: {e}")
                yield _sse({"type": "token", "text": "Unable to retrieve an answer right now. Please try again."})
                yield _sse({"type": "done", "sources": [], "session_id": session_id, "cached": False})

        return StreamingResponse(
            generate_and_save_stream(), media_type="text/event-stream"
        )
    except Exception as e:
        print(f"/chat/stream error: {type(e).__name__}: {e}")

        async def err_stream():
            yield _sse({"type": "token", "text": "Unable to retrieve an answer right now. Please try again."})
            yield _sse({"type": "done", "sources": [], "session_id": session_id, "cached": False})

        return StreamingResponse(err_stream(), media_type="text/event-stream")

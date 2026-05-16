import os
import sys
import uuid
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Language detection
from langdetect import detect
from langgraph.config import get_stream_writer
from pydantic import BaseModel
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

# IMplementing Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent.lang_graph import RagState, compile_all
from vectorDB.search import CACHE_COLLECTION_NAME, dense_embedder, qdrant_client

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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


@app.get("/health")
async def health_chk():
    return {"status": "ok"}


@app.post("/chat")
@limiter.limit("15/minute")
async def users_chat(
    request: Request,
    generated_session_id: Annotated[str, Depends(generate)],
    message: str | None = None,
    session_id: str | None = None,
    chat_data: ChatRequest | None = Body(default=None),
):
    message = resolve_message(message, chat_data)
    session_id = resolve_session_id(generated_session_id, session_id, chat_data)

    try:
        user_lang = detect(message)
        print(f"User Language: {user_lang}")
    except:
        user_lang = "en"

    query_vector = dense_embedder.embed_query(message)
    cache_results = qdrant_client.query_points(
        collection_name=CACHE_COLLECTION_NAME,
        query=query_vector,
        limit=1,
        query_filter=Filter(
            must=[FieldCondition(key="language", match=MatchValue(value=user_lang))]
        ),
    ).points

    if cache_results and cache_results[0].score > 0.90:  # Lowered from 0.95 to 0.85
        print(f"✅ CACHE HIT! Match Confidence: {cache_results[0].score:.2f}")
        return cache_results[0].payload["answer"]

    if cache_results:
        print(
            f"❌ CACHE MISS. Closest match was only {cache_results[0].score:.2f} (Needs > 0.85)"
        )
    else:
        print("❌ CACHE MISS. No previous questions found.")

    print("Asking processor to think...")
    graph = compile_all()
    config = {"configurable": {"thread_id": session_id}}
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": message}], "session_id": session_id},
        config,
    )
    final_answer = result["messages"][-1].content

    if "I am an AI assistant specialized" not in final_answer:
        qdrant_client.upsert(
            collection_name=CACHE_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=query_vector,
                    payload={
                        "question": message,
                        "answer": final_answer,
                        "language": user_lang,
                    },
                )
            ],
        )

    return final_answer


@app.post("/chat/stream")
@limiter.limit("15/minute")
async def users_chat_stream(
    request: Request,
    generated_session_id: Annotated[str, Depends(generate)],
    message: str | None = None,
    session_id: str | None = None,
    chat_data: ChatRequest | None = Body(default=None),
):
    message = resolve_message(message, chat_data)
    session_id = resolve_session_id(generated_session_id, session_id, chat_data)

    try:
        user_lang = detect(message)
        print(f"User Language: {user_lang}")
    except:
        user_lang = "en"

    query_vector = dense_embedder.embed_query(message)
    cache_results = qdrant_client.query_points(
        collection_name=CACHE_COLLECTION_NAME,
        query=query_vector,
        limit=1,
        query_filter=Filter(
            must=[FieldCondition(key="language", match=MatchValue(value=user_lang))]
        ),
    ).points

    # 1. CACHE HIT: Instantly yield the full saved answer
    if cache_results and cache_results[0].score > 0.90:
        print(f"✅ CACHE HIT! Streaming instantly...")

        async def cached_stream():
            yield cache_results[0].payload["answer"]

        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    # 2. CACHE MISS: We must construct a real-time generator
    print("❌ CACHE MISS. Generating and streaming live...")
    graph = compile_all()
    config = {"configurable": {"thread_id": session_id}}

    # This handles the chunk-by-chunk streaming
    async def generate_and_save_stream():
        final_answer = ""
        try:
            # stream_mode="updates" yields {node_name: state_update} dicts.
            # It is stable across LangGraph versions and avoids the Pydantic v2
            # incompatibility that stream_mode="messages" has with message chunk
            # construction. The frontend applies its own typewriter effect anyway.
            async for update in graph.astream(
                {
                    "messages": [{"role": "user", "content": message}],
                    "session_id": session_id,
                },
                config,
                stream_mode="updates",
            ):
                # Only care about the generator node's output
                if "generator" not in update:
                    continue

                messages = update["generator"].get("messages", [])
                for msg in messages:
                    content = getattr(msg, "content", "")
                    # content can be str or list[dict] for multimodal models
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict)
                        )
                    if content:
                        yield content
                        final_answer += content

            # Finished — write answer to Qdrant semantic cache
            if final_answer and "I am an AI assistant specialized" not in final_answer:
                qdrant_client.upsert(
                    collection_name=CACHE_COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=query_vector,
                            payload={
                                "question": message,
                                "answer": final_answer,
                                "language": user_lang,
                            },
                        )
                    ],
                )
        except Exception as e:
            print(f"Streaming error: {e}")

    # Return the generator wrapped in a Streaming HTTP response
    return StreamingResponse(generate_and_save_stream(), media_type="text/event-stream")


# # How does the Computer Centre digitize things?
# # what are the Major Functions and Works of ISO?
# # what is the vision of iso?


# from fastapi import FastAPI, Depends, Request
# from pydantic import BaseModel
# from typing import Annotated
# import uuid
# import os
# import sys

# # Core Logic Imports (Assuming these match your directory structure)
# from agent.lang_graph import RagState, compile_all
# from vectorDB.search import dense_embedder, qdrant_client, CACHE_COLLECTION_NAME
# from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

# # FastAPI Utilities
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware

# # Rate Limiting
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded

# # Language detection
# from langdetect import detect

# app = FastAPI()

# # --- 1. CORS CONFIGURATION (Crucial for Mobile-to-Mac connection) ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# limiter = Limiter(key_func=get_remote_address)
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# # --- 2. REQUEST MODELS (For JSON Body parsing) ---
# class ChatRequest(BaseModel):
#     message: str

# @app.get("/health")
# async def health_chk():
#     # If your phone browser sees {"status":"ok"}, the bridge is working!
#     return {"status": "ok"}

# # --- 3. THE UPDATED CHAT ENDPOINT ---
# @app.post("/chat")
# @limiter.limit("20/minute") # Increased limit for testing
# async def users_chat(request: Request, chat_data: ChatRequest):
#     # Extracts the text from the JSON body sent by the AI Edge Gallery
#     message = chat_data.message
#     session_id = str(uuid.uuid4())

#     try:
#         user_lang = detect(message)
#     except:
#         user_lang = "en"

#     query_vector = dense_embedder.embed_query(message)

#     # Qdrant Cache Logic
#     cache_results = qdrant_client.query_points(
#         collection_name=CACHE_COLLECTION_NAME,
#         query=query_vector,
#         limit=1,
#         query_filter=Filter(
#             must=[FieldCondition(key="language", match=MatchValue(value=user_lang))]
#         )
#     ).points

#     if cache_results and cache_results[0].score > 0.90:
#         print(f"✅ CACHE HIT! Match: {cache_results[0].score:.2f}")
#         return cache_results[0].payload["answer"]

#     # LangGraph Execution
#     print(f"❌ CACHE MISS. Processing: {message}")
#     graph = compile_all()
#     config = {"configurable": {"thread_id": session_id}}

#     try:
#         result = await graph.ainvoke(
#             {"messages": [{"role": "user", "content": message}], "session_id": session_id},
#             config
#         )
#         final_answer = result["messages"][-1].content
#     except Exception as e:
#         print(f"Error in LangGraph: {e}")
#         return f"Processor Error: {str(e)}"

#     # Save to Qdrant Cache
#     if "I am an AI assistant specialized" not in final_answer:
#         qdrant_client.upsert(
#             collection_name=CACHE_COLLECTION_NAME,
#             points=[
#                 PointStruct(
#                     id=str(uuid.uuid4()),
#                     vector=query_vector,
#                     payload={"question": message, "answer": final_answer, "language": user_lang}
#                 )
#             ]
#         )

#     return final_answer

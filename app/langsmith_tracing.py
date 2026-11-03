"""LangSmith tracing for latency, tokens, and OpenRouter cost per chat run.

Opt-in: set LANGSMITH_API_KEY and leave LANGSMITH_TRACING unset (auto-on) or
explicitly true. Explicit false / missing key keeps tracing off. Never logs the
API key. Question text is sent to LangSmith as run inputs unless
LANGSMITH_HIDE_INPUTS=true.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator


_DEFAULT_PROJECT = "govt-rag"
_setup_done = False
_enabled = False


def langsmith_enabled() -> bool:
    return _enabled


def setup_langsmith() -> bool:
    """Configure env for LangSmith after dotenv. Safe to call more than once."""
    global _setup_done, _enabled
    if _setup_done:
        return _enabled
    _setup_done = True

    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    explicit = (
        os.getenv("LANGSMITH_TRACING")
        or os.getenv("LANGCHAIN_TRACING_V2")
        or ""
    ).strip().lower()

    if explicit in {"0", "false", "no", "off"}:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        _enabled = False
        print("langsmith: tracing off")
        return False

    if not api_key:
        if explicit in {"1", "true", "yes", "on"}:
            print(
                "langsmith: LANGSMITH_TRACING is on but LANGSMITH_API_KEY is missing; "
                "tracing disabled"
            )
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        _enabled = False
        return False

    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or _DEFAULT_PROJECT
    )
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ.setdefault("LANGCHAIN_PROJECT", project)

    # Cloud Run can finish the response before background uploads flush.
    if os.getenv("K_SERVICE") and os.getenv("LANGCHAIN_CALLBACKS_BACKGROUND") is None:
        os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"

    endpoint = (os.getenv("LANGSMITH_ENDPOINT") or "").strip()
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint

    _enabled = True
    workspace = (os.getenv("LANGSMITH_WORKSPACE_ID") or "").strip()
    hide = (os.getenv("LANGSMITH_HIDE_INPUTS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    print(
        f"langsmith: tracing on project={project}"
        + (f" workspace={workspace}" if workspace else "")
        + (" hide_inputs" if hide else "")
    )
    return True


def _truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def base_metadata() -> dict[str, Any]:
    """Stable fields attached to every chat / graph run."""
    from vectorDB.rerankers import selected_name

    meta: dict[str, Any] = {
        "service": "manual-ai-rag",
        "git_sha": os.getenv("GIT_SHA") or "dev",
        "revision": os.getenv("K_REVISION") or "local",
        "reranker": selected_name(),
        "llm_model": os.getenv("OPENROUTER_LLM_MODEL") or "google/gemini-2.5-flash",
        "embedding_model": os.getenv("OPENROUTER_EMBEDDING_MODEL")
        or "qwen/qwen3-embedding-8b",
        "doc_version": os.getenv("DOC_VERSION") or "",
        "ls_provider": "openrouter",
        "ls_model_name": os.getenv("OPENROUTER_LLM_MODEL") or "google/gemini-2.5-flash",
    }
    return meta


def graph_run_config(
    *,
    session_id: str,
    language: str,
    request_id: str | None = None,
    cached: bool = False,
    stream: bool = False,
    extra_metadata: dict[str, Any] | None = None,
    extra_tags: list[str] | None = None,
) -> dict[str, Any]:
    """RunnableConfig for graph.ainvoke / astream_events (tags + cost metadata)."""
    tags = [
        "govt-rag",
        f"lang:{language or 'en'}",
        f"reranker:{(os.getenv('RERANKER') or 'minilm').strip().lower()}",
        "stream" if stream else "sync",
        "cache_hit" if cached else "cache_miss",
    ]
    if extra_tags:
        tags.extend(extra_tags)

    metadata = base_metadata()
    metadata.update(
        {
            "session_id": session_id,
            "answer_language": language,
            "cache_hit": cached,
            "stream": stream,
        }
    )
    if request_id:
        metadata["request_id"] = request_id
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "run_name": "govt-rag-chat",
        "tags": tags,
        "metadata": metadata,
        "configurable": {"thread_id": session_id},
    }


@contextmanager
def chat_tracing(
    *,
    session_id: str,
    language: str,
    request_id: str | None = None,
    cached: bool = False,
    stream: bool = False,
    extra_tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Root LangSmith run for one /chat request. Yields the graph config."""
    config = graph_run_config(
        session_id=session_id,
        language=language,
        request_id=request_id,
        cached=cached,
        stream=stream,
        extra_tags=extra_tags,
        extra_metadata=extra_metadata,
    )
    if not _enabled:
        yield config
        return

    from langsmith import trace, tracing_context

    project = os.getenv("LANGSMITH_PROJECT") or _DEFAULT_PROJECT
    ctx_kwargs: dict[str, Any] = {
        "enabled": True,
        "project_name": project,
        "tags": list(config["tags"]),
        "metadata": dict(config["metadata"]),
    }
    if _truthy("LANGSMITH_HIDE_INPUTS"):
        ctx_kwargs["hide_inputs"] = True
        ctx_kwargs["hide_outputs"] = _truthy("LANGSMITH_HIDE_OUTPUTS")

    run_name = "chat.cache_hit" if cached else "chat.miss"
    with tracing_context(**ctx_kwargs):
        with trace(
            name=run_name,
            run_type="chain",
            inputs={"cached": cached, "stream": stream, "language": language},
            metadata=dict(config["metadata"]),
            tags=list(config["tags"]),
            project_name=project,
        ):
            yield config


def reset_langsmith_for_tests() -> None:
    """Allow tests to re-run setup_langsmith after mutating env."""
    global _setup_done, _enabled
    _setup_done = False
    _enabled = False


def record_llm_usage(response, model: str, *, role: str = "chat") -> dict[str, Any]:
    """Attach token + cost usage to the current LangSmith run (OpenRouter)."""
    usage = getattr(response, "usage_metadata", None) or {}
    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or {}
    # OpenRouter sometimes nests native usage under response_metadata.
    native = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}

    input_tokens = int(
        usage.get("input_tokens")
        or token_usage.get("prompt_tokens")
        or native.get("prompt_tokens")
        or 0
    )
    output_tokens = int(
        usage.get("output_tokens")
        or token_usage.get("completion_tokens")
        or native.get("completion_tokens")
        or 0
    )
    total_tokens = int(
        usage.get("total_tokens")
        or token_usage.get("total_tokens")
        or native.get("total_tokens")
        or (input_tokens + output_tokens)
    )

    cost = usage.get("total_cost")
    if cost is None:
        cost = native.get("cost")
    if cost is None:
        cost = token_usage.get("cost")
    try:
        total_cost = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        total_cost = None

    payload: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if total_cost is not None:
        payload["total_cost"] = total_cost

    if not _enabled:
        return payload

    try:
        from langsmith import get_current_run_tree

        run = get_current_run_tree()
        if run is None:
            return payload
        run.add_metadata(
            {
                "ls_provider": "openrouter",
                "ls_model_name": model,
                "openrouter_role": role,
            }
        )
        run.set(usage_metadata=payload)
    except Exception as exc:
        print(f"langsmith: usage annotate skipped ({type(exc).__name__})")
    return payload


def record_embedding_usage(
    *,
    model: str,
    prompt_tokens: int = 0,
    total_cost: float | None = None,
) -> None:
    if not _enabled:
        return
    try:
        from langsmith import get_current_run_tree

        run = get_current_run_tree()
        if run is None:
            return
        usage: dict[str, Any] = {
            "input_tokens": int(prompt_tokens),
            "output_tokens": 0,
            "total_tokens": int(prompt_tokens),
        }
        if total_cost is not None:
            usage["total_cost"] = float(total_cost)
        run.add_metadata(
            {
                "ls_provider": "openrouter",
                "ls_model_name": model,
                "openrouter_role": "embedding",
            }
        )
        run.set(usage_metadata=usage)
    except Exception as exc:
        print(f"langsmith: embed usage skipped ({type(exc).__name__})")


def traceable_if_available(**kwargs):
    """@traceable when langsmith is importable; otherwise a no-op decorator.

    Applied at import time. When tracing is off, LangSmith's decorator is a
    cheap no-op at call time.
    """

    def deco(fn):
        try:
            from langsmith import traceable

            return traceable(**kwargs)(fn)
        except Exception:
            return fn

    return deco


def tracing_or_null(**kwargs):
    """Return tracing_context when enabled, else a nullcontext."""
    if not _enabled:
        return nullcontext()
    from langsmith import tracing_context

    return tracing_context(**kwargs)

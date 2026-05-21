import os
import sys

from dotenv import load_dotenv

# Add the parent folder to Python's path so we can import vectorDB
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, MessagesState, StateGraph

from vectorDB.search import run_hybrid_search

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    # streaming=True removed — with Pydantic v2, ainvoke() on a streaming LLM
    # returns AIMessageChunk instead of AIMessage. LangGraph's add_messages
    # reducer then tries to merge it using positional args (AIMessage(chunk.content))
    # which Pydantic v2 rejects with: "BaseModel.__init__() takes 1 positional argument".
    # stream_mode="updates" on graph.astream() handles response delivery instead.
)


class RagState(MessagesState):
    session_id: str
    context: str
    search_query: str  # Rewritten, search-optimised query set by query_rewriter_node


# ── Node 1: Query Rewriter ─────────────────────────────────────────────────────
async def query_rewriter_node(state: RagState):
    """
    Rewrites the user's raw, conversational question into a concise,
    keyword-rich search query that aligns better with how government
    documents are worded.  This single step significantly improves recall.
    """
    user_query = state["messages"][-1].content

    prompt = f"""You are a search-query optimizer for a UP Government document database.
The database contains PDFs about government portals, departments, schemes, FAQs,
canal/irrigation policy, IT systems, establishment rules, and legal acts.

Rewrite the user's question into a short, keyword-dense search query (max 20 words)
that will retrieve the most relevant document chunks from the database.

Rules:
- Keep specific names (portal names, department names, act names, scheme names) exactly as given.
- Remove filler words like "can you tell me", "what is", "please explain", etc.
- Focus on the core information need — nouns and domain terms matter most.
- Output ONLY the rewritten query. No explanation. No punctuation at the end.
- IMPORTANT: You MUST write the search query in the SAME LANGUAGE as the user's question. If the user asks in Hindi, output the rewritten query in Hindi.

User Question: {user_query}
Optimized Search Query:"""

    response = await llm.ainvoke(prompt)
    search_query = response.content.strip()
    print(f"🔍 Rewritten: '{user_query}' → '{search_query}'")
    return {"search_query": search_query}


# ── Node 2: Retriever ──────────────────────────────────────────────────────────
def retriever_node(state: RagState):
    # Use the rewritten query; fall back to the raw message if missing
    query = state.get("search_query") or state["messages"][-1].content

    context = run_hybrid_search(query)

    # GUARDRAIL: empty results or very low cross-encoder score → off-topic query
    if not context or context[0].score < -3.0:
        return {"context": "REJECT"}

    context_text = "\n\n".join(
        [
            f"Document: {r.payload['document-id']}\n"
            f"Page: {r.payload['page']}\n"
            f"Text: {r.payload['text']}"
            for r in context
        ]
    )
    return {"context": context_text}


# ── Node 3: Generator ──────────────────────────────────────────────────────────
async def generator_node(state: RagState):
    # GUARDRAIL CATCH: skip the LLM entirely for off-topic queries
    if state["context"] == "REJECT":
        return {
            "messages": [
                AIMessage(
                    content="I am an AI assistant specialized in UP Government Data. "
                    "I cannot answer queries about unrelated topics."
                )
            ]
        }

    prompt = f"""You are a helpful assistant. Answer the user's question strictly based
on the context provided below. Do not answer anything outside of the given context.
If the context does not contain enough information to answer, say so honestly.

Context:
{state["context"]}

Question: {state["messages"][-1].content}
Answer:"""

    response = await llm.ainvoke(prompt)
    return {"messages": [response]}


# ── Graph Assembly ─────────────────────────────────────────────────────────────
def compile_all():
    graph = StateGraph(RagState)

    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)

    # Flow: rewrite → retrieve (with re-ranking inside) → generate
    graph.add_edge(START, "query_rewriter")
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "generator")

    return graph.compile()

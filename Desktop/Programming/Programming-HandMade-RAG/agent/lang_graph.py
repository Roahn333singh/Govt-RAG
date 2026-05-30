import os
import sys

from dotenv import load_dotenv

# Add the parent folder to Python's path so we can import vectorDB
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from vectorDB.search import run_hybrid_search

load_dotenv(override=True)


_llm_cache = {}

def get_llm(temperature=0.7):
    global _llm_cache
    if temperature not in _llm_cache:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm_cache[temperature] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )
    return _llm_cache[temperature]


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

The documents in the database are written in a mix of Devanagari Hindi and English.
Some chapters use Devanagari terms (e.g., "निविदा समिति", "करोड़"), while others use English terms (e.g., "tender committee", "crore").

Optimize the user's question into a bilingual, keyword-dense search query (max 20 words) for retrieving the most relevant document chunks.

Rules:
1. Strip all conversational filler (e.g., "can you tell me", "what is", "how to").
2. Translate and pair key technical terms and concepts into BOTH proper English and Devanagari Hindi (e.g., "tender निविदा", "committee समिति", "bid capacity निविदा क्षमता").
3. For monetary amounts and numbers, include both forms (e.g. "10 crore 10 करोड़", "50 lakh 50 लाख", "10000").
4. Output ONLY the optimized bilingual search query, separated by spaces. No explanation, no punctuation.

User Question: {user_query}
Optimized Search Query:"""

    response = await get_llm(temperature=0.0).ainvoke(prompt)
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

    prompt = f"""You are a helpful assistant for UP Government Data. 
Answer the user's question primarily based on the context provided below. 
If the user asks for the definition of a term (like 'Arbitration' or 'Tender') that appears in the context, you may use your general knowledge to define it, but you MUST then explain how it is used within the provided context.
If the question is completely unrelated to the context, say that you don't have enough information.

Context:
{state["context"]}

Question: {state["messages"][-1].content}
Answer:"""

    response = await get_llm(temperature=0.1).ainvoke(prompt)
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

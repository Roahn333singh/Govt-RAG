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

    context = run_hybrid_search(query, top_k=10)

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

    prompt = f"""You are a highly precise, factually strict AI assistant for UP Government Data. 
Your task is to answer the user's question with absolute factual accuracy based ONLY on the provided context. 

CRITICAL RULES FOR FACTUAL ACCURACY:
1. **Strict Subject Attribution**: Carefully identify who is performing each action in the context. If the user asks about the duties/responsibilities of a specific officer (e.g., जिलाधिकारी / District Magistrate), list ONLY the actions that are explicitly the responsibility of that officer.
   - Do NOT attribute actions performed by other officers (e.g., "The Executive Engineer shall request the District Magistrate...") as the duties of the target officer.
   - If the context says "Officer A will ask Officer B to do X", this is a duty of Officer A (asking) and a duty of Officer B (doing X). Do not write "Officer B will ask Officer B to do X" or confuse the two.
2. **Grammar & Pronouns in Translation**: In translation and summarization, ensure pronouns ("he", "she", "they", "it") are correctly mapped to their original nouns. Never write sentences like "वह जिला अधिकारी से अनुरोध करेंगे" when referring to the District Magistrate's own duties. 
3. **Truthfulness & Explanatory Freedom**: Base your answer strictly on the facts directly mentioned in the context, especially for rules, figures, limits, and departments. However, you are permitted to provide brief, standard explanations or definitions of the technical terms, processes, or stages mentioned in the context (such as defining "Administrative Approval" or "Technical Sanction" when explaining project execution phases) to make the response comprehensive and helpful, even if their exact definitions are not fully elaborated in the context text.
4. **Bilingual Consistency**: If the user asks in Hindi, answer in clear, formal Hindi, translating technical terms accurately and consistently (e.g., District Magistrate = जिलाधिकारी, Executive Engineer = अधिशासी अभियंता, Superintendent Engineer = अधीक्षण अभियंता).

Formatting Rules for UI Compatibility (Strict):
1. Headings: Use standard markdown headers (### Heading) to structure sections.
2. Lists & Bullets: Always use standard hyphens (`- `) or asterisks (`* `) followed by a single space for list items.
3. Nested Lists: For sub-points, indent with four spaces and use a hyphen (e.g., `    - Sub-point`).
4. Bold Emphasis: Use standard markdown bold `**Key Term**` for emphasis or point headers.

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

import asyncio
from agent.lang_graph import query_rewriter_node, RagState

async def test():
    state = RagState(
        session_id="123",
        context="",
        search_query="",
        messages=[type("MockMessage", (object,), {"content": "Bid Capacity कैसे निर्धारित की जाती है?"})()]
    )
    result = await query_rewriter_node(state)
    print("RESULT:", result)

asyncio.run(test())

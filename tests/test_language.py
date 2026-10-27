from langchain_core.messages import AIMessage, HumanMessage

from agent.guards import detect_answer_language

E8 = (
    "Kya koi Sub-Divisional Officer ya Assistant Engineer (AE) directly "
    "tender accept kar sakta hai, aur iski limit kya hai?"
)


def test_devanagari_is_hindi():
    assert detect_answer_language("सामान्य भविष्य निधि के नियम क्या हैं?") == "hi"


def test_english_stays_english():
    assert detect_answer_language("What is the Pragati portal used for?") == "en"


def test_a_quoted_hindi_term_does_not_flip_english():
    assert detect_answer_language("What does अनुच्छेद 310 say about tenure?") == "en"


def test_e8_hinglish_is_answered_in_english():
    assert detect_answer_language(E8) == "en"


def test_roman_hindi_function_words_can_still_select_hindi():
    assert detect_answer_language("kya hai yeh aur kaise kaam karta hai yeh") == "hi"


async def test_generator_states_the_answer_language(monkeypatch):
    from agent import lang_graph

    seen = {}

    class Capture:
        async def ainvoke(self, prompt):
            seen["prompt"] = prompt
            return AIMessage(content="Answer. Sources: Doc.pdf, Page: 1")

    monkeypatch.setattr(lang_graph, "get_llm", lambda temperature=0.7: Capture())
    await lang_graph.generator_node(
        {
            "messages": [HumanMessage(content=E8)],
            "context": "Document: Nivida.pdf\nPage: 3\nText: AE cannot accept.",
            "answer_language": "en",
            "sources": [],
        }
    )
    assert "Answer in English. Quote Hindi terms only in brackets." in seen["prompt"]

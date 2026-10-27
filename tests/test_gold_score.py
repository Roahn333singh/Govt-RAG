import json

from eval.run_gold import load_items, score_item


def test_gold_file_covers_the_domains():
    items = load_items(None)
    assert len(items) >= 180
    domains = {i["domain"] for i in items}
    for name in ("portals", "establishment", "financial", "nivida", "accounts", "canals", "nicd", "rti", "out_of_scope", "injection"):
        assert name in domains
    assert sum(1 for i in items if i["core"]) >= 40
    assert all({"id", "question", "lang", "domain", "expected_keywords", "expected_numbers", "must_refuse", "core"} <= set(i) for i in items)


def test_scorer_labels():
    item = {"lang": "en", "must_refuse": False, "expected_keywords": ["Pragati", "portal"], "expected_numbers": ["500"]}
    assert score_item(item, "Pragati portal allows 500 KB.", []) == "correct"
    assert score_item(item, "Something unrelated.", []) == "wrong"
    assert score_item(item, "I am an AI assistant specialized in UP Government Data.", []) == "false_refusal"
    refused = {"must_refuse": True, "lang": "en", "expected_keywords": [], "expected_numbers": []}
    assert score_item(refused, "I can only answer questions about UP Government manuals.", []) == "refused"
    assert score_item(refused, "BANANA", []) == "wrong"
    hindi = {"lang": "hi", "must_refuse": False, "expected_keywords": ["निविदा"], "expected_numbers": []}
    assert score_item(hindi, "The tender rules apply.", []) == "language_mismatch"

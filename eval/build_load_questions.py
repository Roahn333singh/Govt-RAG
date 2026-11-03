"""Build eval/load_questions.jsonl for load_test.py.

Takes every answerable golden question (prod50 + the rest, minus adversarial),
rewords it so the text differs from golden/prod50 phrasing, and splits items
into two disjoint halves: A (reranker run) and B (no-reranker run). The same
gold fields are kept so prod50_test.score() works unchanged.

  python3 eval/build_load_questions.py
"""
import csv, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EN = ["As per the manual, {q}", "Quick check: {q}", "Can you tell me this. {q}",
      "I need this for my office work. {q}", "Please confirm from the rules: {q}",
      "According to the government guidelines, {q}"]
HI = ["नियमों के अनुसार बताइए: {q}", "कृपया स्पष्ट करें: {q}", "मुझे यह जानना है। {q}",
      "मैनुअल के हिसाब से, {q}"]


def reword(q, lang, i):
    t = (HI if lang == "hi" else EN)[i % (len(HI) if lang == "hi" else len(EN))]
    if lang != "hi" and "{q}" in t and t.endswith(", {q}"):
        q = q[0].lower() + q[1:]
    return t.format(q=q)


def main():
    rows = list(csv.DictReader(open(ROOT / "eval/golden200.csv", encoding="utf-8-sig")))
    rows = [r for r in rows if r["answerable"] == "True" and r["type"] != "adversarial"]
    rng = random.Random(42)
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        try:
            facts = json.loads(r["must_include_facts"] or "[]")
        except ValueError:
            facts = []
        out.append({"id": r["id"], "half": "A" if i % 2 == 0 else "B",
                    "question": reword(r["question"].strip(), r["language"], i),
                    "golden_question": r["question"], "language": r["language"],
                    "type": r["type"], "source_doc": r["source_doc"],
                    "source_page_or_section": r["source_page_or_section"],
                    "evidence_quote": r["evidence_quote"], "must_include_facts": facts})
    p = ROOT / "eval/load_questions.jsonl"
    p.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    for h in "AB":
        hs = [o for o in out if o["half"] == h]
        print(h, len(hs), "en", sum(o["language"] == "en" for o in hs), "hi", sum(o["language"] == "hi" for o in hs))
    print("wrote", p, len(out))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Patch Qdrant chunks flagged by the 25 Oct 2026 retest report.

  - IGRS FAQ digit meanings (garbled OCR → clean Hindi + English)
  - AE tender limit: surface GO 24 Apr 1996 ₹10,000 next to old ₹7,500
  - Disciplinary PPT: deemed suspension + grounds (Kruti Dev → Unicode)

Usage:
  .venv/bin/python scripts/patch_retest_fixes.py --dry-run
  .venv/bin/python scripts/patch_retest_fixes.py --clear-cache
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv(override=True)

from qdrant_client.models import PointStruct, SparseVector

from patch_point_text import COLLECTION, DENSE_NAME, SPARSE_NAME, clear_cache, get_client

IGRS_DIGITS = """मुख्यमंत्री सन्दर्भ और ऑनलाइन सन्दर्भ को सन्दर्भ संख्या से कैसे पहचानें?

14-अंकीय IGRS (समाधान / Jan-Sunwai) सन्दर्भ संख्या के प्रथम अंक से स्रोत जाना जाता है:
‘1’ = मुख्यमंत्री सन्दर्भ (Chief Minister reference)
‘2’ = जिलाधिकारी सन्दर्भ
‘3’ = तहसील दिवस सन्दर्भ
‘4’ = ऑनलाइन सन्दर्भ (Online reference)
‘5’ = CSC/लोकवाणी सन्दर्भ
‘6’ = PG पोर्टल सन्दर्भ

उदाहरण: 11166160021769 मुख्यमंत्री सन्दर्भ है और 40018316000393 ऑनलाइन सन्दर्भ है।

‘आख्या’ (Aakhya) श्रेणी क्या है?
‘आख्या’ का अर्थ है कि प्रेषक अधिकारी ने किसी सन्दर्भ के सम्बन्ध में उपयोगकर्ता से रिपोर्ट / स्पष्टीकरण (report or explanation) माँगा है। मुख्यमंत्री सन्दर्भों में आख्या का आशय सक्षम स्तर से हस्ताक्षरित व स्कैन रिपोर्ट होता है।
"""

DEEMED_SUSPENSION = """Deemed suspension (निलंबित समझा जाना) — UP Government Servants (Discipline and Appeal) Rules:

(3) यदि कोई सरकारी सेवक 48 घंटे से अधिक अवधि के लिये अभिरक्षा (Custody) में निरुद्ध (Detain) रहा हो (चाहे आपराधिक आरोप में अथवा अन्यथा), तो वह ऐसे निरोध की दिनांक से निलंबन पर समझा जायेगा (deemed to be suspended)। अभिरक्षा से मुक्त होने पर कर्मचारी सक्षम अधिकारी को इसकी सूचना देगा और निलंबन के विरुद्ध अभ्यावेदन दे सकेगा। सक्षम अधिकारी विचारोपरान्त निलंबन को जारी रख सकता है या समाप्त कर सकता है।

(4) यदि कर्मचारी न्यायालय द्वारा किसी आरोप में दोषी पाया गया हो और इसके फलस्वरूप 48 घंटे से अधिक के कारावास की सजा दी गई हो और यदि उसे न्यायालय के निर्णय के आधार पर dismissal या removal का दण्ड नहीं दिया गया हो, तो वह सजा की दिनांक से ही निलंबित समझा जायेगा।

English: A government servant is deemed to be suspended when detained in custody for a period exceeding 48 hours, or when convicted and sentenced to imprisonment exceeding 48 hours (unless already dismissed or removed from service).
"""

DISCIPLINE_GROUNDS = """Primary grounds / reasons for initiating disciplinary action against a government servant
(UP Government Servants Discipline and Appeal Rules / establishment practice):

अनुशासनिक कार्यवाही शुरू करने के प्रमुख आधार:
- Misconduct / अवचार
- Negligence of duty / कर्तव्य में लापरवाही
- Corruption / भ्रष्टाचार
- Misappropriation of government funds / सरकारी धन का गबन या दुर्विनियोग
- Insubordination / अवज्ञा
- Conduct unbecoming of a government servant / सरकारी सेवक के अनुपयुक्त आचरण
- Grave misconduct or causing financial loss to Government (also relevant under CSR Article 351-A for pension action)

Misconduct facts are framed as charges in a charge-sheet (आरोप पत्र) under the disciplinary rules.
"""

# point_id → either full replacement text, or (old_substr, new_substr)
FULL_REPLACEMENTS: dict[str, str] = {
    "24a76122-7855-4562-b276-8a7d718c3d29": IGRS_DIGITS,
    "68d7a538-90b4-4290-a31a-bb799535a95a": DEEMED_SUSPENSION,
    "bf26ef24-????": DISCIPLINE_GROUNDS,  # resolved at runtime
}

AE_OLD = (
    "They may also delegate to Sub-Divisional Officers or Assistant Engineer "
    "selected by name the power to accept tender not exceeding Rs. 7,500/-."
)
AE_NEW = (
    "They may also delegate to Sub-Divisional Officers or Assistant Engineer "
    "selected by name the power to accept tender not exceeding Rs. 7,500/- "
    "(NOTE: GO No. 2-717/दस-96-17141/85 dated 24 April 1996 raised this "
    "delegated AE/SDO tender-acceptance limit from Rs. 7,500/- to Rs. 10,000/-; "
    "for work orders above Rs. 10,000/- the AE/SDO needs prior approval of the "
    "Divisional Officer)."
)

AE_POINT_IDS = [
    "09345c7a-57e3-4ca1-b688-6ed3864e2cec",
    "188bb936-32bb-42f0-b713-7dafa216516f",
    "1b782dac-b04b-4577-ad17-58398b7613bd",
    "ae0038f4-332b-4675-ae28-94aef04f860f",
]


def upsert_text(client, dense, sparse, point_id: str, new_text: str, dry_run: bool) -> bool:
    records = client.retrieve(
        collection_name=COLLECTION,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )
    if not records:
        print(f"❌ missing {point_id}")
        return False
    point = records[0]
    payload = dict(point.payload or {})
    print(
        f"── {point_id} doc={payload.get('document-id')} page={payload.get('page')} "
        f"chars {len(payload.get('text') or '')}→{len(new_text)}"
    )
    if dry_run:
        print("   ", new_text[:180].replace("\n", " "))
        return True
    payload["text"] = new_text
    if hasattr(dense, "embed_documents"):
        dense_vec = dense.embed_documents([new_text])[0]
    else:
        dense_vec = dense.embed_query(new_text)
    sp = list(sparse.embed([new_text]))[0]
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point.id,
                payload=payload,
                vector={
                    DENSE_NAME: dense_vec,
                    SPARSE_NAME: SparseVector(
                        indices=list(sp.indices), values=list(sp.values)
                    ),
                },
            )
        ],
    )
    print("   ✅ upserted")
    return True


def resolve_prefix(client, prefix: str) -> str | None:
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for p in points:
            if str(p.id).startswith(prefix):
                return str(p.id)
        if offset is None:
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    client = get_client()
    dense = sparse = None
    if not args.dry_run:
        from vectorDB import search

        search.init_services()
        dense = search.dense_embedder
        sparse = search.sparse_embedder

    ok = 0

    grounds_id = resolve_prefix(client, "bf26ef24")
    full = {
        "24a76122-7855-4562-b276-8a7d718c3d29": IGRS_DIGITS,
        "68d7a538-90b4-4290-a31a-bb799535a95a": DEEMED_SUSPENSION,
    }
    if grounds_id:
        full[grounds_id] = DISCIPLINE_GROUNDS
    else:
        print("⚠️  grounds point bf26ef24… not found")

    for pid, text in full.items():
        if upsert_text(client, dense, sparse, pid, text, args.dry_run):
            ok += 1

    for pid in AE_POINT_IDS:
        records = client.retrieve(
            collection_name=COLLECTION, ids=[pid], with_payload=True, with_vectors=False
        )
        if not records:
            print(f"❌ missing AE point {pid}")
            continue
        text = (records[0].payload or {}).get("text") or ""
        if AE_OLD not in text:
            if "Rs. 10,000/-" in text and "24 April 1996" in text:
                print(f"⏭️  already patched {pid}")
                continue
            print(f"⚠️  AE old string missing in {pid}")
            continue
        new_text = text.replace(AE_OLD, AE_NEW)
        if upsert_text(client, dense, sparse, pid, new_text, args.dry_run):
            ok += 1

    print(f"\nDone: {ok} patches {'previewed' if args.dry_run else 'applied'}")
    if args.clear_cache and not args.dry_run and ok:
        clear_cache(client)


if __name__ == "__main__":
    main()

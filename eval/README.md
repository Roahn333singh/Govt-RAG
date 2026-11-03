# Golden-set evaluation

This scores the service that is already deployed. It does not change retrieval, prompts, or caching in the running app.

`source_doc` and `evidence_quote` are grader-only. They are never sent to `/chat` or to the judge.

## Sets

- `eval/golden200.csv` is the full set (`--set full`).
- `eval/review_sample25.csv` is the 25-row subset (`--set sample25`).

Quotes and source documents that belong together are joined with ` || `. Each part has to appear in a chunk from one of the listed source documents.

## Commands

Use the project virtualenv.

Retrieval only (in-process, production rewriter / hybrid search / RRF merge / one rerank). The semantic cache is not used, because this path never calls the cache lookup.

```bash
.venv/bin/python eval/run_eval.py --retrieval-only --set sample25
```

Answers against a no-traffic revision. The key is `RAG_EVAL_API_KEY` (sent as `X-API-Key` and not logged). Each item gets a new `session_id`. If `RAG_EVAL_ADMIN_TOKEN` is set, the exact question is purged from the semantic cache first. The in-memory exact cache on an instance still lasts five minutes, so a second pass of the same questions on a warm instance can be a cache hit.

```bash
export RAG_EVAL_API_KEY=...
export OPENROUTER_API_KEY=...
export RAG_EVAL_ADMIN_TOKEN=...   # optional
.venv/bin/python eval/run_eval.py \
  --answer-only --set sample25 \
  --url https://candidate---manual-ai-rag-nszha5soya-ey.a.run.app \
  --max-cost 0.30
```

Both, full retrieval and a 25-row answer sample, then fail on a regression:

```bash
.venv/bin/python eval/run_eval.py \
  --retrieval-set full --answer-set sample25 \
  --url "$EVAL_BASE_URL" --gate
```

Write the current numbers into `eval/baseline.json`:

```bash
.venv/bin/python eval/run_eval.py --retrieval-only --set full --update-baseline
```

Reports land in `eval/results/<timestamp>_<git sha>/` as `items.jsonl`, `summary.json`, and `report.md`.

Compare rerankers on one shared pool of 20 chunks per item (retrieval runs once; each backend only reranks that pool):

```bash
.venv/bin/python eval/run_eval.py --reranker all --set sample25
```

`--reranker` is `minilm`, `mmarco`, `gte`, `none`, or `all`. The report is `eval/results/<timestamp>_rerank_compare/compare.md`. A suggested threshold is printed for each backend and is not applied. `none` skips the reject floor. Each backend appends its own row to `eval/history.csv`.

## Cloud Storage

A finished `eval/run_eval.py` run uploads to `gs://govt-rag-evals` (project `project-0ff01dde-a087-4c96-b04`, `europe-west3`). The bucket is not created by the script. If it is missing, the upload prints a warning and the eval result stands. `--no-upload` skips the upload. A pull request skips it even without that flag.

```bash
python eval/run_eval.py --set sample25
```

That command runs the sample and uploads. The monitor reads:

- `runs/<run_id>/summary.json` — the eval summary plus `run`, `metrics`, `stages`, `types`, `thresholds`, `gate`, and `deltas`
- `runs/<run_id>/report.md` and, when a comparison was run, `compare.md`
- `runs/<run_id>/failing_items.json` — items that missed the top 8 or were wrongly refused
- `history.csv` and `history.json`
- `latest.json`, written last, with `{run_id, timestamp_ist, path}`

Unmeasured metrics are `null`. `deltas.recall_at_8` is the change versus `eval/baseline.json` in points, or `null` when there is no baseline.

## Cost

The judge is `google/gemini-2.5-flash` at temperature 0 (`--judge-model` or `EVAL_JUDGE_MODEL`). Results are cached in `eval/judge_cache.jsonl`, keyed by a hash of the item, the answer, the model, and the prompt version. A repeat with the same answer does not call the judge again.

Before any judge call the script prints an estimate (about $0.002 per uncached item at the list rates coded in `eval/answer_eval.py`) and exits if that estimate is over `--max-cost` (default `$0.30`). A cold full-set judge run is above that cap; raise `--max-cost` when you mean to spend it. `sample25` fits under the default.

## Gate

`eval/thresholds.yaml`:

- recall@8 drop of more than 3 points versus `eval/baseline.json`
- faithfulness below 0.90
- false refusal above 5%
- correct refusal below 90%
- p95 latency above 15 seconds
- error rate above 2%

A metric that this run did not measure is skipped, not failed.

## What retrieval records

For every answerable item, four stages:

1. Qdrant RRF candidates for each phrasing (raw question, English rewrite, Hindi rewrite)
2. The merged pool (`RERANK_POOL`, 20)
3. The reranked top 8 (`_TOP_K`)
4. The chunks still left after `REJECT_THRESHOLD`

Hit, recall, MRR, and source-document hit are computed per stage. The report also counts items whose quote was in the pool but not in the reranked 8, and items the reject threshold then removed. Every item, including unanswerable ones, logs its top rerank score.

## Run history

Every finished run appends one row to `eval/history.csv` (created with a header on first write). Old rows are copied through, not rewritten. Rates use four decimal places, latency is integer milliseconds, and a measure this run did not take is left blank.

The last column is `reranker` (`minilm` when a run did not record one). A file written before that column existed gets the header cell on the next append; its existing data rows stay as they are. A comparison run appends one row per backend. `latency_p50_ms` and `latency_p95_ms` stay the chat latencies, so a retrieval-only comparison leaves them blank. Rerank latency is in `compare.md`.

`RERANKER=minilm` is the code default when the variable is unset (FastEmbed `ms-marco-MiniLM-L-6-v2`). `deploy.sh` sets `RERANKER=gte` (`Alibaba-NLP/gte-multilingual-reranker-base`). `mmarco` and `gte` are ONNX models baked into the image. `colbert` late-interacts via Qdrant MaxSim on `COLBERT_COLLECTION` (`jinaai/jina-colbert-v2`, built by `scripts/index_colbert.py` into a new collection — never the live one). `COLBERT_REJECT_THRESHOLD` is the MaxSim floor (permissive until calibrated). `none` keeps RRF order and skips the reject threshold. `REJECT_THRESHOLD` overrides model-specific floors. A backend that fails to load stops startup.

Tag a ColBERT prod50 compare run (cache off) and upload for the dashboard:

```bash
export RERANKER=colbert COLBERT_COLLECTION=my-collection-colbert
export CACHE_SIM_THRESHOLD=1.1
# After calibrate_colbert_threshold.py:
# export COLBERT_REJECT_THRESHOLD=...
uv run python eval/run_eval.py --set prod50 \
  --variant colbert-jina-v2 \
  --compare-to gte=eval/results/<gte-run-dir> \
  --compare-to none=eval/results/<none-run-dir>
```

Time each backend on this machine (20 documents, 512 tokens, 2 threads, 30 runs). A peak RSS over 6 GiB or a p95 over 5 seconds is flagged:

```bash
.venv/bin/python scripts/bench_reranker.py
```

A mixed run (full retrieval, sample25 answers) records `eval_set` as the retrieval set, `full`.

`dataset_version` is the first 12 hex characters of the SHA-256 of `eval/golden200.csv`.

Question types in the CSV map onto history columns like this:

| golden200.csv `type` | history column |
|---|---|
| numeric | recall_at_8_numeric |
| single_fact | recall_at_8_single_fact |
| paraphrase_dense | recall_at_8_paraphrase |
| exact_term_bm25 | recall_at_8_exact_term |
| multi_hop | recall_at_8_multi_hop |
| procedural | recall_at_8_procedural |
| comparison | recall_at_8_comparison |

`unanswerable` and `adversarial` have no recall column. `target_url_kind` is `local` for no URL or localhost, `tagged` when the host contains `---`, and `prod` otherwise. `EVAL_URL_KIND` overrides that. `EVAL_REVISION` fills `cloud_run_revision`. `PR_NUMBER` fills `pr_number`.

Rebuild the file from saved reports:

```bash
.venv/bin/python eval/run_eval.py --backfill
```

`--backfill` replaces `history.csv`. A normal run only appends.

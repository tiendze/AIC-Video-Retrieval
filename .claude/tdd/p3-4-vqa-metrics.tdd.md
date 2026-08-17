# TDD evidence — P3.4 `eval/metrics/vqa.py`

**Source plan:** [`.claude/plans/p3-vqa-submission.plan.md`](../plans/p3-vqa-submission.plan.md) §P3.4 · **Trước:** [`p3-3-build-qa-eval-set.tdd.md`](./p3-3-build-qa-eval-set.tdd.md)

**Scope decision (confirmed with chủ dự án via `AskUserQuestion` before implementation, 2026-08-17):** P3.4 has two parts in the plan — (1) the pure scoring module `eval/metrics/vqa.py`, and (2) wiring it into `eval/run.py` against a live `QaService` run over the frozen `configs/eval/aic_qa_v1.json` (91 records) to produce the real 7-metric table. Part (2) requires either real Gemini API spend against all 91 records, or an offline predictions format not yet defined. Chủ dự án chose: **chỉ làm phần (1) trong lượt này** — pure module + full test coverage, following the exact same pattern as `eval/metrics/retrieval.py` and `eval/metrics/trake.py` (neither of which call their respective I/O layers, Milvus/TrakeService, from inside the metrics module). Wiring is left for a follow-up step.

## User journeys (from the plan / PRD §6.7)

- As the evaluator, I can compute `Grounded Accuracy` — the headline VQA metric — from a set of scored queries, and it is defined as text-correct AND evidence-in-segment, not either alone.
- As the evaluator, I can separate retrieval error from VLM reasoning error via `Retrieval-conditioned Accuracy`, computed only over queries where retrieval already found the right keyframe.
- As the evaluator, I can read `Hallucination Rate` alongside `Grounded Accuracy` — a confident answer given zero correct keyframes in context is flagged, independent of whether its text happens to be right.
- As the evaluator, I never get a silently misleading 0.0/100% when there was nothing to average (empty input raises `ValueError`, per CLAUDE.md §7 and every other metrics module in this repo).

## Task report

| Task | Summary | Command | Result |
|---|---|---|---|
| Scoring module | `eval/metrics/vqa.py` — `normalize_answer`, `exact_match`, `token_f1`, `VqaGroundTruthQuery`, `VqaContextKeyframe`, `VqaPrediction`, `VqaQueryResult`, `score_query`, and 6 aggregate functions (`exact_match_rate`, `mean_token_f1`, `grounded_accuracy`, `retrieval_conditioned_accuracy`, `hallucination_rate`, `abstention_rate`) + `mean_llm_judge_score` | `./.venv/Scripts/python.exe -m pytest tests/test_vqa_metrics.py --cov=eval.metrics.vqa --cov-report=term-missing -q` | 28 passed, 100% coverage |
| Regression | Nothing existing broken | `./.venv/Scripts/python.exe -m pytest tests/ -q` | 739 passed, 4 skipped, 35 deselected |

RED (before implementation existed):
```
ModuleNotFoundError: No module named 'eval.metrics.vqa'
ERROR tests/test_vqa_metrics.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!
```

GREEN (after implementation): `28 passed`, `eval/metrics/vqa.py` 100% line coverage; full suite `739 passed, 4 skipped, 35 deselected` — up from the pre-P3.4 baseline with zero regressions.

## Test specification

| # | What is guaranteed | Test | Result |
|---|---|---|---|
| 1 | Normalization lowercases + strips punctuation | `test_normalize_answer_lowercases_and_strips_punctuation` | PASS |
| 2 | English articles ("a"/"an"/"the") are dropped | `test_normalize_answer_removes_english_articles` | PASS |
| 3 | Vietnamese diacritics are preserved (NOT stripped — "bỏ dấu câu" means punctuation, not tone marks) | `test_normalize_answer_preserves_vietnamese_diacritics` | PASS |
| 4 | Exact match after normalization | `test_exact_match_true_after_normalization` | PASS |
| 5 | Exact match false for different words | `test_exact_match_false_for_different_words` | PASS |
| 6 | Token-F1 identical answers = 1.0 | `test_token_f1_identical_answers_scores_1` | PASS |
| 7 | Token-F1 disjoint answers = 0.0 | `test_token_f1_disjoint_answers_scores_0` | PASS |
| 8 | Token-F1 partial overlap is strictly between 0 and 1 | `test_token_f1_partial_overlap_scores_between_0_and_1` | PASS |
| 9 | Token-F1 empty prediction vs non-empty reference = 0.0 | `test_token_f1_empty_prediction_scores_0` | PASS |
| 10 | Token-F1 both empty = 1.0 (edge case, closes coverage) | `test_token_f1_both_empty_scores_1` | PASS |
| 11 | Correct answer + evidence inside the correct segment ⇒ `grounded_correct=True` | `test_score_query_correct_answer_with_grounded_evidence` | PASS |
| 12 | Correct answer TEXT but cited evidence outside the segment ⇒ `evidence_grounded=False`, `grounded_correct=False`, while `retrieval_hit` still `True` (context did contain a correct keyframe) — the "right answer, wrong citation" case the plan's Grounded Accuracy definition exists to catch | `test_score_query_correct_text_but_evidence_outside_segment_is_not_grounded` | PASS |
| 13 | Wrong answer text is never `grounded_correct`, even with perfectly-placed evidence | `test_score_query_wrong_answer_is_never_grounded_correct_even_with_right_evidence` | PASS |
| 14 | `retrieval_hit=False` when nothing in context touches the ground-truth segment | `test_score_query_retrieval_hit_false_when_context_never_touches_segment` | PASS |
| 15 | `confident` requires `status=="answered"` AND `confidence >= threshold` — abstention with high raw confidence and `unavailable` both score `confident=False` | `test_score_query_confident_requires_answered_status_and_threshold` | PASS |
| 16 | `status="unavailable"` (Gemini down) scores as a normal wrong answer, not a crash | `test_score_query_unavailable_status_scores_as_wrong_not_a_crash` | PASS |
| 17 | `exact_match_rate` averages over results | `test_exact_match_rate_averages_over_results` | PASS |
| 18 | `exact_match_rate` raises on empty input | `test_exact_match_rate_raises_on_empty_results` | PASS |
| 19 | `mean_token_f1` averages over results | `test_mean_token_f1_averages_over_results` | PASS |
| 20 | `grounded_accuracy` requires both `exact_match` AND `evidence_grounded` | `test_grounded_accuracy_requires_both_exact_match_and_evidence_grounded` | PASS |
| 21 | `retrieval_conditioned_accuracy` only counts queries with `retrieval_hit=True` | `test_retrieval_conditioned_accuracy_only_counts_retrieval_hits` | PASS |
| 22 | `retrieval_conditioned_accuracy` raises when no query ever had a retrieval hit (distinct from a real 0%) | `test_retrieval_conditioned_accuracy_raises_when_no_query_had_a_retrieval_hit` | PASS |
| 23 | `hallucination_rate` = `confident AND NOT retrieval_hit` | `test_hallucination_rate_is_confident_and_not_retrieval_hit` | PASS |
| 24 | `abstention_rate` counts `status=="insufficient_evidence"` | `test_abstention_rate_counts_insufficient_evidence_status` | PASS |
| 25 | `abstention_rate` raises on empty input | `test_abstention_rate_raises_on_empty_results` | PASS |
| 26 | `mean_llm_judge_score` averages 1-5 rubric scores | `test_mean_llm_judge_score_averages_scores` | PASS |
| 27 | `mean_llm_judge_score` raises on empty input | `test_mean_llm_judge_score_raises_on_empty` | PASS |
| 28 | `mean_llm_judge_score` raises on an out-of-range score (0 or 6) | `test_mean_llm_judge_score_raises_on_out_of_range_score` | PASS |

## Coverage and known gaps

```
Name                  Stmts   Miss  Cover   Missing
---------------------------------------------------
eval\metrics\vqa.py      96      0   100%
---------------------------------------------------
```

**Design decisions taken during implementation** (not spelled out by the plan verbatim, flagged here so they can be overridden):

1. **`grounded_accuracy` uses `exact_match` (not `token_f1`) as the "trả lời đúng" signal.** The plan table lists EM and Token-F1 as separate rows but does not say which one Grounded Accuracy's "đúng" refers to. EM was chosen as the stricter, deterministic gate — Token-F1 remains available as its own softer metric (`mean_token_f1`) for anyone who wants a partial-credit view. **Flagging this as a judgment call**, not a plan-confirmed decision.
2. **`hallucination_rate` is defined at the eval level as `confident AND NOT retrieval_hit`** (context never contained a correct keyframe at all), which is a different concept from the gate-level "ungrounded" in `apply_grounding_gate`'s docstring (`app/services/vqa_service.py`, P3.2 — a citation outside what the model was shown). The gate-level concept needs no ground truth and runs at request time; this eval-level concept needs ground truth and only makes sense offline. Both are legitimately called "grounded"/"hallucination" in different files — documented here to prevent conflating them later.
3. **`normalize_answer` only strips punctuation, not Vietnamese diacritics.** "bỏ dấu câu" in the plan means "remove punctuation marks", not "bỏ dấu" (remove diacritics) — verified against Vietnamese, since stripping diacritics would collapse distinct words (e.g. "đen" vs "den").
4. **English-article removal (`a`/`an`/`the`) is a documented no-op for this dataset.** `configs/eval/aic_qa_v1.json` reference answers are Vietnamese, which has no direct equivalent; kept for SQuAD-style convention compliance and in case an English answer variant is ever scored.
5. **No confidence-threshold config file added yet.** `score_query` takes `confidence_threshold` as a required parameter (no hardcoded default) per CLAUDE.md §6 "không magic number" — but since nothing calls this function yet (wiring is out of scope), a `configs/eval/vqa_metrics.yaml` threshold value would currently be unused dead config. Deferred to the wiring step.

**Còn thiếu (out of scope for this round, per the confirmed scope decision above):**
- No `eval/run.py` integration — no `vqa_run.py` (the `trake_run.py` equivalent), no CLI flag, no real 7-metric table has been produced from `configs/eval/aic_qa_v1.json`'s 91 approved records.
- `mean_llm_judge_score` is a pure aggregate only; there is no Gemini-calling judge client yet (the plan's "Gemini chấm rubric 1-5" I/O piece).
- The plan's exit-criteria checkboxes for `Grounded Accuracy` / `Hallucination Rate` hitting PRD §6.7 targets remain unchecked — no real number has been produced yet, only the scoring logic that will produce one.

## Merge evidence

Checkpoint commits on `feature/p3-vqa-submission`:
- `c872e5c` — `test: add reproducer for P3.4 VQA eval metrics` (RED, validated: collection `ModuleNotFoundError`)
- `8717bfe` — `feat: implement eval/metrics/vqa.py (P3.4 VQA scoring functions)` (GREEN, validated: 27 passed at time of commit; full suite 739 passed / 4 skipped / 0 failed)
- `4cb83ca` — closing edge-case test (`test_token_f1_both_empty_scores_1`, coverage 99%→100%) + this evidence report

---

## Addendum — wiring into `eval/run.py` (2026-08-17, same day, confirmed via `AskUserQuestion`)

**Scope decision:** after the pure module above, chủ dự án explicitly chose to continue with the `eval/run.py` wiring step in the same session rather than stopping at the pure module.

### Task report

| Task | Summary | Command | Result |
|---|---|---|---|
| Orchestration module | `eval/vqa_run.py` — `VqaEvalRecord`, `load_qa_eval_set`, `QaAnswerer` protocol, `run_vqa_queries`, `VqaGroupMetrics`/`VqaReportMetrics`, `aggregate_vqa_metrics`, `run_vqa_eval`, `render_vqa_report_md`. Mirrors `eval/trake_run.py`'s split from `eval/run.py`. | `./.venv/Scripts/python.exe -m pytest tests/test_vqa_run.py --cov=eval.vqa_run --cov-report=term-missing -q` | 13 passed, 100% coverage |
| CLI wiring | `eval/run.py`: `score_vqa`/`vqa_eval_set_path`/`vqa_top_k`/`vqa_confidence_threshold` added to `EvalRunConfig`; `_build_thumbnail_service`/`_build_qa_service` (mirrors `app/main.py`'s real production wiring); `write_report`/`_write_final_report` thread `vqa_metrics` through to the SAME `report.md`/`metrics.json`, same "None = not run" contract as `trake`. | `./.venv/Scripts/python.exe -m pytest tests/test_eval_run.py -q` | 26 passed |
| Run config | `configs/eval/qa_val.yaml` — `score_vqa: true`, points at `configs/eval/aic_qa_v1.json`; sanity-parsed through `_load_config` | `python -c "from eval.run import _load_config; _load_config(Path('configs/eval/qa_val.yaml'))"` | `score_vqa=True`, fields load correctly |
| Regression | Nothing existing broken | `./.venv/Scripts/python.exe -m pytest tests/ -q` | 757 passed, 4 skipped, 35 deselected |

RED (`eval/vqa_run.py`, before it existed):
```
ModuleNotFoundError: No module named 'eval.vqa_run'
```
RED (`eval/run.py` wiring, before `EvalRunConfig`/`write_report` had the new fields):
```
TypeError: EvalRunConfig.__init__() got an unexpected keyword argument 'score_vqa'
AttributeError: 'EvalRunConfig' object has no attribute 'score_vqa'
(plus 2 write_report-section assertion failures)
```
GREEN: `eval/vqa_run.py` 13 passed / 100% coverage; `eval/run.py` wiring 26 passed; combined `eval/metrics/vqa.py` + `eval/vqa_run.py` = 171 statements, 100% coverage, 67 tests; full suite 757 passed / 4 skipped / 0 failed.

### Test specification — `eval/vqa_run.py` (`tests/test_vqa_run.py`)

| # | What is guaranteed | Test | Result |
|---|---|---|---|
| 1 | Raw JSON dict records convert to `VqaEvalRecord` | `test_load_qa_eval_set_converts_raw_json_records` | PASS |
| 2 | Each record's `question` and the caller's `top_k` reach `answerer.answer_question` | `test_run_vqa_queries_sends_question_and_top_k` | PASS |
| 3 | Cited evidence ids resolve to context keyframes and score as grounded | `test_run_vqa_queries_resolves_evidence_ids_to_context_keyframes` | PASS |
| 4 | An evidence id absent from the grid is dropped, not raised — scores as ungrounded | `test_run_vqa_queries_drops_evidence_ids_missing_from_the_grid_without_raising` | PASS |
| 5 | One `VqaQueryResult` per record, in order | `test_run_vqa_queries_one_result_per_record_in_order` | PASS |
| 6 | `aggregate_vqa_metrics` computes the overall group correctly | `test_aggregate_vqa_metrics_computes_overall_group` | PASS |
| 7 | Results split correctly by `answerable_by_ocr` flag | `test_aggregate_vqa_metrics_splits_by_answerable_by_ocr_flag` | PASS |
| 8 | An empty OCR-flag subset reports `None`, not a misleading 0.0 | `test_aggregate_vqa_metrics_ocr_group_is_none_when_no_such_record_exists` | PASS |
| 9 | `retrieval_conditioned_accuracy` is `None` in the group when nothing had a retrieval hit | `test_aggregate_vqa_metrics_retrieval_conditioned_accuracy_is_none_without_a_retrieval_hit` | PASS |
| 10 | `run_vqa_eval` raises on an empty record list | `test_run_vqa_eval_raises_on_empty_records` | PASS |
| 11 | `run_vqa_eval` returns a full `VqaReportMetrics` | `test_run_vqa_eval_returns_report_metrics` | PASS |
| 12 | Rendered report includes the headline metric names and formatted numbers | `test_render_vqa_report_md_includes_headline_numbers` | PASS |
| 13 | Rendered report shows `n/a` (not a crash) when `retrieval_conditioned_accuracy` is `None` | `test_render_vqa_report_md_shows_n_a_when_retrieval_conditioned_accuracy_is_none` | PASS |

### Test specification — `eval/run.py` wiring (`tests/test_eval_run.py`)

| # | What is guaranteed | Test | Result |
|---|---|---|---|
| 14 | `write_report` appends a VQA section to `report.md` and `metrics.json` when `vqa_metrics` is given | `test_write_report_appends_vqa_section_when_vqa_metrics_given` | PASS |
| 15 | `write_report` omits the VQA section (`"vqa": null`) when `vqa_metrics` is not given | `test_write_report_omits_vqa_section_when_vqa_metrics_not_given` | PASS |
| 16 | `_load_config` reads `score_vqa`/`vqa_eval_set_path`/`vqa_top_k`/`vqa_confidence_threshold` | `test_load_config_reads_vqa_fields` | PASS |
| 17 | `_load_config` defaults `score_vqa` to `False` when absent | `test_load_config_defaults_vqa_fields_when_absent` | PASS |

## Coverage and known gaps

```
Name                  Stmts   Miss  Cover
------------------------------------------
eval\metrics\vqa.py      96      0   100%
eval\vqa_run.py          75      0   100%
------------------------------------------
TOTAL                   171      0   100%
```

`_build_qa_service`/`_build_thumbnail_service`/the `score_vqa` branch inside `eval/run.py::_main()` itself are **not** covered by any test — same deliberate gap as `_build_trake_service` (never unit-tested either): they construct real `GeminiVqaClient`/`ThumbnailService`/`SearchService` instances, which need a live Gemini key, Milvus, Mongo, and a GPU-loaded BEiT-3 checkpoint. `_build_qa_service` was sanity-checked by import + structural review against `app/main.py`'s identical `_build_qa_service` (same field names, same construction order), not by a mocked-out test — mirroring how `_build_trake_service` was verified in P2.3.

**Còn thiếu (real, not yet done):**
- The actual `python -m eval.run --config configs/eval/qa_val.yaml` run has **not** been executed — no real `Grounded Accuracy`/`Hallucination Rate` number exists yet, only the code that will produce one. Running it needs `docker compose up -d` (Milvus/Mongo/ES), a GPU for BEiT-3, and spends real Gemini API quota on all 91 frozen records once per run — prepared for chủ dự án to run themselves (CLAUDE.md §10), not run by the agent.
- `mean_llm_judge_score` (in `eval/metrics/vqa.py`) still has no Gemini-calling judge client wired to it anywhere — the "LLM-as-Judge" row of the plan's metric table is reachable in code but not yet produced by any run.
- `vqa_confidence_threshold: 0.5` in `configs/eval/qa_val.yaml` is an unvalidated placeholder (flagged `⚠️ [Cần đo]` in the config file itself) — needs tuning against a real confidence-score distribution once the first live run exists.

## Merge evidence — wiring addendum

Three more checkpoint commits on `feature/p3-vqa-submission`:
- `cdfd776` — `test: add reproducer for P3.4 eval/vqa_run.py orchestration` (RED, validated: `ModuleNotFoundError: No module named 'eval.vqa_run'`)
- `9a1bbeb` — `feat: implement eval/vqa_run.py (P3.4 VQA evaluation orchestration)` (GREEN, validated: 13 passed, 100% coverage on first attempt)
- `5ae9376` — `test: add reproducer for P3.4 eval/run.py VQA wiring` (RED, validated: `TypeError`/`AttributeError` on the not-yet-added `EvalRunConfig` fields, plus 2 `write_report` assertion failures)
- `82588c5` — `feat: wire P3.4 VQA scoring into eval/run.py (score_vqa flag)` (GREEN, validated: `tests/test_eval_run.py` 26 passed; full suite 757 passed / 4 skipped / 0 failed)

# ASR-Assisted KIS Evaluation Implementation Plan

**Goal:** Build a standalone, reproducible ASR gate evaluation runner that measures video retrieval, temporal evidence, and paired Text-only versus Text+ASR keyframe outcomes without fabricating ground truth.

## Global Constraints

- This optional plan is independent of the ASR-assisted KIS runtime; postponing it must not block runtime.
- Use standalone `python -m eval.asr_run`; do not add ASR options to `eval.run` or ASR records to `configs/eval/aic_eval_v1.json`.
- Ground truth comes only from official results or independent visual/manual review, never from ASR hits being evaluated.
- `relevant_video_ids` is required for video recall/gate survival; reviewed `start_sec`/`end_sec` and stored absolute `{video_id, frame_idx}` answers are optional, separately eligible subsets.
- Never derive a relevant or submitted `frame_idx` from timestamp.
- Report Video Recall@1/@5/@10/@20, Gate Survival, Segment Overlap, paired Text-only/Text+ASR AIC Final Score and Recall@K, Median Rank/Rank-in-Grid, and p50/p95 latency; split by `essential`, `supportive`, `irrelevant` and render zero-denominator subsets as `not measured`.
- Capture eval-set version/hash, ASR index/model identity, active search configuration/hash, query count, and distinct ground-truth videos.
- Do not create real labels, freeze artifacts, start services, import data, load models, or run evaluation automatically. Until a reviewed artifact exists and the runner completes, quality remains **not measured**.

## Owner prerequisite

The owner creates and independently reviews `configs/eval/asr_gate_v1.json`, then freezes its exact bytes in `configs/eval/asr_gate_v1.json.sha256`. Missing or mismatched artifacts must fail before stores or FG-CLIP are constructed.

## Implementation sequence

1. Define validated immutable ASR evaluation records and pure contract tests.
2. Implement pure ASR gate, temporal-evidence, and paired KIS metrics with explicit denominators.
3. Capture ASR index/model identity through read-only store queries; reject mixed or missing identity.
4. Build the standalone timed runner and auditable `metrics.json`, `report.md`, and `per_query.csv` bundle.
5. Document the owner-operated freeze/run workflow and verify it without real labels or live services.

## Measurement boundary

Video metrics use all approved records. Segment Overlap uses only records with reviewed `start_sec`/`end_sec`. Keyframe/AIC metrics use only records with reviewed `relevant_answers`. Missing eligibility is `not measured`, not zero; operational impressions are not retrieval-quality claims.

## Manual runner after implementation and owner review

```powershell
./.venv/Scripts/python.exe -m eval.asr_run --config configs/eval/asr_gate_v1.yaml
```

The owner runs this resource-bearing command only after the ASR index, matching model identity, Milvus/Mongo keyframes, local FG-CLIP checkpoint, and Elasticsearch/Milvus/MongoDB are ready. Codex does not launch it automatically.

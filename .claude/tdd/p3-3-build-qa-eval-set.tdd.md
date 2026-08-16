# TDD evidence — P3.3 `eval/build_qa_eval_set.py`

**Source plan:** [`.claude/plans/p3-vqa-submission.plan.md`](../plans/p3-vqa-submission.plan.md) §P3.3.

## Cập nhật sau khi chạy thật (2026-08-16, cùng ngày)

Chủ dự án chạy `generate --limit-shots 5` với `qwen2.5:7b-instruct` (default
lúc viết evidence report này) và phát hiện một câu hỏi mà chính
`reference_answer` của nó là lời từ chối ("không thể trả lời bằng cách xem
caption") thay vì một đáp án thật — model hỏi điều không có căn cứ trong
caption, đúng thứ `_QA_SYSTEM_PROMPT` cấm, rồi thay vì tự bỏ câu đó thì
nhét lời từ chối vào field `reference_answer`. Ollama's `format=<schema>`
ép đúng CẤU TRÚC JSON (test #10 xác nhận điều này) nhưng không ép được NỘI
DUNG tuân theo system prompt — đúng cảnh báo đã ghi sẵn ở `references.md`
§5.6 ("ép schema qua grammar constraint không đảm bảo 100% nội dung hợp
lý"), giờ có bằng chứng thực nghiệm cụ thể.

**Đổi default sang `qwen3:8b`** (`configs/eval/build_qa_eval_set.yaml`,
`BuildQaEvalSetConfig.ollama_model`) — thế hệ mới hơn, tài liệu Ollama ghi
mạnh hơn về instruction-following, vẫn vừa VRAM 8GB (5.2GB). Không đổi test
suite: mọi test dùng `"qwen2.5:7b-instruct"` như một giá trị fixture bất kỳ
để kiểm tra luồng dữ liệu (config → generator → record), không assert đây
PHẢI là default — đổi default không làm test nào sai.

**Chưa verify lại bằng chạy thật** với `qwen3:8b` — chủ dự án cần
`ollama pull qwen3:8b` rồi chạy lại `--limit-shots 5` để xác nhận model mới
có tránh được lỗi này không. Nếu vẫn còn, vấn đề nằm ở prompt (thiếu ví dụ
mẫu / thiếu chỉ dẫn rõ "không có câu trả lời thật thì tự bỏ câu, đừng viết
lời từ chối vào answer"), không phải ở kích cỡ model.

## Xác nhận bằng chạy thật toàn bộ (2026-08-16, cùng ngày)

`qwen3:8b` được chạy lại `--limit-shots 5` (kết quả sạch, 0 lỗi tự-thừa-nhận-
không-trả-lời-được) rồi chạy `generate` đầy đủ trên cả 173 shot đã duyệt:
**346 QA candidate, 0 shot thất bại.** Quét tự động tất cả 346 record (regex
tìm cụm "không thể trả lời"/"không rõ"/...) cho kết quả **0/346** — hết hẳn
lớp lỗi mà `qwen2.5:7b-instruct` mắc phải.

Chủ dự án chạy `review --limit 100` (subcommand mới, thêm sau khi tìm ra
lỗi gate đóng băng dưới đây) rồi `freeze`. File thật trên đĩa, verify lại
bằng cách tự tính sha256 và so khớp với `.sha256`:

```
configs/eval/aic_qa_v1.json           -- 91 record
configs/eval/aic_qa_v1.json.sha256    -- df553d24532d9e08db6eba65b5904792c4e621b95f6504dce77de91129930a36 (khớp, tự tính lại xác nhận)
configs/eval/aic_qa_v1.quality_report.md:
  total generated: 346
  total reviewed: 100
  approved: 91
  tỉ lệ loại khi review: 9.0%
  tỉ lệ answerable_by_ocr: 1.1%
```

**Lỗi phát hiện giữa chừng, đã sửa bằng TDD riêng** (commit `7dc4a71`):
`compute_qa_quality_report` bản đầu bắt buộc TOÀN BỘ 346 candidate phải có
`approved` khác `null` mới cho freeze -- trái với chính plan P3.3 ("Bộ chưa
review chỉ dùng để dò lỗi"). Sửa để chỉ cần ≥100 record ĐÃ REVIEW, phần còn
lại bị loại khỏi bộ đóng băng thay vì chặn `freeze`. 3 test mới
(`TestComputeQaQualityReport::test_ignores_extra_unreviewed_records_when_
enough_are_reviewed`, `test_raises_when_a_larger_batch_still_has_too_few_
reviewed`) khoá lại hành vi đúng.

Subcommand `review` (commit `442f978`) được thêm SAU khi chủ dự án thử tự
sửa tay JSON và thấy không khả thi ở quy mô 100-346 record -- vòng lặp
terminal (mở ảnh, hỏi y/n/skip/quit, ghi lại sau mỗi câu) thay cho việc sửa
tay, không thay đổi việc AI không được tự động hoá quyết định approve/reject
(CLAUDE.md §3) -- quyết định thật vẫn do chủ dự án gõ trong lúc review.

**P3.3 coi là hoàn thành** theo đúng "Verify" của plan: file bộ QA có
version + hash ✓, số mẫu đã review ghi rõ (100/346) ✓, tỉ lệ
`answerable_by_ocr` được báo cáo (1.1%) ✓.

## Design decisions made mid-implementation (not silently assumed)

The plan says "dùng LLM sinh câu hỏi" without naming a model. Two decisions
were surfaced to chủ dự án via `AskUserQuestion` before writing any code,
because CLAUDE.md §3 requires the question-generation model to differ from
the VQA answering model (`GeminiVqaClient`, gemini-3.6-flash) — picking a
model is a product decision, not a coding detail:

1. **OpenAI vs. local LLM** — chủ dự án has no budget for a paid API
   (2026-08-16) → local model via **Ollama**, not OpenAI (even though
   `openai_api_key` was already a required `Settings` field, it was never
   wired into any code path — confirmed by grep before proposing it).
2. **Vision (reads the keyframe image) vs. text-only (reads the existing
   caption)** — chủ dự án chose **text-only**, to keep VRAM usage low and
   sequential with BEiT-3 (CLAUDE.md §1.4). Traded off explicitly: this
   means question quality depends on how much visual detail Gemini's
   one-sentence caption already captured; the plan's own example ("biển
   hiệu ghi tên gì") may not always be answerable from caption alone. Not
   silently absorbed — recorded here and in the module docstring.

Model: `qwen2.5:7b-instruct` (chủ dự án's choice, ~4.7GB q4_K_M, fits
RTX 3070 8GB run sequentially, decent Vietnamese).

`ollama`'s structured-output API (`format=<model_json_schema()>`) was not
in `references.md` — fetched `github.com/ollama/ollama-python` (README does
NOT document it) and `ollama.com/blog/structured-outputs` (has the real
example) before writing `OllamaQaGenerator`, then added `references.md`
§5.6 in the same change (CLAUDE.md §5).

## User journeys (from the plan)

- As chủ dự án, I get 1-2 visually-grounded questions per already-reviewed
  shot, generated by a model that never answers VQA questions, so
  `Grounded Accuracy` never measures a model grading itself.
- As the reviewer, I can flag `answerable_by_ocr` per approved question so
  later reports can separate the OCR channel's credit from the VLM's.
- As chủ dự án, freezing refuses silently-wrong states: unreviewed
  candidates, fewer than 100 reviewed samples, an approved record missing
  its OCR flag, or a generator model that matches the VQA answering model.

## Task report

| Task | Summary | Command | Result |
|---|---|---|---|
| P3.3 | Implemented `load_approved_source_records`, `QaGeneration`/`QaItem` schema, `OllamaQaGenerator` (retry+backoff on transport errors only), `_generate_qa_candidates` (resume/checkpoint), `compute_qa_quality_report` (unreviewed/min-sample/OCR-flag gates), `assert_generator_differs_from_vqa_model` (self-grading guard), `freeze_qa_records`, `_freeze_qa_candidates`, CLI (`generate`/`freeze`) | `./.venv/Scripts/python.exe -m pytest tests/test_build_qa_eval_set.py -v` | 31 passed |
| Full suite regression check | No existing test broken by the new module | `./.venv/Scripts/python.exe -m pytest -q` | 708 passed, 4 skipped, 35 deselected |

RED (before implementation existed):
```
ModuleNotFoundError: No module named 'eval.build_qa_eval_set'
ERROR tests/test_build_qa_eval_set.py
```

GREEN (after implementation):
```
31 passed in 0.92s
```

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 1 | Only `approved: true` source records are used for generation | `TestLoadApprovedSourceRecords::test_keeps_only_approved_true_records` | unit | PASS |
| 2 | `QaGeneration` accepts exactly 1 or 2 items | `TestQaGenerationSchema::test_accepts_one_or_two_items` | unit | PASS |
| 3 | `QaGeneration` rejects 0 items | `TestQaGenerationSchema::test_rejects_zero_items` | unit | PASS |
| 4 | `QaGeneration` rejects >2 items | `TestQaGenerationSchema::test_rejects_more_than_two_items` | unit | PASS |
| 5 | Transient `httpx.ConnectError` is retried, then succeeds | `TestOllamaQaGeneratorRetry::test_retries_on_transient_connection_error_then_succeeds` | unit | PASS |
| 6 | Exhausted retries raise `RuntimeError` with attempt count | `TestOllamaQaGeneratorRetry::test_raises_runtime_error_after_exhausting_retries` | unit | PASS |
| 7 | `ollama.ResponseError` (model not pulled / bad request) is NOT retried — propagates immediately | `TestOllamaQaGeneratorRetry::test_does_not_retry_ollama_response_error` | unit | PASS |
| 8 | Malformed JSON (`pydantic.ValidationError`) is NOT retried | `TestOllamaQaGeneratorRetry::test_does_not_retry_malformed_json_validation_error` | unit | PASS |
| 9 | `OllamaQaGenerator.__init__` builds a real client without a network call | `TestOllamaQaGeneratorInit::test_init_builds_a_client_for_the_configured_host_without_a_network_call` | unit | PASS |
| 10 | `_call_once` sends the caption + `QaGeneration.model_json_schema()` and parses the response | `TestOllamaQaGeneratorCallOnce::test_call_once_sends_caption_and_schema_and_parses_response` | unit | PASS |
| 11 | One record is produced per generated `QaItem` (2 shots × 2 items = 4 records) with correct field mapping | `TestGenerateQaCandidates::test_produces_one_record_per_generated_item` | unit | PASS |
| 12 | Generation failure for one shot is recorded in `failed`, not silently skipped | `TestGenerateQaCandidates::test_generation_failure_is_reported_not_silently_skipped` | unit | PASS |
| 13 | `on_progress` checkpoints fire on failure too | `TestGenerateQaCandidates::test_on_progress_checkpoints_on_failure_too` | unit | PASS |
| 14 | Resume skips shots already in `existing_records` — not re-billed | `TestGenerateQaCandidates::test_resume_skips_shots_already_in_existing_records` | unit | PASS |
| 15 | `limit_shots` truncates the source-record list | `TestGenerateQaCandidates::test_limit_shots_truncates_source_records` | unit | PASS |
| 16 | `on_progress` checkpoints after each shot on success | `TestGenerateQaCandidates::test_on_progress_checkpoints_after_each_shot` | unit | PASS |
| 17 | Unreviewed candidate (`approved: null`) blocks freeze | `TestComputeQaQualityReport::test_raises_on_unreviewed_record` | unit | PASS |
| 18 | Fewer than `min_reviewed_samples` reviewed blocks freeze | `TestComputeQaQualityReport::test_raises_when_fewer_than_minimum_reviewed_samples` | unit | PASS |
| 19 | Approved record missing `answerable_by_ocr` blocks freeze | `TestComputeQaQualityReport::test_raises_when_approved_record_missing_answerable_by_ocr_flag` | unit | PASS |
| 20 | Rejection rate and OCR-answerable rate computed correctly | `TestComputeQaQualityReport::test_computes_rejection_rate_and_ocr_rate` | unit | PASS |
| 21 | Frozen hash changes when content changes | `TestFreezeQaRecords::test_hash_changes_when_content_changes` | unit | PASS |
| 22 | Provenance fields (`generator_model`, `source_caption`, `approved`) never leak into the frozen ground truth | `TestFreezeQaRecords::test_provenance_fields_do_not_leak_into_frozen_output` | unit | PASS |
| 23 | Freeze refuses when the generator model matches the VQA answering model (self-grading guard) | `TestAssertGeneratorDiffersFromVqaModel::test_raises_when_generator_matches_vqa_answering_model` | unit | PASS |
| 24 | Freeze proceeds when models differ | `TestAssertGeneratorDiffersFromVqaModel::test_passes_when_generator_differs_from_vqa_answering_model` | unit | PASS |
| 25 | `_freeze_qa_candidates` composes gate + self-grading guard + freeze + report correctly end to end | `TestFreezeQaCandidatesIntegration::test_freezes_successfully_when_reviewed_and_generator_differs` | unit | PASS |
| 26 | `_freeze_qa_candidates` refuses on self-grading | `TestFreezeQaCandidatesIntegration::test_refuses_to_freeze_when_generator_matches_vqa_model` | unit | PASS |
| 27 | `_freeze_qa_candidates` refuses below the minimum sample gate | `TestFreezeQaCandidatesIntegration::test_refuses_to_freeze_below_minimum_reviewed_samples` | unit | PASS |
| 28 | Candidates doc round-trips through a resume write | `TestWriteQaCandidatesDoc::test_round_trips_through_resume` | unit | PASS |
| 29 | YAML config loads into `BuildQaEvalSetConfig` | `TestLoadConfig::test_reads_yaml_into_dataclass` | unit | PASS |
| 30 | `_run_freeze` CLI path writes `.json`/`.sha256`/`.quality_report.md` on success | `TestRunFreezeCli::test_writes_frozen_json_sha256_and_quality_report` | unit | PASS |
| 31 | `_run_freeze` CLI path returns exit code 1 and writes nothing when the quality gate fails | `TestRunFreezeCli::test_returns_nonzero_and_writes_nothing_when_quality_gate_fails` | unit | PASS |

## Coverage and known gaps

`./.venv/Scripts/python.exe -m pytest tests/test_build_qa_eval_set.py --cov=eval.build_qa_eval_set --cov-report=term-missing`:

```
eval\build_qa_eval_set.py     186     32    83%   420-443, 447-474, 499-505, 509
```

Uncovered lines are `_parse_args` (argparse plumbing), `_run_generate`
(needs a live Ollama server — same category of gap as
`build_eval_set.py`'s own `_run_generate`, which is also untested there),
and the `main()`/`__main__` entry point. `_run_freeze` — the only CLI path
that does not require a live external service — IS covered
(`TestRunFreezeCli`).

**Explicitly out of scope for P3.3** (belongs to P3.4 — `eval/metrics/
vqa.py`, per the plan):
- Computing Grounded Accuracy / Hallucination Rate / etc. from the frozen
  `aic_qa_v1.json` — this module only produces the ground-truth file.
- Actually running `generate` against a real Ollama server, or `freeze`
  against real reviewed data — per CLAUDE.md §10, both are prepared for
  chủ dự án to run, not auto-executed here. No `configs/eval/
  aic_qa_v1.candidates.json` or `configs/eval/aic_qa_v1.json` exists yet.

## Merge evidence

Two checkpoint commits on `feature/p3-vqa-submission`:
- `32e14ae` — `test: add RED reproducer for P3.3 build_qa_eval_set` (RED, validated: `ModuleNotFoundError` on collection)
- `b65f51d` — `feat: implement build_qa_eval_set.py (P3.3 QA eval-set generation)` (GREEN, validated: 31/31 passed, full suite 708 passed/4 skipped/0 failed)

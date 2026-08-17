# TDD evidence — P3.2 grounding gate + `POST /api/qa`

**Source plan:** [`.claude/plans/p3-vqa-submission.plan.md`](../plans/p3-vqa-submission.plan.md) §P3.2 · **Trước:** [`p3-1-vqa-service.tdd.md`](./p3-1-vqa-service.tdd.md)

## User journeys (from the plan)

- As the operator, I never see an answer citing a keyframe the model was not actually shown — a fabricated citation collapses the answer to "không đủ căn cứ", so I am not misled while being timed.
- As the operator, if Gemini is down or slow I still get the keyframe grid plus a readable notice, so US-5 stays partially usable instead of erroring out.
- As the evaluator, `Grounded Accuracy` and `Hallucination Rate` become measurable at all, because "grounded" now has an enforced definition rather than being whatever the model claimed.

## Task report

| Task | Summary | Command | Result |
|---|---|---|---|
| Gate + orchestration | `apply_grounding_gate`, `GroundedAnswer`, `QaResult`, `QaService`, `KeyframeImageUnavailable`, `ThumbnailImageLoader` added to `app/services/vqa_service.py` | `./.venv/Scripts/python.exe -m pytest tests/test_qa_service.py tests/test_vqa_service.py -q` | 39 passed |
| Endpoint | `app/api/qa.py` + `app/schemas/{request,response}/qa.py` + `get_qa_service`, wired in `app/main.py` | `./.venv/Scripts/python.exe -m pytest tests/test_qa_api.py -q` | 8 passed |
| Regression | Nothing existing broken, including P3.1's 14 tests | `./.venv/Scripts/python.exe -m pytest -q` | 677 passed, 4 skipped, 35 deselected |

RED (before implementation existed):
```
ImportError: cannot import name 'qa' from 'app.api'
ERROR tests/test_qa_service.py
ERROR tests/test_qa_api.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
```

GREEN (after implementation): `39 passed` (service) then `8 passed` (API); full suite `677 passed, 4 skipped` — up from P3.1's `644 passed, 4 skipped`, i.e. +33 with zero regressions.

## Test specification

### Grounding gate — `tests/test_qa_service.py::TestGroundingGate`

| # | What is guaranteed | Test | Result |
|---|---|---|---|
| 1 | Evidence naming a keyframe outside the shown set is rejected | `test_vqa_rejects_evidence_outside_retrieved_set` ⭐ *(plan-named)* | PASS |
| 2 | One fabricated id rejects the **whole** answer, even alongside a valid one | `test_a_single_fabricated_id_rejects_the_whole_answer` | PASS |
| 3 | Fully-inside evidence is accepted and citations preserved in order | `test_accepts_an_answer_whose_evidence_is_fully_inside_the_set` | PASS |
| 4 | A confident answer citing nothing is not grounded (no `frame_id` ⇒ unsubmittable) | `test_confident_answer_citing_nothing_is_not_grounded` | PASS |
| 5 | The model's own abstention passes through as `insufficient_evidence` | `test_model_abstention_passes_through_as_insufficient_evidence` | PASS |
| 6 | `confidence`/`reasoning` survive rejection unchanged (P3.4 needs them) | `test_preserves_confidence_and_reasoning_even_when_rejected` | PASS |
| 7 | An empty shown-set rejects every citation | `test_empty_retrieved_set_rejects_every_citation` | PASS |
| 8 | Id matching is exact, not prefix (`kf1` ∉ `{kf10, kf11}`) | `test_keyframe_id_matching_is_exact_not_prefix` | PASS |

### `QaService` — happy path, grounding integration, degraded mode, cost control

| # | What is guaranteed | Test | Result |
|---|---|---|---|
| 9 | Answer + grid returned together, no notice | `TestQaServiceHappyPath::test_returns_answered_with_the_keyframe_grid` | PASS |
| 10 | Retrieval uses `semantic` mode and the requested `top_k` | `::test_retrieves_with_semantic_mode_and_the_requested_top_k` | PASS |
| 11 | Only retrieved keyframes reach the model | `::test_only_retrieved_keyframes_are_handed_to_the_model` | PASS |
| 12 | Fabricated evidence is rejected but the grid survives | `TestQaServiceGroundingIntegration::test_fabricated_evidence_is_rejected_but_the_grid_survives` | PASS |
| 13 | The gate uses the set **sent**, not the set **retrieved** — a keyframe whose image failed to load cannot be cited | `::test_gate_uses_the_set_actually_sent_not_the_set_retrieved` | PASS |
| 14 | Gemini failure still returns the keyframe grid + notice | `TestQaServiceDegradedMode::test_vqa_failure_still_returns_keyframe_grid` ⭐ *(plan-named)* | PASS |
| 15 | A timeout takes the same degraded path | `::test_vqa_timeout_is_the_same_degraded_path` | PASS |
| 16 | One unloadable image is skipped, not fatal | `::test_a_keyframe_with_no_image_is_skipped_not_fatal` | PASS |
| 17 | Zero retrieval results ⇒ the model is never called (no background-knowledge answers) | `::test_retrieval_returning_nothing_never_calls_the_model` | PASS |
| 18 | Every image failing ⇒ the model is never called | `::test_every_image_failing_never_calls_the_model` | PASS |
| 19 | A **retrieval** failure propagates instead of masquerading as degraded mode | `::test_a_search_failure_is_not_swallowed_into_degraded_mode` | PASS |
| 20 | At most `max_keyframes_to_model` images reach the model; the grid still shows all `top_k` | `TestQaServiceCostControl::test_sends_at_most_max_keyframes_to_the_model` | PASS |
| 21 | Images are loaded only for keyframes actually sent | `::test_images_are_only_loaded_for_keyframes_actually_sent` | PASS |
| 22 | Citing a grid keyframe beyond the cap is rejected | `::test_citing_a_keyframe_in_the_grid_but_beyond_the_cap_is_rejected` | PASS |

### `ThumbnailImageLoader` + `POST /api/qa`

| # | What is guaranteed | Test | Result |
|---|---|---|---|
| 23 | Loads the image ThumbnailService resolves | `TestThumbnailImageLoader::test_loads_the_image_the_thumbnail_service_resolves` | PASS |
| 24 | A resolver failure becomes `KeyframeImageUnavailable` | `::test_translates_a_resolver_failure_into_keyframe_image_unavailable` | PASS |
| 25 | An undecodable file becomes `KeyframeImageUnavailable` | `::test_translates_an_undecodable_file_into_keyframe_image_unavailable` | PASS |
| 26 | 200 + answer + grid on the happy path | `tests/test_qa_api.py::TestQaEndpoint::test_returns_the_answer_and_the_grid` | PASS |
| 27 | **Gemini down ⇒ HTTP 200 + grid + notice, never 5xx** | `::test_vqa_failure_still_returns_keyframe_grid` ⭐ *(plan's experiment, automated)* | PASS |
| 28 | Grid entries carry `thumbnail_url`/`image_url` like `/api/search` | `::test_keyframes_carry_thumbnail_and_image_urls` | PASS |
| 29 | `top_k` defaults to `DEFAULT_TOP_K` when omitted | `::test_applies_the_default_top_k_when_omitted` | PASS |
| 30 | An explicit `top_k` passes through | `::test_passes_an_explicit_top_k_through` | PASS |
| 31 | Empty question ⇒ 422 | `::test_rejects_an_empty_question` | PASS |
| 32 | `top_k` above the 100-line submission cap ⇒ 422 | `::test_rejects_a_top_k_above_the_submission_cap` | PASS |
| 33 | Missing `question` ⇒ 422 | `::test_rejects_a_missing_question_field` | PASS |

## Coverage and known gaps

```
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
app\api\qa.py                    15      0   100%
app\services\vqa_service.py     159      5    97%   187-190, 205-214
```

Uncovered lines are still only `GeminiVqaClient.__init__` (constructs a real `genai.Client`) and `_call_once` (the live `interactions.create` call) — the same deliberate gap documented in P3.1, matching `GeminiOcrClient`/`GeminiQueryRewriter`.

**Design decisions taken during implementation** (not spelled out by the plan, flagged here so they can be overridden):

1. **Images come from `ThumbnailService`** (`thumbnail_max_px: 320`), not the full-res corpus. The plan does not say where VQA images come from; reading thumbnails bounds the payload for free from a cache the search grid already warms.
2. **`max_keyframes_to_model: 8`** added to `configs/qa/vqa.yaml`. The plan says "prompt chỉ chứa top-k keyframe", but `top_k` is a request parameter bounded by `MAX_TOP_K=100`; a separate cap was needed so one question cannot send 100 images. The grid still returns all `top_k`. **This number is a guess and should be tuned against `Grounded Accuracy` in P3.4.**
3. **No `confidence` threshold in the gate.** P3.4 owns thresholding (`configs/`-driven per the plan); a gate that dropped low-confidence-but-grounded answers would corrupt `Abstention Rate`.
4. **Zero citations is rejected** even for a confident answer, because the AIC QA submission line is `<video_id>, <frame_id>, <answer>` — an uncited answer has no `frame_id` and cannot be submitted at all.

**Still open / out of scope for P3.2:**
- The manual experiment "chặn mạng tới Gemini → `POST /api/qa` vẫn trả 200" is automated at the service and router level (#14, #27) but has **not** been run against a live server with real network blocking.
- No UI yet — the plan's "hiển thị keyframe bằng chứng cạnh câu trả lời" needs a frontend panel; the API already returns `evidence_keyframe_ids` + the grid to support it.
- P3.3 (QA eval set) is where the CLAUDE.md invariant "model sinh nhãn ≠ model bị đánh giá" must be honoured — the answering model is Gemini, so the question generator must not be.

## Merge evidence

Two checkpoint commits on `feature/p3-vqa-submission`:
- `abbd7f0` — `test: add RED reproducer for P3.2 grounding gate + POST /api/qa` (RED, validated: collection `ImportError` on both files)
- `ceecfb0` — `feat: add P3.2 grounding gate + POST /api/qa` (GREEN, validated: 39 + 8 passed; full suite 677 passed / 4 skipped / 0 failed)

# TDD evidence — P3.1 `vqa_service.py`

**Source plan:** [`.claude/plans/p3-vqa-submission.plan.md`](../plans/p3-vqa-submission.plan.md) §P3.1.

## User journeys (from the plan)

- As the operator, when I ask a question, the system answers using ONLY the keyframes it actually retrieved for me — never the model's background knowledge.
- As the operator, I get back a structured `{answer, confidence, evidence_keyframe_ids[], reasoning}` I can act on programmatically, not free text.
- As the operator, if the retrieved keyframes don't contain enough visual evidence, the model tells me so ("Không đủ căn cứ") instead of guessing — this is what `Hallucination Rate` measures.

## Task report

| Task | Summary | Command | Result |
|---|---|---|---|
| P3.1 | Implemented `VqaAnswer` schema, `RetrievedKeyframe`, `_build_input_parts`, `GeminiVqaClient` (retry+backoff), `VqaService` (async timeout wrapper, raises `VqaServiceError`) | `./.venv/Scripts/python.exe -m pytest tests/test_vqa_service.py -v` | 14 passed |
| Full suite regression check | No existing test broken by the new module | `./.venv/Scripts/python.exe -m pytest -q` | 644 passed, 4 skipped, 35 deselected |

RED (before implementation existed):
```
ModuleNotFoundError: No module named 'app.services.vqa_service'
ERROR tests/test_vqa_service.py
```

GREEN (after implementation):
```
14 passed in 0.40s
```

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 1 | The Gemini prompt's image/keyframe-id parts reference exactly the retrieved keyframes, in order — no other keyframe can appear | `tests/test_vqa_service.py::TestBuildInputParts::test_vqa_prompt_contains_only_retrieved_keyframes` | unit | PASS |
| 2 | A keyframe_id that was never passed in never appears anywhere in the built prompt | `TestBuildInputParts::test_prompt_never_references_a_keyframe_outside_the_retrieved_list` | unit | PASS |
| 3 | The question text is included in the prompt | `TestBuildInputParts::test_question_text_is_included` | unit | PASS |
| 4 | Zero retrieved keyframes still produces a valid (text-only) prompt | `TestBuildInputParts::test_empty_keyframe_list_still_produces_a_prompt_with_no_images` | unit | PASS |
| 5 | `VqaAnswer.model_json_schema()` exposes `answer`/`confidence`/`evidence_keyframe_ids`/`reasoning`, with `evidence_keyframe_ids` optional (has a default) | `TestVqaAnswerSchema::test_vqa_output_matches_json_schema` | unit | PASS |
| 6 | `confidence` outside `[0, 1]` is rejected by the schema | `TestVqaAnswerSchema::test_confidence_must_be_within_0_and_1` | unit | PASS |
| 7 | The model can answer "Không đủ căn cứ" with empty evidence and low confidence — the schema does not reject or special-case this | `TestVqaAnswerSchema::test_vqa_can_return_insufficient_evidence` | unit | PASS |
| 8 | `evidence_keyframe_ids` defaults to `[]` when omitted from the payload | `TestVqaAnswerSchema::test_evidence_keyframe_ids_defaults_to_empty_when_omitted` | unit | PASS |
| 9 | `VqaService.answer()` returns the client's `VqaAnswer` unchanged on success | `TestVqaService::test_returns_the_client_answer_on_success` | unit (async) | PASS |
| 10 | Gemini failure after exhausted retries surfaces as `VqaServiceError`, not a bare exception | `TestVqaService::test_raises_vqa_service_error_when_gemini_fails_after_exhausted_retries` | unit (async) | PASS |
| 11 | Malformed structured output surfaces as `VqaServiceError` | `TestVqaService::test_raises_vqa_service_error_when_output_is_malformed` | unit (async) | PASS |
| 12 | Exceeding `timeout_sec` surfaces as `VqaServiceError`, not a hang | `TestVqaService::test_raises_vqa_service_error_when_the_call_exceeds_the_configured_timeout` | unit (async) | PASS |
| 13 | `GeminiVqaClient` retries on transient network errors (`ConnectionError`/`OSError`/`TimeoutError`) and succeeds once the transient condition clears | `TestGeminiVqaClientRetry::test_retries_on_transient_error_then_succeeds` | unit | PASS |
| 14 | `GeminiVqaClient` raises `RuntimeError` with an attempt count after exhausting retries | `TestGeminiVqaClientRetry::test_raises_runtime_error_after_exhausting_retries` | unit | PASS |

## Coverage and known gaps

`./.venv/Scripts/python.exe -m pytest tests/test_vqa_service.py --cov=app.services.vqa_service --cov-report=term-missing`:

```
app\services\vqa_service.py      74      5    93%   168-171, 186-195
```

Uncovered lines are `GeminiVqaClient.__init__` (constructs a real `genai.Client`) and `_call_once` (the actual `interactions.create` network call) — both require a live Gemini API key and are intentionally not exercised by unit tests, same gap as the existing `GeminiOcrClient`/`GeminiQueryRewriter` (neither has a direct test of `_call_once` either; both are exercised only through their retry wrapper via `monkeypatch`).

**Explicitly out of scope for P3.1** (belongs to P3.2 — grounding gate, per the plan):
- Verifying `evidence_keyframe_ids ⊆` the retrieved set and downgrading to "insufficient evidence" if not.
- Gemini-down/timeout → falling back to a keyframe grid + notice instead of an error (that behavior now lives one layer up, as `VqaServiceError`, which P3.2's caller is expected to catch).
- The `/api/qa` endpoint and its wiring into `app/main.py`/`app/dependencies.py`.
- `configs/qa/vqa.yaml` is added but not yet loaded anywhere (no startup wiring exists until P3.2 builds the endpoint).

## Merge evidence

Two checkpoint commits on `feature/p3-vqa-submission`:
- `9ee3713` — `test: add RED reproducer for P3.1 vqa_service` (RED, validated: `ModuleNotFoundError` on collection)
- `467a304` — `feat: implement VqaService/GeminiVqaClient (P3.1)` (GREEN, validated: 14/14 passed, full suite 644 passed/4 skipped/0 failed)

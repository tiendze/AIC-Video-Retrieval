# ASR-Assisted KIS Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, reproducible ASR gate evaluation runner that measures video retrieval, temporal evidence, and paired Text-only versus Text+ASR keyframe outcomes without fabricating ground truth.

**Architecture:** Pure evaluation types and metrics live in `eval/asr_eval.py`; the standalone `eval.asr_run` entry point loads only an independently reviewed ASR artifact, constructs the same production services as online KIS, and writes an auditable report bundle. Video-level metrics use every approved record, temporal overlap uses only records with reviewed intervals, and keyframe/AIC metrics use only records with reviewed absolute `relevant_answers`.

**Tech Stack:** Python 3.10, dataclasses, JSON/YAML/CSV, FastAPI service-layer objects, Elasticsearch/Milvus/MongoDB clients, Pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-asr-assisted-kis-design.md`

## Global Constraints

- This is an optional plan independent of `2026-08-23-asr-assisted-kis-runtime.md`; postponing it must not block the runtime feature.
- Use the standalone entry point `python -m eval.asr_run`; do not add ASR options to `eval.run`.
- Do not append ASR records to `configs/eval/aic_eval_v1.json`.
- Ground truth comes only from official results or independent visual/manual review, never from the ASR retrieval output being evaluated.
- Allowed `asr_utility` values are exactly `essential`, `supportive`, and `irrelevant`.
- Every scored record has `review_status: approved`; duplicate `query_id` or duplicate public question pairs fail validation.
- `relevant_video_ids` is required for ASR Video Recall and Gate Survival.
- `start_sec` and `end_sec` are optional as a pair and gate Segment Overlap participation.
- `relevant_answers` is optional and contains stored absolute `{video_id, frame_idx}` pairs; only records containing it participate in end-to-end AIC Final Score/Recall@K.
- Never derive a relevant or submitted `frame_idx` from timestamp.
- Report ASR Video Recall@1/@5/@10/@20, Gate Survival Rate, Segment Overlap Rate, paired Text-only versus Text+ASR AIC Final Score and Recall@K, Median Rank/Rank-in-Grid, and p50/p95 latency.
- Split every retrieval metric by `essential`, `supportive`, and `irrelevant`, reporting `n=0` groups as not measured rather than zero quality.
- Report eval-set version/hash, query count, distinct ground-truth video count, ASR index/model identity, and active search configuration/hash.
- The ASR model used to create labels or choose ground truth must not be inferred from the evaluated top hits.
- Do not create real labels, freeze an artifact, start services, import data, load models, or run full evaluation automatically. Those are owner-reviewed/manual operations.
- Until a reviewed artifact exists and the runner completes, the official quality statement remains **not measured**.

## Manual Prerequisite Outside Agent Execution

The project owner creates `configs/eval/asr_gate_v1.json`, reviews every record independently, then writes its SHA-256 to `configs/eval/asr_gate_v1.json.sha256`. The implementation deliberately does not create this real artifact: generated or guessed relevance labels would invalidate the evaluation. Before that file exists, the runner must exit clearly without constructing stores or loading FG-CLIP.

## File Structure

- Create `eval/asr_eval.py`: validated record types, metric result types, query scoring, aggregation, and percentile helpers; no store construction or filesystem writes.
- Create `eval/asr_run.py`: standalone configuration, artifact/hash loading, production service wiring, timed paired execution, identity capture, report writing, and CLI.
- Create `configs/eval/asr_gate_v1.yaml`: versioned runner settings pointing to the owner-reviewed artifact.
- Create `tests/fixtures/asr_eval_sample.json`: deterministic test-only records; never used as real ground truth.
- Create `tests/test_asr_eval.py`: contract and pure metric tests.
- Create `tests/test_asr_run.py`: orchestration, identity, output, and fail-fast tests using fakes.
- Modify `app/stores/mongo_client.py`: read-only distinct ASR model identity lookup for report provenance.
- Modify `tests/test_mongo_client.py`: store identity lookup test.
- Modify `README.md` and `docs/PRD.md`: standalone manual command, label boundary, and “not measured” state.

---

### Task 1: Define and Validate the Independent ASR Evaluation Contract

**Files:**
- Create: `eval/asr_eval.py`
- Create: `tests/fixtures/asr_eval_sample.json`
- Create: `tests/test_asr_eval.py`

**Interfaces:**
- Produces: `AsrUtility = Literal["essential", "supportive", "irrelevant"]`.
- Produces: immutable `RelevantAnswer(video_id: str, frame_idx: int)` and `AsrEvalRecord`.
- Produces: `load_asr_eval_records(raw: Sequence[Mapping[str, Any]]) -> list[AsrEvalRecord]`.
- Produces: validated records with trimmed non-empty queries, unique relevant videos/answers, paired interval bounds, approved review state, and unique public-question identity.

- [ ] **Step 1: Create a deterministic test-only fixture**

Create `tests/fixtures/asr_eval_sample.json` with two records:

```json
[
  {
    "query_id": "sample-essential-1",
    "text_query": "aerial view of a dam under rain",
    "asr_query": "công trình thủy lợi đập hồ chứa",
    "relevant_video_ids": ["L21_V001"],
    "relevant_answers": [{"video_id": "L21_V001", "frame_idx": 3855}],
    "start_sec": 151.0,
    "end_sec": 159.0,
    "asr_utility": "essential",
    "label_source": "test_fixture_independent_review",
    "review_status": "approved"
  },
  {
    "query_id": "sample-irrelevant-1",
    "text_query": "a presenter standing in a studio",
    "asr_query": "bản tin thời sự",
    "relevant_video_ids": ["L21_V002"],
    "asr_utility": "irrelevant",
    "label_source": "test_fixture_independent_review",
    "review_status": "approved"
  }
]
```

This fixture proves optional interval/answer behavior only; its IDs are not a quality claim.

- [ ] **Step 2: Write failing loader tests for valid optional fields**

Define the two test helpers once:

```python
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "asr_eval_sample.json"


def load_fixture() -> list[dict[str, object]]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def valid_raw_record() -> dict[str, object]:
    return dict(load_fixture()[0])
```

```python
def test_load_records_preserves_absolute_answers_and_optional_measurement_subsets():
    records = load_asr_eval_records(load_fixture())
    assert records[0].relevant_answers == (
        RelevantAnswer(video_id="L21_V001", frame_idx=3855),
    )
    assert records[0].has_temporal_label is True
    assert records[0].has_keyframe_labels is True
    assert records[1].has_temporal_label is False
    assert records[1].has_keyframe_labels is False
```

- [ ] **Step 3: Write failing validation tests for invalid or circular labels**

```python
@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"asr_utility": "unknown"}, "asr_utility"),
        ({"review_status": "draft"}, "review_status must be approved"),
        ({"start_sec": 10.0, "end_sec": None}, "start_sec and end_sec"),
        ({"start_sec": 20.0, "end_sec": 10.0}, "end_sec must be"),
        ({"relevant_video_ids": []}, "relevant_video_ids"),
        (
            {"relevant_answers": [{"video_id": "OTHER", "frame_idx": 1}]},
            "answer video_id must be relevant",
        ),
        ({"label_source": "asr_top_hit"}, "label_source cannot be retrieval-derived"),
    ],
)
def test_invalid_records_fail_with_actionable_messages(mutation, message):
    raw = valid_raw_record() | mutation
    with pytest.raises(ValueError, match=message):
        load_asr_eval_records([raw])


def test_duplicate_query_id_and_duplicate_query_pair_fail():
    first = valid_raw_record()
    with pytest.raises(ValueError, match="duplicate query_id"):
        load_asr_eval_records([first, first | {"text_query": "another visual query"}])
    with pytest.raises(ValueError, match="duplicate public question"):
        load_asr_eval_records([first, first | {"query_id": "second-id"}])
```

Reject normalized label sources `asr_top_hit`, `retrieval_result`, and `model_under_evaluation`; accept other non-empty sources so official/manual source names remain auditable without inventing a closed enum absent from the spec.

- [ ] **Step 4: Run contract tests and confirm RED**

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_asr_eval.py -q
```

Expected: collection fails because `eval.asr_eval` does not exist.

- [ ] **Step 5: Implement immutable types and explicit validation**

```python
AsrUtility = Literal["essential", "supportive", "irrelevant"]


@dataclass(frozen=True)
class RelevantAnswer:
    video_id: str
    frame_idx: int


@dataclass(frozen=True)
class AsrEvalRecord:
    query_id: str
    text_query: str
    asr_query: str
    relevant_video_ids: tuple[str, ...]
    relevant_answers: tuple[RelevantAnswer, ...]
    start_sec: float | None
    end_sec: float | None
    asr_utility: AsrUtility
    label_source: str
    review_status: Literal["approved"]

    @property
    def has_temporal_label(self) -> bool:
        return self.start_sec is not None and self.end_sec is not None

    @property
    def has_keyframe_labels(self) -> bool:
        return bool(self.relevant_answers)
```

Parse records through small named helpers that raise `ValueError` containing `query_id` and the invalid field. Deduplicate repeated relevant IDs/answers in first-seen order; reject duplicate records after trimming and Unicode case-folding the Text/ASR query pair.

- [ ] **Step 6: Run contract tests and style checks**

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_asr_eval.py -q
./.venv/Scripts/python.exe -m ruff check eval/asr_eval.py tests/test_asr_eval.py
./.venv/Scripts/python.exe -m black --check eval/asr_eval.py tests/test_asr_eval.py
```

Expected: PASS.

- [ ] **Step 7: Commit the eval contract**

```powershell
git add eval/asr_eval.py tests/fixtures/asr_eval_sample.json tests/test_asr_eval.py
git commit -m "feat: define ASR gate evaluation contract"
```

---

### Task 2: Implement Pure ASR Gate and Paired KIS Metrics

**Files:**
- Modify: `eval/asr_eval.py`
- Modify: `tests/test_asr_eval.py`
- Reuse: `eval/metrics/retrieval.py`
- Reuse: `eval/metrics/latency.py`

**Interfaces:**
- Consumes: `AsrGateResult`, `KisCandidatePool`, and validated `AsrEvalRecord`.
- Produces: `AsrQueryOutcome`, `UtilityMetrics`, and `AsrEvalMetrics` dataclasses.
- Produces: `score_asr_query(record, gate, text_only_pool, gated_pool, asr_latency_ms, stage1_latency_ms) -> AsrQueryOutcome`.
- Produces: `aggregate_asr_outcomes(outcomes, video_recall_ks, keyframe_recall_ks, configured_gate_k) -> AsrEvalMetrics`.

Use these concrete helpers throughout the Task 2 tests:

```python
_BASE_RECORD = AsrEvalRecord(
    query_id="q1",
    text_query="a dam under rain",
    asr_query="công trình thủy lợi",
    relevant_video_ids=("V1",),
    relevant_answers=(),
    start_sec=None,
    end_sec=None,
    asr_utility="essential",
    label_source="independent_manual_review",
    review_status="approved",
)


def record(**changes) -> AsrEvalRecord:
    return dataclasses.replace(_BASE_RECORD, **changes)


def hit(video_id: str, frame_idx: int) -> SearchHit:
    return SearchHit(
        keyframe_id=f"{video_id}_f{frame_idx}",
        video_id=video_id,
        score=1.0,
        frame_idx=frame_idx,
        timestamp=float(frame_idx),
        thumbnail_path="",
    )


def pool(*hits: SearchHit) -> KisCandidatePool:
    return KisCandidatePool(
        candidate_ids=[item.keyframe_id for item in hits],
        text_candidate_count=len(hits),
        ocr_candidate_count=0,
        results=list(hits),
    )


def gate(*video_ids: str) -> AsrGateResult:
    return AsrGateResult(
        query="công trình thủy lợi",
        segment_hit_count=len(video_ids),
        videos=[
            AsrVideoEvidence(
                video_id=video_id,
                video_rank=rank,
                best_bm25_score=float(len(video_ids) - rank + 1),
                segments=[],
            )
            for rank, video_id in enumerate(video_ids, start=1)
        ],
    )


def gate_with_segment(video_id: str, start: float, end: float) -> AsrGateResult:
    return AsrGateResult(
        query="công trình thủy lợi",
        segment_hit_count=1,
        videos=[
            AsrVideoEvidence(
                video_id=video_id,
                video_rank=1,
                best_bm25_score=1.0,
                segments=[
                    AsrEvidenceSegment("segment-1", "evidence", start, end, 1.0)
                ],
            )
        ],
    )


def score(eval_record: AsrEvalRecord, asr_gate: AsrGateResult) -> AsrQueryOutcome:
    return score_asr_query(
        eval_record,
        asr_gate,
        text_only_pool=pool(),
        gated_pool=pool(),
        asr_latency_ms=1.0,
        stage1_latency_ms=2.0,
    )
```

- [ ] **Step 1: Write failing gate-metric tests**

```python
def test_video_recall_and_gate_survival_use_ranked_gate_videos():
    outcome = score_asr_query(
        record=record(relevant_video_ids=("V2",)),
        gate=gate("V1", "V2", "V3"),
        text_only_pool=pool(),
        gated_pool=pool(),
        asr_latency_ms=12.0,
        stage1_latency_ms=40.0,
    )
    assert outcome.relevant_video_rank == 2
    assert outcome.video_hits == {1: False, 5: True, 10: True, 20: True}
    assert outcome.gate_survived is True
```

- [ ] **Step 2: Write failing temporal-overlap tests**

```python
def test_segment_overlap_requires_same_video_and_intersecting_time_ranges():
    labeled = record(relevant_video_ids=("V2",), start_sec=151.0, end_sec=159.0)
    matching = gate_with_segment("V2", 158.0, 162.0)
    wrong_video = gate_with_segment("V1", 151.0, 159.0)
    assert score(labeled, matching).segment_overlap is True
    assert score(labeled, wrong_video).segment_overlap is False


def test_missing_time_label_is_excluded_not_scored_false():
    assert score(record(start_sec=None, end_sec=None), gate("V1")).segment_overlap is None
```

Two closed intervals overlap when `max(label_start, evidence_start) <= min(label_end, evidence_end)`.

- [ ] **Step 3: Write failing paired keyframe metric tests**

```python
def test_keyframe_metrics_use_exact_video_and_absolute_frame_idx():
    labeled = record(relevant_answers=(RelevantAnswer("V2", 12345),))
    outcome = score_asr_query(
        labeled,
        gate("V2"),
        text_only_pool=pool(hit("V1", 12345), hit("V2", 12345)),
        gated_pool=pool(hit("V2", 999), hit("V2", 12345)),
        asr_latency_ms=10.0,
        stage1_latency_ms=30.0,
    )
    assert outcome.text_only_rank == 2
    assert outcome.text_asr_rank == 2


def test_records_without_answers_do_not_enter_keyframe_denominator():
    metrics = aggregate_asr_outcomes(
        [score(record(relevant_answers=()), gate("V1"))],
        video_recall_ks=(1, 5, 10, 20),
        keyframe_recall_ks=(1, 5, 20, 50, 100),
        configured_gate_k=20,
    )
    assert metrics.overall.keyframe_query_count == 0
    assert metrics.overall.text_only_aic_final_score is None
    assert metrics.overall.text_asr_aic_final_score is None
```

- [ ] **Step 4: Run metric tests and confirm RED**

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_asr_eval.py -q
```

Expected: new scoring/aggregation names are not defined.

- [ ] **Step 5: Implement per-query outcome scoring without timestamp-to-frame conversion**

```python
def _first_answer_rank(
    hits: Sequence[SearchHit], answers: frozenset[tuple[str, int]]
) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if (hit.video_id, hit.frame_idx) in answers:
            return rank
    return None


def _recall_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k
```

`score_asr_query()` must never inspect keyframe timestamps to manufacture an answer. Use gate evidence timestamps only for Segment Overlap.

- [ ] **Step 6: Implement aggregation with explicit denominators and utility splits**

For each overall/utility group, report:

```python
@dataclass(frozen=True)
class UtilityMetrics:
    query_count: int
    distinct_relevant_video_count: int
    video_recall_at: dict[int, float | None]
    gate_survival_rate: float | None
    temporal_query_count: int
    segment_overlap_rate: float | None
    keyframe_query_count: int
    text_only_recall_at: dict[int, float | None]
    text_asr_recall_at: dict[int, float | None]
    text_only_aic_final_score: float | None
    text_asr_aic_final_score: float | None
    text_only_median_rank: float | None
    text_asr_median_rank: float | None
    asr_latency_p50_ms: float | None
    asr_latency_p95_ms: float | None
    stage1_latency_p50_ms: float | None
    stage1_latency_p95_ms: float | None
```

Compute AIC Final Score as the mean of Recall@`[1, 5, 20, 50, 100]`. Use `None` for any metric whose eligible denominator is zero. Build `by_utility` with all three keys even when a group has zero records.

- [ ] **Step 7: Run pure metric tests and style checks**

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_asr_eval.py -q
./.venv/Scripts/python.exe -m ruff check eval/asr_eval.py tests/test_asr_eval.py
./.venv/Scripts/python.exe -m black --check eval/asr_eval.py tests/test_asr_eval.py
```

Expected: PASS.

- [ ] **Step 8: Commit pure metrics**

```powershell
git add eval/asr_eval.py tests/test_asr_eval.py
git commit -m "feat: score ASR gate evaluation outcomes"
```

---

### Task 3: Capture ASR Model and Index Identity for Reproducibility

**Files:**
- Modify: `app/stores/mongo_client.py:189-200`
- Modify: `tests/test_mongo_client.py`
- Modify: `eval/asr_run.py`
- Create: `tests/test_asr_run.py`

**Interfaces:**
- Produces: `MongoStore.get_asr_model_ids() -> set[str]` using MongoDB `distinct("model_id")`.
- Produces: `AsrRunIdentity(index_name: str, configured_model_id: str, stored_model_ids: tuple[str, ...], search_config_sha256: str)`.
- Rejects: empty or mixed stored model identities before executing queries.

- [ ] **Step 1: Write a failing Mongo identity test**

Extend the existing fake collection before adding the test:

```python
class _FakeKeyframesCollection:
    def __init__(self, docs: Sequence[dict[str, Any]]) -> None:
        # Keep existing fields.
        self.distinct_result: list[str] = []
        self.distinct_calls: list[tuple[str, dict]] = []

    async def distinct(self, key: str, filter: dict) -> list[str]:
        self.distinct_calls.append((key, filter))
        return list(self.distinct_result)
```

```python
async def test_get_asr_model_ids_reads_distinct_nonblank_values(monkeypatch):
    store = _store(monkeypatch, [])
    store.asr_runs.distinct_result = ["models/PhoWhisper-large-ct2", ""]
    assert await store.get_asr_model_ids() == {"models/PhoWhisper-large-ct2"}
    assert store.asr_runs.distinct_calls == [("model_id", {})]
```

- [ ] **Step 2: Implement the read-only identity lookup**

```python
async def get_asr_model_ids(self) -> set[str]:
    values = await self.asr_runs.distinct("model_id", {})
    return {str(value).strip() for value in values if str(value).strip()}
```

- [ ] **Step 3: Write failing runner identity tests**

Define the read-only fake used by both identity tests:

```python
class FakeMongoStore:
    def __init__(self, model_ids: set[str]) -> None:
        self.model_ids = model_ids

    async def get_asr_model_ids(self) -> set[str]:
        return set(self.model_ids)
```

```python
async def test_resolve_identity_rejects_mixed_asr_models():
    mongo = FakeMongoStore({"model-a", "model-b"})
    with pytest.raises(ValueError, match="mixed ASR model identities"):
        await resolve_asr_identity(
            mongo_store=mongo,
            asr_index_name="asr_segments",
            configured_model_id="model-a",
            search_config_bytes=b"default_asr_video_candidate_k: 20\n",
        )


async def test_resolve_identity_records_index_model_and_config_hash():
    identity = await resolve_asr_identity(
        FakeMongoStore({"model-a"}),
        "asr_segments",
        "model-a",
        b"search config",
    )
    assert identity.index_name == "asr_segments"
    assert identity.stored_model_ids == ("model-a",)
    assert identity.search_config_sha256 == hashlib.sha256(b"search config").hexdigest()
```

- [ ] **Step 4: Implement identity resolution in the standalone runner module**

```python
@dataclass(frozen=True)
class AsrRunIdentity:
    index_name: str
    configured_model_id: str
    stored_model_ids: tuple[str, ...]
    search_config_sha256: str


async def resolve_asr_identity(
    mongo_store: MongoStore,
    asr_index_name: str,
    configured_model_id: str,
    search_config_bytes: bytes,
) -> AsrRunIdentity:
    stored = tuple(sorted(await mongo_store.get_asr_model_ids()))
    if len(stored) != 1:
        raise ValueError(f"mixed ASR model identities or no identity: {stored}")
    if stored[0] != configured_model_id:
        raise ValueError(
            f"ASR model mismatch: configured={configured_model_id!r}, stored={stored[0]!r}"
        )
    return AsrRunIdentity(
        index_name=asr_index_name,
        configured_model_id=configured_model_id,
        stored_model_ids=stored,
        search_config_sha256=hashlib.sha256(search_config_bytes).hexdigest(),
    )
```

- [ ] **Step 5: Run store/identity tests and formatting checks**

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_mongo_client.py tests/test_asr_run.py -q
./.venv/Scripts/python.exe -m ruff check app/stores/mongo_client.py eval/asr_run.py tests/test_mongo_client.py tests/test_asr_run.py
./.venv/Scripts/python.exe -m black --check app/stores/mongo_client.py eval/asr_run.py tests/test_mongo_client.py tests/test_asr_run.py
```

Expected: PASS.

- [ ] **Step 6: Commit identity provenance**

```powershell
git add app/stores/mongo_client.py eval/asr_run.py tests/test_mongo_client.py tests/test_asr_run.py
git commit -m "feat: capture ASR evaluation identity"
```

---

### Task 4: Build the Standalone Timed Runner and Auditable Report Bundle

**Files:**
- Create: `configs/eval/asr_gate_v1.yaml`
- Modify: `eval/asr_run.py`
- Modify: `tests/test_asr_run.py`

**Interfaces:**
- Produces: `AsrEvalRunConfig.from_mapping(mapping) -> AsrEvalRunConfig`.
- Produces: `run_asr_evaluation(config, services, records) -> tuple[list[AsrQueryOutcome], AsrEvalMetrics]`.
- Produces: `write_asr_report(output_dir, config, identity, artifact_hash, outcomes, metrics)` writing `metrics.json`, `report.md`, and `per_query.csv`.
- Produces: `_run_validated(config, records, artifact_hash) -> int`, which is the only boundary allowed to construct live services.
- Produces: `main_for_args(argv: Sequence[str]) -> int` for tests and `main() -> None` for `python -m`.
- Produces: CLI `python -m eval.asr_run --config configs/eval/asr_gate_v1.yaml`.

- [ ] **Step 1: Create exact standalone runner configuration**

```yaml
eval_set_version: asr_gate_v1
eval_set_path: configs/eval/asr_gate_v1.json
eval_set_sha256_path: configs/eval/asr_gate_v1.json.sha256
search_config_path: configs/search/search.yaml
asr_config_path: configs/ingest/asr.yaml
reports_dir: reports/asr
video_recall_ks: [1, 5, 10, 20]
keyframe_recall_ks: [1, 5, 20, 50, 100]
asr_video_top_k: 20
text_candidate_k: 300
ocr_candidate_k: 100
```

The config deliberately contains no OCR query field because the approved record contract compares Text-only against Text+ASR. OCR remains available in runtime but is not silently introduced into this paired ablation.

- [ ] **Step 2: Write failing fail-fast tests before service construction**

Define the config writer used by the fail-fast test:

```python
def write_config(tmp_path: Path, eval_set_path: str) -> Path:
    config_path = tmp_path / "asr_eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "eval_set_version": "asr_gate_v1",
                "eval_set_path": eval_set_path,
                "eval_set_sha256_path": f"{eval_set_path}.sha256",
                "search_config_path": "configs/search/search.yaml",
                "asr_config_path": "configs/ingest/asr.yaml",
                "reports_dir": str(tmp_path / "reports"),
                "video_recall_ks": [1, 5, 10, 20],
                "keyframe_recall_ks": [1, 5, 20, 50, 100],
                "asr_video_top_k": 20,
                "text_candidate_k": 300,
                "ocr_candidate_k": 100,
            }
        ),
        encoding="utf-8",
    )
    return config_path
```

```python
def test_main_fails_before_building_services_when_eval_artifact_is_missing(tmp_path, monkeypatch):
    config_path = write_config(tmp_path, eval_set_path="missing.json")
    build_services = Mock()
    monkeypatch.setattr(asr_run, "_build_services", build_services)
    assert asr_run.main_for_args(["--config", str(config_path)]) == 2
    build_services.assert_not_called()


def test_artifact_hash_mismatch_fails_before_query_execution(tmp_path):
    artifact = tmp_path / "records.json"
    artifact.write_text("[]", encoding="utf-8")
    hash_path = tmp_path / "records.json.sha256"
    hash_path.write_text("deadbeef", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_records(artifact, hash_path)
```

- [ ] **Step 3: Implement config validation and frozen artifact loading**

```python
@dataclass(frozen=True)
class AsrEvalRunConfig:
    eval_set_version: str
    eval_set_path: Path
    eval_set_sha256_path: Path
    search_config_path: Path
    asr_config_path: Path
    reports_dir: Path
    video_recall_ks: tuple[int, ...]
    keyframe_recall_ks: tuple[int, ...]
    asr_video_top_k: int
    text_candidate_k: int
    ocr_candidate_k: int
```

Resolve relative paths against repository root. `load_frozen_records()` reads artifact bytes once, compares lowercase SHA-256 exactly, parses JSON, and then calls `load_asr_eval_records()`.

Implement the testable CLI boundary so missing/invalid artifacts return 2 before `_build_services()`:

```python
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main_for_args(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    try:
        config = load_run_config(args.config)
        records, artifact_hash = load_frozen_records(
            config.eval_set_path,
            config.eval_set_sha256_path,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("ASR evaluation input rejected: %s", exc)
        return 2
    return asyncio.run(_run_validated(config, records, artifact_hash))


def main() -> None:
    raise SystemExit(main_for_args(sys.argv[1:]))
```

- [ ] **Step 4: Write failing orchestration tests for one gate call and paired Stage 1 calls per record**

Define a concrete record/config and fakes that expose the call contract:

```python
CONFIG = AsrEvalRunConfig(
    eval_set_version="asr_gate_v1",
    eval_set_path=Path("records.json"),
    eval_set_sha256_path=Path("records.json.sha256"),
    search_config_path=Path("configs/search/search.yaml"),
    asr_config_path=Path("configs/ingest/asr.yaml"),
    reports_dir=Path("reports/asr"),
    video_recall_ks=(1, 5, 10, 20),
    keyframe_recall_ks=(1, 5, 20, 50, 100),
    asr_video_top_k=20,
    text_candidate_k=300,
    ocr_candidate_k=100,
)
RECORD = AsrEvalRecord(
    query_id="q1",
    text_query="a dam under rain",
    asr_query="công trình thủy lợi",
    relevant_video_ids=("V1",),
    relevant_answers=(RelevantAnswer("V1", 10),),
    start_sec=1.0,
    end_sec=2.0,
    asr_utility="essential",
    label_source="independent_manual_review",
    review_status="approved",
)


def run_hit(video_id: str, frame_idx: int) -> SearchHit:
    return SearchHit(
        keyframe_id=f"{video_id}_f{frame_idx}",
        video_id=video_id,
        score=1.0,
        frame_idx=frame_idx,
        timestamp=1.5,
        thumbnail_path="",
    )


def run_pool(*hits: SearchHit, asr_gate: AsrGateResult | None = None) -> KisCandidatePool:
    return KisCandidatePool(
        candidate_ids=[item.keyframe_id for item in hits],
        text_candidate_count=len(hits),
        ocr_candidate_count=0,
        results=list(hits),
        asr_gate=asr_gate,
    )


def run_gate() -> AsrGateResult:
    return AsrGateResult(
        query=RECORD.asr_query,
        segment_hit_count=1,
        videos=[
            AsrVideoEvidence(
                video_id="V1",
                video_rank=1,
                best_bm25_score=1.0,
                segments=[AsrEvidenceSegment("s1", "evidence", 1.0, 2.0, 1.0)],
            )
        ],
    )


class FakeTimedAsrService:
    last_latency_ms: float | None = None


class FakeSearchService:
    def __init__(self, timer: FakeTimedAsrService) -> None:
        self.timer = timer
        self.calls: list[dict[str, object]] = []

    async def retrieve_kis_candidates(self, **kwargs) -> KisCandidatePool:
        self.calls.append(dict(kwargs))
        if kwargs["asr_query"] is None:
            return run_pool(run_hit("V1", 10))
        self.timer.last_latency_ms = 1.0
        return run_pool(run_hit("V1", 10), asr_gate=run_gate())


class FakeServices:
    def __init__(self) -> None:
        self.timed_asr_service = FakeTimedAsrService()
        self.search_service = FakeSearchService(self.timed_asr_service)
```

```python
async def test_runner_executes_text_only_and_text_plus_asr_once_per_record():
    services = FakeServices()
    outcomes, _ = await run_asr_evaluation(CONFIG, services, [RECORD])
    assert services.search_service.calls == [
        {
            "text_query": RECORD.text_query,
            "text_top_k": 300,
            "ocr_query": None,
            "ocr_top_k": 100,
            "asr_query": None,
            "asr_video_top_k": 20,
        },
        {
            "text_query": RECORD.text_query,
            "text_top_k": 300,
            "ocr_query": None,
            "ocr_top_k": 100,
            "asr_query": RECORD.asr_query,
            "asr_video_top_k": 20,
        },
    ]
    assert outcomes[0].asr_latency_ms >= 0
    assert outcomes[0].stage1_latency_ms >= outcomes[0].asr_latency_ms
```

- [ ] **Step 5: Implement a timing proxy so active Stage 1 queries ASR only once**

```python
class TimedAsrRetrievalService:
    def __init__(self, delegate: AsrRetrievalService) -> None:
        self._delegate = delegate
        self.last_latency_ms: float | None = None

    async def retrieve(self, query: str, video_top_k: int) -> AsrGateResult:
        start = time.perf_counter()
        try:
            return await self._delegate.retrieve(query, video_top_k)
        finally:
            self.last_latency_ms = (time.perf_counter() - start) * 1000
```

Construct `SearchService` with this proxy. For each record, call Text-only once, reset `last_latency_ms`, time the complete Text+ASR Stage 1 call once, and read ASR-only latency from the proxy. Never issue a separate warm-up query that changes the measured call count.

- [ ] **Step 6: Write failing report tests for provenance, eligibility counts, utility splits, and not-measured values**

```python
def test_report_bundle_contains_provenance_and_explicit_denominators(tmp_path):
    identity = AsrRunIdentity(
        index_name="asr_segments",
        configured_model_id="model-a",
        stored_model_ids=("model-a",),
        search_config_sha256="config-hash",
    )
    outcome = score_asr_query(
        RECORD,
        run_gate(),
        text_only_pool=run_pool(run_hit("V1", 10)),
        gated_pool=run_pool(run_hit("V1", 10), asr_gate=run_gate()),
        asr_latency_ms=1.0,
        stage1_latency_ms=2.0,
    )
    report_metrics = aggregate_asr_outcomes(
        [outcome],
        video_recall_ks=CONFIG.video_recall_ks,
        keyframe_recall_ks=CONFIG.keyframe_recall_ks,
        configured_gate_k=CONFIG.asr_video_top_k,
    )
    write_asr_report(tmp_path, CONFIG, identity, "abc123", [outcome], report_metrics)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["eval_set_version"] == "asr_gate_v1"
    assert metrics["eval_set_sha256"] == "abc123"
    assert metrics["identity"]["index_name"] == "asr_segments"
    assert metrics["overall"]["query_count"] == 1
    assert metrics["overall"]["keyframe_query_count"] == 1
    assert metrics["by_utility"]["supportive"]["text_asr_aic_final_score"] is None
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "per_query.csv").exists()
```

- [ ] **Step 7: Implement report serialization and timestamped output directory**

Write JSON with stable sorted keys, a concise Markdown table for overall and utility splits, and one CSV row per query containing label eligibility, ranks, hits, overlap, and latencies. Create output directories as:

```python
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
git_sha = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
output_dir = config.reports_dir / f"{stamp}_{git_sha}"
```

The Markdown heading must say `not measured` when an eligible denominator is zero; never render `0.000` for missing labels.

- [ ] **Step 8: Build production services only after artifact validation**

Reuse the repository's FG-CLIP, translator, Milvus, OCR ES, ASR ES, and Mongo configuration paths. Do not import `app.main.app` because module import constructs the FastAPI object and couples the CLI to application startup; call focused builder functions or extract shared builders without changing their runtime behavior. Close both Elasticsearch clients and MongoDB in `finally`.

```python
async def _run_validated(
    config: AsrEvalRunConfig,
    records: Sequence[AsrEvalRecord],
    artifact_hash: str,
) -> int:
    services = await _build_services(config)
    try:
        outcomes, metrics = await run_asr_evaluation(config, services, records)
        output_dir = create_output_dir(config.reports_dir)
        write_asr_report(
            output_dir,
            config,
            services.identity,
            artifact_hash,
            outcomes,
            metrics,
        )
        logger.info("ASR evaluation report written to %s", output_dir)
        return 0
    finally:
        await services.close()
```

- [ ] **Step 9: Run standalone runner tests and style checks**

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_asr_eval.py tests/test_asr_run.py tests/test_mongo_client.py -q
./.venv/Scripts/python.exe -m ruff check eval/asr_eval.py eval/asr_run.py app/stores/mongo_client.py tests/test_asr_eval.py tests/test_asr_run.py tests/test_mongo_client.py
./.venv/Scripts/python.exe -m black --check eval/asr_eval.py eval/asr_run.py app/stores/mongo_client.py tests/test_asr_eval.py tests/test_asr_run.py tests/test_mongo_client.py
```

Expected: PASS without stores, GPU, or real labels because tests use fakes and the test fixture.

- [ ] **Step 10: Commit the standalone runner**

```powershell
git add configs/eval/asr_gate_v1.yaml eval/asr_run.py tests/test_asr_run.py
git commit -m "feat: add standalone ASR evaluation runner"
```

---

### Task 5: Document the Manual Freeze-and-Run Workflow and Verify the Optional Plan

**Files:**
- Modify: `README.md:108-156`
- Modify: `docs/PRD.md:320-458,585-609,641-657`

**Interfaces:**
- Consumes: the contract, runner, and report bundle from Tasks 1–4.
- Produces: an owner-operated workflow that cannot be mistaken for an automatically measured result.

- [ ] **Step 1: Document label review and freeze commands**

Document that the owner first creates/reviews `configs/eval/asr_gate_v1.json`, then freezes the exact bytes with:

```powershell
$hash = (Get-FileHash 'configs/eval/asr_gate_v1.json' -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -Path 'configs/eval/asr_gate_v1.json.sha256' -Value $hash -Encoding utf8NoBOM
```

State that changing any frozen record requires a new eval-set version and a paired rerun; the file must not be edited after seeing new metrics under the same version.

- [ ] **Step 2: Document the resource-bearing manual runner command**

```powershell
./.venv/Scripts/python.exe -m eval.asr_run --config configs/eval/asr_gate_v1.yaml
```

State prerequisites: populated `asr_segments`, matching `asr_runs.model_id`, populated `keyframes_fgclip_base` and Mongo keyframe metadata, local FG-CLIP checkpoint, Elasticsearch/Milvus/MongoDB running, and sufficient RTX 3070 8 GB availability for the existing online encoder. The owner runs this command; Codex does not launch it automatically.

- [ ] **Step 3: Document report interpretation boundaries**

Record these exact rules:

```text
Video metrics use all approved records.
Segment Overlap uses only records with reviewed start_sec/end_sec.
Keyframe/AIC metrics use only records with reviewed relevant_answers.
A missing eligibility subset is “not measured,” not zero.
Operational impressions are not a retrieval-quality claim.
```

- [ ] **Step 4: Run all lightweight evaluation checks**

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_asr_eval.py tests/test_asr_run.py tests/test_mongo_client.py -q
./.venv/Scripts/python.exe -m ruff check eval/asr_eval.py eval/asr_run.py app/stores/mongo_client.py tests/test_asr_eval.py tests/test_asr_run.py tests/test_mongo_client.py
./.venv/Scripts/python.exe -m black --check eval/asr_eval.py eval/asr_run.py app/stores/mongo_client.py tests/test_asr_eval.py tests/test_asr_run.py tests/test_mongo_client.py
git diff --check
```

Expected: all pass without real services or labels.

- [ ] **Step 5: Verify fail-fast behavior with the intentionally absent owner artifact**

```powershell
./.venv/Scripts/python.exe -m eval.asr_run --config configs/eval/asr_gate_v1.yaml
```

Expected before owner labeling: exit code 2 and an actionable message naming missing `configs/eval/asr_gate_v1.json`; no encoder/store construction and no report directory creation.

- [ ] **Step 6: Audit final scope without running the full evaluation**

```powershell
git status --short
git diff --stat
git diff -- eval/asr_eval.py eval/asr_run.py configs/eval/asr_gate_v1.yaml app/stores/mongo_client.py tests/test_asr_eval.py tests/test_asr_run.py tests/test_mongo_client.py README.md docs/PRD.md
```

Confirm manually:

- no real or generated relevance label was invented;
- the runner is independent of `eval.run`;
- absent/mismatched artifacts fail before expensive setup;
- report provenance includes artifact/config/model/index identity;
- every metric carries its eligible denominator;
- the repository still says ASR quality is “not measured” until the owner runs the frozen artifact.

- [ ] **Step 7: Commit evaluation documentation**

```powershell
git add README.md docs/PRD.md
git commit -m "docs: add ASR evaluation protocol"
```

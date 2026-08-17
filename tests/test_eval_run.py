"""RED reproducer for eval/run.py (plan §P0.9) -- the single evaluation
entry point. Tests the pure/semi-pure orchestration functions with fake
stores/searchers (no Docker, no GPU, no real Milvus/ES/Mongo) -- the actual
`python -m eval.run --config configs/eval/aic_pilot.yaml` run against real
data is a manual step (CLAUDE.md §10), not exercised here.
"""

from __future__ import annotations

from typing import Any

import pytest

from datasets.base import GroundTruthSegment
from eval.run import (
    EvalRunConfig,
    PerQueryResult,
    _load_config,
    aggregate_channel,
    build_keyframe_pool,
    build_leakage_records,
    compute_channel_metrics,
    filter_ground_truth_to_available,
    keyframes_in_window,
    load_pilot_available,
    score_channel,
    write_report,
)


def _gt(
    video_id="L21_V001", start=10.0, end=12.0, order_index=0, caption="x"
) -> GroundTruthSegment:
    return GroundTruthSegment(
        video_id=video_id, start_sec=start, end_sec=end, order_index=order_index, caption=caption
    )


# --------------------------------------------------------------------------
# load_pilot_available / filter_ground_truth_to_available
# --------------------------------------------------------------------------


def test_load_pilot_available_reads_nonblank_lines_as_a_set(tmp_path):
    path = tmp_path / "pilot_available_v1.txt"
    path.write_text("L21_V001\nL21_V002\n\n", encoding="utf-8")

    assert load_pilot_available(path) == frozenset({"L21_V001", "L21_V002"})


def test_load_pilot_available_raises_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pilot_available(tmp_path / "missing.txt")


def test_filter_ground_truth_to_available_drops_unavailable_video_ids():
    segments = [_gt(video_id="L21_V001"), _gt(video_id="L21_V999")]

    result = filter_ground_truth_to_available(segments, frozenset({"L21_V001"}))

    assert [s.video_id for s in result] == ["L21_V001"]


# --------------------------------------------------------------------------
# score_channel -- drives a searcher (SearchService/RandomBaseline/
# BM25Baseline all share the async `search(query, top_k) -> hits` shape)
# --------------------------------------------------------------------------


class _FakeSearcher:
    def __init__(self, hits: list[Any]) -> None:
        self._hits = hits
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return self._hits


class _Hit:
    def __init__(self, video_id: str, timestamp: float) -> None:
        self.video_id = video_id
        self.timestamp = timestamp


async def test_score_channel_computes_video_and_segment_rank_per_query():
    searcher = _FakeSearcher([_Hit("L21_V999", 0.0), _Hit("L21_V001", 11.0)])
    queries = [_gt(video_id="L21_V001", start=10.0, end=12.0)]

    result = await score_channel(searcher, queries, top_k=5)

    assert len(result) == 1
    gt, rank_video, rank_segment = result[0]
    assert gt.video_id == "L21_V001"
    assert rank_video == 2
    assert rank_segment == 2


async def test_score_channel_accepts_dict_shaped_hits_from_es():
    # BM25Baseline/EsStore hits are raw dicts, not dataclasses.
    searcher = _FakeSearcher([{"video_id": "L21_V001", "timestamp": 11.0}])
    queries = [_gt(video_id="L21_V001", start=10.0, end=12.0)]

    result = await score_channel(searcher, queries, top_k=5)

    assert result[0][1] == 1  # rank_video
    assert result[0][2] == 1  # rank_segment


async def test_score_channel_calls_searcher_with_each_query_caption_and_top_k():
    searcher = _FakeSearcher([])
    queries = [_gt(video_id="L21_V001", caption="chay rung")]

    await score_channel(searcher, queries, top_k=7)

    assert searcher.calls == [("chay rung", 7)]


async def test_score_channel_sends_the_transformed_query_when_one_is_given():
    # The VI->EN sidecar path: the searcher must receive the English text.
    searcher = _FakeSearcher([])
    queries = [_gt(video_id="L21_V001", caption="chay rung")]

    await score_channel(searcher, queries, top_k=7, query_transform=lambda gt: "forest fire")

    assert searcher.calls == [("forest fire", 7)]


async def test_score_channel_still_scores_against_the_untransformed_ground_truth():
    # Only the QUERY is translated -- video_id/start_sec/end_sec must keep
    # deciding the hit, so a translated run stays comparable to the baseline.
    searcher = _FakeSearcher([_Hit("L21_V001", 11.0)])
    queries = [_gt(video_id="L21_V001", start=10.0, end=12.0, caption="chay rung")]

    result = await score_channel(
        searcher, queries, top_k=5, query_transform=lambda gt: "forest fire"
    )

    gt, rank_video, rank_segment = result[0]
    assert gt.caption == "chay rung"  # ground truth untouched
    assert rank_video == 1
    assert rank_segment == 1


# --------------------------------------------------------------------------
# compute_channel_metrics
# --------------------------------------------------------------------------


def test_compute_channel_metrics_reports_recall_at_every_configured_k():
    metrics = compute_channel_metrics(
        "semantic", "segment", [1, None, 30], top_ks=(1, 5, 20, 50, 100)
    )

    assert metrics.recall_at_k[1] == pytest.approx(1 / 3)
    assert metrics.recall_at_k[50] == pytest.approx(2 / 3)
    assert metrics.num_queries == 3
    assert metrics.num_found == 2


# --------------------------------------------------------------------------
# aggregate_channel -- video+segment metrics AND author/batch breakdown
# --------------------------------------------------------------------------


def test_aggregate_channel_reports_recall_by_author_and_batch():
    scored = [
        (_gt(video_id="L21_V001"), 1, 1),
        (_gt(video_id="L22_V001"), None, None),
    ]
    author_map = {"L21_V001": "author_a", "L22_V001": "author_b"}

    video_m, segment_m, by_author, by_batch = aggregate_channel(
        "semantic", scored, top_ks=(1, 5, 20, 50, 100), author_map=author_map, breakdown_k=1
    )

    assert video_m.level == "video"
    assert segment_m.level == "segment"
    assert by_author == {"author_a": pytest.approx(1.0), "author_b": pytest.approx(0.0)}
    assert by_batch == {"L21": pytest.approx(1.0), "L22": pytest.approx(0.0)}


# --------------------------------------------------------------------------
# keyframes_in_window / build_leakage_records
# --------------------------------------------------------------------------


def test_keyframes_in_window_filters_by_inclusive_timestamp_range():
    docs = [
        {"keyframe_id": "a", "timestamp": 9.0},
        {"keyframe_id": "b", "timestamp": 10.5},
        {"keyframe_id": "c", "timestamp": 13.0},
    ]

    assert keyframes_in_window(docs, start_sec=10.0, end_sec=12.0) == ["b"]


class _FakeMongoStore:
    def __init__(self, keyframes_by_video: dict[str, list[dict]]) -> None:
        self._keyframes_by_video = keyframes_by_video
        self.calls: list[str] = []

    async def get_keyframes_for_video(self, video_id: str) -> list[dict]:
        self.calls.append(video_id)
        return self._keyframes_by_video.get(video_id, [])


class _FakeEsStore:
    def __init__(self, ocr_by_id: dict[str, str]) -> None:
        self._ocr_by_id = ocr_by_id

    async def get_by_ids(self, keyframe_ids):
        return {kid: self._ocr_by_id[kid] for kid in keyframe_ids if kid in self._ocr_by_id}


async def test_build_leakage_records_joins_mongo_window_with_es_ocr_text():
    mongo = _FakeMongoStore({"L21_V001": [{"keyframe_id": "k1", "timestamp": 11.0}]})
    es = _FakeEsStore({"k1": "tin tuc"})
    ground_truth = [_gt(video_id="L21_V001", start=10.0, end=12.0, caption="chay rung")]

    records = await build_leakage_records(ground_truth, mongo, es)

    assert len(records) == 1
    assert records[0].caption == "chay rung"
    assert records[0].ocr_texts == ["tin tuc"]


async def test_build_leakage_records_caches_mongo_lookup_per_video():
    mongo = _FakeMongoStore({"L21_V001": [{"keyframe_id": "k1", "timestamp": 11.0}]})
    es = _FakeEsStore({})
    ground_truth = [
        _gt(video_id="L21_V001", start=10.0, end=12.0, order_index=0),
        _gt(video_id="L21_V001", start=10.0, end=12.0, order_index=1),
    ]

    await build_leakage_records(ground_truth, mongo, es)

    assert mongo.calls == ["L21_V001"]  # not called twice for the same video


# --------------------------------------------------------------------------
# build_keyframe_pool
# --------------------------------------------------------------------------


class _FakeScrollMongoStore:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    async def scroll_all_keyframes(self):
        for doc in self._docs:
            yield doc


async def test_build_keyframe_pool_converts_every_scrolled_document():
    mongo = _FakeScrollMongoStore([{"keyframe_id": "a", "video_id": "v1", "timestamp": 1.0}])

    pool = await build_keyframe_pool(mongo)

    assert len(pool) == 1
    assert pool[0].keyframe_id == "a"


# --------------------------------------------------------------------------
# write_report
# --------------------------------------------------------------------------


def test_write_report_creates_metrics_json_report_md_and_per_query_csv(tmp_path):
    video_metrics = compute_channel_metrics(
        "semantic", "video", [1, None], top_ks=(1, 5, 20, 50, 100)
    )
    segment_metrics = compute_channel_metrics(
        "semantic", "segment", [1, None], top_ks=(1, 5, 20, 50, 100)
    )
    per_query = [
        PerQueryResult(
            video_id="L21_V001",
            order_index=0,
            caption="x",
            batch="L21",
            author="a",
            channel="semantic",
            rank_video=1,
            rank_segment=1,
        )
    ]

    write_report(
        tmp_path,
        git_sha="abc123",
        eval_set_version="v1",
        eval_set_hash="deadbeef",
        num_video_ids=1,
        num_queries=2,
        seed=42,
        channel_metrics=[video_metrics, segment_metrics],
        author_breakdown={"semantic": {"a": 1.0}},
        batch_breakdown={"semantic": {"L21": 1.0}},
        leakage_report=None,
        per_query=per_query,
    )

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "per_query.csv").exists()
    metrics_doc = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "abc123" in metrics_doc
    assert "v1" in metrics_doc


def test_report_md_labels_median_rank_as_rank_in_grid(tmp_path):
    # plan §P1.7, PRD §6.6: "Rank-in-Grid" is the position of the correct
    # hit in the grid the operator has to scroll to -- exactly what
    # median_rank already aggregates (first_hit_rank per query). This test
    # locks in that the report makes that mapping explicit by NAME, not just
    # by number, so a reader of report.md can trace it back to PRD §6.6
    # without reading eval/run.py's source.
    video_metrics = compute_channel_metrics("semantic", "video", [1, 3], top_ks=(1, 5, 20, 50, 100))
    segment_metrics = compute_channel_metrics(
        "semantic", "segment", [1, 3], top_ks=(1, 5, 20, 50, 100)
    )

    write_report(
        tmp_path,
        git_sha="abc123",
        eval_set_version="v1",
        eval_set_hash="deadbeef",
        num_video_ids=1,
        num_queries=2,
        seed=42,
        channel_metrics=[video_metrics, segment_metrics],
        author_breakdown={},
        batch_breakdown={},
        leakage_report=None,
        per_query=[],
    )

    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Rank-in-Grid" in report_md


def test_write_report_appends_trake_section_when_trake_groups_given(tmp_path):
    # plan §P2.3: score_trake: true adds a TRAKE section to the SAME
    # report.md/metrics.json a KIS/OCR run already writes -- one report per
    # eval/run.py invocation, not a second file to keep in sync.
    from eval.trake_run import TrakeGroupMetrics

    video_metrics = compute_channel_metrics(
        "semantic", "video", [1, None], top_ks=(1, 5, 20, 50, 100)
    )
    segment_metrics = compute_channel_metrics(
        "semantic", "segment", [1, None], top_ks=(1, 5, 20, 50, 100)
    )
    trake_groups = [
        TrakeGroupMetrics(
            sequence_length=3,
            num_queries=1,
            num_found=1,
            sequence_exact_match=1.0,
            video_level_recall_at_k=1.0,
            per_event_recall_at_k=(1.0, 1.0, 1.0),
            mean_kendall_tau=1.0,
            mean_offset_sec=0.0,
            wall_clock_sec=0.1,
        )
    ]

    write_report(
        tmp_path,
        git_sha="abc123",
        eval_set_version="v1",
        eval_set_hash="deadbeef",
        num_video_ids=1,
        num_queries=2,
        seed=42,
        channel_metrics=[video_metrics, segment_metrics],
        author_breakdown={},
        batch_breakdown={},
        leakage_report=None,
        per_query=[],
        trake_groups=trake_groups,
    )

    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "TRAKE metrics" in report_md
    metrics_doc = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "sequence_exact_match" in metrics_doc


def test_write_report_omits_trake_section_when_trake_groups_not_given(tmp_path):
    video_metrics = compute_channel_metrics("semantic", "video", [1], top_ks=(1, 5, 20, 50, 100))
    segment_metrics = compute_channel_metrics(
        "semantic", "segment", [1], top_ks=(1, 5, 20, 50, 100)
    )

    write_report(
        tmp_path,
        git_sha="abc123",
        eval_set_version="v1",
        eval_set_hash="deadbeef",
        num_video_ids=1,
        num_queries=1,
        seed=42,
        channel_metrics=[video_metrics, segment_metrics],
        author_breakdown={},
        batch_breakdown={},
        leakage_report=None,
        per_query=[],
    )

    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "TRAKE metrics" not in report_md


def test_write_report_appends_trake_ablation_section_when_cells_given(tmp_path):
    # plan §P2.4: score_trake_ablation: true adds a λ-ablation section to the
    # SAME report.md/metrics.json, same pattern as trake_groups (P2.3).
    from eval.trake_run import TrakeAblationCell, TrakeGroupMetrics

    video_metrics = compute_channel_metrics("semantic", "video", [1], top_ks=(1, 5, 20, 50, 100))
    segment_metrics = compute_channel_metrics(
        "semantic", "segment", [1], top_ks=(1, 5, 20, 50, 100)
    )
    ablation_cells = [
        TrakeAblationCell(
            lambda_penalty=0.0,
            metrics=TrakeGroupMetrics(
                sequence_length=3,
                num_queries=1,
                num_found=1,
                sequence_exact_match=1.0,
                video_level_recall_at_k=1.0,
                per_event_recall_at_k=(1.0, 1.0, 1.0),
                mean_kendall_tau=1.0,
                mean_offset_sec=0.0,
                wall_clock_sec=0.1,
            ),
        )
    ]

    write_report(
        tmp_path,
        git_sha="abc123",
        eval_set_version="v1",
        eval_set_hash="deadbeef",
        num_video_ids=1,
        num_queries=1,
        seed=42,
        channel_metrics=[video_metrics, segment_metrics],
        author_breakdown={},
        batch_breakdown={},
        leakage_report=None,
        per_query=[],
        trake_ablation_cells=ablation_cells,
    )

    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "DANTE λ ablation" in report_md
    metrics_doc = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "lambda_penalty" in metrics_doc


def test_write_report_omits_trake_ablation_section_when_cells_not_given(tmp_path):
    video_metrics = compute_channel_metrics("semantic", "video", [1], top_ks=(1, 5, 20, 50, 100))
    segment_metrics = compute_channel_metrics(
        "semantic", "segment", [1], top_ks=(1, 5, 20, 50, 100)
    )

    write_report(
        tmp_path,
        git_sha="abc123",
        eval_set_version="v1",
        eval_set_hash="deadbeef",
        num_video_ids=1,
        num_queries=1,
        seed=42,
        channel_metrics=[video_metrics, segment_metrics],
        author_breakdown={},
        batch_breakdown={},
        leakage_report=None,
        per_query=[],
    )

    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "DANTE λ ablation" not in report_md


# --------------------------------------------------------------------------
# _load_config -- P2.4 fields
# --------------------------------------------------------------------------


def test_load_config_reads_trake_ablation_fields(tmp_path):
    config_path = tmp_path / "trake_ablation.yaml"
    config_path.write_text(
        "eval_set_path: configs/eval/aic_eval_v1.json\n"
        "pilot_available_path: configs/eval/pilot_available_v1.txt\n"
        "score_trake_ablation: true\n"
        "trake_ablation_lambda_values: [0.0, 0.001, 0.01]\n",
        encoding="utf-8",
    )

    config = _load_config(config_path)

    assert config.score_trake_ablation is True
    assert config.trake_ablation_lambda_values == (0.0, 0.001, 0.01)


def test_write_report_appends_vqa_section_when_vqa_metrics_given(tmp_path):
    # plan §P3.4: score_vqa: true adds a VQA section to the SAME
    # report.md/metrics.json a KIS/OCR run already writes.
    from eval.vqa_run import VqaGroupMetrics, VqaReportMetrics

    video_metrics = compute_channel_metrics(
        "semantic", "video", [1, None], top_ks=(1, 5, 20, 50, 100)
    )
    segment_metrics = compute_channel_metrics(
        "semantic", "segment", [1, None], top_ks=(1, 5, 20, 50, 100)
    )
    group = VqaGroupMetrics(
        num_queries=1,
        exact_match=1.0,
        token_f1=1.0,
        grounded_accuracy=1.0,
        retrieval_conditioned_accuracy=1.0,
        hallucination_rate=0.0,
        abstention_rate=0.0,
    )
    vqa_metrics = VqaReportMetrics(
        overall=group, answerable_by_ocr=None, not_answerable_by_ocr=group, wall_clock_sec=0.1
    )

    write_report(
        tmp_path,
        git_sha="abc123",
        eval_set_version="v1",
        eval_set_hash="deadbeef",
        num_video_ids=1,
        num_queries=2,
        seed=42,
        channel_metrics=[video_metrics, segment_metrics],
        author_breakdown={},
        batch_breakdown={},
        leakage_report=None,
        per_query=[],
        vqa_metrics=vqa_metrics,
    )

    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "VQA metrics" in report_md
    metrics_doc = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "grounded_accuracy" in metrics_doc


def test_write_report_omits_vqa_section_when_vqa_metrics_not_given(tmp_path):
    video_metrics = compute_channel_metrics("semantic", "video", [1], top_ks=(1, 5, 20, 50, 100))
    segment_metrics = compute_channel_metrics(
        "semantic", "segment", [1], top_ks=(1, 5, 20, 50, 100)
    )

    write_report(
        tmp_path,
        git_sha="abc123",
        eval_set_version="v1",
        eval_set_hash="deadbeef",
        num_video_ids=1,
        num_queries=1,
        seed=42,
        channel_metrics=[video_metrics, segment_metrics],
        author_breakdown={},
        batch_breakdown={},
        leakage_report=None,
        per_query=[],
    )

    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "VQA metrics" not in report_md
    metrics_doc = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert '"vqa": null' in metrics_doc


# --------------------------------------------------------------------------
# _load_config -- P3.4 fields
# --------------------------------------------------------------------------


def test_load_config_reads_vqa_fields(tmp_path):
    config_path = tmp_path / "aic_qa_v1.yaml"
    config_path.write_text(
        "eval_set_path: configs/eval/aic_eval_v1.json\n"
        "pilot_available_path: configs/eval/pilot_available_v1.txt\n"
        "score_vqa: true\n"
        "vqa_eval_set_path: configs/eval/aic_qa_v1.json\n"
        "vqa_top_k: 20\n"
        "vqa_confidence_threshold: 0.7\n",
        encoding="utf-8",
    )

    config = _load_config(config_path)

    assert config.score_vqa is True
    assert config.vqa_eval_set_path == "configs/eval/aic_qa_v1.json"
    assert config.vqa_top_k == 20
    assert config.vqa_confidence_threshold == pytest.approx(0.7)


def test_load_config_defaults_vqa_fields_when_absent(tmp_path):
    config_path = tmp_path / "aic_pilot.yaml"
    config_path.write_text(
        "eval_set_path: configs/eval/aic_eval_v1.json\n"
        "pilot_available_path: configs/eval/pilot_available_v1.txt\n",
        encoding="utf-8",
    )

    config = _load_config(config_path)

    assert config.score_vqa is False


def test_load_config_defaults_trake_ablation_fields_when_absent(tmp_path):
    config_path = tmp_path / "aic_pilot.yaml"
    config_path.write_text(
        "eval_set_path: configs/eval/aic_eval_v1.json\n"
        "pilot_available_path: configs/eval/pilot_available_v1.txt\n",
        encoding="utf-8",
    )

    config = _load_config(config_path)

    assert config.score_trake_ablation is False
    assert config.trake_ablation_lambda_values == EvalRunConfig().trake_ablation_lambda_values

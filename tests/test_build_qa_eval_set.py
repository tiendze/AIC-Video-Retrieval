"""RED reproducer for eval/build_qa_eval_set.py (P3.3, .claude/plans/
p3-vqa-submission.plan.md).

Reuses the ALREADY-SAMPLED, ALREADY-REVIEWED shot universe from
`configs/eval/aic_eval_v1.candidates.json` (P0.4') -- no re-sampling here,
"cung cach lay mau, cung seed" per the plan. Generation is text-only (reads
the shot's existing caption, not the keyframe image -- chu du an decision
2026-08-16: no local vision model, keep VRAM usage low and sequential with
BEiT-3, CLAUDE.md §1.4), via a local Ollama model (references.md §5.6),
deliberately NOT Gemini/OpenAI (self-grading guard, CLAUDE.md §3).

Pure functions tested directly. Ollama is Protocol-faked (tests/test_ocr.py
pattern) for orchestration tests -- no local Ollama server needed to run
this suite.
"""

from __future__ import annotations

import argparse
import json

import pytest
from pydantic import ValidationError

from eval.build_qa_eval_set import (
    BuildQaEvalSetConfig,
    EvalSetBuildError,
    QaGeneration,
    QaItem,
    _freeze_qa_candidates,
    _generate_qa_candidates,
    _load_config,
    _run_freeze,
    _write_qa_candidates_doc,
    assert_generator_differs_from_vqa_model,
    compute_qa_quality_report,
    freeze_qa_records,
    load_approved_source_records,
)


def _source_record(
    video_id: str,
    shot_id: int,
    caption: str = "Một phóng viên đứng trước toà nhà quốc hội đang tường thuật trực tiếp.",
    approved: bool | None = True,
) -> dict:
    return {
        "video_id": video_id,
        "shot_id": shot_id,
        "start_sec": float(shot_id),
        "end_sec": float(shot_id) + 2.0,
        "order_index": shot_id,
        "caption": caption,
        "task": "kis",
        "keyframe_id": f"{video_id}_f{shot_id}",
        "image_path": f"/data/keyframe/{video_id}/{video_id}_{shot_id}.jpg",
        "batch": video_id.split("_V", 1)[0],
        "author": "Author A",
        "auto_filter": "kept",
        "approved": approved,
    }


class TestLoadApprovedSourceRecords:
    def test_keeps_only_approved_true_records(self, tmp_path):
        candidates_path = tmp_path / "aic_eval_v1.candidates.json"
        candidates_path.write_text(
            json.dumps(
                {
                    "records": [
                        _source_record("L21_V001", 1, approved=True),
                        _source_record("L21_V001", 2, approved=False),
                        _source_record("L21_V001", 3, approved=None),
                    ]
                }
            ),
            encoding="utf-8",
        )

        result = load_approved_source_records(candidates_path)

        assert [r["shot_id"] for r in result] == [1]


class TestQaGenerationSchema:
    def test_accepts_one_or_two_items(self):
        one = QaGeneration.model_validate({"items": [{"question": "q", "reference_answer": "a"}]})
        two = QaGeneration.model_validate(
            {
                "items": [
                    {"question": "q1", "reference_answer": "a1"},
                    {"question": "q2", "reference_answer": "a2"},
                ]
            }
        )
        assert len(one.items) == 1
        assert len(two.items) == 2

    def test_rejects_zero_items(self):
        with pytest.raises(ValidationError):
            QaGeneration.model_validate({"items": []})

    def test_rejects_more_than_two_items(self):
        items = [{"question": f"q{i}", "reference_answer": f"a{i}"} for i in range(3)]
        with pytest.raises(ValidationError):
            QaGeneration.model_validate({"items": items})


class TestOllamaQaGeneratorRetry:
    def _generator(self, max_retries=3, retry_base_delay_seconds=0.0):
        from eval.build_qa_eval_set import OllamaQaGenerator

        config = BuildQaEvalSetConfig(
            max_retries=max_retries, retry_base_delay_seconds=retry_base_delay_seconds
        )
        generator = object.__new__(OllamaQaGenerator)  # bypass __init__: no real Ollama server
        object.__setattr__(generator, "_config", config)
        return generator

    def test_retries_on_transient_connection_error_then_succeeds(self, monkeypatch):
        import httpx

        generator = self._generator()
        calls = {"count": 0}

        def _flaky_call_once(caption):
            calls["count"] += 1
            if calls["count"] < 3:
                raise httpx.ConnectError("simulated transient network error")
            return [QaItem(question="q", reference_answer="a")]

        monkeypatch.setattr(generator, "_call_once", _flaky_call_once)

        items = generator.generate("một caption")

        assert [i.question for i in items] == ["q"]
        assert calls["count"] == 3

    def test_raises_runtime_error_after_exhausting_retries(self, monkeypatch):
        import httpx

        generator = self._generator(max_retries=2)

        def _always_times_out(caption):
            raise httpx.TimeoutException("simulated permanent timeout")

        monkeypatch.setattr(generator, "_call_once", _always_times_out)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            generator.generate("một caption")

    def test_does_not_retry_ollama_response_error(self, monkeypatch):
        """A model-not-pulled / invalid-request error from the Ollama server
        is not transient -- retrying it wastes 2*retry_base_delay_seconds for
        nothing. Must propagate immediately (same split as
        build_eval_set.py's GeminiCaptionGenerator: only real
        transport-level failures are retried)."""
        from ollama import ResponseError

        generator = self._generator(max_retries=3)
        calls = {"count": 0}

        def _always_response_error(caption):
            calls["count"] += 1
            raise ResponseError("model 'qwen2.5:7b-instruct' not found")

        monkeypatch.setattr(generator, "_call_once", _always_response_error)

        with pytest.raises(ResponseError):
            generator.generate("một caption")
        assert calls["count"] == 1

    def test_does_not_retry_malformed_json_validation_error(self, monkeypatch):
        generator = self._generator(max_retries=3)
        calls = {"count": 0}

        def _always_malformed(caption):
            calls["count"] += 1
            QaGeneration.model_validate_json("{not valid schema}")  # raises ValidationError

        monkeypatch.setattr(generator, "_call_once", _always_malformed)

        with pytest.raises(ValidationError):
            generator.generate("một caption")
        assert calls["count"] == 1


class TestOllamaQaGeneratorInit:
    def test_init_builds_a_client_for_the_configured_host_without_a_network_call(self):
        """`ollama.Client(host=...)` only builds an httpx client -- it does
        not connect until a method like `.chat()` is called, so this is
        safe to construct without a live Ollama server."""
        from eval.build_qa_eval_set import OllamaQaGenerator

        config = BuildQaEvalSetConfig(ollama_host="http://localhost:11434")

        generator = OllamaQaGenerator(config)

        assert generator._config is config
        assert generator._client is not None


class TestOllamaQaGeneratorCallOnce:
    def test_call_once_sends_caption_and_schema_and_parses_response(self, monkeypatch):
        from eval.build_qa_eval_set import OllamaQaGenerator

        config = BuildQaEvalSetConfig(ollama_model="qwen2.5:7b-instruct")
        generator = object.__new__(OllamaQaGenerator)
        object.__setattr__(generator, "_config", config)

        recorded = {}

        class _FakeMessage:
            content = json.dumps(
                {"items": [{"question": "Biển hiệu ghi gì?", "reference_answer": "Quốc hội"}]}
            )

        class _FakeResponse:
            message = _FakeMessage()

        class _FakeClient:
            def chat(self, *, model, messages, format):
                recorded["model"] = model
                recorded["messages"] = messages
                recorded["format"] = format
                return _FakeResponse()

        object.__setattr__(generator, "_client", _FakeClient())

        items = generator._call_once("Một phóng viên đứng trước toà nhà quốc hội.")

        assert recorded["model"] == "qwen2.5:7b-instruct"
        assert recorded["format"] == QaGeneration.model_json_schema()
        assert any("quốc hội" in str(m.get("content", "")) for m in recorded["messages"])
        assert items == [QaItem(question="Biển hiệu ghi gì?", reference_answer="Quốc hội")]


class _FakeQaGenerator:
    def __init__(self, items_per_call=1, fail_first_n=0):
        self.calls = 0
        self._items_per_call = items_per_call
        self._fail_first_n = fail_first_n

    def generate(self, caption):
        self.calls += 1
        if self.calls <= self._fail_first_n:
            raise RuntimeError("simulated QA generation failure")
        return [
            QaItem(question=f"Câu hỏi {i} về: {caption}", reference_answer=f"Đáp án {i}")
            for i in range(self._items_per_call)
        ]


class TestGenerateQaCandidates:
    def test_produces_one_record_per_generated_item(self):
        source_records = [_source_record("L21_V001", 1), _source_record("L21_V001", 2)]
        config = BuildQaEvalSetConfig()

        result = _generate_qa_candidates(
            config,
            source_records=source_records,
            qa_generator=_FakeQaGenerator(items_per_call=2),
        )

        assert len(result["records"]) == 4  # 2 shots x 2 items each
        record = result["records"][0]
        assert record["video_id"] == "L21_V001"
        assert record["shot_id"] == 1
        assert record["keyframe_id"] == "L21_V001_f1"
        assert record["source_caption"] == source_records[0]["caption"]
        assert record["question"].startswith("Câu hỏi")
        assert record["answerable_by_ocr"] is None
        assert record["approved"] is None
        assert result["failed"] == []

    def test_generation_failure_is_reported_not_silently_skipped(self):
        source_records = [_source_record("L21_V001", 1)]
        config = BuildQaEvalSetConfig()

        result = _generate_qa_candidates(
            config,
            source_records=source_records,
            qa_generator=_FakeQaGenerator(fail_first_n=1),
        )

        assert result["failed"] == ["L21_V001_shot1"]
        assert result["records"] == []

    def test_on_progress_checkpoints_on_failure_too(self):
        source_records = [_source_record("L21_V001", 1), _source_record("L21_V001", 2)]
        config = BuildQaEvalSetConfig()
        checkpoints = []

        _generate_qa_candidates(
            config,
            source_records=source_records,
            qa_generator=_FakeQaGenerator(fail_first_n=1),
            on_progress=lambda records, failed: checkpoints.append((len(records), len(failed))),
        )

        # shot 1 fails -> checkpoint fires with 0 records/1 failed; shot 2
        # succeeds -> checkpoint fires with 1 record/1 failed.
        assert checkpoints == [(0, 1), (1, 1)]

    def test_resume_skips_shots_already_in_existing_records(self):
        source_records = [_source_record("L21_V001", 1), _source_record("L21_V001", 2)]
        existing_records = [
            {
                "video_id": "L21_V001",
                "shot_id": 1,
                "start_sec": 1.0,
                "end_sec": 3.0,
                "keyframe_id": "L21_V001_f1",
                "question": "câu hỏi từ lần chạy trước",
                "reference_answer": "đáp án cũ",
                "source_caption": source_records[0]["caption"],
                "generator_model": "qwen2.5:7b-instruct",
                "answerable_by_ocr": None,
                "approved": None,
            }
        ]
        config = BuildQaEvalSetConfig()
        generator = _FakeQaGenerator()

        result = _generate_qa_candidates(
            config,
            source_records=source_records,
            qa_generator=generator,
            existing_records=existing_records,
        )

        assert generator.calls == 1  # only shot 2 -- shot 1 not re-billed
        assert len(result["records"]) == 2
        by_shot = {r["shot_id"]: r for r in result["records"]}
        assert by_shot[1]["question"] == "câu hỏi từ lần chạy trước"
        assert by_shot[2]["question"] != "câu hỏi từ lần chạy trước"

    def test_limit_shots_truncates_source_records(self):
        source_records = [_source_record("L21_V001", i) for i in range(1, 6)]
        config = BuildQaEvalSetConfig()

        result = _generate_qa_candidates(
            config,
            source_records=source_records,
            qa_generator=_FakeQaGenerator(),
            limit_shots=2,
        )

        assert len(result["records"]) == 2

    def test_on_progress_checkpoints_after_each_shot(self):
        source_records = [_source_record("L21_V001", i) for i in range(1, 4)]
        config = BuildQaEvalSetConfig()
        checkpoints = []

        _generate_qa_candidates(
            config,
            source_records=source_records,
            qa_generator=_FakeQaGenerator(),
            on_progress=lambda records, failed: checkpoints.append(len(records)),
        )

        assert checkpoints == [1, 2, 3]


class TestComputeQaQualityReport:
    def _approved(self, n, answerable_by_ocr=False):
        return [
            {
                "video_id": "v",
                "shot_id": i,
                "question": f"q{i}",
                "approved": True,
                "answerable_by_ocr": answerable_by_ocr,
            }
            for i in range(n)
        ]

    def test_ignores_extra_unreviewed_records_when_enough_are_reviewed(self):
        """Plan §P3.3: 'Đóng băng bộ đã review... Bộ chưa review chỉ dùng
        để dò lỗi' -- reviewing 100 of a larger generated batch (e.g. 346)
        must be enough; the un-reviewed remainder must NOT block freezing,
        only get silently excluded from the frozen set and the report."""
        candidates = self._approved(100)
        candidates.append({"video_id": "v", "shot_id": 999, "question": "q", "approved": None})

        report = compute_qa_quality_report(candidates, min_reviewed_samples=100)

        assert report.total_generated == 101
        assert report.total_reviewed == 100
        assert report.approved_count == 100

    def test_raises_when_fewer_than_minimum_reviewed_samples(self):
        candidates = self._approved(50)

        with pytest.raises(EvalSetBuildError, match="50"):
            compute_qa_quality_report(candidates, min_reviewed_samples=100)

    def test_raises_when_a_larger_batch_still_has_too_few_reviewed(self):
        """Same gate, but the batch is much bigger than the review floor
        (mirrors a real 346-candidate `generate` run where only ~30 were
        reviewed so far) -- must still cite the actual reviewed count, not
        the total generated."""
        candidates = self._approved(30)
        candidates.extend(
            {"video_id": "v", "shot_id": 2000 + i, "question": "q", "approved": None}
            for i in range(316)
        )

        with pytest.raises(EvalSetBuildError, match="30"):
            compute_qa_quality_report(candidates, min_reviewed_samples=100)

    def test_raises_when_approved_record_missing_answerable_by_ocr_flag(self):
        candidates = self._approved(100)
        candidates[0]["answerable_by_ocr"] = None

        with pytest.raises(EvalSetBuildError, match="answerable_by_ocr"):
            compute_qa_quality_report(candidates, min_reviewed_samples=100)

    def test_computes_rejection_rate_and_ocr_rate(self):
        approved_ocr = self._approved(20, answerable_by_ocr=True)
        approved_visual = self._approved(30, answerable_by_ocr=False)
        for i, record in enumerate(approved_visual):
            record["shot_id"] = 1000 + i
        rejected = [
            {"video_id": "v", "shot_id": 2000 + i, "question": "q", "approved": False}
            for i in range(50)
        ]
        candidates = approved_ocr + approved_visual + rejected

        report = compute_qa_quality_report(candidates, min_reviewed_samples=100)

        assert report.total_generated == 100
        assert report.total_reviewed == 100
        assert report.approved_count == 50
        assert report.rejection_rate == pytest.approx(0.5)
        assert report.answerable_by_ocr_rate == pytest.approx(20 / 50)


class TestFreezeQaRecords:
    def _approved_record(self, **overrides):
        record = {
            "video_id": "L21_V001",
            "start_sec": 1.0,
            "end_sec": 3.0,
            "keyframe_id": "L21_V001_f1",
            "question": "Biển hiệu ghi gì?",
            "reference_answer": "Quốc hội",
            "answerable_by_ocr": False,
            "source_caption": "unused provenance field",
            "generator_model": "qwen2.5:7b-instruct",
            "approved": True,
        }
        record.update(overrides)
        return record

    def test_hash_changes_when_content_changes(self):
        _, digest_a = freeze_qa_records([self._approved_record(question="A?")])
        _, digest_b = freeze_qa_records([self._approved_record(question="B?")])

        assert digest_a != digest_b

    def test_provenance_fields_do_not_leak_into_frozen_output(self):
        json_text, _ = freeze_qa_records([self._approved_record()])

        frozen = json.loads(json_text)
        assert "generator_model" not in frozen[0]
        assert "source_caption" not in frozen[0]
        assert "approved" not in frozen[0]
        assert frozen[0]["question"] == "Biển hiệu ghi gì?"
        assert frozen[0]["answerable_by_ocr"] is False


class TestAssertGeneratorDiffersFromVqaModel:
    def test_raises_when_generator_matches_vqa_answering_model(self):
        records = [{"generator_model": "gemini-3.6-flash"}]

        with pytest.raises(EvalSetBuildError, match="self"):
            assert_generator_differs_from_vqa_model(records, vqa_model="gemini-3.6-flash")

    def test_passes_when_generator_differs_from_vqa_answering_model(self):
        records = [{"generator_model": "qwen2.5:7b-instruct"}]

        assert (
            assert_generator_differs_from_vqa_model(records, vqa_model="gemini-3.6-flash") is None
        )


class TestFreezeQaCandidatesIntegration:
    def _candidates_doc(self, generator_model="qwen2.5:7b-instruct"):
        records = [
            {
                "video_id": "v",
                "shot_id": i,
                "start_sec": float(i),
                "end_sec": float(i) + 2.0,
                "keyframe_id": f"v_f{i}",
                "question": f"q{i}",
                "reference_answer": f"a{i}",
                "source_caption": f"caption {i}",
                "generator_model": generator_model,
                "answerable_by_ocr": i % 2 == 0,
                "approved": True,
            }
            for i in range(100)
        ]
        return {
            "eval_set_version": "v1",
            "git_sha": "abc123",
            "generated_at": "2026-08-16T00:00:00+00:00",
            "records": records,
        }

    def test_freezes_successfully_when_reviewed_and_generator_differs(self):
        json_text, digest, report_md = _freeze_qa_candidates(
            self._candidates_doc(), vqa_model="gemini-3.6-flash", min_reviewed_samples=100
        )

        frozen = json.loads(json_text)
        assert len(frozen) == 100
        assert len(digest) == 64  # sha256 hex digest
        assert "approved: 100" in report_md
        assert "tỉ lệ trả lời được bằng OCR" in report_md

    def test_refuses_to_freeze_when_generator_matches_vqa_model(self):
        doc = self._candidates_doc(generator_model="gemini-3.6-flash")

        with pytest.raises(EvalSetBuildError, match="self"):
            _freeze_qa_candidates(doc, vqa_model="gemini-3.6-flash", min_reviewed_samples=100)

    def test_refuses_to_freeze_below_minimum_reviewed_samples(self):
        doc = self._candidates_doc()
        doc["records"] = doc["records"][:50]

        with pytest.raises(EvalSetBuildError, match="50"):
            _freeze_qa_candidates(doc, vqa_model="gemini-3.6-flash", min_reviewed_samples=100)


class TestWriteQaCandidatesDoc:
    def test_round_trips_through_resume(self, tmp_path):
        out_path = tmp_path / "aic_qa_v1.candidates.json"
        config = BuildQaEvalSetConfig(ollama_model="qwen2.5:7b-instruct")
        records = [{"video_id": "L21_V001", "shot_id": 1, "question": "q", "approved": None}]

        _write_qa_candidates_doc(out_path, config, records, failed=["x"])

        doc = json.loads(out_path.read_text(encoding="utf-8"))
        assert doc["records"] == records
        assert doc["failed"] == ["x"]
        assert doc["config"]["ollama_model"] == "qwen2.5:7b-instruct"


class TestLoadConfig:
    def test_reads_yaml_into_dataclass(self, tmp_path):
        config_path = tmp_path / "build_qa_eval_set.yaml"
        config_path.write_text(
            "ollama_model: qwen2.5:14b-instruct\nmin_reviewed_samples: 200\n",
            encoding="utf-8",
        )

        config = _load_config(config_path)

        assert config.ollama_model == "qwen2.5:14b-instruct"
        assert config.min_reviewed_samples == 200


class TestRunFreezeCli:
    def test_writes_frozen_json_sha256_and_quality_report(self, tmp_path):
        config_path = tmp_path / "build_qa_eval_set.yaml"
        config_path.write_text("min_reviewed_samples: 2\n", encoding="utf-8")

        vqa_config_path = tmp_path / "vqa.yaml"
        vqa_config_path.write_text("gemini_model: gemini-3.6-flash\n", encoding="utf-8")

        candidates_path = tmp_path / "aic_qa_v1.candidates.json"
        candidates_path.write_text(
            json.dumps(
                {
                    "eval_set_version": "v1",
                    "git_sha": "abc123",
                    "generated_at": "2026-08-16T00:00:00+00:00",
                    "records": [
                        {
                            "video_id": "v",
                            "shot_id": i,
                            "start_sec": float(i),
                            "end_sec": float(i) + 2.0,
                            "keyframe_id": f"v_f{i}",
                            "question": f"q{i}",
                            "reference_answer": f"a{i}",
                            "source_caption": f"caption {i}",
                            "generator_model": "qwen2.5:7b-instruct",
                            "answerable_by_ocr": False,
                            "approved": True,
                        }
                        for i in range(2)
                    ],
                }
            ),
            encoding="utf-8",
        )
        out_path = tmp_path / "aic_qa_v1.json"
        args = argparse.Namespace(
            config=config_path,
            candidates=candidates_path,
            out=out_path,
            vqa_config=vqa_config_path,
        )

        exit_code = _run_freeze(args)

        assert exit_code == 0
        assert out_path.exists()
        assert out_path.with_name(out_path.name + ".sha256").exists()
        assert out_path.with_name(out_path.stem + ".quality_report.md").exists()

    def test_returns_nonzero_and_writes_nothing_when_quality_gate_fails(self, tmp_path):
        config_path = tmp_path / "build_qa_eval_set.yaml"
        config_path.write_text("min_reviewed_samples: 100\n", encoding="utf-8")

        vqa_config_path = tmp_path / "vqa.yaml"
        vqa_config_path.write_text("gemini_model: gemini-3.6-flash\n", encoding="utf-8")

        candidates_path = tmp_path / "aic_qa_v1.candidates.json"
        candidates_path.write_text(
            json.dumps(
                {
                    "eval_set_version": "v1",
                    "git_sha": "abc123",
                    "generated_at": "2026-08-16T00:00:00+00:00",
                    "records": [
                        {
                            "video_id": "v",
                            "shot_id": 1,
                            "start_sec": 1.0,
                            "end_sec": 3.0,
                            "keyframe_id": "v_f1",
                            "question": "q",
                            "reference_answer": "a",
                            "source_caption": "caption",
                            "generator_model": "qwen2.5:7b-instruct",
                            "answerable_by_ocr": False,
                            "approved": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        out_path = tmp_path / "aic_qa_v1.json"
        args = argparse.Namespace(
            config=config_path,
            candidates=candidates_path,
            out=out_path,
            vqa_config=vqa_config_path,
        )

        exit_code = _run_freeze(args)

        assert exit_code == 1
        assert not out_path.exists()

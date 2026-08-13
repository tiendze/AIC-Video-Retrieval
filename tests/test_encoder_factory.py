"""RED reproducer for app/encoders/factory.py -- pure dispatch logic, no
torch/GPU needed. Uses monkeypatched sys.modules stand-ins for
app.encoders.beit3 / app.encoders.siglip2 so this runs in the default test
tier (neither real encoder's heavy deps need to be installed to prove the
factory picks the right one and passes the right arguments)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from app.encoders.factory import build_encoder, load_encoder_config


def _write_yaml(path: Path, **fields) -> Path:
    path.write_text(yaml.safe_dump(fields), encoding="utf-8")
    return path


def _install_fake_encoder_module(monkeypatch, module_name: str, class_name: str):
    """Registers a fake `app.encoders.<module_name>` module exposing a class
    that just records the kwargs it was constructed with -- lets us assert
    the factory forwarded config fields correctly without importing torch."""
    calls: list[dict] = []

    class _FakeEncoder:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.kwargs = kwargs

    module = types.ModuleType(f"app.encoders.{module_name}")
    setattr(module, class_name, _FakeEncoder)
    monkeypatch.setitem(sys.modules, f"app.encoders.{module_name}", module)
    return calls


class TestLoadEncoderConfig:
    def test_reads_yaml_file_as_dict(self, tmp_path):
        config_path = _write_yaml(tmp_path / "x.yaml", kind="beit3", model_id="m")

        assert load_encoder_config(config_path) == {"kind": "beit3", "model_id": "m"}


class TestBuildEncoderDispatch:
    def test_kind_beit3_builds_beit3encoder_with_repo_root_resolved_paths(self, tmp_path, monkeypatch):
        calls = _install_fake_encoder_module(monkeypatch, "beit3", "BEiT3Encoder")
        config_path = _write_yaml(
            tmp_path / "beit3.yaml",
            kind="beit3",
            checkpoint_path="models/beit3/x.pth",
            sentencepiece_model_path="models/beit3/x.spm",
            device="cuda",
            batch_size=16,
            model_id="beit3_base_itc_patch16_224",
        )

        build_encoder(config_path)

        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["checkpoint_path"].endswith("models/beit3/x.pth") or kwargs[
            "checkpoint_path"
        ].endswith("models\\beit3\\x.pth")
        assert kwargs["device"] == "cuda"
        assert kwargs["batch_size"] == 16
        assert kwargs["model_id"] == "beit3_base_itc_patch16_224"

    def test_kind_siglip2_builds_siglip2encoder_with_hf_fields(self, tmp_path, monkeypatch):
        calls = _install_fake_encoder_module(monkeypatch, "siglip2", "SigLip2Encoder")
        config_path = _write_yaml(
            tmp_path / "siglip2.yaml",
            kind="siglip2",
            hf_repo_id="google/siglip2-so400m-patch14-384",
            hf_revision="deadbeef",
            device="cuda",
            batch_size=16,
            model_id="siglip2_so400m_p14_384",
            preprocess_workers=8,
        )

        build_encoder(config_path)

        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["hf_repo_id"] == "google/siglip2-so400m-patch14-384"
        assert kwargs["hf_revision"] == "deadbeef"
        assert kwargs["preprocess_workers"] == 8

    def test_device_argument_overrides_config_device(self, tmp_path, monkeypatch):
        calls = _install_fake_encoder_module(monkeypatch, "siglip2", "SigLip2Encoder")
        config_path = _write_yaml(
            tmp_path / "siglip2.yaml",
            kind="siglip2",
            hf_repo_id="r",
            hf_revision="rev",
            device="cuda",
            batch_size=1,
            model_id="m",
            preprocess_workers=1,
        )

        build_encoder(config_path, device="cpu")

        assert calls[0]["device"] == "cpu"

    def test_unknown_kind_raises_value_error_listing_supported_kinds(self, tmp_path):
        config_path = _write_yaml(tmp_path / "x.yaml", kind="pe_core")

        with pytest.raises(ValueError, match="pe_core"):
            build_encoder(config_path)

    def test_missing_kind_raises_value_error(self, tmp_path):
        config_path = _write_yaml(tmp_path / "x.yaml", model_id="m")

        with pytest.raises(ValueError, match="unknown encoder kind"):
            build_encoder(config_path)

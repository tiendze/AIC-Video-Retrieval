from __future__ import annotations

import sys
from types import SimpleNamespace

from scripts import build_abe_concept_bank as script


def test_default_cli_targets_fg_clip_one_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_abe_concept_bank"])

    args = script._parse_args()

    assert args.encoder == "fg_clip"
    assert args.config is None
    assert args.concepts == script._DEFAULT_CONCEPTS_PATH
    assert args.output is None
    assert script._DEFAULT_OUTPUTS["fg_clip"].name == "concept_bank_fgclip_b16.pt"


def test_default_fg_clip_config_loads_from_repo_config():
    args = SimpleNamespace(encoder="fg_clip", config=None)

    config_path, config = script._load_requested_config(args)

    assert config_path.name == "fg_clip.yaml"
    assert config["kind"] == "fg_clip"
    assert config["model_id"] == "fgclip_base"
    assert config["embedding_dim"] == 512

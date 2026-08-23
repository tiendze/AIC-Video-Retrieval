"""Safety-contract tests for owner-operated ASR import/reset CLI help."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import import_asr_out as import_script
from scripts import n3_reset_asr_stores as reset_script


def test_import_help_describes_routine_non_reset_import(capsys):
    with pytest.raises(SystemExit) as caught:
        import_script._parse_args(["--help"])

    assert caught.value.code == 0
    help_text = capsys.readouterr().out.lower()
    assert "routine import" in help_text
    assert "does not delete" in help_text
    assert "n3_reset_asr_stores.py" not in help_text


def test_reset_help_requires_owner_review_and_explicit_restore_inputs(capsys):
    with pytest.raises(SystemExit) as caught:
        reset_script._parse_args(["--help"])

    assert caught.value.code == 0
    help_text = capsys.readouterr().out.lower()
    assert "destructive" in help_text
    assert "owner" in help_text
    assert "--confirm-owner-reviewed-reset" in help_text
    assert "--elasticsearch-backup" in help_text
    assert "--milvus-backup" in help_text
    assert "--restore-procedure" in help_text
    assert "safe to run" not in help_text


def test_reset_parser_accepts_only_explicit_existing_nonempty_inputs(tmp_path: Path, capsys):
    with pytest.raises(SystemExit):
        reset_script._parse_args([])
    error_text = capsys.readouterr().err
    assert "--confirm-owner-reviewed-reset" in error_text

    elasticsearch_backup = tmp_path / "asr-index-backup.json"
    milvus_backup = tmp_path / "transcript-vectors-backup.json"
    restore_procedure = tmp_path / "restore-procedure.md"
    elasticsearch_backup.write_text("{}", encoding="utf-8")
    milvus_backup.write_text("{}", encoding="utf-8")
    restore_procedure.write_text("Reviewed restore steps", encoding="utf-8")

    args = reset_script._parse_args(
        [
            "--confirm-owner-reviewed-reset",
            "--elasticsearch-backup",
            str(elasticsearch_backup),
            "--milvus-backup",
            str(milvus_backup),
            "--restore-procedure",
            str(restore_procedure),
        ]
    )

    assert args.confirm_owner_reviewed_reset is True
    assert args.elasticsearch_backup == elasticsearch_backup
    assert args.milvus_backup == milvus_backup
    assert args.restore_procedure == restore_procedure

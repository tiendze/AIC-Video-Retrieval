# Code Review — P0.3

**Reviewer:** ecc:code-reviewer (sub-agent) · **Ngày:** 2026-07-26 · **Phạm vi:** `git diff a6a2168..HEAD` (2 commit, DatasetAdapter Protocol + YouCook2Adapter)
**Verdict:** APPROVE — không có CRITICAL/HIGH/MEDIUM. 2 LOW.

---

## [LOW] `architecture.md` §5.2 chưa cập nhật theo implementation

**File:** `docs/architecture.md:461`

Doc vẫn ghi `from typing import Iterator, Protocol, Sequence`, nhưng `datasets/base.py` dùng `from collections.abc import Iterator, Sequence` (đúng, vì ruff `UP035` flag `typing.Iterator`/`Sequence` là deprecated alias trên `target-version py310` — reviewer xác nhận không đổi hành vi Protocol vì `Protocol` này không `@runtime_checkable`). CLAUDE.md §2: "Code lệch docs → sửa docs trong cùng change" — đây là vi phạm nhỏ.

**Trạng thái:** ✅ Fixed — cập nhật snippet trong `architecture.md`.

---

## [LOW] Chưa xử lý lỗi khi file annotation thiếu field / hỏng

**File:** `datasets/youcook2.py:45-49` (`_load_database`) và 2 method lặp

`FileNotFoundError`, `json.JSONDecodeError`, `KeyError` (thiếu `subset`/`duration`/`annotations`/...) sẽ propagate thô. Không phải bug — chưa có caller (ingest pipeline) nào wire vào adapter này ở diff này, nên chưa cần quyết định retry/skip/raise theo CLAUDE.md §7.

**Trạng thái:** 📝 Ghi nhận, không fix ngay — sẽ xử lý khi P0.7 (ingest pipeline) wire `YouCook2Adapter` vào, lúc đó mới biết chính xác lỗi nên retry/skip/raise theo `video_id` nào.

# Code Review — P0.1 + P0.2

**Reviewer:** ecc:code-reviewer (sub-agent) · **Ngày:** 2026-07-26 · **Phạm vi:** `git diff 1bc891c..HEAD` (8 commit, P0.1 scaffold + P0.2 docker-compose)
**Verdict:** APPROVE — không có CRITICAL/HIGH. 1 MEDIUM + 2 LOW cần xử lý.

---

## [MEDIUM] Clone `coccoc-tokenizer` không pin version

**File:** `docker/elasticsearch/Dockerfile:21-22`

```dockerfile
RUN git clone --depth 1 https://github.com/duydo/coccoc-tokenizer.git \
    && mkdir coccoc-tokenizer/build
```

Mọi artifact ngoài khác trong Dockerfile đều pin version (`ES_VERSION=9.3.0`, `MVN_VERSION=3.9.9`, plugin tag `v${ES_VERSION}`), riêng thư viện native C++/JNI này clone từ default branch — không tag/commit. `docker compose build --no-cache` sau này (máy khác, cache miss) sẽ lấy bất kỳ `HEAD` nào tại thời điểm đó. Đây chính là thư viện gây crash `UnsatisfiedLinkError` đã fix một lần — thay đổi ngược dòng có thể làm lệch tokenization tiếng Việt âm thầm (hỏng `hybrid` search reproducibility) mà không có CI nào bắt được, vì `tests/integration/` bị loại khỏi `pytest` mặc định.

**Fix:** pin `--branch <commit-sha>`. Repo không có tag nào (`git tags` rỗng) → pin theo commit SHA mới nhất: `a5294b78a638e107c9adffd7fe2fbd31400f790d` (2023-05-10, xác nhận qua GitHub API).

**Trạng thái:** ✅ Fixed.

---

## [LOW] `cors_origins` khai báo nhưng chưa gắn vào `create_app()`

**File:** `app/config.py:26`, `app/main.py:15-16`

`Settings.cors_origins` tồn tại với default nhưng `create_app()` chưa gắn `CORSMiddleware`. Không phải bug (P0.1/P0.2 chưa có route nào), nhưng là config chết, không có test canh giữ.

**Trạng thái:** 📝 Ghi nhận, không fix ngay — sẽ gắn `CORSMiddleware` khi P0.8 thêm route đầu tiên (`/api/search`, `/health`). Fix sớm hơn là speculative (YAGNI — CLAUDE.md coding-style).

---

## [LOW] Test file thiếu type annotation trên chữ ký hàm

**Files:** `tests/test_config.py`, `tests/test_main.py`, `tests/integration/test_elasticsearch_vi_analyzer.py`

CLAUDE.md §6: "type annotation trên mọi chữ ký hàm" — không có ngoại lệ cho test. Các hàm test mới chưa annotate `monkeypatch: pytest.MonkeyPatch`, `tmp_path: Path`, `-> None`. Code production (`app/config.py`, `app/main.py`) đã tuân thủ đúng.

**Trạng thái:** ✅ Fixed.

---

## Ghi chú phụ (không phải finding code)

Reviewer phát hiện `.env` local (không commit, đã xác nhận qua `git log --all -p`) có `GEMINI_API_KEY` dài khác với placeholder ban đầu tôi tạo (`dev-placeholder-fill-in-real-key`, 32 ký tự) — hiện là 53 ký tự. Không rõ đây là key thật do người dùng tự điền hay gì khác; không phải vấn đề code (file đã gitignore đúng), chỉ cần người chủ dự án tự xác nhận/xoay vòng key nếu cần.

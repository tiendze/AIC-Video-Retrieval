# Error Handling & Secrets

> Tách từ `CLAUDE.md` §7. Quy tắc bảo mật chung (không hardcode secret, validate lúc khởi động, che token trong log, CORS theo môi trường) nằm ở `.claude/rules/ecc/common/security.md` · `python/security.md` · `python/fastapi.md`. Dưới đây là phần riêng của dự án.

- Mọi API ngoài (Gemini, OpenAI, image search): **timeout + retry exponential backoff + fallback rõ ràng**.
- Lỗi nhánh phụ thuộc mạng **không được giết request chính** — VQA lỗi vẫn trả keyframe grid kèm thông báo.
- **`except Exception: pass` bị cấm.** Bắt exception cụ thể, log kèm `video_id` / `keyframe_id` / `request_id`, rồi quyết định retry / skip / raise.
- **Thiếu dữ liệu phải fail rõ ràng, không trả rỗng** — bộ nhãn chưa tồn tại ⇒ raise, đừng để `eval/` in bảng toàn số 0 trông như "hệ thống hỏng".
- Lỗi hiện ra UI phải đọc được bởi người đang bị tính giờ — không dump traceback.
- Secret nạp qua `pydantic-settings` (`app/config.py`: `Settings` / `CorsSettings`), validate ngay lúc khởi động.
- Ảnh upload: giới hạn size, check MIME, không lưu vĩnh viễn.
- **URL từ nguồn ngoài không bao giờ nhận từ client** — server giữ URL thật trong cache, client chỉ gửi ID tham chiếu (pattern QUEST nhánh 2, `app/services/quest_service.py`). Chống SSRF.

## Liên quan

- Schema lỗi `{detail, error_code, request_id}` và ranh giới tầng: [`code-architecture.md`](code-architecture.md)
- Fallback của QUEST / degraded mode: [`pipeline-and-retrieval.md`](pipeline-and-retrieval.md)
- "Thiếu dữ liệu fail rõ ràng" áp vào eval ra sao: [`evaluation-and-reporting.md`](evaluation-and-reporting.md)

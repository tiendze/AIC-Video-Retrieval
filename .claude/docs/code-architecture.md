# Kiến trúc riêng của dự án

> Tách từ `CLAUDE.md` §6. Chỉ ghi thứ RIÊNG của repo này — quy tắc code chung (PEP 8, black/isort/ruff, immutability, giới hạn kích thước hàm/file, FastAPI DI + `response_model`) nằm ở `.claude/rules/ecc/`. Cây thư mục đầy đủ: `docs/architecture.md` §6.

**Hai Protocol là hàng rào chống rủi ro, không được bypass:** `MultimodalEncoder` (rút lui khỏi BEiT-3 không phải sửa pipeline) và `DatasetAdapter` (đổi/mở rộng dataset chỉ thêm class). Pipeline gọi qua interface, không import implementation.

- Ranh giới tầng: `app/api/` routers **mỏng** · `app/services/` business logic · `app/stores/` chỉ nói chuyện với DB · `eval/run.py` là entry point **duy nhất** của evaluation.
- Encoder/model nạp qua **lifespan**, không nạp trong route. Lỗi API theo schema `{detail, error_code, request_id}`.
- **Không magic number** — ngưỡng, timeout, `top_k`, `λ`, `k` của RRF → `configs/`.
- Encoder contract: `np.ndarray` shape `(N, embedding_dim)`, `float32`, **đã L2-normalize** — ghi trong docstring, kiểm bằng test.
- Docstring/comment viết bằng **tiếng Anh**; tên biến/hàm/file luôn tiếng Anh.

## Liên quan

- Invariants về encoder / `model_id` / collection: `CLAUDE.md` §3
- Encoder nào đang wire, xung khắc dependency: [`libraries-api.md`](libraries-api.md)
- Schema lỗi `{detail, error_code, request_id}` dùng ra sao: [`error-handling.md`](error-handling.md)

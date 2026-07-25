# AIC-Video-Retrieval

Hệ thống truy vấn & hỏi đáp video đa phương thức cho AI Challenge TP.HCM — hỗ trợ 4 task: Textual KIS, Video KIS, QA (VQA) và TRAKE.

Baseline: [arXiv:2512.13169 — *Integrated Semantic and Temporal Alignment for Interactive Video Retrieval*](https://arxiv.org/abs/2512.13169)

## Tài liệu

| File | Nội dung |
|---|---|
| [`docs/PRD.md`](./docs/PRD.md) | Bối cảnh, mục tiêu, scope theo phase, features, **tiêu chuẩn đánh giá (Evaluation Metrics)**, NFR, rủi ro |
| [`docs/architecture.md`](./docs/architecture.md) | Thiết kế hệ thống, 6 sơ đồ Mermaid luồng dữ liệu, schema 3 store, API contract, deployment |
| [`docs/references.md`](./docs/references.md) | Thư viện, snippet API mới nhất lấy từ `llms.txt`, phương án tối ưu giai đoạn 2 |

## Ngăn xếp công nghệ

**Pipeline:** TransNetV2 (shot detection) → BEiT-3 (multimodal embedding) → Gemini (OCR)
**Lưu trữ:** Milvus (vector) · Elasticsearch + `vi_analyzer` (OCR text) · MongoDB (metadata)
**Ứng dụng:** FastAPI + Pydantic · React + Vite + TypeScript
**Dataset giai đoạn 1:** YouCook2 (2.000 video, 176 giờ) — dữ liệu AIC bổ sung sau

## Cài đặt & chạy (P0.1)

```powershell
python -m venv .venv
./.venv/Scripts/pip.exe install -e ".[dev]"

Copy-Item .env.example .env   # rồi điền GEMINI_API_KEY / OPENAI_API_KEY thật

./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m black --check .
./.venv/Scripts/python.exe -m pytest --cov --cov-report=term-missing
```

Thiếu `GEMINI_API_KEY` hoặc `OPENAI_API_KEY` trong `.env` → app báo `ValidationError` rõ ràng ngay lúc khởi động (`Settings()` trong `app/main.py` lifespan), không phải lỗi mơ hồ giữa chừng.

## Trạng thái

🚧 P0 — Nền móng đang thực thi. P0.1 (scaffold, config, FastAPI app) đã xong. Xem [`.claude/plans/p0-foundation.plan.md`](./.claude/plans/p0-foundation.plan.md).
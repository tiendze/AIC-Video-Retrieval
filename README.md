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
**Dataset:** AIC 2025 công khai (Kaggle `aresusayhi/ai-challenge-2025` v2) — **873 video, 130,7 giờ**, tin tức tiếng Việt. Tải bằng `python download.py` vào `data/`. P0 chạy pilot `L21`–`L24` (128 video), mở rộng đủ 873 ở P4. ⚠️ Dataset **không kèm nhãn truy vấn** — bộ eval là nhãn tự sinh, xem PRD §6.1

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

## Hạ tầng lưu trữ (P0.2)

```powershell
docker compose up -d --build   # lần đầu build image Elasticsearch mất vài phút (compile native lib)
docker compose ps               # cả 5 container phải "healthy"
```

Verify `vi_analyzer` tách theo từ ghép (không chạy trong `pytest` mặc định vì cần Docker):

```powershell
./.venv/Scripts/python.exe -m pytest tests/integration -m integration -v
```

## Trạng thái

🚧 P0 — Nền móng đang thực thi. P0.1 (scaffold, config, FastAPI app) và P0.2 (docker-compose: Milvus + Elasticsearch `vi_analyzer` + MongoDB) đã xong. Xem [`.claude/plans/p0-foundation.plan.md`](./.claude/plans/p0-foundation.plan.md).
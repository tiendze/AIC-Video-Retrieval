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
**Dataset:** AIC 2026 batch 1 công khai (nội dung trùng batch 1 của AIC 2025) — **873 video, 130,7 giờ**, tin tức tiếng Việt, nằm ở `data/`. Đã tải: `Videos_*/video/*.mp4` · `media-info` · `map-keyframes`. **Cố ý không tải** `Keyframes_*` / `objects` / `clip-features-32` — dự án tự trích bằng model của mình. P0 chạy pilot `L21`–`L24` (128 video), mở rộng đủ 873 ở P4. ⚠️ **Batch 2 chưa phát hành.** ⚠️ Dataset **không kèm nhãn truy vấn** — bộ eval là nhãn tự sinh, xem PRD §6.1

**Đề thi:** vòng sơ tuyển AIC 2026 — 3 task (`Textual KIS` · `QA` · `TRAKE`, **không còn Video KIS**), nộp tối đa **100 đáp án xếp hạng** mỗi truy vấn, điểm = `mean(R@k)` với k ∈ {1,5,20,50,100}. Đáp án là `<video_id>, <frame_id>`. Toàn văn: `Thong tin vong So tuyen AIC2026.pdf`; tóm tắt: `docs/references.md` §1.2

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

## Frontend (P1.6)

React + Vite + TypeScript, `frontend/`. UI theo danh sách năng lực chức năng của paper baseline (không có mẫu thiết kế riêng — chủ dự án mở gate P1.5, xem `docs/architecture.md` §8).

```powershell
cd frontend
npm install
npm run dev      # http://localhost:5173 -- proxy /api -> http://localhost:8000
npm run test     # Vitest + React Testing Library
npm run build    # tsc -b && vite build
```

Cần backend sống ở `:8000` để dùng thật (`./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000`) + docker-compose stack (Milvus/Elasticsearch/MongoDB). Ảnh keyframe được resize + cache theo yêu cầu qua `GET /api/keyframes/{id}/thumbnail` -- lần xem đầu mỗi ảnh chậm hơn (đang sinh cache), lần sau tức thì. Warm cache trước cho toàn corpus (tốn thời gian, không tự chạy):

```powershell
python scripts/build_thumbnails.py --limit-videos 5   # đo thử trước
python scripts/build_thumbnails.py                    # full corpus
```

## Chẩn đoán không gian embedding (điều tra `semantic` < `random`)

Báo cáo `reports/20260810T161049Z_a40aac8` cho thấy kênh `semantic` (AIC 0.0451)
thấp hơn baseline `random` (0.0809). Hai công cụ dưới đây tách "text tower không
đọc được tiếng Việt" khỏi "phía ảnh/index hỏng" — **không re-embed gì cả** và
**không sửa bộ nhãn** (`aic_eval_v1.json` + `.sha256` giữ nguyên).

Bước 0 — đo mode collapse **không cần bản dịch nào**, không tốn quota:

```powershell
docker compose up -d
./.venv/Scripts/python.exe -m scripts.diagnose_embedding_space
```

`collapsed: true` kèm `query_cosine_mean` cao và `union_ratio` thấp là đã đủ kết
luận không gian vector suy biến. Bước 1–2 chỉ cần khi muốn tách tiếp "text tower
hỏng" khỏi "phía ảnh hỏng".

Bước 1 — sinh sidecar truy vấn tiếng Anh. Mặc định dùng **dịch offline miễn phí**
(`VietAI/envit5-translation`, tải ~1GB lần đầu rồi chạy không cần mạng, không cần
API key):

```powershell
./.venv/Scripts/python.exe -m scripts.translate_eval_queries --limit 5   # thử
./.venv/Scripts/python.exe -m scripts.translate_eval_queries             # 173 câu
```

Đổi model dịch: sửa `local_model_id` trong `configs/eval/query_translation.yaml`
(ví dụ `vinai/vinai-translate-vi2en-v2` — model này không cần tag ngôn ngữ nên
đặt `local_source_prefix`/`local_target_prefix` thành `""`). Dùng Gemini thay thế:
`--translator gemini`. Script resume được và ghi `translator_model` cho **từng
dòng**, nên sidecar sinh dở bằng Gemini rồi dịch nốt bằng model offline vẫn truy
được nguồn gốc.

Bước 2 — đo lại kèm truy vấn tiếng Anh (cần Milvus + GPU):

```powershell
docker compose up -d
./.venv/Scripts/python.exe -m scripts.diagnose_embedding_space `
    --translations configs/eval/aic_eval_v1.queries_en.json
```

Kết quả: `reports/diagnostics/<timestamp>_<git_sha>/embedding_space_{vi,en}.{json,md}`.

Bước 3 — chấm kênh `semantic@en` cạnh `semantic` trong cùng một báo cáo: đặt
`query_translation_path: configs/eval/aic_eval_v1.queries_en.json` trong
`configs/eval/aic_pilot.yaml` rồi chạy lại `python -m eval.run`. Để trống khoá
này là tắt hẳn tính năng, hành vi không đổi.

## Trạng thái

🚧 P0 — Nền móng đang thực thi. P0.1 (scaffold, config, FastAPI app) và P0.2 (docker-compose: Milvus + Elasticsearch `vi_analyzer` + MongoDB) đã xong. Xem [`.claude/plans/p0-foundation.plan.md`](./.claude/plans/p0-foundation.plan.md).
# AIC-Video-Retrieval

Hệ thống truy vấn & hỏi đáp video đa phương thức cho AI Challenge TP.HCM — vòng sơ tuyển AIC 2026, 3 task: `Textual KIS` · `QA (VQA)` · `TRAKE`.

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

## Cài đặt & chạy

Yêu cầu **Python 3.10** (`pyproject.toml` pin cứng `>=3.10,<3.11` — BEiT-3's `timm==0.4.12` không tương thích 3.11+).

```powershell
python -m venv .venv
./.venv/Scripts/pip.exe install -e ".[dev]"

Copy-Item .env.example .env   # rồi điền GEMINI_API_KEY / OPENAI_API_KEY thật

./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m black --check .
./.venv/Scripts/python.exe -m pytest --cov --cov-report=term-missing
```

Thiếu `GEMINI_API_KEY` hoặc `OPENAI_API_KEY` trong `.env` → app báo `ValidationError` rõ ràng ngay lúc khởi động (`Settings()` trong `app/main.py` lifespan), không phải lỗi mơ hồ giữa chừng.

`[dev]` là đủ để chạy API server (với dữ liệu đã ingest sẵn), chạy test suite, và lint. Các việc nặng hơn cần thêm extras riêng (`pip install -e ".[extra]"`, có thể kết hợp `".[extra1,extra2]"`):

| Extra | Cần khi nào | Kéo theo |
|---|---|---|
| `data` | Tải dataset từ Kaggle | `kagglehub` |
| `beit3` | Chạy encoder BEiT-3 (production, GPU khuyến nghị) | `torch`/`torchvision`/`timm`/`torchscale`/`transformers` |
| `ingest` | Chạy pipeline ingest (`pipelines/import_embed_out.py`) | `beit3` + `opencv-python-headless`/`imagehash`/`pillow` |
| `ocr` | Sinh OCR bằng Gemini | `ingest` + `google-genai` |
| `eval` | Chạy `eval/build_eval_set.py` (sinh bộ nhãn) | = `ocr` |
| `asr` | Trích transcript bằng `faster-whisper` | `ingest` + `faster-whisper`/`ctranslate2` |
| `asr-embed` | Embed transcript ASR | `ingest` + `sentence-transformers` |
| `baseline-clip` | Baseline CLIP bắt buộc (PRD §6.3) | xem ghi chú xung đột `timm` trong `pyproject.toml` — cần `pip install --no-deps open-clip-torch==3.3.0` sau |
| `encoder-candidates` | Thử nghiệm encoder ứng viên (SigLIP2/PE-Core) | **venv riêng** (`.venv-candidate`) — xung đột pin `transformers`/`torch` với `beit3`, xem `.claude/plans/encoder-swap-siglip2-pecore.plan.md` |

`.venv-candidate/` đã có sẵn trong repo cho thử nghiệm encoder — không dùng chung venv với `.venv/`.

## Hạ tầng lưu trữ (P0.2)

```powershell
docker compose up -d --build   # lần đầu build image Elasticsearch mất vài phút (compile native lib)
docker compose ps               # cả 5 container phải "healthy"
```

Verify `vi_analyzer` tách theo từ ghép (không chạy trong `pytest` mặc định vì cần Docker):

```powershell
./.venv/Scripts/python.exe -m pytest tests/integration -m integration -v
```

## Frontend

React 19 + Vite + TypeScript, `frontend/`. UI theo danh sách năng lực chức năng của paper baseline (không có mẫu thiết kế riêng — chủ dự án mở gate P1.5, xem `docs/architecture.md` §8).

Ba chế độ tìm kiếm sống trong `ModeSelector` (phím tắt `1`/`2`/`3`):

- **Semantic** — tìm theo văn bản/ảnh qua encoder BEiT-3, có nút "✨ AI enhancer" (QUEST nhánh 1 — Gemini viết lại truy vấn mơ hồ thành mô tả trực quan, người vận hành tự chọn dùng `q₀` hay `q_r`) và "🌐 Tìm ảnh mẫu trên web" (QUEST nhánh 2 — tìm ảnh thực thể lạ trên web, người dùng tự chọn ảnh rồi image-to-image search).
- **OCR** — tìm theo chữ xuất hiện trong khung hình (Elasticsearch + `vi_analyzer`); tự động disable nếu Elasticsearch không sống.
- **DANTE** — nhập chuỗi N sự kiện theo thứ tự (`EventSequenceInput`), trả về video khớp cả chuỗi đúng thứ tự thời gian (`TrakeResultList`) — dùng cho task `TRAKE`.

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

- ✅ **P0 — Nền móng** xong (2026-08-11).
- ✅ **P1 — Tìm kiếm cơ bản** đóng 2026-08-13 (semantic + image-to-image + OCR + fusion, UI dùng được thật).
- 🟡 **P2 — QUEST & DANTE** gần xong (2026-08-15/16): DANTE (`dante_service.py`), `POST /api/trake` + bảng ablation `λ` (đã chọn `λ=0.001`), QUEST 2 nhánh (Gemini rewrite + web image search) đều chạy end-to-end kèm UI. Còn thiếu cột `N=7` trong bảng ablation (cần sinh lại bộ eval, đã hoãn có chủ đích).
- ⬜ **P3 — VQA & nộp bài**, **P4 — Mở rộng toàn kho & tối ưu** chưa bắt đầu.

Chi tiết từng phase + số đo thật: [`.claude/plans/roadmap.plan.md`](./.claude/plans/roadmap.plan.md).

## Directory `app/` — đường online query

`app/` là đường **online (serving)** — FastAPI nhận request, gọi service, trả kết quả. Đối lập với `pipelines/`/`scripts/` (đường **offline ingest**, xem `.claude/docs/pipeline-and-retrieval.md`). Router mỏng, không chứa business logic; service không phụ thuộc FastAPI, test được độc lập; store là client mỏng bọc SDK, filter luôn chạy trong engine (không lọc sau khi lấy kết quả về, CLAUDE.md — Invariants).

```
app/
├── main.py                  # app factory + lifespan: dựng Settings, encoder, 3 store, services; đăng ký router; CORS
├── config.py                 # Settings / CorsSettings (pydantic-settings, đọc .env)
├── dependencies.py            # Depends() provider — đọc instance từ request.app.state, router không tự new client
│
├── api/                       # router tầng HTTP — validate input, gọi service, response_model
│   ├── search.py                #   POST /api/search · POST /api/search/image
│   ├── trake.py                  #   POST /api/trake (DANTE)
│   ├── quest.py                    #   POST /api/quest/rewrite-query · /image-candidates · /image-search
│   ├── videos.py                    #   GET /api/videos/{id} · /api/videos/{id}/keyframes
│   ├── keyframes.py                  #   GET /api/keyframes/{id}/thumbnail · /image
│   ├── health.py                      #   GET /health
│   └── errors.py                       #   chuẩn hoá response lỗi
│
├── services/                   # business logic thuần Python, test không cần FastAPI
│   ├── search_service.py          #   semantic (BEiT-3 + QueryTranslator VI→EN) + image-to-image + OCR + fusion
│   ├── dante_service.py             #   DP O(NT) tìm chuỗi N sự kiện đúng thứ tự — công thức Eq (1)–(5) của paper
│   ├── trake_service.py               #   cầu nối POST /api/trake → dante_service, duyệt mọi video `complete`
│   ├── quest_service.py                 #   QUEST 2 nhánh: GeminiQueryRewriter (viết lại truy vấn) + tìm ảnh mẫu web (DDGS + httpx)
│   ├── query_translator.py                #   VI→EN cho kênh semantic — BEiT-3 text tower chỉ học cặp tiếng Anh
│   ├── thumbnail_service.py                 #   resize + cache ảnh keyframe theo yêu cầu
│   └── health_service.py                     #   kiểm tra 3 store còn sống
│
├── stores/                     # client mỏng bọc SDK từng store
│   ├── milvus_client.py           #   vector search
│   ├── es_client.py                 #   OCR text search (Elasticsearch + vi_analyzer)
│   └── mongo_client.py                #   metadata (frame_idx, global_index, s_v/e_v...)
│
├── encoders/                   # MultimodalEncoder Protocol + implementation; L2-normalize ngay tại encoder
│   ├── base.py                    #   Protocol chung (encode_text/encode_image)
│   ├── beit3.py                     #   encoder production hiện hành
│   ├── siglip2.py                     #   ứng viên thay thế — đang đánh giá, CHƯA phải production
│   ├── clip_baseline.py                 #   baseline CLIP bắt buộc (PRD §6.3)
│   ├── bge_m3.py                          #   text-to-text riêng cho kênh ASR transcript (không dùng chung BEiT-3)
│   ├── factory.py                           #   dispatch theo `kind` trong `configs/model/*.yaml` — hiện chỉ `scripts/import_embed_out.py` gọi qua đây, `app/main.py` vẫn dựng `BEiT3Encoder` trực tiếp
│   └── vendor/beit3_upstream/                 #   code BEiT-3 gốc vendor vào nguyên trạng, không sửa
│
└── schemas/                   # Pydantic — response_model bắt buộc trên mọi route (CLAUDE.md, FastAPI DI rule)
    ├── request/                  #   SearchRequest, TrakeRequest, QuestRequest...
    └── response/                  #   SearchResponse, TrakeResponse, HealthResponse...
```

Muốn chạy được đường này cần: docker-compose stack sống (§ trên) + `.env` đủ key + dữ liệu đã ingest vào Milvus/Elasticsearch/MongoDB (đường offline, `pipelines/import_embed_out.py` — xem `.claude/docs/pipeline-and-retrieval.md`, không tự chạy từ README này).
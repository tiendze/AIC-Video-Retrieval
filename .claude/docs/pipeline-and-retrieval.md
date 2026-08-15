# Pipeline & Retrieval

> Tách từ `CLAUDE.md` §4. Sơ đồ luồng dữ liệu đầy đủ: `docs/architecture.md` §3.

**Ảnh và chữ nằm trong CÙNG một không gian vector — mọi thứ khác là hệ quả.**

## Ingest — corpus sống nạp bằng đường import, không tự trích

Entry point `pipelines/import_embed_out.py` (CLI `scripts/import_embed_out.py`): keyframe tải sẵn ở `data/keyframe/<batch>/<video_id>/`, shot boundary ở `data/shots.json`, embedding tính trên Kaggle ở `data/embed_out/` (BEiT-3) · `data/embed_out_siglip2/`. `shot_id` gán bằng cách tra `frame_idx` vào `[start_frame, end_frame]` của `shots.json`; `global_index` / `s_v` / `e_v` / `timestamp` / `fps` / metadata **dẫn xuất tại bước import**.

- ⚠️ `pipelines/ingest.py` + `shot_detect.py` (đường TransNetV2 tự trích shot + 4 keyframe/shot, P0.7) **còn trong repo nhưng KHÔNG dùng cho corpus hiện tại**. Muốn đổi hành vi ingest thì sửa `import_embed_out.py`, đừng sửa hai file kia.
- Keyframe import không kèm bản resize (`thumbnail_path == image_path`) ⇒ thumbnail do `ThumbnailService` sinh theo yêu cầu + cache đĩa.
- **ASR đã ingest nhưng CHƯA nối vào retrieval** — `pipelines/asr.py` · `import_asr_out.py` · `embed_transcripts.py` ghi transcript vào ES/MongoDB, nhưng **không search mode nào đọc nó**. Đừng coi ASR là kênh tìm kiếm, và **đừng fuse vào ranking khi chưa đo trên bộ eval** (bài học RRF dưới đây).
- Lỗi từng video/keyframe → error list riêng + `ingest_status = "failed"`, **pipeline chạy tiếp**; checkpoint theo `video_id`.
- Việc GPU chạy local (encode lại, ASR, OCR): trên 8GB VRAM **chạy tuần tự an toàn hơn song song**.
- **Lọc frame có chữ + dedupe perceptual hash TRƯỚC khi gọi Gemini OCR** (risk R2 — chi phí token). Video tin tức: **dedupe mới là bước tiết kiệm chính**, không phải lọc chữ.

## Retrieval

- `semantic` · `ocr` là **tham số của một hàm**, không phải hai code path sao chép. `ocr` là BM25 thuần qua Elasticsearch — không dịch, không tự fuse vào ranking.
- **Không fuse RRF hai kênh lệch chất lượng.** `hybrid` semantic⊕ocr bỏ hẳn 2026-08-12 (lý do: `docs/architecture.md` §10 + docstring `app/services/search_service.py`). RRF (`k = 60`) chỉ dùng cho **hai kênh cùng loại, chất lượng ngang nhau**; kém hơn kênh đơn tốt nhất trên bộ eval thì chỉnh trọng số hoặc bỏ.
- **QUEST giữ người trong vòng lặp** — hiện cả `q₀` lẫn `q_r`; người dùng **tự chọn** ảnh mẫu `I*` ở nhánh 2, không auto-pick. Cả hai nhánh có timeout → fallback về search thường, không treo UI.
- **DANTE** dùng `running max` + **backtracking** để trả đúng N keyframe. `λ` là config (`configs/search/dante.yaml`), không hard-code.
- **Degraded mode phải chạy được** — tắt Elasticsearch ⇒ `semantic-only`; mất mạng ⇒ semantic + DANTE vẫn đủ, chỉ QUEST / OCR / VQA ngừng.

## Liên quan

- Invariants ràng buộc pipeline (`frame_idx`, `global_index`, idempotent, khử trùng `video_id`): `CLAUDE.md` §3
- API của từng thư viện dùng ở đây: [`libraries-api.md`](libraries-api.md)
- Timeout / fallback / chống SSRF khi gọi ra ngoài: [`error-handling.md`](error-handling.md)

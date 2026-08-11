# P1 — Tìm kiếm cơ bản

> **Quay lại:** [`roadmap.plan.md`](./roadmap.plan.md) · **Trước:** [`p0-foundation.plan.md`](./p0-foundation.plan.md) · **Tiếp theo:** [`p2-quest-dante.plan.md`](./p2-quest-dante.plan.md)
> **Trạng thái:** ⬜ chưa bắt đầu · **Điều kiện vào:** P0 đã có bảng metrics thật

**Exit criteria (PRD §4):** đạt mục tiêu Recall@10 trong PRD §6.7; UI dùng được để thao tác thật.

Phase này bổ sung hai kênh còn lại (image-to-image, OCR) và hợp nhất chúng. **Câu hỏi trung tâm:** fusion có thực sự đóng góp gì không — nếu không, bỏ hẳn F4 chứ không giữ vì "nghe hợp lý".

---

## P1.1 — Image-to-image search — ✅ xong 2026-08-11 (code + integration test đều PASS)

**Làm gì:** `POST /api/search/image` — người vận hành thấy một ảnh giống thứ cần tìm thì upload lên (US-3).

- `multipart/form-data`. Encode bằng **nhánh vision của CÙNG encoder**, **CÙNG preprocessing** với lúc ingest — vi phạm chỗ này là hỏng âm thầm, không crash.
- Bảo mật: giới hạn kích thước file (10 MiB, `configs/search/search.yaml`), check MIME type (jpeg/png/webp), **không lưu vĩnh viễn** (giải mã trong RAM, không ghi đĩa).

**Đã làm:**
- `app/services/search_service.py`: `search_by_image()` — dùng chung `_search_milvus_vector()` với `semantic` (refactor DRY), không đi qua `QueryTranslator`.
- `app/api/search.py`: route mới, validate MIME → validate size → decode ảnh (bắt `UnidentifiedImageError`) → gọi service.
- `configs/search/search.yaml` + `app/schemas/request/search.py`: thêm `MAX_UPLOAD_SIZE_BYTES`, `ALLOWED_UPLOAD_MIME_TYPES`.
- `pyproject.toml`: thêm `python-multipart==0.0.20` (bắt buộc cho `File()`/`Form()` của FastAPI, thiếu thì lỗi ngay lúc import).

**Code review 2026-08-11 (`ecc:code-reviewer` agent) — 2 HIGH, 2 MEDIUM đã fix:**
- Nhánh `except UnidentifiedImageError` gốc bỏ sót `OSError` thường (file JPEG bị cắt cụt) và `Image.DecompressionBombError` (ảnh khai kích thước khổng lồ trong file bé) — cả hai lọt xuống 500 thay vì 400. Sửa: `except (OSError, Image.DecompressionBombError)`.
- Body upload bị Starlette đọc/spool hết vào bộ nhớ **trước khi** route kịp check `MAX_UPLOAD_SIZE_BYTES` (python-multipart không có trần tổng kích thước). Thêm `register_upload_guard()` — middleware chặn theo header `Content-Length` khai láo trước khi multipart parser chạm vào body; route vẫn giữ check cũ làm lớp phòng thủ thứ hai (trường hợp thiếu/giả `Content-Length`).
- `Content-Type` chỉ là nhãn caller tự khai, `Image.open()` nhận diện theo magic bytes — thêm bước so `image.format` với `Content-Type` đã khai, không khớp thì 415.
- Test hardcode `10*1024*1024+1` thay bằng import `MAX_UPLOAD_SIZE_BYTES` từ config (tránh lệch âm thầm khi đổi config).

**Verify:**
- `tests/test_search_api.py` — 21 test (bao gồm 7 case mới từ review: truncated JPEG, decompression bomb, MIME giả mạo, file field thiếu, upload 0 byte, top_k biên trên, content-length guard chặn trước khi đọc body). Toàn bộ PASS. Coverage `app/api/search.py` + `app/services/search_service.py` = 100%.
- `test_uploaded_keyframe_retrieves_itself_as_top1` — `tests/integration/test_search_service_image.py`, chạy thật 2026-08-11 trên docker-compose stack sống (Milvus/Mongo/ES healthy) + checkpoint BEiT-3 thật (`models/beit3/`, CUDA available) theo lệnh chủ dự án cho phép chạy:
  ```
  BEIT3_CHECKPOINT_PATH=models/beit3/beit3_base_itc_patch16_224.pth \
  BEIT3_SPM_PATH=models/beit3/beit3.spm \
  pytest -m "integration and model" tests/integration/test_search_service_image.py -v
  ```
  Kết quả: **1 passed in 17.33s**. Ảnh đáp án đứng top-1 giữa 2 keyframe (1 decoy), cosine self-similarity ≈ 1.0 — preprocessing lúc query khớp chính xác lúc "ingest".

---

## ~~P1.2 — Nhánh TEXT của ingest (OCR)~~ → **ĐÃ CHUYỂN VỀ P0.7** (2026-07-29)

Chủ dự án quyết kéo nhánh OCR về P0. Nội dung task nay nằm ở **`p0-foundation.plan.md` § P0.7**, mục *"Nhánh TEXT — `pipelines/ocr.py`"*. Lý do đầy đủ: `roadmap.plan.md` §8.

Tóm tắt: R4 đóng (video tin tức đầy chữ) nên OCR ở P0 **đo được ngay**; R2 ở quy mô pilot là khoản chấp nhận được (~13,4 triệu token, hoặc ~6,7 triệu nếu hạ 384×384). Đổi lại P0 có kênh đối chứng thứ hai và **phát hiện được bệnh rò rỉ nhãn trước khi xây tiếp**.

**Việc còn lại ở P1 — mở rộng, không phải xây mới:**

- Đo **chi phí/lợi ích** của kênh OCR trên bộ eval đầy đủ: bao nhiêu % truy vấn được cải thiện nhờ OCR, so với số token đã đốt. Không đáng thì cân nhắc OCR cục bộ (PaddleOCR/EasyOCR, `references.md` §8).
- Cân nhắc **Batch API** khi chuẩn bị mở rộng ra toàn kho ở P4.
- Chốt quyết định hạ ảnh xuống 384×384 dựa trên số đo chất lượng OCR thật từ P0.

---

## P1.3 — Hybrid fusion (RRF)

**Làm gì:** hợp nhất kênh semantic và kênh OCR.

> **Rẻ hơn bản trước.** Cả hai kênh đã chạy và **đã có số đo riêng lẻ từ P0.9**, nên P1.3 chỉ còn phần fusion. Đây cũng là lý do hybrid **không** bị kéo về P0 cùng OCR: so sánh "hybrid vs kênh đơn tốt nhất" chỉ có nghĩa khi hai kênh đơn đã được đo tử tế trước.

- Ba mode `semantic` · `ocr` · `hybrid` là **tham số** của `search_service`, **không** phải ba code path sao chép nhau.
- `RRF(d) = Σ_c 1 / (k + rank_c(d))`, mặc định `k = 60` — đặt trong `configs/`, không hard-code. Trọng số mỗi kênh cấu hình được.

**Verify — đây là test quyết định số phận của F4:**
- `test_hybrid_not_worse_than_best_single_channel` — chạy trên bộ eval. Hybrid **kém hơn** kênh đơn tốt nhất ⇒ chỉnh trọng số, vẫn kém ⇒ **bỏ hẳn F4** và ghi lý do vào báo cáo phase.

⚠️ **Kỳ vọng đã đảo so với bản trước.** Trên YouCook2 kênh OCR gần như rỗng nên hybrid dự kiến ≈ semantic. Trên video tin tức, kênh OCR **giàu tín hiệu**, nên hybrid có cơ hội thật sự vượt semantic — và ngược lại, nếu hybrid **vẫn** ≈ semantic thì đó là tín hiệu đáng nghi, phải điều tra chứ đừng chấp nhận.

⚠️ Cảnh báo riêng cho bộ nhãn tự sinh: caption do Gemini sinh **từ chính keyframe đáp án**, mà keyframe đó có thể chứa chữ. Nếu Gemini chép lại chữ trên màn hình vào caption thì kênh OCR sẽ khớp gần như hoàn hảo và hybrid trông xuất sắc một cách giả tạo. **Đối chiếu với kết quả BM25 ở P0.9** — BM25 cao bất thường là dấu hiệu của đúng bệnh này. Kết quả vẫn phải báo cáo đúng như đo được, không cố làm cho hybrid "trông tốt hơn".

---

## P1.4 — Baseline so sánh + chốt encoder

**Làm gì:** hoàn thiện bảng baseline bắt buộc của PRD §6.3.

| Baseline | Mục đích | Có từ |
|---|---|---|
| Random | Sàn tuyệt đối | P0.9 |
| BM25 trên caption | Kiểm tra bài toán có "rò rỉ" giải được bằng từ khoá thuần | P0.9 |
| **CLIP ViT-B/32** | Baseline multimodal phổ biến | **P1.4** |
| BEiT-3 (hoặc SigLIP nếu R1 buộc rút lui) | Hệ thống chính | P0.5 |
| BEiT-3 + Hybrid RRF | Đo **đóng góp riêng** của fusion | P1.3 |

**Nếu P0.5 đã phải rút lui sang SigLIP:** đây là lúc làm **spike BEiT-3 lần 2** (timebox mới, tách khỏi đường găng) và so số trực tiếp SigLIP ↔ BEiT-3. Số đo quyết định, không phải "paper nói vậy".

**Verify:** bảng 5 dòng, cùng bộ eval, cùng seed, cùng mẫu số video.

---

## P1.5 — 🛑 GATE: DỪNG VÀ HỎI VỀ THIẾT KẾ UI

**Không được viết bất kỳ dòng UI nào trước bước này.** (CLAUDE.md §9 · PRD §9 · architecture.md §8)

Mô tả UI trong paper baseline và PRD §F9 chỉ là **danh sách năng lực chức năng**, **không** phải bản thiết kế phải sao chép. Chủ dự án đã xác nhận sẽ gửi mẫu thiết kế mong muốn trong P1.

**Điều kiện qua gate:** đã nhận mẫu thiết kế từ chủ dự án.

---

## P1.6 — Frontend

**Làm gì:** React + Vite + TypeScript **bám theo mẫu thiết kế đã nhận**, không tự sáng tác.

Ràng buộc **kỹ thuật** (độc lập với hình thức giao diện, giữ nguyên trong mọi thiết kế):
- **Virtualized rendering** cho lưới hàng nghìn thumbnail
- **Shortcut bàn phím** cho thao tác nóng (submit, next/prev, đổi mode)
- **Đổi mode không mất truy vấn đang gõ**
- `video_id` + timestamp hiển thị **ngay trên** thumbnail, không cần hover
- Phục vụ thumbnail nén, tải ảnh gốc theo yêu cầu

**Verify:**
- Grid mượt với ≥ 1.000 thumbnail (đo FPS lúc cuộn)
- Đổi mode giữa lúc đang gõ → ô nhập giữ nguyên nội dung
- Chạy thử một phiên tìm kiếm thật, bấm giờ

---

## P1.7 — Đo latency & Rank-in-Grid

**Làm gì:** lấp các ô ⚠️ trong PRD §6.6 bằng số thật.

| Chế độ | Mục tiêu p50 | Mục tiêu p95 |
|---|---|---|
| Semantic | ≤ 150 ms | ≤ 300 ms |
| OCR (ES) | ≤ 100 ms | ≤ 250 ms |
| Hybrid (RRF) | ≤ 250 ms | ≤ 500 ms |

- Đo từ lúc nhận request đến lúc trả response, **không tính render**.
- **`Rank-in-Grid`**: vị trí của kết quả đúng trong lưới mà người vận hành phải cuộn tới. Chạy tự động trên toàn bộ bộ eval — proxy rẻ cho `TTFC`, tương quan mạnh nhưng không cần tổ chức phiên thao tác có người.
- Mỗi request search log: mode · độ trễ · số kết quả · cache hit/miss.

**Verify:** bảng latency thật thay cho ⚠️ trong PRD §6.6.

---

## P1.8 — Degraded mode

**Làm gì:** kiểm chứng yêu cầu T5 — hệ thống sống sót khi thiếu thành phần.

- Tắt Elasticsearch ⇒ tự chuyển **semantic-only**, không crash, thông báo rõ ràng trên UI (giải phóng ~2 GB RAM).
- Mất mạng ⇒ semantic search vẫn chạy đầy đủ; chỉ QUEST / OCR / VQA ngừng.

**Verify:** test tích hợp có tắt ES; `test_search_falls_back_to_semantic_when_es_unavailable`.

---

## Định nghĩa "xong" của P1

- [ ] Recall@10 đạt mục tiêu PRD §6.7 (bảng đã hiệu chỉnh ở P0.10)
- [ ] `test_hybrid_not_worse_than_best_single_channel` pass — hoặc F4 đã bị bỏ kèm lý do
- [ ] Bảng so sánh 5 baseline đầy đủ
- [ ] Encoder chính đã chốt (BEiT-3 hay SigLIP) **dựa trên số đo**
- [ ] Bảng latency p50/p95 thật
- [ ] UI dùng được để thao tác thật, bám mẫu thiết kế đã nhận
- [ ] Degraded mode chạy được
- [ ] Báo cáo phase theo template CLAUDE.md §8

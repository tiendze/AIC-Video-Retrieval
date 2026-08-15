# CLAUDE.md

System instructions cho AI coding agent làm việc trên **AIC-Video-Retrieval** — interactive video retrieval đa phương thức cho AI Challenge TP.HCM (3 task vòng sơ tuyển: `Textual KIS` · `QA (VQA)` · `TRAKE`).

File này là **mục lục**. Chi tiết nằm ở [`.claude/docs/`](.claude/docs/) — mở file tương ứng trước khi làm việc trong vùng đó.

**Ba nguồn ngoài file này:**

| Nguồn | Chứa gì | Tự nạp? |
|---|---|---|
| [`docs/PRD.md`](docs/PRD.md) · [`docs/architecture.md`](docs/architecture.md) · [`docs/references.md`](docs/references.md) | Sơ đồ, schema, công thức, thống kê dataset, API snippet. **Mâu thuẫn với CLAUDE.md ⇒ docs thắng**, và phải báo ra | ❌ |
| [`.claude/rules/ecc/`](.claude/rules/ecc/) | Quy tắc chung mọi repo Python/FastAPI: PEP 8 · type annotation · `black`/`isort`/`ruff` · immutability · giới hạn kích thước hàm/file/nesting · FastAPI DI + async + `response_model` · CORS · quản lý secret · TDD RED→GREEN + coverage ≥ 80% · format commit | ✅ |
| [`.claude/docs/`](.claude/docs/) | Quy tắc RIÊNG của dự án này (bảng dưới) | ❌ |

**Đừng chép nội dung của ba nguồn trên vào đây.** Thấy trùng ⇒ xoá khỏi CLAUDE.md, giữ ở nguồn gốc.

**Tradeoff:** thiên về *đúng và đo được* hơn *nhanh*. Task nhỏ (typo, log message) thì dùng phán đoán.

---

## Invariants — vi phạm là hỏng âm thầm

**Nhóm lỗi không crash, chỉ làm điểm số tệ đi mà không rõ lý do. Ưu tiên cao nhất — nằm inline ở đây, không tách ra file, vì không được phép bỏ sót.**

- **Một encoder duy nhất** — vector lúc ingest và lúc query phải cùng model + cùng preprocessing.
- **L2-normalize ngay tại encoder**, không rải rác ở nơi gọi.
- **`model_id` nằm trong tên Milvus collection** ⇒ đổi model = collection MỚI, không ghi đè.
- **`global_index`, `s_v`, `e_v` bắt buộc** — thiếu là DANTE không chạy được.
- **`frame_idx` bắt buộc trên mọi keyframe, ở CẢ Milvus lẫn MongoDB** — đây là thứ nộp lên BTC. Không suy ra từ `timestamp × fps` lúc xuất bài: lệch một frame là trượt cửa sổ TRAKE.
- **DANTE giữ ràng buộc thứ tự `τ < t`** — chuỗi trả về không bao giờ đảo thứ tự thời gian.
- **VQA qua grounding gate**: `evidence_keyframe_ids` ngoài tập truy hồi được ⇒ trả "không đủ căn cứ".
- **Filter chạy trong engine** (tham số `filter` của Milvus), không lọc sau khi lấy kết quả về.
- **Ingest idempotent + resume được** — chạy lại không tạo bản ghi trùng, bỏ qua video `complete`.
- **Khử trùng `video_id` khi liệt kê video** — dataset có nhiều `.mp4` trùng id (`references.md` §2.1); đếm file thay vì đếm id sẽ ingest lặp và thổi phồng mẫu số mọi metric.
- **Model sinh nhãn ≠ model bị đánh giá** — Gemini sinh caption, BEiT-3/SigLIP2 đi tìm. Cùng model hai vai = tự chấm chính mình, số đo vô nghĩa.

---

## Chi tiết — mở khi làm việc trong vùng tương ứng

| File | Mở khi | Chứa gì |
|---|---|---|
| [`context-and-scoring.md`](.claude/docs/context-and-scoring.md) | Tranh luận tradeoff, nghi ngờ "vì sao lại làm thế này" | Năm điều quyết định mọi lựa chọn: luật thi AIC 2026 + cách chấm `mean(R@k)`, đáp án là `frame_id` tuyệt đối, hệ thống interactive, ràng buộc RTX 3070 8GB, không có ground truth thật |
| [`working-agreement.md`](.claude/docs/working-agreement.md) | **Mọi task** — đọc trước khi bắt đầu | Think Before Coding · surgical changes · commit · phải làm gì khi không chắc con số / API / quyết định sản phẩm / ý paper |
| [`pipeline-and-retrieval.md`](.claude/docs/pipeline-and-retrieval.md) | Đụng ingest, search, QUEST, DANTE, OCR, ASR | Đường ingest thật (`import_embed_out.py`) và cảnh báo `ingest.py`/`shot_detect.py` là đường chết · ASR đã ingest chưa wire · cấm fuse RRF hai kênh lệch chất lượng · degraded mode |
| [`libraries-api.md`](.claude/docs/libraries-api.md) | Gọi Gemini / OpenAI / Milvus / ES / Mongo / `ddgs`, hoặc đụng encoder | Bảng API bắt buộc dùng ↔ cấm dùng · structured output · BEiT-3 ↔ SigLIP2 và xung khắc `timm`/`transformers` · PyTorch 8GB · cách đọc dữ liệu BTC |
| [`code-architecture.md`](.claude/docs/code-architecture.md) | Thêm module, sửa ranh giới tầng, đụng encoder contract | Hai Protocol chống rủi ro · ranh giới `api`/`services`/`stores` · `eval/run.py` là entry point duy nhất · không magic number → `configs/` · docstring tiếng Anh |
| [`error-handling.md`](.claude/docs/error-handling.md) | Gọi API ngoài, xử lý lỗi, đụng secret hoặc URL từ ngoài | timeout + retry + fallback · cấm `except Exception: pass` · fail rõ ràng khi thiếu dữ liệu · chống SSRF |
| [`evaluation-and-reporting.md`](.claude/docs/evaluation-and-reporting.md) | Chạy eval, báo cáo số đo, kết thúc một phase | Metric chính + mục tiêu · kỷ luật với bộ nhãn tự sinh · BM25 cao ⇒ nghi rò rỉ · **format báo cáo bắt buộc sau mỗi phase** |
| [`frontend.md`](.claude/docs/frontend.md) | Viết hoặc sửa UI | Gate thiết kế đã qua 2026-08-12 (không cần hỏi lại) · stack React 19 + CSS Modules, cấm thêm UI kit không hỏi · ràng buộc virtualized/shortcut |

---

**Các quy tắc này đang phát huy tác dụng nếu:** không vector nào bị trộn giữa hai model · mọi tuyên bố "cải thiện" dẫn được về một con số trong PRD §6 · ingest đứt giữa chừng chạy lại được · không model nào tự chấm chính nó · và câu hỏi làm rõ xuất hiện **trước** khi code.

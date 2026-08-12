# CLAUDE.md

System instructions cho AI coding agent làm việc trên **AIC-Video-Retrieval** — interactive video retrieval đa phương thức cho AI Challenge TP.HCM.

Nguồn sự thật: [`docs/PRD.md`](docs/PRD.md) · [`docs/architecture.md`](docs/architecture.md) · [`docs/references.md`](docs/references.md). File này chỉ chứa **quy tắc** — sơ đồ, schema, công thức, thống kê dataset, API snippet đều nằm trong docs, **đừng chép lại vào đây**. Mâu thuẫn giữa hai bên ⇒ **docs thắng**, và phải báo mâu thuẫn ra.

**Tradeoff:** thiên về *đúng và đo được* hơn *nhanh*. Task nhỏ (typo, log message) thì dùng phán đoán.

---

## 1. Bối cảnh — bốn điều quyết định mọi lựa chọn

1. **3 task thi vòng sơ tuyển AIC 2026:** `Textual KIS` · `QA (VQA)` · `TRAKE`. **Không còn Video KIS.** Mỗi truy vấn nộp **tối đa 100 đáp án đã xếp hạng**; điểm = trung bình `R@k` với **k ∈ {1, 5, 20, 50, 100}** (`references.md` §1.2). ⇒ **thứ hạng mới là thứ ăn điểm**, không phải tốc độ; latency tụt xuống yêu cầu vận hành.
2. **Đáp án là `<video_id>, <frame_id>`** — `frame_id` là **chỉ số frame tuyệt đối**, không phải giây, không phải số thứ tự keyframe. TRAKE chấm theo cửa sổ `[sⱼ, eⱼ]` thường **< 10 frame** ⇒ sai một phép quy đổi là mất điểm im lặng.
3. **Hệ thống interactive** — con người chốt đáp án cuối. Đừng tự động hoá chỗ paper cố ý để người vận hành quyết định.
4. **Ràng buộc cứng:** một máy, **RTX 3070 8GB VRAM**. ⚠️ Đổi từ RTX 2080Ti 11GB khi chuyển từ server sang laptop cá nhân (2026-08-01) — **ít VRAM hơn**, batch size/ngân sách phải thắt lại chứ không nới ra.
5. **Không có ground truth thật.** Dữ liệu AIC 2026 công khai (chi tiết: `references.md` §2) **không kèm nhãn truy vấn nào**; đáp án chỉ biết qua nộp bài. Bộ eval là **nhãn tự sinh** (giao thức: PRD §6.1) ⇒ mọi con số là **tương đối**, và phải hiệu chỉnh qua nộp bài BTC (PRD §6.9).

Baseline [arXiv:2512.13169](https://arxiv.org/abs/2512.13169) đóng góp **QUEST** và **DANTE**, nhưng **không công bố metric định lượng nào**. Vì vậy `eval/` là điều kiện để mọi tuyên bố "cải thiện" có nghĩa.

---

## 2. Think Before Coding

**Không đoán. Không giấu chỗ mơ hồ. Nêu tradeoff.**

- Nói rõ assumption trước khi implement. Không chắc thì hỏi.
- Nhiều cách hiểu → trình bày cả hai, đừng im lặng chọn một.
- Có cách đơn giản hơn → nói ra. Được phép phản biện.
- Code lệch docs → sửa docs trong **cùng** change, không để hai bên trôi khỏi nhau.
- Mỗi thay đổi ảnh hưởng kết quả phải trả lời được: *"đo bằng metric nào trong PRD §6?"*

---

## 3. Invariants — vi phạm là hỏng âm thầm

**Nhóm lỗi không crash, chỉ làm điểm số tệ đi mà không rõ lý do. Ưu tiên cao nhất.**

- **Một encoder duy nhất.** Vector lúc `ingest` và lúc `query` phải sinh từ cùng model + cùng preprocessing.
- **L2-normalize mọi embedding ngay tại encoder**, không rải rác ở nơi gọi.
- **`model_id` nằm trong tên Milvus collection.** Đổi embedding model ⇒ collection MỚI, không ghi đè.
- **`global_index`, `s_v`, `e_v` là bắt buộc** — thiếu là DANTE không chạy được.
- **`frame_idx` bắt buộc đi kèm mọi keyframe, ở CẢ Milvus lẫn MongoDB.** Đây là thứ nộp lên BTC. Không suy ra từ `timestamp × fps` lúc xuất bài — làm tròn sai một frame là trượt cửa sổ TRAKE.
- **DANTE giữ ràng buộc thứ tự `τ < t`.** Chuỗi trả về không bao giờ đảo thứ tự thời gian.
- **VQA phải qua grounding gate**: `evidence_keyframe_ids` nằm ngoài tập keyframe truy hồi được ⇒ trả "không đủ căn cứ".
- **Filter chạy trong engine** (tham số `filter` của Milvus), không lọc sau khi lấy kết quả về.
- **Ingest idempotent + resume được**: chạy lại không tạo bản ghi trùng, bỏ qua video đã `complete`.
- **Khử trùng `video_id` khi liệt kê video.** Dataset có nhiều file `.mp4` trùng id (`references.md` §2.1) — đếm file thay vì đếm id sẽ ingest lặp và thổi phồng mẫu số của mọi báo cáo metric.
- **Model sinh nhãn ≠ model bị đánh giá.** Gemini sinh caption/câu hỏi, BEiT-3 (hoặc SigLIP) đi tìm. Để cùng một model làm cả hai vai là hệ thống tự chấm chính nó — số đo vô nghĩa.

---

## 4. Pipeline & Retrieval

**Ảnh và chữ nằm trong CÙNG một không gian vector — mọi thứ khác là hệ quả.** Sơ đồ đầy đủ: `architecture.md` §3.

**Ingest** — 3 nhánh (VISION → Milvus · TEXT → Elasticsearch · METADATA → MongoDB). Trên 8GB VRAM **chạy tuần tự an toàn hơn song song**.

- **Tự trích keyframe bằng TransNetV2** + công thức 4-keyframe/shot (`PRD.md` §F1) — bám baseline. Keyframe BTC trích sẵn **chỉ dùng làm đối chứng**, không phải nguồn ingest.
- Gán `global_index` liên tục + ghi `s_v` / `e_v` **cùng lúc** với việc ghi keyframe.
- **Luôn lọc frame có chữ + dedupe perceptual hash TRƯỚC khi gọi Gemini OCR** (risk R2 — chi phí token, quy đổi ở `references.md` §5). Với video tin tức, **dedupe mới là bước tiết kiệm chính**, không phải lọc chữ.
- Video lỗi → error list riêng + `ingest_status = "failed"`, **pipeline chạy tiếp**. Checkpoint theo `video_id`.

**Retrieval:**

- Hai mode `semantic` · `ocr` là **tham số**, không phải hai code path sao chép nhau. `ocr` là BM25 thuần qua Elasticsearch, không dịch, không fuse tự động vào ranking.
- **Không fuse RRF hai kênh lệch chất lượng** (bài học 2026-08-12: `hybrid` semantic⊕ocr đã bị bỏ hẳn — RRF fuse theo rank kéo kênh mạnh xuống bất kể trọng số khi kênh kia yếu, và bộ nhãn tự sinh còn cấu trúc không đo công bằng được OCR vì prompt sinh caption cấm chép chữ màn hình). RRF (`k = 60`) chỉ dùng khi fuse **hai kênh cùng loại, chất lượng ngang nhau** (vd. hai encoder semantic) — nếu kém hơn kênh đơn tốt nhất trên bộ eval thì chỉnh trọng số hoặc bỏ hẳn.
- **QUEST giữ người trong vòng lặp**: hiển thị cả `q₀` lẫn `q_r`; người dùng **tự chọn** ảnh mẫu `I*` ở nhánh 2, không auto-pick. Cả hai nhánh có timeout → fallback về search thường, không treo UI.
- **DANTE** dùng bản `running max` + có **backtracking** để trả đúng N keyframe. `λ` là config, không hard-code.
- **Degraded mode phải chạy được**: tắt Elasticsearch ⇒ `semantic-only`; mất mạng ⇒ semantic search và DANTE vẫn đủ, chỉ QUEST / OCR / VQA ngừng.

---

## 5. Thư viện & API

**Đã fetch kiểm chứng 2026-07-25. Phần lớn tutorial trên mạng đang dạy API CŨ — đừng viết theo trí nhớ.**

| Thư viện | Bắt buộc dùng | KHÔNG dùng |
|---|---|---|
| Gemini (`google-genai`) | `client.interactions.create(...)` | `generate_content(...)`, `generation_config` |
| OpenAI (`openai`) | `client.responses.parse(..., text_format=Model)` | `chat.completions.create` |
| Milvus (`pymilvus`) | `MilvusClient` (v2.4+) | ORM API cũ |
| Elasticsearch | `from_=`; **bulk helpers** khi ingest | index từng document trong vòng lặp |
| MongoDB (`pymongo`) | **bulk write** + unique index `keyframe_id` / `video_id` | insert từng document; client đồng bộ trong `async def` |

- **OpenAI structured output phải xử lý nhánh `item.type == "refusal"`** trước khi đọc `item.parsed`.
- **Structured output cho cả OCR, VQA và sinh nhãn** qua `model_json_schema()` — không parse text tự do bằng regex.
- **TransNetV2**: bản `inference-pytorch/`, input `torch.uint8` shape `[B, T, 27, 48, 3]` → resize frame về **48×27**.
- **BEiT-3**: dùng `beit3_base_itc`. **Timebox 1 ngày** — quá thì chuyển `SigLIPEncoder` qua Protocol và ghi nhận là sai lệch so với baseline.
- **PyTorch trên 8GB VRAM**: `torch.no_grad()` + autocast fp16, batch size configurable, giải phóng cache giữa các giai đoạn.
- **Đọc dữ liệu BTC:** mọi giá trị trong CSV/JSON của BTC là **chuỗi** kể cả số; `publish_date` là **`DD/MM/YYYY`** không phải ISO-8601; tên thư mục **không nhất quán** nên phải glob. Chi tiết: `references.md` §2.
- Model ID (`gemini-3.6-flash`, `gpt-5.6`) **xác nhận lại lúc implement**. Cần API chưa có trong `references.md` → fetch `llms.txt` rồi cập nhật `references.md` trong cùng change.

---

## 6. Kiến trúc & Code Style

**Hai Protocol là hàng rào chống rủi ro, không được bypass:** `MultimodalEncoder` (chống risk R1 — BEiT-3 khó cài) và `DatasetAdapter` (đổi/mở rộng dataset chỉ thêm class, không sửa pipeline). Pipeline chỉ gọi qua interface, không import implementation trực tiếp.

- Ranh giới tầng: `app/api/` routers **mỏng** (validate + uỷ quyền) · `app/services/` chứa toàn bộ business logic · `app/stores/` chỉ nói chuyện với DB · `eval/run.py` là entry point **duy nhất** của evaluation. Cây thư mục: `architecture.md` §6.
- FastAPI: `create_app()`, model nạp qua **lifespan**, client inject qua `Depends` (không khởi tạo trong route), `async def` + async client cho endpoint I/O, mọi endpoint khai báo `response_model`, lỗi theo schema `{detail, error_code, request_id}`.
- Python: **PEP 8 + type annotation trên mọi chữ ký hàm**. Format `black` · `isort` · lint `ruff`.
- Immutable trước: `@dataclass(frozen=True)` cho value object. Không mutate tham số đầu vào.
- Hàm < 50 dòng · file ≤ 800 dòng · nesting ≤ 4 tầng (dùng early return).
- **Không magic number.** Ngưỡng, timeout, `top_k`, `λ`, `k` của RRF → `configs/`.
- Encoder contract: `np.ndarray` shape `(N, embedding_dim)`, `dtype=float32`, **đã L2-normalize** — ghi trong docstring và kiểm bằng test.
- Docstring/comment viết bằng **tiếng Anh**. Tên biến/hàm/file luôn tiếng Anh.

---

## 7. Error Handling & Secrets

- Mọi lời gọi API ngoài (Gemini, OpenAI, image search): **timeout + retry exponential backoff + fallback rõ ràng**.
- Lỗi nhánh phụ thuộc mạng **không được giết request chính**: VQA lỗi → vẫn trả keyframe grid kèm thông báo.
- **`except Exception: pass` bị cấm.** Bắt exception cụ thể, log kèm `video_id` / `keyframe_id` / `request_id`, rồi quyết định retry / skip / raise.
- **Thiếu dữ liệu phải fail rõ ràng, không trả rỗng.** Ví dụ: bộ nhãn chưa tồn tại ⇒ raise, đừng trả list rỗng khiến `eval/` in bảng metric toàn số 0 trông như "hệ thống hỏng".
- Lỗi hiện ra UI phải đọc được bởi người đang bị tính giờ — không dump traceback.
- API key nạp qua `pydantic-settings` từ `.env`, **validate ngay lúc khởi động**, không bao giờ commit. Log phải **che** key / token / authorization header.
- CORS theo môi trường, không kết hợp wildcard với credentials. Ảnh upload: giới hạn size, check MIME type, không lưu vĩnh viễn.

---

## 8. Goal-Driven Execution & Testing

**Biến task thành tiêu chí verify được, rồi lặp cho tới khi đạt.**

- "Thêm hybrid search" → *"viết test chứng minh hybrid ≥ kênh đơn tốt nhất, rồi làm nó pass"*.
- "Fix bug DANTE" → *"viết test tái hiện chuỗi bị đảo thứ tự, rồi làm nó pass"*.
- TDD: RED → GREEN → refactor. Coverage **≥ 80%**. Tên test mô tả hành vi: `test_encoder_output_is_l2_normalized`.
- **Không sang phase sau khi phase trước chưa có số đo** — bài học trực tiếp từ khoảng trống của paper.
- Metric chính: `Recall@K` / `MRR` (báo cáo **cả** video-level và segment-level) · `Sequence Exact-Match` + `Kendall's tau` (TRAKE) · `Grounded Accuracy` + `Hallucination Rate` (VQA). Mục tiêu cụ thể: PRD §6.7.

**Kỷ luật riêng vì nhãn là nhãn tự sinh:**

- **Không sửa bộ eval để cải thiện số.** Đổi bộ eval ⇒ tăng version và chạy lại toàn bộ baseline. Ranh giới giữa *"sửa nhãn sai"* và *"sửa nhãn cho số đẹp"* rất mỏng ⇒ lý do sửa phải viết ra **trước** khi nhìn số mới.
- Báo cáo metric luôn kèm: git SHA · model checkpoint · version index · **`eval_set_version` + hash bộ nhãn** · **số `video_id` duy nhất đánh giá được** · số truy vấn · seed.
- **BM25 cao bất thường ⇒ nghi bộ nhãn rò rỉ, không phải hệ thống giỏi** (caption sinh từ chính keyframe đáp án). Điều tra trước khi ăn mừng.

**Bắt buộc: báo cáo sau mỗi phase.** Kết thúc một phase (P0–P4 trong PRD §4) hoặc một bước lớn trong plan, **dừng lại giải thích trước khi đi tiếp** — viết bằng tiếng Việt, đủ để người đọc hiểu mà không cần mở code:

```
Đã làm gì   — module/file nào được tạo hoặc sửa, và vì sao cần chúng
Input       — dữ liệu / config / tham số đi vào, lấy từ đâu, định dạng ra sao
Output      — sinh ra cái gì (file · collection · index · metrics), nằm ở đâu
Cách verify — lệnh hoặc test chứng minh nó chạy đúng, kèm kết quả thực tế
Còn thiếu   — phần chưa xong, giả định đang treo, rủi ro đã phát hiện
```

- Giải thích **luồng dữ liệu**, không liệt kê diff. Người đọc cần hiểu *dữ liệu biến đổi thế nào*, không cần đọc lại code.
- Số liệu phải là **kết quả chạy thật**. Chưa chạy được thì ghi rõ "chưa đo", không suy đoán.

> Build / run commands: ghi vào `README.md` khi scaffold xong, **không** đưa vào file này.

---

## 9. Frontend — DỪNG LẠI VÀ HỎI

**Trước khi viết bất kỳ dòng UI nào, phải hỏi người chủ dự án về thiết kế giao diện.** Họ sẽ gửi mẫu mong muốn.

Mô tả UI trong paper và PRD §F9 chỉ là **danh sách năng lực chức năng**, không phải bản thiết kế phải sao chép. Ràng buộc kỹ thuật vẫn giữ: virtualized rendering cho lưới hàng nghìn thumbnail · shortcut bàn phím · đổi mode không mất truy vấn đang gõ.

---

## 10. Surgical Changes & Khi không chắc

- **Lệnh tốn tài nguyên (Docker `build`/`up`/`down`; train/inference model; ingest hàng loạt) → chuẩn bị sẵn script, nhưng để người chủ dự án tự chạy.** Đưa ra lệnh cụ thể, không tự ý thực thi.
- **Server không cho cài thư viện trên host — mọi lệnh cần thư viện phải chạy trong container đã cài sẵn.** Không tự cài package vào host, không tự exec vào container. Chỉ chuẩn bị script/lệnh (kèm ghi chú "chạy trong container").
- Chỉ đụng thứ cần đụng. Không "cải thiện" code/comment/format lân cận. Dead code có sẵn thì **báo**, đừng tự xoá.
- Commit `<type>: <description>` (`feat` · `fix` · `refactor` · `docs` · `test` · `chore` · `perf` · `ci`), atomic, không commit khi test đỏ.
- Không chắc **con số** → `[Cần điền]` / `⚠️ [Cần đo]`, đừng bịa. Câu hỏi đang treo: PRD §10.
- Không chắc **API thư viện** → fetch docs / `llms.txt`.
- Không chắc **quyết định sản phẩm** (format file nộp, thiết kế UI, giao thức sinh nhãn) → hỏi.
- Không chắc paper nói gì → `architecture.md` §10 phân biệt 📄 **[Paper]** với 🔧 **[Giả định]**. Sửa 🔧 thoải mái; sửa 📄 phải nêu lý do.

---

**Các quy tắc này đang phát huy tác dụng nếu:** không có vector nào bị trộn giữa hai model · mọi tuyên bố "cải thiện" dẫn được về một con số trong PRD §6 · ingest đứt giữa chừng chạy lại được · không có model nào tự chấm chính nó · và câu hỏi làm rõ xuất hiện **trước** khi code.

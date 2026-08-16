# CLAUDE.md

System instructions cho AI coding agent trên **AIC-Video-Retrieval** — interactive video retrieval đa phương thức cho AI Challenge TP.HCM (3 task vòng sơ tuyển: `Textual KIS` · `QA (VQA)` · `TRAKE`).

| Nguồn ngoài file này | Chứa gì | Tự nạp? |
|---|---|---|
| [`docs/PRD.md`](docs/PRD.md) · [`architecture.md`](docs/architecture.md) · [`references.md`](docs/references.md) | Sơ đồ, schema, công thức, thống kê dataset, API snippet. **Mâu thuẫn với file này ⇒ docs thắng**, và phải báo ra | ❌ mở khi cần |
| [`.claude/rules/ecc/`](.claude/rules/ecc/) | Quy tắc chung mọi repo Python/FastAPI: PEP 8 · type annotation · `black`/`isort`/`ruff` · immutability · giới hạn kích thước hàm/file · FastAPI DI + `response_model` · TDD RED→GREEN + coverage ≥ 80% · format commit | ✅ |

**Tradeoff:** thiên về *đúng và đo được* hơn *nhanh*. Task nhỏ (typo, log message) thì dùng phán đoán.

## §1 — Bối cảnh & cách chấm điểm

1. **3 task vòng sơ tuyển AIC 2026** (không còn Video KIS). Mỗi truy vấn nộp **tối đa 100 đáp án đã xếp hạng**, điểm = trung bình `R@k`, k ∈ {1,5,20,50,100} ⇒ **thứ hạng là thứ ăn điểm**, latency chỉ là yêu cầu vận hành.
2. **Đáp án là `<video_id>, <frame_id>`** — `frame_id` là **chỉ số frame tuyệt đối**, không phải giây, không phải số thứ tự keyframe. TRAKE chấm theo cửa sổ thường **< 10 frame** ⇒ sai một phép quy đổi là mất điểm im lặng.
3. **Hệ thống interactive** — người chốt đáp án cuối. Đừng tự động hoá chỗ paper cố ý để người vận hành quyết định.
4. **Một máy, RTX 3070 8GB VRAM** — batch size phải thắt chứ không nới; việc GPU local (encode lại, ASR, OCR) **chạy tuần tự an toàn hơn song song**.
5. **Không có ground truth thật.** Bộ eval là **nhãn tự sinh** (PRD §6.1) ⇒ mọi con số là **tương đối**, phải hiệu chỉnh qua nộp bài BTC (PRD §6.9).

Baseline [arXiv:2512.13169](https://arxiv.org/abs/2512.13169) đóng góp **QUEST** + **DANTE** nhưng **không công bố metric định lượng nào** ⇒ `eval/` là điều kiện để mọi tuyên bố "cải thiện" có nghĩa.

## §2 — Think before coding

**Không đoán. Không giấu chỗ mơ hồ. Nêu tradeoff.**

- Nói rõ assumption trước khi implement; nhiều cách hiểu → trình bày cả hai, đừng im lặng chọn một. Có cách đơn giản hơn → nói ra; được phép phản biện.
- Code lệch docs → sửa docs trong **cùng** change.
- Mỗi thay đổi ảnh hưởng kết quả phải trả lời được: *"đo bằng metric nào trong PRD §6?"*

## §3 — Invariants — vi phạm là hỏng âm thầm

**Nhóm lỗi không crash, chỉ làm điểm số tệ đi mà không rõ lý do. Ưu tiên cao nhất.**

- **Một encoder duy nhất** — vector lúc ingest và lúc query phải cùng model + cùng preprocessing.
- **L2-normalize ngay tại encoder**, không rải rác ở nơi gọi.
- **`model_id` nằm trong tên Milvus collection** ⇒ đổi model = collection MỚI, không ghi đè.
- **`global_index`, `s_v`, `e_v` bắt buộc** — thiếu là DANTE không chạy được.
- **`frame_idx` bắt buộc trên mọi keyframe, ở CẢ Milvus lẫn MongoDB** — đây là thứ nộp lên BTC. Không suy ra từ `timestamp × fps` lúc xuất bài: lệch một frame là trượt cửa sổ TRAKE.
- **DANTE giữ ràng buộc thứ tự `τ < t`** — chuỗi trả về không bao giờ đảo thứ tự thời gian.
- **VQA qua grounding gate**: `evidence_keyframe_ids` nằm ngoài tập keyframe **đã thực sự cấp cho model** ⇒ trả "không đủ căn cứ".
- **Filter chạy trong engine** (tham số `filter` của Milvus), không lọc sau khi lấy kết quả về.
- **Ingest idempotent + resume được** — chạy lại không tạo bản ghi trùng, bỏ qua video `complete`.
- **Khử trùng `video_id` khi liệt kê video** — dataset có nhiều `.mp4` trùng id (`references.md` §2.1); đếm file thay vì đếm id sẽ ingest lặp và thổi phồng mẫu số mọi metric.
- **Model sinh nhãn ≠ model bị đánh giá** — Gemini sinh caption, BEiT-3/SigLIP2 đi tìm. Cùng model hai vai = tự chấm chính mình, số đo vô nghĩa.

## §4 — Pipeline & retrieval

**Ảnh và chữ nằm trong CÙNG một không gian vector — mọi thứ khác là hệ quả.**

**Ingest — corpus sống nạp bằng đường import, không tự trích.** Entry point `pipelines/import_embed_out.py`: keyframe ở `data/keyframe/<batch>/<video_id>/`, shot boundary ở `data/shots.json`, embedding tính sẵn trên Kaggle ở `data/embed_out/`. `global_index` / `s_v` / `e_v` / `timestamp` / `fps` **dẫn xuất tại bước import**.

- ⚠️ `pipelines/ingest.py` + `shot_detect.py` (đường TransNetV2) **còn trong repo nhưng KHÔNG dùng** cho corpus hiện tại. Đổi hành vi ingest ⇒ sửa `import_embed_out.py`.
- Keyframe import không kèm bản resize (`thumbnail_path == image_path`) ⇒ `ThumbnailService` sinh theo yêu cầu + cache đĩa.
- **ASR đã ingest nhưng CHƯA nối vào retrieval** — không search mode nào đọc nó. Đừng coi là kênh tìm kiếm, đừng fuse vào ranking khi chưa đo trên bộ eval.
- Lỗi từng video/keyframe → error list riêng + `ingest_status="failed"`, pipeline **chạy tiếp**; checkpoint theo `video_id`.
- **Lọc frame có chữ + dedupe perceptual hash TRƯỚC khi gọi Gemini OCR** (risk R2 — chi phí token). Video tin tức: **dedupe mới là bước tiết kiệm chính**, không phải lọc chữ.

**Retrieval:**

- `semantic` · `ocr` là **tham số của một hàm**, không phải hai code path sao chép. `ocr` là BM25 thuần qua Elasticsearch — không dịch, không tự fuse vào ranking.
- **Không fuse RRF hai kênh lệch chất lượng.** `hybrid` semantic⊕ocr bỏ hẳn 2026-08-12. RRF (`k=60`) chỉ dùng cho **hai kênh cùng loại, chất lượng ngang nhau**.
- **QUEST giữ người trong vòng lặp** — hiện cả `q₀` lẫn `q_r`; người dùng **tự chọn** ảnh mẫu `I*` ở nhánh 2, không auto-pick. Cả hai nhánh có timeout → fallback, không treo UI.
- **DANTE** dùng `running max` + **backtracking** để trả đúng N keyframe. `λ` đọc từ `configs/search/dante.yaml`.
- **Degraded mode phải chạy được** — tắt Elasticsearch ⇒ `semantic-only`; mất mạng ⇒ semantic + DANTE vẫn đủ, chỉ QUEST / OCR / VQA ngừng.

## §5 — Thư viện & API

**Phần lớn tutorial trên mạng đang dạy API CŨ — đừng viết theo trí nhớ.**

| Thư viện | Bắt buộc dùng | KHÔNG dùng |
|---|---|---|
| Gemini (`google-genai`) | `client.interactions.create(...)` | `generate_content(...)`, `generation_config` |
| OpenAI (`openai`) | `client.responses.parse(..., text_format=Model)` | `chat.completions.create` |
| Milvus (`pymilvus`) | `MilvusClient` (v2.4+) | ORM API cũ |
| Elasticsearch | `from_=`; **bulk helpers** khi ingest | index từng document trong vòng lặp |
| MongoDB (`pymongo`) | **bulk write** + unique index | insert từng document; client đồng bộ trong `async def` |
| `ddgs` (QUEST nhánh 2) | `DDGS().images()` bọc `asyncio.to_thread` | gọi thẳng trong `async def` — không có async client |

- 🛑 **`gemini-2.5-flash` KHÔNG dùng cho bất kỳ nhánh nào cần structured output** (OCR · VQA · sinh nhãn). Thực nghiệm 2026-08-09: nó **lờ hoàn toàn `response_format`**, trả văn bản thường thay vì JSON đúng schema, trong khi cùng request thì `gemini-3.6-flash` trả đúng ⇒ **`gemini-3.6-flash` là bắt buộc, không phải tuỳ chọn** (`references.md` §5.1).
- **Structured output qua `model_json_schema()`** — không parse text tự do bằng regex. OpenAI phải xử lý nhánh `item.type == "refusal"` **trước** khi đọc `item.parsed`.
- **Encoder:** `BEiT3Encoder` đang wire trong `app/main.py`; `SigLip2Encoder` là ứng viên thay thế, chọn qua `app/encoders/factory.py`. Hai encoder **xung khắc dependency** (`timm==0.4.12` vs `transformers>=4.49`) — đừng import cả hai trong một process.
- **PyTorch trên 8GB VRAM:** `torch.no_grad()` + autocast fp16, batch size configurable, giải phóng cache giữa các giai đoạn.
- **Đọc dữ liệu BTC:** mọi giá trị CSV/JSON là **chuỗi** kể cả số; `publish_date` là **`DD/MM/YYYY`**; tên thư mục **không nhất quán** nên phải glob (`references.md` §2).
- Model ID **xác nhận lại lúc implement**. Cần API chưa có trong `references.md` → fetch `llms.txt` rồi cập nhật `references.md` trong cùng change.

## §6 — Kiến trúc code

**Hai Protocol là hàng rào chống rủi ro, không được bypass:** `MultimodalEncoder` (rút lui khỏi BEiT-3 không phải sửa pipeline) và `DatasetAdapter` (đổi/mở rộng dataset chỉ thêm class). Pipeline gọi qua interface, không import implementation.

- Ranh giới tầng: `app/api/` routers **mỏng** · `app/services/` business logic · `app/stores/` chỉ nói chuyện với DB · `eval/run.py` là entry point **duy nhất** của evaluation.
- Encoder/model nạp qua **lifespan**, không nạp trong route. Lỗi API theo schema `{detail, error_code, request_id}`.
- **Không magic number** — ngưỡng, timeout, `top_k`, `λ`, `k` của RRF → `configs/`.
- Encoder contract: `np.ndarray` shape `(N, embedding_dim)`, `float32`, **đã L2-normalize** — ghi trong docstring, kiểm bằng test.
- Docstring/comment viết bằng **tiếng Anh**; tên biến/hàm/file luôn tiếng Anh.

## §7 — Lỗi & secret

- Mọi API ngoài (Gemini, OpenAI, image search): **timeout + retry exponential backoff + fallback rõ ràng**.
- Lỗi nhánh phụ thuộc mạng **không được giết request chính** — VQA lỗi vẫn trả keyframe grid kèm thông báo.
- **`except Exception: pass` bị cấm.** Bắt exception cụ thể, log kèm `video_id` / `keyframe_id` / `request_id`, rồi quyết định retry / skip / raise.
- **Thiếu dữ liệu phải fail rõ ràng, không trả rỗng** — bộ nhãn chưa tồn tại ⇒ raise, đừng để `eval/` in bảng toàn số 0 trông như "hệ thống hỏng".
- Lỗi hiện ra UI phải đọc được bởi người đang bị tính giờ — không dump traceback.
- Secret nạp qua `pydantic-settings` (`app/config.py`), validate ngay lúc khởi động.
- Ảnh upload: giới hạn size, check MIME, không lưu vĩnh viễn. **URL từ nguồn ngoài không bao giờ nhận từ client** — server giữ URL thật trong cache, client chỉ gửi ID tham chiếu (pattern QUEST nhánh 2). Chống SSRF.

## §8 — Đo đạc & báo cáo phase

**Biến task thành tiêu chí verify được, rồi lặp cho tới khi đạt** — *"Fix bug DANTE"* → *"viết test tái hiện chuỗi bị đảo thứ tự, rồi làm nó pass"*.

- **Không sang phase sau khi phase trước chưa có số đo** — bài học trực tiếp từ khoảng trống của paper.
- Metric chính: `Recall@K` / `MRR` (báo cáo **cả** video-level và segment-level) · `Sequence Exact-Match` + `Kendall's tau` (TRAKE) · `Grounded Accuracy` + `Hallucination Rate` (VQA). Mục tiêu: PRD §6.7.
- **Không sửa bộ eval để cải thiện số.** Đổi bộ eval ⇒ tăng version + chạy lại toàn bộ baseline; lý do sửa phải viết ra **trước** khi nhìn số mới.
- Báo cáo metric luôn kèm: git SHA · model checkpoint · `eval_set_version` + hash bộ nhãn · **số `video_id` duy nhất đánh giá được** · số truy vấn · seed.
- **BM25 cao bất thường ⇒ nghi bộ nhãn rò rỉ**, không phải hệ thống giỏi. Điều tra trước khi ăn mừng.

**Bắt buộc — kết thúc mỗi phase (P0–P4) hoặc mỗi bước lớn trong plan, dừng lại giải thích bằng tiếng Việt trước khi đi tiếp:**

```
Đã làm gì   — module/file nào được tạo hoặc sửa, và vì sao cần chúng
Input       — dữ liệu / config / tham số đi vào, lấy từ đâu, định dạng ra sao
Output      — sinh ra cái gì (file · collection · index · metrics), nằm ở đâu
Cách verify — lệnh hoặc test chứng minh nó chạy đúng, kèm kết quả thực tế
Còn thiếu   — phần chưa xong, giả định đang treo, rủi ro đã phát hiện
```

Giải thích **luồng dữ liệu**, không liệt kê diff. Số liệu phải là **kết quả chạy thật** — chưa chạy được thì ghi "chưa đo", không suy đoán. Build/run commands nằm ở `README.md`.

## §9 — Frontend

**Gate thiết kế UI đã qua 2026-08-12** — build theo danh sách năng lực chức năng của paper, **không cần hỏi lại trước khi viết UI**; chủ dự án yêu cầu chỉnh giao diện thì cứ chỉnh.

Stack `frontend/`: React 19 + Vite + TypeScript · **CSS Modules viết tay, KHÔNG UI kit, KHÔNG state library** (cố ý — để lần restyle sau không phải gỡ abstraction của thư viện nào). Test: Vitest + React Testing Library. Lint: `oxlint`.

- Thêm thư viện UI mới (component kit, styling framework, state manager) ⇒ **phải hỏi trước**.
- **Ràng buộc kỹ thuật:** virtualized rendering cho lưới nhiều thumbnail · shortcut bàn phím · đổi mode không mất truy vấn đang gõ · phục vụ thumbnail nén, tải ảnh gốc theo yêu cầu.
- Tính năng chưa có backend: hiển thị **disabled kèm badge phase** (`"P2"`/`"P3"`), không ẩn hẳn.

## §10 — Surgical changes & khi không chắc

- **Lệnh tốn tài nguyên (Docker `build`/`up`/`down`; train/inference model; ingest hàng loạt) → chuẩn bị script, để chủ dự án tự chạy.** Đưa lệnh cụ thể, không tự ý thực thi.
- Chỉ đụng thứ cần đụng. Không "cải thiện" code/comment/format lân cận. Dead code có sẵn thì **báo**, đừng tự xoá.
- Commit atomic, **không commit khi test đỏ**.
- Không chắc **con số** → `[Cần điền]` / `⚠️ [Cần đo]`, đừng bịa. Câu hỏi treo: PRD §10.
- Không chắc **API thư viện** → fetch docs / `llms.txt`.
- Không chắc **quyết định sản phẩm** (format file nộp, giao thức sinh nhãn, thêm dependency lớn) → hỏi.
- Không chắc paper nói gì → `docs/architecture.md` §10 phân biệt 📄 **[Paper]** với 🔧 **[Giả định]**. Sửa 🔧 thoải mái; sửa 📄 phải nêu lý do.

**Các quy tắc này đang phát huy tác dụng nếu:** không vector nào bị trộn giữa hai model · mọi tuyên bố "cải thiện" dẫn được về một con số trong PRD §6 · ingest đứt giữa chừng chạy lại được · không model nào tự chấm chính nó · và câu hỏi làm rõ xuất hiện **trước** khi code.

# Research — Kênh ASR (Automatic Speech Recognition)

> **Trạng thái: ĐÃ DUYỆT 2026-08-02** — thành **P0.11** (`.claude/plans/p0-foundation.plan.md` §P0.11). `PRD.md` §2.3 + Q3 và `references.md` §5.3/§8/§9 **đã sửa xong**. Còn `architecture.md` §3.2/§6/§10 — sửa **cùng change với code** (CLAUDE.md §2).
>
> Phạm vi: thêm nhánh ASR vào pipeline ingest, song song nhánh OCR (P0.7) và nhánh VISION đã có.
>
> ---
>
> ### ⚠️ File này STALE 5 chỗ — đọc phải đối chiếu
>
> Nó viết **trước khi P0.7 nối xong OCR**. Đã kiểm lại working tree; kế hoạch thật đi theo cột bên phải:
>
> | § | File này nói | Sự thật |
> |---|---|---|
> | §2.1:47 | "nhánh OCR **chưa** được gọi từ `ingest_video()`" | **Sai.** Đã nối đầy đủ — `ingest.py:414` gọi `_run_ocr_branch` (`:298-363`), chuỗi trạng thái chạy tới `complete` |
> | §2.1:46 | hằng số `_SKIP_STATUSES` | Tên thật là **`_TERMINAL_STATUSES`** (`ingest.py:68`) `= {ocr_done, complete}` |
> | §6 | "`EsStore` nhận `index_name` ⇒ **dùng lại được**" | **Sai.** `ensure_index()` + `bulk_index()` **hard-code** mapping OCR và field `ocr_text` (`es_client.py:58-84`) ⇒ index thứ hai cần **store class riêng** |
> | §6.2 | tách `ocr_status`/`asr_status` trên `VideoDocument` | **Đã đổi hướng.** `upsert_video()` là **full `replace_one`** ⇒ field mới bị ingest chạy lại ghi đè im lặng. Dùng **collection Mongo riêng `asr_runs`** — cũng có nghĩa không sửa dòng nào của code ingest đang chạy |
> | §8 | đánh số ASR là "P0.8" | **Đụng tên** — P0.8 đã là search + `/health`. Dùng **P0.11** |
>
> Ngoài ra §4 ngầm giả định `VideoInfo` có `duration` — **không có**, nó chỉ gồm `fps` + `frame_count` (`video_io.py:23-26`).
>
> ### ✅ Ba [Cần verify] của §4 đã đóng (fetch 2026-08-02, chi tiết `references.md` §5.3)
>
> - `faster-whisper` decode bằng **PyAV (bundle FFmpeg)** ⇒ **đọc thẳng `.mp4`, không cần extract audio** — đúng như §4 phỏng đoán.
> - `Segment` **thật sự có** `no_speech_prob` + `avg_logprob` ⇒ bộ lọc ảo giác §5a chạy được.
> - **PhoWhisper-large KHÔNG có bản CTranslate2** và không phơi `no_speech_prob` ⇒ không phải baseline, là bước nâng cấp có đo.
>
> ### 🔄 §3 đã được mở rộng
>
> §3 kết luận "index mức segment" — **vẫn giữ**. Nhưng **BM25-only không phải end state**: lời nói là văn nói tự phát, khoảng cách paraphrase **lớn hơn OCR về mặt cấu trúc** (chữ trên màn hình đã ở dạng từ khoá, lời dẫn thì không). Nên P0.11 tách **P0.11a** (→ ES/BM25) và **P0.11b** (→ Milvus/dense, cần encoder thứ ba vì BEiT-3 là image–text chứ không phải text–text).
>
> ### ✅ §9 — bốn câu hỏi treo đã chốt
>
> **A** đảo non-goal: có · **B** `large-v3` làm baseline · **C** P0.11, ngoài đường găng, **trước** eval · **D** có làm tập truy vấn ASR-aware, kèm **đối chứng âm** (chỉ giữ truy vấn mà thực thể được nói ra nhưng **không có trong index OCR**) — phép đo duy nhất không làm giả được.

---

## 1. Vì sao ASR đáng làm

| Bằng chứng | Nguồn |
|---|---|
| Dữ liệu là video **tin tức tiếng Việt** (HTV — *60 Giây Official*), **có tiếng nói** và chữ trên màn hình | `references.md:86` |
| ASR được xếp **ưu tiên cao nhất** trong danh sách tối ưu giai đoạn 2 | `references.md:744` |
| Lý do: lời dẫn nói ra nhiều thứ **không** hiện thành chữ trên màn hình (tên người, địa danh, số liệu) ⇒ **bổ trợ**, không trùng lặp với OCR | `references.md:739` |
| Quy mô: **873 video, 470.428 giây ≈ 130,7 giờ** | `references.md:92-94` |

**Dataset KHÔNG kèm transcript/phụ đề sẵn.** `datasets/aic2026.py:75-84,149-150` chỉ đọc `author, title, publish_date, watch_url, channel_id, length` từ `media-info`. ⇒ phải tự chạy ASR, không có đường tắt parse file có sẵn.

**Hệ quả thực tế:** đây là job compute **một lần, chạy nền, không phụ thuộc Gemini API key** — nên khởi động được ngay trong lúc P0.7 còn chờ key.

---

## 2. Hiện trạng code liên quan

### 2.1 Pipeline ingest

Entry: `scripts/run_ingest.py:126` → `run_ingest()` (`pipelines/ingest.py:272`) → `ingest_video()` (`pipelines/ingest.py:236-269`).

Luồng hiện tại trong `ingest_video()`:

```
next_global_index_start()        ingest.py:84-86
  → upsert VideoDocument (STATUS_DECODING)   ingest.py:251
  → detect_shots(source_path)    ingest.py:253   (decode lần 1)
  → read_video_info(source_path) ingest.py:254
  → keyframe_frame_indices()     ingest.py:256
  → extract_keyframe_images()    ingest.py:257   (decode lần 2, cùng decoder để frame_idx không lệch)
  → build_extracted_keyframes()  ingest.py:117-146  (gán global_index liên tục)
  → save_keyframe_images()       ingest.py:149-159
  → mongo.bulk_insert_keyframes()ingest.py:263
  → embed_and_write()            ingest.py:265   (BEiT-3 → Milvus)
  → upsert VideoDocument (STATUS_EMBEDDED, s_v/e_v)  ingest.py:267-269
```

- **Resume**: `_SKIP_STATUSES = {"embedded", "ocr_done", "complete"}` (`ingest.py:57`), đọc từ `mongo.get_video()` (`ingest.py:289`).
- **Nhánh OCR (`pipelines/ocr.py`) chưa được gọi từ `ingest_video()`** — cố ý, vì cần `GEMINI_API_KEY` thật và tốn tiền mỗi call (`ingest.py:1-17`). `STATUS_EMBEDDED` hiện là trạng thái tận cùng thực tế.

### 2.2 Đơn vị dữ liệu ở các store

| Store | Đơn vị | Field |
|---|---|---|
| MongoDB `keyframes` | keyframe | `keyframe_id, video_id, shot_id, frame_idx, timestamp, global_index, image_path, thumbnail_path` (`mongo_client.py:28-36`) |
| MongoDB `videos` | video | `video_id, s_v, e_v, duration, fps, num_keyframes, ingest_status, author, batch, title, publish_date, watch_url, channel_id` (`mongo_client.py:40-53`) |
| Milvus `keyframes_{model_id}` | keyframe | `keyframe_id` (PK), `embedding`, `video_id`, `global_index`, `frame_idx`, `timestamp`, `author`, `batch` (`milvus_client.py:62-104`) |
| Elasticsearch | keyframe | `OcrDocument`: `keyframe_id, ocr_text, video_id, timestamp` (`es_client.py:21-26`) |

⚠️ **Không tồn tại đơn vị mức shot/segment.** `shot_id` chỉ là string gắn trên `KeyframeDocument`, sinh ở `ingest.py:130`. Cặp `Shot(start_frame, end_frame)` chỉ sống trong RAM lúc `ingest_video()` chạy, **không được ghi lại ở đâu**.

⚠️ **`OcrDocument` thiếu `frame_idx`** trong khi Mongo và Milvus đều có. Lệch với tinh thần invariant CLAUDE.md §3 (`frame_idx` là thứ nộp BTC). Nên vá cùng lúc làm ASR, đừng nhân bản lỗi sang index thứ hai.

### 2.3 Config & dependency

- `app/config.py` (pydantic-settings): `gemini_api_key, openai_api_key, milvus_uri, milvus_token, elasticsearch_url, elastic_api_key, mongo_uri, cors_origins, app_env` — **không có key nào cho ASR/audio**.
- `configs/`: `model/beit3.yaml`, `model/transnetv2.yaml`, `stores/milvus.yaml`, `ingest/ocr.yaml`. Loader là `_load_yaml()` cục bộ (`scripts/run_ingest.py:87-89`, `yaml.safe_load`), không phải pydantic-settings.
- `pyproject.toml`: `torch==2.4.1`, `torchvision==0.19.1`. **Không có bất kỳ thư viện audio nào** (không `ffmpeg-python`, `av`, `librosa`, `soundfile`, `faster-whisper`). `pipelines/video_io.py:17` dùng `cv2.VideoCapture` — OpenCV **không đọc audio track**.
- GPU: **RTX 3070 8GB** — `CLAUDE.md` §1.4, `PRD.md:595`, `architecture.md:591` đã thống nhất (không còn mâu thuẫn với commit `2a4de45`).

---

## 3. Quyết định kiến trúc quan trọng nhất — đơn vị lưu transcript

**Khuyến nghị: index ở mức SEGMENT, không ép về keyframe.**

Repo hiện tại đẩy về hướng keyframe theo quán tính (mọi store đều mức keyframe). Đây là chỗ dễ hỏng chất lượng nhất.

**Lý do không ép về keyframe:**

1. Một câu lời dẫn dài 4–8 giây thường **trải qua nhiều shot**; ngược lại một shot studio 15 giây chứa 3–4 câu. Cắt transcript theo ranh giới keyframe làm **vỡ câu** ⇒ BM25 mất ngữ cảnh, đúng thứ ăn điểm của kênh text.
2. Segment → keyframe **suy ra được** (lọc `timestamp ∈ [start_time, end_time]`). Keyframe → segment **không suy ngược được**. Segment là biểu diễn không mất thông tin.

**Schema đề xuất** — ES index thứ hai, mỗi document = 1 ASR segment, mang cả hai hệ quy chiếu:

```
video_id         keyword
segment_id       keyword
transcript_text  text, analyzer = vi_analyzer   (dùng lại của OCR, es_client.py:44-66)
start_time       float   # giây — hiển thị & debug
end_time         float
start_frame_idx  long    # frame tuyệt đối — thứ nộp BTC
end_frame_idx    long
```

**Tầng retrieval** expand segment → keyframe: lấy keyframe có `timestamp` trong cửa sổ (mở rộng, xem §5b), đưa vào RRF cùng kênh semantic + OCR. Fan-out này là **tính năng**: một câu nói khớp truy vấn tự nhiên đề cử nhiều frame ứng viên.

---

## 4. Chọn model

| Bậc | Model | Đánh đổi |
|---|---|---|
| **Mặc định (baseline)** | `faster-whisper` + `large-v3`, compute type `int8_float16` | Chín, ổn định, VRAM ~3GB (thoải mái trên 8GB). Timestamp segment-level trôi ~0,5–2s — chấp nhận được vì ta expand cửa sổ. ⚠️ **[Cần đo]** VRAM và tốc độ thực tế |
| **Nâng chất** | `PhoWhisper-large` (VinAI, fine-tune tiếng Việt) | WER tiếng Việt tốt hơn Whisper gốc. Rủi ro: bản HF transformers chậm hơn CTranslate2; fine-tune **có thể làm lệch timestamp**. ⚠️ **[Cần verify]** có bản convert CTranslate2 chưa |

**Cách làm an toàn:** bọc bằng `Protocol` y hệt `OcrClient` (`pipelines/ocr.py:66-70`) — đã là pattern của repo. Chạy `large-v3` trước để có đường cơ sở chạy được, swap `PhoWhisper` sau và **so bằng số**. Không cam kết vào một model trước khi đo.

**Không dùng Gemini/OpenAI audio API:**
- 130,7 giờ audio là chi phí token lớn cho thứ chạy local miễn phí.
- Local giữ được degraded mode (CLAUDE.md §4: *mất mạng ⇒ semantic search và DANTE vẫn đủ*) — transcript đã nằm sẵn trong ES.

**Trích audio:** `faster-whisper` decode audio thẳng từ `.mp4` qua PyAV ⇒ có thể **không cần bước extract riêng**. ⚠️ **[Cần verify]** — nếu đúng thì chỉ thêm 1 dependency thay vì 2.

> CLAUDE.md §5 bắt buộc: **fetch docs `faster-whisper`, xác nhận API, cập nhật `references.md` §5 trong cùng change.** Hiện `references.md:739` mới có tên thư viện, chưa có snippet đã kiểm chứng như các thư viện khác.

---

## 5. Ba cạm bẫy chất lượng — quan trọng hơn chọn model

### a) Whisper ảo giác trên đoạn không có lời

Bản tin có nhạc hiệu, B-roll im lặng, tiếng ồn hiện trường. Whisper hay bịa câu lặp ở những đoạn này. **Text bịa vào ES là ô nhiễm trực tiếp kênh BM25** — hỏng âm thầm đúng kiểu CLAUDE.md §3 cảnh báo.

**Bắt buộc:** bật VAD filter + loại segment theo ngưỡng `no_speech_prob` / `avg_logprob`. Hai ngưỡng vào `configs/ingest/asr.yaml`, không hard-code.

### b) Lệch pha lời nói ↔ hình ảnh

Phát thanh viên nói *"Thủ tướng thăm nhà máy X"* khi hình vẫn là studio; B-roll nhà máy đến **sau** vài giây. Expand segment → keyframe bằng đúng `[start_time, end_time]` sẽ liên tục trả về mặt phát thanh viên thay vì cảnh cần tìm.

**Cần cửa sổ mở rộng bất đối xứng, lệch về phía sau.** Độ lệch cụ thể: ⚠️ **[Cần đo]** — vào config, tune trên bộ eval.

### c) Bộ eval hiện tại sẽ đánh giá THẤP kênh ASR một cách hệ thống

**Đây là rủi ro phương pháp luận lớn nhất.** Nhãn eval sinh từ **caption của keyframe** (PRD §6.1) — mô tả *thị giác*. Truy vấn sinh ra sẽ mô tả thứ **nhìn thấy được**, trong khi giá trị riêng của ASR nằm ở thứ **chỉ nghe thấy** (tên riêng, số liệu).

⇒ Chạy bộ eval hiện tại rồi kết luận *"ASR không cải thiện"* là **kết luận sai do thước đo lệch**, không phải do ASR kém.

Muốn đo đúng cần một tập truy vấn con sinh từ transcript. Nhưng khi đó phải giữ kỷ luật CLAUDE.md §8: *model sinh nhãn ≠ model bị đánh giá* — và cảnh giác rò rỉ, vì truy vấn sinh từ transcript rồi đi tìm bằng chính transcript là **tự chấm chính mình**.

---

## 6. Điểm cắm vào code

| Việc | Vị trí |
|---|---|
| Nguồn audio | `VideoRef.source_path` (`datasets/base.py:9`, gán ở `datasets/aic2026.py:78`) — **không cần sửa adapter**, chính là file `.mp4` shot detect đang dùng |
| Orchestrator mới | `scripts/run_asr.py` — sao khung từ `scripts/run_ingest.py:126` |
| Pipeline mới | `pipelines/asr.py` — theo cấu trúc `pipelines/ocr.py`: Protocol client + dataclass config mirror yaml + `run_asr()` trả summary |
| Config mới | `configs/ingest/asr.yaml` — đúng convention `configs/ingest/ocr.yaml` |
| Store | `EsStore` nhận `index_name` qua constructor (`es_client.py:32-34`) ⇒ **dùng lại được**, chỉ thêm mapping + dataclass segment. Giữ nguyên `vi_analyzer` để OCR và ASR cùng chuẩn hoá tiếng Việt |
| Trạng thái | `mongo.upsert_video()` (`mongo_client.py:100-104`) — nơi duy nhất ghi `ingest_status` |

### 6.1 Chạy pass riêng, KHÔNG nhồi vào `ingest_video()`

**Lý do VRAM.** `ingest_video()` (`ingest.py:236`) đã giữ TransNetV2 + BEiT-3. Thêm Whisper large vào cùng vòng lặp per-video nghĩa là load/unload model **873 lần**. CLAUDE.md §4 *"chạy tuần tự an toàn hơn song song"* nên hiểu là **tuần tự theo giai đoạn**: quét hết 873 video bằng Whisper (load model 1 lần), rồi mới sang việc khác.

### 6.2 Vấn đề trạng thái — phải sửa trước

`ingest_status` hiện là **chuỗi tuyến tính** (`ingest.py:40-45`, `_SKIP_STATUSES` ở `:57`). OCR và ASR là **hai nhánh phụ độc lập**: một video có thể xong ASR mà chưa xong OCR và ngược lại. Nhét `STATUS_ASR_DONE` vào chuỗi đó sẽ **đá nhau với `STATUS_OCR_DONE`**.

**Đề xuất:** tách field trên `VideoDocument` — `ingest_status` giữ nguyên cho nhánh vision, thêm `ocr_status` và `asr_status`.

> Vấn đề này **đã tồn tại sẵn với OCR**, chỉ chưa lộ vì OCR chưa nối. Sửa một lần cho cả hai.

---

## 7. Mâu thuẫn docs phải xử lý trước khi code

| # | Mâu thuẫn | Vị trí | Hành động |
|---|---|---|---|
| 1 | ASR được ghi là **non-goal v1** rõ ràng | `PRD.md:133` | Làm ASR bây giờ là **đảo quyết định** — phải sửa PRD trong cùng change |
| 2 | Q3 *"Có kênh audio/phụ đề không?"* còn là **câu hỏi mở** | `PRD.md:682` | Quyết định này chính là câu trả lời: **có audio, KHÔNG có phụ đề sẵn** ⇒ đóng Q3 |
| 3 | `architecture.md` **không có một chữ nào** về audio/ASR — sơ đồ §3 chỉ 3 nhánh, §6 cây thư mục không có `asr.py` | `architecture.md` | Thêm nhánh thứ 4 + entry §10 đánh dấu 🔧 **[Giả định]** (không phải 📄 [Paper] — baseline arXiv:2512.13169 không có ASR) |
| 4 | Ô ASR còn ghi *"AIC 2025"* | `references.md:739` | Sót từ đợt đổi tên, sửa luôn |

---

## 8. Thứ tự thực hiện đề xuất (P0.8)

1. **Fetch docs `faster-whisper`**, xác nhận API + model id, cập nhật `references.md` §5 — timebox **nửa ngày**.
2. **Sửa docs trước khi code** (PRD §2.3, Q3, architecture §3/§6/§10, references §8) — để code không lệch docs.
3. **Tách `ocr_status` / `asr_status`** khỏi `ingest_status` — vá luôn cho OCR.
4. **TDD `pipelines/asr.py`** với fake `AsrClient`, không cần GPU:
   - test VAD filter loại đúng đoạn im lặng
   - test loại segment ảo giác theo ngưỡng
   - test quy đổi `time → frame_idx` dùng `fps` từ `read_video_info` — **không làm tròn ẩu** (CLAUDE.md §3: trượt cửa sổ TRAKE)
5. **Chạy thử 1 video**, đọc transcript bằng mắt, chỉnh ngưỡng.
6. **Pilot ~10 video**, rồi mới full 873. Chuẩn bị script/lệnh để **chủ dự án tự chạy** (CLAUDE.md §10 — job nặng).
7. **Hoãn quyết định fusion RRF** cho tới khi `eval/` tồn tại.

### ⚠️ Cảnh báo phạm vi

CLAUDE.md §8: *không sang phase sau khi phase trước chưa có số đo*. Hiện **chưa có `eval/`** ⇒ mọi tuyên bố *"ASR cải thiện retrieval"* **chưa kiểm chứng được**.

- Xây nhánh **ingest** ASR bây giờ: **hợp lý** (job compute dài, chạy nền, không phụ thuộc Gemini key).
- Bật vào **retrieval** và tuyên bố cải thiện: **phải đợi eval** — và đợi cả tập truy vấn ASR-aware ở §5c.

---

## 9. Câu hỏi cần chủ dự án chốt

| # | Câu hỏi | Trạng thái |
|---|---|---|
| A | Đồng ý đảo non-goal `PRD.md:133` để đưa ASR vào v1? | ⬜ chưa chốt |
| B | `large-v3` làm baseline trước, PhoWhisper là bước nâng cấp có đo — hay PhoWhisper ngay từ đầu? | ⬜ chưa chốt |
| C | ASR là **P0.8** (trước eval) hay đẩy sau khi `eval/` xong? | ⬜ chưa chốt |
| D | Có làm tập truy vấn eval ASR-aware (§5c) không, hay chấp nhận đo lệch? | ⬜ chưa chốt |

---

## 10. Ghi chú về độ tin cậy của file này

- Mọi khẳng định về **code hiện có** đều kèm `file:line`, đọc trực tiếp từ working tree ngày 2026-08-02.
- Mọi khẳng định về **thư viện ASR** (VRAM, tốc độ, API `faster-whisper`, PhoWhisper CTranslate2) là **CHƯA fetch docs kiểm chứng** — đánh dấu ⚠️ [Cần đo] / [Cần verify]. Phải làm bước §8.1 trước khi tin.
- **Chưa có số đo thực tế nào** — không có benchmark, không có transcript mẫu. Toàn bộ §4–§5 là thiết kế trên giấy.

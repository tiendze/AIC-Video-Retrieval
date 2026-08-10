# P0 — Nền móng

> **Quay lại:** [`roadmap.plan.md`](./roadmap.plan.md) · **Tiếp theo:** [`p1-search.plan.md`](./p1-search.plan.md)
> **Trạng thái:** 🟡 đang chạy — **P0.1 → P0.6 xong**; **VISION + ASR đã ingest xong toàn kho 873 video** (qua đường Kaggle-offline-embed + P0.11a); **OCR *live* (`pipelines/ocr.py`, gọi Gemini) hoãn có chủ đích** (chủ dự án 2026-08-09, quota không còn là lý do — quyết định sản phẩm); ⭐ **2026-08-10: `ocr_keyframes` đã nạp đầy toàn kho 873 video qua đường import offline riêng** (`data/ocr_results.jsonl` — OCR đã chạy từ trước, ngoài `pipelines/ocr.py` — xem `pipelines/import_ocr_results.py`), quyết định hoãn live-OCR ở trên **vẫn giữ nguyên**, nhưng kênh `ocr` giờ có dữ liệu thật để P0.8 dùng; **P0.4′ đã `freeze` `aic_eval_v1` v1 — NHƯNG là bộ nhãn RÚT GỌN, review tay bị BỎ QUA có chủ đích (xem cảnh báo dưới)**

| Bước | Trạng thái |
|---|---|
| P0.1 scaffold | ✅ xong |
| P0.2 docker-compose | ✅ xong (host port đã trả về mặc định `19530` / `9200` / `9000` khi chuyển từ server về laptop) |
| P0.3 `DatasetAdapter` + `AIC2026Adapter` | ✅ xong — đã đổi tên từ `AIC2025Adapter` và sửa theo layout mới |
| P0.4 downloader | ❌ đã bỏ (2026-07-29) |
| P0.4′ sinh bộ eval | 🟡 **`aic_eval_v1.json` đã freeze 2026-08-10 nhưng RÚT GỌN — 173/~800 shot dự kiến, review tay BỊ BỎ QUA** (chủ dự án 2026-08-10, hết quota Gemini + thời gian cấp bách). Xem chi tiết ngay dưới bảng. |
| P0.5 `BEiT3Encoder` | ✅ xong — checkpoint thật đã tải + verify 2026-08-01: `embedding_dim=768`, 6/6 test pass, batch=8 ảnh ~1,1GB VRAM đỉnh trên RTX 3070 |
| P0.6 ba store client | ✅ xong — đã bổ sung field `frame_idx` vào Milvus (2026-08-01, xem dưới) |
| **P0.7 ingest** | 🟡 **VISION xong toàn kho (221.903 keyframe, 873 video `complete`) · TEXT/OCR *live* (`pipelines/ocr.py`) hoãn có chủ đích (2026-08-09) · ⭐ nhưng `ocr_keyframes` đã nạp đầy 2026-08-10 qua import offline riêng (`data/ocr_results.jsonl`, ngoài `pipelines/ocr.py`) — xem `pipelines/import_ocr_results.py`** |
| P0.8 search + `/health` | ⬜ |
| P0.9 `eval/run.py` | 🟡 code xong 2026-08-10, chạy thật trên dữ liệu chưa (xem ghi chú dưới) |
| P0.10 báo cáo phase | ⬜ |
| P0.11a ASR → ES (BM25) | ✅ xong 2026-08-09 — toàn kho 873 video, 31.719 segment, verify khớp Mongo/ES |
| P0.11b ASR → Milvus (dense) | ⬜ **optional** — chủ dự án 2026-08-07 (và tái xác nhận 2026-08-09): không cần cho lúc này |

**Exit criteria (PRD §4):** search một truy vấn bất kỳ ra kết quả từ đầu đến cuối; `eval/run.py` chạy được và in ra Recall@k **dù số còn thấp**.

**Mục tiêu thật của phase này không phải "làm cho nó tốt" mà là "làm cho nó đo được".** Con số đầu tiên dù xấu vẫn có giá trị hơn một hệ thống chưa từng được đo — đây chính là khoảng trống của paper baseline (PRD §1.3).

---

## Đường găng

```
P0.1 scaffold ──► P0.3 adapter ──┬──► P0.5 encoder BEiT-3 (timebox 1 ngày)
P0.2 docker-compose ──► P0.6 stores ──┴──► P0.7 ingest ──► P0.4′ sinh nhãn ──► P0.8 search ──► P0.9 eval ──► P0.10 báo cáo
                                                │
                                                └──► P0.11a ASR→ES ──► P0.11b ASR→Milvus     (ngoài đường găng, P0.11b optional)
```

> **P0.11 không nằm trên đường găng.** Nó chỉ cần `.mp4` (đọc `fps`/`frame_count` thẳng bằng cv2, **không đọc Mongo**), nên không phụ thuộc P0.7 đã xong hay chưa — chỉ vướng VRAM, không vướng dữ liệu. Đường găng P0 vẫn kết thúc ở P0.10.
>
> **Cập nhật 2026-08-07 — P0.11b optional.** Chủ dự án: bước `bge-m3` encode (dense semantic cho ASR) tốn thời gian, không cấp thiết bằng phần còn lại. P0.11a (transcribe → Elasticsearch/BM25) vẫn giữ nguyên kế hoạch; P0.11b chuyển thành **làm sau nếu còn dư thời gian**, không phải điều kiện để coi P0.11 là "xong". Không cần code nào thay đổi để làm việc này khả thi — `scripts/embed_transcripts.py` vốn đã là lệnh CLI tách rời (§P0.11b bên dưới), không nằm trong `run_ingest.py` hay bất kỳ đường chạy bắt buộc nào.

> **Cập nhật 2026-07-29 — đổi dataset.** Dữ liệu công khai đã tải xong về `data/` (873 video, 130,7 giờ). Task **P0.4 (downloader `yt-dlp`) bị bỏ** — video đã là file `.mp4` cục bộ. Thay vào đó là **P0.4′ — sinh bộ eval**, vì dataset không kèm nhãn nào.
>
> **Cập nhật 2026-08-01 — ba việc, đều đã làm xong:**
>
> 1. **Đề vòng sơ tuyển AIC 2026 đã công bố** (`Thong tin vong So tuyen AIC2026.pdf`). Ba thay đổi chạm vào P0: **bỏ Video KIS**; **bỏ phạt theo thời gian**, thay bằng nộp 100 đáp án xếp hạng chấm `mean(R@k)`, k ∈ {1,5,20,50,100}; **đáp án là `frame_id` tuyệt đối**, cửa sổ TRAKE < 10 frame. Chi tiết: `references.md` §1.2.
> 2. **Dataset đổi tên sang AIC 2026 batch 1** (nội dung không đổi) và **layout đã khác**: `data/Videos_<batch>/video/*.mp4` phẳng, `data/media-info-aic25-b1/media-info/`, `data/map-keyframes-aic25-b1/map-keyframes/`. Bản tải hiện tại **873 file / 873 id** — không còn 88 file trùng id. `Keyframes_*` / `objects` / `clip-features-32` **cố ý không tải**, dự án tự trích.
> 3. **Chuyển từ server về laptop:** host port docker-compose trả về mặc định `19530` / `9200` / `9000`, và `mongo` bỏ hack `entrypoint: ["mongod"]` (hack đó chỉ cần cho rootless Docker trên NFS của server).
>
> ⚠️ **Batch 2 chưa phát hành.** Mọi con số của P0 là của batch 1 và sẽ phải đo lại.

**Ràng buộc thứ tự cứng mới:** `P0.4′` phải chạy **sau** `P0.7`, vì nhãn sinh từ shot do TransNetV2 cắt ra (PRD §6.1.1). Không còn việc dài nào chạy nền được — ba việc nặng (dựng BEiT-3, giải mã 27,9 giờ video, sinh + review nhãn) nối tiếp nhau. Cần cắt thì cắt **phạm vi pilot**, đừng cắt bước review nhãn.

**Phạm vi P0:** pilot **`L21`–`L24` = 128 video, 27,9 giờ** (mốc tham chiếu: 26.003 keyframe do BTC trích sẵn; số keyframe của mình ⚠️ chưa đo).

---

## P0.1 — Scaffold dự án

**Làm gì:** dựng bộ khung theo đúng cây thư mục `architecture.md` §6.

- `pyproject.toml` — **pin version** mọi dependency (PRD §7.5). Tooling: `black` · `isort` · `ruff` · `pytest` · `pytest-cov`.
- Cây thư mục: `app/{api,services,encoders,stores,schemas}` · `pipelines/` · `datasets/` · `eval/{metrics,baselines}` · `configs/{ingest,eval,model}` · `scripts/` · `tests/`.
- `app/config.py` — `pydantic-settings`, nạp từ `.env`, **validate ngay lúc khởi động**. Thiếu API key ⇒ fail rõ ràng lúc start, không fail giữa lúc ingest chạy được 3 tiếng.
- `.env.example` (commit) + `.env` (không commit). Log phải **che** key / token / authorization header.
- `app/main.py` — `create_app()`, model nạp qua **lifespan**.

**Verify:**
```powershell
ruff check .            # sạch
black --check .         # sạch
pytest                  # chạy được (kể cả khi mới có test placeholder)
```
Xoá `GEMINI_API_KEY` khỏi `.env` → app phải báo lỗi rõ ràng lúc khởi động.

**Ghi vào `README.md`** phần build/run command khi bước này xong (theo ghi chú cuối CLAUDE.md — **không** đưa vào CLAUDE.md).

---

## P0.2 — `docker-compose.yml` cho ba store

**Làm gì:** dựng hạ tầng lưu trữ trên một máy (architecture.md §9.1).

| Service | Ghi chú |
|---|---|
| **Milvus standalone** | Kèm `etcd` + `MinIO`. ~4 GB RAM |
| **Elasticsearch 9.3.0** | Phải **build image riêng** để cài plugin `elasticsearch-analysis-vietnamese` **9.3.0** (bảng tương thích: plugin 9.3.0 ↔ ES 9.0.0–9.3.0, `references.md` §4.3). `ES_JAVA_OPTS=-Xms2g -Xmx2g` — giới hạn cứng, R3 |
| **MongoDB** | ~1 GB RAM |

Encoder + FastAPI **chạy native**, không trong container — truyền GPU vào container trên Windows là phức tạp không cần thiết.

**Verify:**
```powershell
docker compose up -d
# ES đã có plugin tiếng Việt chưa — kỳ vọng tách theo TỪ GHÉP:
# ["cộng hòa","xã hội","chủ nghĩa","việt nam"]
```
Gọi `POST localhost:9200/_analyze` với `{"analyzer":"vi_analyzer","text":"Cộng hòa Xã hội chủ nghĩa Việt Nam"}`.

> **Lịch sử port (2026-08-01):** khi chạy trên server dùng chung `gpu156`, host port bị đổi sang `19540` / `9201` / `9004` vì `19530` / `9200` / `9000` đã bị stack khác chiếm, và `mongo` phải bỏ qua `docker-entrypoint.sh`. Nay dự án chạy trên laptop cá nhân nên **đã trả về mặc định** `19530` / `9200` / `9000` và `mongo` dùng entrypoint chuẩn. Nếu quay lại server thì remap lại, đừng sửa mặc định trong repo.

**Còn phụ thuộc:** `/health` (P0.8) mới kiểm được cả ba store cùng lúc.

---

## P0.3 — `DatasetAdapter` + `AIC2026Adapter` ✅ **xong**

**Làm gì:** hàng rào chống rủi ro T3 — đổi/mở rộng dataset chỉ thêm class, không sửa pipeline.

- `datasets/base.py`: Protocol `DatasetAdapter` + value object `VideoRef`, `GroundTruthSegment` (`@dataclass(frozen=True)`, chữ ký chính xác ở `architecture.md` §5.2). **Chữ ký giữ nguyên** — đây là bằng chứng T3 đúng, và nó đã được nghiệm thu thật: đổi dataset 2025 → 2026 (đổi cả layout thư mục) **không sửa một dòng nào** của `base.py`.
- `datasets/aic2026.py` — thay `datasets/aic2025.py`. Hai phương thức lấy dữ liệu từ **hai nguồn khác nhau**:

| Phương thức | Nguồn |
|---|---|
| `iter_videos(split)` | Glob `Video*/video/*.mp4` ở **mọi độ sâu** dưới `data/` + đọc `media-info/<video_id>.json`. `split` ∈ `"pilot"` (`L21`–`L24`) \| `"all"` |
| `load_ground_truth(split)` | Đọc bộ nhãn **tự sinh** `configs/eval/aic_eval_v1.json` (sinh ở P0.4′) |
| `load_reference_keyframes(video_id)` | Đọc `map-keyframes/<video_id>.csv` — **nguồn đối chứng độc lập duy nhất** cho chất lượng cắt shot (R10), dùng ở P0.7 |

**Năm cái bẫy phải xử lý trong adapter:**

1. **Trùng `video_id`** — bản tải hiện tại 873 file / 873 id, nhưng bản Kaggle trước có 961 file / 873 id (`Videos_L25_a1`, `Videos_L25_b` lặp id). Khử trùng **vẫn giữ**: bỏ đi thì một lần tải lại khác layout là mẫu số metric bị thổi phồng mà không có gì báo động.
2. **Tên thư mục không nhất quán** và **độ sâu thay đổi giữa các bản phát hành** (`Videos/Videos/Videos_L21_a/…` trước, `Videos_L21_a/…` bây giờ; `Video_L30_a` số ít). Phải **glob**, không dựng đường dẫn bằng chuỗi định dạng.
3. **`duration_sec` lấy từ `media-info.length`**, đơn vị **giây** (int).
4. **`publish_date` là `DD/MM/YYYY`**, không phải ISO-8601 — parse tường minh, không để thư viện đoán.
5. ⭐ **Metadata có thể thiếu.** Đề AIC 2026 ghi rõ *"một số video có thể không có file metadata tương ứng"*. Khi thiếu: `duration_sec` suy từ dòng cuối `map-keyframes` (`frame_idx / fps` — là **cận dưới**, chỉ dùng để lập kế hoạch, **không** dùng làm mốc nộp bài), `category` để `None`. Thiếu **cả hai** nguồn ⇒ raise, không đoán.

- `category` của `VideoRef` ánh xạ sang **`author`** (tên kênh), thay cho `recipe_type` của YouCook2 (PRD §6.1.4).
- `load_ground_truth` **fail rõ ràng** nếu `aic_eval_v1.json` chưa tồn tại — không trả list rỗng, vì list rỗng sẽ khiến `eval/run.py` in ra bảng metric toàn số 0 trông như "hệ thống hỏng" thay vì "chưa sinh nhãn".

**Verify:** test trên fixture rút gọn `tests/fixtures/aic2026_sample/` (3 video, có **cố ý** một id trùng, một thư mục tên `Video_*` số ít, và một video **không có** metadata) + `tests/fixtures/aic2026_no_metadata/`:
- `test_iter_videos_deduplicates_repeated_video_id` — 4 file → 3 `VideoRef`
- `test_iter_videos_handles_singular_video_directory_name`
- `test_iter_videos_finds_shards_directly_under_dataset_root` — layout phẳng của bản 2026
- `test_pilot_split_contains_only_l21_to_l24`
- `test_duration_sec_is_read_from_metadata_length_field`
- `test_duration_sec_falls_back_to_map_keyframes_when_metadata_missing`
- `test_iter_videos_raises_when_no_metadata_and_no_map_keyframes`
- `test_publish_date_parsed_as_day_first` · `test_publish_date_raises_when_metadata_missing`
- `test_load_reference_keyframes_casts_every_column`
- `test_load_ground_truth_raises_when_eval_set_missing`
- `test_video_ref_is_immutable`

**Xoá kèm:** `datasets/aic2025.py`, `tests/test_aic2025_adapter.py` (đã xoá 2026-08-01), và trước đó `datasets/youcook2.py` + fixture YouCook2.

---

## ~~P0.4 — Downloader video~~ → **BỎ** (2026-07-29)

Dataset AIC 2025 phát hành **file `.mp4` trực tiếp**, không phải URL YouTube. Không còn khâu tải, R5 đóng. **Xoá `scripts/download_videos.py` + `tests/test_download_videos.py`.**

Vẫn giữ một ý tưởng từ task cũ: **đóng băng danh sách video thực sự ingest được** thành `configs/eval/pilot_available_v1.txt` + hash, làm mẫu số chính thức cho mọi báo cáo metric. Nay danh sách này sinh ra ở cuối **P0.7** (video nào ingest `complete`) chứ không phải ở khâu tải.

---

## P0.4′ — Sinh bộ eval ⭐ **task mới, chạy SAU P0.7**

> 🟡 **2026-08-09: code xong (`eval/build_eval_set.py` + `tests/test_build_eval_set.py`, 33 test xanh, coverage 81%).** Chi tiết quyết định + kiểm chứng: `.claude/plans/p0-4-build-eval-set.plan.md`.
>
> ⚠️ **2026-08-10 — `generate` chạy thật nhưng dừng giữa chừng vì hết quota Gemini: 173/~800 task hoàn thành** (144 KIS + 29 TRAKE, 267 shot `failed`, 61 video khác nhau — xem `configs/eval/aic_eval_v1.candidates.json`). Vì thời gian cấp bách, chủ dự án quyết định (2026-08-10) **BỎ QUA bước review tay** thay vì chờ review ≥ 100 mẫu như kế hoạch gốc: đã set `approved=true` cho toàn bộ 173 candidate không xem lại, rồi `freeze` thẳng → `configs/eval/aic_eval_v1.json` (sha256 `008f368e…`), `.sha256`, `.quality_report.md`.
>
> **Hệ quả cần nhớ khi đọc số liệu P0.9 trở đi:**
> - Cổng chất lượng PRD §6.1.3 (loại > 30% khi review) **không có ý nghĩa ở v1** — `rejection_rate` báo cáo là 0.0% vì không ai từ chối gì, không phải vì caption thật sự tốt 100%. **Không dùng con số này để kết luận chất lượng nhãn.**
> - Bộ nhãn v1 **nhỏ hơn nhiều** so với mục tiêu ban đầu (~600 KIS + 40×5 TRAKE): chỉ 173/800 task, tất cả rơi vào 61/873 video. Metric P0.9 tính trên v1 sẽ có **phương sai cao** và không đại diện tốt cho toàn kho.
> - Phép thử rò rỉ nhãn `caption` ↔ `ocr_text` vẫn **chưa chạy được** (OCR hoãn) — không đổi so với trước.
> - **Kế hoạch nối tiếp:** khi quota Gemini hồi phục, chạy lại đúng lệnh cũ (`python -m eval.build_eval_set generate`) — script tự resume, không trả tiền lại cho 173 record đã có (xem docstring `eval/build_eval_set.py:20-27`). Sau khi có đủ mẫu, **review tay thật** rồi `freeze` lại thành `v2`, và mọi báo cáo metric dùng `v1` phải được chạy lại với `v2` trước khi coi là số liệu chính thức.

**Làm gì:** dataset **không kèm nhãn nào**. Không có nhãn thì `eval/run.py` ở P0.9 không có gì để chấm, và cả dự án mất khả năng chứng minh mọi tuyên bố "cải thiện".

- `eval/build_eval_set.py` — thực hiện giao thức **PRD §6.1.2**, 5 bước: lấy mẫu shot (phân tầng theo `batch` + `author`, seed cố định) → Gemini sinh caption tiếng Việt → lọc caption chung chung → **review tay ≥ 100 mẫu** → đóng băng `configs/eval/aic_eval_v1.json` + hash.
- Ngữ cảnh đưa vào prompt sinh caption: keyframe giữa shot + `media-info.title`. ~~`objects[].detection_class_entities`~~ **bỏ** — thư mục `objects/` cố ý không tải, pipeline tự chạy chưa xây (PRD §6.1.2 đã cập nhật 2026-08-09).
- Gemini: `client.interactions.create(...)`, **structured output** qua `model_json_schema()` — không parse regex. Model `gemini-3.6-flash`. ⚠️ Đã thử `gemini-2.5-flash` (rẻ hơn) 2026-08-09 rồi đảo lại cùng ngày — model đó **lờ hoàn toàn `response_format`**, verify bằng test trực tiếp trên ảnh thật (xem `references.md` §5.1).
- Quy mô v1: ~600 shot, **toàn kho 873 video** (đổi từ pilot `L21`–`L24` — cả kho đã ingest xong tính đến 2026-08-09), phân bổ tầng **căn bậc hai**. Thêm 40 chuỗi × 5 shot liên tiếp cho TRAKE, sinh cùng đợt.

**🛑 Ràng buộc bất biến — model sinh nhãn phải KHÁC model bị đánh giá.** Gemini sinh caption, BEiT-3/SigLIP đi tìm. Vi phạm điều này thì mọi con số ở P0.9 vô nghĩa, và đây đúng là loại lỗi *không crash, chỉ làm điểm số sai một cách khó truy vết*.

**Đầu ra bắt buộc kèm bộ nhãn** (PRD §6.1.3 — không đo cái thước thì không biết số đo đáng tin đến đâu):
- Tỉ lệ caption bị loại khi review tay
- ⚠️ Độ đồng thuận giữa 2 người review trên ≥ 30 mẫu chồng nhau — **`not_measured` ở v1** (chỉ 1 reviewer, chủ dự án 2026-08-09)
- Tỉ lệ caption "chung chung" — đo bằng proxy Jaccard trên tập từ nội dung (chưa có search engine thật ở P0 để tìm ngược)

**Verify:**
- `test_eval_set_sampling_is_deterministic_given_seed`
- `test_eval_set_is_stratified_across_batch_and_author`
- `test_eval_set_hash_changes_when_content_changes`
- `test_trake_sequences_are_contiguous_and_time_ordered`
- (+ 29 test khác trong `tests/test_build_eval_set.py`, xem file đó)

**Rủi ro cần theo dõi (R6, R9):** đây là mắt xích yếu nhất của cả P0. Nếu tỉ lệ caption bị loại khi review > 30%, **dừng lại báo cáo** — bộ nhãn kém thì mọi metric phía sau chỉ là nhiễu. Cổng này đã code cứng trong `eval/build_eval_set.py freeze` (từ chối ghi file nếu vượt ngưỡng).

**Known gap:** phép thử rò rỉ nhãn `caption` ↔ `ocr_text` (PRD §6.1.3/§6.1.2, dùng để phát hiện Gemini chép chữ trên màn hình) **chưa chạy được** — nhánh OCR đang hoãn (chủ dự án 2026-08-09), `ocr_keyframes` index rỗng. Đo bổ sung khi OCR chạy.

**Còn lại để coi P0.4′ "xong":** chạy `python -m eval.build_eval_set generate --limit-shots 5` để mắt thấy tai nghe caption thật (đã chạy 1 lần 2026-08-09, phát hiện `gemini-2.5-flash` không dùng được — đã sửa về `gemini-3.6-flash`, cần chạy lại `--limit-shots 5` để xác nhận), rồi chạy full, review tay ≥ 100 mẫu, `freeze`, và báo cáo phase theo CLAUDE.md §8.

---

## P0.5 — `MultimodalEncoder` + `BEiT3Encoder` ⏱ **timebox 1 ngày**

**Làm gì:** trái tim hệ thống — đưa ảnh và chữ về **cùng một không gian vector**.

- `app/encoders/base.py`: Protocol `MultimodalEncoder` (chữ ký chính xác ở `architecture.md` §5.1) — `embedding_dim`, `model_id`, `encode_image`, `encode_text`.
- `app/encoders/beit3.py`: theo quyết định của chủ dự án, **bám baseline trước**. Cần: clone `microsoft/unilm`, `torchscale`, `timm` **bản pin cũ**, tokenizer SentencePiece `beit3.spm`, checkpoint **`beit3_base_itc`** (bản `_itc` cho retrieval, `base` chứ không phải `large` — bám baseline; ⚠️ **2026-08-01: máy đổi sang RTX 3070 8GB** — `base` giờ không chỉ là lựa chọn bám baseline mà còn là giới hạn VRAM thật, `large` rủi ro OOM cao hơn nhiều so với trên 2080Ti 11GB).
- Inference: `torch.no_grad()` + autocast fp16, batch size configurable, giải phóng cache giữa các giai đoạn.
- **L2-normalize ngay trong encoder**, không để nơi gọi tự chuẩn hoá.

**🛑 Quy tắc timebox:** quá **1 ngày** chưa chạy được ⇒ dừng, viết `app/encoders/siglip.py`, và **ghi nhận rõ trong báo cáo phase đây là sai lệch so với baseline**. Không đốt tuần thứ hai vào R1.

**Verify (viết test TRƯỚC):**
- `test_encoder_output_is_l2_normalized` — norm mỗi hàng ≈ 1.0
- `test_encode_text_and_image_share_embedding_dim`
- `test_encoder_returns_float32_array_of_shape_n_by_dim`
- `test_same_input_gives_same_vector` — tính tất định

**Đầu ra bắt buộc:** ghi lại `embedding_dim` thực tế → **trả lời Q4 của PRD** và cập nhật `architecture.md` §4.1 nếu khác 768.

---

## P0.6 — Ba store client

**Làm gì:** `app/stores/` — tầng duy nhất được nói chuyện với DB.

**`milvus_client.py`** (dùng `MilvusClient` v2.4+, **không** ORM API cũ):
- Tên collection **phải chứa `model_id`** — đổi embedding model ⇒ collection MỚI. Đây là ràng buộc ở cấp thiết kế, không phụ thuộc kỷ luật người viết code.
- Schema đúng `architecture.md` §4.1: `keyframe_id` (PK), `embedding`, `video_id`, `global_index`, **`frame_idx`**, `timestamp`, `author`, `batch`.

> ⭐ **Bổ sung 2026-08-01 — field `frame_idx`.** Đề AIC 2026 nộp `<video_id>, <frame_id>` và cửa sổ chấm TRAKE < 10 frame, nên chỉ số frame tuyệt đối phải đi kèm **kết quả search**, không được tính lại bằng `timestamp × fps` lúc xuất bài (làm tròn sai một frame là trượt cửa sổ). MongoDB đã có sẵn `frame_idx` từ đầu; Milvus vừa được bổ sung. **Phải sửa trước P0.7** — sửa sau nghĩa là ingest lại từ đầu.
- Index HNSW, `metric_type="COSINE"`, `M=16`, `efConstruction=200`; search `params={"ef": 64}`.
- **Filter đặt trong tham số `filter` của `search()`**, không lọc sau khi lấy kết quả về.

**`mongo_client.py`**: unique index trên `keyframe_id` và `video_id` — đây chính là cơ chế bảo đảm idempotent (T1). Ingest dùng **bulk write**. Từ `async def` phải dùng client bất đồng bộ.

**`es_client.py`**: mapping với `vi_analyzer` + `asciifolding` `preserve_original: true`; ingest dùng **bulk helpers**, không index từng document.

**Verify:**
- `test_collection_name_changes_when_model_id_changes`
- `test_duplicate_keyframe_id_does_not_create_second_record`
- `test_search_filter_is_passed_to_engine_not_applied_after`

---

## P0.7 — Ingest pipeline (VISION + **TEXT/OCR** + METADATA)

> ✅ **Khớp PRD §F1, không còn lệch.** Chủ dự án quyết ngày 2026-07-29 kéo nhánh TEXT (OCR) từ P1 về P0 — lý do đầy đủ ở `roadmap.plan.md` §8. Đây là **cả ba nhánh** của sơ đồ ingest (`architecture.md` §3.2).

**Làm gì:** biến video thô thành dữ liệu tìm kiếm được.

| Module | Nội dung |
|---|---|
| `pipelines/shot_detect.py` | TransNetV2 bản **`inference-pytorch/`**. Input bắt buộc `torch.uint8` shape `[B, T, 27, 48, 3]` ⇒ resize frame về **48×27** trước khi đưa vào (model tự assert). Không train lại |
| `pipelines/keyframe.py` | 4 keyframe mỗi shot theo **đúng công thức PRD §F1**: `k_extract = { K_(a + ⌊i × (b − a) / 3⌋), ∀i ∈ {0,1,2,3} }` |
| `pipelines/ingest.py` | Điều phối + checkpoint. Gán `global_index` **liên tục**, ghi `s_v` / `e_v` / **`frame_idx`** / `fps` **cùng lúc** với việc ghi keyframe |
| `pipelines/embed.py` | Encode theo lô + ghi Milvus (kèm `frame_idx`) |
| **`pipelines/ocr.py`** | **Mới — kéo từ P1.2 về.** Xem khối dưới |

### Nhánh TEXT — `pipelines/ocr.py`

**Thứ tự các bước là điểm mấu chốt về chi phí:**

```
keyframe → [1] lọc frame CÓ CHỮ → [2] dedupe perceptual hash → [3] Gemini OCR → [4] index ES
```

- **Bước [1] và [2] chạy TRƯỚC khi gọi Gemini** (R2). Keyframe 1280×720 ⇒ 2 tile ⇒ **516 token/ảnh**.
- ⚠️ **Thứ tự quan trọng của hai bước lọc đã đảo so với YouCook2.** Video tin tức có chữ ở **gần như mọi frame** nên bước [1] cắt được rất ít; **bước [2] dedupe mới là chỗ tiết kiệm chính** — bản tin giữ nguyên khung đồ hoạ, logo, thanh chạy chữ suốt nhiều giây liền. Đo và báo cáo tỉ lệ cắt của **từng** bước riêng biệt, đừng gộp.
- **Đòn bẩy rẻ nhất: hạ ảnh về ≤ 384×384** ⇒ còn 258 token/ảnh, cắt đôi chi phí. Phải kiểm chứng trước rằng **chữ nhỏ vẫn đọc được** — tên riêng và địa danh trong bản tin thường ở cỡ chữ nhỏ dưới đáy khung hình.
- **Ngân sách pilot `L21`–`L24`: ~13,4 triệu token** (hoặc ~6,7 triệu nếu hạ 384×384). **Đặt trần chi phí trong `configs/`** — vượt trần thì dừng và báo, đừng đốt tiếp.
- Gemini: `client.interactions.create(...)`, **structured output** qua `model_json_schema()` — không parse regex. Timeout + retry exponential backoff. **Lỗi OCR không được giết cả pipeline** — nhánh VISION vẫn phải hoàn tất.
- Index vào ES bằng **bulk helpers**, mapping `vi_analyzer` + `asciifolding` `preserve_original` (P0.6).
- `ingest_status` đi qua `ocr_done` trước khi tới `complete`.

**Checkpoint:** `ingest_status` ∈ `pending` / `decoding` / `embedded` / `ocr_done` / `complete` / `failed`. Khởi động bỏ qua mọi video `complete`.

**Xử lý lỗi:** video lỗi → error list riêng + `ingest_status = "failed"`, **pipeline chạy tiếp**. Cấm `except Exception: pass` — bắt exception cụ thể, log kèm `video_id`.

**Trên 8 GB VRAM: chạy tuần tự từng nhánh an toàn hơn song song.**

### ⭐ Ràng buộc mới của P0.7 từ đề AIC 2026 (2026-08-01)

**`frame_idx` là dữ liệu hạng nhất, không phải trường phụ.** Đáp án nộp lên BTC là `<video_id>, <frame_id>` với `frame_id` là **chỉ số frame tuyệt đối**. Vì vậy ingest phải:

1. **Ghi `frame_idx` thật của mỗi keyframe** — lấy từ chính vòng giải mã (số thứ tự frame đã đọc), **không** suy ra từ `timestamp × fps` về sau. `timestamp` là đại lượng *dẫn xuất* (`frame_idx / fps`), không phải ngược lại.
2. **Ghi `fps` của video vào MongoDB** (`VideoDocument.fps` đã có sẵn) — đây là hằng số quy đổi duy nhất, và `eval/` lẫn bộ xuất bài phải dùng **cùng một** giá trị đó.
3. **Ghi `frame_idx` vào cả Milvus lẫn MongoDB.** Milvus để bộ xuất bài không phải round-trip; MongoDB để hiển thị và kiểm tra chéo.

⚠️ **Vì sao đây là loại lỗi nguy hiểm nhất trong nhóm "không crash":** cửa sổ chấm của TRAKE thường **dưới 10 frame** (< 0,4 giây ở 25 fps). Lệch một frame do làm tròn không làm hỏng bất kỳ test nào, không sinh log, không làm Recall xấu đi — nó chỉ làm điểm thi bằng 0 ở đúng những truy vấn khó nhất.

**Verify:**
- `test_keyframe_indices_match_paper_formula` — so với giá trị tính tay cho vài khoảng `[a,b]`
- `test_global_index_is_contiguous_across_videos`
- `test_s_v_and_e_v_match_min_max_global_index`
- `test_duplicate_video_id_is_ingested_only_once` (xem P0.3 — bản tải hiện tại không còn file trùng, nhưng ràng buộc phải giữ)
- ⭐ `test_frame_idx_comes_from_decoder_not_from_timestamp_times_fps` — dựng video có `fps` không nguyên (ví dụ 29,97) và chứng minh `frame_idx` khớp chính xác, trong khi `round(timestamp × fps)` lệch
- ⭐ `test_timestamp_is_derived_from_frame_idx_and_fps` — một chiều duy nhất, không có đường ngược
- Thực nghiệm: ingest 5 video → kill giữa chừng → chạy lại ⇒ không bản ghi trùng, video `complete` bị bỏ qua

**Ba đầu ra bắt buộc của bước này** (đều là dữ kiện cả P0 đang chờ):

1. **Trả lời Q6 — số keyframe thật.** TransNetV2 + công thức §F1 sinh ra bao nhiêu keyframe trên pilot? Ngoại suy cho 873 video. Mốc đối chiếu: BTC trích sẵn 26.003 keyframe cho pilot, 177.321 cho toàn kho. **Lệch nhiều so với mốc này là tín hiệu cần điều tra**, không phải chuyện đương nhiên — video tin tức cắt cảnh rất dày. Đo xong thì cập nhật lại PRD §6.6 (ngân sách R2, đĩa, thời gian ingest).
2. **Đo `tIoU` với `map-keyframes` của BTC (R10).** Nhãn eval sinh từ chính shot của mình, nên metric ở P0.9 **không phạt được lỗi cắt shot**. Đây là nguồn đối chứng độc lập duy nhất — và từ 2026-08-01 nó là thành phần BTC **duy nhất** còn tải về (`Keyframes_*` / `objects` / `clip-features-32` đã cố ý bỏ). Đọc qua `AIC2026Adapter.load_reference_keyframes()`. Báo cáo kèm **phân bố độ dài shot** (trung vị + p10/p90) — shot quá ngắn làm Recall segment-level đẹp một cách giả tạo.
3. **`configs/eval/pilot_available_v1.txt` + hash** — danh sách video ingest `complete`, là mẫu số chính thức của mọi báo cáo metric.
4. **Số token OCR thực tế đã tiêu**, tách theo bước lọc: bao nhiêu keyframe vào [1], còn lại sau [1], còn lại sau [2], bao nhiêu thực sự gọi Gemini. Đây là **bằng chứng R2 đã được giảm thiểu** — không có bảng này thì không biết dedupe có tác dụng hay không.

**Verify thêm cho nhánh TEXT:**
- `test_ocr_skips_frames_without_text`
- `test_ocr_deduplicates_near_identical_frames_before_api_call` — mock Gemini, đếm số lần gọi
- `test_ocr_failure_does_not_abort_ingest` — nhánh VISION vẫn `complete`
- `test_ocr_stops_when_token_budget_exceeded`
- Smoke test lấy thẳng keyframe thật từ pilot (**không cần tự chuẩn bị bộ ảnh có chữ** — R4 đã đóng): kiểm `vi_analyzer` tách từ ghép đúng, và `asciifolding` khớp khi gõ **không dấu**.

---

## P0.8 — Semantic search + `/health`

**Làm gì:** đóng vòng end-to-end đầu tiên.

- `app/services/search_service.py` — mode `semantic` **và `ocr`** (nhánh TEXT đã có dữ liệu từ P0.7). **Ba mode là tham số**, không phải ba code path sao chép nhau. Mode `hybrid` để lại P1.3 — nó cần cả hai kênh đã có số đo riêng lẻ thì so sánh mới có nghĩa.
- `app/api/search.py` — `POST /api/search`. Router **mỏng**: validate + uỷ quyền. `response_model` bắt buộc. Encoder và store client inject qua `Depends`, **không** khởi tạo trong route.
- `app/api/health.py` — `GET /health` kiểm tra kết nối cả ba store.
- Lỗi trả theo schema thống nhất `{detail, error_code, request_id}`. Lỗi hiện ra UI phải **đọc được bởi người đang bị tính giờ** — không dump traceback.

**Verify:** gõ một `caption` bất kỳ từ `aic_eval_v1.json` → trả về grid keyframe kèm `score`, `video_id`, `timestamp`, `keyframe_id`, đường dẫn thumbnail.

---

## P0.9 — `eval/run.py` ⭐ phần quan trọng nhất của P0

> 🟡 **2026-08-10: code xong, TDD RED→GREEN từng module, 79 test mới xanh (ruff sạch).** Gồm `eval/metrics/retrieval.py` (Recall@k/MRR/mAP/nDCG@10/median-rank/AIC Final Score + luật hit point-in-segment, cả video-level lẫn segment-level), `eval/metrics/leakage.py` (phép thử rò rỉ nhãn caption↔ocr_text), `eval/baselines/random.py` + `bm25.py`, `configs/eval/aic_pilot.yaml`, và `eval/run.py` (orchestration). Hai method mới ở store: `MongoStore.scroll_all_keyframes()` (pool cho random baseline) + `EsStore.get_by_ids()` (tra OCR text theo `keyframe_id` cho phép thử rò rỉ nhãn).
>
> **Quyết định đã xác nhận với chủ dự án (2026-08-10):** `bm25.py` là wrapper mỏng quanh `EsStore.search()` sẵn có (Elasticsearch mặc định *là* BM25) — số của baseline `bm25` và kênh `ocr` sẽ TRÙNG nhau, đây là chủ đích (bm25 đóng vai sàn so sánh, không phải kênh thứ ba).
>
> **Chưa chạy thật** (CLAUDE.md §10 — cần BEiT-3 GPU + Milvus/ES/Mongo thật đang chạy, để chủ dự án tự chạy):
> ```powershell
> docker compose up -d
> python -m eval.run --config configs/eval/aic_pilot.yaml
> ```
> Sau khi chạy xong, đọc `reports/<timestamp>_<git_sha>/report.md` để lấy bảng số thật cho P0.10.

**Làm gì:** biến hệ thống từ "chạy được" thành "đo được".

- `eval/run.py` là **entry point duy nhất** của evaluation.
- **Quy tắc hit — point-in-segment** (PRD §6.2): keyframe `k` hit segment `g` ⟺ `g.start ≤ timestamp(k) ≤ g.end` **và** `video_id(k) == video_id(g)`.
- Báo cáo **cả hai mức**: video-level và segment-level. Chênh lệch giữa hai mức chỉ ra bệnh cụ thể — video-level cao mà segment-level thấp nghĩa là lỗi ở lấy mẫu keyframe, không phải ở embedding.
- `eval/metrics/retrieval.py`: **`AIC Final Score`** · Recall@**{1,5,20,50,100}** · MRR · mAP · nDCG@10 · Median Rank · Recall@20 tách theo **`author`** và **`batch`** (thay `recipe_type`, PRD §6.1.4).

⭐ **Bộ K đã đổi theo đề AIC 2026: `{1, 5, 20, 50, 100}`** — bỏ `K=10`, thêm `K=50`. Đây đúng là bộ K mà BTC chấm. Thêm `AIC Final Score = mean(Recall@k)` trên 5 mốc đó: **đây là con số duy nhất so sánh trực tiếp được với bảng điểm cuộc thi**, và cũng là con số dùng để so giữa các cấu hình từ P0 trở đi (PRD §6.3).
- `eval/baselines/`: `random.py` (sàn tuyệt đối) + `bm25.py`.

- **Báo cáo kênh `semantic` và kênh `ocr` tách riêng** — cả hai đã có dữ liệu từ P0.7. Đây là lợi ích chính của việc kéo OCR về P0: với nhãn tự sinh, một kênh đơn độc rất khó biết đúng hay sai; **chỗ hai kênh độc lập bất đồng với nhau là chỗ đáng điều tra**.

⚠️ **Với bộ nhãn tự sinh, `bm25.py` đổi vai từ phụ thành chính.** Caption do Gemini sinh **từ chính keyframe đáp án**; nếu OCR hoặc metadata chứa lại đúng từ ngữ đó thì bài toán "rò rỉ" và giải được bằng từ khoá thuần. **BM25 cao bất thường ⇒ bộ nhãn hỏng, không phải hệ thống giỏi** — quay lại sửa P0.4′ chứ đừng ăn mừng.

🔬 **Phép thử rò rỉ nhãn — làm được ngay ở P0 nhờ có kênh OCR.** Đo **độ chồng lấp từ vựng giữa `caption` và `ocr_text` của chính keyframe đáp án**. Chồng lấp cao nghĩa là Gemini đã chép chữ trên màn hình vào caption, và khi đó điểm số của kênh OCR chỉ là hệ thống tìm lại đúng thứ nó vừa tự viết ra. Đây là lý do quan trọng nhất khiến OCR đáng nằm ở P0 thay vì P1: **phát hiện bộ nhãn hỏng trước khi xây ba phase lên trên nó**. Vượt ngưỡng ⇒ sửa prompt sinh caption ở P0.4′ (cấm chép chữ hiển thị), sinh lại, tăng version bộ nhãn.

⚠️ **`mAP` và `nDCG` phải đọc dè dặt hơn `Recall@K`**: bộ nhãn chỉ có nhãn dương (biết keyframe nào đúng), không có nhãn âm (không biết keyframe nào chắc chắn sai) — PRD §1.4.

**Đầu ra:** `reports/<timestamp>_<git_sha>/` gồm `metrics.json` · `report.md` · `per_query.csv`, **bắt buộc** ghi kèm: git SHA · tên checkpoint · version index · **`eval_set_version` + hash bộ nhãn** · **số `video_id` duy nhất thực tế đánh giá được** (đọc từ `pilot_available_v1.txt`) · số truy vấn · seed.

**Verify:**
```powershell
python -m eval.run --config configs/eval/aic_pilot.yaml
```
Hệ thống phải **vượt xa** baseline random. Không vượt ⇒ **có bug**, không phải "cần tối ưu".

---

## P0.10 — Báo cáo phase & hiệu chỉnh mục tiêu

**Làm gì:**
1. Viết báo cáo theo template CLAUDE.md §8 (Đã làm gì · Input · Output · Cách verify · Còn thiếu).
2. **Hiệu chỉnh bảng PRD §6.7** — các mốc hiện tại là 🔧 giả định "sẽ hiệu chỉnh lại sau lần đo đầu tiên ở P0". Đây chính là lần đo đó.
3. Cập nhật `architecture.md` §4.1 nếu `embedding_dim` khác 768 (Q4).
4. Cập nhật trạng thái P0 trong `roadmap.plan.md`.

**🛑 Không sang P1 khi chưa có bảng số này.**

---

## P0.11 — Nhánh ASR (kênh truy hồi thứ tư)

> 🔧 **Quyết định chủ dự án 2026-08-02 — đảo non-goal `PRD.md:133`.** Baseline arXiv:2512.13169 **không** có ASR; đây là bổ sung của dự án, không phải của paper.
> **Ngoài đường găng P0.** Không chặn P0.8/P0.9/P0.10. Đánh số **P0.11** vì P0.8 đã thuộc về search + `/health`.

**Vì sao làm ngay bây giờ, không phải chen ngang:** OCR là **lời gọi API** (hết quota Gemini ⇒ đứng), ASR là **GPU thuần offline** (không cần key, không tốn quota). Hai việc chặn nhau ở **hai tài nguyên khác nhau** ⇒ chạy ASR lúc này không đánh đổi gì. Video đang ở `embedded`, mà `_TERMINAL_STATUSES` chỉ gồm `{ocr_done, complete}` ⇒ có quota là chạy lại `run_ingest.py`, vào thẳng `resume_ocr_for_video`, **không mất tiến độ**.

**Vì sao đáng làm:** dữ liệu là bản tin tiếng Việt **có lời dẫn** (`references.md:86`); lời dẫn nói ra nhiều thứ **không hiện thành chữ** — tên người, địa danh, số liệu ⇒ **bổ trợ** OCR chứ không trùng. `references.md:744` xếp ASR ưu tiên cao nhất của giai đoạn 2.

### Đã fetch kiểm chứng (CLAUDE.md §5) — cập nhật `references.md` §5 trong cùng change

- `faster-whisper` decode audio bằng **PyAV (bundle FFmpeg)** ⇒ **đọc thẳng `.mp4`, không cần bước extract audio riêng**.
- `transcribe(...) -> Tuple[Iterable[Segment], TranscriptionInfo]` — `Iterable` là **generator lazy**, không chạy gì tới khi iterate.
- `Segment` có `start, end, text, no_speech_prob, avg_logprob` — hai trường sau là thứ bộ lọc ảo giác cần.
- Benchmark chính thức **RTX 3070 Ti**: audio 13 phút → ~1 phút; VRAM 2926–6090MB ⇒ vừa 8GB.
- **PhoWhisper-large KHÔNG có bản CTranslate2** (đọc model card) và không phơi `no_speech_prob` ⇒ là bước nâng cấp **có đo** về sau, không phải baseline.

### P0.11a — Transcribe → Elasticsearch (BM25)

| Module | Nội dung |
|---|---|
| `pipelines/asr.py` | Soi gương `pipelines/ocr.py`: `AsrConfig` (mirror yaml) · `AsrSegment` · Protocol `AsrClient` · `FasterWhisperClient` (lazy import) · hàm thuần `is_hallucination` / `to_frame_window` · `run_asr()` trả `AsrRunSummary` |
| `configs/ingest/asr.yaml` | Đúng convention `ocr.yaml`: phẳng, key trùng tên field 1:1, mỗi key có comment nêu lý do |
| `scripts/run_asr.py` | Sao khung `run_ingest.py`. **Dựng model 1 lần ngoài vòng lặp** |
| `app/stores/es_client.py` | Thêm `AsrSegmentDocument` + `AsrEsStore`. ⚠️ `EsStore` hiện **hard-code** mapping OCR và field `ocr_text` (`:58-84`) ⇒ **không dùng lại được**, phải có store riêng. Trích `_VI_ANALYSIS_SETTINGS` ra hằng số để hai index dùng chung `vi_analyzer` |
| `app/stores/mongo_client.py` | Thêm collection **`asr_runs`** + `AsrRunDocument(video_id, asr_status, num_segments, model_id, fps)` |

**Index ở mức SEGMENT, không ép về keyframe.** Một câu lời dẫn 4–8s trải nhiều shot; cắt theo ranh giới keyframe làm **vỡ câu** ⇒ BM25 mất ngữ cảnh. Segment → keyframe suy ra được, chiều ngược lại thì không.

**⚠️ Trạng thái ASR để ở collection RIÊNG, không thêm field vào `VideoDocument`.** `upsert_video()` là **full `replace_one`** (`mongo_client.py:100-104`) ⇒ field ASR thêm vào đó sẽ bị ingest chạy lại **ghi đè im lặng**. Collection riêng cũng có nghĩa **không sửa một dòng nào của code ingest đang chạy**.

**⚠️ Quy đổi `time → frame_idx`.** `video_io.py:68-71` ghi `frame_idx → timestamp` là chiều hợp lệ duy nhất. Ta đi ngược, nên chốt luật:

```
start_frame_idx = max(0,               floor(start_time * fps))
end_frame_idx   = min(frame_count - 1, ceil (end_time   * fps))
```

**floor biên trái, ceil biên phải** — cửa sổ chỉ **nới rộng**, không bao giờ đánh rơi frame. `round()` bị cấm ở đây. Hai số này là **cửa sổ truy hồi, KHÔNG phải đáp án nộp BTC** — `frame_id` nộp bài luôn lấy từ `frame_idx` của chính keyframe trong Mongo/Milvus.

**⚠️ Lọc ảo giác — text bịa vào ES là ô nhiễm trực tiếp kênh BM25.** Bản tin có nhạc hiệu, B-roll im lặng, tiếng ồn hiện trường; Whisper hay bịa câu lặp ở đó. Bật `vad_filter=True` + loại segment khi `no_speech_prob > ngưỡng` **hoặc** `avg_logprob < ngưỡng` **hoặc** text quá ngắn. Đặt `condition_on_previous_text=False` để chặn vòng lặp ảo giác. **Mọi ngưỡng vào `configs/ingest/asr.yaml`**, không hard-code.

### P0.11b — Transcript → Milvus (dense) — 🔧 optional (chủ dự án 2026-08-07)

> Làm sau nếu còn dư thời gian, không chặn P0.11 "xong" hay bất kỳ phase nào khác — xem cập nhật ở đầu §P0.11.

**BM25-only không phải end state cho ASR.** Khác biệt cấu trúc so với OCR:

- **Chữ trên màn hình được viết ra để đọc** — tiêu đề, chạy chữ, tên nhân vật. Nó **đã ở dạng từ khoá**; người tìm sẽ gõ gần đúng những từ đó.
- **Lời nói là văn nói tự phát** — dài dòng, đầy liên từ và đại từ, diễn đạt tuỳ ý. **Không ai gõ truy vấn bằng những từ đó.**

⇒ **khoảng cách paraphrase của ASR lớn hơn OCR về mặt cấu trúc.** Lập luận "danh từ riêng IDF cao ⇒ BM25 mạnh" chỉ đúng cho phần **tên riêng/số liệu**, không cover phần mô tả sự kiện — mà đó là phần lớn lời dẫn. BM25 vẫn **cần**, chỉ là **không đủ**.

**Không dùng lại được BEiT-3:** `beit3_base_itc` huấn luyện **image–text** contrastive, không phải text–text; tokenizer là `XLMRobertaTokenizer` (`app/encoders/beit3.py:53`) đa ngôn ngữ ở tầng token nhưng **alignment học trên caption tiếng Anh** ⇒ vector tiếng Việt lệch. Cần **encoder thứ ba** — đề xuất `bge-m3`, ⚠️ **[Cần verify]** lúc implement — vào collection riêng `transcripts_{model_id}`.

Không vi phạm invariant *"một encoder duy nhất"* (CLAUDE.md §3): invariant đó nói *ingest và query **trong cùng một kênh** phải cùng model*, không cấm kênh text–text có encoder riêng. Nhưng `model_id` **bắt buộc** nằm trong tên collection.

**Vì sao tách khỏi P0.11a, không phải để hoãn:** transcription là phần **đắt và không đảo ngược** (~10h GPU/873 video); embedding đọc lại `transcript_text` từ ES bằng scroll API nên **không cần chạy lại Whisper**. Nếu ngưỡng lọc ảo giác sai thì phải transcribe lại — lúc đó embedding cũng bỏ đi. Chốt transcript trước rồi mới embed là thứ tự rẻ hơn. Trên 8GB, chạy tuần tự Whisper (~3GB) rồi `bge-m3` (~2,5GB) an toàn hơn nạp cả hai.

### Đo "ASR có cải thiện không" — bốn bước

⚠️ **Bẫy phải tránh:** bộ eval sinh từ **caption thị giác** (PRD §6.1) ⇒ truy vấn mô tả thứ **nhìn thấy được**, trong khi giá trị riêng của ASR là thứ **chỉ nghe thấy**. Chạy bộ eval hiện tại rồi kết luận *"ASR không cải thiện"* là **sai do thước đo lệch**, không phải do ASR kém.

| Bước | Đo gì | Cần `eval/` chưa? |
|---|---|---|
| **1** | **Hallucination Rate** (100 segment mẫu có seed, người nghe) · **WER** (người tự gõ 10–15 phút audio — *ground truth thật duy nhất của cả dự án*) · **Timestamp drift** (30 segment, `mean \|Δt\|`) | ❌ **làm được ngay sau khi chạy L21** |
| **2** | **Cổng "không được làm hại"**: `semantic` vs `+OCR` vs `+OCR+ASR` trên bộ eval **hiện có**. Câu hỏi là *"ASR có làm tụt không"*, không phải *"có tăng không"* | ✅ sau P0.9 |
| **3** | Tập truy vấn ASR-aware. **Bắt buộc**: Gemini sinh · BEiT-3+BM25 tìm (tách vai, CLAUDE.md §3) · cấm chép nguyên văn · **báo cáo độ chồng lấp từ vựng** — điểm dồn vào nhóm chồng lấp cao ⇒ **rò rỉ nhãn, không phải hệ thống giỏi** | ✅ |
| **4** | **Đối chứng âm — phép đo duy nhất không làm giả được**: chỉ giữ truy vấn mà thực thể **được nói ra** nhưng **không có trong index OCR**. `semantic-only` phải gần mức ngẫu nhiên; **độ nhảy khi bật ASR chính là giá trị của ASR** | ✅ |

Bước 3 bắt diễn đạt lại — đúng chỗ BM25 yếu nhất. Nên **báo cáo tách hai nhóm chồng lấp cao/thấp**: nhóm thấp tụt mạnh chính là **bằng chứng định lượng** để quyết P0.11b, thay vì tranh luận cảm tính.

Timestamp drift ở Bước 1 chính là con số điền vào **cửa sổ mở rộng bất đối xứng lệch về sau** — phát thanh viên nói *"Thủ tướng thăm nhà máy X"* khi hình còn là studio, B-roll đến sau vài giây.

**Kèm mọi con số:** git SHA · `asr_model_id` + `compute_type` + ngưỡng đã dùng · `eval_set_version` + hash · **số `video_id` duy nhất** · số truy vấn · seed.

### Verify

```powershell
pytest tests/test_asr.py -v
python -c "import yaml; from pipelines.asr import AsrConfig; print(AsrConfig(**yaml.safe_load(open('configs/ingest/asr.yaml',encoding='utf-8'))))"
```

Test bám convention `tests/test_ocr.py` (không fixture, double là `_Fake*`, `async def` không marker) — **không cần GPU, không cần ES**:
- `test_hallucination_filter_drops_segment_above_no_speech_threshold`
- `test_hallucination_filter_drops_segment_below_avg_logprob_threshold`
- ⭐ `test_frame_window_uses_floor_and_ceil_so_no_frame_is_dropped` — chống trượt cửa sổ TRAKE
- `test_frame_window_clamped_to_video_frame_count`
- `test_asr_indexes_only_kept_segments_to_elasticsearch`
- `test_asr_thresholds_come_from_config_not_hardcoded`
- `test_asr_summary_counts_each_drop_reason_separately`

**Lệnh chạy thật — chuẩn bị sẵn, chủ dự án tự chạy** (CLAUDE.md §10):
```powershell
pip install -e ".[asr]"
python scripts/run_asr.py --batches L21 --limit 1   # đọc transcript bằng mắt trước
python scripts/run_asr.py --batches L21             # 29 video
```
⚠️ **Không chạy song song với ingest** khi ingest còn giữ TransNetV2 + BEiT-3.

### Docs phải sửa trong cùng change (CLAUDE.md §2)

`PRD.md:133` đảo non-goal · `PRD.md:682` đóng **Q3** (*có audio, KHÔNG có phụ đề sẵn*) · `architecture.md` §3.2 thêm nhánh **thứ 4** (gắn từ node `A`, **không** từ `G` — audio đến từ `.mp4`, không từ keyframe) + §6 thêm `asr.py` + §10 thêm dòng 🔧 · `references.md:739` sửa `AIC 2025` → `AIC 2026` + thêm **§5.3 faster-whisper**.

⚠️ `docs/research.md` **stale 4 chỗ**, đọc phải cảnh giác: (1) OCR *đã* nối vào `ingest_video()` rồi; (2) hằng số tên `_TERMINAL_STATUSES` chứ không phải `_SKIP_STATUSES`; (3) `EsStore` **không** dùng lại được; (4) `VideoInfo` không có `duration`.

### Định nghĩa "xong" của P0.11

**Bắt buộc (P0.11a):**

- [ ] `pytest tests/test_asr.py` xanh, coverage ≥ 80%
- [ ] L21 (29 video) chạy xong, `asr_runs.asr_status == "done"` đủ số
- [ ] **Hallucination Rate + WER + timestamp drift đã đo bằng người** — không phải suy đoán
- [ ] Ngưỡng trong `asr.yaml` đã chỉnh theo số đo, **lý do chỉnh viết ra TRƯỚC khi nhìn số mới**
- [ ] Docs đã sửa đồng bộ, `research.md` đã đánh dấu các chỗ stale
- [ ] Báo cáo phase theo template CLAUDE.md §8

**Optional (P0.11b, 🔧 chủ dự án 2026-08-07 — không cần cho P0.11 "xong"):**

- [ ] `pytest tests/test_embed_transcripts.py` xanh
- [ ] `scripts/embed_transcripts.py` chạy xong trên L21, `transcripts_{model_id}` có đủ segment
- [ ] Bước 3/4 ở bảng đo phía trên (`semantic` vs `+OCR` vs `+OCR+ASR`, đối chứng âm) đã chạy — nếu bỏ P0.11b thì hai bước này cũng bỏ theo

---

## Định nghĩa "xong" của P0

- [ ] `docker compose up -d` → `/health` xanh cả ba store
- [ ] `python -m eval.run --config configs/eval/aic_pilot.yaml` ra bảng metrics thật
- [ ] Ingest kill giữa chừng → chạy lại tiếp đúng chỗ, không trùng bản ghi
- [ ] **Cả ba nhánh ingest chạy được** (VISION + TEXT/OCR + METADATA) — khớp PRD §F1
- [ ] **Bảng chi phí OCR theo từng bước lọc** + số token thực tiêu, trong ngân sách đã đặt
- [ ] **`vi_analyzer` kiểm chứng trên dữ liệu thật**, gõ không dấu vẫn khớp
- [ ] **Metric kênh `semantic` và kênh `ocr` báo cáo tách riêng**
- [ ] **Phép thử rò rỉ nhãn đã chạy** (chồng lấp `caption` ↔ `ocr_text`), kết luận ghi rõ
- [ ] Coverage ≥ 80%, `ruff` sạch
- [ ] Bảng PRD §6.7 đã có số đo thật thay cho giả định
- [ ] Q4, **Q6** đã trả lời; trạng thái R1 (BEiT-3 dựng được hay phải rút lui) đã chốt và ghi rõ
- [ ] Báo cáo phase đã viết

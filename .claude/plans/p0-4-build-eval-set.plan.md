# P0.4′ — `eval/build_eval_set.py` (sinh bộ nhãn eval v1)

## Context

Dataset AIC 2026 **không kèm nhãn truy vấn nào**. Không có nhãn thì `eval/run.py` (P0.9) không có gì để chấm, và mọi tuyên bố "cải thiện" trong dự án đều không kiểm chứng được — đây chính là khoảng trống mà baseline arXiv:2512.13169 để lại và dự án này phải lấp.

Ba nhánh ingest đã sẵn sàng làm nguyên liệu: Milvus có 221.903 keyframe embedding (BEiT-3), Mongo có 873 video `complete` + 221.903 keyframe kèm `shot_id`/`frame_idx`/`image_path`, `data/shots.json` có 102.706 ranh giới shot thật. Nhánh OCR bị **hoãn có chủ đích** (chủ dự án chốt 2026-08-09) — hệ quả: phép thử rò rỉ nhãn `caption` ↔ `ocr_text` (PRD §6.1.3) chưa chạy được, phải ghi là known gap chứ không lờ đi.

Kết quả cần đạt: `configs/eval/aic_eval_v1.json` đóng băng kèm hash — **mẫu số chính thức** của mọi báo cáo metric từ đây về sau.

> 🔴 **Đảo quyết định 2026-08-09 (cùng ngày, sau khi chạy thử thật):** dòng "Model sinh caption = `gemini-2.5-flash`" bên dưới **đã bị đảo lại thành `gemini-3.6-flash`**. Chạy `generate --limit-shots 5` thật phát hiện 2.5-flash trả **văn bản thường** thay vì JSON (5/5 shot lỗi `pydantic.ValidationError: json_invalid`). Verify bằng test trực tiếp qua SDK đã cài: cùng ảnh, cùng request, đúng request-shape (kiểm bằng `TypeAdapter(ResponseFormat).validate_python(...)` — validate OK, không phải lỗi code) — 2.5-flash bỏ qua `response_format`, 3.6-flash trả đúng JSON. Kết luận: `response_format` là tính năng model 3.x, không phải mọi model trong bảng "Interactions API supported models" đều enforce. Giữ nguyên bảng quyết định gốc bên dưới để thấy lý do ban đầu, chỉ đảo riêng dòng model — chi tiết đầy đủ ở "Sửa bổ sung" cuối file.

## Quyết định đã chốt (chủ dự án, 2026-08-09) — ~~model~~ xem đảo lại ở trên

| Quyết định | Giá trị | Vì sao |
|---|---|---|
| Phạm vi lấy mẫu | **Toàn kho 873 video** (không phải pilot L21–L24) | Cả 873 video đã ingest xong, khác thời điểm viết PRD ⇒ lệch PRD §6.1.2, **phải sửa docs trong cùng change** |
| Phân bổ tầng | **Căn bậc hai** số shot | L26/ViVU TV chiếm ~50% kho (51.055/102.706 shot); tỉ lệ thuần ⇒ 298/600 mẫu dồn vào một kênh, chia đều ⇒ bỏ phí tầng lớn. √ cho 156 |
| Reviewer | **1 người** | Metric "độ đồng thuận 2 reviewer" (PRD §6.1.3) **không đo được** ⇒ ghi `not_measured` trong báo cáo chất lượng, tuyệt đối không bịa số |
| Nhãn TRAKE | **Có** — thêm quota chuỗi liên tiếp | Bộ eval là artifact đóng băng; bổ sung sau ⇒ tăng `eval_set_version` ⇒ chạy lại toàn bộ baseline. Chi phí thêm ~200 ảnh ≈ 50K token, không đáng kể |
| Model sinh caption | **`gemini-2.5-flash`** (thay `gemini-3.6-flash`) | Rẻ hơn 5× input / 3× output; đồng bộ sẵn cho nhánh OCR sau này (~$69 → ~$14). ⚠️ Rủi ro kèm theo ở mục "Còn thiếu" |

### Kiểm chứng model (fetch 2026-08-09, CLAUDE.md §5)

- `gemini-2.5-flash` **có** trong bảng model được **Interactions API** hỗ trợ (`https://ai.google.dev/gemini-api/docs/interactions.md.txt` liệt kê `Gemini 2.5 Flash | Model | gemini-2.5-flash`). Không có ghi chú nào bắt model 2.5 phải dùng `generate_content` ⇒ không vi phạm ràng buộc "cấm `generate_content`".
- Giá (`https://ai.google.dev/gemini-api/docs/pricing`, paid tier, /1M token):

  | Model | Input | Output |
  |---|---|---|
  | `gemini-3.6-flash` | $1,50 | $7,50 |
  | **`gemini-2.5-flash`** | **$0,30** | **$2,50** |

- Ước tính riêng bước này: ~800 ảnh × 258 token + prompt ≈ 286K input, ~32K output ⇒ **≈ $0,17** (so với ≈ $0,67 ở 3.6-flash).
- ⚠️ **Chưa kiểm chứng:** cú pháp gửi *ảnh + structured output cùng lúc* không nằm trên trang overview của Interactions API — trang này chỉ trỏ sang hai feature guide riêng. Snippet trong `references.md` §5.1 (đã fetch 2026-07-25) vẫn là nguồn duy nhất; xác nhận lại khi chạy `--limit-shots 5`.

## Mâu thuẫn PRD ↔ thực tế phải xử lý

PRD §6.1.2 bước 2 yêu cầu đưa `objects[].detection_class_entities` vào ngữ cảnh prompt. **Dữ liệu này không tồn tại**: thư mục `objects/` của BTC cố ý không tải (`docs/references.md` §2 "Chưa tải — và có chủ đích"), và object detection tự chạy được ghi là "chưa cam kết" — không có `pipelines/object*.py` nào. ⇒ Ngữ cảnh chỉ còn **ảnh keyframe + `media-info.title`**. Sửa PRD §6.1.2 trong cùng change (CLAUDE.md: code lệch docs ⇒ sửa docs cùng lúc).

## Dữ liệu nguồn (đã verify trực tiếp)

- `data/shots.json` — `{"videos":[{"video_name":"L21_V001.mp4","shots":[{"shot_id":int 1..N liên tục,"start_frame","end_frame","start_sec","end_sec"}]}]}`. 873 video, 102.706 shot, không video nào thiếu. **Nguồn thẩm quyền** cho `start_sec`/`end_sec`/`order_index`.
- Mongo `keyframes` — `shot_id` dạng `"L21_V001_shot1"` khớp đúng đánh số `shots.json` (verify: `L21_V001_f3` frame=3 ∈ shot 1 `[0,8]`). Đọc qua `MongoStore.get_keyframes_for_video()` (`app/stores/mongo_client.py:125`), gom nhóm theo `shot_id` trong bộ nhớ — 1 query/video, không phải 1 query/shot.
- ⚠️ **Shot rỗng có thật**: L21_V001 có 337 shot nhưng chỉ 336 shot có keyframe. Phải log + đếm tường minh, cấm bỏ qua im lặng.
- `AIC2026Adapter.iter_videos("all")` → `VideoRef.title` + `.category` (=author). 873/873 video đủ cả hai.
- Ảnh keyframe có thật trên đĩa tại `image_path` (vd `data/keyframe/L21/L21_V001/L21_V001_3.jpg`).
- Phân bố shot pilot: trung vị 2,77s · p10 1,30s · p90 10,53s ⇒ ngưỡng `min_shot_duration_sec` loại flash-cut là hợp lý.

## Tệp tạo mới

### 1. `configs/eval/build_eval_set.yaml` ↔ `BuildEvalSetConfig`

Bám đúng style `configs/ingest/ocr.yaml`: phẳng, key trùng field 1:1, header ghi `Config(**yaml.safe_load(...))`, **mỗi field một comment nêu lý do**.

| Field | Giá trị | Lý do |
|---|---|---|
| `seed` | `20260809` | Tính tất định; CLAUDE.md cấm magic number ⇒ nằm ở config |
| `split` | `all` | Toàn kho 873 video |
| `kis_sample_size` | `600` | Quy mô v1 theo PRD §6.1.2 |
| `trake_sequence_count` | `40` | Số chuỗi liên tiếp |
| `trake_sequence_length` | `5` | N ∈ {3,5,7}, chọn giữa |
| `min_shot_duration_sec` | `1.0` | Dưới p10 (1,30s); shot < 1s là flash-cut, vô dụng làm đích truy hồi |
| `max_image_side` | `384` | Bám `ocr.yaml:max_ocr_side` — 258 token/ảnh thay vì 516 |
| `min_content_words` | `4` | Lọc chung chung tầng 1 ("Một người đang nói") |
| `near_duplicate_jaccard` | `0.6` | Ngưỡng Jaccard trên tập từ nội dung |
| `max_near_duplicates` | `3` | Caption khớp > 3 caption khác ⇒ chung chung (hiện thực hoá "khớp > X shot" của PRD §6.1.3) |
| `vietnamese_stopwords` | list | Trích từ nội dung; là dữ liệu nên nằm ở YAML |
| `max_retries` / `retry_base_delay_seconds` | `3` / `1.0` | Y hệt `ocr.yaml` |
| `gemini_model` | `gemini-2.5-flash` | Đã fetch kiểm chứng 2026-08-09: Interactions API hỗ trợ, $0,30/$2,50 per 1M. Khác `ocr.yaml` (đang để `gemini-3.6-flash`) — **cố ý**, không đồng bộ hoá vội vì OCR chưa chạy |

### 2. `eval/build_eval_set.py`

Một file tự chứa, đúng cây thư mục `docs/architecture.md:571-575` (ngang hàng `run.py`, **không** tách `pipelines/` + `scripts/`). Khung CLI sao y `scripts/import_asr_out.py`: docstring + `Usage:` + `WARNING:` (§10 — gọi Gemini hàng loạt, người chủ dự án tự chạy), boilerplate UTF-8 reconfigure, `_REPO_ROOT`, `_parse_args()`, `_load_yaml()`, async `_main() -> int`, `main()` wrapper.

**Hai subcommand** — vì bước 4 (review tay) không tự động hoá được:

```
python -m eval.build_eval_set generate --config configs/eval/build_eval_set.yaml
#   → configs/eval/aic_eval_v1.candidates.json   (người điền cột "approved")
python -m eval.build_eval_set freeze   --candidates configs/eval/aic_eval_v1.candidates.json
#   → configs/eval/aic_eval_v1.json + .sha256 + .quality_report.md
```

**Hàm thuần (test không cần mock):**

```python
stratum_key(video_id: str, author: str) -> tuple[str, str]
allocate_by_sqrt(stratum_sizes: Mapping[K, int], total: int) -> dict[K, int]
eligible_shots(shots: Sequence[dict], min_duration_sec: float) -> list[dict]
sample_kis_shots(by_stratum: Mapping[K, list[ShotRef]], quotas: Mapping[K, int],
                 rng: random.Random) -> list[ShotRef]
sample_trake_sequences(shots_by_video: Mapping[str, list[dict]], count: int,
                       length: int, rng: random.Random) -> list[list[ShotRef]]
select_middle_keyframe(keyframe_docs: Sequence[dict]) -> dict   # sort frame_idx, lấy len//2
content_words(caption: str, stopwords: frozenset[str]) -> frozenset[str]
flag_generic_captions(captions: Sequence[str], config: BuildEvalSetConfig) -> list[str]
extract_trake_sequences(segments: Sequence[GroundTruthSegment], n: int)
                        -> list[list[GroundTruthSegment]]
freeze_records(approved: Sequence[dict]) -> tuple[str, str]      # (json_text, sha256_hex)
compute_quality_report(candidates: Sequence[dict]) -> QualityReport
```

**Ranh giới I/O — Protocol + Fake (bám `tests/test_ocr.py:50-67`):**

```python
class CaptionGenerator(Protocol):
    def generate(self, image: Image.Image, title: str) -> str: ...

class GeminiCaptionGenerator:            # lazy `from google import genai` trong __init__
    ...                                   # client.interactions.create(...),
                                          # response_format schema = CaptionGeneration.model_json_schema()
                                          # retry chỉ bắt (TimeoutError, ConnectionError, OSError),
                                          # backoff retry_base_delay_seconds * 2**attempt
                                          # KHÔNG thêm nhánh `refusal` — repo chưa có tiền lệ cho Gemini

class CaptionGeneration(BaseModel):       # pydantic, KHÔNG dataclass — bám OcrExtraction
    caption: str = Field(description="...")

class KeyframeSource(Protocol):           # MongoStore thoả cấu trúc sẵn
    async def get_keyframes_for_video(self, video_id: str) -> list[dict]: ...
```

**Schema `aic_eval_v1.candidates.json`:**

```json
{ "eval_set_version": "v1", "generated_at": "...", "git_sha": "...",
  "config": { "...echo nguyên config đã dùng..." },
  "records": [{
     "video_id": "L21_V001", "shot_id": 42, "start_sec": 130.5, "end_sec": 134.2,
     "order_index": 42, "caption": "...", "task": "kis|trake",
     "keyframe_id": "L21_V001_f3915", "image_path": "...",
     "batch": "L21", "author": "60 Giây Official",
     "auto_filter": "kept|generic_short|generic_duplicate",
     "approved": null
  }] }
```

Người review điền `approved: true/false`. Bản ghi `auto_filter != "kept"` để sẵn `approved: false` nhưng **giữ lại trong file** để tính tỉ lệ loại.

**File đóng băng giữ đúng 5 field** của `GroundTruthSegment` (`{video_id, start_sec, end_sec, order_index, caption}`) ⇒ `AIC2026Adapter.load_ground_truth()` (`datasets/aic2026.py:87`) đọc được **không cần sửa một dòng nào**. Chuỗi TRAKE **suy ra được**, không cần field riêng: `extract_trake_sequences()` gom theo `video_id`, sắp theo `order_index`, tìm dãy `order_index` liên tiếp — vì shot TRAKE được lấy mẫu liên tiếp nên `order_index` của chúng là số nguyên liền nhau.

Hash: bám `scripts/run_ingest.py:141-163` — `hashlib.sha256(content.encode("utf-8")).hexdigest()`, ghi sidecar `<path>.sha256` chứa `digest + "\n"` (hex trần, giống `configs/eval/pilot_available_v1.txt.sha256`).

**Fail rõ ràng, không trả rỗng** (CLAUDE.md §7): `freeze` gặp bản ghi `approved: null` ⇒ raise, không im lặng bỏ qua; shot không có keyframe ⇒ đếm + log kèm `video_id`/`shot_id`; lỗi 1 shot trong vòng lặp ⇒ append vào `failed`, chạy tiếp, `return 1 if failed else 0`.

### 3. `tests/test_build_eval_set.py`

Bám style `tests/test_import_asr_out.py`: `class Test<Behavior>:`, helper `_`-prefix thay fixture, `tmp_path` sẵn có, `pytest.raises(..., match=...)`.

**4 test PRD/plan bắt buộc:**
- `test_eval_set_sampling_is_deterministic_given_seed`
- `test_eval_set_is_stratified_across_batch_and_author`
- `test_eval_set_hash_changes_when_content_changes`
- `test_trake_sequences_are_contiguous_and_time_ordered`

**Thêm (đều là chỗ hỏng âm thầm):**
- `test_sqrt_allocation_gives_small_stratum_more_than_proportional`
- `test_allocation_sums_to_requested_total`
- `test_middle_keyframe_is_picked_by_frame_idx_order`
- `test_shot_without_keyframes_is_reported_not_silently_skipped`
- `test_caption_below_min_content_words_is_flagged_generic`
- `test_near_duplicate_captions_are_flagged_generic`
- `test_specific_caption_is_not_flagged`
- `test_trake_sample_yields_contiguous_shot_ids`
- `test_frozen_json_matches_ground_truth_segment_schema` (load ngược bằng `AIC2026Adapter`)
- `test_freeze_raises_on_unreviewed_record`
- `test_caption_generator_retries_on_transient_error`
- `test_quality_report_marks_inter_reviewer_agreement_not_measured`

### 4. Sửa docs (cùng change)

- `docs/PRD.md` §6.1.2 — bỏ `objects[].detection_class_entities` khỏi ngữ cảnh bước 2 (kèm lý do: chưa tải, self-run chưa cam kết); ghi phạm vi đổi pilot → toàn kho 873; ghi phân bổ √ và quota TRAKE.
- `docs/PRD.md` §6.1.3 — ghi rõ độ đồng thuận 2 reviewer là `not_measured` ở v1 (1 người review).
- `.claude/plans/p0-foundation.plan.md` P0.4′ — đồng bộ ba điểm trên + đánh dấu OCR hoãn là known gap của phép thử rò rỉ nhãn.
- `pyproject.toml` — `eval/` cần `google-genai` + `pillow`; thêm extra `eval = ["aic-video-retrieval[ocr]"]` (đã pin `google-genai==2.16.0`, `pillow==11.0.0` ở `[ocr]`).
- `docs/references.md` §5.1 — thêm bảng giá đã fetch 2026-08-09 (`gemini-2.5-flash` $0,30/$2,50 vs `gemini-3.6-flash` $1,50/$7,50) và ghi nhận `gemini-2.5-flash` được Interactions API hỗ trợ. CLAUDE.md §5 bắt buộc: fetch API mới ⇒ cập nhật `references.md` **trong cùng change**. Kèm ghi chú ước tính lại chi phí OCR: ~45,8M token ⇒ ~$69 (3.6) so với ~$14 (2.5).

## Thứ tự thực thi (TDD: RED → GREEN)

1. `configs/eval/build_eval_set.yaml` + `BuildEvalSetConfig` dataclass.
2. `tests/test_build_eval_set.py` — viết trước, đỏ.
3. Hàm thuần (allocation, sampling, generic-filter, TRAKE, hash) → xanh.
4. Protocol + `GeminiCaptionGenerator` + orchestration async + CLI.
5. Sửa docs.
6. **Dừng lại, giao lệnh `generate` cho chủ dự án tự chạy** (§10 — gọi Gemini hàng loạt).
7. Sau khi người review xong ⇒ chạy `freeze` ⇒ báo cáo phase theo template CLAUDE.md §8.

## Verify

```powershell
# 1. Test + lint + coverage (chạy được ngay, không cần Gemini/Mongo)
./.venv/Scripts/python.exe -m pytest tests/test_build_eval_set.py -v
./.venv/Scripts/python.exe -m pytest --cov --cov-report=term-missing   # >= 80%
./.venv/Scripts/python.exe -m ruff check . ; ./.venv/Scripts/python.exe -m black --check .

# 2a. BAT BUOC truoc khi chay full: thu 5 mau, doc caption tieng Viet xem co dung khong
#     (gemini-2.5-flash re hon nhung yeu hon - phai mat thay tai nghe trươc khi dot 800 anh)
python -m eval.build_eval_set generate --config configs/eval/build_eval_set.yaml --limit-shots 5

# 2b. Sinh candidate day du - CHU DU AN TU CHAY (goi Gemini ~800 anh, can Mongo up)
python -m eval.build_eval_set generate --config configs/eval/build_eval_set.yaml

# 3. Nguoi review dien cot "approved" trong configs/eval/aic_eval_v1.candidates.json (>=100 mau)

# 4. Dong bang
python -m eval.build_eval_set freeze --candidates configs/eval/aic_eval_v1.candidates.json

# 5. Kiem cheo: adapter doc lai duoc file da dong bang
python -c "from pathlib import Path; from datasets.aic2026 import AIC2026Adapter; a=AIC2026Adapter(Path('data'), Path('configs/eval/aic_eval_v1.json')); gt=a.load_ground_truth('all'); print(len(gt), gt[0])"
```

**Cổng chất lượng (PRD §6.1.3 / R6):** tỉ lệ caption bị loại khi review **> 30% ⇒ dừng lại báo cáo**, không đóng băng — bộ nhãn kém thì mọi metric phía sau chỉ là nhiễu.

## Còn thiếu / rủi ro đã biết

- **Phép thử rò rỉ nhãn `caption` ↔ `ocr_text` chưa chạy được** vì nhánh OCR đang hoãn. Đây là lá chắn chính chống "BM25 cao bất thường" ở P0.9 ⇒ ghi rõ known gap trong báo cáo phase; khi nào chạy OCR thì đo bổ sung.
- **Độ đồng thuận 2 reviewer không đo được** (1 người) ⇒ `not_measured`, không bịa.
- Ngưỡng `near_duplicate_jaccard` / `max_near_duplicates` là 🔧 giả định ban đầu, **phải hiệu chỉnh sau lần chạy thật đầu tiên** và ghi lý do hiệu chỉnh **trước** khi nhìn số mới (CLAUDE.md §8).
- ~~⚠️ Rủi ro của việc chọn `gemini-2.5-flash`~~ **đã xảy ra thật, nặng hơn dự đoán** — không phải "caption chung chung hơn" mà là **không trả JSON, 5/5 shot lỗi ngay từ lần thử đầu**. Bước `--limit-shots 5` bắt buộc đã đúng chỗ: bắt được lỗi này trước khi chạy 800 mẫu, chi phí bỏ đi chỉ vài cent thay vì cả batch. Đã đảo về `gemini-3.6-flash` — xem "Sửa bổ sung" cuối file.
- Chênh lệch chi phí thực tế của quyết định này chỉ ~$0,50 cho bước eval — phần tiết kiệm lớn (~$55) nằm ở nhánh OCR, **chưa hiện thực hoá** vì OCR đang hoãn.

**Sửa bổ sung 2026-08-09 #1 (phát hiện trước khi chạy thật, chủ dự án hỏi thẳng):** `generate` ban đầu KHÔNG resume được — lỗi hết quota Gemini (`google.genai.errors.ClientError`, verify trực tiếp source `google-genai/errors.py`) không phải `OSError`/`RuntimeError`/`ValueError`/`TimeoutError`/`ConnectionError` nên xuyên qua mọi nhánh `except`, làm crash trước khi file candidates từng được ghi ⇒ mất toàn bộ caption đã trả tiền. Đã sửa: `_generate_candidates` nhận `existing_records` (bỏ qua shot đã có caption, không gọi lại Gemini) + `on_progress` checkpoint ghi file **sau mỗi video** thay vì chỉ một lần ở cuối. `_run_generate` tự đọc `--out` nếu đã tồn tại để resume. Test: `test_resume_skips_shots_already_in_existing_records`, `test_on_progress_checkpoints_after_each_video`.

**Sửa bổ sung 2026-08-09 #2 (chạy `--limit-shots 5` thật, 5/5 lỗi):** `gemini-2.5-flash` không dùng được cho structured output — trả `pydantic.ValidationError: json_invalid` cho cả 5 shot vì model trả text thường thay vì JSON. Verify nguyên nhân bằng script chẩn đoán gọi trực tiếp cùng ảnh + cùng request cho cả hai model qua chính SDK đã cài (`google.genai`): `gemini-2.5-flash` → `output_text` là câu tiếng Việt thường; `gemini-3.6-flash` → `output_text` là `{"caption": "..."}` đúng schema. Đã loại trừ khả năng lỗi request (dùng `TypeAdapter(ResponseFormat).validate_python(...)` của chính SDK để xác nhận request-shape hợp lệ). Đã đảo `gemini_model` về `gemini-3.6-flash` ở `configs/eval/build_eval_set.yaml` + default `BuildEvalSetConfig`, đồng bộ `docs/PRD.md` §6.1.2, `docs/references.md` §5.1, `p0-foundation.plan.md`.

✅ **Xác nhận 2026-08-09: chạy lại `--limit-shots 5` với `gemini-3.6-flash` — sinh đúng.** Chủ dự án xác nhận trực tiếp. Bước "mắt thấy tai nghe caption thật" đã qua — sẵn sàng chạy `generate` đầy đủ (không giới hạn), rồi review tay, rồi `freeze`.

**Sửa bổ sung 2026-08-09 #3 (chạy `generate` đầy đủ, crash sau 6 caption mới):** Gặp `HTTP 429` (rate-limit, không phải hết ngân sách tháng — thông điệp lỗi ghi `limit: 5` cho `generate_content_free_tier_requests`, dấu hiệu key đang ở free tier). Lỗi này là `google.genai._gaos.lib.compat_errors.RateLimitError`, verify trực tiếp qua SDK đã cài: **kế thừa một hệ phân cấp HOÀN TOÀN khác** `google.genai.errors.APIError` (module public) — hai class `APIError` trùng tên nhưng không liên quan gì nhau. Không phải `OSError`/`RuntimeError`/`ValueError` nên vẫn crash. **Tệ hơn:** checkpoint khi đó chỉ lưu SAU MỖI VIDEO, không phải sau mỗi shot — video đang xử lý dở có nhiều shot nên 6 caption vừa sinh (đã trả tiền) mất sạch dù cơ chế resume #1 đã có. Đã sửa: (a) checkpoint chuyển sang **sau MỖI shot** (cả thành công lẫn thất bại); (b) except clause nới thành `except Exception` có chủ đích, log rõ + ghi vào `failed`, không phải `except Exception: pass` (CLAUDE.md §7) — lý do: SDK có 2 hệ exception song song không tương thích nhau và không có type ổn định để bắt riêng. Test: `test_on_progress_checkpoints_within_a_single_multi_shot_video`, `test_unexpected_exception_type_is_recorded_not_left_uncaught`.

⚠️ **Vấn đề gốc chưa giải quyết được bằng code:** giới hạn `5 req` của free tier là hạn chế phía Google, không phải bug. Sinh ~640 caption ở tốc độ này gần như chắc chắn sẽ liên tục dính 429 (mỗi shot ~6-7s, vượt xa ngưỡng free tier). Cần chủ dự án tự kiểm tra billing trên Google AI Studio/Cloud Console cho key đang dùng.

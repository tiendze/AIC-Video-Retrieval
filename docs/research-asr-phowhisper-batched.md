# Research — Nâng cấp kênh ASR: PhoWhisper + batched inference

> **Trạng thái: ĐỀ XUẤT — chưa duyệt, chưa thực thi.** Không có dòng code nào được sửa khi viết file này.
>
> **Quan hệ với [`research.md`](research.md):** file đó là hồ sơ P0.11 (dựng nhánh ASR, đã duyệt 2026-08-02). File này là **bước tiếp theo** trả lời đúng câu hỏi mà `research.md` §9-B để treo: *"`large-v3` làm baseline trước, PhoWhisper là bước nâng cấp **có đo**"*. Đây là lúc đo.
>
> **Không ghi đè `research.md`** — nó ghi lại quyết định đã duyệt, đè lên là mất hồ sơ.

**Ngày:** 2026-08-05 · **Phase:** P0.11a-v2 · **Branch:** `feature/p0-foundation`

---

## 1. Phạm vi

### Giữ nguyên — không đụng tới

Luồng dữ liệu end-to-end **không đổi**, đây là ràng buộc cứng của đề xuất:

```
video .mp4 ──► ASR ──► lọc ảo giác ──► AsrSegment ──► Elasticsearch (asr_segments)
                                            │                    │
                                            │                    ├──► asr_transcripts.json
                                            └──► MongoDB          └──► asr_by_shot.json
                                                 (asr_runs)
```

- Ingest vào ES + Mongo: **giữ nguyên**
- Xuất `asr_transcripts.json` + `asr_by_shot.json`: **giữ nguyên**
- Schema `AsrSegmentDocument` (`app/stores/es_client.py:103-119`): **giữ nguyên**
- Bộ lọc ảo giác 3 bước (`pipelines/asr.py:158-170`): **giữ nguyên cơ chế**, chỉ có thể phải chỉnh ngưỡng (§5.3)
- Resume/idempotent qua `asr_runs.asr_status` (`scripts/run_asr.py:282-285`): **giữ nguyên**

### Trong phạm vi

| # | Thay đổi | Mục tiêu |
|---|---|---|
| **A** | Đổi checkpoint sang **PhoWhisper-large (bản CTranslate2)** | Giảm lỗi tên riêng/địa danh tiếng Việt |
| **B** | Bật **`BatchedInferencePipeline`, `batch_size=8`** | Giảm thời gian chạy |

Hai thay đổi **độc lập về kỹ thuật** (A đổi trọng số, B đổi cách gọi decoder) nhưng **phải đo tách nhau** — gộp lại rồi thấy số tốt hơn thì không biết công của cái nào.

### Ngoài phạm vi (ghi lại để không quên)

- `hotwords` / `initial_prompt` với gazetteer địa danh — nhắm đúng lỗi "Cửu Long", rẻ hơn đổi model, nhưng là **thay đổi thứ ba** cần đo riêng.
- `Zipformer-30M-RNNT-6000h` (WER 7.97% VLSP2025-Public, ~100x realtime): license `cc-by-nc-nd-4.0`, chạy trên `sherpa-onnx` chứ không phải CTranslate2, **không phơi `no_speech_prob`/`avg_logprob`** ⇒ phá thẳng bộ lọc ảo giác. Ứng viên nghiêm túc cho sau này, không phải bây giờ.
- `Qwen3-ASR-1.7B` (Apache-2.0, 01/2026): chưa công bố WER tiếng Việt riêng, chi phí tích hợp cao.

---

## 2. Hiện trạng — số đo thật

Từ log chạy L21 ngày 2026-08-04 (`scripts/run_asr.py:136-160`):

| Chỉ số | Giá trị |
|---|---|
| Số video | 28 |
| Tổng wall-clock | **1h 22m 45s** |
| Trung bình / video | 2m 57s |
| Tổng audio | 8h 31m 08s |
| Tốc độ | **6.2x realtime** |
| Video chậm nhất | `L21_V015` |

Cấu hình sinh ra số này (`configs/ingest/asr.yaml`): `large-v3` · `float16` · `beam_size=5` · `language=vi` · `vad_filter=true` · `condition_on_previous_text=false`.

**Chưa có số đo nào về chất lượng.** Không WER, không tỉ lệ tên riêng đúng. `tests/test_asr.py` chỉ test hàm thuần (lọc ảo giác, quy đổi frame, timing) bằng fake client — **không lần nào chạy model thật**. Đây là lỗ hổng phải bịt trước, xem §6.

Triệu chứng đã quan sát bằng mắt: địa danh **"Cửu Long" → "Cổ long"**. Đây là lỗi *thiên lệch từ vựng*, không phải lỗi âm học — model nghe đúng âm nhưng chọn sai chuỗi ký tự vì tiếng Việt không nằm đủ dày trong dữ liệu huấn luyện của Whisper gốc.

---

## 3. ⚠️ Đính chính hai chỗ trong docs

Đã fetch kiểm chứng **2026-08-05**. Hai khẳng định sau **không còn đúng**:

| Vị trí | Nội dung sai | Sự thật |
|---|---|---|
| `references.md:715` | *"PhoWhisper-large KHÔNG dùng được với faster-whisper — không có bản convert CTranslate2, và không phơi `no_speech_prob`"* | **Sai cả hai vế** — xem dưới |
| `research.md:27` | Nhắc lại y hệt khẳng định trên | Kế thừa lỗi |

**Vế 1 — "không có bản CTranslate2":** hiện có ít nhất ba bản công khai:

- [`quocphu/PhoWhisper-ct2-FasterWhisper`](https://huggingface.co/quocphu/PhoWhisper-ct2-FasterWhisper) — gộp 5 kích cỡ
- [`kiendt/PhoWhisper-large-ct2`](https://huggingface.co/kiendt/PhoWhisper-large-ct2)
- [`qbsmlabs/PhoWhisper-large`](https://huggingface.co/qbsmlabs/PhoWhisper-large)

Hoặc tự convert từ bản gốc BSD-3, một lệnh:

```bash
ct2-transformers-converter --model vinai/PhoWhisper-large \
    --output_dir models/PhoWhisper-large-ct2 \
    --copy_files tokenizer_config.json --quantization float16
```

**Vế 2 — "không phơi `no_speech_prob`":** hiểu nhầm về kiến trúc. `no_speech_prob` và `avg_logprob` là **output của bộ giải mã faster-whisper**, không phải thuộc tính của checkpoint. Mọi model chạy qua `WhisperModel` đều có hai field này, kể cả bản fine-tune. Đã xác nhận thêm: **chế độ batched cũng trả đủ hai field** (`faster_whisper/transcribe.py`, `Segment(... avg_logprob=..., no_speech_prob=...)`).

⇒ Bộ lọc ảo giác **chạy được với PhoWhisper**. Rào cản mà `references.md` dựng lên không tồn tại.

> **Hành động:** hai chỗ trên phải sửa **trong cùng change với code** (CLAUDE.md §2). Chưa sửa khi viết file này vì phạm vi là "chỉ viết research.md".

---

## 4. Thay đổi A — PhoWhisper-large (CTranslate2)

### 4.1 Bằng chứng

WER (%) từ [repo chính thức VinAI](https://github.com/VinAIResearch/PhoWhisper), thấp hơn = tốt hơn:

| Model | Tham số | CMV-Vi | VIVOS | VLSP2020 T1 | VLSP2020 T2 |
|---|---|---|---|---|---|
| PhoWhisper-small | 244M | 11.08 | 6.33 | 15.93 | 32.96 |
| PhoWhisper-medium | 769M | 8.27 | 4.97 | 14.12 | 26.85 |
| **PhoWhisper-large** | **1.55B** | **8.14** | **4.67** | **13.75** | **26.68** |

Nền tảng: fine-tune Whisper trên **844 giờ tiếng Việt đa vùng miền**. Đây chính là lý do kỳ vọng nó thắng ở địa danh — "Cửu Long", "Hậu Giang", "Bạc Liêu" nằm trong phân phối huấn luyện của nó, không nằm trong của Whisper gốc.

**⚠️ Bảng trên KHÔNG chứa Whisper large-v3 để so.** VinAI công bố so với large-v2 (kiến trúc gốc của PhoWhisper). Không có nguồn công khai nào đo PhoWhisper-large ↔ Whisper large-v3 trên cùng tập tiếng Việt. ⇒ **[Cần đo]** — đây là lý do §6 tồn tại, không được coi bảng này là bằng chứng đủ.

Đối chứng ngược đáng biết: trên audio **chuyển mã Việt–Anh**, Whisper large-v3 lại thắng PhoWhisper ([ViMedCSS](https://arxiv.org/pdf/2602.12911)). Bản tin HTV có chèn tên riêng tiếng Anh (tên nước, tổ chức) ⇒ rủi ro thật, phải nằm trong tập đo.

### 4.2 Kiến trúc — vì sao rẻ

`AsrClient` Protocol (`pipelines/asr.py:82-86`) đã dựng sẵn hàng rào, docstring của nó viết đúng chữ *"PhoWhisper"*. PhoWhisper-ct2 chạy qua **chính lớp `FasterWhisperClient` hiện có** — cùng `WhisperModel`, chỉ khác đường dẫn checkpoint.

⇒ **Không cần class mới, không cần Protocol mới.** Đây là điểm mấu chốt phân biệt PhoWhisper với Zipformer.

### 4.3 Rủi ro riêng của bản fine-tune

| Rủi ro | Vì sao | Giảm thiểu |
|---|---|---|
| **Timestamp trôi** — PhoWhisper fine-tune chủ yếu trên câu ngắn, khả năng dự đoán token timestamp cho long-form có thể suy giảm | Ảnh hưởng `start_frame_idx`/`end_frame_idx` (`pipelines/asr.py:182-188`) | **Thay đổi B giải quyết luôn**: batched mode ép `without_timestamps=True`, biên segment lấy từ VAD chứ không từ token ⇒ A và B có **cộng hưởng**, không xung đột |
| **Phân phối `avg_logprob` khác** | Model khác ⇒ độ tự tin khác ⇒ ngưỡng `-1.0` (`asr.yaml:45`) có thể loại nhầm hoặc để lọt | Đo lại tỉ lệ drop theo từng lý do; log đã có sẵn (`scripts/run_asr.py:322-335`) |
| **Bản convert của bên thứ ba** | Không kiểm soát được quantization/tokenizer họ dùng | Ưu tiên **tự convert** từ `vinai/PhoWhisper-large` (BSD-3) bằng lệnh ở §3 — kiểm soát được, và tránh phụ thuộc repo có thể bị xoá |
| **Nền là large-v2, không phải v3** | Mất các cải tiến của v3 ngoài phần tiếng Việt | Chấp nhận có ý thức; đây chính là thứ §6 phải đo |

### 4.4 VRAM

PhoWhisper-large = 1.55B tham số, fp16 ≈ **3.1GB** trọng số — ngang large-v3. Không có áp lực mới lên 8GB.

---

## 5. Thay đổi B — batched inference, `batch_size=8`

### 5.1 Cơ chế

Hiện tại `pipelines/asr.py:206` gọi `WhisperModel.transcribe()` — giải mã **tuần tự** từng cửa sổ 30s.

> **Làm rõ chữ "tuần tự"** — dự án dùng từ này cho ba tầng, đừng nhầm:
>
> | Tầng | Hiện tại | Đề xuất |
> |---|---|---|
> | 1 — nhánh (VISION → OCR → ASR) | tuần tự | **giữ nguyên** — đây là luật VRAM `CLAUDE.md` §4 |
> | 2 — video (`run_asr.py:281` `for video in videos:`) | tuần tự | **giữ nguyên** — song song hoá sẽ nạp 2 model ⇒ nổ 8GB |
> | 3 — cửa sổ audio *bên trong* 1 video | tuần tự | **đổi thành batched** ← đây |

Bộ giải mã Whisper là autoregressive: mỗi token là một lượt đọc toàn bộ ~3GB trọng số từ VRAM. Ở `batch_size=1` nút thắt là **băng thông bộ nhớ**, các nhân CUDA đứng chơi. Batching gộp 8 chunk (do VAD cắt, **của cùng một video**) vào một lượt — đọc trọng số một lần, dùng chung cho cả 8.

```
Video 18 phút ──► VAD bỏ im lặng ──► ~30 chunk (mỗi chunk ≤30s tiếng nói)
                                          │
      batch_size=8:  [c1..c8] ──► [c9..c16] ──► [c17..c24] ──► [c25..c30]
                     4 lượt, thay vì 30 lượt
```

### 5.2 Kỳ vọng

| Nguồn | Con số |
|---|---|
| [faster-whisper README](https://github.com/SYSTRAN/faster-whisper) — large-v3, `batch_size=16` | 46m12s → **25m50s**, và WER *giảm* 14.801 → 13.527 |
| [mobiusml](https://mobiusml.github.io/batched_whisper_blog/) | **2–4x** |
| `beam_size` 5 → 1–2 (tuỳ chọn, độc lập) | thêm ~1.5x |

⇒ Dự kiến **15–25x realtime**, tức 8h31m audio còn **~20–35 phút** thay vì 1h22m. ⚠️ **[Cần đo]** — ngoại suy, chưa chạy.

**`batch_size=8` chứ không phải 16:** README chạy GPU lớn hơn 3070 8GB. Thêm nữa, tăng batch **không cho tăng tốc tuyến tính** — encoder luôn nhận đúng 30s mel nên chunk ngắn bị đệm cho đủ, và decoder chỉ xong khi cả 8 chuỗi xong. 8 → 16 thường chỉ thêm ~20–30% trong khi VRAM tăng thật. Bắt đầu ở 8, đo, rồi mới nâng.

Ước tính VRAM: ~3GB trọng số + ~1.5–2.5GB activation ở `batch_size=8` ⇒ vừa 8GB. ⚠️ **[Cần đo]** bằng `nvidia-smi` lúc chạy thật.

### 5.3 Tham số bị vô hiệu trong batched mode

Đọc từ `faster_whisper/transcribe.py`, batched mode **hard-code** các giá trị sau, bỏ qua thứ ta truyền vào:

| Tham số | Bị ép thành | Hệ quả cho ta |
|---|---|---|
| `condition_on_previous_text` | `False` | **Không mất gì** — `asr.yaml:35` vốn đã đặt `false` để chống vòng lặp ảo giác |
| `without_timestamps` | `True` | Biên segment lấy từ VAD chunk, **không** từ token timestamp — xem §7 |
| `max_initial_timestamp` | `0.0` | — |
| `prompt_reset_on_temperature` | `0.5` | — |

⚠️ **`condition_on_previous_text` trong `asr.yaml` sẽ trở thành config chết** khi bật batched. Phải ghi chú ngay tại chỗ trong YAML, nếu không người sau sẽ chỉnh nó và tưởng có tác dụng — đúng kiểu hỏng âm thầm `CLAUDE.md` §3 cảnh báo.

`hotwords` và `beam_size` **vẫn có tác dụng** trong batched mode (cả hai đường đều dồn vào cùng `TranscriptionOptions`).

### 5.4 Rủi ro đã có người báo

[Issue #1179](https://github.com/SYSTRAN/faster-whisper/issues/1179): batched làm **mất hẳn segment** và **rò `initial_prompt` vào output** trên audio đa ngôn ngữ (Hà Lan lẫn tiếng Anh). Ta pin `language=vi` nên rủi ro thấp hơn, **nhưng bản tin có chèn tiếng Anh**. Chưa rõ đã fix ở version nào.

⇒ Đây là lý do phải giữ **đường lui**: `batched` là cờ bật/tắt trong config, không phải hành vi cứng.

---

## 6. Điều kiện chặn — phải có bộ đo trước

**Hiện không có cách nào biết model nào tốt hơn.** Không WER, không transcript chuẩn. Mọi so sánh bây giờ là cảm tính — đúng thứ `CLAUDE.md` §8 cấm: *"không sang phase sau khi phase trước chưa có số đo"*.

### 6.1 Bộ nhãn tối thiểu

- **~10 phút audio**, lấy từ **3 video L21** khác nhau, ưu tiên đoạn dày địa danh và có chèn tên riêng tiếng Anh (để bắt rủi ro §4.1).
- **Gõ tay** transcript chuẩn. Không dùng model nào sinh ra nó — `CLAUDE.md` §3: *model sinh nhãn ≠ model bị đánh giá*. Ở đây con người là người sinh nhãn, sạch.
- Lưu vào `eval/asr_gold/` kèm `eval_set_version` + hash, theo đúng kỷ luật báo cáo `CLAUDE.md` §8.

### 6.2 Hai metric, không phải một

| Metric | Vì sao cần |
|---|---|
| **WER tổng** | Chỉ số chuẩn, so được với bảng của VinAI |
| **Proper-noun recall** — tỉ lệ tên riêng/địa danh trong nhãn xuất hiện đúng trong transcript | **Đây mới là thứ phản ánh vấn đề đang gặp.** "Cửu Long" là 1 từ trong ~2000 từ; WER tổng có thể đứng yên trong khi chỉ số này nhảy rõ. Chỉ nhìn WER là **bỏ sót đúng cái mình đang sửa** |

Cộng thêm, lấy sẵn từ instrumentation đã có (`pipelines/asr.py:113-133`): `speed_ratio`, `mean_wall_sec`, và tỉ lệ drop theo từng lý do.

### 6.3 Ma trận chạy — 4 lần trên cùng 10 phút

| # | Model | Mode | Trả lời câu hỏi |
|---|---|---|---|
| 1 | large-v3 | tuần tự | Đường cơ sở (tái lập số cũ) |
| 2 | large-v3 | batched, bs=8 | **Batching có làm hỏng chất lượng không?** (issue #1179) |
| 3 | PhoWhisper-large-ct2 | tuần tự | **PhoWhisper có thật sự tốt hơn cho tiếng Việt không?** |
| 4 | PhoWhisper-large-ct2 | batched, bs=8 | Cấu hình đích |

Chạy 1↔2 và 3↔4 cô lập ảnh hưởng của B; 1↔3 và 2↔4 cô lập ảnh hưởng của A. **Không được nhảy thẳng tới 4.**

**Tiêu chí chấp nhận:**

- Cấu hình 4 thắng cấu hình 1 ở **proper-noun recall**, và **không tệ hơn** ở WER tổng.
- Cấu hình 4 đạt **≥ 12x realtime** (gần gấp đôi 6.2x hiện tại).
- Không có video nào bị **mất segment** so với cấu hình 1 (bắt lỗi issue #1179).

Không đạt ⇒ giữ `large-v3`, ghi lại số đo, không âm thầm bỏ qua.

---

## 7. Ảnh hưởng tới data contract

**Không có field nào thay đổi.** `AsrSegment`, `AsrSegmentDocument`, hai file JSON giữ nguyên hình dạng. Nhưng **giá trị** thay đổi theo hai cách phải nói rõ:

### 7.1 Biên segment sẽ khác — không phải bug

Batched mode lấy biên từ **VAD chunk** thay vì token timestamp của Whisper. Hệ quả dây chuyền:

```
biên segment đổi ──► start_frame_idx/end_frame_idx đổi   (asr.py:182-188)
                 └─► số shot khớp được trong asr_by_shot.json đổi   (export_asr.py:123-160)
```

⇒ **Phải chạy lại toàn bộ L21**, không được trộn kết quả cũ với mới. Rất may `segment_id = f"{video_id}_seg{index}"` (`asr.py:259`) là `_id` của ES, nên chạy lại **ghi đè sạch** thay vì nhân bản — trừ khi số segment giảm, lúc đó segment thừa của lần cũ **còn sót lại trong index**.

> ⚠️ **Việc phải làm kèm:** xoá index `asr_segments` trước khi chạy lại, hoặc xoá theo `video_id`. Nếu không, `asr_transcripts.json` (đọc ngược từ ES qua `scroll_all()`) sẽ trộn segment của hai model. Đây là **rủi ro ô nhiễm dữ liệu thật**, không phải lo xa.

Giả định *"segment không chồng lấn, sắp xếp tăng dần theo thời gian"* mà `_transcript_for_shots` (`export_asr.py:123-160`) dựa vào **vẫn đúng** — batched ghép kết quả lại theo trình tự thời gian trước khi trả ra.

### 7.2 `model_id` provenance phải đổi theo

`AsrRunDocument.model_id` (`app/stores/mongo_client.py:49`) và `meta.model_id` trong cả hai file JSON (`export_asr.py:103`, `:219`) hiện lấy từ `asr_config.model_size` = `"large-v3"`.

Đổi model mà không đổi chuỗi này ⇒ **không phân biệt được file JSON nào do model nào sinh ra** — vi phạm `CLAUDE.md` §8 (*mọi artifact báo cáo phải nêu model checkpoint*).

---

## 8. Điểm cắm code

### 8.1 Bảng thay đổi

| # | File | Vị trí | Thay đổi |
|---|---|---|---|
| 1 | `pipelines/asr.py` | `:48` | Đổi tên `model_size` → **`model_id`** (giá trị giờ là repo HF / đường dẫn, không phải "kích cỡ") |
| 2 | `pipelines/asr.py` | `:48-57` | Thêm `batched: bool = True`, `batch_size: int = 8` |
| 3 | `pipelines/asr.py` | `:197-203` | `__init__` dựng `BatchedInferencePipeline` khi `config.batched` |
| 4 | `pipelines/asr.py` | `:205-212` | `transcribe()` truyền `batch_size` khi ở chế độ batched |
| 5 | `configs/ingest/asr.yaml` | `:10` | `model_size:` → `model_id:`, giá trị mới + **ghi chú `condition_on_previous_text` bị vô hiệu** |
| 6 | `scripts/run_asr.py` | `:310`, `:353`, `:377` | `asr_config.model_size` → `asr_config.model_id` |
| 7 | `pyproject.toml` | `:49-52` | Giữ `faster-whisper==1.2.1` (đã có batching từ 1.1) — **không đổi** |
| 8 | `docs/references.md` | `:715` | Sửa khẳng định sai (§3) |
| 9 | `docs/research.md` | `:27` | Sửa khẳng định sai kế thừa (§3) |
| 10 | `tests/test_asr.py` | — | Thêm test cho nhánh batched (fake client, không cần GPU) |

**`tests/` không dùng `model_size`** (đã grep toàn repo) ⇒ đổi tên chỉ chạm 6 vị trí ở mục 1/5/6. Rẻ hơn dự kiến.

### 8.2 Phác thảo — CHƯA áp dụng

`configs/ingest/asr.yaml`:

```yaml
# Checkpoint CTranslate2. Đổi từ large-v3 (2026-08-05): PhoWhisper fine-tune
# 844h tiếng Việt đa vùng miền -- nhắm lỗi địa danh ("Cửu Long" -> "Cổ long").
# Giá trị này cũng là chuỗi provenance ghi vào asr_runs.model_id và meta của
# hai file JSON -- đổi checkpoint BẮT BUỘC đổi chuỗi này.
model_id: models/PhoWhisper-large-ct2

# Batched inference (faster-whisper >= 1.1). Gộp các chunk do VAD cắt của CÙNG
# một video vào một lượt decode -- KHÔNG phải chạy song song nhiều video, và
# KHÔNG vi phạm luật "tuần tự" của CLAUDE.md §4 (vẫn một model, một GPU context).
# Để false nếu batched làm mất segment -- xem issue faster-whisper #1179.
batched: true

# 8 chứ không phải 16 như README: 3070 chỉ có 8GB, và tăng batch không cho tăng
# tốc tuyến tính (chunk ngắn bị đệm đủ 30s, decoder chờ chuỗi dài nhất).
batch_size: 8

# ⚠️ VÔ HIỆU khi batched=true -- faster-whisper hard-code False trong chế độ
# batched. Giữ lại cho đường lui batched=false. Đổi giá trị này khi đang bật
# batched sẽ KHÔNG có tác dụng gì.
condition_on_previous_text: false
```

`pipelines/asr.py`:

```python
def __init__(self, config: AsrConfig = _DEFAULT_ASR_CONFIG) -> None:
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model = WhisperModel(
        config.model_id, device=config.device, compute_type=config.compute_type
    )
    # Batched decodes VAD chunks of the same video in parallel; sequential is
    # kept as an escape hatch (faster-whisper#1179 reports dropped segments).
    self._model = BatchedInferencePipeline(model=model) if config.batched else model
    self._config = config

def transcribe(self, video_path: str) -> list[AsrSegment]:
    # `batch_size` is rejected by WhisperModel.transcribe(), so it is only
    # passed on the batched path.
    extra = {"batch_size": self._config.batch_size} if self._config.batched else {}
    segments, _info = self._model.transcribe(
        video_path,
        beam_size=self._config.beam_size,
        language=self._config.language,
        vad_filter=self._config.vad_filter,
        condition_on_previous_text=self._config.condition_on_previous_text,
        **extra,
    )
    ...  # phần dựng AsrSegment giữ nguyên
```

⚠️ **[Cần verify lúc implement]** — batched mode bật VAD mặc định; cần xác nhận truyền `vad_filter=True` tường minh không gây lỗi trùng tham số ở faster-whisper 1.2.1.

---

## 9. Rủi ro

| # | Rủi ro | Mức | Giảm thiểu |
|---|---|---|---|
| R1 | Batched làm mất segment (issue #1179) | **Cao** | Cờ `batched` tắt được; tiêu chí chấp nhận §6.3 so số segment trực tiếp |
| R2 | Index ES trộn kết quả hai model | **Cao** | Xoá index trước khi chạy lại (§7.1) — thao tác **bắt buộc**, không phải tuỳ chọn |
| R3 | PhoWhisper kém hơn ở đoạn chèn tiếng Anh | Trung bình | Đưa đoạn có tiếng Anh vào bộ nhãn §6.1 |
| R4 | Ngưỡng `avg_logprob = -1.0` không hợp với model mới | Trung bình | Đo lại tỉ lệ drop; chỉnh ngưỡng **trước** khi nhìn WER, không phải sau (`CLAUDE.md` §8) |
| R5 | OOM ở `batch_size=8` | Thấp | Giảm về 4; hoặc `compute_type: int8_float16` |
| R6 | Bản convert bên thứ ba khác bản gốc | Thấp | Tự convert (§3) |
| R7 | Biên segment đổi ⇒ số shot khớp đổi, bị đọc nhầm thành hồi quy | Trung bình | Ghi rõ trong báo cáo phase; so **cùng cấu hình**, không so chéo |

---

## 10. Thứ tự thực hiện

1. **Dựng bộ nhãn §6.1** — 10 phút transcript gõ tay. *Đây là điều kiện chặn, không phải bước tuỳ chọn.*
2. Viết script đo WER + proper-noun recall (`eval/asr_wer.py`), test bằng nhãn giả trước.
3. Convert PhoWhisper sang CT2 (§3) — **lệnh nặng, chủ dự án tự chạy** (`CLAUDE.md` §10).
4. Sửa `docs/references.md:715` + `docs/research.md:27` (§3).
5. Code thay đổi A + B sau cờ config, TDD với fake client (không cần GPU).
6. Chạy ma trận 4 cấu hình §6.3 trên 10 phút audio. **Dừng lại đọc số.**
7. Nếu đạt tiêu chí: xoá index `asr_segments`, chạy lại toàn bộ L21, xuất lại hai file JSON.
8. **Báo cáo phase** theo mẫu `CLAUDE.md` §8.

---

## 11. Câu hỏi cần chủ dự án chốt

| # | Câu hỏi | Trạng thái |
|---|---|---|
| A | Chấp nhận bỏ ~2–3 giờ gõ tay 10 phút transcript làm bộ nhãn? Không có nó thì mọi so sánh là cảm tính | ⬜ chưa chốt |
| B | Tự convert PhoWhisper sang CT2, hay tải bản có sẵn của bên thứ ba? | ⬜ chưa chốt |
| C | Làm cả A+B cùng lúc, hay tách hai commit để đo riêng? (khuyến nghị: **tách**) | ⬜ chưa chốt |
| D | `hotwords` + gazetteer địa danh — làm luôn đợt này hay để đợt sau? Nó nhắm đúng "Cửu Long" và rẻ hơn đổi model | ⬜ chưa chốt |
| E | Chỉ chạy lại L21, hay nhân dịp này chạy luôn L22–L24? | ⬜ chưa chốt |

---

## 12. Độ tin cậy của file này

- Mọi khẳng định về **code hiện có** kèm `file:line`, đọc trực tiếp từ working tree ngày **2026-08-05**.
- Mọi khẳng định về **thư viện/model** đã fetch kiểm chứng ngày **2026-08-05** — nguồn liệt kê §13. Khác với `research.md` §10 (thời điểm đó chưa fetch).
- **Số đo hiện trạng §2 là kết quả chạy thật** (log L21, 2026-08-04).
- **Mọi con số kỳ vọng §5.2 là ngoại suy** — đánh dấu ⚠️ [Cần đo]. Chưa chạy PhoWhisper hay batched lần nào trên máy này.
- **Chưa có số đo chất lượng nào, cho bất kỳ model nào.** Đó là toàn bộ lý do §6 tồn tại.

---

## 13. Nguồn

- [VinAIResearch/PhoWhisper](https://github.com/VinAIResearch/PhoWhisper) — bảng WER §4.1
- [vinai/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large) — license BSD-3, 844h
- [quocphu/PhoWhisper-ct2-FasterWhisper](https://huggingface.co/quocphu/PhoWhisper-ct2-FasterWhisper) · [kiendt/PhoWhisper-large-ct2](https://huggingface.co/kiendt/PhoWhisper-large-ct2) · [qbsmlabs/PhoWhisper-large](https://huggingface.co/qbsmlabs/PhoWhisper-large) — bản CT2
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — API batched, benchmark
- [faster-whisper#1179](https://github.com/SYSTRAN/faster-whisper/issues/1179) — rủi ro R1
- [DeepWiki — transcription options](https://deepwiki.com/SYSTRAN/faster-whisper/4.3-transcription-options-and-configuration) — `hotwords` dùng được ở cả hai chế độ
- [mobiusml — Speeding up Whisper](https://mobiusml.github.io/batched_whisper_blog/) — cơ chế batching
- [ViMedCSS (arXiv 2602.12911)](https://arxiv.org/pdf/2602.12911) — đối chứng code-switching
- [hynt/Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) · [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — ngoài phạm vi §1

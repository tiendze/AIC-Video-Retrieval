# Plan: Nâng cấp kênh ASR — PhoWhisper + batched inference

**Ngày lập:** 2026-08-07 · **Branch:** `feature/p0-foundation` · **Phase:** P0.11a-v2
**Nguồn:** [`docs/research-asr-phowhisper-batched.md`](../../docs/research-asr-phowhisper-batched.md) (ĐỀ XUẤT, chưa duyệt)
**Trạng thái:** 🟢 **ĐÃ DUYỆT — đường nhanh (chủ dự án 2026-08-07):** *"cứ thực thi ingest vào đi chưa cần nhãn tay đâu mình sẽ tự kiểm tra trước, rồi sau đó mình sẽ test bằng nhãn sau, tại giờ mình đang cần gấp"*

> ⚠️ **Ghi vào hồ sơ, không được quên:** đường nhanh này **đổi cả model lẫn batched mà KHÔNG có số đo chất lượng nào**. Đây là quyết định có ý thức của chủ dự án vì lý do thời gian, **không phải** vì đã chứng minh được PhoWhisper tốt hơn. Cho tới khi §NHỊP 3 chạy xong, mọi phát biểu kiểu *"PhoWhisper tốt hơn"* đều **chưa có căn cứ** — chỉ được nói *"đã đổi sang PhoWhisper, chưa đo"*. Nhịp 3 **vẫn còn nợ**, xem §12.

---

## 1. Tóm tắt

Research doc đề xuất hai thay đổi: **A** đổi checkpoint sang PhoWhisper-large (CTranslate2) để sửa lỗi địa danh, **B** bật `BatchedInferencePipeline` với `batch_size=8` để tăng tốc. Cả hai chạy qua **chính lớp `FasterWhisperClient` hiện có** — không cần class mới, không cần Protocol mới.

Nhưng research doc §6 tự đặt một **điều kiện chặn**: dự án hiện **không có bất kỳ số đo chất lượng ASR nào**, nên không có cách nào biết PhoWhisper tốt hơn hay tệ hơn. `CLAUDE.md` §8 cấm sang phase sau khi phase trước chưa có số đo.

**Phát hiện quan trọng nhất — lối thoát khỏi thế bí:** hai thay đổi A và B **không cùng mức ràng buộc đo lường**.

| | Câu hỏi cần trả lời | Cần bộ nhãn người? |
|---|---|---|
| **B** (batched) | *"Có làm hỏng so với hành vi đã được chấp nhận không?"* | **KHÔNG** — mốc so sánh là chính output tuần tự đã chạy 2026-08-04 |
| **A** (PhoWhisper) | *"Có đúng hơn không?"* | **CÓ** — đây là tuyên bố về chân lý, phải có nhãn người |

⇒ **B đo được ngay hôm nay mà không cần gõ một chữ nhãn nào.** A vẫn bị chặn.

**Trạng thái code — verify trực tiếp 2026-08-07:** chưa có dòng nào của research doc được áp dụng. Số dòng trong research doc **vẫn đúng** (`asr.py:48`, `:201`; `asr.yaml:10`; `run_asr.py:310/353/377`). `tests/test_asr.py` không chứa `model_size`/`batched`/`batch_size` nào.

---

## 2. Phát hiện mới — research doc chưa nêu

Đọc trực tiếp working tree và `.venv/Lib/site-packages/faster_whisper/transcribe.py` (xác nhận `1.2.1`, khớp pin `pyproject.toml:51`).

### F1 — 🔴 CHẶN: đổi model rồi chạy lại L21 sẽ bỏ qua sạch 28 video, rồi ghi provenance sai

`scripts/run_asr.py:282-285`:

```python
existing = await mongo.get_asr_run(video.video_id)
if existing is not None and existing.get("asr_status") == STATUS_DONE:
    skipped.append(video.video_id)
    continue
```

Điều kiện skip **chỉ nhìn `asr_status`, không so `model_id`**. Sau khi đổi `asr.yaml` sang PhoWhisper và chạy lại L21: cả 28 video đều `done` ⇒ **bỏ qua sạch, không transcribe lại gì**. Nhưng vòng lặp vẫn chạy tiếp tới `run_asr.py:377`:

```python
export_ok = await _dump_json_artifacts(es, args, asr_config.model_size)
```

⇒ hai file JSON được ghi lại với `meta.model_id = "models/PhoWhisper-large-ct2"` **đè lên dữ liệu do `large-v3` sinh ra**. Đây là **hỏng âm thầm đúng loại `CLAUDE.md` §3 cảnh báo**: không crash, không log lỗi, chỉ có một file JSON nói dối về nguồn gốc của chính nó. Research §7.2 nhìn ra vấn đề provenance nhưng bỏ sót cơ chế tạo ra nó.

May mắn: `asr_runs` **đã lưu sẵn `model_id`** (`mongo_client.py:49`, `:164`) ⇒ sửa được mà không đổi schema.

### F2 — 🟠 Batched làm `avg_logprob`/`no_speech_prob` thô đi một bậc

- Đường tuần tự (`transcribe.py:1360-1364`): mỗi `Segment` mang giá trị **của riêng nó**.
- Đường batched (`transcribe.py:144-148`): mọi subsegment tách ra từ **cùng một VAD chunk** dùng chung **một** cặp giá trị.

Hệ quả cho bộ lọc ảo giác (`pipelines/asr.py:158-170`): độ hạt của quyết định drop nhảy từ *segment* lên *chunk*. Một chunk 25 giây chứa 4 câu sẽ **bị loại hoặc giữ cả cụm**. ⇒ **tỉ lệ drop chắc chắn đổi vì cơ chế, không phải vì model** — đọc sai chỗ này sẽ quy nhầm công/tội cho PhoWhisper. Research R4 chỉ nêu nguyên nhân từ checkpoint; thực tế còn nguyên nhân thứ hai, độc lập, đến từ B.

### F3 — 🟡 Tổ hợp `batched=true` + `vad_filter=false` là lỗi runtime, không phải cấu hình hợp lệ

`transcribe.py:413-419` — trong batched mode, không có `clip_timestamps` và `vad_filter=False`, audio > 30 giây thì:

```python
raise RuntimeError("No clip timestamps found. Set 'vad_filter' to True or provide 'clip_timestamps'.")
```

Mọi video của ta đều > 30 giây ⇒ tổ hợp này **luôn nổ ở video đầu tiên**, sau khi đã nạp model và tốn VRAM. Phải **fail ngay lúc dựng `AsrConfig`**. Test được, không cần GPU.

### F4 — 🟢 Đã giải quyết mục `⚠️ [Cần verify]` của research §8.2

Đọc `transcribe.py:254-299` (chữ ký `BatchedInferencePipeline.transcribe`):

- `vad_filter: bool = True` — **là tham số hợp lệ**, truyền tường minh **không gây lỗi trùng**.
- `condition_on_previous_text` — **có trong chữ ký** nhưng nằm dưới mục docstring `Unused Arguments` (`:351-364`) ⇒ nhận rồi bỏ qua, không raise. Xác nhận cảnh báo "config chết" của research §5.3 là đúng.
- `batch_size` — chỉ có ở đường batched; `WhisperModel.transcribe` (`:747-790`) **không có** ⇒ cách `**extra` ở research §8.2 là **đúng và bắt buộc**.
- `BatchedInferencePipeline.__init__(self, model)` (`:112-115`) — chỉ nhận đúng một đối số.
- `hotwords` có ở **cả hai** chữ ký (`:296`, `:788`) ⇒ khẳng định của research nay verify được bằng source cục bộ.

⇒ **Phác thảo code ở research §8.2 chạy được nguyên xi. Không còn ẩn số API nào.**

### F5 — 🟠 Rủi ro ô nhiễm R2 lan sang Milvus, không chỉ Elasticsearch

Research §7.1 chỉ nói tới index ES. Nhưng P0.11b đọc segment từ ES rồi upsert vào Milvus `transcripts_bge_m3` theo khoá `segment_id` (`milvus_client.py:244-260`). Segment thừa của lần chạy cũ **ở lại trong Milvus** y hệt như ở lại trong ES.

Bất đối xứng đáng chú ý: `MilvusStore` **có** `drop_collection()` (`milvus_client.py:66-74`) nhưng `TranscriptMilvusStore` **không có**; `AsrEsStore` **không có method xoá nào cả** (`es_client.py:121-202`).

⇒ Muốn thực hiện R2 mitigation "xoá index trước khi chạy lại" thì hiện **không có API nào trong repo làm được**.

---

## 3. Sự căng thẳng phải chốt: §6 nói "chặn", chủ dự án nói "đổi đi"

| | Hướng | Chi phí | Được | Mất |
|---|---|---|---|---|
| **(a)** | Làm đủ §10: gõ nhãn → viết eval → convert → code → chạy ma trận → chạy lại L21 | ~2–3h gõ tay + ~2–3h GPU | Mọi tuyên bố đều có số. Đúng chuẩn dự án | Chậm nhất. Rào cản 3 tiếng gõ tay đứng trước cả thay đổi không cần nhãn |
| **(b)** | Code A+B sau cờ config, **default giữ nguyên hành vi cũ**, hoãn đo | ~2h code, 0 GPU | Rẻ, có test, không vi phạm §8 (hành vi không đổi) | **Không giải quyết yêu cầu** — model vẫn `large-v3`. Cờ nằm đó không ai bật |
| **(c)** | Chỉ làm B, hoãn A | ~2h code + ~40 phút GPU | Ăn ngay phần tăng tốc; ít rủi ro chất lượng nhất | Không sửa "Cửu Long → Cổ long" — **đúng triệu chứng khởi nguồn** |

### ✅ ĐÃ CHỌN: **(d) đường nhanh** — chủ dự án 2026-08-07

**Chốt:** code cả A+B rồi **bật luôn cả hai** (`model_id: PhoWhisper` + `batched: true`), chạy lại L21 ngay, **chủ dự án tự kiểm tra transcript bằng mắt**. Bộ nhãn tay + `eval/` **hoãn sang nhịp 3, làm sau**.

**Cái được:** có transcript PhoWhisper trong tay hôm nay thay vì sau 3–4 tiếng gõ nhãn.

**Cái mất — phải ghi ra, không được quên:**

- Không biết PhoWhisper **thật sự** tốt hơn hay tệ hơn `large-v3`. Kiểm tra bằng mắt bắt được lỗi thô ("Cửu Long" đúng chưa) nhưng **không bắt được hồi quy phân tán** — ví dụ WER tổng tệ đi 3% trải đều khắp nơi thì mắt thường không thấy.
- Không cô lập được A với B. Nếu kết quả tệ, **không biết tại model hay tại batching** — phải quay lại chạy ma trận 4 cấu hình mới biết.
- research §4.1 cảnh báo trên audio **chuyển mã Việt–Anh**, `large-v3` **thắng** PhoWhisper. Bản tin HTV có chèn tên riêng tiếng Anh ⇒ khi kiểm tra bằng mắt, **cố ý tìm đoạn có tiếng Anh mà xem**, đừng chỉ xem đoạn địa danh.

**Ba lớp bảo vệ giữ lại — đều không tốn thời gian người:**

| Lớp | Vì sao giữ dù đang gấp |
|---|---|
| **F1 — skip theo `model_id`** | Không sửa thì chạy lại L21 **bỏ qua sạch 28 video**, không transcribe gì, rồi vẫn ghi JSON mang `model_id` mới ⇒ tưởng đã chạy PhoWhisper mà thực ra vẫn là transcript cũ. **Đây là cái bẫy tốn thời gian nhất khi vội** |
| **F5/R2 — xoá ES index + Milvus collection trước khi chạy** | Không xoá thì `asr_transcripts.json` trộn segment của hai model ⇒ kiểm tra bằng mắt thấy dữ liệu lẫn lộn không hiểu nổi, mất công debug oan |
| **So số segment trước/sau** | **Miễn phí** — `data/asr_transcripts.json` cũ đã có `num_segments` từng video. So một lệnh là bắt được issue #1179 (batched nuốt segment), thứ mắt thường **không** thấy vì transcript còn lại vẫn đọc trôi chảy |

> **Giữ nguyên cờ `batched`** làm đường lui: gặp lỗi mất segment thì đổi `false` chạy lại, không phải sửa code.

<details>
<summary>Khuyến nghị ban đầu (hướng lai 3 nhịp) — giữ lại làm hồ sơ, KHÔNG còn là đường đang đi</summary>

- **Nhịp 1 — code (không GPU, không nhãn).** Đưa cả A lẫn B vào sau cờ config, `asr.yaml` **giữ nguyên `large-v3` + `batched: false`** ⇒ hành vi mặc định không đổi ⇒ không có gì để đo, không vi phạm §8. Kèm sửa F1/F3/F5 và 2 chỗ sai trong docs.
- **Nhịp 2 — đo B bằng self-drift, không cần nhãn.** Chạy `large-v3` tuần tự vs batched trên cùng 3 clip. Mốc so sánh là **output tuần tự đã được chấp nhận 2026-08-04**, không phải chân lý — nên **không** phải "model tự chấm chính nó" (`CLAUDE.md` §3 cấm cùng model làm cả vai sinh nhãn lẫn vai bị đánh giá; ở đây không ai sinh nhãn, chỉ đo *độ lệch* so với baseline hiện hành). Đạt tiêu chí ⇒ bật `batched: true`. **Chủ dự án ăn phần tăng tốc mà không phải gõ chữ nào.**
- **Nhịp 3 — A vẫn bị chặn bởi nhãn người, không có đường vòng.** Nhưng chi phí gõ tay **giảm được ~60%** — xem câu **A** ở §4.

> **Phải nói thẳng:** nhịp 3 **không rút gọn thêm được nữa**. Muốn biết PhoWhisper có sửa được "Cửu Long" hay không thì phải có người nghe và ghi ra "Cửu Long". Bỏ qua bước này thì đổi model là **một thay đổi không đo được** — nó có thể đang làm hỏng chỗ khác mà ta không bao giờ biết (research §4.1: trên audio chuyển mã Việt–Anh, `large-v3` **thắng** PhoWhisper; bản tin HTV có chèn tên riêng tiếng Anh).

</details>

---

## 4. Quyết định — trạng thái sau khi chủ dự án chốt đường nhanh (2026-08-07)

| # | Trạng thái |
|---|---|
| **A** bộ nhãn tay | 🔵 **HOÃN sang nhịp 3** — chủ dự án tự kiểm tra bằng mắt trước |
| **B** nguồn CT2 | ⬜ **cần chốt trước khi chạy** — xem bảng dưới |
| **C** tách commit | ✅ **gộp** — đường nhanh bật cả hai cùng lúc (đánh đổi: không cô lập được A với B, đã ghi ở §3) |
| **D** hotwords | 🔵 **HOÃN** — vẫn là ứng viên tốt nếu PhoWhisper không sửa được "Cửu Long" |
| **E** phạm vi chạy | ✅ **L21 trước** — tập duy nhất có đường cơ sở để so số segment |
| **F** chỗ để nhãn/báo cáo | 🔵 **HOÃN cùng A** |
| **G** `eval/run.py` | 🔵 **HOÃN cùng A** — nhưng khi làm thì **vẫn phải là `eval/run.py`**, không đặt tạm ở `scripts/` |

5 câu của research §11, cộng 2 câu mới sinh ra từ §2 — lý do đầy đủ giữ lại bên dưới làm hồ sơ:

| # | Câu hỏi | Khuyến nghị | Lý do |
|---|---|---|---|
| **A** | Chấp nhận bỏ ~2–3h gõ tay 10 phút transcript? | **Có, nhưng gõ ít hơn nhiều:** danh sách tên riêng/địa danh cho **đủ 10 phút** (chỉ nghe và ghi tên, ~30–40 phút) + transcript đầy đủ chỉ cho **5 phút** (~1h) | Hai metric có chi phí nhãn rất khác nhau. **Proper-noun recall mới là metric phản ánh đúng vấn đề** (research §6.2), mà nhãn của nó rẻ hơn transcript đầy đủ cả bậc. WER chỉ cần đủ bắt hồi quy tổng thể. Cắt ~60% công gõ mà không mất metric nào |
| **B** | Tự convert CT2 hay tải bản bên thứ ba? | **Tự convert** từ `vinai/PhoWhisper-large` (BSD-3), ghi `sha256` output vào báo cáo | Research R6. Ba repo bên thứ ba không kiểm soát được quantization/tokenizer và có thể bị xoá. Lệnh chỉ một dòng |
| **C** | A+B cùng lúc hay tách hai commit? | **Tách theo *lần đo*, không theo *code*.** Code cả hai vào một commit; tách riêng **commit đổi giá trị default trong `asr.yaml`** — một commit bật `batched`, một commit đổi `model_id`, mỗi commit kèm số đo trong message | Research khuyến nghị tách để quy công đúng — mà thứ quyết định điều đó là giá trị YAML lúc chạy, không phải ranh giới commit Python. Tách code thành 2 commit chỉ tạo commit trung gian không ai chạy |
| **D** | `hotwords` + gazetteer làm luôn hay để sau? | **Để sau — nhưng ngay sau nhịp 3, không để trôi** | Là **thay đổi thứ ba**, gộp vào là mất khả năng quy công. Sau khi có bộ nhãn + `eval/`, đo hotwords chỉ tốn ~1h GPU. Đã verify chạy được ở cả hai chế độ (F4) |
| **E** | Chỉ chạy lại L21 hay cả L22–L24? | **L21 trước, một mình** | L21 là **tập duy nhất có đường cơ sở để so** (log 2026-08-04). L22–L24 chưa từng chạy ASR ⇒ chạy chúng là *dữ liệu mới*, không phải *so sánh*; gộp vào làm mọi số tổng hợp không diễn giải được |
| **F** *(mới)* | Bộ nhãn + kết quả đo để ở đâu? `eval/` là **package Python** (`pyproject.toml:62`), nhét JSON vào là trộn code với dữ liệu | **Nhãn:** `configs/eval/asr_gold_v1.json` (cùng chỗ `aic_eval_v1.json` đã có) · **Clip audio:** `data/eval/asr_clips/` (không commit) · **Báo cáo:** `data/eval/asr_runs/` (không commit) | Bám đúng chỗ repo đã đặt bộ eval. Nhãn phải **commit được** (cần version + hash, `CLAUDE.md` §8); audio và output thì không |
| **G** *(mới)* | `CLAUDE.md` §6: *"`eval/run.py` là entry point **duy nhất** của evaluation"* — nhưng nó **chưa tồn tại**. Tạo ngay hay đặt tạm CLI ở `scripts/`? | **Tạo `eval/run.py` ngay, với subcommand `asr`** | Đây là phép đo đầu tiên của dự án — đặt sai chỗ bây giờ thì mọi phép đo sau bám theo cái sai. `argparse` subparsers ~50 dòng. Đặt ở `scripts/` là tạo entry point cạnh tranh ngay lúc luật vừa có cơ hội được tuân thủ lần đầu |

---

## 5. Patterns to Mirror

| Việc | Mirror từ | Chi tiết cần bám |
|---|---|---|
| Thêm field `AsrConfig` | `pipelines/asr.py:43-58` | `@dataclass(frozen=True)`, default **khớp đúng** `asr.yaml`; docstring trỏ YAML là nguồn sự thật |
| Chú thích YAML | `configs/ingest/asr.yaml:31-35` | Mỗi tham số một khối comment **giải thích vì sao**, không phải nó là gì |
| Cờ bật/tắt backend | `pipelines/asr.py:197-203` | Import lazy trong `__init__`, giữ lý do "only needed when this class is used" |
| Hàm thuần + CLI mỏng | `pipelines/asr.py:113-133` ↔ `scripts/run_asr.py:136-160` | Logic ở `pipelines/`, script chỉ gọi + log. **Quy tắc resume (F1) phải theo pattern này** — hiện nằm inline trong script nên không test được |
| Test double | `tests/test_asr.py:35-41` | Class `_Fake*` viết tay, **không fixture**, không `unittest.mock` |
| Tên test | `tests/test_asr.py:64` | Mô tả **hành vi và lý do**: `test_frame_window_uses_floor_and_ceil_so_no_frame_is_dropped` |
| Comment giải thích số trong test | `tests/test_asr.py:65-67` | Viết thẳng phép tính vào comment, nói rõ *cái sai nào bị test này bắt* |
| Xoá dữ liệu cũ trước khi ghi nguồn mới | `app/stores/milvus_client.py:66-74` | Drop + tạo lại rỗng ngay để caller dùng tiếp. **Copy tinh thần này cho `TranscriptMilvusStore` và `AsrEsStore`** |
| Fail loud khi thiếu dữ liệu | `pipelines/export_asr.py:85-89` | `raise ValueError` kèm câu gợi ý hành động, không trả rỗng |
| Module đo lường thuần | `pipelines/verify_embed_parity.py:23-45` | Exception riêng, dataclass báo cáo có `@property passed`, hàm thuần nhận `Callable` để inject phần nặng |
| Docstring CLI | `scripts/run_asr.py:1-33` | `WARNING: GPU-heavy ... prepared here for you to run yourself` + `Prerequisites:` + `Usage:` |

---

## 6. Files to Change

| File | Action | Vì sao |
|---|---|---|
| `pipelines/asr.py` | **UPDATE** | `:48` `model_size`→`model_id`; `:48-58` thêm `batched`/`batch_size`; validate tổ hợp (F3); `:197-203` dựng `BatchedInferencePipeline`; `:205-212` `**extra`; hàm thuần quyết định resume (F1) |
| `configs/ingest/asr.yaml` | **UPDATE** | `:10` đổi khoá thành `model_id`; thêm `batched`/`batch_size`; `:31-35` cảnh báo `condition_on_previous_text` **bị vô hiệu khi batched** |
| `scripts/run_asr.py` | **UPDATE** | `:310/353/377` → `.model_id`; `:282-285` skip **theo cả `model_id`** (F1); thêm `--force`; `:21` docstring còn ghi cứng "large-v3" |
| `tests/test_asr.py` | **UPDATE** | Test cờ batched + resume model-aware + validate tổ hợp. Fake client, **không GPU** |
| `app/stores/es_client.py` | **UPDATE** | `AsrEsStore` thêm `delete_by_video()` / `recreate_index()` — hiện **không có API xoá nào** (F5), mà R2 mức Cao |
| `app/stores/milvus_client.py` | **UPDATE** | `TranscriptMilvusStore.drop_collection()`, mirror `MilvusStore:66-74` (F5) |
| `eval/run.py` | **CREATE** | Entry point **duy nhất** của evaluation (`CLAUDE.md` §6), subcommand `asr` — xem câu G |
| `eval/metrics/asr_quality.py` | **CREATE** | Hàm thuần: chuẩn hoá tiếng Việt (NFC), WER có breakdown, proper-noun recall, drift giữa hai lần chạy |
| `tests/test_asr_quality.py` | **CREATE** | TDD cho module trên. Không GPU, không nhãn thật |
| `configs/eval/asr_gold_v1.json` | **CREATE** *(chủ dự án gõ)* | Bộ nhãn người. Commit được, có `eval_set_version` |
| `configs/eval/asr_matrix.yaml` | **CREATE** | 4 cấu hình ma trận §6.3 + ngưỡng chấp nhận — **không magic number** |
| `pyproject.toml` | **UPDATE** | Thêm extra `eval` (thư viện WER). Nhóm `asr` **giữ nguyên** `faster-whisper==1.2.1` |
| `docs/references.md` | **UPDATE** | `:715` khẳng định sai; §5.3 bổ sung khối API `BatchedInferencePipeline` |
| `docs/research.md` | **UPDATE** | `:27`, `:133`, `:234` — **đính chính, không xoá lịch sử** |
| `docs/research-asr-phowhisper-batched.md` | **UPDATE** | ĐỀ XUẤT → ĐÃ DUYỆT; §8.2 gỡ `⚠️ [Cần verify]` (F4); bổ sung F1–F5; §11 điền quyết định |
| `pipelines/export_asr.py` · `app/stores/mongo_client.py` · `pipelines/embed_transcripts.py` | **KHÔNG ĐỔI** | `model_id` đã đúng tên ở cả hai chỗ |

**Không tạo:** class encoder mới, Protocol mới, code path thứ hai cho batched. `AsrClient` Protocol (`asr.py:82-86`) đã dựng sẵn hàng rào và docstring của nó **đã viết đúng chữ "PhoWhisper"** từ P0.11a.

---

## 7. Tasks

### NHỊP 1 — Code + docs (không GPU, không nhãn) → **rồi chạy luôn**

> **Đường nhanh:** T3 giữ default cũ trong lúc code để test hồi quy chứng minh không làm hỏng gì; **T5b** cuối nhịp mới lật default sang PhoWhisper + batched. Lật ở bước riêng để nếu chạy hỏng thì `git revert` đúng một commit là về trạng thái chạy được.

| # | Task | Validate |
|---|---|---|
| **T1** | Sửa 2 khẳng định sai: `references.md:715`, `research.md:27`/`:133`. Ghi rõ "đính chính 2026-08-07", giữ câu cũ dạng gạch bỏ để không mất hồ sơ | `rg "KHÔNG có bản convert CTranslate2" docs/` → rỗng |
| **T2** | **RED** — test `AsrConfig` mới: `model_id` tồn tại + `model_size` biến mất; `batched=True`+`vad_filter=False` ⇒ raise (F3); `batch_size ≥ 1` | `pytest tests/test_asr.py -x` đỏ đúng 3 test |
| **T3** | **GREEN** — `model_size`→`model_id: str = "large-v3"` (**giữ giá trị cũ**); `batched: bool = False` (**mặc định tắt** — đường lui là mặc định cho tới khi có số đo, R1); `batch_size: int = 8`; `__post_init__` validate F3 | T2 xanh |
| **T4** | **RED→GREEN** — nhánh batched: fake backend ghi lại kwargs; `batched=False` ⇒ **không** có `batch_size`, `batched=True` ⇒ **có**. Implement theo research §8.2 (nay đã verify, F4) | `pytest tests/test_asr.py -k batched` |
| **T5** | Đổi tên ở `asr.yaml` + `run_asr.py`. **Sửa F1:** tách `should_transcribe(existing_run, model_id, force) -> bool` thành hàm thuần trong `pipelines/asr.py`, skip **chỉ khi** `asr_status == done` **và** `model_id` khớp. Thêm `--force` | `test_video_transcribed_again_when_model_id_changed` + `test_video_skipped_when_same_model_already_done` — **RED trước** |
| **T6** | API xoá cho R2 (F5): `AsrEsStore.delete_by_video()` / `recreate_index()` (delete-by-query **trong engine**), `TranscriptMilvusStore.drop_collection()` | Test fake client kiểm đúng tên index/collection; integration test có marker |
| **T5b** | **Lật default** trong `asr.yaml`: `model_id: models/PhoWhisper-large-ct2` + `batched: true`. **Commit riêng**, message ghi rõ *"chưa đo — kiểm tra bằng mắt, số đo nợ ở nhịp 3"* | Test hồi quy của T2–T5 vẫn phải xanh (chúng test hành vi theo config, không phụ thuộc giá trị default) |
| **T6b** | Script so số segment trước/sau, đọc `data/asr_transcripts.*.json` — **hàm thuần, không GPU, không nhãn** | Bắt issue #1179. Miễn phí vì file cũ đã có `num_segments` từng video |
| **T7** | Cập nhật research doc + báo cáo phase nhịp 1 | Mẫu `CLAUDE.md` §8, **tiếng Việt**. **Ghi rõ: chưa có số đo chất lượng** |

### NHỊP 2 — Chạy thật (chủ dự án tự chạy) — **đường nhanh, không nhãn**

| # | Task | Ghi chú |
|---|---|---|
| **N1** | Chuẩn bị checkpoint PhoWhisper-ct2 (tự convert hoặc tải — câu **B** §4) | Lệnh ở §8 |
| **N2** | **Sao lưu** `data/asr_transcripts.json` → `asr_transcripts.large-v3.json` | **Làm TRƯỚC khi xoá bất cứ thứ gì.** Đây là đường cơ sở duy nhất để so số segment |
| **N3** | **Xoá** ES index `asr_segments` + Milvus collection `transcripts_bge_m3` | R2 + F5, **bắt buộc**, không phải tuỳ chọn |
| **N4** | Chạy `scripts/run_asr.py --batches L21`, xuất JSON tên mang model | F1 đã sửa nên không cần `--force`, nhưng có sẵn nếu muốn |
| **N5** | **Kiểm tra bằng mắt** — mở transcript đọc | ⚠️ **Cố ý tìm đoạn có chèn tiếng Anh mà xem**, không chỉ xem địa danh (research §4.1: `large-v3` thắng PhoWhisper ở audio chuyển mã) |
| **N6** | Chạy script T6b so số segment | Segment tụt mạnh ở video nào ⇒ nghi #1179 ⇒ đổi `batched: false` chạy lại video đó |
| **N7** | Chạy lại `scripts/embed_transcripts.py` (P0.11b, optional) | Segment mới chưa có vector |

### NHỊP 3 — Đo bằng nhãn (🔵 **NỢ LẠI**, làm sau khi hết gấp)

> Chủ dự án 2026-08-07 hoãn nhịp này để đi nhanh. **Không được coi là đã xong.** Chừng nào chưa chạy, câu trả lời cho *"PhoWhisper có tốt hơn không"* vẫn là **"chưa đo"**, không phải "có".

| # | Task | Ghi chú |
|---|---|---|
| **T8** | Chuẩn bị 3 clip cố định → `data/eval/asr_clips/*.wav` (16kHz mono). Ưu tiên đoạn **dày địa danh** + đoạn **có chèn tên riêng tiếng Anh** | **Cắt ra file chứ không dùng `clip_timestamps`**: input phải byte-identical giữa 4 cấu hình, và `clip_timestamps` vô hiệu hoá `vad_filter` ở đường batched (`transcribe.py:341-343`). ⚠️ `speed_ratio` trên `.wav` **không so trực tiếp** với 6,2x của log cũ (đo trên `.mp4`, gồm chi phí demux) |
| **T9** | **RED→GREEN** `eval/metrics/asr_quality.py`: `normalize_vi` (**NFC trước tiên** — tiếng Việt có hai dạng tổ hợp Unicode, thiếu bước này hai chuỗi nhìn giống hệt lại so ra khác; **giữ nguyên dấu thanh** vì dấu chính là thứ đang sai) · `word_error_rate` → breakdown 3 loại lỗi · `proper_noun_recall` → **phải trả `missing`** · `transcript_drift` | Coverage ≥ 80%. ⚠️ Thư viện WER: khuyến nghị `jiwer` nhưng **fetch docs xác nhận API/version lúc implement** (`CLAUDE.md` §5); **tắt transform mặc định của jiwer**, để `normalize_vi` chạy trước |
| **T10** | `eval/run.py` subcommand `asr` (câu **G** — vẫn phải là `eval/run.py`) | **Gọi thẳng `FasterWhisperClient` + bộ lọc, KHÔNG chạm ES/Mongo**. Báo cáo kèm git SHA · `model_id` · `eval_set_version` + hash · seed |
| **T12** | **Chủ dự án gõ** `configs/eval/asr_gold_v1.json` | **Không model nào được sinh file này.** Sửa nhãn về sau ⇒ tăng version + chạy lại baseline, **lý do sửa viết ra trước khi nhìn số mới** |
| **T14** | Chạy ma trận 4 cấu hình, **DỪNG LẠI ĐỌC SỐ** | Cô lập A với B — thứ đường nhanh đã đánh đổi. **R4+F2:** chỉnh ngưỡng `avg_logprob` **TRƯỚC khi nhìn WER** |
| **T16** | Báo cáo phase | Nếu số cho thấy PhoWhisper **tệ hơn** ⇒ quay về `large-v3`, và đó là kết quả hợp lệ, không phải thất bại |

---

## 8. Validation & lệnh chạy

**Chạy được ngay, không GPU:**

```powershell
rg -n "model_size" pipelines/ scripts/ configs/ tests/     # kỳ vọng: rỗng
rg -n "KHÔNG có bản convert CTranslate2" docs/             # kỳ vọng: rỗng
pytest tests/test_asr.py tests/test_asr_quality.py -v
pytest --cov=pipelines --cov=eval --cov=app.stores --cov-report=term-missing
black --check . ; isort --check-only . ; ruff check .
```

**Lệnh tốn tài nguyên — CHUẨN BỊ SẴN, chủ dự án tự chạy (`CLAUDE.md` §10):**

```bash
# T13 — convert PhoWhisper sang CTranslate2
ct2-transformers-converter --model vinai/PhoWhisper-large \
    --output_dir models/PhoWhisper-large-ct2 \
    --copy_files tokenizer_config.json --quantization float16

# T8 — cắt clip đo (thay start/duration bằng đoạn đã chọn)
ffmpeg -ss 312 -t 210 -i data/Videos_L21_a/video/L21_V015.mp4 \
       -vn -ac 1 -ar 16000 data/eval/asr_clips/L21_V015_c0.wav

# T11/T14 — chạy ma trận đo (KHÔNG chạm ES/Mongo)
python eval/run.py asr --matrix configs/eval/asr_matrix.yaml \
       --gold configs/eval/asr_gold_v1.json --out data/eval/asr_runs/

# T15 bước 1 — sao lưu đường cơ sở TRƯỚC khi xoá bất cứ thứ gì
cp data/asr_transcripts.json data/asr_transcripts.large-v3.json

# T15 bước 2 — xoá index ES (R2, BẮT BUỘC)
curl -X DELETE "http://localhost:9200/asr_segments"

# T15 bước 3 — xoá collection Milvus (F5, research doc bỏ sót)

# T15 bước 5 — chạy lại L21
python scripts/run_asr.py --batches L21 \
       --output-json data/asr_transcripts.phowhisper.json \
       --output-shot-json data/asr_by_shot.phowhisper.json

# T15 bước 6 — dựng lại vector transcript (P0.11b)
python scripts/embed_transcripts.py
```

> ⚠️ Cổng ES và cách xoá collection Milvus **xác nhận lại từ `docker-compose.yml` / `configs/stores/milvus.yaml` lúc implement** — chưa verify lúc lập kế hoạch, đừng chép nguyên lệnh trên.

---

## 9. Acceptance criteria

**Nhịp 1 (code) — không có tiêu chí về số đo, vì hành vi không đổi:**

- [ ] `rg "model_size"` trả rỗng trên `pipelines/` `scripts/` `configs/` `tests/`
- [ ] `asr.yaml` mặc định vẫn `large-v3` + `batched: false` ⇒ chạy lại L21 cho **đúng** kết quả cũ
- [ ] Test chứng minh `batch_size` chỉ đi đường batched
- [ ] Test chứng minh video được transcribe lại khi `model_id` đổi (**F1**)
- [ ] Test chứng minh `batched=true` + `vad_filter=false` raise ngay lúc dựng config (**F3**)
- [ ] `AsrEsStore` và `TranscriptMilvusStore` có API xoá, kèm test (**F5**)
- [ ] Coverage ≥ 80%; `black`/`isort`/`ruff` sạch
- [ ] `references.md:715` và `research.md:27`/`:133` đã đính chính

**Nhịp 2 (bật batched):**

- [ ] Cấu hình 2 đạt **≥ 12x realtime** trên clip
- [ ] **Không clip nào mất segment** so với cấu hình 1 (issue #1179)
- [ ] `transcript_drift` 2↔1 trong ngưỡng ghi ở `asr_matrix.yaml` (**chốt ngưỡng trước khi chạy**)
- [ ] VRAM đỉnh `nvidia-smi` < 8GB (R5)
- [ ] Tỉ lệ drop báo cáo kèm **giải thích F2**, không bị đọc nhầm thành hồi quy

**Nhịp 3 (đổi model):**

- [ ] Cấu hình 4 **thắng** cấu hình 1 ở **proper-noun recall**, **không tệ hơn** ở WER tổng
- [ ] "Cửu Long" không còn trong `missing` của `ProperNounReport`
- [ ] Ngưỡng `avg_logprob` xem xét lại **trước** khi nhìn WER, lý do chỉnh viết ra trước
- [ ] Mọi báo cáo kèm: git SHA · `model_id` · `eval_set_version` + hash · số clip · seed
- [ ] Không đạt ⇒ **giữ `large-v3`**, số đo vẫn ghi vào docs

---

## 10. Risks

| # | Rủi ro | Mức | Giảm thiểu |
|---|---|---|---|
| **F1** | Chạy lại với model mới bị skip sạch, JSON ghi provenance sai | 🔴 **Cao** | T5 skip theo cả `model_id`; test hồi quy bắt buộc; `--force` |
| R2 | Index ES trộn segment của hai model | 🔴 **Cao** | T6 API xoá; T15 xếp việc xoá **trước** việc chạy |
| **F5** | Milvus `transcripts_bge_m3` giữ vector segment cũ ⇒ search trả text của model cũ | 🔴 **Cao** | T6 `drop_collection()`; T15 bước 3 + 6 |
| R1 | Batched làm mất segment (#1179), bản tin có chèn tiếng Anh | 🟠 Cao | `batched` là cờ, **mặc định `false`**; tiêu chí nhịp 2 so số segment trực tiếp |
| **F2** | Tỉ lệ drop đổi vì batched gộp `avg_logprob` theo chunk, bị quy nhầm cho PhoWhisper | 🟠 Trung bình | Ma trận 4 cấu hình cô lập được; ghi F2 vào báo cáo **trước** khi đọc số |
| R4 | Ngưỡng `-1.0` không hợp model mới | 🟠 Trung bình | Chỉnh ngưỡng **trước** khi nhìn WER, viết lý do trước |
| R7 | Biên segment đổi ⇒ số shot khớp đổi, bị đọc thành hồi quy | 🟠 Trung bình | Ghi rõ trong báo cáo; chỉ so **cùng cấu hình** |
| R3 | PhoWhisper kém hơn ở đoạn chèn tiếng Anh | 🟠 Trung bình | Bắt buộc có một clip chứa tên riêng tiếng Anh |
| — | Bộ nhãn 10 phút quá nhỏ ⇒ WER có khoảng tin cậy rộng | 🟠 Trung bình | **Nói rõ trong mọi báo cáo**; proper-noun recall là metric quyết định |
| **F3** | `batched` + `vad_filter=false` nổ giữa chừng sau khi đã nạp model | 🟡 Thấp | Validate ngay lúc dựng `AsrConfig` |
| R5 | OOM ở `batch_size=8` trên 8GB | 🟡 Thấp | Hạ về 4; hoặc `compute_type: int8_float16` |
| R6 | Bản convert bên thứ ba khác bản gốc | 🟡 Thấp | Tự convert, ghi sha256 |
| — | Nhịp 3 bị bỏ dở, `batched` bật mà model không bao giờ đổi | 🟠 Trung bình | **Cái giá có ý thức** của hướng lai. Nếu chủ dự án không định gõ nhãn thì **nói ra ngay**, ghi vào docs rằng lỗi "Cửu Long" là **vấn đề đã biết, chưa xử lý** — không để trôi thành "hình như đã fix" |

---

## 11. Làm được ngay vs chủ dự án tự chạy

| Không cần GPU, không cần nhãn — làm ngay | Chủ dự án tự chạy (`CLAUDE.md` §10) |
|---|---|
| T1 sửa docs | T8 `ffmpeg` cắt clip |
| T2–T4 đổi tên field, thêm cờ, TDD fake client | T11 chạy đo B (2 cấu hình × 3 clip) |
| T5 sửa F1 + `--force` | T12 **gõ bộ nhãn** — không ai làm thay được |
| T6 API xoá cho ES/Milvus | T13 convert PhoWhisper sang CT2 |
| T9 `eval/metrics/asr_quality.py` + test | T14 chạy đủ ma trận 4 cấu hình |
| T10 `eval/run.py` (đọc config, ghi báo cáo) | T15 xoá index/collection + chạy lại L21 + embed lại |
| T7 cập nhật research doc | |

---

## 12. Còn thiếu / giả định đang treo

1. **7 quyết định ở §4 chưa chốt** — A đắt nhất (công gõ tay), G ảnh hưởng kiến trúc `eval/` lâu dài. Code không nên bắt đầu trước khi A, C, F, G có câu trả lời.
2. **Chưa đo gì cả trên máy này.** Mọi con số kỳ vọng (15–25x, VRAM 1,5–2,5GB activation) là ngoại suy từ README của faster-whisper chạy trên GPU khác.
3. **Bộ nhãn 10 phút là mẫu nhỏ.** Đủ bắt lỗi hệ thống ("Cửu Long" sai ở mọi lần xuất hiện), **không đủ** để tuyên bố chênh lệch WER 0,5%.
4. **Chưa xác nhận version + API thư viện WER** — phải fetch docs lúc implement (`CLAUDE.md` §5).
5. **Chưa xác nhận lệnh xoá Milvus collection và cổng ES** từ `docker-compose.yml` / `configs/stores/milvus.yaml`.
6. **`eval/run.py` chưa tồn tại** — phép đo đầu tiên của dự án; nó sẽ định hình mọi phép đo về sau (Recall@K, TRAKE, VQA). Thiết kế subcommand phải nghĩ tới điều đó, không chỉ nghĩ cho ASR.
7. **`hotwords` + gazetteer (câu D) đang bị hoãn** — nhắm **đúng** triệu chứng khởi nguồn và rẻ hơn đổi model. Đã verify chạy được ở cả hai chế độ (F4). Nếu nhịp 3 cho kết quả PhoWhisper không thắng rõ, đây là thứ nên thử **trước khi** kết luận vấn đề khó.
8. **Working tree đang có 3 file ASR sửa chưa commit** (`pipelines/asr.py`, `scripts/run_asr.py`, `tests/test_asr.py` — công việc P0.11a khác). Phải **commit hoặc stash trước** khi bắt đầu T2, nếu không hai luồng thay đổi trộn vào nhau trong cùng một diff.

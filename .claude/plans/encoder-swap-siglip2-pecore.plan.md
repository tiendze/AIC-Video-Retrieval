# Plan: Đổi encoder semantic — SigLIP2 so400m-384 (A1) + PE-Core-bigG-448 (A2)

**Ngày lập:** 2026-08-13 · **Branch đề xuất:** `feature/encoder-swap` · **Nguồn:** `docs/research-encoder-vietnamese.md`
**Trạng thái:** ĐỀ XUẤT — chưa sửa dòng code nào, chưa đốt giờ GPU nào.

## Phạm vi — chủ dự án chốt 2026-08-13

Chỉ đụng vào **đúng hai file cũ**, và cả hai đều theo kiểu **cộng thêm, mặc định tắt**:

| File cũ | Cách đụng | Hành vi khi không bật cờ |
|---|---|---|
| `scripts/import_embed_out.py` | Thêm cờ `--encoder` (mặc định `beit3`) | **Y hệt hôm nay**, không đổi một byte hành vi |
| `eval/run.py` | Thêm kênh ứng viên (mặc định tắt, như `score_clip_baseline`) | **Y hệt hôm nay** |

**KHÔNG đụng:** `app/main.py` · `scripts/run_ingest.py` · `app/services/search_service.py` · `pipelines/ingest.py`. Đường production và đường ingest gốc giữ nguyên vẹn — đây là lý do plan này **bỏ** phần "gom factory 3 chỗ" của research §7.1 và chỉ gom **2 chỗ mới phát sinh**.

Mọi thứ còn lại là **file mới**: encoder class, config model, notebook, test.

---

## Summary

Thay vision tower `beit3_base_itc_patch16_224` (768-dim, 224px, 2022) bằng **hai ứng viên chạy song song**: `google/siglip2-so400m-patch14-384` (1152-dim, đa ngữ 109 tiếng) và `timm/PE-Core-bigG-14-448` (1280-dim, text tower chỉ tiếng Anh — encoder MERVIN dùng để đạt 79/88). Embed **toàn bộ 221.903 keyframe** trên Kaggle theo đúng đường artifact đã dùng cho BEiT-3, import về Milvus thành **collection riêng cho mỗi `model_id`**, rồi đo bằng `eval/run.py` trên đúng bộ nhãn `aic_eval_v1.json`.

Kết thúc plan phải trả lời được **một câu hỏi số học**: kênh ứng viên có thắng `semantic_live_en = 0.4809` (AIC Final Score, segment-level, 173 query, 873 video) hay không. Cộng thêm một phép đo gần như miễn phí mà chỉ A1 cho phép: **dịch VI→EN vs không dịch trên cùng bộ vector ảnh** (research §6.3) — đây là bằng chứng dứt điểm cho giả thuyết §3 của research doc.

**Thứ tự ưu tiên trong plan này bị chi phối bởi một rủi ro duy nhất:** `pyproject.toml` pin `timm==0.4.12` + `transformers==4.44.2` cho BEiT-3, còn PE-Core là **timm-backed tower** (`vit_pe_core_gigantic_patch14_448`, chỉ có ở timm ≥ ~1.0.19) và SigLIP2 cần `transformers ≥ 4.49`. Lối thoát cũ ("ViT-B/32 không bao giờ import timm lúc runtime", `pyproject.toml:73-85`) **không áp dụng được cho A2**. Vì vậy **Task 0 là thí nghiệm dependency, chạy trước mọi thứ khác**, tốn ~1–2 giờ CPU chứ không phải hàng chục giờ GPU.

---

## Sự thật đã verify 2026-08-13 (dùng thẳng, không verify lại)

| | A1 | A2 |
|---|---|---|
| Checkpoint | `google/siglip2-so400m-patch14-384` | `timm/PE-Core-bigG-14-448` |
| License | apache-2.0 ✓ | **apache-2.0 ✓** (đóng research §9 Q1) |
| `embedding_dim` | **1152** (`vision/text hidden_size = projection_size = 1152`) | **1280** |
| Ảnh vào | 384×384, patch 14 | 448×448, patch 14 |
| Text tower | Gemma tokenizer, `vocab_size=256000`, **đa ngữ 109 tiếng** | English-only, `context_length = 72`, width 1280, 0.47B |
| Cách load | `transformers.AutoModel/AutoProcessor` + `get_image_features` / `get_text_features` | `open_clip.create_model_and_transforms('hf-hub:timm/PE-Core-bigG-14-448')` + `open_clip.get_tokenizer(...)` |
| Ràng buộc thư viện | `transformers >= 4.49` (config ghi `4.49.0.dev0`) | `open_clip` + **timm ≥ ~1.0.19** (vision tower uỷ quyền cho timm) |
| Dung lượng Milvus | 221.903 × 1152 × 4 = **~1,02 GB** | 221.903 × 1280 × 4 = **~1,14 GB** |

Cộng collection đang có: BEiT-3 768-dim ~682 MB + CLIP 512-dim ~455 MB ⇒ **tổng 4 collection ~3,3 GB vector thô** chưa kể index HNSW. Chấp nhận được; dọn collection thua cuộc sau khi chốt.

---

## Patterns to Mirror

| Cần làm | Mẫu có sẵn | Điểm phải bắt chước |
|---|---|---|
| Hợp đồng encoder | `app/encoders/base.py:23-45` | `(N, dim)` · `float32` · **L2-normalize BÊN TRONG** · deterministic. Protocol structural, không kế thừa |
| Class encoder A2 (open_clip) | `app/encoders/clip_baseline.py:56-123` | Normalize thủ công `features / features.norm(dim=-1, keepdim=True)`; `torch.no_grad()` + autocast fp16; **KHÔNG** gọi `empty_cache()` mỗi batch |
| Throughput ảnh | `app/encoders/clip_baseline.py:29-41,102-111` | `ThreadPoolExecutor` cho preprocess — **nút thắt là decode JPEG trên CPU, không phải GPU** (đo thật 2026-08-12: batch 64→256 chỉ +27%, VRAM phẳng 1–1,5 GB) |
| File config encoder | `configs/model/clip.yaml:1-34` | Comment cảnh báo `model_id` ↔ tên collection; ghi số đo thật vào comment `batch_size`; mọi hằng ở config, không magic number |
| CLI embed resume được | `scripts/embed_clip_baseline.py:98-120,152-174,190-213` | Checkpoint text theo `video_id`; `--limit-videos` để đo throughput trước; đọc keyframe doc từ Mongo → dựng `KeyframeVector` → `milvus.upsert` |
| Thêm kênh eval encoder riêng | `eval/run.py:671-718` + `:859-864` + `configs/eval/aic_pilot.yaml` | Encoder riêng + collection riêng + **import LAZY bên trong hàm** để report thường không cần cài extra; wiring `query_translator` giống `semantic_live_en` để so công bằng |
| Tên collection theo model | `app/stores/milvus_client.py:56` | `f"keyframes_{model_id}"` — đổi `model_id` là ra collection mới, không ghi đè |
| Hợp đồng artifact Kaggle | `.claude/plans/kaggle-offline-embed.plan.md` §3 | `manifest.json` + `probe/` + `vectors/<video_id>.{npy,jsonl}`; **thứ tự dòng jsonl = hợp đồng sắp xếp**; `keyframe_id = {video_id}_f{frame_idx}`; ghi file tạm rồi `os.replace` |
| Cấu trúc notebook Kaggle | `notebooks/kaggle_embed_keyframes.ipynb` cell 7→13 | Hằng dồn về một cell cấu hình · kiểm kê fail-loud · `%%writefile` script · **smoke test trước full run** · cell sinh manifest chạy cuối cùng |
| Cổng parity | `pipelines/verify_embed_parity.py:60-88` + `scripts/import_embed_out.py:118-155` | Lệch ⇒ `sys.exit(1)`, **chặn import**, không cảnh báo rồi cho qua |
| Test hợp đồng encoder | `tests/test_clip_encoder.py` | `pytest.importorskip` + `pytestmark = pytest.mark.model`; 6 test: shape/dtype · L2-norm · dim · determinism · empty image · empty text |
| Test import/parity thuần | `tests/test_import_embed_out.py`, `tests/test_verify_embed_parity.py` | Test không cần GPU/DB, chạy trong bộ mặc định (`addopts = -m 'not integration and not model'`) |

---

## 🔴 Quyết định chiến lược #1 — xung đột dependency

`eval/run.py:913` dựng `BEiT3Encoder`, `:921` dựng `ClipEncoder`, cả hai **trong một process**. Với A2 điều này bất khả thi vì `timm==0.4.12` (BEiT-3 vendor cần) và `timm≥1.0.19` (PE-Core cần) loại trừ nhau.

Bề mặt phụ thuộc timm của BEiT-3 vendor **đã kiểm, chỉ có 2 dòng**:

```
app/encoders/vendor/beit3_upstream/modeling_retrieval.py:17  from timm.models.registry import register_model
app/encoders/vendor/beit3_upstream/modeling_utils.py:14      from timm.models.layers import trunc_normal_ as __call_trunc_normal_
```

Cả hai đều là đường import mà timm 1.0.x **còn giữ shim deprecated** ⇒ phương án (c) có cơ hội thật, không phải hy vọng hão.

| Phương án | Ưu | Nhược | Đánh giá |
|---|---|---|---|
| **(a) Tách env theo họ encoder, merge report** | Không đụng pin production; A1+A2 chắc chắn chạy được | Hai lần chạy `eval/run.py` ⇒ phải chứng minh comparable; cần eval/run.py chạy được **không có BEiT-3** (hiện không) | **Fallback chính thức** |
| **(b) So với con số 0.4809 trong report cũ** | Rẻ nhất | Vi phạm tinh thần CLAUDE.md §8 (khác git SHA, khác thời điểm index); nếu Milvus/dữ liệu đổi giữa hai lần thì không ai biết | **BÁC BỎ** |
| **(c) Nâng pin `timm`/`transformers`, verify BEiT-3 còn chạy** | Một process, một report, so trực tiếp, đơn giản nhất về sau | Rủi ro: 2 đường import trên còn shim trong timm 1.0.x hay không **⚠️[Cần verify]**; đổi pin là đụng production | **Ưu tiên thử trước** |

**Chốt:** Task 0 thử (c) trong ~1–2 giờ. (c) xanh ⇒ đi (c). (c) đỏ ⇒ rơi về (a), và khi đó Task 9 phải làm thêm phần "eval/run.py chạy được khi không có encoder chính". Dù đi hướng nào, **Task 9 vẫn tách stores khỏi encoder chính** vì nó cũng cho phép chạy report chỉ-ứng-viên mà không nạp BEiT-3 vào 8GB VRAM.

Ràng buộc bổ sung dù chọn hướng nào: hai lần chạy phải **cùng `git_sha`, cùng `eval_set_hash`, cùng `pilot_available_v1.txt`, cùng ngày**, và report tổng hợp phải ghi rõ đây là hai process khác nhau (CLAUDE.md §8).

## 🔴 Quyết định chiến lược #2 — import lần thứ hai KHÔNG được chạm MongoDB

Phát hiện khi đọc code, không có trong research doc. **Đã verify từng dòng:**

- `scripts/import_embed_out.py:253` gọi `is_video_already_imported(...)`, mà Mongo hiện đã có đủ 873 video ở `ingest_status = "complete"` ⇒ chạy lại nguyên trạng cho artifact SigLIP2 sẽ **skip sạch 873/873 video, ghi 0 vector**, và log vẫn xanh.
- Nếu "chữa" bằng cách bỏ check đó thì tệ hơn: `next_global_index_start(mongo)` (`:243`) trả về `max+1` ⇒ **`global_index` trong collection mới lệch hoàn toàn với `s_v`/`e_v` đang lưu ở Mongo** ⇒ DANTE đọc sai cửa sổ, filter theo `global_index` sai — hỏng âm thầm, đúng loại lỗi CLAUDE.md §3 cảnh báo.

**Chốt:** thêm chế độ `--vectors-only`: đọc `global_index`/`frame_idx`/`timestamp`/`author`/`batch` **từ MongoDB theo `keyframe_id`** (nguồn sự thật đã có), chỉ ghi Milvus, **không ghi Mongo một dòng nào**. Đây chính là pattern `scripts/embed_clip_baseline.py:152-174` đã dùng — chỉ khác là vector đến từ file thay vì từ GPU local.

---

## Files to Change

| File | Action | Vì sao |
|---|---|---|
| `app/encoders/siglip2.py` | CREATE | `SigLip2Encoder` (A1) — transformers `AutoModel` + `AutoProcessor`, `get_image_features`/`get_text_features`, normalize trong class |
| `app/encoders/pe_core.py` | CREATE | `PeCoreEncoder` (A2) — `open_clip` hf-hub, gần như bản sao cấu trúc `clip_baseline.py` |
| `app/encoders/factory.py` | CREATE | Dispatch theo trường `kind` trong yaml, **import lazy trong từng nhánh**. Chỉ **2 caller**: `scripts/import_embed_out.py` và `eval/run.py`. Tồn tại để hai chỗ đó không chép logic dispatch của nhau — **không** phải bản gom 3 chỗ của research §7.1 (`app/main.py`, `scripts/run_ingest.py` giữ nguyên `BEiT3Encoder` trực tiếp) |
| `configs/model/siglip2.yaml` | CREATE | `kind`, `hf_repo_id`, `hf_revision` (commit SHA pin), `model_id`, `batch_size`, `preprocess_workers`, `device`, `max_text_tokens` |
| `configs/model/pe_core.yaml` | CREATE | Như trên + `open_clip_model_name` |
| `configs/model/beit3.yaml` | UPDATE | Thêm `kind: beit3` để factory nhận diện. Không đổi giá trị nào khác |
| `configs/model/clip.yaml` | UPDATE | Thêm `kind: open_clip` (giữ nguyên phần còn lại) |
| `configs/eval/aic_candidates.yaml` | CREATE | Config eval cho vòng đo ứng viên; **không sửa `aic_pilot.yaml`** để report cũ còn tái lập được |
| `eval/run.py` | UPDATE | Thêm kênh ứng viên tổng quát; tách dựng stores khỏi encoder chính; cờ bật/tắt kênh BEiT-3 |
| `pipelines/verify_embed_parity.py` | UPDATE | Thêm `verify_hf_weight_identity()` + `verify_image_probe()`; **giữ nguyên** `verify_checksums()` cũ cho BEiT-3 |
| `scripts/verify_embed_parity.py` | UPDATE | Bỏ hardcode `beit3.yaml`, nhận `--model-config` |
| `scripts/import_embed_out.py` | UPDATE | Thêm cờ `--encoder {beit3,siglip2,pe_core}` (**mặc định `beit3` ⇒ hành vi cũ y nguyên**). Cờ này kéo theo `--vectors-only` cho nhánh ứng viên |
| `pipelines/import_embed_out.py` | UPDATE | Thêm hàm dựng `KeyframeVector` từ Mongo doc + vector artifact (nhánh vectors-only), có kiểm khớp tập `keyframe_id` |
| `app/main.py` | **KHÔNG ĐỤNG** | Chủ dự án chốt 2026-08-13: đường production giữ nguyên `BEiT3Encoder` |
| `scripts/run_ingest.py` | **KHÔNG ĐỤNG** | Như trên — đường ingest gốc giữ nguyên |
| `notebooks/kaggle_embed_siglip2.ipynb` | CREATE | Notebook A1, kernel Kaggle mặc định, **không dựng conda py310** |
| `notebooks/kaggle_embed_pecore.ipynb` | CREATE | Notebook A2 (deps khác hẳn A1 ⇒ notebook riêng, không tham số hoá chung) |
| `pyproject.toml` | UPDATE | Thêm extra `encoder-candidates`; cập nhật comment `:73-85` (lối thoát `--no-deps` không còn phủ được A2) |
| `tests/test_siglip2_encoder.py` | CREATE | Bộ test hợp đồng, mirror `test_clip_encoder.py` |
| `tests/test_pe_core_encoder.py` | CREATE | Như trên |
| `tests/test_encoder_factory.py` | CREATE | Unit thuần (không GPU): dispatch đúng `kind`, raise rõ ràng với `kind` lạ, không import module nặng khi không cần |
| `tests/test_verify_embed_parity.py` | UPDATE | Case cho manifest v2 + image probe + chặn khi lệch revision |
| `tests/test_import_embed_out.py` | UPDATE | Case `--vectors-only`: không ghi Mongo, `global_index` lấy từ Mongo, lệch tập `keyframe_id` thì raise |
| `tests/test_eval_run.py` | UPDATE | Case config có kênh ứng viên; case chạy được khi tắt kênh chính |
| `docs/research-encoder-vietnamese.md` | UPDATE | Điền các ô ⚠️[Cần verify] §5 (dim, license), đóng §9 Q1, ghi lại việc chủ dự án **override §6.1** (không sàng lọc tập con) |
| `docs/references.md` | UPDATE | Thêm mục API SigLIP2 (transformers) + PE-Core (open_clip hf-hub) — CLAUDE.md §5 bắt buộc |
| `CLAUDE.md`, `docs/architecture.md`, `docs/PRD.md` | UPDATE (điều kiện) | Chỉ khi ứng viên thắng và đổi encoder production — chi tiết ở mục "Mâu thuẫn với docs" |
| Không tạo | — | Không tạo module preprocessing dùng chung (research §7.2: seam nằm ở class encoder, đúng thiết kế); không tạo script embed local cho A1/A2 (embed ở Kaggle) |

---

## Tasks

> Thứ tự chọn theo nguyên tắc: **chạm rủi ro lớn nhất sớm nhất**, và không treo job GPU dài trước khi có số throughput thật.

### T0 — Thí nghiệm dependency (CHẶN mọi task khác) · ~1–2h CPU · Risk: 🔴 cao

- **Action:**
  1. Dựng env sạch `.venv-cand` (py310, đúng `requires-python`): `torch==2.4.1`, `transformers>=4.49`, `timm>=1.0.19`, `open-clip-torch==3.3.0` (lần này **có** deps), `sentencepiece`, `pillow`.
  2. Trong env đó: load A1 bằng `AutoModel.from_pretrained(..., revision=<commit sha>)`, load A2 bằng `open_clip.create_model_and_transforms('hf-hub:timm/PE-Core-bigG-14-448')`. In `embedding_dim` thực tế, đối chiếu 1152 / 1280.
  3. Thử phương án **(c)**: trong một env thứ hai, giữ nguyên extra `beit3` nhưng nâng `timm` lên 1.0.x và `transformers` lên 4.49, rồi chạy `pytest -m model tests/test_beit3_encoder.py -v`. Điểm nghi ngờ cụ thể là đúng 2 dòng import đã liệt kê ở "Quyết định chiến lược #1".
  4. Ghi kết quả (pass/fail + version cụ thể) vào `docs/research-encoder-vietnamese.md` mục mới "§11 Kết quả spike dependency".
- **Mirror:** comment lý luận ở `pyproject.toml:68-92` — viết lại lý do mới theo đúng giọng đó, đừng để comment cũ nói sai.
- **Validate:** `python -c "import torch,timm,transformers,open_clip;print(...)"` in đủ version; `pytest -m model tests/test_beit3_encoder.py -v` (nhánh c) hoặc ghi rõ traceback nếu đỏ.
- **Đầu ra chốt:** chọn (a) hay (c). **Không viết code encoder trước khi có kết luận này.**

### T1 — Encoder factory (phạm vi hẹp) · Risk: thấp

- **Action:** `app/encoders/factory.py::build_encoder(config_path: Path, device_override: str | None = None) -> MultimodalEncoder`, dispatch theo `kind` ∈ `{beit3, open_clip, siglip2, pe_core}`. **Import implementation lazy bên trong từng nhánh** — nếu factory import cả 4 ở module level thì chính nó lại rơi vào bẫy T0. `kind` lạ ⇒ raise `ValueError` liệt kê các kind hợp lệ (CLAUDE.md §7: fail rõ ràng).
- **Chỉ 2 caller:** `scripts/import_embed_out.py` (T5) và `eval/run.py` (T9). Lý do tồn tại: hai chỗ đó đều phải chọn encoder theo cờ, gom lại để không chép dispatch hai lần. **`app/main.py` và `scripts/run_ingest.py` KHÔNG đụng tới** — chủ dự án chốt 2026-08-13.
- **Nợ kỹ thuật ghi ra, không giấu:** research §7.1 nói phải gom cả 3 chỗ hardcode `BEiT3Encoder` vì rủi ro lệch encoder giữa ingest và query. Plan này **cố ý chỉ gom 2**, đổi lấy việc không đụng đường production đang chạy. Rủi ro còn lại được chặn ở chỗ khác: cổng parity T4 (`hf_revision` + hash weight + image probe) mới là thứ thật sự bắt "vector ingest ≠ vector query". Nếu về sau chốt đổi encoder production (Q1) thì **phải trả nốt món nợ này** trong cùng change đó.
- **Mirror:** `eval/run.py:671-691` (lazy import + đọc yaml + truyền tham số theo tên).
- **Validate:** `pytest tests/test_encoder_factory.py -q` xanh; `git diff app/main.py scripts/run_ingest.py` **rỗng**.

### T2 — `SigLip2Encoder` (A1) + config + test hợp đồng · Risk: trung bình

- **Action:** class theo Protocol; `AutoModel.from_pretrained(repo, revision=...)` + `AutoProcessor.from_pretrained(repo, revision=...)`; `encode_image` → `get_image_features`, `encode_text` → `get_text_features`; normalize thủ công; `torch.no_grad()` + autocast fp16 trên cuda; `ThreadPoolExecutor` cho phần processor ảnh.
- ⚠️ **Bẫy phải verify lúc implement, không viết theo trí nhớ (CLAUDE.md §5):** SigLIP/SigLIP2 được train với text **pad tới `max_length` cố định (64)**; gọi processor với padding mặc định cho ra vector **sai âm thầm** — đúng loại bẫy như "QuickGELU mismatch" đã ghi ở `configs/model/clip.yaml:7-14`. Phải fetch docs xác nhận tham số đúng (`padding="max_length"`, `max_length`), ghi vào `docs/references.md` và vào comment của `configs/model/siglip2.yaml`.
- `model_id: siglip2_so400m_p14_384` ⇒ collection `keyframes_siglip2_so400m_p14_384`.
- **Mirror:** `app/encoders/clip_baseline.py` toàn bộ cấu trúc; `tests/test_clip_encoder.py` cho test (đổi assert dim thành **1152**).
- **Validate:** `pytest -m model tests/test_siglip2_encoder.py -v` trong env ứng viên (chạy được cả trên CPU với ảnh 96×64 giả). Thêm một kiểm tra tỉnh táo: cosine giữa `encode_text("a fire truck")` và `encode_image(<ảnh xe cứu hoả thật>)` phải cao hơn cosine với một ảnh bất kỳ — bắt lỗi "load được nhưng lệch không gian".

### T3 — `PeCoreEncoder` (A2) + config + test hợp đồng · Risk: trung bình

- **Action:** như T2 nhưng qua `open_clip`; `encode_image`/`encode_text` + normalize thủ công; assert dim **1280**.
- ⚠️ `context_length = 72` — phải đo **tỉ lệ query bị cắt** trên 173 caption đã dịch EN và in ra khi eval; caption bị cắt là chất lượng tụt vì lý do không liên quan tới vision tower.
- **Mirror:** `app/encoders/clip_baseline.py:59-77` gần như một-đối-một.
- **Validate:** `pytest -m model tests/test_pe_core_encoder.py -v`; in ra `model.visual.__class__` để xác nhận đúng là timm-backed tower, không phải fallback.

### T4 — Hợp đồng artifact v2 + parity probe thiết kế lại · Risk: 🔴 cao

Parity probe cũ hash file checkpoint local (`pipelines/verify_embed_parity.py:60-74`). A1/A2 tải từ HF hub, **không có file checkpoint trong repo để hash**. Cơ chế thay thế:

- **Action — `manifest.json` v2** (thêm field, không xoá field cũ để BEiT-3 artifact cũ vẫn đọc được):

| Field mới | Dùng để |
|---|---|
| `manifest_version: 2` | Phân biệt hai thế hệ artifact |
| `encoder_kind` | Chọn nhánh verify (`beit3` dùng nhánh cũ) |
| `hf_repo_id`, `hf_revision` | **Commit SHA ghim cứng**, truyền vào `from_pretrained(revision=...)` ở CẢ Kaggle lẫn local. Lệch ⇒ chặn |
| `weight_files_sha256` | `{tên_file: sha256}` của các file `.safetensors` **thực sự được load**, lấy từ HF cache. Bản thay thế trực tiếp cho `checkpoint_sha256` |
| `preprocess` | Chuỗi mô tả resize/normalize/tokenizer + `max_text_tokens` — truy vết khi probe fail |
| `library_versions` | `{torch, transformers, open_clip, timm}` |
| `image_probe` | (mới, xem dưới) |

- **Action — thêm image probe.** Trước đây local **không có ảnh** nên chỉ probe được text (plan cũ §3). Giờ **ảnh đã có local** ⇒ probe mạnh hơn hẳn: chọn **16 `keyframe_id` cố định** (ghi thẳng vào manifest), Kaggle lưu vector ảnh của chúng, local encode lại đúng 16 ảnh đó và so cosine. Text probe **giữ nguyên** như cũ.
- **Ngưỡng:** text probe giữ `0.9999`. Image probe đặt ngưỡng **riêng**, đưa vào config chứ không hard-code, và **quyết định giá trị TRƯỚC khi nhìn số** (kỷ luật CLAUDE.md §8): đề xuất `0.999` vì fp16 autocast trên GPU Kaggle (T4/P100) khác GPU local (3070) gây trôi số ở chữ số thứ 4. ⚠️[Cần đo] độ trôi thật ở smoke test; nếu thực tế còn tệ hơn 0.999 thì **không hạ ngưỡng cho qua** mà chuyển probe sang **fp32 hai đầu** (đúng cách chữa) rồi đo lại.
- Lệch bất kỳ ⇒ `sys.exit(1)`, **chặn import**. Không có nhánh cảnh báo-rồi-chạy-tiếp.
- **Mirror:** `pipelines/verify_embed_parity.py:60-88` (kiểu hàm raise `EmbedParityError`), `scripts/verify_embed_parity.py` (kiểu in report + exit).
- **Validate:** `pytest tests/test_verify_embed_parity.py -q` — thêm case: revision lệch ⇒ raise; thiếu `weight_files_sha256` ⇒ raise (không im lặng bỏ qua); image probe dưới ngưỡng ⇒ raise.

### T5 — Cờ `--encoder` cho `import_embed_out.py` · Risk: 🔴 cao (hỏng âm thầm)

**Nguyên tắc thiết kế cờ:** `--encoder beit3` là **mặc định** và phải cho ra **đúng luồng đang chạy hôm nay, không lệch một byte**. Nhánh ứng viên là đường rẽ mới bên cạnh, không phải viết lại đường cũ.

```
python scripts/import_embed_out.py                                   # y hệt hôm nay
python scripts/import_embed_out.py --encoder siglip2 --embed-out-dir data/embed_out_siglip2
```

- **Action:**
  1. `--encoder {beit3,siglip2,pe_core}` → ánh xạ sang `configs/model/<x>.yaml`, dựng encoder qua factory T1. `beit3` giữ nguyên `_BEIT3_CONFIG_PATH` hiện có (`:77`) làm giá trị mặc định ⇒ **`git diff` không đổi hành vi nhánh cũ**.
  2. Nhánh parity chọn theo `manifest.encoder_kind`: `beit3` → `verify_checksums` cũ (`:124`) nguyên vẹn; ứng viên → nhánh mới của T4 (`hf_revision` + `weight_files_sha256` + image probe).
  3. **`--encoder` khác `beit3` ⇒ tự động bật chế độ vectors-only**, và đây là điểm mấu chốt: bỏ qua `is_video_already_imported` (`:253`), **không** gọi `next_global_index_start` (`:243`), **không** ghi `keyframes`/`videos` ở Mongo. Thay vào đó: với mỗi `video_id`, đọc keyframe doc từ Mongo, join theo `keyframe_id` với dòng jsonl, ghi `KeyframeVector` vào collection mới. Buộc kèm nhau chứ không để thành hai cờ độc lập — `--encoder siglip2` mà quên `--vectors-only` sẽ ghi đè `global_index` của toàn bộ corpus, không được để lỗi đó nằm trong tầm với của người gõ lệnh.
  4. **Kiểm khớp bắt buộc, fail loud:** tập `keyframe_id` trong artifact phải **bằng đúng** tập trong Mongo cho video đó (thiếu / thừa / lệch đều raise, kèm `video_id` + vài id ví dụ). Không có bước này thì một artifact lệch nguồn keyframe sẽ import ra collection thủng lỗ chỗ mà eval vẫn chạy và chỉ ra số thấp — đúng loại "hỏng âm thầm" CLAUDE.md §3 nói tới.
  5. `--reset` ở nhánh ứng viên chỉ được drop **collection của `model_id` đang import**; `_reset_stores` hiện tại (`:210-217`) xoá sạch `mongo.keyframes` + `mongo.videos` ⇒ **tuyệt đối không đi vào nhánh đó** khi `--encoder != beit3`. Đây là đường phá dữ liệu nguy hiểm nhất trong toàn plan.
  6. `scripts/verify_embed_parity.py` nhận cờ tương đương để chạy gate riêng trước khi import.
- **Mirror:** `scripts/embed_clip_baseline.py:152-174` (Mongo doc → `KeyframeVector`), `pipelines/import_embed_out.py:94-103` (kiểm số hàng .npy == số dòng .jsonl).
- **Validate:** `pytest tests/test_import_embed_out.py -q` với case mới; rồi chạy thật `python scripts/import_embed_out.py --model-config configs/model/siglip2.yaml --vectors-only --limit-videos 1` và kiểm `collection.row_count` tăng đúng số keyframe của video đó, `db.keyframes.count()` **không đổi**.

### T6 — Notebook Kaggle A1 + smoke test bắt buộc · Risk: trung bình

- **Action:** `notebooks/kaggle_embed_siglip2.ipynb`, cấu trúc cell bám `kaggle_embed_keyframes.ipynb`:
  1. MD mục tiêu/cảnh báo · 2. `nvidia-smi` (in rõ **T4 hay P100** — quyết định fp16 có lợi hay không) · 3. clone repo qua Secret `GITHUB_TOKEN` · 4. `pip install` transformers≥4.49 + torch của Kaggle · 5. cell cấu hình hằng · 6. kiểm kê file ảnh (parse tên, khớp thư mục cha, bắt trùng) · 7. `%%writefile` script embed · 8. **smoke 1.000 keyframe có bấm giờ** · 9. **cell chốt quyết định** in ra `keyframe/s` và giờ ước tính cho 221.903 · 10. full run · 11. kiểm output · 12. manifest v2 + probe (text + image).
  2. **Bỏ hẳn env conda py310.** A1 chạy trên kernel mặc định của Kaggle — `app/encoders/__init__.py` **rỗng** (đã kiểm) nên `import app.encoders.siglip2` không kéo theo beit3. Đây là đơn giản hoá lớn so với notebook cũ.
  3. ⚠️ **Không** `pip install -e .` (pyproject `requires-python >=3.10,<3.11` sẽ chặn trên kernel 3.11/3.12). Chỉ thêm repo vào `sys.path`.
  4. ⚠️ **Bẫy shadow module:** repo có package tên `datasets/` (đã kiểm: `datasets/__init__.py`, `aic2026.py`, `base.py`) trùng tên thư viện HF `datasets`. `sys.path.insert(0, repo)` sẽ khiến `import datasets` bắt phải package của repo. ⇒ **append vào cuối `sys.path`**, hoặc nạp module encoder bằng `importlib.util.spec_from_file_location`. Ghi cảnh báo này ngay trong cell.
- **Cổng bắt buộc:** chỉ bấm cell full run sau khi số smoke được ghi vào notebook. **Không ghi số giờ nào trong plan này** — ⚠️[Cần đo].
- **Đòn bẩy throughput theo thứ tự thử:** `preprocess_workers` (nút thắt thật, theo đo 2026-08-12) → `batch_size` → chia đôi danh sách video chạy 2 session song song (artifact shard theo video nên merge là copy file) → cuối cùng mới tới `Image.draft()`, và **chỉ khi** cosine draft-vs-full ≥ 0,999 và ghi vào `manifest.preprocess`.
- **Ràng buộc lịch:** Kaggle **30h GPU/tuần** cho **cả hai** ứng viên × 221.903 ảnh. Nếu smoke cho thấy A1 + A2 > 30h ⇒ dừng lại hỏi chủ dự án, ưu tiên A1 (research §5: cược an toàn), A2 sang tuần sau.
- **Validate:** cell 11 in số vector (kỳ vọng **221.903**), số video (**873**), dung lượng (~**1,02 GB**), `‖v‖₂ ∈ [0,999; 1,001]` trên mẫu ngẫu nhiên mỗi shard.

### T7 — Notebook Kaggle A2 · Risk: trung bình

- Như T6 nhưng deps `open-clip-torch` + `timm≥1.0.19`; assert `model.visual` là timm tower; dung lượng kỳ vọng ~**1,14 GB**.
- **Chạy sau khi T6 có số**, để biết còn bao nhiêu quota GPU. Notebook riêng vì deps xung khắc với A1 về mặt tối giản môi trường, và vì cell cấu hình khác hẳn.

### T8 — Import A1 về Milvus qua cổng parity · Risk: cao

- **Action:** tải `embed_out_siglip2/` về, chạy `scripts/verify_embed_parity.py --model-config configs/model/siglip2.yaml` (phải xanh), rồi `scripts/import_embed_out.py --model-config configs/model/siglip2.yaml --vectors-only`. **Lệnh do chủ dự án tự chạy** (CLAUDE.md §10) — agent chỉ chuẩn bị.
- **Validate:** `row_count` collection `keyframes_siglip2_so400m_p14_384` = **221.903**; `db.keyframes.count()` không đổi; spot-check 5 `keyframe_id` bất kỳ có `global_index` **giống hệt** trong hai collection BEiT-3 và SigLIP2.

### T9 — Kênh ứng viên trong `eval/run.py` · Risk: trung bình

- **Action:**
  1. Tổng quát hoá `_build_clip_search_service` (`eval/run.py:671-718`) thành `_build_candidate_search_service(settings, es, mongo, model_config_path, use_translator: bool)` — dùng factory T1, lazy import giữ nguyên tinh thần.
  2. `EvalRunConfig` thêm `candidate_channels: tuple[dict, ...]`, mỗi phần tử `{name, model_config, translate}`. Giữ `score_clip_baseline` **nguyên vẹn** để report cũ tái lập được.
  3. Tách `_build_stores` (`:621-637`) sao cho Mongo/ES/random/bm25 **không phụ thuộc encoder chính**; thêm `score_primary_channels: bool` — false ⇒ không dựng `BEiT3Encoder` chút nào. Đây là phần bắt buộc nếu T0 rơi vào phương án (a), và vẫn hữu ích ở (c) (report chỉ-ứng-viên không phải nạp BEiT-3 vào 8GB VRAM).
  4. `configs/eval/aic_candidates.yaml` khai báo các kênh:

| Kênh | Encoder | Dịch VI→EN | Trả lời câu hỏi gì |
|---|---|---|---|
| `semantic_live_en` | BEiT-3 | có | Mốc so sánh **0.4809** (chỉ chạy được ở phương án c) |
| `siglip2_en` | A1 | có | Vision tower mạnh hơn có ăn điểm không |
| `siglip2_vi` | A1 | **không** | **Phép đo quý nhất** — 0 đồng embed thêm, bác bỏ/xác nhận giả thuyết research §3 |
| `pecore_en` | A2 | có | Encoder MERVIN dùng, text tower English-only nên bắt buộc dịch |
| `random`, `bm25` | — | — | Giữ nguyên, sàn tham chiếu |

  5. In thêm vào report: **tỉ lệ query bị truncation** cho mỗi kênh (A2 72 token, A1 64 token) — nếu cao thì mọi kết luận phải đọc kèm số này.
- **Mirror:** `eval/run.py:859-864` (cách cắm kênh vào `_run_all_channels`), `configs/eval/aic_pilot.yaml` (cách viết comment giải thích cờ).
- **Validate:** `pytest tests/test_eval_run.py -q`; rồi `python -m eval.run --config configs/eval/aic_candidates.yaml` (chủ dự án chạy) sinh `reports/<ts>_<sha>/`.

### T10 — Import A2 + đo · Risk: thấp (lặp lại T8+T9 cho A2)

### T11 — Báo cáo phase + đồng bộ docs · Risk: thấp

- **Action:** viết báo cáo theo đúng 5 mục CLAUDE.md §8 (Đã làm gì · Input · Output · Cách verify · Còn thiếu), **bằng tiếng Việt**, kèm bảng AIC Score các kênh + git SHA + `eval_set_hash` + số video_id + 173 query + seed + **mật độ keyframe theo batch** (kaggle plan §2) + ghi rõ report chạy trong một hay hai process.
- Cập nhật docs theo mục "Mâu thuẫn với docs" dưới đây, **trong cùng change**.

---

## Validation

```powershell
# 1) Tĩnh — chạy được trên máy trần, không GPU, không DB
ruff check . ; black --check . ; isort --check-only .
pytest -q                      # addopts đã loại 'model' và 'integration'
pytest --cov=app --cov=pipelines --cov=eval --cov-report=term-missing -q   # >= 80%

# 2) Hợp đồng encoder — trong env ứng viên (T0 quyết định env nào)
pytest -m model tests/test_siglip2_encoder.py -v
pytest -m model tests/test_pe_core_encoder.py -v
pytest -m model tests/test_beit3_encoder.py -v      # bắt buộc nếu đi phương án (c)

# 3) Chứng minh nhánh cũ KHÔNG đổi hành vi (chạy TRƯỚC khi làm gì khác)
git diff --stat app/main.py scripts/run_ingest.py app/services/search_service.py pipelines/ingest.py   # phải rỗng
python scripts/import_embed_out.py --help          # 'beit3' hiện ra là default của --encoder

# 4) Cổng parity — chặn import khi lệch
python scripts/verify_embed_parity.py --encoder siglip2 --embed-out-dir data/embed_out_siglip2
python scripts/verify_embed_parity.py --encoder pe_core --embed-out-dir data/embed_out_pecore

# 5) Import (chủ dự án tự chạy — CLAUDE.md §10)
python scripts/import_embed_out.py --encoder siglip2 --embed-out-dir data/embed_out_siglip2 --limit-videos 1   # smoke
python scripts/import_embed_out.py --encoder siglip2 --embed-out-dir data/embed_out_siglip2                    # full

# 6) Đo
python -m eval.run --config configs/eval/aic_pilot.yaml        # report cũ phải tái lập y hệt
python -m eval.run --config configs/eval/aic_candidates.yaml   # report ứng viên
```

**Bằng chứng phải in ra bằng số thật (không suy đoán):**

| Hạng mục | Kỳ vọng |
|---|---|
| keyframe/s ở smoke Kaggle, A1 và A2 | ⚠️[Cần đo] |
| Tổng giờ GPU thực tế mỗi ứng viên | ⚠️[Cần đo] — phải < ngân sách 30h/tuần cho cả hai |
| Vector trong artifact / video có output | 221.903 / 873 |
| `row_count` mỗi collection mới | 221.903 |
| min cosine text probe / image probe | ≥ 0,9999 / ≥ 0,999 (⚠️[Cần đo] độ trôi thật) |
| Tỉ lệ query bị truncation | ⚠️[Cần đo] |
| AIC Score segment-level từng kênh | ⚠️[Cần đo] |

---

## Risks

| # | Rủi ro | Khả năng | Tác động | Giảm thiểu |
|---|---|---|---|---|
| R1 | `timm==0.4.12` (BEiT-3) ⊥ `timm≥1.0.19` (PE-Core) — không sống chung một process | **Chắc chắn** | Không dựng được report một-process | **T0 chạy trước tiên**; (c) nếu vendor BEiT-3 sống với timm 1.0.x, ngược lại (a) tách env + ràng buộc cùng git SHA/eval_set_hash |
| R2 | Import lần hai **skip sạch** vì Mongo đã `complete`; hoặc `global_index` lệch nếu bỏ check | **Chắc chắn nếu không xử lý** | Hỏng âm thầm: collection rỗng hoặc DANTE đọc sai cửa sổ | T5 `--vectors-only` + kiểm khớp tập `keyframe_id` fail-loud |
| R3 | SigLIP2 text pad sai (`padding="max_length"`) ⇒ vector text sai âm thầm | Cao | Điểm tệ vì lý do sai, kết luận sai theo | Fetch docs lúc implement (CLAUDE.md §5); test tỉnh táo cosine ảnh–chữ ở T2; ghi vào `references.md` + comment config |
| R4 | Parity probe cũ vô dụng với model tải từ HF hub | Chắc chắn | Mất bằng chứng "một encoder duy nhất" | T4: pin `hf_revision` + hash `.safetensors` trong cache + **thêm image probe** (giờ đã có ảnh local) |
| R5 | Trôi số fp16 giữa GPU Kaggle và 3070 làm image probe fail | Trung bình | Chặn import nhầm | Probe chạy **fp32 hai đầu** nếu 0,999 không đạt; **không** hạ ngưỡng để cho qua |
| R6 | Vượt ngân sách 30h GPU/tuần với 2 ứng viên × 221.903 ảnh | Trung bình | Trượt lịch | Cổng quyết định sau smoke; ưu tiên A1; chia 2 session (shard theo video, resume sẵn có) |
| R7 | Session Kaggle bị cắt giữa full run | Trung bình | Mất công | Resume theo video + `os.replace` (đã có ở notebook cũ) |
| R8 | Repo có package `datasets/` che khuất `datasets` của HF trên Kaggle | Trung bình | Import lỗi khó hiểu | Append `sys.path` ở **cuối**, hoặc nạp module bằng `importlib`; ghi cảnh báo trong cell |
| R9 | `pip install -e .` fail trên kernel Kaggle (requires-python `<3.11`) | Cao | Notebook chết ở cell cài | Không install repo, chỉ `sys.path` |
| R10 | Truncation query (A2 72 tok / A1 64 tok) so với BEiT-3 64 tok | Trung bình | Điểm tụt vì độ dài, không phải vì vision tower | Đo và in tỉ lệ truncation trong report; đọc kết luận kèm số đó |
| R11 | `--reset` ở nhánh ứng viên đi nhầm vào `_reset_stores` (`:210-217`) ⇒ **xoá sạch `mongo.keyframes` + `mongo.videos`** của cả corpus | Thấp nhưng **thảm hoạ** | Mất toàn bộ metadata 873 video, phải ingest lại từ đầu | T5.5: nhánh ứng viên không bao giờ gọi `_reset_stores`; có test chứng minh `--encoder siglip2 --reset` **không** chạm hai collection Mongo đó |
| R11b | Sửa `import_embed_out.py`/`eval/run.py` làm lệch hành vi nhánh BEiT-3 cũ | Trung bình | Report cũ không tái lập được | Mặc định cờ = hành vi cũ; test cũ phải xanh **không sửa**; kiểm `git diff` 4 file production rỗng |
| R12 | Delta AIC Score nhỏ, 173 query không đủ phân biệt | Trung bình | Kết luận "cải thiện" thiếu căn cứ | Báo cáo thêm số query cải thiện / xấu đi từ `per_query.csv` (dấu hiệu nhất quán), không chỉ một con số tổng |
| R13 | Nhãn tự sinh — mọi số là tương đối | Chắc chắn | Kết luận có thể lệch khi thi thật | CLAUDE.md §1.5 / §8: ghi rõ trong report, hiệu chỉnh qua nộp bài BTC |
| R14 | Đĩa Milvus phình (~3,3 GB vector thô + index) | Thấp | Hết dung lượng | Dọn collection thua cuộc sau khi chốt; kiểm dung lượng trước T8 |

---

## Acceptance

- [ ] **T0 có kết luận viết ra**: chọn (a) hay (c), kèm version thư viện cụ thể và kết quả `pytest -m model tests/test_beit3_encoder.py`.
- [ ] **`git diff app/main.py scripts/run_ingest.py app/services/search_service.py pipelines/ingest.py` RỖNG** — bằng chứng phạm vi được tôn trọng.
- [ ] **Hành vi nhánh cũ không đổi:** `python scripts/import_embed_out.py` (không cờ) và `python -m eval.run --config configs/eval/aic_pilot.yaml` chạy ra kết quả y như trước; toàn bộ test cũ xanh **không sửa một test nào**.
- [ ] `SigLip2Encoder.embedding_dim == 1152`, `PeCoreEncoder.embedding_dim == 1280`; cả hai pass đủ 6 test hợp đồng (shape/dtype · L2-norm · dim · determinism · empty image · empty text).
- [ ] `pytest -q` xanh, coverage ≥ 80%.
- [ ] Parity gate **chặn thật**: có test chứng minh lệch `hf_revision` / lệch `weight_files_sha256` / image probe dưới ngưỡng đều raise, không có nhánh cảnh báo-rồi-chạy-tiếp.
- [ ] `--vectors-only` có test chứng minh **không ghi một dòng nào vào MongoDB**, và `global_index` trong collection mới **bằng đúng** giá trị đang có ở Mongo.
- [ ] Artifact mỗi ứng viên: **221.903 vector / 873 video**, `‖v‖₂ ≈ 1`, manifest v2 đủ field.
- [ ] `row_count` = **221.903** cho `keyframes_siglip2_so400m_p14_384` và `keyframes_pe_core_bigg14_448`.
- [ ] Một `reports/<ts>_<sha>/` chứa đủ các kênh `siglip2_en` · `siglip2_vi` · `pecore_en` (+ `semantic_live_en` nếu phương án c), kèm `git_sha` · `eval_set_hash` · `num_video_ids` · 173 query · seed · mật độ keyframe theo batch.
- [ ] **Tiêu chí "có cải thiện" viết thành số:** một ứng viên chỉ được gọi là cải thiện khi **AIC Final Score segment-level > 0.4809** trên đúng `aic_eval_v1.json` (173 query, cùng `pilot_available_v1.txt`), **và** `per_query.csv` cho thấy số query cải thiện nhiều hơn số query xấu đi. Bằng nhau hoặc thấp hơn ⇒ ghi là **không cải thiện**, giữ BEiT-3, không diễn giải lại.
- [ ] So sánh `siglip2_en` vs `siglip2_vi` được ghi rõ: nó đóng hay mở giả thuyết research §3 ("model hiểu tiếng Việt có thể là hướng sai").
- [ ] Báo cáo tiếng Việt 5 mục theo CLAUDE.md §8, mọi số là số chạy thật; chỗ chưa chạy ghi "chưa đo".
- [ ] Docs đã đồng bộ trong cùng change (mục dưới).

---

## Câu hỏi còn treo

| # | Câu hỏi | Vì sao chưa quyết được | Ảnh hưởng |
|---|---|---|---|
| Q1 | Nếu ứng viên thắng, có đổi **encoder production** (`app/main.py`) ngay trong plan này không, hay giữ BEiT-3 tới khi đo xong TRAKE/VQA? | Đây là quyết định sản phẩm, không phải kỹ thuật (CLAUDE.md §10) | Quyết định phạm vi T11 và mức độ sửa `CLAUDE.md`/`architecture.md` |
| Q2 | GPU Kaggle sẽ là **T4 hay P100**? | Kaggle phân bổ theo phiên | P100 fp16 yếu ⇒ throughput và độ trôi parity khác hẳn; ảnh hưởng cả R5 lẫn R6 |
| Q3 | Ngưỡng cosine cho **image probe** đặt bao nhiêu là đúng? | Chưa có số đo độ trôi giữa hai GPU | Đặt lỏng ⇒ mất tác dụng gác; đặt chặt ⇒ chặn nhầm. Đề xuất 0,999 + fallback fp32, cần chủ dự án xác nhận |
| Q4 | **Hạn nộp vòng sơ tuyển** còn bao lâu? (research §9 Q4 vẫn treo) | Chủ dự án chưa trả lời | Quyết định có chạy cả A1+A2 trong tuần này hay chỉ A1 |
| Q5 | Chạy full A1 và A2 **song song hai session Kaggle** hay tuần tự? | Phụ thuộc quota còn lại và số smoke | Song song rút ngắn lịch nhưng đốt quota nhanh gấp đôi |
| Q6 | Có nên đo thêm chế độ `pecore_vi` (không dịch) dù text tower English-only? | Gần như chắc chắn tệ, nhưng miễn phí và là đối chứng cho `siglip2_vi` | Chỉ tốn vài phút eval; đề xuất **có**, chờ xác nhận |
| Q7 | Nếu cả hai ứng viên thua, có thử tiếp C1 `viclip-ot` (rẻ, trùng thông số BEiT-3) hay dừng hướng đổi encoder? | Ngoài phạm vi plan này | Quyết định bước tiếp theo sau T11 |

---

## Mâu thuẫn với docs (phải sửa trong cùng change — CLAUDE.md §2)

| Doc | Chỗ mâu thuẫn | Sửa thế nào | Khi nào |
|---|---|---|---|
| `docs/research-encoder-vietnamese.md` §5 | Bảng ứng viên còn ⚠️[Cần verify] dim và license của A1/A2 | Điền **1152 / 1280**, **apache-2.0 cả hai**; ghi cách load đã xác nhận | **Ngay T2/T3** |
| `docs/research-encoder-vietnamese.md` §9 Q1 | "License PE-Core là gì?" | **Đóng**: apache-2.0, rủi ro FAIR-NC không tồn tại | Ngay |
| `docs/research-encoder-vietnamese.md` §6.1 | Đòi sàng lọc tập con ~25k keyframe trước | Chủ dự án **override** 2026-08-13: chạy full 221.903 để số so trực tiếp với 0.4809. Ghi rõ quyết định + lý do, đừng để doc mâu thuẫn với việc đang làm | Ngay |
| `docs/research-encoder-vietnamese.md` §7 | Không nhắc xung đột `timm`/`transformers` | Thêm mục §11 ghi kết quả spike T0 | T0 |
| `docs/references.md` §3.x | Chưa có mục API cho SigLIP2 / PE-Core | Thêm mục mới: chữ ký load, tham số padding/max_length, `context_length`, version thư viện tối thiểu (CLAUDE.md §5 bắt buộc fetch rồi ghi) | T2/T3 |
| `pyproject.toml:73-85` | Comment khẳng định lối thoát `--no-deps open-clip-torch` là đủ vì "ViT-B/32 never imports timm at runtime" | **Sai với PE-Core** (timm-backed tower). Sửa comment nêu rõ giới hạn của lối thoát đó và cách xử lý mới | T0/T3 |
| `.claude/plans/kaggle-offline-embed.plan.md` §9.2 | "Không có ảnh ở local" | Đã lỗi thời — ảnh đã tải về, và chính điều đó cho phép **image parity probe**. Plan cũ là tài liệu lịch sử ⇒ **không sửa nội dung**, chỉ ghi chú trong plan này (đang làm) | — |
| `CLAUDE.md` §5 (bảng thư viện) | "BEiT-3: dùng `beit3_base_itc`, timebox 1 ngày, quá thì chuyển `SigLIPEncoder`" | Nếu Q1 chốt đổi encoder production: cập nhật bảng + ghi nhận **sai lệch có chủ đích so với baseline paper** | Điều kiện, T11 |
| `docs/architecture.md` §10 | BEiT-3 gắn nhãn 📄 **[Paper]** | Đổi encoder chính = lệch khỏi paper ⇒ phải nêu lý do bằng số đo, không sửa lặng lẽ | Điều kiện, T11 |
| `docs/PRD.md` §6.3 / risk R1 | R1 mô tả SigLIP chỉ là *fallback* khi BEiT-3 khó cài | Thực tế là **nâng cấp có chủ đích, có số đo**, không phải fallback. Viết lại đoạn R1 cho khớp | Điều kiện, T11 |
| `configs/eval/aic_pilot.yaml` | Tên file "pilot" đã không còn đúng phạm vi (đã ghi chú sẵn trong file) | Không đụng trong change này; `aic_candidates.yaml` mới đặt tên đúng ngay từ đầu | — |

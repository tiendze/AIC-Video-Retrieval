# Research — Thay encoder semantic: từ BEiT-3 sang vision tower mạnh hơn

> **Trạng thái: ĐỀ XUẤT, CHƯA THỰC THI** — soạn 2026-08-12, nhánh `feature/p1-search`.
> Chưa có dòng code nào được sửa. Chưa có phép đo mới nào được chạy.
> Mọi con số trong §2 là **đo thật đã có trong repo**; mọi con số trong §4 là **trích từ paper ngoài**;
> mọi thứ gắn ⚠️ **[Cần verify]** là chưa tự tay kiểm chứng — phải fetch docs trước khi implement (CLAUDE.md §5).

**Câu hỏi khởi nguồn (chủ dự án, 2026-08-12):**
> "Semantic đang dùng BEiT-3 nhưng nó không hiểu tiếng Việt, nên đã thêm `VietAI/envit5-translation` dịch VI→EN rồi đưa cho BEiT-3, kết quả vượt trội (0.48). Nhưng hybrid RRF vẫn tệ hơn single channel. Tìm model hiểu tiếng Việt tốt nhất hiện tại, ứng cử viên để đột phá điểm."

**Kết luận ngắn:** câu hỏi tách được thành hai vấn đề **độc lập**, và câu trả lời cho vấn đề chính
*không phải* "model hiểu tiếng Việt tốt nhất".

---

## 1. Hai vấn đề độc lập — đừng gộp

| # | Vấn đề | Nguyên nhân | Cách chữa | Chi phí |
|---|---|---|---|---|
| A | Hybrid RRF < single channel | Kênh OCR quá yếu + fuse theo rank + weight 1:1 | Chỉnh weight/độ sâu candidate, hoặc bỏ hẳn | Rẻ, không cần re-embed |
| B | Trần điểm của semantic ở 0.48 | Nghi vision tower (BEiT-3 *base*, 224px, 2022) | Đổi encoder | Đắt, re-embed 221.903 keyframe |

Đổi encoder **không** chữa được (A). Thực tế nó làm (A) **tệ hơn**: encoder mạnh hơn ⇒ chênh lệch
semantic-vs-OCR giãn ra ⇒ RRF weight 1:1 kéo xuống mạnh hơn nữa.

---

## 2. Bằng chứng đã đo trong repo (không phải suy đoán)

### 2.1 Số đo hybrid

Nguồn: `.claude/plans/p1-search.plan.md:90-97`, mirror ở `configs/eval/aic_pilot.yaml:33-36`.
Đo 2026-08-12, corpus đầy đủ 873 video, `split=all`, docker-compose thật + checkpoint BEiT-3 thật.
Metric: **AIC Final Score (segment-level)**, 173 query.

| Run | semantic | ocr | hybrid |
|---|---|---|---|
| 1 — semantic **không** dịch | 0.0012 | 0.0254 | 0.0162 |
| 2 — semantic **dịch** VI→EN | **0.4809** | 0.0254 | 0.4428 |

Số phụ, hiệu ứng dịch cô lập trên riêng kênh semantic (`app/services/query_translator.py:8-9`,
report khác, cùng tập 173 query / cùng collection): **0.0451 (VI) → 0.6208 (EN)**.

> ⚠️ Hai cặp số VI/EN này không khớp nhau (0.0012→0.4809 vs 0.0451→0.6208). Chúng đến từ **hai
> report khác nhau** — nhiều khả năng khác `split` hoặc khác cấp độ metric (video-level vs
> segment-level). **[Cần verify]** trước khi trích dẫn ra ngoài: mở lại hai thư mục
> `reports/<timestamp>_<git_sha>/metrics.json` tương ứng và ghi rõ cấu hình của từng cái.

### 2.2 Vì sao hybrid buộc phải tệ hơn — đọc từ code

`app/services/search_service.py:206-233`:

```
RRF(d) = Σ_c  weight_c / (k + rank_c(d))
```

- **Fuse theo rank, không theo score magnitude.** Một hit rác của OCR đứng hạng 1 trong list của
  nó đóng góp `1/(60+1)` — **đúng bằng** hit tốt nhất của semantic. Với chất lượng chênh **~19 lần**
  (0.4809 vs 0.0254), kết quả này là tất yếu về mặt toán học, không phải bug.
- **Hai kênh cùng `top_k`.** `_search_hybrid` (`search_service.py:192-203`) truyền nguyên `top_k`
  cho cả hai nhánh ⇒ kênh yếu vẫn được cấp đủ `top_k` phiếu bầu.
- **Weight mặc định 1.0/1.0.** `configs/search/search.yaml`: `hybrid_rrf_k: 60`,
  `hybrid_semantic_weight: 1.0`, `hybrid_ocr_weight: 1.0`. Chưa từng thử hạ `hybrid_ocr_weight`
  — P1.3 nhảy thẳng từ "đo thấy tệ" sang "bỏ F4 khỏi eval mặc định"
  (`configs/eval/aic_pilot.yaml:43` → `modes: [semantic, ocr]`).
- Document vắng mặt ở một kênh ghi 0 cho kênh đó (`scores.get(hit.keyframe_id, 0.0)`,
  `search_service.py:218`) — **không** phạt thêm. Nên lực kéo xuống đến từ hit yếu của OCR chiếm
  slot hạng cao, chứ không phải semantic bị trừ điểm vì OCR thiếu.

Chẩn đoán này đã được ghi sẵn trong docstring module (`search_service.py:33-39`).

### 2.3 Quy mô phải re-embed

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Keyframe | **221.903** | `.claude/plans/kaggle-offline-embed.plan.md:32`, `pipelines/import_embed_out.py:6` |
| Video `complete` | **873** | `.claude/plans/p0-foundation.plan.md:15` |
| Query trong eval set | **173** | `configs/eval/aic_eval_v1.json` |
| Collection hiện tại | `keyframes_beit3_base_itc_patch16_224` | `app/stores/milvus_client.py:56` |

Đổi `model_id` ⇒ Milvus tự tạo collection **mới**, collection cũ nguyên vẹn (invariant CLAUDE.md §3).
Nghĩa là: **không tái sử dụng được một vector nào**, phải embed lại toàn bộ 221.903 keyframe.

---

## 3. Vì sao "model hiểu tiếng Việt" có thể là hướng sai

Phía text **đã được giải quyết** bằng envit5: 0.4809. Phần chưa được kiểm chứng là **vision tower**:
`beit3_base_itc_patch16_224` — bản *base* (~220M), **224px**, kiến trúc 2022, `embedding_dim = 768`
(`app/encoders/beit3.py:32-33`).

Hai rủi ro của hướng "native Vietnamese":

1. **Tháp text đa ngữ thường yếu hơn tháp tiếng Anh của cùng model** — dung lượng bị chia cho
   89–109 ngôn ngữ. Bỏ envit5 để lấy "native VI" có thể **đổi 0.48 lấy số thấp hơn**.
2. **Không đội mạnh nào ở AIC 2025 đặt cược vào đó** (xem §4). Họ đặt cược vào vision tower to hơn
   + độ phân giải cao hơn.

⚠️ Đây là **giả thuyết**, chưa đo. Cách rẻ nhất để bác bỏ/xác nhận: SigLIP 2 là đa ngữ sẵn, nên sau
khi embed **một lần**, có thể đo cả hai chế độ (dịch / không dịch) mà **không tốn thêm chi phí embed
nào** — vì chỉ phía query đổi, vector ảnh dùng chung (xem §6.3).

---

## 4. Bằng chứng từ chính sân chơi AIC HCMC 2025

| Hệ thống | Vision-text encoder | Xử lý tiếng Việt | Kết quả công bố |
|---|---|---|---|
| **MERVIN** ([arXiv 2605.16120](https://arxiv.org/html/2605.16120)) | **PE-Core-bigG-14-448** (Meta Perception Encoder) | Không dịch; dùng `dangvantuan/vietnamese-embedding` cho **kênh text riêng** (transcript/summary), không phải kênh ảnh–chữ | **79/88 điểm** vòng sơ tuyển, metric "Mean of Top-k R-Scores", k∈{1,5,20,50,100} |
| **Vortex** ([arXiv 2606.19682](https://arxiv.org/html/2606.19682)) | **CLIP DFN5B + SigLIP2**, fuse **RRF k=60**, N=2 kênh | Không nêu | Tiến trình sơ tuyển 20.6 → 27.8 → 31.2 |
| **U-CESE** ([arXiv 2605.23274](https://arxiv.org/html/2605.23274v1)) | MobileCLIP | Không nêu; dùng Vintern-1B để caption ảnh | Không công bố số định lượng |
| **Baseline của repo này** ([arXiv 2512.13169](https://arxiv.org/abs/2512.13169)) | BEiT-3 | — | **Không công bố metric nào** |

Ba quan sát:

- Hai hệ thống mạnh nhất dùng vision tower **lớn hơn nhiều** và **độ phân giải cao hơn nhiều**
  (448px vs 224px của ta).
- **Vortex fuse hai kênh cùng loại** (CLIP + SigLIP2, đều là semantic ảnh–chữ, chất lượng tương
  đương) — khác hẳn ta đang fuse semantic mạnh với OCR yếu. Đây là gợi ý rằng RRF *đúng cách* là
  fuse các kênh **ngang tài**, và có thể là hướng cứu F4 sau này.
- MERVIN dùng model tiếng Việt cho **kênh text**, không phải kênh ảnh–chữ. Đúng như phân tích §3.

---

## 5. Bảng ứng viên

Ràng buộc: RTX 3070 8GB tại chỗ · Kaggle/Colab dùng thoải mái (chủ dự án xác nhận 2026-08-12) ·
221.903 keyframe · ưu tiên license Apache-2.0.

### Hạng A — nâng vision tower, giữ envit5 (**ĐÃ CHỌN, 2 cược**)

| | Model | Params | Res | Dim | License | Tiếng Việt |
|---|---|---|---|---|---|---|
| **A1** | `google/siglip2-so400m-patch14-384` | ~400M (vision) | 384 | ⚠️ **[Cần verify]** (nghi 1152) | ⚠️ **[Cần verify]** (nghi Apache-2.0) | **Native, 109 ngôn ngữ**, tokenizer Gemma |
| **A2** | `timm/PE-Core-bigG-14-448` (Meta Perception Encoder, remapped for OpenCLIP) | ~1.8B (vision tower `vit_pe_core_gigantic_patch14_448`, timm-backed) | 448 | **1280** ✅ verified 2026-08-14 | **apache-2.0** ✅ verified 2026-08-14, không FAIR-NC | Text tower **chỉ tiếng Anh** (custom CLIP BPE, `context_length=72`, vocab 49408) ⇒ **bắt buộc giữ envit5** |

- **A1 là cược an toàn**: tỷ lệ chất lượng/chi phí tốt nhất, Apache-2.0 (nếu verify đúng),
  và đa ngữ nên test được cả hai chế độ query miễn phí.
- **A2 là cược đánh cược**: chính là encoder MERVIN dùng để đạt 79/88. Nhưng nặng nhất, và
  **license là rủi ro thật** — nếu là non-commercial/FAIR-NC thì phải đối chiếu điều lệ BTC.

> ✅ **Đóng 2026-08-14** (encoder-swap plan §T3/T7): checkpoint chính xác là
> `timm/PE-Core-bigG-14-448` (bản OpenCLIP-remapped của Meta, không phải repo gốc
> `facebookresearch/perception_models`), fetch trực tiếp
> `https://huggingface.co/api/models/timm/PE-Core-bigG-14-448` +
> `.../raw/main/open_clip_config.json` (không đoán/tin lại nghiên cứu cũ): license
> **apache-2.0**, `embedding_dim=1280`, load qua
> `open_clip.create_model_and_transforms("hf-hub:timm/PE-Core-bigG-14-448")`.
>
> ⚠️ Phát hiện thêm ngoài phạm vi câu hỏi ban đầu: `open_clip==3.3.0`'s hf-hub loader
> **không cho pin `revision`** ở thời điểm load (verify trực tiếp source
> `src/open_clip/factory.py`/`pretrained.py`) — khác `SigLip2Encoder`
> (`AutoModel.from_pretrained(..., revision=...)` ghim được). `PeCoreEncoder`
> (`app/encoders/pe_core.py`) vì vậy không nhận tham số `hf_revision`; cổng an toàn
> chống drift dựa vào `weight_files_sha256` + image/text probe (T4), không phải pin
> tại thời điểm load. Xem chi tiết trong docstring của `app/encoders/pe_core.py`.

### Hạng B — native Vietnamese, bỏ envit5 (chưa chọn)

| | Model | Ghi chú |
|---|---|---|
| B1 | `jinaai/jina-clip-v2` | 0.9B, [89 ngôn ngữ](https://jina.ai/news/jina-clip-v2-multilingual-multimodal-embeddings-for-text-and-images/), **512px**, Matryoshka cắt 1024→64 dim (tiết kiệm Milvus). ⚠️ weights **CC-BY-NC** — chủ dự án chọn ưu tiên Apache-2.0 ⇒ **tạm gác** |
| B2 | [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | 2B, 30+ ngôn ngữ có tiếng Việt, MMEB-V2 **73.2**, dim tới 2048. Nhưng là embedder nền **VLM**: mỗi ảnh chạy qua LLM ⇒ chi phí/ảnh cao hơn CLIP tower cả bậc. **Rủi ro lịch trình cao** với 221.903 ảnh |

### Hạng C — thí nghiệm rẻ (chưa chọn)

| | Model | Ghi chú |
|---|---|---|
| C1 | [`minhnguyent546/viclip-ot`](https://arxiv.org/html/2602.22678) | VLM nền tảng **đầu tiên cho ảnh–chữ tiếng Việt**. **221M, 768-dim, 224px — trùng khít thông số BEiT-3**, nên là phép A/B rẻ nhất có thể (không phải đổi gì ngoài class encoder) |

**Đánh giá thẳng về C1 — không kỳ vọng đột phá.** Train chỉ trên **13.100 ảnh / 61.241 caption**
(UIT-OpenViIC), backbone DINOv3 ViT-B/16. Con số zero-shot Crossmodal-3600 họ báo (56.85, so với
jina-clip-v2 53.91 và Qwen3-VL-Embedding-2B 55.40) đo trên benchmark **caption ngắn, phân bố gần y
hệt dữ liệu train của họ** — không đại diện cho keyframe tin tức + query mô tả dài của AIC. Nếu có
slot rảnh thì test, nhưng đừng đặt cược chính vào nó.

---

## 6. Giao thức đo — bắt buộc đọc trước khi chạy

### 6.1 Không được so số tập con với 0.4809

Re-embed đủ 221.903 keyframe cho **mỗi** ứng viên là lãng phí. Sàng lọc trên tập con trước:

1. Chọn corpus con = **toàn bộ video có ground-truth** trong `aic_eval_v1.json` + ~100 video nhiễu
   ngẫu nhiên (seed cố định, ghi vào config) ≈ **~25k keyframe**, rẻ hơn ~10 lần.
2. **Chạy lại baseline BEiT-3 trên đúng tập con đó** — vector đã có sẵn trong Milvus, chỉ cần lọc,
   gần như miễn phí. Không có mốc này thì mọi so sánh vô nghĩa.
3. Đo ứng viên trên tập con → chỉ ứng viên **thắng BEiT-3 trên tập con** mới được re-embed toàn kho.
4. ⚠️ Số trên tập con **không so được** với 0.4809 (mẫu số distractor nhỏ hơn ⇒ recall bị thổi lên).
   Chỉ dùng để **xếp hạng tương đối**. Phải ghi rõ điều này trong mọi report (CLAUDE.md §8).

### 6.2 Kỷ luật báo cáo (CLAUDE.md §8)

Mỗi lần đo phải kèm: git SHA · model checkpoint · version index · `eval_set_version` + hash bộ nhãn ·
số `video_id` duy nhất đánh giá được · số truy vấn · seed. Thêm cho lần này: **kích thước corpus con**
và **danh sách video nhiễu**, vì nếu không thì hai lần đo không so được với nhau.

### 6.3 Mẹo: SigLIP 2 test "dịch vs không dịch" tốn 0 đồng embed

Chỉ **phía query** đổi (`_search_semantic` gọi translator hay không). Vector ảnh dùng chung. Nên sau
một lần embed A1, chạy eval hai lần với `query_translator` bật/tắt là ra ngay câu trả lời cho §3.
Đây là phép đo có giá trị nhất trong toàn bộ kế hoạch này, và nó gần như miễn phí.

### 6.4 Đo throughput trước khi treo job dài

**Không ước lượng bằng cảm tính.** Trước khi treo Kaggle job cho 221.903 keyframe: chạy timed pilot
trên **1.000 keyframe**, lấy ảnh/giây thật, rồi nhân lên. Ghi con số vào plan. Tôi cố tình **không**
đưa ước lượng giờ ở đây vì sẽ là số bịa (CLAUDE.md §10).

### 6.5 Dung lượng Milvus (tính được, không phải đoán)

`221.903 × dim × 4 bytes` (float32):

| dim | Dung lượng |
|---|---|
| 768 (BEiT-3 hiện tại) | ~682 MB |
| 1152 (nếu A1 đúng như nghi) | ~1,02 GB |
| 1280 | ~1,14 GB |

Mỗi ứng viên là **một collection riêng cùng tồn tại**. Ba collection ≈ 2,8 GB — chấp nhận được,
nhưng phải dọn collection thua cuộc sau khi chốt.

---

## 7. Việc phải làm trong repo (chưa làm)

### 7.1 Gom factory encoder — bắt buộc làm trước

Hiện có **3 chỗ hardcode `BEiT3Encoder(...)`**, mỗi chỗ tự load `configs/model/beit3.yaml`:

| File | Hàm |
|---|---|
| `app/main.py:34-42` | `_build_encoder` (production, gắn vào `app.state.encoder` qua lifespan `:85`) |
| `scripts/run_ingest.py:113-122` | `_build_encoder` |
| `eval/run.py:539-547` | `_build_encoder` |

Không gom lại thì mỗi lần đổi ứng viên phải sửa 3 chỗ, và **rủi ro lệch encoder giữa ingest và query**
— vi phạm invariant số 1 của CLAUDE.md §3 ("một encoder duy nhất"). Phải có **một factory chọn
encoder theo config**.

### 7.2 Encoder mới phải thoả `MultimodalEncoder`

`app/encoders/base.py:23-45` — Protocol structural, không cần kế thừa. Contract:
`(N, embedding_dim)` · `float32` · **L2-normalize bên trong encoder** · deterministic.
Test hợp đồng có sẵn: `tests/test_beit3_encoder.py` — encoder mới phải pass bộ tương đương.

**Lưu ý:** preprocessing nằm gọn **bên trong** `BEiT3Encoder` (`beit3.py:54-62`: resize 224 BICUBIC,
normalize mean/std 0.5, `XLMRobertaTokenizer`, `_MAX_TEXT_TOKENS = 64`). Không có module
`preprocessing.py` dùng chung ⇒ encoder mới phải tự viết resize/normalize/tokenize của nó. Đây là
**đúng thiết kế** (seam nằm ở class encoder), không phải nợ kỹ thuật cần dọn.

### 7.3 Bỏ dịch = đổi wiring, không đụng SearchService

`SearchService.__init__` đã nhận `query_translator: QueryTranslator | None = None`
(`search_service.py:91`), check `if self._query_translator is not None` (`:124`). Truyền `None` là
xong — **không sửa dòng nào** trong `SearchService`.

### 7.4 Đường embed offline đã có sẵn

`notebooks/kaggle_embed_keyframes.ipynb` → `data/embed_out/vectors/<video_id>.{npy,jsonl}` →
`scripts/import_embed_out.py` / `pipelines/import_embed_out.py` → parity check
`pipelines/verify_embed_parity.py`. Đây là đường đã dùng cho 221.903 keyframe hiện tại, dùng lại
nguyên vẹn. **Nhớ chạy `verify_embed_parity` cho encoder mới** — nó chính là cái bắt lỗi
"vector ingest ≠ vector query".

### 7.5 Không cần làm lại

`configs/eval/pilot_available_v1.txt` và trạng thái `ingest_status` trong Mongo là **model-agnostic**
⇒ giữ nguyên. Nhưng mọi report `eval/run.py` cũ **không so được** với encoder mới cho tới khi có
baseline mới.

---

## 8. Track song song — chữa hybrid (rẻ, độc lập, chưa làm)

Không cần chờ encoder. Ba việc theo thứ tự tăng dần công sức:

1. **Hạ `hybrid_ocr_weight`** xuống 0.1–0.2 trong `configs/search/search.yaml`, chạy lại F4.
   Rẻ nhất, chưa từng thử.
2. **Cắt độ sâu candidate của OCR** — hiện cả hai kênh nhận cùng `top_k`
   (`search_service.py:193-194`). Cho OCR ít slot hơn.
3. **Fuse hai kênh ngang tài** thay vì mạnh–yếu, giống Vortex (CLIP + SigLIP2). Nếu §7 xong và có
   2 encoder tốt, đây là cách F4 có thể thắng thật.

CLAUDE.md §4 nói rõ: "Hybrid không được kém hơn kênh đơn tốt nhất. Kém thì chỉnh trọng số **hoặc bỏ
hẳn**." P1.3 đã chọn "bỏ hẳn" mà **chưa thử chỉnh trọng số** — bước (1) là món nợ cần trả.

---

## 9. Câu hỏi còn treo

| # | Câu hỏi | Ảnh hưởng |
|---|---|---|
| ~~Q1~~ | ~~License của PE-Core-bigG-14-448 là gì?~~ | **Đóng 2026-08-14**: `apache-2.0`, không FAIR-NC — xem §5 |
| Q2 | `embedding_dim` thật của A1/A2? | A2 đóng 2026-08-14: **1280**. A1 (SigLIP 2) vẫn ghi "1152" trong bảng §5 nhưng KHÔNG có nguồn fetch nào trong doc này xác nhận lại — người triển khai A1 nên đối chiếu, ngoài phạm vi thay đổi này |
| Q3 | Hai cặp số VI/EN ở §2.1 vì sao lệch? | Ảnh hưởng con số nào được dùng làm baseline chính thức |
| Q4 | Hạn nộp vòng sơ tuyển còn bao lâu? | Chủ dự án chưa trả lời trực tiếp; quyết định có nên chạy cả A1+A2 hay chỉ A1 |
| Q5 | Query thi thật dài/phức tạp cỡ nào so với 173 caption tự sinh? | Nếu lệch nhiều, mọi số ở đây lệch theo (CLAUDE.md §1.5) |

---

## 10. Quyết định đã chốt (chủ dự án, 2026-08-12)

- ✅ Compute: Kaggle/Colab **dùng thoải mái** — mở đường cho model lớn.
- ✅ Cách chạy: **treo embedding trên Kaggle, local tiếp tục bước tiếp theo song song**.
- ✅ Đặt cược **cả 2 model Hạng A**: A1 SigLIP 2 so400m-384 + A2 PE-Core-bigG-14-448.
- ✅ License: ưu tiên **Apache-2.0** ⇒ jina-clip-v2 (CC-BY-NC) **tạm gác**.
- ⏸️ Chưa thực thi. Tài liệu này là đầu vào cho một plan riêng.

---

## Nguồn

- [MERVIN: A Unified Framework for Multimodal Event Retrieval in Vietnamese News Videos](https://arxiv.org/html/2605.16120)
- [Vortex: Multi-Modal Fusion System for Intelligent Video Retrieval](https://arxiv.org/html/2606.19682)
- [U-CESE: Unified Clip-based Event Search Engine for AI Challenge HCMC 2025](https://arxiv.org/html/2605.23274v1)
- [Integrated Semantic and Temporal Alignment for Interactive Video Retrieval (baseline của repo)](https://arxiv.org/abs/2512.13169)
- [ViCLIP-OT: The First Foundation Vision-Language Model for Vietnamese Image–Text Retrieval](https://arxiv.org/html/2602.22678)
- [SigLIP 2: Multilingual Vision-Language Encoders (paper)](https://arxiv.org/pdf/2502.14786)
- [SigLIP 2: A better multilingual vision language encoder (HF blog)](https://huggingface.co/blog/siglip2)
- [jina-clip-v2: Multilingual Multimodal Embeddings for Text and Images](https://jina.ai/news/jina-clip-v2-multilingual-multimodal-embeddings-for-text-and-images/)
- [jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval](https://arxiv.org/abs/2506.18902)
- [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)

# Plan: Đưa FG-CLIP Phrase-to-Patch (KIS/PPLR) vào task TRAKE

**Ngày lập:** 2026-08-27 · **Branch:** `PPLR` · **Base:** `76be1e7 Update FG-CLIP Phrase-to-Patch Local Retrieval Ver1`
**Phạm vi:** chỉ TRAKE. **KHÔNG dùng `eval/`** (chủ dự án chốt 2026-08-27) ⇒ mọi verify là unit test + đối chứng thủ công trên truy vấn thật.
**Trạng thái:** 🟢 **ĐÃ TRIỂN KHAI (2026-08-28)** T1+T2+T3 trên `PPLR`, chưa merge `main`, chưa test trên corpus dense đầy đủ (còn thiếu 79 video — xem §"Trạng thái dữ liệu"). Ghi chú triển khai:

- `SearchService` có thêm 2 property công khai (`local_embedding_store`, `kis_config`) để `TrakeService` dùng chung, không chỉ `local_embedding_store` như bản nháp đầu.
- `KISLocalEmbeddingStore._VideoLocalFiles` đổi tên công khai thành `VideoLocalFiles` (không có nơi nào khác phụ thuộc tên cũ) để `TrakeService` type-hint được.
- Batching theo `local_score_batch_size` (mặc định 256) áp dụng cho cả video đọc trọn — không phải video chia theo pool, mà là chia MaxSim theo lô hàng bên trong video đó (tránh dồn GPU với video dài).
- Test mới: 9 test trong `tests/test_trake_service.py` (bao gồm test số học chốt trực tiếp bất biến "local_z mean ≈ 0 mỗi video"), 3 test trong `tests/test_trake_api.py`, 3 test trong `tests/test_fg_clip_encoder.py` (cần `torch`, chưa chạy được trong môi trường agent). Toàn bộ 188 test Python liên quan + 193 test frontend đã chạy PASS trong môi trường agent (đã cài thêm `pydantic`/`fastapi`/`pymongo`/`pymilvus`/`elasticsearch`/`pytest-asyncio`/`python-multipart` — nhẹ, không phải `torch`).
- Phát hiện lỗi có sẵn **không liên quan** đến thay đổi này: `tests/test_trake_embedding_cache.py::test_trake_request_uses_loaded_cache_without_per_video_milvus_fetching` gọi `TrakeService(...)` thiếu `search_service` — đã tồn tại trước khi đụng vào code này (xác nhận bằng `git stash`), không sửa trong change này.
- F9 (cảnh báo cắt 77 token) đã làm: `count_text_tokens`/`max_text_tokens` trên `FGClipEncoder`, cảnh báo qua `TrakeEventDiagnostics.text_truncated` + badge UI. **Chưa** làm phần "tách câu tự động" (đó là nhánh `main`'s `split_into_sentences`, xem F8/F9 — vẫn treo).
- Chưa làm: R4 (merge `PPLR`↔`main`), đối chứng A/B thủ công trên truy vấn thật (§5.4 — cần `torch` + Milvus/ES/Mongo chạy thật, môi trường agent không có).

**Cập nhật 2026-08-28 (đợt việc riêng, xem plan `oke-n-thi-u-t-i-steady-llama.md`):** B8 (`score_videos`) có hai lỗi độc lập với đợt phrase-to-patch này, cả hai đều không liên quan local rerank (B2b) — B2b không đụng `score_videos`:

1. `TrakeService._rank_evidence` là tổng không chặn trên qua các event, trong khi `coverage` mà nó cộng chung là phân số ∈[0,1] ⇒ với truy vấn 5-7 event, `evidence` áp đảo `coverage` 2.5-3.5×. Đã sửa: chia `_rank_evidence` cho `len(events)`.
2. `smooth_max` (`logsumexp` của top 3 chuỗi mỗi video) cũng không chặn trên — video có nhiều chuỗi *khác nhau thật sự* (không phải trùng lặp gần, `sequence_nms` chỉ gộp lệch <10 frame) cùng điểm được cộng bonus miễn phí tới `log(3)≈1,10`, có thể đè video chỉ có đúng một chuỗi sạch. Phát hiện từ báo cáo thực tế "đáp án đúng nằm phía dưới" sau khi đã sửa lỗi 1. Đây là hành vi **có chủ đích, có test khẳng định** trước đó (`test_several_good_sequences_outrank_one_lucky_one`), không phải oversight như lỗi 1. Đã sửa: thêm `video_smooth_max_bonus_cap: 0.3` chặn trần bonus.

`event_text_top_k` cũng hạ 1000→300 trong đợt này.

---

## 1. Tóm tắt

`PPLR` thêm đúng **một** năng lực cho KIS: rerank Stage 1 bằng **MaxSim phrase-to-patch** trên đặc trưng FG-CLIP dense 8x8 đọc sẵn từ đĩa (`app/services/kis_local_retrieval.py`, 436 dòng + `configs/search/search.yaml: kis.local_retrieval`). Operator gõ thêm các *local phrase* (`"red seasoning packet"`) bên dưới mỗi text query; mỗi keyframe ứng viên được chấm `mean_phrase(max_patch(cos))` trên 64 patch, rồi pool Stage 1 được sắp lại theo điểm đó.

TRAKE **đã** dùng KIS ở bước B1 (`TrakeService._retrieve_one_pool` gọi thẳng `SearchService.retrieve_kis_candidates`), nên về nguyên tắc "TRAKE thừa hưởng cái gì làm KIS giỏi". Nhưng thực tế hôm nay **TRAKE không hưởng gì cả**, vì hai lý do độc lập nhau (F1, F2 dưới).

Plan này gồm 3 bước: **T1** nối đường truyền local phrase vào B1; **T2** — phần ăn điểm thật — đưa MaxSim vào chỗ TRAKE chọn *frame nào trả lời event*; **T3** diagnostics/UI.

---

## 2. Phát hiện khi đọc code (đã verify trực tiếp trên working tree)

### F1 — 🔴 TRAKE không truyền `local_phrases_by_query`, tính năng mới im lặng hoàn toàn

`app/services/trake_service.py: _retrieve_one_pool` liệt kê đủ Text/OCR/Title/ASR nhưng **không có** tham số `local_phrases_by_query` mà `PPLR` vừa thêm vào `retrieve_kis_candidates`. Chuỗi request đứt ở mọi tầng phía trên:

- `app/schemas/request/trake.py: TrakeEventRequest` — không có field local phrase.
- `app/api/trake.py: _to_event_query` — không map.
- `frontend/src/api/types.ts: TrakeEventPayload` — không có field (KIS đã có `local_phrases_by_query?: string[][]`).
- `frontend/src/components/EventSequenceInput.tsx` — 4 kênh cứng trong `CHANNELS`, không có ô phrase.

⇒ Sửa T1 là cơ học, rủi ro thấp.

### F2 — 🔴 Chỗ TRAKE ăn điểm KHÔNG nằm ở KIS pool, mà ở B2 backfill — nơi local rerank không chạm tới

Đây là phát hiện quan trọng nhất của plan này.

Trong TRAKE, KIS chỉ quyết định **video nào** thành ứng viên và cấp `kis_rank` làm evidence. **Frame nào trả lời event** lại do `TrakeService._local_top_indices` quyết: nó `argpartition` trên ma trận cosine **global** từ `TrakeEmbeddingCache` (`_normalized_scores_for_candidates`), tức vẫn là vector FG-CLIP toàn ảnh — đúng thứ mà phrase-to-patch sinh ra để sửa.

Mà TRAKE chấm theo cửa sổ **< 10 frame tuyệt đối** (CLAUDE.md §1.2): chọn đúng video mà lệch shot là mất điểm. ⇒ **Nếu chỉ làm T1, kỳ vọng cải thiện cho điểm TRAKE gần như bằng 0**; nó chỉ đổi tập video ứng viên.

### F3 — 🟠 Rủi ro thầm: video thiếu dense feature bị đẩy khỏi pool

`SearchService._local_rerank_stage1_results` đặt `0 if has_local_score else 1` ở đầu sort key ⇒ mọi hit **không có** local embedding rơi xuống **sau toàn bộ** hit có điểm, rồi `results = results[:text_top_k]` cắt. Với TRAKE (`event_text_top_k: 1000`, union còn thêm OCR/Title) phần bị cắt chính là nhóm thiếu feature.

Đo thật hôm nay:

```
data/fgclip_dense_8x8: 873 thư mục video, chỉ 794 có embeddings.npy
Thiếu 79 video — L21: 29, L22: 31, L25: 19
```

(Phần còn lại đang nằm trong `fgclip_dense_8x8-20260827T135900Z-1-003.zip` chưa giải nén hết vào `data/`.)

⇒ Bật local phrase khi corpus dense chưa đủ = **âm thầm loại ~9% video khỏi ứng viên**, đúng loại lỗi CLAUDE.md §3 gọi là "hỏng âm thầm". Phải preflight trước khi bật mặc định.

### F4 — 🟠 Ngân sách I/O: 64 KB mỗi frame, bắt buộc chặn số video rerank

`metadata.json`: `dtype float16`, `64 x 512` ⇒ **65 536 byte/keyframe**. Corpus đầy đủ 221 903 keyframe ≈ **14,2 GB**. Store dùng `np.load(mmap_mode="r")` nên đọc theo hàng, nhưng random-read 64 KB/frame vẫn là chi phí thật.

Ước lượng cho T2 với default hiện tại (`max_candidate_videos: 500`, `bucket_top_k: 30`, 5 event):
`500 × 5 × 30 × 64 KB ≈ 4,8 GB I/O mỗi request` ⇒ **không chấp nhận được cho hệ interactive**.

Thêm nữa `kis.local_retrieval.mmap_cache_size: 64` < 500 ⇒ LRU thrash (đóng/mở mmap liên tục). Cache chỉ giữ handle + dict `frame_idx→row` (~700 int/video) nên nâng lên 512 gần như miễn phí RAM.

### F5 — 🟡 TRAKE 1 ô text/event, KIS align phrase theo index text query

`normalize_local_phrase_groups(text_queries, local_phrases_by_query)` ghép theo **chỉ số**. UI TRAKE mỗi event chỉ có 1 ô text (`TrakeEventDraft.text`), nhưng backend `TrakeEventRequest` vẫn cho `text_queries` nhiều phần tử (prompt-ensemble ở `_encode_events`).

**Quyết định đề xuất:** một event có **một** danh sách local phrase; khi gửi sang KIS thì nhân bản cho mọi text query của event: `local_phrases_by_query = [phrases] * len(text_queries)`. Lý do: phrase mô tả *vùng ảnh của khoảnh khắc*, không phụ thuộc cách diễn đạt câu; KIS lấy `mean` qua các group nên nhân bản không làm lệch thang điểm.

**Ràng buộc phải nêu trên UI:** event **không có text query** thì phrase bị bỏ qua (KIS yêu cầu `text_queries` không rỗng) ⇒ disable ô phrase + hint, đừng để operator gõ vào chỗ vô tác dụng.

### F6 — 🟡 `TrakeFrameRefiner` (B6) là dead code

`app/services/trake_frame_refiner.py` (301 dòng) + `tests/test_trake_frame_refiner.py` tồn tại nhưng **không nơi nào import** ngoài test, và `configs/search/trake.yaml` **không có** section `frame_refine` mà `FrameRefineConfig.from_mapping` đòi. Báo theo CLAUDE.md §10, **không xoá**. Nó giải bài toán khác (decode video xuống dưới lưới keyframe) — ngoài phạm vi plan này, nhưng nếu sau này bật thì T2 là bước ngay trước nó trong cùng chuỗi.

### F7 — 🟡 `main` đã revert pipeline TRAKE, và có việc translator mà `PPLR` chưa có

`git log PPLR..main`: `8986cbd Revert "feat: TRAKE candidate-set chaining pipeline"` cùng nhánh `feat/llm-query-translator` (`904fe5c`, `eee71af`: glossary + sentence-split decomposition). `git log main..PPLR` chỉ có đúng 1 commit KIS.

⇒ `PPLR` là nơi duy nhất còn pipeline TRAKE. Merge về `main` sau này phải xử lý revert đó **và** hợp nhất translator. Plan này làm việc trên `PPLR`.

### F8 — 🔴 Bẫy merge: `split_into_sentences` của `main` làm **lệch index** `local_phrases_by_query` của `PPLR`

`main:app/services/search_service.py:253` mở đầu `retrieve_kis_candidates` bằng:

```python
text_query_sentences = split_into_sentences(text_query) if text_query else []
text_queries = _normalize_stage1_text_queries(None, [*text_query_sentences, *(text_queries or ())])
```

Tức `main` **chèn thêm phần tử vào đầu** `text_queries`. Còn `PPLR` ghép phrase theo **chỉ số** (`normalize_local_phrase_groups(text_queries, local_phrases_by_query)`). Merge thẳng hai nhánh ⇒ phrase group của query 0 rơi vào câu tách đầu tiên, các group sau lệch hết — **không crash, chỉ chấm sai vùng ảnh**.

⇒ Khi merge phải quyết định lại: hoặc phrase gắn theo **query gốc** rồi nhân bản cho mọi câu tách ra từ nó, hoặc bỏ align-theo-index. Ghi vào R4.

### F9 — 🟠 Text của event TRAKE bị cắt 77 token âm thầm, và bản vá của `main` **không chạm tới TRAKE**

`app/encoders/fg_clip.py:231-233` — `max_length=77, truncation=True`, không cảnh báo. Commit `eee71af` đo được **9/25 truy vấn SOTUYEN1 vượt cap**, và với `query-p1-16-trake.txt` phần bị cắt đúng là **event E3, một trong ba keypoint bắt buộc**.

Nhưng bản vá đó chỉ chạy khi `text_query` khác `None`, trong khi `TrakeService._retrieve_one_pool` gọi `text_query=None, text_queries=[...]` ⇒ **kể cả trên `main`, event TRAKE vẫn không được tách câu**. Event mô tả dài vẫn mất đuôi.

⇒ Rẻ và độc lập với local retrieval: cho event text đi qua `split_into_sentences` (sau khi merge), hoặc tối thiểu **log cảnh báo khi event vượt cap** thay vì cắt im lặng.

---

## 3. Nguyên tắc thiết kế cho T2 (chỗ dễ làm hỏng nhất)

Vấn đề: nếu cộng thẳng điểm local vào `CandidateFrame.score` mà chỉ một phần video được rerank (do F4 buộc phải cap), thì **điểm sequence giữa các video không còn so sánh được** ⇒ B8/B9 xếp hạng sai. Đúng loại lỗi CLAUDE.md §4 cấm ("không fuse hai kênh lệch chất lượng").

**Nguyên tắc:** *local retrieval đổi **frame nào** trả lời event, KHÔNG đổi cách so video này với video kia.*

Cách thực hiện: z-score điểm local **trong nội bộ (video, event)** rồi cộng có trọng số:

```
frame.score = semantic_z + local_weight * local_z_in_video + kis_evidence_weight * evidence
local_z_in_video = zscore(local_score) trên đúng pool ứng viên của (video, event)
```

`local_z_in_video` có **trung bình 0 theo từng video** ⇒ tổng điểm kỳ vọng của một video không bị đẩy lên/xuống so với video không được rerank. Nhờ đó **được phép cap** số video rerank (F4) mà không phá tính so sánh chéo — video ngoài cap giữ nguyên hành vi hôm nay (`local_z = 0`).

**Chấm cả video một lượt, không lọc pool trước.** Ý định ban đầu là lấy `pool_multiplier × bucket_top_k` theo global rồi mới rerank local. Số đo 2026-08-27 (§5.0) bác ý đó: frame mà MaxSim chọn nằm ở hạng **p50 = 1, p90 ≈ 21, nhưng max = 273–308** theo bảng xếp hạng global — tức ~10% trường hợp nằm ngoài mọi pool hợp lý, và có trường hợp nằm tận đáy video. Mà chi phí thì **bằng nhau**: `50 video × 300 frame × 64 KB ≈ 960 MB`, đúng bằng ước lượng `50 × 5 event × 60 frame`, vì bucket của các event chồng lấn nhau. Đọc trọn một file `embeddings.npy` còn là I/O **tuần tự** thay vì 300 lần random-read 64 KB.

⇒ Với mỗi video ứng viên trong top-V: nạp trọn mảng dense một lần, một `einsum` chấm **mọi frame × mọi phrase của mọi event**, rồi thả. Bỏ hẳn knob `pool_multiplier`, bỏ luôn phần đuôi rủi ro.

---

## 4. Các bước thực hiện

### T1 — Nối local phrase vào B1 (per-event KIS)

| File | Việc |
|---|---|
| `app/schemas/request/trake.py` | `TrakeEventRequest.local_phrases: list[str]`; validator strip/dedupe casefold — **tái dùng** `_normalize_local_phrase_group` của `schemas/request/search.py`, không chép lại |
| `app/services/trake_service.py` | `TrakeEventQuery.local_phrases: tuple[str, ...]`; `_retrieve_one_pool` truyền `local_phrases_by_query=[phrases] * len(text_queries)` (F5) |
| `app/api/trake.py` | `_to_event_query` map field |
| `frontend/src/api/types.ts` | `TrakeEventPayload.local_phrases?: string[]` |
| `frontend/src/hooks/useTrake.ts` | `TrakeEventDraft.localPhrases: string[]` + add/remove/change; `toPayload` bỏ phrase rỗng, bỏ toàn bộ khi `text` rỗng |
| `frontend/src/components/EventSequenceInput.tsx` | Khối "Local phrases" dưới ô Text mỗi event, mượn markup/CSS đã có ở `SearchBar.tsx` |

**Test:** `tests/test_trake_service.py` — `_FakeSearchService.retrieve_kis_candidates(**kwargs)` đã ghi lại kwargs sẵn ⇒ assert `local_phrases_by_query` đi qua đúng, và event không text thì **không** gửi phrase. `tests/test_trake_api.py` — round-trip schema. Frontend: `EventSequenceInput.test.tsx`, `useTrake.test.tsx`.

### T2 — MaxSim ở tầng bucket (B2/B4) ★ phần chính

1. **Chia sẻ store, không tạo bản thứ hai.** `app/main.py` dựng **một** `KISLocalEmbeddingStore.from_config(kis_config)` rồi inject vào **cả** `SearchService(local_embedding_store=...)` (tham số đã có sẵn) **và** `TrakeService(local_embedding_store=...)` (thêm mới). Tránh parse `manifest.csv` 221 904 dòng hai lần và tránh hai LRU mmap rời nhau.
2. **`TrakeService._build_buckets`** — khi event có local phrase **và** video nằm trong top-V:
   - nạp **trọn** mảng dense của video một lần (§3), chấm mọi frame × mọi phrase của mọi event bằng **`maxsim_query_scores` có sẵn** — tái dùng `app/services/kis_local_retrieval.py`, **không viết lại MaxSim**. Cần thêm một đường "chấm cả video" bên cạnh `LocalPhraseReranker.rerank` (vốn nhận danh sách `LocalEmbeddingRef` rời);
   - z-score `local_score` trong nội bộ (video, event); frame thiếu embedding ⇒ `local_z = 0.0` + đếm vào log, **không loại** (khác KIS: ở đây loại nghĩa là mất frame ứng viên);
   - `frame.score = semantic_z + local_weight * local_z + evidence_weight * evidence`;
   - `temporal_nms` + cắt `bucket_top_k` như cũ, giữ nguyên sort theo timestamp.
3. **Chọn top-V** theo `_video_rank_evidence` (đã có sẵn), cắt bằng `local_rerank.max_videos`.
4. **`CandidateFrame`** (`app/services/trake_chain.py`) thêm `local_score: float | None = None`, `local_z: float = 0.0` — có default ⇒ không vỡ chỗ gọi/test cũ.
5. **Không gọi MaxSim từ event loop** — `_chain_candidate_videos` vốn đã nằm trong `asyncio.to_thread`, giữ nguyên.
6. **Một dòng log** cùng dạng `[TRAKE TIMING]`: `local_videos=`, `local_frames_scored=`, `local_missing=`, `local_io_ms=`, `local_scoring_ms=`.

**Config mới — `configs/search/trake.yaml`** (không magic number, CLAUDE.md §6):

```yaml
# --- B2b: local phrase-to-patch rerank trong bucket -----------------------
# Chỉ chạy khi event có local phrase. local_z được z-score TRONG NỘI BỘ
# (video, event) => trung bình 0 mỗi video => cap max_videos không phá tính
# so sánh điểm sequence giữa các video (plan §3).
local_rerank:
  enabled: true
  max_videos: 50          # top-V theo KIS evidence. ~50 x 300 frame x 64KB ~ 960MB I/O tuan tu
  weight: 0.5             # frame.score += weight * local_z   [Cần đo]
```

Và **nâng** `configs/search/search.yaml: kis.local_retrieval.mmap_cache_size` từ `64` → `512` (F4) — cache chỉ giữ handle + index, không giữ dữ liệu.

**Test** (`tests/test_trake_service.py`, hoặc tách `tests/test_trake_local_rerank.py`):
- frame có local score cao nhưng global hạng thấp **được** vào bucket và thắng khi `weight` đủ lớn;
- video ngoài `max_videos` ⇒ điểm frame **y hệt** khi tắt local (bảo chứng tính so sánh chéo);
- video thiếu `embeddings.npy` ⇒ vẫn chained, `local_z = 0`, có log, **không raise**;
- `local_z` mean ≈ 0 trong mỗi (video, event) — chốt trực tiếp nguyên tắc §3;
- thứ tự thời gian của sequence không đảo sau rerank (invariant CLAUDE.md §3).

### T3 — Diagnostics + UI

- `app/schemas/response/trake.py: TrakeKeyframeResponse` thêm `local_score`, `local_z` (cùng nhóm `semantic_z`, `kis_rank` đang có).
- `frontend/src/components/TrakeResultList.tsx`: hiện `local` cạnh `z` / `kis` để operator đang bị tính giờ phân biệt "encoder thích frame này" với "phrase khớp vùng ảnh".
- Badge cảnh báo khi `local_missing > 0` cho video đang xem.

---

## 5. Cách verify (không dùng `eval/`)

**5.0 — Số đo cấu trúc đã chạy thật (2026-08-27, numpy, 3 video L25 ~300 keyframe, 300 hướng truy vấn lấy từ patch vector của một video khác):**

| Đo | Kết quả |
|---|---|
| cosine giữa 2 keyframe bất kỳ **cùng video** (proxy global = mean-pool patch) | p50 = 0,76 · p90 = 0,85 · max = 1,000 |
| `argmax` global **khác** `argmax` MaxSim | **50–55%** số hướng truy vấn |
| hạng của frame MaxSim-chọn theo xếp hạng global | p50 = 1 · p90 ≈ 21 · **max = 273–308** |
| độ tương phản `(top1 − median)/std` | global 3,5 → **MaxSim 5,1** |

⚠️ **Đây là proxy cấu trúc, KHÔNG phải phrase thật** — hướng truy vấn là patch vector (phía ảnh), còn phrase thật là embedding phía chữ và có modality gap. Con số nói được: trong cùng một video, không gian global bị nén (p50 0,76) và điểm patch-level chọn ra frame khác với global ở khoảng nửa số trường hợp. Nó **không** chứng minh frame nào đúng hơn. Muốn số thật phải chạy encoder trên máy có `torch`.

1. **Preflight coverage** — chạy trước khi bật mặc định:

```bash
cd data/fgclip_dense_8x8 && for d in L*/*/; do [ -f "$d/embeddings.npy" ] || echo "${d%/}"; done | wc -l
```

Kỳ vọng `0`. Hôm nay là **79** (F3).

2. **Unit test** — chủ dự án chạy trong venv của mình (môi trường agent không có `.venv`, thiếu `torch`/`fastapi`):

```bash
./.venv/Scripts/python.exe -m pytest tests/test_trake_service.py tests/test_trake_api.py tests/test_kis_local_retrieval.py -q
```

3. **Frontend:**

```bash
cd frontend && npx vitest run
```

4. **Đối chứng thủ công A/B** trên 2–3 truy vấn TRAKE thật: chạy một lần **không** phrase, một lần **có** phrase, so `[TRAKE TIMING]` + vị trí video/sequence đúng trong danh sách. Đây là **quan sát**, không phải metric — mọi phát biểu chỉ được viết là *"đã đổi, chưa đo"* cho tới khi có bộ nhãn (CLAUDE.md §8).

---

## 6. Rủi ro & câu hỏi treo

| # | Rủi ro | Xử lý |
|---|---|---|
| R1 | Thiếu 79 video dense ⇒ mất ứng viên âm thầm (F3) | Preflight §5.1 trước khi bật; T2 **không** loại frame thiếu feature, chỉ log |
| R2 | I/O 64 KB/frame (F4) | `local_rerank.max_videos` + `pool_multiplier`; đo `local_io_ms` trước khi nới |
| R3 | Phá tính so sánh điểm chéo video | Nguyên tắc z nội bộ video §3 + test "video ngoài cap điểm không đổi" |
| R4 | `PPLR` ↔ `main` phân kỳ (F7) **và** merge làm lệch index phrase (F8) | ⚠️ **Cần chủ dự án chốt**: merge `PPLR` về `main` (revert cái revert + hợp nhất translator) trước hay sau T1–T3? Dù chọn hướng nào, F8 phải xử lý trong chính lần merge đó |
| R6 | Event TRAKE dài bị cắt 77 token âm thầm (F9) | Việc rẻ, độc lập: log cảnh báo khi vượt cap; tách câu sau khi merge |
| R5 | `local_weight = 0.5` là số đặt tay | ⚠️ **[Cần đo]** — không có căn cứ paper/eval; đánh dấu như `abe_concept_top_p` đã làm |

**Câu hỏi cho chủ dự án:**

1. `local_rerank.max_videos: 50` có chấp nhận được về độ trễ không, hay ưu tiên recall (raise) / tốc độ (lower)?
2. Làm T1 trước rồi dừng xem log, hay làm liền T1+T2? (T1 một mình gần như không đổi điểm TRAKE — F2.)
3. Hướng xử lý phân kỳ `PPLR` / `main` (R4).

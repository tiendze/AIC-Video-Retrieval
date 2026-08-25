# Research — Refine truy vấn BTC (dịch, decomposition, Query Compiler)

**Ngày:** 2026-08-24 · **Nhánh:** `feat/llm-query-translator` · **Người hỏi:** chủ dự án
**Câu hỏi gốc:** refine truy vấn BTC thành *một query duy nhất tối ưu hơn*, hay *nhiều query qua decomposition* — dựa trên bộ đề mẫu `SOTUYEN1-bo-de-thi/`.

> **Trạng thái (cập nhật 2026-08-25):** Phần 7 (dịch VI→EN qua LLM, `GeminiTranslator`+`FallbackTranslator`) đã merge ở `904fe5c` nhưng **hiện không dùng được — hết quota Gemini**, `configs/model/query_translator.yaml` đã flip về `kind: local_nmt`. Thay vào đó, hai fix **không cần LLM** vừa merge: **`GlossaryCorrectedTranslator`** (sửa 5 lỗi dịch EnViT5 đã đo, word-boundary matching sau code review) và **Stage 1 OR-union qua tách câu** (`split_into_sentences()`, xoá hoàn toàn class lỗi truncation — đo 0/25 mảnh còn vượt trần). `QueryCompiler`/decomposition bằng LLM ở Stage 2 (§6) **vẫn CHƯA triển khai**. Không sang phase sau khi phase trước chưa có số đo (CLAUDE.md §8) — mọi con số Recall/MRR trong tài liệu này là **chưa đo**, trừ khi ghi rõ ngược lại. **Phát hiện mới:** `eval/run.py` chưa từng đo `retrieve_kis_candidates`/`rerank_kis_candidates` (chỉ đo `_search_semantic()` đơn giản hơn) — nghĩa là chưa có kênh eval nào đo được hiệu quả thật của cả hai fix này.

---

## 1. Vì sao cần nghiên cứu

Baseline arXiv:2512.13169 đề xuất QUEST (LLM rewrite một câu) nhưng **không công bố số đo nào**, và repo hiện tại chỉ có `LocalNmtTranslator` (EnViT5) làm bước dịch VI→EN duy nhất trước khi encode. Câu hỏi *"refine thành một query hay nhiều query"* không tách biệt được nếu không có dữ liệu thật — nên bước đầu tiên là đo trên chính 25 câu đề mẫu (`SOTUYEN1-bo-de-thi/`, giờ nằm trên nhánh này, commit `d4869e5`) thay vì suy đoán.

**Phát hiện dẫn đường (đo được, không phải giả định):** câu hỏi thật không phải "một hay nhiều", mà là *refine ở tầng nào* — vì hai lỗi đo được (truncation, mistranslation) đòi hỏi hai cách sửa khác nhau, và kiến trúc rerank hiện tại (§4) chỉ hỗ trợ đúng một trong hai nếu làm ẩu.

---

## 2. Phương pháp

- Đọc toàn bộ 25 file `SOTUYEN1-bo-de-thi/query-p1-*.txt` (14 KIS, 4 QA, 1 TRAKE — phát hiện `p1-8` và `p1-14` **giống nhau từng byte** ⇒ chỉ **24 truy vấn duy nhất**, đây là mẫu số đúng khi báo cáo bất kỳ số đo nào trên bộ này).
- Chạy đúng đường production: `VietAI/envit5-translation` (`models/fgclip_base` tokenizer, `max_text_tokens: 77`, `truncation=True`) trên cả 25 câu — script chẩn đoán một lần, không nằm trong repo (không có caller nào khác cần nó).
- Pass `ecc:code-reviewer` soát đường query hiện tại (`kis_scoring.py`, `search_service.py`, `quest_service.py`) để xác nhận **thứ có thật** trong code, không suy đoán từ tên hàm.
- Tra cứu 9 nguồn học thuật qua `WebSearch`/`WebFetch` (⚠️ MCP `firecrawl`/`exa` mà skill `deep-research` khuyến nghị không có sẵn ở máy này — dùng công cụ thay thế, ghi rõ để không lẫn với coverage đầy đủ hơn).

---

## 3. Ba phát hiện đo được trên 25 đề BTC

### 3.1 — 9/25 truy vấn (36%) bị cắt cụt trước khi encode

FG-CLIP (`configs/model/fg_clip.yaml:max_text_tokens: 77`, `truncation=True`) là encoder đang wire thật trong `app/main.py:65` (⚠️ CLAUDE.md §5 ghi "BEiT3Encoder đang wire" — **stale**, chưa sửa, xem §9 câu hỏi treo). Sau khi dịch VI→EN bằng EnViT5, 9 câu vượt trần 77 token:

| File | Token EN | Cắt mất | Giữ lại |
|---|---:|---:|---:|
| `p1-23-kis` | 110 | 33 | 70,0% |
| `p1-6-kis` | 105 | 28 | 73,3% |
| `p1-18-kis` | 105 | 28 | 73,3% |
| `p1-16-trake` | 91 | 14 | 84,6% |
| `p1-17-qa` | 90 | 13 | 85,6% |
| `p1-8-kis` / `p1-14-kis` | 86 | 9 | 89,5% |
| `p1-24-kis` | 84 | 7 | 91,7% |
| `p1-7-kis` | 79 | 2 | 97,5% |

Phần bị cắt luôn là **đuôi câu** — trong đề BTC đuôi câu thường mang chi tiết phân biệt duy nhất:
- `p1-23-kis`: mất toàn bộ sơ đồ 3 tầng (thứ duy nhất nhận dạng được slide).
- `p1-18-kis`: mất "chén nước chấm nhỏ với 2 miếng ớt".
- `p1-16-trake`: mất trọn **sự kiện E3** — một trong ba mốc TRAKE bắt buộc phải nộp.

Đây là lỗi im lặng đúng nghĩa CLAUDE.md §3 — không exception, không log, không dấu hiệu UI.

### 3.2 — EnViT5 dịch hỏng đúng danh từ mang tín hiệu truy xuất

| Đề | Tiếng Việt | EnViT5 dịch ra | Hậu quả |
|---|---|---|---|
| `p1-16-trake` | con lân | *phosphor* | Chủ thể 3 sự kiện thành nguyên tố hoá học |
| `p1-16-trake` | kẻng đồng | *the phosphor dancer* | E3 mất vật thể cần tìm |
| `p1-24-kis` | lia cam | *a single orange peel* | Động tác máy quay thành "vỏ cam" |
| `p1-24-kis` | cây lục bình | *greenery* | Thực thể định danh cả phóng sự bị xoá |
| `p1-20-kis` | cầm dù | *carrying parachutes* | Ô thành dù nhảy — không keyframe nào khớp |
| `p1-25-kis` | trống cơ | *muscle drum kit* | Vô nghĩa |
| `p1-23-kis` | mũi tên vàng/xanh ngọc | *yellow/yellow arrows* | Sai màu, trong truy vấn mà màu là tất cả |
| `p1-19-kis` | con lân | *The animal* | Chủ thể bị làm mờ |
| `p1-01-kis` | hơn 5 người | *five people* | Sai số đếm |
| `p1-04-kis` | (mệnh đề thứ 2, ~29 từ) | *(mất hẳn)* | NMT bỏ nguyên một mệnh đề |
| `p1-22-kis` | áo dài | *gown* | Mất đặc trưng trang phục |

Cộng dồn 3.1 + 3.2: **14/24 truy vấn duy nhất (~58%)** suy giảm đo được trước khi retrieval bắt đầu.

### 3.3 — Đề BTC có cấu trúc, và cấu trúc đó đòi hỏi kênh xử lý khác nhau

| Đặc tính | Số truy vấn | Ví dụ | Kênh đúng |
|---|---:|---|---|
| Chuỗi thời gian nhiều cảnh | 15/24 | "bắt đầu bằng… sau đó… kết thúc" | decompose → events |
| Gắn thuộc tính vào vật thể | 14/24 | "áo vàng quần đen", "kéo màu đen" | ABE pairs (`app/services/abe.py`) |
| Ràng buộc đếm / thứ tự | 12/24 | "3 người đội nón đỏ", "nhất, nhì, ba" | người xác minh — **không** vào fusion tự động |
| Neo chữ trong khung hình | 7/24 | "London Zoo", "giá dầu mazut" | OCR / BM25 |
| Từ ngữ động tác máy quay | 6/24 | "quay chậm", "lia cam", "zoom xa dần" | loại khỏi query thị giác |
| Phủ định / loại trừ | 3/24 | "chỉ có một người đeo kính" | người xác minh |
| Cần tri thức ngoài hình ảnh | 2/24 | "Tên của con đèo là gì?" | ASR gate |

Ba hàng "người xác minh" / "loại khỏi query" là lý do §6 không được decompose ngây thơ — keyframe tĩnh không thể hiện "quay chậm", FG-CLIP không đếm được "3 người", không xử lý được "chỉ có một".

---

## 4. Ràng buộc từ code hiện có (đã xác minh, không phải suy đoán)

Pass `ecc:code-reviewer` sửa hai giả định ban đầu của tôi:

- **`SearchService.search_kis()` là stub luôn raise** (`app/services/search_service.py:211-216`). Đường sống thật là **hai lệnh gọi tách biệt**: `retrieve_kis_candidates()` (`:218-465`, Stage 1) và `rerank_kis_candidates()` (`:519-684`, Stage 2).
- **Stage 1 chỉ nhận MỘT `text_query: str` và MỘT `ocr_query: str`** (`app/schemas/request/search.py:91-96`, không phải `Sequence`), hợp qua `_union_stage1_candidate_ids()` (`:918-929`).
- **Stage 2 KHÔNG bao giờ nới pool.** `rerank_kis_candidates()` nhận `Sequence[KisQuery]` nhưng chỉ xếp lại trên `candidate_pool_ids` cố định do caller truyền vào (`:584-587`) — mọi `KisQuery` dù bao nhiêu đều rerank trên đúng tập ứng viên Stage 1 đã tìm ra, không tìm thêm được gì mới.
- **Fusion là geometric mean không trọng số, không cơ chế bỏ qua.** `geometric_mean_score()` (`app/services/kis_scoring.py:244-253`) nhận `Sequence[float]` bất kỳ, clamp ở `epsilon=1e-8` (`configs/search/search.yaml:33`). Với 2 sub-query, một percentile bị kẹp về `1e-8`: `√(1e-8 × 0,9) ≈ 9,5e-5` — thấp hơn ~3.000 lần một ứng viên tầm thường-đều-đều `0,3`. Áp dụng **hai lần**: geomean trong từng modality (`_build_text_modality_score`, `search_service.py:1114`), rồi geomean lần nữa giữa các modality (`:663-665`).
- **Dead code không nên xây trên:** `KisSearchRequest` (`app/schemas/request/search.py:79-81`), `_union_candidate_ids()` (`search_service.py:902-915`, 0 caller), `_reciprocal_rank_fusion()` (`:1211-1240`, giữ lại có chủ đích cho tương lai, không dùng cho decomposition vì nó là rank-fusion khác loại kênh, không phải same-type geomean).

**Hệ quả kiến trúc bắt buộc:** decomposition phải là **OR ở Stage 1** (hợp pool ⇒ tăng recall) và **AND ở Stage 2** (geomean ⇒ tăng precision). Hôm nay repo chỉ có AND, và chỉ ở Stage 2 — nghĩa là decomposition tự phát hôm nay chỉ đảo thứ tự ứng viên đã có, không cứu được truy vấn mà Stage 1 chưa từng khui ra.

---

## 5. Vì sao mỗi phương án đơn lẻ đều thua

**Chỉ "một query tối ưu hơn":** thua vì trần token (§3.1) — 9/25 truy vấn không vừa 77 token ngay cả sau khi nén, muốn vừa phải vứt đúng nội dung cần cứu. Thêm nữa, CLIP-family hành xử như bag-of-words và làm nhoè liên kết thuộc tính–vật thể ([Yuksekgonul et al., "CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally"](https://openreview.net/pdf/981efbd7b189575f82ed4626e813cb3bc2d0148e.pdf)) — đúng cái mà 14/24 đề (§3.3) dựa vào.

**Chỉ "decomposition":** thua vì hai lý do đo được ở §4 — Stage 2 không cứu recall, và AND-fusion không trọng số có thể giết một đáp án đúng chỉ vì một sub-query không biểu diễn được.

---

## 6. Phương án — Query Compiler (CHƯA triển khai)

Một lần gọi `gemini-3.6-flash` (bắt buộc, không phải mặc định — CLAUDE.md §5: `gemini-2.5-flash` lờ `response_format`), structured output qua `model_json_schema()`, sinh nhiều output có kiểu thay vì một câu:

```python
class CompiledQuery(BaseModel):
    stage1_queries: list[str]       # 1–3 câu tiếng Anh, mỗi câu ≤ 60 token → OR-union Stage 1
    subqueries:     list[str]       # mệnh đề thị giác rời → AND-geomean Stage 2
    abe_pairs:      list[AbePair]   # (attribute, object) → app/services/abe.py scorer sẵn có
    ocr_terms:      list[str]       # chuỗi hiếm, giữ nguyên tiếng Việt → BM25
    asr_terms:      list[str]       # từ khoá lời nói → ASR video gate
    events:         list[str]       # mô tả sự kiện theo thứ tự → DANTE / TRAKE
    unverifiable:   list[str]       # đếm / phủ định / thứ tự → CHỈ hiển thị cho người, không vào score
    dropped:        list[str]       # từ ngữ máy quay đã loại, để người kiểm chứng
```

**Bốn luật rút ra từ số đo, không từ sở thích:**

1. Trần **60 token** mỗi câu `stage1_queries` — chừa biên dưới 77 để chênh lệch tokenizer không đẩy tràn.
2. `unverifiable` **không bao giờ** vào `geometric_mean_score()` — mệnh đề đếm/phủ định/thứ tự (12/24 và 3/24 đề, §3.3) sẽ giết đáp án đúng theo đúng cơ chế đã đo ở §4.
3. `dropped` giữ lại từ ngữ máy quay bị loại (6/24 đề) để người vận hành thấy đã bỏ gì — không tự động, đúng tinh thần "đừng tự động hoá chỗ paper cố ý để người vận hành quyết định" (CLAUDE.md §1.3, cùng lý do `QueryRewriteService` không tự gọi `SearchService`, `app/services/quest_service.py:20-29`).
4. `subqueries` decompose bằng ngôn ngữ nào phải khớp với `_prepare_text_inputs()` (`search_service.py:764-771`) — hàm này dịch **vô điều kiện** mọi query khi `_query_translator` khác `None`, và `KisQuery` (`kis_scoring.py:37-42`) chưa có cờ bỏ qua dịch. Nếu Compiler phát tiếng Anh trực tiếp, cần thêm cờ đó trước — nếu không `LocalNmtTranslator`/`GeminiTranslator` sẽ "dịch" tiếng Anh sang tiếng Anh, hỏng im lặng, không raise.

**Việc code cần làm (chưa làm), theo tỉ lệ giá trị/rủi ro:**

| # | Thay đổi | Vì sao | Rủi ro |
|---|---|---|---|
| 1 | Stage 1 nhận `text_queries: Sequence[str]`, hợp pool | Thay đổi *duy nhất* mở khoá recall — không có nó, decomposition chỉ đảo thứ tự | Thấp — `_union_stage1_candidate_ids` đã có khuôn |
| 2 | Service `QueryCompiler` mới, theo pattern `QueryRewriteService` | Timeout + fallback + catch hẹp đã chứng minh ở `app/services/quest_service.py:341-361` | Thấp — fallback về đường EnViT5/`GeminiTranslator` hiện có ⇒ degraded mode vẫn chạy (CLAUDE.md §4) |
| 3 | Cờ bỏ qua dịch trên `KisQuery` | Không có nó, sub-query tiếng Anh bị dịch lại và hỏng im lặng | Trung bình — chạm contract dùng chung với sidecar eval (`eval/query_translation.py`) |

---

## 7. Đã triển khai vs chưa triển khai

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Dịch VI→EN qua LLM (`GeminiTranslator`) thay EnViT5 | ✅ **Đã merge**, nhánh này | `app/services/query_translator.py:171-230`, commit `904fe5c` |
| `FallbackTranslator` (Gemini primary, EnViT5 fallback, timeout) | ✅ **Đã merge** | `app/services/query_translator.py:232-308` |
| `build_query_translator()` factory dùng chung 3 entry point | ✅ **Đã merge** | `app/main.py`, `eval/run.py`, `scripts/translate_eval_queries.py` |
| Prompt dịch ép 5 luật (đầy đủ, giữ số đếm, văn hoá→nhìn thấy được, thuật ngữ máy quay, không bịa) | ✅ **Đã merge** | `app/services/query_translator.py:150-169` |
| Test hành vi (RED→GREEN, 24 test case) | ✅ **Đã merge, pass** | `tests/test_llm_translator.py` |
| Code review (0 CRITICAL, 1 HIGH đã sửa, 2 MEDIUM đã sửa) | ✅ **Đã xong** | Xem §9.2 |
| Stage 1 nhận nhiều `text_query` qua tách câu (OR-union, không LLM) | ✅ **Đã merge** | `app/services/query_decomposition.py::split_into_sentences`, wire ở `search_service.py::retrieve_kis_candidates` |
| `GlossaryCorrectedTranslator` (sửa lỗi dịch EnViT5 đã đo, không LLM) | ✅ **Đã merge** | `app/services/query_translator.py`, word-boundary matching sau code review |
| `CompiledQuery` / `QueryCompiler` service (decomposition bằng LLM) | ❌ **Chưa làm** | Chỉ có thiết kế ở §6, cần quota Gemini |
| Cờ bỏ qua dịch trên `KisQuery` | ❌ **Chưa làm** | — |
| Kênh eval `semantic@compiled` | ❌ **Chưa làm** | — |
| **Kênh eval đo `retrieve_kis_candidates`/`rerank_kis_candidates`** | ❌ **Chưa có, phát hiện mới** | `eval/run.py` chỉ đo `_search_semantic()` — chưa từng đo Stage 1/2 KIS thật |
| **Đo Recall/MRR với `GeminiTranslator`** | ❌ **Chưa đo, hiện không khả thi** | Hết quota Gemini |
| **Đo Recall/MRR với glossary + tách câu** | ❌ **Chưa đo** | Cần kênh eval mới ở trên trước |

---

## 8. Cách đo — bắt buộc trước khi tuyên bố "cải thiện"

CLAUDE.md §8: không sang phase sau khi phase trước chưa có số đo, và mọi tuyên bố "cải thiện" phải trả lời được *đo bằng metric nào trong PRD §6*.

**Ablation tối thiểu** (tách hai giả thuyết dễ bị gộp làm một — "LLM dịch tốt hơn" khác "decomposition có tác dụng"):

1. EnViT5 nguyên trạng (baseline đã có, `semantic@en`)
2. → đổi sang `GeminiTranslator` (đã code xong, **chưa đo**) — kênh mới `semantic@compiled_translate`
3. → thêm OR-union Stage 1 (chưa code)
4. → thêm decomposition Stage 2 (chưa code)

**Sàn / mục tiêu hiện tại** (PRD §6.7, dưới sàn nghĩa là có bug chứ không phải "cần tối ưu"):

| Metric | Sàn | Mục tiêu v1 |
|---|---:|---:|
| AIC Final Score (segment) | 0,48 | 0,60 |
| Recall@1 (segment) | 0,21 | 0,26 |
| Recall@20 (segment) | 0,54 | 0,68 |
| Recall@20 (video) | 0,67 | 0,84 |
| MRR (segment) | 0,28 | 0,35 |

Mỗi báo cáo số đo phải kèm: git SHA · `eval_set_version` + hash bộ nhãn · số `video_id` duy nhất đánh giá được · số truy vấn · seed (CLAUDE.md §8).

**Rủi ro rò rỉ phải chặn trước khi ăn mừng:** bộ eval là nhãn Gemini tự sinh (PRD §6.1). Nếu `GeminiTranslator`/`QueryCompiler` cũng là Gemini, nó có thể học "nói giọng của nhãn" và thổi phồng điểm mà không cải thiện gì trên đề thật — vi phạm CLAUDE.md §3 *"model sinh nhãn ≠ model bị đánh giá"*. Cách chặn: đổi model một trong hai vai, hoặc chỉ tin số sau khi hiệu chỉnh qua nộp bài BTC (PRD §6.9).

---

## 9. Việc cần chốt trước khi làm tiếp

### 9.1 — Câu hỏi treo cho chủ dự án

| # | Câu hỏi | Trạng thái |
|---|---|---|
| A | Triển khai Stage 1 OR-union + `QueryCompiler` decomposition (§6), hay dừng ở dịch LLM đã merge và đo trước? | ⬜ chưa chốt |
| B | `unverifiable` (đếm/phủ định/thứ tự) hiện checklist cho người vận hành — đúng UX mong muốn, hay cần tự động hoá khác? | ⬜ chưa chốt |
| C | Dùng model khác Gemini cho một trong hai vai (sinh nhãn eval / Query Compiler) để tránh rò rẻ (§8), hay chấp nhận rủi ro và hiệu chỉnh qua BTC? | ⬜ chưa chốt |

### 9.2 — Đã sửa trong lần review này (không còn treo)

Pass `ecc:code-reviewer` trên `GeminiTranslator`/`FallbackTranslator` trả về 0 CRITICAL, 1 HIGH, 2 MEDIUM — cả ba **đã sửa và commit** (`904fe5c`):
- HIGH: `FallbackTranslator` bắt exception quá hẹp (rủi ro crash thay vì fallback khi lỗi mạng thật) → đổi sang `except Exception` có chủ đích, giải thích trong docstring.
- MEDIUM: mỗi lần timeout tạo `ThreadPoolExecutor` mới không giới hạn → đổi sang pool dùng chung `max_workers=4`.
- MEDIUM: `eval/run.py` fetch API key vô điều kiện, build `Settings()` lặp 3 lần → truyền `settings` xuyên suốt, gate theo `kind`.

### 9.3 — Doc lệch code (chưa sửa, ngoài phạm vi lần này)

CLAUDE.md §5 ghi *"`BEiT3Encoder` đang wire trong `app/main.py`"* — sai. `app/main.py:65` build **FG-CLIP** (`_build_encoder()` gọi `build_encoder(_FG_CLIP_CONFIG_PATH)`). Theo CLAUDE.md §2, code lệch docs phải sửa docs trong cùng change — chưa làm vì nằm ngoài phạm vi "chỉ dịch" đã chốt; cần chủ dự án xác nhận trước khi sửa CLAUDE.md.

---

## 10. Nguồn tham khảo

1. [MERVIN: A Unified Framework for Multimodal Event Retrieval in Vietnamese News Videos](https://arxiv.org/abs/2605.16120) — tiền lệ gần nhất: đúng miền tin tức tiếng Việt, đúng ba task AIC HCMC, dùng cả ba bước translate + rewrite + decompose. Đạt 79/88 điểm vòng sơ tuyển AIC HCMC 2025.
2. [When Should Queries Be Decomposed? A Stage-Aware Study of Query Decomposition for Multi-Condition Retrieval](https://arxiv.org/pdf/2606.08577) — decomposition giúp ở tầng rerank nhưng có thể **hại** ở tầng first-stage retrieval. Cơ sở lý thuyết cho việc tách OR/AND theo tầng ở §4–§6.
3. [Not All Queries Need Rewriting: When Prompt-Only LLM Refinement Helps and Hurts Dense Retrieval](https://arxiv.org/html/2603.13301) — rewrite làm giảm 9% trên miền có từ vựng đã khớp sẵn; trần lợi ích chọn lọc per-query chỉ ~3 điểm. Lý do không rewrite vô tội vạ.
4. [Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval](https://arxiv.org/html/2506.10202) — decompose thành prequel/current/sequel, fusion theo trọng số nghịch entropy, +8 NDCG trên MultiVENT. Cảnh báo trực tiếp: "mean over captions underperformed the most" — không dùng trung bình cộng đơn giản.
5. [CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally](https://openreview.net/pdf/981efbd7b189575f82ed4626e813cb3bc2d0148e.pdf) — thông tin gắn kết thuộc tính–vật thể **có** trong embedding, chỉ mất ở bước căn chỉnh cross-modal.
6. [ABE-CLIP: Training-Free Attribute Binding Enhancement for Compositional Image-Text Matching](https://arxiv.org/pdf/2512.17178) — chính là ABE đang có trong repo (`app/services/abe.py`); xác nhận `abe_pairs` là kênh đúng cho 14/24 đề gắn thuộc tính.
7. [Enhanced Multimodal Video Retrieval System: Integrating Query Expansion and Cross-modal Temporal Event Retrieval](https://arxiv.org/html/2512.06334) — Gemini sinh nhiều query tương đương rồi **hợp** top-k. Không công bố số đo định lượng cho phần query expansion, chỉ case study — không trích dẫn như bằng chứng định lượng.
8. [MADTempo: An Interactive System for Multi-Event Temporal Video Retrieval with Query Augmentation](https://arxiv.org/pdf/2512.12929) — chưa fetch được nội dung đầy đủ (PDF quá lớn), chỉ liệt kê tham khảo.
9. [Results of the 2025 Video Browser Showdown](https://arxiv.org/abs/2509.12000) — bối cảnh hệ thống interactive; ViFi và SnapSeek 2.0 dùng temporal "now-and-then" input.

---

## 11. Ghi chú về độ tin cậy của file này

- §3 (token truncation, mistranslation) là **số đo thật**, chạy trực tiếp `models/fgclip_base` + `VietAI/envit5-translation` trên cả 25 file `SOTUYEN1-bo-de-thi/`, tại git `5191098` (commit gốc trước khi nhánh này tách ra). Script chẩn đoán không nằm trong repo (chạy một lần, không có caller khác cần).
- §4 (ràng buộc code) là **đọc trực tiếp working tree**, đã qua một vòng `ecc:code-reviewer` sửa premise sai (§4 đã ghi rõ hai chỗ tôi sửa lại).
- §6 (Query Compiler) là **thiết kế trên giấy, chưa code, chưa đo**. Bốn luật trong §6 suy ra từ số đo ở §3–§4, không phải sở thích, nhưng bản thân thiết kế chưa qua implementation hay eval.
- §7 dòng "Đã triển khai" (`GeminiTranslator`, `FallbackTranslator`) đã qua TDD RED→GREEN (`tests/test_llm_translator.py`, 24 case) và code review (0 CRITICAL/1 HIGH đã sửa/2 MEDIUM đã sửa), nhưng **chưa có số Recall/MRR nào** — "đã merge" không đồng nghĩa "đã đo tốt hơn". §8 là bước bắt buộc kế tiếp.
- 9 nguồn ở §10 tra qua `WebSearch`/`WebFetch`, không phải `firecrawl`/`exa` mà skill `deep-research` khuyến nghị (MCP không có sẵn ở máy này) — coverage có thể hẹp hơn nếu chạy lại với đúng công cụ.

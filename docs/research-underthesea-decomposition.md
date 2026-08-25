# Research + Plan — Typed query decomposition bằng `underthesea`

**Ngày:** 2026-08-25 · **Nhánh:** `feat/llm-query-translator`
**Trạng thái:** ✅ **Dependency `underthesea` ĐÃ ĐƯỢC CHỦ DỰ ÁN DUYỆT (2026-08-25)** — trả lời câu hỏi B của [`research-query-decomposition-methods.md`](research-query-decomposition-methods.md) §6.

**Phạm vi:** triển khai §4.2 của doc trên — thay `split_into_sentences()` (regex dấu câu) bằng decomposer sinh **subquery có kiểu** (event / attribute-object / location / distinctive-clue), không dùng LLM API.

> **Chưa duyệt, vẫn treo:** câu A (thí nghiệm `walk_short_pos` 248 token), C (dựng kênh eval Stage 1 trước hay code trước), D (trần số subquery), E (sửa 3 chỗ docs lệch code). Doc này **không** giả định chúng đã chốt — xem §7.

---

## 1. Sự thật đã xác minh về `underthesea`

Fetch từ PyPI JSON API + readthedocs + repo `undertheseanlp/chunking` ngày 2026-08-25. **Không viết theo trí nhớ** (CLAUDE.md §5).

### 1.1 — Phiên bản & dependency

| | |
|---|---|
| **Bản mới nhất** | `9.5.0` |
| **Python yêu cầu** | `>=3.10` |
| **Dependency lõi** | `Click>=6.0` · `tqdm` · `requests` · `joblib` · `PyYAML` · `underthesea_core>=3.3.0` · `huggingface-hub` |
| **torch / transformers** | ⚠️ **CHỈ trong extra `[deep]`** — bản cài mặc định **không** kéo torch/transformers |

### 1.2 — ✅ Tương thích với repo: không xung đột

| Ràng buộc repo | Giá trị | `underthesea` cần | Kết luận |
|---|---|---|---|
| `requires-python` (`pyproject.toml:5`) | `>=3.10,<3.11` | `>=3.10` | ✅ Khớp |
| Python thực tế trong `.venv` | `3.10.11` | | ✅ |
| `transformers` (đường FG-CLIP) | `4.44.2` | không đụng (base install) | ✅ Không xung đột |
| `torch` | `2.4.1` | không đụng (base install) | ✅ Không xung đột |
| `huggingface-hub` | `0.36.2`, **đã là dependency sẵn** (`pyproject.toml:123`) | có, không pin phiên bản | ✅ Không phải dependency mới |

**Kết luận quan trọng:** cài bằng `pip install underthesea` (**KHÔNG** `underthesea[deep]`) thì đây là dependency **thuần Python, nhẹ, chạy CPU** — không đụng đến mâu thuẫn `timm==0.4.12` vs `transformers>=4.49` (CLAUDE.md §5), không chiếm VRAM của FG-CLIP.

> ⚠️ **Một điểm phải kiểm lúc cài:** `underthesea_core` là extension biên dịch sẵn (Rust). Nếu không có wheel cho `cp310-win_amd64` thì pip sẽ cố build từ nguồn và cần toolchain. **[Cần verify lúc cài thật]** — nếu fail, đây là lý do chính đáng để cân nhắc lại.

### 1.3 — API thật (verbatim từ readthedocs)

```python
from underthesea import sent_tokenize, word_tokenize, pos_tag, chunk, ner
```

| Hàm | Trả về | Dùng vào việc gì ở đây |
|---|---|---|
| `sent_tokenize(text)` | `list[str]` | **Thay thế trực tiếp regex hiện tại** — tách câu có hiểu ngữ cảnh (không cắt nhầm sau "TP.", "v.v.") |
| `pos_tag(text)` | `list[(word, pos)]` | Không dùng trực tiếp — `chunk()` đã bao gồm |
| `chunk(text)` | `list[(word, pos_tag, chunk_tag)]` | **Nguồn chính** của typed subquery |
| `ner(text)` | `list[(word, pos, chunk_tag, entity_tag)]` | Ứng viên cho `distinctive_clue` (tên riêng) — xem §3.4 |

### 1.4 — 🛑 Tagset là **IOB2**, không phải nhãn phẳng

Đây là chỗ dễ viết sai code nhất. Repo `undertheseanlp/chunking` liệt kê **21 nhãn**:

```
B-AP, B-MP, B-NP, B-PP, B-QP, B-TP, B-VP, B-WH, B-WP, B-XP,
I-AP, I-MP, I-NP, I-PP, I-QP, I-VP, I-WH, I-WP, I-XP, N-NP, O
```

Nghĩa là `chunk()` trả về **token-level tag có tiền tố `B-`/`I-`**, không phải cụm đã gom sẵn. Code phải **tự gom** `B-NP` + các `I-NP` liền sau thành một danh ngữ. Bỏ qua bước này ⇒ mỗi token thành một subquery, hỏng hoàn toàn.

Nhãn đáng chú ý ngoài NP/VP/PP/AP:

| Nhãn | Phỏng đoán ý nghĩa | Vì sao quan trọng ở đây |
|---|---|---|
| `QP` | Quantifier phrase (cụm số lượng) | ⭐ Có thể bắt được **ràng buộc đếm** ("3 người đội nón đỏ") — thứ trước đây đánh giá là chỉ LLM làm được. Xem §3.5 |
| `TP` | Time phrase (cụm thời gian) | Hỗ trợ tách event theo thời gian, bổ trợ §4.1 của doc mẹ |
| `MP`, `WH`, `WP`, `XP`, `N-NP` | Chưa rõ | **[Cần verify]** — không tìm được định nghĩa tagset đầy đủ |

⚠️ Cột "phỏng đoán" đúng nghĩa là **phỏng đoán**, suy từ tên nhãn. Phải in thử `chunk()` trên vài câu đề BTC để xác nhận **trước khi** viết logic phụ thuộc vào `QP`/`TP`.

### 1.5 — ⚠️ F1 của chunker chỉ ~77% — con số quyết định cả thiết kế

Repo `undertheseanlp/chunking` gắn badge **F1 ≈ 77%**. Không có bảng chi tiết theo từng nhãn (link "Detail Report" trỏ Google Sheets, không fetch được).

**Đây không phải chi tiết phụ, nó ràng buộc thiết kế:** cứ ~4 ranh giới cụm thì có ~1 cái sai. Và con số 77% đo trên **văn viết chuẩn** — đề BTC là văn mô tả cảnh quay, **lệch domain**, nên thực tế có thể thấp hơn.

Ba hệ quả bắt buộc:

1. **Chỉ dùng cho OR-union Stage 1, tuyệt đối không cho AND-fusion Stage 2.** `geometric_mean_score()` (`kis_scoring.py:273-282`) kéo cả điểm về ~0 khi một sub-query trượt — nhồi cụm chunk sai vào đó là tự bắn vào chân.
2. **Luôn giữ truy vấn gốc làm subquery #0** (§3.6). Chunk sai chỉ được phép *thêm* nhiễu, không được phép *lấy mất* hành vi baseline.
3. **Phải đo accuracy chunk trên chính 24 đề BTC trước khi tin** — không tin badge 77% của domain khác.

---

## 2. Ràng buộc từ code hiện có

Trích từ trace đã thực hiện; chi tiết đầy đủ ở [`research-query-decomposition-methods.md`](research-query-decomposition-methods.md) §2.

| Ràng buộc | Vị trí | Ý nghĩa với thiết kế này |
|---|---|---|
| Điểm cắm duy nhất | `search_service.py:253` | Thay `split_into_sentences(text_query)` bằng hàm khác trả `list[str]` ⇒ toàn bộ union/fuse phía sau **không cần sửa** |
| Thứ tự **TÁCH → DỊCH** | `:253` rồi `:301` | ✅ Đúng thứ tự cần: chunk trên **tiếng Việt** rồi mới dịch từng mảnh. Không phải đảo gì |
| Điểm hiển thị là **SUM** cosine | `:1181` | ⚠️ Thêm subquery **không** trung tính: candidate khớp nhiều mảnh rác có thể vượt candidate khớp một mảnh tốt. Xem §6 rủi ro |
| Có `text_queries` ⇒ bỏ geomean text/ocr/title | `:451-453` | Bật decomposition = tắt đóng góp xếp hạng của OCR/Title ở Stage 1 |
| **Không có trần số subquery** | không tồn tại | Mỗi subquery = 1 lần gọi Milvus riêng ⇒ chunking sinh 8–12 mảnh sẽ nhân latency lên chừng đó. **Phải thêm trần trong cùng change** |
| Không có Protocol cho decomposer | — | Phải tự định nghĩa; đây là khoảng trống kiến trúc thật |
| `KisQuery` không có cờ skip-translate | `kis_scoring.py:38-43` | Chưa chạm tới ở phase này (chỉ làm Stage 1), nhưng sẽ chặn khi mở rộng sang Stage 2 |

---

## 3. Thiết kế

### 3.1 — Module mới: `app/services/query_chunking.py`

**Không sửa `query_decomposition.py`.** Giữ nguyên `split_into_sentences()` làm **fallback** khi underthesea lỗi/không cài được — đúng tinh thần degraded mode (CLAUDE.md §4).

```
app/services/query_decomposition.py   ← GIỮ NGUYÊN, thành fallback
app/services/query_chunking.py        ← MỚI: chunk → typed subquery
```

### 3.2 — Kiểu dữ liệu

```python
class SubqueryKind(str, Enum):
    ORIGINAL         = "original"          # truy vấn gốc, luôn có (§3.6)
    EVENT            = "event"             # NP + VP liền kề
    ATTRIBUTE_OBJECT = "attribute_object"  # NP có AP bổ nghĩa
    LOCATION         = "location"          # PP chỉ nơi chốn
    DISTINCTIVE_CLUE = "distinctive_clue"  # NP hiếm / tên riêng
    UNVERIFIABLE     = "unverifiable"      # QP đếm — KHÔNG vào scoring (§3.5)

@dataclass(frozen=True)
class TypedSubquery:
    kind: SubqueryKind
    text: str                     # tiếng Việt, chưa dịch
    source_span: tuple[int, int]  # vị trí token trong câu gốc, để debug/UI
```

`frozen=True` theo `.claude/rules/ecc/python/coding-style.md`.

### 3.3 — Thuật toán gom IOB → cụm

```
1. chunk(text) -> [(word, pos, iob_tag), ...]
2. Gom: gặp "B-X" mở cụm mới loại X; "I-X" nối vào cụm đang mở; "O" đóng cụm.
3. Từ chuỗi cụm, dựng subquery theo luật §3.4.
4. Lọc: bỏ cụm chỉ gồm stopword; bỏ cụm < 2 token (trừ tên riêng từ ner()).
5. Trần số subquery (config) — cắt theo thứ tự ưu tiên §3.7.
```

### 3.4 — Luật dựng subquery

| Kiểu | Luật | Ví dụ |
|---|---|---|
| `EVENT` | `NP` + `VP` liền kề (cho phép `PP` xen giữa) | "người đàn ông" + "cầm ô bước qua đường" |
| `ATTRIBUTE_OBJECT` | `NP` chứa `AP`, **hoặc** `NP` + `AP` liền sau | "chén nước chấm nhỏ", "áo dài vàng" |
| `LOCATION` | `PP` mà giới từ thuộc tập nơi chốn (`trên`, `dưới`, `trong`, `ngoài`, `trước`, `sau`, `cạnh`, `giữa`, `tại`, `ở`) | "trên sân khấu" |
| `DISTINCTIVE_CLUE` | `NP` mà `ner()` gán nhãn thực thể, **hoặc** IDF cao (§4.3 doc mẹ — phase sau) | "kẻng đồng", "London Zoo" |
| `UNVERIFIABLE` | Cụm `QP` | "hơn 5 người", "3 chiếc nón đỏ" |

⚠️ Luật `LOCATION` dùng **danh sách giới từ nơi chốn tự soạn**, vì `PP` không phân biệt nơi chốn với các loại giới ngữ khác. Danh sách này là magic list ⇒ đưa vào `configs/`, không hard-code (CLAUDE.md §6).

### 3.5 — `UNVERIFIABLE` — thu hồi một phần thiết kế `CompiledQuery`

Nhãn `QP` có thể bắt được ràng buộc đếm mà không cần LLM. Đây là **thu hồi một phần** trường `unverifiable` của `CompiledQuery` (§6 doc `research-query-refinement.md`) — thứ trước đây đánh giá là chỉ LLM làm được.

**Luật cứng, không thương lượng:** subquery `UNVERIFIABLE` **không bao giờ** đi vào `text_queries` hay `geometric_mean_score()`. Nó chỉ:
- hiển thị cho người vận hành như một checklist ("truy vấn yêu cầu **hơn 5 người** — tự kiểm khi duyệt kết quả"),
- và **bị loại khỏi** các subquery khác để không làm nhiễu vector.

FG-CLIP không đếm được; đưa "3 người" vào vector chỉ làm loãng. Lý do đã ghi ở `research-query-refinement.md:126`.

⚠️ Toàn bộ §3.5 phụ thuộc vào giả định `QP` = quantifier phrase — **[Cần verify]** trước khi code (§1.4).

### 3.6 — Truy vấn gốc luôn là subquery #0 ⭐ luật an toàn quan trọng nhất

```python
subqueries = [TypedSubquery(ORIGINAL, text, ...), *chunk_derived]
```

**Vì sao bắt buộc:** chunker F1 ~77% (§1.5). Nếu chunk sai hết, OR-union vẫn còn truy vấn gốc ⇒ **sàn dưới của hệ thống mới không thấp hơn hệ thống cũ về recall**. Không có luật này, một câu bị chunk hỏng có thể mất sạch candidate đúng — và hỏng **âm thầm**, không ai biết.

⚠️ **Nhưng đây chỉ là sàn cho RECALL, không phải cho RANKING.** Vì điểm hiển thị là SUM (§2), thêm nhiều mảnh rác vẫn có thể đẩy candidate nhiễu lên trên. Sàn recall ≠ sàn precision. Đây là lý do §5 phải đo cả `Recall@K` lẫn `MRR`, không chỉ recall.

### 3.7 — Trần số subquery + thứ tự ưu tiên

Vì mỗi subquery = 1 lần Milvus search và hiện **không có trần nào**:

```yaml
# configs/search/decomposition.yaml  (mới)
max_subqueries: 6          # gồm cả ORIGINAL
chunker: underthesea       # underthesea | sentences (fallback)
location_prepositions: [trên, dưới, trong, ngoài, trước, sau, cạnh, giữa, tại, ở]
min_chunk_tokens: 2
```

Khi vượt trần, cắt theo ưu tiên: `ORIGINAL` (không bao giờ cắt) > `DISTINCTIVE_CLUE` > `EVENT` > `ATTRIBUTE_OBJECT` > `LOCATION`. `UNVERIFIABLE` không tính vào trần vì không đi search.

---

## 4. Kế hoạch triển khai (TDD RED → GREEN)

Theo `.claude/rules/ecc/common/testing.md` + CLAUDE.md §8.

| Bước | Việc | File | Test viết TRƯỚC |
|---|---|---|---|
| **1** | Thêm dependency + smoke test API thật | `pyproject.toml` | `test_underthesea_smoke.py` — `chunk()` trả 3-tuple, tag có tiền tố `B-`/`I-`; **in thật tagset trên 3 câu đề BTC** để đóng `[Cần verify]` §1.4 |
| **2** | Gom IOB → cụm | `query_chunking.py` | `test_iob_grouping` — `B-NP I-NP I-NP` → **1** cụm chứ không phải 3; `O` đóng cụm; tag lạ (`N-NP`) không crash |
| **3** | Luật dựng subquery | `query_chunking.py` | `test_event_from_np_vp`, `test_attribute_object`, `test_location_only_place_prepositions`, `test_qp_becomes_unverifiable_and_never_searched` |
| **4** | Luật an toàn | `query_chunking.py` | `test_original_query_always_first` ⭐; `test_falls_back_to_sentences_when_chunker_raises`; `test_respects_max_subqueries_cap` |
| **5** | Cắm vào Stage 1 | `search_service.py:253` | Sửa test đang khoá (§4.1) — **có chủ đích, ghi lý do trong commit** |
| **6** | Config | `configs/search/decomposition.yaml` | `test_config_loads` |

### 4.1 — Test sẽ phải sửa (behaviour-lock có chủ đích, không phải tai nạn)

| Test | Khoá gì | Xử lý |
|---|---|---|
| `test_search_service.py:543,648` | Câu đơn → **đúng 1** `encode_text` + 1 Milvus search | ⚠️ **Sẽ vỡ** — câu đơn giờ sinh ORIGINAL + các cụm. Phải viết lại kỳ vọng |
| `test_search_service.py:616-620` | Câu 2 mệnh đề → đúng 2 mảnh | ⚠️ **Sẽ vỡ** — sẽ ra nhiều hơn 2 |
| `test_search_service.py:652-697` | **Công thức SUM** cosine | ✅ Không vỡ nếu không đụng fusion. **Không được sửa test này để "cho qua"** |
| `test_search_service.py:777-880` | Thứ tự sort + giá trị số | ✅ Không vỡ (dùng `text_queries` tường minh, không qua decomposer) |
| `tests/test_query_decomposition.py` (11 test) | Regex hiện tại | ✅ Không vỡ — `split_into_sentences` giữ nguyên làm fallback |

> **Luật:** sửa test vì hành vi **cố ý** đổi thì được; sửa test để che một hồi quy thì không. Mỗi test sửa phải kèm lý do trong commit body (tinh thần CLAUDE.md §8: "không sửa bộ eval để cải thiện số").

---

## 5. Cách đo — điều kiện để tuyên bố "tốt hơn"

### 5.1 — ⚠️ Chặn cứng: kênh eval hiện tại KHÔNG đo được thứ này

`eval/run.py` chỉ đo `_search_semantic()`, mà hàm đó **không gọi** `split_into_sentences` (`search_service.py:922-926`). Nghĩa là **không có cách nào** đo decomposition bằng kênh eval hiện có — kể cả decomposition đã merge từ trước.

⇒ **Bước 0.5 của doc mẹ (dựng kênh eval đo `retrieve_kis_candidates` thật) là điều kiện tiên quyết**, không phải việc làm sau. Đây là câu hỏi C còn treo (§7).

### 5.2 — Đo chất lượng chunker RIÊNG, trước khi đo retrieval

Tách hai giả thuyết dễ bị gộp: *"chunker chạy đúng"* khác *"chunker giúp tăng recall"*.

```
Trên 24 truy vấn BTC duy nhất:
  - % câu chunk ra ít nhất 1 EVENT hợp lệ
  - % cụm NP/VP đúng khi soi tay (mẫu 50 cụm)   ← đối chiếu badge F1 77%
  - phân bố số subquery/truy vấn (kiểm trần §3.7 có bị chạm không)
  - độ dài token mỗi subquery sau khi dịch      ← còn mảnh nào > 40 token không
```

Nếu tỉ lệ thứ hai thấp hơn ~70%, dừng lại — chunker lệch domain quá nặng, không đáng đi tiếp.

### 5.3 — Ablation retrieval

```
baseline : split_into_sentences hiện tại
+ chunk  : underthesea typed subquery
```

Metric: `Recall@K` **và** `MRR` (segment + video level, PRD §6.4). Recall một mình không đủ — §3.6 giải thích sàn recall không kéo theo sàn ranking.

Mỗi báo cáo kèm: git SHA · `eval_set_version` + hash bộ nhãn · số `video_id` duy nhất · số truy vấn · seed (CLAUDE.md §8).

### 5.4 — Đo latency, vì nó sẽ tăng

Mỗi subquery = 1 lần gọi Milvus. Đo p50/p95 của `retrieve_kis_candidates` trước/sau, đối chiếu ngân sách latency KIS (PRD §1.2). Đây là lý do trần §3.7 phải có ngay từ đầu, không để sau.

**Điểm cộng phương pháp luận:** cách này **không** dùng LLM ⇒ không dính rủi ro "model sinh nhãn ≠ model bị đánh giá" (CLAUDE.md §3) — bộ eval là nhãn Gemini tự sinh, còn chunker thì không biết gì về Gemini.

---

## 6. Rủi ro đã nhận diện

| # | Rủi ro | Mức | Giảm thiểu |
|---|---|---|---|
| 1 | Chunker F1 ~77%, lệch domain (văn viết chuẩn → văn mô tả cảnh quay) | **Cao** | §3.6 luật ORIGINAL #0 (sàn recall) + §5.2 đo riêng trước khi tin |
| 2 | SUM scoring: nhiều mảnh rác đẩy candidate nhiễu lên | **Cao** | Trần §3.7 + đo **MRR** không chỉ Recall (§5.3) |
| 3 | Latency nhân theo số subquery | Trung bình | Trần `max_subqueries: 6`, đo p95 (§5.4) |
| 4 | Bật decomposition ⇒ tắt đóng góp xếp hạng OCR/Title (`:451-453`) | Trung bình | Chưa có cách sửa gọn; **phải ghi vào báo cáo**, không giấu |
| 5 | `underthesea_core` không có wheel cp310-win → build fail | Trung bình | Kiểm ngay bước 1; fail thì fallback `sent_tokenize`-only hoặc xét lại |
| 6 | Giả định `QP`/`TP` sai (§1.4) | Thấp | Bước 1 in tagset thật, đóng `[Cần verify]` trước khi code §3.5 |
| 7 | Thêm dependency giữa mùa thi | Thấp | Base install nhẹ, thuần Python, có fallback; đã được duyệt |

---

## 7. Câu hỏi còn treo (chưa chốt, doc này KHÔNG giả định đã chốt)

| # | Câu hỏi | Ảnh hưởng tới doc này |
|---|---|---|
| **A** | Chạy thí nghiệm `walk_short_pos` (248 token) trước? | Nếu đường 248 dùng được, động cơ "chia nhỏ vì trần token" yếu đi — nhưng §1.4 doc mẹ cho thấy decompose **vẫn** đáng làm vì CLIP chỉ chú ý ~20 token đầu. Thiết kế này **không đổ** nếu A cho kết quả tích cực |
| **C** | Dựng kênh eval Stage 1 trước hay code trước? | ⭐ **Chặn cứng §5.** Code xong mà không đo được thì không biết có tốt hơn không |
| **D** | Trần số subquery bao nhiêu? | §3.7 đề xuất `6` — **con số này chưa có cơ sở đo**. Cần chốt hoặc chấp nhận là giá trị khởi điểm để tinh chỉnh |
| **E** | Sửa 3 chỗ docs lệch code (§2.2 doc mẹ) ngay? | Chỗ #1 (SUM vs MAX trong docstring `query_decomposition.py:16-17`) nằm đúng file đang đụng ⇒ **nên sửa trong cùng change này** (CLAUDE.md §2) |

---

## 8. Nguồn

1. [underthesea trên PyPI](https://pypi.org/project/underthesea/) — bản 9.5.0, `requires_python >=3.10`, dependency lõi không gồm torch/transformers (fetch qua PyPI JSON API 2026-08-25).
2. [underthesea — readthedocs](https://underthesea.readthedocs.io/en/latest/readme.html) — API `sent_tokenize` / `word_tokenize` / `pos_tag` / `chunk` / `ner` và cấu trúc trả về.
3. [undertheseanlp/chunking](https://github.com/undertheseanlp/chunking) — tagset IOB2 21 nhãn, badge **F1 ≈ 77%**.
4. [VnCoreNLP (NAACL 2018)](https://aclanthology.org/N18-5012/) — phương án thay thế nếu chunking nông không đủ; mạnh hơn ở dependency parsing nhưng là Java.
5. [ABE-CLIP](https://arxiv.org/pdf/2512.17178) — cơ sở cho kênh `attribute_object` → `app/services/abe.py`.
6. [`research-query-decomposition-methods.md`](research-query-decomposition-methods.md) — doc mẹ, §4.2 là mục được triển khai ở đây; §1 chứa phát hiện FG-CLIP 248 token liên quan tới câu hỏi A.

---

## 9. Ghi chú độ tin cậy

- **§1 là sự thật đã fetch**, không phải trí nhớ — PyPI JSON API + readthedocs + GitHub repo, ngày 2026-08-25. Trừ hai chỗ đánh dấu `[Cần verify]`: ý nghĩa nhãn `QP`/`TP`/`MP`/`WH`/`WP`/`XP`/`N-NP` (§1.4) và wheel `underthesea_core` cho cp310-win (§1.2).
- **§1.2 (tương thích) đối chiếu working tree thật**: `pyproject.toml:5,123`, `python 3.10.11`, `transformers 4.44.2`, `huggingface_hub 0.36.2` — chạy lệnh, không suy đoán.
- **§2 lấy từ trace code đã thực hiện**, mọi ràng buộc có `file:line`.
- **§3–§4 là thiết kế, CHƯA code, CHƯA đo.** Con số `max_subqueries: 6` (§3.7) và danh sách giới từ nơi chốn (§3.4) là **lựa chọn khởi điểm chưa có cơ sở đo**, không phải kết quả thực nghiệm.
- **Không có con số Recall/MRR nào** — và theo §5.1 thì hiện **chưa có kênh eval nào đo được**. Mọi tuyên bố "tốt hơn" chỉ có nghĩa sau khi câu hỏi C được chốt và kênh đó tồn tại.
- F1 77% ở §1.5 là badge của repo chunking trên **domain văn viết chuẩn**, không phải trên đề BTC. Coi là **trần trên**, không phải kỳ vọng.

# Research — Phương pháp query decomposition không phụ thuộc LLM API

**Ngày:** 2026-08-25 · **Nhánh:** `feat/llm-query-translator` · **Người hỏi:** chủ dự án
**Câu hỏi gốc:** decomposition hiện tại chỉ tách câu theo `.` `!` `?` / xuống dòng — đề xuất 3–5 phương pháp hiệu quả hơn, ưu tiên rule-based / NLP nhẹ / local model vì hết quota Gemini.

**Ràng buộc chủ dự án đặt ra:** Stage 1 dùng FG-CLIP giới hạn 77 token · mục tiêu chính là **tăng Recall Stage 1** · subquery search độc lập rồi OR-union · query BTC chứa nhiều event theo thời gian + subject/action/attribute/object/location + distinctive clue.

**Quan hệ với doc khác:** [`research-query-refinement.md`](research-query-refinement.md) trả lời *"refine thành một query hay nhiều query"* và thiết kế `QueryCompiler` **dùng Gemini**. File này trả lời câu hỏi kế tiếp: *decompose bằng gì khi **không** có quota LLM*.

> **Công cụ:** MCP `firecrawl`/`exa` mà skill `deep-research` khuyến nghị **không có** trên máy này — dùng `WebSearch`/`WebFetch` thay thế, coverage có thể hẹp hơn. §1 và §2 **không** đến từ web mà từ đọc trực tiếp checkpoint + code trong repo.

---

## 1. ⚠️ Phát hiện đảo tiền đề: trần 77 token là do CONFIG, không phải do model

Phải xử lý **trước** mọi phương pháp decomposition, vì nó có thể xoá phần lớn lý do decompose.

### 1.1 — Checkpoint local đã có sẵn đường 248 token

`models/fgclip_base/modeling_clip.py:1155-1163` — lớp text embedding của FG-CLIP giữ **ba** bảng position embedding, không phải một:

```python
self.position_embedding     = nn.Embedding(config.max_position_embeddings, embed_dim)  # 77
self.position_embedding_res = nn.Embedding(248, embed_dim)
self.position_embedding_ori = nn.Embedding(248, embed_dim)
self.mask1 = torch.zeros([248, 1]); self.mask1[:20, :] = 1   # 20 vị trí đầu giữ nguyên
self.mask2 = torch.zeros([248, 1]); self.mask2[20:, :] = 1   # phần còn lại nội suy
```

`forward()` chọn nhánh theo cờ `walk_short_pos` (`:1185-1192`): `True` → bảng 77; `False` → tổ hợp hai bảng 248. Trọng số của cả hai bảng **đã nằm trong `model.safetensors`** trên đĩa — không cần tải thêm gì.

Khớp với paper: [FG-CLIP](https://arxiv.org/html/2505.05071v2) mở rộng text encoder lên **248 token** bằng nội suy tuyến tính (giữ 20 vị trí đầu, nội suy ×4 phần sau), huấn luyện với cả short + long caption.

### 1.2 — Repo đang khoá ở nhánh 77 tại bốn tầng

| Tầng | Vị trí | Giá trị |
|---|---|---|
| Config | `configs/model/fg_clip.yaml:13` | `max_text_tokens: 77` |
| Factory | `app/encoders/factory.py:84` | `config.get("max_text_tokens", 77)` |
| Tokenizer | `app/encoders/fg_clip.py:227-229` | `max_length=77`, `truncation=True` |
| **Model call** | `app/encoders/fg_clip.py:182` | `get_text_features(..., walk_short_pos=True)` |

Dòng cuối quan trọng nhất: kể cả nâng `max_text_tokens` lên 248, `walk_short_pos=True` vẫn đọc bảng 77 ⇒ sai position embedding hoặc `IndexError`. Phải đổi **cả hai**, không chỉ config.

Truncation xảy ra ở `_tokenize()` (`app/encoders/fg_clip.py:224-232`) — tầng tokenizer HuggingFace của chính encoder, **cắt im lặng, không log/raise/warning nào**.

### 1.3 — 🛑 Cạm bẫy: `walk_short_pos=False` đổi luôn PROJECTION HEAD

Chỗ dễ hỏng âm thầm đúng nghĩa CLAUDE.md §3. `models/fgclip_base/modeling_fgclip.py:186-189`:

```python
if walk_short_pos:
    text_features = self.text_projection(pooled_output)        # không gian A
else:
    text_features = self.text_filip_projection(pooled_output)  # không gian B — KHÁC
```

Trong khi corpus vector sinh bằng `get_image_features` → `self.visual_projection` (`modeling_fgclip.py:111`), tức **không gian A**.

**Hệ quả:** flip `walk_short_pos=False` để lấy 248 token đẩy query vector sang `text_filip_projection`. Nếu head này không đồng không gian với `visual_projection`, hệ thống vẫn chạy, vẫn trả 100 kết quả, vẫn không raise — chỉ là **toàn bộ ranking vô nghĩa**. Đúng lớp lỗi CLAUDE.md §3 liệt kê đầu tiên.

FG-CLIP dùng `text_filip_projection` cho căn chỉnh fine-grained với **dense image features**, nhưng paper cũng mô tả long-caption loss căn chỉnh long text với global image. **Hai cách hiểu mâu thuẫn, không giải quyết được bằng cách đọc paper.** Phải đo.

**⚠️ [Cần đo] — thí nghiệm rẻ, chạy trước mọi việc ở §4:**

```
Lấy ~20 keyframe đã có vector trong Milvus + caption tiếng Anh tương ứng.
Encode mỗi caption 2 cách:
  (a) walk_short_pos=True,  max_length=77   ← đường hiện tại
  (b) walk_short_pos=False, max_length=248  ← đường dài
So cosine(text_vec, image_vec_của_chính_keyframe_đó):
  (b) ≈ (a) hoặc cao hơn      ⇒ cùng không gian, đường 248 dùng được
  (b) rơi về ~0 / ngẫu nhiên  ⇒ khác không gian, KHÔNG dùng được, phải decompose thật
```

Chi phí: một script chẩn đoán, vài phút GPU. Giá trị: quyết định luôn §4.1/§4.4 có cần thiết không.

### 1.4 — Kể cả 248 token chạy được, decomposition VẪN có giá trị riêng

Đừng vội kết luận "mở 248 token là xong". [CLIP Is Shortsighted (arXiv:2602.22419)](https://arxiv.org/abs/2602.22419) đo được: text encoder họ CLIP **dồn attention vào câu đầu và token đầu**, làm yếu alignment phần còn lại — tác giả gọi là "shortcut" hình thành lúc train. [Long-CLIP](https://arxiv.org/html/2403.15378v1) đo độ dài **hiệu dụng** của CLIP chỉ ~**20 token**, dù cửa sổ danh nghĩa 77.

Ghép lại: một truy vấn 200 token đưa vào một lần thì **phần đuôi gần như không đóng góp vào vector** — kể cả khi không bị cắt.

> Trần token chỉ là **triệu chứng**. Bệnh thật là CLIP không phân bổ đều attention theo chiều dài. Decomposition sửa bệnh; nâng trần token chỉ sửa triệu chứng.

Đây là lập luận độc lập ủng hộ tiếp tục decompose, **và** là lý do mỗi subquery nên **ngắn** (~20–40 token), không phải "vừa đủ 77".

---

## 2. Hiện trạng code — đã trace, có ba chỗ lệch docs

### 2.1 — Đường đi thật của một truy vấn KIS Stage 1

```
text_query ──split_into_sentences()──> [câu₁, câu₂, …]        search_service.py:253
           ──_normalize_stage1_text_queries()──> dedupe, giữ thứ tự      :254-256, :1066-1076
           ──_prepare_text_inputs()──> dịch VI→EN TỪNG mảnh riêng        :301, :928-935
           ──encoder.encode_text()──> 1 lần batch cho cả list            :302
           ──milvus.search()──> N lần gọi RIÊNG, song song qua gather    :303-308
           ──_stage1_text_candidate_union()──> OR theo (video_id, frame_idx)  :1119-1147
           ──_stage1_text_scores_for_union()──> điểm hiển thị            :1150-1200
```

**Thứ tự là TÁCH → DỊCH**, không phải dịch → tách. Đúng thiết kế: dịch cả câu dài rồi mới cắt sẽ mất nội dung im lặng.

**Không có trần số subquery** ở bất kỳ đâu — `_normalize_stage1_text_queries`, `KisConfig` (`kis_scoring.py:47-63`), `configs/search/search.yaml` đều không giới hạn. Mỗi subquery = 1 lần gọi Milvus riêng ⇒ **latency tăng tuyến tính theo số mảnh, không ai chặn**.

### 2.2 — Ba chỗ code lệch docs (phải sửa trong cùng change, CLAUDE.md §2)

| # | Docs nói | Code thật | Ảnh hưởng |
|---|---|---|---|
| 1 | `query_decomposition.py:16-17` — union "by keeping the **highest** cosine score" | `search_service.py:1181` — `fused_score += raw_cosine`, tức **SUM** | Docstring sai. SUM nghĩa là candidate khớp **nhiều mảnh** xếp trên candidate khớp **một mảnh rất mạnh** — hành vi khác hẳn MAX, và đó là hành vi test đang khoá (`test_search_service.py:652-697`) |
| 2 | `research-query-refinement.md:90` — "Stage 1 chỉ nhận MỘT `text_query`, không phải `Sequence`" | `search_service.py:239` — `text_queries: Sequence[str]` đã có; `app/schemas/request/search.py:94` đã có `text_queries: list[str]` **từ trước nhánh này** | Doc mô tả state cũ. Việc "#1 Stage 1 nhận Sequence" trong bảng §6 doc đó **đã xong rồi**, không còn là việc cần làm |
| 3 | CLAUDE.md §5 — "`BEiT3Encoder` đang wire trong `app/main.py`" | `app/main.py:64-65` build **FG-CLIP** | Đã được ghi nhận là stale ở `research-query-refinement.md:207`, vẫn chưa sửa. Cần chủ dự án xác nhận trước khi sửa CLAUDE.md |

### 2.3 — Hai hành vi bất ngờ, không phải bug nhưng phải biết trước khi đụng vào

- **Chỉ `text_query` được tách.** `text_queries[i]` do operator gõ trực tiếp vào nhiều ô **không** đi qua `split_into_sentences` (`:253` chỉ nhận `text_query`). `ocr_query`, `title_query`, `asr_query` cũng không. Nghĩa là operator gõ một câu dài nhiều mệnh đề vào ô thứ hai thì nó **không** được cứu khỏi truncation.
- **Có `text_queries` là geometric-mean giữa text/ocr/title bị bỏ qua hoàn toàn** (`:451-453`). OCR/Title vẫn được ghi vào `Stage1RetrievalScore` và `source_modalities`, nhưng **không ảnh hưởng `display_score` hay thứ tự sort**. Tức là: bật decomposition = tắt đóng góp xếp hạng của OCR/Title ở Stage 1. Đây là hệ quả kiến trúc chưa ai đo.

### 2.4 — Điểm cắm và cái giá phải trả

**Điểm cắm duy nhất:** `search_service.py:253` — thay `split_into_sentences(text_query)` bằng bất kỳ hàm nào trả `list[str]`. Toàn bộ union / fuse / dedupe phía sau **không cần sửa gì**.

**Không có Protocol nào cho decomposer** — `split_into_sentences` là hàm thuần, không phải class implement interface. Muốn subquery **có kiểu** (event / attribute-object / location / clue) thì phải thiết kế schema mới và map vào các kênh sẵn có; đây là khoảng trống kiến trúc thật.

**Test sẽ vỡ khi đổi decomposer** (đây là behaviour-lock có chủ đích, không phải tai nạn):

| Test | Khoá gì |
|---|---|
| `test_search_service.py:543,648` | Câu đơn → **đúng 1** `encode_text` + **1** Milvus search |
| `test_search_service.py:616-620` | Câu 2 mệnh đề → **đúng 2** mảnh, **2** Milvus search |
| `test_search_service.py:652-697` | **Công thức SUM** cosine, giá trị số cụ thể |
| `test_search_service.py:777-880` | Thứ tự sort + `score == 2*(0.5**0.5)` |
| `tests/test_query_decomposition.py` (11 test) | Hành vi regex hiện tại |

### 2.5 — Dependency hiện có (quyết định phương pháp nào cắm được)

`pyproject.toml`: **không có** `underthesea`, `pyvi`, `VnCoreNLP`, `spacy`, PhoBERT — grep 0 match. Pin liên quan: `torch==2.4.1`, `transformers==4.44.2` (đường production `beit3`/`fg-clip`); `transformers>=4.49` chỉ trong extra `encoder-candidates` (venv riêng, xung khắc `timm==0.4.12`).

⇒ Mọi phương pháp cần thư viện NLP tiếng Việt đều là **thêm dependency mới, phải hỏi** (CLAUDE.md §10).

---

## 3. Vì sao tách theo dấu câu là chưa đủ

`app/services/query_decomposition.py:35` hiện dùng đúng một regex:

```python
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
```

Ba lớp thất bại, đối chiếu thống kê §3.3 của [`research-query-refinement.md`](research-query-refinement.md):

| Lớp thất bại | Ví dụ trong đề BTC | Vì sao regex trượt |
|---|---|---|
| **Nhiều event trong MỘT câu** | "…bắt đầu bằng cảnh X, **sau đó** Y, **cuối cùng** Z." | Không có `.` giữa các event ⇒ 3 event dồn vào 1 subquery |
| **Câu đơn nhưng quá tải** | 1 câu 90 token: subject + action + 3 attribute + location | Không có ranh giới nào ⇒ trả nguyên câu, vẫn bị cắt |
| **Distinctive clue chìm giữa câu** | "…chén nước chấm nhỏ với 2 miếng ớt" ở mệnh đề 3 | Clue quý nhất trộn chung ngữ cảnh nhiễu, loãng vector |

15/24 truy vấn duy nhất có chuỗi thời gian nhiều cảnh; 14/24 có gắn thuộc tính vào vật thể — chính là hai lớp đầu.

---

## 4. Năm phương pháp đề xuất

Xếp theo **tỉ lệ giá trị / rủi ro**, không theo độ "thông minh".

### 4.1 — Tách mệnh đề theo liên từ thời gian tiếng Việt ⭐ làm trước

**Ý tưởng:** mở rộng regex từ "dấu kết câu" sang "dấu kết câu **+ liên từ trình tự**". Đề BTC viết theo văn phong tường thuật rất đều, mốc event gần như luôn được đánh dấu bằng một tập từ nối hữu hạn:

`đầu tiên` · `ban đầu` · `mở đầu` · `bắt đầu bằng` · `sau đó` · `tiếp theo` · `tiếp đến` · `rồi` · `kế tiếp` · `sau cùng` · `cuối cùng` · `kết thúc bằng` · `trong khi` · `đồng thời` · `khi` · `lúc`

**Vì sao hợp:**
- Đánh thẳng lớp thất bại #1 (15/24) — lớp lớn nhất.
- Mỗi mệnh đề tách ra là một `event` ứng viên, **dùng lại được cho `TrakeService`/DANTE** mà không phải viết decomposer thứ hai.
- Ranh giới tách giữ nguyên thứ tự thời gian ⇒ không phá bất biến `τ < t` của DANTE.

| | |
|---|---|
| **Dependency mới** | Không |
| **Chi phí runtime** | ~0 (regex) |
| **Rủi ro** | Thấp. Từ nối đa nghĩa (`rồi` cũng là trợ từ "xong rồi") ⇒ chỉ tách khi từ nối đứng **đầu mệnh đề** sau dấu phẩy, dùng word-boundary như `GlossaryCorrectedTranslator` đã làm (`query_translator.py:347`) |
| **Test phải sửa** | `test_query_decomposition.py`, `test_search_service.py:616-620` |

**Kiểm chứng trước khi code (10 phút):** đếm trên 24 truy vấn BTC xem bao nhiêu câu chứa ít nhất một từ nối trong tập trên, và sau khi tách còn bao nhiêu mảnh > 40 token. Con số thứ nhất thấp ⇒ phương pháp không đáng làm, biết ngay, không phải sau khi code xong.

---

### 4.2 — Chunking cú pháp qua `underthesea` → subquery CÓ KIỂU ⭐⭐ giá trị cao nhất

> ✅ **ĐÃ DUYỆT 2026-08-25** (câu hỏi B §6). Thiết kế chi tiết + kế hoạch TDD: [`research-underthesea-decomposition.md`](research-underthesea-decomposition.md).
> Hai số đã xác minh sau khi duyệt, **sửa lại đánh giá rủi ro bên dưới**: underthesea `9.5.0` base install **không** kéo torch/transformers ⇒ rủi ro dependency thấp hơn dự kiến; nhưng chunker chỉ đạt **F1 ≈ 77%** trên văn viết chuẩn ⇒ rủi ro chất lượng **cao hơn** dự kiến.

**Ý tưởng:** phương pháp thật sự trả lời "hiểu được event/action/attribute" mà không cần LLM. [`underthesea`](https://underthesea.readthedocs.io/en/latest/readme.html) là toolkit NLP tiếng Việt thuần Python, `pip install underthesea`, chạy CPU, có sẵn word segmentation · POS tagging · **chunking** (`NP` danh ngữ · `VP` động ngữ · `PP` giới ngữ · `AP` tính ngữ).

Từ chuỗi chunk, dựng subquery **có kiểu** thay vì mảnh văn bản vô danh:

| Kiểu subquery | Dựng từ | Ví dụ | Kênh đích |
|---|---|---|---|
| `event` | `NP + VP` liền kề | "người đàn ông cầm ô bước qua đường" | Stage 1 semantic + DANTE |
| `attribute_object` | `NP` có `AP` bổ nghĩa | "chén nước chấm nhỏ", "áo dài vàng" | Stage 1 semantic + **ABE** (`app/services/abe.py` đã có) |
| `location` | `PP` chỉ nơi chốn | "trên sân khấu", "trước cổng chợ" | Stage 1 semantic |
| `distinctive_clue` | `NP` có IDF cao (§4.3) | "kẻng đồng", "sơ đồ ba tầng" | Stage 1 semantic + OCR/BM25 |

**Vì sao mạnh hơn §4.1:** giải lớp thất bại #2 và #3 — câu đơn quá tải bị xé thành danh ngữ/động ngữ ngắn; distinctive clue tách **riêng thành subquery độc lập** thay vì chìm giữa ngữ cảnh. Mỗi subquery rơi vào vùng ~20 token — đúng vùng hiệu dụng của CLIP (§1.4).

**Điểm cộng kiến trúc:** `abe_pairs` trong thiết kế `CompiledQuery` vốn định để Gemini sinh. Chunker sinh được `attribute_object` **offline** ⇒ mở khoá `app/services/abe.py` không cần quota. Đây là phần "cứu" được nhiều nhất từ thiết kế Query Compiler đã bỏ dở.

| | |
|---|---|
| **Dependency mới** | `underthesea` (thuần Python, CPU, không đụng VRAM) — ⚠️ **cần chủ dự án duyệt** (CLAUDE.md §10) |
| **Chi phí runtime** | Vài chục ms/query trên CPU, không tranh GPU với FG-CLIP |
| **Rủi ro** | Trung bình. Chunker train trên văn viết chuẩn; đề BTC là văn mô tả, có thể lệch domain. **Phải đo accuracy chunk trên chính 24 đề BTC trước khi tin** |
| **Ràng buộc thứ tự** | Phải chunk trên **tiếng Việt** rồi mới dịch từng mảnh — giữ đúng thứ tự "tách → dịch" hiện có (§2.1), không đảo |

**Lựa chọn thay thế:** [`VnCoreNLP`](https://github.com/vncorenlp/VnCoreNLP) ([NAACL 2018](https://aclanthology.org/N18-5012/)) mạnh hơn ở **dependency parsing** (rút quan hệ subject–verb–object chính xác hơn chunking nông), nhưng là **Java**, phải chạy JVM subprocess — thêm điểm hỏng vận hành giữa lúc thi. Chunking nông của underthesea là đủ; chỉ nâng lên nếu đo thấy chunk sai nhiều.

---

### 4.3 — Distinctive-clue subquery theo IDF, tận dụng Elasticsearch đã có

**Ý tưởng:** không phải mọi mảnh đều đáng search. Trong "một người phụ nữ mặc áo dài đứng cạnh **cây lục bình**", cụm `cây lục bình` mang gần như toàn bộ khả năng phân biệt; `một người phụ nữ` khớp hàng chục nghìn keyframe. Dùng **IDF tính từ index Elasticsearch đang có** chấm điểm hiếm cho từng danh ngữ, rồi phát **một subquery riêng chỉ gồm clue hiếm nhất**.

Cơ sở: IDF `log(N/df(t))` đề cao đúng term hiếm, và dense retrieval **không có cơ chế tương đương** — term xuất hiện trong 3/50.000 document được BM25 đẩy lên đầu, còn embedding thì làm loãng nó vào vector trung bình.

**Điểm cộng:** định tuyến tự nhiên — clue IDF rất cao thường là chữ trên màn hình hoặc tên riêng ⇒ đẩy sang kênh **OCR/BM25** thay vì semantic, đúng phân luồng 7/24 truy vấn "neo chữ trong khung hình".

| | |
|---|---|
| **Dependency mới** | Không (dùng `app/stores/es_client.py` sẵn có) |
| **Rủi ro** | Thấp–trung bình. IDF tính trên corpus tiếng Việt hay tiếng Anh? — phải thống nhất với thứ đang index, nếu không IDF vô nghĩa |
| **Phụ thuộc** | Cần §4.2 (hoặc ít nhất bộ tách danh ngữ) để có ứng viên chấm điểm |
| **⚠️ Tương tác** | Xem §2.3 — bật decomposition làm OCR/Title mất đóng góp xếp hạng ở Stage 1; phương pháp này đẩy clue **vào** OCR nên hai hiệu ứng có thể triệt tiêu nhau. **Phải đo cùng nhau, không đo riêng** |

---

### 4.4 — Cửa sổ trượt theo ngân sách token, có chồng lấn (lưới an toàn)

**Ý tưởng:** sau §4.1/§4.2 vẫn có thể còn mảnh vượt trần. Thay vì `truncation=True` cắt cụt im lặng, cắt thành nhiều cửa sổ ≤ ~40 token **chồng lấn ~25%** để không ranh giới nào xé đôi một cụm từ.

**Vì sao cần:** phương pháp duy nhất **đảm bảo bằng cấu trúc** rằng không token nào bị vứt — hai phương pháp trên phụ thuộc văn phong đề, cái này thì không.

| | |
|---|---|
| **Dependency mới** | Không |
| **Rủi ro** | Thấp về recall, nhưng **tăng số lần gọi Milvus** ⇒ latency. Hiện **không có trần số subquery** (§2.1) — phải thêm trần cùng lúc |
| **Vai trò** | Fallback, **không** phải phương pháp chính — cắt máy móc tạo mảnh không tự nhiên, chất lượng kém hơn §4.2 |

---

### 4.5 — Local LLM thay Gemini (Qwen quantized qua llama.cpp)

**Ý tưởng:** giữ nguyên thiết kế `QueryCompiler` nhưng đổi backend sang model local, xoá ràng buộc quota.

**Vì sao cân nhắc:** phương án duy nhất lấy lại **đầy đủ** thiết kế `CompiledQuery` — gồm `unverifiable` (đếm/phủ định) và `dropped` (từ ngữ máy quay), hai trường mà rule-based và chunking **không** làm được vì đòi hiểu ngữ nghĩa phủ định và ý đồ quay phim.

**Vì sao xếp cuối:**
- **VRAM.** RTX 3070 8GB đang nuôi FG-CLIP. Nhét thêm LLM 4B quantized vào cùng GPU ⇒ tranh chấp đúng lúc bị tính giờ. CLAUDE.md §1.4: việc GPU local "chạy tuần tự an toàn hơn song song".
- **Chất lượng tiếng Việt chưa biết.** Không tìm được số đo Vietnamese-specific nào cho Qwen 4B ở mức quantization vừa 8GB — **[Cần đo]**, không giả định.
- **Latency.** Một lượt sinh JSON có cấu trúc trên GPU chia sẻ, giữa lúc thi, là rủi ro vận hành thật.

| | |
|---|---|
| **Dependency mới** | `llama-cpp-python` / `llama-server` + GGUF vài GB — **dependency lớn, phải hỏi** |
| **Rủi ro** | Cao (VRAM + chất lượng + latency, cả ba chưa đo) |
| **Khi nào đáng làm** | Chỉ khi §4.1–§4.3 đã đo và vẫn hụt mục tiêu PRD §6.7 |

---

## 5. Khuyến nghị — thứ tự thực hiện

**Không làm cả năm.** Thứ tự dưới tách được từng giả thuyết, đúng CLAUDE.md §8:

| Bước | Việc | Chi phí | Vì sao đặt ở đây |
|---|---|---|---|
| **0** | **Thí nghiệm `walk_short_pos` (§1.3)** | vài phút GPU | Nếu đường 248 token dùng được, phần lớn động cơ decompose-vì-truncation biến mất. Rẻ nhất, thông tin nhiều nhất |
| **0.5** | **Dựng kênh eval đo `retrieve_kis_candidates` thật** | 1 task | ⚠️ **Chặn cứng.** `eval/run.py` hiện chỉ đo `_search_semantic()` — hàm này **không** gọi `split_into_sentences` (§2.1), nghĩa là kênh eval hiện tại **không thể** đo decomposition dù chỉ một lần |
| **1** | §4.1 — liên từ thời gian | nhỏ | Lớp thất bại lớn nhất (15/24), 0 dependency |
| **2** | §4.2 — chunking underthesea | trung bình | Giá trị cao nhất; cần duyệt dependency + đo accuracy chunk trước |
| **3** | §4.3 — IDF distinctive clue | nhỏ | Cần §4.2 làm đầu vào; đo kèm hiệu ứng §2.3 |
| **4** | §4.4 — sliding window + trần số subquery | nhỏ | Chỉ bật cho mảnh còn vượt trần sau 1–3 |
| **5** | §4.5 — local LLM | lớn | Chỉ khi 1–4 đã đo và vẫn hụt |

**Bước 0.5 là điều kiện tiên quyết thật sự, và nghiêm trọng hơn tưởng.** `_search_semantic()` — hàm duy nhất mà `eval/run.py` đo — **không đi qua đường decomposition**. Nghĩa là ba fix đã merge trên nhánh này (`GeminiTranslator`, `GlossaryCorrectedTranslator`, tách câu Stage 1) không chỉ *chưa* được đo, mà *không thể* được đo bằng kênh eval hiện có. Thêm phương pháp thứ tư vào đó chỉ làm dày phần chưa kiểm chứng.

**Ablation tối thiểu** (mỗi dòng bật thêm đúng một thứ):

```
baseline   : tách câu hiện tại (đã merge, CHƯA đo, KHÔNG đo được)
+ §4.1     : + liên từ thời gian
+ §4.2     : + chunking có kiểu
+ §4.3     : + distinctive clue theo IDF
```

Mỗi báo cáo kèm: git SHA · `eval_set_version` + hash bộ nhãn · số `video_id` duy nhất · số truy vấn · seed (CLAUDE.md §8).

**Điểm cộng chưa ai nói ra:** bộ eval là nhãn Gemini tự sinh (PRD §6.1). Bốn phương pháp §4.1–§4.4 **không** dùng LLM ⇒ **không** dính rủi ro "model sinh nhãn ≠ model bị đánh giá" (CLAUDE.md §3). Đây là ưu thế phương pháp luận thật của hướng rule-based, không chỉ là giải pháp tình thế vì hết quota. §4.5 thì dính, phải xử lý riêng.

---

## 6. Câu hỏi cần chủ dự án chốt

| # | Câu hỏi | Vì sao cần chốt trước khi code |
|---|---|---|
| A | Chạy thí nghiệm `walk_short_pos` (§1.3) trước, hay bỏ qua và decompose luôn? | Kết quả có thể xoá phần lớn động cơ của §4.1/§4.4 |
| ~~B~~ | ~~Duyệt thêm dependency `underthesea`?~~ | ✅ **ĐÃ DUYỆT 2026-08-25** → [`research-underthesea-decomposition.md`](research-underthesea-decomposition.md) |
| C | Dựng kênh eval Stage 1 thật (bước 0.5) trước, hay code decomposition trước rồi đo sau? | Đo sau = không biết cái nào đóng góp gì — đúng lỗi mà `eval/` sinh ra để tránh |
| D | Trần số subquery/query là bao nhiêu? | Hiện **không có trần nào** (§2.1); mỗi subquery = 1 Milvus search, ảnh hưởng latency KIS (PRD §1.2) |
| E | Sửa 3 chỗ docs lệch code ở §2.2 ngay, hay để cùng change sau? | CLAUDE.md §2 bắt sửa docs trong **cùng** change; #1 (SUM vs MAX) dễ khiến người sau hiểu sai hành vi ranking |

---

## 7. Nguồn tham khảo

1. [FG-CLIP: Fine-Grained Visual and Textual Alignment](https://arxiv.org/html/2505.05071v2) — mở rộng text encoder lên 248 token bằng nội suy position embedding (giữ 20 vị trí đầu, nội suy ×4 phần sau). Cơ sở §1.1.
2. [CLIP Is Shortsighted: Paying Attention Beyond the First Sentence](https://arxiv.org/abs/2602.22419) — CLIP dồn attention vào câu đầu, làm yếu alignment phần còn lại. Lập luận cốt lõi §1.4.
3. [Long-CLIP: Unlocking the Long-Text Capability of CLIP](https://arxiv.org/html/2403.15378v1) — độ dài **hiệu dụng** của CLIP chỉ ~20 token dù cửa sổ 77; kỹ thuật giữ-20-nội-suy-phần-sau mà FG-CLIP dùng lại.
4. [Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval](https://arxiv.org/abs/2506.10202) — decompose prequel/current/sequel, +8 NDCG trên MultiVENT. ⚠️ Dùng LLM ⇒ chỉ tham khảo **cấu trúc phân loại event**, không phải phương pháp triển khai ở đây.
5. [VnCoreNLP: A Vietnamese NLP Toolkit (NAACL 2018)](https://aclanthology.org/N18-5012/) — word segmentation · POS · NER · dependency parsing. Phương án thay thế §4.2.
6. [underthesea — Vietnamese NLP Toolkit](https://underthesea.readthedocs.io/en/latest/readme.html) · [chunking experiments](https://github.com/undertheseanlp/chunking) — chunking NP/VP/PP/AP thuần Python. Nền §4.2.
7. [Building a Semantic Role Labelling System for Vietnamese](https://axi.lims.ac.uk/paper/1705.04038) — quy SRL tiếng Việt về dependency parsing; nền lý thuyết cho rút subject–action–object.
8. [ABE-CLIP: Training-Free Attribute Binding Enhancement](https://arxiv.org/pdf/2512.17178) — chính là `app/services/abe.py` sẵn có; xác nhận `attribute_object` của §4.2 là kênh đúng.
9. [TULIP: Token-length Upgraded CLIP (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/54f117f842d47c2007b501bddd04c0e7-Paper-Conference.pdf) — hướng khác vượt trần 77 token bằng relative positional encoding; tham khảo, không áp dụng ngay vì phải train lại.

---

## 8. Ghi chú về độ tin cậy của file này

- **§1 và §2 là phần mạnh nhất** — đọc trực tiếp `models/fgclip_base/modeling_clip.py`, `modeling_fgclip.py`, `app/encoders/fg_clip.py`, `app/services/search_service.py`, `pyproject.toml` trên working tree, mọi khẳng định có `file:line`. **Nhưng** kết luận "đường 248 token có dùng được không" thì **chưa đo** — §1.3 nêu rõ hai cách hiểu mâu thuẫn và thí nghiệm cần chạy.
- **§4 là đề xuất thiết kế, chưa code, chưa đo.** Các con số 15/24, 14/24, 7/24 lấy từ [`research-query-refinement.md`](research-query-refinement.md) §3.3 (đã đo thật tại git `5191098`), nhưng **hiệu quả dự kiến của từng phương pháp là suy luận, không phải số đo**.
- **Không có con số Recall/MRR nào trong file này**, vì kênh eval đo Stage 1 thật chưa tồn tại — và đường eval hiện có (`_search_semantic`) không đi qua decomposition. Mọi tuyên bố "phương pháp X tốt hơn Y" chỉ có nghĩa sau bước 0.5.
- Nguồn §7 tra qua `WebSearch`/`WebFetch`, **không** phải `firecrawl`/`exa` (MCP không có trên máy) — coverage có thể hẹp hơn nếu chạy lại với đúng công cụ.

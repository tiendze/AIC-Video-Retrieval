# PRD — Hệ thống Truy vấn & Hỏi đáp Video Đa phương thức

> **Tên dự án:** AIC-Video-Retrieval
> **Phiên bản:** 0.1 (bản khởi tạo)
> **Ngày:** 2026-07-25
> **Trạng thái:** Draft — chờ review
> **Baseline:** [arXiv:2512.13169 — *Integrated Semantic and Temporal Alignment for Interactive Video Retrieval*](https://arxiv.org/abs/2512.13169) (team AIO_Owlgorithms, HCMC AI Challenge 2025)

---

## Quy ước đọc tài liệu

| Ký hiệu | Ý nghĩa |
|---|---|
| 📄 **[Paper]** | Thông tin trích trực tiếp từ paper baseline — đã kiểm chứng |
| 🔧 **[Giả định]** | Quyết định của dự án, paper **không** nêu — có thể thay đổi |
| ⚠️ **[Cần đo]** | Con số ước lượng, phải benchmark lại trên phần cứng thật |

---

## 1. Bối cảnh

### 1.1 Bài toán

Hội thi **AI Challenge TP.HCM (AIC)** yêu cầu xây dựng công cụ tìm kiếm sự kiện trong một kho video lớn. Ban tổ chức đưa ra truy vấn, thí sinh dùng công cụ tự phát triển để định vị đúng khoảnh khắc và nộp kết quả. Cuộc thi có 4 dạng nhiệm vụ:

| Task | Đầu vào | Đầu ra cần tìm |
|---|---|---|
| **Textual KIS** | Mô tả sự kiện bằng ngôn ngữ tự nhiên (vòng chung kết: lộ dần từng phần) | Đúng video + đúng khoảnh khắc |
| **Video KIS** | Một clip ngắn (≤ 20 giây) được chiếu, **không được ghi hình** | Đúng video + đúng khoảnh khắc |
| **QA (VQA)** | Câu hỏi về nội dung hình ảnh trong kho video | Câu trả lời + bằng chứng (keyframe) |
| **TRAKE** | Một chuỗi N sự kiện **theo thứ tự thời gian** | Đúng video + chuỗi N keyframe đúng thứ tự |

Đặc thù quyết định thiết kế: **điểm số giảm theo thời gian trả lời và theo số lần nộp sai**. Nghĩa là hệ thống không chỉ cần đúng — nó cần đúng *nhanh*, và giao diện phải cho phép người vận hành duyệt kết quả trong vài giây.

### 1.2 Hai hạn chế mà baseline nhắm tới

📄 **[Paper]** Paper baseline xác định hai điểm nghẽn của các hệ thống truy vấn video hiện có:

1. **Embedding model bị "đóng băng" (frozen)** — các model như CLIP/BEiT-3 chỉ biết những gì có trong dữ liệu huấn luyện. Truy vấn chứa thực thể mới (paper gọi là *Out-of-Knowledge — OOK*, ví dụ "Labubu", "Babythree") sẽ thất bại hoàn toàn vì model không có biểu diễn cho khái niệm đó. Paper giải quyết bằng module **QUEST**.

2. **Truy vấn theo chuỗi thời gian rất tốn kém** — tìm N sự kiện đúng thứ tự trong T keyframe theo cách vét cạn là tổ hợp. Paper giải quyết bằng thuật toán quy hoạch động **DANTE**, đưa độ phức tạp về `O(NT)`.

### 1.3 Khoảng trống của baseline mà dự án này phải lấp

Đây là phần quan trọng nhất của bối cảnh. Paper mô tả kiến trúc rất đầy đủ nhưng **không công bố bất kỳ số liệu định lượng nào**:

| Thiếu | Hệ quả |
|---|---|
| Không có Recall@k, MRR, mAP | Không biết hệ thống tốt đến đâu, không so sánh được thay đổi |
| Không có số liệu latency | Không có ngân sách thời gian để tối ưu |
| Không nêu thống kê dataset (số video, số keyframe) | Không ước lượng được chi phí hạ tầng |
| Không có ablation cho tham số `λ` của DANTE | Phải tự dò lại |
| Không nêu checkpoint BEiT-3 cụ thể | Rủi ro không tái lập được |
| Không so sánh với baseline nào khác | Không biết đóng góp thực sự đến từ đâu |

Paper chỉ báo cáo xếp hạng **định tính** ở vòng chung kết: TRAKE *Outstanding*, Textual KIS *Excellent*, Video KIS *Very Good*, QA *Very Good*.

→ **Mục §6 (Tiêu chuẩn đánh giá) của tài liệu này chính là phần lấp khoảng trống đó**, và là điều kiện tiên quyết để mọi tối ưu về sau có ý nghĩa.

### 1.4 Tình trạng dữ liệu hiện tại

Chưa có dữ liệu chính thức từ ban tổ chức. Giai đoạn 1 dùng **YouCook2** làm dataset thay thế:

| Thuộc tính | Giá trị |
|---|---|
| Số video | 2.000 video YouTube nấu ăn |
| Tổng thời lượng | 176 giờ |
| Số công thức (category) | 89 |
| Thời lượng trung bình | 5,26 phút/video |
| Segment trung bình mỗi video | 7,7 procedure segment |
| Caption | Tiếng Anh, dạng mệnh lệnh, ~8,8 từ |
| File nhãn | `youcookii_annotations_trainval.json` |
| Split | train / val / test — **nhãn của test bị giữ kín** |

**Vì sao YouCook2 phù hợp:** mỗi video được annotate sẵn một chuỗi *procedure segment có thứ tự thời gian*, kèm caption. Đây là ground truth miễn phí cho hai thứ khó kiếm nhất:
- Chuỗi segment có thứ tự → ground truth cho **TRAKE / DANTE**
- Caption từng segment → ground truth cho **Textual KIS**
- `recipe_type` → nhãn để kiểm thử bộ lọc metadata

**Ba điểm lệch so với dữ liệu AIC — phải ghi nhớ khi diễn giải kết quả:**

1. **Caption tiếng Anh**, còn AIC là tiếng Việt → `vi_analyzer` của Elasticsearch chưa phát huy tác dụng ở giai đoạn 1.
2. **Video nấu ăn gần như không có chữ trên màn hình**, khác hẳn video tin tức của AIC → nhánh OCR/Elasticsearch sẽ ở trạng thái *"đã xây nhưng chưa kiểm chứng được"*.
3. **Không có nhãn QA** → phải sinh bộ câu hỏi bằng LLM từ caption, kèm vòng review thủ công.

---

## 2. Mục tiêu

### 2.1 Mục tiêu nghiệp vụ

| ID | Mục tiêu | Đo bằng |
|---|---|---|
| **G1** | Tái hiện được kiến trúc baseline, chạy end-to-end trên dữ liệu thật | Hoàn tất một truy vấn Textual KIS từ UI ra kết quả |
| **G2** | Có bộ số đo tin cậy để mọi thay đổi đều chứng minh được là cải thiện | Chạy được `eval/` một lệnh, ra bảng metrics đầy đủ |
| **G3** | Hỗ trợ đủ 4 task thi (Textual KIS, Video KIS, QA, TRAKE) | Mỗi task có endpoint + đường xuất file nộp |
| **G4** | Người vận hành tìm ra đáp án đúng trong ngân sách thời gian thi | `Time-To-First-Correct` (xem §6.6) |

### 2.2 Mục tiêu kỹ thuật

| ID | Mục tiêu |
|---|---|
| **T1** | Pipeline ingest **idempotent và resume được** — đứt giữa chừng chạy lại không hỏng dữ liệu, không xử lý lại phần đã xong |
| **T2** | Backbone embedding **thay được** mà không sửa pipeline (qua `MultimodalEncoder` Protocol) |
| **T3** | Dataset **thay được** mà không sửa pipeline (qua `DatasetAdapter`) — chuyển YouCook2 → AIC chỉ cần thêm một adapter |
| **T4** | Toàn bộ tham số ảnh hưởng kết quả nằm trong file config, không hard-code |
| **T5** | Hệ thống vẫn phục vụ được truy vấn semantic khi mất mạng (Gemini/OpenAI không khả dụng) |

### 2.3 Ngoài phạm vi (Non-goals)

Ghi rõ để tránh trôi phạm vi:

- ❌ **Không** fine-tune / train lại backbone ở giai đoạn 1 — chỉ dùng checkpoint có sẵn
- ❌ **Không** làm multi-tenant, phân quyền, quản lý người dùng — đây là công cụ nội bộ chạy LAN
- ❌ **Không** xử lý kênh audio/ASR ở v1 (đưa vào phương án tối ưu giai đoạn 2)
- ❌ **Không** tự động hoá hoàn toàn — hệ thống là *interactive*, con người vẫn là người quyết định đáp án cuối
- ❌ **Không** tối ưu hạ tầng phân tán / Kubernetes — mục tiêu là chạy tốt trên **một máy**

---

## 3. Người dùng & User stories

### 3.1 Persona

| Persona | Mô tả | Điều họ quan tâm nhất |
|---|---|---|
| **P1 — Người vận hành khi thi** (chính) | Thí sinh ngồi trước màn hình lúc thi, đọc truy vấn của BTC và thao tác công cụ | Tốc độ. Mỗi giây đều mất điểm. Cần shortcut bàn phím, grid load nhanh, đổi mode tức thì |
| **P2 — Kỹ sư pipeline** | Người chạy ingest, quản lý index, theo dõi log | Ingest không đứt, resume được, biết đang ở đâu trong 176 giờ video |
| **P3 — Người đánh giá** | Người chạy benchmark, so sánh cấu hình | Chạy eval một lệnh, kết quả tái lập được, so sánh được giữa các lần chạy |

### 3.2 User stories

| ID | Story | Acceptance criteria |
|---|---|---|
| **US-1** | *P1:* Tôi gõ mô tả sự kiện bằng tiếng Việt và thấy ngay lưới keyframe liên quan | Kết quả trả về trong ngân sách §7.1; lưới hiển thị ≥ 50 keyframe kèm `video_id` và mốc thời gian |
| **US-2** | *P1:* Khi truy vấn chứa tên riêng lạ mà hệ thống không hiểu, tôi bấm một nút để hệ thống tự viết lại truy vấn | Nút "AI Enhance" trả về truy vấn đã viết lại; hiển thị **cả hai** phiên bản để người dùng chọn |
| **US-3** | *P1:* Tôi thấy một ảnh giống thứ cần tìm, tôi upload nó lên để tìm ảnh tương tự | Upload ảnh → image-to-image search dùng **chung** không gian embedding với text search |
| **US-4** | *P1:* Với TRAKE, tôi nhập N mô tả sự kiện theo thứ tự và nhận về chuỗi keyframe đúng thứ tự trong cùng một video | Trả về top-k video, mỗi video kèm đúng N keyframe, thứ tự thời gian không đảo |
| **US-5** | *P1:* Tôi hỏi một câu hỏi về nội dung và nhận câu trả lời **kèm keyframe làm bằng chứng** | Câu trả lời luôn đi kèm ≥ 1 keyframe nguồn; không có keyframe thì không trả lời |
| **US-6** | *P1:* Tôi chốt đáp án và xuất ra file đúng định dạng nộp | Xuất file theo format BTC; validate trước khi ghi |
| **US-7** | *P2:* Tôi chạy ingest, máy mất điện, tôi chạy lại và nó tiếp tục từ chỗ dở | Chạy lại không tạo bản ghi trùng; video đã xong bị bỏ qua |
| **US-8** | *P3:* Tôi chạy một lệnh và có bảng metrics đầy đủ để đưa vào báo cáo | `python -m eval.run --config ...` sinh ra bảng markdown + JSON |

---

## 4. Phạm vi theo giai đoạn

| Phase | Nội dung | Điều kiện hoàn thành (exit criteria) |
|---|---|---|
| **P0 — Nền móng** | Dựng `docker-compose` (Milvus + ES + MongoDB); `MultimodalEncoder` + `DatasetAdapter`; ingest **val split YouCook2 (457 video)** | Search một truy vấn bất kỳ ra kết quả từ đầu đến cuối; `eval/` chạy được và in ra Recall@k dù số còn thấp |
| **P1 — Tìm kiếm cơ bản** | Semantic search (text→image, image→image); OCR search qua ES; hybrid fusion (RRF); UI tối thiểu | Đạt mục tiêu Recall@10 trong §6.7; UI dùng được để thao tác thật |
| **P2 — Hai đóng góp của paper** | QUEST (2 nhánh); DANTE (TRAKE) + ablation `λ` | TRAKE `Sequence Exact-Match` đạt mục tiêu §6.7; có bảng ablation `λ` |
| **P3 — VQA & nộp bài** | VQA answering qua Gemini có grounding; submission exporter | `Grounded Accuracy` đạt mục tiêu §6.7; file nộp qua validate |
| **P4 — Dữ liệu thật & tối ưu** | Nạp dữ liệu AIC; kích hoạt `vi_analyzer`; đo lại toàn bộ; tối ưu theo số đo | Toàn bộ metrics chạy lại trên dữ liệu AIC; báo cáo so sánh |

**Nguyên tắc xuyên suốt:** không sang phase sau khi phase trước chưa có số đo. Đây là bài học trực tiếp từ khoảng trống của paper.

---

## 5. Tính năng

### F1 — Pipeline ingest *(P0)*

Biến video thô thành dữ liệu tìm kiếm được.

**Luồng** 📄 **[Paper]**:
1. **Shot segmentation** bằng **TransNetV2** — phát hiện điểm chuyển cảnh
2. **Trích keyframe**: 4 vị trí mỗi shot (đầu, hai điểm giữa, cuối), theo công thức paper:
   ```
   k_extract = { K_(a + ⌊i × (b − a) / 3⌋) , ∀i ∈ {0,1,2,3} }
   ```
   với `[a, b]` là khoảng frame của shot
3. **OCR** bằng **Gemini API** → `{keyframe_id: ocr_text}`
4. **Embedding** bằng **BEiT-3**, **L2-normalize**, ghi vào Milvus
5. **Index** song song vào 3 store (chi tiết schema xem `architecture.md`)

**Acceptance criteria:**
- [ ] Idempotent: chạy lại trên video đã xử lý → không tạo bản ghi trùng
- [ ] Resume: checkpoint theo `video_id`; đứt giữa chừng chạy lại tiếp tục đúng chỗ
- [ ] Ghi log video lỗi vào một file riêng, **không** làm dừng cả pipeline
- [ ] Gán `global_index` liên tục cho keyframe; ghi `s_v`, `e_v` cho mỗi video (**bắt buộc** — DANTE không chạy được nếu thiếu)

### F2 — Semantic search *(P1)*

Tìm kiếm trong không gian embedding chung.

- **text → image**: mã hoá truy vấn bằng nhánh text của BEiT-3, ANN search trên Milvus (COSINE)
- **image → image**: người dùng upload ảnh, mã hoá bằng nhánh vision **cùng một model**, search tương tự

**Acceptance criteria:**
- [ ] Hai chế độ dùng **chung một không gian vector** — đây là điều kiện để multimodal hoạt động
- [ ] Hỗ trợ lọc theo `video_id` / `recipe_type` ngay trong truy vấn Milvus (không lọc sau)
- [ ] Trả về `score`, `video_id`, `timestamp`, `keyframe_id`, đường dẫn thumbnail

### F3 — OCR / keyword search *(P1)*

Tìm theo chữ xuất hiện trên màn hình, qua Elasticsearch.

**Acceptance criteria:**
- [ ] Index dùng `vi_analyzer` từ plugin `elasticsearch-analysis-vietnamese`
- [ ] Có `asciifolding` với `preserve_original` để khớp cả khi người dùng gõ không dấu
- [ ] ⚠️ Trên YouCook2 nhánh này gần như vô dụng (không có chữ trên màn hình) — chỉ đo thật khi có dữ liệu AIC

### F4 — Hybrid fusion *(P1)*

🔧 **[Giả định]** Paper mô tả Semantic và OCR là hai **mode riêng biệt**, không nêu cơ chế hợp nhất. Dự án bổ sung fusion vì hai kênh này bổ trợ nhau.

- Hợp nhất bằng **Reciprocal Rank Fusion**: `RRF(d) = Σ_c 1 / (k + rank_c(d))`, mặc định `k = 60`
- Trọng số mỗi kênh cấu hình được

**Acceptance criteria:**
- [ ] Hybrid **không được kém hơn** kênh tốt nhất trong hai kênh đơn lẻ trên bộ eval — nếu kém, phải chỉnh trọng số hoặc bỏ

### F5 — QUEST *(P2)*

📄 **[Paper]** Xử lý truy vấn Out-of-Knowledge, hai nhánh:

| Nhánh | Cách hoạt động |
|---|---|
| **Nhánh 1 — Viết lại truy vấn** | Đưa `q₀` qua LLM (GPT-4o-mini / Gemini) để sinh `q_r` mô tả trực quan và chi tiết hơn → search bằng BEiT-3 như bình thường |
| **Nhánh 2 — Tìm ảnh mẫu bên ngoài** | Tìm ảnh của thực thể lạ trên web → chọn ảnh mẫu `I*` → mã hoá bằng BEiT-3 → image-to-image search |

**Acceptance criteria:**
- [ ] Luôn hiển thị truy vấn gốc **và** truy vấn đã viết lại — người vận hành phải thấy LLM đã làm gì
- [ ] Nhánh 2 cho phép người dùng **chọn thủ công** ảnh mẫu, không tự động chọn ảnh đầu tiên
- [ ] Cả hai nhánh có timeout; hết timeout thì fallback về search thường, **không** treo UI

### F6 — DANTE *(P2)*

📄 **[Paper]** Quy hoạch động cho TRAKE. Công thức nguyên văn:

```
(1)  S[i,t]      = cosine_similarity(u_i, E[t])
(2)  DP[i,t]     = S[i,t] + max_{τ ∈ [s_v, t−1]} ( DP[i−1,τ] − λ(t − τ) )
(3)  running_max = max( running_max, DP[i−1,t−1] + λ(t−1) )
(4)  DP[i,t]     = S[i,t] + running_max − λt
(5)  DANTE[v]    = max_{t ∈ [s_v, e_v]} DP[N,t]
```

- `u_i`: embedding của sự kiện thứ `i` trong N sự kiện; `E[t]`: embedding keyframe tại index toàn cục `t`
- `λ`: phạt khoảng cách thời gian — ưu tiên các sự kiện gần nhau, giữ đúng trật tự tường thuật
- Eq (3)(4) là bản tối ưu bằng *running max*, đưa `O(N·T²)` → **`O(NT)`**
- 📄 Paper đề xuất `λ = 0.001` cho khoảng cách index 3–15, `λ = 0.01` cho khoảng cách 1–3

**Acceptance criteria:**
- [ ] Ràng buộc thứ tự nghiêm ngặt: `τ < t` — chuỗi trả về **không bao giờ** đảo thứ tự thời gian
- [ ] Có backtracking để trả về đúng N keyframe, không chỉ điểm số
- [ ] `λ` là tham số cấu hình; có script sweep (§6.4)
- [ ] Verify độ phức tạp bằng thực nghiệm: thời gian chạy tuyến tính theo T

### F7 — VQA answering *(P3)*

Trả lời câu hỏi dựa trên keyframe truy hồi được. Dùng **Gemini API** làm mô hình chính.

**Acceptance criteria:**
- [ ] Prompt chỉ chứa top-k keyframe **thực sự truy hồi được** — không để model trả lời từ kiến thức nền
- [ ] Đầu ra là JSON có cấu trúc: `{answer, confidence, evidence_keyframe_ids[], reasoning}`
- [ ] Model **phải** được phép trả lời "không đủ căn cứ" — đây là cơ chế chống bịa (đo bằng `Hallucination Rate`, §6.5)
- [ ] Hiển thị keyframe bằng chứng cạnh câu trả lời để người vận hành tự kiểm chứng

### F8 — Submission exporter *(P3)*

Xuất file kết quả theo định dạng BTC.

⚠️ **Định dạng nộp chính xác chưa công bố** — trang hướng dẫn AIC ghi *"quy tắc tính điểm sẽ được thông báo riêng"*. Vì vậy:

**Acceptance criteria:**
- [ ] Tách rời **dữ liệu kết quả** và **bộ định dạng (formatter)** — đổi format chỉ sửa formatter
- [ ] Có validator chạy trước khi ghi file
- [ ] Hỗ trợ cả 4 dạng task

### F9 — Giao diện tương tác *(P1, hoàn thiện ở P3)*

📄 Paper mô tả UI gồm: (A) ô nhập + nút AI enhancer; (B) chọn mode Semantic/OCR/DANTE; (C) lưới kết quả + nút "Play YouTube" / "Find Similar Images"; (D) bộ lọc metadata + upload ảnh.

> ⚠️ **Xem §9 — Ràng buộc quy trình.** Đây **chỉ là tham chiếu chức năng**, không phải bản thiết kế phải làm theo.

**Acceptance criteria (chức năng, không phải hình thức):**
- [ ] Đổi mode không mất truy vấn đang gõ
- [ ] Lưới kết quả dùng virtualized rendering — mượt với hàng nghìn thumbnail
- [ ] Có shortcut bàn phím cho các thao tác nóng (submit, next/prev, đổi mode)
- [ ] Hiển thị `video_id` + timestamp ngay trên mỗi thumbnail, không cần hover

---

## 6. ⭐ Tiêu chuẩn đánh giá (Evaluation Metrics)

> Đây là phần lấp khoảng trống lớn nhất của paper baseline (§1.3). Mọi tuyên bố "cải thiện" trong dự án phải dẫn được về một con số trong mục này.

### 6.0 Nguyên tắc

1. **Mọi metric phải chỉ ra được nguồn ground truth cụ thể.** Metric nào không có nguồn nhãn thì phải ghi rõ *"cần sinh nhãn"* kèm cách sinh.
2. **Bộ đánh giá cố định và phiên bản hoá.** Dùng **val split YouCook2** (nhãn test bị giữ kín). Danh sách truy vấn eval lưu trong repo, có hash để phát hiện thay đổi.
3. **Luôn báo cáo cùng baseline.** Một con số đơn độc không có ý nghĩa.
4. **Báo cáo kèm mẫu số thực tế.** Video YouCook2 tải hỏng thì ghi rõ đánh giá trên bao nhiêu video, không im lặng bỏ qua.

### 6.1 Nguồn ground truth

Từ `youcookii_annotations_trainval.json`, cấu trúc:

```
database → { video_id: {
    duration, subset, recipe_type, video_url,
    annotations: [ { segment: [start_sec, end_sec], id, sentence } ]
} }
```

| Trường | Dùng làm ground truth cho |
|---|---|
| `annotations[].sentence` | Truy vấn Textual KIS |
| `annotations[].segment` | Đáp án đúng (khoảng thời gian) |
| Thứ tự `annotations[]` | Chuỗi sự kiện cho TRAKE |
| `recipe_type` | Nhãn kiểm thử bộ lọc metadata |
| `video_url` | Truy vết video nguồn |

### 6.2 Định nghĩa "hit" — nền tảng của mọi metric

Ground truth là một **khoảng thời gian**, còn hệ thống trả về một **keyframe**. Cần quy tắc quy đổi rõ ràng, nếu không mọi con số đều vô nghĩa.

**Quy tắc chọn — Hit theo điểm (point-in-segment):**

> Keyframe `k` được coi là **hit** cho segment `g` khi và chỉ khi
> `g.start ≤ timestamp(k) ≤ g.end` **và** `video_id(k) == video_id(g)`

**Vì sao chọn cách này thay vì tIoU(shot, segment) ≥ 0.5:**
- Sản phẩm trả về *keyframe*, không trả về *đoạn*. Đo trực tiếp thứ hệ thống thực sự sinh ra.
- Ranh giới shot của TransNetV2 và ranh giới procedure segment là hai khái niệm khác nhau — một bước nấu ăn có thể trải qua nhiều shot. Ép tIoU giữa chúng sẽ phạt oan hệ thống.
- Khớp cách chấm của cuộc thi: BTC kiểm tra frame nộp lên có nằm trong đoạn đúng hay không.

**Metric phụ trợ (báo cáo kèm, không dùng để chấm chính):** `tIoU(shot(k), g)` — dùng để chẩn đoán chất lượng shot segmentation, không phải chất lượng retrieval.

**Hai mức đo — bắt buộc báo cáo cả hai:**

| Mức | Điều kiện hit | Diễn giải |
|---|---|---|
| **Video-level** | Đúng `video_id` | Hệ thống có tìm đúng video không |
| **Segment-level** | Đúng `video_id` **và** timestamp nằm trong segment | Hệ thống có tìm đúng khoảnh khắc không |

Chênh lệch giữa hai mức chỉ ra bệnh cụ thể: video-level cao mà segment-level thấp nghĩa là *tìm đúng video nhưng sai khoảnh khắc* — vấn đề nằm ở lấy mẫu keyframe hoặc độ mịn thời gian, không phải ở embedding.

### 6.3 Nhóm A — Metrics truy hồi (Textual KIS / Video KIS)

**Bộ đánh giá:** mỗi `sentence` trong val split là một truy vấn; ground truth là `segment` tương ứng.

| Metric | Công thức / định nghĩa | Vì sao cần |
|---|---|---|
| **Recall@K**, K ∈ {1, 5, 10, 20, 100} | Tỉ lệ truy vấn có ít nhất một hit trong top-K | Metric chính. K=1 đo độ chính xác đỉnh; K=100 đo khả năng "có trong tầm với" của người vận hành khi cuộn lưới |
| **MRR** | Trung bình của `1 / rank_hit_đầu_tiên` trên toàn bộ truy vấn | Nhạy với vị trí kết quả đúng — sát với trải nghiệm thi thực tế |
| **mAP** | Trung bình Average Precision | Xét mọi kết quả đúng, không chỉ cái đầu tiên |
| **nDCG@10** | Discounted cumulative gain chuẩn hoá | Có tính đến thứ hạng của nhiều kết quả đúng |
| **Median Rank** | Trung vị thứ hạng của hit đầu tiên | Không bị kéo lệch bởi outlier như Mean Rank |
| **Recall@K theo nhóm** | Recall@10 tách theo `recipe_type` | Phát hiện hệ thống yếu ở loại nội dung nào |

**Baselines bắt buộc so sánh:**

| Baseline | Mục đích |
|---|---|
| Random | Sàn tuyệt đối — nếu không vượt xa cái này thì có bug |
| BM25 trên caption | Kiểm tra bài toán có bị "rò rỉ" giải được bằng từ khoá thuần hay không |
| CLIP ViT-B/32 | Baseline multimodal phổ biến, dễ chạy |
| **BEiT-3** | Hệ thống chính |
| BEiT-3 + Hybrid (RRF) | Đo đóng góp riêng của fusion |

### 6.4 Nhóm B — Metrics TRAKE / DANTE

**Bộ đánh giá:** mỗi video val có ≥ 3 segment → lấy N segment liên tiếp làm chuỗi truy vấn (N ∈ {3, 5, 7}), caption của chúng là N sự kiện đầu vào.

| Metric | Định nghĩa | Vì sao cần |
|---|---|---|
| **Sequence Exact-Match** | Tỉ lệ truy vấn mà **cả N** keyframe đều hit **và** đúng thứ tự | Metric chính — đây đúng là điều kiện chấm điểm của TRAKE |
| **Per-event Recall@K** | Recall@K tính riêng cho từng sự kiện `i` trong chuỗi | Chỉ ra sự kiện thứ mấy hay hỏng (thường là sự kiện đầu/cuối) |
| **Video-level Recall@K** | Video đúng có nằm trong top-K video mà DANTE chấm điểm cao nhất không | Tách lỗi "chọn sai video" khỏi lỗi "chọn sai frame trong video đúng" |
| **Kendall's tau** | Tương quan thứ hạng giữa thứ tự dự đoán và thứ tự thật | Đo riêng chất lượng *sắp thứ tự* — đây chính là thứ `λ` điều khiển |
| **Mean Temporal Offset** | Trung bình khoảng cách tuyệt đối giữa `timestamp(k_i)` và tâm segment `g_i`, tính bằng giây | Sai bao xa khi sai — quan trọng vì lệch 2 giây khác hẳn lệch 2 phút |
| **Thời gian chạy theo T** | Wall-clock theo số keyframe | Kiểm chứng thực nghiệm tuyên bố `O(NT)` của paper |

**Ablation `λ` — bắt buộc.** Paper đưa ra `λ = 0.001` và `λ = 0.01` nhưng không có bảng ablation nào. Dự án phải sweep:

```
λ ∈ {0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1}
```

Báo cáo `Sequence Exact-Match` và `Kendall's tau` cho từng giá trị, tách theo N ∈ {3, 5, 7}. Trường hợp `λ = 0` là quan trọng nhất: nó cho biết **hình phạt thời gian có thực sự đóng góp gì không** — nếu `λ = 0` cho kết quả tương đương thì phần cốt lõi của DANTE chỉ là ràng buộc thứ tự, không phải hình phạt khoảng cách.

### 6.5 Nhóm C — Metrics QA / VQA

⚠️ **YouCook2 không có nhãn QA.** Cách sinh bộ đánh giá:

1. Với mỗi segment, dùng LLM sinh 1–2 câu hỏi **chỉ trả lời được bằng thông tin thị giác** trong segment đó (ví dụ *"Nguyên liệu nào được cho vào chảo ở bước này?"*), kèm đáp án tham chiếu.
2. **Review thủ công ít nhất 100 mẫu** — loại câu hỏi trả lời được bằng kiến thức nền mà không cần nhìn video.
3. Đóng băng bộ QA đã review làm bộ đánh giá chính thức; bộ chưa review chỉ dùng để dò lỗi.

| Metric | Định nghĩa |
|---|---|
| **Exact Match (EM)** | Trùng khít sau chuẩn hoá (lowercase, bỏ dấu câu, bỏ mạo từ) |
| **Token-F1** | F1 trên tập token — chấp nhận diễn đạt khác |
| **Grounded Accuracy** ⭐ | Trả lời đúng **VÀ** ≥ 1 keyframe bằng chứng nằm trong segment đúng. Đây là metric chính |
| **Retrieval-conditioned Accuracy** | Accuracy chỉ tính trên các câu mà retrieval đã đúng — tách lỗi retrieval khỏi lỗi suy luận của VLM |
| **Hallucination Rate** ⭐ | Tỉ lệ hệ thống đưa câu trả lời tự tin (`confidence ≥ ngưỡng`) trong khi **không** có keyframe đúng nào trong ngữ cảnh |
| **Abstention Rate** | Tỉ lệ trả lời "không đủ căn cứ" — đọc kèm Hallucination Rate |
| **LLM-as-Judge** | Gemini chấm theo rubric 1–5. **Metric tham khảo**, không dùng để so sánh chính vì có thiên lệch tự đánh giá |

**Vì sao `Grounded Accuracy` là metric chính:** một hệ thống VQA trả lời đúng nhờ kiến thức nền của LLM (chứ không nhờ nhìn video) là vô dụng trong cuộc thi — BTC yêu cầu chỉ ra bằng chứng. Cặp `Grounded Accuracy` ↑ / `Hallucination Rate` ↓ mới mô tả đúng chất lượng hệ thống.

### 6.6 Nhóm D — Metrics hệ thống & chế độ thi

**Độ trễ** (đo từ lúc nhận request đến lúc trả response, không tính render):

| Chế độ | p50 | p95 |
|---|---|---|
| Semantic | ⚠️ ≤ 150 ms | ⚠️ ≤ 300 ms |
| OCR (ES) | ⚠️ ≤ 100 ms | ⚠️ ≤ 250 ms |
| Hybrid (RRF) | ⚠️ ≤ 250 ms | ⚠️ ≤ 500 ms |
| DANTE (N=5) | ⚠️ ≤ 1 s | ⚠️ ≤ 2 s |
| QUEST (có gọi LLM) | ⚠️ ≤ 3 s | ⚠️ ≤ 6 s |
| VQA answering | ⚠️ ≤ 5 s | ⚠️ ≤ 10 s |

**Metrics mô phỏng chế độ thi** — đây là thứ gần nhất với điểm số thật:

| Metric | Định nghĩa | Cách đo |
|---|---|---|
| **Time-To-First-Correct (TTFC)** | Số giây từ lúc nhận truy vấn đến lúc người vận hành nộp đáp án đúng | Phiên thao tác thật có bấm giờ, ≥ 20 truy vấn, ≥ 2 người vận hành |
| **Submissions-To-Correct (STC)** | Số lần nộp cho tới khi đúng | Cùng phiên trên |
| **Success Rate @ 300s** | Tỉ lệ truy vấn giải được trong 5 phút | Cùng phiên trên |
| **Rank-in-Grid** | Vị trí của kết quả đúng trong lưới mà người vận hành phải cuộn tới | Log tự động — proxy rẻ cho TTFC, đo được không cần con người |

`Rank-in-Grid` là metric thay thế thực dụng: chạy tự động trên toàn bộ bộ eval, tương quan mạnh với TTFC nhưng không tốn công tổ chức phiên thao tác.

**Hiệu quả hạ tầng:**

| Metric | Mục tiêu |
|---|---|
| Ingest throughput | ⚠️ ≥ 100 keyframe/s ở bước embedding (BEiT-3 base, fp16, RTX 3070) |
| Thời gian ingest val split (457 video) | ⚠️ ≤ 6 giờ |
| Thời gian ingest toàn bộ YouCook2 (2.000 video) | ⚠️ ≤ 24 giờ |
| Dung lượng vector index | ⚠️ ~1,2 GB (giả định 400k keyframe × 768 chiều × 4 byte) |
| Dung lượng thumbnail | ⚠️ ~16 GB (400k × ~40 KB) |
| RAM lúc phục vụ truy vấn | ≤ 12 GB cho cả Milvus + ES + MongoDB |
| Chi phí Gemini OCR | Token/1.000 keyframe — Gemini tính **258 token cho mỗi tile 768×768** |

### 6.7 Bảng mục tiêu v1

🔧 **[Giả định]** Paper không công bố số nào, nên các mốc dưới đây là mục tiêu tự đặt dựa trên kết quả thường thấy của retrieval video trên YouCook2. **Sẽ hiệu chỉnh lại sau lần đo đầu tiên ở P0.**

| Metric | Sàn (phải đạt) | Mục tiêu v1 | Kỳ vọng |
|---|---|---|---|
| Recall@1 (segment-level) | ≥ 0,05 | **≥ 0,12** | ≥ 0,20 |
| Recall@10 (segment-level) | ≥ 0,20 | **≥ 0,35** | ≥ 0,50 |
| Recall@10 (video-level) | ≥ 0,40 | **≥ 0,60** | ≥ 0,75 |
| MRR | ≥ 0,10 | **≥ 0,20** | ≥ 0,30 |
| TRAKE Sequence Exact-Match (N=3) | ≥ 0,10 | **≥ 0,25** | ≥ 0,40 |
| TRAKE Kendall's tau | ≥ 0,50 | **≥ 0,75** | ≥ 0,90 |
| VQA Grounded Accuracy | ≥ 0,25 | **≥ 0,45** | ≥ 0,60 |
| VQA Hallucination Rate | ≤ 0,30 | **≤ 0,15** | ≤ 0,08 |
| Semantic search p95 | ≤ 800 ms | **≤ 300 ms** | ≤ 150 ms |

**Cách đọc bảng:** cột *Sàn* là ngưỡng dưới — không đạt nghĩa là có bug, không phải "cần tối ưu". Cột *Mục tiêu v1* là mốc để coi phase hoàn thành.

### 6.8 Giao thức đánh giá

Để kết quả tái lập được và so sánh được giữa các lần chạy:

1. **Một lệnh:** `python -m eval.run --config configs/eval/youcook2_val.yaml`
2. **Đầu ra:** `reports/<timestamp>_<git_sha>/` gồm `metrics.json`, `report.md`, `per_query.csv`
3. **Bắt buộc ghi kèm mọi báo cáo:** git SHA, tên checkpoint model, phiên bản index, số video thực tế đánh giá được, seed
4. **Cố định seed** cho mọi bước có ngẫu nhiên (lấy mẫu bộ eval, sinh QA)
5. **Không sửa bộ eval để cải thiện số.** Đổi bộ eval thì tăng phiên bản và chạy lại toàn bộ baseline

---

## 7. Yêu cầu phi chức năng

### 7.1 Hiệu năng
- Ngân sách độ trễ theo §6.6
- Lưới kết quả phải render mượt với ≥ 1.000 thumbnail (virtualized rendering)
- Truy vấn Milvus phải lọc metadata **trong** engine, không lọc sau khi lấy về

### 7.2 Tài nguyên (ràng buộc cứng: RTX 3070 8GB VRAM, một máy)

| Tài nguyên | Yêu cầu |
|---|---|
| VRAM | ≤ 8 GB → dùng `beit3_base_itc` thay vì `large`; inference fp16; batch size cấu hình được |
| RAM hệ thống | Tối thiểu 16 GB, **khuyến nghị 32 GB** — ba store chạy cùng máy là sức ép lên RAM chứ không phải VRAM |
| JVM heap của Elasticsearch | Giới hạn cứng (≤ 2 GB ở môi trường dev) |
| Đĩa | ⚠️ ~40 GB cho thumbnail + index + video đã tải (chưa tính video gốc) |
| Chế độ suy giảm | Phải chạy được ở chế độ **semantic-only** khi tắt Elasticsearch |

### 7.3 Độ tin cậy
- Ingest **idempotent** và **resume được** (T1) — bắt buộc, không phải "nice to have", vì ingest kéo dài hàng chục giờ
- Video lỗi ghi vào danh sách riêng, pipeline vẫn chạy tiếp
- Mọi lời gọi API bên ngoài có timeout + retry với exponential backoff
- **Mất mạng không làm chết hệ thống**: semantic search và DANTE chạy hoàn toàn cục bộ; chỉ QUEST, OCR và VQA cần mạng

### 7.4 Bảo mật
- API key nạp từ biến môi trường / `.env`, **không bao giờ** commit (`.gitignore` đã có sẵn quy tắc Python)
- Log phải **che** API key, token và authorization header
- CORS cấu hình theo môi trường, không dùng wildcard kèm credentials (theo `.claude/rules/ecc/python/fastapi.md`)
- Ảnh người dùng upload: giới hạn kích thước, kiểm tra MIME type, không lưu vĩnh viễn

### 7.5 Khả năng tái lập
- Pin phiên bản mọi dependency (`requirements.txt` / `pyproject.toml` có lock)
- Ghi lại checkpoint hash của model
- Cố định seed
- Phiên bản hoá index: đổi model embedding **bắt buộc** phải tạo collection Milvus mới, không ghi đè — vector từ hai model khác nhau không so sánh được

### 7.6 Khả năng bảo trì
- Tuân thủ `.claude/rules/ecc/python/`: PEP 8, type annotation trên mọi chữ ký hàm, black + isort + ruff
- Tuân thủ `.claude/rules/ecc/python/fastapi.md`: `create_app()`, router mỏng, logic nằm ở service, schema request/response tách biệt, DI qua `Depends`
- `MultimodalEncoder` và `DatasetAdapter` là Protocol — thay implementation không sửa nơi gọi

### 7.7 Khả năng quan sát
- Structured logging (JSON) cho pipeline ingest, có tiến độ theo `video_id`
- Mỗi request search log lại: mode, độ trễ, số kết quả, cache hit/miss
- Endpoint `/health` kiểm tra kết nối cả ba store

---

## 8. Rủi ro & giảm thiểu

| # | Rủi ro | Mức | Giảm thiểu |
|---|---|---|---|
| **R1** | **BEiT-3 khó cài** — không có trong HuggingFace Transformers, phải clone `microsoft/unilm`, cần `torchscale` + `timm` bản pin + tokenizer SentencePiece `beit3.spm`. Rất dễ xung đột với torch mới trên Windows | 🔴 Cao | Thiết kế `MultimodalEncoder` Protocol **ngay từ ngày đầu**; chuẩn bị sẵn implementation SigLIP/open_clip làm phương án thay thế; **timebox 1 ngày** cho việc dựng BEiT-3, quá thì chuyển phương án và ghi nhận là sai lệch so với baseline |
| **R2** | **Chi phí & rate-limit của Gemini OCR** — ước tính 240k–500k keyframe cho toàn bộ YouCook2; gọi OCR từng frame là không khả thi cả về tiền lẫn quota | 🔴 Cao | (a) Dedupe keyframe gần trùng bằng perceptual hash trước khi gọi; (b) lọc sơ bộ bằng bộ phát hiện chữ rẻ tiền, chỉ gửi frame **có** chữ; (c) dùng Batch API; (d) giai đoạn 1 chỉ OCR trên val split |
| **R3** | **RAM khi chạy Milvus + Elasticsearch + MongoDB trên cùng một máy** | 🟡 Trung bình | Giới hạn JVM heap của ES; yêu cầu RAM ≥ 16 GB; hỗ trợ chế độ semantic-only để tắt hẳn ES khi cần |
| **R4** | **YouCook2 không có chữ trên màn hình** → không kiểm chứng được nhánh OCR | 🟡 Trung bình | Đánh dấu rõ nhánh OCR là *"đã xây, chưa kiểm chứng"*; **không** báo cáo metric OCR trên YouCook2; chuẩn bị bộ test nhỏ có chữ để smoke test |
| **R5** | **Link YouTube của YouCook2 bị gỡ** → thiếu video, mẫu số đánh giá teo dần | 🟡 Trung bình | Log video tải hỏng; **mọi báo cáo metric ghi kèm số video thực tế đánh giá được**; đóng băng danh sách video đã tải thành công làm bộ eval chuẩn |
| **R6** | **Không có nhãn QA** cho YouCook2 | 🟡 Trung bình | Sinh bằng LLM + review thủ công ≥ 100 mẫu (§6.5); ghi rõ đây là nhãn tổng hợp, không phải nhãn người |
| **R7** | **Ingest 176 giờ video đứt giữa chừng** | 🟡 Trung bình | Checkpoint theo `video_id`; ingest idempotent (T1); chia shard chạy song song nhiều tiến trình |
| **R8** | **Định dạng file nộp của BTC chưa công bố** | 🟡 Trung bình | Tách formatter khỏi dữ liệu kết quả (F8); giả định format sai chỉ tốn công sửa một module |
| **R9** | **Dữ liệu AIC thật có phân bố rất khác YouCook2** — tin tức tiếng Việt vs. nấu ăn tiếng Anh. Tối ưu trên YouCook2 có thể không chuyển giao được | 🟡 Trung bình | Coi mọi số đo ở P0–P3 là **tương đối**, dùng để so sánh giữa các cấu hình chứ không phải dự đoán điểm thi; ưu tiên các cải tiến mang tính kiến trúc hơn là tinh chỉnh siêu tham số |

---

## 9. Ràng buộc quy trình

> **Về giao diện frontend:** khi tới bước triển khai UI, **phải hỏi ý người chủ dự án về thiết kế giao diện trước** — họ sẽ gửi mẫu thiết kế mong muốn.
>
> Mô tả UI trong §F9 và trong paper baseline chỉ dùng làm **tham chiếu chức năng** (những thao tác giao diện cần hỗ trợ), **không** phải bản thiết kế phải sao chép. Không tự dựng UI giống paper.

Các quyết định đã chốt khác:

| Hạng mục | Quyết định |
|---|---|
| Ngôn ngữ tài liệu | Tiếng Việt, giữ thuật ngữ kỹ thuật tiếng Anh |
| Phạm vi | Đủ 4 task AIC |
| Dataset | YouCook2 trước, dữ liệu AIC bổ sung sau |
| Storage | Đúng baseline: Milvus + Elasticsearch + MongoDB |
| Backend | FastAPI + Pydantic |
| Frontend | React + Vite + TypeScript (thiết kế chốt sau) |
| VQA answering | Gemini API làm chính |
| Compute | Một GPU local RTX 3070 8 GB |

---

## 10. Câu hỏi mở

| # | Câu hỏi | Cần trả lời trước |
|---|---|---|
| Q1 | Định dạng file nộp chính xác của BTC? | P3 |
| Q2 | Dữ liệu AIC có bao nhiêu video / giờ? Ảnh hưởng trực tiếp tới ước lượng hạ tầng | P4 |
| Q3 | Có kênh audio/phụ đề không? Nếu có thì ASR đáng đưa lên sớm | P4 |
| Q4 | Số chiều embedding thực tế của `beit3_base_itc` (768 hay khác)? | P0 |
| Q5 | Lúc thi có mạng internet ổn định không? Ảnh hưởng tới việc QUEST và VQA có dùng được không | P3 |

---

## Tài liệu liên quan

- [`architecture.md`](./architecture.md) — Thiết kế hệ thống, sơ đồ luồng dữ liệu, schema
- [`references.md`](./references.md) — Thư viện, API, nguồn tham khảo

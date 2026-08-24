# AIC-Video-Retrieval

Hệ thống truy vấn video đa phương thức cho vòng sơ tuyển AI Challenge TP.HCM
2026. Dự án tập trung vào 3 task còn lại của đề: `Textual KIS`, `QA/VQA` và
`TRAKE`; `Video KIS` đã bị bỏ khỏi vòng sơ tuyển.

Tài liệu setup chi tiết nằm ở [`README_setup.md`](./README_setup.md). README này
giải thích hệ thống đang có gì, KIS đang chạy ra sao, TRAKE hiện tại có gì, các
công thức tính điểm và cách xếp hạng kết quả.

## Tổng quan hệ thống

### Bài toán và định dạng nộp

Mỗi truy vấn được nộp tối đa 100 đáp án đã xếp hạng.

| Task | Input | Output nộp |
|---|---|---|
| `Textual KIS` | Mô tả khoảnh khắc bằng ngôn ngữ tự nhiên | `<video_id>,<frame_id>` |
| `QA/VQA` | Mô tả khoảnh khắc + câu hỏi | `<video_id>,<frame_id>,<answer>` |
| `TRAKE` | Chuỗi N sự kiện theo thứ tự thời gian | `<video_id>,<frame_id_1>,...,<frame_id_N>` |

`frame_id` trong file nộp luôn lấy từ `frame_idx` đã lưu cùng keyframe lúc ingest.
Hệ thống không tính lại `frame_id` bằng `timestamp * fps`, vì sai số làm tròn có
thể làm lệch khỏi cửa sổ chấm rất hẹp của BTC.

### Thành phần chính

| Tầng | Thành phần hiện có | Vai trò |
|---|---|---|
| Frontend | React + Vite + TypeScript | UI tìm kiếm, chọn đáp án, xem keyframe, export CSV |
| API | FastAPI + Pydantic | Router mỏng: validate request, gọi service, chuẩn hóa lỗi |
| Encoder | `FG-CLIP Base ViT-B/16`, `model_id=fgclip_base`, 512 chiều | Mã hóa text/image vào cùng không gian vector, L2-normalize |
| Vector store | Milvus collection `keyframes_fgclip_base` | ANN search cosine cho keyframe |
| Text store | Elasticsearch | OCR text `ocr_keyframes` và transcript ASR `asr_segments` |
| Metadata store | MongoDB | `video_id`, `frame_idx`, `timestamp`, `global_index`, `s_v/e_v`, title, path ảnh |
| VQA/QUEST | Gemini + web image search | Trả lời QA có evidence, rewrite query, tìm ảnh mẫu web |
| TRAKE | `TrakeEmbeddingCache` + DANTE DP | Tìm chuỗi N keyframe đúng thứ tự trong cùng video |

### Dữ liệu hiện tại

Dự án dùng AIC 2026 batch 1 công khai, tương ứng batch 1 của AIC 2025:

- 873 video tin tức tiếng Việt, khoảng 130,7 giờ.
- Dataset công khai không có ground truth truy vấn; eval offline dùng nhãn tự sinh.
- Corpus online hiện đi qua keyframe/vector đã import vào Milvus/MongoDB. Runtime
  backend hiện dùng FG-CLIP, không còn dùng BEiT-3 làm encoder phục vụ chính.
- ASR là nhánh bổ trợ: transcript được index riêng trong Elasticsearch và chỉ dùng
  làm video gate/evidence cho KIS, không trực tiếp tạo điểm keyframe.

## Luồng offline

Offline ingest biến video/keyframe thành dữ liệu truy hồi được.

Luồng thiết kế đầy đủ:

```text
video -> shot detection -> keyframe extraction -> image embedding -> Milvus
                                  |-> OCR -> Elasticsearch
                                  |-> metadata/frame_idx/global_index -> MongoDB
video -> ASR transcript -> Elasticsearch asr_segments
```

### Công thức trích keyframe

Với mỗi shot có khoảng frame $[a, b]$, hệ thống lấy 4 vị trí:

$$
k_i = a + \left\lfloor \frac{i(b-a)}{3} \right\rfloor,\quad i \in \{0,1,2,3\}
$$

Mỗi keyframe được gán:

- `frame_idx`: frame tuyệt đối trong video, dùng để nộp bài.
- `timestamp`: thời điểm giây để hiển thị và đánh giá offline.
- `global_index`: chỉ số liên tục trên toàn corpus, dùng để slice video khi chạy TRAKE.
- `s_v`, `e_v`: global index đầu/cuối của video `v`, lưu trong MongoDB.

### Công thức embedding/search cơ bản

Encoder trả vector đã L2-normalize:

$$
u = \operatorname{normalize}(\operatorname{encode\_text}(q))
$$

$$
e_t = \operatorname{normalize}(\operatorname{encode\_image}(k_t))
$$

$$
\operatorname{score}_{\text{text-image}}(t)
= \cos(u,e_t)
= u^\top e_t
$$

Milvus dùng HNSW với cosine. Config hiện tại:

```text
M = 16
efConstruction = 200
search_ef = 128
```

## KIS hiện có gì

KIS hiện có hai đường:

1. Search đơn giản qua `/api/search`:
   - `mode=semantic`: text -> FG-CLIP text embedding -> Milvus cosine search.
   - `mode=ocr`: text -> Elasticsearch BM25 trên OCR -> join MongoDB để lấy `frame_idx`.
   - `/api/search/image`: ảnh upload -> FG-CLIP image embedding -> Milvus cosine search.

2. KIS hai giai đoạn qua `/api/search/kis/candidates` và
   `/api/search/kis/rerank`:
   - Stage 1 tạo candidate pool recall cao bằng Text, OCR, Title và optional ASR gate.
   - Stage 2 rerank pool bằng Text/OCR/ABE subqueries rồi xuất Top-K cuối.

### KIS Stage 1: tạo candidate pool

Input Stage 1 có thể gồm:

- `text_query` hoặc `text_queries`: mô tả hình ảnh/khoảnh khắc.
- `ocr_query`: chữ xuất hiện trên màn hình.
- `title_query`: từ/cụm từ trong tiêu đề video.
- `asr_query` hoặc `asr_queries`: từ/cụm từ nghe được trong audio transcript.

Các giới hạn chính nằm trong `configs/search/search.yaml`:

```text
default_text_candidate_k = 300
default_ocr_candidate_k = 100
default_title_candidate_k = 100
max_candidate_k = 1000
max_top_k = 100
```

#### Stage 1 Text

Mỗi text query được dịch VI->EN nếu có translator, rồi encode bằng FG-CLIP:

$$
u_j = \operatorname{normalize}(\operatorname{encode\_text}(\operatorname{translate}(q_j)))
$$

$$
s_j(d) = \cos(u_j,e_d)
$$

Với nhiều text query, điểm hiển thị Stage 1 hiện là tổng cosine:

$$
\operatorname{score}_{\text{text-stage1}}(d) = \sum_j s_j(d)
$$

Candidate từ nhiều query được union và dedupe theo cặp `(video_id, frame_idx)`.
Nếu Stage 1 có Text, ranking Stage 1 ưu tiên điểm text fused này; OCR/Title vẫn
được giữ trong metadata nguồn candidate để người vận hành biết vì sao item xuất
hiện.

#### Stage 1 OCR

OCR đi qua Elasticsearch BM25, nhưng không dùng BM25 thô trực tiếp. Hệ thống lấy
sâu hơn trước rồi rescoring:

$$
\operatorname{retrieval\_k}
= \max(\operatorname{ocr\_top\_k} \cdot \operatorname{candidate\_multiplier},
\operatorname{candidate\_min})
$$

Với cấu hình mặc định:

$$
\operatorname{candidate\_multiplier}=5,\quad \operatorname{candidate\_min}=100
$$

Với mỗi OCR hit:

$$
\operatorname{ref} =
\begin{cases}
P_{95}(\{\operatorname{BM25}(d) \mid \operatorname{BM25}(d)>0\}), & \text{nếu đủ mẫu} \\
\max(\{\operatorname{BM25}(d) \mid \operatorname{BM25}(d)>0\}), & \text{ngược lại}
\end{cases}
$$

$$
\operatorname{bm25\_norm}(d)
= \operatorname{clamp}\left(\frac{\operatorname{BM25}(d)}{\operatorname{ref}}, 0, 1\right)
$$

$$
\operatorname{score}_{\text{ocr-query}}(d)
= \operatorname{clamp}(\operatorname{bm25\_norm}(d)\cdot \operatorname{coverage}(d),0,1)
$$

Trong đó $\operatorname{coverage}(d)$ là tỷ lệ token query khớp OCR text, có
fuzzy match cho tiếng Việt.

OCR Stage 1 sort theo:

```text
score_ocr_query giảm dần
raw_bm25 giảm dần
initial_rank tăng dần
```

#### Stage 1 Title

Title search chạy lexical trong MongoDB, không tạo index riêng:

$$
\operatorname{title\_score}(d)=
\begin{cases}
1.0, & \text{nếu khớp một token} \\
2.0 + N, & \text{nếu khớp cụm } N \text{ token} \\
0, & \text{nếu không khớp}
\end{cases}
$$

Video title được sort theo `title_score` giảm dần, rồi `video_id` tăng dần. Sau
đó hệ thống lấy keyframe trong các video đó theo kiểu round-robin để pool không
bị một video dài chiếm hết.

#### Stage 1 ASR gate

ASR không phải kênh score. Nó là detector/filter deterministic trên transcript:

```text
query -> normalize text -> scan asr_segments -> matched video_ids + timestamp evidence
```

Luật detect chính:

- 1 token: token phải xuất hiện trong transcript.
- Cụm nhiều token: ưu tiên contiguous phrase.
- Query dài hơn: số token khớp / số token query phải >= `asr_token_coverage_threshold`
  (`0.8` mặc định).

$$
\operatorname{coverage}_{\text{asr}}(q, x)
= \frac{|\operatorname{tokens}(q)\cap \operatorname{tokens}(x)|}
       {|\operatorname{tokens}(q)|}
$$

$$
\operatorname{detected}(q,x)
= \mathbb{I}\left[\operatorname{coverage}_{\text{asr}}(q,x) \ge 0.8\right]
$$

`asr_match_mode`:

```text
any -> video đạt nếu khớp ít nhất một ASR query
all -> video đạt nếu khớp mọi ASR query
```

ASR gate trả evidence theo `start_time/end_time`, nhưng đáp án nộp vẫn lấy
keyframe stored gần midpoint của evidence nhất. ASR không đóng góp vào
`final_score` của Stage 2 và không có ASR-only submission pool.

### Rank percentile dùng trong KIS

Stage 2 không cộng trực tiếp cosine, BM25 và ABE raw score vì các thang đo khác
nhau. Mỗi query/modality trước hết được xếp hạng trong cùng candidate pool, rồi
đổi rank thành percentile:

$$
\operatorname{rank\_percentile}(d)
= 1 - \frac{\operatorname{rank}(d)-1}{M-1}
$$

Trong đó `M` là kích thước candidate pool hợp lệ. Top rank có percentile `1.0`,
worst rank có percentile `0.0`. Các item đồng điểm nhận cùng rank tương đương;
nếu tất cả đều không có điểm dương thì percentile về `0.0`.

### KIS Stage 2: rerank candidate pool

Stage 2 nhận `candidate_ids` từ Stage 1 và danh sách `subqueries`, gồm 3 loại:

- `text`: rerank bằng cosine text-image.
- `ocr`: rerank bằng BM25 normalized * coverage.
- `abe`: rerank attribute-object binding bằng patch alignment.

#### Text modality

Với từng text subquery:

$$
s_j(d) = \cos(u_j,e_d)
$$

$$
p_j(d) = \operatorname{rank\_percentile}(s_j(d))
$$

Điểm Text modality là geometric mean của các percentile:

$$
\operatorname{score}_{\text{text}}(d)
= \operatorname{GM}(p_1(d),\ldots,p_n(d))
$$

#### OCR modality

Với từng OCR subquery:

$$
\operatorname{score}_{\text{ocr-query},j}(d)
= \operatorname{clamp}(
  \operatorname{normalize\_bm25}(\operatorname{BM25}_j(d))\cdot
  \operatorname{coverage}_j(d),
  0,1
)
$$

$$
p_j(d)=\operatorname{rank\_percentile}(\operatorname{score}_{\text{ocr-query},j}(d))
$$

$$
\operatorname{score}_{\text{ocr}}(d)=\operatorname{GM}(p_1(d),\ldots,p_n(d))
$$

#### ABE modality

ABE dùng patch embedding của ảnh và token embedding của text để xử lý truy vấn
kiểu thuộc tính-vật thể, ví dụ "áo đỏ", "xe trắng".

Với một ABE query:

$$
\operatorname{base\_local}(d)
= \frac{1}{|T|}
  \sum_{\tau \in T}
  \frac{1}{K}
  \sum_{p \in \operatorname{TopK}(\tau,d)}
  \cos(\mathbf{z}_\tau,\mathbf{p})
$$

$$
\operatorname{refined\_local}(d)
= \frac{1}{|R|}
  \sum_{\rho \in R}
  \frac{1}{K}
  \sum_{p \in \operatorname{TopK}(\rho,d)}
  \cos(\mathbf{z}_\rho,\mathbf{p})
$$

$$
\operatorname{binding\_delta}(d)
= |\operatorname{refined\_local}(d)-\operatorname{base\_local}(d)|
$$

$$
\operatorname{local\_score}(d)
= \operatorname{refined\_local}(d)+\operatorname{binding\_delta}(d)
$$

$$
\operatorname{global\_abe}(d)
= \cos(\mathbf{g}_{\text{text}},\mathbf{g}_{\text{image},d})
$$

$$
\operatorname{raw\_abe}(d)
= (1-w)\operatorname{local\_score}(d)
  + w\operatorname{global\_abe}(d)
$$

`w = abe_global_weight = 0.3`, `patch_top_k = 5`, `concept_top_p = 5` theo
`configs/search/search.yaml`.

Sau đó:

$$
p_j(d)=\operatorname{rank\_percentile}(\operatorname{raw\_abe}_j(d))
$$

$$
\operatorname{score}_{\text{abe}}(d)=\operatorname{GM}(p_1(d),\ldots,p_n(d))
$$

#### Final KIS score

Mỗi modality đang bật trả một score trong `[0, 1]`. Điểm cuối:

$$
\operatorname{final\_score}(d)
= \operatorname{GM}(
  \operatorname{score}_{\text{text}}(d),
  \operatorname{score}_{\text{ocr}}(d),
  \operatorname{score}_{\text{abe}}(d)
)
$$

Geometric mean dùng epsilon `1e-8` để tránh `log(0)`:

$$
\operatorname{GM}(x_1,\ldots,x_n)
= \exp\left(
  \frac{1}{n}\sum_{i=1}^{n}\log(\max(x_i,\varepsilon))
\right),
\quad \varepsilon=10^{-8}
$$

Ranking cuối của Stage 2:

```text
sort candidate by final_score giảm dần
return Top final_top_k, với final_top_k <= 100
```

Frontend vẫn cho người vận hành chọn thủ công. Khi export KIS, các lựa chọn thủ
công đứng trước, có thể sweep thêm frame quanh keyframe đã chọn, rồi mới dùng
ranking tự động làm fallback cho đủ tối đa 100 dòng.

## TRAKE hiện có gì

TRAKE hiện đã có endpoint `/api/trake`, service online và cache embedding:

- API nhận `events[]` theo thứ tự thời gian, `top_k`, `threshold`,
  `lambda_penalty`.
- Khi FastAPI startup, `TrakeEmbeddingCache` load embedding toàn bộ video có
  `ingest_status=complete` từ Milvus vào RAM.
- Mỗi request chỉ encode các event text, nhân ma trận với cache, rồi chạy DANTE
  trên slice `[s_v, e_v]` của từng video.
- Một video có thể đóng góp nhiều sequence candidate nếu nhiều chuỗi vượt
  threshold.
- Kết quả trả về gồm `video_id`, `score`, đúng N keyframe theo thứ tự, mỗi
  keyframe có `keyframe_id`, `frame_idx`, `timestamp` và debug score/penalty.

Config hiện tại:

```text
default_top_k = 10
max_top_k = 100
default_score_threshold = 0.0
default lambda_penalty nếu client không gửi = 0.01
```

### TRAKE/DANTE formula

Gọi:

- `N`: số event trong query.
- `T_v`: số keyframe của video `v`.
- `u_i`: embedding của event thứ `i`.
- `E_t`: embedding keyframe thứ `t` trong video.
- `time[t]`: timestamp giây của keyframe `t`.
- `lambda`: hệ số phạt khoảng cách thời gian.

Ma trận tương đồng:

$$
S[i,t] = \cos(u_i,E_t)=u_i^\top E_t
$$

DP theo timestamp giây:

$$
DP[0,t]=S[0,t]
$$

$$
DP[i,t]
= S[i,t] +
\max_{\substack{\tau < t \\ \operatorname{time}[\tau] < \operatorname{time}[t]}}
\left(
  DP[i-1,\tau]
  - \lambda(\operatorname{time}[t]-\operatorname{time}[\tau])
\right)
$$

Công thức trên được triển khai bằng running max để giữ độ phức tạp `O(N*T_v)`:

$$
\operatorname{running\_max}_i(t)
= \max_{\substack{\tau < t \\ \operatorname{time}[\tau] < \operatorname{time}[t]}}
\left(DP[i-1,\tau]+\lambda\operatorname{time}[\tau]\right)
$$

$$
DP[i,t]
= S[i,t] + \operatorname{running\_max}_i(t) - \lambda\operatorname{time}[t]
$$

Ràng buộc `time[tau] < time[t]` đảm bảo event sau phải nằm sau event trước theo
thời gian thật, không chỉ theo global index.

Điểm của một sequence:

$$
\operatorname{score}(\operatorname{sequence}) = DP[N-1,t_{\text{end}}]
$$

Hệ thống backtrack từ `t_end` để lấy đúng N global indices, rồi map về keyframe
metadata trong cache.

### TRAKE ranking

Với mỗi video:

```text
chạy DANTE trên slice video v
giữ các sequence có score >= threshold
```

Sau đó gộp toàn bộ sequence candidates của mọi video:

```text
sort all sequences by score giảm dần
return Top-K
```

Vì ranking là trên sequence, không phải trên video duy nhất, cùng một `video_id`
có thể xuất hiện nhiều lần nếu nó có nhiều chuỗi khớp tốt.

## Công thức chấm/eval

### KIS official-style score

BTC cho tối đa 100 đáp án đã xếp hạng. Bộ eval nội bộ mô phỏng công thức bằng
các mốc:

$$
\mathcal{K}=\{1,5,20,50,100\}
$$

$$
\operatorname{AICFinalScore}
= \frac{1}{|\mathcal{K}|}\sum_{k\in\mathcal{K}}\operatorname{Recall}@k
$$

Với Textual KIS, một hit đúng khi đúng video và frame/timestamp nằm trong đoạn
ground truth:

$$
\operatorname{hit}(d,g)
= \mathbb{I}\left[
  \operatorname{video\_id}_d=\operatorname{video\_id}_g
  \land
  s_g \le \operatorname{timestamp}_d \le e_g
\right]
$$

Vì mỗi query eval hiện có một đáp án đúng, `Recall@k` là:

$$
\operatorname{Recall}@k
= \frac{|\{q \mid \operatorname{first\_hit\_rank}(q)\le k\}|}{|\mathcal{Q}|}
$$

Các metric bổ trợ:

$$
\operatorname{MRR}
= \frac{1}{|\mathcal{Q}|}
  \sum_{q\in\mathcal{Q}}
  \begin{cases}
    \frac{1}{r_q}, & r_q \text{ tồn tại} \\
    0, & \text{không tìm thấy}
  \end{cases}
$$

Vì mỗi query chỉ có một relevant item:

$$
\operatorname{mAP}=\operatorname{MRR}
$$

$$
\operatorname{nDCG@}k
= \frac{1}{|\mathcal{Q}|}
  \sum_{q\in\mathcal{Q}}
  \begin{cases}
    \frac{1}{\log_2(r_q+1)}, & r_q \le k \\
    0, & r_q > k \text{ hoặc không tìm thấy}
  \end{cases}
$$

### TRAKE official-style score

Theo đề, sai `video_id` thì sequence nhận 0. Nếu đúng video, điểm trên một đáp án
TRAKE là tỷ lệ event frame rơi vào đúng cửa sổ:

$$
R
= \frac{1}{N}\sum_{j=1}^{N}
\mathbb{I}\left[\operatorname{frame}_j \in [s_j,e_j]\right]
$$

Eval nội bộ có thêm:

- `Sequence Exact-Match`: tỷ lệ query mà đúng video và cả N event đều hit.
- `Video Recall@k`: ground-truth video có nằm trong Top-K không.
- `Per-event Recall@k`: event thứ `j` hit trong Top-K hay không.
- `Kendall tau`: đo thứ tự bằng cách gán mỗi retrieved keyframe tới ground-truth
  segment center gần nhất, rồi tính tau-b.
- `Mean temporal offset`: trung bình khoảng cách tuyệt đối tới tâm segment đúng.

## Ranking đang được dùng ở đâu

| Đường | Đơn vị ranking | Score dùng để sort |
|---|---|---|
| `/api/search`, `semantic` | keyframe | cosine từ Milvus |
| `/api/search`, `ocr` | keyframe | BM25 Elasticsearch, sau đó join metadata |
| `/api/search/image` | keyframe | cosine image-image từ Milvus |
| `/api/search/kis/candidates` | keyframe candidate | Text fused cosine nếu có Text; nếu không thì geomean rank percentile OCR/Title; ASR là gate/evidence |
| `/api/search/kis/rerank` | keyframe | `final_score = GM(active modality scores)` |
| `/api/trake` | sequence N keyframe | DANTE DP score |
| Frontend KIS export | dòng CSV | manual selections trước, frame sweep quanh lựa chọn, ranking fallback sau |
| Frontend TRAKE export | dòng CSV | selected sequences trước, ranking fallback sau |

Hybrid RRF giữa Semantic và OCR từng được thử nhưng hiện không wire vào ranking
online vì đo thực nghiệm làm kéo kênh mạnh xuống. Hàm RRF vẫn còn trong code để
dành cho trường hợp sau này fuse hai semantic encoder cùng loại:

$$
\operatorname{RRF}(d)
= \sum_c \frac{w_c}{k+\operatorname{rank}_c(d)}
$$

## API chính

| Method | Endpoint | Mục đích |
|---|---|---|
| `POST` | `/api/search` | Search đơn giản `semantic` hoặc `ocr` |
| `POST` | `/api/search/image` | Image-to-image search |
| `POST` | `/api/search/kis/candidates` | Stage 1 KIS candidate pool |
| `POST` | `/api/search/kis/rerank` | Stage 2 KIS reranking |
| `POST` | `/api/trake` | TRAKE/DANTE sequence search |
| `POST` | `/api/quest/rewrite-query` | Rewrite truy vấn OOK |
| `POST` | `/api/quest/image-candidates` | Tìm ảnh mẫu web |
| `POST` | `/api/qa` | VQA answer có evidence |
| `GET` | `/api/keyframes/{id}/thumbnail` | Thumbnail cached |
| `GET` | `/api/keyframes/{id}/image` | Ảnh keyframe gốc |
| `GET` | `/api/videos/{id}` | Metadata video |
| `GET` | `/api/videos/{id}/keyframes` | Keyframe quanh video |
| `GET` | `/health` | Store health + ASR readiness |

## Chạy nhanh

```powershell
docker compose up -d --build
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Test/lint:

```powershell
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m black --check .
./.venv/Scripts/python.exe -m pytest
```

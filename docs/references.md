# Tài liệu tham khảo — AIC Video Retrieval

> **Phiên bản:** 0.1 · **Ngày kiểm chứng nguồn:** 2026-07-25
> **Tài liệu liên quan:** [`PRD.md`](./PRD.md) · [`architecture.md`](./architecture.md)

## Nguyên tắc chọn thư viện

**Bám thư viện của paper baseline trước, tối ưu sau.** Giai đoạn 1 dùng đúng ngăn xếp mà paper mô tả để kết quả so sánh được với baseline. Các phương án thay thế được liệt kê ở §8 nhưng **chưa dùng ở v1**.

**Về cách thu thập thông tin trong tài liệu này:** với mỗi thư viện, đã thử fetch file `llms.txt` ở gốc trang docs trước. Nếu có, snippet API bên dưới lấy trực tiếp từ đó (nội dung mới nhất do chính nhà cung cấp phát hành cho LLM đọc). Nếu không có, đã fallback về trang docs chính thức và ghi rõ trong bảng §9.

> ⚠️ **Lưu ý về model ID:** các ID model trong tài liệu này (`gemini-3.6-flash`, `gpt-5.6`) là những gì docs chính thức đang hiển thị tại ngày kiểm chứng. Hãy xác nhận lại lúc implement — nhà cung cấp đổi tên model thường xuyên.

---

## 1. Paper gốc & bối cảnh cuộc thi

### 1.1 Paper baseline

**Integrated Semantic and Temporal Alignment for Interactive Video Retrieval**
Thanh-Danh Luu, Le-Vu Nguyen Dinh, Duc-Thien Tran, Duy-Bao Bui, Nam-Tien Le, Tinh-Anh Nguyen Nhu (team AIO_Owlgorithms) · cs.MM · 15/12/2025

- Trang abstract: https://arxiv.org/abs/2512.13169
- Bản HTML đầy đủ: https://arxiv.org/html/2512.13169v1

**Đóng góp chính:**

| | Nội dung |
|---|---|
| **QUEST** | Xử lý truy vấn Out-of-Knowledge bằng 2 nhánh: LLM viết lại truy vấn + tìm ảnh mẫu bên ngoài |
| **DANTE** | Quy hoạch động căn chỉnh chuỗi sự kiện theo thời gian, `O(NT)` |

**Kết quả công bố:** chỉ định tính (TRAKE *Outstanding*, Textual KIS *Excellent*, Video KIS *Very Good*, QA *Very Good*). **Không có metric định lượng nào** — xem §1.3 và §6 của [`PRD.md`](./PRD.md).

### 1.2 Cuộc thi

- Trang chính thức: https://aichallenge.hochiminhcity.gov.vn/en/home
- Hướng dẫn: https://aichallenge.hochiminhcity.gov.vn/en/huong-dan

> ✅ **Cập nhật 2026-08-01 — đề vòng sơ tuyển AIC 2026 đã công bố** (`Thong tin vong So tuyen AIC2026.pdf` ở gốc repo). Mục này viết lại theo đề chính thức; phần cũ (4 task, phạt theo thời gian) là của AIC 2025 và **không còn áp dụng**.

**Ba dạng nhiệm vụ ở vòng sơ tuyển:**

| Task | Đầu vào | Định dạng đáp án `rᵢ` |
|---|---|---|
| **Textual KIS** | Mô tả sự kiện bằng ngôn ngữ tự nhiên — vòng sơ tuyển cung cấp **sẵn và trọn vẹn**, không lộ dần | `<video_id>, <frame_id>` |
| **Q&A (VQA)** | Mô tả sự kiện + một câu hỏi về thông tin trong sự kiện đó. Trả lời được bằng **tiếng Việt hoặc tiếng Anh** | `<video_id>, <frame_id>, <answer>` |
| **TRAKE** | Chuỗi N khoảnh khắc ngữ nghĩa có cấu trúc, theo thứ tự thời gian | `<video_id>, <frame_id₁>, …, <frame_idₙ>` |

**Video KIS đã bị bỏ** khỏi vòng sơ tuyển AIC 2026.

⚠️ **`frame_id` là chỉ số frame tuyệt đối trong video**, không phải giây và không phải số thứ tự keyframe. Đề nói rõ "semantic keyframe" (khoảnh khắc mang ý nghĩa nội dung) **khác** "I-Frame" (khung hình kỹ thuật của thuật toán nén).

### Cách chấm — đây là thay đổi lớn nhất so với AIC 2025

Mỗi truy vấn được nộp **tối đa 100 đáp án đã xếp hạng**. Mỗi đáp án nhận một **R-Score ∈ [0, 1]**:

- **Textual KIS:** `R = I(vᵢ = GTᵥ ∧ idᵢ ∈ [s, e])` — đúng video **và** frame nằm trong đoạn đáp án.
- **Q&A:** như trên, cộng thêm điều kiện `answer` khớp **về mặt ngữ nghĩa**.
- **TRAKE:** sai `video_id` ⇒ **0 điểm ngay lập tức**. Đúng video thì `R = (1/N) · Σⱼ I(idᵢ,ⱼ ∈ [sⱼ, eⱼ])` — tỉ lệ khoảnh khắc khớp. ⚠️ Mỗi đoạn `[sⱼ, eⱼ]` **thường dưới 10 frame**.

**Điểm cuối cùng của một truy vấn:**

```
R@k        = max(R-Score của k đáp án đầu tiên),   k ∈ {1, 5, 20, 50, 100}
FinalScore = (R@1 + R@5 + R@20 + R@50 + R@100) / 5
```

Ví dụ trong đề: đáp án #1 có R = 0,5; đáp án #3 có R = 0,8 (cao nhất) ⇒ `(0,5 + 0,8 + 0,8 + 0,8 + 0,8) / 5 = 0,74`.

**Ba hệ quả bắt buộc nhớ khi thiết kế:**

1. **Không còn phạt theo thời gian trả lời hay số lần nộp sai.** Latency không còn là yêu cầu *chức năng*; nó chỉ còn là yêu cầu vận hành (người ngồi duyệt 100 kết quả).
2. **`R@1` chiếm 1/5 tổng điểm** trong khi `R@5`…`R@100` thường bằng nhau. Nghĩa là **đẩy đáp án đúng lên hạng 1** đáng giá gần bằng toàn bộ phần còn lại — đây chính là thứ `MRR` và `Recall@1` đo.
3. **Bộ K của `eval/` phải là `{1, 5, 20, 50, 100}`**, đúng bộ K mà BTC chấm. Đo `Recall@10` mà không đo `Recall@20`/`@50` là đo lệch thước.

---

## 2. Dataset — AIC 2026 batch 1 (dữ liệu công khai)

> ✅ **Cập nhật 2026-08-01.** Đề vòng sơ tuyển ghi rõ: dữ liệu đợt 1 của AIC 2026 **chính là batch 1 của AIC 2025** — nội dung không đổi, chỉ đổi tên cuộc thi. **Batch 2 sẽ được bổ sung sau**, nên mọi con số dưới đây là của batch 1 và sẽ phải đo lại khi batch 2 về.
>
> Số liệu **đo trực tiếp trên đĩa ngày 2026-08-01**, không phải ước lượng.

- Gốc dữ liệu trên máy: **`data/`** (bố cục phẳng — xem §2.2)
- Nội dung: video **tin tức tiếng Việt** (HTV — *60 Giây Official*, …), có tiếng nói và **chữ trên màn hình**

### 2.1 Thống kê

| Thuộc tính | Giá trị đo được |
|---|---|
| Số `video_id` duy nhất | **873** (batch `L21` … `L30`) |
| Số file `.mp4` | **873** — bản tải hiện tại **không còn** thư mục trùng id (`Videos_L25_a1` / `Videos_L25_b` của bản Kaggle cũ đã không còn) |
| Tổng thời lượng | **470.428 giây ≈ 130,7 giờ** (tổng trường `length` của `media-info`) |
| Thời lượng trung bình | ~9,0 phút/video |
| Keyframe **BTC trích sẵn** | 177.321 ảnh JPG 1280×720 — ⚠️ **chưa tải về**, xem §2.2 |

**Bẫy khi đếm video vẫn giữ nguyên trong code.** Bản tải hiện tại 1 file = 1 id, nhưng bản Kaggle trước đó có 961 file / 873 id. Adapter **vẫn khử trùng theo `video_id`** — bỏ bước này thì một lần tải lại khác layout là mọi mẫu số metric bị thổi phồng mà không có gì báo động.

### 2.2 Bố cục thư mục

**Đã tải về (`data/`):**

```
data/
├── Videos_<batch>/video/<video_id>.mp4              # 873 file .mp4
├── media-info-aic25-b1/media-info/<video_id>.json   # 873 file metadata YouTube
└── map-keyframes-aic25-b1/map-keyframes/<video_id>.csv   # 873 file, n,pts_time,fps,frame_idx
```

**Chưa tải — và có chủ đích (quyết định chủ dự án 2026-08-01):**

| Thành phần BTC | Vì sao không tải |
|---|---|
| `Keyframes_<batch>/` | Dự án **tự trích keyframe bằng TransNetV2** (PRD §F1). Keyframe BTC chỉ là đối chứng |
| `objects/` | Object detection **tự chạy bằng model của mình** |
| `clip-features-32/` | Vector chính do encoder của dự án sinh; feature BTC chỉ dùng đối chứng |

⇒ **`map-keyframes` là nguồn đối chứng độc lập duy nhất còn lại** cho chất lượng cắt shot (R10, PRD §6.2). Đừng bỏ nó.

`<batch>` gồm: `L21` `L22` `L23` `L24` `L25` `L26_a`…`L26_e` `L27` `L28` `L29` `L30`. Tên thư mục **không nhất quán** giữa các bản phát hành (bản trước có `Video_L30_a` số ít và lồng thêm một tầng `Videos/Videos/`). Adapter phải **glob**, không dựng đường dẫn bằng chuỗi định dạng — `AIC2026Adapter` glob `Video*/video/*.mp4` ở mọi độ sâu chính vì lý do này.

⚠️ **Đề AIC 2026 ghi rõ: "một số video có thể không có file metadata tương ứng."** Bản tải hiện tại đủ 873/873, nhưng adapter vẫn phải chịu được thiếu — khi thiếu, `duration_sec` suy từ dòng cuối `map-keyframes` (`frame_idx / fps`, là **cận dưới**) và `author` để `None`. Thiếu **cả hai** nguồn thì raise, không đoán.

### 2.3 Bốn nhánh `features/`

**`map-keyframes/<video_id>.csv`** — cầu nối giữa chỉ số keyframe và thời gian:

```csv
n,pts_time,fps,frame_idx
1,0.0,30.0,0
2,3.0,30.0,90
3,8.7,30.0,261
```

`n` khớp **1-1** với tên file `NNN.jpg` (đã verify: `L21_V001` 307 ảnh ↔ 307 dòng; `L26_V050` 149 ↔ 149; `L30_V090` 86 ↔ 86). `pts_time` là timestamp giây — nguồn `timestamp` miễn phí cho keyframe của BTC.

**`clip-features-32/<video_id>.npy`** — `shape = (n_keyframes, 512)`, `dtype = float16`. 512 chiều ⇒ **CLIP ViT-B/32**.

✅ **Đã xác nhận L2-normalize** (đo 2026-07-29 trên `L21_V001`, `L26_V050`, `L30_V090`): norm mỗi hàng ∈ [0,9995 · 1,0005], trung bình 1,0000 — sai lệch đúng bằng mức làm tròn của float16. Dùng được trực tiếp với `COSINE` / `IP`.

⚠️ **Checkpoint và preprocessing chính xác không được công bố** — không biết là bản OpenAI gốc hay bản `open_clip` huấn luyện trên LAION. Nên feature này chỉ dùng để **đối chứng**, không dùng làm vector chính (PRD §10/Q8b).

**`media-info/<video_id>.json`** — metadata YouTube của video gốc:

```json
{
  "author": "60 Giây Official",
  "channel_id": "UCRjzfa1E0gA50lvDQipbDMg",
  "channel_url": "https://www.youtube.com/channel/UCRjzfa1E0gA50lvDQipbDMg",
  "description": "...",
  "keywords": ["..."],
  "length": 1262,
  "publish_date": "01/08/2024",
  "thumbnail_url": "https://i.ytimg.com/vi/<yt_id>/sddefault.jpg",
  "title": "60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024",
  "watch_url": "https://youtube.com/watch?v=<yt_id>"
}
```

Đây là nguồn cho `VideoRef.duration_sec` (trường `length`, đơn vị **giây**) và cho các trường lọc metadata trong MongoDB/Milvus.

⚠️ Hai bẫy định dạng: `publish_date` là chuỗi **`DD/MM/YYYY`** (`01/08/2024` = 1 tháng 8), **không** phải ISO-8601 — phải parse tường minh, đừng để thư viện đoán. Và **không có trường nào tương đương `recipe_type`** của YouCook2 — trường phân nhóm thay thế là `author` (kênh) và `batch` (`L21`…`L30`).

**`objects/<video_id>/NNN.json`** — object detection cho **từng** keyframe của BTC:

```json
{
  "detection_scores": ["0.79673874", "0.6866252"],
  "detection_class_entities": ["Lantern", "Skyscraper"],
  "detection_class_names": ["/m/01jfsr", "/m/079cl"],
  "detection_boxes": [["0.46860334", "0.36642352", "0.63618606", "0.46714845"]],
  "detection_class_labels": ["84", "379"]
}
```

Nhãn theo bộ **Open Images** (`/m/...`). ⚠️ Mọi giá trị lưu dưới dạng **chuỗi** kể cả số — phải ép kiểu khi đọc. `detection_boxes` chuẩn hoá theo thứ tự `[ymin, xmin, ymax, xmax]`.

### 2.4 Không có ground truth trong dataset

**Dataset không chứa bất kỳ file nhãn truy vấn nào** — không có caption↔segment, không có chuỗi sự kiện TRAKE, không có cặp QA. Đã quét toàn bộ cây thư mục: không có file `.csv`/`.json`/`.txt`/`.md` nào ở cấp gốc.

Đây là đặc thù của cuộc thi: phần công khai chỉ dùng để **xây index và truy hồi**; đáp án của bộ truy vấn thi **chỉ biết được qua việc nộp bài** lên hệ thống BTC. Hệ quả trực tiếp: mọi metric offline của dự án phải chạy trên **bộ eval tự sinh** — giao thức sinh nhãn nằm ở §6.1 của [`PRD.md`](./PRD.md).

### 2.5 Quan hệ với pipeline của dự án

Dự án **tự trích keyframe bằng TransNetV2** theo đúng baseline (quyết định của chủ dự án, 2026-07-29), **không** dùng `Keyframes_<batch>/` làm nguồn ingest. Dữ liệu BTC trích sẵn vẫn có bốn công dụng:

| Tài nguyên BTC | Dùng làm gì |
|---|---|
| `Keyframes_<batch>/` + `map-keyframes/` | **Đối chứng** chất lượng shot detection của mình (metric phụ tIoU, PRD §6.2); phương án rút lui nếu TransNetV2 không dựng được |
| `clip-features-32/` | Sanity-check. ⚠️ **Không** so trực tiếp với hệ thống chính ở mức segment — chúng gắn với keyframe của BTC, không phải keyframe của mình. Baseline CLIP ViT-B/32 trong PRD §6.3 phải chạy **bằng encoder của mình trên keyframe của mình** thì so sánh mới công bằng |
| `media-info/` | Nguồn metadata video, nạp thẳng vào MongoDB |
| `objects/` | Kênh truy hồi bổ sung tiềm năng (**chưa cam kết**) — gắn với keyframe BTC nên cần ánh xạ theo timestamp |

---

## 3. Model lõi

### 3.1 TransNetV2 — phát hiện chuyển cảnh

📄 Paper dùng để cắt video thành shot.

- Repo: https://github.com/soCzech/TransNetV2
- Paper: [*TransNet V2: An effective deep network architecture for fast shot transition detection*](https://arxiv.org/abs/2008.04838)
- `llms.txt`: ❌ không có

**Cài đặt:** không có pip package — phải clone repo. Có hai bản inference: thư mục `inference/` (TensorFlow) và `inference-pytorch/` (PyTorch). **Dùng bản PyTorch** để đồng nhất với phần còn lại của dự án.

**API (trích từ `inference-pytorch/transnetv2_pytorch.py`):**

```python
class TransNetV2(nn.Module):
    def __init__(self, F=16, L=3, S=2, D=1024,
                 use_many_hot_targets=True,
                 use_frame_similarity=True,
                 use_color_histograms=True,
                 use_mean_pooling=False,
                 dropout_rate=0.5, ...)

    def forward(self, inputs): ...
```

**Ràng buộc đầu vào — rất quan trọng:** tensor `torch.uint8` shape `[B, T, 27, 48, 3]`. Model tự assert và báo lỗi `"incorrect input type and/or shape"` nếu sai. Nghĩa là phải resize frame về **48×27** (rộng × cao) trước khi đưa vào — đây cũng là lý do bước giải mã video rẻ hơn nhiều so với giải mã full-res.

> Không cần train lại. README ghi rõ: *"You do not need to train the network, use code and instructions in inference folder to detect shots in your videos."*

**✅ Cập nhật 2026-08-01 — checkpoint đã dựng xong (P0.7).** Không có gói pip lẫn checkpoint PyTorch chính chủ. Đã xác nhận: trọng số TensorFlow gốc được commit thẳng qua Git LFS trong `inference/transnetv2-weights/`; convert sang PyTorch bằng chính `convert_weights.py` của tác giả (`scripts/setup_transnetv2.sh`, chạy trong venv tạm với TensorFlow 2.17 — tương thích ngược tốt với SavedModel lưu bằng TF 2.1). Kiến trúc model vendor thẳng vào `pipelines/vendor/transnetv2_upstream/` (MIT license), không import qua pip.

- Checkpoint: `models/transnetv2/transnetv2-pytorch-weights.pth` (~30,5 MB)
- SHA256 (tự sinh lúc convert, không phải checksum công bố sẵn — dùng để phát hiện lệch khi chạy lại script):
  `eed5336d5d6a013c67f5863505a26e7e835053e64a9ce413d6b089ccba07bb53`
- Verify: `TRANSNETV2_CHECKPOINT_PATH=models/transnetv2/transnetv2-pytorch-weights.pth pytest -m model tests/test_shot_detect_model.py -v` → pass trên video pilot thật `L21_V001.mp4`.

### 3.2 BEiT-3 — encoder đa phương thức

📄 Paper dùng làm encoder chính, đưa ảnh và chữ về chung không gian vector.

- Repo: https://github.com/microsoft/unilm/tree/master/beit3
- Hướng dẫn retrieval: https://github.com/microsoft/unilm/blob/master/beit3/get_started/get_started_for_retrieval.md
- Paper: [*Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks*](https://arxiv.org/abs/2208.10442)
- `llms.txt`: ❌ không có

**⚠️ Đây là rủi ro kỹ thuật số 1 của dự án (R1 trong PRD).** BEiT-3 **không có** trong HuggingFace Transformers. Cần:

| Phụ thuộc | Ghi chú |
|---|---|
| `torchscale` | Thư viện của Microsoft, chứa kiến trúc Multiway Transformer |
| `timm` | Repo BEiT-3 được xây trên timm; thường cần **bản pin cũ** |
| `beit3.spm` | Tokenizer SentencePiece — tải riêng, không đi kèm code |
| Checkpoint | Tải từ trang release của repo unilm |

**Chọn biến thể nào:** repo phân biệt rõ hai nhóm — `beit3_base` / `beit3_large` (deep fusion, cho các task vision-language cần suy luận chéo) và **`beit3_base_itc` / `beit3_large_itc`** (image-text contrastive). Với **image-text retrieval, dùng bản `_itc`** — repo ghi rõ nhóm này cho kết quả tốt hơn ở task retrieval.

**Với RTX 3070 8GB: vẫn chọn `beit3_base_itc`**, không phải `large` — giờ vừa bám baseline vừa là giới hạn VRAM thật (đổi từ RTX 2080Ti 11GB khi chuyển máy, 2026-08-01).

**Phương án rút lui:** nếu quá 1 ngày chưa dựng được, chuyển sang SigLIP hoặc open_clip qua `MultimodalEncoder` Protocol (§5.1 của [`architecture.md`](./architecture.md)) và ghi nhận đây là sai lệch so với baseline.

---

## 4. Hạ tầng lưu trữ & tìm kiếm

### 4.1 Milvus — vector database

- Docs: https://milvus.io/docs
- **`llms.txt`: ✅ https://milvus.io/llms.txt**
- Python SDK reference: https://milvus.io/api-reference/pymilvus/v2.6.x/About.md

> Docs nhấn mạnh: **luôn dùng interface `MilvusClient`** (từ v2.4+), tránh ORM API cũ.

**Cài đặt:** `pip install pymilvus`

**Kết nối** (nguồn: https://milvus.io/docs/single-vector-search.md):

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)
```

**Tạo schema và collection** (nguồn: https://milvus.io/docs/create-collection.md):

```python
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR, max_length=512)

index_params = client.prepare_index_params()
index_params.add_index(field_name="my_id", index_type="AUTOINDEX")
index_params.add_index(
    field_name="my_vector",
    index_type="AUTOINDEX",
    metric_type="COSINE",
)

client.create_collection(
    collection_name="customized_setup_1",
    schema=schema,
    index_params=index_params,
)
```

**Index HNSW** (nguồn: https://milvus.io/docs/hnsw.md):

```python
index_params.add_index(
    field_name="your_vector_field_name",
    index_type="HNSW",
    index_name="vector_index",
    metric_type="L2",          # dự án dùng COSINE
    params={"M": 64, "efConstruction": 100},
)
```

Khoảng khuyến nghị của docs: `M ∈ [5, 100]`, `efConstruction ∈ [50, 500]`.

**Tìm kiếm:**

```python
search_params = {"params": {"ef": 10}}

res = client.search(
    collection_name="your_collection_name",
    anns_field="vector_field",
    data=[[0.1, 0.2, 0.3, 0.4, 0.5]],
    limit=10,
    search_params=search_params,
    output_fields=["video_id", "timestamp"],
)
```

**Ghi chú cho dự án:** với `COSINE`, giá trị lớn hơn = giống hơn, dải `[-1, 1]`. Vì embedding đã L2-normalize nên `COSINE` và `IP` cho cùng thứ hạng — vẫn khai báo `COSINE` cho rõ nghĩa. Bộ lọc metadata phải đặt trong tham số `filter` của `search()`, **không** lọc sau khi lấy kết quả về.

**Trang docs quan trọng khác:**
- Dense vector: https://milvus.io/docs/dense-vector.md
- IVF-FLAT: https://milvus.io/docs/ivf-flat.md
- Insert / Update / Delete: https://milvus.io/docs/insert-update-delete.md

### 4.2 Elasticsearch — tìm kiếm văn bản

- Docs: https://www.elastic.co/docs
- **`llms.txt`: ✅ https://www.elastic.co/docs/llms.txt** (đúng chuẩn spec — có heading, mô tả, danh sách link kèm chú thích)
- Python client: https://www.elastic.co/docs/reference/elasticsearch/clients/python
- Text analysis: https://www.elastic.co/docs/reference/text-analysis

**Cài đặt:** `pip install elasticsearch`

**Client và tìm kiếm** (nguồn: trang Python client ở trên):

```python
import os
from elasticsearch import Elasticsearch

client = Elasticsearch(
    hosts=[os.getenv("ELASTICSEARCH_URL")],
    api_key=os.getenv("ELASTIC_API_KEY"),
)

resp = client.search(
    index="my-index-000001",
    from_=40,
    size=20,
    query={"term": {"user.id": "kimchy"}},
)
```

> Lưu ý cú pháp: tham số phân trang là `from_` (có dấu gạch dưới) vì `from` là từ khoá Python. Với ingest lượng lớn, dùng **bulk helpers** thay vì gọi index từng document.

### 4.3 Plugin phân tích tiếng Việt

📄 Paper dùng "Vietnamese Analysis Plugin" cho Elasticsearch.

- Repo: https://github.com/duydo/elasticsearch-analysis-vietnamese
- `llms.txt`: ❌ không có
- Dựa trên bộ tách từ C++ của đội CocCoc

**Cung cấp:** analyzer `vi_analyzer`, tokenizer `vi_tokenizer`, token filter `vi_stop`.

**⚠️ Cài đặt KHÔNG chỉ là install jar.** Plugin gọi bộ tách từ C++ của CocCoc qua JNI —
thiếu thư viện native `libcoccoc_tokenizer_jni.so` trên `java.library.path` thì
Elasticsearch **crash lúc khởi động** với `java.lang.UnsatisfiedLinkError: no
coccoc_tokenizer_jni in java.library.path` (phát hiện thực nghiệm lúc build P0.2,
2026-07-26 — lỗi này không lộ ra khi chỉ đọc `pom.xml`).

Build đúng cần 3 bước (xem `Dockerfile` chính thức của tác giả tại tag
`v9.3.0`, không có trong repo docs nhưng có trong chính repo plugin):

```sh
# 1. Build thư viện native (C++/cmake) — https://github.com/duydo/coccoc-tokenizer
git clone --depth 1 https://github.com/duydo/coccoc-tokenizer.git
cmake -DJAVA_HOME=$JAVA_HOME -DBUILD_JAVA=1 -S coccoc-tokenizer -B coccoc-tokenizer/build
cmake --build coccoc-tokenizer/build --target install
# → /usr/local/lib/libcoccoc_tokenizer_jni.so + /usr/local/share/tokenizer/dicts

# 2. Build plugin jar (Maven, cùng phiên bản Elasticsearch)
mvn --batch-mode -Dmaven.test.skip -e package -DprojectVersion=9.3.0

# 3. Cài cả hai vào Elasticsearch
cp libcoccoc_tokenizer_jni.so /usr/lib/
cp -r tokenizer/dicts /usr/local/share/tokenizer/dicts
bin/elasticsearch-plugin install file://target/releases/elasticsearch-analysis-vietnamese-9.3.0.zip
```

Dockerfile 3-stage đầy đủ (build native lib + build jar + image cuối) đã lưu ở
`docker/elasticsearch/Dockerfile` của dự án — dùng lại thay vì viết lại từ đầu.

**Mapping cơ bản:**

```json
PUT my-vi-index-00001
{
  "settings": {
    "analysis": {
      "analyzer": {
        "my_vi_analyzer": {
          "type": "vi_analyzer",
          "keep_punctuation": true,
          "stopwords": ["rất", "những"]
        }
      }
    }
  }
}
```

**Mapping có `asciifolding`** — dùng bản này cho dự án, vì người vận hành hay gõ không dấu khi thi gấp:

```json
PUT my-vi-index-00002
{
  "settings": {
    "analysis": {
      "analyzer": {
        "my_vi_analyzer": {
          "tokenizer": "vi_tokenizer",
          "filter": ["lowercase", "ascii_folding"]
        }
      },
      "filter": {
        "ascii_folding": {
          "type": "asciifolding",
          "preserve_original": true
        }
      }
    }
  }
}
```

**Bảng tương thích phiên bản** — ràng buộc trực tiếp việc chọn version Elasticsearch trong `docker-compose.yml`:

| Plugin | Elasticsearch |
|---|---|
| 9.3.0 | 9.0.0 – 9.3.0 |
| 8.7.0 | 8.7.0 |
| 8.4.0 | 8.4.0 – 8.7.1 |
| 8.0.0 | 8.0.0 – 8.0.x |
| 7.16.1 | 7.16 – 7.17.1 |

Ví dụ tách từ: `"Cộng hòa Xã hội chủ nghĩa Việt Nam"` → `["cộng hòa", "xã hội", "chủ nghĩa", "việt nam"]` — tách theo **từ ghép**, không tách theo âm tiết. Đây chính là giá trị của plugin so với analyzer mặc định.

### 4.4 MongoDB / PyMongo — metadata

- Docs: https://www.mongodb.com/docs
- **`llms.txt`: ✅ https://www.mongodb.com/docs/llms.txt**
- **`llms.txt` riêng cho PyMongo: ✅ https://www.mongodb.com/docs/languages/python/pymongo-driver/current/llms.txt**

**Cài đặt:** `pip install pymongo`

**Các trang cần dùng (lấy từ `llms.txt` của PyMongo):**

| Chủ đề | URL |
|---|---|
| Tạo `MongoClient` | https://www.mongodb.com/docs/languages/python/pymongo-driver/current/connect/mongoclient.md |
| Insert documents | https://www.mongodb.com/docs/languages/python/pymongo-driver/current/crud/insert.md |
| **Bulk write** (dùng cho ingest) | https://www.mongodb.com/docs/languages/python/pymongo-driver/current/crud/bulk-write.md |
| Update documents | https://www.mongodb.com/docs/languages/python/pymongo-driver/current/crud/update.md |
| Query / find | https://www.mongodb.com/docs/languages/python/pymongo-driver/current/crud/query/find.md |
| **Indexes** | https://www.mongodb.com/docs/languages/python/pymongo-driver/current/indexes.md |

**Ghi chú cho dự án:**
- Tạo **unique index** trên `keyframe_id` và `video_id` — đây là cơ chế bảo đảm tính idempotent của ingest (yêu cầu T1)
- Ingest dùng **bulk write**, không insert từng document
- Từ FastAPI (`async def`) phải dùng client bất đồng bộ, không dùng `pymongo` đồng bộ (theo `.claude/rules/ecc/python/fastapi.md`)

---

## 5. LLM / VLM API

### 5.1 Google Gemini — OCR + VQA

📄 Paper dùng Gemini cho OCR. 🔧 Dự án dùng thêm cho VQA answering.

- Docs: https://ai.google.dev/gemini-api/docs
- **`llms.txt`: ✅ https://ai.google.dev/gemini-api/docs/llms.txt** (các trang con có hậu tố `.md.txt`)

**Cài đặt:** `pip install google-genai`

**⚠️ API hiện tại là Interactions API (`client.interactions.create`)**, không phải `generate_content` như các hướng dẫn cũ trên mạng.

**Gửi ảnh + prompt — ảnh inline** (nguồn: https://ai.google.dev/gemini-api/docs/image-understanding.md.txt):

```python
import base64
from google import genai

with open('path/to/small-sample.jpg', 'rb') as f:
    image_bytes = f.read()

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=[
        {"type": "text", "text": "Caption this image."},
        {
            "type": "image",
            "data": base64.b64encode(image_bytes).decode('utf-8'),
            "mime_type": "image/jpeg",
        },
    ],
)
print(interaction.output_text)
```

**Gửi ảnh qua File API** (dùng khi ảnh lớn hoặc tái sử dụng nhiều lần):

```python
my_file = client.files.upload(file="path/to/sample.jpg")

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=[
        {"type": "text", "text": "Caption this image."},
        {"type": "image", "uri": my_file.uri, "mime_type": my_file.mime_type},
    ],
)
```

**Đầu ra JSON có cấu trúc** (nguồn: https://ai.google.dev/gemini-api/docs/structured-output.md.txt) — dùng cho cả OCR và VQA:

```python
from pydantic import BaseModel, Field
from google import genai

class Recipe(BaseModel):
    recipe_name: str = Field(description="The name of the recipe.")
    ingredients: list[str]

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="your prompt here",
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": Recipe.model_json_schema(),
    },
)

recipe = Recipe.model_validate_json(interaction.output_text)
```

**💰 Chi phí token — con số quyết định tính khả thi của nhánh OCR:**

> *"Larger images are tiled into 768x768 pixel tiles, each costing 258 tokens."* Ảnh nhỏ hơn 384×384 tính 258 token.

Keyframe của AIC 2026 là **1280×720** ⇒ cắt thành **2 tile** 768×768 ⇒ **516 token/ảnh**. Hạ về ≤ 384×384 trước khi gửi thì còn **258 token/ảnh** — đây là đòn bẩy chi phí rẻ nhất, nhưng phải kiểm chứng chữ nhỏ trên màn hình có còn đọc được không.

Quy đổi trên số keyframe **BTC trích sẵn** (mốc tham chiếu — số keyframe do TransNetV2 của dự án sinh ra ⚠️ **[Cần đo]**, xem PRD §8/R2):

| Phạm vi | Keyframe | @516 token | @258 token |
|---|---|---|---|
| Pilot `L21`–`L24` (128 video) | 26.003 | ~13,4 triệu | ~6,7 triệu |
| Toàn bộ (873 video) | 177.321 | ~91,5 triệu | ~45,8 triệu |

Đây chính là cơ sở của rủi ro R2 trong PRD và lý do phải lọc frame có chữ + dedupe **trước** khi gọi. Khác YouCook2 ở một điểm quan trọng: video tin tức có **tỉ lệ frame chứa chữ rất cao**, nên bước lọc [1] sẽ cắt được ít hơn kỳ vọng — bước dedupe [2] mới là chỗ tiết kiệm chính (bản tin giữ nguyên khung đồ hoạ suốt nhiều giây liền).

**💵 Giá model (fetch 2026-08-09, `https://ai.google.dev/gemini-api/docs/pricing`, paid tier, USD/1M token):**

| Model | Input | Output | Ghi chú |
|---|---|---|---|
| `gemini-3.6-flash` | $1,50 | $7,50 | Mặc định `configs/ingest/ocr.yaml` (nhánh OCR, đang hoãn) |
| `gemini-2.5-flash` | **$0,30** | **$2,50** | Dùng cho `eval/build_eval_set.py` (sinh caption bộ nhãn), chủ dự án chốt 2026-08-09 |

Đã kiểm chứng `gemini-2.5-flash` **có** trong bảng model được **Interactions API** hỗ trợ (`https://ai.google.dev/gemini-api/docs/interactions.md.txt`: *"Gemini 2.5 Flash | Model | gemini-2.5-flash"*) — không bị ép dùng `generate_content`.

🛑 **Nhưng KHÔNG dùng `gemini-2.5-flash` cho bất kỳ nhánh nào cần structured output (OCR, VQA, sinh nhãn).** Test thực nghiệm 2026-08-09 (`eval/build_eval_set.py`, cùng ảnh + cùng request cho cả hai model): `gemini-2.5-flash` **lờ hoàn toàn tham số `response_format`** — trả về văn bản thường thay vì JSON đúng schema, dù request đã verify đúng shape qua chính SDK (không phải lỗi code). `gemini-3.6-flash` trả đúng JSON `{"caption": "..."}` với nguyên văn cùng request. Interactions API hỗ trợ `response_format` như một *tính năng model 3.x*, không phải mọi model được liệt kê trong bảng "supported models" đều enforce nó. ⇒ **`gemini-3.6-flash` là lựa chọn bắt buộc**, không phải tuỳ chọn, cho mọi lệnh gọi Gemini có `response_format` trong dự án này. Bảng giá bên dưới giữ lại làm tư liệu, không phải khuyến nghị dùng 2.5-flash.

**Trang khác cần dùng:**
- Batch API (giảm chi phí ingest hàng loạt): https://ai.google.dev/gemini-api/docs/batch-api.md.txt
- Embeddings: https://ai.google.dev/gemini-api/docs/embeddings.md.txt
- Get started: https://ai.google.dev/gemini-api/docs/get-started.md.txt

### 5.2 OpenAI — viết lại truy vấn (QUEST nhánh 1)

📄 Paper dùng GPT-4o-mini cho nút "AI enhancer".

- Docs: https://developers.openai.com/api
- **`llms.txt`: ✅ https://developers.openai.com/api/llms.txt** (có thêm bản đầy đủ `llms-full.txt`)

**Cài đặt:** `pip install openai`

**API hiện tại là Responses API.** Cho QUEST nhánh 1, dùng structured output để nhận về truy vấn đã viết lại ở dạng có kiểu (nguồn: https://developers.openai.com/api/docs/guides/structured-outputs.md):

```python
from pydantic import BaseModel

class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

response = client.responses.parse(
    model="gpt-5.6",
    input=[
        {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
        {"role": "user", "content": "how can I solve 8x + 7 = -23"},
    ],
    text_format=MathReasoning,
)
```

**Đọc kết quả — phải xử lý cả trường hợp model từ chối:**

```python
for output in response.output:
    if output.type != "message":
        continue
    for item in output.content:
        if item.type == "refusal":
            print(item.refusal)
            continue
        if not item.parsed:
            raise Exception("Could not parse response")
        print(item.parsed)
```

**Trang khác:**
- Responses API overview: https://developers.openai.com/api/reference/responses/overview.md
- Images và vision: https://developers.openai.com/api/docs/guides/images-vision.md
- Quickstart: https://developers.openai.com/api/docs/quickstart.md

---

### 5.3 faster-whisper — ASR

🔧 Dự án (P0.11). Baseline arXiv:2512.13169 **không** có ASR.

- Repo: https://github.com/SYSTRAN/faster-whisper
- **`llms.txt`: ❌ không có** — nguồn là README + đọc thẳng `faster_whisper/transcribe.py`. Đã fetch kiểm chứng **2026-08-02**.

**Cài đặt:** `pip install faster-whisper` — kéo theo `ctranslate2`, `av`, `tokenizers`, `onnxruntime` (VAD). GPU cần **cuBLAS + cuDNN 9 (CUDA 12)**. ✅ Version pin đã xác nhận 2026-08-02 (`pyproject.toml` nhóm `asr`): `faster-whisper==1.2.1` (bản mới nhất trên PyPI tại thời điểm implement, phát hành 2025-10-31), kéo theo `ctranslate2>=4.0,<5`, `huggingface-hub>=0.21`, `tokenizers>=0.13,<1`, `onnxruntime>=1.14,<2`, `av>=11`.

**⚠️ Không cần bước trích audio riêng.** README: *"The audio is decoded with the Python library PyAV which bundles the FFmpeg libraries"* ⇒ `transcribe()` nhận **thẳng đường dẫn `.mp4`**, không phải extract `.wav` trước.

**Khởi tạo + transcribe** (nguồn: README + `faster_whisper/transcribe.py`):

```python
from faster_whisper import WhisperModel

# WhisperModel.__init__(model_size_or_path, device="auto", device_index=0,
#     compute_type="default", cpu_threads=0, num_workers=1,
#     download_root=None, local_files_only=False, revision=None, ...)
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

segments, info = model.transcribe(
    "data/Videos_L21_a/video/L21_V001.mp4",   # .mp4 trực tiếp
    language="vi",
    beam_size=5,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500},
    condition_on_previous_text=False,   # chống vòng lặp ảo giác
)
```

**⚠️ `segments` là generator LAZY** — `transcribe()` trả `Tuple[Iterable[Segment], TranscriptionInfo]` và **chưa chạy gì** cho tới khi iterate. Đừng đo thời gian ở dòng `transcribe(...)`.

**Kiểu dữ liệu trả về** (nguồn: `faster_whisper/transcribe.py`):

```python
@dataclass
class Segment:
    id: int; seek: int
    start: float; end: float          # giây
    text: str
    tokens: List[int]
    avg_logprob: float                # ← lọc ảo giác
    compression_ratio: float
    no_speech_prob: float             # ← lọc ảo giác
    words: Optional[List[Word]]       # chỉ khi word_timestamps=True
    temperature: Optional[float]

@dataclass
class Word:
    start: float; end: float; word: str; probability: float

@dataclass
class TranscriptionInfo:
    language: str; language_probability: float
    duration: float; duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionOptions
    vad_options: VadOptions
```

**Tham số `transcribe()` liên quan tới chất lượng** (mặc định trong ngoặc): `no_speech_threshold` (0.6) · `log_prob_threshold` (−1.0) · `compression_ratio_threshold` (2.4) · `condition_on_previous_text` (**True** — nên đặt `False`) · `hallucination_silence_threshold` (None) · `word_timestamps` (False) · `hotwords` (None).

**💰 Chi phí — con số quyết định tính khả thi:** chạy **cục bộ, miễn phí**, không tốn quota Gemini. Benchmark chính thức trong README trên **RTX 3070 Ti** (audio 13 phút):

| compute_type | Thời gian | VRAM |
|---|---|---|
| fp16 | 1m03s | ~6090 MB |
| int8 | 59s | ~2926 MB |

⇒ Ngoại suy 130,7 giờ (873 video) ≈ **10–12 giờ GPU**; pilot L21 (29 video) ≈ **[Cần đo]**. Vừa RTX 3070 8GB.

~~**⚠️ PhoWhisper-large KHÔNG dùng được với faster-whisper.** Model card [vinai/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large) chỉ có bản HF `transformers` — **không có bản convert CTranslate2**, và không phơi `no_speech_prob` nên **không chạy được bộ lọc ảo giác**.~~

**✅ Đính chính 2026-08-07 — sai cả hai vế** (fetch kiểm chứng 2026-08-05, chi tiết `docs/research-asr-phowhisper-batched.md` §3): **có** bản CTranslate2 công khai (`quocphu/PhoWhisper-ct2-FasterWhisper`, `kiendt/PhoWhisper-large-ct2`, `qbsmlabs/PhoWhisper-large`) hoặc tự convert một lệnh (`ct2-transformers-converter --model vinai/PhoWhisper-large ...`). `no_speech_prob`/`avg_logprob` là **output của bộ giải mã faster-whisper**, không phải thuộc tính checkpoint — mọi model qua `WhisperModel` đều có, kể cả bản fine-tune, kể cả chế độ batched. Bộ lọc ảo giác **chạy được với PhoWhisper**. Là bước nâng cấp **có đo** về sau, không phải baseline.

---

### 5.4 `bge-m3` — encoder dense cho transcript (P0.11b)

🔧 Dự án (P0.11b). Đã fetch kiểm chứng 2026-08-02 (model card HuggingFace + trang PyPI `sentence-transformers`).

- Model card: https://huggingface.co/BAAI/bge-m3
- **Dimension: 1024.** Max sequence length: 8192 token.
- **Cài đặt:** `pip install sentence-transformers==5.6.1` (đã pin trong `pyproject.toml` nhóm `asr-embed`) — deps tương thích với `transformers==4.44.2` đã pin cho BEiT-3 (`sentence-transformers` yêu cầu `transformers<6,>=4.41`).
- **Dùng qua `sentence-transformers`**, không dùng `FlagEmbedding` (package gốc của BAAI) — API đơn giản hơn vì chỉ cần vector dense, không cần sparse/ColBERT-style multi-vector mà `FlagEmbedding` mặc định trả về:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("BAAI/bge-m3", device="cuda")
  embeddings = model.encode(sentences)  # (N, 1024)
  ```
- ⚠️ **Không rõ output có tự L2-normalize hay không** — model card không ghi rõ. `Bgem3Encoder` (app/encoders/bge_m3.py) **tự chuẩn hoá lại**, không tin vào default của thư viện (CLAUDE.md §3).
- Đa ngôn ngữ ">100 ngôn ngữ hoạt động" — tiếng Việt không được liệt kê tường minh trong danh sách mẫu của model card, nhưng bge-m3 vốn huấn luyện đa ngôn ngữ quy mô lớn; **chất lượng tiếng Việt cụ thể [Cần đo]** khi có bộ eval ASR-aware (P0.11 Bước 3).
- VRAM/tốc độ: **[Cần đo]** — chưa benchmark riêng trên RTX 3070; kế hoạch chạy tuần tự sau Whisper (không cùng lúc) trên 8GB.

---

### 5.5 `ddgs` — tìm ảnh mẫu web (QUEST nhánh 2)

🔧 Dự án (P2.5). Đã fetch kiểm chứng 2026-08-14 (PyPI + GitHub `deedy5/ddgs`) — thay cho phương án ban đầu, xem `docs/research-quest-llm-options.md` để biết lý do loại Google Custom Search JSON API (đóng cửa khách mới) và Bing Image Search API (Microsoft khai tử 11/08/2025).

- PyPI: https://pypi.org/project/ddgs/ · GitHub: https://github.com/deedy5/ddgs
- **Cài đặt:** `pip install ddgs` — package đổi tên từ `duckduckgo_search` cũ, **import mới là `from ddgs import DDGS`**, không phải `duckduckgo_search`.
- ⚠️ **Không phải API chính thức** — scraper kết quả tìm kiếm, tự khai "for educational purposes only" trong README. Không có SLA, không công bố giới hạn tần suất chính thức (cộng đồng báo bot-detection kích hoạt quanh ~30 request/phút/IP). Chấp nhận được cho QUEST nhánh 2 CHỈ VÌ plan đã bắt buộc timeout + fallback về search thường khi nhánh này lỗi (CLAUDE.md §4/§7) — không dùng pattern này cho bất kỳ chỗ nào coi nó là nguồn dữ liệu bắt buộc phải có.
- **Đồng bộ, KHÔNG có async client** (`AsyncDDGS` không tồn tại) — bắt buộc bọc `asyncio.to_thread(...)` khi gọi từ route `async def`, không được gọi trực tiếp (`.claude/rules/ecc/python/fastapi.md`: không gọi blocking I/O từ async route).
- **API tìm ảnh:**
  ```python
  from ddgs import DDGS

  results = DDGS().images(
      query="Labubu",
      max_results=10,
      safesearch="moderate",
  )
  # -> list[dict]: mỗi dict có title, image (URL ảnh gốc), thumbnail (URL),
  #    url (trang nguồn), height, width, source
  ```
- **Exception (`ddgs.exceptions`):** `DDGSException` (base) · `RatelimitException` (vượt giới hạn tần suất) · `TimeoutException` (hết thời gian chờ) — bắt cụ thể từng loại, không `except Exception: pass` (CLAUDE.md §7).
- **SSRF:** kết quả trả về `image`/`thumbnail` là URL của bên thứ ba (không phải do dự án host) — không bao giờ để backend tự động tải các URL này về theo yêu cầu của người dùng cuối; nếu cần hiển thị ảnh, hiển thị trực tiếp URL cho frontend load (giống ảnh ứng viên, không phải ảnh do server fetch), hoặc nếu backend cần tải để mã hoá bằng BEiT-3 thì phải giới hạn: chỉ tải chính các URL mà `DDGS().images()` vừa trả về trong cùng request, không nhận URL tuỳ ý từ client.

### 5.6 Ollama (local LLM) — sinh câu hỏi QA (P3.3)

🔧 Dự án (P3.3, `.claude/plans/p3-vqa-submission.plan.md`). Máy chủ dự án không có ngân sách gọi API trả phí (OpenAI) — chọn LLM chạy local qua Ollama để sinh câu hỏi, **cố ý khác Gemini** (model đang trả lời VQA, `configs/qa/vqa.yaml`) để tránh tự chấm chính mình (CLAUDE.md §3). Đã fetch kiểm chứng 2026-08-16 (`github.com/ollama/ollama-python` + `ollama.com/blog/structured-outputs`) — README của package KHÔNG document structured output, chỉ blog post có.

- Cài đặt: `pip install ollama` (package `ollama`, không phải REST tự gọi tay). Yêu cầu **Ollama server đang chạy local** (mặc định `http://localhost:11434`) và model đã `ollama pull` trước — script KHÔNG tự tải model (CLAUDE.md §10: lệnh tốn tài nguyên chuẩn bị sẵn cho chủ dự án tự chạy).
- Model đã chọn cho P3.3: `qwen3:8b` (5.2GB, vừa VRAM RTX 3070 8GB, chạy tuần tự không cạnh tranh với BEiT-3, CLAUDE.md §1.4).
  - ⚠️ ĐỔI từ `qwen2.5:7b-instruct` (2026-08-16, cùng ngày thử nghiệm) — chạy thật với `--limit-shots 5` cho ra một câu hỏi mà chính `reference_answer` của nó thừa nhận "không thể trả lời bằng cách xem caption" (model hỏi điều không có căn cứ rồi nhét lời từ chối vào field answer thay vì tự bỏ câu, đúng rủi ro đã cảnh báo ở dòng dưới). `ollama.com/library/qwen3` ghi model thế hệ mới hơn, mạnh hơn về instruction-following — đúng năng lực đang thiếu.
- **Structured output qua `format=<model_json_schema()>`** — cùng tinh thần `model_json_schema()` của Gemini/OpenAI (CLAUDE.md §5: không parse text tự do bằng regex), KHÔNG cần tool-calling hay JSON-mode kiểu cũ:

  ```python
  from ollama import Client
  from pydantic import BaseModel

  class Country(BaseModel):
      name: str
      capital: str
      languages: list[str]

  client = Client(host="http://localhost:11434")
  response = client.chat(
      model="qwen3:8b",
      messages=[{"role": "user", "content": "Tell me about Canada."}],
      format=Country.model_json_schema(),
  )
  country = Country.model_validate_json(response.message.content)
  ```

- Đọc kết quả qua `response.message.content` (chuỗi JSON) rồi `Model.model_validate_json(...)` — khác Gemini (`interaction.output_text`) và OpenAI (`item.parsed`), không dùng nhầm accessor của SDK khác.
- Không có refusal branch riêng như OpenAI — model local không tự chối; vẫn phải bọc `model_validate_json` trong try/except vì model có thể trả JSON không khớp schema (đặc biệt model nhỏ, ép schema qua grammar constraint không đảm bảo 100% nội dung hợp lý). **Xác nhận bằng thực nghiệm 2026-08-16** với `qwen2.5:7b-instruct`: schema (đúng cấu trúc JSON) được tuân thủ, nhưng NỘI DUNG field `reference_answer` lại là một câu từ chối thay vì một đáp án thật — ép schema không ép được ngữ nghĩa.
- Lỗi kết nối (server Ollama chưa chạy, model chưa pull) là `httpx`-based exception hoặc `ollama.ResponseError`/`ConnectionError` tuỳ nguyên nhân — bắt cụ thể, không `except Exception: pass` (CLAUDE.md §7).

---

## 6. Backend

### 6.1 FastAPI

- Docs: https://fastapi.tiangolo.com
- **`llms.txt`: ❌ không có** — đã thử cả `https://fastapi.tiangolo.com/llms.txt` và `https://fastapi.tiangolo.com/llms-full.txt`, cả hai trả về **HTTP 404**. Dùng docs chính thức.

**Cài đặt:** `pip install "fastapi[standard]"`

**Trang cần dùng:**

| Chủ đề | URL |
|---|---|
| Dependency injection | https://fastapi.tiangolo.com/tutorial/dependencies/ |
| Async / `async def` | https://fastapi.tiangolo.com/async/ |
| Response model | https://fastapi.tiangolo.com/tutorial/response-model/ |
| Upload file (cho `/search/image`) | https://fastapi.tiangolo.com/tutorial/request-files/ |
| Lifespan (nạp model lúc khởi động) | https://fastapi.tiangolo.com/advanced/events/ |
| Background tasks | https://fastapi.tiangolo.com/tutorial/background-tasks/ |
| CORS | https://fastapi.tiangolo.com/tutorial/cors/ |

**Ràng buộc của dự án** (từ `.claude/rules/ecc/python/fastapi.md`):
- Khởi tạo app trong `create_app()`
- Router **mỏng** — logic nằm ở service layer
- Schema request / update / response **tách riêng**
- Encoder và store client inject qua `Depends`, **không** khởi tạo trong route handler
- Endpoint có I/O dùng `async def` với client bất đồng bộ — **không** gọi `requests` hay client đồng bộ từ route async
- CORS theo môi trường; không kết hợp wildcard origin với credentials
- Log phải che API key, token, authorization header

### 6.2 Pydantic

- Docs: https://docs.pydantic.dev
- **`llms.txt`: ✅ https://pydantic.dev/docs/validation/latest/llms.txt** (`docs.pydantic.dev/latest/llms.txt` trả 301 về URL này)

**Cài đặt:** `pip install pydantic pydantic-settings`

| Chủ đề | URL |
|---|---|
| `BaseModel` | https://pydantic.dev/docs/validation/latest/api/pydantic/base_model/index.md |
| Validators | https://pydantic.dev/docs/validation/latest/concepts/validators/index.md |
| `pydantic-settings` | https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/index.md |

**Dùng trong dự án:** `pydantic-settings` cho `app/config.py` (nạp API key từ `.env`); `BaseModel` cho toàn bộ schema request/response; `model_json_schema()` để sinh schema cho structured output của Gemini/OpenAI (§5).

### 6.3 PyTorch

- Docs: https://docs.pytorch.org
- **`llms.txt`: ❌ không có** (đã thử `https://docs.pytorch.org/llms.txt` → HTTP 404)

**Cài đặt:** cài bản khớp CUDA của máy, xem https://pytorch.org/get-started/locally/

**Điểm cần lưu ý cho RTX 3070 8GB:** dùng `torch.no_grad()` và autocast fp16 cho toàn bộ inference; batch size cấu hình được; giải phóng cache giữa các giai đoạn pipeline.

### 6.4 HuggingFace Transformers

Chỉ dùng khi cần **encoder dự phòng** (SigLIP/CLIP) theo rủi ro R1.

- Docs: https://huggingface.co/docs/transformers
- **`llms.txt`: ✅ https://huggingface.co/docs/transformers/llms.txt**

| Chủ đề | URL |
|---|---|
| Auto classes | https://huggingface.co/docs/transformers/v5.14.0/model_doc/auto.md |
| Multimodal processors | https://huggingface.co/docs/transformers/v5.14.0/multimodal_processing.md |
| **SigLIP** (ứng viên thay BEiT-3) | https://huggingface.co/docs/transformers/v5.14.0/model_doc/siglip.md |
| **CLIP** (baseline so sánh) | https://huggingface.co/docs/transformers/v5.14.0/model_doc/clip.md |
| ColPali | https://huggingface.co/docs/transformers/v5.14.0/model_doc/colpali.md |

CLIP ở đây phục vụ **hai mục đích**: vừa là encoder dự phòng, vừa là baseline bắt buộc trong bảng so sánh §6.3 của PRD.

### 6.5 open_clip

Dùng cho `ClipEncoder` (`app/encoders/clip_baseline.py`, baseline P1.4) và `PeCoreEncoder`
(`app/encoders/pe_core.py`, candidate A2 của
`.claude/plans/encoder-swap-siglip2-pecore.plan.md`).

- Repo: https://github.com/mlfoundations/open_clip
- **`llms.txt`: chưa kiểm chứng**
- Package pin: `open-clip-torch==3.3.0` (`pyproject.toml` extras `baseline-clip`,
  `encoder-candidates`)

**Load một checkpoint tuỳ ý từ HF Hub** (không cần đăng ký kiến trúc trong open_clip's local
registry) — dùng cho `PeCoreEncoder`:

```python
model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:timm/PE-Core-bigG-14-448")
tokenizer = open_clip.get_tokenizer("hf-hub:timm/PE-Core-bigG-14-448")
```

Config (`vision_cfg`/`text_cfg`/`preprocess_cfg`, bao gồm `context_length`) đọc thẳng từ
`open_clip_config.json` trên repo HF — không cần khai lại thủ công.

⚠️ **Không pin được `revision` cho load kiểu `hf-hub:`** — verified 2026-08-14 bằng cách fetch
trực tiếp source `open-clip-torch==3.3.0`
(`src/open_clip/factory.py`, `src/open_clip/pretrained.py`): nhánh `hf-hub:` của
`create_model_and_transforms`/`create_model` gọi
`download_pretrained_from_hf(model_id, cache_dir=cache_dir)` — **không truyền `revision`**, dù
`download_pretrained_from_hf()` bản thân nó CÓ tham số `revision` (dùng được nếu tự gọi
`huggingface_hub.hf_hub_download(..., revision=...)` thủ công thay vì qua
`create_model_and_transforms`). Khác hẳn `transformers.AutoModel.from_pretrained(revision=...)`
(§6.4, dùng cho `SigLip2Encoder`). Hệ quả cho encoder nào load kiểu `hf-hub:`: không có
`revision=` để ghim, phải dựa vào `weight_files_sha256` + probe cosine để bắt drift *sau khi
đã xảy ra*, không phải chặn từ lúc load — xem `app/encoders/pe_core.py` module docstring.

**PE-Core-bigG-14-448 cụ thể** — verified 2026-08-14 qua
`https://huggingface.co/api/models/timm/PE-Core-bigG-14-448` +
`.../raw/main/open_clip_config.json`:

| Trường | Giá trị |
|---|---|
| License | `apache-2.0`, không FAIR-NC/non-commercial |
| `embed_dim` | 1280 |
| `vision_cfg.image_size` | 448 |
| `vision_cfg.timm_model_name` | `vit_pe_core_gigantic_patch14_448` (timm-backed ⇒ cần `timm>=1.0.19`, không phải kiến trúc open_clip thuần như `ViT-B-32-quickgelu`) |
| `text_cfg.context_length` | 72 |
| `text_cfg` tokenizer | `custom_text: true`, CLIP BPE chuẩn (vocab 49408) — **English-only**, không phải tokenizer đa ngữ |
| File weight ưu tiên hash | `open_clip_model.safetensors` (có sẵn trên repo, ưu tiên hơn `.bin`) |

**Dùng trong dự án:** `ClipEncoder` (baseline P1.4, ViT-B/32) và `PeCoreEncoder` (candidate A2).

---

## 7. Frontend

Chỉ liệt kê ở mức tối thiểu — **thiết kế giao diện sẽ chốt riêng với người chủ dự án** (xem §8 của [`architecture.md`](./architecture.md)).

| Thư viện | Docs | `llms.txt` |
|---|---|---|
| React | https://react.dev | chưa kiểm chứng |
| Vite | https://vite.dev | chưa kiểm chứng |
| TypeScript | https://www.typescriptlang.org/docs/ | chưa kiểm chứng |

Ràng buộc kỹ thuật (độc lập với thiết kế): virtualized rendering cho lưới hàng nghìn thumbnail; shortcut bàn phím; đổi mode không mất truy vấn đang gõ.

---

## 8. Phương án tối ưu giai đoạn 2

> **Toàn bộ mục này CHƯA dùng ở v1.** Ghi lại để không phải nghiên cứu lại khi tới lúc tối ưu.

| Lĩnh vực | Ứng viên | Khi nào cân nhắc |
|---|---|---|
| **Vector search** | [FAISS](https://github.com/facebookresearch/faiss), [Qdrant](https://qdrant.tech) | Nếu Milvus quá nặng cho một máy. FAISS in-process bỏ được hẳn một service |
| **Encoder** | [SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip), [open_clip](https://github.com/mlfoundations/open_clip), [jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2) | **Cũng là phương án rút lui khẩn cấp cho R1.** jina-clip-v2 đa ngôn ngữ — có thể hợp hơn với truy vấn tiếng Việt |
| **OCR** | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Thay Gemini OCR để **cắt chi phí R2**. Chạy cục bộ, không quota, nhưng chất lượng tiếng Việt thường kém hơn |
| ~~**Kênh ASR**~~ → 🔧 **đã kéo về v1** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [WhisperX](https://github.com/m-bain/whisperX) | Thêm một kênh truy hồi hoàn toàn mới từ lời thoại. **Rất giá trị với AIC 2026**: bản tin có lời dẫn dày đặc, nói ra nhiều thứ **không** hiện thành chữ trên màn hình (tên người, địa danh, số liệu) — bổ trợ chứ không trùng lặp với kênh OCR. **2026-08-02: không còn thuộc giai đoạn 2** — đã thành **P0.11** (`PRD.md` §2.3). API đã fetch kiểm chứng ở **§5.3** |
| **Reranking** | Cross-encoder / ColBERT-style | Khi Recall@100 cao nhưng Recall@1 thấp — nghĩa là đáp án có trong tầm với, chỉ xếp hạng sai |
| **Tăng tốc inference** | ONNX Runtime, TensorRT | Khi ingest là nút thắt |
| **Nén vector** | Product Quantization, scalar quantization | Khi index không vừa RAM |

**Thứ tự ưu tiên đề xuất:** ASR (mở ra kênh mới, giá trị cao nhất trên video tin tức) → reranking → nén vector → tăng tốc inference.

---

## 9. Bảng tổng hợp trạng thái `llms.txt`

Đã fetch kiểm chứng trực tiếp ngày **2026-07-25**:

| Thư viện | `llms.txt` | URL | Ghi chú |
|---|:---:|---|---|
| Milvus | ✅ | https://milvus.io/llms.txt | Chỉ mục docs đầy đủ theo chủ đề |
| Elasticsearch | ✅ | https://www.elastic.co/docs/llms.txt | Đúng chuẩn spec llms.txt |
| MongoDB | ✅ | https://www.mongodb.com/docs/llms.txt | Có bản riêng cho PyMongo |
| Gemini API | ✅ | https://ai.google.dev/gemini-api/docs/llms.txt | Trang con dùng hậu tố `.md.txt` |
| OpenAI | ✅ | https://developers.openai.com/api/llms.txt | Có thêm `llms-full.txt` |
| HF Transformers | ✅ | https://huggingface.co/docs/transformers/llms.txt | |
| Pydantic | ✅ | https://pydantic.dev/docs/validation/latest/llms.txt | `docs.pydantic.dev/latest/llms.txt` trả 301 về đây |
| **FastAPI** | ❌ | — | `llms.txt` và `llms-full.txt` đều **404** → dùng docs chính thức |
| **PyTorch** | ❌ | — | `docs.pytorch.org/llms.txt` **404** → dùng docs chính thức |
| **TransNetV2** | ❌ | — | Chỉ có GitHub → đọc README + source |
| **BEiT-3** | ❌ | — | Chỉ có `microsoft/unilm` → đọc README + get-started |
| ES Vietnamese plugin | ❌ | — | Chỉ có GitHub README |
| **faster-whisper** | ❌ | — | Chỉ có GitHub → đọc README + `faster_whisper/transcribe.py`. Fetch **2026-08-02**, xem §5.3 |

**Nên kiểm tra lại bảng này định kỳ** — `llms.txt` đang được các nhà cung cấp bổ sung dần, và URL có thể đổi (Pydantic đã đổi một lần, để lại redirect 301).

---

## 10. Ghi chú thu thập

Những điều phát hiện được nhờ đọc `llms.txt` thay vì viết theo trí nhớ — đáng lưu ý vì chúng khác với hầu hết hướng dẫn đang lưu hành trên mạng:

1. **Gemini đã chuyển sang Interactions API.** `client.interactions.create(...)` với `input=[...]` và `response_format={...}`, không còn là `generate_content(...)` với `generation_config`. Code chép từ blog cũ sẽ không chạy.
2. **OpenAI dùng Responses API.** `client.responses.parse(model=..., input=[...], text_format=PydanticModel)`, khác hẳn `chat.completions.create`. Kết quả đọc qua `item.parsed`, và **phải xử lý nhánh `item.type == "refusal"`**.
3. **Milvus khuyến nghị rõ ràng dùng `MilvusClient`**, không dùng ORM API cũ — nhiều tutorial vẫn đang dạy API cũ.
4. **Số token ảnh của Gemini (258 token mỗi tile 768×768)** là con số cụ thể quyết định tính khả thi của nhánh OCR. Không có nó thì không ước lượng được rủi ro R2.
5. **Pydantic đã đổi domain docs** — `docs.pydantic.dev` giờ redirect 301 sang `pydantic.dev/docs/validation/...`.

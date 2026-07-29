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

**Bốn dạng nhiệm vụ:** Textual KIS, Video KIS (clip ≤ 20 giây, không được ghi hình), QA, TRAKE.

**Cách chấm:** điểm tối đa mỗi truy vấn bằng nhau; điểm giảm theo thời gian trả lời và số lần nộp sai. Trang hướng dẫn ghi *"quy tắc tính điểm sẽ được thông báo riêng"* — **định dạng file nộp chính xác chưa công bố**.

---

## 2. Dataset — AIC 2025 (dữ liệu công khai)

> Số liệu dưới đây **đo trực tiếp trên đĩa ngày 2026-07-29**, không phải ước lượng.

- Nguồn: Kaggle `aresusayhi/ai-challenge-2025`, version **2** — tải bằng `download.py` (`kagglehub`)
- Gốc dữ liệu trên máy: `data/datasets/aresusayhi/ai-challenge-2025/versions/2/`
- Dung lượng: **123 GB**
- Nội dung: video **tin tức tiếng Việt** (HTV — *60 Giây Official*, …), có tiếng nói và **chữ trên màn hình**

### 2.1 Thống kê

| Thuộc tính | Giá trị đo được |
|---|---|
| Số `video_id` duy nhất | **873** (batch `L21` … `L30`) |
| Số file `.mp4` | 961 — chênh 88 file vì `Videos_L25_a1` và `Videos_L25_b` là **bản sao trùng id** của `Videos_L25_a` |
| Tổng thời lượng | **470.428 giây ≈ 130,7 giờ** (tổng trường `length` của `media-info`) |
| Thời lượng trung bình | ~9,0 phút/video |
| Keyframe **BTC trích sẵn** | **177.321** ảnh JPG **1280×720** (~203 ảnh/video, ~1 ảnh mỗi 2,65 giây) |

**Bẫy khi đếm video:** đếm file `.mp4` sẽ ra 961 và tính trùng 88 video. Luôn khử trùng theo `video_id` — `DatasetAdapter` phải đảm bảo điều này, và mọi báo cáo metric lấy mẫu số từ số `video_id` duy nhất.

### 2.2 Bố cục thư mục

```
versions/2/
├── Videos/Videos/Videos_<batch>/video/<video_id>.mp4    # 961 file, 873 id duy nhất
├── Keyframes_<batch>/keyframes/<video_id>/NNN.jpg       # 177.321 ảnh 1280×720
└── features/
    ├── map-keyframes/<video_id>.csv        # n,pts_time,fps,frame_idx
    ├── clip-features-32/<video_id>.npy     # (n_keyframes, 512) float16 — CLIP ViT-B/32
    ├── media-info/<video_id>.json          # metadata YouTube
    └── objects/<video_id>/NNN.json         # object detection từng keyframe
```

`<batch>` gồm: `L21` `L22` `L23` `L24` `L25` `L26_a`…`L26_e` `L27` `L28` `L29` `L30`. Tên thư mục **không nhất quán** — thư mục video dùng tiền tố `Videos_` trừ `Video_L30_a` (số ít), thư mục keyframe dùng `Keyframes_`. Adapter phải **glob** chứ không dựng đường dẫn bằng chuỗi định dạng.

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

**Với RTX 3070 8GB: chọn `beit3_base_itc`**, không phải `large`.

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

Keyframe của AIC 2025 là **1280×720** ⇒ cắt thành **2 tile** 768×768 ⇒ **516 token/ảnh**. Hạ về ≤ 384×384 trước khi gửi thì còn **258 token/ảnh** — đây là đòn bẩy chi phí rẻ nhất, nhưng phải kiểm chứng chữ nhỏ trên màn hình có còn đọc được không.

Quy đổi trên số keyframe **BTC trích sẵn** (mốc tham chiếu — số keyframe do TransNetV2 của dự án sinh ra ⚠️ **[Cần đo]**, xem PRD §8/R2):

| Phạm vi | Keyframe | @516 token | @258 token |
|---|---|---|---|
| Pilot `L21`–`L24` (128 video) | 26.003 | ~13,4 triệu | ~6,7 triệu |
| Toàn bộ (873 video) | 177.321 | ~91,5 triệu | ~45,8 triệu |

Đây chính là cơ sở của rủi ro R2 trong PRD và lý do phải lọc frame có chữ + dedupe **trước** khi gọi. Khác YouCook2 ở một điểm quan trọng: video tin tức có **tỉ lệ frame chứa chữ rất cao**, nên bước lọc [1] sẽ cắt được ít hơn kỳ vọng — bước dedupe [2] mới là chỗ tiết kiệm chính (bản tin giữ nguyên khung đồ hoạ suốt nhiều giây liền).

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
| **Kênh ASR** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [WhisperX](https://github.com/m-bain/whisperX) | Thêm một kênh truy hồi hoàn toàn mới từ lời thoại. **Rất giá trị với AIC 2025**: bản tin có lời dẫn dày đặc, nói ra nhiều thứ **không** hiện thành chữ trên màn hình (tên người, địa danh, số liệu) — bổ trợ chứ không trùng lặp với kênh OCR |
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

**Nên kiểm tra lại bảng này định kỳ** — `llms.txt` đang được các nhà cung cấp bổ sung dần, và URL có thể đổi (Pydantic đã đổi một lần, để lại redirect 301).

---

## 10. Ghi chú thu thập

Những điều phát hiện được nhờ đọc `llms.txt` thay vì viết theo trí nhớ — đáng lưu ý vì chúng khác với hầu hết hướng dẫn đang lưu hành trên mạng:

1. **Gemini đã chuyển sang Interactions API.** `client.interactions.create(...)` với `input=[...]` và `response_format={...}`, không còn là `generate_content(...)` với `generation_config`. Code chép từ blog cũ sẽ không chạy.
2. **OpenAI dùng Responses API.** `client.responses.parse(model=..., input=[...], text_format=PydanticModel)`, khác hẳn `chat.completions.create`. Kết quả đọc qua `item.parsed`, và **phải xử lý nhánh `item.type == "refusal"`**.
3. **Milvus khuyến nghị rõ ràng dùng `MilvusClient`**, không dùng ORM API cũ — nhiều tutorial vẫn đang dạy API cũ.
4. **Số token ảnh của Gemini (258 token mỗi tile 768×768)** là con số cụ thể quyết định tính khả thi của nhánh OCR. Không có nó thì không ước lượng được rủi ro R2.
5. **Pydantic đã đổi domain docs** — `docs.pydantic.dev` giờ redirect 301 sang `pydantic.dev/docs/validation/...`.

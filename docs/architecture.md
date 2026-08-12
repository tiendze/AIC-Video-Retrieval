# Kiến trúc hệ thống — AIC Video Retrieval

> **Phiên bản:** 0.2 · **Ngày:** 2026-07-29 (bản 0.1: 2026-07-25)
> **Thay đổi 0.1 → 0.2:** dataset chuyển từ YouCook2 sang **AIC 2025 công khai**. Ảnh hưởng: §1 · §3.2 · §4.1 · §4.3 · §5.2 (**viết lại**) · §6 · §10.
> **Baseline:** [arXiv:2512.13169](https://arxiv.org/abs/2512.13169)
> **Tài liệu liên quan:** [`PRD.md`](./PRD.md) · [`references.md`](./references.md)

Ký hiệu: 📄 **[Paper]** = trích từ baseline · 🔧 **[Giả định]** = quyết định của dự án.

---

## 1. Tư tưởng thiết kế

Hệ thống được tổ chức quanh **một ý tưởng cốt lõi**: đưa ảnh và văn bản về **cùng một không gian vector**. Khi đã ở chung không gian, mọi thứ khác trở thành hệ quả:

- Tìm ảnh bằng câu chữ = mã hoá câu chữ rồi tìm láng giềng gần nhất
- Tìm ảnh bằng ảnh = mã hoá ảnh rồi tìm láng giềng gần nhất
- Tìm chuỗi sự kiện theo thời gian = quy hoạch động trên ma trận tương đồng trong chính không gian đó

Đây là lý do sơ đồ §3.3 (luồng multimodal) là sơ đồ trung tâm của tài liệu này: nó cho thấy **encoder dùng lúc index và encoder dùng lúc query phải là một** — nếu hai bên lệch nhau, toàn hệ thống vô nghĩa.

Ba nguyên tắc kiến trúc kéo theo:

| Nguyên tắc | Vì sao | Hiện thực hoá |
|---|---|---|
| **Tách encoder khỏi pipeline** | BEiT-3 rất khó cài (rủi ro R1 trong PRD). Phải đổi được backbone mà không viết lại pipeline | `MultimodalEncoder` Protocol (§5.1) |
| **Tách dataset khỏi pipeline** | Hôm nay pilot `L21`–`L24`, mai là đủ 873 video, mốt là batch mới BTC phát hành | `DatasetAdapter` Protocol (§5.2) |
| **Tách offline khỏi online** | Ingest chạy hàng chục giờ; query phải trả lời trong vài trăm ms. Hai chế độ vận hành hoàn toàn khác nhau | Hai luồng riêng (§3.2 và §3.3), giao nhau duy nhất tại tầng lưu trữ |

---

## 2. Tổng quan hệ thống

### 2.1 Sơ đồ container

```mermaid
flowchart LR
    subgraph client["Tầng người dùng"]
        UI["Web UI<br/>React + Vite + TypeScript"]
    end

    subgraph backend["Tầng ứng dụng"]
        API["FastAPI Gateway<br/>routers mỏng"]
        SVC["Service Layer<br/>search / trake / qa / ingest"]
        ENC["Encoder Service<br/>BEiT-3 trên GPU"]
    end

    subgraph storage["Tầng lưu trữ"]
        MV[("Milvus<br/>vector embedding")]
        ES[("Elasticsearch<br/>OCR text + vi_analyzer")]
        MG[("MongoDB<br/>metadata + đường dẫn")]
        FS[("Ổ đĩa<br/>keyframe JPEG")]
    end

    subgraph external["Dịch vụ ngoài"]
        GEM["Gemini API<br/>OCR + VQA"]
        OAI["OpenAI API<br/>viết lại truy vấn"]
        IMG["Tìm ảnh web<br/>QUEST nhánh 2"]
    end

    UI -->|"HTTP / JSON"| API
    API --> SVC
    SVC --> ENC
    SVC --> MV
    SVC --> ES
    SVC --> MG
    SVC -.->|"phụ thuộc mạng"| GEM
    SVC -.->|"phụ thuộc mạng"| OAI
    SVC -.->|"phụ thuộc mạng"| IMG
    UI -->|"tải thumbnail"| FS

    classDef ext fill:#fff4e6,stroke:#d97706,stroke-width:2px,color:#7c2d12
    classDef store fill:#eef6ff,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    class GEM,OAI,IMG ext
    class MV,ES,MG,FS store
```

**Đường nét đứt = phụ thuộc mạng.** Đây là ranh giới suy giảm chức năng: khi mất mạng, semantic search và DANTE vẫn chạy đầy đủ; chỉ QUEST, OCR và VQA ngừng hoạt động (yêu cầu T5 trong PRD).

### 2.2 Bảng thành phần

| Thành phần | Công nghệ | Nguồn | Vai trò |
|---|---|---|---|
| Shot segmentation | TransNetV2 | 📄 Paper | Cắt video thành shot |
| Visual/Text encoder | BEiT-3 (`beit3_base_itc`) | 📄 Paper | Đưa ảnh và chữ về chung không gian vector |
| OCR | Gemini API | 📄 Paper | Đọc chữ trên keyframe |
| Vector DB | Milvus | 📄 Paper | ANN search theo cosine |
| Text search | Elasticsearch + `vi_analyzer` | 📄 Paper | BM25 trên OCR text tiếng Việt |
| Metadata | MongoDB | 📄 Paper | Đường dẫn ảnh, thông tin video |
| Viết lại truy vấn | GPT-4o-mini / Gemini | 📄 Paper | QUEST nhánh 1 |
| VQA answering | Gemini API | 🔧 Giả định | Sinh câu trả lời có grounding |
| Backend | FastAPI + Pydantic | 🔧 Giả định | Paper không nêu framework |
| Frontend | React + Vite + TS | 🔧 Giả định | Paper không nêu framework |

⚠️ **Nguồn dữ liệu hiện đang chạy thật (2026-08-07):** hai dòng Shot segmentation / Visual-Text encoder ở trên mô tả kiến trúc baseline. Dữ liệu thật trong Milvus/MongoDB lúc này đến từ `data/embed_out/` — BEiT-3 encode **trên Kaggle** (không phải TransNetV2 + encoder chạy cục bộ như sơ đồ §3.2 vẽ), keyframe input do thành viên nhóm cung cấp sẵn. Chi tiết: `.claude/plans/kaggle-offline-embed.plan.md`, import bằng `scripts/import_embed_out.py`.

---

## 3. Luồng dữ liệu

### 3.1 Hai chế độ vận hành

| | **Offline — Ingest** | **Online — Query** |
|---|---|---|
| Tần suất | Một lần cho mỗi video | Mỗi request người dùng |
| Ngân sách thời gian | Hàng giờ | Hàng trăm mili-giây |
| Nút thắt | Giải mã video, gọi OCR | Gọi encoder, ANN search |
| Với tầng lưu trữ | **Ghi** | **Đọc** |

Chúng giao nhau tại đúng một điểm: **tầng lưu trữ**, và bắt buộc dùng **cùng một encoder**.

### 3.2 Luồng Ingest (offline)

```mermaid
flowchart TD
    A["Video nguồn<br/>AIC 2026 batch 1 · file mp4 cục bộ"] --> B["DatasetAdapter<br/>liệt kê video + metadata<br/>khử trùng video_id"]
    B --> CK{"Đã xử lý xong?<br/>kiểm tra checkpoint"}
    CK -->|"rồi"| SKIP["Bỏ qua"]
    CK -->|"chưa"| C["Giải mã video<br/>ffmpeg tới 48x27 uint8"]

    C --> D["TransNetV2<br/>phát hiện chuyển cảnh"]
    D --> E["Danh sách shot<br/>các khoảng frame a tới b"]
    E --> F["Trích 4 keyframe mỗi shot<br/>đầu · giữa · giữa · cuối"]
    F --> G["Gán global_index<br/>và tính s_v, e_v"]

    G --> H1["Nhánh THỊ GIÁC"]
    G --> H2["Nhánh VĂN BẢN"]
    G --> H3["Nhánh METADATA"]
    A --> H4["Nhánh ASR (P0.11a)<br/>độc lập checkpoint VISION/OCR"]

    H1 --> I1["BEiT-3 image encoder"]
    I1 --> J1["L2 normalize"]
    J1 --> K1[("Milvus<br/>keyframe_id + vector")]

    H2 --> P["Lọc frame có chữ<br/>+ dedupe perceptual hash"]
    P --> I2["Gemini OCR"]
    I2 --> K2[("Elasticsearch<br/>keyframe_id + ocr_text")]

    H3 --> I3["Lưu JPEG + sinh thumbnail"]
    I3 --> K3[("MongoDB<br/>đường dẫn + metadata")]

    H4 --> P4["faster-whisper transcribe<br/>(.mp4 trực tiếp qua PyAV)"]
    P4 --> I4["Lọc ảo giác<br/>no_speech_prob / avg_logprob / độ dài text"]
    I4 --> K4[("Elasticsearch<br/>segment_id + text + frame window")]

    K4 -.->|"scripts/embed_transcripts.py<br/>chạy riêng, sau — KHÔNG transcribe lại"| P5["bge-m3 encode<br/>(đọc text lại từ ES qua scroll API)"]
    P5 --> J5["L2 normalize"]
    J5 --> K5[("Milvus<br/>segment_id + vector dense (P0.11b)")]

    K1 --> Z["Ghi checkpoint theo video_id"]
    K2 --> Z
    K3 --> Z
    Z --> B
    K4 --> Z4["Ghi asr_runs.asr_status (MongoDB)"]

    ERR["Danh sách video lỗi"]
    C -.->|"lỗi giải mã"| ERR
    I2 -.->|"lỗi API"| ERR

    classDef store fill:#eef6ff,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef risk fill:#fff4e6,stroke:#d97706,stroke-width:2px,color:#7c2d12
    class K1,K2,K3,K4,K5 store
    class P,ERR risk
```

**Các điểm cần chú ý trong sơ đồ này:**

1. **Node `Gán global_index và tính s_v, e_v`** — 📄 Paper lưu ở MongoDB `{video_id, s_v, e_v}`. Đây là **điều kiện tiên quyết của DANTE**: thuật toán quy hoạch động chạy trên khoảng index `[s_v, e_v]` của từng video. Bỏ bước này thì DANTE không thể triển khai.

2. **Node `Lọc frame có chữ + dedupe`** 🔧 — paper gọi Gemini OCR trên **mọi** keyframe; điều này bất khả thi về chi phí và quota (rủi ro R2). Keyframe 1280×720 tốn **516 token/ảnh**, và chỉ riêng 177k keyframe BTC trích sẵn đã là ~91,5 triệu token. Bước lọc này là bổ sung của dự án. ⚠️ Với video tin tức, **`dedupe` mới là bước tiết kiệm chính** chứ không phải `lọc frame có chữ` — bản tin có chữ ở gần như mọi frame, nhưng giữ nguyên khung đồ hoạ suốt nhiều giây liền.

5. **Node `DatasetAdapter`** — ngoài việc liệt kê video, nó phải **khử trùng `video_id`**: dataset có 961 file `.mp4` nhưng chỉ 873 id duy nhất (`Videos_L25_a1` và `Videos_L25_b` lặp lại id của `Videos_L25_a`). Bỏ bước này thì 88 video bị ingest hai lần, và mẫu số của mọi báo cáo metric bị thổi phồng.

3. **Vòng lặp checkpoint** — `Z → B` khép kín để pipeline resume được (yêu cầu T1).

4. **Ba nhánh song song** — chúng độc lập, có thể chạy đồng thời hoặc tách thành ba lần chạy riêng. Trên máy 8GB VRAM, chạy tuần tự từng nhánh thường an toàn hơn.

8. **Nguồn dữ liệu của lượt ingest hiện tại (2026-08-07)** 🔧 — sơ đồ trên mô tả pipeline baseline (`C → D → E → F` chạy TransNetV2 cục bộ). Trên thực tế, 873 video / 221.903 keyframe hiện có trong Milvus/MongoDB được import từ `data/embed_out/`: vector BEiT-3 encode **trên Kaggle**, không chạy lại `D`/`E`/`F`/`I1` cục bộ. `global_index`, `s_v`/`e_v` và `shot_id` (node `G`) vẫn được tính đúng như sơ đồ nhưng lúc **import**, bằng `pipelines/import_embed_out.py` — `shot_id` dùng shot boundary thật từ `data/shots.json` thay vì gọi lại TransNetV2. Quyết định và chi tiết đầy đủ: `.claude/plans/kaggle-offline-embed.plan.md`.

6. **Nhánh ASR (`H4`) tách khỏi node `A`, không phải `G`** 🔧 — bổ sung 2026-08-02, ngoài đường găng P0 (`.claude/plans/p0-foundation.plan.md` §P0.11). Nó chỉ cần `.mp4` (đọc `fps`/`frame_count` bằng cv2, không đọc Mongo), nên chạy độc lập với checkpoint VISION/OCR — video ở trạng thái `embedded` hay `complete` không ảnh hưởng gì tới việc ASR đã chạy hay chưa. Trạng thái của nó ghi ở collection **riêng** `asr_runs`, không phải field trên `VideoDocument` (`upsert_video()` là `replace_one` toàn phần — thêm field ASR vào đó sẽ bị ingest ghi đè im lặng). `start_frame_idx`/`end_frame_idx` trong `Elasticsearch` là **cửa sổ truy hồi** suy từ `start_time`/`end_time` qua floor/ceil, không phải `frame_id` nộp BTC.

7. **`K4 -.-> P5` là đường nét đứt có chủ đích** 🔧 — bổ sung P0.11b, 2026-08-02. Khác với ba nhánh còn lại (chạy trong cùng một lượt `ingest_video()`), bước `bge-m3 encode` chạy ở **một lời gọi CLI riêng** (`scripts/embed_transcripts.py`), **bất kỳ lúc nào sau khi** P0.11a đã index xong — không đồng bộ với transcribe. Lý do: transcribe tốn và không đảo ngược (~10h GPU/873 video); nếu sau này đo Hallucination Rate/WER (P0.11 Bước 1) ra ngưỡng lọc sai phải transcribe lại, thì embedding — vốn rẻ và **đọc lại `text` từ ES qua scroll API, không đụng tới audio gốc** — cũng phải làm lại theo, nhưng không kéo theo việc phải transcribe lại lần hai. Chưa có checkpoint riêng cho bước embed — chạy lại `embed_transcripts.py` sẽ re-embed toàn bộ (an toàn vì `upsert` theo `segment_id`, chỉ tốn thêm thời gian, không tạo bản ghi trùng).

### 3.3 ⭐ Luồng truy vấn Multimodal (online)

Đây là sơ đồ trung tâm. Nó thể hiện đúng luồng: **Upload (Image + Text) → Trích đặc trưng (Vision & Text) → Multimodal Embedding → Vector DB → Retrieval → VLM/LLM → Kết quả**.

```mermaid
flowchart TD
    subgraph input["1 · ĐẦU VÀO NGƯỜI DÙNG"]
        Q1["Truy vấn văn bản<br/>mô tả sự kiện hoặc câu hỏi"]
        Q2["Ảnh upload<br/>ảnh mẫu tuỳ chọn"]
    end

    subgraph feat["2 · TRÍCH XUẤT ĐẶC TRƯNG"]
        TE["Text Encoder<br/>nhánh ngôn ngữ BEiT-3"]
        VE["Vision Encoder<br/>nhánh thị giác BEiT-3"]
    end

    subgraph emb["3 · KHÔNG GIAN EMBEDDING ĐA PHƯƠNG THỨC"]
        SHARED{{"Vector d chiều đã L2-normalize<br/>ẢNH và CHỮ nằm chung một không gian"}}
    end

    subgraph store["4 · TẦNG LƯU TRỮ"]
        MV[("Milvus<br/>ANN · COSINE · HNSW")]
        ES[("Elasticsearch<br/>BM25 trên OCR")]
        MG[("MongoDB<br/>metadata")]
    end

    subgraph ret["5 · TRUY HỒI & XẾP HẠNG"]
        ANN["Top-K láng giềng gần nhất"]
        BM["Top-K khớp từ khoá"]
        FUSE["Hợp nhất RRF<br/>+ lọc metadata"]
        HYD["Bổ sung thông tin<br/>đường dẫn · timestamp · video_id"]
    end

    subgraph gen["6 · TẦNG SINH CÂU TRẢ LỜI"]
        VLM["Gemini VLM<br/>keyframe + câu hỏi"]
        GUARD{"Bằng chứng<br/>có đủ mạnh?"}
    end

    subgraph out["7 · KẾT QUẢ"]
        R1["Lưới keyframe đã xếp hạng"]
        R2["Câu trả lời + keyframe bằng chứng"]
        R3["Không đủ căn cứ"]
    end

    INGEST["Keyframe lúc ingest"] -.->|"CÙNG MỘT ENCODER"| VE

    Q1 --> TE
    Q2 --> VE
    TE --> SHARED
    VE --> SHARED

    SHARED --> MV
    Q1 --> ES

    MV --> ANN
    ES --> BM
    ANN --> FUSE
    BM --> FUSE
    FUSE --> HYD
    MG --> HYD

    HYD --> R1
    HYD --> VLM
    VLM --> GUARD
    GUARD -->|"đủ"| R2
    GUARD -->|"không đủ"| R3

    classDef hl fill:#fef3c7,stroke:#d97706,stroke-width:3px,color:#78350f
    classDef store2 fill:#eef6ff,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    class SHARED hl
    class MV,ES,MG store2
```

**Ba điều sơ đồ này muốn nói:**

1. **Khối 3 là trái tim hệ thống.** Text encoder và vision encoder đổ vào *cùng một* không gian vector. Đây là lý do một truy vấn chữ có thể tìm ra ảnh, và một ảnh upload có thể tìm ra ảnh khác — không cần hai index riêng.

2. **Mũi tên nét đứt `Keyframe lúc ingest → Vision Encoder`** là ràng buộc quan trọng nhất và dễ vi phạm nhất: vector trong Milvus được sinh lúc ingest, vector truy vấn được sinh lúc query. Nếu hai bên dùng model hoặc tiền xử lý khác nhau, kết quả sẽ sai một cách âm thầm — không crash, chỉ là điểm số tệ không rõ lý do. Vì vậy §7.5 của PRD quy định: **đổi model embedding phải tạo collection Milvus mới**.

3. **Cổng `GUARD`** đứng giữa VLM và kết quả. Hệ thống được phép nói "không đủ căn cứ". Không có cổng này thì `Hallucination Rate` (metric §6.5 của PRD) sẽ không kiểm soát được.

### 3.4 QUEST — xử lý truy vấn Out-of-Knowledge

📄 **[Paper]** Kích hoạt khi truy vấn chứa thực thể mà model embedding chưa từng thấy (ví dụ trong paper: *"Labubu"*, *"Babythree"*).

```mermaid
flowchart TD
    Q0["Truy vấn gốc q0<br/>chứa thực thể lạ"] --> DEC{"Người vận hành<br/>chọn nhánh"}

    DEC -->|"Nhánh 1"| B1["LLM viết lại truy vấn<br/>GPT-4o-mini hoặc Gemini"]
    B1 --> QR["Truy vấn đã viết lại qr<br/>mô tả trực quan, chi tiết hơn"]
    QR --> SHOW1["Hiển thị cả q0 và qr<br/>người dùng chọn"]
    SHOW1 --> TE["BEiT-3 Text Encoder"]

    DEC -->|"Nhánh 2"| B2["Tìm ảnh trên web"]
    B2 --> CAND["Danh sách ảnh ứng viên"]
    CAND --> PICK["Người dùng chọn ảnh mẫu I*"]
    PICK --> VE["BEiT-3 Vision Encoder"]

    TE --> SEARCH["ANN search trên Milvus"]
    VE --> SEARCH
    SEARCH --> RES["Lưới keyframe kết quả"]

    B1 -.->|"timeout hoặc lỗi API"| FB["Fallback:<br/>search thường bằng q0"]
    B2 -.->|"timeout hoặc lỗi API"| FB
    FB --> RES

    classDef human fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#064e3b
    class SHOW1,PICK,DEC human
```

**Node màu xanh là điểm con người can thiệp** — đây là chủ ý thiết kế, không phải thiếu tự động hoá. Trong cuộc thi, người vận hành biết ngữ cảnh mà hệ thống không biết; ép tự động chọn ảnh mẫu đầu tiên sẽ mất nhiều hơn được.

### 3.5 DANTE — căn chỉnh thời gian cho TRAKE

📄 **[Paper]** Quy hoạch động tìm chuỗi N sự kiện đúng thứ tự trong một video.

```mermaid
flowchart TD
    IN["N mô tả sự kiện<br/>theo thứ tự thời gian"] --> ENC["BEiT-3 Text Encoder"]
    ENC --> U["N vector truy vấn<br/>u1 tới uN"]

    U --> S["Ma trận tương đồng S<br/>S[i,t] = cosine giữa ui và E[t]"]
    MG[("MongoDB<br/>s_v và e_v của mỗi video")] --> LOOP
    S --> LOOP["Duyệt từng video v"]

    LOOP --> DP["Quy hoạch động trên khoảng s_v tới e_v<br/>DP[i,t] = S[i,t] + running_max − lambda*t"]
    DP --> SC["Điểm của video<br/>DANTE[v] = max DP[N,t]"]
    SC --> BT["Backtrack<br/>lấy đúng N keyframe"]
    BT --> TOP["Top-k video<br/>kèm chuỗi keyframe đúng thứ tự"]

    LOOP -->|"video tiếp theo"| LOOP
```

**Công thức nguyên văn từ paper:**

```
(1)  S[i,t]      = cosine_similarity(u_i, E[t])
(2)  DP[i,t]     = S[i,t] + max_{τ ∈ [s_v, t−1]} ( DP[i−1,τ] − λ(t − τ) )
(3)  running_max = max( running_max, DP[i−1,t−1] + λ(t−1) )
(4)  DP[i,t]     = S[i,t] + running_max − λt
(5)  DANTE[v]    = max_{t ∈ [s_v, e_v]} DP[N,t]
```

**Cách đọc:**

| Thành phần | Ý nghĩa |
|---|---|
| `τ ∈ [s_v, t−1]` | Sự kiện `i−1` phải xảy ra **trước** sự kiện `i`. Đây là ràng buộc thứ tự — phần cốt lõi của TRAKE |
| `−λ(t − τ)` | Phạt khoảng cách thời gian. Càng xa càng bị trừ điểm, nên chuỗi kết quả có xu hướng gom lại gần nhau |
| Eq (3)(4) | Mẹo *running max*: thay vì quét lại toàn bộ `τ` cho mỗi `t` (chi phí `O(T²)` mỗi hàng), duy trì một biến max chạy dần. Tách `λ(t−τ)` thành `λt − λτ` cho phép đưa `λτ` vào trong max và `λt` ra ngoài |
| Độ phức tạp | `O(N(e_v − s_v + 1))` mỗi video, tổng `O(NT)` |

**Giá trị `λ` theo paper:** `0.001` khi khoảng cách index giữa các sự kiện là 3–15; `0.01` khi khoảng cách 1–3. Paper **không có bảng ablation** — xem §6.4 của PRD cho kế hoạch sweep bắt buộc.

### 3.6 Trình tự end-to-end của một truy vấn VQA

```mermaid
sequenceDiagram
    autonumber
    actor U as Người vận hành
    participant UI as Web UI
    participant API as FastAPI
    participant E as Encoder GPU
    participant M as Milvus
    participant DB as MongoDB
    participant G as Gemini

    U->>UI: Nhập câu hỏi và ảnh tuỳ chọn
    UI->>API: POST /api/qa
    API->>API: Validate bằng Pydantic

    alt Cache hit
        API-->>UI: Trả kết quả đã cache
    else Cache miss
        API->>E: encode_text của câu hỏi
        E-->>API: vector đã L2-normalize
        API->>M: search với vector, top_k, filter
        M-->>API: keyframe_id kèm điểm số
        API->>DB: lấy metadata theo keyframe_id
        DB-->>API: đường dẫn ảnh, timestamp, video_id

        API->>G: interactions.create với ảnh, câu hỏi, JSON schema
        Note over API,G: timeout 10s · retry backoff
        alt Gemini trả lời được
            G-->>API: answer, confidence, evidence_ids
            API->>API: Kiểm tra evidence_ids nằm trong tập keyframe đã truy hồi
            API-->>UI: Câu trả lời kèm keyframe bằng chứng
        else Gemini lỗi hoặc timeout
            API-->>UI: Chỉ trả lưới keyframe, báo VQA tạm không khả dụng
        end
    end

    UI-->>U: Hiển thị kết quả
    U->>UI: Chốt đáp án
    UI->>API: POST /api/submissions/export
```

**Bước kiểm tra `evidence_ids nằm trong tập keyframe đã truy hồi`** là chốt chặn chống bịa: nếu model viện dẫn keyframe không có trong ngữ cảnh được cấp, câu trả lời bị đánh dấu không đáng tin. Không có bước này thì `Grounded Accuracy` không đo được.

---

## 4. Thiết kế tầng lưu trữ

📄 **[Paper]** phân chia ba store như dưới. Cột "Nguồn" đánh dấu 🔧 là các trường dự án bổ sung vì paper không nêu nhưng thiếu thì không triển khai được.

### 4.1 Milvus — vector

| Trường | Kiểu | Nguồn | Ghi chú |
|---|---|---|---|
| `keyframe_id` | `VARCHAR(64)`, primary key | 📄 | Định danh dạng `{video_id}_{global_index}` |
| `embedding` | `FLOAT_VECTOR(d)` | 📄 | Đã L2-normalize; `d` = 768 với `beit3_base_itc` (xác nhận lại lúc load checkpoint — Q4 trong PRD) |
| `video_id` | `VARCHAR(32)` | 🔧 | Để lọc **trong** engine, không lọc sau khi lấy về |
| `global_index` | `INT64` | 🔧 | **Bắt buộc cho DANTE** — chỉ số toàn cục của keyframe |
| `frame_idx` | `INT64` | 🔧 | **Bắt buộc để nộp bài** — chỉ số frame tuyệt đối trong video. Đây đúng là `<frame_id>` mà BTC chấm (PRD §1.1). Mang theo trong kết quả search để bộ xuất bài **không phải** tính lại từ `timestamp × fps` |
| `timestamp` | `FLOAT` | 🔧 | Giây tính từ đầu video — dùng cho hiển thị và cho quy tắc hit của `eval/`, **không** dùng để nộp bài |
| `author` | `VARCHAR(128)` | 🔧 | Tên kênh, từ `media-info.author` — lọc theo nhóm nội dung |
| `batch` | `VARCHAR(8)` | 🔧 | `L21` … `L30` — lọc theo lô phát hành, cũng là đơn vị chia shard khi ingest |

> Hai trường `author` / `batch` **thay cho `recipe_type`** của bản 0.1. Dataset AIC không có trường phân loại nội dung nào tương đương (PRD §6.1.4).

**Index:** HNSW, `metric_type = "COSINE"`, `M = 16`, `efConstruction = 200` (khoảng khuyến nghị của Milvus: `M ∈ [5,100]`, `efConstruction ∈ [50,500]`). Search dùng `params = {"ef": 64}`, chỉnh theo cân bằng recall/latency.

> ⚠️ Vì embedding đã L2-normalize, `COSINE` và `IP` cho cùng thứ hạng. Vẫn khai báo `COSINE` cho rõ ràng ngữ nghĩa.

**Collection thứ hai — `transcripts_{model_id}` (P0.11b, 🔧 dự án 2026-08-02).** Kênh ASR dense, tách hẳn khỏi `keyframes_{model_id}` — khác cả schema lẫn encoder (`bge-m3`, text-to-text, không phải `beit3_base_itc`). Không vi phạm invariant "một encoder duy nhất" (CLAUDE.md §3): invariant đó nói ingest/query **trong cùng một kênh** phải cùng model, không cấm kênh khác dùng encoder riêng.

| Trường | Kiểu | Ghi chú |
|---|---|---|
| `segment_id` | `VARCHAR(64)`, primary key | Dạng `{video_id}_seg{index}`, cùng quy ước với `AsrSegmentDocument` ở Elasticsearch |
| `embedding` | `FLOAT_VECTOR(1024)` | Đã L2-normalize; `d = 1024` với `bge-m3` |
| `video_id` | `VARCHAR(32)` | Lọc trong engine |
| `start_frame_idx` / `end_frame_idx` | `INT64` | **Cửa sổ truy hồi**, suy từ `start_time`/`end_time` qua floor/ceil — không phải `frame_id` nộp BTC (frame_id luôn lấy từ keyframe) |
| `start_time` / `end_time` | `FLOAT` | Giây, để hiển thị |

### 4.2 Elasticsearch — OCR text

| Trường | Kiểu | Nguồn |
|---|---|---|
| `keyframe_id` | `keyword` | 📄 |
| `ocr_text` | `text`, analyzer `vi_analyzer` | 📄 |
| `video_id` | `keyword` | 🔧 |
| `timestamp` | `float` | 🔧 |

Mapping dùng plugin `elasticsearch-analysis-vietnamese`, kèm `asciifolding` với `preserve_original: true` để khớp cả khi người vận hành gõ không dấu — thao tác rất phổ biến khi thi gấp. Cú pháp chi tiết xem [`references.md`](./references.md).

### 4.3 MongoDB — metadata

**Collection `keyframes`:**

| Trường | Nguồn |
|---|---|
| `keyframe_id` (unique index) | 📄 |
| `image_path` | 📄 |
| `video_id`, `shot_id`, `frame_idx`, `timestamp`, `global_index`, `thumbnail_path` | 🔧 |

**Collection `videos`:**

| Trường | Nguồn | Ghi chú |
|---|---|---|
| `video_id` (unique index) | 📄 | |
| `s_v` — global_index đầu tiên | 📄 | **Bắt buộc cho DANTE** |
| `e_v` — global_index cuối cùng | 📄 | **Bắt buộc cho DANTE** |
| `duration`, `fps`, `num_keyframes`, `ingest_status`, `ingested_at` | 🔧 | `ingest_status` là cơ chế checkpoint |
| `author`, `batch`, `title`, `publish_date`, `watch_url`, `channel_id` | 🔧 | Nạp từ `features/media-info/<video_id>.json`. ⚠️ `duration` lấy từ trường `length` (đơn vị **giây**); `publish_date` là chuỗi **`DD/MM/YYYY`** — parse tường minh, không để thư viện đoán ISO-8601 |

`ingest_status` nhận các giá trị `pending` / `decoding` / `embedded` / `ocr_done` / `complete` / `failed`. Đây chính là hiện thực của yêu cầu resume (T1): pipeline khởi động sẽ bỏ qua mọi video đang ở trạng thái `complete`.

---

## 5. Các lớp trừu tượng bắt buộc

### 5.1 `MultimodalEncoder`

Đây là hàng rào chống rủi ro R1 (BEiT-3 khó cài). Mọi nơi trong pipeline chỉ được gọi qua interface này.

```python
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


class MultimodalEncoder(Protocol):
    """Đưa ảnh và văn bản về cùng một không gian vector đã L2-normalize."""

    @property
    def embedding_dim(self) -> int: ...

    @property
    def model_id(self) -> str:
        """Định danh dùng để đặt tên collection Milvus — đổi model là đổi collection."""
        ...

    def encode_image(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Trả về mảng shape (N, embedding_dim), dtype float32, đã L2-normalize."""
        ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Trả về mảng shape (N, embedding_dim), dtype float32, đã L2-normalize."""
        ...
```

Implementation dự kiến: `BEiT3Encoder` (chính, theo paper), `SigLIPEncoder` / `OpenClipEncoder` (dự phòng).

**Ràng buộc bất biến:** `model_id` phải là một phần trong tên collection Milvus. Nó khiến việc trộn nhầm vector của hai model trở nên bất khả thi ở cấp thiết kế, chứ không phụ thuộc vào kỷ luật của người viết code.

### 5.2 `DatasetAdapter`

Cho phép đổi dataset — mở rộng pilot → toàn kho, hoặc nạp batch mới BTC phát hành — bằng cách thêm một class, không sửa pipeline (yêu cầu T3).

```python
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VideoRef:
    video_id: str
    source_path: str
    duration_sec: float
    category: str | None = None


@dataclass(frozen=True)
class GroundTruthSegment:
    video_id: str
    start_sec: float
    end_sec: float
    order_index: int      # thứ tự trong chuỗi — nguồn nhãn cho TRAKE
    caption: str          # nguồn truy vấn cho Textual KIS


class DatasetAdapter(Protocol):
    def iter_videos(self, split: str) -> Iterator[VideoRef]: ...
    def load_ground_truth(self, split: str) -> Sequence[GroundTruthSegment]: ...
```

`AIC2026Adapter` là implementation hiện tại. Nó khác `YouCook2Adapter` của bản 0.1 ở một điểm bản chất: **hai phương thức lấy dữ liệu từ hai nguồn khác nhau**.

| Phương thức | Nguồn |
|---|---|
| `iter_videos(split)` | Glob `Videos/Videos/Videos_*/video/*.mp4` + đọc `features/media-info/<video_id>.json`. `split` nhận `"pilot"` (`L21`–`L24`) hoặc `"all"`. **Bắt buộc khử trùng `video_id`** — 961 file, 873 id |
| `load_ground_truth(split)` | Đọc bộ nhãn **tự sinh** `configs/eval/aic_eval_v1.json` (PRD §6.1). Dataset gốc **không** có nhãn |

Hai hệ quả thiết kế:

1. **`load_ground_truth` chỉ dùng được sau khi đã ingest**, vì nhãn được sinh từ shot do TransNetV2 cắt ra. Đây là đảo chiều so với YouCook2 (nhãn có trước, ingest sau) — `eval/run.py` phải fail rõ ràng nếu bộ nhãn chưa tồn tại, không được trả bảng metric rỗng.
2. Chữ ký `VideoRef` / `GroundTruthSegment` / `DatasetAdapter` **không đổi một dòng nào** so với bản 0.1. Đó chính là bằng chứng T3 được thiết kế đúng — điều mà bản 0.1 định kiểm tra mãi tới P4.

⚠️ Đường dẫn phải dựng bằng **glob**, không bằng chuỗi định dạng: tên thư mục không nhất quán (`Videos_L21_a` … nhưng `Video_L30_a` số ít). Chi tiết: `references.md` §2.2.

---

## 6. Cấu trúc thư mục

Tuân thủ `.claude/rules/ecc/python/fastapi.md` (`create_app()`, router mỏng, logic ở service, schema tách biệt, DI qua `Depends`).

```
AIC-Video-Retrieval/
├── docs/                       PRD.md · architecture.md · references.md
├── app/
│   ├── main.py                 create_app()
│   ├── config.py               pydantic-settings, nạp từ .env
│   ├── api/                    routers MỎNG — chỉ validate và uỷ quyền
│   │   ├── search.py  trake.py  qa.py  keyframes.py  submissions.py  health.py
│   ├── services/               nơi chứa toàn bộ logic nghiệp vụ
│   │   ├── search_service.py       semantic · ocr (hybrid RRF bỏ 2026-08-12 — bộ nhãn không đo được kênh OCR công bằng)
│   │   ├── quest_service.py        QUEST 2 nhánh
│   │   ├── dante_service.py        quy hoạch động TRAKE
│   │   ├── vqa_service.py          Gemini + kiểm tra grounding
│   │   └── submission_service.py   xuất file nộp
│   ├── encoders/               base.py (Protocol) · beit3.py · siglip.py
│   ├── stores/                 milvus_client.py · es_client.py · mongo_client.py
│   ├── schemas/                request/ và response/ TÁCH RIÊNG
│   └── dependencies.py         Depends: encoder, các store client
├── pipelines/
│   ├── ingest.py               điều phối, quản lý checkpoint
│   ├── shot_detect.py          TransNetV2
│   ├── keyframe.py             lấy mẫu 4 vị trí mỗi shot
│   ├── ocr.py                  lọc + dedupe + Gemini
│   ├── embed.py                encode theo lô + ghi Milvus
│   ├── asr.py                  faster-whisper + lọc ảo giác (P0.11a, ngoài đường găng P0)
│   ├── embed_transcripts.py    bge-m3 encode transcript → Milvus (P0.11b)
│   └── export_asr.py           dump index asr_segments ra 1 file JSON chia sẻ (P0.11a)
├── datasets/                   base.py (Protocol) · aic2026.py
├── eval/
│   ├── run.py                  điểm vào duy nhất
│   ├── build_eval_set.py       sinh bộ nhãn tự động (PRD §6.1) — dataset KHÔNG kèm nhãn
│   ├── metrics/                retrieval.py · trake.py · vqa.py
│   └── baselines/              random.py · bm25.py · clip.py
├── configs/                    ingest/ · eval/ · model/
├── frontend/                   React + Vite + TS (P1.6 — xem §8)
├── docker-compose.yml          Milvus · Elasticsearch · MongoDB
└── tests/
```

---

## 7. API contract

| Method | Endpoint | Mục đích | Ghi chú |
|---|---|---|---|
| `POST` | `/api/search` | Tìm bằng văn bản | Body: `query`, `mode` (`semantic` \| `ocr`), `top_k`, `filters` |
| `POST` | `/api/search/image` | Tìm bằng ảnh upload | `multipart/form-data`; dùng **chung** không gian vector |
| `POST` | `/api/quest` | Viết lại truy vấn / tìm ảnh mẫu | Trả về **cả** truy vấn gốc và bản viết lại |
| `POST` | `/api/trake` | Chuỗi N sự kiện | Body: `events[]`, `lambda_penalty`, `top_k` |
| `POST` | `/api/qa` | Hỏi đáp VQA | Trả `answer`, `confidence`, `evidence_keyframe_ids[]` |
| `GET` | `/api/keyframes/{id}` | Chi tiết một keyframe | ⬜ chưa xây (P2/P3) |
| ✅ `GET` | `/api/keyframes/{id}/thumbnail` | Ảnh thumbnail đã resize | P1.6. Resize on-demand + cache đĩa (`app/services/thumbnail_service.py`) — corpus nhập từ Kaggle không có thumbnail sẵn (`thumbnail_path == image_path`). `keyframe_id` validate bằng regex trước khi chạm filesystem |
| ✅ `GET` | `/api/keyframes/{id}/image` | Ảnh gốc full-res | P1.6. Tải theo yêu cầu, không resize |
| ✅ `GET` | `/api/videos/{id}` | Thông tin video kèm `s_v`, `e_v` | P1.6. Wrapper mỏng trên `MongoStore.get_video()`. `watch_url` phục vụ nút "Play YouTube" |
| ✅ `GET` | `/api/videos/{id}/keyframes` | Toàn bộ keyframe của một video | P1.6. Wrapper mỏng trên `MongoStore.get_keyframes_for_video()` — không có trong paper, phục vụ panel keyframe lân cận trong `KeyframeDetail` |
| `POST` | `/api/submissions/export` | Xuất file nộp | Validate trước khi ghi |
| `GET` | `/health` | Kiểm tra kết nối 3 store | |

**Quy ước chung:**
- Mọi endpoint khai báo `response_model` (theo rules ECC)
- Endpoint có I/O dùng `async def` với client bất đồng bộ
- Encoder và store client inject qua `Depends`, **không** khởi tạo trong route handler
- Lỗi trả về theo một schema thống nhất: `{detail, error_code, request_id}`

---

## 8. Ghi chú về Frontend

**Cập nhật 2026-08-12:** chủ dự án mở gate P1.5, cho phép build UI theo danh sách năng lực chức năng của paper thay vì chờ mẫu thiết kế riêng — *"cứ build UI theo như paper baseline cho mình đi, mình cho phép điều đó. Có gì sau này mình sẽ kêu chỉnh sửa lại UI sau"*. Chi tiết quyết định: `.claude/plans/p1-search.plan.md` §P1.5.

Mô tả UI trong paper baseline — ô nhập kèm nút AI enhancer, mode selector Semantic/OCR/DANTE, lưới kết quả, nút "Play YouTube" / "Find Similar Images", bộ lọc metadata, upload ảnh — chỉ là **danh sách năng lực chức năng cần hỗ trợ**, không phải bản thiết kế phải sao chép. Vì không có mẫu thiết kế, P1.6 chọn CSS viết tay (không UI kit) để lần restyle sau không phải gỡ abstraction của thư viện nào.

QUEST (AI enhancer)/DANTE/VQA chưa có backend (P2/P3) — hiển thị trong UI ở trạng thái **disabled kèm badge "P2"/"P3"** thay vì ẩn hẳn, để bố cục chốt một lần và P2/P3 chỉ việc nối handler thật vào.

Các ràng buộc **kỹ thuật** (độc lập với hình thức giao diện) vẫn giữ nguyên:
- Virtualized rendering cho lưới hàng nghìn thumbnail
- Có shortcut bàn phím cho thao tác nóng
- Đổi mode không mất truy vấn đang gõ
- Phục vụ thumbnail nén, tải ảnh gốc theo yêu cầu

---

## 9. Triển khai

### 9.1 Bố trí trên một máy (RTX 3070 8GB, đổi từ RTX 2080Ti 11GB khi chuyển máy 2026-08-01)

| Chạy trong Docker | Chạy native |
|---|---|
| Milvus standalone (kèm etcd, MinIO) | FastAPI app |
| Elasticsearch + plugin `vi_analyzer` | Encoder service (cần truy cập GPU trực tiếp) |
| MongoDB | Pipeline ingest |

**Vì sao encoder chạy native:** truyền GPU vào container trên Windows thêm một lớp phức tạp không cần thiết cho môi trường một máy.

### 9.2 Ngân sách tài nguyên

| Thành phần | RAM | Ghi chú |
|---|---|---|
| Elasticsearch | ≤ 2 GB heap | Đặt cứng qua `ES_JAVA_OPTS` |
| Milvus standalone | ~4 GB | Tăng theo kích thước index |
| MongoDB | ~1 GB | |
| FastAPI + encoder | ~2 GB RAM và ≤ 8 GB VRAM | |
| **Tổng** | **≥ 16 GB, khuyến nghị 32 GB** | Sức ép nằm ở RAM hệ thống, không phải VRAM |

**Chế độ suy giảm:** tắt Elasticsearch để giải phóng ~2 GB; hệ thống chuyển sang semantic-only, mất chức năng OCR search.

### 9.3 Vòng đời index

```mermaid
stateDiagram-v2
    [*] --> Empty
    Empty --> Ingesting: khởi động pipeline
    Ingesting --> Ingesting: checkpoint từng video
    Ingesting --> Interrupted: lỗi hoặc mất điện
    Interrupted --> Ingesting: chạy lại, bỏ qua video complete
    Ingesting --> Ready: hoàn tất
    Ready --> Serving: API sẵn sàng
    Serving --> Ready: nạp thêm video mới
    Ready --> Empty: đổi model embedding, bắt buộc tạo collection MỚI
```

Nhánh cuối là ràng buộc quan trọng nhất trong sơ đồ: **đổi model embedding không bao giờ được ghi đè collection cũ**. Vector của hai model khác nhau không so sánh được, và lỗi này không gây crash — nó chỉ làm điểm số tệ đi một cách khó truy vết.

---

## 10. Truy vết quyết định thiết kế

| Quyết định | Nguồn | Lý do |
|---|---|---|
| TransNetV2 cắt shot | 📄 Paper | Bám baseline |
| 4 keyframe mỗi shot | 📄 Paper | Công thức nguyên văn, xem §F1 của PRD |
| BEiT-3 làm encoder | 📄 Paper | Bám baseline; có `MultimodalEncoder` để rút lui |
| Milvus + ES + MongoDB | 📄 Paper | Quyết định đã chốt: bám đúng baseline |
| DANTE cho TRAKE | 📄 Paper | Công thức và độ phức tạp giữ nguyên |
| QUEST 2 nhánh | 📄 Paper | Giữ nguyên, thêm điểm can thiệp của người dùng |
| Hybrid fusion RRF (semantic+OCR) | 🔧 Dự án, **bỏ 2026-08-12** | Đo được: RRF fuse theo rank kéo kênh mạnh xuống khi trộn kênh yếu; và bộ nhãn `aic_eval_v1` cấm caption chép chữ màn hình nên không đo công bằng được kênh OCR (eval/build_eval_set.py prompt). `_reciprocal_rank_fusion` giữ lại làm hàm thuần, chưa wire — để dành fuse hai encoder semantic cùng loại sau này (docs/research-encoder-vietnamese.md §8.3) |
| OCR = mode TEXT/BM25 riêng, không fuse | 🔧 Chủ dự án 2026-08-12 | Đúng cách các đội AIC mạnh nhất (Vortex, MERVIN) dùng OCR: filter/mode do người vận hành chọn, không phải kênh ranking tự động |
| Lọc + dedupe trước OCR | 🔧 Dự án | Chi phí Gemini trên ~400k keyframe không khả thi (R2) |
| `MultimodalEncoder` Protocol | 🔧 Dự án | Chống rủi ro cài đặt BEiT-3 (R1) |
| `DatasetAdapter` Protocol | 🔧 Dự án | Đổi/mở rộng dataset không sửa pipeline (T3) |
| Tự trích keyframe bằng TransNetV2, **không** dùng keyframe BTC trích sẵn | 🔧 Chủ dự án 2026-07-29 | Bám baseline. Đổi lại: tốn công giải mã 130,7 giờ video, và nhãn eval sinh từ chính shot của mình nên metric không phạt được lỗi cắt shot (rủi ro R10) |
| Nguồn keyframe/vector hiện tại: BEiT-3 encode **trên Kaggle**, keyframe do thành viên nhóm cung cấp (không phải TransNetV2 tự trích + encode cục bộ như dòng trên) | 🔧 Chủ dự án 2026-08-04 | RTX 3070 8GB không đủ chạy encode 221k keyframe trong thời gian hợp lý; Kaggle cấp GPU rảnh. Nhóm đã kiểm tra kỹ bộ keyframe cung cấp, không đối chứng lại TransNetV2. Hợp đồng artifact + parity probe: `.claude/plans/kaggle-offline-embed.plan.md` |
| Bộ eval **nhãn tự sinh** | 🔧 Chủ dự án 2026-07-29 | Dataset công khai không kèm nhãn; đáp án thật chỉ biết qua nộp bài. Không có nhãn thì không đo được, mà không đo được thì vi phạm nguyên tắc xuyên suốt của dự án |
| `author` / `batch` thay `recipe_type` | 🔧 Dự án | Dataset AIC không có trường phân loại nội dung tương đương |
| Cổng grounding trước khi trả lời | 🔧 Dự án | Cần thiết để đo `Hallucination Rate` |
| FastAPI + React/Vite | 🔧 Dự án | Paper không nêu framework |
| Nhánh ASR (P0.11, faster-whisper) — đảo non-goal | 🔧 Chủ dự án 2026-08-02 | Baseline không có ASR; bản tin có lời dẫn nói ra nhiều thứ không hiện thành chữ (tên người, địa danh, số liệu) — bổ trợ OCR. Là job GPU offline không tốn quota Gemini nên chạy được đúng lúc nhánh OCR đang chờ quota, không chặn P0.8-P0.10 |
| `bge-m3` làm encoder thứ ba cho transcript (P0.11b), không dùng lại BEiT-3 | 🔧 Dự án 2026-08-02 | `beit3_base_itc` học contrastive ảnh-chữ, không phải chữ-chữ; alignment học trên caption tiếng Anh nên vector tiếng Việt lệch. Lời nói tự phát (paraphrase cao) khó khớp BM25 hơn OCR nhiều — cần kênh dense riêng |

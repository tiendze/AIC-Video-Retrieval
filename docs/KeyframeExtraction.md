# Batch Shot Boundary Detection + CAPS Keyframe Extraction

README này mô tả pipeline đang được triển khai trong notebook:

```text
Video
  -> Shot Boundary Detection
  -> Final shots
  -> CAPS Keyframe Extraction
  -> Keyframe images + metadata
```

Mục tiêu của pipeline là:

1. Chia mỗi video thành các shot tương đối chính xác.
2. Chỉ gọi OmniShotCut ở những vùng khó thay vì chạy trên toàn bộ video.
3. Chọn số keyframe thích nghi cho từng shot.
4. Giữ lại frame đại diện, frame có thay đổi ngắn, frame có chữ và frame bổ sung diversity.
5. Chạy theo batch trên toàn bộ dataset và hỗ trợ resume/sharding.

---

# 1. Cấu trúc dữ liệu

Notebook tìm video theo dạng:

```text
/kaggle/input/ai_challange_2026/
├── Videos_L21_a/
│   └── video/
│       ├── L21_V001.mp4
│       └── ...
├── Videos_L30_a/
│   └── video/
│       ├── L30_V001.mp4
│       └── ...
└── map-keyframes-aic25-b1/
    └── map-keyframes/
        ├── L21_V001.csv
        ├── L21_V002.csv
        └── ...
```

Video và CSV được ghép theo `video_id`:

```text
L21_V001.mp4 <-> L21_V001.csv
```

---

# 2. Vai trò của `map-keyframes`

`map-keyframes` chỉ dùng để kiểm tra quan hệ giữa:

- `frame_idx`;
- `pts_time`;
- FPS của video.

Nó không được dùng làm ground-truth keyframe cho CAPS.

Với mỗi dòng trong CSV, notebook kiểm tra sai số thời gian:

$$
e_t = \left|\text{pts\_time} - \frac{\text{frame\_idx}}{\text{fps}_{\text{video}}}\right|
$$

và sai số frame:

$$
e_f = \left|\text{frame\_idx} - \operatorname{round}(\text{pts\_time}\cdot\text{fps}_{\text{video}})\right|
$$

Kết quả được tổng hợp thành:

```text
PASS
WARN
FAIL
MISSING
```

Trong phiên bản tối ưu, visual seek check mặc định được tắt để tránh mở và seek video quá nhiều lần.

---

# 3. Pipeline tổng quát

```text
Dataset discovery
    ↓
Video <-> map CSV matching
    ↓
Map timing alignment check
    ↓
TransNetV2 chạy toàn video
    ↓
Strong boundary + suspicious candidate
    ↓
Merge suspicious windows
    ↓
OmniShotCut chạy trên vùng khó
    ↓
Merge TransNetV2 + OmniShotCut boundaries
    ↓
Final shots
    ↓
CAPS candidate sampling
    ↓
DINOv2 embeddings
    ↓
Raw temporal visual change
    ↓
Adaptive frame budget
    ↓
Anchor + text + peaks + rare state + diversity
    ↓
Original-FPS quality refinement
    ↓
Keyframe images + metadata
```

---

# 4. Shot Boundary Detection

## 4.1. TransNetV2 là detector chính

TransNetV2 chạy trên toàn bộ video và trả một probability cho mỗi frame:

$$
p_t^{T} \in [0,1]
$$

Trong đó:

- giá trị thấp: frame ít có khả năng là boundary;
- giá trị cao: frame có khả năng nằm tại điểm chuyển shot.

Một boundary mạnh được chấp nhận khi:

$$
p_t^{T} \geq \tau_{\text{strong}}
$$

Mặc định:

$$
\tau_{\text{strong}} = 0.5
$$

Các boundary mạnh tạo ra danh sách TransNetV2 shots ban đầu.

---

## 4.2. Suspicious candidate là gì?

Suspicious candidate là một vị trí chưa đủ chắc chắn để trở thành final boundary, nhưng có dấu hiệu đáng ngờ và cần OmniShotCut kiểm tra thêm.

Notebook tạo suspicious candidate từ hai nguồn chính.

### Nguồn A: weak TransNetV2 peak

Một weak peak thỏa:

$$
\tau_{\text{weak}} \leq p_t^{T} < \tau_{\text{strong}}
$$

Ví dụ với profile `balanced`:

```text
0.35 <= TransNet score < 0.50
```

Weak peak còn phải:

- là local maximum;
- đủ xa strong boundary đã có;
- đạt prominence tối thiểu;
- không vượt quá số candidate tối đa theo phút video trong profile `balanced`.

Weak peak không tự trở thành final boundary. Nó chỉ tạo một vùng cần kiểm tra.

### Nguồn B: DINOv2 anomaly trong shot dài

TransNetV2 đôi khi gộp một đoạn dài dù bên trong có thay đổi thị giác đáng kể.

Trong mỗi TransNetV2 shot đủ dài, notebook lấy các frame thưa và tính:

$$
d_i^{D} = 1 - \cos(z_i,z_{i-1})
$$

Trong đó:

- $z_i$: DINOv2 embedding của candidate thứ $i$;
- $d_i^{D}$ lớn: hai candidate liên tiếp khác nhau rõ rệt.

Baseline robust:

$$
m_D = \operatorname{median}(d^{D})
$$

Median Absolute Deviation:

$$
\operatorname{MAD}(d^{D}) = \operatorname{median}\left(\left|d_i^{D}-m_D\right|\right)
$$

Ngưỡng anomaly:

$$
\tau_D = m_D + \lambda_D\operatorname{MAD}(d^{D})
$$

Một local peak vượt ngưỡng và có prominence đủ lớn sẽ trở thành suspicious candidate.

### Tóm tắt suspicious candidate

Một vị trí được gọi là suspicious nếu nó thuộc ít nhất một trường hợp:

```text
1. TransNetV2 có weak peak.
2. DINOv2 thấy thay đổi mạnh bên trong một TransNetV2 shot dài.
```

---

## 4.3. Tạo suspicious window

Mỗi suspicious candidate tại frame $c$ tạo một cửa sổ:

$$
W_c = [c-r_{\text{before}},\;c+r_{\text{after}}]
$$

Ví dụ profile `balanced` dùng khoảng:

```text
1.5 giây trước candidate
1.5 giây sau candidate
```

Mục đích là cho OmniShotCut đủ ngữ cảnh để quyết định:

- có boundary thật hay không;
- boundary nằm chính xác ở frame nào;
- có gradual transition hay không.

---

## 4.4. Merge suspicious windows

Nhiều candidate gần nhau có thể tạo các window chồng lặp.

Ví dụ:

```text
Candidate A -> window [100, 190]
Candidate B -> window [170, 260]
```

Hai window này chồng nhau nên được merge thành:

```text
Merged window -> [100, 260]
```

Hai window cũng được merge nếu khoảng trống giữa chúng nhỏ hơn:

```text
MERGE_WINDOW_GAP_SEC
```

Thuật toán đơn giản:

1. Sắp window theo `start_frame`.
2. Lấy window đầu làm group hiện tại.
3. Nếu window tiếp theo bắt đầu trước `current_end + gap`, gộp vào group.
4. Nếu không, đóng group cũ và tạo group mới.
5. Mỗi merged window lưu lại toàn bộ reason và score mạnh nhất.

Sau merge, mỗi window có:

```text
start_frame
end_frame
center_frames
reasons
max_transnet_score
max_dino_score
router_score
```

Điều này tránh chạy OmniShotCut nhiều lần trên cùng một vùng video.

---

## 4.5. Chọn window để chạy OmniShotCut

Trong profile `exhaustive`:

```text
Chạy tất cả merged suspicious windows.
```

Trong profile `balanced`:

```text
Xếp window theo router score.
Chỉ giữ tối đa MAX_OMNISHOT_WINDOWS_PER_VIDEO window mạnh nhất.
```

Profile mặc định hiện tại là:

```python
RUNTIME_PROFILE = "balanced"
```

`balanced` nhanh hơn nhưng có thể bỏ một số candidate yếu. Khi cần kiểm tra độ bao phủ tối đa, dùng:

```python
RUNTIME_PROFILE = "exhaustive"
```

---

## 4.6. OmniShotCut xử lý vùng khó

OmniShotCut chỉ chạy trên các suspicious windows đã được chọn.

Nếu OmniShotCut trả boundary local tại frame $b_{\text{local}}$, notebook đổi về frame toàn video:

$$
b_{\text{global}} = w_{\text{start}} + b_{\text{local}}
$$

Trong đó $w_{\text{start}}$ là frame bắt đầu của suspicious window.

---

## 4.7. Merge TransNetV2 và OmniShotCut boundary

Tất cả boundary được sắp theo frame index.

Nếu hai boundary cách nhau không quá:

```text
LOCALIZATION_TOLERANCE_FRAMES
```

thì chúng được xem là cùng một boundary.

Nguồn cuối cùng:

```text
transnet  : chỉ TransNetV2 phát hiện
omnishot : chỉ OmniShotCut phát hiện
both      : cả hai model phát hiện gần nhau
```

Nếu source là `both`, notebook dùng vị trí đại diện của nhóm boundary gần nhau.

Sau merge, boundary quá gần đầu shot trước hoặc tạo một shot quá ngắn sẽ bị loại bằng:

```text
MIN_SHOT_DURATION_SEC
```

---

## 4.8. Tạo final shots

Giả sử các final boundary là:

$$
0 < b_1 < b_2 < \cdots < b_M < N
$$

với $N$ là tổng số frame.

Các shot cuối là:

$$
s_1 = [0,b_1)
$$

$$
s_k = [b_{k-1},b_k)
$$

$$
s_{M+1} = [b_M,N)
$$

Đây là đầu vào trực tiếp của CAPS.

---

# 5. CAPS: Adaptive Candidate Selection

CAPS không xử lý toàn bộ frame trong shot. Nó chỉ lấy một tập candidate thưa, sau đó chọn một số keyframe cuối cùng theo nội dung của shot.

```text
Shot
  -> sparse candidates
  -> DINOv2 embeddings
  -> temporal visual change
  -> adaptive budget
  -> frame selection
  -> quality refinement
```

---

## 5.1. Bước 1: lấy candidate thưa

Với shot $s=[t_a,t_b)$, duration là:

$$
T_s = t_b-t_a
$$

Sampling mặc định:

```text
Shot dưới 2 giây : khoảng 6 FPS
Shot dài hơn     : khoảng 2 FPS
```

Số candidate gần đúng:

$$
N_s \approx T_s r_s
$$

Trong đó $r_s$ là sampling rate của shot.

Candidate timestamp:

$$
t_i = t_a + \left(i+\frac{1}{2}\right)\frac{1}{r_s}
$$

Frame index:

$$
n_i = \operatorname{round}(t_i\cdot\text{FPS})
$$

Candidate quá sát final boundary bị loại để tránh lấy frame đang transition.

---

## 5.2. Bước 2: trích xuất DINOv2 feature

Mỗi candidate frame $I_i$ được mã hóa:

$$
z_i = \operatorname{DINOv2}(I_i)
$$

Embedding được chuẩn hóa L2:

$$
\hat{z}_i = \frac{z_i}{\|z_i\|_2}
$$

Sau chuẩn hóa, cosine similarity được tính bằng:

$$
\cos(\hat{z}_i,\hat{z}_j)=\hat{z}_i^{\top}\hat{z}_j
$$

DINOv2 được dùng để mô tả visual state và cấu trúc cảnh.

---

## 5.3. Bước 3: temporal visual change

Mức thay đổi giữa hai candidate liên tiếp:

$$
d_i = 1 - \cos(z_i,z_{i-1})
$$

Ý nghĩa:

```text
d_i nhỏ -> hai frame gần giống nhau
d_i lớn -> trạng thái thị giác thay đổi mạnh
```

### Không dùng median filter mặc định

Pipeline hiện tại dùng raw change trực tiếp.

Lý do: một panel chữ hoặc một trạng thái ngắn thường tạo tín hiệu:

```text
thấp -> cao -> thấp
```

Median filter kích thước 3 có thể xóa chính peak ngắn này.

Median và MAD chỉ được dùng để tính ngưỡng robust.

Baseline:

$$
m_s = \operatorname{median}(d)
$$

MAD:

$$
\operatorname{MAD}(d)=\operatorname{median}\left(\left|d_i-m_s\right|\right)
$$

Ngưỡng peak:

$$
\tau_s = m_s + \lambda\operatorname{MAD}(d)
$$

Một novelty peak cần:

```text
1. Cao hơn threshold.
2. Là local maximum.
3. Có prominence đủ lớn.
4. Cách peak khác ít nhất MIN_PEAK_DISTANCE_SEC.
```

Peak score:

$$
S_{\text{peak}}(i)=\operatorname{prominence}(i)\cdot Q_i
$$

Trong đó $Q_i$ là quality score của candidate.

---

# 6. Adaptive frame budget

CAPS không quyết định số keyframe chỉ bằng duration.

Ví dụ:

```text
Shot 30 giây đứng yên có thể chỉ cần 1 frame.
Shot 8 giây có nhiều thao tác có thể cần 4 frame.
```

Tổng lượng visual change vượt baseline:

$$
L_s = \sum_i \max\left(0,d_i-\operatorname{median}(d)\right)
$$

Budget:

$$
K_s = \operatorname{clip}\left(1+\left\lfloor\frac{L_s}{\gamma}+0.5\right\rfloor,\;1,\;K_{\max}\right)
$$

Trong đó:

- $K_s$: số keyframe cuối của shot;
- $K_{\max}$: giới hạn keyframe tối đa;
- $\gamma$: lượng visual change cần để cấp thêm một frame.

Ý nghĩa:

```text
L_s gần 0 -> K_s = 1
L_s lớn   -> shot nhận thêm frame
```

Nếu text rescue được bật và shot có textness cao, budget có thể được nâng tối thiểu lên 2 để text frame không phải cạnh tranh với anchor duy nhất.

---

# 7. Chọn frame trong budget

## 7.1. Representative anchor

Tính centroid embedding:

$$
\bar{z}_s = \frac{1}{N_s}\sum_i z_i
$$

Chọn candidate gần centroid nhất:

$$
f_{\text{rep}} = \operatorname*{arg\,max}_i \cos(z_i,\bar{z}_s)
$$

Anchor là frame đại diện cho trạng thái phổ biến nhất của shot và luôn chiếm một slot.

---

## 7.2. Text-aware frame

Nếu shot có vùng giống chữ đủ mạnh, chọn:

$$
f_{\text{text}} = \operatorname*{arg\,max}_i T_iQ_i
$$

Trong đó:

- $T_i$: textness score;
- $Q_i$: quality score.

Đây là heuristic phát hiện vùng giống chữ, không phải OCR recognition.

---

## 7.3. Temporal surprise peaks

Các novelty peak được xếp theo:

$$
S_{\text{peak}}(i)=\operatorname{prominence}(i)Q_i
$$

Peak tốt nhất được thêm lần lượt cho đến khi:

```text
Đủ K_s frame
hoặc
Không còn peak hợp lệ
```

---

## 7.4. Rare-state rescue

Một trạng thái hiếm thường nằm xa centroid embedding:

$$
R_i = \left(1-\cos(z_i,\bar{z}_s)\right)Q_i
$$

Nếu $R_i$ nổi bật đủ mạnh và budget còn chỗ, frame đó được giữ trước diversity filling.

---

## 7.5. Diversity filling

Nếu vẫn chưa đủ budget:

$$
|S| < K_s
$$

CAPS chọn candidate mới bằng:

$$
f^{*}=\operatorname*{arg\,max}_{i\notin S} Q_i\left[\alpha\min_{j\in S}D_{\text{visual}}(i,j)+(1-\alpha)\min_{j\in S}D_{\text{time}}(i,j)\right]
$$

Visual distance:

$$
D_{\text{visual}}(i,j)=1-\cos(z_i,z_j)
$$

Temporal distance:

$$
D_{\text{time}}(i,j)=\frac{|t_i-t_j|}{T_s}
$$

Mặc định:

$$
\alpha=0.7
$$

Nghĩa là:

```text
70% ưu tiên visual diversity
30% ưu tiên temporal coverage
```

Hàm `min` kiểm tra candidate mới có quá gần bất kỳ frame nào đã chọn hay không.

---

# 8. Quality refinement ở original FPS

Timestamp candidate có thể đúng về nội dung nhưng frame chính xác tại timestamp đó có thể:

- motion blur;
- quá tối;
- cháy sáng;
- nằm sát transition;
- panel chữ chưa đứng yên.

Vì vậy notebook quay lại video gốc và xét một cửa sổ nhỏ quanh timestamp.

Với frame thông thường:

$$
[t_i-r,\;t_i+r]
$$

Với text frame, cửa sổ được mở dài hơn về phía sau để chờ panel hoàn thành animation.

Quality score:

$$
Q(f)=w_1Q_{\text{sharp}}-w_2Q_{\text{dark}}-w_3Q_{\text{overexposed}}-w_4Q_{\text{transition}}
$$

Sharpness:

$$
Q_{\text{sharp}}=\operatorname{Var}(\nabla^2 I)
$$

Dark ratio:

$$
Q_{\text{dark}}=\frac{|\{p\mid I(p)<T_{\text{dark}}\}|}{|\{p\}|}
$$

Overexposed ratio:

$$
Q_{\text{overexposed}}=\frac{|\{p\mid I(p)>T_{\text{bright}}\}|}{|\{p\}|}
$$

Refinement chỉ thay timestamp nhẹ trong cùng khoảnh khắc, không thay đổi budget hay vai trò frame.

---

# 9. Cấu hình profile

## Balanced

```python
RUNTIME_PROFILE = "balanced"
```

Đặc điểm:

- weak threshold cao hơn;
- yêu cầu prominence;
- giới hạn weak candidate theo phút;
- tối đa 12 OmniShotCut windows mỗi video;
- phù hợp chạy toàn dataset nhanh hơn.

## Exhaustive

```python
RUNTIME_PROFILE = "exhaustive"
```

Đặc điểm:

- weak threshold thấp hơn;
- không giới hạn suspicious windows;
- giữ nhiều DINO anomaly hơn;
- phù hợp kiểm tra độ bao phủ và validation.

Hai profile giữ nguyên cấu trúc pipeline. Khác biệt nằm ở số candidate được chuyển đến OmniShotCut.

---

# 10. Tối ưu cho T4 x2

Notebook phân bố model:

```text
GPU 0: TransNetV2 + OmniShotCut
GPU 1: DINOv2
```

Các tối ưu chính:

1. Shared sparse timeline để tái sử dụng DINOv2 embedding.
2. Batch DINOv2 lớn hơn.
3. CPU decode và GPU inference có prefetch.
4. Merge suspicious windows trước khi gọi OmniShotCut.
5. Merge quality-refinement intervals và decode mỗi vùng một lần.
6. Dùng NVENC để tạo OmniShotCut clip khi Kaggle FFmpeg hỗ trợ.
7. Resume theo từng video.
8. Hỗ trợ chia shard.

Các tối ưu này thay đổi cách thực thi và cache, nhưng không thay đổi các công thức selection chính của CAPS.

---

# 11. Output

```text
batch_keyframe_output_v3/
├── run_config.json
├── video_inventory.csv
├── dataset_summary.csv
├── all_boundaries.csv
├── all_keyframes_metadata.csv
├── map_alignment_report.csv
├── errors.csv
├── omnishot_ranking.csv
├── top_omnishot_boundary_plot.png
├── keyframes/
│   ├── L21_V001/
│   │   ├── L21_V001_s0001_k01_anchor_....jpg
│   │   └── ...
│   └── ...
└── per_video/
    ├── L21_V001/
    │   ├── transnet_predictions.csv
    │   ├── transnet_shots.csv
    │   ├── suspicious_windows.csv
    │   ├── omnishot_boundaries_raw.csv
    │   ├── final_boundaries.csv
    │   ├── final_shots.csv
    │   ├── caps_shot_summary.csv
    │   ├── keyframes_metadata.csv
    │   ├── summary.json
    │   └── _SUCCESS.json
    └── ...
```

`top_omnishot_boundary_plot.png` gồm:

- đường probability của TransNetV2;
- strong threshold;
- final boundary của TransNetV2;
- final boundary có OmniShotCut tham gia;
- tên video ở tiêu đề.

---

# 12. Resume và sharding

Một video hoàn tất khi có:

```text
per_video/<video_id>/_SUCCESS.json
```

Nếu:

```python
RESUME = True
```

notebook bỏ qua video đã hoàn tất.

Chia thành nhiều shard:

```python
NUM_SHARDS = 4
SHARD_INDEX = 0
```

Sau đó chạy `SHARD_INDEX` từ 0 đến 3.

---

# 13. Các tham số quan trọng

| Tham số | Balanced mặc định | Ý nghĩa |
|---|---:|---|
| `TRANSNET_STRONG_THRESHOLD` | 0.50 | Boundary mạnh |
| `TRANSNET_WEAK_THRESHOLD` | 0.35 | Weak TransNet candidate |
| `TRANSNET_WEAK_MIN_PROMINENCE` | 0.03 | Độ nổi bật tối thiểu của weak peak |
| `LONG_SHOT_MIN_SEC` | 10.0 | Shot đủ dài để chạy DINO anomaly |
| `MAX_OMNISHOT_WINDOWS_PER_VIDEO` | 12 | Số window tối đa gọi OmniShotCut |
| `BASE_SAMPLE_FPS` | 2.0 | Shared sparse timeline |
| `SHORT_SHOT_SAMPLE_FPS` | 6.0 | Candidate FPS cho shot ngắn |
| `LAMBDA_MAD` | 1.5 | Threshold temporal peak |
| `MIN_PEAK_DISTANCE_SEC` | 0.75 | Khoảng cách tối thiểu giữa peak |
| `BUDGET_GAMMA` | 0.20 | Visual change cho mỗi frame bổ sung |
| `K_MAX` | 6 | Keyframe tối đa mỗi shot |
| `DIVERSITY_ALPHA` | 0.70 | Trọng số visual/time diversity |
| `REFINEMENT_RADIUS_SEC` | 0.25 | Quality refinement window |

---

# 14. Những điểm cần đánh giá bằng ablation

1. Fixed one frame per shot.
2. Fixed three frames per shot.
3. Representative anchor.
4. Anchor + temporal peaks.
5. Adaptive path-length budget.
6. Diversity filling.
7. Rare-state rescue.
8. Original-FPS quality refinement.
9. Text rescue bật/tắt.
10. `balanced` so với `exhaustive` cho Shot Boundary Detection.

Metrics nên theo dõi:

- số keyframe trung bình mỗi shot;
- keyframe mỗi phút;
- state coverage;
- OCR recall;
- duplicate rate;
- retrieval Recall@K;
- số boundary source `transnet`, `omnishot`, `both`;
- thời gian xử lý mỗi video.

---

# 15. Giới hạn hiện tại

1. DINO anomaly có thể phản ứng với camera pan hoặc zoom dù không có boundary thật.
2. Textness heuristic không thay thế OCR detector.
3. `balanced` có thể bỏ suspicious candidate yếu vì giới hạn compute.
4. OmniShotCut localization phụ thuộc clip extraction và codec.
5. Adaptive budget cần được chứng minh bằng ablation, không nên tự động gọi là novel.
6. Final shot quality cần được kiểm tra trên một validation subset có annotation thủ công.

# Roadmap — AIC-Video-Retrieval

> **Ngày lập:** 2026-07-25 · **Trạng thái:** chưa thực thi (repo mới có `docs/`)
> **Nguồn sự thật:** [`docs/PRD.md`](../../docs/PRD.md) · [`docs/architecture.md`](../../docs/architecture.md) · [`docs/references.md`](../../docs/references.md)
> Mâu thuẫn giữa plan và docs ⇒ **docs thắng**, và phải báo mâu thuẫn ra.

Đây là điểm vào duy nhất của kế hoạch. Mỗi phase có một file riêng; file này chỉ giữ **toàn cảnh, thứ tự, và trạng thái**.

---

## 1. Bảng phase

| Phase | Tên | File chi tiết | Exit criteria (PRD §4) | Trạng thái |
|---|---|---|---|---|
| **P0** | Nền móng | [`p0-foundation.plan.md`](./p0-foundation.plan.md) | Search một truy vấn ra kết quả end-to-end; `eval/run.py` chạy được và in Recall@k dù số còn thấp | 🟡 đang làm |
| **P1** | Tìm kiếm cơ bản | [`p1-search.plan.md`](./p1-search.plan.md) | Recall@10 đạt mục tiêu PRD §6.7; UI dùng được để thao tác thật | ⬜ chưa bắt đầu |
| **P2** | Hai đóng góp của paper | [`p2-quest-dante.plan.md`](./p2-quest-dante.plan.md) | TRAKE `Sequence Exact-Match` đạt §6.7; **có bảng ablation λ** | ⬜ chưa bắt đầu |
| **P3** | VQA & nộp bài | [`p3-vqa-submission.plan.md`](./p3-vqa-submission.plan.md) | `Grounded Accuracy` đạt §6.7; file nộp qua validate | ⬜ chưa bắt đầu |
| **P4** | Mở rộng toàn kho & tối ưu | [`p4-aic-optimize.plan.md`](./p4-aic-optimize.plan.md) | Toàn bộ metrics chạy lại trên đủ 873 video; có báo cáo so sánh pilot ↔ full | ⬜ chưa bắt đầu |

Ký hiệu trạng thái: ⬜ chưa bắt đầu · 🟡 đang làm · ✅ xong (có số đo) · 🛑 bị chặn

---

## 2. Quyết định đã chốt với chủ dự án

### 2.1 Cập nhật 2026-07-29 — đổi dataset

Dữ liệu công khai **AIC 2026 batch 1** đã tải xong về `data/` (873 video, 130,7 giờ). **Phương án dataset thay thế YouCook2 bị bỏ.**

> **Cập nhật 2026-08-01.** Đề vòng sơ tuyển AIC 2026 đã công bố (`Thong tin vong So tuyen AIC2026.pdf`): **3 task** (bỏ Video KIS), nộp **100 đáp án xếp hạng**/truy vấn chấm `mean(R@k)` với k ∈ {1,5,20,50,100}, đáp án là **`frame_id` tuyệt đối**. Dữ liệu đợt 1 chính là batch 1 của AIC 2025; **batch 2 chưa phát hành**. Chỉ tải `Videos` + `media-info` + `map-keyframes` — keyframe/object/CLIP feature tự trích. Chi tiết: `references.md` §1.2 và §2.

| Quyết định | Chọn | Hệ quả lên kế hoạch |
|---|---|---|
| **Nguồn ground truth** | **Tự sinh nhãn đơn giản** | Dataset không kèm nhãn nào; đáp án thật chỉ biết qua nộp bài lên BTC. ⇒ **Task mới P0.4′** (sinh bộ eval) thay chỗ P0.4 downloader. Giao thức: PRD §6.1 |
| **Nguồn keyframe** | **Bám baseline** — TransNetV2 + tự trích 4 keyframe/shot | Keyframe BTC trích sẵn (177k ảnh) **không** dùng làm nguồn ingest, chỉ làm đối chứng. P0.7 giữ nguyên độ nặng; đường găng dịch từ *tải video* sang *giải mã video* |
| **Phạm vi P0** | **Pilot `L21`–`L24`** (128 video, 27,9 giờ) | Vòng lặp ingest→search→eval đủ nhanh để sửa sai. Mở rộng đủ 873 video ở P4 |

**Ba việc biến mất khỏi kế hoạch:** tải video bằng `yt-dlp` (P0.4) · chờ dữ liệu AIC để bắt đầu P4 · trạng thái *"nhánh OCR đã xây nhưng chưa kiểm chứng"*.

**Một việc mới xuất hiện:** sinh + review bộ nhãn, và **đo chất lượng chính bộ nhãn đó** (PRD §6.1.3).

### 2.2 Quyết định gốc (2026-07-25)

| Quyết định | Chọn | Hệ quả lên kế hoạch |
|---|---|---|
| ~~**Dữ liệu**~~ | ~~Chưa có annotation lẫn video~~ | **Thay bằng §2.1** — dữ liệu đã có đủ, chỉ thiếu nhãn |
| **Encoder** | **BEiT-3 trước** theo đúng baseline; chỉ chuyển SigLIP khi thực sự không dựng được | `MultimodalEncoder` Protocol viết từ ngày đầu (không phải "làm sau"); BEiT-3 có **timebox 1 ngày** |
| **Cấu trúc plan** | 1 roadmap + 5 file phase | Chính là bộ file này |
| **Frontend** | Chủ dự án sẽ gửi mẫu thiết kế sớm, trong P1 | P1 có checkpoint **🛑 DỪNG-HỎI** cứng trước dòng UI đầu tiên (task P1.5) |

---

## 3. Đường găng

```
P0.1 scaffold ──► P0.3 adapter ──┬──► P0.5 encoder BEiT-3 (timebox 1 ngày) ─┐
                                 │                                          ├──► P0.7 ingest ──► P0.4′ sinh nhãn ──► P0.8 search ──► P0.9 eval
     P0.2 docker-compose ──► P0.6 stores ──────────────────────────────────-┘      (VISION + TEXT/OCR + METADATA)      (semantic + ocr)
```

**Đường găng đã đổi hình so với bản 2026-07-25.** Trước đây việc dài nhất là *tải ~40 giờ video* và nó chạy nền song song với mọi thứ khác. Nay video đã có sẵn trên đĩa, và thay vào đó xuất hiện một **ràng buộc thứ tự cứng mới**:

> **P0.4′ (sinh nhãn) phải chạy SAU P0.7 (ingest).**

Lý do: nhãn được sinh từ **shot do TransNetV2 cắt ra**, mà shot chỉ tồn tại sau khi ingest xong (PRD §6.1.1). Đây là đảo chiều so với YouCook2, nơi nhãn có sẵn trước khi viết dòng code đầu tiên.

**Hệ quả thực tế:** không còn việc dài nào chạy nền được để giấu chi phí. Ba việc nặng — dựng BEiT-3 (P0.5), giải mã 27,9 giờ video (P0.7), sinh + review nhãn (P0.4′) — **nối tiếp nhau**. P0 vì thế dài hơn ước lượng ban đầu; nếu cần cắt, cắt phạm vi pilot xuống dưới 128 video chứ đừng cắt bước review nhãn.

---

## 4. Quy tắc xuyên suốt

1. **Không sang phase sau khi phase trước chưa có số đo.** Bài học trực tiếp từ khoảng trống của paper baseline (PRD §1.3): paper mô tả kiến trúc đầy đủ nhưng **không công bố một metric định lượng nào**, nên không ai biết đóng góp thực sự đến từ đâu.
2. **Báo cáo sau mỗi phase**, bằng tiếng Việt, theo template CLAUDE.md §8:
   ```
   Đã làm gì   — module/file nào được tạo hoặc sửa, và vì sao cần chúng
   Input       — dữ liệu / config / tham số đi vào, lấy từ đâu, định dạng ra sao
   Output      — sinh ra cái gì (file · collection · index · metrics), nằm ở đâu
   Cách verify — lệnh hoặc test chứng minh nó chạy đúng, kèm kết quả thực tế
   Còn thiếu   — phần chưa xong, giả định đang treo, rủi ro đã phát hiện
   ```
   Số liệu phải là **kết quả chạy thật**. Chưa chạy được thì ghi rõ "chưa đo", không suy đoán.
3. **TDD**: RED → GREEN → refactor. Coverage ≥ 80%. Tên test mô tả hành vi.
4. **Không sửa bộ eval để cải thiện số.** Đổi bộ eval ⇒ tăng version và chạy lại toàn bộ baseline.
5. Mọi tham số ảnh hưởng kết quả (`λ`, `k` của RRF, `top_k`, timeout, ngưỡng) nằm trong `configs/`, **không hard-code**.
6. Commit `<type>: <description>`, atomic, **không commit khi test đỏ**.
7. Mọi báo cáo metric ghi kèm: git SHA · model checkpoint · version index · **số video thực tế đánh giá được** · seed.

---

## 5. Invariants — vi phạm là hỏng âm thầm

Nhóm lỗi không crash, chỉ làm điểm số tệ đi mà không rõ lý do. Kiểm tra ở **mọi** phase:

- **Một encoder duy nhất** — vector lúc `ingest` và lúc `query` sinh từ cùng model + cùng preprocessing.
- **L2-normalize ngay tại encoder**, không rải rác ở nơi gọi.
- **`model_id` nằm trong tên collection Milvus** — đổi embedding model ⇒ tạo collection MỚI, không ghi đè.
- **`global_index`, `s_v`, `e_v` là bắt buộc** — thiếu là DANTE không chạy được.
- **DANTE giữ ràng buộc `τ < t`** — chuỗi trả về không bao giờ đảo thứ tự thời gian.
- **VQA phải qua grounding gate** — `evidence_keyframe_ids` nằm ngoài tập truy hồi ⇒ trả "không đủ căn cứ".
- **Filter chạy trong engine** (tham số `filter` của Milvus), không lọc sau khi lấy kết quả về.
- **Ingest idempotent + resume được.**

---

## 6. Rủi ro theo dõi

| # | Rủi ro | Phase liên quan | Trạng thái | Giảm thiểu |
|---|---|---|---|---|
| **R1** | BEiT-3 khó cài (không có trên HF Transformers) | P0.5 | 🔴 đang mở | Timebox 1 ngày → fallback `SigLIPEncoder` qua Protocol, ghi nhận là sai lệch baseline |
| **R2** | Chi phí / rate-limit Gemini OCR — keyframe 1280×720 tốn **516 token/ảnh**; riêng 177k keyframe BTC đã là ~91,5 triệu token. Video tin tức có chữ ở gần như mọi frame | **P0.7** (kéo từ P1.2 về) | 🔴 đang mở | **Dedupe perceptual hash là đòn bẩy chính** (bản tin giữ nguyên khung nhiều giây); hạ ảnh về ≤ 384×384 còn 258 token sau khi kiểm chứng chữ nhỏ vẫn đọc được; Batch API; **P0 chỉ OCR pilot `L21`–`L24` — ngân sách ~13,4 triệu token, hoặc ~6,7 triệu nếu hạ 384×384**. Đặt trần chi phí trong `configs/`, vượt trần thì dừng chứ đừng đốt tiếp |
| **R3** | RAM khi chạy Milvus + ES + MongoDB cùng máy | P0.2 | 🟡 | Giới hạn `ES_JAVA_OPTS` ≤ 2 GB; hỗ trợ chế độ semantic-only |
| **R4** | ~~YouCook2 không có chữ trên màn hình~~ | — | ✅ **đóng 2026-07-29** | Dataset AIC là tin tức tiếng Việt, đầy chữ trên màn hình ⇒ OCR + `vi_analyzer` kiểm chứng được **ngay từ P0**, không phải chờ P4.2 |
| **R5** | ~~Link YouTube bị gỡ~~ | — | ✅ **đóng 2026-07-29** | Dataset phát hành file `.mp4` trực tiếp. Rủi ro thay thế (nhỏ): 88 file trùng `video_id` — adapter phải khử trùng, nếu không mẫu số bị thổi phồng |
| **R6** | **Không có nhãn cho bất kỳ task nào** (không chỉ QA) | **P0.4′** | 🔴 đang mở | Sinh theo PRD §6.1 + review tay ≥ 100 mẫu; **đo chất lượng chính bộ nhãn** (§6.1.3); model sinh nhãn ≠ model bị đánh giá; hiệu chỉnh qua nộp bài BTC (§6.9) |
| **R7** | Ingest đứt giữa chừng | P0.7 | 🟡 | Checkpoint theo `video_id` qua `ingest_status`; chia shard theo `batch` |
| **R8** | Định dạng file nộp của BTC chưa công bố | P3.5 | 🟡 | Tách formatter khỏi dữ liệu kết quả |
| **R9** | **Nhãn tự sinh ≠ truy vấn thật của BTC** (trước đây R9 là lệch miền dữ liệu; miền nay đã đúng, lệch chuyển sang phân bố truy vấn) | P0.4′ → P4 | 🔴 đang mở | Coi mọi số đo là **tương đối**; ưu tiên cải tiến kiến trúc hơn tinh chỉnh siêu tham số; bắt buộc hiệu chỉnh qua §6.9 |
| **R10** | **Nhãn và hệ thống dùng chung một bộ cắt shot** ⇒ lỗi shot detection không bị metric phạt | P0.7 → P0.9 | 🟡 mới | Đo `tIoU` giữa shot của mình và `map-keyframes` BTC — nguồn đối chứng độc lập duy nhất; báo cáo kèm phân bố độ dài shot |

---

## 7. Câu hỏi mở (PRD §10) — nơi sẽ trả lời

| # | Câu hỏi | Trả lời ở |
|---|---|---|
| Q1 | Định dạng file nộp chính xác của BTC? | P3.5 (chờ BTC công bố) |
| ~~Q2~~ | ~~Dữ liệu AIC có bao nhiêu video / giờ?~~ | ✅ **đã trả lời 2026-07-29:** 873 video · 130,7 giờ · 123 GB |
| Q3 | Có kênh audio / phụ đề không? | P4.5 — bản tin có lời dẫn nên gần như chắc chắn có audio; còn phải xác nhận chất lượng |
| Q4 | `embedding_dim` thực tế của `beit3_base_itc`? | **P0.5** — ghi lại ngay khi load checkpoint |
| Q5 | Lúc thi có mạng ổn định không? | P3 (quyết định QUEST/VQA có dùng được không) |
| **Q6** | **Số keyframe thật do TransNetV2 + §F1 sinh ra?** Quyết định lại ngân sách R2 / đĩa / thời gian ingest | **P0.7** — đo trên pilot rồi ngoại suy |
| **Q7** | Hệ thống nộp bài BTC mở khi nào, cho nộp thử mấy lần? | P3 — quyết định §6.9 có khả thi không |
| ~~Q8a~~ | ~~`clip-features-32` đã L2-normalize chưa?~~ | ✅ **đã trả lời 2026-07-29: rồi** (norm ≈ 1,0000) |
| **Q8b** | `clip-features-32` dùng checkpoint CLIP nào? | P0 — chỉ cần nếu muốn dùng feature BTC làm đối chứng số học |

---

## 8. Điểm lệch có chủ ý so với PRD — **cần chủ dự án phán quyết**

| Lệch | PRD nói | Plan đề xuất | Lý do |
|---|---|---|---|
| ~~Vị trí nhánh OCR~~ | F1 (P0) gồm cả bước 3 "OCR bằng Gemini API" | ~~Đẩy nhánh TEXT sang P1~~ | ✅ **ĐÃ ĐÓNG 2026-07-29 — chủ dự án quyết kéo OCR về P0.** Plan nay **khớp PRD**, không còn lệch |

**Không còn điểm lệch nào với PRD.**

### Vì sao quyết định này đúng

Điểm lệch trên sinh ra từ hai lý do, và **cả hai đều đã mất hiệu lực**:

- **R4 (YouCook2 không có chữ)** → đóng. Video tin tức đầy chữ trên màn hình, OCR ở P0 đo được ngay.
- **R2 (chi phí)** → vẫn còn, nhưng ở quy mô pilot là khoản chấp nhận được: **~13,4 triệu token** ở 1280×720, hoặc **~6,7 triệu** nếu hạ về ≤ 384×384.

Đổi lại, P0 nhận được ba thứ mà bản trước phải chờ tới P1/P4:

1. **Kênh đối chứng thứ hai cho `eval/run.py`.** Với nhãn tự sinh, một kênh đơn độc rất khó biết đúng hay sai; hai kênh độc lập bất đồng với nhau ở đâu là chỉ dấu điều tra.
2. **Phát hiện sớm bệnh rò rỉ nhãn.** Gemini sinh caption *từ chính keyframe đáp án*; nếu nó chép lại chữ trên màn hình thì kênh OCR sẽ khớp gần như hoàn hảo. Có OCR ở P0 mới thấy được bệnh này **trước** khi xây tiếp ba phase lên trên một bộ nhãn hỏng.
3. **`vi_analyzer` kiểm chứng thực chiến ngay từ P0**, thay vì mang một thành phần chưa từng chạy thật đi suốt bốn phase.

**Vẫn giữ ở P1:** hybrid fusion RRF (F4, task P1.3). Nó cần **cả hai** kênh đã ổn định và đã có số đo riêng lẻ thì so sánh mới có nghĩa — mà đó chính là thứ P0.9 sẽ cung cấp. P1.3 nhờ vậy rẻ hơn trước.

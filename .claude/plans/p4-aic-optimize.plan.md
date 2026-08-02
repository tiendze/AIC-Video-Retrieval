# P4 — Mở rộng toàn kho & tối ưu

> **Quay lại:** [`roadmap.plan.md`](./roadmap.plan.md) · **Trước:** [`p3-vqa-submission.plan.md`](./p3-vqa-submission.plan.md)
> **Trạng thái:** ⬜ chưa bắt đầu · **Điều kiện vào:** P3 xong

> **Viết lại 2026-07-29 — đổi trục phase.** Bản trước giả định P0–P3 chạy trên YouCook2 và P4 mới là lúc "nạp dữ liệu AIC". Dữ liệu AIC nay đã có từ P0, nên P4 **không còn là bước đổi dataset** mà là bước **mở rộng quy mô**: từ pilot 128 video lên đủ **873 video**.
>
> **Ba task cũ đã hoàn thành sớm hoặc mất lý do tồn tại:** `AICAdapter` (làm ở P0.3) · "kích hoạt `vi_analyzer` thực chiến" (làm ở **P0.7**, R4 đóng từ P0) · "collection AIC tách khỏi collection YouCook2" (không còn collection YouCook2 nào).

**Exit criteria (PRD §4):** toàn bộ metrics chạy lại trên đủ 873 video; có báo cáo so sánh pilot ↔ full.

**Phase này là bài kiểm tra thật của toàn bộ kiến trúc** — nhưng theo một trục khác bản trước: không phải *"đổi dataset có tốn công không"* mà **"hệ thống có chịu được gấp 7 lần dữ liệu không"**.

---

## P4.1 — Ingest nốt 745 video còn lại

**Làm gì:** mở rộng từ pilot `L21`–`L24` (128 video) lên đủ **873 video / 130,7 giờ**.

- Chạy bằng đúng `AIC2026Adapter` với `split="all"` — **không thêm class mới, không sửa pipeline**. Đây chính là phép thử T3 mà bản trước định làm ở P4.1.
- Chia shard theo `batch` (`L21` … `L30`), chạy song song nhiều tiến trình; checkpoint theo `video_id` qua `ingest_status` (R7).
- ⚠️ **Ước lượng lại trước khi chạy.** Số keyframe thật đo được ở P0.7 (Q6) × ~7 = quy mô full. Nếu vượt ngân sách đĩa/RAM ở PRD §6.6 thì phải xử lý **trước**, không phải sau khi chạy 20 tiếng.

**Verify:**
- **Ingest full chạy bằng đúng `pipelines/ingest.py` cũ, không sửa dòng nào.** Phải sửa ⇒ ghi rõ đã sửa gì và vì sao Protocol chưa đủ.
- `test_all_split_yields_873_unique_video_ids` — khử trùng 88 file `.mp4` lặp id.
- Kill giữa chừng → chạy lại tiếp đúng chỗ, không bản ghi trùng.

---

## P4.2 — Mở rộng bộ eval theo quy mô mới

**Làm gì:** bộ nhãn `aic_eval_v1` chỉ lấy mẫu từ pilot. Kho lớn gấp 7 lần thì bộ eval cũ không còn đại diện.

- Sinh `aic_eval_v2` lấy mẫu **phân tầng trên cả 873 video** (theo `batch` + `author`), cùng giao thức PRD §6.1.
- **Giữ nguyên `aic_eval_v1`** và chạy lại nó trên index mới — đây là cách duy nhất tách bạch *"số đổi vì kho to hơn"* khỏi *"số đổi vì bộ nhãn khác"*.
- Đo lại chất lượng bộ nhãn (PRD §6.1.3) trên v2.

**⚠️ Điều cần lường trước:** Recall **gần như chắc chắn giảm** khi kho tăng từ 128 lên 873 video — nhiều keyframe nhiễu hơn để lẫn vào. Đó là **kết quả bình thường**, không phải hồi quy. Cái cần nhìn là *mức giảm*, và so nó với mức giảm của các baseline (random, BM25, CLIP) trong cùng điều kiện.

---

## P4.3 — Collection mới, không ghi đè

**Làm gì:** giữ bất biến quan trọng nhất của vòng đời index (architecture.md §9.3).

- Đổi embedding model ⇒ **collection Milvus mới**, không ghi đè. Vector của hai model khác nhau **không so sánh được**, và lỗi này không crash — chỉ làm điểm số tệ đi khó truy vết.
- Index pilot và index full nên là **hai collection tách biệt**, không phải một collection được ghi thêm — để còn chạy lại được số đo P0–P3 mà đối chứng.

**Verify:** `test_collection_name_contains_model_id`; liệt kê collection trong Milvus thấy tách bạch pilot / full.

---

## P4.4 — Chạy lại toàn bộ metrics + báo cáo so sánh

**Làm gì:** đo lại **tất cả** ở quy mô đầy đủ — retrieval (§6.3), TRAKE (§6.4), VQA (§6.5), hệ thống (§6.6).

Báo cáo so sánh **pilot (128 video) ↔ full (873 video)**, kèm cảnh báo **R9** ở dạng đã cập nhật:

> Số đo là **tương đối** — dùng để so sánh giữa các cấu hình, **không** phải dự đoán điểm thi. Lý do nay không còn là lệch miền dữ liệu (miền đã đúng), mà là **nhãn tự sinh khác truy vấn thật của BTC**: caption do Gemini sinh sạch và trực quan hơn.

Những chỗ cần chú ý khi diễn giải:
- **Tách hai nguồn biến động**: kho to hơn (P4.1) và bộ nhãn khác (P4.2). Chạy `aic_eval_v1` trên index full là mấu chốt để tách.
- Latency ở §6.6 phải đo lại — ANN search trên index gấp 7 lần không còn cùng ngân sách thời gian.
- Ưu tiên giữ các cải tiến mang tính **kiến trúc**; nghi ngờ các cải tiến chỉ là tinh chỉnh siêu tham số trên pilot.

**Verify:** bảng so sánh đầy đủ, cùng format báo cáo, kèm git SHA / checkpoint / version index / **`eval_set_version` + hash** / **số `video_id` duy nhất đánh giá được** / số truy vấn / seed cho **cả hai** lần chạy.

---

## P4.4′ — Hiệu chỉnh với hệ thống nộp bài của BTC (PRD §6.9)

**Làm gì:** đây là lần duy nhất dự án chạm được vào **nhãn thật**. Không làm bước này thì mọi con số ở trên vẫn chỉ là hệ thống tự chấm chính nó.

1. Chạy hệ thống trên bộ truy vấn công khai của BTC, nộp, ghi lại đúng/sai.
2. Chạy **cùng cấu hình đó** trên bộ eval tự sinh.
3. Báo cáo **độ chênh**. Chênh lớn ⇒ bộ nhãn tự sinh đang đo sai thứ cần đo ⇒ sửa giao thức PRD §6.1, **không** sửa hệ thống cho khớp.
4. Mọi kết luận "cải thiện" rút ra chỉ từ nhãn tự sinh mà **không sống sót** qua bước 3 → **phải rút lại**.

**Phụ thuộc Q7** (hệ thống nộp bài mở khi nào, cho nộp thử mấy lần). Nếu không mở kịp, ghi rõ trong báo cáo rằng **toàn bộ số đo của dự án chưa từng được đối chiếu với nhãn thật** — đó là một giới hạn phải nói ra, không phải chi tiết bỏ qua được.

---

## P4.5 — Backlog tối ưu

**Thứ tự ưu tiên** (`references.md` §8) — mỗi hạng mục chỉ được coi là "cải thiện" nếu dẫn được về một con số trong PRD §6:

| # | Tối ưu | Khi nào làm | Kỳ vọng |
|---|---|---|---|
| 1 | **Kênh ASR** (`faster-whisper` / `WhisperX`) | Nếu Q3 xác nhận có audio | Mở ra **kênh truy hồi hoàn toàn mới** từ lời thoại — giá trị cao nhất |
| 2 | **Reranking** (cross-encoder / ColBERT-style) | Khi Recall@100 cao nhưng Recall@1 thấp — đáp án có trong tầm với, chỉ xếp hạng sai | Cải thiện MRR / Recall@1 |
| 3 | **Nén vector** (PQ / scalar quantization) | Khi index không vừa RAM | Giảm RAM, đánh đổi recall |
| 4 | **Tăng tốc inference** (ONNX Runtime / TensorRT) | Khi ingest là nút thắt | Giảm thời gian ingest |
| — | OCR cục bộ (PaddleOCR / EasyOCR) | Nếu chi phí Gemini OCR trên dữ liệu AIC vượt ngân sách | Cắt chi phí R2, đánh đổi chất lượng |
| — | Encoder đa ngôn ngữ (`jina-clip-v2`) | Nếu truy vấn tiếng Việt kém rõ rệt so với tiếng Anh | Có thể hợp hơn với AIC |

**Nguyên tắc:** làm **một** tối ưu, đo, báo cáo, rồi mới làm cái tiếp theo. Gộp nhiều thay đổi vào một lần đo là mất khả năng quy trách nhiệm cho từng thay đổi.

---

## Chuẩn bị cho ngày thi

Ngoài tối ưu số đo, phase này phải chuẩn bị **chế độ vận hành thật**:

- Đo `Time-To-First-Correct` / `Submissions-To-Correct` / `Success Rate @ 300s` bằng **phiên thao tác thật có bấm giờ**, ≥ 20 truy vấn, ≥ 2 người vận hành (PRD §6.6). Đây là thứ gần nhất với điểm số thật.
- Diễn tập **degraded mode**: mất mạng lúc thi ⇒ semantic search + DANTE vẫn đủ dùng; QUEST / OCR / VQA ngừng (Q5).
- Chốt format file nộp khi BTC công bố (Q1) — chỉ sửa formatter.

---

## Định nghĩa "xong" của P4

- [ ] Ingest đủ **873 video** chạy bằng `pipelines/ingest.py` cũ, **không sửa dòng nào** (hoặc ghi rõ đã phải sửa gì và vì sao Protocol chưa đủ)
- [ ] `aic_eval_v2` sinh xong, có version + hash + số đo chất lượng nhãn (PRD §6.1.3)
- [ ] `aic_eval_v1` chạy lại trên index full — để tách "kho to hơn" khỏi "nhãn khác"
- [ ] Collection pilot và collection full tách bạch trong Milvus
- [ ] Toàn bộ metrics chạy lại ở quy mô 873 video, **latency đo lại**
- [ ] Báo cáo so sánh pilot ↔ full kèm cảnh báo R9
- [ ] **P4.4′ đã chạy** — hoặc ghi rõ chưa đối chiếu được với nhãn thật (Q7)
- [ ] Q3, Q6 đã trả lời
- [ ] Có ít nhất một phiên đo TTFC thật
- [ ] Báo cáo phase theo template CLAUDE.md §8

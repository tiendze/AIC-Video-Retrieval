# Đo đạc, kỷ luật eval & báo cáo phase

> Tách từ `CLAUDE.md` §8. Quy tắc TDD chung (RED→GREEN→refactor, coverage ≥ 80%, tên test mô tả hành vi) nằm ở `.claude/rules/ecc/common/testing.md` · `python/testing.md`. Dưới đây là phần riêng của dự án.

**Biến task thành tiêu chí verify được, rồi lặp cho tới khi đạt** — *"Fix bug DANTE"* → *"viết test tái hiện chuỗi bị đảo thứ tự, rồi làm nó pass"*.

- **Không sang phase sau khi phase trước chưa có số đo** — bài học trực tiếp từ khoảng trống của paper.
- Metric chính: `Recall@K` / `MRR` (báo cáo **cả** video-level và segment-level) · `Sequence Exact-Match` + `Kendall's tau` (TRAKE) · `Grounded Accuracy` + `Hallucination Rate` (VQA). Mục tiêu: PRD §6.7.

## Kỷ luật riêng vì nhãn là nhãn tự sinh

- **Không sửa bộ eval để cải thiện số.** Đổi bộ eval ⇒ tăng version + chạy lại toàn bộ baseline. Ranh giới giữa *"sửa nhãn sai"* và *"sửa nhãn cho số đẹp"* rất mỏng ⇒ lý do sửa phải viết ra **trước** khi nhìn số mới.
- Báo cáo metric luôn kèm: git SHA · model checkpoint · version index · **`eval_set_version` + hash bộ nhãn** · **số `video_id` duy nhất đánh giá được** · số truy vấn · seed.
- **BM25 cao bất thường ⇒ nghi bộ nhãn rò rỉ, không phải hệ thống giỏi** (caption sinh từ chính keyframe đáp án). Điều tra trước khi ăn mừng.

## Bắt buộc: báo cáo sau mỗi phase

Kết thúc một phase (P0–P4) hoặc mỗi bước lớn trong plan — **dừng lại giải thích trước khi đi tiếp**, viết bằng tiếng Việt, đủ hiểu mà không cần mở code:

```
Đã làm gì   — module/file nào được tạo hoặc sửa, và vì sao cần chúng
Input       — dữ liệu / config / tham số đi vào, lấy từ đâu, định dạng ra sao
Output      — sinh ra cái gì (file · collection · index · metrics), nằm ở đâu
Cách verify — lệnh hoặc test chứng minh nó chạy đúng, kèm kết quả thực tế
Còn thiếu   — phần chưa xong, giả định đang treo, rủi ro đã phát hiện
```

Giải thích **luồng dữ liệu**, không liệt kê diff. Số liệu phải là **kết quả chạy thật** — chưa chạy được thì ghi "chưa đo", không suy đoán.

> Build / run commands nằm ở `README.md`, **không** đưa vào tài liệu này.

## Liên quan

- Vì sao nhãn là nhãn tự sinh (điều 5): [`context-and-scoring.md`](context-and-scoring.md)
- `eval/run.py` là entry point duy nhất: [`code-architecture.md`](code-architecture.md)
- "Thiếu dữ liệu phải fail rõ ràng, không trả rỗng": [`error-handling.md`](error-handling.md)

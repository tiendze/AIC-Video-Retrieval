# Cách làm việc & khi không chắc

> Gộp từ `CLAUDE.md` §2 (Think Before Coding) + §10 (Surgical Changes & Khi không chắc) — cả hai đều là quy tắc hành vi của agent, không phải quy tắc code.

## Think Before Coding

**Không đoán. Không giấu chỗ mơ hồ. Nêu tradeoff.**

- Nói rõ assumption trước khi implement; nhiều cách hiểu → trình bày cả hai, đừng im lặng chọn một.
- Có cách đơn giản hơn → nói ra. Được phép phản biện.
- Code lệch docs → sửa docs trong **cùng** change.
- Mỗi thay đổi ảnh hưởng kết quả phải trả lời được: *"đo bằng metric nào trong PRD §6?"*

## Surgical Changes

- **Lệnh tốn tài nguyên (Docker `build`/`up`/`down`; train/inference model; ingest hàng loạt) → chuẩn bị script, để chủ dự án tự chạy.** Đưa lệnh cụ thể, không tự ý thực thi.
- Chỉ đụng thứ cần đụng. Không "cải thiện" code/comment/format lân cận. Dead code có sẵn thì **báo**, đừng tự xoá.
- Commit atomic, **không commit khi test đỏ** (format `<type>: <description>` xem `.claude/rules/ecc/common/git-workflow.md`).

## Khi không chắc

- Không chắc **con số** → `[Cần điền]` / `⚠️ [Cần đo]`, đừng bịa. Câu hỏi treo: PRD §10.
- Không chắc **API thư viện** → fetch docs / `llms.txt`.
- Không chắc **quyết định sản phẩm** (format file nộp, giao thức sinh nhãn, thêm dependency lớn) → hỏi.
- Không chắc paper nói gì → `docs/architecture.md` §10 phân biệt 📄 **[Paper]** với 🔧 **[Giả định]**. Sửa 🔧 thoải mái; sửa 📄 phải nêu lý do.

## Liên quan

- Format báo cáo bắt buộc sau mỗi phase: [`evaluation-and-reporting.md`](evaluation-and-reporting.md)
- Thêm thư viện UI phải hỏi trước: [`frontend.md`](frontend.md)
- Cập nhật `docs/references.md` khi fetch API mới: [`libraries-api.md`](libraries-api.md)

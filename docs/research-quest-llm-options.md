# Khảo sát: LLM nào cho QUEST nhánh 1 khi không có kinh phí OpenAI

**Ngày:** 2026-08-14 · **Bối cảnh:** chuẩn bị P2.5 (`.claude/plans/p2-quest-dante.plan.md`) · **Người hỏi:** chủ dự án

**Câu hỏi:** plan P2.5 nói nhánh 1 dùng OpenAI (`client.responses.parse`), nhưng không có kinh phí mua token. Có phương án thay thế nào vẫn tốt không?

> **Cập nhật 2026-08-14 (bản 2):** chủ dự án phản hồi **free tier `gemini-3.6-flash` hết hạn mức rất nhanh, dùng một tí là hết** — đây là kinh nghiệm thực tế đã va phải khi chạy OCR/sinh nhãn. Phương án A (Gemini) vì vậy **bị hạ khỏi khuyến nghị**; bản 1 của tài liệu này đã sai khi cho rằng chi phí thấp là đủ để chốt. Xem mục 3 và 5 đã viết lại.

---

## 1. Chi phí QUEST vẫn nhỏ — nhưng "nhỏ" không giải quyết được "hết quota"

Giữ lại phân tích này vì nó vẫn đúng và vẫn hữu ích để hiểu bài toán:

| | Nhánh OCR / sinh nhãn (nơi chi phí thật nằm) | QUEST nhánh 1 |
|---|---|---|
| Input | **Ảnh**, hàng trăm nghìn keyframe | **Chữ**, một câu truy vấn ngắn |
| Số lần gọi | Hàng loạt, chạy batch | **1 lần / 1 truy vấn người dùng gõ** |
| Khi nào chạy | Lúc ingest (một lần, rất lớn) | Lúc thi, tương tác, người vận hành bấm |

Theo bảng giá `gemini-3.6-flash` trong [`references.md` §5](./references.md) ($1,50 / $7,50 mỗi 1M token in/out): 1 lần gọi ≈ **$0,001**, 1 000 lần ≈ **~$1**. ⚠️ **[Cần verify]** con số 200 token vào / 100 token ra là giả định của tôi, chưa đo.

**Nhưng đây là chi phí TRẢ TIỀN.** Vấn đề của chủ dự án là **không có thẻ/ngân sách để trả**, nên phải sống trong free tier — mà free tier thì tính theo **số request/phút và số request/ngày**, không tính theo độ rẻ của token. QUEST rẻ đến mấy cũng vẫn bị chặn khi hết lượt gọi trong ngày.

⇒ **Bài toán thật không phải "rẻ hay đắt", mà là "có phụ thuộc quota của người khác hay không".**

---

## 2. Điều kiện thực tế trong repo (đã kiểm chứng, không phải suy đoán)

| Sự thật | Nguồn |
|---|---|
| **PRD §F5 đã cho phép Gemini rồi** — nguyên văn: "Đưa `q₀` qua LLM (**GPT-4o-mini / Gemini**)" | `docs/PRD.md` §F5 |
| Plan P2.5 thu hẹp lại thành OpenAI — đây là **plan tự thu hẹp**, không phải ràng buộc từ PRD | `.claude/plans/p2-quest-dante.plan.md` §P2.5 |
| **OpenAI hiện KHÔNG được dùng ở bất kỳ đâu trong code** — chỉ tồn tại như 1 field bắt buộc | `grep openai *.py`: chỉ có `app/config.py` + test |
| ⚠️ **`openai_api_key` đang là field BẮT BUỘC của `Settings`** → thiếu nó là app **không khởi động được** | `app/config.py:30`, `tests/test_config.py:28` |
| Dự án **đã chạy model HF cục bộ trên GPU** rồi (`LocalNmtTranslator` cho VI→EN) — pattern local **có sẵn**, không phải làm mới | `app/services/query_translator.py` |
| Ràng buộc cứng: **1 máy, RTX 3070 8GB VRAM** | CLAUDE.md §1 |
| Ngân sách latency QUEST: **≤ 3 s (p50) / ≤ 6 s (p95)** | `docs/PRD.md` §6 |
| QUEST **bắt buộc có fallback về search thường** khi lỗi/timeout — mất QUEST không giết hệ thống | `docs/PRD.md` §F5 |

---

## 3. Các phương án (đã viết lại sau phản hồi về quota)

### ❌ Phương án A — Gemini free tier — **loại**

Chủ dự án đã va thực tế: hết hạn mức quá nhanh. Không xây một tính năng dùng lúc thi trên một quota có thể cạn giữa chừng. Giữ lại làm đường lui nếu có key trả phí sau này.

### ⭐ Phương án B — LLM chạy cục bộ — **khuyến nghị mới**

Đây là phương án duy nhất **không phụ thuộc quota của ai cả**, và dự án đã có sẵn pattern chạy model HF trên GPU (`LocalNmtTranslator`).

**Quan trọng — chọn đúng cỡ model, đừng mặc định 7B:**

| Cỡ | Ví dụ | ⚠️ VRAM ước tính **[Cần đo]** | Ghi chú |
|---|---|---|---|
| 1,5B | Qwen2.5-1.5B-Instruct | ~1 GB (4-bit) / ~3 GB (fp16) | Đủ cho việc mở rộng truy vấn đơn giản |
| 3B | Qwen2.5-3B-Instruct | ~2 GB (4-bit) / ~6 GB (fp16) | Cân bằng tốt nhất, nhiều khả năng là lựa chọn đúng |
| 7B | Qwen2.5-7B-Instruct | ~5 GB (4-bit) | Tranh VRAM gay gắt với BEiT-3 + NMT đang chạy |

⚠️ **[Cần đo]** — mọi con số VRAM/latency ở trên là ước lượng, **tôi chưa đo trên máy bạn**. Máy đang phải nạp đồng thời: BEiT-3 + `LocalNmtTranslator` (+ model QUEST mới). Phải đo thật trước khi chốt cỡ model.

**Việc QUEST nhánh 1 cần làm thực ra rất nhẹ:** nhận `q₀` → viết lại thành mô tả trực quan chi tiết hơn. Đây là *query expansion*, không phải suy luận phức tạp — nên model nhỏ có cửa. Cộng thêm: đã có sẵn timeout + fallback bắt buộc (PRD §F5) nên model chậm cũng không treo UI.

**Lợi ích phụ đáng kể:** chạy cục bộ ⇒ **không cần mạng** ⇒ trả lời luôn được PRD §10 Q5 ("lúc thi có mạng ổn định không") và khớp yêu cầu degraded mode của CLAUDE.md §4.

### ⚠️ Phương án C — free tier của nhà cung cấp khác

Ứng viên có thể tra: Groq · OpenRouter (free models) · Cloudflare Workers AI · Together AI · HuggingFace Inference API.

⚠️ **[Cần verify]** — tôi **không khẳng định** điều khoản free tier hiện tại của bất kỳ cái nào; phải tự kiểm tra lúc implement. **Rủi ro cốt lõi vẫn nguyên**: đã là free tier của người khác thì vẫn có thể cạn đúng ngày thi, đúng như đã xảy ra với Gemini. Chỉ nên dùng làm **kênh phụ**, không phải kênh chính.

---

## 4. ⚠️ Sự thật khó chịu: model nhỏ cục bộ yếu ở ĐÚNG việc QUEST sinh ra để làm

Đây là điểm tôi phải nói thẳng, kể cả khi nó làm phương án khuyến nghị kém hấp dẫn đi.

QUEST nhánh 1 tồn tại để xử lý **Out-of-Knowledge** — truy vấn chứa thực thể **model embedding chưa từng thấy** (PRD §F5 lấy ví dụ "Labubu", "Babythree"). Nhưng:

- Nếu LLM viết lại truy vấn **cũng không biết** "Labubu" là gì, nó chỉ có thể **bịa** hoặc trả về mô tả vô dụng.
- Model nhỏ (1,5B–3B), cutoff cũ hơn ⇒ **càng ít khả năng biết** thực thể mới so với GPT-4o-mini/Gemini.
- Nghĩa là: đúng những truy vấn OOK khó nhất — nơi QUEST đáng lẽ toả sáng — lại là nơi model cục bộ nhỏ hụt hơi nhất.

**Hệ quả về thứ tự ưu tiên:** **nhánh 2 (tìm ảnh mẫu trên web) mới là nửa mạnh hơn của QUEST** cho bài toán OOK — nó đi ra web, nơi có thông tin cập nhật, thay vì dựa vào trí nhớ đóng băng của một LLM. Nhánh 2 **không cần LLM chút nào**, chỉ cần image search + để người vận hành tự chọn `I*` (đúng như PRD §F5 và CLAUDE.md §4 đã yêu cầu: *không* auto-pick).

⚠️ **[Cần verify]** nhánh 2 cần API image search nào miễn phí được — chưa khảo sát, là câu hỏi riêng.

---

## 5. Khuyến nghị (đã viết lại)

1. **Nhánh 1 → LLM cục bộ cỡ nhỏ (Qwen2.5-3B-Instruct là điểm khởi đầu hợp lý)**, không dùng API trả phí, không dựa free tier. ⚠️ Phải **đo VRAM + latency thật** trên 3070 (cùng lúc với BEiT-3 + NMT đang nạp) trước khi chốt cỡ model.
2. **Cân nhắc đảo thứ tự: làm nhánh 2 trước nhánh 1.** Nhánh 2 giải OOK tốt hơn về bản chất và **không cần LLM**, nên không vướng vào đúng cái ràng buộc kinh phí đang bàn.
3. **Sửa `app/config.py`: `openai_api_key` bắt buộc → optional** (`SecretStr | None = None`). Hiện thiếu key này là app **không boot được**, dù OpenAI không dùng ở đâu. Việc này phải làm dù chọn phương án nào.
4. **Cập nhật `.claude/plans/p2-quest-dante.plan.md` §P2.5**: bỏ "OpenAI `responses.parse`" + `test_quest_handles_openai_refusal_branch`, thay bằng contract của model cục bộ, kèm lý do (ràng buộc kinh phí, không phải sở thích kỹ thuật).

---

## 6. Câu hỏi lớn hơn cần chủ dự án cân nhắc — **P2.5 có đáng làm ngay bây giờ không?**

Nêu ra vì CLAUDE.md §2 cho phép phản biện, và mấy dữ kiện sau xếp cạnh nhau thì đáng suy nghĩ:

- **Không đo được:** bộ eval hiện tại là caption tin tức tiếng Việt, **gần như không có thực thể OOK nào** ⇒ không có cách nào chứng minh QUEST cải thiện được gì (CLAUDE.md §8: "không sang phase sau khi phase trước chưa có số đo").
- **Bị ràng buộc kinh phí:** đúng nửa cần LLM lại là nửa đang thiếu ngân sách.
- **Trong khi đó P3** (VQA + **định dạng file nộp**) đụng thẳng vào thứ ăn điểm — CLAUDE.md §1: "thứ hạng mới là thứ ăn điểm". Sai định dạng nộp là mất điểm im lặng toàn bộ.
- P2 đã đạt cả hai chỉ tiêu chính (Sequence Exact-Match, Kendall's tau — xem plan §P2.3/§P2.4), QUEST là hạng mục còn lại **không có exit criteria định lượng**.

**Không phải đề nghị bỏ QUEST** — mà là: có thể **P3 nên đi trước**, rồi quay lại P2.5 khi đã chốt được cách đo (hoặc chấp nhận rõ ràng rằng QUEST là tính năng "hỗ trợ người vận hành", không kèm số).

---

## 7. Câu hỏi còn treo (cần chủ dự án quyết)

- [ ] Chốt phương án B (LLM cục bộ) hay tạm hoãn nhánh 1?
- [ ] Có đồng ý làm **nhánh 2 trước nhánh 1** không?
- [ ] Có đồng ý đổi `openai_api_key` thành optional không — hay giữ lại vì P3 (VQA) có thể vẫn cần?
- [ ] **P2.5 làm ngay, hay nhảy sang P3 trước rồi quay lại?** (xem mục 6)
- [ ] Nếu chọn LLM cục bộ: có chấp nhận thêm ~2–6 GB VRAM vào budget 8GB không, hay muốn tôi khảo sát riêng phương án chỉ nạp model khi cần rồi giải phóng?

# P3 — VQA & nộp bài

> **Quay lại:** [`roadmap.plan.md`](./roadmap.plan.md) · **Trước:** [`p2-quest-dante.plan.md`](./p2-quest-dante.plan.md) · **Tiếp theo:** [`p4-aic-optimize.plan.md`](./p4-aic-optimize.plan.md)
> **Trạng thái:** ⬜ chưa bắt đầu · **Điều kiện vào:** P2 đã có bảng ablation `λ`

**Exit criteria (PRD §4):** `Grounded Accuracy` đạt mục tiêu §6.7; file nộp qua validate.

**Tư tưởng của phase này:** một hệ thống VQA trả lời đúng nhờ *kiến thức nền của LLM* (chứ không nhờ nhìn video) là **vô dụng** trong cuộc thi — BTC yêu cầu chỉ ra bằng chứng. Vì vậy mọi thiết kế ở đây xoay quanh việc buộc câu trả lời phải dẫn được về keyframe thật.

---

## P3.1 — `vqa_service.py` ✅ hoàn thành (2026-08-16)

**Làm gì:** trả lời câu hỏi dựa trên keyframe **thực sự truy hồi được** (PRD §F7 · architecture.md §3.6).

- Prompt **chỉ chứa top-k keyframe đã truy hồi** — không để model trả lời từ kiến thức nền.
- Gemini: `client.interactions.create(...)` với ảnh inline hoặc File API; **structured output** qua `model_json_schema()`.
- Schema đầu ra: `{answer, confidence, evidence_keyframe_ids[], reasoning}`.
- Model **phải được phép** trả lời "không đủ căn cứ" — đây là cơ chế chống bịa, đo bằng `Hallucination Rate`.
- Timeout 10s + retry exponential backoff.

**Verify:**
- `test_vqa_prompt_contains_only_retrieved_keyframes`
- `test_vqa_output_matches_json_schema`
- `test_vqa_can_return_insufficient_evidence`

---

## P3.2 — Grounding gate ✅ hoàn thành (2026-08-16)

**Làm gì:** chốt chặn giữa VLM và kết quả (architecture.md §3.6, bước "Kiểm tra evidence_ids").

- Sau khi Gemini trả lời: kiểm tra **mọi** `evidence_keyframe_ids` có nằm trong tập keyframe đã cấp cho model không. Nằm ngoài ⇒ câu trả lời **không đáng tin** ⇒ trả "không đủ căn cứ".
- Không có bước này thì `Grounded Accuracy` **không đo được** và `Hallucination Rate` không kiểm soát được.
- **Lỗi nhánh mạng không giết request chính:** Gemini lỗi / timeout ⇒ vẫn trả **grid keyframe** kèm thông báo "VQA tạm không khả dụng" (US-5 vẫn dùng được một phần).
- Hiển thị keyframe bằng chứng **cạnh** câu trả lời để người vận hành tự kiểm chứng.

**Verify:**
- `test_vqa_rejects_evidence_outside_retrieved_set`
- `test_vqa_failure_still_returns_keyframe_grid`
- Thực nghiệm: chặn mạng tới Gemini → `POST /api/qa` vẫn trả 200 kèm grid + thông báo

---

## P3.3 — Sinh bộ eval QA (R6) ✅ hoàn thành (2026-08-16)

**Trạng thái thật, đã verify bằng file thật trên đĩa (không suy đoán):**
`configs/eval/aic_qa_v1.json` (91 record) + `.json.sha256` +
`.quality_report.md` đã tồn tại, hash tự tính khớp file `.sha256`
(`df553d2...`). Số liệu từ `.quality_report.md`:

| | |
|---|---|
| eval_set_version | v1 |
| Tổng đã sinh (`generate`) | 346 |
| Đã review | 100 |
| Approved (đóng băng) | 91 |
| Tỉ lệ loại khi review | 9.0% |
| Tỉ lệ `answerable_by_ocr` (trong số approved) | 1.1% |

`eval/build_qa_eval_set.py` đã viết + test (35/35 pass, xem
`.claude/tdd/p3-3-build-qa-eval-set.tdd.md`).

**Quyết định khác với mô tả gốc bên dưới** (chốt qua `AskUserQuestion`
2026-08-16, không đoán):
- **Model sinh câu hỏi = Ollama local, KHÔNG phải OpenAI** — chủ dự án không có ngân sách API trả phí. `openai_api_key` vẫn còn trong `Settings` nhưng chưa từng được dùng ở đâu.
- **Chỉ đọc caption (text), KHÔNG nhìn ảnh keyframe** — giữ VRAM thấp, chạy tuần tự với BEiT-3 (CLAUDE.md §1.4). Đánh đổi: chất lượng câu hỏi phụ thuộc caption Gemini đã viết, có thể không đủ chi tiết cho câu hỏi kiểu "biển hiệu ghi gì".
- **Không sample lại** — dùng thẳng các record `approved: true` sẵn có trong `configs/eval/aic_eval_v1.candidates.json` (P0.4') làm shot universe, đúng tinh thần "cùng cách lấy mẫu, cùng seed" nhưng bằng cách tái dùng thay vì gọi lại `sample_kis_shots`/`sample_trake_sequences`.
- **ĐỔI model `qwen2.5:7b-instruct` → `qwen3:8b`** (cùng ngày, sau khi chạy `--limit-shots 5` thật): model cũ sinh một câu hỏi mà chính `reference_answer` của nó thừa nhận không trả lời được từ caption — ép JSON schema không ép được model tuân thủ system prompt "chỉ hỏi điều trả lời được". `qwen3:8b` chạy lại 346 shot thật, **0 lỗi kiểu đó** — xác nhận bằng chạy thật, không chỉ suy đoán.
- **Sửa lỗi gate đóng băng** (phát hiện khi chủ dự án chạy `freeze` thật lần đầu): code ban đầu bắt buộc review 100% số record đã sinh (346), trái với plan ("bộ chưa review chỉ dùng để dò lỗi"). Đã sửa để chỉ cần ≥100 record ĐÃ REVIEW, phần chưa review bị loại khỏi bộ đóng băng chứ không chặn `freeze` — commit `7dc4a71`.
- **Thêm subcommand `review`** (`eval/build_qa_eval_set.py review --limit N`) — vòng lặp terminal tự mở ảnh keyframe, hỏi y/n/skip/quit, ghi lại sau mỗi câu — thay cho việc chủ dự án phải tự sửa tay JSON.



**Làm gì:** dataset AIC **không có nhãn QA** — cũng như không có nhãn nào khác (PRD §1.4). Phải tự sinh (PRD §6.5).

> **Tận dụng hạ tầng có sẵn.** Bộ eval retrieval đã được sinh từ P0.4′ bằng đúng cơ chế này. P3.3 **mở rộng** `eval/build_eval_set.py` thêm nhánh QA, không dựng lại từ đầu — cùng cách lấy mẫu, cùng seed, cùng quy trình đóng băng.

1. Với mỗi shot đã có caption, dùng LLM sinh 1–2 câu hỏi **chỉ trả lời được bằng thông tin thị giác** trong shot đó (ví dụ *"Biển hiệu phía sau người dẫn ghi tên cơ quan nào?"*), kèm đáp án tham chiếu.
2. **Review thủ công ít nhất 100 mẫu** — loại bỏ câu hỏi trả lời được bằng kiến thức nền mà không cần nhìn video.
3. **Đóng băng** bộ đã review làm bộ đánh giá chính thức (có version + hash). Bộ chưa review chỉ dùng để dò lỗi.
4. Cố định seed lúc sinh.

**⚠️ Bẫy riêng của video tin tức — phải xử lý ở bước review.** Rất nhiều câu trả lời hiện thành **chữ trên màn hình**, nên câu hỏi kiểu *"tỉnh nào đang được nhắc tới?"* trả lời được bằng OCR mà không cần hiểu hình ảnh. Đó không phải lỗi, nhưng khi đọc `Grounded Accuracy` phải **tách được phần công của kênh OCR khỏi phần công của VLM**, nếu không sẽ quy nhầm công lao. Cách làm: gắn cờ `answerable_by_ocr` cho từng mẫu lúc review, rồi báo cáo metric **tách theo cờ này**.

**Ghi rõ trong mọi báo cáo:** đây là **nhãn tổng hợp**, không phải nhãn người. Model sinh câu hỏi phải **khác** model trả lời — nếu cả hai đều là Gemini thì `Grounded Accuracy` đang đo Gemini tự chấm chính nó.

**Verify:** file bộ QA có version + hash; số mẫu đã review ghi rõ; tỉ lệ `answerable_by_ocr` được báo cáo.

---

## P3.4 — `eval/metrics/vqa.py`

| Metric | Định nghĩa |
|---|---|
| Exact Match (EM) | Trùng khít sau chuẩn hoá (lowercase, bỏ dấu câu, bỏ mạo từ) |
| Token-F1 | F1 trên tập token — chấp nhận diễn đạt khác |
| **Grounded Accuracy** ⭐ | Trả lời đúng **VÀ** ≥ 1 keyframe bằng chứng nằm trong segment đúng. **Metric chính** |
| Retrieval-conditioned Accuracy | Chỉ tính trên câu mà retrieval đã đúng — tách lỗi retrieval khỏi lỗi suy luận của VLM |
| **Hallucination Rate** ⭐ | Tỉ lệ trả lời tự tin (`confidence ≥ ngưỡng`) trong khi **không** có keyframe đúng nào trong ngữ cảnh |
| Abstention Rate | Tỉ lệ trả "không đủ căn cứ" — đọc **kèm** Hallucination Rate |
| LLM-as-Judge | Gemini chấm rubric 1–5. **Chỉ tham khảo** — có thiên lệch tự đánh giá, không dùng để so sánh chính |

Ngưỡng `confidence` đặt trong `configs/`, không hard-code.

**Cách đọc:** cặp `Grounded Accuracy` ↑ / `Hallucination Rate` ↓ mới mô tả đúng chất lượng. Một mình Accuracy cao mà Hallucination cao là hệ thống đang đoán bừa.

**Verify:** chạy qua `eval/run.py`, ra bảng đủ 7 metric.

---

## P3.5 — Submission exporter (F8)

**Làm gì:** xuất file kết quả theo định dạng BTC.

⚠️ **Định dạng nộp chính xác chưa công bố** (Q1, R8) — trang hướng dẫn AIC ghi *"quy tắc tính điểm sẽ được thông báo riêng"*. Vì vậy thiết kế phải chịu được việc đoán sai:

- **Tách rời dữ liệu kết quả và bộ formatter.** Đổi format ⇒ chỉ sửa formatter, không đụng service.
- **Validator chạy trước khi ghi file.**
- Hỗ trợ đủ **3 dạng task của vòng sơ tuyển AIC 2026**: Textual KIS · QA · TRAKE. **Video KIS đã bị bỏ.**

> ✅ **Cập nhật 2026-08-01 — định dạng nộp đã công bố, Q1 của PRD §10 đã đóng.** Mỗi dòng: `<video_id>, <frame_id>` (KIS) · `<video_id>, <frame_id>, <answer>` (QA) · `<video_id>, <frame_id₁>, …, <frame_idₙ>` (TRAKE). **Tối đa 100 dòng mỗi truy vấn, thứ tự dòng chính là thứ hạng.** Điểm = `mean(R@k)`, k ∈ {1,5,20,50,100}. Chi tiết: `references.md` §1.2, PRD §F8.
>
> Ba việc phải làm đúng, đều là loại lỗi **không crash**:
> - `frame_id` lấy thẳng từ `frame_idx` đã lưu lúc ingest, **không** tính lại từ `timestamp × fps`.
> - Validator chặn khi > 100 dòng, và cảnh báo khi N frame của một dòng TRAKE **không cùng** `video_id` (sai video ⇒ mất trắng cả truy vấn).
> - Thứ tự dòng phải đúng thứ tự xếp hạng của hệ thống — đảo thứ tự là mất `R@1` mà không có dấu hiệu nào.
- `POST /api/submissions/export` (US-6).

**Verify:**
- `test_changing_format_only_touches_formatter` — thay formatter giả định, service không đổi
- `test_export_rejects_invalid_submission_before_writing`
- Xuất thử đủ 4 task với dữ liệu giả định

---

## Định nghĩa "xong" của P3

- [ ] `Grounded Accuracy` đạt mục tiêu PRD §6.7
- [ ] `Hallucination Rate` dưới ngưỡng §6.7
- [x] Bộ eval QA đã review ≥ 100 mẫu, đóng băng, có hash — `configs/eval/aic_qa_v1.json` (91 record đã duyệt/100 đã review/346 đã sinh), hash verify khớp `.sha256` (2026-08-16)
- [ ] Grounding gate chặn được evidence bịa (có test)
- [ ] Gemini chết → request chính vẫn sống (có test)
- [ ] Exporter đủ 4 task, có validator
- [ ] Q1 (format nộp) — ghi rõ trạng thái: đã có từ BTC hay vẫn đang giả định
- [ ] Q5 (mạng lúc thi) — nếu không có mạng ổn định thì ghi rõ QUEST/VQA là tính năng *có điều kiện*
- [ ] Báo cáo phase theo template CLAUDE.md §8

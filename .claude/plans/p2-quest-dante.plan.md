# P2 — Hai đóng góp của paper: QUEST & DANTE

> **Quay lại:** [`roadmap.plan.md`](./roadmap.plan.md) · **Trước:** [`p1-search.plan.md`](./p1-search.plan.md) · **Tiếp theo:** [`p3-vqa-submission.plan.md`](./p3-vqa-submission.plan.md)
> **Trạng thái:** ⬜ chưa bắt đầu · **Điều kiện vào:** P1 đã đạt mục tiêu Recall@10

**Exit criteria (PRD §4):** TRAKE `Sequence Exact-Match` đạt mục tiêu §6.7; **có bảng ablation `λ`**.

Đây là phase tái hiện hai đóng góp chính của paper baseline. **Giá trị riêng của dự án nằm ở P2.4** — paper đưa ra hai giá trị `λ` nhưng không có bảng ablation nào, nên không ai biết hình phạt thời gian thực sự đóng góp bao nhiêu.

**Điều kiện tiên quyết từ P0:** `global_index` liên tục + `s_v` / `e_v` trong MongoDB. Thiếu là DANTE không chạy được.

---

## P2.1 — `dante_service.py` ✅ hoàn thành (2026-08-14)

**Làm gì:** quy hoạch động tìm chuỗi N sự kiện đúng thứ tự trong một video.

**Công thức nguyên văn từ paper** (PRD §F6 · architecture.md §3.5):

```
(1)  S[i,t]      = cosine_similarity(u_i, E[t])
(2)  DP[i,t]     = S[i,t] + max_{τ ∈ [s_v, t−1]} ( DP[i−1,τ] − λ(t − τ) )
(3)  running_max = max( running_max, DP[i−1,t−1] + λ(t−1) )
(4)  DP[i,t]     = S[i,t] + running_max − λt
(5)  DANTE[v]    = max_{t ∈ [s_v, e_v]} DP[N,t]
```

Yêu cầu triển khai:
- Dùng bản **running max** — Eq (3)(4), đưa `O(N·T²)` → **`O(NT)`**. Mẹo: tách `λ(t−τ)` thành `λt − λτ`, đưa `λτ` vào trong max và `λt` ra ngoài.
- **Ràng buộc thứ tự `τ < t` là bất biến** — chuỗi trả về **không bao giờ** đảo thứ tự thời gian.
- **Có backtracking** để trả về đúng N keyframe, không chỉ điểm số.
- `λ` đọc từ `configs/`, **không hard-code**.
- DP chạy trên khoảng `[s_v, e_v]` của từng video, lấy từ MongoDB.

**Verify (TDD — viết RED trước):**
- `test_dante_never_returns_out_of_order_sequence` — dựng ma trận similarity cố tình dụ thuật toán chọn thứ tự đảo; kết quả vẫn phải tăng dần theo `global_index`
- `test_dante_returns_exactly_n_keyframes` — backtracking đủ N
- `test_dante_runtime_is_linear_in_t` — nhân đôi T ⇒ thời gian tăng ~2×, **không** ~4×. Đây là kiểm chứng thực nghiệm tuyên bố `O(NT)` mà paper không chứng minh bằng số
- `test_lambda_zero_reduces_to_pure_order_constraint`

**Kết quả thực tế:**
- Files: `app/services/dante_service.py` (`dante_search()`, `compute_similarity_matrix()`, `DanteResult` frozen dataclass), `configs/search/dante.yaml` (`lambda_penalty: 0.001`), `tests/test_dante_service.py`
- `.venv/Scripts/python.exe -m pytest tests/test_dante_service.py -v --cov=app.services.dante_service --cov-report=term-missing` → **18/18 passed**, coverage module **100%** (59/59 statements)
- Running-max DP (Eq 3–4) đạt O(NT), có backtracking, giữ bất biến `τ < t`; `λ` đọc từ config, không hard-code
- Bug N=0 (IndexError) phát hiện lúc code review đã fix ở commit `870434c` — nay có test riêng `test_raises_on_zero_events_instead_of_crashing`, raise `DanteServiceError` thay vì crash
- Regression toàn repo trước đó: 514 passed, 4 skipped, không có test nào đỏ

---

## P2.2 — `POST /api/trake` + bộ eval TRAKE ✅ hoàn thành (2026-08-14)

**Làm gì:** đưa DANTE ra API và dựng ground truth.

- Endpoint: body `events[]`, `lambda_penalty`, `top_k`. Trả top-k video, mỗi video kèm đúng N keyframe **đúng thứ tự thời gian** (US-4).
- **Bộ eval:** mỗi video val có ≥ 3 segment → lấy **N segment liên tiếp** làm chuỗi truy vấn, N ∈ {3, 5, 7}; caption của chúng là N sự kiện đầu vào. Nguồn nhãn: `order_index` của `GroundTruthSegment` (từ P0.3).
- Cố định seed khi lấy mẫu bộ eval.

**Verify:** gọi endpoint với một chuỗi thật từ val → trả về đúng cấu trúc, thứ tự không đảo.

**Kết quả thực tế:**
- **Bộ eval N-segment-liên-tiếp đã có sẵn từ P0.4** (`eval/build_eval_set.extract_trake_sequences(segments, n)`, dùng `order_index` của `GroundTruthSegment`) — P2.2 không phải viết lại phần này, chỉ cần endpoint.
- Files mới: `app/services/trake_service.py` (`TrakeService`, `TrakeVideoResult`, `TrakeKeyframe`), `app/api/trake.py` (router), `app/schemas/request|response/trake.py`, `configs/search/trake.yaml`. Sửa: `app/stores/milvus_client.py` (+`MilvusStore.get_vectors_for_video()` — `query()` filter chính xác trả về `embedding` thô, khác `search()` là ANN không trả `embedding`), `app/dependencies.py` (+`get_trake_service`), `app/main.py` (đăng ký router).
- **Thiết kế:** `TrakeService.search()` encode N câu mô tả sự kiện → duyệt **mọi** video `ingest_status == "complete"` (đúng "duyệt từng video v" trong architecture.md §3.5 — paper không lọc ứng viên trước) → lấy vector keyframe trong `[s_v, e_v]` từ Milvus → chạy `dante_search` → xếp hạng theo `DANTE[v]`, trả top-k.
- **Bug phát hiện qua TDD (RED thật, không phải giả định):** `dante_search` tái tạo `global_index = s_v + local_column_index`, giả định `[s_v, e_v]` liên tục tuyệt đối (đúng bất biến CLAUDE.md §3). Test đầu tiên dựng dữ liệu global_index thưa (0 và 9, thiếu 1–8) để mô phỏng "cách xa nhau" đã vi phạm chính bất biến đó, khiến service map nhầm keyframe (`KeyError`). Đã thêm `_is_dense_global_index_range()` guard: video có khoảng trống global_index bị bỏ qua + log cảnh báo, không crash — cùng lúc sửa lại test cho đúng bất biến (dùng frame đệm lấp đầy) thay vì né nó.
- Video không đủ N frame, có khoảng trống global_index, hoặc DANTE không tìm được chuỗi hữu hạn (điểm `-inf`) → bị bỏ qua + log `video_id`, không làm hỏng cả request (CLAUDE.md §7).
- `.venv/Scripts/python.exe -m pytest tests/test_trake_service.py tests/test_trake_api.py tests/test_milvus_client.py --cov=... --cov-report=term-missing` → **27/27 passed**, coverage **100%** trên `trake_service.py` (67/67 dòng), `app/api/trake.py`, `app/schemas/request/trake.py`, `app/schemas/response/trake.py`
- Toàn bộ suite: `534 passed, 3 skipped` (không regression so với baseline 514 trước P2.1+P2.2), `ruff check` + `black --check` sạch
- Git checkpoint theo TDD: `75a19d2` (RED — test), `472ceb5` (GREEN — implementation)

**Còn thiếu / rủi ro đã biết:**
- Chưa đo latency thật của `TrakeService.search()` khi duyệt toàn bộ corpus (873 video) — thiết kế "duyệt từng video" đúng theo paper nhưng chưa benchmark; TRAKE không bị ràng buộc latency chặt như KIS/QA (PRD §1.2) nên chấp nhận được ở P2.2, sẽ đo lại nếu P2.3/P2.4 cho thấy chậm.
- Chưa test integration thật với Milvus/MongoDB sống (chỉ Fake) — theo CLAUDE.md §10, việc chạy thật cần Docker do chủ dự án tự thực hiện.
- `eval/metrics/trake.py` (Sequence Exact-Match, Kendall's tau) là phạm vi P2.3, chưa làm.

---

## P2.3 — `eval/metrics/trake.py`

**Làm gì:** đo chất lượng TRAKE (PRD §6.4). Chạy qua **cùng một entry point** `eval/run.py`.

| Metric | Vì sao cần |
|---|---|
| **Sequence Exact-Match** ⭐ | Cả N keyframe đều hit **và** đúng thứ tự — đúng điều kiện chấm điểm của TRAKE |
| Per-event Recall@K | Chỉ ra sự kiện thứ mấy hay hỏng (thường là đầu/cuối chuỗi) |
| Video-level Recall@K | Tách lỗi "chọn sai video" khỏi lỗi "chọn sai frame trong video đúng" |
| **Kendall's tau** ⭐ | Đo riêng chất lượng *sắp thứ tự* — đây chính là thứ `λ` điều khiển |
| Mean Temporal Offset | Sai bao xa khi sai (giây) — lệch 2 giây khác hẳn lệch 2 phút |
| Thời gian chạy theo T | Kiểm chứng `O(NT)` |

**Verify:** `python -m eval.run --config configs/eval/trake_val.yaml` ra bảng đầy đủ, kèm git SHA / checkpoint / mẫu số / seed.

---

## P2.4 — ⭐ Ablation `λ` — phần paper không có

**Làm gì:** sweep bắt buộc (PRD §6.4).

```
λ ∈ {0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1}   ×   N ∈ {3, 5, 7}
```

Báo cáo `Sequence Exact-Match` và `Kendall's tau` cho từng ô.

**Trường hợp `λ = 0` là quan trọng nhất.** Nó trả lời câu hỏi: *hình phạt khoảng cách thời gian có thực sự đóng góp gì không?* Nếu `λ = 0` cho kết quả tương đương, thì phần cốt lõi của DANTE chỉ là **ràng buộc thứ tự**, không phải hình phạt khoảng cách — và **phải nói thẳng điều đó trong báo cáo**, kể cả khi nó mâu thuẫn với paper.

Paper đề xuất `λ = 0.001` (khoảng cách index 3–15) và `λ = 0.01` (khoảng cách 1–3) — đối chiếu xem số đo có ủng hộ không.

**Verify:** bảng ablation 7 × 3; `configs/` ghi lại giá trị `λ` được chọn cuối cùng kèm lý do.

---

## P2.5 — `quest_service.py` (2 nhánh)

**Làm gì:** xử lý truy vấn Out-of-Knowledge — truy vấn chứa thực thể mà model embedding chưa từng thấy (PRD §F5 · architecture.md §3.4).

**Nhánh 1 — viết lại truy vấn:**
- `q₀` → LLM sinh `q_r` mô tả trực quan, chi tiết hơn → search bằng encoder như bình thường.
- OpenAI: `client.responses.parse(..., text_format=Model)`. **Bắt buộc xử lý nhánh `item.type == "refusal"`** trước khi đọc `item.parsed`.
- **Hiển thị CẢ `q₀` LẪN `q_r`** — người vận hành phải thấy LLM đã làm gì (US-2).

**Nhánh 2 — tìm ảnh mẫu bên ngoài:**
- Tìm ảnh của thực thể lạ trên web → hiện danh sách ứng viên → **người dùng TỰ CHỌN** ảnh mẫu `I*` → image-to-image search.
- **Không auto-pick ảnh đầu tiên.** Người vận hành biết ngữ cảnh mà hệ thống không biết — ép tự động sẽ mất nhiều hơn được.

**Cả hai nhánh:**
- Timeout + retry exponential backoff → hết timeout thì **fallback về search thường bằng `q₀`**, **không treo UI**.
- Lỗi mạng ở nhánh phụ **không được giết request chính**.

**Verify:**
- `test_quest_falls_back_to_plain_search_on_timeout`
- `test_quest_handles_openai_refusal_branch`
- `test_quest_response_contains_both_original_and_rewritten_query`
- Thực nghiệm: ngắt mạng → UI vẫn trả kết quả search thường trong ngân sách latency

---

## Định nghĩa "xong" của P2

- [ ] `Sequence Exact-Match (N=3)` đạt mục tiêu PRD §6.7
- [ ] `Kendall's tau` đạt mục tiêu §6.7
- [ ] **Bảng ablation `λ` đầy đủ 7 × 3**, có kết luận rõ về `λ = 0`
- [ ] Chứng minh thực nghiệm `O(NT)` (đồ thị thời gian theo T)
- [ ] QUEST hai nhánh chạy được, có fallback, hiển thị `q₀` + `q_r`
- [ ] Coverage ≥ 80%
- [ ] Báo cáo phase theo template CLAUDE.md §8

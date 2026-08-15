# Bối cảnh & cách chấm điểm

> Tách từ `CLAUDE.md` §1. Đây là năm điều quyết định mọi lựa chọn kỹ thuật của dự án — đọc trước khi tranh luận về tradeoff nào đó.

## Năm điều quyết định mọi lựa chọn

1. **3 task vòng sơ tuyển AIC 2026:** `Textual KIS` · `QA (VQA)` · `TRAKE` (**không còn Video KIS**). Mỗi truy vấn nộp **tối đa 100 đáp án đã xếp hạng**, điểm = trung bình `R@k`, k ∈ {1, 5, 20, 50, 100} (`references.md` §1.2) ⇒ **thứ hạng là thứ ăn điểm**, latency chỉ còn là yêu cầu vận hành.
2. **Đáp án là `<video_id>, <frame_id>`** — `frame_id` là **chỉ số frame tuyệt đối**, không phải giây, không phải số thứ tự keyframe. TRAKE chấm theo cửa sổ thường **< 10 frame** ⇒ sai một phép quy đổi là mất điểm im lặng.
3. **Hệ thống interactive** — người chốt đáp án cuối. Đừng tự động hoá chỗ paper cố ý để người vận hành quyết định.
4. **Một máy, RTX 3070 8GB VRAM** (đổi từ 2080Ti 11GB khi sang laptop cá nhân 2026-08-01 — **ít VRAM hơn**, batch size phải thắt chứ không nới).
5. **Không có ground truth thật.** Dataset không kèm nhãn truy vấn; đáp án chỉ biết qua nộp bài. Bộ eval là **nhãn tự sinh** (PRD §6.1) ⇒ mọi con số là **tương đối**, phải hiệu chỉnh qua nộp bài BTC (PRD §6.9).

Baseline [arXiv:2512.13169](https://arxiv.org/abs/2512.13169) đóng góp **QUEST** và **DANTE** nhưng **không công bố metric định lượng nào** ⇒ `eval/` là điều kiện để mọi tuyên bố "cải thiện" có nghĩa.

## Liên quan

- Cách đo và kỷ luật báo cáo: [`evaluation-and-reporting.md`](evaluation-and-reporting.md)
- Hệ quả lên pipeline: [`pipeline-and-retrieval.md`](pipeline-and-retrieval.md)
- Invariants sinh ra từ điều 2 (`frame_idx`): `CLAUDE.md` §3

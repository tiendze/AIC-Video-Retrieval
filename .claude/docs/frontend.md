# Frontend

> Tách từ `CLAUDE.md` §9. Quy ước đặt tên JS/TS chung (camelCase, PascalCase, `use` prefix cho hook) nằm ở `.claude/rules/ecc/common/coding-style.md`. Ghi chú thiết kế đầy đủ: `docs/architecture.md` §8 · `.claude/plans/p1-search.plan.md` §P1.5.

**Gate thiết kế UI đã qua 2026-08-12** (plan P1.5) — chủ dự án cho phép build theo danh sách năng lực chức năng của paper, không có mẫu thiết kế riêng: *"có gì sau này mình sẽ kêu chỉnh sửa lại UI sau"*. **Không cần hỏi lại trước khi viết UI**; chủ dự án yêu cầu chỉnh giao diện thì cứ chỉnh, đó là kịch bản đã lường trước.

Stack ở `frontend/`: React 19 + Vite + TypeScript · **CSS viết tay (CSS Modules), KHÔNG UI kit, KHÔNG state library** — cố ý, để lần restyle sau không phải gỡ abstraction của thư viện nào. Test: Vitest + React Testing Library. Lint: `oxlint`.

- Thêm thư viện UI mới (component kit, styling framework, state manager) ⇒ **phải hỏi trước**, vì nó phá đúng lý do của quyết định trên.
- **Ràng buộc kỹ thuật:** virtualized rendering cho lưới nhiều thumbnail (`@tanstack/react-virtual`) · shortcut bàn phím · đổi mode không mất truy vấn đang gõ · phục vụ thumbnail nén, tải ảnh gốc theo yêu cầu.
- Tính năng chưa có backend hiển thị **disabled kèm badge phase** (`"P2"` / `"P3"`), không ẩn hẳn — bố cục chốt một lần, phase sau chỉ nối handler thật vào.

## Liên quan

- Backend mà UI gọi (mode, degraded mode): [`pipeline-and-retrieval.md`](pipeline-and-retrieval.md)
- Lỗi hiện ra UI phải đọc được, không dump traceback: [`error-handling.md`](error-handling.md)
- Vì sao "hệ thống interactive" (điều 3): [`context-and-scoring.md`](context-and-scoring.md)

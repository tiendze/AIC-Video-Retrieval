# Thư viện & API

> Tách từ `CLAUDE.md` §5. Snippet đầy đủ + link `llms.txt`: `docs/references.md` §5–6.

**Phần lớn tutorial trên mạng đang dạy API CŨ — đừng viết theo trí nhớ.**

| Thư viện | Bắt buộc dùng | KHÔNG dùng |
|---|---|---|
| Gemini (`google-genai`) | `client.interactions.create(...)` | `generate_content(...)`, `generation_config` |
| OpenAI (`openai`) | `client.responses.parse(..., text_format=Model)` | `chat.completions.create` |
| Milvus (`pymilvus`) | `MilvusClient` (v2.4+) | ORM API cũ |
| Elasticsearch | `from_=`; **bulk helpers** khi ingest | index từng document trong vòng lặp |
| MongoDB (`pymongo`) | **bulk write** + unique index `keyframe_id` / `video_id` | insert từng document; client đồng bộ trong `async def` |
| `ddgs` (QUEST nhánh 2) | `DDGS().images()` bọc `asyncio.to_thread` | gọi thẳng trong `async def` — không có async client |

- **OpenAI structured output phải xử lý nhánh `item.type == "refusal"`** trước khi đọc `item.parsed`.
- **Structured output cho OCR, VQA và sinh nhãn** qua `model_json_schema()` — không parse text tự do bằng regex.
- **Encoder:** `BEiT3Encoder` (`beit3_base_itc`) đang wire trong `app/main.py`; `SigLip2Encoder` là ứng viên thay thế, chọn qua `app/encoders/factory.py` (`kind:` trong `configs/model/*.yaml`). Hai encoder **xung khắc dependency** (`timm==0.4.12` vs `transformers>=4.49`) — đừng import cả hai trong một process.
- **PyTorch trên 8GB VRAM**: `torch.no_grad()` + autocast fp16, batch size configurable, giải phóng cache giữa các giai đoạn.
- **Đọc dữ liệu BTC:** mọi giá trị CSV/JSON là **chuỗi** kể cả số; `publish_date` là **`DD/MM/YYYY`**; tên thư mục **không nhất quán** nên phải glob (`docs/references.md` §2).
- Model ID (`gemini-3.6-flash`, `gpt-5.6`) **xác nhận lại lúc implement**. Cần API chưa có trong `docs/references.md` → fetch `llms.txt` rồi cập nhật `docs/references.md` trong cùng change.

## Liên quan

- Contract của encoder (shape, dtype, L2-normalize): [`code-architecture.md`](code-architecture.md)
- Nơi các thư viện này được gọi: [`pipeline-and-retrieval.md`](pipeline-and-retrieval.md)
- Timeout / retry / fallback bắt buộc khi gọi API ngoài: [`error-handling.md`](error-handling.md)

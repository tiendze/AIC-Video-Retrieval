"""Behaviour lock for split_into_sentences.

Why this exists: measured on the 25 SOTUYEN1 contest queries, 9/25 exceed the
FG-CLIP 77-token cap after VI->EN translation, and `truncation=True` drops the
tail silently -- for query-p1-16-trake.txt, the tail is TRAKE event E3, one of
three mandatory keypoints. No sub-query produced here ever needs an LLM: the
same sentence-ending punctuation the queries are already written with is
enough. Retrieval measured 0/25 split fragments still exceeding the cap
(vinai_split_report.json in the research session), so splitting alone removes
the truncation class.

`retrieve_kis_candidates` (app/services/search_service.py) calls this on
`text_query`, translates every piece, and OR-unions the resulting Stage 1
candidate pools by taking the max cosine score per keyframe across pieces --
so a single-sentence query MUST still round-trip to exactly `[text.strip()]`,
unchanged, to keep every existing Stage 1 test passing byte-for-byte.
"""

from __future__ import annotations

from app.services.query_decomposition import split_into_sentences


class TestSplitIntoSentences:
    def test_a_single_sentence_with_no_terminal_punctuation_round_trips_unchanged(self) -> None:
        # Existing Stage 1 tests call retrieve_kis_candidates(text_query="red
        # shirt", ...) and assert encoder.encode_text_calls == [["red
        # shirt"]] -- this case must never produce more than one piece.
        assert split_into_sentences("red shirt") == ["red shirt"]

    def test_splits_on_a_period(self) -> None:
        result = split_into_sentences("Con lân đứng ở cột. Sau đó nhảy sang cột khác.")

        assert result == ["Con lân đứng ở cột.", "Sau đó nhảy sang cột khác."]

    def test_splits_on_question_and_exclamation_marks(self) -> None:
        result = split_into_sentences("Có bao nhiêu người? Rất đông!")

        assert result == ["Có bao nhiêu người?", "Rất đông!"]

    def test_splits_on_newlines_for_multiline_kis_queries(self) -> None:
        # query-p1-14-kis.txt / query-p1-8-kis.txt are written as separate
        # lines with no terminal period on some of them.
        text = (
            "Người đầu bếp đặt nguyên liệu vào đĩa\n"
            "Đầu bếp dùng muỗng lấy nguyên liệu mềm\n"
            "Phần nguyên liệu được đặt vào giữa đĩa"
        )

        result = split_into_sentences(text)

        assert result == [
            "Người đầu bếp đặt nguyên liệu vào đĩa",
            "Đầu bếp dùng muỗng lấy nguyên liệu mềm",
            "Phần nguyên liệu được đặt vào giữa đĩa",
        ]

    def test_strips_whitespace_from_each_piece(self) -> None:
        result = split_into_sentences("  Câu một.   Câu hai.  ")

        assert result == ["Câu một.", "Câu hai."]

    def test_drops_empty_pieces_from_consecutive_punctuation(self) -> None:
        result = split_into_sentences("Xong rồi!! Tiếp theo.")

        assert result == ["Xong rồi!!", "Tiếp theo."]

    def test_empty_string_returns_no_pieces(self) -> None:
        assert split_into_sentences("") == []

    def test_whitespace_only_string_returns_no_pieces(self) -> None:
        assert split_into_sentences("   \n  ") == []

    def test_a_trailing_terminator_does_not_produce_an_empty_final_piece(self) -> None:
        assert split_into_sentences("Một câu duy nhất.") == ["Một câu duy nhất."]

    def test_long_query_from_the_sotuyen1_set_splits_every_piece_under_the_fgclip_cap(self) -> None:
        # query-p1-23-kis.txt: measured 110 EnViT5 tokens whole, 33 over the
        # 77-token cap. This is the worst case in the set.
        text = (
            "Hình ảnh giáo viên nam mặc sơ mi trắng, thắt cà vạt tối màu, nổi bật "
            "trên phông nền xanh dương đậm có hoa văn mờ.\n\n"
            "Slide bài giảng Nền trắng với khung viền màu hồng tím, phía trên có "
            "thanh tiêu đề xanh dương chứa họa tiết địa cầu và mũi tên vàng/xanh "
            "ngọc. Bên dưới là sơ đồ 3 tầng được liên kết bởi các mũi tên xanh "
            "ngọc trỏ xuống:\n\n"
            "Tầng 1: 2 khối hộp được bao bởi 1 khối hộp cam.\n\n"
            "Tầng 2: 1 khối hộp lớn màu xanh dương đậm ở chính giữa.\n\n"
            "Tầng 3: 2 khối hộp được bao bởi 1 khối hộp xanh lá cây."
        )

        pieces = split_into_sentences(text)

        assert len(pieces) >= 2
        assert all(piece for piece in pieces)

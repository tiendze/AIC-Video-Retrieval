"""RED reproducer for pipelines/text_normalize.py -- Vietnamese spoken-number
normalization for ASR transcript text (2026-08-08, real L21_V001 evidence).
Pure function, no I/O.
"""

from __future__ import annotations

from pipelines.text_normalize import normalize_spoken_numbers


class TestMarkerDrivenMultiWordNumbers:
    """Numbers containing an unambiguous marker word (mươi/trăm/nghìn/
    triệu/phẩy/mười/linh/lẻ) always convert -- these markers have no other
    common meaning in Vietnamese."""

    def test_real_example_sau_muoi_giay(self):
        # docs/ evidence: "chương trình sáu mươi giây" -> "chương trình 60 giây"
        assert normalize_spoken_numbers("chương trình sáu mươi giây") == "chương trình 60 giây"

    def test_real_example_hai_muoi_lan(self):
        assert normalize_spoken_numbers("gấp gần hai mươi lần") == "gấp gần 20 lần"

    def test_real_example_ba_muoi_mot_thang_bay(self):
        # "ba mươi mốt tháng bảy" -> "31 tháng 7" (day AND month both convert)
        assert normalize_spoken_numbers("ngày ba mươi mốt tháng bảy") == "ngày 31 tháng 7"

    def test_real_example_hai_muoi_hai(self):
        assert normalize_spoken_numbers("triển lãm lần thứ hai mươi hai") == "triển lãm lần thứ 22"

    def test_ten_alone(self):
        assert normalize_spoken_numbers("mười người") == "10 người"

    def test_ten_plus_unit(self):
        assert normalize_spoken_numbers("mười lăm phút") == "15 phút"

    def test_hundred_bare(self):
        assert normalize_spoken_numbers("năm trăm người") == "500 người"

    def test_hundred_with_zero_connector(self):
        # "năm trăm linh năm" -> 505: "trăm" + "linh" are both unambiguous markers.
        assert normalize_spoken_numbers("năm trăm linh năm") == "505"

    def test_hundred_with_tens(self):
        assert normalize_spoken_numbers("năm trăm hai mươi mốt") == "521"

    def test_thousand_combined_with_hundred_and_tens(self):
        # "hai nghìn không trăm hai mươi sáu" -> 2026 (spoken year reading)
        assert normalize_spoken_numbers("hai nghìn không trăm hai mươi sáu") == "2026"

    def test_million(self):
        assert normalize_spoken_numbers("một triệu đồng") == "1000000 đồng"


class TestDecimalViaPhay:
    def test_real_example_nam_phay_bay(self):
        # docs/ evidence: "năm phẩy bảy xen ti mét" -> "5.7 xen ti mét"
        assert normalize_spoken_numbers("năm phẩy bảy xen ti mét") == "5.7 xen ti mét"

    def test_decimal_with_marker_driven_fraction(self):
        assert normalize_spoken_numbers("hai mươi phẩy năm") == "20.5"

    def test_phay_not_followed_by_a_number_stays_as_is(self):
        # "phẩy" here isn't a decimal point (no fraction word after it) --
        # the integer part must fall through to the normal paths unchanged.
        assert normalize_spoken_numbers("năm phẩy à quên") == "năm phẩy à quên"


class TestBareUnitAdjacentDigits:
    """A single ambiguous digit-word converts ONLY when sitting right next
    to a whitelisted unit noun -- this is what recovers "tháng bảy" (July)
    from a bare "bảy" without a mươi/trăm marker."""

    def test_real_example_thang_bay(self):
        assert (
            normalize_spoken_numbers("kéo dài đến hết ngày tháng bảy")
            == "kéo dài đến hết ngày tháng 7"
        )

    def test_real_example_thang_chin(self):
        assert (
            normalize_spoken_numbers("nghỉ lễ quốc khánh hai tháng chín")
            == "nghỉ lễ quốc khánh 2 tháng 9"
        )

    def test_age_unit(self):
        assert normalize_spoken_numbers("cháu bé ba tuổi") == "cháu bé 3 tuổi"

    def test_unambiguous_digit_without_unit_still_stays_word(self):
        # "sáu" has no other meaning, but with no marker AND no adjacent
        # whitelisted unit we still leave it alone -- the function's only
        # trigger conditions are marker-driven or unit-adjacent, by design.
        assert normalize_spoken_numbers("số sáu may mắn") == "số sáu may mắn"


class TestAmbiguousWordsNeverConvertBare:
    """The real risk this module exists to avoid: corrupting a common
    non-numeric phrase because a digit-word also happens to spell it."""

    def test_nam_nay_year_not_converted(self):
        assert (
            normalize_spoken_numbers("năm nay thời tiết khắc nghiệt")
            == "năm nay thời tiết khắc nghiệt"
        )

    def test_khong_negation_not_converted(self):
        assert normalize_spoken_numbers("không có gì thay đổi") == "không có gì thay đổi"

    def test_mot_article_not_converted_without_unit(self):
        assert (
            normalize_spoken_numbers("một người đàn ông đang đi bộ")
            == "một người đàn ông đang đi bộ"
        )

    def test_ba_father_not_converted(self):
        assert normalize_spoken_numbers("ba mẹ đưa con đi học") == "ba mẹ đưa con đi học"

    def test_nam_year_word_before_marker_driven_number_stays_year(self):
        # Spoken year reading: "năm" here means "year", not "5" -- the
        # marker-driven phrase after it converts on its own.
        assert normalize_spoken_numbers("năm hai nghìn không trăm hai mươi sáu") == "năm 2026"

    def test_nam_never_converts_even_next_to_whitelisted_unit(self):
        # Deliberate under-conversion (module docstring): "năm giờ" (5
        # o'clock) stays as words rather than risk "năm" elsewhere meaning
        # "year".
        assert normalize_spoken_numbers("bây giờ là năm giờ") == "bây giờ là năm giờ"


class TestNoNumbersPassThroughUnchanged:
    def test_plain_sentence(self):
        text = "phóng viên đứng trước tòa nhà chính phủ đưa tin"
        assert normalize_spoken_numbers(text) == text

    def test_empty_string(self):
        assert normalize_spoken_numbers("") == ""

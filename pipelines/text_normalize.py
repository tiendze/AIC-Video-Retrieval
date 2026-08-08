"""Normalizes Vietnamese spoken numbers to digits in ASR transcript text
(2026-08-08 -- real data from L21 showed "sáu mươi giây", "năm phẩy bảy
xentimét", "ba mươi mốt tháng bảy" going into Elasticsearch as words, so a
query typed as "60 giây" or "31/7" never matches).

Deliberately conservative: several digit-words are highly ambiguous in
Vietnamese outside a numeric context -- "năm" (5 vs "year") is the worst
offender ("năm tháng" alone can mean "5 months" OR the idiom "years and
months" for time passing; "năm nay"/"năm ngoái"/"năm học" are all "year", not
"5"). A bare ambiguous word is converted ONLY when either (a) it is part of
a multi-word number phrase with an unambiguous marker ("mươi"/"trăm"/
"nghìn"/"triệu"/"phẩy"/"mười"/"linh"/"lẻ" -- these words have no other
common meaning), or (b) it sits immediately next to a small whitelist of
unit nouns (_UNIT_WHITELIST) AND isn't in _NEVER_BARE. Anything else is left
as spoken words -- missing a conversion is cheap, corrupting an unrelated
sentence ("năm nay" -> "5 nay") is not.
"""

from __future__ import annotations

_DIGIT_WORDS: dict[str, int] = {
    "không": 0,
    "một": 1,
    "mốt": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "nhăm": 5,
    "sáu": 6,
    "bảy": 7,
    "bẩy": 7,
    "tám": 8,
    "chín": 9,
}
_TEN = "mười"
_TEN_MULTIPLIER = "mươi"
_HUNDRED = "trăm"
_THOUSAND = {"nghìn", "ngàn"}
_MILLION = "triệu"
_ZERO_CONNECTOR = {"linh", "lẻ"}
_DECIMAL_POINT = "phẩy"

# Words never treated as a BARE digit even next to a whitelisted unit --
# their non-numeric meaning is too common in this dataset's news-broadcast
# register to risk (see module docstring: "năm tháng" the idiom). Still
# converts fine as part of a marker-driven multi-word phrase (e.g. "năm
# mươi" -> "50" is safe: "mươi" is unambiguous) or right next to "phẩy"
# (decimal point has no other meaning either), so this only blocks the
# *bare, unit-adjacent* single-word case.
_NEVER_BARE = {"năm"}

# Nouns whose presence right before/after a bare digit-word is enough
# evidence that the word is really a number, not its other meaning --
# curated conservatively from what appears in this dataset (news broadcast
# durations, dates, ages, counts), not an exhaustive dictionary.
_UNIT_WHITELIST = {
    "giây",
    "phút",
    "giờ",
    "ngày",
    "tháng",
    "tuổi",
    "lần",
    "mét",
    "km",
    "cm",
    "kg",
    "độ",
}


def _parse_tens_units(tokens: list[str], i: int, *, after_hundred: bool) -> tuple[int | None, int]:
    """0-99, optionally with a 'linh/lẻ <digit>' zero-connector right after
    a hundreds group (e.g. "trăm linh năm" -> 105's last two digits)."""
    if (
        after_hundred
        and i < len(tokens)
        and tokens[i] in _ZERO_CONNECTOR
        and i + 1 < len(tokens)
        and tokens[i + 1] in _DIGIT_WORDS
    ):
        return _DIGIT_WORDS[tokens[i + 1]], i + 2

    if i < len(tokens) and tokens[i] == _TEN:
        if i + 1 < len(tokens) and tokens[i + 1] in _DIGIT_WORDS:
            return 10 + _DIGIT_WORDS[tokens[i + 1]], i + 2
        return 10, i + 1

    if (
        i < len(tokens)
        and tokens[i] in _DIGIT_WORDS
        and i + 1 < len(tokens)
        and tokens[i + 1] == _TEN_MULTIPLIER
    ):
        tens = _DIGIT_WORDS[tokens[i]] * 10
        i += 2
        if i < len(tokens) and tokens[i] in _DIGIT_WORDS:
            return tens + _DIGIT_WORDS[tokens[i]], i + 1
        return tens, i

    return None, i


def _parse_group_under_1000(tokens: list[str], i: int) -> tuple[int | None, int]:
    hundreds = 0
    after_hundred = False
    if (
        i < len(tokens)
        and tokens[i] in _DIGIT_WORDS
        and i + 1 < len(tokens)
        and tokens[i + 1] == _HUNDRED
    ):
        hundreds = _DIGIT_WORDS[tokens[i]] * 100
        i += 2
        after_hundred = True

    rest, next_i = _parse_tens_units(tokens, i, after_hundred=after_hundred)
    if rest is None:
        return (hundreds, i) if after_hundred else (None, i)
    return hundreds + rest, next_i


def _parse_marker_driven_integer(tokens: list[str], start: int) -> tuple[int, int] | None:
    """Parses a number phrase that contains at least one unambiguous marker
    word (mươi/trăm/nghìn/triệu/mười/linh/lẻ). Returns None if tokens[start]
    doesn't begin one -- a BARE single digit word never matches here, by
    design (see the bare-unit-adjacent branch in `normalize_spoken_numbers`
    for that case)."""
    i = start
    total = 0

    if (
        i < len(tokens)
        and tokens[i] in _DIGIT_WORDS
        and i + 1 < len(tokens)
        and tokens[i + 1] == _MILLION
    ):
        total += _DIGIT_WORDS[tokens[i]] * 1_000_000
        i += 2

    if (
        i < len(tokens)
        and tokens[i] in _DIGIT_WORDS
        and i + 1 < len(tokens)
        and tokens[i + 1] in _THOUSAND
    ):
        total += _DIGIT_WORDS[tokens[i]] * 1000
        i += 2

    group, next_i = _parse_group_under_1000(tokens, i)
    if group is None:
        return (total, i) if total > 0 else None
    return total + group, next_i


def _looks_like_marker_driven(tokens: list[str], start: int) -> bool:
    """True when the run tokens[start:] actually uses a marker word, as
    opposed to `_parse_marker_driven_integer` falling through to its bare-
    digit branch for a single ambiguous word."""
    if start < len(tokens) and tokens[start] == _TEN:
        return True
    if start + 1 < len(tokens) and tokens[start + 1] in (
        _TEN_MULTIPLIER,
        _HUNDRED,
        *_THOUSAND,
        _MILLION,
    ):
        return True
    return False


def _parse_marker_driven_or_bare(tokens: list[str], i: int) -> tuple[int, int] | None:
    """Marker-driven phrase if present, else a single bare digit word --
    used ONLY by the decimal path below, where the adjacent 'phẩy' already
    removes the ambiguity that keeps a bare digit out of the general path
    (nobody says "năm phẩy" to mean anything but a decimal five-point)."""
    if _looks_like_marker_driven(tokens, i):
        parsed = _parse_marker_driven_integer(tokens, i)
        if parsed is not None:
            return parsed
    if i < len(tokens) and tokens[i] in _DIGIT_WORDS:
        return _DIGIT_WORDS[tokens[i]], i + 1
    return None


def _try_parse_decimal(tokens: list[str], i: int) -> tuple[str, int] | None:
    """'<int> phẩy <int>' -> "value.value". Tried before the general
    marker-driven/bare-unit-adjacent paths, since 'phẩy' is the strongest
    marker of all (decimal point, no other meaning) and justifies allowing
    a bare digit -- even one normally in _NEVER_BARE -- on either side."""
    integer_part = _parse_marker_driven_or_bare(tokens, i)
    if integer_part is None:
        return None
    int_value, mid_i = integer_part
    if mid_i >= len(tokens) or tokens[mid_i] != _DECIMAL_POINT:
        return None

    fraction_part = _parse_marker_driven_or_bare(tokens, mid_i + 1)
    if fraction_part is None:
        return None
    frac_value, next_i = fraction_part
    return f"{int_value}.{frac_value}", next_i


def normalize_spoken_numbers(text: str) -> str:
    """Converts Vietnamese spoken-number word sequences to digits. See
    module docstring for the deliberate conservatism around ambiguous words.
    """
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        decimal = _try_parse_decimal(tokens, i)
        if decimal is not None:
            text_value, next_i = decimal
            out.append(text_value)
            i = next_i
            continue

        if _looks_like_marker_driven(tokens, i):
            parsed = _parse_marker_driven_integer(tokens, i)
            if parsed is not None:
                value, next_i = parsed
                out.append(str(value))
                i = next_i
                continue

        token = tokens[i]
        if (
            token in _DIGIT_WORDS
            and token not in _NEVER_BARE
            and (
                (i > 0 and tokens[i - 1] in _UNIT_WHITELIST)
                or (i + 1 < len(tokens) and tokens[i + 1] in _UNIT_WHITELIST)
            )
        ):
            out.append(str(_DIGIT_WORDS[token]))
        else:
            out.append(token)
        i += 1

    return " ".join(out)

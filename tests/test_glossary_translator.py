"""Behaviour lock for GlossaryCorrectedTranslator.

Why this exists: measured on the 25 SOTUYEN1 contest queries (git 5191098),
EnViT5 mistranslates a small set of Vietnamese cultural/cinematography terms
into specific, reproducible wrong English -- `con lân` -> "phosphor", `cầm
dù` -> "carrying parachutes", `lia cam` -> "a single orange peel", `lục
bình` -> "greenery", `trống cơ` -> "muscle drum kit". These are not one-off
sentence quirks: the same Vietnamese term produces the same wrong English
every time, so a deterministic post-translation correction fixes them without
an LLM (currently unavailable -- no Gemini quota, see FallbackTranslator).

Each rule matches on WORD BOUNDARIES on both sides, and is gated on the
ORIGINAL VIETNAMESE containing the trigger as a whole word/phrase -- code
review 2026-08-25 found the first version's bare-substring matching could
corrupt a CORRECT translation: `vi_trigger="lân"` also matches inside "lân
cận" (vicinity), and `en_search="phosphor"` also matches inside
"phosphorescent" (a real, unrelated, correctly-translated Vietnamese term).
`DEFAULT_GLOSSARY` uses the actual multi-word phrase measured for each
failure specifically to stay clear of these homographs.
"""

from __future__ import annotations

from app.services.query_translator import (
    DEFAULT_GLOSSARY,
    GlossaryCorrectedTranslator,
    GlossaryRule,
)


class _FakeTranslator:
    def __init__(self, english: str, model: str = "fake-model") -> None:
        self._english = english
        self._model = model
        self.calls: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model

    def translate(self, vietnamese: str) -> str:
        self.calls.append(vietnamese)
        return self._english


class TestGlossaryRuleApplication:
    def test_replaces_the_wrong_phrase_when_the_vietnamese_trigger_is_present(self) -> None:
        rule = GlossaryRule(vi_trigger="con lân", en_search="phosphor", en_replacement="lion dance costume")
        translator = GlossaryCorrectedTranslator(
            _FakeTranslator("a close-up of a white phosphor with a red nose"), rules=(rule,)
        )

        result = translator.translate("Con lân màu trắng, mũi đỏ.")

        assert result == "a close-up of a white lion dance costume with a red nose"

    def test_leaves_the_translation_alone_when_the_vietnamese_trigger_is_absent(self) -> None:
        # The English word could coincidentally appear for an unrelated reason
        # -- only correct it when we know WHY it's wrong.
        rule = GlossaryRule(vi_trigger="con lân", en_search="phosphor", en_replacement="lion dance costume")
        translator = GlossaryCorrectedTranslator(
            _FakeTranslator("a lab technician measuring phosphor levels"), rules=(rule,)
        )

        result = translator.translate("Người kỹ thuật viên đo nồng độ.")

        assert result == "a lab technician measuring phosphor levels"

    def test_leaves_the_translation_alone_when_the_english_search_term_is_absent(self) -> None:
        rule = GlossaryRule(vi_trigger="con lân", en_search="phosphor", en_replacement="lion dance costume")
        translator = GlossaryCorrectedTranslator(
            _FakeTranslator("a close-up of a white lion with a red nose"), rules=(rule,)
        )

        result = translator.translate("Con lân màu trắng, mũi đỏ.")

        assert result == "a close-up of a white lion with a red nose"

    def test_match_is_case_insensitive_on_both_sides(self) -> None:
        rule = GlossaryRule(vi_trigger="con lân", en_search="phosphor", en_replacement="lion dance costume")
        translator = GlossaryCorrectedTranslator(
            _FakeTranslator("Phosphor dancers entered the stage"), rules=(rule,)
        )

        result = translator.translate("Đội múa Con Lân bước ra sân khấu.")

        assert result == "lion dance costume dancers entered the stage"

    def test_matches_the_exact_measured_plural_form_only(self) -> None:
        # "parachutes" (plural) is the only form EnViT5 was ever measured to
        # produce -- the rule targets that literal string, not a guessed stem.
        rule = GlossaryRule(vi_trigger="cầm dù", en_search="parachutes", en_replacement="umbrellas")
        translator = GlossaryCorrectedTranslator(
            _FakeTranslator("two people carrying parachutes in the rain"), rules=(rule,)
        )

        result = translator.translate("Hai người cầm dù trong cơn mưa.")

        assert result == "two people carrying umbrellas in the rain"

    def test_does_not_corrupt_a_correct_translation_where_the_search_term_is_only_a_substring(
        self,
    ) -> None:
        # The exact false positive code review found: "con lân" is also a
        # substring of "con lân tinh" (phosphorescence, an unrelated term),
        # and "phosphor" is a substring of "phosphorescent" -- a CORRECT
        # EnViT5 output. Word-boundary matching on the English side must
        # refuse to touch it.
        rule = GlossaryRule(vi_trigger="con lân", en_search="phosphor", en_replacement="lion dance costume")
        translator = GlossaryCorrectedTranslator(
            _FakeTranslator("a phosphorescent toy glowing in the dark"), rules=(rule,)
        )

        result = translator.translate("Một con lân tinh phát sáng trong bóng tối.")

        assert result == "a phosphorescent toy glowing in the dark"

    def test_does_not_fire_on_an_unrelated_word_that_shares_the_bare_syllable(self) -> None:
        # "dù" alone means "although" -- an extremely common word. The rule
        # must require the full "cầm dù" phrase, not just the syllable "dù".
        rule = GlossaryRule(vi_trigger="cầm dù", en_search="parachutes", en_replacement="umbrellas")
        translator = GlossaryCorrectedTranslator(
            _FakeTranslator("although it is raining, the parade continues"), rules=(rule,)
        )

        result = translator.translate("Dù trời mưa, đoàn diễu hành vẫn tiếp tục.")

        assert result == "although it is raining, the parade continues"

    def test_applies_multiple_independent_rules_in_one_translation(self) -> None:
        rules = (
            GlossaryRule(vi_trigger="con lân", en_search="phosphor", en_replacement="lion"),
            GlossaryRule(vi_trigger="cầm dù", en_search="parachutes", en_replacement="umbrellas"),
        )
        translator = GlossaryCorrectedTranslator(
            _FakeTranslator("a phosphor dancer next to a vendor selling parachutes"), rules=rules
        )

        result = translator.translate("Con lân đứng cạnh người cầm dù bán hàng.")

        assert result == "a lion dancer next to a vendor selling umbrellas"

    def test_delegates_model_id_to_the_wrapped_translator(self) -> None:
        translator = GlossaryCorrectedTranslator(
            _FakeTranslator("x", model="VietAI/envit5-translation"), rules=()
        )

        assert translator.model_id == "VietAI/envit5-translation"

    def test_empty_rule_set_is_a_pure_passthrough(self) -> None:
        translator = GlossaryCorrectedTranslator(_FakeTranslator("unchanged text"), rules=())

        assert translator.translate("bất kỳ câu nào") == "unchanged text"


class TestDefaultGlossary:
    def test_covers_every_term_measured_wrong_in_the_sotuyen1_query_set(self) -> None:
        # Locks the glossary to the actual measured failures (module docstring
        # of app/services/query_translator.py), not an arbitrary word list.
        triggers = {rule.vi_trigger for rule in DEFAULT_GLOSSARY}

        assert triggers == {"con lân", "cầm dù", "lia cam", "lục bình", "trống cơ"}

    def test_fixes_the_exact_lan_mistranslation_measured_from_envit5(self) -> None:
        translator = GlossaryCorrectedTranslator(
            _FakeTranslator(
                "The video begins with a near-head photo of a white phosphor with a "
                "red nose, next to a white flag with a red border."
            ),
            rules=DEFAULT_GLOSSARY,
        )

        result = translator.translate(
            "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, bên cạnh "
            "lá cờ trắng viền đỏ."
        )

        assert "phosphor" not in result.lower()

"""QueryTranslator — VI->EN query-time translation for the `semantic` search
channel.

BEiT-3's image-text contrastive stage was trained on English pairs, so a
Vietnamese query is off-distribution for the text tower even though the image
tower is untouched (plan §P0.9). Confirmed 2026-08-11 by scoring the SAME
Milvus collection with English queries (eval/query_translation.py's
`semantic@en` sidecar channel): AIC Final Score went from 0.0451 (Vietnamese)
to 0.6208 (English) on the same 173 queries. Translating the query at search
time, not swapping the encoder, keeps the existing collection valid -- no
re-embedding, no new `model_id` (CLAUDE.md §3).

Shared by two callers so a sidecar generated for the eval diagnostic and a
live production query are translated identically: `app.services.search_service`
(production, `mode="semantic"` only -- the `ocr` index holds Vietnamese text,
so translating an OCR query would measure translation loss instead of the
retrieval channel) and `scripts/translate_eval_queries.py` (offline sidecar
generation for the `semantic@en` eval channel).
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class QueryTranslator(Protocol):
    @property
    def model_id(self) -> str:
        """Stamped onto sidecar rows for provenance (eval/query_translation.py
        `TranslatedQuery.translator_model`)."""
        ...

    def translate(self, vietnamese: str) -> str: ...


def strip_target_prefix(decoded: str, target_prefix: str) -> str:
    """EnViT5 echoes the target-language tag it was trained with, so a decode
    comes back as `"en: A mother donkey..."`. Leaving the tag in would make it
    part of the query text that gets embedded and searched.

    Only strips at the start and only the configured tag -- a caption that
    legitimately contains "en:" mid-sentence is left alone. Models that need no
    tag (VinAI-Translate) get `target_prefix=""` and pass through untouched.
    """
    if not target_prefix:
        return decoded.strip()
    text = decoded.strip()
    return text[len(target_prefix) :].strip() if text.startswith(target_prefix) else text


class LocalNmtTranslator:
    """Offline VI->EN translation via a local seq2seq model -- no API key, no
    quota, no network after the first HuggingFace download.

    Default is `VietAI/envit5-translation` (EnViT5, trained on MTet + PhoMT):
    a T5-base-sized seq2seq that reports state-of-the-art en<->vi BLEU while
    staying small enough to run on CPU, so it does not compete with BEiT-3 for
    the 8GB of VRAM (CLAUDE.md §1.4). Greedy decoding (no sampling) keeps
    translation deterministic -- the same query must translate identically on
    every call, otherwise the same search issued twice could return different
    results.

    Model swap (e.g. to a more Vietnamese-specialized NMT model) is a config
    change only -- `configs/model/query_translator.yaml` -- since every
    caller talks to the `QueryTranslator` Protocol, not this class directly.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        source_prefix: str,
        target_prefix: str,
        max_new_tokens: int,
    ) -> None:
        import torch  # lazy: only needed when this class is used
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._model_id = model_id
        self._torch = torch
        self._device = torch.device(device)
        self._source_prefix = source_prefix
        self._target_prefix = target_prefix
        self._max_new_tokens = max_new_tokens
        logger.info("Loading %s on %s ...", model_id, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self._model.to(self._device).eval()

    @property
    def model_id(self) -> str:
        return self._model_id

    def translate(self, vietnamese: str) -> str:
        prompt = f"{self._source_prefix}{vietnamese}"
        encoded = self._tokenizer(prompt, return_tensors="pt", padding=True)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        english = strip_target_prefix(decoded, self._target_prefix)
        if not english:
            # An empty translation would be embedded as an empty query and
            # silently score 0 -- fail the search instead (CLAUDE.md §7).
            raise ValueError(f"empty translation for {vietnamese!r} (decoded={decoded!r})")
        return english

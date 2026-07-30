"""Vendored, trimmed subset of microsoft/unilm/beit3 model-building code.

Source: https://github.com/microsoft/unilm/tree/master/beit3 (commit tracked in
docs/references.md §3.2). BEiT-3 is not published as an installable package
(PRD risk R1), so the pieces needed to *load a pretrained checkpoint and run
inference* are ported here verbatim/trimmed rather than reimplemented.

Only the retrieval (image-text contrastive) path is kept — VQA, NLVR2,
captioning and image-classification heads from the upstream repo are dropped
because this project only needs BEiT3ForRetrieval (YAGNI).

Copyright (c) 2023 Microsoft. Licensed under the MIT License (see
https://github.com/microsoft/unilm/blob/master/LICENSE).
"""

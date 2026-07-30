# --------------------------------------------------------
# Trimmed from microsoft/unilm/beit3/modeling_finetune.py — only
# BEiT3ForRetrieval and its `_patch16_224_retrieval` registration are kept.
# The upstream file also defines VQA/NLVR2/captioning/image-classification
# heads that this project does not use (YAGNI).
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks
# (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

from .modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from .utils import ClipLoss, get_rank, get_world_size


class BEiT3ForRetrieval(BEiT3Wrapper):
    def __init__(self, args, **kwargs):
        super().__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        self.criterion = ClipLoss(rank=get_rank(), world_size=get_world_size())
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self, image=None, text_description=None, padding_mask=None, only_infer=False, **kwargs
    ):
        if image is not None:
            outputs = self.beit3(
                textual_tokens=None, visual_tokens=image, text_padding_position=None
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_cls = F.normalize(vision_cls, dim=-1)
        else:
            vision_cls = None

        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None

        if only_infer:
            return vision_cls, language_cls
        loss, logits_per_image, logits_per_text = self.criterion(
            vision_cls, language_cls, self.logit_scale.exp()
        )
        return loss, vision_cls, language_cls


@register_model
def beit3_base_patch16_224_retrieval(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    return BEiT3ForRetrieval(args, **kwargs)


@register_model
def beit3_large_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    return BEiT3ForRetrieval(args, **kwargs)

# --------------------------------------------------------
# Trimmed from microsoft/unilm/beit3/utils.py — only what BEiT3ForRetrieval
# needs for single-process inference is kept (training/distributed/logging
# utilities from the upstream file are dropped, YAGNI).
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks
# (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class ClipLoss(nn.Module):
    """Contrastive loss used at BEiT-3 ITC training time.

    Unused at inference (only_infer=True short-circuits before this runs), kept
    only because BEiT3ForRetrieval.__init__ constructs it unconditionally.
    """

    def __init__(self, cache_labels=False, rank=0, world_size=1):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss, logits_per_image, logits_per_text


def load_state_dict(model, state_dict, prefix="", ignore_missing="relative_position_index"):
    """Load a checkpoint state_dict into model, tolerating expected missing keys."""
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    model_name = model.__class__.__name__
    if len(warn_missing_keys) > 0:
        print(f"Weights of {model_name} not initialized from checkpoint: {warn_missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Weights from checkpoint not used in {model_name}: {unexpected_keys}")
    if len(ignore_missing_keys) > 0:
        print(f"Ignored missing weights of {model.__class__.__name__}: {ignore_missing_keys}")
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))

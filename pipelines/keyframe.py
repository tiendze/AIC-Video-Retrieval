"""4-keyframe-per-shot sampling — PRD §F1.

    k_extract = { K_(a + floor(i * (b - a) / 3)) , for all i in {0,1,2,3} }

`[a, b]` is a shot's inclusive frame range (pipelines.shot_detect.Shot). The
formula is written with *set* notation, so a short shot (span < 3) collapses
to fewer than 4 distinct frame indices instead of repeating one frame as
multiple duplicate keyframe records.

Only computes which frame indices to extract — assigning `global_index`,
`s_v`/`e_v`, and writing the actual keyframe images/embeddings happens in
pipelines/ingest.py, which orchestrates this across a whole video.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipelines.shot_detect import Shot

_SAMPLES_PER_SHOT = 4


def keyframe_frame_indices(shot: Shot) -> list[int]:
    """Returns the ascending, deduplicated frame indices to extract from one shot."""
    span = shot.end_frame - shot.start_frame
    indices = {shot.start_frame + (i * span) // 3 for i in range(_SAMPLES_PER_SHOT)}
    return sorted(indices)


def keyframe_indices_for_shots(shots: Sequence[Shot]) -> list[int]:
    """Flattens keyframe indices across all of a video's shots, in shot order —
    the order pipelines/ingest.py assigns contiguous `global_index` against."""
    indices: list[int] = []
    for shot in shots:
        indices.extend(keyframe_frame_indices(shot))
    return indices

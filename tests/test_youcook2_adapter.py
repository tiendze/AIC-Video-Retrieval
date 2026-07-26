from pathlib import Path

from datasets.youcook2 import YouCook2Adapter

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "youcookii_annotations_fixture.json"


def make_adapter() -> YouCook2Adapter:
    return YouCook2Adapter(annotations_path=FIXTURE_PATH, video_dir=Path("data/videos"))


def test_iter_videos_returns_only_requested_split() -> None:
    adapter = make_adapter()

    val_videos = list(adapter.iter_videos(split="validation"))

    assert {v.video_id for v in val_videos} == {"vid001", "vid002"}
    assert all(v.duration_sec > 0 for v in val_videos)


def test_ground_truth_order_index_follows_annotation_order() -> None:
    adapter = make_adapter()

    segments = adapter.load_ground_truth(split="validation")
    vid001_segments = [s for s in segments if s.video_id == "vid001"]
    vid001_segments.sort(key=lambda s: s.order_index)

    assert [s.order_index for s in vid001_segments] == [0, 1, 2, 3]
    assert [s.caption for s in vid001_segments] == [
        "cut the onion",
        "heat the pan",
        "add the onion to the pan",
        "stir until golden",
    ]
    assert vid001_segments[0].start_sec == 0.0
    assert vid001_segments[0].end_sec == 10.0


def test_load_ground_truth_excludes_other_splits() -> None:
    adapter = make_adapter()

    segments = adapter.load_ground_truth(split="validation")

    assert all(s.video_id != "vid003" for s in segments)
    assert len(segments) == 7  # 4 (vid001) + 3 (vid002)

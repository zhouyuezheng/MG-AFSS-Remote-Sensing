from ultralytics.afss.scheduler import (
    classify_score,
    select_active_images,
    select_easy_forced_review,
    select_moderate_forced_coverage,
)
from ultralytics.afss.state import AFSSImageState


def make_state(im_file: str, task_score: float, last_used_epoch: int) -> AFSSImageState:
    return AFSSImageState(im_file=im_file, task_score=task_score, last_used_epoch=last_used_epoch)


def test_classify_score_boundaries():
    assert classify_score(0.54, 0.55, 0.85) == "hard"
    assert classify_score(0.55, 0.55, 0.85) == "moderate"
    assert classify_score(0.85, 0.55, 0.85) == "moderate"
    assert classify_score(0.851, 0.55, 0.85) == "easy"


def test_easy_forced_review_picks_long_unseen_samples_first():
    states = [
        make_state("recent.jpg", 0.91, last_used_epoch=10),
        make_state("stale.jpg", 0.92, last_used_epoch=1),
        make_state("oldest.jpg", 0.93, last_used_epoch=0),
    ]

    assert select_easy_forced_review(states, current_epoch=12, forced_gap=10) == ["oldest.jpg", "stale.jpg"]


def test_moderate_forced_coverage_respects_configured_gap():
    states = [
        make_state("due.jpg", 0.70, last_used_epoch=2),
        make_state("overdue.jpg", 0.71, last_used_epoch=1),
        make_state("recent.jpg", 0.72, last_used_epoch=4),
    ]

    assert select_moderate_forced_coverage(states, current_epoch=5, forced_gap=3) == ["overdue.jpg", "due.jpg"]


def test_select_active_images_always_keeps_hard_samples():
    states = [
        make_state("hard.jpg", 0.20, last_used_epoch=11),
        make_state("moderate.jpg", 0.70, last_used_epoch=2),
        make_state("easy.jpg", 0.90, last_used_epoch=1),
    ]

    selected = select_active_images(
        states,
        current_epoch=12,
        easy_ratio=0.0,
        moderate_ratio=0.0,
        easy_forced_gap=10,
        moderate_forced_gap=3,
        moderate_threshold=0.55,
        easy_threshold=0.85,
    )

    assert selected == ["easy.jpg", "hard.jpg", "moderate.jpg"]

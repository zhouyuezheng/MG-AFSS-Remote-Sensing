from ultralytics.afss.io import dump_active_list, load_state, save_state


def test_state_round_trip(tmp_path):
    state_path = tmp_path / "state.json"
    sample = {"a.jpg": {"task_score": 0.7, "level": "moderate"}}
    save_state(state_path, sample)
    assert load_state(state_path) == sample


def test_load_state_returns_safe_default_for_missing_file(tmp_path):
    assert load_state(tmp_path / "missing.json") == {}


def test_dump_active_list_can_sort_paths_deterministically(tmp_path):
    active_path = tmp_path / "afss" / "active.txt"
    dump_active_list(active_path, ["b.jpg", "a.jpg"], sort_paths=True)

    assert active_path.read_text() == "a.jpg\nb.jpg\n"


def test_dump_active_list_preserves_expected_paths_exactly(tmp_path):
    active_path = tmp_path / "afss" / "active.txt"
    expected_paths = ["train/c.jpg", "train/a.jpg"]
    dump_active_list(active_path, expected_paths)

    assert active_path.read_text().splitlines() == expected_paths

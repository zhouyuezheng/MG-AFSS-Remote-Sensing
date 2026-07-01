from pathlib import Path
from types import SimpleNamespace

from ultralytics.afss.state import AFSSImageState
from ultralytics.models.yolo.detect import afss_train
from ultralytics.models.yolo.detect.afss_train import AFSSDetectionTrainer


def make_trainer_stub(tmp_path, afss: bool = True) -> AFSSDetectionTrainer:
    trainer = AFSSDetectionTrainer.__new__(AFSSDetectionTrainer)
    trainer.args = SimpleNamespace(
        afss=afss,
        afss_warmup_epochs=20,
        afss_update_interval=5,
        afss_easy_ratio=0.02,
        afss_moderate_ratio=0.40,
        afss_easy_forced_gap=10,
        afss_moderate_forced_gap=3,
        afss_conf=0.25,
        afss_save_refresh_json=False,
        afss_thresholds={"detect": [0.55, 0.85]},
    )
    trainer.save_dir = tmp_path / "detect" / "train"
    trainer.afss_dir = trainer.save_dir / "afss"
    trainer.afss_state_path = trainer.afss_dir / "state.json"
    trainer.afss_state = {}
    trainer.afss_enabled = afss
    trainer.afss_active_list_path = None
    trainer.afss_active_list_epoch = None
    trainer.batch_size = 4
    trainer.world_size = 1
    trainer.data = {"train": "train.txt"}
    trainer.train_loader = "original-loader"
    trainer.resume = False
    trainer.start_epoch = 0
    return trainer


def test_afss_trainer_writes_active_list_under_run_dir(tmp_path):
    trainer = make_trainer_stub(tmp_path)
    active_path = trainer._write_active_train_list(["b.jpg", "a.jpg"], epoch=3)

    assert active_path == trainer.afss_dir / "train_epoch0003.txt"
    assert active_path.read_text() == "a.jpg\nb.jpg\n"


def test_afss_trainer_uses_full_dataset_during_warmup():
    trainer = make_trainer_stub(tmp_path=Path("/tmp"))
    assert trainer._use_full_dataset(epoch=19) is True
    assert trainer._use_full_dataset(epoch=20) is False


def test_afss_trainer_refreshes_on_configured_interval():
    trainer = make_trainer_stub(tmp_path=Path("/tmp"))
    assert trainer._should_refresh_afss(epoch=19) is False
    assert trainer._should_refresh_afss(epoch=20) is True
    assert trainer._should_refresh_afss(epoch=21) is False
    assert trainer._should_refresh_afss(epoch=25) is True


def test_afss_trainer_saves_and_restores_state(tmp_path):
    trainer = make_trainer_stub(tmp_path)
    trainer.afss_state = {"a.jpg": AFSSImageState(im_file="a.jpg", task_score=0.7, level="moderate")}

    trainer._save_afss_state()
    trainer.afss_state = {}
    trainer._load_afss_state()

    assert trainer.afss_state["a.jpg"].task_score == 0.7
    assert trainer.afss_state["a.jpg"].level == "moderate"


def test_afss_trainer_rebuilds_only_its_train_loader(tmp_path):
    trainer = make_trainer_stub(tmp_path)
    calls = []

    def fake_get_dataloader(dataset_path, batch_size, rank, mode):
        calls.append((dataset_path, batch_size, rank, mode))
        return "afss-loader"

    trainer.get_dataloader = fake_get_dataloader
    trainer._rebuild_train_loader_from_list(trainer.afss_dir / "train_epoch0001.txt")

    assert trainer.train_loader == "afss-loader"
    assert calls == [(str(trainer.afss_dir / "train_epoch0001.txt"), 4, -1, "train")]


def test_afss_trainer_serializes_resume_metadata(tmp_path):
    trainer = make_trainer_stub(tmp_path)
    trainer.afss_state = {"a.jpg": AFSSImageState(im_file="a.jpg", task_score=0.7, level="moderate")}
    trainer.afss_active_list_epoch = 20
    trainer.afss_active_list_path = trainer.afss_dir / "train_epoch0020.txt"

    payload = trainer._get_afss_resume_metadata()

    assert payload["active_list_epoch"] == 20
    assert payload["active_list_name"] == "train_epoch0020.txt"
    assert payload["state"]["a.jpg"]["task_score"] == 0.7


def test_afss_trainer_resume_after_warmup_restores_last_active_list(tmp_path):
    trainer = make_trainer_stub(tmp_path)
    trainer.resume = True
    trainer.start_epoch = 22
    calls = []

    active_list = trainer.afss_dir / "train_epoch0020.txt"
    active_list.parent.mkdir(parents=True, exist_ok=True)
    active_list.write_text("a.jpg\n", encoding="utf-8")

    def fake_get_dataloader(dataset_path, batch_size, rank, mode):
        calls.append((dataset_path, batch_size, rank, mode))
        return "resumed-afss-loader"

    trainer.get_dataloader = fake_get_dataloader
    trainer._restore_afss_resume_state({"afss_resume": {"active_list_name": "train_epoch0020.txt", "active_list_epoch": 20}})

    assert trainer.train_loader == "resumed-afss-loader"
    assert calls == [(str(active_list), 4, -1, "train")]


def test_afss_trainer_resume_during_warmup_keeps_full_dataset_loader(tmp_path):
    trainer = make_trainer_stub(tmp_path)
    trainer.resume = True
    trainer.start_epoch = 10
    calls = []

    def fake_get_dataloader(dataset_path, batch_size, rank, mode):
        calls.append((dataset_path, batch_size, rank, mode))
        return "resumed-afss-loader"

    trainer.get_dataloader = fake_get_dataloader
    trainer._restore_afss_resume_state({"afss_resume": {"active_list_name": "train_epoch0020.txt", "active_list_epoch": 20}})

    assert trainer.train_loader == "original-loader"
    assert calls == []


def test_afss_trainer_logs_level_counts_and_deltas(tmp_path, monkeypatch):
    trainer = make_trainer_stub(tmp_path)
    trainer.afss_state = {
        "easy_a.jpg": AFSSImageState(im_file="easy_a.jpg", level="easy"),
        "easy_b.jpg": AFSSImageState(im_file="easy_b.jpg", level="easy"),
        "mid.jpg": AFSSImageState(im_file="mid.jpg", level="moderate"),
        "hard.jpg": AFSSImageState(im_file="hard.jpg", level="hard"),
    }
    messages = []

    monkeypatch.setattr(afss_train.LOGGER, "info", messages.append)
    trainer._log_afss_update_summary(
        epoch=20,
        previous_counts={"easy": 0, "moderate": 3, "hard": 1},
        previous_active_count=4,
        active_images=["easy_a.jpg", "mid.jpg", "hard.jpg"],
    )

    assert len(messages) == 1
    assert "\x1b[" in messages[0]
    assert "AFSS update epoch=20" in messages[0]
    assert "easy=2 (+2)" in messages[0]
    assert "middle=1 (-2)" in messages[0]
    assert "hard=1 (+0)" in messages[0]
    assert "active=3 (-1)" in messages[0]


def test_afss_trainer_logs_colored_startup_summary_once(tmp_path, monkeypatch):
    trainer = make_trainer_stub(tmp_path)
    trainer.afss_startup_summary_logged = False
    messages = []

    monkeypatch.setattr(afss_train.LOGGER, "info", messages.append)

    trainer._on_afss_train_start(trainer)
    trainer._on_afss_train_start(trainer)

    assert len(messages) == 1
    assert "\x1b[" in messages[0]
    assert "AFSS Summary" in messages[0]
    assert "afss_conf=0.25" in messages[0]
    assert "afss_thresholds={'detect': [0.55, 0.85]}" in messages[0]
    assert "afss_save_refresh_json=False" in messages[0]


def test_afss_trainer_uses_afss_conf_for_evaluator(tmp_path, monkeypatch):
    trainer = make_trainer_stub(tmp_path)
    trainer.callbacks = {"on_val_start": []}
    trainer.args.conf = 0.001

    captured = {}

    class FakeEvaluator:
        def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
            captured["dataloader"] = dataloader
            captured["save_dir"] = save_dir
            captured["args_conf"] = args.conf
            captured["args_split"] = args.split
            captured["callbacks"] = _callbacks

    monkeypatch.setattr(afss_train, "AFSSDetectionEvaluator", FakeEvaluator)
    trainer._get_full_train_eval_loader = lambda: "full-train-loader"

    trainer._get_afss_evaluator()

    assert captured["dataloader"] == "full-train-loader"
    assert captured["save_dir"] == trainer.afss_dir / "eval"
    assert captured["args_conf"] == 0.25
    assert captured["args_split"] == "train"
    assert captured["callbacks"] == trainer.callbacks


def test_afss_trainer_writes_sorted_refresh_snapshot(tmp_path):
    trainer = make_trainer_stub(tmp_path)
    trainer.args.afss_save_refresh_json = True
    trainer.afss_state = {
        "easy.jpg": AFSSImageState(
            im_file="easy.jpg",
            last_used_epoch=0,
            last_eval_epoch=1,
            task_score=0.95,
            level="easy",
            metrics={"box": {"precision": 0.95, "recall": 0.90}},
        ),
        "moderate.jpg": AFSSImageState(
            im_file="moderate.jpg",
            last_used_epoch=1,
            last_eval_epoch=1,
            task_score=0.60,
            level="moderate",
            metrics={"box": {"precision": 0.60, "recall": 0.70}},
        ),
        "hard.jpg": AFSSImageState(
            im_file="hard.jpg",
            last_used_epoch=1,
            last_eval_epoch=1,
            task_score=0.10,
            level="hard",
            metrics={"box": {"precision": 0.10, "recall": 0.20}},
        ),
    }

    snapshot_path = trainer._write_afss_refresh_snapshot(epoch=1, active_images=["moderate.jpg", "hard.jpg"])

    assert snapshot_path == trainer.afss_dir / "refresh_epoch0001.json"
    payload = afss_train.load_state(snapshot_path)
    assert payload["epoch"] == 1
    assert payload["thresholds"] == {"moderate": 0.55, "easy": 0.85}
    assert payload["counts"] == {"easy": 1, "moderate": 1, "hard": 1}
    assert payload["active_count"] == 2
    assert payload["active_images"] == ["hard.jpg", "moderate.jpg"]
    assert [item["im_file"] for item in payload["images"]] == ["easy.jpg", "moderate.jpg", "hard.jpg"]
    assert [item["level"] for item in payload["images"]] == ["easy", "moderate", "hard"]

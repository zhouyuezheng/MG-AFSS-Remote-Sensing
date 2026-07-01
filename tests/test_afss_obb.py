from types import SimpleNamespace

from ultralytics.afss.io import load_state
from ultralytics.afss.state import AFSSImageState
from ultralytics.models.yolo.obb.afss_train import AFSSOBBTrainer
from ultralytics.models.yolo.obb.afss_val import AFSSOBBEvaluator, aggregate_image_metrics


def make_trainer_stub(tmp_path) -> AFSSOBBTrainer:
    trainer = AFSSOBBTrainer.__new__(AFSSOBBTrainer)
    trainer.args = SimpleNamespace(
        afss=True,
        afss_warmup_epochs=20,
        afss_update_interval=5,
        afss_easy_ratio=0.02,
        afss_moderate_ratio=0.40,
        afss_easy_forced_gap=10,
        afss_moderate_forced_gap=3,
        afss_conf=0.25,
        afss_save_refresh_json=True,
        afss_thresholds={"obb": [0.55, 0.85]},
    )
    trainer.afss_task_name = "obb"
    trainer.afss_dir = tmp_path / "obb" / "train" / "afss"
    trainer.afss_state = {}
    return trainer


def test_obb_aggregate_image_metrics_counts_precision_and_recall():
    result = aggregate_image_metrics(num_gt=4, num_pred=5, matched_gt=3, matched_pred=3)
    assert result["obb"]["recall"] == 0.75
    assert result["obb"]["precision"] == 0.6


def test_obb_evaluator_collects_payload_by_image_file(tmp_path):
    evaluator = AFSSOBBEvaluator(save_dir=tmp_path)
    evaluator.store_image_metrics("sample.jpg", aggregate_image_metrics(num_gt=4, num_pred=4, matched_gt=3, matched_pred=3))

    assert evaluator.image_results["sample.jpg"]["metrics"]["obb"]["recall"] == 0.75
    assert evaluator.image_results["sample.jpg"]["task_score"] == 0.75


def test_obb_trainer_uses_rotated_thresholds_for_state_and_snapshot(tmp_path):
    trainer = make_trainer_stub(tmp_path)
    trainer.afss_state = {
        "easy.jpg": AFSSImageState(im_file="easy.jpg", task_score=0.0, level="hard"),
        "hard.jpg": AFSSImageState(im_file="hard.jpg", task_score=0.0, level="easy"),
    }

    trainer._update_afss_state_from_results(
        {
            "easy.jpg": {"metrics": {"obb": {"precision": 0.95, "recall": 0.90}}, "task_score": 0.90},
            "hard.jpg": {"metrics": {"obb": {"precision": 0.20, "recall": 0.10}}, "task_score": 0.10},
        },
        epoch=5,
    )
    snapshot_path = trainer._write_afss_refresh_snapshot(epoch=5, active_images=["hard.jpg"])

    assert trainer.afss_state["easy.jpg"].level == "easy"
    assert trainer.afss_state["hard.jpg"].level == "hard"
    assert snapshot_path == trainer.afss_dir / "refresh_epoch0005.json"
    payload = load_state(snapshot_path)
    assert payload["task"] == "obb"
    assert payload["thresholds"] == {"moderate": 0.55, "easy": 0.85}

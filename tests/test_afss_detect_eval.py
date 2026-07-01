from ultralytics.models.yolo.detect.afss_val import AFSSDetectionEvaluator, aggregate_image_metrics


def test_aggregate_image_metrics_detect_counts_precision_and_recall():
    result = aggregate_image_metrics(num_gt=4, num_pred=5, matched_gt=3, matched_pred=3)
    assert result["box"]["recall"] == 0.75
    assert result["box"]["precision"] == 0.6


def test_aggregate_image_metrics_background_only_image_is_perfect():
    result = aggregate_image_metrics(num_gt=0, num_pred=0, matched_gt=0, matched_pred=0)
    assert result["box"]["recall"] == 1.0
    assert result["box"]["precision"] == 1.0


def test_aggregate_image_metrics_no_prediction_image_has_zero_recall():
    result = aggregate_image_metrics(num_gt=3, num_pred=0, matched_gt=0, matched_pred=0)
    assert result["box"]["recall"] == 0.0
    assert result["box"]["precision"] == 0.0


def test_aggregate_image_metrics_perfect_match_image_is_perfect():
    result = aggregate_image_metrics(num_gt=2, num_pred=2, matched_gt=2, matched_pred=2)
    assert result["box"]["recall"] == 1.0
    assert result["box"]["precision"] == 1.0


def test_detect_evaluator_collects_payload_by_image_file(tmp_path):
    evaluator = AFSSDetectionEvaluator(save_dir=tmp_path)
    evaluator.store_image_metrics("sample.jpg", aggregate_image_metrics(num_gt=4, num_pred=4, matched_gt=3, matched_pred=3))

    assert evaluator.image_results["sample.jpg"]["metrics"]["box"]["recall"] == 0.75
    assert evaluator.image_results["sample.jpg"]["task_score"] == 0.75

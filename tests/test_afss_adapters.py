from ultralytics.afss.adapters import DetectAdapter, OBBAdapter, SegmentAdapter, PoseAdapter


def test_detect_adapter_uses_min_box_precision_recall():
    metrics = {"box": {"precision": 0.9, "recall": 0.7}}
    assert DetectAdapter().score(metrics) == 0.7


def test_segment_adapter_uses_joint_minimum():
    metrics = {
        "box": {"precision": 0.95, "recall": 0.92},
        "mask": {"precision": 0.80, "recall": 0.60},
    }
    assert SegmentAdapter().score(metrics) == 0.60


def test_obb_adapter_uses_rotated_metrics_minimum():
    metrics = {"obb": {"precision": 0.82, "recall": 0.73}}
    assert OBBAdapter().score(metrics) == 0.73


def test_pose_adapter_uses_joint_box_pose_minimum():
    metrics = {
        "box": {"precision": 0.93, "recall": 0.90},
        "pose": {"precision": 0.78, "recall": 0.66},
    }
    assert PoseAdapter().score(metrics) == 0.66

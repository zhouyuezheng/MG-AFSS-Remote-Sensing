# Ultralytics Configuration Patch

Add the following AFSS options to the patched Ultralytics `ultralytics/cfg/default.yaml`.

```yaml
# AFSS
afss: False
afss_warmup_epochs: 20
afss_update_interval: 5
afss_easy_ratio: 0.02
afss_moderate_ratio: 0.40
afss_easy_forced_gap: 10
afss_moderate_forced_gap: 3
afss_conf: 0.25
afss_save_refresh_json: False
afss_thresholds:
  detect: [0.55, 0.85]
  obb: [0.55, 0.85]
  segment: [0.55, 0.85]
  pose: [0.55, 0.85]
```

The submitted paper uses the HBB (`detect`) and OBB (`obb`) branches. Segment and pose thresholds are retained because the AFSS adapter layer supports them, but they are not part of the reported Remote Sensing experiments.

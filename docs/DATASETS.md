# Dataset Notes

The manuscript uses four public third-party remote-sensing object-detection datasets:

| Dataset | Task in manuscript | Notes |
|---|---|---|
| NWPU VHR-10 | HBB | Main three-seed evaluation and additional NWPU settings |
| UCAS-AOD | HBB | Other-dataset validation |
| HRSC2016 | OBB | Oriented-box boundary case |
| ShipRSImageNet | OBB | Oriented-box boundary case |

Raw imagery and annotations are not redistributed in this release candidate. Obtain each dataset from the original provider and follow its license or usage terms.

## Expected YOLO-Style Layout

The training entry expects a standard Ultralytics dataset yaml, for example:

```yaml
path: /path/to/nwpu_vhr10_full
train: images/train
val: images/val
nc: 10
names:
  0: airplane
  1: ship
  2: storage_tank
  3: baseball_diamond
  4: tennis_court
  5: basketball_court
  6: ground_track_field
  7: harbor
  8: bridge
  9: vehicle
```

For OBB experiments, labels should follow the Ultralytics YOLO OBB format expected by the patched runtime.

## Recommended Practice

- Keep raw third-party datasets outside the code repository.
- Create local `data.yaml` files for each dataset.
- Record dataset protocol decisions, such as whether background/negative images are retained.
- Do not mix results across different dataset protocols.

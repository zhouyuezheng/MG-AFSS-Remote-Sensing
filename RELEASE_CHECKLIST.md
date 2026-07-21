# Release Checklist

Updated: 2026-07-22

This checklist records what was included, excluded, and validated before preparing the public MG-AFSS repository accompanying the Remote Sensing manuscript.

## Included

| Path | Purpose |
|---|---|
| `mg_afss/` | Public training entry and path helpers |
| `ultralytics_patch/` | AFSS scheduler, HBB/OBB evaluator and trainer patch files, and config notes |
| `tools/` | Environment check, result collection, dataset-yaml helper, and resolution audit |
| `tests/` | Focused AFSS tests for scheduler, IO, HBB, and OBB behavior |
| `docs/` | Dataset and reproduction notes |
| `README.md`, `NOTICE.md`, `LICENSE`, `requirements.txt` | Public package metadata |

## Excluded

| Excluded item | Reason |
|---|---|
| Raw datasets | Third-party data should be obtained from original providers |
| Trained checkpoints and pretrained weights | Large files and potential license constraints |
| Full `experiments/` folders | Local paths, logs, and exploratory run history |
| Paper figures and processed numerical results | Withheld while the manuscript is under review |
| Full Ultralytics fork | Release candidate provides only the patch files |
| Qualitative detection manifests | Depend on local third-party images and local checkpoint paths |
| Project archives and temporary files | Not needed for reproduction |

## Validation Performed

From the release repository root:

```powershell
python -m py_compile (Get-ChildItem -Recurse -File . -Include *.py | ForEach-Object { $_.FullName })
```

Results:

- Python syntax check passed.
- Text and binary scans for local drive paths, private usernames, old local fork names, and credential-like tokens found no actionable matches outside the AGPL license text.
- No internal local-fork references were found in source files or documentation.

From the release repository root:

```powershell
python -m mg_afss.run_mg_afss_train --help
```

Results:

- `--help` exited with code 0.
- Direct execution of `tools/check_env.py` works after adding the release root to `sys.path`; without a patched runtime it reports the expected missing-runtime issue instead of an import failure.
- A temporary patched Ultralytics runtime was created by overlaying `ultralytics_patch/` on an installed Ultralytics package. In that runtime, AFSS trainer imports succeeded and the focused test suite reported `32 passed`. The local development PyTorch build still lacks `torch.nn.functional.scaled_dot_product_attention`, so formal training should use the manuscript environment or another recent PyTorch/CUDA stack.

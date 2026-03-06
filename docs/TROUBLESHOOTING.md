# Troubleshooting

- CUDA OOM: reduce batch size, enable gradient accumulation, or lower resolution.
- Path issues: keep `configs/config_local.py` locally (not committed).
- Determinism: fix seeds and keep `pip_freeze.txt` from runs.
- Filenames: avoid spaces/brackets in script names.

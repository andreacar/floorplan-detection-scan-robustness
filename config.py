"""Compatibility shim.

The configuration source of truth lives in `configs/config.py`, but many modules
(including the external RT-DETR code) import `config` from the repo root.
Keep that import stable by re-exporting everything.
"""

from configs.config import *  # noqa: F401,F403

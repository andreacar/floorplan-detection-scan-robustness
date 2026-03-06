from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Optional


def _maybe_prepend_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

def _forget_module(prefix: str) -> None:
    # Ensure import resolution re-evaluates sys.path (especially for namespace packages).
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


def _has_main_3_experiments(parent: Path) -> bool:
    return (parent / "RT_DETR_final" / "main_3_experiments.py").is_file()


def _candidate_parents(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []

    env = os.environ.get("RT_DETR_FINAL_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser())

    # Common local layouts.
    candidates.append(repo_root)
    # Sibling checkout (e.g. ../RT_DETR_final or ../SomeRepo/RT_DETR_final).
    if repo_root.parent.is_dir():
        candidates.append(repo_root.parent)
        try:
            for child in repo_root.parent.iterdir():
                if not child.is_dir():
                    continue
                # Keep this cheap: only consider directories that already contain the package.
                if (child / "RT_DETR_final").is_dir():
                    candidates.append(child)
        except OSError:
            # Best-effort scanning; ignore permission/IO issues.
            pass
    candidates.append(repo_root / "external")
    candidates.append(repo_root / "third_party")
    candidates.append(repo_root / "vendor")

    return candidates


def _ensure_rtdetr_final_importable(repo_root: Path) -> Optional[Path]:
    """
    Ensure the `RT_DETR_final` package can be imported.

    Returns the path that was added to sys.path, if any.
    """
    try:
        importlib.import_module("RT_DETR_final")
        return None
    except ModuleNotFoundError:
        pass

    for parent in _candidate_parents(repo_root):
        # If the env var points directly at the package dir, use its parent.
        if parent.is_dir() and (parent / "__init__.py").is_file() and parent.name == "RT_DETR_final":
            _maybe_prepend_sys_path(parent.parent)
        else:
            pkg_dir = parent / "RT_DETR_final"
            if pkg_dir.is_dir():
                _maybe_prepend_sys_path(parent)
                return parent

        try:
            importlib.import_module("RT_DETR_final")
            return parent
        except ModuleNotFoundError:
            continue

    return None


def import_core_main_3_experiments(repo_root: str | Path):
    repo_root_path = Path(repo_root).resolve()

    # If the caller provided an explicit checkout root, prefer it (and ensure legacy
    # imports like `eval.*` work when `eval/` lives under `RT_DETR_final/`).
    env_root = os.environ.get("RT_DETR_FINAL_DIR", "").strip()
    if env_root:
        env_root_path = Path(env_root).expanduser()
        if env_root_path.is_dir():
            _maybe_prepend_sys_path(env_root_path)
            if (env_root_path / "RT_DETR_final").is_dir():
                _maybe_prepend_sys_path(env_root_path / "RT_DETR_final")

    added = _ensure_rtdetr_final_importable(repo_root_path)
    # Some research repos import `data.*`, `models.*`, etc. as top-level packages.
    # In the CubiCasaVec layout those live under `RT_DETR_final/`, so we also add
    # that directory itself to sys.path when present.
    try:
        rtdetr_pkg = importlib.import_module("RT_DETR_final")
        rtdetr_paths = list(getattr(rtdetr_pkg, "__path__", []))
        if rtdetr_paths:
            rtdetr_dir = Path(rtdetr_paths[0]).resolve()
            # Prefer adding the directory above RT_DETR_final, since in the CubiCasaVec layout
            # top-level modules like `data/` and `eval/` live next to the package.
            cubic_root = rtdetr_dir.parent
            cubic_root_path = str(cubic_root)
            _maybe_prepend_sys_path(cubic_root)

            # Some checkouts keep `data/` and/or `eval/` *inside* RT_DETR_final/.
            # Add that directory too when present so imports like `import eval.*` resolve.
            if (rtdetr_dir / "data").is_dir() or (rtdetr_dir / "eval").is_dir():
                _maybe_prepend_sys_path(rtdetr_dir)
    except ModuleNotFoundError:
        # Will be handled below with a clearer error.
        pass
    try:
        return importlib.import_module("RT_DETR_final.main_3_experiments")
    except ModuleNotFoundError as e:
        # If `RT_DETR_final` is importable but incomplete (e.g. this repo includes only
        # a subset like `experiment_recipes/`), try to locate a full checkout that
        # contains `main_3_experiments.py` and re-import from there.
        missing_name = getattr(e, "name", "") or ""
        if missing_name in {"RT_DETR_final.main_3_experiments", "RT_DETR_final"}:
            for parent in _candidate_parents(repo_root_path):
                if not parent.is_dir():
                    continue
                if not _has_main_3_experiments(parent):
                    continue
                _maybe_prepend_sys_path(parent)
                if (parent / "RT_DETR_final").is_dir():
                    _maybe_prepend_sys_path(parent / "RT_DETR_final")
                _forget_module("RT_DETR_final")
                try:
                    return importlib.import_module("RT_DETR_final.main_3_experiments")
                except ModuleNotFoundError as inner:
                    # If we found the file but importing it fails due to a different missing
                    # dependency (numpy/torch/etc.), surface the real missing module rather
                    # than masking it as "RT_DETR_final missing".
                    inner_name = getattr(inner, "name", "") or ""
                    if inner_name and inner_name not in {"RT_DETR_final", "RT_DETR_final.main_3_experiments"}:
                        msg = (
                            "Found `RT_DETR_final/main_3_experiments.py`, but importing it failed.\n"
                            f"Missing Python module: {inner_name}\n\n"
                            "Fix: activate the correct environment (see `environment.yml`) and/or install the\n"
                            "dependencies required by your RT-DETR checkout.\n"
                        )
                        raise ModuleNotFoundError(msg) from inner
                    continue

        # If the missing module is not RT_DETR_final itself, surface a better hint.
        if missing_name and missing_name not in {"RT_DETR_final", "RT_DETR_final.main_3_experiments"}:
            # Special-case the common research-repo pattern where `data` lives under RT_DETR_final/.
            if missing_name == "data":
                hint = (
                    "Imported `RT_DETR_final`, but it failed to import `data.*`.\n"
                    "This usually means the RT_DETR repo expects to be run from its own project root.\n\n"
                    "Fix: ensure the directory that contains `data/` is on `sys.path`.\n"
                    "For CubiCasaVec this is typically `<...>/RT_DETR_final/`.\n"
                )
                raise ModuleNotFoundError(hint) from e
            raise

        searched = [
            str(p.resolve()) for p in _candidate_parents(repo_root_path) if p.exists()
        ]
        hint = ""
        try:
            pkg = importlib.import_module("RT_DETR_final")
            pkg_paths = list(getattr(pkg, "__path__", []))
            if pkg_paths and not (Path(pkg_paths[0]) / "main_3_experiments.py").is_file():
                hint += (
                    "Found `RT_DETR_final`, but it does not contain `main_3_experiments.py`.\n"
                    "This repo snapshot appears to vendor only partial RT-DETR artifacts.\n\n"
                )
        except Exception:
            pass
        hint = (
            hint
            + "Missing dependency `RT_DETR_final`.\n\n"
            + "Expected one of these layouts:\n"
            + f"- {repo_root_path}/RT_DETR_final/\n"
            + f"- {repo_root_path}/external/RT_DETR_final/\n"
            + f"- {repo_root_path}/third_party/RT_DETR_final/\n\n"
            + "Fix options:\n"
            + "1) Clone/copy the RT_DETR_final code into one of the folders above.\n"
            + "2) Or set `RT_DETR_FINAL_DIR` to the directory that *contains* `RT_DETR_final/`.\n\n"
            + f"Repo root: {repo_root_path}\n"
            + f"Searched parents: {searched}\n"
        )
        if added is not None:
            hint += f"Added to sys.path: {added}\n"
        raise ModuleNotFoundError(hint) from e
    except Exception as e:  # pragma: no cover
        msg = str(e)
        if isinstance(e, ValueError) and "BASE_DIR is not set" in msg:
            hint = (
                f"{msg}\n\n"
                "Fix options:\n"
                "1) Set `DATASET_BASE_DIR` (or `BASE_DIR`) to your CubiCasa dataset root.\n"
                "2) Or set `DEFAULT_BASE_DIR` in `configs/config.py`.\n"
                "3) Or fetch the published Zenodo dataset once into the default cache.\n\n"
                "Examples:\n"
                "- python training/train.py --preset clean --base-dir /path/to/cubicasa\n"
                "- python training/train.py --preset clean --set DATASET_BASE_DIR=/path/to/cubicasa\n"
                "- python tools/fetch_dataset.py\n"
            )
            raise ValueError(hint) from e
        raise

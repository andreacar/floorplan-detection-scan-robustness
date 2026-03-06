import os
from pathlib import Path

def get_artifacts_root():
    root = os.environ.get("ARTIFACTS_DIR", "artifacts")
    return Path(root)

def figures_dir():
    d = get_artifacts_root() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d

def tables_dir():
    d = get_artifacts_root() / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d

def figure_path(name):
    p = figures_dir() / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def table_path(name):
    p = tables_dir() / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

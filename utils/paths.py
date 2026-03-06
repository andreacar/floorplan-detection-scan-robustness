import os
from typing import List
from config import BASE_DIR

DATA_ROOTS = {
    "colorful": os.path.join(BASE_DIR, "colorful"),
    "high_quality": os.path.join(BASE_DIR, "high_quality"),
    "high_quality_architectural": os.path.join(BASE_DIR, "high_quality_architectural"),
}


def resolve_path(txt_line: str):
    line = txt_line.strip()
    if not line:
        return None
    if line.endswith("/"):
        line = line[:-1]
    if line.startswith("/"):
        line = line[1:]

    parts = line.split("/")
    dataset = parts[0]
    rest = "/".join(parts[1:])
    if dataset not in DATA_ROOTS:
        return None
    return os.path.join(DATA_ROOTS[dataset], rest)


def load_split_list(path: str) -> List[str]:
    out = []
    with open(path, "r") as f:
        for line in f:
            r = resolve_path(line)
            if r and os.path.isdir(r):
                out.append(r)
            else:
                print(f"[SKIP] {line.strip()}")
    return out

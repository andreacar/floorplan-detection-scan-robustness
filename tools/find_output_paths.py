import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "paper_experiments",
    ROOT / "evaluation",
    ROOT / "training",
    ROOT / "mitigation",
]

PATTERNS = [
    r"savefig\(",
    r"to_csv\(",
    r"write_text\(",
    r"open\(",
    r"np\.save",
    r"torch\.save",
    r"artifacts",
    r"outputs",
    r"figures",
    r"tables",
    r"paper_fig",
    r"paper_tab",
    r"runs/",
]

hits = []

for base in TARGETS:
    if not base.exists():
        continue
    for py in base.rglob("*.py"):
        txt = py.read_text(encoding="utf-8", errors="replace")
        if any(re.search(p, txt) for p in PATTERNS):
            strings = re.findall(
                r"[\"']([^\"']*(?:fig|plot|table|output|artifact|run|save|dump|result)[^\"']*)[\"']",
                txt,
                flags=re.I,
            )
            strings = [x for x in strings if len(x) < 140]
            hits.append((str(py.relative_to(ROOT)), sorted(set(strings))[:20]))

print("# Potential output path hints (file -> strings)")
for f, ss in hits:
    print("\n==", f)
    for x in ss:
        print("  ", x)

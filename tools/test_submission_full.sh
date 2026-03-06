#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-gdino}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/5] Strict submission quickcheck"
conda run -n "$ENV_NAME" python tools/quickcheck_submission.py --strict

echo "[2/5] Paper runner dry-check"
conda run -n "$ENV_NAME" python tools/run_paper.py paper_all --dry

echo "[3/5] Variant-generation smoke check (1 folder)"
CONFIG_OUT="$(conda run -n "$ENV_NAME" python -c "import config as c; print(c.BASE_DIR); print(c.TEST_TXT)")"
BASE_DIR="$(printf '%s\n' "$CONFIG_OUT" | sed -n '1p')"
TEST_TXT_PATH="$(printf '%s\n' "$CONFIG_OUT" | sed -n '2p')"

if [[ ! -f "$TEST_TXT_PATH" ]]; then
  echo "TEST_TXT not found: $TEST_TXT_PATH"
  exit 1
fi

SMOKE_FOLDER=""
while IFS= read -r line; do
  [[ -z "${line// }" ]] && continue
  rel="${line#/}"
  rel="${rel%/}"
  abs="$BASE_DIR/$rel"

  if [[ -f "$abs/graph.json" && -f "$abs/F1_scaled.png" && -f "$abs/model_baked.png" ]]; then
    SMOKE_FOLDER="$abs"
    break
  fi
done < "$TEST_TXT_PATH"

if [[ -z "$SMOKE_FOLDER" ]]; then
  echo "No folder found for variant-generation smoke test."
  exit 1
fi

echo "Using folder: $SMOKE_FOLDER"
conda run -n "$ENV_NAME" python diagnostic/render_4_variants.py --root-data "$SMOKE_FOLDER" --save-mode dataset --jobs 1

for fn in \
  01_svg_clean.png \
  02_scan_raw.png \
  03_scan_inside_boxes.png \
  04_svg_clean_plus_scan_outside.png \
  05_scan_boxes_polygon.png; do
  if [[ ! -f "$SMOKE_FOLDER/four_final_variants/$fn" ]]; then
    echo "Missing generated variant: $SMOKE_FOLDER/four_final_variants/$fn"
    exit 1
  fi
done

echo "[4/5] Checkpoint inference smoke check (A/B on C/D for 1 test sample)"
SMOKE_PICKER="/tmp/pick_submission_smoke_line.py"
cat > "$SMOKE_PICKER" <<'PY'
import os
import config as c
from utils.paths import resolve_path
from transformers import AutoImageProcessor
from data.dataset import GraphRTDetrDataset

processor = AutoImageProcessor.from_pretrained(c.BACKBONE)
for raw in open(c.TEST_TXT, "r", encoding="utf-8"):
    line = raw.strip()
    if not line:
        continue
    p = resolve_path(line)
    if not p:
        continue
    if not (
        os.path.isfile(os.path.join(p, "graph.json"))
        and os.path.isfile(os.path.join(p, "four_final_variants/03_scan_inside_boxes.png"))
        and os.path.isfile(os.path.join(p, "four_final_variants/04_svg_clean_plus_scan_outside.png"))
    ):
        continue
    try:
        c.IMAGE_FILENAME = "four_final_variants/03_scan_inside_boxes.png"
        ds = GraphRTDetrDataset([p], processor, "hierarchy_config.py", augment=False)
        if len(ds) > 0:
            print(line)
            break
    except Exception:
        continue
PY

SMOKE_LINE="$(conda run -n "$ENV_NAME" python "$SMOKE_PICKER" | grep '^/' | head -n 1 || true)"
rm -f "$SMOKE_PICKER"

if [[ -z "$SMOKE_LINE" ]]; then
  echo "No test sample with C/D variants found in TEST_TXT."
  exit 1
fi

SMOKE_TXT="/tmp/test_submission_smoke.txt"
printf '%s\n' "$SMOKE_LINE" > "$SMOKE_TXT"
OUT_DIR="/tmp/ab_on_cd_smoke_submission"
rm -rf "$OUT_DIR"

TEST_TXT="$SMOKE_TXT" conda run -n "$ENV_NAME" python -m evaluation.test_AB_on_CD \
  --model-a-dir stable_runs/20260115_134958/exp1_clean/checkpoints/best \
  --model-b-dir stable_runs/20260115_134958/exp2_scanned/checkpoints/best \
  --out-dir "$OUT_DIR"

echo "[5/5] Smoke metrics summary"
python3 - <<'PY'
import glob
import json
import os

out_dir = '/tmp/ab_on_cd_smoke_submission'
files = sorted(glob.glob(os.path.join(out_dir, '*_summary.json')))
if not files:
    raise SystemExit('No summary files found in ' + out_dir)

for p in files:
    d = json.load(open(p, 'r', encoding='utf-8'))
    o = d.get('overall', {})
    print(os.path.basename(p))
    print(f"  AP={o.get('AP', 0.0):.6f} AP50={o.get('AP50', 0.0):.6f} AP75={o.get('AP75', 0.0):.6f} AR100={o.get('AR_100', 0.0):.6f}")
PY

echo "All submission smoke checks passed."

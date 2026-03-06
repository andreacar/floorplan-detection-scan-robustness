#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import os
import sys

import pandas as pd
import statsmodels.formula.api as smf


def main():
    path = os.path.join("paper_experiments", "out", "degradation", "degradation_sweep.json")
    if not os.path.exists(path):
        print(f"Missing {path}; run degradation_sweep.py first.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        records = pd.DataFrame(json.load(f))

    if records.empty:
        print("No degradation sweep records found (empty file).", file=sys.stderr)
        sys.exit(1)

    model = smf.mixedlm("recall85_deg ~ dist_roi", records, groups=records["layout_id"])
    result = model.fit()
    print(result.summary())


if __name__ == "__main__":
    main()

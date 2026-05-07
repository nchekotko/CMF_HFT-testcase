"""Build figures + tables from a list of result dirs.

Default behaviour: pick up V1/V2 (and V3 if present), point each at its
``summary.pkl``, and use V2's ``calibration.pkl`` for the calibration plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cmf_mm.reports.generate import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report artefacts")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out", default="results/report")
    args = parser.parse_args()

    root = Path(args.results_root)
    summary_paths: dict[str, Path] = {}
    candidates = [("V1 vanilla", "v1"), ("V2 microprice", "v2"), ("V3 asymmetric", "v3")]
    for label, sub in candidates:
        p = root / sub / "summary.pkl"
        if p.exists():
            summary_paths[label] = p
    if not summary_paths:
        raise SystemExit(f"No summary.pkl files found under {root}/v1, v2, v3")

    cal_path = root / "v2" / "calibration.pkl"
    if not cal_path.exists():
        cal_path = next(iter(p.parent / "calibration.pkl" for p in summary_paths.values()), None)
        if cal_path is None or not cal_path.exists():
            cal_path = None

    generate_report(summary_paths=summary_paths, calibration_path=cal_path, out_dir=args.out)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()

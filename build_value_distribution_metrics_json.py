import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKBOOK_PATH = ROOT / "data" / "results_raw_data_with_category_division.xlsx"
OUT_PATH = ROOT / "data" / "value_distribution_metrics.json"

# Four-category mapping used in Figure 1.
CATEGORY_MAP = {
    "Normal": "Normal",
    "Skewed": "Skewed",
    "Uniform": "Uniform",
    "Two-level uniform": "Multi regime",
    "2-level uniform": "Multi regime",
    "Bimodal": "Multi regime",
    "Multimodal": "Multi regime",
}

ORDER = ["Normal", "Skewed", "Uniform", "Multi regime"]


def main() -> None:
    df = pd.read_excel(WORKBOOK_PATH, sheet_name="Correlations")
    df.columns = [str(c).strip() for c in df.columns]
    df["Data distribution"] = df["Data distribution"].astype(str).str.strip()

    # Only include "without spike" rows as requested.
    spike_col = "Dominant spike"
    if spike_col in df.columns:
        df[spike_col] = df[spike_col].astype(str).str.strip().str.lower()
        df = df[df[spike_col] == "no"].copy()

    df["distribution_group"] = df["Data distribution"].map(CATEGORY_MAP)
    filtered = df[df["distribution_group"].notna()].copy()

    for col in ["nrmse", "nmae", "RTA1pct", "RTA5pct", "RTA10pct"]:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

    grouped = (
        filtered.groupby("distribution_group")
        .agg(
            n=("distribution_group", "size"),
            nrmse=("nrmse", "mean"),
            nmae=("nmae", "mean"),
            rta1_pct=("RTA1pct", lambda s: float(s.mean() * 100.0)),
            rta5_pct=("RTA5pct", lambda s: float(s.mean() * 100.0)),
            rta10_pct=("RTA10pct", lambda s: float(s.mean() * 100.0)),
        )
        .reset_index()
    )

    by_name = {row["distribution_group"]: row for _, row in grouped.iterrows()}

    categories = []
    for name in ORDER:
        row = by_name.get(name)
        if row is None:
            categories.append(
                {
                    "name": name,
                    "n": 0,
                    "nrmse": None,
                    "nmae": None,
                    "rta1_pct": None,
                    "rta5_pct": None,
                    "rta10_pct": None,
                }
            )
            continue

        categories.append(
            {
                "name": name,
                "n": int(row["n"]),
                "nrmse": float(row["nrmse"]),
                "nmae": float(row["nmae"]),
                "rta1_pct": float(row["rta1_pct"]),
                "rta5_pct": float(row["rta5_pct"]),
                "rta10_pct": float(row["rta10_pct"]),
            }
        )

    payload = {
        "source_file": str(WORKBOOK_PATH.name),
        "source_sheet": "Correlations",
        "source_filter": "Dominant spike = No",
        "order": ORDER,
        "category_rules": "Uniform = Uniform only; Multi regime = Bimodal + Multimodal + Two-level uniform + 2-level uniform; excluded from this figure: Discrete and Too little continuous data; only rows without dominant spike are included",
        "categories": categories,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

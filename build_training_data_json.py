import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_PARQUET = Path("e:/Surrogate_model_test_runs/surrogate_test_sabine/data/SSEREP 10K/Model_Results.parquet")
OUTPUT_JSON = Path("e:/Surrogate_model_test_runs/surrogate_models_multi-sector_ESOM/data/training_value_distributions.json")


def safe_float(value):
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def summarize_outcome(group: pd.DataFrame):
    values = pd.to_numeric(group["value"], errors="coerce").dropna().astype(float)
    if values.empty:
        return None

    std = float(values.std())
    if std <= 0:
        return None

    counts = values.value_counts()
    recurring = sorted(float(v) for v, c in counts.items() if c >= 20)

    # Keep a fixed 50-bin histogram, similar to main.py.
    hist_counts, edges = np.histogram(values, bins=50)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()

    unit = ""
    if "Unit" in group.columns:
        unit = str(group["Unit"].iloc[0] or "")

    tech = ""
    if "Technology_name" in group.columns:
        tech = str(group["Technology_name"].iloc[0] or "")

    tech_id = ""
    if "technology" in group.columns:
        tech_id = str(group["technology"].iloc[0] or "")

    return {
        "outcome": str(group["Outcome"].iloc[0]),
        "technology_name": tech,
        "technology_id": tech_id,
        "unit": unit,
        "n": int(values.shape[0]),
        "mean": safe_float(values.mean()),
        "std": safe_float(std),
        "min": safe_float(values.min()),
        "max": safe_float(values.max()),
        "bin_centers": [safe_float(v) for v in centers],
        "bin_counts": [int(v) for v in hist_counts.tolist()],
        "recurring_values": [safe_float(v) for v in recurring[:25]],
    }


def main():
    if not SOURCE_PARQUET.exists():
        raise FileNotFoundError(f"Source parquet not found: {SOURCE_PARQUET}")

    df = pd.read_parquet(SOURCE_PARQUET, columns=["Outcome", "value", "Technology_name", "Unit", "technology"])
    df = df.dropna(subset=["Outcome"])

    targets = []
    for _, group in df.groupby("Outcome", sort=True):
        summary = summarize_outcome(group)
        if summary is not None:
            targets.append(summary)

    payload = {
        "source": str(SOURCE_PARQUET),
        "target_count_with_variation": len(targets),
        "targets": targets,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON} with {len(targets)} varying targets")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd


PRED_DIR = Path(
    "e:/Surrogate_model_test_runs/surrogate_test_sabine/Single_target/outputs/test regression methods/pred_vs_actual_plots"
)
HYBRID_PRED_DIR = Path(
    "e:/Surrogate_model_test_runs/surrogate_test_sabine/Single_target/outputs/effect adding classifier/pred_vs_actual_plots/XGB_HYBRID"
)
JOINT_LOG_CSV = Path(
    "e:/Surrogate_model_test_runs/surrogate_models_multi-sector_ESOM/data/joint_learning_informed_classifier_run_log.csv"
)
OUTPUT_JSON = Path(
    "e:/Surrogate_model_test_runs/surrogate_models_multi-sector_ESOM/data/regression_constraint_violations.json"
)

METHODS = ["XGB_single", "AE_MLP", "AE_GaussianProcess"]
HYBRID_METHOD = "XGB_HYBRID_tuned"
METHOD_LABELS = {
    "XGB_single": "XGB",
    "AE_MLP": "MLP",
    "AE_GaussianProcess": "GP",
    HYBRID_METHOD: "XGB+classifier",
    "Informed_XGB_HYBRID": "informed_classifier",
}

CONSTRAINT_EPS = 1e-6


def parse_file_name(name: str, allowed_methods=None):
    if not name.endswith(".csv"):
        return None
    stem = name[:-4]
    parts = stem.split("_")
    if len(parts) < 6:
        return None

    var = parts[0]
    if var not in {"techStocks", "techUseNet"}:
        return None

    period = parts[1]
    tech = f"{parts[2]}_{parts[3]}"
    commodity = parts[4]
    method = "_".join(parts[5:])
    if allowed_methods is not None and method not in allowed_methods:
        return None

    return {
        "var": var,
        "period": period,
        "tech": tech,
        "commodity": commodity,
        "method": method,
    }


def load_cap2act_lookup():
    import importlib.util

    module_path = Path("e:/Surrogate_model_test_runs/surrogate_test_sabine/Single_target/cap2act_constraints.py")
    spec = importlib.util.spec_from_file_location("cap2act_constraints", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_cap2act


def load_y_pred(path: Path):
    try:
        df = pd.read_csv(path, usecols=["y_pred"])
    except Exception:
        return None
    vals = pd.to_numeric(df["y_pred"], errors="coerce").dropna().to_numpy(dtype=float)
    return vals if len(vals) else None


def build_index(pred_dir: Path, methods):
    index = {m: {} for m in methods}
    allowed = set(methods)
    for path in pred_dir.glob("*.csv"):
        meta = parse_file_name(path.name, allowed_methods=allowed)
        if not meta:
            continue
        key = (meta["period"], meta["tech"], meta["commodity"])
        index[meta["method"]].setdefault(key, {})[meta["var"]] = path
    return index


def method_pair_keys(index, method):
    pairs = index.get(method, {})
    return {k for k, d in pairs.items() if "techStocks" in d and "techUseNet" in d}


def compute_violation_for_pairs(index, method, pair_keys, get_cap2act):
    pair_violations = []
    for key in sorted(pair_keys):
        period, tech, _commodity = key
        stock_path = index[method][key]["techStocks"]
        use_path = index[method][key]["techUseNet"]

        y_stock = load_y_pred(stock_path)
        y_use = load_y_pred(use_path)
        if y_stock is None or y_use is None:
            continue

        n = min(len(y_stock), len(y_use))
        if n <= 0:
            continue

        y_stock = y_stock[:n]
        y_use = y_use[:n]

        try:
            k_val = float(get_cap2act(tech))
        except Exception:
            continue

        viol_pct = 100.0 * float((y_use > (k_val * y_stock + CONSTRAINT_EPS)).sum()) / float(n)
        pair_violations.append(viol_pct)

    mean_viol = float(np.mean(pair_violations)) if pair_violations else None
    return len(pair_violations), mean_viol


def compute_violation_detail_map(index, method, pair_keys, get_cap2act):
    details = {}
    for key in sorted(pair_keys):
        period, tech, _commodity = key
        stock_path = index[method][key]["techStocks"]
        use_path = index[method][key]["techUseNet"]

        y_stock = load_y_pred(stock_path)
        y_use = load_y_pred(use_path)
        if y_stock is None or y_use is None:
            continue

        n = min(len(y_stock), len(y_use))
        if n <= 0:
            continue

        y_stock = y_stock[:n]
        y_use = y_use[:n]

        try:
            k_val = float(get_cap2act(tech))
        except Exception:
            continue

        viol_pct = 100.0 * float((y_use > (k_val * y_stock + CONSTRAINT_EPS)).sum()) / float(n)
        details[(period, tech)] = {
            "pair_key": [period, tech],
            "constraint_violations_pct": float(viol_pct),
            "violates_constraint": bool(viol_pct > 0.0),
        }
    return details


def select_one_key_per_period_tech(pair_keys, allowed_period_tech):
    chosen = {}
    for key in sorted(pair_keys):
        period, tech, _commodity = key
        pt = (period, tech)
        if pt not in allowed_period_tech:
            continue
        if pt not in chosen:
            chosen[pt] = key
    return set(chosen.values())


def parse_period_tech_from_target(target_variable: str):
    parts = str(target_variable or "").split()
    if len(parts) < 3:
        return None
    var = parts[0]
    if var not in {"techStocks", "techUseNet"}:
        return None
    period = parts[1]
    tech = parts[2]
    return var, period, tech


def load_informed_violation_map(path: Path):
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    rows = df[df["surrogate_method"].astype(str) == "Informed_XGB_HYBRID"]

    pairs = {}
    for _, row in rows.iterrows():
        parsed = parse_period_tech_from_target(row.get("target_variable"))
        if not parsed:
            continue
        var, period, tech = parsed
        key = (str(period), str(tech))
        entry = pairs.setdefault(key, {"has_stock": False, "has_use": False, "viol_pct": None})
        if var == "techStocks":
            entry["has_stock"] = True
        elif var == "techUseNet":
            entry["has_use"] = True
            v = pd.to_numeric(pd.Series([row.get("constraint_violations_pct")]), errors="coerce").iloc[0]
            if pd.notna(v):
                entry["viol_pct"] = float(v)

    informed_map = {}
    for key, entry in pairs.items():
        if entry["has_stock"] and entry["has_use"] and entry["viol_pct"] is not None:
            informed_map[key] = entry["viol_pct"]
    return informed_map


def normalize_pct_values(values):
    if not values:
        return values
    max_v = max(values)
    if max_v <= 1.0:
        return [v * 100.0 for v in values]
    return values


def main():
    if not PRED_DIR.exists():
        raise FileNotFoundError(f"Prediction folder not found: {PRED_DIR}")
    if not HYBRID_PRED_DIR.exists():
        raise FileNotFoundError(f"Hybrid prediction folder not found: {HYBRID_PRED_DIR}")

    get_cap2act = load_cap2act_lookup()

    index = build_index(PRED_DIR, METHODS)

    hybrid_index = build_index(HYBRID_PRED_DIR, [HYBRID_METHOD])
    hybrid_pairs = method_pair_keys(hybrid_index, HYBRID_METHOD)
    hybrid_pairs_period_tech = {(p, t) for (p, t, _c) in hybrid_pairs}

    informed_viol_map = load_informed_violation_map(JOINT_LOG_CSV)
    informed_pairs = set(informed_viol_map.keys())

    common_hybrid_informed = hybrid_pairs_period_tech.intersection(informed_pairs)

    method_rows = []
    comparable_detail_rows = {}
    for method in METHODS:
        method_keys = method_pair_keys(index, method)
        comparable_keys = select_one_key_per_period_tech(method_keys, common_hybrid_informed)
        used_pairs, mean_viol = compute_violation_for_pairs(index, method, comparable_keys, get_cap2act)
        comparable_detail_rows[method] = compute_violation_detail_map(index, method, comparable_keys, get_cap2act)
        method_rows.append(
            {
                "method_key": method,
                "method": METHOD_LABELS.get(method, method),
                "pair_count": used_pairs,
                "constraint_violations_pct": mean_viol,
            }
        )

    hybrid_common_keys = select_one_key_per_period_tech(hybrid_pairs, common_hybrid_informed)
    hybrid_used_pairs, hybrid_mean_viol = compute_violation_for_pairs(
        hybrid_index,
        HYBRID_METHOD,
        hybrid_common_keys,
        get_cap2act,
    )
    comparable_detail_rows[HYBRID_METHOD] = compute_violation_detail_map(
        hybrid_index,
        HYBRID_METHOD,
        hybrid_common_keys,
        get_cap2act,
    )

    informed_vals = normalize_pct_values([informed_viol_map[k] for k in common_hybrid_informed if k in informed_viol_map])
    informed_mean_viol = float(np.mean(informed_vals)) if informed_vals else None

    comparable_rows = []
    for period, tech in sorted(common_hybrid_informed):
        xgb_detail = comparable_detail_rows.get("XGB_single", {}).get((period, tech))
        hybrid_detail = comparable_detail_rows.get(HYBRID_METHOD, {}).get((period, tech))
        informed_pct_raw = informed_viol_map.get((period, tech))
        informed_pct = None
        if informed_pct_raw is not None:
            normalized_vals = normalize_pct_values([informed_pct_raw])
            informed_pct = normalized_vals[0] if normalized_vals else None

        comparable_rows.append(
            {
                "period": period,
                "technology": tech,
                "XGB": {
                    "constraint_violations_pct": xgb_detail["constraint_violations_pct"] if xgb_detail else None,
                    "violates_constraint": bool(xgb_detail and xgb_detail["violates_constraint"]),
                },
                "XGB+classifier": {
                    "constraint_violations_pct": hybrid_detail["constraint_violations_pct"] if hybrid_detail else None,
                    "violates_constraint": bool(hybrid_detail and hybrid_detail["violates_constraint"]),
                },
                "informed_classifier": {
                    "constraint_violations_pct": informed_pct,
                    "violates_constraint": bool(informed_pct is not None and informed_pct > 0.0),
                },
            }
        )

    method_rows.extend(
        [
            {
                "method_key": HYBRID_METHOD,
                "method": METHOD_LABELS.get(HYBRID_METHOD, HYBRID_METHOD),
                "pair_count": hybrid_used_pairs,
                "constraint_violations_pct": hybrid_mean_viol,
            },
            {
                "method_key": "Informed_XGB_HYBRID",
                "method": METHOD_LABELS.get("Informed_XGB_HYBRID", "Informed_XGB_HYBRID"),
                "pair_count": len(common_hybrid_informed),
                "constraint_violations_pct": informed_mean_viol,
            },
        ]
    )

    payload = {
        "definition": "100 * mean over common tech-period pairs of fraction(y_pred_use > K * y_pred_stock + 1e-6)",
        "common_pair_count": len(common_hybrid_informed),
        "methods": method_rows,
        "comparable_pair_rows": comparable_rows,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON} with {len(common_hybrid_informed)} common pairs")


if __name__ == "__main__":
    main()

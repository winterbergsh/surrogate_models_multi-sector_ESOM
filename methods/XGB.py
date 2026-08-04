import os
import time
import numpy as np
import torch
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    make_scorer,
)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

from Single_target.settings_single import (
    SAVE_DIR,
    SEED,
    MAX_TRIALS,
    N_SPLITS,
    TEST_SIZE,
    get_tech_unit,
    get_scaler,
    X_SCALING,
    get_save_dir,
)

from Single_target.data_prep_single import prepare_data
from Single_target.results_summary import log_run, calculate_smape

# ============ user switches ============
N_ITER_TUNING = MAX_TRIALS        # number of random param samples
N_FOLDS_CV = N_SPLITS            # CV folds
# =======================================

# Predefined default XGB params (if tuning is disabled)
DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 10.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # "n_jobs": 8,
}

# Use method-specific SAVE_DIR for XGB (test regression methods)
_SAVE_DIR = get_save_dir("XGB_single")

# folder for trained models (consistent with AE)
TRAINED_DIR = os.path.join(_SAVE_DIR, "trained_models")
os.makedirs(TRAINED_DIR, exist_ok=True)

# folder for prediction-vs-real data and plots (consistent with AE)
PVR_DIR = os.path.join(_SAVE_DIR, "pred_vs_actual_plots")
os.makedirs(PVR_DIR, exist_ok=True)


def safe(x):
    """Make a string safe for use in paths / filenames."""
    return "none" if x is None else str(x).replace("/", "_").replace(" ", "_")


def benchmark_xgb_predict(model, X, n_repeats: int = 100):
    """
    Benchmark XGBoost model.predict on a given feature matrix X.

    Returns:
        per_sample  (seconds per sample)
    """
    # Warm-up
    _ = model.predict(X)

    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = model.predict(X)
    elapsed = time.perf_counter() - start

    batch_time = elapsed / n_repeats
    per_sample = batch_time / X.shape[0]

    print("\n=== XGB Inference Runtime ===")
    print(f"Batch time (predict):   {batch_time:.6f} s")
    print(f"Per-sample (predict):   {per_sample:.3e} s")

    return per_sample


def set_seeds(seed: int = 7) -> None:
    import random

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    res = y_true - y_pred
    return {
        "RMSE": float(np.sqrt(np.mean(res**2))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _resolve_cores_per_job(cores_per_job) -> int:
    """Normalize and cap cores-per-job to a safe positive integer."""
    try:
        requested = int(cores_per_job)
    except (TypeError, ValueError):
        requested = 1

    requested = max(1, requested)
    available = os.cpu_count() or requested
    return min(requested, available)


def _resolve_cv_parallel_jobs(cores_per_job) -> int:
    """Use the per-job core budget for sklearn CV/search workers."""
    return _resolve_cores_per_job(cores_per_job)



def run_xgb_single(
    target_variable,
    node,
    commodity,
    technology,
    period,
    cores_per_job: int = 1,
    filter_constant_output: bool = False,
    sample_size=None,
    overwrite: bool = False,
) -> None:
    """
    Run single-target XGBoost for a specific target and context.
    Uses grid search without extensive tuning (single iteration).
    
    Args:
        filter_constant_output: If True, skips constant output targets.
        sample_size: If provided, limit the dataset to this many samples.
        overwrite: If True, will re-run even if output files already exist.
    """
    set_seeds(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    cores_per_job = _resolve_cores_per_job(cores_per_job)
    cv_parallel_jobs = _resolve_cv_parallel_jobs(cores_per_job)

    surrogate_method = "XGB_single"

    # build key for this combination (used in filenames)
    # Include sample_size in filename if it's specified and not 4000
    sample_suffix = f"_sample{sample_size}" if sample_size is not None and sample_size != 4000 else ""
    # Build filename: target_variable_commodity_method
    parts = [safe(target_variable), safe(commodity), safe(surrogate_method)]
    parts = [p for p in parts if p]  # Remove empty strings
    folder_name = f"{'_'.join(parts)}{sample_suffix}"

    # if prediction-vs-real CSV already exists: skip this run (unless overwrite=True)
    pvr_csv_path = os.path.join(PVR_DIR, f"{folder_name}.csv")
    if os.path.exists(pvr_csv_path) and not overwrite:
        print(
            "This combination of surrogate model and target variable is already run: "
            f"{folder_name}"
        )
        return

    # ---------------- data ----------------
    print("\n=== XGBoost single-target run ===")
    print(f"Target variable: {target_variable}")
    print(f"Node:            {node}")
    print(f"Commodity:       {commodity}")
    print(f"Technology:      {technology}")
    print(f"Period:          {period}")
    print(f"Cores per job:   {cores_per_job}")
    print(f"CV workers/job:  {cv_parallel_jobs}")

    x_tr, y_tr, x_te, y_te = prepare_data(
        target_variable,
        node=node,
        period=period,
        technology=technology,
        commodity=commodity,
        sample_size=sample_size,
    )
    print(f"Training samples: {len(x_tr)} | Test samples: {len(x_te)}")

    Xtr = x_tr.detach().cpu().numpy()
    Xte = x_te.detach().cpu().numpy()
    Ytr = y_tr.detach().cpu().numpy().ravel()
    Yte = y_te.detach().cpu().numpy().ravel()
    
    # Use training data only for std_target
    std_target = float(np.std(Ytr))

    # Check uniqueness/variance using TRAIN ONLY
    amount_different_values = len(np.unique(Ytr))
    if filter_constant_output and amount_different_values == 1:
        print(f"✗ Skipping: constant output in training (only {amount_different_values} unique value)")
        return

    # ---------------- scaling: X only (using centralized scaling from settings_single.py) ----------------
    scaler = get_scaler()
    if scaler is None:
        Xtr_s = Xtr
        Xte_s = Xte
    else:
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

    # -------- training: hyperparameter tuning with 6-fold CV --------
    print("\n[ XGBoost hyperparameter tuning with 6-fold cross-validation ]")
    print(f"RandomizedSearchCV iterations: {N_ITER_TUNING}")
    
    t0 = time.perf_counter()

    param_distributions = {
        "n_estimators": randint(400, 4000),
        "learning_rate": loguniform(1e-3, 2e-1),
        "max_depth": randint(3, 10),
        "min_child_weight": loguniform(1e-1, 1e2),
        "subsample": uniform(0.6, 0.4),        # [0.6, 1.0]
        "colsample_bytree": uniform(0.6, 0.4), # [0.6, 1.0]
        "reg_alpha": loguniform(1e-9, 1e1),
        "reg_lambda": loguniform(1e-2, 1e2),
    }

    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=1,
    )

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=N_ITER_TUNING,
        scoring=rmse_scorer,
        cv=N_FOLDS_CV,
        verbose=1,
        n_jobs=cv_parallel_jobs,
        refit=True,
        random_state=SEED,
    )

    search.fit(Xtr_s, Ytr)
    training_time = time.perf_counter() - t0

    model = search.best_estimator_
    best_params = search.best_params_

    print(f"\nBest RMSE (CV) = {-search.best_score_:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print(f"\nTraining time (incl. tuning with CV): {training_time:.2f} s")

    running_time = benchmark_xgb_predict(model, Xte_s, n_repeats=100)
    n_iter_used = N_ITER_TUNING  # Number of hyperparameter combinations tried

    # ---------------- evaluation ----------------
    yhat_tr = model.predict(Xtr_s)
    yhat_te = model.predict(Xte_s)

    m_tr = eval_metrics(Ytr, yhat_tr)
    m_te = eval_metrics(Yte, yhat_te)
    
    # Extract internal R² from training metrics
    r2_internal = m_tr["R2"]
    
    # Get raw metrics from eval_metrics
    rmse = m_te["RMSE"]
    mae = m_te["MAE"]
    
    # Calculate standardized metrics
    rmse_std = float(rmse / std_target) if std_target > 0 else None
    mae_std = float(mae / std_target) if std_target > 0 else None
    
    # Calculate range-normalized metrics
    value_range = float(np.max(Yte) - np.min(Yte))
    nrmse_range = float(rmse / value_range) if value_range > 0 else None
    nmae_range = float(mae / value_range) if value_range > 0 else None
    
    # Calculate RTA (Ratio of True predictions within α% of true value)
    # RTA_α = 1/n ∑(1(|ŷ_i - y_i| ≤ α|y_i|))
    # For y_i = 0: α|y_i| = 0, so requires exact prediction (ŷ_i = 0)
    # For y_i ≠ 0: α|y_i| is the tolerance threshold
    rta1_pct = 0.0
    rta5_pct = 0.0
    rta10_pct = 0.0
    
    try:
        abs_errors = np.abs(yhat_te - Yte)
        true_vals_abs = np.abs(Yte)
        
        # Apply threshold uniformly: |ŷ_i - y_i| ≤ α|y_i|
        rta1_pct = float(np.sum(abs_errors <= 0.01 * true_vals_abs) / len(Yte))
        rta5_pct = float(np.sum(abs_errors <= 0.05 * true_vals_abs) / len(Yte))
        rta10_pct = float(np.sum(abs_errors <= 0.10 * true_vals_abs) / len(Yte))
    except Exception as e:
        pass  # Keep initialized values (all 0)
    
    # Calculate 95th percentile absolute error
    mae_95p = float(np.percentile(np.abs(Yte - yhat_te), 95))
    
    # Calculate %minmax: portion of y values at min or max (if 2+ values)
    y_min = Ytr.min()
    y_max = Ytr.max()
    count_min = np.sum(Ytr == y_min)
    count_max = np.sum(Ytr == y_max)
    
    # Only count if 2 or more values at min/max
    count_minmax = 0
    if count_min >= 2:
        count_minmax += count_min
    if count_max >= 2 and y_min != y_max:  # Don't double count if all values are same
        count_minmax += count_max
    
    pct_minmax = float(count_minmax / len(Ytr)) if len(Ytr) > 0 else 0.0
    
    # Calculate amount of different values: number of unique values
    amount_different_values = len(np.unique(Ytr))
    
    # Calculate SMAPE
    smape_val = calculate_smape(Yte, yhat_te)

    # Calculate ±5% metrics (with defensive initialization)
    within5pct = 0
    min5pct = 0
    max5pct = 0
    
    try:
        y_mean_abs = abs(np.mean(Yte))
        if y_mean_abs > 0:
            threshold = y_mean_abs * 0.05
            valid_mask = ~np.isnan(yhat_te)
            if valid_mask.sum() > 0:
                within5pct = int(np.sum(np.abs(yhat_te[valid_mask] - Yte[valid_mask]) <= threshold))
                min5pct = int(np.sum(yhat_te[valid_mask] < Yte[valid_mask] - threshold))
                max5pct = int(np.sum(yhat_te[valid_mask] > Yte[valid_mask] + threshold))
    except Exception:
        pass  # Keep initialized values (all 0)
    
    # Calculate zero-value prediction metrics (handle NaN)
    true_value_zeros_test = 0
    pred_neg05_to_0_test = 0
    pred_0_to_pos05_test = 0
    pred_correct_0 = 0
    
    try:
        true_value_zeros_test = int(np.sum(Yte == 0))
        zero_mask = (Yte == 0) & ~np.isnan(yhat_te)
        if zero_mask.sum() > 0:
            pred_neg05_to_0_test = int(np.sum((yhat_te[zero_mask] >= -0.5) & (yhat_te[zero_mask] < 0)))
            pred_0_to_pos05_test = int(np.sum((yhat_te[zero_mask] > 0) & (yhat_te[zero_mask] <= 0.5)))
        pred_correct_0 = int(np.sum((Yte == 0) & (yhat_te == 0)))
    except Exception:
        pass  # Keep initialized values (all 0)

    print("\n=== XGBoost single-target metrics (FINAL) ===")
    print(
        f"Train  | RMSE={m_tr['RMSE']:.6f}  MAE={m_tr['MAE']:.6f}  "
        f"Bias={m_tr['Bias']:.6f}  R²={m_tr['R2']:.6f}"
    )
    print(
        f"Test   | RMSE={m_te['RMSE']:.6f}  MAE={m_te['MAE']:.6f}  MAE_95p={mae_95p:.6f}  "
        f"Bias={m_te['Bias']:.6f}  R²={m_te['R2']:.6f}  %MinMax={pct_minmax:.6f}  #DiffValues={amount_different_values}"
    )
    
    # DEBUG: Print metrics before logging

    # ---------------- save model + scaler + plot ----------------
    # one joblib per (target,node,commodity,technology,period,method)
    model_path = os.path.join(TRAINED_DIR, f"{folder_name}.joblib")
    scaler_path = os.path.join(TRAINED_DIR, f"{folder_name}_scaler.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to:   {model_path}")
    print(f"Scaler saved to:  {scaler_path}")

    # prediction vs real CSV
    import pandas as pd
    from Single_target.settings_single import get_outcome_meta
    
    outcome_meta = get_outcome_meta(target_variable)
    variable = outcome_meta.get("variable", target_variable)

    df_pvr = pd.DataFrame(
        {
            "y_true": Yte,
            "y_pred": yhat_te,
            "target_variable": target_variable,
            "variable": variable,
            "node": node,
            "commodity": commodity,
            "technology": technology,
            "period": period,
            "surrogate_method": surrogate_method,
        }
    )
    df_pvr.to_csv(pvr_csv_path, index=False)

    print(f"Saved prediction vs real data to: {pvr_csv_path}")

    # Get unit from technology
    unit = get_tech_unit(technology) if technology else None

    # ---------------- log to CSV (final runs only) ----------------
    log_run(
        surrogate_method=surrogate_method,
        target_variable=target_variable,
        node=node,
        period=period,
        technology=technology,
        commodity=commodity,
        unit=unit,
        sample_size=sample_size,
        rmse=rmse,
        nrmse=rmse_std,
        nrmse_range=nrmse_range,
        mae=mae,
        nmae=mae_std,
        nmae_range=nmae_range,
        RTA1pct=rta1_pct,
        RTA5pct=rta5_pct,
        RTA10pct=rta10_pct,
        training_time_s=training_time,
        running_time_s=running_time,
        std_target=std_target,
        count_min=int(count_min),
        count_max=int(count_max),
        amount_different_values=amount_different_values,
        n_iter=n_iter_used,
        hyperparameters=best_params,
    )


def _run_xgb_target_job(
    target_job,
    cores_per_job,
    filter_constant_output=False,
    sample_size=None,
    overwrite=False,
):
    """Worker wrapper for one run_xgb_single job."""
    try:
        if isinstance(target_job, dict):
            kwargs = {
                "target_variable": target_job.get("target_variable"),
                "node": target_job.get("node"),
                "commodity": target_job.get("commodity"),
                "technology": target_job.get("technology"),
                "period": target_job.get("period"),
            }
        else:
            if not isinstance(target_job, (tuple, list)) or len(target_job) != 5:
                raise ValueError(
                    "Each target job must be either a dict with keys "
                    "(target_variable, node, commodity, technology, period) "
                    "or a 5-item tuple/list in that order."
                )
            kwargs = {
                "target_variable": target_job[0],
                "node": target_job[1],
                "commodity": target_job[2],
                "technology": target_job[3],
                "period": target_job[4],
            }

        run_xgb_single(
            **kwargs,
            cores_per_job=cores_per_job,
            filter_constant_output=filter_constant_output,
            sample_size=sample_size,
            overwrite=overwrite,
        )

        return {"ok": True, "target_variable": kwargs.get("target_variable")}
    except Exception as exc:
        return {
            "ok": False,
            "target_variable": kwargs.get("target_variable") if "kwargs" in locals() else None,
            "error": str(exc),
        }


def run_multiple_targets_xgb(
    target_jobs,
    cores_per_job,
    filter_constant_output=False,
    sample_size=None,
    overwrite=False,
):
    """
    Run multiple single-target XGB jobs in parallel while preventing oversubscription.

    Parallel target jobs are computed as floor(total_cpus / cores_per_job), capped to >= 1.
    Within each job, CV/search uses sklearn workers bounded by cores_per_job.
    """
    if not target_jobs:
        print("No target jobs provided for run_multiple_targets_xgb().")
        return []

    resolved_cores_per_job = _resolve_cores_per_job(cores_per_job)
    total_cpus = os.cpu_count() or resolved_cores_per_job
    n_parallel_targets = max(1, total_cpus // resolved_cores_per_job)

    print("\n=== XGB Multi-Target Parallel Execution ===")
    print(f"Total available CPUs:            {total_cpus}")
    print(f"Cores per target job:            {resolved_cores_per_job}")
    print(f"Parallel target jobs launched:   {n_parallel_targets}")
    print(f"Total target jobs queued:        {len(target_jobs)}")

    results = Parallel(n_jobs=n_parallel_targets, backend="loky")(
        delayed(_run_xgb_target_job)(
            target_job=job,
            cores_per_job=resolved_cores_per_job,
            filter_constant_output=filter_constant_output,
            sample_size=sample_size,
            overwrite=overwrite,
        )
        for job in target_jobs
    )

    n_ok = sum(1 for r in results if r.get("ok"))
    n_fail = len(results) - n_ok
    print(f"Finished XGB multi-target run: {n_ok} succeeded, {n_fail} failed.")
    if n_fail > 0:
        for item in results:
            if not item.get("ok"):
                print(
                    f"  - Failed target={item.get('target_variable')}: {item.get('error')}"
                )

    return results


if __name__ == "__main__":
    raise RuntimeError(
        "run_xgb_single(...) must be called with explicit arguments "
        "(target_variable, node, commodity, technology, period). "
        "Use this module via your Streamlit main.py."
    )

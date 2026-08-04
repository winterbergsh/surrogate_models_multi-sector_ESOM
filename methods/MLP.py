import os
import time
import numpy as np
import torch
import pandas as pd
import joblib  # NEW

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from autoemulate import AutoEmulate

from Single_target.settings_single import (
    SAVE_DIR,
    SEED,
    LOG_LEVEL,
    N_BOOTSTRAPS,
    MAX_TRIALS,
    N_SPLITS,
    TEST_SIZE,
    get_tech_unit,
    get_save_dir,
)

from Single_target.data_prep_single import prepare_data
from Single_target.results_summary import log_run, calculate_smape

BEST_FAMILY = "MLP"


def _resolve_cores_per_job(cores_per_job) -> int:
    """Normalize and cap cores-per-job to a safe positive integer."""
    try:
        requested = int(cores_per_job)
    except (TypeError, ValueError):
        requested = 1

    requested = max(1, requested)
    available = os.cpu_count() or requested
    return min(requested, available)

# MLP hyperparameter configuration for AutoEmulate
MLP_CONFIG = {
    "MLP": {
        # architecture
        "hidden_layers": [1, 2, 3],
        "hidden_units": [32, 64, 128],
        # activation function
        "activation": ["relu"],
        # regularisation
        "dropout": [0.0, 0.1, 0.2],
        "weight_decay": [1e-5, 1e-4],
        # optimiser settings
        "lr": [1e-3, 5e-4, 1e-4],
        # training settings
        "batch_size": [32, 64],
        "epochs": [50,100]
    }
}

# Use method-specific SAVE_DIR for MLP
_SAVE_DIR = get_save_dir(BEST_FAMILY)

TRAINED_DIR = os.path.join(_SAVE_DIR, "trained_models")
os.makedirs(TRAINED_DIR, exist_ok=True)

# folder for prediction-vs-real data and plots
PVR_DIR = os.path.join(_SAVE_DIR, "pred_vs_actual_plots")
os.makedirs(PVR_DIR, exist_ok=True)


def safe(x):
    """Make a string safe for use in paths / filenames."""
    return "none" if x is None else str(x).replace("/", "_").replace(" ", "_")


def benchmark_emulator(model, x, n_repeats: int = 100) -> float:
    """Timing benchmark for predict; returns per-sample time (s)."""
    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        _ = model.predict(x)  # warm-up

    if x.is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = model.predict(x)

    if x.is_cuda:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    per_call = elapsed / n_repeats
    per_sample = per_call / x.shape[0]

    print("\n=== Emulator Runtime ===")
    print(f"Batch time:   {per_call:.6f} s")
    print(f"Per-sample:   {per_sample:.3e} s")

    return per_sample


def run_autoemulate(
    target_variable,
    node,
    commodity,
    technology,
    period,
    cores_per_job: int = 16,
    best_family: str = BEST_FAMILY,
    filter_constant_output: bool = False,
    sample_size=None,
    overwrite: bool = False,
) -> None:
    """
    Main entry point to run AutoEmulate (MLP) for a single target.
    Uses 6 folds cross-validation with a single model (n_iter=1).
    
    Args:
        filter_constant_output: If True, skips constant output targets.
        sample_size: If provided, limit the dataset to this many samples.
        overwrite: If True, will re-run even if output files already exist.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    cores_per_job = _resolve_cores_per_job(cores_per_job)
    torch.set_num_threads(cores_per_job)

    print("\n=== AutoEmulate single-target run (MLP) ===")
    print(f"Target variable: {target_variable}")
    print(f"Node:            {node}")
    print(f"Commodity:       {commodity}")
    print(f"Technology:      {technology}")
    print(f"Period:          {period}")
    print(f"Cores per job: {cores_per_job}")
    print(f"Emulator family: {best_family}")

    # build key for this combination (used for model dir and pred-vs-real file)
    surrogate_method = f"AE_{BEST_FAMILY}"
    # Include sample_size in filename if it's specified and not 4000
    sample_suffix = f"_sample{sample_size}" if sample_size is not None and sample_size != 4000 else ""
    # Build filename: target_variable_commodity_method
    parts = [safe(target_variable), safe(commodity), safe(surrogate_method)]
    parts = [p for p in parts if p]  # Remove empty strings
    folder_name = f"{'_'.join(parts)}{sample_suffix}"

    # if prediction-vs-real CSV already exists: skip this run (unless overwrite=True)
    pvr_path = os.path.join(PVR_DIR, f"{folder_name}.csv")
    if os.path.exists(pvr_path) and not overwrite:
        msg = (
            "This combination of surrogate model and target variable is already run: "
            f"{folder_name}"
        )
        print(msg)
        return

    # ----- Data preparation -----
    x_tr, y_tr, x_te, y_te = prepare_data(
        target_variable,
        node=node,
        period=period,
        technology=technology,
        commodity=commodity,
        sample_size=sample_size,
    )
    print(f"Training samples: {len(x_tr)} | Test samples: {len(x_te)}")

    Ytr = y_tr.detach().cpu().numpy().ravel()
    Yte = y_te.detach().cpu().numpy().ravel()
    
    # Use training data only for std_target and constant output filtering
    std_target = float(np.std(Ytr))

    # Check if target has variance (skip constant outputs if filter enabled)
    # Use TRAIN ONLY for this check
    amount_different_values = len(np.unique(Ytr))
    if filter_constant_output and amount_different_values == 1:
        print(f"✗ Skipping: constant output (only {amount_different_values} unique value)")
        return

    # ----- Train chosen family -----
    print(f"\nTraining family: {best_family}")
    t0 = time.perf_counter()
    n_iter_to_use = MAX_TRIALS  # Hyperparameter tuning with 6-fold CV
    # Set n_bootstraps=0 for fair comparison
    n_bootstraps_to_use = 0
    
    # Use MLP_CONFIG for hyperparameter search space
    models_config = MLP_CONFIG
    
    ae = AutoEmulate(
        x_tr,
        y_tr,
        models=models_config,
        n_iter=n_iter_to_use,
        n_splits=N_SPLITS,
        n_bootstraps=n_bootstraps_to_use,
        shuffle=True,
        random_seed=SEED,
        log_level=LOG_LEVEL,
    )
    training_time = time.perf_counter() - t0
    print(f"Training time: {training_time:.2f} s ({training_time / 60:.2f} min)")

    # summary + best model
    summary = ae.summarise()
    best = ae.best_result()

    best_row = summary.loc[summary["model_name"] == best.model_name].iloc[0]
    best_hyperparams = best_row["params"]

    print("\n=== AutoEmulate summary ===")
    print(summary)
    print("\n=== Best hyperparameters (GaussianProcess) ===")
    print(best_hyperparams)

    # ----- Predictions on test set -----
    with torch.no_grad():
        out = best.model.predict(x_te)

        # GaussianProcess-like models: out.mean is a tensor
        if hasattr(out, "mean") and not callable(out.mean):
            pred_tensor = out.mean
        else:
            pred_tensor = out

        y_pred = pred_tensor.detach().cpu().numpy().flatten()

    y_true = y_te.cpu().numpy().flatten()

    # ----- Metrics -----
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    
    # Calculate standardized metrics
    rmse_std = float(rmse / std_target) if std_target > 0 else None
    mae_std = float(mae / std_target) if std_target > 0 else None
    
    # Calculate range-normalized metrics
    value_range = float(np.max(y_true) - np.min(y_true))
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
        abs_errors = np.abs(y_pred - y_true)
        true_vals_abs = np.abs(y_true)
        
        # Apply threshold uniformly: |ŷ_i - y_i| ≤ α|y_i|
        rta1_pct = float(np.sum(abs_errors <= 0.01 * true_vals_abs) / len(y_true))
        rta5_pct = float(np.sum(abs_errors <= 0.05 * true_vals_abs) / len(y_true))
        rta10_pct = float(np.sum(abs_errors <= 0.10 * true_vals_abs) / len(y_true))
    except Exception as e:
        pass  # Keep initialized values (all 0)
    
    # ----- Calculate internal R² (on training data) -----
    with torch.no_grad():
        out_train = best.model.predict(x_tr)
        if hasattr(out_train, "mean") and not callable(out_train.mean):
            pred_train_tensor = out_train.mean
        else:
            pred_train_tensor = out_train
        y_pred_train = pred_train_tensor.detach().cpu().numpy().flatten()
    
    y_train_true = y_tr.cpu().numpy().flatten()
    r2_internal = r2_score(y_train_true, y_pred_train)
    
    # Calculate 95th percentile absolute error
    mae_95p = float(np.percentile(np.abs(y_true - y_pred), 95))
    
    # Calculate %minmax: portion of y values at min or max (if 2+ values)
    y_min = y_train_true.min()
    y_max = y_train_true.max()
    count_min = np.sum(y_train_true == y_min)
    count_max = np.sum(y_train_true == y_max)
    
    # Only count if 2 or more values at min/max
    count_minmax = 0
    if count_min >= 2:
        count_minmax += count_min
    if count_max >= 2 and y_min != y_max:  # Don't double count if all values are same
        count_minmax += count_max
    
    pct_minmax = float(count_minmax / len(y_train_true)) if len(y_train_true) > 0 else 0.0
    
    # Calculate %unique: portion of unique values
    n_unique = len(np.unique(y_train_true))
    amount_different_values = len(np.unique(y_train_true))
    
    # Calculate SMAPE
    smape_val = calculate_smape(y_true, y_pred)


    # Calculate �5% metrics (with defensive initialization)
    within5pct = 0
    min5pct = 0
    max5pct = 0
    
    try:
        y_mean_abs = abs(np.mean(y_true))
        if y_mean_abs > 0:
            threshold = y_mean_abs * 0.05
            valid_mask = ~np.isnan(y_pred)
            if valid_mask.sum() > 0:
                within5pct = int(np.sum(np.abs(y_pred[valid_mask] - y_true[valid_mask]) <= threshold))
                min5pct = int(np.sum(y_pred[valid_mask] < y_true[valid_mask] - threshold))
                max5pct = int(np.sum(y_pred[valid_mask] > y_true[valid_mask] + threshold))
    except Exception as e:
        pass  # Keep initialized values (all 0)
    
    # Calculate zero-value prediction metrics (handle NaN)
    true_value_zeros_test = 0
    pred_neg05_to_0_test = 0
    pred_0_to_pos05_test = 0
    pred_correct_0 = 0
    
    try:
        true_value_zeros_test = int(np.sum(y_true == 0))
        zero_mask = (y_true == 0) & ~np.isnan(y_pred)
        if zero_mask.sum() > 0:
            pred_neg05_to_0_test = int(np.sum((y_pred[zero_mask] >= -0.5) & (y_pred[zero_mask] < 0)))
            pred_0_to_pos05_test = int(np.sum((y_pred[zero_mask] > 0) & (y_pred[zero_mask] <= 0.5)))
        pred_correct_0 = int(np.sum((y_true == 0) & (y_pred == 0)))
    except Exception as e:
        pass  # Keep initialized values (all 0)
    print("\n=== Holdout Performance ===")
    print(f"R²:   {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"SMAPE: {smape_val:.3f}%")
    print(f"Bias: {bias:.3f}")
    print(f"%MinMax: {pct_minmax:.3f}")
    print(f"#DiffValues: {amount_different_values}")

    # ----- Runtime benchmark -----
    running_time = benchmark_emulator(best.model, x_te, n_repeats=100)
    n_iter_used = MAX_TRIALS  # Number of hyperparameter combinations tried

    # ----- Save model in trained_models/<folder_name> -----
    model_dir = os.path.join(TRAINED_DIR, folder_name)
    saved_path = ae.save(best, model_dir, use_timestamp=True)
    print(f"\nSaved AutoEmulate model bundle to: {saved_path}")

    # additionally: save a single joblib file with full key in the filename
    joblib_path = os.path.join(TRAINED_DIR, f"{folder_name}.joblib")
    joblib.dump(best, joblib_path)
    print(f"Saved joblib snapshot to: {joblib_path}")

    # ----- Save prediction vs real data in pred_vs_actual_plots/<folder_name>.csv -----
    from Single_target.settings_single import get_outcome_meta
    outcome_meta = get_outcome_meta(target_variable)
    variable = outcome_meta.get("variable", target_variable)
    
    df_pvr = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "target_variable": target_variable,
            "variable": variable,
            "node": node,
            "commodity": commodity,
            "technology": technology,
            "period": period,
            "surrogate_method": surrogate_method,
            "std": std_target,
        }
    )
    df_pvr.to_csv(pvr_path, index=False)

    print(f"Saved prediction vs real data to: {pvr_path}")

    # ----- Log run -----
    # Get unit from technology
    unit = get_tech_unit(technology) if technology else None
    
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
        hyperparameters=best_hyperparams,
    )


# Alias for main.py compatibility
run_mlppytorch_single = run_autoemulate

if __name__ == "__main__":
    raise RuntimeError(
        "run_autoemulate(...) must be called with explicit arguments "
        "(target_variable, node, commodity, technology, period). "
        "Use this module via your Streamlit main.py."
    )

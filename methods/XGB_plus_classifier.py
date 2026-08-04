import os
import time
import csv
from datetime import datetime
import numpy as np
import torch
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    make_scorer,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import randint, uniform, loguniform

from Single_target.settings_single import (
    SEED,
    MAX_TRIALS,
    N_SPLITS,
    get_tech_unit,
    get_scaler,
    get_save_dir,
)
from Single_target.data_prep_single import prepare_data
from Single_target.results_summary import log_run, calculate_smape, log_classifier_run, CLASSIFIER_METRICS_CSV, CLASSIFIER_METRICS_HEADER

# ============ user switches ============
N_ITER_TUNING = MAX_TRIALS        # number of random param samples
N_FOLDS_CV = N_SPLITS             # CV folds
# ======================================

DEFAULT_CORES_PER_JOB = 1

# Set SAVE_DIR based on surrogate method
SURROGATE_METHOD_NAME = "XGB_HYBRID"
SAVE_DIR = get_save_dir(SURROGATE_METHOD_NAME)

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
}

# folders - separate for XGB_HYBRID model
TRAINED_DIR = os.path.join(SAVE_DIR, "trained_models", "XGB_HYBRID")
os.makedirs(TRAINED_DIR, exist_ok=True)

PVR_DIR = os.path.join(SAVE_DIR, "pred_vs_actual_plots", "XGB_HYBRID")
os.makedirs(PVR_DIR, exist_ok=True)

# Use new path structure from settings_single
from Single_target.settings_single import ST_MODELS_DIR
CLS_DIR = os.path.join(ST_MODELS_DIR, "classifiers")
HYB_DIR = os.path.join(ST_MODELS_DIR, "hybrid")
os.makedirs(CLS_DIR, exist_ok=True)
os.makedirs(HYB_DIR, exist_ok=True)


def safe(x):
    return "none" if x is None else str(x).replace("/", "_").replace(" ", "_")


def set_seeds(seed: int = 43) -> None:
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _resolve_cores_per_job(cores_per_job) -> int:
    """Normalize and cap cores-per-job to a safe positive integer."""
    try:
        requested = int(cores_per_job)
    except (TypeError, ValueError):
        requested = DEFAULT_CORES_PER_JOB

    requested = max(1, requested)
    available = os.cpu_count() or requested
    return min(requested, available)


def _resolve_cv_parallel_jobs(cores_per_job) -> int:
    """Use the per-job core budget for sklearn CV/search workers."""
    return _resolve_cores_per_job(cores_per_job)


def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    res = y_true - y_pred
    rmse = float(np.sqrt(np.mean(res**2)))
    return {
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def benchmark_hybrid_predict(clf, reg, X, categories, n_repeats: int = 100):
    # warm-up
    _ = clf.predict(X)
    _ = reg.predict(X[: min(len(X), 10)]) if len(X) > 0 else None

    start = time.perf_counter()
    for _ in range(n_repeats):
        pred_cls = clf.predict(X)
        
        # If predictions are 2D (probabilities), take argmax to get class labels
        if pred_cls.ndim == 2:
            pred_cls = np.argmax(pred_cls, axis=1)
        
        y_pred = np.empty(X.shape[0], dtype=float)
        
        n_categories = len(categories)
        
        # Assign category values to predicted categories
        for i, cat_val in enumerate(categories):
            mask = (pred_cls == i)
            y_pred[mask] = cat_val
        
        # For REST predictions, use regressor
        rest_mask = (pred_cls == n_categories)
        if np.any(rest_mask):
            y_pred[rest_mask] = reg.predict(X[rest_mask])
    elapsed = time.perf_counter() - start

    batch_time = elapsed / n_repeats
    per_sample = batch_time / X.shape[0]

    print("\n=== Hybrid Inference Runtime ===")
    print(f"Batch time (hybrid):   {batch_time:.6f} s")
    print(f"Per-sample (hybrid):   {per_sample:.3e} s")
    return per_sample


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


# =========================
# CLASSIFIER HELPERS
# =========================
def identify_categories(y: np.ndarray, threshold: int = 20) -> tuple:
    """
        Identify which values in y are 'categories' (appear at least threshold times).
    Returns:
            - categories: list of unique values that appear at least threshold times
      - category_indices: boolean mask for samples in categories
      - rest_indices: boolean mask for samples not in categories
    """
    y = np.asarray(y).ravel()
    unique_vals, counts = np.unique(y, return_counts=True)
    
    # Find values that appear at least threshold times
    categories = unique_vals[counts >= threshold]
    
    # Create masks
    category_indices = np.zeros(len(y), dtype=bool)
    for cat in categories:
        category_indices |= (y == cat)
    
    rest_indices = ~category_indices
    
    return list(categories), category_indices, rest_indices


def make_category_classes(y: np.ndarray, categories: list) -> np.ndarray:
    """
    Create class labels for category-based classification:
    - 0, 1, 2, ... for each category
    - len(categories) for REST samples
    """
    y = np.asarray(y).ravel()
    n_categories = len(categories)
    cls = np.full(len(y), n_categories, dtype=np.int64)  # default to REST
    
    for i, cat in enumerate(categories):
        cls[y == cat] = i
    
    return cls


def plot_confusion_matrix(cm, out_path, title):
    n_classes = cm.shape[0]
    class_names = [str(i) for i in range(n_classes - 1)] + ["REST"]
    
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_true_vs_pred_regression(y_true, y_pred, out_path, title):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    
    # Remove background (make it white and remove grey background)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Plot scatter
    ax.scatter(y_true, y_pred, s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    
    # Diagonal line (reference)
    ax.plot([lo, hi], [lo, hi], 'k-', linewidth=1.5, label='Perfect prediction')
    
    # Add box outline (spines)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Remove grid or make it minimal
    ax.grid(False)
    
    ax.set_xlabel("True value", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted value", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def train_classifier(Xtr_s, Ytr, threshold: int = 20, cores_per_job: int = DEFAULT_CORES_PER_JOB):
    """
    Train a classifier that identifies category vs REST samples.
    Categories are values appearing at least threshold times in training set.
    """
    categories, cat_indices, rest_indices = identify_categories(Ytr, threshold=threshold)
    y_tr_cls = make_category_classes(Ytr, categories)
    cv_parallel_jobs = _resolve_cv_parallel_jobs(cores_per_job)
    
    n_classes = len(categories) + 1  # +1 for REST

    print("\n[ Classifier tuning enabled ]")

    param_distributions = {
        "n_estimators": randint(100, 1000),
        "learning_rate": loguniform(1e-3, 2e-1),
        "max_depth": randint(3, 6),
        "min_child_weight": loguniform(1e-1, 1e2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": loguniform(1e-9, 1e1),
        "reg_lambda": loguniform(1e-2, 1e2),
    }

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
    )

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=N_ITER_TUNING,
        scoring="accuracy",
        cv=N_FOLDS_CV,
        verbose=1,
        n_jobs=cv_parallel_jobs,
        refit=True,
        random_state=SEED,
    )

    search.fit(Xtr_s, y_tr_cls)

    clf = search.best_estimator_
    best_params = search.best_params_

    print(f"\nBest accuracy (CV) = {search.best_score_:.4f}")
    print("Best classifier params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return clf, categories, y_tr_cls, best_params


def hybrid_predict(clf, reg, X_s, categories):
    """
    Hybrid prediction:
    - If classifier predicts a category (0 to n_categories-1), return that category value
    - If classifier predicts REST (n_categories), use regression to predict
    """
    pred_cls = clf.predict(X_s)
    
    # If predictions are 2D (probabilities), take argmax to get class labels
    if pred_cls.ndim == 2:
        pred_cls = np.argmax(pred_cls, axis=1)
    
    y_pred = np.empty(X_s.shape[0], dtype=float)
    
    n_categories = len(categories)
    
    # Assign category values to predicted categories
    for i, cat_val in enumerate(categories):
        mask = (pred_cls == i)
        y_pred[mask] = cat_val
    
    # For REST predictions, use regressor
    rest_mask = (pred_cls == n_categories)
    
    if np.any(rest_mask):
        y_pred[rest_mask] = reg.predict(X_s[rest_mask])

    return y_pred, pred_cls


def run_xgb_single(
    target_variable,
    node,
    commodity,
    technology,
    period,
    tune_hyperparameters: bool = True,
    filter_constant_output: bool = False,
    sample_size=None,
    cores_per_job: int = DEFAULT_CORES_PER_JOB,
    overwrite: bool = False,
) -> None:
    """
    HYBRID model:
      - classifier (min/mid/max) trained on ALL training data
      - regressor trained ONLY on 'middle' training samples
      - inference: classifier decides min/max/mid; mid uses regressor prediction; min/max become constants
      - r2_external is computed on FULL test set using hybrid predictions
    """
    set_seeds(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    cores_per_job = _resolve_cores_per_job(cores_per_job)
    cv_parallel_jobs = _resolve_cv_parallel_jobs(cores_per_job)

    tuning_suffix = "_tuned" if tune_hyperparameters else "_quick"
    surrogate_method = f"XGB_HYBRID{tuning_suffix}"

    sample_suffix = f"_sample{sample_size}" if sample_size is not None and sample_size != 4000 else ""
    parts = [safe(target_variable), safe(commodity), safe(surrogate_method)]
    parts = [p for p in parts if p]  # Remove empty strings
    folder_name = f"{'_'.join(parts)}{sample_suffix}"

    pvr_csv_path = os.path.join(PVR_DIR, f"{folder_name}.csv")
    if os.path.exists(pvr_csv_path) and not overwrite:
        print("This combination of surrogate model and target variable is already run: " f"{folder_name}")
        return

    print("\n=== XGBoost HYBRID single-target run ===")
    print(f"Tune hyperparameters (regressor): {tune_hyperparameters}")
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

    print(f"DEBUG DATA: About to convert tensors to numpy")
    Xtr = x_tr.detach().cpu().numpy()
    Xte = x_te.detach().cpu().numpy()
    Ytr = y_tr.detach().cpu().numpy().ravel()
    Yte = y_te.detach().cpu().numpy().ravel()
    print(f"DEBUG DATA: Tensor conversion successful - Xtr shape={Xtr.shape}, Ytr shape={Ytr.shape}, Xte shape={Xte.shape}, Yte shape={Yte.shape}")

    std_target = float(np.std(Ytr))

    amount_different_values = len(np.unique(Ytr))
    if filter_constant_output and amount_different_values == 1:
        print(f"✗ Skipping: constant output in training (only {amount_different_values} unique value)")
        return
    
    # Also check for extremely imbalanced data (one value dominates >95%)
    if filter_constant_output and amount_different_values > 1:
        unique_vals, counts = np.unique(Ytr, return_counts=True)
        max_count_ratio = counts.max() / len(Ytr)
        if max_count_ratio > 0.95:
            dominant_val = unique_vals[counts.argmax()]
            print(f"✗ Skipping: extremely imbalanced output (value {dominant_val} appears in {max_count_ratio*100:.1f}% of training data)")
            return

    # scaling: X only
    print(f"DEBUG DATA: About to get scaler")
    scaler = get_scaler()
    print(f"DEBUG DATA: Got scaler, scaler type={type(scaler)}")
    
    if scaler is None:
        print(f"DEBUG DATA: Scaler is None, using raw data")
        Xtr_s = Xtr
        Xte_s = Xte
    else:
        print(f"DEBUG DATA: About to fit_transform on Xtr shape={Xtr.shape}")
        Xtr_s = scaler.fit_transform(Xtr)
        print(f"DEBUG DATA: fit_transform successful, Xtr_s shape={Xtr_s.shape}")
        print(f"DEBUG DATA: About to transform on Xte shape={Xte.shape}")
        Xte_s = scaler.transform(Xte)
        print(f"DEBUG DATA: transform successful, Xte_s shape={Xte_s.shape}")

    # Start timing: include both classifier and regressor training
    t0 = time.perf_counter()
    
    # =========================
    # 1) Train classifier on ALL train - Categories vs REST
    # =========================
    print(f"DEBUG CLASSIFIER: About to call train_classifier")
    print(f"DEBUG CLASSIFIER: Xtr_s shape={Xtr_s.shape}, Ytr shape={Ytr.shape}")
    
    CATEGORY_THRESHOLD = 20  # Values appearing >= 20 times are categories
    clf, categories, y_tr_cls, classifier_best_params = train_classifier(
        Xtr_s,
        Ytr,
        threshold=CATEGORY_THRESHOLD,
        cores_per_job=cores_per_job,
    )
    
    print(f"DEBUG CLASSIFIER: train_classifier returned successfully")
    print(f"DEBUG CLASSIFIER: categories={categories}, y_tr_cls shape={y_tr_cls.shape}")
    
    # Check if there are any recurring categories (values appearing > threshold times)
    # If no categories found, all data is "REST" (continuous/uniform) → pure regression only
    if len(categories) == 0:
        print(f"\n" + "="*80)
        print(f"INFO: No recurring categories found (all values appear <= threshold).")
        print(f"All data is continuous/uniform (REST category only). NO CLASSIFIER NEEDED.")
        print(f"Using PURE REGRESSION on all {len(Ytr)} training samples.")
        print(f"="*80 + "\n")
        
        # ========= PURE REGRESSION PATH =========
        # Train regressor on ALL data
        print("\n[ XGBoost hyperparameter tuning with 6-fold cross-validation ]")

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

        reg = search.best_estimator_
        best_params = search.best_params_

        print(f"\nBest RMSE (CV) = {-search.best_score_:.4f}")
        print("Best params:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        print(f"\nTraining time (incl. tuning with CV): {training_time:.2f} s")

        # Predict with pure regression
        yhat_tr = reg.predict(Xtr_s)
        yhat_te = reg.predict(Xte_s)
        
        # All predictions are from regression (no classifier)
        pred_cls_te = np.full(len(Yte), len(categories))
        
        # Evaluate metrics on test set
        m_te = eval_metrics(Yte, yhat_te)
        m_tr = eval_metrics(Ytr, yhat_tr)
        
        rmse = m_te["RMSE"]
        mae = m_te["MAE"]
        
        # Calculate three normalization methods:
        # 1. Standard deviation normalization (current)
        rmse_std = float(rmse / std_target) if std_target > 0 else None
        mae_std = float(mae / std_target) if std_target > 0 else None
        
        # 2. Range normalization (max - min)
        value_range = float(np.max(Yte) - np.min(Yte))
        nrmse_range = float(rmse / value_range) if value_range > 0 else None
        nmae_range = float(mae / value_range) if value_range > 0 else None
        
        # 3. Mean normalization (mean of absolute values)
        mean_target = float(np.mean(np.abs(Ytr)))
        rmse_mean = float(rmse / mean_target) if mean_target > 0 else None
        mae_mean = float(mae / mean_target) if mean_target > 0 else None
        
        # Calculate RTA metrics
        rta1_pct = 0.0
        rta5_pct = 0.0
        rta10_pct = 0.0
        try:
            abs_errors = np.abs(yhat_te - Yte)
            true_vals_abs = np.abs(Yte)
            rta1_pct = float(np.sum(abs_errors <= 0.01 * true_vals_abs) / len(Yte)) if len(Yte) > 0 else 0.0
            rta5_pct = float(np.sum(abs_errors <= 0.05 * true_vals_abs) / len(Yte)) if len(Yte) > 0 else 0.0
            rta10_pct = float(np.sum(abs_errors <= 0.10 * true_vals_abs) / len(Yte)) if len(Yte) > 0 else 0.0
        except:
            pass
        
        mae_95p = float(np.percentile(np.abs(Yte - yhat_te), 95))
        smape_val = calculate_smape(Yte, yhat_te)
        
        y_min = Ytr.min()
        y_max = Ytr.max()
        count_min = np.sum(Ytr == y_min)
        count_max = np.sum(Ytr == y_max)
        count_minmax = 0
        if count_min >= 2:
            count_minmax += count_min
        if count_max >= 2 and y_min != y_max:
            count_minmax += count_max
        pct_minmax = float(count_minmax / len(Ytr)) if len(Ytr) > 0 else 0.0
        amount_different_values = len(np.unique(Ytr))
        
        print(f"\n=== Pure Regression Metrics (FINAL) ===")
        print(f"Train  | RMSE={m_tr['RMSE']:.6f}  MAE={m_tr['MAE']:.6f}  Bias={m_tr['Bias']:.6f}  R2={m_tr['R2']:.6f}")
        print(f"Test   | RMSE={m_te['RMSE']:.6f}  MAE={m_te['MAE']:.6f}  MAE_95p={mae_95p:.6f}  Bias={m_te['Bias']:.6f}  R2={m_te['R2']:.6f}  SMAPE={smape_val:.3f}%")
        
        # Runtime benchmark for pure regression (regressor only, no classifier)
        running_time = benchmark_xgb_predict(reg, Xte_s, n_repeats=100)
        n_iter_used = N_ITER_TUNING  # Number of hyperparameter combinations tried
        
        # Save models
        reg_path = os.path.join(TRAINED_DIR, f"{folder_name}_regressor.joblib")
        scaler_path = os.path.join(TRAINED_DIR, f"{folder_name}_scaler.joblib")
        joblib.dump(reg, reg_path)
        joblib.dump(scaler, scaler_path)
        print(f"Saved models to {TRAINED_DIR}/")
        
        # Save predictions
        from Single_target.settings_single import get_outcome_meta
        outcome_meta = get_outcome_meta(target_variable)
        variable = outcome_meta.get("variable", target_variable)
        
        import pandas as pd
        df_pvr = pd.DataFrame({
            "y_true": Yte,
            "y_pred": yhat_te,
            "pred_class": pred_cls_te.astype(int),
            "target_variable": target_variable,
            "variable": variable,
            "node": node,
            "commodity": commodity,
            "technology": technology,
            "period": period,
            "surrogate_method": surrogate_method,
        })
        df_pvr.to_csv(pvr_csv_path, index=False)
        
        # Save plot
        plot_path = os.path.join(PVR_DIR, f"{folder_name}_true_vs_pred_hybrid.png")
        plot_true_vs_pred_regression(Yte, yhat_te, plot_path, title="XGB 2-fold (Pure Regression)")
        
        # Metrics for logging
        REG_cat = len(Yte)
        CLAS_cat = 0
        RTA1reg = int(np.sum(np.abs(yhat_te - Yte) <= 0.01 * np.abs(Yte)))
        RTA5reg = int(np.sum(np.abs(yhat_te - Yte) <= 0.05 * np.abs(Yte)))
        RTA10reg = int(np.sum(np.abs(yhat_te - Yte) <= 0.10 * np.abs(Yte)))
        classifier_accuracy_categories = 0.0
        accuracy_reg_1pct = float(RTA1reg / len(Yte)) if len(Yte) > 0 else 0.0
        accuracy_reg_5pct = float(RTA5reg / len(Yte)) if len(Yte) > 0 else 0.0
        accuracy_reg_10pct = float(RTA10reg / len(Yte)) if len(Yte) > 0 else 0.0
        
        unit = get_tech_unit(technology) if technology else None
        
        # Log results
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
            hyperparameters={
                "regressor": best_params,
                "pure_regression": True,
                "categories_threshold": CATEGORY_THRESHOLD,
            },
        )
        
        try:
            log_classifier_run(
                surrogate_method=surrogate_method,
                target_variable=target_variable,
                node=node,
                period=period,
                technology=technology,
                commodity=commodity,
                unit=unit,
                rmse=rmse,
                nrmse=rmse_std,
                nrmse_range=nrmse_range,
                nrmse_mean=rmse_mean,
                mae=mae,
                nmae=mae_std,
                nmae_range=nmae_range,
                nmae_mean=mae_mean,
                RTA1pct=rta1_pct,
                RTA5pct=rta5_pct,
                RTA10pct=rta10_pct,
                training_time_s=training_time,
                running_time_s=running_time,
                std_target=std_target,
                classifier_accuracy=None,  # N/A - pure regression (no classifier used)
                categories=[],
                REG_cat=REG_cat,
                RTA1reg=RTA1reg,
                RTA5reg=RTA5reg,
                RTA10reg=RTA10reg,
                CLAS_cat=CLAS_cat,
                classifier_accuracy_categories=classifier_accuracy_categories,
                accuracy_reg_1pct=accuracy_reg_1pct,
                accuracy_reg_5pct=accuracy_reg_5pct,
                accuracy_reg_10pct=accuracy_reg_10pct,
                hyperparameters={
                    "regressor": best_params,
                    "pure_regression": True,
                    "categories_threshold": CATEGORY_THRESHOLD,
                },
            )
        except Exception as e:
            print(f"Warning: log_classifier_run failed: {e}")
        
        return  # Return early - pure regression path complete
    
    # ========= HYBRID PATH (2+ categories) =========
    print(f"\nINFO: {len(categories)} categories found. Using HYBRID (classifier + regressor) approach.")
    
    # Identify categories in test set
    print(f"DEBUG CLASSIFIER: About to call make_category_classes for test set")
    y_te_cls_true = make_category_classes(Yte, categories)
    print(f"DEBUG CLASSIFIER: make_category_classes returned, y_te_cls_true shape={y_te_cls_true.shape}")
    
    print(f"DEBUG CLASSIFIER: About to predict on test set")
    y_te_cls_pred_raw = clf.predict(Xte_s)
    print(f"DEBUG CLASSIFIER: clf.predict returned shape={y_te_cls_pred_raw.shape}, ndim={y_te_cls_pred_raw.ndim}")
    
    # If predictions are 2D (probabilities), take argmax to get class labels
    if y_te_cls_pred_raw.ndim == 2:
        print(f"DEBUG CLASSIFIER: Converting 2D predictions to class labels via argmax")
        y_te_cls_pred = np.argmax(y_te_cls_pred_raw, axis=1)
    else:
        y_te_cls_pred = y_te_cls_pred_raw
    print(f"DEBUG CLASSIFIER: After conversion, y_te_cls_pred shape={y_te_cls_pred.shape}")
    
    # Ensure both are 1D arrays for sklearn metrics
    y_te_cls_true = np.asarray(y_te_cls_true).ravel()
    y_te_cls_pred = np.asarray(y_te_cls_pred).ravel()
    
    print(f"DEBUG CLASSIFIER: After ravel - y_te_cls_true shape={y_te_cls_true.shape}, y_te_cls_pred shape={y_te_cls_pred.shape}")
    
    n_classes = len(categories) + 1
    print(f"DEBUG CLASSIFIER: About to calculate accuracy_score with n_classes={n_classes}")
    clf_acc = accuracy_score(y_te_cls_true, y_te_cls_pred)
    print(f"DEBUG CLASSIFIER: accuracy_score returned clf_acc={clf_acc}")
    
    print(f"DEBUG CLASSIFIER: About to calculate confusion_matrix")
    clf_cm = confusion_matrix(y_te_cls_true, y_te_cls_pred, labels=list(range(n_classes)))
    print(f"DEBUG CLASSIFIER: confusion_matrix returned, shape={clf_cm.shape}")
    
    class_names = [f"Cat_{val}" for val in categories] + ["REST"]
    print("\n=== Classifier (Category vs REST) ===")
    print(f"Categories (>=20 occurrences): {categories}")
    print(f"Total classes: {n_classes} (categories + REST)")
    print(f"Classifier accuracy (test): {clf_acc:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred) labels={class_names}:")
    print(clf_cm)
    print("\nClassification report (test):")
    print(
        classification_report(
            y_te_cls_true,
            y_te_cls_pred,
            labels=list(range(n_classes)),
            target_names=class_names,
            zero_division=0,
        )
    )

    # Save classifier outputs
    clf_path = os.path.join(TRAINED_DIR, f"{folder_name}_classifier.joblib")
    joblib.dump(clf, clf_path)

    cm_path = os.path.join(CLS_DIR, f"{folder_name}_class_confusion_matrix.png")
    plot_confusion_matrix(clf_cm, cm_path, f"Classifier confusion matrix (test)\nacc={clf_acc:.4f}")

    # =========================
    # 2) Train regressor ONLY on REST samples (from TRAIN)
    # =========================
    print(f"DEBUG START: About to extract REST samples")
    print(f"DEBUG START: y_tr_cls type={type(y_tr_cls)}, shape={y_tr_cls.shape}")
    print(f"DEBUG START: categories={categories}, len={len(categories)}")
    
    rest_mask = (y_tr_cls == len(categories))
    print(f"DEBUG 1: rest_mask type={type(rest_mask)}, len={len(rest_mask)}, sum={np.sum(rest_mask)}")
    print(f"DEBUG 1: Xtr_s type={type(Xtr_s)}, shape={Xtr_s.shape}")
    print(f"DEBUG 1: Ytr type={type(Ytr)}, shape={Ytr.shape}")
    
    # Explicitly get REST indices and extract data
    rest_count = int(np.sum(rest_mask))
    print(f"DEBUG 2: rest_count={rest_count}")
    
    Xtr_rest = np.zeros((rest_count, Xtr_s.shape[1]), dtype=np.float32)
    Ytr_rest = np.zeros(rest_count, dtype=np.float32)
    print(f"DEBUG 3: Xtr_rest shape={Xtr_rest.shape}, Ytr_rest shape={Ytr_rest.shape}")
    
    rest_idx = 0
    for i in range(len(Xtr_s)):
        if rest_mask[i]:
            Xtr_rest[rest_idx] = Xtr_s[i].astype(np.float32)
            Ytr_rest[rest_idx] = Ytr[i].astype(np.float32)
            rest_idx += 1
    
    print(f"DEBUG 4: After loop - rest_idx={rest_idx}, Xtr_rest.shape={Xtr_rest.shape}, Ytr_rest.shape={Ytr_rest.shape}")
    
    # Validate shapes
    print(f"DEBUG: Xtr_rest shape={Xtr_rest.shape}, Ytr_rest shape={Ytr_rest.shape}")
    assert Xtr_rest.ndim == 2, f"Xtr_rest should be 2D, got shape {Xtr_rest.shape}"
    assert Ytr_rest.ndim == 1, f"Ytr_rest should be 1D, got shape {Ytr_rest.shape}"
    assert Xtr_rest.shape[0] == Ytr_rest.shape[0], f"Shape mismatch: Xtr_rest {Xtr_rest.shape} vs Ytr_rest {Ytr_rest.shape}"

    if len(Ytr_rest) < 10:
        print(f"\n⚠ WARNING: Too few REST samples ({len(Ytr_rest)}) to train regressor reliably.")
        print(f"   Attempting to retry with lower category threshold...")
        
        # Try with lower threshold
        lower_threshold = max(1, CATEGORY_THRESHOLD // 2)
        print(f"   Retrying with CATEGORY_THRESHOLD={lower_threshold} (was {CATEGORY_THRESHOLD})")
        
        clf, categories, y_tr_cls, classifier_best_params = train_classifier(
            Xtr_s,
            Ytr,
            threshold=lower_threshold,
            cores_per_job=cores_per_job,
        )
        rest_mask = (y_tr_cls == len(categories))
        rest_count = int(np.sum(rest_mask))
        
        if rest_count < 10:
            print(f"✗ Still insufficient REST samples ({rest_count}) even with lower threshold.")
            print(f"   Skipping this target (data is too imbalanced for hybrid classifier).")
            return
        else:
            print(f"✓ Successfully found {rest_count} REST samples with lower threshold.")
            Xtr_rest = np.zeros((rest_count, Xtr_s.shape[1]), dtype=np.float32)
            Ytr_rest = np.zeros(rest_count, dtype=np.float32)
            rest_idx = 0
            for i in range(len(Xtr_s)):
                if rest_mask[i]:
                    Xtr_rest[rest_idx] = Xtr_s[i].astype(np.float32)
                    Ytr_rest[rest_idx] = Ytr[i].astype(np.float32)
                    rest_idx += 1

    print(f"\n=== Regressor training on REST only ===")
    print(f"REST samples (train): {len(Ytr_rest)} / {len(Ytr)}")
    print(f"Category samples (train): {len(Ytr) - len(Ytr_rest)}")

    if tune_hyperparameters:
        print("\n[ Regressor tuning enabled ]")

        param_distributions = {
            "n_estimators": randint(400, 4000),
            "learning_rate": loguniform(1e-3, 2e-1),
            "max_depth": randint(3, 10),
            "min_child_weight": loguniform(1e-1, 1e2),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
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

        search.fit(Xtr_rest, Ytr_rest)
        training_time = time.perf_counter() - t0

        reg = search.best_estimator_
        best_params = search.best_params_

        print(f"\nBest RMSE (cv) = {-search.best_score_:.4f}")
        print("Best regressor params:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        print(f"\nTraining time (incl. tuning): {training_time:.2f} s")
    else:
        print("\n[ Regressor tuning disabled - using DEFAULT_XGB_PARAMS ]")

        t0 = time.perf_counter()

        base = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS, random_state=SEED, n_jobs=1)

        def rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

        rmse_scorer = make_scorer(rmse, greater_is_better=False)

        _ = cross_val_score(
            estimator=base,
            X=Xtr_rest,
            y=Ytr_rest,
            cv=N_FOLDS_CV,
            scoring=rmse_scorer,
            n_jobs=cv_parallel_jobs,
            verbose=1,
        )

        reg = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS, random_state=SEED, n_jobs=1)
        reg.fit(Xtr_rest, Ytr_rest)

        training_time = time.perf_counter() - t0
        best_params = DEFAULT_XGB_PARAMS.copy()
        print(f"Training time (no tuning, with CV on REST only): {training_time:.2f} s")

    # =========================
    # 3) HYBRID inference and metrics
    # =========================
    print(f"DEBUG INFERENCE: About to call hybrid_predict on test set")
    print(f"DEBUG INFERENCE: clf type={type(clf)}, reg type={type(reg)}, Xte_s shape={Xte_s.shape}, categories={categories}")
    
    try:
        yhat_te_hybrid, pred_cls_te = hybrid_predict(clf, reg, Xte_s, categories)
        print(f"DEBUG INFERENCE: hybrid_predict on test set successful, yhat_te_hybrid shape={yhat_te_hybrid.shape}, pred_cls_te shape={pred_cls_te.shape}")
    except Exception as e:
        print(f"DEBUG INFERENCE: ERROR in hybrid_predict on test set: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"DEBUG INFERENCE: About to call hybrid_predict on train set")
    try:
        yhat_tr_hybrid, pred_cls_tr = hybrid_predict(clf, reg, Xtr_s, categories)
        print(f"DEBUG INFERENCE: hybrid_predict on train set successful, yhat_tr_hybrid shape={yhat_tr_hybrid.shape}, pred_cls_tr shape={pred_cls_tr.shape}")
    except Exception as e:
        print(f"DEBUG INFERENCE: ERROR in hybrid_predict on train set: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"DEBUG INFERENCE: About to call eval_metrics on test set")
    m_te = eval_metrics(Yte, yhat_te_hybrid)
    print(f"DEBUG INFERENCE: eval_metrics on test set successful: {m_te}")
    
    print(f"DEBUG INFERENCE: About to call eval_metrics on train set")
    m_tr = eval_metrics(Ytr, yhat_tr_hybrid)
    print(f"DEBUG INFERENCE: eval_metrics on train set successful: {m_tr}")

    r2_internal = m_tr["R2"]
    
    # Get raw metrics
    rmse = m_te["RMSE"]
    mae = m_te["MAE"]
    
    # Calculate three normalization methods:
    # 1. Standard deviation normalization (current)
    rmse_std = float(rmse / std_target) if std_target > 0 else None
    mae_std = float(mae / std_target) if std_target > 0 else None
    
    # 2. Range normalization (max - min)
    value_range = float(np.max(Yte) - np.min(Yte))
    nrmse_range = float(rmse / value_range) if value_range > 0 else None
    nmae_range = float(mae / value_range) if value_range > 0 else None
    
    # 3. Mean normalization (mean of absolute values)
    mean_target = float(np.mean(np.abs(Ytr)))
    rmse_mean = float(rmse / mean_target) if mean_target > 0 else None
    mae_mean = float(mae / mean_target) if mean_target > 0 else None
    
    # Calculate RTA (Ratio of True predictions within α% of true value)
    # RTA_α = 1/n ∑(1(|ŷ_i - y_i| ≤ α|y_i|))
    # For y_i = 0: α|y_i| = 0, so requires exact prediction (ŷ_i = 0)
    # For y_i ≠ 0: α|y_i| is the tolerance threshold
    rta1_pct = 0.0
    rta5_pct = 0.0
    rta10_pct = 0.0
    
    try:
        abs_errors = np.abs(yhat_te_hybrid - Yte)
        true_vals_abs = np.abs(Yte)
        
        # Apply threshold uniformly: |ŷ_i - y_i| ≤ α|y_i|
        rta1_pct = float(np.sum(abs_errors <= 0.01 * true_vals_abs) / len(Yte))
        rta5_pct = float(np.sum(abs_errors <= 0.05 * true_vals_abs) / len(Yte))
        rta10_pct = float(np.sum(abs_errors <= 0.10 * true_vals_abs) / len(Yte))
    except Exception as e:
        pass  # Keep initialized values (all 0)
    
    mae_95p = float(np.percentile(np.abs(Yte - yhat_te_hybrid), 95))
    smape_val = calculate_smape(Yte, yhat_te_hybrid)

    # Calculate ±5% metrics (with defensive initialization)
    within5pct = 0
    min5pct = 0
    max5pct = 0
    
    try:
        y_mean_abs = abs(np.mean(Yte))
        if y_mean_abs > 0:
            threshold = y_mean_abs * 0.05
            valid_mask = ~np.isnan(yhat_te_hybrid)
            if valid_mask.sum() > 0:
                within5pct = int(np.sum(np.abs(yhat_te_hybrid[valid_mask] - Yte[valid_mask]) <= threshold))
                min5pct = int(np.sum(yhat_te_hybrid[valid_mask] < Yte[valid_mask] - threshold))
                max5pct = int(np.sum(yhat_te_hybrid[valid_mask] > Yte[valid_mask] + threshold))
    except Exception:
        pass  # Keep initialized values (all 0)
    
    # Calculate zero-value prediction metrics (handle NaN)
    true_value_zeros_test = 0
    pred_neg05_to_0_test = 0
    pred_0_to_pos05_test = 0
    pred_correct_0 = 0
    
    try:
        true_value_zeros_test = int(np.sum(Yte == 0))
        zero_mask = (Yte == 0) & ~np.isnan(yhat_te_hybrid)
        if zero_mask.sum() > 0:
            pred_neg05_to_0_test = int(np.sum((yhat_te_hybrid[zero_mask] >= -0.5) & (yhat_te_hybrid[zero_mask] < 0)))
            pred_0_to_pos05_test = int(np.sum((yhat_te_hybrid[zero_mask] > 0) & (yhat_te_hybrid[zero_mask] <= 0.5)))
        pred_correct_0 = int(np.sum((Yte == 0) & (yhat_te_hybrid == 0)))
    except Exception:
        pass  # Keep initialized values (all 0)

    # %minmax on TRAIN labels (same as your existing logic)
    y_min = Ytr.min()
    y_max = Ytr.max()
    count_min = np.sum(Ytr == y_min)
    count_max = np.sum(Ytr == y_max)
    count_minmax = 0
    if count_min >= 2:
        count_minmax += count_min
    if count_max >= 2 and y_min != y_max:
        count_minmax += count_max
    pct_minmax = float(count_minmax / len(Ytr)) if len(Ytr) > 0 else 0.0
    amount_different_values = len(np.unique(Ytr))

    print("\n=== HYBRID metrics (FINAL) ===")
    print(
        f"Train  | RMSE={m_tr['RMSE']:.6f}  MAE={m_tr['MAE']:.6f}  "
        f"Bias={m_tr['Bias']:.6f}  R²={m_tr['R2']:.6f}"
    )
    print(
        f"Test   | RMSE={m_te['RMSE']:.6f}  MAE={m_te['MAE']:.6f}  MAE_95p={mae_95p:.6f}  "
        f"Bias={m_te['Bias']:.6f}  R²={m_te['R2']:.6f}  SMAPE={smape_val:.3f}%  "
        f"%MinMax={pct_minmax:.6f}  #DiffValues={amount_different_values}"
    )

    # runtime benchmark on hybrid
    running_time = benchmark_hybrid_predict(clf, reg, Xte_s, categories, n_repeats=100)
    n_iter_used = N_ITER_TUNING if tune_hyperparameters else 1

    # =========================
    # 4) Save models and outputs
    # =========================
    reg_path = os.path.join(TRAINED_DIR, f"{folder_name}_regressor.joblib")
    scaler_path = os.path.join(TRAINED_DIR, f"{folder_name}_scaler.joblib")
    categories_path = os.path.join(TRAINED_DIR, f"{folder_name}_categories.joblib")

    joblib.dump(reg, reg_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({"categories": categories}, categories_path)

    print("\n=== Saved hybrid model artifacts ===")
    print(f"Classifier: {clf_path}")
    print(f"Regressor:  {reg_path}")
    print(f"Scaler:     {scaler_path}")
    print(f"Categories: {categories_path}")
    print(f"CM plot:    {cm_path}")

    # prediction vs real CSV (HYBRID)
    from Single_target.settings_single import get_outcome_meta
    outcome_meta = get_outcome_meta(target_variable)
    variable = outcome_meta.get("variable", target_variable)

    import pandas as pd
    df_pvr = pd.DataFrame(
        {
            "y_true": Yte,
            "y_pred": yhat_te_hybrid,
            "pred_class": pred_cls_te.astype(int),  # 0=min, 1=middle, 2=max (predicted)
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
    print(f"\nSaved prediction vs real data to: {pvr_csv_path}")

    # regression plot (HYBRID)
    plot_path = os.path.join(PVR_DIR, f"{folder_name}_true_vs_pred_hybrid.png")
    plot_true_vs_pred_regression(
        Yte,
        yhat_te_hybrid,
        plot_path,
        title="XGB 2-fold",
    )
    print(f"Saved hybrid true-vs-pred plot to: {plot_path}")

    # =========================
    # 5) Calculate additional metrics for regression and category splits
    # =========================
    
    # Identify REST vs category samples in test set
    n_categories = len(categories)
    rest_mask_test = (pred_cls_te == n_categories)
    clas_mask_test = (pred_cls_te < n_categories)
    
    REG_cat = int(np.sum(rest_mask_test))  # Number of REST samples
    CLAS_cat = int(np.sum(clas_mask_test))  # Number of category samples
    
    # Calculate RTA metrics for REST samples only
    RTA1reg = 0
    RTA5reg = 0
    RTA10reg = 0
    
    if REG_cat > 0:
        rest_true = Yte[rest_mask_test]
        rest_pred = yhat_te_hybrid[rest_mask_test]
        abs_errors_rest = np.abs(rest_pred - rest_true)
        true_vals_abs_rest = np.abs(rest_true)
        
        RTA1reg = int(np.sum(abs_errors_rest <= 0.01 * true_vals_abs_rest))
        RTA5reg = int(np.sum(abs_errors_rest <= 0.05 * true_vals_abs_rest))
        RTA10reg = int(np.sum(abs_errors_rest <= 0.10 * true_vals_abs_rest))
    
    # Calculate classifier accuracy for categories only (excluding REST)
    classifier_accuracy_categories = 0.0
    if CLAS_cat > 0:
        clas_true = y_te_cls_true[clas_mask_test]
        clas_pred = y_te_cls_pred[clas_mask_test]
        classifier_accuracy_categories = float(accuracy_score(clas_true, clas_pred))
    
    # Calculate combined accuracy metrics: (correctly classified categories + correctly predicted REST) / total
    num_correctly_classified_categories = int(CLAS_cat * classifier_accuracy_categories)
    accuracy_reg_1pct = float((num_correctly_classified_categories + RTA1reg) / len(Yte))
    accuracy_reg_5pct = float((num_correctly_classified_categories + RTA5reg) / len(Yte))
    accuracy_reg_10pct = float((num_correctly_classified_categories + RTA10reg) / len(Yte))
    
    # =========================
    # 6) Log run metrics
    # =========================
    unit = get_tech_unit(technology) if technology else None
    
    # Log to standard surrogate_run_log.csv
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
        hyperparameters={
            "classifier": {
                **classifier_best_params,
                "categories": categories,
            },
            "regressor": best_params,
            "categories_threshold": CATEGORY_THRESHOLD,
            "classifier_acc_test": float(clf_acc),
        },
    )
    
    # Log to classifier-specific CSV (2fold_performance.csv)
    print(f"DEBUG: About to call log_classifier_run")
    try:
        log_classifier_run(
            surrogate_method=surrogate_method,
            target_variable=target_variable,
            node=node,
            period=period,
            technology=technology,
            commodity=commodity,
            unit=unit,
            rmse=rmse,
            nrmse=rmse_std,
            nrmse_range=nrmse_range,
            nrmse_mean=rmse_mean,
            mae=mae,
            nmae=mae_std,
            nmae_range=nmae_range,
            nmae_mean=mae_mean,
            RTA1pct=rta1_pct,
            RTA5pct=rta5_pct,
            RTA10pct=rta10_pct,
            training_time_s=training_time,
            running_time_s=running_time,
            std_target=std_target,
            classifier_accuracy=float(clf_acc),
            categories=categories,
            REG_cat=REG_cat,
            RTA1reg=RTA1reg,
            RTA5reg=RTA5reg,
            RTA10reg=RTA10reg,
            CLAS_cat=CLAS_cat,
            classifier_accuracy_categories=classifier_accuracy_categories,
            accuracy_reg_1pct=accuracy_reg_1pct,
            accuracy_reg_5pct=accuracy_reg_5pct,
            accuracy_reg_10pct=accuracy_reg_10pct,
            hyperparameters={
                "classifier": {
                    **classifier_best_params,
                    "categories": categories,
                },
                "regressor": best_params,
                "categories_threshold": CATEGORY_THRESHOLD,
            },
        )
        print(f"DEBUG: log_classifier_run completed successfully")
    except Exception as e:
        print(f"DEBUG ERROR: log_classifier_run failed with: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    raise RuntimeError(
        "run_xgb_single(...) must be called with explicit arguments "
        "(target_variable, node, commodity, technology, period). "
        "Use this module via your Streamlit main.py."
    )

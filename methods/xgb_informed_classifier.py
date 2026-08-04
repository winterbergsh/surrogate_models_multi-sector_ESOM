"""
Informed XGB Hybrid Model

Same base logic as XGB_HYBRID, with informed learning for techUseNet.

Logic:
1. Predict techStocks with the same classifier + regressor hybrid method.
2. Convert techUseNet into a bounded use/stock ratio.
3. Train the ratio target with the same classifier + regressor hybrid method.
4. If the ratio target has too few REST samples, use classifier-only fallback.
5. Convert predicted ratio back with sigmoid.
6. Enforce: techUseNet_pred <= K * techStocks_pred.

Important:
- techStocks keeps the original XGB_HYBRID skip behavior.
- classifier-only fallback is only allowed for the informed use-ratio model.
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import joblib

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    make_scorer,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from scipy.stats import randint, loguniform, uniform

from Single_target.settings_single import (
    SEED,
    MAX_TRIALS,
    N_SPLITS,
    get_save_dir,
    get_scaler,
    get_tech_unit,
    INPUT_PATH,
    TRAINING_DATA_DIR,
    JOINT_LEARNING_DIR,
    JL_MODELS_DIR,
    JL_PREDICTIONS_DIR,
    JL_METRICS_DIR,
)

from Single_target.data_prep_single import prepare_data
from Single_target.cap2act_constraints import get_cap2act


# ============================================================
# Settings
# ============================================================

N_ITER_TUNING = MAX_TRIALS
N_FOLDS_CV = N_SPLITS
CATEGORY_THRESHOLD = 20

MODEL_LABEL = "Informed_XGB_HYBRID"
FILENAME_PREFIX = "Informed_XGB_HYBRID"

CONSTRAINT_EPS = 1e-6

JL_LOG_CSV = os.path.join(
    JOINT_LEARNING_DIR,
    "joint_learning_informed_classifier_run_log.csv",
)

os.makedirs(JL_MODELS_DIR, exist_ok=True)
os.makedirs(JL_PREDICTIONS_DIR, exist_ok=True)
os.makedirs(JL_METRICS_DIR, exist_ok=True)


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


PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 1000),
    "learning_rate": loguniform(1e-3, 2e-1),
    "max_depth": randint(3, 6),
    "min_child_weight": loguniform(1e-1, 1e2),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_alpha": loguniform(1e-9, 1e1),
    "reg_lambda": loguniform(1e-2, 1e2),
}


WEATHER_DEPENDENT_K_MAP = {
    "PNL01_14": {1: 10.5, 2: 9.5, 3: 6.6, 4: 7.9, 5: 7.2, 6: 7.5, 7: 6.2, 8: 6.2},
    "PNL01_15": {1: 18.1, 2: 16.7, 3: 17.2, 4: 19.5, 5: 21.2, 6: 18.8, 7: 16.5, 8: 16.5},
    "PNL01_16": {1: 4.2, 2: 3.3, 3: 3.8, 4: 3.5, 5: 2.7, 6: 3.4, 7: 3.3, 8: 3.3},
    "PNL01_17": {1: 4.2, 2: 3.3, 3: 3.8, 4: 3.5, 5: 2.7, 6: 3.4, 7: 3.3, 8: 3.3},
}

classifier_best_params = {}

# ============================================================
# General helpers
# ============================================================

def safe(x):
    return "none" if x is None else str(x).replace("/", "_").replace(" ", "_")


def set_seeds(seed: int = SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def sigmoid(z):
    z = np.asarray(z)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def rmse_func(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred))

    std_target = float(np.std(y_true))
    value_range = float(np.max(y_true) - np.min(y_true))
    mean_target = float(np.mean(np.abs(y_true)))

    nrmse = float(rmse / std_target) if std_target > 0 else None
    nmae = float(mae / std_target) if std_target > 0 else None
    nrmse_range = float(rmse / value_range) if value_range > 0 else None
    nmae_range = float(mae / value_range) if value_range > 0 else None
    nrmse_mean = float(rmse / mean_target) if mean_target > 0 else None
    nmae_mean = float(mae / mean_target) if mean_target > 0 else None

    abs_errors = np.abs(y_pred - y_true)
    true_abs = np.abs(y_true)

    rta1 = float(np.sum(abs_errors <= 0.01 * true_abs) / len(y_true)) if len(y_true) > 0 else 0.0
    rta5 = float(np.sum(abs_errors <= 0.05 * true_abs) / len(y_true)) if len(y_true) > 0 else 0.0
    rta10 = float(np.sum(abs_errors <= 0.10 * true_abs) / len(y_true)) if len(y_true) > 0 else 0.0

    smape_den = np.abs(y_true) + np.abs(y_pred)
    smape = float(
        np.mean(
            np.where(
                smape_den == 0,
                0.0,
                2.0 * np.abs(y_pred - y_true) / smape_den,
            )
        ) * 100.0
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "R2": r2,
        "NRMSE": nrmse,
        "NMAE": nmae,
        "NRMSE_RANGE": nrmse_range,
        "NMAE_RANGE": nmae_range,
        "NRMSE_MEAN": nrmse_mean,
        "NMAE_MEAN": nmae_mean,
        "RTA1pct": rta1,
        "RTA5pct": rta5,
        "RTA10pct": rta10,
        "SMAPE": smape,
    }


def _values_match(left, right):
    if pd.isna(left) and pd.isna(right):
        return True

    left_is_number = isinstance(left, (int, float, np.integer, np.floating))
    right_is_number = isinstance(right, (int, float, np.integer, np.floating))

    if left_is_number and right_is_number:
        return bool(np.isclose(float(left), float(right), equal_nan=True))

    return str(left) == str(right)


def _baseline_stock_prediction_path(stock_target, commodity, tune_hyperparameters, sample_size):
    baseline_dir = os.path.join(
        get_save_dir("XGB_HYBRID"),
        "pred_vs_actual_plots",
        "XGB_HYBRID",
    )

    method_suffix = "XGB_HYBRID_tuned" if tune_hyperparameters else "XGB_HYBRID_quick"
    sample_suffix = f"_sample{sample_size}" if sample_size is not None and sample_size != 4000 else ""

    candidates = [
        os.path.join(baseline_dir, f"{safe(stock_target)}_{safe(commodity)}_{method_suffix}.csv"),
        os.path.join(baseline_dir, f"{safe(stock_target)}_{safe(commodity)}{sample_suffix}_{method_suffix}.csv"),
        os.path.join(baseline_dir, f"{safe(stock_target)}_{safe(commodity)}_{method_suffix}{sample_suffix}.csv"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find baseline techStocks predictions for '{stock_target}'. Checked: {candidates}"
    )


def _load_baseline_stock_predictions(
    stock_target,
    y_test_stock,
    node,
    period,
    technology,
    commodity,
    tune_hyperparameters,
    sample_size,
):
    prediction_path = _baseline_stock_prediction_path(
        stock_target,
        commodity,
        tune_hyperparameters,
        sample_size,
    )

    df_pred = pd.read_csv(prediction_path)

    actual_col = "techStocks_actual" if "techStocks_actual" in df_pred.columns else "y_true"
    pred_col = "techStocks_predicted" if "techStocks_predicted" in df_pred.columns else "y_pred"
    pred_class_col = "techStocks_pred_class" if "techStocks_pred_class" in df_pred.columns else "pred_class"

    missing_columns = [
        col for col in [actual_col, pred_col] if col not in df_pred.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Baseline techStocks prediction file is missing required columns: {missing_columns}"
        )

    if len(df_pred) != len(y_test_stock):
        raise ValueError(
            f"Baseline techStocks prediction row count mismatch for '{stock_target}': "
            f"expected {len(y_test_stock)}, found {len(df_pred)}"
        )

    loaded_actual = pd.to_numeric(df_pred[actual_col], errors="coerce").to_numpy(dtype=float)
    loaded_predicted = pd.to_numeric(df_pred[pred_col], errors="coerce").to_numpy(dtype=float)

    if not np.allclose(loaded_actual, np.asarray(y_test_stock, dtype=float), equal_nan=True):
        raise ValueError(
            f"Baseline techStocks rows for '{stock_target}' do not align with the current test split."
        )

    if pred_class_col in df_pred.columns:
        pred_classes = pd.to_numeric(df_pred[pred_class_col], errors="coerce").fillna(-1).astype(int).to_numpy()
    else:
        pred_classes = np.full(len(df_pred), -1, dtype=int)

    metadata_checks = {
        "target_variable": stock_target,
        "node": node,
        "period": period,
        "technology": technology,
        "commodity": commodity,
    }
    for column, expected_value in metadata_checks.items():
        if column in df_pred.columns and len(df_pred) > 0:
            actual_value = df_pred[column].iloc[0]
            if not _values_match(actual_value, expected_value):
                raise ValueError(
                    f"Baseline techStocks file mismatch for '{column}'. Expected {expected_value!r}, found {actual_value!r}."
                )

    metrics_path = os.path.join(get_save_dir("XGB_HYBRID"), "2fold_performance.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Baseline metrics file not found: {metrics_path}")

    df_metrics = pd.read_csv(metrics_path)
    df_metrics = df_metrics[df_metrics["target_variable"].astype(str) == str(stock_target)].copy()

    if "period" in df_metrics.columns:
        df_metrics = df_metrics[df_metrics["period"].apply(lambda x: _values_match(x, period))]
    if "technology" in df_metrics.columns:
        df_metrics = df_metrics[df_metrics["technology"].apply(lambda x: _values_match(x, technology))]
    if "commodity" in df_metrics.columns:
        df_metrics = df_metrics[df_metrics["commodity"].apply(lambda x: _values_match(x, commodity))]
    if "node" in df_metrics.columns:
        df_metrics = df_metrics[df_metrics["node"].apply(lambda x: _values_match(x, node))]
    if tune_hyperparameters and "surrogate_method" in df_metrics.columns:
        df_metrics = df_metrics[
            df_metrics["surrogate_method"].astype(str).str.contains("tuned", na=False)
        ]

    if df_metrics.empty:
        raise ValueError(f"Could not find a matching baseline metrics row for '{stock_target}' in {metrics_path}")

    if "timestamp" in df_metrics.columns:
        df_metrics = df_metrics.sort_values("timestamp", na_position="last")

    baseline_metrics_row = df_metrics.iloc[-1].to_dict()

    return {
        "prediction_path": prediction_path,
        "metrics_path": metrics_path,
        "loaded_actual": loaded_actual,
        "loaded_predicted": loaded_predicted,
        "pred_classes": pred_classes,
        "baseline_metrics_row": baseline_metrics_row,
    }


# ============================================================
# Hybrid classifier-regressor helpers
# ============================================================

def identify_categories(y: np.ndarray, threshold: int = CATEGORY_THRESHOLD):
    y = np.asarray(y).ravel()
    unique_vals, counts = np.unique(y, return_counts=True)
    categories = unique_vals[counts >= threshold]

    category_indices = np.zeros(len(y), dtype=bool)
    for cat in categories:
        category_indices |= y == cat

    rest_indices = ~category_indices
    return list(categories), category_indices, rest_indices


def make_category_classes(y: np.ndarray, categories: list):
    y = np.asarray(y).ravel()
    n_categories = len(categories)
    cls = np.full(len(y), n_categories, dtype=np.int64)

    for i, cat in enumerate(categories):
        cls[y == cat] = i

    return cls

def safe_accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)

    y_pred = y_pred.ravel()
    return accuracy_score(y_true, y_pred)

def train_classifier(X_train, y_train, threshold: int = CATEGORY_THRESHOLD):
    categories, _, _ = identify_categories(y_train, threshold=threshold)
    y_train_cls = make_category_classes(y_train, categories)
    n_classes = len(categories) + 1

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
        scoring=make_scorer(safe_accuracy_score),
        cv=N_FOLDS_CV,
        verbose=1,
        n_jobs=-1,
        refit=True,
        random_state=SEED,
    )

    search.fit(X_train, y_train_cls)

    clf = search.best_estimator_
    best_params = search.best_params_

    return clf, categories, y_train_cls, best_params


def hybrid_predict(clf, reg, X, categories, fallback_value=None):
    pred_cls = clf.predict(X)

    if pred_cls.ndim == 2:
        pred_cls = np.argmax(pred_cls, axis=1)

    y_pred = np.empty(X.shape[0], dtype=float)
    n_categories = len(categories)

    for i, cat_val in enumerate(categories):
        mask = pred_cls == i
        y_pred[mask] = cat_val

    rest_mask = pred_cls == n_categories

    if np.any(rest_mask):
        if reg is not None:
            y_pred[rest_mask] = reg.predict(X[rest_mask])
        else:
            y_pred[rest_mask] = fallback_value

    return y_pred, pred_cls


def benchmark_xgb_predict(model, X, n_repeats: int = 100):
    _ = model.predict(X)

    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = model.predict(X)

    elapsed = time.perf_counter() - start
    return (elapsed / n_repeats) / X.shape[0]


def benchmark_hybrid_predict(clf, reg, X, categories, fallback_value=None, n_repeats: int = 100):
    _ = clf.predict(X)

    if reg is not None:
        _ = reg.predict(X[: min(len(X), 10)])

    start = time.perf_counter()

    for _ in range(n_repeats):
        _ = hybrid_predict(
            clf,
            reg,
            X,
            categories,
            fallback_value=fallback_value,
        )

    elapsed = time.perf_counter() - start
    return (elapsed / n_repeats) / X.shape[0]


def benchmark_informed_xgb_predict(
    y_stock_pred,
    use_ratio_result,
    X,
    K,
    n_repeats: int = 100,
):
    K = np.asarray(K, dtype=float)
    y_stock_pred = np.asarray(y_stock_pred, dtype=float).ravel()

    if len(y_stock_pred) != len(X):
        raise ValueError("Baseline stock predictions must align with the inference feature matrix.")

    def _predict_once():
        if use_ratio_result["clf"] is None:
            z_use = use_ratio_result["reg"].predict(X)
        else:
            z_use, _ = hybrid_predict(
                use_ratio_result["clf"],
                use_ratio_result["reg"],
                X,
                use_ratio_result["categories"],
                fallback_value=use_ratio_result["fallback_value"],
            )

        ratio = sigmoid(z_use)
        y_use = np.clip(y_stock_pred * K * ratio, 0, None)

        return y_use

    _ = _predict_once()

    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = _predict_once()

    elapsed = time.perf_counter() - start
    return (elapsed / n_repeats) / X.shape[0]


def train_xgb_hybrid_target(
    X_train,
    y_train,
    X_test,
    y_test=None,
    tune_hyperparameters: bool = True,
    allow_classifier_only_fallback: bool = False,
):
    """
    Mirrors the original XGB_HYBRID logic.

    If allow_classifier_only_fallback=False:
        - Skip when REST samples are insufficient.

    If allow_classifier_only_fallback=True:
        - Use classifier-only fallback when REST samples are insufficient.
        - This should only be used for the informed use-ratio target.
    """

    y_train = np.asarray(y_train).ravel()
    categories, _, _ = identify_categories(y_train, threshold=CATEGORY_THRESHOLD)

    clf = None
    reg = None
    best_params = {}
    fallback_value = float(np.mean(y_train))

    clf_acc_train = None
    clf_acc_test = None
    clf_cm_test = None
    classification_report_test = None

    pure_regression = False
    classifier_only_fallback = False
    active_threshold = CATEGORY_THRESHOLD

    # ========================================================
    # Pure regression path
    # ========================================================
    if len(categories) == 0:
        print("No recurring categories found. Using pure regression.")

        pure_regression = True
        X_train_reg = X_train
        y_train_reg = y_train

    # ========================================================
    # Hybrid path
    # ========================================================
    else:
        print(f"{len(categories)} recurring categories found. Using hybrid classifier-regressor.")

        clf, categories, y_train_cls, classifier_best_params = train_classifier(
            X_train,
            y_train,
            threshold=CATEGORY_THRESHOLD,
        )

        rest_mask = y_train_cls == len(categories)
        rest_count = int(np.sum(rest_mask))

        if rest_count < 10:
            if not allow_classifier_only_fallback:
                print(f"Too few REST samples ({rest_count}). Skipping this target.")
                return None

            print(f"Too few REST samples ({rest_count}). Using classifier-only fallback.")

            classifier_only_fallback = True
            reg = None
            best_params = {
                "classifier_only_fallback": True,
                "fallback_value": fallback_value,
                "rest_count": rest_count,
            }

            y_pred_train, pred_cls_train = hybrid_predict(
                clf,
                reg,
                X_train,
                categories,
                fallback_value=fallback_value,
            )

            y_pred_test, pred_cls_test = hybrid_predict(
                clf,
                reg,
                X_test,
                categories,
                fallback_value=fallback_value,
            )

            y_train_cls_true = make_category_classes(y_train, categories)
            clf_acc_train = accuracy_score(y_train_cls_true, pred_cls_train)

            if y_test is not None:
                y_test_cls_true = make_category_classes(y_test, categories)
                n_classes = len(categories) + 1

                clf_acc_test = accuracy_score(y_test_cls_true, pred_cls_test)
                clf_cm_test = confusion_matrix(
                    y_test_cls_true,
                    pred_cls_test,
                    labels=list(range(n_classes)),
                )

                class_names = [f"Cat_{val}" for val in categories] + ["REST"]
                classification_report_test = classification_report(
                    y_test_cls_true,
                    pred_cls_test,
                    labels=list(range(n_classes)),
                    target_names=class_names,
                    zero_division=0,
                )

            running_time_single_target = benchmark_hybrid_predict(
                clf,
                reg,
                X_test,
                categories,
                fallback_value=fallback_value,
                n_repeats=100,
            )

            return {
                "clf": clf,
                "reg": reg,
                "categories": categories,
                "best_params": best_params,
                "fallback_value": fallback_value,
                "classifier_only_fallback": classifier_only_fallback,
                "y_pred_train": y_pred_train,
                "y_pred_test": y_pred_test,
                "pred_cls_train": pred_cls_train,
                "pred_cls_test": pred_cls_test,
                "classifier_accuracy_train": clf_acc_train,
                "classifier_accuracy_test": clf_acc_test,
                "confusion_matrix_test": clf_cm_test,
                "classification_report_test": classification_report_test,
                "pure_regression": pure_regression,
                "categories_threshold": active_threshold,
                "running_time_single_target_s": running_time_single_target,
            }

        X_train_reg = X_train[rest_mask]
        y_train_reg = y_train[rest_mask]

    # ========================================================
    # Train regressor
    # ========================================================

    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=1,
    )

    rmse_scorer = make_scorer(rmse_func, greater_is_better=False)

    if tune_hyperparameters:
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=N_ITER_TUNING,
            scoring=rmse_scorer,
            cv=N_FOLDS_CV,
            verbose=1,
            n_jobs=-1,
            refit=True,
            random_state=SEED,
        )

        search.fit(X_train_reg, y_train_reg)
        reg = search.best_estimator_
        best_params = search.best_params_

    else:
        reg = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS, random_state=SEED, n_jobs=1)
        reg.fit(X_train_reg, y_train_reg)
        best_params = DEFAULT_XGB_PARAMS.copy()

    # ========================================================
    # Predict
    # ========================================================

    if pure_regression:
        y_pred_train = reg.predict(X_train)
        y_pred_test = reg.predict(X_test)

        pred_cls_train = np.full(len(y_pred_train), -1, dtype=int)
        pred_cls_test = np.full(len(y_pred_test), -1, dtype=int)

        running_time_single_target = benchmark_xgb_predict(
            reg,
            X_test,
            n_repeats=100,
        )

    else:
        y_pred_train, pred_cls_train = hybrid_predict(
            clf,
            reg,
            X_train,
            categories,
            fallback_value=fallback_value,
        )

        y_pred_test, pred_cls_test = hybrid_predict(
            clf,
            reg,
            X_test,
            categories,
            fallback_value=fallback_value,
        )

        y_train_cls_true = make_category_classes(y_train, categories)
        clf_acc_train = accuracy_score(y_train_cls_true, pred_cls_train)

        if y_test is not None:
            y_test_cls_true = make_category_classes(y_test, categories)
            y_test_cls_true = np.asarray(y_test_cls_true).ravel()
            pred_cls_test = np.asarray(pred_cls_test).ravel()

            n_classes = len(categories) + 1
            clf_acc_test = accuracy_score(y_test_cls_true, pred_cls_test)
            clf_cm_test = confusion_matrix(
                y_test_cls_true,
                pred_cls_test,
                labels=list(range(n_classes)),
            )

            class_names = [f"Cat_{val}" for val in categories] + ["REST"]
            classification_report_test = classification_report(
                y_test_cls_true,
                pred_cls_test,
                labels=list(range(n_classes)),
                target_names=class_names,
                zero_division=0,
            )

            print("\n=== Classifier test metrics ===")
            print(f"Classifier accuracy test: {clf_acc_test:.4f}")
            print("Confusion matrix:")
            print(clf_cm_test)
            print("Classification report:")
            print(classification_report_test)

        running_time_single_target = benchmark_hybrid_predict(
            clf,
            reg,
            X_test,
            categories,
            fallback_value=fallback_value,
            n_repeats=100,
        )

    return {
        "clf": clf,
        "reg": reg,
        "categories": categories,
        "best_params": best_params,
        "fallback_value": fallback_value,
        "classifier_only_fallback": classifier_only_fallback,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "pred_cls_train": pred_cls_train,
        "pred_cls_test": pred_cls_test,
        "classifier_accuracy_train": clf_acc_train,
        "classifier_accuracy_test": clf_acc_test,
        "confusion_matrix_test": clf_cm_test,
        "classification_report_test": classification_report_test,
        "pure_regression": pure_regression,
        "categories_threshold": active_threshold,
        "running_time_single_target_s": running_time_single_target,
    }


# ============================================================
# Weather helpers
# ============================================================

def _load_wind_weather_conditions() -> dict:
    try:
        if INPUT_PATH.lower().endswith(".parquet"):
            df = pd.read_parquet(INPUT_PATH, columns=["Variant", "Weather Conditions"])
        else:
            df = pd.read_excel(
                INPUT_PATH,
                engine="openpyxl",
                usecols=["Variant", "Weather Conditions"],
            )
        df = df.dropna(subset=["Variant", "Weather Conditions"])

        return dict(
            zip(
                df["Variant"].astype(str),
                pd.to_numeric(df["Weather Conditions"], errors="coerce"),
            )
        )

    except Exception as e:
        print(f"Warning: Could not load wind weather conditions from {INPUT_PATH}: {e}")
        return {}


def get_weather_regimes(
    technology,
    period,
    node=None,
    commodity=None,
    sample_size=None,
):
    if technology not in WEATHER_DEPENDENT_K_MAP:
        raise ValueError(
            f"get_weather_regimes() was called for non-weather-dependent technology '{technology}'."
        )

    weather_map = _load_wind_weather_conditions()

    if not weather_map:
        raise ValueError(
            f"{technology} is weather-dependent, but the weather lookup table could not be loaded from {INPUT_PATH}."
        )

    stock_target = f"techStocks {period} {technology}"

    merged_path = os.path.join(
        TRAINING_DATA_DIR,
        f"training_merged_{safe(stock_target)}_{safe(commodity)}.parquet",
    )

    if not os.path.exists(merged_path):
        prepare_data(
            stock_target,
            node=node,
            period=period,
            technology=technology,
            commodity=commodity,
            sample_size=sample_size,
        )

    if not os.path.exists(merged_path):
        raise ValueError(
            f"{technology} is weather-dependent, but merged stock data was not found: {merged_path}"
        )

    merged_df = pd.read_parquet(merged_path)

    if sample_size is not None and len(merged_df) > sample_size:
        merged_df = merged_df.sample(n=sample_size, random_state=SEED)

    if "Variant" not in merged_df.columns:
        raise ValueError(
            f"Merged stock data for '{stock_target}' does not contain a Variant column: {merged_path}"
        )

    variant_list = merged_df["Variant"].astype(str).to_numpy()

    if len(variant_list) < 10:
        raise ValueError(
            f"Too few samples to derive weather regimes for '{stock_target}' from {merged_path}."
        )

    test_size_to_use = 600 if len(variant_list) >= 700 else int(len(variant_list) * 0.15)

    variant_train, variant_test = train_test_split(
        variant_list,
        test_size=test_size_to_use,
        random_state=SEED,
        shuffle=True,
    )

    def _map_variants_to_regimes(variants, split_name):
        regimes = []
        missing = []

        for variant in variants:
            weather_val = weather_map.get(str(variant), np.nan)

            if pd.isna(weather_val):
                missing.append(str(variant))
                continue

            regimes.append(int(np.clip(np.floor(float(weather_val)), 1, 8)))

        if missing:
            raise ValueError(
                f"{len(missing)} variants in the {split_name} split were not found in the weather lookup table. "
                f"First missing variants: {missing[:5]}"
            )

        return np.asarray(regimes, dtype=int)

    y_train_regime = _map_variants_to_regimes(variant_train, "train")
    y_test_regime = _map_variants_to_regimes(variant_test, "test")

    print(f"Loaded weather regimes for {technology}, period {period}")
    return y_train_regime, y_test_regime


def get_k_array(technology, y_regime, static_k):
    if technology not in WEATHER_DEPENDENT_K_MAP:
        return np.full(len(y_regime), static_k, dtype=float)

    k_map = WEATHER_DEPENDENT_K_MAP[technology]
    return np.array([k_map[int(r)] for r in y_regime], dtype=float)


# ============================================================
# Metric split helpers
# ============================================================

def compute_regression_category_split_metrics(y_true, y_pred, pred_cls, categories):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    pred_cls = np.asarray(pred_cls).ravel()

    if len(categories) == 0:
        rest_mask = np.ones(len(y_true), dtype=bool)
        clas_mask = np.zeros(len(y_true), dtype=bool)
    else:
        n_categories = len(categories)
        rest_mask = pred_cls == n_categories
        clas_mask = pred_cls < n_categories

    reg_cat = int(np.sum(rest_mask))
    clas_cat = int(np.sum(clas_mask))

    rta1reg = 0
    rta5reg = 0
    rta10reg = 0

    if reg_cat > 0:
        rest_true = y_true[rest_mask]
        rest_pred = y_pred[rest_mask]
        abs_errors_rest = np.abs(rest_pred - rest_true)
        true_vals_abs_rest = np.abs(rest_true)

        rta1reg = int(np.sum(abs_errors_rest <= 0.01 * true_vals_abs_rest))
        rta5reg = int(np.sum(abs_errors_rest <= 0.05 * true_vals_abs_rest))
        rta10reg = int(np.sum(abs_errors_rest <= 0.10 * true_vals_abs_rest))

    return {
        "REG_cat": reg_cat,
        "CLAS_cat": clas_cat,
        "RTA1reg": rta1reg,
        "RTA5reg": rta5reg,
        "RTA10reg": rta10reg,
    }


# ============================================================
# Main model
# ============================================================

def run_joint_xgboost(
    technology: str,
    period: int,
    node=None,
    commodity=None,
    tune_hyperparameters: bool = True,
    filter_constant_output: bool = False,
    sample_size=None,
    overwrite: bool = False,
    cores_per_job: int | None = None,
) -> None:

    set_seeds(SEED)

    print("\n" + "=" * 80)
    print(f"Informed XGB HYBRID - {technology} - {period}")
    print("=" * 80)

    stock_target = f"techStocks {period} {technology}"
    use_target = f"techUseNet {period} {technology}"

    tuning_suffix = "_tuned" if tune_hyperparameters else "_quick"
    sample_suffix = f"_sample{sample_size}" if sample_size is not None and sample_size != 4000 else ""

    filename_base = f"{FILENAME_PREFIX}{tuning_suffix}_{safe(technology)}_{period}{sample_suffix}"

    pred_path = os.path.join(JL_PREDICTIONS_DIR, f"{filename_base}.csv")
    metrics_path = os.path.join(JL_METRICS_DIR, f"{filename_base}.csv")

    if os.path.exists(pred_path) and os.path.exists(metrics_path) and not overwrite:
        print(f"Run already exists: {filename_base}")
        return

    # ========================================================
    # 1. Load data
    # ========================================================

    print("\n[1/6] Loading data")

    try:
        x_train, y_train_stock, x_test, y_test_stock = prepare_data(
            stock_target,
            node=node,
            period=period,
            technology=technology,
            commodity=commodity,
            sample_size=sample_size,
        )

        _, y_train_use, _, y_test_use = prepare_data(
            use_target,
            node=node,
            period=period,
            technology=technology,
            commodity=commodity,
            sample_size=sample_size,
        )

    except ValueError as e:
        print(f"Skipping {stock_target} and {use_target}: data not available")
        print(f"Error: {e}")
        return

    X_train = to_numpy(x_train)
    X_test = to_numpy(x_test)

    y_train_stock = to_numpy(y_train_stock).ravel()
    y_test_stock = to_numpy(y_test_stock).ravel()
    y_train_use = to_numpy(y_train_use).ravel()
    y_test_use = to_numpy(y_test_use).ravel()

    min_train_len = min(len(X_train), len(y_train_stock), len(y_train_use))
    min_test_len = min(len(X_test), len(y_test_stock), len(y_test_use))

    X_train = X_train[:min_train_len]
    y_train_stock = y_train_stock[:min_train_len]
    y_train_use = y_train_use[:min_train_len]

    X_test = X_test[:min_test_len]
    y_test_stock = y_test_stock[:min_test_len]
    y_test_use = y_test_use[:min_test_len]

    if filter_constant_output:
        if np.std(y_train_stock) < 1e-6 or np.std(y_train_use) < 1e-6:
            print("Skipping because stock or use output is constant.")
            return

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    # ========================================================
    # 2. Scale features
    # ========================================================

    print("\n[2/6] Scaling features")

    scaler = get_scaler()

    if scaler is None:
        X_train_s = X_train
        X_test_s = X_test
    else:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

    # ========================================================
    # 3. Load stock baseline predictions
    # ========================================================

    print("\n[3/6] Loading baseline techStocks predictions")

    baseline_stock = _load_baseline_stock_predictions(
        stock_target=stock_target,
        y_test_stock=y_test_stock,
        node=node,
        period=period,
        technology=technology,
        commodity=commodity,
        tune_hyperparameters=tune_hyperparameters,
        sample_size=sample_size,
    )

    y_pred_stock_test = np.clip(baseline_stock["loaded_predicted"], 0, None)
    stock_pred_cls_test = baseline_stock["pred_classes"]
    stock_metrics_row = baseline_stock["baseline_metrics_row"]

    print(
        f"Loaded stock baseline from: {baseline_stock['prediction_path']}\n"
        f"Stock test: RMSE={float(stock_metrics_row.get('rmse', np.nan)):.6f}, "
        f"MAE={float(stock_metrics_row.get('mae', np.nan)):.6f}"
    )

    # ========================================================
    # 4. Create informed ratio target
    # ========================================================

    print("\n[4/6] Creating informed techUseNet ratio target")

    static_k = get_cap2act(technology)
    is_weather_dependent = technology in WEATHER_DEPENDENT_K_MAP

    if is_weather_dependent:
        y_regime_train, y_regime_test = get_weather_regimes(
            technology=technology,
            period=period,
            node=node,
            commodity=commodity,
            sample_size=sample_size,
        )

        y_regime_train = y_regime_train[:len(y_train_use)]
        y_regime_test = y_regime_test[:len(y_test_use)]

        if len(y_regime_train) != len(y_train_use) or len(y_regime_test) != len(y_test_use):
            raise ValueError(
                "Weather regime lengths do not match train/test target lengths after alignment."
            )

    else:
        y_regime_train = np.zeros(len(y_train_use), dtype=int)
        y_regime_test = np.zeros(len(y_test_use), dtype=int)

    K_train = get_k_array(technology, y_regime_train, static_k)
    K_test = get_k_array(technology, y_regime_test, static_k)

    z_train = np.zeros_like(y_train_use, dtype=float)

    for i in range(len(y_train_use)):
        if y_train_stock[i] > 1e-6 and K_train[i] > 0:
            ratio = y_train_use[i] / y_train_stock[i]
            ratio = min(K_train[i], ratio)
            ratio = max(0.0, ratio)
            z_train[i] = logit(ratio / K_train[i])
        else:
            z_train[i] = logit(1e-6)

    # ========================================================
    # 5. Train informed use-ratio model
    # ========================================================

    print("\n[5/6] Training informed use-ratio hybrid model")

    use_training_start = time.perf_counter()

    use_ratio_result = train_xgb_hybrid_target(
        X_train_s,
        z_train,
        X_test_s,
        y_test=None,
        tune_hyperparameters=tune_hyperparameters,
        allow_classifier_only_fallback=True,
    )

    if use_ratio_result is None:
        print("Skipping informed run because use-ratio model could not be trained.")
        return

    training_time = time.perf_counter() - use_training_start

    z_pred_train = use_ratio_result["y_pred_train"]
    z_pred_test = use_ratio_result["y_pred_test"]

    ratio_pred_train = sigmoid(z_pred_train)
    ratio_pred_test = sigmoid(z_pred_test)

    y_pred_use_train = y_train_stock * K_train * ratio_pred_train
    y_pred_use_test = y_pred_stock_test * K_test * ratio_pred_test

    y_pred_use_train = np.clip(y_pred_use_train, 0, None)
    y_pred_use_test = np.clip(y_pred_use_test, 0, None)

    use_metrics = eval_metrics(y_test_use, y_pred_use_test)

    print(
        f"Use test: RMSE={use_metrics['RMSE']:.6f}, "
        f"MAE={use_metrics['MAE']:.6f}, "
        f"R2={use_metrics['R2']:.6f}"
    )

    # ========================================================
    # 6. Constraint violations and runtime
    # ========================================================

    print("\n[6/6] Calculating constraint violations and runtime")

    violations = y_pred_use_test > (K_test * y_pred_stock_test + CONSTRAINT_EPS)
    violation_pct = float(100.0 * np.sum(violations) / len(y_pred_use_test))

    print(f"Constraint violations: {violation_pct:.6f}%")

    running_time = benchmark_informed_xgb_predict(
        y_stock_pred=y_pred_stock_test,
        use_ratio_result=use_ratio_result,
        X=X_test_s,
        K=K_test,
        n_repeats=100,
    )

    use_split = compute_regression_category_split_metrics(
        y_test_use,
        y_pred_use_test,
        use_ratio_result["pred_cls_test"],
        use_ratio_result["categories"],
    )

    # ========================================================
    # Save models
    # ========================================================

    use_reg_path = os.path.join(JL_MODELS_DIR, f"{filename_base}_use_ratio_regressor.joblib")
    use_clf_path = os.path.join(JL_MODELS_DIR, f"{filename_base}_use_ratio_classifier.joblib")
    scaler_path = os.path.join(JL_MODELS_DIR, f"{filename_base}_scaler.joblib")
    meta_path = os.path.join(JL_MODELS_DIR, f"{filename_base}_metadata.joblib")

    if use_ratio_result["reg"] is not None:
        joblib.dump(use_ratio_result["reg"], use_reg_path)

    joblib.dump(scaler, scaler_path)

    if use_ratio_result["clf"] is not None:
        joblib.dump(use_ratio_result["clf"], use_clf_path)

    joblib.dump(
        {
            "technology": technology,
            "period": period,
            "static_k": static_k,
            "is_weather_dependent": is_weather_dependent,
            "use_ratio_categories": use_ratio_result["categories"],
            "use_ratio_best_params": use_ratio_result["best_params"],
            "use_ratio_pure_regression": use_ratio_result["pure_regression"],
            "use_ratio_classifier_only_fallback": use_ratio_result["classifier_only_fallback"],
            "use_ratio_fallback_value": use_ratio_result["fallback_value"],
            "use_ratio_categories_threshold": use_ratio_result["categories_threshold"],
            "stock_source_prediction_path": baseline_stock["prediction_path"],
            "stock_source_metrics_path": baseline_stock["metrics_path"],
            "stock_source_surrogate_method": stock_metrics_row.get("surrogate_method"),
            "weather_regime_test": y_regime_test,
        },
        meta_path,
    )

    # ========================================================
    # Save predictions
    # ========================================================

    df_pred = pd.DataFrame(
        {
            "techStocks_actual": y_test_stock,
            "techStocks_predicted": y_pred_stock_test,
            "techStocks_pred_class": stock_pred_cls_test.astype(int),
            "techUseNet_actual": y_test_use,
            "techUseNet_predicted": y_pred_use_test,
            "use_ratio_latent_pred": z_pred_test,
            "use_ratio_sigmoid_pred": ratio_pred_test,
            "use_ratio_pred_class": use_ratio_result["pred_cls_test"].astype(int),
            "K": K_test,
            "weather_regime": y_regime_test,
            "is_weather_dependent": is_weather_dependent,
            "constraint_limit": K_test * y_pred_stock_test,
            "constraint_violation": violations.astype(int),
            "use_ratio_classifier_only_fallback": use_ratio_result["classifier_only_fallback"],
            "stock_source_prediction_path": baseline_stock["prediction_path"],
            "technology": technology,
            "period": period,
            "model": MODEL_LABEL,
        }
    )

    df_pred.to_csv(pred_path, index=False)

    # ========================================================
    # Save metrics
    # ========================================================

    unit = get_tech_unit(technology) if technology else None

    stock_row = {
        "timestamp": pd.Timestamp.now(),
        "surrogate_method": MODEL_LABEL,
        "target_variable": stock_target,
        "node": stock_metrics_row.get("node", node),
        "period": period,
        "technology": technology,
        "commodity": stock_metrics_row.get("commodity", commodity),
        "unit": unit,
        "sample_size": sample_size,
        "rmse": stock_metrics_row.get("rmse"),
        "nrmse": stock_metrics_row.get("nrmse"),
        "nrmse_range": stock_metrics_row.get("nrmse_range"),
        "nrmse_mean": stock_metrics_row.get("nrmse_mean"),
        "mae": stock_metrics_row.get("mae"),
        "nmae": stock_metrics_row.get("nmae"),
        "nmae_range": stock_metrics_row.get("nmae_range"),
        "nmae_mean": stock_metrics_row.get("nmae_mean"),
        "R2": stock_metrics_row.get("R2"),
        "Bias": stock_metrics_row.get("Bias"),
        "SMAPE": stock_metrics_row.get("SMAPE"),
        "RTA1pct": stock_metrics_row.get("RTA1pct"),
        "RTA5pct": stock_metrics_row.get("RTA5pct"),
        "RTA10pct": stock_metrics_row.get("RTA10pct"),
        "constraint_violations_pct": stock_metrics_row.get("constraint_violations_pct"),
        "training_time_s": stock_metrics_row.get("training_time_s"),
        "running_time_s": stock_metrics_row.get("running_time_s"),
        "single_target_running_time_s": stock_metrics_row.get("running_time_s"),
        "classifier_accuracy": stock_metrics_row.get("classifier_accuracy"),
        "classifier_accuracy_train": None,
        "categories": stock_metrics_row.get("categories"),
        "categories_threshold": None,
        "pure_regression": stock_metrics_row.get("n_categories", 0) == 0,
        "classifier_only_fallback": False,
        "fallback_value": None,
        "REG_cat": stock_metrics_row.get("REG_cat"),
        "CLAS_cat": stock_metrics_row.get("CLAS_cat"),
        "RTA1reg": stock_metrics_row.get("RTA1reg"),
        "RTA5reg": stock_metrics_row.get("RTA5reg"),
        "RTA10reg": stock_metrics_row.get("RTA10reg"),
        "hyperparameters": stock_metrics_row.get("hyperparameters"),
        "stock_source_prediction_path": baseline_stock["prediction_path"],
        "stock_source_metrics_path": baseline_stock["metrics_path"],
        "stock_source_surrogate_method": stock_metrics_row.get("surrogate_method"),
    }

    use_row = {
        "timestamp": pd.Timestamp.now(),
        "surrogate_method": MODEL_LABEL,
        "target_variable": use_target,
        "node": node,
        "period": period,
        "technology": technology,
        "commodity": commodity,
        "unit": unit,
        "sample_size": sample_size,
        "rmse": use_metrics["RMSE"],
        "nrmse": use_metrics["NRMSE"],
        "nrmse_range": use_metrics["NRMSE_RANGE"],
        "nrmse_mean": use_metrics["NRMSE_MEAN"],
        "mae": use_metrics["MAE"],
        "nmae": use_metrics["NMAE"],
        "nmae_range": use_metrics["NMAE_RANGE"],
        "nmae_mean": use_metrics["NMAE_MEAN"],
        "R2": use_metrics["R2"],
        "Bias": use_metrics["Bias"],
        "SMAPE": use_metrics["SMAPE"],
        "RTA1pct": use_metrics["RTA1pct"],
        "RTA5pct": use_metrics["RTA5pct"],
        "RTA10pct": use_metrics["RTA10pct"],
        "constraint_violations_pct": violation_pct,
        "training_time_s": training_time,
        "running_time_s": running_time,
        "single_target_running_time_s": use_ratio_result["running_time_single_target_s"],
        "classifier_accuracy": use_ratio_result["classifier_accuracy_test"],
        "classifier_accuracy_train": use_ratio_result["classifier_accuracy_train"],
        "categories": str(use_ratio_result["categories"]),
        "categories_threshold": use_ratio_result["categories_threshold"],
        "pure_regression": use_ratio_result["pure_regression"],
        "classifier_only_fallback": use_ratio_result["classifier_only_fallback"],
        "fallback_value": use_ratio_result["fallback_value"],
        "REG_cat": use_split["REG_cat"],
        "CLAS_cat": use_split["CLAS_cat"],
        "RTA1reg": use_split["RTA1reg"],
        "RTA5reg": use_split["RTA5reg"],
        "RTA10reg": use_split["RTA10reg"],
        "hyperparameters": str(use_ratio_result["best_params"]),
        "stock_source_prediction_path": baseline_stock["prediction_path"],
        "stock_source_metrics_path": baseline_stock["metrics_path"],
        "stock_source_surrogate_method": stock_metrics_row.get("surrogate_method"),
    }

    df_metrics = pd.DataFrame([stock_row, use_row])
    df_metrics.to_csv(metrics_path, index=False)

    if os.path.exists(JL_LOG_CSV):
        df_metrics.to_csv(JL_LOG_CSV, mode="a", header=False, index=False)
    else:
        df_metrics.to_csv(JL_LOG_CSV, index=False)

    print("\n=== FINAL RESULTS ===")
    print(f"Stock RMSE: {float(stock_metrics_row.get('rmse', np.nan)):.6f}")
    print(f"Use RMSE:   {use_metrics['RMSE']:.6f}")
    print(f"Violations: {violation_pct:.6f}%")
    print(f"Runtime:    {running_time:.3e} s/sample")
    print(f"Saved predictions to: {pred_path}")
    print(f"Saved metrics to:     {metrics_path}")
    print(f"Saved models to:      {JL_MODELS_DIR}")


if __name__ == "__main__":
    raise RuntimeError(
        "Call run_joint_xgboost(...) from main.py or Streamlit with explicit arguments."
    )
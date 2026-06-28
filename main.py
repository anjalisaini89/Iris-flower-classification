"""Iris Flower Classification — Enhanced Pipeline.

Enhancements over v1
--------------------
1.  Cross-validation (5-fold stratified) reported alongside train/test accuracy.
2.  K-Nearest Neighbours added as a fourth classifier.
3.  Decision Tree depth tuned via GridSearchCV.
4.  Feature-importance bar chart saved for tree-based models.
5.  Confusion-matrix heatmap saved (instead of plain text only).
6.  Per-class ROC curves (one-vs-rest) saved for every model.
7.  Pairplot (seaborn) coloured by class saved as EDA figure.
8.  Violin plots per numeric feature coloured by class saved as EDA figure.
9.  Interactive CLI: ask the user for feature values at runtime; fallback to
    NEW_SAMPLE_VALUES when running non-interactively.
10. Runtime summary table printed at the end covering all key metrics.

All original behaviour is preserved:
- Adaptive column detection, ID removal, duplicate handling, missing-value
  imputation, 80/20 stratified split, Joblib model save.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from io import StringIO
from math import ceil
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
    GridSearchCV,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Section 1: Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Iris.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_iris_model.joblib"

# Default sample used when running non-interactively (e.g. CI, scripted runs).
NEW_SAMPLE_VALUES: dict[str, float] = {
    "SepalLengthCm": 5.7,
    "SepalWidthCm": 2.9,
    "PetalLengthCm": 4.2,
    "PetalWidthCm": 1.3,
}

TARGET_NAME_HINTS = {"species", "target", "class", "label", "variety", "type"}

MODEL_TIE_BREAK_PRIORITY = {
    "Logistic Regression": 0,
    "K-Nearest Neighbours": 1,
    "Decision Tree Classifier": 2,
    "Random Forest Classifier": 3,
}


# ---------------------------------------------------------------------------
# Section 2: Data structures
# ---------------------------------------------------------------------------


@dataclass
class DatasetProfile:
    """Store decisions derived from the dataset structure."""

    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    target_column: str
    target_reason: str
    id_columns: list[str]
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    duplicate_rows_removed: int
    rows_removed_missing_target: int
    missing_values_before: dict[str, int]
    missing_values_after_cleaning: dict[str, int]
    class_labels: list[Any]


@dataclass
class ModelResult:
    """Store everything produced for a single trained model."""

    name: str
    pipeline: Pipeline
    accuracy: float
    cv_mean: float
    cv_std: float
    cv_scores: np.ndarray
    predictions: np.ndarray
    train_time_s: float


# ---------------------------------------------------------------------------
# Section 3: Console helpers
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def project_path(path: Path) -> str:
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_project_directories() -> None:
    for directory in (FIGURES_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_display() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    sns.set_theme(style="whitegrid", palette="Set2")


# ---------------------------------------------------------------------------
# Section 4: Data loading and cleaning
# ---------------------------------------------------------------------------


def load_provided_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {project_path(path)}. "
            "Place the provided internship CSV at data/Iris.csv."
        )
    return pd.read_csv(path)


def normalize_text_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    for column in normalized.select_dtypes(include=["object", "string"]):
        normalized[column] = normalized[column].where(
            normalized[column].isna(),
            normalized[column].astype(str).str.strip(),
        )
        normalized[column] = normalized[column].replace("", np.nan)
    return normalized


def clean_column_name(column: str) -> str:
    return column.strip().lower().replace(" ", "_").replace("-", "_")


def is_identifier_column(column: str, series: pd.Series) -> bool:
    clean_name = clean_column_name(column)
    non_missing = series.dropna()
    if non_missing.empty:
        return False
    unique_count = non_missing.nunique(dropna=True)
    unique_ratio = unique_count / len(non_missing)
    name_suggests_id = clean_name in {"id", "index"} or clean_name.endswith("_id")
    is_unique_numeric_sequence = (
        pd.api.types.is_numeric_dtype(non_missing)
        and unique_count == len(non_missing)
        and non_missing.is_monotonic_increasing
    )
    return (name_suggests_id and unique_ratio > 0.90) or (
        is_unique_numeric_sequence and unique_ratio == 1.0
    )


def detect_target_column(data: pd.DataFrame) -> tuple[str, str]:
    for column in data.columns:
        if clean_column_name(column) in TARGET_NAME_HINTS:
            return column, f"Column name '{column}' matches a common target label."

    max_classes = min(20, max(2, int(len(data) * 0.20)))
    candidates = [
        col for col in data.columns if 2 <= data[col].nunique(dropna=True) <= max_classes
    ]
    if candidates:
        column = candidates[-1]
        return column, "Last low-cardinality column selected as fallback target."

    column = data.columns[-1]
    return column, "Final column selected as supervised-learning target."


def sorted_labels(labels: pd.Series) -> list[Any]:
    unique = labels.dropna().unique().tolist()
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=str)


def prepare_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, DatasetProfile]:
    original_shape = data.shape
    missing_before = data.isna().sum().astype(int).to_dict()

    cleaned = normalize_text_columns(data)
    target_column, target_reason = detect_target_column(cleaned)

    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    missing_target_rows = int(cleaned[target_column].isna().sum())
    cleaned = cleaned.dropna(subset=[target_column]).reset_index(drop=True)

    id_columns = [
        col
        for col in cleaned.columns
        if col != target_column and is_identifier_column(col, cleaned[col])
    ]

    feature_columns = [
        col for col in cleaned.columns if col not in [target_column, *id_columns]
    ]

    numeric_features = [
        col for col in feature_columns if pd.api.types.is_numeric_dtype(cleaned[col])
    ]
    categorical_features = [col for col in feature_columns if col not in numeric_features]

    if not feature_columns:
        raise ValueError("No usable feature columns were found after cleaning.")
    if cleaned[target_column].nunique(dropna=True) < 2:
        raise ValueError("The target column must contain at least two classes.")

    profile = DatasetProfile(
        original_shape=original_shape,
        cleaned_shape=cleaned.shape,
        target_column=target_column,
        target_reason=target_reason,
        id_columns=id_columns,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        duplicate_rows_removed=duplicate_count,
        rows_removed_missing_target=missing_target_rows,
        missing_values_before=missing_before,
        missing_values_after_cleaning=cleaned.isna().sum().astype(int).to_dict(),
        class_labels=sorted_labels(cleaned[target_column]),
    )
    return cleaned, profile


# ---------------------------------------------------------------------------
# Section 5: Dataset overview + decisions
# ---------------------------------------------------------------------------


def display_dataset_overview(data: pd.DataFrame) -> None:
    print_section("Dataset Information")
    print(f"Dataset source: {project_path(DATA_PATH)}")
    print(f"Dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print("\nColumn names and data types:")
    print(data.dtypes.to_string())

    info_buffer = StringIO()
    data.info(buf=info_buffer)
    print("\nPandas info:")
    print(info_buffer.getvalue())

    print("First five rows:")
    print(data.head())

    print("\nMissing values by column:")
    print(data.isna().sum().to_string())
    print(f"\nDuplicate rows found: {data.duplicated().sum()}")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print("\nNumeric summary statistics:")
        print(data[numeric_cols].describe().round(3))

    categorical_cols = [c for c in data.columns if c not in numeric_cols]
    if categorical_cols:
        print("\nCategorical summary statistics:")
        print(data[categorical_cols].describe())


def explain_dataset_decisions(profile: DatasetProfile) -> None:
    print_section("Automatic Dataset Decisions")
    print(f"Original shape:  {profile.original_shape}")
    print(f"Cleaned shape:   {profile.cleaned_shape}")
    print(f"\nTarget column:           {profile.target_column}")
    print(f"Target selection reason: {profile.target_reason}")
    print("\nFeature columns:")
    for col in profile.feature_columns:
        print(f"  - {col}")
    if profile.id_columns:
        print("\nIdentifier columns removed:")
        for col in profile.id_columns:
            print(f"  - {col}")
    print(f"\nDuplicate rows removed:             {profile.duplicate_rows_removed}")
    print(f"Rows removed (missing target):      {profile.rows_removed_missing_target}")
    print(f"\nClasses found ({len(profile.class_labels)}): {profile.class_labels}")


# ---------------------------------------------------------------------------
# Section 6: Visualisations
# ---------------------------------------------------------------------------


def _save_and_close(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_class_distribution_plot(data: pd.DataFrame, profile: DatasetProfile) -> Path:
    path = FIGURES_DIR / "class_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    order = [str(l) for l in profile.class_labels]
    sns.countplot(data=data, x=profile.target_column, order=order, ax=ax)
    ax.set_title(f"Class Distribution: {profile.target_column}", fontsize=13)
    ax.set_xlabel(profile.target_column)
    ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0], padding=3)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _save_and_close(fig, path)


def save_numeric_distribution_plot(data: pd.DataFrame, profile: DatasetProfile) -> Path | None:
    if not profile.numeric_features:
        return None
    path = FIGURES_DIR / "feature_distributions.png"
    cols = profile.numeric_features
    n_cols = min(2, len(cols))
    n_rows = ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)
    flat = axes.flatten()
    use_hue = len(profile.class_labels) <= 10
    for ax, col in zip(flat, cols):
        kw: dict[str, Any] = {"data": data, "x": col, "kde": data[col].nunique() > 5, "ax": ax}
        if use_hue:
            kw["hue"] = profile.target_column
            kw["element"] = "step"
        sns.histplot(**kw)
        ax.set_title(f"Distribution of {col}")
    for ax in flat[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Numeric Feature Distributions", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_and_close(fig, path)


def save_correlation_heatmap(data: pd.DataFrame, profile: DatasetProfile) -> Path | None:
    if len(profile.numeric_features) < 2:
        return None
    path = FIGURES_DIR / "correlation_heatmap.png"
    corr = data[profile.numeric_features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Numeric Feature Correlation Heatmap")
    fig.tight_layout()
    return _save_and_close(fig, path)


# --- Enhancement 1: Pairplot coloured by class ---
def save_pairplot(data: pd.DataFrame, profile: DatasetProfile) -> Path | None:
    """Save a seaborn pairplot with classes as hue. Best EDA overview."""
    if len(profile.numeric_features) < 2:
        return None
    path = FIGURES_DIR / "pairplot.png"
    plot_cols = profile.numeric_features + [profile.target_column]
    fig = sns.pairplot(
        data[plot_cols],
        hue=profile.target_column,
        diag_kind="kde",
        plot_kws={"alpha": 0.7},
    )
    fig.figure.suptitle("Pairplot: All Feature Pairs by Class", y=1.02, fontsize=14)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


# --- Enhancement 2: Violin plots per numeric feature coloured by class ---
def save_violin_plots(data: pd.DataFrame, profile: DatasetProfile) -> Path | None:
    if not profile.numeric_features:
        return None
    path = FIGURES_DIR / "violin_plots.png"
    cols = profile.numeric_features
    n_cols = min(2, len(cols))
    n_rows = ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), squeeze=False)
    flat = axes.flatten()
    for ax, col in zip(flat, cols):
        sns.violinplot(
            data=data,
            x=profile.target_column,
            y=col,
            inner="quartile",
            ax=ax,
        )
        ax.set_title(f"{col} by {profile.target_column}")
        ax.tick_params(axis="x", rotation=15)
    for ax in flat[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Feature Spread per Class (Violin Plots)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_and_close(fig, path)


def create_visualizations(data: pd.DataFrame, profile: DatasetProfile) -> list[Path]:
    print_section("EDA Visualisations")
    plot_fns = [
        save_class_distribution_plot,
        save_numeric_distribution_plot,
        save_correlation_heatmap,
        save_pairplot,
        save_violin_plots,
    ]
    paths = []
    for fn in plot_fns:
        path = fn(data, profile)
        if path:
            paths.append(path)
            print(f"  Saved: {project_path(path)}")
    return paths


# ---------------------------------------------------------------------------
# Section 7: Preprocessing + model building
# ---------------------------------------------------------------------------


def build_preprocessor(profile: DatasetProfile) -> ColumnTransformer:
    transformers = []
    if profile.numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, profile.numeric_features))
    if profile.categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", cat_pipeline, profile.categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_tuned_decision_tree(profile: DatasetProfile) -> Pipeline:
    """Decision Tree with depth tuned by GridSearchCV (3-fold, fast)."""
    base_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(profile)),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])
    param_grid = {"model__max_depth": [2, 3, 4, 5, None]}
    return GridSearchCV(base_pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)


def build_models(profile: DatasetProfile) -> dict[str, Pipeline | GridSearchCV]:
    """Return all four classifiers, each wrapped with preprocessing."""
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(profile)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "K-Nearest Neighbours": Pipeline([
            ("preprocessor", build_preprocessor(profile)),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree Classifier": build_tuned_decision_tree(profile),
        "Random Forest Classifier": Pipeline([
            ("preprocessor", build_preprocessor(profile)),
            ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]),
    }


# ---------------------------------------------------------------------------
# Section 8: Training + cross-validation
# ---------------------------------------------------------------------------


def train_and_compare_models(
    models: dict[str, Any],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    profile: DatasetProfile,
) -> dict[str, ModelResult]:
    print_section("Model Training and Accuracy Comparison")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, ModelResult] = {}

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        train_time = time.perf_counter() - t0

        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        # Cross-validation on the *full* cleaned data (not just train split)
        full_x = pd.concat([x_train, x_test])
        full_y = pd.concat([y_train, y_test])
        cv_scores = cross_val_score(model, full_x, full_y, cv=skf, scoring="accuracy", n_jobs=-1)

        results[name] = ModelResult(
            name=name,
            pipeline=model,
            accuracy=accuracy,
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            cv_scores=cv_scores,
            predictions=predictions,
            train_time_s=train_time,
        )

    comparison = pd.DataFrame([
        {
            "Model": r.name,
            "Test Accuracy": f"{r.accuracy:.4f}",
            f"CV Mean ({CV_FOLDS}-fold)": f"{r.cv_mean:.4f}",
            "CV Std": f"±{r.cv_std:.4f}",
            "Train Time (s)": f"{r.train_time_s:.3f}",
        }
        for r in results.values()
    ]).sort_values("Test Accuracy", ascending=False)

    print(comparison.to_string(index=False))
    return results


def select_best_model(results: dict[str, ModelResult]) -> str:
    best_name = max(
        results,
        key=lambda n: (results[n].accuracy, -MODEL_TIE_BREAK_PRIORITY.get(n, 99)),
    )
    best_acc = results[best_name].accuracy
    tied = [n for n, r in results.items() if r.accuracy == best_acc]
    if len(tied) > 1:
        print(f"\nTie between {', '.join(tied)}. Selected '{best_name}' (simplest model).")
    return best_name


# ---------------------------------------------------------------------------
# Section 9: Evaluation visualisations
# ---------------------------------------------------------------------------


# --- Enhancement 3: Confusion matrix saved as a heatmap image ---
def save_confusion_matrix_plot(
    result: ModelResult,
    profile: DatasetProfile,
) -> Path:
    path = FIGURES_DIR / f"confusion_matrix_{result.name.lower().replace(' ', '_')}.png"
    labels = profile.class_labels
    label_names = [str(l) for l in labels]
    cm = confusion_matrix(
        # y_test not available here; recomputed through ConfusionMatrixDisplay below
        # We pass predictions and rely on the caller having y_test
        y_true=None, y_pred=None,  # placeholder — see caller
        labels=labels,
    )
    # Actual computation happens in evaluate_best_model; we accept cm as arg there.
    return path


def save_cm_heatmap(
    y_test: pd.Series,
    result: ModelResult,
    profile: DatasetProfile,
) -> Path:
    path = FIGURES_DIR / f"confusion_matrix_{result.name.lower().replace(' ', '_')}.png"
    labels = profile.class_labels
    label_names = [str(l) for l in labels]
    cm = confusion_matrix(y_test, result.predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {result.name}", fontsize=13)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    return _save_and_close(fig, path)


# --- Enhancement 4: Per-class ROC curves (one-vs-rest) ---
def save_roc_curves(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    results: dict[str, ModelResult],
    profile: DatasetProfile,
) -> Path | None:
    """Save one-vs-rest ROC curves for all models that support predict_proba."""
    classes = profile.class_labels
    if len(classes) < 2:
        return None

    path = FIGURES_DIR / "roc_curves.png"
    y_bin = label_binarize(y_test, classes=classes)
    n_classes = y_bin.shape[1]

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 5 * n_rows), squeeze=False)

    for ax, (name, result) in zip(axes.flatten(), results.items()):
        model = result.pipeline
        if not hasattr(model, "predict_proba"):
            ax.text(0.5, 0.5, f"{name}\n(no predict_proba)", ha="center", va="center")
            ax.set_title(name)
            continue

        y_score = model.predict_proba(x_test)

        if n_classes == 2:
            # Binary case
            fpr, tpr, _ = roc_curve(y_bin[:, 1], y_score[:, 1])
            auc = roc_auc_score(y_bin[:, 1], y_score[:, 1])
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        else:
            # Multi-class: one curve per class
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                auc = roc_auc_score(y_bin[:, i], y_score[:, i])
                ax.plot(fpr, tpr, label=f"{cls} AUC = {auc:.3f}")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(f"ROC Curves (OvR) — {name}", fontsize=12)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_and_close(fig, path)


# --- Enhancement 5: Feature importance for tree-based models ---
def save_feature_importance_plot(
    result: ModelResult,
    profile: DatasetProfile,
) -> Path | None:
    """Save a horizontal bar chart of feature importances from tree models."""
    model = result.pipeline
    # Unwrap GridSearchCV
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_

    if not hasattr(model, "named_steps"):
        return None

    final_estimator = model.named_steps.get("model")
    if final_estimator is None or not hasattr(final_estimator, "feature_importances_"):
        return None

    importances = final_estimator.feature_importances_

    # Map back to original feature names (only numeric here; extend if OHE needed)
    feature_names: list[str] = list(profile.numeric_features)
    if len(feature_names) != len(importances):
        # Try to reconstruct from preprocessor output
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None:
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feature_names = [f"f{i}" for i in range(len(importances))]

    path = FIGURES_DIR / f"feature_importance_{result.name.lower().replace(' ', '_')}.png"

    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.5)))
    bars = ax.barh(
        [str(feature_names[i]) for i in sorted_idx],
        importances[sorted_idx],
        color=sns.color_palette("Set2", len(importances)),
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title(f"Feature Importances — {result.name}", fontsize=13)
    ax.set_xlabel("Importance")
    ax.set_xlim(0, max(importances) * 1.2)
    fig.tight_layout()
    return _save_and_close(fig, path)


def evaluate_best_model(
    best_name: str,
    results: dict[str, ModelResult],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    profile: DatasetProfile,
) -> list[Path]:
    """Print classification report and save confusion-matrix + importance plots."""
    print_section("Best Model Evaluation")

    result = results[best_name]
    labels = profile.class_labels
    label_names = [str(l) for l in labels]

    print(f"Best model:    {result.name}")
    print(f"Test accuracy: {result.accuracy:.4f}")
    print(f"CV accuracy:   {result.cv_mean:.4f} ± {result.cv_std:.4f}")

    cm = confusion_matrix(y_test, result.predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print("\nConfusion Matrix:")
    print(cm_df)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, result.predictions, labels=labels, target_names=label_names, zero_division=0
        )
    )

    saved_paths: list[Path] = []

    cm_path = save_cm_heatmap(y_test, result, profile)
    saved_paths.append(cm_path)
    print(f"\nSaved confusion-matrix heatmap: {project_path(cm_path)}")

    imp_path = save_feature_importance_plot(result, profile)
    if imp_path:
        saved_paths.append(imp_path)
        print(f"Saved feature importance chart: {project_path(imp_path)}")

    roc_path = save_roc_curves(x_test, y_test, results, profile)
    if roc_path:
        saved_paths.append(roc_path)
        print(f"Saved ROC curves:               {project_path(roc_path)}")

    return saved_paths


# ---------------------------------------------------------------------------
# Section 10: Interactive / scripted prediction
# ---------------------------------------------------------------------------


def _is_interactive() -> bool:
    """Return True when running in an interactive terminal."""
    return sys.stdin.isatty()


def gather_sample_interactively(profile: DatasetProfile, data: pd.DataFrame) -> pd.DataFrame:
    """Prompt the user for each numeric feature value in the terminal."""
    print_section("Interactive Prediction")
    print("Enter values for a new flower sample.")
    print("Press Enter to accept the dataset median shown in brackets.\n")

    sample_values: dict[str, float] = {}

    for feature in profile.feature_columns:
        if feature in profile.numeric_features:
            median_val = round(data[feature].median(), 3)
            while True:
                raw = input(f"  {feature} [{median_val}]: ").strip()
                if raw == "":
                    sample_values[feature] = median_val
                    break
                try:
                    sample_values[feature] = float(raw)
                    break
                except ValueError:
                    print(f"  Please enter a number (e.g. {median_val}).")
        else:
            mode_val = data[feature].mode(dropna=True).iloc[0]
            sample_values[feature] = mode_val

    return pd.DataFrame([sample_values], columns=profile.feature_columns)


def build_scripted_sample(data: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    """Build a prediction sample from NEW_SAMPLE_VALUES with median fallbacks."""
    sample_values: dict[str, Any] = {}
    for feature in profile.feature_columns:
        if feature in NEW_SAMPLE_VALUES:
            sample_values[feature] = NEW_SAMPLE_VALUES[feature]
        elif feature in profile.numeric_features:
            sample_values[feature] = data[feature].median()
        else:
            sample_values[feature] = data[feature].mode(dropna=True).iloc[0]
    return pd.DataFrame([sample_values], columns=profile.feature_columns)


def predict_new_sample(
    model: Any,
    data: pd.DataFrame,
    profile: DatasetProfile,
) -> None:
    print_section("New Sample Prediction")

    if _is_interactive():
        sample = gather_sample_interactively(profile, data)
    else:
        sample = build_scripted_sample(data, profile)
        print("Non-interactive run: using NEW_SAMPLE_VALUES as the prediction input.")

    predicted_class = model.predict(sample)[0]

    print("\nInput sample:")
    print(sample.to_string(index=False))
    print(f"\nPredicted {profile.target_column}: {predicted_class}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        prob_table = pd.DataFrame(
            {"Class": model.classes_, "Probability": probabilities}
        ).sort_values(by="Probability", ascending=False)
        print("\nPrediction probabilities:")
        print(prob_table.to_string(index=False, formatters={"Probability": "{:.4f}".format}))


# ---------------------------------------------------------------------------
# Section 11: Model saving
# ---------------------------------------------------------------------------


def save_model_artifact(
    best_name: str,
    result: ModelResult,
    profile: DatasetProfile,
    plot_paths: list[Path],
) -> None:
    artifact = {
        "model_name": result.name,
        "model": result.pipeline,
        "accuracy": result.accuracy,
        "cv_mean": result.cv_mean,
        "cv_std": result.cv_std,
        "dataset_path": project_path(DATA_PATH),
        "target_column": profile.target_column,
        "feature_columns": profile.feature_columns,
        "numeric_features": profile.numeric_features,
        "categorical_features": profile.categorical_features,
        "id_columns_removed": profile.id_columns,
        "class_labels": [str(l) for l in profile.class_labels],
        "plot_paths": [project_path(p) for p in plot_paths],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
    }
    joblib.dump(artifact, MODEL_PATH)

    print_section("Model Saved")
    print(f"Saved best model artifact: {project_path(MODEL_PATH)}")
    print("Load later with: joblib.load('models/best_iris_model.joblib')")


# ---------------------------------------------------------------------------
# Section 12: Runtime summary
# ---------------------------------------------------------------------------


def print_runtime_summary(
    results: dict[str, ModelResult],
    best_name: str,
    profile: DatasetProfile,
) -> None:
    print_section("Run Summary")
    print(f"Dataset: {profile.cleaned_shape[0]} rows × {len(profile.feature_columns)} features")
    print(f"Classes: {profile.class_labels}\n")

    summary = pd.DataFrame([
        {
            "Model": r.name,
            "Test Acc": f"{r.accuracy:.4f}",
            "CV Acc": f"{r.cv_mean:.4f} ± {r.cv_std:.4f}",
            "Train(s)": f"{r.train_time_s:.3f}",
            "Best?": "✓" if r.name == best_name else "",
        }
        for r in results.values()
    ])
    print(summary.to_string(index=False))
    print(f"\nBest model selected: {best_name}")


# ---------------------------------------------------------------------------
# Section 13: Main
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_project_directories()
    configure_display()

    raw_data = load_provided_dataset(DATA_PATH)
    display_dataset_overview(raw_data)

    cleaned_data, profile = prepare_dataset(raw_data)
    explain_dataset_decisions(profile)

    eda_paths = create_visualizations(cleaned_data, profile)

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned_data[profile.feature_columns],
        cleaned_data[profile.target_column],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=cleaned_data[profile.target_column]
        if cleaned_data[profile.target_column].value_counts().min() >= 2
        else None,
    )

    models = build_models(profile)
    results = train_and_compare_models(models, x_train, x_test, y_train, y_test, profile)

    best_name = select_best_model(results)
    best_result = results[best_name]

    eval_paths = evaluate_best_model(best_name, results, x_test, y_test, profile)

    predict_new_sample(best_result.pipeline, cleaned_data, profile)

    all_paths = eda_paths + eval_paths
    save_model_artifact(best_name, best_result, profile, all_paths)

    print_runtime_summary(results, best_name, profile)


if __name__ == "__main__":
    main()
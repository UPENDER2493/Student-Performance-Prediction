# ============================================================
# WEEK 6 - STUDENT PERFORMANCE
# STEP 8: MODEL EVALUATION & ERROR ANALYSIS
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "student-mat-cleaned.csv"

REPORT_DIR = PROJECT_DIR / "reports"
VIS_DIR = PROJECT_DIR / "visualizations"

TARGET = "G3"
TEST_SIZE = 0.20
RANDOM_STATE = 42

REPORT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("WEEK 6 - STUDENT PERFORMANCE")
print("STEP 8: MODEL EVALUATION & ERROR ANALYSIS")
print("=" * 80)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("1. DATASET LOADING")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print(f"Dataset path : {DATA_PATH}")
print(f"Rows         : {df.shape[0]}")
print(f"Columns      : {df.shape[1]}")

if TARGET not in df.columns:
    raise ValueError("Target G3 is missing from the dataset.")

print("✓ Dataset loaded successfully.")


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 80)
print("2. FEATURE / TARGET PREPARATION")
print("=" * 80)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_full = X.copy()

X_pregrade = X.drop(
    columns=["G1", "G2"],
    errors="ignore"
).copy()

print(f"Full Information features : {X_full.shape[1]}")
print(f"Pre-Grade features        : {X_pregrade.shape[1]}")
print(f"Target observations       : {len(y)}")


# ============================================================
# 4. SAME TRAIN / TEST SPLITS AS STEP 7
# ============================================================

print("\n" + "=" * 80)
print("3. TRAIN / TEST SPLIT")
print("=" * 80)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
    X_pregrade,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("Full Information:")
print(f"  Train: {len(X_train_full)}")
print(f"  Test : {len(X_test_full)}")

print("\nPre-Grade Prediction:")
print(f"  Train: {len(X_train_pre)}")
print(f"  Test : {len(X_test_pre)}")

print("✓ Same random state and test size as Step 7.")


# ============================================================
# 5. PREPROCESSOR
# ============================================================

def build_preprocessor(X_data):

    numeric_features = X_data.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X_data.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# 6. BUILD RANDOM FOREST PIPELINES
# ============================================================

print("\n" + "=" * 80)
print("4. BUILDING BEST BASELINE MODELS")
print("=" * 80)

rf_full = Pipeline(
    steps=[
        (
            "preprocessing",
            build_preprocessor(X_train_full)
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

rf_pregrade = Pipeline(
    steps=[
        (
            "preprocessing",
            build_preprocessor(X_train_pre)
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

print("✓ Full Information Random Forest created.")
print("✓ Pre-Grade Random Forest created.")


# ============================================================
# 7. TRAIN MODELS
# ============================================================

print("\n" + "=" * 80)
print("5. MODEL TRAINING")
print("=" * 80)

print("Training Full Information Random Forest...")
rf_full.fit(X_train_full, y_train_full)
print("✓ Full Information model trained.")

print("\nTraining Pre-Grade Random Forest...")
rf_pregrade.fit(X_train_pre, y_train_pre)
print("✓ Pre-Grade model trained.")


# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("6. GENERATING TEST PREDICTIONS")
print("=" * 80)

pred_full = rf_full.predict(X_test_full)
pred_pregrade = rf_pregrade.predict(X_test_pre)

print(f"Full Information predictions : {len(pred_full)}")
print(f"Pre-Grade predictions        : {len(pred_pregrade)}")

print("✓ Predictions generated.")


# ============================================================
# 9. PERFORMANCE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("7. MODEL PERFORMANCE SUMMARY")
print("=" * 80)


def calculate_metrics(y_true, y_pred):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(
            y_true,
            y_pred
        )
    }


metrics_full = calculate_metrics(
    y_test_full,
    pred_full
)

metrics_pregrade = calculate_metrics(
    y_test_pre,
    pred_pregrade
)

evaluation_summary = pd.DataFrame(
    [
        {
            "Scenario": "Full Information",
            **metrics_full
        },
        {
            "Scenario": "Pre-Grade Prediction",
            **metrics_pregrade
        }
    ]
)

print(
    evaluation_summary
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 10. ERROR ANALYSIS DATAFRAME
# ============================================================

print("\n" + "=" * 80)
print("8. ERROR ANALYSIS")
print("=" * 80)


def create_error_dataframe(
    X_test,
    y_true,
    predictions,
    scenario_name
):

    error_df = X_test.copy()

    error_df["Actual_G3"] = y_true.to_numpy()

    error_df["Predicted_G3"] = predictions

    error_df["Error"] = (
        error_df["Actual_G3"]
        - error_df["Predicted_G3"]
    )

    error_df["Absolute_Error"] = (
        error_df["Error"]
        .abs()
    )

    error_df["Squared_Error"] = (
        error_df["Error"] ** 2
    )

    error_df["Scenario"] = scenario_name

    return error_df


errors_full = create_error_dataframe(
    X_test_full,
    y_test_full,
    pred_full,
    "Full Information"
)

errors_pregrade = create_error_dataframe(
    X_test_pre,
    y_test_pre,
    pred_pregrade,
    "Pre-Grade Prediction"
)


# ============================================================
# 11. ERROR STATISTICS
# ============================================================

print("\nFull Information Error Statistics:")

print(
    f"  Mean Error          : "
    f"{errors_full['Error'].mean():.4f}"
)

print(
    f"  Mean Absolute Error : "
    f"{errors_full['Absolute_Error'].mean():.4f}"
)

print(
    f"  Max Absolute Error  : "
    f"{errors_full['Absolute_Error'].max():.4f}"
)

print(
    f"  Error Std Dev       : "
    f"{errors_full['Error'].std():.4f}"
)


print("\nPre-Grade Prediction Error Statistics:")

print(
    f"  Mean Error          : "
    f"{errors_pregrade['Error'].mean():.4f}"
)

print(
    f"  Mean Absolute Error : "
    f"{errors_pregrade['Absolute_Error'].mean():.4f}"
)

print(
    f"  Max Absolute Error  : "
    f"{errors_pregrade['Absolute_Error'].max():.4f}"
)

print(
    f"  Error Std Dev       : "
    f"{errors_pregrade['Error'].std():.4f}"
)


# ============================================================
# 12. LARGEST PREDICTION ERRORS
# ============================================================

print("\n" + "=" * 80)
print("9. LARGEST PREDICTION ERRORS")
print("=" * 80)

largest_errors_full = (
    errors_full
    .sort_values(
        by="Absolute_Error",
        ascending=False
    )
    .head(10)
)

largest_errors_pregrade = (
    errors_pregrade
    .sort_values(
        by="Absolute_Error",
        ascending=False
    )
    .head(10)
)

print("\nFull Information - Top 10 Errors:")

print(
    largest_errors_full[
        [
            "Actual_G3",
            "Predicted_G3",
            "Error",
            "Absolute_Error"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print("\nPre-Grade Prediction - Top 10 Errors:")

print(
    largest_errors_pregrade[
        [
            "Actual_G3",
            "Predicted_G3",
            "Error",
            "Absolute_Error"
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 13. ERROR BY ACTUAL G3 RANGE
# ============================================================

print("\n" + "=" * 80)
print("10. ERROR BY ACTUAL G3 RANGE")
print("=" * 80)

bins = [-1, 5, 10, 15, 20]
labels = [
    "0-5",
    "6-10",
    "11-15",
    "16-20"
]


def error_by_grade_range(error_df):

    temp = error_df.copy()

    temp["Actual_G3_Range"] = pd.cut(
        temp["Actual_G3"],
        bins=bins,
        labels=labels
    )

    grouped = (
        temp
        .groupby(
            "Actual_G3_Range",
            observed=False
        )
        .agg(
            Count=("Actual_G3", "size"),
            Mean_Absolute_Error=(
                "Absolute_Error",
                "mean"
            ),
            Mean_Error=("Error", "mean"),
            RMSE=(
                "Squared_Error",
                lambda x: np.sqrt(x.mean())
            )
        )
        .reset_index()
    )

    return grouped


range_full = error_by_grade_range(
    errors_full
)

range_pregrade = error_by_grade_range(
    errors_pregrade
)

print("\nFull Information:")

print(
    range_full
    .round(4)
    .to_string(index=False)
)

print("\nPre-Grade Prediction:")

print(
    range_pregrade
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 14. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("11. RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)


def get_feature_importance(
    pipeline,
    scenario_name
):

    preprocessor = pipeline.named_steps[
        "preprocessing"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df["Scenario"] = scenario_name

    return importance_df


importance_full = get_feature_importance(
    rf_full,
    "Full Information"
)

importance_pregrade = get_feature_importance(
    rf_pregrade,
    "Pre-Grade Prediction"
)


print("\nTop 15 Full Information Features:")

print(
    importance_full
    .head(15)[
        ["Feature", "Importance"]
    ]
    .round(6)
    .to_string(index=False)
)

print("\nTop 15 Pre-Grade Features:")

print(
    importance_pregrade
    .head(15)[
        ["Feature", "Importance"]
    ]
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 15. SAVE CSV REPORTS
# ============================================================

print("\n" + "=" * 80)
print("12. SAVING REPORTS")
print("=" * 80)

evaluation_summary_path = (
    REPORT_DIR /
    "step8_evaluation_summary.csv"
)

evaluation_summary.to_csv(
    evaluation_summary_path,
    index=False
)

errors_full_path = (
    REPORT_DIR /
    "step8_full_information_errors.csv"
)

errors_full.to_csv(
    errors_full_path,
    index=False
)

errors_pregrade_path = (
    REPORT_DIR /
    "step8_pregrade_errors.csv"
)

errors_pregrade.to_csv(
    errors_pregrade_path,
    index=False
)

largest_full_path = (
    REPORT_DIR /
    "step8_largest_full_information_errors.csv"
)

largest_errors_full.to_csv(
    largest_full_path,
    index=False
)

largest_pregrade_path = (
    REPORT_DIR /
    "step8_largest_pregrade_errors.csv"
)

largest_errors_pregrade.to_csv(
    largest_pregrade_path,
    index=False
)

range_full_path = (
    REPORT_DIR /
    "step8_error_by_grade_range_full.csv"
)

range_full.to_csv(
    range_full_path,
    index=False
)

range_pregrade_path = (
    REPORT_DIR /
    "step8_error_by_grade_range_pregrade.csv"
)

range_pregrade.to_csv(
    range_pregrade_path,
    index=False
)

importance_full_path = (
    REPORT_DIR /
    "step8_feature_importance_full.csv"
)

importance_full.to_csv(
    importance_full_path,
    index=False
)

importance_pregrade_path = (
    REPORT_DIR /
    "step8_feature_importance_pregrade.csv"
)

importance_pregrade.to_csv(
    importance_pregrade_path,
    index=False
)

print(f"✓ {evaluation_summary_path.name}")
print(f"✓ {errors_full_path.name}")
print(f"✓ {errors_pregrade_path.name}")
print(f"✓ {largest_full_path.name}")
print(f"✓ {largest_pregrade_path.name}")
print(f"✓ {range_full_path.name}")
print(f"✓ {range_pregrade_path.name}")
print(f"✓ {importance_full_path.name}")
print(f"✓ {importance_pregrade_path.name}")


# ============================================================
# 16. VISUALIZATION 1 - ACTUAL VS PREDICTED
# ============================================================

print("\n" + "=" * 80)
print("13. CREATING VISUALIZATIONS")
print("=" * 80)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_full,
    pred_full,
    alpha=0.7
)

min_value = min(
    y_test_full.min(),
    pred_full.min()
)

max_value = max(
    y_test_full.max(),
    pred_full.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title(
    "Random Forest - Actual vs Predicted G3\n"
    "Full Information"
)

plt.tight_layout()

plt.savefig(
    VIS_DIR / "16_rf_actual_vs_predicted_full.png",
    dpi=300
)

plt.close()


# ============================================================
# 17. VISUALIZATION 2 - ACTUAL VS PREDICTED PRE-GRADE
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_pre,
    pred_pregrade,
    alpha=0.7
)

min_value = min(
    y_test_pre.min(),
    pred_pregrade.min()
)

max_value = max(
    y_test_pre.max(),
    pred_pregrade.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title(
    "Random Forest - Actual vs Predicted G3\n"
    "Pre-Grade Prediction"
)

plt.tight_layout()

plt.savefig(
    VIS_DIR / "17_rf_actual_vs_predicted_pregrade.png",
    dpi=300
)

plt.close()


# ============================================================
# 18. VISUALIZATION 3 - RESIDUALS VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    pred_full,
    errors_full["Error"],
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted G3")
plt.ylabel("Residual (Actual - Predicted)")
plt.title(
    "Random Forest Residuals vs Predicted G3\n"
    "Full Information"
)

plt.tight_layout()

plt.savefig(
    VIS_DIR / "18_rf_residuals_full.png",
    dpi=300
)

plt.close()


# ============================================================
# 19. VISUALIZATION 4 - RESIDUAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

sns.histplot(
    errors_full["Error"],
    kde=True
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title(
    "Random Forest Residual Distribution\n"
    "Full Information"
)

plt.tight_layout()

plt.savefig(
    VIS_DIR / "19_rf_residual_distribution_full.png",
    dpi=300
)

plt.close()


# ============================================================
# 20. VISUALIZATION 5 - ABSOLUTE ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

sns.histplot(
    errors_full["Absolute_Error"],
    kde=True
)

plt.xlabel("Absolute Prediction Error")
plt.ylabel("Frequency")
plt.title(
    "Random Forest Absolute Error Distribution\n"
    "Full Information"
)

plt.tight_layout()

plt.savefig(
    VIS_DIR / "20_rf_absolute_error_distribution_full.png",
    dpi=300
)

plt.close()


# ============================================================
# 21. VISUALIZATION 6 - ERROR BY GRADE RANGE
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    range(len(range_full)),
    range_full["Mean_Absolute_Error"]
)

plt.xticks(
    range(len(range_full)),
    range_full["Actual_G3_Range"]
)

plt.xlabel("Actual G3 Range")
plt.ylabel("Mean Absolute Error")
plt.title(
    "Random Forest Error by Actual G3 Range\n"
    "Full Information"
)

plt.tight_layout()

plt.savefig(
    VIS_DIR / "21_rf_error_by_grade_range_full.png",
    dpi=300
)

plt.close()


# ============================================================
# 22. VISUALIZATION 7 - FEATURE IMPORTANCE
# ============================================================

top_features = (
    importance_full
    .head(15)
    .sort_values(
        by="Importance",
        ascending=True
    )
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(
    "Top 15 Random Forest Features\n"
    "Full Information"
)

plt.tight_layout()

plt.savefig(
    VIS_DIR / "22_rf_feature_importance_full.png",
    dpi=300
)

plt.close()


# ============================================================
# 23. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("14. FINAL VERIFICATION")
print("=" * 80)

required_reports = [
    "step8_evaluation_summary.csv",
    "step8_full_information_errors.csv",
    "step8_pregrade_errors.csv",
    "step8_largest_full_information_errors.csv",
    "step8_largest_pregrade_errors.csv",
    "step8_error_by_grade_range_full.csv",
    "step8_error_by_grade_range_pregrade.csv",
    "step8_feature_importance_full.csv",
    "step8_feature_importance_pregrade.csv"
]

required_visualizations = [
    "16_rf_actual_vs_predicted_full.png",
    "17_rf_actual_vs_predicted_pregrade.png",
    "18_rf_residuals_full.png",
    "19_rf_residual_distribution_full.png",
    "20_rf_absolute_error_distribution_full.png",
    "21_rf_error_by_grade_range_full.png",
    "22_rf_feature_importance_full.png"
]

for filename in required_reports:

    path = REPORT_DIR / filename

    if path.exists():
        print(f"✓ Report: {filename}")
    else:
        print(f"✗ Missing report: {filename}")


for filename in required_visualizations:

    path = VIS_DIR / filename

    if path.exists():
        print(f"✓ Visualization: {filename}")
    else:
        print(f"✗ Missing visualization: {filename}")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("STEP 8 MODEL EVALUATION & ERROR ANALYSIS COMPLETED")
print("=" * 80)

print("\nNext step: analyze the actual results before model improvement.")
print("=" * 80)

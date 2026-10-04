import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from joblib import load
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 80)
print("WEEK 6 - STUDENT PERFORMANCE")
print("STEP 11A: FINAL MODEL ANALYSIS & EXPLAINABILITY")
print("=" * 80)

DATA_PATH = Path("data/student-mat-cleaned.csv")
MODEL_PATH = Path(
    "models/student_performance_final_model.joblib"
)

REPORT_DIR = Path("reports")
VIS_DIR = Path("visualizations")

REPORT_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)

TARGET = "G3"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print(f"Target  : {TARGET}")

print("✓ Dataset loaded.")


# ============================================================
# 3. RECREATE SAME TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("2. RECREATING TEST SET")
print("=" * 80)

from sklearn.model_selection import train_test_split

(
    X_train,
    X_test,
    y_train,
    y_test
) = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

print("✓ Same split as final model.")


# ============================================================
# 4. LOAD FINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("3. LOADING FINAL MODEL")
print("=" * 80)

model = load(MODEL_PATH)

print(f"✓ Loaded: {MODEL_PATH}")


# ============================================================
# 5. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("4. GENERATING FINAL PREDICTIONS")
print("=" * 80)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

print("✓ Predictions generated.")


# ============================================================
# 6. PERFORMANCE VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("5. PERFORMANCE VERIFICATION")
print("=" * 80)

test_mse = mean_squared_error(
    y_test,
    test_predictions
)

test_metrics = {
    "MAE": mean_absolute_error(
        y_test,
        test_predictions
    ),
    "MSE": test_mse,
    "RMSE": np.sqrt(test_mse),
    "R2": r2_score(
        y_test,
        test_predictions
    )
}

for metric, value in test_metrics.items():
    print(f"{metric:<6}: {value:.4f}")


# ============================================================
# 7. CREATE ERROR DATAFRAME
# ============================================================

print("\n" + "=" * 80)
print("6. ERROR ANALYSIS")
print("=" * 80)

error_df = X_test.copy()

error_df["Actual_G3"] = y_test.values
error_df["Predicted_G3"] = test_predictions

error_df["Error"] = (
    error_df["Actual_G3"]
    -
    error_df["Predicted_G3"]
)

error_df["Absolute_Error"] = (
    error_df["Error"]
    .abs()
)

error_df["Squared_Error"] = (
    error_df["Error"] ** 2
)

error_df = error_df.reset_index(
    names="Original_Index"
)

print(
    f"Mean error       : "
    f"{error_df['Error'].mean():.4f}"
)

print(
    f"Mean absolute error : "
    f"{error_df['Absolute_Error'].mean():.4f}"
)

print(
    f"Maximum absolute error : "
    f"{error_df['Absolute_Error'].max():.4f}"
)


# ============================================================
# 8. LARGEST PREDICTION ERRORS
# ============================================================

print("\n" + "=" * 80)
print("7. LARGEST PREDICTION ERRORS")
print("=" * 80)

largest_errors = (
    error_df
    .sort_values(
        by="Absolute_Error",
        ascending=False
    )
    .head(15)
)

print(
    largest_errors[
        [
            "Original_Index",
            "Actual_G3",
            "Predicted_G3",
            "Error",
            "Absolute_Error"
        ]
    ]
    .round(3)
    .to_string(index=False)
)

largest_errors_path = (
    REPORT_DIR /
    "step11_largest_prediction_errors.csv"
)

largest_errors.to_csv(
    largest_errors_path,
    index=False
)

print(
    f"\n✓ {largest_errors_path.name}"
)


# ============================================================
# 9. ERROR BY GRADE RANGE
# ============================================================

print("\n" + "=" * 80)
print("8. ERROR BY ACTUAL GRADE RANGE")
print("=" * 80)

def grade_range(g3):

    if g3 <= 5:
        return "0-5"

    elif g3 <= 10:
        return "6-10"

    elif g3 <= 15:
        return "11-15"

    else:
        return "16-20"


error_df["Grade_Range"] = (
    error_df["Actual_G3"]
    .apply(grade_range)
)

range_summary = (
    error_df
    .groupby("Grade_Range", observed=True)
    .agg(
        Count=("Actual_G3", "size"),
        MAE=("Absolute_Error", "mean"),
        Mean_Error=("Error", "mean"),
        RMSE=(
            "Squared_Error",
            lambda x: np.sqrt(x.mean())
        )
    )
    .reset_index()
)

range_order = [
    "0-5",
    "6-10",
    "11-15",
    "16-20"
]

range_summary["Grade_Range"] = pd.Categorical(
    range_summary["Grade_Range"],
    categories=range_order,
    ordered=True
)

range_summary = (
    range_summary
    .sort_values("Grade_Range")
)

print(
    range_summary
    .round(4)
    .to_string(index=False)
)

range_path = (
    REPORT_DIR /
    "step11_error_by_grade_range.csv"
)

range_summary.to_csv(
    range_path,
    index=False
)

print(
    f"\n✓ {range_path.name}"
)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("9. FINAL MODEL FEATURE IMPORTANCE")
print("=" * 80)

preprocessor = model.named_steps[
    "preprocessing"
]

random_forest = model.named_steps[
    "model"
]

feature_names = (
    preprocessor
    .get_feature_names_out()
)

importance_values = (
    random_forest
    .feature_importances_
)

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance_values
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

importance_df["Importance_Percent"] = (
    importance_df["Importance"] * 100
)

print("\nTop 20 features:")

print(
    importance_df
    .head(20)
    .round(4)
    .to_string(index=False)
)

importance_path = (
    REPORT_DIR /
    "step11_final_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print(
    f"\n✓ {importance_path.name}"
)


# ============================================================
# 11. TOP FEATURE PLOT
# ============================================================

print("\n" + "=" * 80)
print("10. FEATURE IMPORTANCE VISUALIZATION")
print("=" * 80)

top_features = (
    importance_df
    .head(15)
    .sort_values(
        by="Importance"
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(
    "Random Forest Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Features Influencing Student Performance Prediction"
)

plt.tight_layout()

feature_plot_path = (
    VIS_DIR /
    "23_final_feature_importance.png"
)

plt.savefig(
    feature_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"✓ {feature_plot_path.name}"
)


# ============================================================
# 12. ACTUAL VS PREDICTED
# ============================================================

print("\n" + "=" * 80)
print("11. ACTUAL VS PREDICTED VISUALIZATION")
print("=" * 80)

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.7
)

minimum = min(
    y_test.min(),
    test_predictions.min()
)

maximum = max(
    y_test.max(),
    test_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual G3"
)

plt.ylabel(
    "Predicted G3"
)

plt.title(
    "Final Random Forest: Actual vs Predicted G3"
)

plt.tight_layout()

actual_predicted_path = (
    VIS_DIR /
    "24_final_actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"✓ {actual_predicted_path.name}"
)


# ============================================================
# 13. RESIDUAL PLOT
# ============================================================

print("\n" + "=" * 80)
print("12. RESIDUAL ANALYSIS VISUALIZATION")
print("=" * 80)

residuals = (
    y_test.values
    -
    test_predictions
)

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    test_predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted G3"
)

plt.ylabel(
    "Residual (Actual - Predicted)"
)

plt.title(
    "Final Random Forest Residual Analysis"
)

plt.tight_layout()

residual_path = (
    VIS_DIR /
    "25_final_residual_analysis.png"
)

plt.savefig(
    residual_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"✓ {residual_path.name}"
)


# ============================================================
# 14. ERROR DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("13. ERROR DISTRIBUTION VISUALIZATION")
print("=" * 80)

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    residuals,
    bins=15,
    edgecolor="black"
)

plt.axvline(
    x=0,
    linestyle="--"
)

plt.xlabel(
    "Prediction Error (Actual - Predicted)"
)

plt.ylabel(
    "Number of Students"
)

plt.title(
    "Distribution of Final Model Prediction Errors"
)

plt.tight_layout()

error_distribution_path = (
    VIS_DIR /
    "26_final_error_distribution.png"
)

plt.savefig(
    error_distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"✓ {error_distribution_path.name}"
)


# ============================================================
# 15. ERROR BY GRADE RANGE PLOT
# ============================================================

print("\n" + "=" * 80)
print("14. ERROR BY GRADE RANGE VISUALIZATION")
print("=" * 80)

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    range_summary["Grade_Range"].astype(str),
    range_summary["MAE"]
)

plt.xlabel(
    "Actual G3 Grade Range"
)

plt.ylabel(
    "Mean Absolute Error"
)

plt.title(
    "Prediction Error Across Student Grade Ranges"
)

plt.tight_layout()

grade_error_plot_path = (
    VIS_DIR /
    "27_final_error_by_grade_range.png"
)

plt.savefig(
    grade_error_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"✓ {grade_error_plot_path.name}"
)


# ============================================================
# 16. SAVE COMPLETE ERROR DATA
# ============================================================

error_path = (
    REPORT_DIR /
    "step11_final_test_error_analysis.csv"
)

error_df.to_csv(
    error_path,
    index=False
)

print(
    f"\n✓ {error_path.name}"
)


# ============================================================
# 17. SAVE FINAL ANALYSIS SUMMARY
# ============================================================

analysis_summary = pd.DataFrame(
    [
        {
            "Metric": "Test MAE",
            "Value": test_metrics["MAE"]
        },
        {
            "Metric": "Test MSE",
            "Value": test_metrics["MSE"]
        },
        {
            "Metric": "Test RMSE",
            "Value": test_metrics["RMSE"]
        },
        {
            "Metric": "Test R2",
            "Value": test_metrics["R2"]
        },
        {
            "Metric": "Mean Prediction Error",
            "Value": error_df["Error"].mean()
        },
        {
            "Metric": "Maximum Absolute Error",
            "Value": error_df["Absolute_Error"].max()
        },
        {
            "Metric": "Number of Test Samples",
            "Value": len(y_test)
        },
        {
            "Metric": "Number of Model Features",
            "Value": len(feature_names)
        }
    ]
)

analysis_summary_path = (
    REPORT_DIR /
    "step11_final_analysis_summary.csv"
)

analysis_summary.to_csv(
    analysis_summary_path,
    index=False
)

print(
    f"✓ {analysis_summary_path.name}"
)


# ============================================================
# 18. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("15. STEP 11 FILE VERIFICATION")
print("=" * 80)

required_files = [
    REPORT_DIR /
    "step11_largest_prediction_errors.csv",

    REPORT_DIR /
    "step11_error_by_grade_range.csv",

    REPORT_DIR /
    "step11_final_feature_importance.csv",

    REPORT_DIR /
    "step11_final_test_error_analysis.csv",

    REPORT_DIR /
    "step11_final_analysis_summary.csv",

    VIS_DIR /
    "23_final_feature_importance.png",

    VIS_DIR /
    "24_final_actual_vs_predicted.png",

    VIS_DIR /
    "25_final_residual_analysis.png",

    VIS_DIR /
    "26_final_error_distribution.png",

    VIS_DIR /
    "27_final_error_by_grade_range.png"
]

all_files_exist = True

for path in required_files:

    if path.exists():

        print(
            f"✓ {path}"
        )

    else:

        print(
            f"✗ {path}"
        )

        all_files_exist = False


print("\n" + "=" * 80)

if all_files_exist:

    print(
        "STEP 11 FINAL MODEL ANALYSIS "
        "COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "STEP 11 COMPLETED WITH MISSING FILES"
    )

print("=" * 80)

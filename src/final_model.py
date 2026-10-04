import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

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
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

print("=" * 80)
print("WEEK 6 - STUDENT PERFORMANCE")
print("STEP 10: FINAL MODEL SELECTION, TRAINING & SAVING")
print("=" * 80)

DATA_PATH = Path("data/student-mat-cleaned.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

TARGET = "G3"
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. FINAL MODEL DECISION
# ============================================================

print("\n" + "=" * 80)
print("1. FINAL MODEL SELECTION")
print("=" * 80)

print("Selected scenario : Full Information")
print("Selected model    : Random Forest Regressor")
print()
print("Reason:")
print("- Strongest baseline test performance.")
print("- G1 and G2 are available before final grade G3.")
print("- Hyperparameter tuning did not improve test performance.")
print("- Baseline Random Forest remains the evidence-based choice.")

FINAL_PARAMS = {
    "n_estimators": 200,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

print("\nFinal Random Forest parameters:")

for parameter, value in FINAL_PARAMS.items():
    print(f"  {parameter}: {value}")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("2. DATASET LOADING")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print(f"Target  : {TARGET}")

if TARGET not in df.columns:
    raise ValueError("Target G3 is missing.")

print("✓ Dataset loaded successfully.")


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 80)
print("3. FEATURE / TARGET PREPARATION")
print("=" * 80)

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

print(f"Total features : {X.shape[1]}")
print(f"Target rows    : {len(y)}")

if "G3" in X.columns:
    raise ValueError("Target leakage detected: G3 exists in X.")

if "G1" not in X.columns or "G2" not in X.columns:
    raise ValueError("G1 or G2 is missing from Full Information scenario.")

print("✓ G3 excluded from features.")
print("✓ G1 and G2 included as valid available predictors.")


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("4. TRAIN / TEST SPLIT")
print("=" * 80)

(
    X_train,
    X_test,
    y_train,
    y_test
) = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Test size        : {TEST_SIZE * 100:.0f}%")
print(f"Random state     : {RANDOM_STATE}")

print("✓ Same split used in previous modeling steps.")


# ============================================================
# 6. IDENTIFY FEATURE TYPES
# ============================================================

print("\n" + "=" * 80)
print("5. FEATURE TYPES")
print("=" * 80)

numeric_features = (
    X_train
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)

categorical_features = (
    X_train
    .select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    )
    .columns
    .tolist()
)

print(f"Numerical features   : {len(numeric_features)}")
print(f"Categorical features : {len(categorical_features)}")

print("\nNumerical:")
print(", ".join(numeric_features))

print("\nCategorical:")
print(", ".join(categorical_features))


# ============================================================
# 7. BUILD FINAL PREPROCESSOR
# ============================================================

print("\n" + "=" * 80)
print("6. BUILDING PREPROCESSING PIPELINE")
print("=" * 80)

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
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

print("✓ Numerical preprocessing:")
print("  Median imputation → StandardScaler")

print("✓ Categorical preprocessing:")
print("  Most-frequent imputation → OneHotEncoder")

print("✓ Preprocessing is inside the model pipeline.")


# ============================================================
# 8. BUILD FINAL RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("7. BUILDING FINAL RANDOM FOREST")
print("=" * 80)

final_model = RandomForestRegressor(
    n_estimators=FINAL_PARAMS["n_estimators"],
    random_state=FINAL_PARAMS["random_state"],
    n_jobs=FINAL_PARAMS["n_jobs"]
)

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            final_model
        )
    ]
)

print("✓ Final pipeline created.")


# ============================================================
# 9. TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("8. TRAINING FINAL MODEL")
print("=" * 80)

final_pipeline.fit(
    X_train,
    y_train
)

print("✓ Final model trained successfully.")


# ============================================================
# 10. GENERATE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("9. FINAL HOLDOUT TEST EVALUATION")
print("=" * 80)

test_predictions = final_pipeline.predict(
    X_test
)

mse = mean_squared_error(
    y_test,
    test_predictions
)

final_metrics = {
    "MAE": mean_absolute_error(
        y_test,
        test_predictions
    ),
    "MSE": mse,
    "RMSE": np.sqrt(mse),
    "R2": r2_score(
        y_test,
        test_predictions
    )
}

for metric, value in final_metrics.items():
    print(f"{metric:<6}: {value:.4f}")


# ============================================================
# 11. TRAINING PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("10. TRAIN VS TEST PERFORMANCE")
print("=" * 80)

train_predictions = final_pipeline.predict(
    X_train
)

train_mse = mean_squared_error(
    y_train,
    train_predictions
)

train_metrics = {
    "Train_RMSE": np.sqrt(train_mse),
    "Train_R2": r2_score(
        y_train,
        train_predictions
    ),
    "Test_RMSE": final_metrics["RMSE"],
    "Test_R2": final_metrics["R2"]
}

for metric, value in train_metrics.items():
    print(f"{metric:<12}: {value:.4f}")


# ============================================================
# 12. TEST PREDICTION TABLE
# ============================================================

print("\n" + "=" * 80)
print("11. CREATING TEST PREDICTION REPORT")
print("=" * 80)

prediction_report = X_test.copy()

prediction_report["Actual_G3"] = y_test.values
prediction_report["Predicted_G3"] = test_predictions
prediction_report["Error"] = (
    prediction_report["Actual_G3"]
    -
    prediction_report["Predicted_G3"]
)

prediction_report["Absolute_Error"] = (
    prediction_report["Error"]
    .abs()
)

prediction_report = prediction_report.reset_index(
    names="Original_Index"
)

prediction_report_path = (
    REPORT_DIR /
    "final_model_test_predictions.csv"
)

prediction_report.to_csv(
    prediction_report_path,
    index=False
)

print(
    f"✓ {prediction_report_path.name}"
)


# ============================================================
# 13. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("12. SAVING FINAL MODEL")
print("=" * 80)

model_path = (
    MODEL_DIR /
    "student_performance_final_model.joblib"
)

joblib.dump(
    final_pipeline,
    model_path
)

print(f"✓ Model saved: {model_path}")


# ============================================================
# 14. RELOAD MODEL
# ============================================================

print("\n" + "=" * 80)
print("13. MODEL RELOAD VERIFICATION")
print("=" * 80)

loaded_model = joblib.load(
    model_path
)

print("✓ Saved model loaded successfully.")


# ============================================================
# 15. VERIFY RELOADED MODEL
# ============================================================

reloaded_predictions = loaded_model.predict(
    X_test
)

predictions_match = np.allclose(
    test_predictions,
    reloaded_predictions
)

print(
    f"Predictions identical after reload : "
    f"{predictions_match}"
)

if not predictions_match:
    raise ValueError(
        "Reloaded model predictions do not match."
    )

print("✓ Model serialization verified.")


# ============================================================
# 16. SAVE MODEL METADATA
# ============================================================

print("\n" + "=" * 80)
print("14. SAVING MODEL METADATA")
print("=" * 80)

metadata = {
    "project": "Week 6 - Student Performance Prediction",
    "dataset": "UCI Student Performance Dataset",
    "dataset_file": str(DATA_PATH),
    "target": TARGET,
    "problem_type": "Regression",
    "final_scenario": "Full Information",
    "model": "RandomForestRegressor",
    "reason_for_selection": (
        "Baseline Random Forest achieved the strongest "
        "test performance. Hyperparameter tuning did not "
        "improve the holdout test performance."
    ),
    "parameters": {
        key: str(value)
        for key, value in FINAL_PARAMS.items()
    },
    "features": {
        "total": int(X.shape[1]),
        "numerical": numeric_features,
        "categorical": categorical_features,
        "includes_G1": True,
        "includes_G2": True,
        "excludes_G3": True
    },
    "data_split": {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE
    },
    "metrics": {
        key: float(value)
        for key, value in final_metrics.items()
    },
    "train_test_performance": {
        key: float(value)
        for key, value in train_metrics.items()
    },
    "tuning_result": {
        "full_information_baseline_rmse": 2.0150,
        "full_information_tuned_rmse": 2.0196,
        "tuning_improvement_percent": -0.2272
    },
    "model_file": str(model_path),
    "prediction_report": str(prediction_report_path)
}

metadata_path = (
    MODEL_DIR /
    "student_performance_model_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        indent=4
    )

print(
    f"✓ {metadata_path}"
)


# ============================================================
# 17. FINAL MODEL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("15. FINAL MODEL SUMMARY")
print("=" * 80)

summary = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor",
            "Scenario": "Full Information",
            "Features": X.shape[1],
            "MAE": final_metrics["MAE"],
            "MSE": final_metrics["MSE"],
            "RMSE": final_metrics["RMSE"],
            "R2": final_metrics["R2"],
            "Train_RMSE": train_metrics["Train_RMSE"],
            "Train_R2": train_metrics["Train_R2"],
            "N_Estimators": FINAL_PARAMS["n_estimators"],
            "Random_State": FINAL_PARAMS["random_state"]
        }
    ]
)

summary_path = (
    REPORT_DIR /
    "final_model_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    summary.round(4).to_string(index=False)
)

print(
    f"\n✓ {summary_path.name}"
)


# ============================================================
# 18. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("16. FINAL FILE VERIFICATION")
print("=" * 80)

required_files = [
    MODEL_DIR / "student_performance_final_model.joblib",
    MODEL_DIR / "student_performance_model_metadata.json",
    REPORT_DIR / "final_model_test_predictions.csv",
    REPORT_DIR / "final_model_summary.csv"
]

all_files_exist = True

for path in required_files:

    if path.exists():

        size_kb = path.stat().st_size / 1024

        print(
            f"✓ {path} "
            f"({size_kb:.2f} KB)"
        )

    else:

        print(
            f"✗ {path}"
        )

        all_files_exist = False


print("\n" + "=" * 80)

if all_files_exist:
    print(
        "STEP 10 FINAL MODEL TRAINING & "
        "VERIFICATION COMPLETED SUCCESSFULLY"
    )
else:
    print(
        "STEP 10 COMPLETED WITH MISSING FILES"
    )

print("=" * 80)

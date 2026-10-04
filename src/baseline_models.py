# ============================================================
# WEEK 6 - STUDENT PERFORMANCE
# STEP 7: BASELINE REGRESSION MODEL TRAINING & EVALUATION
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "student-mat-cleaned.csv"
REPORT_DIR = PROJECT_DIR / "reports"

TARGET = "G3"
TEST_SIZE = 0.20
RANDOM_STATE = 42


REPORT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("WEEK 6 - STUDENT PERFORMANCE")
print("STEP 7: BASELINE REGRESSION MODEL TRAINING & EVALUATION")
print("=" * 80)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 80)
print("1. DATASET LOADING")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print(f"Dataset path : {DATA_PATH}")
print(f"Rows         : {df.shape[0]}")
print(f"Columns      : {df.shape[1]}")
print(f"Target       : {TARGET}")


# ============================================================
# 3. TARGET / FEATURE SEPARATION
# ============================================================

print("\n" + "=" * 80)
print("2. TARGET / FEATURE SEPARATION")
print("=" * 80)

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Feature rows : {X.shape[0]}")
print(f"Feature cols : {X.shape[1]}")
print(f"Target rows  : {y.shape[0]}")

if TARGET in X.columns:
    raise ValueError("Target G3 is still present in the feature set.")

print("\n✓ Target successfully separated from features.")


# ============================================================
# 4. CREATE TWO MODELING SCENARIOS
# ============================================================

print("\n" + "=" * 80)
print("3. MODELING SCENARIOS")
print("=" * 80)

# Scenario A:
# All available predictors except G3.
X_full = X.copy()

# Scenario B:
# Remove G1 and G2 to simulate prediction before previous
# academic grades are available.
X_pregrade = X.drop(
    columns=["G1", "G2"],
    errors="ignore"
).copy()

print("Scenario A - Full Information")
print(f"Raw features: {X_full.shape[1]}")
print("Includes G1:", "G1" in X_full.columns)
print("Includes G2:", "G2" in X_full.columns)

print("\nScenario B - Pre-Grade Prediction")
print(f"Raw features: {X_pregrade.shape[1]}")
print("Includes G1:", "G1" in X_pregrade.columns)
print("Includes G2:", "G2" in X_pregrade.columns)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("4. TRAIN / TEST SPLIT")
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
print(f"  Training samples: {len(X_train_full)}")
print(f"  Testing samples : {len(X_test_full)}")

print("\nPre-Grade Prediction:")
print(f"  Training samples: {len(X_train_pre)}")
print(f"  Testing samples : {len(X_test_pre)}")

print("\n✓ Same train/test split strategy used for both scenarios.")


# ============================================================
# 6. FUNCTION TO BUILD PREPROCESSOR
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
# 7. DEFINE BASELINE MODELS
# ============================================================

print("\n" + "=" * 80)
print("5. BASELINE MODELS")
print("=" * 80)

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(
        alpha=1.0
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=RANDOM_STATE,
        max_depth=None
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

for model_name in models:
    print(f"✓ {model_name}")

print("\nTotal baseline models:", len(models))


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    scenario_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    # Fit ONLY on training data.
    pipeline.fit(X_train, y_train)

    # Predict test data.
    y_pred = pipeline.predict(X_test)

    # Regression metrics.
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    # Training score for detecting obvious overfitting.
    train_predictions = pipeline.predict(X_train)

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_predictions
        )
    )

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    # Error statistics.
    errors = y_test.to_numpy() - y_pred

    mean_error = errors.mean()
    max_absolute_error = np.abs(errors).max()

    result = {
        "Scenario": scenario_name,
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Train_RMSE": train_rmse,
        "Train_R2": train_r2,
        "Mean_Error": mean_error,
        "Max_Absolute_Error": max_absolute_error,
        "Train_Samples": len(X_train),
        "Test_Samples": len(X_test)
    }

    return result, pipeline, y_pred


# ============================================================
# 9. TRAIN ALL MODELS
# ============================================================

print("\n" + "=" * 80)
print("6. MODEL TRAINING")
print("=" * 80)

results = []

trained_pipelines = {}

predictions = {}

for scenario_name, scenario_data in [
    (
        "Full Information",
        (
            X_train_full,
            X_test_full,
            y_train_full,
            y_test_full
        )
    ),
    (
        "Pre-Grade Prediction",
        (
            X_train_pre,
            X_test_pre,
            y_train_pre,
            y_test_pre
        )
    )
]:

    print("\n" + "-" * 80)
    print(f"SCENARIO: {scenario_name}")
    print("-" * 80)

    X_train, X_test, y_train, y_test = scenario_data

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}...")

        result, pipeline, y_pred = evaluate_model(
            model_name=model_name,
            scenario_name=scenario_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )

        results.append(result)

        key = (
            scenario_name,
            model_name
        )

        trained_pipelines[key] = pipeline
        predictions[key] = y_pred

        print(
            f"  MAE  : {result['MAE']:.4f}"
        )

        print(
            f"  RMSE : {result['RMSE']:.4f}"
        )

        print(
            f"  R²   : {result['R2']:.4f}"
        )


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("7. MODEL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(results)

display_columns = [
    "Scenario",
    "Model",
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "Train_RMSE",
    "Train_R2"
]

print(
    results_df[
        display_columns
    ].round(4).to_string(index=False)
)


# ============================================================
# 11. RANK MODELS
# ============================================================

print("\n" + "=" * 80)
print("8. MODEL RANKING")
print("=" * 80)

ranked_results = results_df.sort_values(
    by=["Scenario", "RMSE"],
    ascending=[True, True]
).copy()

ranked_results["RMSE_Rank"] = (
    ranked_results
    .groupby("Scenario")["RMSE"]
    .rank(
        method="min",
        ascending=True
    )
)

ranked_results["R2_Rank"] = (
    ranked_results
    .groupby("Scenario")["R2"]
    .rank(
        method="min",
        ascending=False
    )
)

print(
    ranked_results[
        [
            "Scenario",
            "Model",
            "RMSE",
            "R2",
            "RMSE_Rank",
            "R2_Rank"
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 12. COMPARE G1/G2 IMPACT
# ============================================================

print("\n" + "=" * 80)
print("9. IMPACT OF INCLUDING G1 AND G2")
print("=" * 80)

comparison = results_df.pivot(
    index="Model",
    columns="Scenario",
    values=["MAE", "RMSE", "R2"]
)

print(
    comparison
    .round(4)
    .to_string()
)


# ============================================================
# 13. BEST MODEL FOR EACH SCENARIO
# ============================================================

print("\n" + "=" * 80)
print("10. BEST BASELINE MODEL")
print("=" * 80)

best_models = {}

for scenario in results_df["Scenario"].unique():

    scenario_results = results_df[
        results_df["Scenario"] == scenario
    ]

    best_row = scenario_results.loc[
        scenario_results["RMSE"].idxmin()
    ]

    best_models[scenario] = {
        "Model": best_row["Model"],
        "RMSE": float(best_row["RMSE"]),
        "MAE": float(best_row["MAE"]),
        "R2": float(best_row["R2"])
    }

    print(f"\n{scenario}:")
    print(f"  Best model: {best_row['Model']}")
    print(f"  RMSE      : {best_row['RMSE']:.4f}")
    print(f"  MAE       : {best_row['MAE']:.4f}")
    print(f"  R²        : {best_row['R2']:.4f}")


# ============================================================
# 14. SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("11. SAVING MODEL RESULTS")
print("=" * 80)

results_path = REPORT_DIR / "baseline_model_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

ranking_path = REPORT_DIR / "baseline_model_ranking.csv"

ranked_results.to_csv(
    ranking_path,
    index=False
)

comparison_path = REPORT_DIR / "g1_g2_model_comparison.csv"

comparison.reset_index().to_csv(
    comparison_path,
    index=False
)

best_models_path = REPORT_DIR / "best_baseline_models.json"

with open(
    best_models_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        best_models,
        file,
        indent=4
    )

print(f"✓ {results_path.name}")
print(f"✓ {ranking_path.name}")
print(f"✓ {comparison_path.name}")
print(f"✓ {best_models_path.name}")


# ============================================================
# 15. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("12. FINAL VERIFICATION")
print("=" * 80)

print(f"✓ Dataset rows       : {len(df)}")
print(f"✓ Models trained     : {len(models)}")
print(f"✓ Scenarios tested   : 2")
print(f"✓ Total evaluations  : {len(results_df)}")
print("✓ Preprocessing fitted only on training data")
print("✓ Test data used only for final evaluation")
print("✓ MAE calculated")
print("✓ MSE calculated")
print("✓ RMSE calculated")
print("✓ R² calculated")
print("✓ Training metrics calculated for overfitting inspection")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("STEP 7 BASELINE MODELING COMPLETED")
print("=" * 80)

print("\nGenerated report files:")

for filename in [
    "baseline_model_results.csv",
    "baseline_model_ranking.csv",
    "g1_g2_model_comparison.csv",
    "best_baseline_models.json"
]:

    path = REPORT_DIR / filename

    if path.exists():
        print(f"✓ {filename}")
    else:
        print(f"✗ {filename}")

print("\n" + "=" * 80)
print("STEP 7 RUN FINISHED")
print("=" * 80)

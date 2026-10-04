# ============================================================
# WEEK 6 - STUDENT PERFORMANCE
# STEP 9: CROSS-VALIDATION & HYPERPARAMETER TUNING
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR /
    "data" /
    "student-mat-cleaned.csv"
)

REPORT_DIR = PROJECT_DIR / "reports"

TARGET = "G3"
TEST_SIZE = 0.20
RANDOM_STATE = 42

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("WEEK 6 - STUDENT PERFORMANCE")
print("STEP 9: CROSS-VALIDATION & HYPERPARAMETER TUNING")
print("=" * 80)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("1. DATASET LOADING")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print(f"Target  : {TARGET}")

if TARGET not in df.columns:
    raise ValueError("Target G3 is missing.")

print("✓ Dataset loaded successfully.")


# ============================================================
# 3. PREPARE TWO FEATURE SCENARIOS
# ============================================================

print("\n" + "=" * 80)
print("2. FEATURE SCENARIOS")
print("=" * 80)

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

# Scenario A:
# G1 and G2 are available when prediction is made.
X_full = X.copy()

# Scenario B:
# Prediction is made before G1 and G2 are available.
X_pregrade = X.drop(
    columns=["G1", "G2"],
    errors="ignore"
).copy()

print(
    f"Full Information features : "
    f"{X_full.shape[1]}"
)

print(
    f"Pre-Grade features        : "
    f"{X_pregrade.shape[1]}"
)

print(
    f"Target observations       : "
    f"{len(y)}"
)


# ============================================================
# 4. SAME TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("3. TRAIN / TEST SPLIT")
print("=" * 80)

(
    X_train_full,
    X_test_full,
    y_train_full,
    y_test_full
) = train_test_split(
    X_full,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

(
    X_train_pre,
    X_test_pre,
    y_train_pre,
    y_test_pre
) = train_test_split(
    X_pregrade,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("Full Information:")
print(
    f"  Train: {len(X_train_full)}"
)
print(
    f"  Test : {len(X_test_full)}"
)

print("\nPre-Grade Prediction:")
print(
    f"  Train: {len(X_train_pre)}"
)
print(
    f"  Test : {len(X_test_pre)}"
)

print("✓ Same split as previous steps.")


# ============================================================
# 5. PREPROCESSOR FUNCTION
# ============================================================

def build_preprocessor(X_data):

    numeric_features = (
        X_data
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X_data
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

    return ColumnTransformer(
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


# ============================================================
# 6. BUILD BASELINE PIPELINE
# ============================================================

def build_random_forest_pipeline(
    X_data
):

    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(
                    X_data
                )
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


# ============================================================
# 7. CROSS-VALIDATION CONFIGURATION
# ============================================================

print("\n" + "=" * 80)
print("4. CROSS-VALIDATION CONFIGURATION")
print("=" * 80)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

print("Cross-validation : 5-Fold KFold")
print("Shuffle           : True")
print(f"Random state      : {RANDOM_STATE}")


# ============================================================
# 8. HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "model__n_estimators": [
        100,
        200,
        400
    ],
    "model__max_depth": [
        None,
        5,
        10,
        15
    ],
    "model__min_samples_split": [
        2,
        5,
        10
    ],
    "model__min_samples_leaf": [
        1,
        2,
        4
    ],
    "model__max_features": [
        1.0,
        "sqrt"
    ]
}

total_combinations = (
    len(param_grid["model__n_estimators"])
    * len(param_grid["model__max_depth"])
    * len(param_grid["model__min_samples_split"])
    * len(param_grid["model__min_samples_leaf"])
    * len(param_grid["model__max_features"])
)

total_fits = (
    total_combinations *
    cv.get_n_splits()
)

print("\nHyperparameter grid:")

for parameter, values in param_grid.items():

    print(
        f"  {parameter}: {values}"
    )

print(
    f"\nTotal parameter combinations : "
    f"{total_combinations}"
)

print(
    f"Total cross-validation fits   : "
    f"{total_fits}"
)

print(
    "\nThis may take some time because "
    "multiple Random Forest models are evaluated."
)


# ============================================================
# 9. TUNING FUNCTION
# ============================================================

def tune_random_forest(
    X_train,
    y_train,
    scenario_name
):

    print("\n" + "-" * 80)
    print(
        f"TUNING: {scenario_name}"
    )
    print("-" * 80)

    pipeline = (
        build_random_forest_pipeline(
            X_train
        )
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(
        X_train,
        y_train
    )

    print(
        f"\n✓ {scenario_name} tuning completed."
    )

    print(
        "\nBest CV RMSE:"
    )

    print(
        f"{-grid_search.best_score_:.4f}"
    )

    print(
        "\nBest parameters:"
    )

    for parameter, value in (
        grid_search
        .best_params_
        .items()
    ):

        print(
            f"  {parameter}: {value}"
        )

    return grid_search


# ============================================================
# 10. RUN TUNING
# ============================================================

print("\n" + "=" * 80)
print("5. STARTING HYPERPARAMETER TUNING")
print("=" * 80)

grid_full = tune_random_forest(
    X_train_full,
    y_train_full,
    "Full Information"
)

grid_pregrade = tune_random_forest(
    X_train_pre,
    y_train_pre,
    "Pre-Grade Prediction"
)


# ============================================================
# 11. EXTRACT CV RESULTS
# ============================================================

print("\n" + "=" * 80)
print("6. CROSS-VALIDATION RESULTS")
print("=" * 80)


def extract_cv_results(
    grid_search,
    scenario_name
):

    results = pd.DataFrame(
        grid_search.cv_results_
    ).copy()

    results[
        "Mean_CV_RMSE"
    ] = -results[
        "mean_test_score"
    ]

    results[
        "Std_CV_RMSE"
    ] = results[
        "std_test_score"
    ]

    results[
        "Mean_Train_RMSE"
    ] = -results[
        "mean_train_score"
    ]

    results[
        "Scenario"
    ] = scenario_name

    selected_columns = [
        "Scenario",
        "Mean_CV_RMSE",
        "Std_CV_RMSE",
        "Mean_Train_RMSE",
        "param_model__n_estimators",
        "param_model__max_depth",
        "param_model__min_samples_split",
        "param_model__min_samples_leaf",
        "param_model__max_features",
        "rank_test_score"
    ]

    return results[
        selected_columns
    ].sort_values(
        by="Mean_CV_RMSE"
    )


cv_results_full = extract_cv_results(
    grid_full,
    "Full Information"
)

cv_results_pregrade = extract_cv_results(
    grid_pregrade,
    "Pre-Grade Prediction"
)


print("\nTop 10 Full Information configurations:")

print(
    cv_results_full
    .head(10)
    .round(4)
    .to_string(index=False)
)

print("\nTop 10 Pre-Grade configurations:")

print(
    cv_results_pregrade
    .head(10)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 12. TEST-SET EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("7. FINAL TEST-SET EVALUATION")
print("=" * 80)

best_full = (
    grid_full
    .best_estimator_
)

best_pregrade = (
    grid_pregrade
    .best_estimator_
)

pred_full = best_full.predict(
    X_test_full
)

pred_pregrade = best_pregrade.predict(
    X_test_pre
)


def evaluate_model(
    y_true,
    predictions,
    scenario
):

    mse = mean_squared_error(
        y_true,
        predictions
    )

    return {
        "Scenario": scenario,
        "MAE": mean_absolute_error(
            y_true,
            predictions
        ),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(
            y_true,
            predictions
        )
    }


tuned_test_results = pd.DataFrame(
    [
        evaluate_model(
            y_test_full,
            pred_full,
            "Full Information"
        ),
        evaluate_model(
            y_test_pre,
            pred_pregrade,
            "Pre-Grade Prediction"
        )
    ]
)

print(
    tuned_test_results
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 13. BASELINE VS TUNED COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("8. BASELINE VS TUNED COMPARISON")
print("=" * 80)

baseline_results = pd.DataFrame(
    [
        {
            "Scenario": "Full Information",
            "Baseline_MAE": 1.2005,
            "Baseline_RMSE": 2.0150,
            "Baseline_R2": 0.8020
        },
        {
            "Scenario": "Pre-Grade Prediction",
            "Baseline_MAE": 2.9938,
            "Baseline_RMSE": 3.7508,
            "Baseline_R2": 0.3139
        }
    ]
)

comparison = baseline_results.merge(
    tuned_test_results,
    on="Scenario"
)

comparison[
    "RMSE_Change"
] = (
    comparison["RMSE"]
    -
    comparison["Baseline_RMSE"]
)

comparison[
    "RMSE_Improvement_Percent"
] = (
    (
        comparison["Baseline_RMSE"]
        -
        comparison["RMSE"]
    )
    /
    comparison["Baseline_RMSE"]
    * 100
)

comparison[
    "R2_Change"
] = (
    comparison["R2"]
    -
    comparison["Baseline_R2"]
)

print(
    comparison
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 14. TRAIN VS TEST PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("9. TRAIN VS TEST PERFORMANCE")
print("=" * 80)


def train_test_performance(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    scenario
):

    train_pred = model.predict(
        X_train
    )

    test_pred = model.predict(
        X_test
    )

    train_mse = mean_squared_error(
        y_train,
        train_pred
    )

    test_mse = mean_squared_error(
        y_test,
        test_pred
    )

    return {
        "Scenario": scenario,
        "Train_RMSE": np.sqrt(
            train_mse
        ),
        "Test_RMSE": np.sqrt(
            test_mse
        ),
        "Train_R2": r2_score(
            y_train,
            train_pred
        ),
        "Test_R2": r2_score(
            y_test,
            test_pred
        )
    }


train_test_results = pd.DataFrame(
    [
        train_test_performance(
            best_full,
            X_train_full,
            y_train_full,
            X_test_full,
            y_test_full,
            "Full Information"
        ),
        train_test_performance(
            best_pregrade,
            X_train_pre,
            y_train_pre,
            X_test_pre,
            y_test_pre,
            "Pre-Grade Prediction"
        )
    ]
)

print(
    train_test_results
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 15. SAVE REPORTS
# ============================================================

print("\n" + "=" * 80)
print("10. SAVING REPORTS")
print("=" * 80)

cv_full_path = (
    REPORT_DIR /
    "step9_cv_results_full.csv"
)

cv_pregrade_path = (
    REPORT_DIR /
    "step9_cv_results_pregrade.csv"
)

comparison_path = (
    REPORT_DIR /
    "step9_baseline_vs_tuned.csv"
)

tuned_test_path = (
    REPORT_DIR /
    "step9_tuned_test_results.csv"
)

train_test_path = (
    REPORT_DIR /
    "step9_train_test_performance.csv"
)

cv_results_full.to_csv(
    cv_full_path,
    index=False
)

cv_results_pregrade.to_csv(
    cv_pregrade_path,
    index=False
)

comparison.to_csv(
    comparison_path,
    index=False
)

tuned_test_results.to_csv(
    tuned_test_path,
    index=False
)

train_test_results.to_csv(
    train_test_path,
    index=False
)

print(
    f"✓ {cv_full_path.name}"
)

print(
    f"✓ {cv_pregrade_path.name}"
)

print(
    f"✓ {comparison_path.name}"
)

print(
    f"✓ {tuned_test_path.name}"
)

print(
    f"✓ {train_test_path.name}"
)


# ============================================================
# 16. SAVE BEST PARAMETERS
# ============================================================

best_parameters = {
    "Full Information": {
        key: str(value)
        for key, value
        in grid_full.best_params_.items()
    },
    "Pre-Grade Prediction": {
        key: str(value)
        for key, value
        in grid_pregrade.best_params_.items()
    }
}

parameters_path = (
    REPORT_DIR /
    "step9_best_parameters.json"
)

with open(
    parameters_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        best_parameters,
        file,
        indent=4
    )

print(
    f"✓ {parameters_path.name}"
)


# ============================================================
# 17. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("11. FINAL VERIFICATION")
print("=" * 80)

required_files = [
    "step9_cv_results_full.csv",
    "step9_cv_results_pregrade.csv",
    "step9_baseline_vs_tuned.csv",
    "step9_tuned_test_results.csv",
    "step9_train_test_performance.csv",
    "step9_best_parameters.json"
]

all_files_exist = True

for filename in required_files:

    path = REPORT_DIR / filename

    if path.exists():

        print(
            f"✓ {filename}"
        )

    else:

        print(
            f"✗ {filename}"
        )

        all_files_exist = False


print("\n" + "=" * 80)

if all_files_exist:

    print(
        "STEP 9 CROSS-VALIDATION & "
        "HYPERPARAMETER TUNING COMPLETED"
    )

else:

    print(
        "STEP 9 COMPLETED WITH MISSING FILES"
    )

print("=" * 80)

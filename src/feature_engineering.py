from pathlib import Path

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# WEEK 6 - STUDENT PERFORMANCE
# STEP 6: FEATURE ENGINEERING & MODELING DATA PREPARATION
# ============================================================

DATA_PATH = Path("data/student-mat-cleaned.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "G3"

print("=" * 80)
print("WEEK 6 - STUDENT PERFORMANCE")
print("STEP 6: FEATURE ENGINEERING & MODELING DATA PREPARATION")
print("=" * 80)


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv(DATA_PATH, sep=";")

print("\n" + "=" * 80)
print("1. DATASET LOADING")
print("=" * 80)

print(f"Dataset path : {DATA_PATH}")
print(f"Rows         : {df.shape[0]}")
print(f"Columns      : {df.shape[1]}")
print(f"Target       : {TARGET}")


# ============================================================
# 2. TARGET / FEATURE SEPARATION
# ============================================================

print("\n" + "=" * 80)
print("2. TARGET / FEATURE SEPARATION")
print("=" * 80)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found.")

X_full = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

print(f"Feature rows : {X_full.shape[0]}")
print(f"Feature cols : {X_full.shape[1]}")
print(f"Target rows  : {y.shape[0]}")

print("\nTarget information:")
print(f"Target dtype : {y.dtype}")
print(f"Target mean  : {y.mean():.3f}")
print(f"Target min   : {y.min()}")
print(f"Target max   : {y.max()}")


# ============================================================
# 3. CHECK FOR TARGET LEAKAGE
# ============================================================

print("\n" + "=" * 80)
print("3. TARGET LEAKAGE CHECK")
print("=" * 80)

if TARGET in X_full.columns:
    print("ERROR: G3 is present in the feature set.")
    raise ValueError("Target leakage detected: G3 is inside X.")
else:
    print("✓ G3 is not present in the feature set.")

leakage_candidates = [
    col for col in X_full.columns
    if col.lower() in {
        "g3",
        "final_grade",
        "finalgrade",
        "target"
    }
]

if leakage_candidates:
    print(f"Potential target-like columns: {leakage_candidates}")
else:
    print("✓ No additional target-like column detected.")


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

print("\n" + "=" * 80)
print("4. FEATURE TYPE IDENTIFICATION")
print("=" * 80)

numeric_features = X_full.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_full.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

print(f"Numerical features  : {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

print("\nNumerical features:")
for feature in numeric_features:
    print(f"  - {feature}")

print("\nCategorical features:")
for feature in categorical_features:
    print(f"  - {feature}")


# ============================================================
# 5. SCENARIO A - FULL INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("5. SCENARIO A - FULL INFORMATION")
print("=" * 80)

print(
    "This scenario includes G1 and G2 because previous-period "
    "grades are available before the final G3 grade."
)

X_full_scenario = X_full.copy()

print(f"Number of features: {X_full_scenario.shape[1]}")

print("\nG1 present:", "G1" in X_full_scenario.columns)
print("G2 present:", "G2" in X_full_scenario.columns)
print("G3 present:", "G3" in X_full_scenario.columns)


# ============================================================
# 6. SCENARIO B - PRE-GRADE PREDICTION
# ============================================================

print("\n" + "=" * 80)
print("6. SCENARIO B - PRE-GRADE PREDICTION")
print("=" * 80)

print(
    "This scenario excludes G1 and G2 to simulate prediction "
    "before previous academic grades are available."
)

excluded_grade_features = ["G1", "G2"]

X_pregrade = X_full.drop(
    columns=excluded_grade_features,
    errors="ignore"
).copy()

print(f"Number of features: {X_pregrade.shape[1]}")

print("\nG1 present:", "G1" in X_pregrade.columns)
print("G2 present:", "G2" in X_pregrade.columns)
print("G3 present:", "G3" in X_pregrade.columns)

if "G3" in X_pregrade.columns:
    raise ValueError("G3 unexpectedly exists in pre-grade features.")


# ============================================================
# 7. FEATURE COUNT COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("7. FEATURE SET COMPARISON")
print("=" * 80)

feature_comparison = pd.DataFrame(
    {
        "Scenario": [
            "Full Information",
            "Pre-Grade Prediction"
        ],
        "Total Features": [
            X_full_scenario.shape[1],
            X_pregrade.shape[1]
        ],
        "G1 Included": [
            "Yes",
            "No"
        ],
        "G2 Included": [
            "Yes",
            "No"
        ],
        "G3 Included": [
            "No",
            "No"
        ]
    }
)

print(feature_comparison.to_string(index=False))

feature_comparison.to_csv(
    REPORT_DIR / "feature_set_comparison.csv",
    index=False
)


# ============================================================
# 8. TRAIN / TEST SPLIT - FULL INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("8. TRAIN / TEST SPLIT - FULL INFORMATION")
print("=" * 80)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full_scenario,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Training samples: {len(X_train_full)}")
print(f"Testing samples : {len(X_test_full)}")
print(f"Training target : {len(y_train_full)}")
print(f"Testing target  : {len(y_test_full)}")

print(
    f"Train percentage: "
    f"{len(X_train_full) / len(df) * 100:.1f}%"
)

print(
    f"Test percentage : "
    f"{len(X_test_full) / len(df) * 100:.1f}%"
)


# ============================================================
# 9. TRAIN / TEST SPLIT - PRE-GRADE
# ============================================================

print("\n" + "=" * 80)
print("9. TRAIN / TEST SPLIT - PRE-GRADE")
print("=" * 80)

X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
    X_pregrade,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Training samples: {len(X_train_pre)}")
print(f"Testing samples : {len(X_test_pre)}")
print(f"Training target : {len(y_train_pre)}")
print(f"Testing target  : {len(y_test_pre)}")


# ============================================================
# 10. VERIFY TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("10. TARGET DISTRIBUTION CHECK")
print("=" * 80)

distribution_check = pd.DataFrame(
    {
        "Dataset": [
            "Full Dataset",
            "Full Train",
            "Full Test"
        ],
        "Count": [
            len(y),
            len(y_train_full),
            len(y_test_full)
        ],
        "Mean G3": [
            y.mean(),
            y_train_full.mean(),
            y_test_full.mean()
        ],
        "Std G3": [
            y.std(),
            y_train_full.std(),
            y_test_full.std()
        ],
        "Min G3": [
            y.min(),
            y_train_full.min(),
            y_test_full.min()
        ],
        "Max G3": [
            y.max(),
            y_train_full.max(),
            y_test_full.max()
        ]
    }
)

print(distribution_check.round(3).to_string(index=False))

distribution_check.to_csv(
    REPORT_DIR / "train_test_target_distribution.csv",
    index=False
)


# ============================================================
# 11. BUILD PREPROCESSING PIPELINES
# ============================================================

print("\n" + "=" * 80)
print("11. PREPROCESSING PIPELINES")
print("=" * 80)

# Feature types for the full scenario
full_numeric_features = X_full_scenario.select_dtypes(
    include=np.number
).columns.tolist()

full_categorical_features = X_full_scenario.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

# Feature types for the pre-grade scenario
pre_numeric_features = X_pregrade.select_dtypes(
    include=np.number
).columns.tolist()

pre_categorical_features = X_pregrade.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()


# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing
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


# Full information preprocessing
preprocessor_full = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            full_numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            full_categorical_features
        )
    ],
    remainder="drop"
)


# Pre-grade preprocessing
preprocessor_pregrade = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            pre_numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            pre_categorical_features
        )
    ],
    remainder="drop"
)


print("✓ Numerical preprocessing:")
print("  - Median imputation")
print("  - StandardScaler")

print("\n✓ Categorical preprocessing:")
print("  - Most-frequent imputation")
print("  - One-hot encoding")
print("  - handle_unknown='ignore'")

print("\n✓ Full-information ColumnTransformer created")
print("✓ Pre-grade ColumnTransformer created")


# ============================================================
# 12. FIT PREPROCESSING ONLY ON TRAINING DATA
# ============================================================

print("\n" + "=" * 80)
print("12. PREPROCESSING FIT CHECK")
print("=" * 80)

print(
    "The preprocessors will be fitted only on training data "
    "to prevent information leakage from the test set."
)

X_train_full_transformed = preprocessor_full.fit_transform(
    X_train_full
)

X_test_full_transformed = preprocessor_full.transform(
    X_test_full
)

X_train_pre_transformed = preprocessor_pregrade.fit_transform(
    X_train_pre
)

X_test_pre_transformed = preprocessor_pregrade.transform(
    X_test_pre
)

print("\nFull-information transformed shapes:")
print(f"Training: {X_train_full_transformed.shape}")
print(f"Testing : {X_test_full_transformed.shape}")

print("\nPre-grade transformed shapes:")
print(f"Training: {X_train_pre_transformed.shape}")
print(f"Testing : {X_test_pre_transformed.shape}")


# ============================================================
# 13. CHECK TRANSFORMED DATA
# ============================================================

print("\n" + "=" * 80)
print("13. TRANSFORMED DATA VALIDATION")
print("=" * 80)

checks = {
    "Full train contains NaN": bool(
        np.isnan(X_train_full_transformed).any()
    ),
    "Full test contains NaN": bool(
        np.isnan(X_test_full_transformed).any()
    ),
    "Pre-grade train contains NaN": bool(
        np.isnan(X_train_pre_transformed).any()
    ),
    "Pre-grade test contains NaN": bool(
        np.isnan(X_test_pre_transformed).any()
    ),
}

for name, result in checks.items():
    print(f"{name}: {result}")

if any(checks.values()):
    raise ValueError(
        "NaN values detected after preprocessing."
    )

print("\n✓ No NaN values detected after preprocessing.")


# ============================================================
# 14. GET FEATURE NAMES AFTER ONE-HOT ENCODING
# ============================================================

print("\n" + "=" * 80)
print("14. TRANSFORMED FEATURE NAMES")
print("=" * 80)

full_feature_names = preprocessor_full.get_feature_names_out()
pre_feature_names = preprocessor_pregrade.get_feature_names_out()

print(
    f"Full-information transformed features : "
    f"{len(full_feature_names)}"
)

print(
    f"Pre-grade transformed features         : "
    f"{len(pre_feature_names)}"
)

print("\nFirst 20 full-information transformed features:")

for feature in full_feature_names[:20]:
    print(f"  - {feature}")

print("\nFirst 20 pre-grade transformed features:")

for feature in pre_feature_names[:20]:
    print(f"  - {feature}")


# ============================================================
# 15. SAVE FEATURE INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("15. SAVING FEATURE INFORMATION")
print("=" * 80)

feature_info = {
    "target": TARGET,
    "dataset": str(DATA_PATH),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "full_information": {
        "raw_feature_count": len(X_full_scenario.columns),
        "numeric_features": full_numeric_features,
        "categorical_features": full_categorical_features,
        "transformed_feature_count": len(full_feature_names),
        "includes_G1": "G1" in X_full_scenario.columns,
        "includes_G2": "G2" in X_full_scenario.columns,
        "includes_G3": "G3" in X_full_scenario.columns
    },
    "pre_grade_prediction": {
        "raw_feature_count": len(X_pregrade.columns),
        "numeric_features": pre_numeric_features,
        "categorical_features": pre_categorical_features,
        "transformed_feature_count": len(pre_feature_names),
        "includes_G1": "G1" in X_pregrade.columns,
        "includes_G2": "G2" in X_pregrade.columns,
        "includes_G3": "G3" in X_pregrade.columns
    }
}

with open(
    REPORT_DIR / "feature_information.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(feature_info, file, indent=4)

pd.DataFrame(
    {
        "feature_name": full_feature_names,
        "scenario": "Full Information"
    }
).to_csv(
    REPORT_DIR / "transformed_features_full.csv",
    index=False
)

pd.DataFrame(
    {
        "feature_name": pre_feature_names,
        "scenario": "Pre-Grade Prediction"
    }
).to_csv(
    REPORT_DIR / "transformed_features_pregrade.csv",
    index=False
)


# ============================================================
# 16. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("16. FINAL VERIFICATION")
print("=" * 80)

print(f"✓ Original dataset rows       : {len(df)}")
print(f"✓ Target                     : {TARGET}")
print(f"✓ Full raw features          : {X_full_scenario.shape[1]}")
print(f"✓ Pre-grade raw features     : {X_pregrade.shape[1]}")
print(f"✓ Full transformed features  : {len(full_feature_names)}")
print(f"✓ Pre-grade transformed      : {len(pre_feature_names)}")
print(f"✓ Train/test ratio           : 80/20")
print(f"✓ Random state               : {RANDOM_STATE}")
print("✓ Target leakage check passed")
print("✓ Preprocessing leakage check passed")
print("✓ Missing-value check passed")
print("✓ Feature preparation completed")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("STEP 6 FEATURE ENGINEERING & MODELING PREPARATION COMPLETED")
print("=" * 80)

print("\nGenerated report files:")

for filename in [
    "feature_set_comparison.csv",
    "train_test_target_distribution.csv",
    "feature_information.json",
    "transformed_features_full.csv",
    "transformed_features_pregrade.csv"
]:
    path = REPORT_DIR / filename

    if path.exists():
        print(f"✓ {filename}")
    else:
        print(f"✗ {filename}")

print("\n" + "=" * 80)
print("STEP 6 RUN FINISHED")
print("=" * 80)

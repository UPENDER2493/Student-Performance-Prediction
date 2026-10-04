from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/student-mat.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH, sep=";")


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("WEEK 6 - STUDENT PERFORMANCE")
print("STEP 3: DATA UNDERSTANDING")
print("=" * 80)


# ============================================================
# 1. BASIC DATASET INFORMATION
# ============================================================

print()
print("=" * 80)
print("1. BASIC DATASET INFORMATION")
print("=" * 80)

print(f"Dataset file       : {DATA_PATH}")
print(f"Number of rows     : {df.shape[0]}")
print(f"Number of columns  : {df.shape[1]}")
print(f"Total cells        : {df.shape[0] * df.shape[1]:,}")


# ============================================================
# 2. TARGET INFORMATION
# ============================================================

print()
print("=" * 80)
print("2. TARGET VARIABLE")
print("=" * 80)

target = "G3"

print("Target name        : G3")
print("Target meaning     : Final mathematics grade")
print("Target range       : 0 to 20")
print("Problem type       : Regression")

print()
print("Target statistics:")
print(df[target].describe().to_string())

print()
print(f"Mean               : {df[target].mean():.3f}")
print(f"Median             : {df[target].median():.3f}")
print(f"Standard deviation : {df[target].std():.3f}")
print(f"Minimum            : {df[target].min()}")
print(f"Maximum            : {df[target].max()}")
print(f"Skewness           : {df[target].skew():.3f}")

print()
print("Target value counts:")
print(df[target].value_counts().sort_index().to_string())


# ============================================================
# 3. DATA TYPES
# ============================================================

print()
print("=" * 80)
print("3. DATA TYPES")
print("=" * 80)

dtype_summary = (
    df.dtypes
    .value_counts()
    .rename_axis("Data_Type")
    .reset_index(name="Feature_Count")
)

print(dtype_summary.to_string(index=False))

dtype_summary.to_csv(
    REPORT_DIR / "data_type_summary.csv",
    index=False
)


# ============================================================
# 4. MISSING VALUES
# ============================================================

print()
print("=" * 80)
print("4. MISSING VALUE ANALYSIS")
print("=" * 80)

missing_table = pd.DataFrame(
    {
        "Feature": df.columns,
        "Missing_Count": df.isna().sum().values,
        "Missing_Percentage": (
            df.isna().mean().values * 100
        ),
    }
)

missing_table["Missing_Percentage"] = (
    missing_table["Missing_Percentage"].round(3)
)

print(missing_table.to_string(index=False))

total_missing = int(df.isna().sum().sum())

print()
print(f"Total missing values: {total_missing}")

missing_table.to_csv(
    REPORT_DIR / "missing_value_analysis.csv",
    index=False
)


# ============================================================
# 5. DUPLICATE ANALYSIS
# ============================================================

print()
print("=" * 80)
print("5. DUPLICATE ANALYSIS")
print("=" * 80)

duplicate_count = int(df.duplicated().sum())
duplicate_percentage = (
    duplicate_count / len(df) * 100
)

print(f"Duplicate rows       : {duplicate_count}")
print(
    f"Duplicate percentage : "
    f"{duplicate_percentage:.3f}%"
)


# ============================================================
# 6. NUMERICAL FEATURES
# ============================================================

print()
print("=" * 80)
print("6. NUMERICAL FEATURES")
print("=" * 80)

numerical_features = df.select_dtypes(
    include=np.number
).columns.tolist()

print(
    f"Number of numerical features: "
    f"{len(numerical_features)}"
)

print()
print(numerical_features)

numerical_summary = df[numerical_features].describe().T

numerical_summary["missing"] = (
    df[numerical_features].isna().sum()
)

numerical_summary["unique"] = (
    df[numerical_features].nunique()
)

print()
print(
    numerical_summary.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

numerical_summary.to_csv(
    REPORT_DIR / "numerical_feature_summary.csv"
)


# ============================================================
# 7. CATEGORICAL FEATURES
# ============================================================

print()
print("=" * 80)
print("7. CATEGORICAL FEATURES")
print("=" * 80)

categorical_features = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print(
    f"Number of categorical features: "
    f"{len(categorical_features)}"
)

print()
print(categorical_features)


# ============================================================
# 8. CATEGORICAL DISTRIBUTIONS
# ============================================================

print()
print("=" * 80)
print("8. CATEGORICAL FEATURE DISTRIBUTIONS")
print("=" * 80)

categorical_records = []

for feature in categorical_features:

    counts = df[feature].value_counts()

    percentages = (
        df[feature]
        .value_counts(normalize=True)
        .mul(100)
    )

    print()
    print(f"--- {feature} ---")

    for category in counts.index:

        count = int(counts[category])
        percentage = float(percentages[category])

        print(
            f"{str(category):20} "
            f"{count:5} "
            f"({percentage:6.2f}%)"
        )

        categorical_records.append(
            {
                "Feature": feature,
                "Category": category,
                "Count": count,
                "Percentage": round(
                    percentage,
                    3
                ),
            }
        )

categorical_distribution = pd.DataFrame(
    categorical_records
)

categorical_distribution.to_csv(
    REPORT_DIR / "categorical_distributions.csv",
    index=False
)


# ============================================================
# 9. UNIQUE VALUES
# ============================================================

print()
print("=" * 80)
print("9. UNIQUE VALUE ANALYSIS")
print("=" * 80)

unique_table = pd.DataFrame(
    {
        "Feature": df.columns,
        "Unique_Values": [
            df[column].nunique()
            for column in df.columns
        ],
    }
)

print(unique_table.to_string(index=False))

unique_table.to_csv(
    REPORT_DIR / "unique_value_analysis.csv",
    index=False
)


# ============================================================
# 10. G1 / G2 / G3 ANALYSIS
# ============================================================

print()
print("=" * 80)
print("10. ACADEMIC GRADE VARIABLES")
print("=" * 80)

grade_features = [
    feature
    for feature in ["G1", "G2", "G3"]
    if feature in df.columns
]

print("Grade variables:")
print(grade_features)

grade_summary = df[grade_features].describe().T

print()
print(
    grade_summary.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

grade_summary.to_csv(
    REPORT_DIR / "academic_grade_summary.csv"
)


# ============================================================
# 11. ZERO GRADE ANALYSIS
# ============================================================

print()
print("=" * 80)
print("11. ZERO-GRADE ANALYSIS")
print("=" * 80)

zero_grade_count = int(
    (df["G3"] == 0).sum()
)

zero_grade_percentage = (
    zero_grade_count / len(df) * 100
)

print(f"G3 = 0 count       : {zero_grade_count}")
print(
    f"G3 = 0 percentage  : "
    f"{zero_grade_percentage:.3f}%"
)

if "G1" in df.columns:
    print(
        f"G1 = 0 count       : "
        f"{int((df['G1'] == 0).sum())}"
    )

if "G2" in df.columns:
    print(
        f"G2 = 0 count       : "
        f"{int((df['G2'] == 0).sum())}"
    )


# ============================================================
# 12. RANGE VALIDATION
# ============================================================

print()
print("=" * 80)
print("12. RANGE VALIDATION")
print("=" * 80)

range_rules = {
    "age": (0, 100),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, None),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}

range_records = []

for feature, (lower, upper) in range_rules.items():

    if feature not in df.columns:
        continue

    series = df[feature]

    if lower is not None:
        lower_violations = int(
            (series < lower).sum()
        )
    else:
        lower_violations = 0

    if upper is not None:
        upper_violations = int(
            (series > upper).sum()
        )
    else:
        upper_violations = 0

    total_violations = (
        lower_violations +
        upper_violations
    )

    print(
        f"{feature:15} "
        f"violations = {total_violations}"
    )

    range_records.append(
        {
            "Feature": feature,
            "Expected_Min": lower,
            "Expected_Max": upper,
            "Lower_Violations": lower_violations,
            "Upper_Violations": upper_violations,
            "Total_Violations": total_violations,
        }
    )

range_validation = pd.DataFrame(
    range_records
)

range_validation.to_csv(
    REPORT_DIR / "range_validation.csv",
    index=False
)


# ============================================================
# 13. TARGET CORRELATION
# ============================================================

print()
print("=" * 80)
print("13. NUMERICAL CORRELATION WITH G3")
print("=" * 80)

correlations = (
    df.corr(numeric_only=True)["G3"]
    .drop("G3")
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

correlation_table = pd.DataFrame(
    {
        "Feature": correlations.index,
        "Correlation": correlations.values,
        "Absolute_Correlation": (
            correlations.abs().values
        ),
    }
)

print(
    correlation_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

correlation_table.to_csv(
    REPORT_DIR / "target_correlation_ranking.csv",
    index=False
)


# ============================================================
# 14. LEAKAGE WARNING
# ============================================================

print()
print("=" * 80)
print("14. POTENTIAL TARGET LEAKAGE CHECK")
print("=" * 80)

print(
    "G1 and G2 are previous-period grades and are "
    "strongly related to the final grade G3."
)

print(
    "They will NOT be removed automatically at this stage."
)

print(
    "We will explicitly compare:"
)

print(
    "1. A model using G1/G2"
)

print(
    "2. A model excluding G1/G2"
)

print(
    "This allows the final report to discuss "
    "prediction timing and leakage risk properly."
)


# ============================================================
# 15. FINAL DATA QUALITY SUMMARY
# ============================================================

print()
print("=" * 80)
print("15. FINAL DATA QUALITY SUMMARY")
print("=" * 80)

print(f"Rows                 : {len(df):,}")
print(f"Columns              : {len(df.columns)}")
print(f"Missing values       : {total_missing:,}")
print(f"Duplicate rows       : {duplicate_count:,}")
print(f"Numerical features   : {len(numerical_features)}")
print(f"Categorical features : {len(categorical_features)}")
print(f"G3 mean              : {df['G3'].mean():.3f}")
print(f"G3 median            : {df['G3'].median():.3f}")
print(f"G3 standard deviation: {df['G3'].std():.3f}")
print(f"G3 skewness          : {df['G3'].skew():.3f}")
print(f"G3 zero count        : {zero_grade_count}")
print(
    f"G3 zero percentage   : "
    f"{zero_grade_percentage:.3f}%"
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 80)
print("STEP 3 DATA UNDERSTANDING COMPLETED")
print("=" * 80)

print()
print("Generated report files:")

report_files = [
    "data_type_summary.csv",
    "missing_value_analysis.csv",
    "numerical_feature_summary.csv",
    "categorical_distributions.csv",
    "unique_value_analysis.csv",
    "academic_grade_summary.csv",
    "range_validation.csv",
    "target_correlation_ranking.csv",
]

for filename in report_files:

    path = REPORT_DIR / filename

    status = "✓" if path.exists() else "✗"

    print(f"{status} {filename}")

print()
print("=" * 80)

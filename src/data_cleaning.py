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
print("STEP 4: DATA CLEANING & OUTLIER INVESTIGATION")
print("=" * 80)


# ============================================================
# 1. INITIAL DATASET STATUS
# ============================================================

print()
print("=" * 80)
print("1. INITIAL DATASET STATUS")
print("=" * 80)

print(f"Rows              : {len(df):,}")
print(f"Columns           : {len(df.columns)}")
print(f"Missing values    : {int(df.isna().sum().sum()):,}")
print(f"Duplicate rows    : {int(df.duplicated().sum()):,}")


# ============================================================
# 2. MISSING VALUE CHECK
# ============================================================

print()
print("=" * 80)
print("2. MISSING VALUE CHECK")
print("=" * 80)

missing = df.isna().sum()

missing_features = missing[missing > 0]

if missing_features.empty:
    print("No missing values detected.")
else:
    print(missing_features.to_string())


# ============================================================
# 3. DUPLICATE CHECK
# ============================================================

print()
print("=" * 80)
print("3. DUPLICATE CHECK")
print("=" * 80)

duplicate_count = int(df.duplicated().sum())

print(f"Duplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    print("Duplicate rows will be removed.")
    df = df.drop_duplicates().reset_index(drop=True)
else:
    print("No duplicate rows require removal.")


# ============================================================
# 4. LOGICAL RANGE VALIDATION
# ============================================================

print()
print("=" * 80)
print("4. LOGICAL RANGE VALIDATION")
print("=" * 80)

range_rules = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 3),
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
invalid_rows = set()

for feature, (lower, upper) in range_rules.items():

    if feature not in df.columns:
        continue

    series = df[feature]

    if lower is not None:
        lower_mask = series < lower
        lower_violations = int(lower_mask.sum())
        invalid_rows.update(
            df.index[lower_mask].tolist()
        )
    else:
        lower_violations = 0

    if upper is not None:
        upper_mask = series > upper
        upper_violations = int(upper_mask.sum())
        invalid_rows.update(
            df.index[upper_mask].tolist()
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

range_validation = pd.DataFrame(range_records)

range_validation.to_csv(
    REPORT_DIR / "cleaning_range_validation.csv",
    index=False
)


# ============================================================
# 5. ZERO-GRADE INVESTIGATION
# ============================================================

print()
print("=" * 80)
print("5. ZERO-GRADE INVESTIGATION")
print("=" * 80)

zero_g3 = df["G3"] == 0

zero_count = int(zero_g3.sum())
zero_percentage = zero_count / len(df) * 100

print(f"G3 = 0 records     : {zero_count}")
print(f"G3 = 0 percentage  : {zero_percentage:.3f}%")

print()
print("G1/G2 values among students with G3 = 0:")

zero_grade_summary = (
    df.loc[
        zero_g3,
        ["G1", "G2", "G3", "absences", "failures"]
    ]
    .describe()
)

print(
    zero_grade_summary.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

print()
print(
    "Decision: G3 = 0 values will NOT be removed "
    "because 0 is within the valid grade range."
)


# ============================================================
# 6. ABSENCES DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("6. ABSENCES DISTRIBUTION")
print("=" * 80)

absence_stats = df["absences"].describe()

print(
    absence_stats.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

print()
print("Selected absence percentiles:")

absence_percentiles = df["absences"].quantile(
    [0.50, 0.75, 0.90, 0.95, 0.99]
)

print(
    absence_percentiles.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# 7. IQR OUTLIER ANALYSIS
# ============================================================

print()
print("=" * 80)
print("7. IQR OUTLIER ANALYSIS")
print("=" * 80)

outlier_features = [
    "absences",
    "G1",
    "G2",
    "G3",
]

outlier_records = []

for feature in outlier_features:

    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (
        (df[feature] < lower_bound)
        | (df[feature] > upper_bound)
    )

    count = int(mask.sum())
    percentage = count / len(df) * 100

    print()
    print(f"Feature              : {feature}")
    print(f"Q1                   : {q1:.3f}")
    print(f"Q3                   : {q3:.3f}")
    print(f"IQR                  : {iqr:.3f}")
    print(f"Lower bound          : {lower_bound:.3f}")
    print(f"Upper bound          : {upper_bound:.3f}")
    print(f"Outlier candidates   : {count}")
    print(f"Percentage           : {percentage:.3f}%")

    outlier_records.append(
        {
            "Feature": feature,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "Lower_Bound": round(lower_bound, 3),
            "Upper_Bound": round(upper_bound, 3),
            "Outlier_Candidates": count,
            "Outlier_Percentage": round(
                percentage,
                3
            ),
        }
    )

outlier_summary = pd.DataFrame(
    outlier_records
)

outlier_summary.to_csv(
    REPORT_DIR / "outlier_analysis.csv",
    index=False
)


# ============================================================
# 8. ABSENCE EXTREME VALUES
# ============================================================

print()
print("=" * 80)
print("8. EXTREME ABSENCE VALUES")
print("=" * 80)

absence_q1 = df["absences"].quantile(0.25)
absence_q3 = df["absences"].quantile(0.75)

absence_iqr = absence_q3 - absence_q1

absence_upper_bound = (
    absence_q3 + 1.5 * absence_iqr
)

absence_outliers = df[
    df["absences"] > absence_upper_bound
].copy()

print(
    f"IQR upper threshold: "
    f"{absence_upper_bound:.3f}"
)

print(
    f"Extreme absence records: "
    f"{len(absence_outliers)}"
)

if not absence_outliers.empty:

    print()
    print("Extreme absence values:")

    print(
        absence_outliers[
            [
                "absences",
                "G1",
                "G2",
                "G3",
                "failures"
            ]
        ]
        .sort_values(
            "absences",
            ascending=False
        )
        .to_string(index=False)
    )

    absence_outliers[
        [
            "school",
            "sex",
            "age",
            "absences",
            "G1",
            "G2",
            "G3",
            "failures"
        ]
    ].to_csv(
        REPORT_DIR / "absence_extreme_records.csv",
        index=False
    )

else:

    print("No extreme absence records found.")


# ============================================================
# 9. IMPACT OF ABSENCE EXTREMES
# ============================================================

print()
print("=" * 80)
print("9. IMPACT OF EXTREME ABSENCES")
print("=" * 80)

normal_absence = df[
    df["absences"] <= absence_upper_bound
]

extreme_absence = df[
    df["absences"] > absence_upper_bound
]

print(
    f"Normal absence group   : "
    f"{len(normal_absence)} students"
)

print(
    f"Extreme absence group  : "
    f"{len(extreme_absence)} students"
)

if len(extreme_absence) > 0:

    normal_mean_g3 = normal_absence["G3"].mean()
    extreme_mean_g3 = extreme_absence["G3"].mean()

    print()
    print(
        f"Normal-group G3 mean   : "
        f"{normal_mean_g3:.3f}"
    )

    print(
        f"Extreme-group G3 mean  : "
        f"{extreme_mean_g3:.3f}"
    )

    print(
        f"Difference in G3 mean  : "
        f"{extreme_mean_g3 - normal_mean_g3:.3f}"
    )

else:

    print("No extreme absence group available.")


# ============================================================
# 10. DECISION ON OUTLIERS
# ============================================================

print()
print("=" * 80)
print("10. OUTLIER HANDLING DECISION")
print("=" * 80)

print(
    "IQR results identify statistical outlier candidates, "
    "not automatically invalid observations."
)

print(
    "Ordinal survey variables will not be removed based "
    "only on IQR rules."
)

print(
    "G1, G2 and G3 values are valid grades within the "
    "dataset's 0-20 scale."
)

print(
    "Absence extremes will be retained unless there is "
    "evidence that they are data-entry errors."
)

print(
    "Therefore, no observations are removed solely because "
    "they are statistical outliers."
)


# ============================================================
# 11. FINAL CLEANED DATASET
# ============================================================

print()
print("=" * 80)
print("11. FINAL CLEANED DATASET")
print("=" * 80)

cleaned_df = df.copy()

cleaned_path = (
    DATA_PATH.parent /
    "student-mat-cleaned.csv"
)

cleaned_df.to_csv(
    cleaned_path,
    index=False
)

print(
    f"Cleaned dataset saved to: "
    f"{cleaned_path}"
)

print(f"Final rows            : {len(cleaned_df):,}")
print(f"Final columns         : {len(cleaned_df.columns)}")
print(
    f"Missing values        : "
    f"{int(cleaned_df.isna().sum().sum())}"
)
print(
    f"Duplicate rows        : "
    f"{int(cleaned_df.duplicated().sum())}"
)


# ============================================================
# 12. CLEANING SUMMARY
# ============================================================

print()
print("=" * 80)
print("12. CLEANING SUMMARY")
print("=" * 80)

summary = pd.DataFrame(
    {
        "Metric": [
            "Initial Rows",
            "Final Rows",
            "Rows Removed",
            "Initial Columns",
            "Final Columns",
            "Missing Values",
            "Duplicate Rows",
            "G3 Zero Records",
            "Absence IQR Candidates",
        ],
        "Value": [
            len(df),
            len(cleaned_df),
            0,
            len(df.columns),
            len(cleaned_df.columns),
            int(cleaned_df.isna().sum().sum()),
            int(cleaned_df.duplicated().sum()),
            zero_count,
            len(absence_outliers),
        ],
    }
)

print(summary.to_string(index=False))

summary.to_csv(
    REPORT_DIR / "cleaning_summary.csv",
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 80)
print("STEP 4 DATA CLEANING & OUTLIER INVESTIGATION COMPLETED")
print("=" * 80)

print()
print("Generated files:")

report_files = [
    "cleaning_range_validation.csv",
    "outlier_analysis.csv",
    "absence_extreme_records.csv",
    "cleaning_summary.csv",
]

for filename in report_files:

    path = REPORT_DIR / filename

    if path.exists():
        print(f"✓ {filename}")
    else:
        print(f"✗ {filename}")

if cleaned_path.exists():
    print("✓ data/student-mat-cleaned.csv")
else:
    print("✗ data/student-mat-cleaned.csv")

print()
print("=" * 80)

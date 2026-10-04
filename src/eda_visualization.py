import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# WEEK 6 - STUDENT PERFORMANCE
# STEP 5: EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
# ============================================================

DATA_PATH = Path("data/student-mat-cleaned.csv")
VIS_DIR = Path("visualizations")
REPORT_DIR = Path("reports")

VIS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")


print("=" * 80)
print("WEEK 6 - STUDENT PERFORMANCE")
print("STEP 5: EXPLORATORY DATA ANALYSIS & VISUALIZATIONS")
print("=" * 80)

print("\nDataset loaded successfully")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 1. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("1. TARGET DISTRIBUTION - G3")
print("=" * 80)

print(f"Mean   : {df['G3'].mean():.3f}")
print(f"Median : {df['G3'].median():.3f}")
print(f"Std    : {df['G3'].std():.3f}")
print(f"Min    : {df['G3'].min():.3f}")
print(f"Max    : {df['G3'].max():.3f}")
print(f"Skew   : {df['G3'].skew():.3f}")

g3_counts = df["G3"].value_counts().sort_index()
print("\nG3 frequency:")
print(g3_counts.to_string())

plt.figure(figsize=(10, 6))
sns.histplot(df["G3"], bins=np.arange(-0.5, 21.5, 1), discrete=True)
plt.axvline(df["G3"].mean(), linestyle="--", label=f"Mean = {df['G3'].mean():.2f}")
plt.axvline(df["G3"].median(), linestyle=":", label=f"Median = {df['G3'].median():.2f}")
plt.title("Distribution of Final Mathematics Grade (G3)")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.savefig(VIS_DIR / "01_g3_distribution.png", dpi=300)
plt.close()


# ============================================================
# 2. G1, G2, G3 COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("2. G1, G2 AND G3 COMPARISON")
print("=" * 80)

grade_summary = df[["G1", "G2", "G3"]].agg(
    ["count", "mean", "std", "min", "median", "max"]
).T

print(grade_summary.round(3).to_string())

grade_summary.to_csv(REPORT_DIR / "grade_comparison_summary.csv")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[["G1", "G2", "G3"]])
plt.title("Comparison of Period Grades: G1, G2 and G3")
plt.xlabel("Grade")
plt.ylabel("Score (0-20)")
plt.tight_layout()
plt.savefig(VIS_DIR / "02_grade_comparison_boxplot.png", dpi=300)
plt.close()


# ============================================================
# 3. ABSENCES DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("3. ABSENCES DISTRIBUTION")
print("=" * 80)

absence_summary = df["absences"].describe(
    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
)

print(absence_summary.round(3).to_string())

print(f"\nStudents with absences > 20: {(df['absences'] > 20).sum()}")
print(
    f"Percentage with absences > 20: "
    f"{(df['absences'] > 20).mean() * 100:.3f}%"
)

plt.figure(figsize=(10, 6))
sns.histplot(df["absences"], bins=30)
plt.axvline(20, linestyle="--", label="IQR upper threshold = 20")
plt.title("Distribution of Student Absences")
plt.xlabel("Number of Absences")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.savefig(VIS_DIR / "03_absences_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df["absences"])
plt.title("Boxplot of Student Absences")
plt.xlabel("Number of Absences")
plt.tight_layout()
plt.savefig(VIS_DIR / "04_absences_boxplot.png", dpi=300)
plt.close()


# ============================================================
# 4. G3 VS G1
# ============================================================

print("\n" + "=" * 80)
print("4. RELATIONSHIP: G3 VS G1")
print("=" * 80)

corr_g1 = df["G1"].corr(df["G3"])
print(f"Pearson correlation (G1, G3): {corr_g1:.4f}")

plt.figure(figsize=(9, 6))
sns.regplot(
    data=df,
    x="G1",
    y="G3",
    scatter_kws={"alpha": 0.55},
    line_kws={}
)
plt.title(f"Final Grade vs First Period Grade (r = {corr_g1:.3f})")
plt.xlabel("G1 - First Period Grade")
plt.ylabel("G3 - Final Grade")
plt.tight_layout()
plt.savefig(VIS_DIR / "05_g3_vs_g1.png", dpi=300)
plt.close()


# ============================================================
# 5. G3 VS G2
# ============================================================

print("\n" + "=" * 80)
print("5. RELATIONSHIP: G3 VS G2")
print("=" * 80)

corr_g2 = df["G2"].corr(df["G3"])
print(f"Pearson correlation (G2, G3): {corr_g2:.4f}")

plt.figure(figsize=(9, 6))
sns.regplot(
    data=df,
    x="G2",
    y="G3",
    scatter_kws={"alpha": 0.55},
    line_kws={}
)
plt.title(f"Final Grade vs Second Period Grade (r = {corr_g2:.3f})")
plt.xlabel("G2 - Second Period Grade")
plt.ylabel("G3 - Final Grade")
plt.tight_layout()
plt.savefig(VIS_DIR / "06_g3_vs_g2.png", dpi=300)
plt.close()


# ============================================================
# 6. G3 VS FAILURES
# ============================================================

print("\n" + "=" * 80)
print("6. FINAL GRADE VS NUMBER OF FAILURES")
print("=" * 80)

failure_summary = (
    df.groupby("failures")["G3"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

print(failure_summary.round(3).to_string(index=False))

failure_summary.to_csv(
    REPORT_DIR / "g3_by_failures_summary.csv",
    index=False
)

plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x="failures", y="G3")
plt.title("Final Grade Distribution by Previous Failures")
plt.xlabel("Number of Previous Failures")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(VIS_DIR / "07_g3_vs_failures.png", dpi=300)
plt.close()


# ============================================================
# 7. G3 VS STUDY TIME
# ============================================================

print("\n" + "=" * 80)
print("7. FINAL GRADE VS STUDY TIME")
print("=" * 80)

studytime_summary = (
    df.groupby("studytime")["G3"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

print(studytime_summary.round(3).to_string(index=False))

studytime_summary.to_csv(
    REPORT_DIR / "g3_by_studytime_summary.csv",
    index=False
)

plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Final Grade Distribution by Weekly Study Time")
plt.xlabel("Study Time Category")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(VIS_DIR / "08_g3_vs_studytime.png", dpi=300)
plt.close()


# ============================================================
# 8. G3 BY GENDER
# ============================================================

print("\n" + "=" * 80)
print("8. FINAL GRADE BY GENDER")
print("=" * 80)

gender_summary = (
    df.groupby("sex")["G3"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

print(gender_summary.round(3).to_string(index=False))

gender_summary.to_csv(
    REPORT_DIR / "g3_by_gender_summary.csv",
    index=False
)

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="sex", y="G3")
plt.title("Final Grade Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(VIS_DIR / "09_g3_by_gender.png", dpi=300)
plt.close()


# ============================================================
# 9. G3 BY SCHOOL
# ============================================================

print("\n" + "=" * 80)
print("9. FINAL GRADE BY SCHOOL")
print("=" * 80)

school_summary = (
    df.groupby("school")["G3"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

print(school_summary.round(3).to_string(index=False))

school_summary.to_csv(
    REPORT_DIR / "g3_by_school_summary.csv",
    index=False
)

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="school", y="G3")
plt.title("Final Grade Distribution by School")
plt.xlabel("School")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(VIS_DIR / "10_g3_by_school.png", dpi=300)
plt.close()


# ============================================================
# 10. G3 BY PARENTAL EDUCATION
# ============================================================

print("\n" + "=" * 80)
print("10. FINAL GRADE VS MOTHER'S EDUCATION")
print("=" * 80)

medu_summary = (
    df.groupby("Medu")["G3"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)

print(medu_summary.round(3).to_string(index=False))

medu_summary.to_csv(
    REPORT_DIR / "g3_by_mother_education.csv",
    index=False
)

plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x="Medu", y="G3")
plt.title("Final Grade by Mother's Education Level")
plt.xlabel("Mother's Education Level")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(VIS_DIR / "11_g3_by_mother_education.png", dpi=300)
plt.close()


# ============================================================
# 11. G3 VS ABSENCES
# ============================================================

print("\n" + "=" * 80)
print("11. RELATIONSHIP: G3 VS ABSENCES")
print("=" * 80)

corr_abs = df["absences"].corr(df["G3"])
print(f"Pearson correlation (absences, G3): {corr_abs:.4f}")

plt.figure(figsize=(9, 6))
sns.regplot(
    data=df,
    x="absences",
    y="G3",
    scatter_kws={"alpha": 0.55},
    line_kws={}
)
plt.title(f"Final Grade vs Absences (r = {corr_abs:.3f})")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(VIS_DIR / "12_g3_vs_absences.png", dpi=300)
plt.close()


# ============================================================
# 12. CORRELATION HEATMAP
# ============================================================

print("\n" + "=" * 80)
print("12. NUMERICAL FEATURE CORRELATION HEATMAP")
print("=" * 80)

numeric_cols = df.select_dtypes(include=np.number).columns
corr_matrix = df[numeric_cols].corr()

target_corr = (
    corr_matrix["G3"]
    .drop("G3")
    .sort_values(key=lambda x: x.abs(), ascending=False)
)

print("\nCorrelation with G3:")
print(target_corr.round(4).to_string())

target_corr.to_csv(
    REPORT_DIR / "eda_target_correlation_ranking.csv",
    header=["correlation"]
)

plt.figure(figsize=(14, 11))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    square=True
)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig(VIS_DIR / "13_correlation_heatmap.png", dpi=300)
plt.close()


# ============================================================
# 13. EXTREME ABSENCE IMPACT
# ============================================================

print("\n" + "=" * 80)
print("13. EXTREME ABSENCE IMPACT")
print("=" * 80)

df["absence_group"] = np.where(
    df["absences"] > 20,
    "Extreme (>20)",
    "Normal (<=20)"
)

absence_group_summary = (
    df.groupby("absence_group", observed=True)["G3"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)

print(absence_group_summary.round(3).to_string(index=False))

absence_group_summary.to_csv(
    REPORT_DIR / "absence_group_impact_summary.csv",
    index=False
)

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="absence_group", y="G3")
plt.title("Final Grade by Absence Group")
plt.xlabel("Absence Group")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(VIS_DIR / "14_absence_group_impact.png", dpi=300)
plt.close()


# ============================================================
# 14. ABSENCE VS G3 - ZOOMED VIEW
# ============================================================

print("\n" + "=" * 80)
print("14. ABSENCE VS G3 - ZOOMED VIEW")
print("=" * 80)

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="absences",
    y="G3",
    hue="absence_group",
    alpha=0.65
)
plt.xlim(-1, 45)
plt.title("Absences vs Final Grade (Zoomed to 45 Absences)")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(VIS_DIR / "15_absences_g3_zoomed.png", dpi=300)
plt.close()


# ============================================================
# 15. SAVE EDA OVERVIEW
# ============================================================

eda_overview = pd.DataFrame(
    {
        "Metric": [
            "Rows",
            "Columns",
            "G3 Mean",
            "G3 Median",
            "G3 Standard Deviation",
            "G3 Minimum",
            "G3 Maximum",
            "G3 Skewness",
            "G1-G3 Correlation",
            "G2-G3 Correlation",
            "Absences-G3 Correlation",
            "G3 Zero Count",
            "G3 Zero Percentage",
            "Absence > 20 Count",
            "Absence > 20 Percentage",
        ],
        "Value": [
            len(df),
            df.shape[1] - 1,
            df["G3"].mean(),
            df["G3"].median(),
            df["G3"].std(),
            df["G3"].min(),
            df["G3"].max(),
            df["G3"].skew(),
            corr_g1,
            corr_g2,
            corr_abs,
            int((df["G3"] == 0).sum()),
            (df["G3"] == 0).mean() * 100,
            int((df["absences"] > 20).sum()),
            (df["absences"] > 20).mean() * 100,
        ],
    }
)

eda_overview.to_csv(
    REPORT_DIR / "eda_overview.csv",
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("STEP 5 EDA COMPLETED")
print("=" * 80)

print(f"\nVisualizations saved in: {VIS_DIR}")
print(f"EDA reports saved in   : {REPORT_DIR}")

print("\nGenerated visualization files:")

for file in sorted(VIS_DIR.glob("*.png")):
    print(f"✓ {file.name}")

print("\nGenerated EDA report files:")

for file in sorted(REPORT_DIR.glob("*.csv")):
    if file.name in {
        "grade_comparison_summary.csv",
        "g3_by_failures_summary.csv",
        "g3_by_studytime_summary.csv",
        "g3_by_gender_summary.csv",
        "g3_by_school_summary.csv",
        "g3_by_mother_education.csv",
        "eda_target_correlation_ranking.csv",
        "absence_group_impact_summary.csv",
        "eda_overview.csv",
    }:
        print(f"✓ {file.name}")

print("\n" + "=" * 80)
print("STEP 5 RUN FINISHED")
print("=" * 80)

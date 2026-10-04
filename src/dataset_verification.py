from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

csv_files = sorted(DATA_DIR.glob("*.csv"))

print()
print("=" * 80)
print("CSV FILES DETECTED")
print("=" * 80)

print(f"Number of CSV files: {len(csv_files)}")

comparison = []

for file in csv_files:

    print()
    print("=" * 80)
    print(f"FILE: {file.name}")
    print("=" * 80)

    # UCI Student Performance files use semicolon separators
    df = pd.read_csv(file, sep=";")

    print()
    print("SHAPE")
    print("-" * 40)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print()
    print("FIRST 5 ROWS")
    print("-" * 40)
    print(df.head().to_string())

    print()
    print("COLUMN NAMES")
    print("-" * 40)

    for number, column in enumerate(df.columns, start=1):
        print(f"{number:2}. {column}")

    print()
    print("DATA TYPES")
    print("-" * 40)
    print(df.dtypes.to_string())

    print()
    print("MISSING VALUES")
    print("-" * 40)
    print(df.isna().sum().to_string())

    total_missing = int(df.isna().sum().sum())

    print()
    print(f"TOTAL MISSING VALUES: {total_missing}")

    duplicate_count = int(df.duplicated().sum())

    print()
    print("DUPLICATE ROWS")
    print("-" * 40)
    print(f"Total duplicate rows: {duplicate_count}")

    print()
    print("UNIQUE VALUES PER COLUMN")
    print("-" * 40)
    print(df.nunique().to_string())

    # --------------------------------------------------------
    # Target analysis
    # --------------------------------------------------------

    if "G3" in df.columns:

        print()
        print("=" * 80)
        print("TARGET VARIABLE: G3")
        print("=" * 80)

        print()
        print("G3 VALUE COUNTS")
        print("-" * 40)
        print(
            df["G3"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print()
        print("G3 DESCRIPTIVE STATISTICS")
        print("-" * 40)
        print(
            df["G3"]
            .describe()
            .to_string()
        )

        print()
        print(f"G3 Mean   : {df['G3'].mean():.3f}")
        print(f"G3 Median : {df['G3'].median():.3f}")
        print(f"G3 Std    : {df['G3'].std():.3f}")
        print(f"G3 Min    : {df['G3'].min()}")
        print(f"G3 Max    : {df['G3'].max()}")

        comparison.append(
            {
                "File": file.name,
                "Rows": df.shape[0],
                "Columns": df.shape[1],
                "Missing_Values": total_missing,
                "Duplicate_Rows": duplicate_count,
                "G3_Mean": round(df["G3"].mean(), 3),
                "G3_Median": round(df["G3"].median(), 3),
                "G3_Std": round(df["G3"].std(), 3),
                "G3_Min": int(df["G3"].min()),
                "G3_Max": int(df["G3"].max()),
            }
        )

# ------------------------------------------------------------
# Dataset comparison
# ------------------------------------------------------------

print()
print("=" * 80)
print("DATASET COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame(comparison)

print()
print(comparison_df.to_string(index=False))

# ------------------------------------------------------------
# Save inventory
# ------------------------------------------------------------

inventory_path = REPORT_DIR / "dataset_inventory.csv"

comparison_df.to_csv(
    inventory_path,
    index=False
)

print()
print("=" * 80)
print("INVENTORY SAVED")
print("=" * 80)

print(inventory_path)

print()
print("=" * 80)
print("DATASET VERIFICATION COMPLETED")
print("=" * 80)

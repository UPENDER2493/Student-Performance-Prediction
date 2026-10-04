# Student Performance Prediction

### Predicting Final Student Mathematics Grades Using Supervised Machine Learning

A complete end-to-end machine learning project that predicts a student's **final mathematics grade (`G3`) on a 0-20 scale** using academic, demographic, social, and behavioral features from the UCI Student Performance dataset.

The project covers the complete machine learning workflow-from dataset verification and exploratory analysis to feature engineering, model comparison, hyperparameter tuning, final evaluation, error analysis, model persistence, and reproducibility.

---

## Project Overview

Student academic performance is influenced by multiple factors such as previous grades, study habits, failures, absences, family background, and social conditions.

This project investigates whether these factors can be used to estimate a student's final mathematics grade.

### Objective

Build and evaluate regression models capable of predicting:

> **Final Mathematics Grade (`G3`) - range: 0 to 20**

The project also compares two prediction scenarios:

- **Full Information:** includes previous-period grades `G1` and `G2`.
- **Pre-Grade:** excludes `G1` and `G2` to evaluate prediction using information available before those grades are known.

---

## Key Results

The final selected model is a **Random Forest Regressor** trained under the Full Information scenario.

| Metric | Final Model |
|---|---:|
| Model | Random Forest Regressor |
| Test MAE | **1.2005** |
| Test MSE | **4.0601** |
| Test RMSE | **2.0150** |
| Test RÂ² | **0.8020** |

The model achieved an **RÂ² of 0.8020**, meaning it explained approximately 80.2% of the variance in final mathematics grades on the held-out test set.

### Scenario Comparison

| Scenario | Model | MAE | RMSE | RÂ² |
|---|---|---:|---:|---:|
| Full Information | Random Forest | **1.2005** | **2.0150** | **0.8020** |
| Pre-Grade | Random Forest | 2.9938 | 3.7508 | 0.3139 |

The comparison demonstrates the strong predictive value of prior academic performance, particularly `G2`.

---

## Dataset

The primary dataset is the **Student Performance dataset**, using the mathematics dataset:

```text
data/student-mat.csv
```

### Dataset Profile

- **395 observations**
- **33 columns**
- **32 predictor/feature columns**
- **1 target column (`G3`)**
- **17 categorical features**
- **16 numerical features**
- **0 missing values**
- **0 duplicate records**

The Portuguese-language dataset is also retained in the repository for reference:

```text
data/student-por.csv
```

A cleaned mathematics dataset is additionally available:

```text
data/student-mat-cleaned.csv
```

---

## Target Variable

The prediction target is:

```text
G3
```

`G3` represents the student's final mathematics grade on a **0-20 scale**.

### Target Statistics

| Statistic | Value |
|---|---:|
| Mean | 10.415 |
| Median | 11 |
| Standard Deviation | 4.581 |
| Minimum | 0 |
| Maximum | 20 |

---

## Machine Learning Workflow

```text
Dataset
   â”‚
   â–¼
Dataset Verification
   â”‚
   â–¼
Data Understanding
   â”‚
   â–¼
Data Cleaning
   â”‚
   â–¼
Exploratory Data Analysis
   â”‚
   â–¼
Feature Engineering
   â”‚
   â–¼
Train/Test Split
   â”‚
   â–¼
Preprocessing Pipeline
   â”‚
   â”œâ”€â”€ Numerical Imputation
   â”œâ”€â”€ Standard Scaling
   â”‚
   â””â”€â”€ Categorical Imputation
       â””â”€â”€ One-Hot Encoding
   â”‚
   â–¼
Baseline Model Comparison
   â”‚
   â”œâ”€â”€ Linear Regression
   â”œâ”€â”€ Ridge Regression
   â”œâ”€â”€ Decision Tree
   â””â”€â”€ Random Forest
   â”‚
   â–¼
Model Evaluation
   â”‚
   â–¼
Hyperparameter Tuning
   â”‚
   â–¼
Final Model Selection
   â”‚
   â–¼
Error Analysis
   â”‚
   â–¼
Model Persistence & Verification
```

---

## Models Evaluated

Four regression algorithms were evaluated:

### 1. Linear Regression

Used as a simple baseline to establish a reference performance level.

### 2. Ridge Regression

An extension of linear regression using L2 regularization to reduce the impact of multicollinearity and overfitting.

### 3. Decision Tree Regressor

A non-linear tree-based model capable of capturing feature interactions.

### 4. Random Forest Regressor

An ensemble of decision trees that generally provides stronger robustness and predictive performance for mixed tabular data.

---

## Baseline Results

### Full Information Scenario

| Model | MAE | RMSE | RÂ² |
|---|---:|---:|---:|
| Linear Regression | 1.6467 | 2.3784 | 0.7241 |
| Ridge Regression | 1.6391 | 2.3715 | 0.7257 |
| Decision Tree | 1.3165 | 2.5705 | 0.6778 |
| **Random Forest** | **1.2005** | **2.0150** | **0.8020** |

Random Forest produced the strongest baseline performance and was therefore selected for further investigation.

---

## Pre-Grade Prediction

To investigate how much predictive power comes from previous grades, a second scenario was created without `G1` and `G2`.

| Model | MAE | RMSE | RÂ² |
|---|---:|---:|---:|
| Linear Regression | 3.3953 | 4.1957 | 0.1415 |
| Ridge Regression | 3.3929 | 4.1934 | 0.1424 |
| Decision Tree | 3.6329 | 4.8379 | -0.1414 |
| **Random Forest** | **2.9938** | **3.7508** | **0.3139** |

This experiment shows that predicting the final grade becomes substantially harder when previous academic grades are unavailable.

---

## Feature Engineering & Preprocessing

The preprocessing pipeline was designed to avoid data leakage by fitting transformations only on the training data.

### Numerical Features

Applied:

- Median imputation
- Standard scaling

### Categorical Features

Applied:

- Most-frequent imputation
- One-hot encoding
- Unknown-category handling

### Train/Test Split

```text
Training set: 316 samples
Testing set:   79 samples
Split ratio:   80/20
Random state:  42
```

---

## Feature Importance

The final Random Forest model identified the following features among the most influential:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `G2` | **0.7872** |
| 2 | `absences` | **0.1143** |
| 3 | `reason_home` | 0.0190 |
| 4 | `age` | 0.0091 |
| 5 | `G1` | 0.0055 |

The dominance of `G2` indicates that a student's previous academic performance is highly informative for predicting the final grade.

Feature importance indicates predictive contribution within the trained model; it **does not establish causation**.

---

## Error Analysis

The final model achieved strong overall performance, but its errors were not evenly distributed across all grade ranges.

### Mean Absolute Error by Grade Range

| Actual Grade Range | Samples | MAE |
|---|---:|---:|
| 0-5 | 9 | 3.0278 |
| 6-10 | 29 | 1.3652 |
| 11-15 | 31 | 0.6668 |
| 16-20 | 10 | 0.7330 |

The model performed best for students in the middle and upper grade ranges and struggled more with very low grades.

This is an important limitation because students with unusually low outcomes can be particularly difficult to predict from the available features.

---

## Hyperparameter Tuning

Random Forest hyperparameters were optimized using:

- `GridSearchCV`
- 5-fold cross-validation
- RMSE-based scoring
- Shuffled K-Fold validation
- `random_state=42`

A total of:

```text
216 parameter combinations Ã— 5 folds
= 1080 fits per scenario
```

were evaluated.

### Tuning Result

Interestingly, hyperparameter tuning did **not** improve the baseline Random Forest on the held-out test set.

Therefore, the simpler baseline configuration was retained as the final model rather than selecting a more complex configuration solely because it had been tuned.

This decision prioritizes **measured validation performance and reproducibility** over unnecessary model complexity.

---

## Final Model

The final model is:

```text
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

Saved model:

```text
models/student_performance_final_model.joblib
```

Metadata:

```text
models/student_performance_model_metadata.json
```

The saved model was reloaded and tested to verify reproducibility.

### Reload Verification

```text
Predictions identical after reload: True
```

---

## Visual Analysis

The project contains 12 final visualizations covering target distribution, relationships, correlations, feature importance, predictions, residuals, and error behavior.

### Target & Grade Analysis

| Visualization | Description |
|---|---|
| `G3_Distribution.png` | Distribution of final mathematics grades |
| `Grade_Comparison.png` | Comparison of G1, G2, and G3 |
| `G3_vs_G1.png` | Relationship between first-period and final grades |
| `G3_vs_G2.png` | Relationship between second-period and final grades |

### Behavioral & Academic Factors

| Visualization | Description |
|---|---|
| `G3_vs_Failures.png` | Final grade by number of previous failures |
| `G3_vs_Studytime.png` | Final grade by weekly study time |
| `G3_vs_Absences.png` | Relationship between absences and final grade |
| `Correlation_Heatmap.png` | Feature correlation overview |

### Model Analysis

| Visualization | Description |
|---|---|
| `Feature_Importance.png` | Final Random Forest feature importance |
| `Actual_vs_Predicted.png` | Actual versus predicted grades |
| `Residual_Analysis.png` | Distribution and behavior of prediction errors |
| `Error_by_Grade_Range.png` | Model error across grade ranges |

---

## Project Structure

```text
Student-Performance-Prediction/
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ student-mat.csv
â”‚   â”œâ”€â”€ student-por.csv
â”‚   â””â”€â”€ student-mat-cleaned.csv
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ student_performance_final_model.joblib
â”‚   â””â”€â”€ student_performance_model_metadata.json
â”‚
â”œâ”€â”€ notebook/
â”‚   â””â”€â”€ Week_6_Student_Performance.ipynb
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ dataset_verification.py
â”‚   â”œâ”€â”€ data_understanding.py
â”‚   â”œâ”€â”€ data_cleaning.py
â”‚   â”œâ”€â”€ eda_visualization.py
â”‚   â”œâ”€â”€ feature_engineering.py
â”‚   â”œâ”€â”€ baseline_models.py
â”‚   â”œâ”€â”€ model_evaluation.py
â”‚   â”œâ”€â”€ model_tuning.py
â”‚   â”œâ”€â”€ final_model.py
â”‚   â””â”€â”€ final_analysis.py
â”‚
â”œâ”€â”€ visualizations/
â”‚   â”œâ”€â”€ Actual_vs_Predicted.png
â”‚   â”œâ”€â”€ Correlation_Heatmap.png
â”‚   â”œâ”€â”€ Error_by_Grade_Range.png
â”‚   â”œâ”€â”€ Feature_Importance.png
â”‚   â”œâ”€â”€ G3_Distribution.png
â”‚   â”œâ”€â”€ G3_vs_Absences.png
â”‚   â”œâ”€â”€ G3_vs_Failures.png
â”‚   â”œâ”€â”€ G3_vs_G1.png
â”‚   â”œâ”€â”€ G3_vs_G2.png
â”‚   â”œâ”€â”€ G3_vs_Studytime.png
â”‚   â”œâ”€â”€ Grade_Comparison.png
â”‚   â””â”€â”€ Residual_Analysis.png
â”‚
â”œâ”€â”€ .gitignore
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â””â”€â”€ Week_6_Student_Performance_Final_Report.pdf
```

---

## Tech Stack

### Programming

- Python 3.14

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Model Persistence

- Joblib

### Documentation

- Jupyter Notebook
- PDF Report

### Development Environment

- Visual Studio Code
- Python Virtual Environment (`.venv`)
- Git & GitHub

---

## Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/UPENDER2493/Student-Performance-Prediction.git
cd Student-Performance-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The project is organized as a sequence of reproducible Python scripts.

Run the workflow from the project root:

```bash
python src/dataset_verification.py
python src/data_understanding.py
python src/data_cleaning.py
python src/eda_visualization.py
python src/feature_engineering.py
python src/baseline_models.py
python src/model_evaluation.py
python src/model_tuning.py
python src/final_model.py
python src/final_analysis.py
```

The executed notebook is also available at:

```text
notebook/Week_6_Student_Performance.ipynb
```

For the most reliable relative-path behavior, run notebook-related work from the project root.

---

## Reproducibility

The project uses fixed random seeds where applicable:

```text
random_state = 42
```

The workflow includes:

- Explicit train/test split
- Training-only preprocessing
- Cross-validation
- Reproducible model configuration
- Saved model artifact
- Saved metadata
- Executed notebook
- Final visualizations
- Final evaluation results

This makes the project easier to reproduce and audit.

---

## Limitations

Although the final model achieved strong performance, several limitations remain:

1. The dataset contains only 395 mathematics-student records.
2. The dataset represents a specific educational context and may not generalize to all schools or populations.
3. `G2` is highly predictive of `G3`, so the Full Information scenario benefits strongly from prior academic performance.
4. The Pre-Grade scenario demonstrates substantially weaker predictive performance.
5. Feature importance should not be interpreted as causal influence.
6. Prediction errors are larger for some low-performing students.
7. The model should be treated as a predictive analysis tool rather than a definitive assessment of student ability.

---

## Future Improvements

Potential extensions include:

- External validation using additional student datasets
- Comparison with Gradient Boosting, XGBoost, or other advanced ensemble models
- More systematic feature selection
- Explainability using SHAP or similar techniques
- Prediction intervals and uncertainty estimation
- Separate models for different academic groups
- Additional temporal or longitudinal student data
- Deployment through an API or web application
- Monitoring model performance after deployment
- Fairness and subgroup performance analysis

---

## Key Takeaways

### 01 - Previous performance matters

`G2` was by far the strongest predictive feature in the final Random Forest model.

### 02 - Random Forest performed best

Among the evaluated baseline models, Random Forest achieved the strongest overall test performance.

### 03 - Prediction without previous grades is harder

Removing `G1` and `G2` reduced the Random Forest RÂ² from:

```text
0.8020 â†’ 0.3139
```

### 04 - Strong overall performance does not mean perfect predictions

The model performed substantially better for middle-range grades than for some very low-grade cases.

### 05 - Evaluation matters more than tuning for its own sake

Hyperparameter tuning did not improve the final held-out performance, so the baseline Random Forest was retained.

---

## Academic Project

**Project:** Student Performance Prediction Using Supervised Machine Learning

**Target:** Final Mathematics Grade (`G3`)

**Problem Type:** Regression

**Final Model:** Random Forest Regressor

**Best Test RÂ²:** 0.8020

**Best Test RMSE:** 2.0150

**Best Test MAE:** 1.2005

---

## Author

**Upender Rajput**

B.Tech Student | AI/ML & Software Development

GitHub: **UPENDER2493**

---

## License

This project is intended primarily for **educational, academic, and portfolio purposes**.

Please refer to the original dataset source and its applicable terms before using the dataset for other purposes.
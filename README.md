# Student Performance Prediction Using Supervised Machine Learning

A supervised machine learning project for predicting students' final mathematics grade (`G3`) using the UCI Student Performance Dataset.

## Project Overview

This project implements a complete supervised machine learning workflow for predicting a student's final mathematics performance.

- **Target:** `G3` â€” Final mathematics grade
- **Target Range:** 0â€“20
- **Problem Type:** Regression
- **Primary Dataset:** UCI Student Performance Dataset
- **Primary File:** `data/student-mat.csv`

The project covers data verification, data cleaning, exploratory data analysis, preprocessing, baseline model development, model evaluation, hyperparameter tuning, final model selection, feature importance analysis, and prediction error analysis.

---

## Dataset

The project uses the UCI Student Performance Dataset.

### Mathematics Dataset

**File:** `data/student-mat.csv`

- 395 student records
- 33 columns
- 32 predictor features
- 1 target variable (`G3`)
- 0 missing values
- 0 duplicate rows

### Portuguese Dataset

**File:** `data/student-por.csv`

The Portuguese-language dataset is retained as a reference/alternative dataset. It was not used for the final mathematics prediction model.

### Cleaned Dataset

**File:** `data/student-mat-cleaned.csv`

This is the validated mathematics dataset used during the modeling workflow.

---

## Target Variable

The target variable is `G3`, representing the student's final mathematics grade.

The target has a range from 0 to 20.

| Statistic | G3 |
|---|---:|
| Mean | 10.415 |
| Standard Deviation | 4.581 |
| Minimum | 0 |
| Median | 11 |
| Maximum | 20 |

---

## Project Objectives

1. Understand the dataset and target variable.
2. Validate data quality.
3. Perform data cleaning and validation.
4. Perform exploratory data analysis.
5. Study relationships between student characteristics and final grades.
6. Build multiple supervised regression models.
7. Compare model performance using MAE, MSE, RMSE and RÂ².
8. Compare models with and without previous grades.
9. Perform hyperparameter tuning.
10. Select the strongest final model.
11. Analyze feature importance.
12. Analyze prediction errors.
13. Document limitations and future improvements.

---

## Machine Learning Workflow

```text
Dataset Selection
       â†“
Data Verification
       â†“
Data Understanding
       â†“
Data Cleaning
       â†“
Exploratory Data Analysis
       â†“
Feature Engineering & Preprocessing
       â†“
Train/Test Split
       â†“
Baseline Models
       â†“
Model Evaluation
       â†“
Hyperparameter Tuning
       â†“
Final Model Selection
       â†“
Feature Importance
       â†“
Error Analysis
       â†“
Final Report
```

---

## Data Quality

The primary mathematics dataset contains:

- 395 rows
- 33 columns
- 0 missing values
- 0 duplicate rows
- No logical range violations identified

The dataset was retained without removing legitimate high-absence observations because extreme absence values were not proven to be invalid.

---

## Exploratory Data Analysis

The analysis examined:

- Final grade distribution
- Previous grades (`G1` and `G2`)
- Failures
- Study time
- Absences
- Gender
- School
- Mother's education
- Feature correlations
- Relationships between numerical variables and `G3`

Important observed relationships included:

- `G2` had the strongest correlation with `G3`.
- `G1` was also strongly correlated with `G3`.
- Previous failures showed a negative relationship with final grade.
- Absences had a very weak linear correlation with `G3`.

---

## Models Evaluated

The following regression models were evaluated:

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

Two modeling scenarios were compared.

### Full Information Scenario

This scenario included all predictor variables, including previous grades `G1` and `G2`.

### Pre-Grade Scenario

This scenario excluded `G1` and `G2` to evaluate prediction using student information available before the earlier grading stages.

---

## Baseline Model Results

### Full Information Scenario

| Model | MAE | MSE | RMSE | RÂ² |
|---|---:|---:|---:|---:|
| Linear Regression | 1.6467 | 5.6566 | 2.3784 | 0.7241 |
| Ridge Regression | 1.6391 | 5.6242 | 2.3715 | 0.7257 |
| Decision Tree | 1.3165 | 6.6076 | 2.5705 | 0.6778 |
| **Random Forest** | **1.2005** | **4.0601** | **2.0150** | **0.8020** |

### Pre-Grade Scenario

| Model | MAE | MSE | RMSE | RÂ² |
|---|---:|---:|---:|---:|
| Linear Regression | 3.3953 | 17.6037 | 4.1957 | 0.1415 |
| Ridge Regression | 3.3929 | 17.5846 | 4.1934 | 0.1424 |
| Decision Tree | 3.6329 | 23.4051 | 4.8379 | -0.1414 |
| **Random Forest** | **2.9938** | **14.0686** | **3.7508** | **0.3139** |

Random Forest achieved the strongest baseline performance in both scenarios.

---

## Final Model

The final selected model is:

**Random Forest Regressor**

Configuration:

```text
n_estimators = 200
random_state = 42
n_jobs = -1
```

The final model was selected based on the baseline comparison because hyperparameter tuning did not improve the test performance.

---

## Final Model Performance

The final model achieved:

| Metric | Test Performance |
|---|---:|
| MAE | **1.2005** |
| MSE | **4.0601** |
| RMSE | **2.0150** |
| RÂ² | **0.8020** |

An RMSE of approximately 2.02 means that predictions typically differ from actual final grades by around two grade points, with larger errors contributing more strongly to the RMSE.

---

## Hyperparameter Tuning

Random Forest hyperparameters were optimized using 5-fold cross-validation with RMSE as the optimization metric.

### Best Full Information Parameters

```text
n_estimators = 400
max_depth = None
max_features = 1.0
min_samples_leaf = 2
min_samples_split = 2
```

Tuned test performance:

- MAE: 1.2123
- MSE: 4.0787
- RMSE: 2.0196
- RÂ²: 0.8011

The tuned model did not improve the baseline Random Forest, so the baseline model was retained as the final model.

---

## Feature Importance

The strongest features in the final model included:

| Feature | Importance |
|---|---:|
| `G2` | 0.7872 |
| `absences` | 0.1143 |
| `reason_home` | 0.0190 |
| `age` | 0.0091 |
| `G1` | 0.0055 |

The very high importance of `G2` indicates that previous performance is highly predictive of the final grade.

**Note:** Feature importance describes predictive contribution and should not be interpreted as proof of causation.

---

## Error Analysis

The final model was particularly accurate for students whose actual grades were in the middle and upper ranges.

The largest prediction errors were concentrated among several low-grade observations.

For example, some students with an actual `G3` of 0 were predicted substantially higher than their actual grade.

This indicates that the model has difficulty identifying some extreme low-performance cases.

---

## Visualizations

The project includes 12 final visualizations:

1. `G3_Distribution.png`
2. `Grade_Comparison.png`
3. `G3_vs_G1.png`
4. `G3_vs_G2.png`
5. `G3_vs_Failures.png`
6. `G3_vs_Studytime.png`
7. `G3_vs_Absences.png`
8. `Correlation_Heatmap.png`
9. `Feature_Importance.png`
10. `Actual_vs_Predicted.png`
11. `Residual_Analysis.png`
12. `Error_by_Grade_Range.png`

---

## Project Structure

```text
Week_6_Student_Performance/
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ student-mat.csv
â”‚   â”œâ”€â”€ student-por.csv
â”‚   â””â”€â”€ student-mat-cleaned.csv
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
â”‚   â””â”€â”€ 12 final visualization files
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ student_performance_final_model.joblib
â”‚   â””â”€â”€ student_performance_model_metadata.json
â”‚
â”‚
â”œâ”€â”€ .gitignore
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
├── notebook/
│   └── Week_6_Student_Performance.ipynb
â””â”€â”€ Week_6_Student_Performance_Final_Report.pdf
```

---

## Technologies Used

- Python 3.14
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- Python-docx
- OpenPyXL

---

## Preprocessing

Numerical features were processed using:

- Median imputation
- StandardScaler

Categorical features were processed using:

- Most-frequent imputation
- One-hot encoding
- Unknown-category handling

The preprocessing pipeline was fitted using the training data only to avoid data leakage.

---

## Train/Test Split

The dataset was divided using:

```text
Training data: 80%
Test data: 20%
random_state: 42
```

This resulted in:

- Training samples: 316
- Test samples: 79

---

## Reproducibility

The project uses fixed random seeds where applicable, particularly:

```text
random_state = 42
```

The final trained model is stored in:

```text
models/student_performance_final_model.joblib
```

Model metadata is stored in:

```text
models/student_performance_model_metadata.json
```

---

## Running the Project

After activating the virtual environment and installing the requirements, the individual workflow scripts can be executed from the project root.

```powershell
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

The scripts reproduce the major stages of the machine learning workflow.

> **Note:** Some workflow scripts generate intermediate analysis files. The final submission retains the curated final visualizations, trained model artifacts, notebook, and final PDF report.

---

## Limitations

1. The dataset contains only 395 mathematics students.
2. The project focuses on predicting mathematics performance.
3. The model is based on historical student information.
4. Previous grades (`G1` and `G2`) are highly predictive, which limits the usefulness of the full-information model for early intervention.
5. Some extreme low-grade cases are difficult for the model to predict accurately.
6. Feature importance describes predictive contribution, not causal relationships.
7. External factors not represented in the dataset may affect student performance.

---

## Future Improvements

Possible future improvements include:

- Testing additional regression algorithms such as Gradient Boosting and XGBoost.
- Applying systematic feature selection.
- Testing ensemble stacking and boosting methods.
- Evaluating the model using repeated cross-validation.
- Developing an early-warning model without previous grades.
- Investigating additional student-level features.
- Improving prediction of extreme low-grade cases.
- Deploying the model through a web-based educational application.

---

## Final Conclusion

The project demonstrates a complete supervised machine learning workflow for student performance prediction.

The Random Forest Regressor achieved the best baseline performance with:

**MAE = 1.2005**

**RMSE = 2.0150**

**RÂ² = 0.8020**

The analysis shows that previous academic performance, particularly `G2`, is the strongest predictive signal for final mathematics grade. However, the model performs less reliably on extreme low-grade cases, highlighting the importance of detailed error analysis alongside aggregate performance metrics.

---

## Academic Context

This project was developed as part of a supervised machine learning coursework/capstone workflow focused on understanding the complete machine learning lifecycle from dataset selection through final model evaluation.

---

## License

This project is intended for educational and academic purposes.

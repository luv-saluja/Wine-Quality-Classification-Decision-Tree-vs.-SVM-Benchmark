# Wine Quality Classification: Decision Tree vs. SVM

A comparative study of **Decision Tree** and **Support Vector Machine (SVM)** classifiers for predicting wine quality, using the UCI White Wine Quality dataset.

## Overview

The original dataset scores wine quality on a scale of 3–9, which results in a highly imbalanced multi-class problem. This project reframes it as a **binary classification** task — a wine is labeled **"good" (1)** if its quality score is ≥ 7, and **"not good" (0)** otherwise — then compares how a Decision Tree and an SVM perform on this task.

## Dataset

- **Source:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) (`winequality-white.csv`)
- **Features:** 11 physicochemical properties (e.g. acidity, residual sugar, chlorides, density, alcohol, etc.)
- **Target:** `quality`, binarized to 1 (score ≥ 7) or 0 (score < 7)

> **Note:** The notebook reads the file from a local Windows path (`D:\winequality-white.csv`). Update this path to point to your own local copy of the dataset before running.

## Workflow

1. **Exploratory Data Analysis (EDA)**
   - Automated profiling report via `ydata_profiling`
   - Null/duplicate checks
   - Class distribution and correlation analysis
   - Bivariate analysis (violin plots of each feature vs. quality)

2. **Preprocessing**
   - Binarization of the target variable
   - Train/test split (80/20, stratified)
   - `SMOTE` oversampling applied within the training folds only (via `imblearn.pipeline.Pipeline`) to address class imbalance

3. **Model Training & Tuning**
   - **Decision Tree** — hyperparameters tuned with `GridSearchCV` (criterion, max depth, min samples split/leaf, ccp_alpha, class weight) over 5-fold stratified CV, optimizing weighted F1
   - **SVM (RBF kernel)** — features scaled with `StandardScaler`, hyperparameters (`C`, `gamma`) tuned with `GridSearchCV`, also optimizing weighted F1

4. **Evaluation**
   - Classification reports (precision, recall, F1)
   - Confusion matrices
   - ROC curve and AUC (Decision Tree)
   - Feature importance: Gini importance vs. permutation importance
   - Custom decision-threshold tuning via cross-validation to maximize F1 on the minority ("good wine") class

## Results

| Model | Best CV Weighted F1 | Test Accuracy | Class 1 (Good) Precision | Class 1 (Good) Recall | Class 1 (Good) F1 |
|---|---|---|---|---|---|
| Decision Tree | 0.817 | 0.83 | 0.58 | 0.74 | 0.65 |
| SVM (RBF) | 0.825 | 0.84 | 0.63 | 0.70 | 0.66 |

Key observations from the EDA:
- Higher-quality wines tend to have **lower density** and **higher alcohol content**.
- The dataset is imbalanced, motivating the binary reframing and use of SMOTE.
- Both models perform similarly overall, with the SVM showing a modest edge in precision on the minority class and the Decision Tree showing higher recall.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
ydata-profiling
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn ydata-profiling
```

## Usage

1. Download the [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) (`winequality-white.csv`).
2. Update the file path in the notebook's data-loading cell to point to your local copy.
3. Run the notebook cells in order: `Comparative_Study_of_DT_and_SVM.ipynb`.

## Project Structure

```
.
├── Comparative_Study_of_DT_and_SVM.ipynb   # Main analysis notebook
├── EDA_output.html                          # Generated automated EDA report (created on run)
└── README.md
```

## License

Add a license of your choice (e.g. MIT) if you plan to share this repository publicly.

# Iris Flower Classification using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-EDA-purple)](https://pandas.pydata.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Provided%20CSV-teal)](data/Iris.csv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete, internship-ready machine learning project that classifies Iris
flowers using the **official CSV dataset provided for the CodeAlpha Data Science
Internship**.

This project intentionally uses only [data/Iris.csv](data/Iris.csv). It does
not download data and does not substitute Scikit-learn's built-in Iris dataset.

## Project Highlights

- Loads the official provided dataset from `data/Iris.csv`.
- Automatically identifies column names and data types.
- Reports dataset shape, missing values, duplicate rows, and summary statistics.
- Detects the target column from the actual dataset structure.
- Removes identifier-like columns, such as `Id`, from model features.
- Removes duplicate rows if present.
- Handles missing feature values with a Scikit-learn preprocessing pipeline.
- Generates EDA visualizations based on the actual columns in the CSV.
- Splits the dataset into 80% training data and 20% testing data.
- Trains Logistic Regression, Decision Tree, and Random Forest classifiers.
- Compares model accuracies and selects the best-performing model.
- Displays accuracy score, confusion matrix, and classification report.
- Predicts the class of a new sample entered in the code.
- Saves the best model artifact using Joblib.

## Project Structure

```text
Iris-flower-classification/
├── data/
│   └── Iris.csv
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── models/
│   ├── .gitkeep
│   └── best_iris_model.joblib
├── outputs/
│   ├── sample_output.txt
│   └── figures/
│       ├── .gitkeep
│       ├── class_distribution.png
│       ├── correlation_heatmap.png
│       └── feature_distributions.png
└── screenshots/
    └── README.md
```

## Dataset Overview

The provided CSV contains 150 rows and 6 columns:

| Column | Type | Role |
| --- | --- | --- |
| `Id` | Integer | Identifier column, excluded from training |
| `SepalLengthCm` | Float | Numeric feature |
| `SepalWidthCm` | Float | Numeric feature |
| `PetalLengthCm` | Float | Numeric feature |
| `PetalWidthCm` | Float | Numeric feature |
| `Species` | Text | Target variable |

The target classes are:

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

## Automatic Decisions Made by the Code

The project is designed to adapt to the dataset instead of assuming fixed
column positions.

| Decision | How it works |
| --- | --- |
| Target selection | Looks for common target names such as `species`, `target`, `class`, or `label`; for this dataset it selects `Species`. |
| ID removal | Removes identifier-like columns that are unique row identifiers; for this dataset it removes `Id`. |
| Feature selection | Uses all non-target, non-ID columns as model features. |
| Missing values | Numeric features use median imputation; categorical features use most-frequent imputation. |
| Categorical features | One-hot encoded when present. |
| Numeric features | Standardized after imputation so Logistic Regression trains reliably. |
| Duplicate rows | Removed before training if found. |
| Train/test split | Uses an 80/20 split and stratifies when every class has enough samples. |

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/Iris-flower-classification.git
cd Iris-flower-classification
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Windows note: if `python` points to an interpreter without `pip`, use a
specific launcher command such as:

```bash
py -3.13 -m pip install -r requirements.txt
py -3.13 main.py
```

## Usage

Run the complete machine learning pipeline:

```bash
python main.py
```

The script will:

1. Load `data/Iris.csv`.
2. Display dataset shape, data types, missing values, duplicate count, and
   summary statistics.
3. Detect the target variable and feature columns.
4. Clean text values, remove duplicates, and remove rows with missing target
   values.
5. Build preprocessing steps based on detected numeric and categorical columns.
6. Save EDA visualizations.
7. Train and compare three classification models.
8. Evaluate the best model.
9. Predict a new flower sample.
10. Save the best model with Joblib.

## Sample Output

```text
Target column: Species
Feature columns selected for modeling:
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm

Columns removed from features because they look like IDs:
- Id

Model Training and Accuracy Comparison
======================================
                   Model  Accuracy
     Logistic Regression  0.933333
Decision Tree Classifier  0.900000
Random Forest Classifier  0.900000

Best Model Evaluation
=====================
Best model: Logistic Regression
Accuracy score: 0.9333

Predicted Species: Iris-versicolor

Model Saved
===========
Saved best model artifact: models/best_iris_model.joblib
```

For a fuller run example, see [outputs/sample_output.txt](outputs/sample_output.txt).

## Visualizations

Running `main.py` generates:

- `outputs/figures/class_distribution.png`
- `outputs/figures/feature_distributions.png`
- `outputs/figures/correlation_heatmap.png`

Use the `screenshots/` folder for GitHub, LinkedIn, or internship submission
screenshots.

## How to Load the Saved Model

```python
import joblib
import pandas as pd

artifact = joblib.load("models/best_iris_model.joblib")
model = artifact["model"]
feature_columns = artifact["feature_columns"]

sample = pd.DataFrame(
    [[5.7, 2.9, 4.2, 1.3]],
    columns=feature_columns,
)

prediction = model.predict(sample)
print(prediction[0])
```

## LinkedIn Project Description

I completed an Iris Flower Classification project as part of my CodeAlpha Data
Science Internship. I used the official provided CSV dataset and built an
end-to-end machine learning pipeline in Python with Pandas, NumPy, Matplotlib,
Seaborn, and Scikit-learn.

The project automatically inspects the dataset, identifies feature and target
columns, removes identifier columns, handles missing values, performs EDA,
generates visualizations, trains multiple classification models, compares model
accuracy, evaluates the best model, predicts a new sample, and saves the final
model using Joblib.

This project strengthened my understanding of supervised classification,
preprocessing pipelines, exploratory data analysis, model evaluation, and
GitHub-ready project documentation.

## Resume Bullet Points

- Built an adaptive Iris flower classification pipeline using Python,
  Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn on the official provided
  internship dataset.
- Automated dataset inspection, target detection, ID-column removal, duplicate
  handling, missing-value preprocessing, EDA visualizations, and model training.
- Trained and compared Logistic Regression, Decision Tree, and Random Forest
  classifiers using an 80/20 stratified train-test split.
- Evaluated the best model with accuracy score, confusion matrix, and
  classification report, then saved the trained pipeline with Joblib.

## Suggested Commit Messages

```text
Initial commit: add official Iris dataset classification project
Add adaptive preprocessing and EDA pipeline
Add model comparison, evaluation, prediction, and Joblib export
Polish README with usage, sample output, and portfolio documentation
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.

# House-Prices-Advanced-Regression-Techniques

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Overview

This repository contains an exploratory data analysis (EDA) notebook for the **Ames Housing dataset**. The goal is to understand the characteristics of the data, identify key patterns, and prepare it for potential use in a machine learning model to predict house prices (`SalePrice`).

The notebook performs a thorough analysis, including data loading, structural inspection, missing value assessment, target variable analysis, and feature type identification. The insights gained from this EDA provide a solid foundation for feature engineering and model selection.

## 📂 Repository Structure

```
Predicting-Electric-Vehicle-Purchases/
├── dataset/
│   ├── train.csv                
│   ├── test.csv                 
│   └── sample_submission.csv 
├── plots/                       # EDA figures (distributions, KDEs, signal charts)
├── outputs/
├── 01_EDA.ipynb
├── train_classifier.py
└── submission.csv
```

## 📊 Dataset

The project uses the Ames Housing dataset, a popular alternative to the Boston Housing dataset for regression tasks. It contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, and is split into two files:

*   **`train.csv`**: The training set, which includes the target variable `SalePrice`.
*   **`test.csv`**: The test set, used for making predictions.
*   **`sample_submission.csv`**: A sample file showing the expected submission format.

The dataset is not included in this repository. Please download it from the [Kaggle House Prices - Advanced Regression Techniques competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place the files in a `./dataset/` directory.

### Attribute Information

The dataset contains a mix of numerical and categorical features:

| Type | Count | Examples |
| :--- | :--- | :--- |
| **Numerical Features** | 36 | `LotArea`, `OverallQual`, `YearBuilt`, `GrLivArea`, `GarageArea` |
| **Categorical/Text Features** | 43 | `MSZoning`, `Neighborhood`, `BldgType`, `KitchenQual`, `SaleCondition` |
| **Identifier** | 1 | `Id` |
| **Target Variable** | 1 | `SalePrice` (the property's sale price in dollars) |

## 🔍 Key Findings from the Analysis

The notebook `analyze_dataset.ipynb` provides a detailed EDA. Some of the most important findings are:

### 1. Data Shape and Structure
*   **Training Set:** 1460 rows and 81 columns.
*   **Test Set:** 1459 rows and 80 columns (missing the `SalePrice` column).
*   The data consists of a mix of 3 `float64`, 35 `int64`, and 43 `object` (string/categorical) data types.

### 2. Target Variable: `SalePrice`
The target variable is continuous and shows a **right-skewed distribution**.
*   **Mean:** $180,921
*   **Median:** $163,000
*   **Range:** $34,900 – $755,000
*   **Skewness:** 1.88

To improve normality for modeling, a **logarithmic transformation (`log1p`)** is applied, which significantly reduces skewness and makes the distribution more symmetric and closer to a normal distribution.

### 3. Missing Values
Missing values are present in both the training and test sets. The most affected columns are:
*   `PoolQC` (over 99% missing)
*   `MiscFeature` (over 96% missing)
*   `Alley` (over 93% missing)
*   `Fence` (over 80% missing)

Many of these likely indicate the absence of a feature (e.g., no pool, no alley, no fence), which is valuable information for feature engineering.

### 4. Train vs. Test Consistency
An analysis of the categorical features shows that some categories appear only in the training set but not in the test set (e.g., `Utilities_NoSeWa`, `Condition2_RRNn`). This is important to note, as some encoding methods might break if they encounter categories in the test set that were not present during training.

## 🛠️ Technologies Used

*   **Python 3.9+**
*   **Jupyter Notebook** for interactive data analysis.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Matplotlib & Seaborn:** For creating informative visualizations.

## 🚀 Getting Started

To run this analysis locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ms20237/House-Prices-Advanced-Regression-Techniques.git
    cd House-Prices-Advanced-Regression-Techniques
    ```

2.  **Install the required dependencies:**
    It is recommended to create a virtual environment.
    ```bash
    pip install pandas numpy matplotlib seaborn jupyter
    ```

3.  **Download the data:**
    Download the dataset files (`train.csv`, `test.csv`, `sample_submission.csv`) from Kaggle and place them into a new directory named `dataset/` in the project root.

    Your project structure should look like this:
    ```
    .
    ├── dataset/
    │   ├── train.csv
    │   ├── test.csv
    │   └── sample_submission.csv
    ├── plots/  (will be created automatically)
    ├── analyze_dataset.ipynb
    ├── train_classifier.py
    └── README.md
    ```

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook analyze_dataset.ipynb
    ```

## 🔮 Next Steps

This EDA provides the groundwork for a predictive modeling pipeline. Potential next steps include:

*   **Feature Engineering:** Create new features from existing ones (e.g., `HouseAge = YrSold - YearBuilt`, `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`).
*   **Imputation:** Develop a strategy to handle missing values based on the insights from this analysis.
*   **Encoding Categoricals:** Convert categorical features into numerical format using One-Hot Encoding or Ordinal Encoding.
*   **Modeling:** Train and evaluate various regression models (e.g., Linear Regression, Ridge, Lasso, XGBoost) on the prepared data.

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
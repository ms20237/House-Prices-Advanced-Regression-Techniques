"""
House Prices: Advanced Regression Techniques
==============================================
End-to-end pipeline: load -> explore -> clean -> feature engineer ->
encode -> train/compare models -> pick best -> predict -> submission.csv

Usage:
    Place train.csv and test.csv in the same folder as this script
    (or edit TRAIN_PATH / TEST_PATH below), then run:

        python house_prices.py

Output:
    outputs/submission.csv
    Console output with CV scores for each model and feature importances.
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# CONFIG
TRAIN_PATH = "./dataset/train.csv"
TEST_PATH = "./dataset/test.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42

# Columns where "NA" in the raw data is a MEANINGFUL CATEGORY (e.g. "no garage"), not a missing value. 
NONE_MEANS_ABSENT_CATEGORICAL = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
    "MasVnrType",
]

# Numeric columns where missing really means "0" (e.g. no basement -> 0 sqft)
NONE_MEANS_ZERO_NUMERIC = [
    "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# LOAD
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    return train, test


# EXPLORE (prints summary stats)
def explore(train):
    print("\n--- SalePrice summary ---")
    print(train["SalePrice"].describe())

    print("\n--- SalePrice skew ---")
    print(f"Raw skew:  {train['SalePrice'].skew():.3f}")
    print(f"Log skew:  {np.log1p(train['SalePrice']).skew():.3f}")

    print("\n--- Top 15 columns by missing value count (train) ---")
    missing = train.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(15))

    print("\n--- Top 10 numeric correlations with SalePrice ---")
    numeric = train.select_dtypes(include=[np.number])
    corr = numeric.corr()["SalePrice"].sort_values(ascending=False)
    print(corr.iloc[1:11])  # skip SalePrice itself


# CLEAN MISSING VALUES (domain-aware, before generic imputation)
def fill_domain_missing(df):
    df = df.copy()

    for col in NONE_MEANS_ABSENT_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in NONE_MEANS_ZERO_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage: impute by neighborhood median (houses in the same
    # neighborhood tend to have similar frontage) rather than a global median
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Electrical has 1 missing value in train -> fill with mode
    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    return df


# FEATURE ENGINEERING
def add_features(df):
    df = df.copy()

    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    df["TotalPorchSF"] = (
        df["OpenPorchSF"].fillna(0)
        + df["3SsnPorch"].fillna(0)
        + df["EnclosedPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
        + df["WoodDeckSF"].fillna(0)
    )

    df["TotalBsmtFinSF"] = df["BsmtFinSF1"].fillna(0) + df["BsmtFinSF2"].fillna(0)

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)
    df["GarageAge"] = df["GarageAge"].fillna(0)

    df["OverallQual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]

    df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"].fillna(0) > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)
    df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"].fillna(0) > 0).astype(int)
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)

    # MSSubClass is numeric-coded but is really a category (see data description)
    df["MSSubClass"] = df["MSSubClass"].astype(str)

    return df


# BUILD PREPROCESSING PIPELINE
def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,     # <-- the fix
        )),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


# TRAIN / COMPARE MODELS
def get_candidate_models():
    return {
        "Ridge": RidgeCV(
            alphas=np.logspace(-2, 3, 40),
            cv=5,
        ),
        "Lasso": LassoCV(
            alphas=np.logspace(-4, -1, 30),
            cv=5, max_iter=20000, n_jobs=-1,
        ),
        "ElasticNet": ElasticNetCV(
            alphas=np.logspace(-4, -1, 20),
            l1_ratio=[0.5, 0.7, 0.9, 0.95, 1.0],
            cv=5, max_iter=20000, n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_features=0.3,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=3,
            max_features="sqrt",
            min_samples_leaf=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=800,
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def compare_models(X, y, preprocessor):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = get_candidate_models()
    results = {}

    for name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        scores = -cross_val_score(
            pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        results[name] = {"Mean RMSLE": scores.mean(), "Std RMSLE": scores.std()}
        print(f"{name:>18}: RMSLE = {scores.mean():.4f} (+/- {scores.std():.4f})")

    results_df = pd.DataFrame(results).T.sort_values("Mean RMSLE")
    return results_df


def main():
    train, test = load_data(TRAIN_PATH, TEST_PATH)

    explore(train)

    test_ids = test["Id"]

    train = fill_domain_missing(train)
    test = fill_domain_missing(test)

    train_feat = add_features(train.drop(columns=["SalePrice", "Id"]))
    test_feat = add_features(test.drop(columns=["Id"]))

    y = np.log1p(train["SalePrice"])
    X = train_feat
    X_test = test_feat

    # Align columns between train/test (in case of category mismatches)
    X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

    preprocessor = build_preprocessor(X)

    print("\n--- Holdout validation (quick sanity check) ---")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    quick_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=20.0)),
    ])
    quick_pipe.fit(X_train, y_train)
    y_pred = quick_pipe.predict(X_valid)
    rmse = root_mean_squared_error(y_valid, y_pred)
    print(f"Ridge holdout RMSLE: {rmse:.4f}")

    print("\n--- 5-fold CV across candidate models ---")
    results_df = compare_models(X, y, preprocessor)
    print("\nModel comparison (sorted best to worst):")
    print(results_df)

    best_model_name = results_df.index[0]
    print(f"\nBest model: {best_model_name}")

    best_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", get_candidate_models()[best_model_name]),
    ])
    best_pipeline.fit(X, y)

    test_predictions_log = best_pipeline.predict(X_test)
    test_predictions = np.expm1(test_predictions_log)
    test_predictions = np.maximum(test_predictions, 0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_predictions})
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"\nSaved submission to: {submission_path}")
    print(submission.head())
    print(f"Submission shape: {submission.shape}")


if __name__ == "__main__":
    main()
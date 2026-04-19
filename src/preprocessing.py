# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

from src.config import RAW_DATA_PATH


TARGET_COLUMN = "performance_band"


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def split_data(df):
    X = df.drop(columns=[TARGET_COLUMN, "performance_score"])  # avoid leakage
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline(X):
    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    return preprocessor


def get_processed_data():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessing_pipeline(X_train)

    return X_train, X_test, y_train, y_test, preprocessor
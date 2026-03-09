import pandas as pd
import numpy as np


def check_date_format(df, date_columns, date_format="%Y-%m-%d"):
    """
    Validate date columns and convert to datetime.
    """

    for col in date_columns:

        if col not in df.columns:
            print(f"{col} not found in dataframe")
            continue

        try:
            df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce")
        except Exception as e:
            print(f"Date conversion failed for {col}: {e}")

    return df


def check_null_values(df, feature_columns):
    """
    Check null values for each feature column.
    """

    null_summary = {}

    for col in feature_columns:

        if col not in df.columns:
            continue

        null_count = df[col].isnull().sum()

        null_summary[col] = null_count

        if null_count > 0:
            print(f"Column {col} has {null_count} null values")

    return null_summary


def identify_categorical_columns(df):
    """
    Identify categorical columns automatically
    """

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return categorical_cols


def cardinality_analysis(df, categorical_cols, high_card_threshold=20):
    """
    Identify high and low cardinality categorical columns
    """

    high_cardinality_cols = []
    low_cardinality_cols = []
    zero_cardinality_cols = []

    for col in categorical_cols:

        unique_count = df[col].nunique()

        if unique_count == 0:
            zero_cardinality_cols.append(col)

        elif unique_count > high_card_threshold:
            high_cardinality_cols.append(col)

        else:
            low_cardinality_cols.append(col)

    return {
        "high_cardinality": high_cardinality_cols,
        "low_cardinality": low_cardinality_cols,
        "zero_cardinality": zero_cardinality_cols
    }


def encode_low_cardinality(df, low_cardinality_cols):
    """
    One-hot encode low cardinality categorical columns
    """

    if len(low_cardinality_cols) == 0:
        return df

    df = pd.get_dummies(
        df,
        columns=low_cardinality_cols,
        drop_first=True
    )

    return df


def create_features(df):
    """
    Main feature engineering pipeline
    """

    df = df.copy()

    # Example feature transformations
    if "feature1" in df.columns and "feature2" in df.columns:
        df["feature_sum"] = df["feature1"] + df["feature2"]

    if "feature3" in df.columns and "feature4" in df.columns:
        df["feature_ratio"] = df["feature3"] / (df["feature4"] + 1)

    return df


def feature_engineering_pipeline(df, date_columns=None):

    if date_columns:
        df = check_date_format(df, date_columns)

    categorical_cols = identify_categorical_columns(df)

    card_results = cardinality_analysis(df, categorical_cols)

    low_card_cols = card_results["low_cardinality"]

    df = encode_low_cardinality(df, low_card_cols)

    df = create_features(df)

    return df, card_results
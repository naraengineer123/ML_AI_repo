import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_features(df):

    df = df.copy()

    # Example Feature Engineering
    df["feature_sum"] = df["feature1"] + df["feature2"]
    df["feature_ratio"] = df["feature3"] / (df["feature4"] + 1)

    return df


def scale_features(X):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler
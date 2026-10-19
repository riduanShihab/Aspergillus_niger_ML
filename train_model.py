import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Load dataset
file_path = "asergillus niger final.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

substrate_mapping = {
    "wheat barn": "wheat bran",
    "rice barn": "rice bran",
    "wheat bran": "wheat bran",
    "rice bran": "rice bran",
    "banana peel": "banana peel",
    "orange peel": "orange peel",
    "pectin": "pectin",
    "potato peel": "potato peel"
}
df["substrate"] = df["substrate"].astype(str).str.strip().str.lower().map(lambda s: substrate_mapping.get(s, s))
for col in ["ph", "time(days)", "temperature", "activity"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["ph", "time(days)", "temperature", "activity", "strain", "substrate"]).reset_index(drop=True)
df["strain"] = df["strain"].astype(str).str.replace(".0", "", regex=False)

features = ["strain", "ph", "substrate", "time(days)", "temperature"]
X = df[features].copy()
y = df["activity"].copy()

categorical = ["strain", "substrate"]
numerical = ["ph", "time(days)", "temperature"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numerical)
    ]
)

preprocessor.fit(X)
X_trans = preprocessor.transform(X)

gpr = GaussianProcessRegressor(
    kernel=C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e3)),
    normalize_y=True,
    random_state=42,
    n_restarts_optimizer=10
)
gpr.fit(X_trans, y)

artifacts = {
    "model": gpr,
    "preprocessor": preprocessor,
    "features": features,
    "categorical": categorical,
    "numerical": numerical,
    "strains": sorted(df["strain"].unique().tolist()),
    "substrates": sorted(df["substrate"].unique().tolist()),
    "ranges": {
        "ph": {"min": float(df["ph"].min()), "max": float(df["ph"].max()), "median": float(df["ph"].median())},
        "time_days": {"min": float(df["time(days)"].min()), "max": float(df["time(days)"].max()), "median": float(df["time(days)"].median())},
        "temperature": {"min": float(df["temperature"].min()), "max": float(df["temperature"].max()), "median": float(df["temperature"].median())}
    }
}

joblib.dump(artifacts, "model_artifacts.joblib")
print("Model artifacts successfully saved to model_artifacts.joblib!")

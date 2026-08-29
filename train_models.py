"""
Multi-dataset training pipeline.

Trains THREE separate models, one per dataset:
  1. Bike price prediction   -> target: price
  2. Car price prediction     -> target: Price
  3. Laptop rating prediction -> target: rating (from num_to_rate, shipping)

Each model follows the required STEPS:
  1. DATA LOADING
  2. DATA UNDERSTANDING
  3. DATA CLEANING (fill + drop)
  4. GRAPHS
  5. DATA ENCODING
  6. TRAIN / TEST SPLIT
  7. SAVE INTO .pkl FILE
"""

import io
import os
import re
import sys
import warnings

# Force UTF-8 output so rupee symbols / unicode in prints don't crash the run
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import joblib

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_number(series: pd.Series) -> pd.Series:
    """Extract the first numeric value from a string series (handles units,
    currency symbols, commas). Returns a float series with NaN where missing."""
    def _one(val):
        if pd.isna(val):
            return np_nan()
        s = str(val)
        # remove currency symbols / whitespace / commas
        s = re.sub(r"[^\d.\-]", " ", s)
        s = s.replace("  ", " ").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else np_nan()

    return series.apply(_one)


def np_nan():
    import numpy as np
    return np.nan


def to_numeric_if_possible(series: pd.Series) -> pd.Series:
    """If an object column contains mostly numbers, coerce it to numeric."""
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.notna().sum() >= 0.5 * len(series):
        return coerced
    return series


# ---------------------------------------------------------------------------
# Generic training routine that applies the 7 steps to any dataset
# ---------------------------------------------------------------------------
def train_model(name, csv_path, target, drop_cols, numeric_features,
                categorical_features, target_is_categorical=False):
    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    # ---- STEP 1: DATA LOADING --------------------------------------------
    print("\n[STEP 1] DATA LOADING")
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path} -> shape: {df.shape}")

    # Drop the unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # ---- STEP 2: DATA UNDERSTANDING --------------------------------------
    print("\n[STEP 2] DATA UNDERSTANDING")
    print("Columns:", list(df.columns))
    print("\nInfo:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print(f"\nTarget column '{target}' sample values:")
    print(df[target].astype(str).head(10).tolist())

    # ---- STEP 3: DATA CLEANING (fill + drop) -----------------------------
    print("\n[STEP 3] DATA CLEANING USING FILL AND DROP")
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")

    # Clean the target: extract numbers / handle NA
    df[target] = extract_number(df[target])
    before = len(df)
    df = df.dropna(subset=[target])
    print(f"Dropped {before - len(df)} rows with missing target. Remaining: {len(df)}")

    # Coerce designated numeric feature columns to numbers
    for col in numeric_features:
        if col in df.columns:
            df[col] = extract_number(df[col])

    # Fill numeric columns with median, categorical with mode
    for col in df.select_dtypes(include=["number"]).columns:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"  Filled numeric '{col}' with median = {median}")

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isnull().any():
            mode = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode)
            print(f"  Filled categorical '{col}' with mode = {mode}")

    keep = [c for c in (numeric_features + categorical_features + [target]) if c in df.columns]
    df = df[keep]
    print(f"Final feature set -> numeric: {numeric_features}, categorical: {categorical_features}")

    # ---- STEP 4: GRAPHS --------------------------------------------------
    print("\n[STEP 4] GRAPHS")
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target], kde=True, color="teal")
    plt.title(f"{name} - Target Distribution ({target})")
    plt.tight_layout()
    tgt_png = os.path.join(GRAPH_DIR, f"{name}_target_distribution.png")
    plt.savefig(tgt_png)
    plt.close()
    print(f"  Saved: {tgt_png}")

    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty and num_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm", fmt=".2f")
        plt.title(f"{name} - Feature Correlation")
        plt.tight_layout()
        corr_png = os.path.join(GRAPH_DIR, f"{name}_correlation.png")
        plt.savefig(corr_png)
        plt.close()
        print(f"  Saved: {corr_png}")

    for col in numeric_features:
        if col in df.columns:
            plt.figure(figsize=(7, 5))
            sns.scatterplot(data=df, x=col, y=target, alpha=0.5, color="purple")
            plt.title(f"{name}: {col} vs {target}")
            plt.tight_layout()
            sc_png = os.path.join(GRAPH_DIR, f"{name}_{col}_vs_{target}.png")
            plt.savefig(sc_png)
            plt.close()
            print(f"  Saved: {sc_png}")

    # ---- STEP 5: DATA ENCODING -------------------------------------------
    print("\n[STEP 5] DATA ENCODING")
    X = df.drop(columns=[target])
    y = df[target]

    present_num = [c for c in numeric_features if c in X.columns]
    present_cat = [c for c in categorical_features if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), present_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), present_cat),
        ]
    )

    label_encoders = {}
    if target_is_categorical or (y.dtype == object) or str(y.dtype).startswith("category"):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        label_encoders["target"] = le
        print(f"  Encoded target '{target}' with LabelEncoder classes: {list(le.classes_)}")

    for col in present_cat:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"  Label-encoded feature '{col}' -> {len(le.classes_)} classes")

    # ---- STEP 6: TRAIN / TEST SPLIT --------------------------------------
    print("\n[STEP 6] TEST TRAIN AND SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    if "target" in label_encoders:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if "target" in label_encoders:
        score = accuracy_score(y_test, y_pred)
        print(f"  Test accuracy: {score:.4f}")
    else:
        score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"  Test R2: {score:.4f} | MAE: {mae:.4f}")

    # ---- STEP 7: SAVE INTO .pkl FILE -------------------------------------
    print("\n[STEP 7] SAVE INTO .pkl FILE")
    artifact = {
        "pipeline": pipeline,
        "numeric_features": present_num,
        "categorical_features": present_cat,
        "target": target,
        "label_encoders": label_encoders,
        "score": float(score),
    }
    pkl_path = os.path.join(BASE_DIR, f"{name}_model.pkl")
    joblib.dump(artifact, pkl_path)
    print(f"  Model saved -> {pkl_path}")

    return pkl_path


# ---------------------------------------------------------------------------
# Dataset-specific configurations
# ---------------------------------------------------------------------------
def main():
    # 1) BIKE PRICE
    # The user requested skipping (dropping) many misspelled columns. Most of
    # those exact names are absent from the file, so we drop the ones that do
    # exist and keep the remaining real feature columns for training.
    bike_drop = [
        "owner", "Engine Description", "Fuel System", "cooling", "Displacement",
        "maximun maelectic strat", "Trip Meter", "peedometer", "0-100 kmph",
        "Rear Suspension", "Front Suspension", "stroke", "bore",
        "umber of Gears", "Fuel Tank Capacity", "Kerb/Wet Weight", "Wheelbase",
        "Ground Clearance", "Seat Height", "verall Height", "Overall Width",
        "Number of CylindersOverall Width", "Seat Height",
    ]
    bike_numeric = ["Maximum Power", "Maximum Torque", "Overall Length",
                    "Overall Width", "Overall Height", "Seat Height",
                    "Ground Clearance", "Wheelbase", "Kerb/Wet Weight",
                    "Fuel Tank Capacity", "Bore", "Stroke", "Number of Gears",
                    "Number of Cylinders", "Electric Start"]
    bike_categorical = ["company_name", "model", "status", "Body Type", "Fuel Type",
                        "Cooling", "Clutch", "Gearbox Type", "Front Brake",
                        "Rear Brake", "Front Suspension", "Rear Suspension",
                        "Speedometer", "Tachometer", "Trip Meter", "Clock"]
    train_model(
        name="bike",
        csv_path=os.path.join(BASE_DIR, "Bike_data.csv"),
        target="price",
        drop_cols=bike_drop,
        numeric_features=bike_numeric,
        categorical_features=bike_categorical,
    )

    # 2) CAR PRICE
    car_drop = ["Transmission", "Mileage", "condition"]
    car_numeric = ["Year", "Engine Size", "Mileage"]
    car_categorical = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]
    train_model(
        name="car",
        csv_path=os.path.join(BASE_DIR, "car_price_prediction_.csv"),
        target="Price",
        drop_cols=car_drop,
        numeric_features=car_numeric,
        categorical_features=car_categorical,
    )

    # 3) LAPTOP RATING (recommendation) from num_to_rate + shipping
    laptop_drop = []
    laptop_numeric = ["num_to_rate"]
    laptop_categorical = ["shiping"]
    train_model(
        name="laptop",
        csv_path=os.path.join(BASE_DIR, "new_egg_gaming_laptops.csv"),
        target="rating",
        drop_cols=laptop_drop,
        numeric_features=laptop_numeric,
        categorical_features=laptop_categorical,
        target_is_categorical=True,
    )

    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED AND SAVED.")
    print("=" * 70)


if __name__ == "__main__":
    main()

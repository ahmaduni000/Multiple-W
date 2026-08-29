"""
Watches Price Prediction - Full Pipeline
=========================================
Steps implemented:
  1. Data Collection
  2. Data Understanding
  3. Data Cleaning (fill & drop missing values)
  4. Visualizations (graphs)
  5. Data Encoding (categorical -> numeric)
  6. Train / Test Split
  7. Model Training
  8. Save model to pickle file

Run:  python main.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for servers/terminals)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "watches (cleaned).csv")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
MODEL_PATH = os.path.join(BASE_DIR, "watch_price_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

os.makedirs(GRAPH_DIR, exist_ok=True)
sns.set_style("whitegrid")


# ----------------------------------------------------------------------------
# STEP 1 & 2: DATA COLLECTION + DATA UNDERSTANDING
# ----------------------------------------------------------------------------
def load_and_understand(path):
    print("\n" + "=" * 70)
    print("STEP 1 & 2: DATA COLLECTION & DATA UNDERSTANDING")
    print("=" * 70)

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    print(f"Raw shape: {df.shape}")
    print("\nColumns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column (raw):")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nPrice summary:")
    print(df["Price"].describe())

    return df


# ----------------------------------------------------------------------------
# STEP 3: DATA CLEANING (FILL & DROP)
# ----------------------------------------------------------------------------
def clean_data(df):
    print("\n" + "=" * 70)
    print("STEP 3: DATA CLEANING (FILL & DROP)")
    print("=" * 70)

    # --- DROP rows with missing TARGET (Price) ---
    before = len(df)
    df = df.dropna(subset=["Price"])
    print(f"Dropped {before - len(df)} rows with missing Price (target).")

    # --- Select useful features ---
    numeric_features = [
        "Year of production", "Face Area", "Water resistance",
        "Watches Sold by the Seller", "Active listing of the seller",
        "Fast Shipper", "Trusted Seller", "Punctuality", "Seller Reviews",
    ]
    categorical_features = [
        "Brand", "Movement", "Case material", "Bracelet material",
        "Condition", "Scope of delivery", "Gender", "Shape",
        "Crystal", "Dial", "Bracelet color", "Clasp", "Availability",
    ]

    # Keep only existing columns
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    # --- FILL numeric missing values with MEDIAN ---
    for col in numeric_features:
        median_val = df[col].median()
        filled = df[col].isna().sum()
        df[col] = df[col].fillna(median_val)
        if filled:
            print(f"Filled {filled} missing in '{col}' with median = {median_val}")

    # --- FILL categorical missing values with MODE ('Unknown' fallback) ---
    for col in categorical_features:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        filled = df[col].isna().sum()
        df[col] = df[col].fillna(mode_val).astype(str).str.strip()
        if filled:
            print(f"Filled {filled} missing in '{col}' with mode = '{mode_val}'")

    df = df.reset_index(drop=True)
    print(f"\nFinal cleaned shape: {df.shape}")
    print(df.head())
    return df, numeric_features, categorical_features


# ----------------------------------------------------------------------------
# STEP 4: VISUALIZATIONS (GRAPHS)
# ----------------------------------------------------------------------------
def visualize(df):
    print("\n" + "=" * 70)
    print("STEP 4: VISUALIZATIONS (GRAPHS)")
    print("=" * 70)

    # 4.1 Price distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df["Price"], bins=50, color="teal")
    plt.title("Price Distribution ($)")
    plt.xlabel("Price ($)")
    plt.tight_layout()
    p1 = os.path.join(GRAPH_DIR, "price_distribution.png")
    plt.savefig(p1)
    plt.close()
    print("Saved:", p1)

    # 4.2 Price by Brand (top 15 by count)
    plt.figure(figsize=(12, 6))
    top_brands = df["Brand"].value_counts().head(15).index
    sns.boxplot(data=df[df["Brand"].isin(top_brands)],
                x="Brand", y="Price", showfliers=False)
    plt.xticks(rotation=45)
    plt.title("Price by Brand (Top 15)")
    plt.tight_layout()
    p2 = os.path.join(GRAPH_DIR, "price_by_brand.png")
    plt.savefig(p2)
    plt.close()
    print("Saved:", p2)

    # 4.3 Price by Movement
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Movement", y="Price", showfliers=False)
    plt.title("Price by Movement Type")
    plt.tight_layout()
    p3 = os.path.join(GRAPH_DIR, "price_by_movement.png")
    plt.savefig(p3)
    plt.close()
    print("Saved:", p3)

    # 4.4 Price by Condition
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Condition", y="Price", showfliers=False)
    plt.xticks(rotation=30)
    plt.title("Price by Condition")
    plt.tight_layout()
    p4 = os.path.join(GRAPH_DIR, "price_by_condition.png")
    plt.savefig(p4)
    plt.close()
    print("Saved:", p4)

    # 4.5 Correlation heatmap (numeric features + Price)
    num_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm",
                fmt=".2f", annot_kws={"size": 7})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    p5 = os.path.join(GRAPH_DIR, "correlation_heatmap.png")
    plt.savefig(p5)
    plt.close()
    print("Saved:", p5)


# ----------------------------------------------------------------------------
# STEP 5: DATA ENCODING
# ----------------------------------------------------------------------------
def encode_data(df, categorical_features):
    print("\n" + "=" * 70)
    print("STEP 5: DATA ENCODING (categorical -> numeric)")
    print("=" * 70)

    df = df.copy()
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col + "_Enc"] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  Encoded '{col}': {len(le.classes_)} unique values")

    # Save encoders for later use in prediction
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print(f"Encoders saved -> {ENCODERS_PATH}")

    return df, encoders


# ----------------------------------------------------------------------------
# STEP 6 & 7: TRAIN/TEST SPLIT + MODEL TRAINING
# ----------------------------------------------------------------------------
def train_model(df, numeric_features, categorical_features):
    print("\n" + "=" * 70)
    print("STEP 6 & 7: TRAIN/TEST SPLIT + MODEL TRAINING")
    print("=" * 70)

    features = numeric_features + [c + "_Enc" for c in categorical_features]
    X = df[features]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    print("\n--- Linear Regression ---")
    print(f"  R2 Score : {r2_score(y_test, lr_pred):.4f}")
    print(f"  MAE      : {mean_absolute_error(y_test, lr_pred):.2f}")
    print(f"  RMSE     : {mean_squared_error(y_test, lr_pred) ** 0.5:.2f}")

    # Model 2: Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("\n--- Random Forest Regressor ---")
    print(f"  R2 Score : {r2_score(y_test, rf_pred):.4f}")
    print(f"  MAE      : {mean_absolute_error(y_test, rf_pred):.2f}")
    print(f"  RMSE     : {mean_squared_error(y_test, rf_pred) ** 0.5:.2f}")

    importances = pd.Series(rf.feature_importances_, index=features)
    print("\nTop 10 feature importances (Random Forest):")
    print(importances.sort_values(ascending=False).head(10))

    best_model = rf if r2_score(y_test, rf_pred) >= r2_score(y_test, lr_pred) else lr
    best_name = "RandomForestRegressor" if best_model is rf else "LinearRegression"
    print(f"\nSelected best model: {best_name}")

    return best_model, scaler, features


# ----------------------------------------------------------------------------
# STEP 8: SAVE MODEL TO PICKLE
# ----------------------------------------------------------------------------
def save_model(model, scaler):
    print("\n" + "=" * 70)
    print("STEP 8: SAVE MODEL TO PICKLE FILE")
    print("=" * 70)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Model saved  -> {MODEL_PATH}")
    print(f"Scaler saved -> {SCALER_PATH}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    df_raw = load_and_understand(DATA_PATH)
    df_clean, num_feats, cat_feats = clean_data(df_raw)
    visualize(df_clean)
    df_enc, _ = encode_data(df_clean, cat_feats)
    model, scaler, _ = train_model(df_enc, num_feats, cat_feats)
    save_model(model, scaler)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

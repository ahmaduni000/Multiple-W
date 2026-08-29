"""
predict.py - Load the saved pickle model and predict watch price.

Example:
    python predict.py
"""

import os
import pickle

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "watch_price_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

NUMERIC_FEATURES = [
    "Year of production", "Face Area", "Water resistance",
    "Watches Sold by the Seller", "Active listing of the seller",
    "Fast Shipper", "Trusted Seller", "Punctuality", "Seller Reviews",
]
CATEGORICAL_FEATURES = [
    "Brand", "Movement", "Case material", "Bracelet material",
    "Condition", "Scope of delivery", "Gender", "Shape",
    "Crystal", "Dial", "Bracelet color", "Clasp", "Availability",
]

# Fallback medians used when a numeric value is not provided
MEDIANS = {
    "Year of production": 2020.0, "Face Area": 692.37,
    "Water resistance": 10.0, "Watches Sold by the Seller": 584.0,
    "Active listing of the seller": 419.0, "Seller Reviews": 448.0,
}


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return model, scaler, encoders


def predict_price(**kwargs):
    """Pass feature values as keyword arguments, e.g.
    predict_price(Brand='Rolex', Movement='Automatic',
                  **{'Case material': 'Steel', ...}).
    Missing features fall back to training median / mode."""
    model, scaler, encoders = load_artifacts()

    row = {}
    for col in NUMERIC_FEATURES:
        val = kwargs.get(col, np.nan)
        row[col] = MEDIANS.get(col, 0.0) if pd.isna(val) else val
    for col in CATEGORICAL_FEATURES:
        val = str(kwargs.get(col, "Unknown"))
        le = encoders.get(col)
        if le is not None:
            # Map unseen labels to the most frequent class (index 0)
            if val in le.classes_:
                row[col + "_Enc"] = int(le.transform([val])[0])
            else:
                row[col + "_Enc"] = 0
        else:
            row[col + "_Enc"] = 0

    features = NUMERIC_FEATURES + [c + "_Enc" for c in CATEGORICAL_FEATURES]
    df = pd.DataFrame([row])[features]

    if hasattr(model, "coef_"):  # linear model needs scaling
        X = scaler.transform(df)
    else:
        X = df.values

    price = model.predict(X)[0]
    return round(float(price), 2)


if __name__ == "__main__":
    price = predict_price(
        Brand="Rolex",
        Movement="Automatic",
        **{"Case material": "Steel", "Bracelet material": "Steel",
           "Condition": "Used (Very good)", "Scope of delivery": "Original box, original papers",
           "Gender": "Men's watch/Unisex", "Shape": "Circular", "Crystal": "Sapphire crystal",
           "Dial": "Black", "Bracelet color": "Black", "Clasp": "Fold clasp",
           "Availability": "Item is in stock", "Year of production": 2022.0,
           "Face Area": 692.37, "Water resistance": 30.0,
           "Watches Sold by the Seller": 100.0, "Active listing of the seller": 50.0,
           "Fast Shipper": 1, "Trusted Seller": 1, "Punctuality": 1, "Seller Reviews": 100.0},
    )
    print(f"Predicted price: ${price}")

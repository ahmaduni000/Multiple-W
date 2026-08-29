# Watches Price Prediction

End-to-end machine learning pipeline that predicts watch prices from specs and
seller metadata (brand, movement, case/bracelet material, condition, year,
face area, water resistance, seller reputation, etc.).

## Project Steps
1. **Data Collection** – load `watches (cleaned).csv` (45,024 rows, 23 cols)
2. **Data Understanding** – shape, dtypes, missing values, duplicates, price stats
3. **Data Cleaning** – drop rows missing the target; fill numeric gaps with median, categorical gaps with mode
4. **Graphs** – price distribution, price by brand/movement/condition, correlation heatmap
5. **Data Encoding** – `LabelEncoder` converts 13 categorical columns to numbers
6. **Train/Test Split** – 80/20 split
7. **Model Training** – Linear Regression + Random Forest (best by R² is saved)
8. **Pickle** – model + scaler + encoders saved as `.pkl`

## Setup
```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run the pipeline
```bash
python main.py
```

## Predict with the saved model
```bash
python predict.py
```

## Outputs
- `watch_price_model.pkl` – trained model
- `scaler.pkl` – feature scaler
- `encoders.pkl` – saved LabelEncoders for categorical features
- `graphs/` – 5 PNG visualizations

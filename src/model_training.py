# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
import joblib
import os
from data_preprocessing import load_and_clean_data
from feature_engineering import create_features

def train_and_save_models(data_path, models_dir="models"):
    """
    Loads data, engineers features, trains the best models (XGBoost and SVC),
    and saves them.
    """
    print("Starting model training process...")
    df = load_and_clean_data(data_path)
    print("Data loaded and cleaned.")
    df = create_features(df)
    print("Features engineered.")

    # --- Regression Model Training (XGBoost) ---
    X1 = df[[
        'volume', 'Price Change', 'Rolling_Std_Close', 'vol_1h', 'vol_mean_6h',
        'vol_std_6h', 'vol_max_6h', 'vol_min_6h', 'hour', 'dayofweek', 'day', 'rsi',
        'high_low_ratio', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]]
    y1 = df['Target_Close']
    X1_train, _, y1_train, _ = train_test_split(X1, y1, test_size=0.2, random_state=42)

    print("Training XGBoost Regressor for price prediction...")
    xgb_pipeline = make_pipeline(
        StandardScaler(),
        XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    )
    xgb_pipeline.fit(X1_train, y1_train)
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(xgb_pipeline, os.path.join(models_dir, 'best_xgb_model.pkl'))
    print("✅ XGBoost Regressor model saved.")

    # --- Classification Model Training (SVC) ---
    X2 = df[['volume', 'Price Change', 'Volatility', 'Rolling_Mean_Close', 'Rolling_Std_Close',
         'vol_mean_6h', 'vol_std_6h', 'vol_max_6h', 'vol_min_6h', 'return_mean_6h',
         'return_std_6h', 'hour', 'dayofweek', 'day', 'rsi', 'macd', 'bb_high',
         'bb_low', 'ema_10', 'ema_30', 'high_low_ratio', 'close_open_diff',
         'close_lag_1', 'volume_lag_1', 'rolling_max_6h', 'rolling_min_6h',
         'price_volatility_interaction', 'hour_sin', 'hour_cos', 'day_sin',
         'day_cos']]
    y2 = df['Target_Movement']
    X2_train, _, y2_train, _ = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

    print("Training SVC for movement prediction...")
    # Scale the features for SVC
    scaler = StandardScaler()
    X2_train_scaled = scaler.fit_transform(X2_train)

    # Define parameter grid for RandomizedSearch based on your notebook
    svc_param_grid = {
        'C': [1, 10, 50, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    
    # Using RandomizedSearchCV to find the best SVC
    random_search = RandomizedSearchCV(
        SVC(probability=True), # probability=True is needed for confidence scores
        param_distributions=svc_param_grid,
        n_iter=10, # As in the notebook
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X2_train_scaled, y2_train)
    
    best_svc = random_search.best_estimator_

    joblib.dump(best_svc, os.path.join(models_dir, 'best_svc_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl')) # Save the scaler used for SVC
    print("✅ Best SVC model and scaler saved.")
    print("\nTraining complete!")

if __name__ == '__main__':
    train_and_save_models(data_path='data/BTCUSDT-1h.csv')
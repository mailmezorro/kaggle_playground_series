#!/usr/bin/env python
"""
Run Optuna hyperparameter tuning for XGBoost with GPU support via xgb.cv.
Requires conda env `xgb_ok` (CUDA11.8 + GPU‑enabled xgboost).
Usage:
  conda activate xgb_ok
  python optuna_xgb.py [--data-dir PATH]
"""
import warnings
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

import os
import argparse
import numpy as np
import pandas as pd
import optuna
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

# ---------- Argument Parsing ----------
parser = argparse.ArgumentParser(description="Optuna XGBoost tuning script")
parser.add_argument("--data-dir", type=str,
                    help="Path to data directory containing s5-e4-preprocessed folder")
args = parser.parse_args()

# ---------- Determine DATA_DIR ----------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
candidates   = [args.data_dir] if args.data_dir else []
candidates  += [
    '/kaggle/input',
    os.path.join(PROJECT_ROOT, 'kaggle', 'input'),
    os.path.join(PROJECT_ROOT, 'input'),
    os.path.join(PROJECT_ROOT, '..', 'kaggle', 'input'),
]
for d in candidates:
    if d and os.path.exists(d):
        DATA_DIR = d
        break
else:
    raise FileNotFoundError("Data directory not found. Use --data-dir to specify it.")

TRAIN_PATH = os.path.join(DATA_DIR, 's5-e4-preprocessed', 's5-e4-train_preprocessed.csv')
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Train file not found at {TRAIN_PATH}")
train_df = pd.read_csv(TRAIN_PATH)

# ---------- Features & Target ----------
TARGET = "Listening_Time_minutes"
X_full = train_df.drop(columns=[TARGET])
y_full = train_df[TARGET].astype(np.float32)

# ---------- Split Dataset ----------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_full, y_full, test_size=0.3, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# ---------- Preprocessing ----------
cat_cols = [
    'Podcast_Name', 'Genre',
    'Publication_Day', 'Publication_Time',
    'Episode_Sentiment'
]
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
enc.fit(X_train[cat_cols])
for df in (X_train, X_val, X_test):
    df[cat_cols] = enc.transform(df[cat_cols]).astype(np.float32)

num_cols = X_train.select_dtypes(include=['int64','float64']).columns
scaler   = RobustScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols]).astype(np.float32)
X_val[num_cols]   = scaler.transform(X_val[num_cols]).astype(np.float32)
X_test[num_cols]  = scaler.transform(X_test[num_cols]).astype(np.float32)

# ---------- Pre‑allocate DMatrix on GPU ----------
dtrain = xgb.DMatrix(X_train.values, label=y_train.values)

# ---------- Define custom folds for xgb.cv ----------
kf    = KFold(n_splits=5, shuffle=True, random_state=42)
folds = [
    (
        np.asarray(tr_idx, dtype=np.int32),
        np.asarray(val_idx, dtype=np.int32)
    )
    for tr_idx, val_idx in kf.split(X_train)
]

# ---------- Objective Function using xgb.cv on GPU ----------
def objective(trial: optuna.Trial) -> float:
    params = {
        'max_depth':        trial.suggest_int('max_depth', 3, 6),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma':            trial.suggest_float('gamma', 0, 5),
        'tree_method':      'hist',       # CPU-hist API, GPU enabled via device
        'device':           'cuda',       # run on GPU
        'eval_metric':      'rmse',
        'random_state':     42,
        'nthread':          1,            # reduce CPU overhead
    }
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        folds=folds,
        num_boost_round=1000,
        early_stopping_rounds=50,
        seed=42,
        metrics='rmse',
        as_pandas=True,
        verbose_eval=False
    )
    return float(cv_results['test-rmse-mean'].min())

# ---------- Main ----------
if __name__ == '__main__':
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=20, timeout=6000)

    print('Best RMSE:', study.best_value)
    print('Best params:', study.best_params)
    joblib.dump(study, 'optuna_study.pkl')
    
    print("Best Parameters:", study.best_params)

    # Train best model on training set
    best_model = XGBRegressor(**study.best_params, enable_categorical=True)
    best_model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = best_model.predict(X_val)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    print(f"RMSE on validation set: {rmse:.4f}")
    print(f"MAE on validation set:  {mae:.4f}")

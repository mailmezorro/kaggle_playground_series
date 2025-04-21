#!/usr/bin/env python
# benchmark_gpu_vs_cpu.py

import argparse
import time

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression

def run_benchmark(n_samples: int, n_features: int, n_rounds: int):
    print(f"\n== Benchmark: {n_samples} Samples × {n_features} Features × {n_rounds} Runden ==")
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,
        random_state=42
    )
    dtrain = xgb.DMatrix(X, label=y)

    base_params = {
        'tree_method':      'hist',            # standard API; Gerät wechselt via 'device'
        'objective':        'reg:squarederror',
        'eval_metric':      'rmse',
        'max_depth':        6,
        'learning_rate':    0.1,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'seed':             42,
    }

    for device in ['cpu', 'cuda']:
        params = base_params.copy()
        params['device'] = device
        label = device.upper()
        print(f"\n→ Starte Training auf {label}…")
        t0 = time.time()
        xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)
        dt = time.time() - t0
        print(f"   Dauer auf {label}: {dt:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vergleicht XGBoost CPU vs. GPU Performance"
    )
    parser.add_argument(
        "--samples", type=int, default=100_000,
        help="Anzahl der Samples (Standard: 100000)"
    )
    parser.add_argument(
        "--features", type=int, default=30,
        help="Anzahl der Features (Standard: 30)"
    )
    parser.add_argument(
        "--rounds", type=int, default=200,
        help="Anzahl der Boosting‑Runden (Standard: 200)"
    )
    args = parser.parse_args()
    run_benchmark(args.samples, args.features, args.rounds)

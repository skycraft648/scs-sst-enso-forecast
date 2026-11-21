"""Training and evaluation for reconstructed monthly SST prediction experiment."""
from __future__ import annotations
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES = ['Month', 'sst_lag1', 'sst_lag3', 'sst_lag6', 'anom34']
DEFAULT_MODELS = ['LinearRegression', 'SVR', 'RandomForest', 'XGBoost', 'SARIMAX', 'LSTM']


def _evaluate(y, pred) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 and np.std(pred) > 0 and np.std(y) > 0 else float('nan')
    return dict(RMSE=float(np.sqrt(mean_squared_error(y, pred))),
                MAE=float(mean_absolute_error(y, pred)),
                R2=float(r2_score(y, pred)), Pearson_r=corr,
                MAPE_percent=float(np.mean(np.abs((y - pred) / np.maximum(np.abs(y), 1e-8))) * 100))


def sarimax_one_step(train_y, train_x, test_y, test_x):
    """Expanding observed-history one-step forecasts, NOT a multi-month forecast.

    For each test month: forecast once with known current exogenous predictors,
    then append the actual observed SST before predicting the following month.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitted = SARIMAX(train_y, exog=train_x, order=(2, 1, 2),
                         seasonal_order=(1, 0, 1, 12), trend='n',
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False, maxiter=80)
        preds = []
        for i in range(len(test_y)):
            ex = np.asarray(test_x[i:i+1], dtype=float)
            preds.append(float(np.asarray(fitted.forecast(steps=1, exog=ex)).reshape(-1)[0]))
            fitted = fitted.append(endog=np.asarray(test_y[i:i+1]), exog=ex, refit=False)
    return np.asarray(preds)


def train_lstm(x_all, y_all, train_end, window=12, seed=42):
    """Train LSTM with train-only fitted scalers and chronological validation.

    The current month's exogenous features and previous observations are passed
    in a 12-step sequence ending at the predicted month. Works for retrospective
    one-step prediction with observed lags, not an ex-ante seasonal forecast.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise RuntimeError('LSTM requires TensorFlow: pip install -r requirements-lstm.txt') from e
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    xsc = StandardScaler().fit(x_all[:train_end])
    ysc = StandardScaler().fit(np.asarray(y_all[:train_end]).reshape(-1, 1))
    xs = xsc.transform(x_all)
    ys = ysc.transform(np.asarray(y_all).reshape(-1, 1)).reshape(-1)
    # All model datasets have the same valid evaluation dates; train rows can
    # include preceding months as warm-up but no test targets are used for fit.
    sequences = np.stack([xs[i-window+1:i+1] for i in range(window-1, len(xs))])
    target = ys[window-1:]
    indices = np.arange(window-1, len(xs))
    train_mask = indices < train_end
    test_mask = ~train_mask
    seq_train, y_train = sequences[train_mask], target[train_mask]
    if len(seq_train) < 30:
        raise ValueError('Not enough training months for LSTM. At least 30 sequences required.')
    val_len = max(1, round(len(seq_train) * .1))
    tr, val = (seq_train[:-val_len], y_train[:-val_len]), (seq_train[-val_len:], y_train[-val_len:])
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, x_all.shape[1])),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(tr[0], tr[1], validation_data=val, epochs=100, batch_size=8,
              shuffle=False, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=10, restore_best_weights=True)])
    preds_scaled = model.predict(sequences[test_mask], verbose=0).reshape(-1, 1)
    preds = ysc.inverse_transform(preds_scaled).reshape(-1)
    return indices[test_mask], preds


def run_experiment(csv_file: str | Path, output_dir: str | Path, models: list[str] | None = None,
                   seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read preprocessed SST; train chronological 80/20 with an LSTM warmup.

    The 12-month warmup restriction is applied to ALL models for fair testing.
    The final months use actual lagged SST from preceding months.
    """
    models = list(models or DEFAULT_MODELS)
    unknown = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f'Unknown models: {unknown}; choices: {DEFAULT_MODELS}')
    df = pd.read_csv(csv_file, parse_dates=['date'])
    required = {'date', 'sst', *FEATURES}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required - set(df.columns)}; run preprocess first.')
    df = df.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(df['date'])
    if dates.duplicated().any() or not dates.dt.to_period('M').equals(pd.Series(pd.period_range(dates.iloc[0], periods=len(dates), freq='M'))):
        raise ValueError('Months must be unique, gap-free, and sorted; rebuild preprocessing.')
    df = df.dropna(subset=FEATURES + ['sst']).reset_index(drop=True)
    if len(df) < 60:
        raise ValueError('At least 60 complete months required for meaningful demo/training.')
    # All algorithms share the identical evaluated subset after 12-month window.
    warmup = 11
    n_eligible = len(df) - warmup
    train_end = warmup + int(n_eligible * 0.8)
    features = df[FEATURES].to_numpy(dtype=float)
    y = df['sst'].to_numpy(dtype=float)
    xtr, xte = features[warmup:train_end], features[train_end:]
    ytr, yte = y[warmup:train_end], y[train_end:]
    test = pd.DataFrame({'date': df.date.iloc[train_end:].to_numpy(), 'actual_sst': yte})
    results = []
    estimators = {
        'LinearRegression': LinearRegression(),
        'SVR': Pipeline([('scale', StandardScaler()), ('reg', SVR(C=10, epsilon=0.1, kernel='rbf'))]),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1, min_samples_leaf=2),
    }
    if 'XGBoost' in models:
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise RuntimeError('XGBoost requires: pip install xgboost') from e
        estimators['XGBoost'] = XGBRegressor(n_estimators=300, max_depth=3,
                                            learning_rate=0.04, subsample=0.85,
                                            colsample_bytree=0.9, objective='reg:squarederror',
                                            random_state=seed, n_jobs=2)
    importance = None
    for name in models:
        if name in estimators:
            model = estimators[name]
            model.fit(xtr, ytr)
            pred = model.predict(xte)
            if name == 'XGBoost':
                importance = model.feature_importances_
        elif name == 'SARIMAX':
            pred = sarimax_one_step(ytr, xtr, yte, xte)
        elif name == 'LSTM':
            ix, pred = train_lstm(features[warmup:], y[warmup:],
                                  train_end - warmup, window=12, seed=seed)
            # train_lstm requires 12 extra feature rows, so its test samples
            # are exactly the test slice if train_end is beyond the warmup.
            assert len(ix) == len(yte), 'Unexpected LSTM test alignment.'
        else:
            raise AssertionError(name)
        if len(pred) != len(yte):
            raise ValueError(f'{name}: different test sample count')
        test[name] = pred
        results.append({'Model': name, **_evaluate(yte, pred)})
    metrics = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
    output_dir = Path(output_dir)
    (output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    test.to_csv(output_dir / 'tables' / 'model_predictions.csv', index=False)
    metrics.to_csv(output_dir / 'tables' / 'model_metrics.csv', index=False)
    if importance is not None:
        pd.DataFrame({'feature': FEATURES, 'importance': importance}).sort_values('importance', ascending=False).to_csv(
            output_dir / 'tables' / 'xgboost_feature_importance.csv', index=False)
    return metrics, test

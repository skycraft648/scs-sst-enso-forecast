from pathlib import Path
import pandas as pd
from sst_project.demo import generate_demo
from sst_project.data import parse_nino34, prepare_csv
from sst_project.models import run_experiment
from sst_project.plots import diagnostic_plots, prediction_plots


def test_parser(tmp_path: Path):
    folder = generate_demo(tmp_path / 'demo', n_months=100)
    df = parse_nino34(folder / 'raw' / 'enso' / 'demo_nino.ascii')
    assert len(df) == 100
    assert list(df.columns) == ['date', 'anom34']


def test_full_smoke_no_deep_learning(tmp_path: Path):
    folder = generate_demo(tmp_path / 'demo', n_months=120)
    processed = tmp_path / 'SST_processed.csv'
    df = prepare_csv(folder / 'raw' / 'sst' / 'demo_monthly_sst.csv',
                     folder / 'raw' / 'enso' / 'demo_nino.ascii', processed)
    assert len(df) == 120
    assert not df.SSTA.isna().any()
    out = tmp_path / 'outputs'
    metrics, preds = run_experiment(processed, out, ['LinearRegression', 'SVR', 'RandomForest', 'XGBoost'])
    assert len(metrics) == 4
    assert len(preds) == 21  # 120 months - 6 lags - 11 LSTM warmup; 80/20 split
    assert set(['actual_sst', 'LinearRegression']).issubset(preds.columns)
    diagnostic_plots(processed, out)
    prediction_plots(out)
    assert (out / 'figures' / 'Figure5_XGBoost_Importance.png').exists()


def test_netcdf_to_monthly_sst(tmp_path: Path):
    import numpy as np
    import xarray as xr
    from sst_project.data import netcdf_to_monthly_sst
    ds = xr.Dataset(
        {'analysed_sst': (('time', 'lat', 'lon'), np.full((2, 3, 3), 300.15), {'units': 'K'})},
        coords={'time': pd.to_datetime(['1993-01-16', '1993-02-16']),
                'lat': [0., 10., 30.], 'lon': [100., 110., 120.]})
    path = tmp_path / 'sample.nc'
    ds.to_netcdf(path, engine='scipy')
    out = netcdf_to_monthly_sst(path, tmp_path / 'monthly.csv', tmp_path / 'map.npz')
    assert len(out) == 2
    assert abs(out.sst.iloc[0] - 27) < 1e-6
    assert (tmp_path / 'map.npz').exists()

"""Explicitly SYNTHETIC data generator for smoke testing. NOT real observations."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_demo(folder: str | Path, seed: int = 42, n_months: int = 384):
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range('1993-01-01', periods=n_months, freq='MS')
    month = dates.month.to_numpy()
    t = np.arange(len(dates))
    nino = 1.2 * np.sin(2 * np.pi * t / 48 + .5) + rng.normal(0, .25, len(dates))
    sst = 27.2 + 2.8 * np.sin(2 * np.pi * (month - 4) / 12) + .08 * nino + rng.normal(0, .3, len(dates))
    (out / 'raw' / 'sst').mkdir(parents=True, exist_ok=True)
    (out / 'raw' / 'enso').mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'date': dates, 'sst': sst}).to_csv(out / 'raw' / 'sst' / 'demo_monthly_sst.csv', index=False)
    # NOAA-style 10 columns: year month sst12 anom12 sst3 anom3 sst4 anom4 sst34 anom34
    lines = ['YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM']
    for dt, anomaly in zip(dates, nino):
        lines.append(f'{dt.year:04d} {dt.month:02d} 25.0 0.0 26.0 0.0 27.0 0.0 26.5 {anomaly:.3f}')
    (out / 'raw' / 'enso' / 'demo_nino.ascii').write_text('\n'.join(lines) + '\n', encoding='utf8')
    (out / 'SYNTHETIC_DATA_ONLY.txt').write_text(
        'WARNING: All files in this demo data folder are synthetic. NOT real SST or NOAA observations.\n'
        'Do not report demo metrics as scientific results.\n', encoding='utf8')
    return out

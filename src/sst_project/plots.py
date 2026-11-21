"""Plots corresponding approximately to figures in the archived paper.

These are newly generated figures; they are NOT copies of the lost originals.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def diagnostic_plots(csv_file: str | Path, output_dir: str | Path,
                     mean_map_npz: str | Path | None = None) -> float:
    df = pd.read_csv(csv_file, parse_dates=['date'])
    out = Path(output_dir) / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 4))
    plt.plot(df.date, df.sst, linewidth=0.9)
    plt.title('South China Sea monthly mean SST')
    plt.ylabel('SST (°C)'); plt.xlabel('Date'); plt.tight_layout()
    plt.savefig(out / 'Figure1_SST_Time_Series.png', dpi=170); plt.close()

    if mean_map_npz and Path(mean_map_npz).is_file():
        mm = np.load(mean_map_npz)
        plt.figure(figsize=(8, 5))
        plt.pcolormesh(mm['lon'], mm['lat'], mm['mean_sst'], shading='auto')
        plt.colorbar(label='Long-term mean SST (°C)')
        plt.title('South China Sea long-term SST map')
        plt.xlabel('Longitude (°E)'); plt.ylabel('Latitude (°N)')
        plt.tight_layout(); plt.savefig(out / 'Figure2_Mean_SST_Map.png', dpi=170); plt.close()

    valid = df[['anom34', 'SSTA']].dropna()
    corr = float(valid.anom34.corr(valid.SSTA))
    plt.figure(figsize=(6, 5))
    plt.scatter(valid.anom34, valid.SSTA, s=13, alpha=.7)
    plt.title(f'Niño3.4 vs SCS SST anomaly (r={corr:.3f})')
    plt.xlabel('Niño3.4 monthly anomaly (°C)')
    plt.ylabel('SCS SST anomaly (°C)')
    plt.tight_layout(); plt.savefig(out / 'Figure3_Nino_SSTA_Scatter.png', dpi=170); plt.close()
    return corr


def prediction_plots(output_dir: str | Path):
    out = Path(output_dir)
    table = out / 'tables' / 'model_predictions.csv'
    if not table.exists():
        raise FileNotFoundError(f'No predictions CSV: {table}')
    df = pd.read_csv(table, parse_dates=['date'])
    plt.figure(figsize=(13, 6))
    plt.plot(df.date, df.actual_sst, label='Observed', color='black', linewidth=2)
    for model in [x for x in df.columns if x not in {'date', 'actual_sst'}]:
        plt.plot(df.date, df[model], label=model, linewidth=1, alpha=.84)
    plt.legend(ncol=3, fontsize=9)
    plt.title('Retrospective one-month-ahead SST prediction comparison')
    plt.ylabel('SST (°C)'); plt.xlabel('Date'); plt.tight_layout()
    plt.savefig(out / 'figures' / 'Figure4_Model_Comparison.png', dpi=170)
    plt.close()
    fi = out / 'tables' / 'xgboost_feature_importance.csv'
    if fi.exists():
        f = pd.read_csv(fi).sort_values('importance')
        plt.figure(figsize=(7, 4))
        plt.barh(f.feature, f.importance)
        plt.xlabel('XGBoost feature importance')
        plt.title('XGBoost feature importance (this reconstruction)')
        plt.tight_layout(); plt.savefig(out / 'figures' / 'Figure5_XGBoost_Importance.png', dpi=170)
        plt.close()

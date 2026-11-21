# South China Sea SST Prediction with ENSO (Reconstructed)

> **Code reconstruction, not an original-code recovery.** The original project source and local SST input data were lost in a hard-drive failure. This repository was rebuilt from the author's earlier paper, workflow notes and project design. It is **not** a byte-for-byte recovery; historical metrics are archived separately and must not be represented as newly reproduced results.

基于历史论文和研究记录**重新编写**的南海月平均海表温度（SST）预测项目，覆盖 **1993–2024 年、0–25°N、105–125°E**，结合 NOAA Niño3.4 异常指数，对比 SARIMAX、Linear Regression、SVR、Random Forest、XGBoost 和 LSTM 六种方法。

**This is a retrospective one-step prediction experiment**, not an operational months-ahead forecast: test-set features include **observed prior-month SST** and the **concurrent-month Niño3.4 anomaly**. If predicting months into the future without these observations, the evaluation setup must be redesigned. Paper results cannot be reproduced exactly without the original dataset and hyperparameters.

## Project scope / 项目内容

- **Inputs:** C3S/ESA CCI satellite SST NetCDF, and NOAA CPC ERSSTv5 Niño3.4 monthly anomaly
- **Domain:** South China Sea (0–25°N, 105–125°E), Jan 1993–Dec 2024
- **Processing:** NetCDF → regional mean monthly SST → join Niño3.4 → calendar month, SST lag 1/3/6 months
- **Diagnostics only:** SST anomaly (SSTA) against 1993–2024 monthly climatology
- **Models:** SARIMAX (2,1,2) × (1,0,1,12), Linear Regression, RBF-SVR, Random Forest, XGBoost, 32-unit LSTM with 12-month window and dropout 0.2
- **Train/test:** chronological 80/20 on the common eligible rows (after feature lags and common 12-month warmup); LSTM uses last 10% of training sequences for validation
- **Metrics:** RMSE, MAE, R², Pearson r, MAPE
- **Figures:** SST time series, SST map (if compatible raw NetCDF supplied), Niño3.4-SSTA relationship, model prediction comparison, XGBoost feature importance

## Repo layout

```text
scs-sst-enso-reconstructed/
├── src/sst_project/
│   ├── cli.py          # Commands / 入口
│   ├── data.py         # NetCDF / NOAA parsing and processing
│   ├── models.py       # 6 models + evaluation
│   ├── plots.py        # Results visualization
│   └── demo.py         # Explicitly synthetic smoke-test data
├── data/
│   ├── raw/sst/        # Place original Copernicus .nc files here (not committed)
│   ├── raw/enso/       # Download NOAA Niño index here (not committed)
│   └── processed/      # Generated CSV (not committed)
├── outputs/{figures,tables}/  # Generated, not committed
├── docs/archived_paper_results.md
├── tests/
├── requirements.txt
└── requirements-lstm.txt
```

## Setup / 环境安装

Use **Python 3.11 or 3.12**, preferably in a new virtual environment. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

For all six models (LSTM additionally requires TensorFlow):

```powershell
pip install -r requirements-lstm.txt
```

On macOS/Linux, create environment with `python3 -m venv .venv` and run `source .venv/bin/activate`.

## Test it without original SST files (synthetic only)

This only proves the engineering workflow executes; **generated data and results are fake/synthetic, not research evidence**.

```powershell
python -m sst_project.cli demo
python -m sst_project.cli preprocess --sst demo_synthetic/raw/sst/demo_monthly_sst.csv --nino demo_synthetic/raw/enso/demo_nino.ascii --output demo_synthetic/SST_processed.csv
python -m sst_project.cli train --data demo_synthetic/SST_processed.csv --output demo_synthetic/outputs --models LinearRegression SVR RandomForest XGBoost
python -m sst_project.cli plot --data demo_synthetic/SST_processed.csv --output demo_synthetic/outputs --map demo_synthetic/mean_sst_map.npz
pytest -q
```

For a more complete demo with SARIMAX and LSTM after installing TensorFlow, omit the `--models ...` option in `train`.

## Run with real data / 使用真实数据

1. Download monthly or daily SST NetCDF (`analysed_sst` or `sst` variable, with time/lat/lon coordinates) from the [Copernicus Climate Data Store satellite SST dataset](https://cds.climate.copernicus.eu/datasets/satellite-sea-surface-temperature?tab=overview) or the same specific product/version originally used, if known. Use 1993–2024. Put `.nc` files in `data/raw/sst/`. **Confirm product version/units/fill values, spatial coverage, land masking, and temporal aggregation against the source metadata.** The original exact product version is not recoverable from paper alone.
2. The NOAA CPC monthly Niño indices are available at [NOAA CPC ERSSTv5 monthly table](https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii); the **last column** is the Niño3.4 anomaly. Run:

```powershell
python -m sst_project.cli download-nino
python -m sst_project.cli extract-sst --source data/raw/sst
python -m sst_project.cli preprocess
python -m sst_project.cli train --models LinearRegression SVR RandomForest XGBoost SARIMAX
python -m sst_project.cli plot
```

3. To train all six models, additionally install TensorFlow and run:

```powershell
pip install -r requirements-lstm.txt
python -m sst_project.cli train
python -m sst_project.cli plot
```

**If you already have a reliable monthly regional-mean SST CSV**, provide `date,sst` columns and skip `extract-sst`:

```powershell
python -m sst_project.cli preprocess --sst path/to/your/monthly_sst.csv
```

Outputs appear in `outputs/tables/` and `outputs/figures/`. Because original raw data are absent, the current repository does **not** include real metrics or result figures. Data and large generated files are excluded in `.gitignore`.

## Reproducibility and limitations / 重要限制

- The full original code, SST NetCDFs, intermediate `SST_processed.csv`, fitted models, and original figures were not recovered. `docs/archived_paper_results.md` records **previously reported paper metrics only**, separate from future runs.
- The paper described “regional averaging,” but did not document a reproducible land-sea mask/weighting and exact original product. This reconstruction uses a **simple unweighted spatial mean**. It may differ from the original processing and should not be called exact replication.
- The paper mentioned linear interpolation of missing Niño values. To avoid use of future values in a prospective setting, this implementation instead applies **forward fill** to a missing Niño series; changes relative to the paper are intentional and documented.
- Target SST is never filled by interpolation. Consecutive monthly SST records are required for meaningful lag features.
- SVR, Random Forest and XGBoost tuning settings were reconstructed choices, **not claims about original hyperparameters**. Differences in model versions, random seeds, and source data mean evaluation metrics will differ.
- Reported paper 80/20 split may not have used the exact same warmup policy. Here all models are evaluated on identical dates for comparison.
- SARIMAX uses observed-history, expanding one-step evaluation with test observations appended after predictions; the input lags use measured SST. Results are **not** a free-running horizon forecast.
- SSTA monthly climatology is calculated over the complete data span **for diagnostics only** and is not used to train predictive models.
- A reported paper correlation (r≈0.059) and performance metrics should not be claimed reproduced until the actual SST/NOAA data are restored and rerun.

## Acknowledgment / Attribution

Original research topic and experimental specification from the author's historical manuscript. This repository's source is a 2026 reconstruction following disk loss. NOAA CPC and C3S/Copernicus are data providers and retain their own data-use terms.

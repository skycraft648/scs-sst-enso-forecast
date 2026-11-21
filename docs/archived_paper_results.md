# Reported metrics from the archived paper — NOT reproduced

The following table is transcribed from the prior project paper, which was
available in the author's ChatGPT file history. Original Python scripts and
input SST NetCDF/CSV were lost due to disk failure. This reconstructed code
has not been validated against those original inputs.

**Do NOT present these values as results produced by this GitHub repository.**

| Model | RMSE (°C) | MAE (°C) | R² | Pearson r | MAPE (%) |
|---|---:|---:|---:|---:|---:|
| SARIMAX | 0.321 | 0.263 | 0.946 | 0.978 | 0.924 |
| Linear Regression | 0.518 | 0.419 | 0.859 | 0.957 | 1.460 |
| SVR | 0.320 | 0.265 | 0.946 | 0.977 | 0.940 |
| Random Forest | 0.406 | 0.290 | 0.913 | 0.967 | 1.013 |
| XGBoost | 0.386 | 0.298 | 0.922 | 0.974 | 1.045 |
| LSTM | 0.303 | 0.250 | 0.952 | 0.980 | 0.878 |

The original paper also reported contemporaneous Niño3.4–SCS SSTA Pearson
correlation r≈0.059, but this value is not reproducible without the original data.
The present reconstruction makes additional implementation choices not
recoverable from the paper: random seeds; default SVR/tree hyperparameters;
missing-index policy (forward fill); simple spatial averaging; expanding
one-step SARIMAX evaluation. These choices may alter outputs materially.

"""Input processing for the reconstructed South China Sea SST project.

NOAA data and Copernicus SST are NOT included in this repository.
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

SST_BOUNDS = {'lat': (0.0, 25.0), 'lon': (105.0, 125.0)}


def parse_nino34(path: str | Path) -> pd.DataFrame:
    """Read the NOAA CPC ERSSTv5 index table; Niño3.4 ANOM is column #10.

    Format: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM.
    The 10th value (zero-index 9) is the Niño3.4 monthly anomaly, not SST.
    """
    records = []
    for line in Path(path).read_text(encoding='utf-8-sig', errors='replace').splitlines():
        parts = line.strip().split()
        if len(parts) < 10 or not re.fullmatch(r'\d{4}', parts[0]):
            continue
        try:
            year, month = int(parts[0]), int(parts[1])
            anom = float(parts[9])
        except ValueError:
            continue
        if 1 <= month <= 12 and -20 < anom < 20:
            records.append({'date': pd.Timestamp(year, month, 1), 'anom34': anom})
    if not records:
        raise ValueError('NOAA index file contains no valid rows; check ERSSTv5 format.')
    df = pd.DataFrame(records).drop_duplicates('date', keep='last').sort_values('date')
    return df.reset_index(drop=True)


def prepare_csv(sst_csv: str | Path, nino_ascii: str | Path, output: str | Path,
                start: str = '1993-01-01', end: str = '2024-12-01') -> pd.DataFrame:
    """Merge monthly SST and NOAA Niño3.4. Requires SST CSV: date,sst.

    Never invent/fill missing SST. For missing Niño3.4 use past-known ffill only;
    missing initial values are left as NA and later rejected by the model stage.
    """
    sst = pd.read_csv(sst_csv)
    cols = {c.lower().strip(): c for c in sst.columns}
    if 'date' not in cols or 'sst' not in cols:
        raise ValueError('Monthly SST CSV must have date and sst columns.')
    sst = sst.rename(columns={cols['date']: 'date', cols['sst']: 'sst'})[['date', 'sst']]
    sst['date'] = pd.to_datetime(sst['date']).dt.to_period('M').dt.to_timestamp()
    sst['sst'] = pd.to_numeric(sst['sst'], errors='coerce')
    if sst['date'].duplicated().any():
        raise ValueError('Duplicate months found in SST CSV. Please deduplicate with explicit rules.')
    sst = sst.sort_values('date').set_index('date')
    sst = sst.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if sst.empty:
        raise ValueError('SST data are empty in the specified period.')
    required_months = pd.date_range(sst.index.min(), sst.index.max(), freq='MS')
    if not sst.index.equals(required_months):
        missing = required_months.difference(sst.index)
        raise ValueError(f'Missing SST months: {list(missing[:10])}. No interpolation is performed.')
    if sst['sst'].isna().any():
        raise ValueError('SST contains NaNs; provide complete observations before training.')
    nino = parse_nino34(nino_ascii).set_index('date')
    merged = sst.join(nino, how='left')
    if merged['anom34'].isna().any():
        print(f'Warning: {merged.anom34.isna().sum()} Niño3.4 values missing; using causal forward fill.')
        merged['anom34'] = merged['anom34'].ffill()
    if merged['anom34'].isna().any():
        raise ValueError('Initial Niño3.4 entries missing; cannot forward-fill before first observation.')
    merged['Year'] = merged.index.year
    merged['Month'] = merged.index.month
    climatology = merged.groupby('Month')['sst'].transform('mean')
    merged['SSTA'] = merged['sst'] - climatology  # diagnostic only: not model input
    for lag in (1, 3, 6):
        merged[f'sst_lag{lag}'] = merged['sst'].shift(lag)
    merged.index.name = 'date'
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.reset_index().to_csv(out, index=False)
    return merged.reset_index()


def netcdf_to_monthly_sst(source: str | Path, output_csv: str | Path,
                          mean_map_npz: str | Path | None = None) -> pd.DataFrame:
    """Extract 0-25 N, 105-125 E monthly SST from one/many GHRSST-style NetCDFs.

    Spatial mean is unweighted for closer alignment with the original report's
    generic 'regional averaging' (the original weighting is not recoverable).
    Geographic metadata and units are inspected rather than silently assumed.
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise RuntimeError('Install xarray and netCDF4 to read NetCDF files.') from e
    source = Path(source)
    paths = sorted(source.rglob('*.nc')) if source.is_dir() else [source]
    paths = [p for p in paths if p.exists() and p.is_file()]
    if not paths:
        raise FileNotFoundError(f'No NetCDF files found at {source}')
    series, maps = [], []
    for path in paths:
        with xr.open_dataset(path, decode_times=True) as ds:
            candidates = ['analysed_sst', 'sst', 'sea_surface_temperature', 'tos']
            var = next((v for v in candidates if v in ds.data_vars), None)
            if not var:
                raise ValueError(f'{path.name}: no expected SST variable {candidates}')
            da = ds[var]
            renames = {}
            for alias, standard in [('latitude', 'lat'), ('longitude', 'lon')]:
                if alias in da.coords or alias in da.dims:
                    renames[alias] = standard
            da = da.rename(renames)
            if not {'lat', 'lon'}.issubset(set(da.coords)):
                raise ValueError(f'{path.name}: lat/lon must be coordinate names')
            if da['lat'].ndim != 1 or da['lon'].ndim != 1:
                raise ValueError('Curvilinear grids need a separate spatial mask implementation.')
            # Normalize longitude for grids using -180 to 180 convention.
            lon = da['lon'] % 360
            da = da.assign_coords(lon=lon).sortby('lon')
            da = da.where((da.lat >= 0) & (da.lat <= 25), drop=True)
            da = da.where((da.lon >= 105) & (da.lon <= 125), drop=True)
            if da.sizes.get('lat', 0) == 0 or da.sizes.get('lon', 0) == 0:
                raise ValueError(f'{path.name}: no data within South China Sea bounds')
            unit = str(da.attrs.get('units', '')).strip().lower()
            if unit in {'k', 'kelvin', 'degk', 'degree_kelvin'}:
                da = da - 273.15
            elif unit in {'c', '°c', 'degc', 'degrees_celsius', 'degree_celsius', 'celsius'}:
                pass
            else:
                sample = float(da.mean(skipna=True).values)
                if sample > 150:
                    da = da - 273.15
                else:
                    print(f'Warning: {path.name} SST units {unit!r} not recognized; treating as Celsius.')
            if 'time' not in da.dims:
                raise ValueError(f'{path.name}: a decoded time dimension is required')
            avg = da.mean(dim=('lat', 'lon'), skipna=True)
            timestamps = pd.to_datetime(avg.time.values)
            values = np.asarray(avg.values, dtype=float).reshape(-1)
            if len(timestamps) != len(values):
                raise ValueError(f'{path.name}: ambiguous additional dimensions')
            for date, value in zip(timestamps, values):
                series.append({'date': pd.Timestamp(date).to_period('M').to_timestamp(), 'sst': value})
            if mean_map_npz is not None:
                # Each map is represented by its per-file time-average; maps must be same grid.
                maps.append((np.asarray(da.lat.values), np.asarray(da.lon.values),
                             np.asarray(da.mean(dim='time', skipna=True).values, dtype=float),
                             da.sizes['time']))
    df = pd.DataFrame(series)
    if df.empty:
        raise ValueError('NetCDF files contain no SST values.')
    # If daily files are supplied, aggregate to calendar month; if monthly files are
    # duplicated, results need manual inspection for appropriate source selection.
    df = df.groupby('date', as_index=False)['sst'].mean().sort_values('date')
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if mean_map_npz and maps:
        lat, lon = maps[0][:2]
        if all(np.array_equal(lat, m[0]) and np.array_equal(lon, m[1]) for m in maps):
            total = np.zeros_like(maps[0][2])
            weight = np.zeros_like(total)
            for _, _, grid, n in maps:
                valid = np.isfinite(grid)
                total[valid] += grid[valid] * n
                weight[valid] += n
            mean = np.divide(total, weight, out=np.full_like(total, np.nan), where=weight > 0)
            dst = Path(mean_map_npz)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, lat=lat, lon=lon, mean_sst=mean)
        else:
            print('Warning: files use different grids; mean map skipped.')
    return df

"""Command line entry point: python -m sst_project.cli --help"""
import argparse
from pathlib import Path
from urllib.request import urlretrieve
from .data import netcdf_to_monthly_sst, prepare_csv
from .models import run_experiment, DEFAULT_MODELS
from .plots import diagnostic_plots, prediction_plots
from .demo import generate_demo

NOAA_URL = 'https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii'


def main():
    parser = argparse.ArgumentParser(description='Reconstructed SCS SST / ENSO project (original code lost).')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('download-nino', help='Download original NOAA ERSSTv5 monthly index')
    p.add_argument('--output', default='data/raw/enso/nino.ascii')
    p = sub.add_parser('extract-sst', help='Extract monthly SCS mean SST from NetCDF file(s)')
    p.add_argument('--source', default='data/raw/sst')
    p.add_argument('--output', default='data/processed/monthly_sst.csv')
    p.add_argument('--map-output', default='data/processed/mean_sst_map.npz')
    p = sub.add_parser('preprocess', help='Merge monthly SST and Niño3.4')
    p.add_argument('--sst', default='data/processed/monthly_sst.csv')
    p.add_argument('--nino', default='data/raw/enso/nino.ascii')
    p.add_argument('--output', default='data/processed/SST_processed.csv')
    p = sub.add_parser('train', help='Train and evaluate models')
    p.add_argument('--data', default='data/processed/SST_processed.csv')
    p.add_argument('--output', default='outputs')
    p.add_argument('--models', nargs='+', choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    p = sub.add_parser('plot', help='Generate figure PNGs from processed data and predictions')
    p.add_argument('--data', default='data/processed/SST_processed.csv')
    p.add_argument('--output', default='outputs')
    p.add_argument('--map', default='data/processed/mean_sst_map.npz')
    p = sub.add_parser('demo', help='Create labeled synthetic data for a smoke test only')
    p.add_argument('--folder', default='demo_synthetic')
    args = parser.parse_args()
    if args.command == 'download-nino':
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(NOAA_URL, str(dest))
        print('NOAA file saved:', dest)
    elif args.command == 'extract-sst':
        df = netcdf_to_monthly_sst(args.source, args.output, args.map_output)
        print('Monthly SST rows:', len(df))
    elif args.command == 'preprocess':
        df = prepare_csv(args.sst, args.nino, args.output)
        print('Processed rows:', len(df), 'output:', args.output)
    elif args.command == 'train':
        metrics, _ = run_experiment(args.data, args.output, args.models)
        print(metrics.round(4).to_string(index=False))
        print('Outputs saved in:', args.output)
    elif args.command == 'plot':
        corr = diagnostic_plots(args.data, args.output, args.map)
        prediction_plots(args.output)
        print('Correlation Niño3.4-SSTA:', round(corr, 4))
    elif args.command == 'demo':
        print('*** SYNTHETIC TEST DATA ONLY — DO NOT USE FOR SCIENTIFIC CLAIMS ***')
        print('Created:', generate_demo(args.folder))


if __name__ == '__main__':
    main()

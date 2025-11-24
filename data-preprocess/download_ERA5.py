import xarray as xr
import numpy as np
import os
import dask
from dask.diagnostics import ProgressBar
import gcsfs
import sys

# Constants
STANDARD_GRAVITY = 9.80665  # m/s²
if len(sys.argv) < 2:
    print("usage: python extract_data.py year")
    exit()
YEAR = str(sys.argv[1])
OUTPUT_FILE = "daily_500Zg_1deg_{}_12UTC_NH.nc".format(YEAR)

# Configure Dask
dask.config.set(scheduler='threads')
from dask.config import set as dask_set_config
dask_set_config({'num_workers': 16})  # Or whatever your CPU can handle

# GCS Zarr access
fs = gcsfs.GCSFileSystem(token='anon')
store = fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
ds = xr.open_dataset(store, engine='zarr', consolidated=True, chunks={'time': 24})
print(ds.variables)
exit()

# Select time, level, and northern hemisphere
ds_sel = ds.sel(time=slice(f'{YEAR}-01-01', f'{YEAR}-12-31'), level=500, latitude=slice(90, 0))
geopotential_500 = ds_sel['geopotential']

# Subsample to fixed hour (12 UTC)
Zg_12UTC = geopotential_500.sel(time=geopotential_500['time'].dt.hour == 12)
Zg_12UTC.name = '500Zg'
bad_times = Zg_12UTC['time'].values[np.isnat(Zg_12UTC['time'].values)]
print("Bad timestamps:", bad_times)

# Prepare time steps
all_times = Zg_12UTC['time'].values
valid_times = all_times[~np.isnat(all_times)]
days = valid_times

# Create output file if it doesn't exist
if not os.path.exists(OUTPUT_FILE):
    first_day = Zg_12UTC.isel(time=0) / STANDARD_GRAVITY
    first_day_ds = first_day.coarsen(latitude=4, longitude=4, boundary='trim').mean().to_dataset(name='500Zg')
    first_day_ds['time'] = [days[0]]
    first_day_ds.to_netcdf(OUTPUT_FILE, mode='w')
    processed_times = {np.datetime_as_string(days[0], unit='s')}
else:
    # Read processed times from existing file
    with xr.open_dataset(OUTPUT_FILE) as existing_ds:
        processed_times = set(np.datetime_as_string(existing_ds['time'].values, unit='s'))

# Process and append remaining time steps
with ProgressBar():
    for i, t in enumerate(days):
        t_str = np.datetime_as_string(t, unit='s')
        if t_str in processed_times:
            continue

        print(f"Processing {t_str} ({i+1}/{len(days)})")
        try:
            Zg = (Zg_12UTC.sel(time=t) / STANDARD_GRAVITY).coarsen(latitude=4, longitude=4, boundary='trim').mean()
            Zg_ds = Zg.to_dataset(name='500Zg')
            Zg_ds['time'] = [t]

            Zg_ds.to_netcdf(OUTPUT_FILE, mode='a')
        except Exception as e:
            print(f"Failed to process {t_str}: {e}")
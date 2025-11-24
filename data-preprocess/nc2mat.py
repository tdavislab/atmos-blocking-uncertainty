from matplotlib import pyplot as plt
import scipy.fft
from scipy.stats import linregress
import scipy
import netCDF4 as nc
import numpy as np
import copy
import os
import sys
from tqdm import tqdm

from data_utils import *

def compute_date_dict(data):
    # data.time_strs[yr] contains [isotime] information
    # To obtain all the unique days throughout the year,
    unique_days = list(set([(time_obj.month, time_obj.day) for time_strs in data.time_strs for time_obj in time_strs]))
    unique_days.sort()
    ndays = len(unique_days)
    day_dict = {(mo, da): i for i, (mo, da) in enumerate(unique_days)}
    
    return day_dict


def compute_DM(data: Data, ltdm=None, ltds=None, nFourier=6):
    """_summary_

    Args:
        data (Data): the netCDF4 raw data
        smooth_data (Data, optional): a smooth version that has been processed. Defaults to None.
        do_linregress (bool, optional): whether to do linear regression to remove the global warming trend. Defaults to True.
    
    Output:
        ltdm_smooth: (np.array((nlats, nlons, ndays))) long-term daily mean at a specific lat/lon location. 
               NOTE: This is smoothed using the FFT
    """
    
    nlats = len(data.lats)
    nlons = len(data.lons)
    
    # data.time_strs[yr] contains [isotime] information
    # To obtain all the unique days throughout the year,
    unique_days = list(set([(time_obj.month, time_obj.day) for time_strs in data.time_strs for time_obj in time_strs]))
    unique_days.sort()
    ndays = len(unique_days)
    day_dict = {(mo, da): i for i, (mo, da) in enumerate(unique_days)}
    
    if ltdm is None or ltds is None:
        dm = np.zeros((nlats, nlons, ndays))
        ds = np.zeros((nlats, nlons, ndays))
        print("Computing the daily mean at each location:")
        for ilat, lat in enumerate(tqdm(data.lats, desc="Latitude loop")):
            for ilon, lon in enumerate(data.lons):
                vals = np.zeros((ndays, ))
                cnts = np.zeros((ndays, ))
                vals_sq = np.zeros((ndays, ))
                for iyear, time_strs_by_year in enumerate(data.time_strs):
                    orders = [day_dict[(time_obj.month, time_obj.day)] for time_obj in time_strs_by_year]
                    for iday, day_order in enumerate(orders):
                        if not data.is_nan(iyear, iday, ilat, ilon):
                            v = data.fields[iyear][iday, ilat, ilon]
                            vals[day_order]   += v
                            vals_sq[day_order] += v * v        
                            cnts[day_order]   += 1
                        
                dm[ilat][ilon][:] = vals / cnts
                with np.errstate(invalid="ignore", divide="ignore"):
                    var = (vals_sq / cnts) - (dm[ilat, ilon, :]**2)
                var = np.clip(var, 0.0, None)
                ds[ilat, ilon, :] = np.sqrt(var)
    else:
        dm = ltdm.copy()
        ds = ltds.copy()
            
    # Next, we compute the smoothed long-term daily mean by computing the first six fourier components
    dm_smooth = np.zeros((nlats, nlons, ndays))
    print("Computing the smoothed daily mean using Fourier Transform, # Fourier harmonics =", nFourier)
    for ilat, lat in enumerate(tqdm(data.lats, desc="Latitude loop")):
        for ilon, lon in enumerate(data.lons):
            dm_yearly_seq_at_latlon = dm[ilat, ilon]
            dm_yearly_seq_at_latlon_ft = scipy.fft.rfft(dm_yearly_seq_at_latlon)
            dm_yearly_seq_at_latlon_ft_smooth = dm_yearly_seq_at_latlon_ft.copy()
            dm_yearly_seq_at_latlon_ft_smooth[nFourier:] = 0
            dm_yearly_seq_at_latlon_smooth = scipy.fft.irfft(dm_yearly_seq_at_latlon_ft_smooth, ndays)
            dm_smooth[ilat, ilon] = np.real(dm_yearly_seq_at_latlon_smooth)
    
    ds_smooth = np.zeros((nlats, nlons, ndays))
    print("Computing the smoothed daily std via smoothing variance with Fourier harmonics =", nFourier)
    for ilat, lat in enumerate(tqdm(data.lats, desc="Latitude loop (std)")):
        for ilon, lon in enumerate(data.lons):
            # work with variance to avoid negative artifacts after iFFT
            var_seq = ds[ilat, ilon]**2
            var_ft = scipy.fft.rfft(var_seq)
            var_ft_smooth = var_ft.copy()
            var_ft_smooth[nFourier:] = 0
            var_smooth = scipy.fft.irfft(var_ft_smooth, ndays)
            var_smooth = np.clip(np.real(var_smooth), 0.0, None)
            ds_smooth[ilat, ilon] = np.sqrt(var_smooth)
    
    return dm, dm_smooth, ds, ds_smooth

def compute_anomaly(data: Data, ltdm_smooth: np.array, data_anomaly_fields=None, do_linregress=False):
    
    def anomalies_weighted_mean(data: np.ndarray, lats: list[float], lons: list[float]):
        """
        Computes the latitude-weighted mean of geopotential anomalies along longitudes,
        ignoring np.nan values in the input data.

        Args:
            data (np.ndarray): Geopotential anomaly data of shape (n_lats, n_lons) at a single time step.
            lats (list[float]): List of latitude values.
            lons (list[float]): List of longitude values.

        Returns:
            np.ndarray: Weighted mean anomaly along each longitude (shape: [n_lons]).
        """
        nlats = len(lats)
        nlons = len(lons)
        assert data.shape == (nlats, nlons)

        lat_weights = np.cos(np.radians(lats))  # shape: [nlats]
        weighted_sum = np.zeros(nlons)
        weight_sum = np.zeros(nlons)

        for i in range(nlats):
            for j in range(nlons):
                val = data[i, j]
                if not np.isnan(val):
                    weighted_sum[j] += val * lat_weights[i]
                    weight_sum[j] += lat_weights[i]

        # Avoid division by zero where all values are NaN
        with np.errstate(invalid='ignore', divide='ignore'):
            weighted_mean = np.divide(weighted_sum, weight_sum)
            weighted_mean[np.isnan(weight_sum)] = np.nan  # If no valid data, return nan

        return weighted_mean

    # data.time_strs[yr] contains [time_obj] information
    # To obtain all the unique days throughout the year,
    unique_days = list(set([(time_obj.month, time_obj.day) for time_strs in data.time_strs for time_obj in time_strs]))
    unique_days.sort()
    day_dict = {(mo, da): i for i, (mo, da) in enumerate(unique_days)}
    
    if data_anomaly_fields is None:
        data_anomaly_fields = copy.deepcopy(data.fields)
        print("Subtracting Long-term Daily Mean to obtain the anomaly data:")
        for ilat, lat in enumerate(tqdm(data.lats, desc="Latitude loop")):
            for ilon, lon in enumerate(data.lons):
                for iyear, time_strs_by_year in enumerate(data.time_strs):
                    orders = [day_dict[(time_obj.month, time_obj.day)] for time_obj in time_strs_by_year]
                    for iday, day_order in enumerate(orders):
                        if not data.is_nan(iyear, iday, ilat, ilon):
                            data_anomaly_fields[iyear][iday, ilat, ilon] = data.fields[iyear][iday, ilat, ilon] - ltdm_smooth[ilat, ilon, day_order]
                        
    if do_linregress:
        data_anomaly_fields_dtrnd = copy.deepcopy(data_anomaly_fields)

        nlats = len(data.lats)
        nlons = len(data.lons)
        ntimes = data.get_total_times()
        means = np.zeros((ntimes, nlons))
        day_cnt = 0
        print("Computing daily weighted anomaly mean for each longitude:")
        for iyear, time_strs_by_year in enumerate(tqdm(data.time_strs, desc="Year loop")):
            for iday, _ in enumerate(time_strs_by_year):
                means[day_cnt] = anomalies_weighted_mean(data_anomaly_fields[iyear][iday, :, :], data.lats, data.lons)
                day_cnt += 1

        print("Detrending the data:")
        means_y = np.average(means, axis=1)
        slope, intercept, rvalue, pvalue, stderr = linregress(np.arange(ntimes), means_y)
        print("Slope, Intercept = ", slope, intercept)
        
        for ilon, lon in enumerate(tqdm(data.lons, desc="Longitude loop")):         
            # linear regression / detrending
            day_cnt = 0
            for iyear, time_strs_by_year in enumerate(data.time_strs):
                for iday, _ in enumerate(time_strs_by_year):
                    data_anomaly_fields_dtrnd[iyear][iday, :, ilon] = data_anomaly_fields[iyear][iday, :, ilon] - slope * (day_cnt) - intercept
                    if np.any(np.isnan(data_anomaly_fields_dtrnd[iyear][iday, :, :])):
                        data_anomaly_fields_dtrnd[iyear][iday, :, :] = interpolate_nans_2d(data_anomaly_fields_dtrnd[iyear][iday, :, :])
                    day_cnt += 1
            
        return data_anomaly_fields, data_anomaly_fields_dtrnd
    else:
        return data_anomaly_fields, None
    

def save_data_anomaly_data(data: Data, data_anomaly_fields: list[np.ndarray], root_path: str) -> None:
    assert len(data.years_list) == len(data_anomaly_fields)
    for iyear, year in enumerate(tqdm(data.years_list, desc="Saving data by years...")):
        year_path = os.path.join(root_path, str(year))
        os.makedirs(year_path, exist_ok=True)
        
        # for each data field at a time step, we need to save the following information
        # day - time: str
        # scalar field: np.ndarray(nlats, nlons)
        # latitudes: np.ndarray(nlats)
        # longitudes: np.ndarray(nlons)
        # The saved data should be easily loadable without additional packages
        for iday, time_obj in enumerate(data.time_strs[iyear]):
            time_str = time_obj.isoformat().replace(":", "=")
            field = data_anomaly_fields[iyear][iday, :, :]
            out_filename = os.path.join(year_path, f"500zg_anom_dtrnd_{time_str}.npz")
            np.savez(out_filename, time=time_obj.isoformat(), data=field, lats=data.lats, lons=data.lons)
            

def compute_normalization(data: Data, data_anomaly_fields: list[np.ndarray], data_std_fields: np.ndarray):
    """
    For every pixel, we normalize the function value on each date within the range of JJA for all years.
    Exact date range: 
        - UKESM: 05/27 - 09/04
        - ERA5:  05/26 - 09/06
    Steps:
    1. Collect the range of data at each pixel within the JJA date range
    2. Compute the normalized field for dates within the JJA date range
      * If the date is outside the range, let's not touch it (to save time)
      * We will not store the instances that are not normalized on saving (if saving the normalized data).
        
    Output: (JJA) normalized data_anomaly_fields.
    """
    if "UKESM" in data.keyword:
        start_month, start_day = 1, 1
        end_month, end_day = 12, 31
    else:
        start_month, start_day = 1, 1
        end_month, end_day = 12, 31
        
    def is_in_season(date):
        """Check if a cftime date is within a month-day range, ignoring year."""
        md = (date.month, date.day)
        start_md = (start_month, start_day)
        end_md = (end_month, end_day)

        if start_md <= end_md:
            return start_md <= md <= end_md
        else:
            # For ranges that span the new year (e.g., Nov–Feb)
            return md >= start_md or md <= end_md
    
    # data.time_strs[yr] contains [isotime] information
    # To obtain all the unique days throughout the year,
    unique_days = list(set([(time_obj.month, time_obj.day) for time_strs in data.time_strs for time_obj in time_strs]))
    unique_days.sort()
    ndays = len(unique_days)
    day_dict = {(mo, da): i for i, (mo, da) in enumerate(unique_days)}
    
    normalized_fields = copy.deepcopy(data_anomaly_fields)
    
    min_value = 100
    std_dev_rate = 1.0 if "UKESM" in data.keyword else 1.0
    
    print("Computing the normalized Zg_anomaly field")
    for ilat, lat in enumerate(tqdm(data.lats, desc="Latitude loop")):
        for ilon, lon in enumerate(data.lons):
            for iyear, time_strs_by_year in enumerate(data.time_strs):
                for iday, time_obj in enumerate(time_strs_by_year):
                    if is_in_season(time_obj) and (not data.is_nan(iyear, iday, ilat, ilon)):
                        val = data_anomaly_fields[iyear][iday, ilat, ilon]
                        normalized_fields[iyear][iday, ilat, ilon] = (val) / max(min_value, std_dev_rate * data_std_fields[ilat, ilon, iday])
    
    return normalized_fields
            
            
def compute_normalization_summer_deprecated(data: Data, data_anomaly_fields: list[np.ndarray]):
    """
    For every pixel, we normalize the function value on each date within the range of JJA for all years.
    Exact date range: 
        - UKESM: 05/27 - 09/04
        - ERA5:  05/26 - 09/06
    Steps:
    1. Collect the range of data at each pixel within the JJA date range
    2. Compute the normalized field for dates within the JJA date range
      * If the date is outside the range, let's not touch it (to save time)
      * We will not store the instances that are not normalized on saving (if saving the normalized data).
        
    Output: (JJA) normalized data_anomaly_fields.
    """
    if "UKESM" in data.keyword:
        start_month, start_day = 5, 27
        end_month, end_day = 9, 4
    else:
        start_month, start_day = 5, 26
        end_month, end_day = 9, 6
        
    def is_in_season(date):
        """Check if a cftime date is within a month-day range, ignoring year."""
        md = (date.month, date.day)
        start_md = (start_month, start_day)
        end_md = (end_month, end_day)

        if start_md <= end_md:
            return start_md <= md <= end_md
        else:
            # For ranges that span the new year (e.g., Nov–Feb)
            return md >= start_md or md <= end_md
    
    # data.time_strs[yr] contains [isotime] information
    # To obtain all the unique days throughout the year,
    unique_days = list(set([(time_obj.month, time_obj.day) for time_strs in data.time_strs for time_obj in time_strs]))
    unique_days.sort()
    ndays = len(unique_days)
    day_dict = {(mo, da): i for i, (mo, da) in enumerate(unique_days)}
    
    value_ranges_at_loc = [[[None, None] for lon in data.lons] for lat in data.lats]
    
    normalized_fields = copy.deepcopy(data_anomaly_fields)
    
    print("Computing the value range at each pixel:")
    for ilat, lat in enumerate(tqdm(data.lats, desc="Latitude loop")):
        for ilon, lon in enumerate(data.lons):
            for iyear, time_strs_by_year in enumerate(data.time_strs):
                for iday, time_obj in enumerate(time_strs_by_year):
                    if is_in_season(time_obj) and (not data.is_nan(iyear, iday, ilat, ilon)):
                        val = data_anomaly_fields[iyear][iday, ilat, ilon]
                        if value_ranges_at_loc[ilat][ilon][0] is None:
                            value_ranges_at_loc[ilat][ilon][0] = val
                            value_ranges_at_loc[ilat][ilon][1] = val
                        else:
                            value_ranges_at_loc[ilat][ilon][0] = min(value_ranges_at_loc[ilat][ilon][0], val)
                            value_ranges_at_loc[ilat][ilon][1] = max(value_ranges_at_loc[ilat][ilon][1], val)
    
    ## Step 2: normalize the value within the JJA range
    print("Normalizing the value by the range at each pixel:")
    for ilat, lat in enumerate(tqdm(data.lats, desc="Latitude loop")):
        for ilon, lon in enumerate(data.lons):
            min_at_loc, max_at_loc = value_ranges_at_loc[ilat][ilon]
            for iyear, time_strs_by_year in enumerate(data.time_strs):
                for iday, time_obj in enumerate(time_strs_by_year):
                    if is_in_season(time_obj) and (not data.is_nan(iyear, iday, ilat, ilon)):
                        val = data_anomaly_fields[iyear][iday, ilat, ilon]
                        normalized_fields[iyear][iday, ilat, ilon] = (val - min_at_loc) / (max_at_loc - min_at_loc)
                    else:
                        normalized_fields[iyear][iday, ilat, ilon] = None
    
    return normalized_fields
    

import matplotlib.pyplot as plt

def debug_vis_ltdm(ltdm, ltdm_smooth, ilat, ilon, sample_lat, sample_lon):
    plt.plot(ltdm[ilat, ilon], label="Original")
    plt.plot(ltdm_smooth[ilat, ilon], label="Smoothed (6 FTs)")
    plt.legend()
    plt.title("Fourier Smoothed Long-term Daily Mean at ({}, {})".format(sample_lat, sample_lon))
    plt.show()
    

def main(argv):
    if len(argv) < 1:
        print("Usage: python nc2mat.py data_fname [st_year] [ed_year] [-normalize]")
        return
    
    fname = argv[0]
    st_year = ed_year = None
    try:
        st_year = int(argv[1])
        ed_year = int(argv[2])
    except:
        print("Using default start and end year:", st_year, ed_year)
        
    normalize = False
    for arg in argv:
        if "normalize" in arg:
            normalize = True
            break
    if normalize:
        print("Will normalize pixel data based on long-term standard deviation.")
    else:
        print("Will NOT normalize pixel data based on long-term standard deviation.")
    
    print("fname:",fname)
    infoClass = DataInfo()
    data = infoClass.create_data(fname, st_year, ed_year, min_lat=0, max_lat=75)
    
    ltdm_dir = "./ltdm"
    os.makedirs(ltdm_dir, exist_ok=True)
    
    ltdm_filename = os.path.join(ltdm_dir, "ltdm_" + os.path.splitext(fname)[0] + ".npz")
    ltds_filename = os.path.join(ltdm_dir, "ltds_" + os.path.splitext(fname)[0] + ".npz")
    
    if not os.path.exists(ltdm_filename) or (not os.path.exists(ltds_filename)):
        ltdm, ltdm_smooth, ltds, ltds_smooth = compute_DM(data)
        np.savez(ltdm_filename, ltdm=ltdm, ltdm_smooth=ltdm_smooth)
        np.savez(ltds_filename, ltds=ltds, ltds_smooth=ltds_smooth)
    else:
        ltdm_load = np.load(ltdm_filename)
        ltdm = ltdm_load['ltdm']
        ltdm_smooth = ltdm_load['ltdm_smooth']
        ltds_load = np.load(ltds_filename)
        ltds = ltds_load['ltds']
        ltds_smooth = ltds_load['ltds_smooth']
    
    # Visualize the ltdm and smoothed ltdm at a specific (lat, lon) location
    smooth_loss = np.zeros(ltdm.shape[:2])
    smooth_diff = ltdm - ltdm_smooth
    for ilat in range(smooth_loss.shape[0]):
        for ilon in range(smooth_loss.shape[1]):
            smooth_loss[ilat, ilon] = np.linalg.norm(smooth_diff[ilat, ilon, :])
    
    # Flatten the array and get indices of the top 10 values
    top_n = 10
    flat_indices = np.argpartition(smooth_loss.ravel(), -top_n)[-top_n:]

    # Convert flat indices back to 2D indices
    top_indices_2d = np.column_stack(np.unravel_index(flat_indices, smooth_loss.shape))

    # Optional: sort by value descending
    top_indices_2d = top_indices_2d[np.argsort(smooth_loss[tuple(top_indices_2d.T)])[::-1]]

    print("Top 10 values and their positions:")
    for idx in top_indices_2d:
        print(f"Value: {smooth_loss[tuple(idx)]:.4f}, Position: {tuple(idx)}")
        # debug_vis_ltdm(ltdm, ltdm_smooth, idx[0], idx[1], data.lats[idx[0]], data.lons[idx[1]])
    
    # Now, we subtract the original field data by ltdm_smooth to obtain the anomaly
    # It is quite optional to subtract the linear regression data (reflecting the global warming effect).
    # The time sequence is not that long, and the global warming effect trend is minor.
    ltdm_anom_filename = os.path.join(ltdm_dir, "ltdm_anom_" + os.path.splitext(fname)[0] + ".npz")
    ltdm_anom_dtrnd_filename = os.path.join(ltdm_dir, "ltdm_anom_dtrnd_" + os.path.splitext(fname)[0] + ".npz")
    day_dict_filename = os.path.join(ltdm_dir, "day_dict_" + os.path.splitext(fname)[0] + ".npz")
    
    if not os.path.exists(day_dict_filename):
        day_dict = compute_date_dict(data)
        np.savez(day_dict_filename, day_dict=np.array(day_dict, dtype=object))
    
    if not (os.path.exists(ltdm_anom_filename) and os.path.exists(ltdm_anom_dtrnd_filename)):
        data_anomaly, data_anomaly_dtrnd = compute_anomaly(data, ltdm_smooth, do_linregress=True)
        if "UKESM" in data.keyword:
            np.savez(ltdm_anom_filename, ltdm_anom=data_anomaly)
            np.savez(ltdm_anom_dtrnd_filename, ltdm_anom_dtrnd=data_anomaly_dtrnd)
        else:
            # Save your list-of-arrays with known ordering
            np.savez(ltdm_anom_filename, ltdm_anom=np.array(data_anomaly, dtype=object))
            np.savez(ltdm_anom_dtrnd_filename, ltdm_anom_dtrnd=np.array(data_anomaly, dtype=object))
    else:
        if "UKESM" in data.keyword:
            ltdm_anom_load = np.load(ltdm_anom_filename)
            data_anomaly = ltdm_anom_load["ltdm_anom"]
            ltdm_anom_dtrnd_load = np.load(ltdm_anom_dtrnd_filename)
            data_anomaly_dtrnd = ltdm_anom_dtrnd_load["ltdm_anom_dtrnd"]
        else:
            ltdm_anom_load = np.load(ltdm_anom_filename, allow_pickle=True)
            data_anomaly = list(ltdm_anom_load["ltdm_anom"])
            ltdm_anom_dtrnd_load = np.load(ltdm_anom_dtrnd_filename, allow_pickle=True)
            data_anomaly_dtrnd = list(ltdm_anom_dtrnd_load["ltdm_anom_dtrnd"])
        
        # data_anomaly, data_anomaly_dtrnd = compute_anomaly(data, ltdm_smooth, data_anomaly, do_linregress=True)
        # np.savez(ltdm_anom_filename, ltdm_anom=data_anomaly)
        # np.savez(ltdm_anom_dtrnd_filename, ltdm_anom_dtrnd=data_anomaly_dtrnd)
    
    # If we normalize all summer data
    if normalize:
        ltdm_anom_dtrnd_normalized_JJA_filename = os.path.join(ltdm_dir, "ltdm_anom_dtrnd_normalized_JJA_" + os.path.splitext(fname)[0] + ".npz")
        
        data_anomaly_dtrnd_normalized_JJA = compute_normalization(data, data_anomaly_dtrnd, ltds_smooth)
        if "UKESM" in data.keyword:
            np.savez(ltdm_anom_dtrnd_normalized_JJA_filename, ltdm_anom_dtrnd_normalized_JJA=data_anomaly_dtrnd_normalized_JJA)
        else:
            # Save your list-of-arrays with known ordering
            np.savez(ltdm_anom_dtrnd_normalized_JJA_filename, ltdm_anom_dtrnd_normalized_JJA=np.array(data_anomaly_dtrnd_normalized_JJA, dtype=object))
            
        final_data_root = "UKESM-normalize" if "UKESM" in data.keyword else "ERA5-normalize"
        os.makedirs(final_data_root, exist_ok=True)
        save_data_anomaly_data(data, data_anomaly_dtrnd_normalized_JJA, final_data_root)
    else:
        # Finally, we save the dtrnd anomaly data (or simply anomaly itself) into a data file for each day
        final_data_root = "UKESM" if "UKESM" in data.keyword else "ERA5"
        os.makedirs(final_data_root, exist_ok=True)
        save_data_anomaly_data(data, data_anomaly_dtrnd, final_data_root)

if __name__ == '__main__':
    main(sys.argv[1:])

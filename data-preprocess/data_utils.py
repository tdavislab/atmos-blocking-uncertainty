import numpy as np
import netCDF4 as nc
import cftime
import fnmatch
from datetime import datetime
from typing import Optional
from scipy.interpolate import griddata

def convert_time_array(time_values, units, calendar="gregorian"):
    """
    Convert an array of time values from NetCDF to (year, month, day) tuples.

    Args:
        time_values (array-like): Numeric time values (e.g., from NetCDF variable).
        units (str): Time units (e.g., "days since 1850-01-01", "hours since 1900-01-01").
        calendar (str): Calendar type ("gregorian", "360_day", etc.).

    Returns:
        List of tuples: [(year, month, day), ...]
    """
    # Use cftime to convert time values
    dates = cftime.num2date(time_values, units=units, calendar=calendar)

    # Extract (year, month, day)
    return list(dates)

def latlon_to_unit_sphere(lat_deg, lon_deg):
    """
    Convert latitude and longitude to 3D Cartesian coordinates on a unit sphere.

    Args:
        lat_deg (float or array-like): Latitude(s) in degrees.
        lon_deg (float or array-like): Longitude(s) in degrees.

    Returns:
        Tuple of (x, y, z) coordinates on the unit sphere.
    """
    # Convert degrees to radians
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    # Spherical to Cartesian conversion
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return x, y, z

def get_start_of_year_iso(st_year: Optional[int]) -> Optional[str]:
    """
    Return the ISO format datetime string for the start of a given year.

    Args:
        st_year (int): The year (e.g., 2025)

    Returns:
        str: ISO 8601 datetime string (e.g., "2025-01-01T00:00:00")
    """
    if st_year is None:
        return None
    dt = datetime(st_year, 1, 1, 0, 0, 0)
    return dt

def get_end_of_year_iso(ed_year: Optional[int]) -> Optional[str]:
    """
    Return the ISO format datetime string for the end of a given year.

    Args:
        st_year (int): The year (e.g., 2025)

    Returns:
        str: ISO 8601 datetime string (e.g., "2025-12-31T23:59:59")
    """
    if ed_year is None:
        return None
    dt = datetime(ed_year, 12, 31, 23, 59, 59)
    return dt

def longitude_difference(lon1, lon2):
    """
    Computes the smallest difference in degrees between two longitudes on a 0–360 scale.

    Parameters:
        lon1 (float): First longitude (in degrees, range 0–360).
        lon2 (float): Second longitude (in degrees, range 0–360).

    Returns:
        float: The minimal angular difference between the two longitudes (0–180).
    """
    lon1 = lon1 % 360
    lon2 = lon2 % 360
    diff = abs(lon1 - lon2)
    return min(diff, 360 - diff)

from scipy.interpolate import griddata

def interpolate_nans_2d(data, method='linear'):
    """
    Fill np.nan entries in a 2D array via interpolation.

    Parameters:
        data : np.ndarray
            2D array with possible np.nan entries.
        method : str
            Interpolation method: 'linear', 'nearest', or 'cubic'.

    Returns:
        np.ndarray : New array with interpolated values.
    """
    x, y = np.indices(data.shape)
    valid_mask = ~np.isnan(data)

    interpolated = griddata(
        (x[valid_mask], y[valid_mask]),
        data[valid_mask],
        (x[~valid_mask], y[~valid_mask]),
        method=method
    )

    filled_data = data.copy()
    filled_data[~valid_mask] = interpolated
    return filled_data

# Example: time in "days since 1850-01-01" with a 360_day calendar
# time_values = np.array([0, 30, 60, 90])  # e.g., days after reference
# units = "days since 1850-01-01"
# calendar = "360_day"

# converted_dates = convert_time_array(time_values, units, calendar)
# print(converted_dates)  # [(1850, 1, 1), (1850, 2, 1), (1850, 3, 1), (1850, 4, 1)]

def binary_search_date(isotime, time_list, get_first=False, cmp_month=False, cmp_day=False):    
    def compare_iso_times(dt1, dt2):
        if dt1.year < dt2.year:
            return -1
        elif dt1.year > dt2.year:
            return 1
        
        if cmp_month:
            if dt1.month < dt2.month:
                return -1
            elif dt1.month > dt2.month:
                return 1
            
        if cmp_day:
            if dt1.day < dt2.day:
                return -1
            elif dt1.day > dt2.day:
                return 1
    
        return 0
    
    head = 0
    tail = len(time_list) - 1
    if isotime is None:
        return head if get_first else tail
        
    while head <= tail:
        mid = (head + tail) >> 1
        t_mid = time_list[mid]
        cmp_date = compare_iso_times(isotime, t_mid)
        if cmp_date < 0:
            tail = mid - 1
        elif cmp_date > 0:
            head = mid + 1
        else:
            if get_first:
                tail = mid - 1
            else:
                head = mid + 1
    return head if get_first else tail

class Data:
    def __init__(self):
        self.filename = None
        self.keyword = None
        self.lat_name = None
        self.lon_name = None
        self.time_name = None
        self.field_name = None
        self.field_info = None
        self.time_unit = None
        self.calendar = None
        self.num_years = 0
        self.lats = None
        self.lons = None
        self.times = None
        self.time_strs = None
        self.fields = None
        self.years_list = None
        self.has_nan = False
        self.missing_value = None
    
    def get_info(self):
        print(self.field_info)
    
    def get_total_times(self):
        ntimes = np.sum([len(each) for each in self.time_strs])
        return ntimes
        
    def is_nan(self, iyear, iday, ilat, ilon):
        return np.isnan(self.fields[iyear][iday, ilat, ilon])
        

class DataInfo:
    def __init__(self):
        self.filenames = ["500zg_1x1_1979-2019_12hr_NHML.nc",
                          "500zg_1x1_1979-2019_12hr_NHML_daymean.nc",
                          "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc",
                          "500zg_JJA_1x1_1979-2019_daymean_LTDManom_EurAR5_dtrnd_wrttime.nc",
                          "psl_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc",
                          "daily_500Zg_1deg_*_12UTC_NH.nc",
                          "daily_500Zg_1deg_*_daily_mean_NH.nc",
                          "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_*-*.nc",
                         ]
        
        self.lat_names = {
            "500zg_1x1_1979-2019_12hr_NHML.nc": "latitude",
            "500zg_1x1_1979-2019_12hr_NHML_daymean.nc": "latitude",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "lat",
            "500zg_JJA_1x1_1979-2019_daymean_LTDManom_EurAR5_dtrnd_wrttime.nc": "latitude",
            "psl_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "lat",
            "daily_500Zg_1deg_*_12UTC_NH.nc": "latitude",
            "daily_500Zg_1deg_*_daily_mean_NH.nc": "latitude",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_*-*.nc": "lat",
        }
        
        self.lon_names = {
            "500zg_1x1_1979-2019_12hr_NHML.nc": "longitude",
            "500zg_1x1_1979-2019_12hr_NHML_daymean.nc": "longitude",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "lon",
            "500zg_JJA_1x1_1979-2019_daymean_LTDManom_EurAR5_dtrnd_wrttime.nc": "longitude",
            "psl_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "lon",
            "daily_500Zg_1deg_*_12UTC_NH.nc": "longitude",
            "daily_500Zg_1deg_*_daily_mean_NH.nc": "longitude",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_*-*.nc": "lon",
        }
        
        self.time_names = {
            "500zg_1x1_1979-2019_12hr_NHML.nc": "time",
            "500zg_1x1_1979-2019_12hr_NHML_daymean.nc": "time",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "time",
            "500zg_JJA_1x1_1979-2019_daymean_LTDManom_EurAR5_dtrnd_wrttime.nc": "time",
            "psl_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "time",
            "daily_500Zg_1deg_*_12UTC_NH.nc": "time",
            "daily_500Zg_1deg_*_daily_mean_NH.nc": "time",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_*-*.nc": "time",
        }
        
        self.field_names = {
            "500zg_1x1_1979-2019_12hr_NHML.nc": "zg",
            "500zg_1x1_1979-2019_12hr_NHML_daymean.nc": "zg",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "zg",
            "500zg_JJA_1x1_1979-2019_daymean_LTDManom_EurAR5_dtrnd_wrttime.nc": "z",
            "psl_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "psl",
            "daily_500Zg_1deg_*_12UTC_NH.nc": "500Zg",
            "daily_500Zg_1deg_*_daily_mean_NH.nc": "500Zg",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_*-*.nc": "zg",
        }
        
        self.field_infos = {
            "500zg_1x1_1979-2019_12hr_NHML.nc": "JJA, 12-hr mean geopotential height, 1979-2019, observation data",
            "500zg_1x1_1979-2019_12hr_NHML_daymean.nc": "JJA, daily mean geopotential height, 1979-2019, observation data",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": 
                "JJA, daily mean geopotential height, 1960-2060, simulation data",
            "500zg_JJA_1x1_1979-2019_daymean_LTDManom_EurAR5_dtrnd_wrttime.nc": 
                "JJA, 12-hr geopotential height anomaly, 1979-2019, observation data, Europe only",
            "psl_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": 
                "JJA, daily mean sea level pressure, 1960-2060, simulation data",
            "daily_500Zg_1deg_*_12UTC_NH.nc": 
                "Year-round, geopotential height at 12:00 UTC at 500 hPA level, date varied by filename, ERA5 observation data",
            "daily_500Zg_1deg_*_daily_mean_NH.nc": 
                "Year-round, daily mean geopotential height at 500 hPA level, date varied by filename, ERA5 observation data",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_*-*.nc": 
                "Year-round, geopotential height at 12:00 UTC at 500 hPA level, date varied by filename, UK-ESM simulation data",
        }
        
        self.time_units = {
            "500zg_1x1_1979-2019_12hr_NHML.nc": "hours since 1900-01-01",
            "500zg_1x1_1979-2019_12hr_NHML_daymean.nc": "hours since 1900-01-01",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "days since 1850-01-01",
            "500zg_JJA_1x1_1979-2019_daymean_LTDManom_EurAR5_dtrnd_wrttime.nc": "days since 1979-06-01 10:30:00",
            "psl_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "days since 1850-01-01",
            "daily_500Zg_1deg_*_12UTC_NH.nc": "hours since 1900-01-01",
            "daily_500Zg_1deg_*_daily_mean_NH.nc": "days since 1979-01-01 00:00:00",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_*-*.nc": "days since 1960-01-01 12:00:00.000000",
        }
        
        self.calendars = {
            "500zg_1x1_1979-2019_12hr_NHML.nc": "gregorian",
            "500zg_1x1_1979-2019_12hr_NHML_daymean.nc": "gregorian",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "360_day",
            "500zg_JJA_1x1_1979-2019_daymean_LTDManom_EurAR5_dtrnd_wrttime.nc": "proleptic_gregorian",
            "psl_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": "360_day",
            "daily_500Zg_1deg_*_12UTC_NH.nc": "proleptic_gregorian",
            "daily_500Zg_1deg_*_daily_mean_NH.nc": "proleptic_gregorian",
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_*-*.nc": "360_day",
        }
        
        self.missing_values = {
            "500zg_1x1_1979-2019_12hr_NHML.nc": np.nan,
            "500zg_1x1_1979-2019_12hr_NHML_daymean.nc": np.nan,
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": np.nan,
            "500zg_JJA_1x1_1979-2019_daymean_LTDManom_EurAR5_dtrnd_wrttime.nc": np.nan,
            "psl_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd.nc": np.nan,
            "daily_500Zg_1deg_*_12UTC_NH.nc": np.nan,
            "daily_500Zg_1deg_*_daily_mean_NH.nc": np.nan,
            "500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_*-*.nc": 1e+20,
        }

        
    def create_data(self, fname, st_year=None, ed_year=None, min_lat=None, min_lon=None, max_lat=None, max_lon=None) -> Data:
        keyword = None
        for name_pattern in self.filenames:
            if fnmatch.fnmatch(fname, name_pattern):
                keyword = name_pattern
        if keyword is None:
            print("Data info/keyword does not exist in data_util:", fname)
            return None
        if (min_lon is not None and min_lon < 0) or (max_lon is not None and max_lon < 0):
            print("We use degrees_east for longitude. It has to be positive.")
            return None
        
        data = Data()
        data.filename = fname
        data.keyword = keyword
        data.lat_name = self.lat_names[keyword]
        data.lon_name = self.lon_names[keyword]
        data.time_name = self.time_names[keyword]
        data.field_name = self.field_names[keyword]
        data.field_info = self.field_infos[keyword]
        data.calendar = self.calendars[keyword]
        data.time_unit = self.time_units[keyword]
        data.missing_value = self.missing_values[keyword]
        
        ds = nc.Dataset(fname)
        
        t = np.asarray(ds.variables[data.time_name])
        t_str = convert_time_array(t, data.time_unit, data.calendar)
        
        print(t_str[0], t_str[364], t_str[728])
        
        lats = np.asarray(ds.variables[data.lat_name])
        lons = np.asarray(ds.variables[data.lon_name])
        
        # We create the slicing information based on the restriction arguments
        # Namely, we can create data with year, latitude, and longitude restrictions
        year_slice_st = 0
        year_slice_ed = len(t)
        lat_slice_st = 0 
        lat_slice_ed = len(lats)
        lon_slice_st = 0
        lon_slice_ed = len(lons)
        
        if st_year is not None or ed_year is not None:
            year_slice_st = binary_search_date(get_start_of_year_iso(st_year), t_str, get_first=True)
            year_slice_ed = binary_search_date(get_end_of_year_iso(ed_year), t_str, get_first=False) + 1
        if min_lat is not None or max_lat is not None:
            for i, lat in enumerate(lats):
                if min_lat is not None and lat >= min_lat:
                    lat_slice_st = i
                    min_lat = None
                if max_lat is not None and lat >= max_lat:
                    lat_slice_ed = i
                    max_lat = None
        if min_lon is not None or max_lon is not None:
            for i, lon in enumerate(lons):
                if min_lon is not None and lon >= min_lon:
                    lon_slice_st = i
                    min_lon = None
                if max_lon is not None and lon >= max_lon:
                    lon_slice_ed = i
                    max_lon = None
        
        print("We process data {} in the following range:".format(keyword))
        try:
            print("Time:", t_str[year_slice_st], t_str[year_slice_ed - 1])
        except:
            print("Time:", st_year, ed_year)
        print("Lats:", lats[lat_slice_st], lats[lat_slice_ed - 1], ", # samples:", lat_slice_ed - lat_slice_st)
        print("Lons:", lons[lon_slice_st], lons[lon_slice_ed - 1], ", # samples:", lon_slice_ed - lon_slice_st)
        
        # we divide the data into years
        data.times = []
        data.time_strs = []
        data.fields = []
        
        # dimension of field: (time, lat, lon)
        field_var = ds.variables[data.field_name]
        z = np.asarray(field_var[year_slice_st:year_slice_ed, lat_slice_st:lat_slice_ed, lon_slice_st:lon_slice_ed])
        data.lats = lats[lat_slice_st:lat_slice_ed]
        data.lons = lons[lon_slice_st:lon_slice_ed]
        # Rather than assigning them to data.times (...), we separate them by years
        times = t[year_slice_st:year_slice_ed]
        time_strs = t_str[year_slice_st:year_slice_ed]
        years_list = [each.year for each in t_str]
        
        data.times = list()
        data.time_strs = list()
        data.years_list = list()
        data.fields = list()
        data.has_nan = list()
        
        years_set = list(set(years_list))
        years_set.sort()
        
        for year in years_set:
            # TODO: separate the dates by years
            year_st = binary_search_date(get_start_of_year_iso(year), time_strs, get_first=True)
            year_ed = binary_search_date(get_end_of_year_iso(year), time_strs, get_first=False) + 1
            
            field_in_year = z[year_st : year_ed, :, :]
            field_dims = field_in_year.shape
            field_size = field_dims[0] * field_dims[1] * field_dims[2]
            data.times.append(times[year_st : year_ed])
            data.time_strs.append(time_strs[year_st : year_ed])
            data.years_list.append(year)
            
            if not np.isnan(self.missing_values[keyword]):
                if self.missing_values[keyword] > 0:
                    field_in_year[field_in_year >= self.missing_values[keyword]] = np.nan
                else:
                    field_in_year[field_in_year <= self.missing_values[keyword]] = np.nan
            
            data.fields.append(field_in_year)
            data.has_nan.append(np.isnan(field_in_year).sum())
            print(year, "# Nan value?", np.isnan(field_in_year).sum(), "Total size:", field_size)    
        
        return data
    
    def get_info(self, fname):
        print(self.field_infos[fname])
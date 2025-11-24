import os
import sys
import numpy as np
import json
import re
import csv
import cftime
from tqdm import tqdm
from matplotlib import pyplot as plt
from region_overlap_util import *
from boxplot_util import *

def load_scalar_fields_from_npz(directory, start_time, end_time, calendar):
    """
    Load a time sequence of scalar fields from .npz files in the given directory.
    Assumes each .npz file contains: 'field', 'lats', and 'lons'.

    Returns:
        fields: List of 2D numpy arrays (scalar fields).
        filenames: List of filenames corresponding to each field.
    """
    def is_in_season(date):
        """Check if a cftime date is within a month-day range, ignoring year."""
        md = (date.month, date.day)
        
        if start_time <= end_time:
            return start_time <= md <= end_time
        else:
            # For ranges that span the new year (e.g., Nov–Feb)
            return md >= start_time or md <= end_time
    
    def from_iso_to_cftime_manual(iso_str, calendar="360_day"):
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})', iso_str)
        if not match:
            raise ValueError("Invalid ISO datetime format")
        y, m, d, H, M, S = map(int, match.groups())
        return cftime.datetime(y, m, d, H, M, S, calendar=calendar)
        
    fields = []
    filenames = []
    dates = []
    lats = None
    lons = None

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".npz"):
            data = np.load(os.path.join(directory, filename))
            datestr = str(data["time"])
            date = from_iso_to_cftime_manual(datestr, calendar)
            if not is_in_season(date):
                continue
            
            fields.append(data["data"])
            filenames.append(filename)
            dates.append(date)
            lats = data["lats"]
            lons = data["lons"]

    return fields, filenames, dates, lats, lons


def load_ground_truth(keyword):
    gt_root = "C:/Users/mingz/Desktop/ResearchProjects/GWMT-Blocking/blocking-data/GTD"
    fname = "UKESM-ground-truth.csv" if "UKESM" in keyword else "ERA5-ground-truth.csv"
    csv_name = os.path.join(gt_root, fname)
    
    ground_truth = {}

    with open(csv_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = (int(row['YEAR']), int(row['MONTH']), int(row['DAY']))
            value = row['GroundTruth']  # Convert to int/float if needed
            ground_truth[key] = int(float(value))

    return ground_truth

def crop_field_to_region(field, lats, lons, lat_range, lon_range):
    """
    Restrict a scalar field to a specified latitude and longitude range.

    Parameters:
        field (np.ndarray): 2D array with shape (lat, lon)
        lats (np.ndarray): 1D array of latitude values
        lons (np.ndarray): 1D array of longitude values
        lat_range (tuple): (lat_min, lat_max)
        lon_range (tuple): (lon_min, lon_max)

    Returns:
        cropped_field (np.ndarray): 2D cropped field
        cropped_lats (np.ndarray): corresponding latitude array
        cropped_lons (np.ndarray): corresponding longitude array
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # Handle possible longitude wrap-around (e.g., -10 to 40 on 0-360 scale)
    if lon_min < 0 and np.any(lons > 180):
        lons = np.where(lons > 180, lons - 360, lons)

    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)

    cropped_field = field[np.ix_(lat_mask, lon_mask)]
    cropped_lats = lats[lat_mask]
    cropped_lons = lons[lon_mask]

    return cropped_field, cropped_lats, cropped_lons

def shift_longitude_deprecated(field, lons):
    """
    Shift a scalar field and 1D longitude array so that 0° longitude is centered.

    Parameters:
        field (2D array): Scalar field with shape (lat, lon).
        lons (1D array): Longitude array with values in [0, 360).

    Returns:
        shifted_field, shifted_lons: Both realigned with 0° longitude at center.
    """
    # Find the index to center 0° longitude
    lons_pos = np.array([each % 360 for each in lons])
    shift_idx = np.argmin(np.abs(lons_pos - 180))
    shifted_field = np.roll(field, -shift_idx, axis=1)
    shifted_lons = np.roll(lons, -shift_idx).copy()
    shifted_lons[shifted_lons > 180] -= 360
    return shifted_field, shifted_lons

def shift_longitude(field, lons):
    """
    Shift the scalar field and longitude array so that the smallest (westernmost)
    longitude is the first entry in the array.

    Parameters:
        field (2D array): Scalar field with shape (lat, lon).
        lons (1D array): Longitude array, in degrees.

    Returns:
        shifted_field (2D array): Field shifted along the longitude axis.
        shifted_lons (1D array): Longitude array starting from westernmost point.
    """
    lons = np.asarray(lons)
    # Wrap longitudes to [-180, 180) if needed
    lons_mod = ((lons + 180) % 360) - 180
    shift_idx = np.argmin(lons_mod)

    shifted_field = np.roll(field, -shift_idx, axis=1)
    shifted_lons = np.roll(lons_mod, -shift_idx)

    return shifted_field, shifted_lons

def shift_longitude_stack(fields, lons):
    """
    Apply shift_longitude to a stack of scalar fields.

    Parameters:
        fields (3D array): Shape (time, lat, lon)
        lons (1D array): Longitude values in [0, 360)

    Returns:
        shifted_fields, shifted_lons
    """
    shifted_fields = []
    for i in range(fields.shape[0]):
        f_shifted, lons_shifted = shift_longitude(fields[i], lons)
        shifted_fields.append(f_shifted)
    return np.stack(shifted_fields), lons_shifted

def generate_contours_from_mask(mask, lats, lons, level=0.5):
    """
    Generate contour polygons from a binary mask using matplotlib.
    Returns a list of Nx2 arrays representing contour lines in (lon, lat) space.
    """
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(lons, lats)
    cs = ax.contour(X, Y, mask.astype(float), levels=[level])
    plt.close(fig)

    contours = []
    for collection in cs.collections:
        for path in collection.get_paths():
            contours.append(path.vertices)
    return contours


def save_contours_to_tempfiles(output_dir, contour_data_dict):
    """
    Save contour polylines to temporary JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}

    for label, contours in contour_data_dict.items():
        label_file = os.path.join(output_dir, f"{label}.json")
        json_data = [polyline.tolist() for polyline in contours]
        with open(label_file, 'w') as f:
            json.dump(json_data, f)
        metadata[label] = label_file

    meta_file = os.path.join(output_dir, "contour_metadata.json")
    with open(meta_file, 'w') as f:
        json.dump(metadata, f)

    return metadata


def load_contours_from_tempfiles(meta_file):
    """
    Load contour polylines from metadata JSON file.
    """
    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    contour_data_dict = {}
    for label, path in metadata.items():
        with open(path, 'r') as f:
            contour_data_dict[label] = [np.array(polyline) for polyline in json.load(f)]

    return contour_data_dict

def extract_contours(data, years, lats, lons, threshold):
    shifted_data, shifted_lons = shift_longitude_stack(data, lons)
    assert len(years) == data.shape[0]
    
    contour_data = {
        str(year): generate_contours_from_mask(shifted_data[iyear] >= threshold, lats, shifted_lons) \
            for iyear, year in enumerate(years)
    }
    return contour_data


def extract_and_save_contour_boxplot_components(data, cbd_scores, years, lats, lons, threshold, output_dir):
    """
    Shift data, extract contours for contour boxplot components, and save them to JSON files.

    Parameters:
        data (3D np.array): Shape (samples, lat, lon), scalar field ensemble.
        cbd_scores (1D np.array): Relaxed cBD scores for each sample.
        lats (1D array): Latitude coordinates.
        lons (1D array): Longitude coordinates (expected in [0, 360)).
        threshold (float): Contour threshold value.
        output_dir (str): Output directory for temporary files.
    """
    # Shift all fields and get aligned longitudes
    shifted_data, shifted_lons = shift_longitude_stack(data, lons)
    assert np.all(data == shifted_data)
    assert np.all(lons == shifted_lons)

    n_samples = shifted_data.shape[0]
    sorted_indices = np.argsort(cbd_scores)[::-1]
    median_idx = sorted_indices[0]
    # if "0527" in output_dir:
    #     print(sorted_indices)
    #     print(cbd_scores)
    # print("Median:", median_idx)
    # print(years)
    central_indices = sorted_indices[:int(0.5 * n_samples)]
    outlier_indices = np.where(cbd_scores == 0)[0]
    # print("Outliers:", outlier_indices)
    inlier_indices = [i for i in range(n_samples) if i not in outlier_indices]

    masks = shifted_data >= threshold
    mean_mask = masks.sum(axis=0) > (n_samples // 2)
    central_union = np.any(masks[central_indices], axis=0)
    central_intersection = np.all(masks[central_indices], axis=0)
    inlier_union = np.any(masks[inlier_indices], axis=0)
    inlier_intersection = np.all(masks[inlier_indices], axis=0)

    band_50_mask = central_union.astype(int) - central_intersection.astype(int)
    band_100_mask = inlier_union.astype(int) - inlier_intersection.astype(int)

    contour_data = {
        "median": generate_contours_from_mask(shifted_data[median_idx] >= threshold, lats, shifted_lons),
        "mean": generate_contours_from_mask(mean_mask, lats, shifted_lons),
        "band_50": generate_contours_from_mask(band_50_mask, lats, shifted_lons),
        "band_100": generate_contours_from_mask(band_100_mask, lats, shifted_lons),
    }
    
    median_mask = np.array(shifted_data[median_idx] >= threshold, dtype=int)
    np.savez(os.path.join(output_dir, "median.npz"), data=median_mask, lats=lats, lons=shifted_lons)
    
    contour_individual = extract_contours(data, years, lats, lons, threshold)
    contour_data.update(contour_individual)

    return save_contours_to_tempfiles(output_dir, contour_data)

def process_contour_boxplot_by_day(fields_dict, dates_dict, lats, lons, day_idx, threshold, output_dir):
    """
    Given a set of yearly fields and dates, extract a specific day's data,
    compute contour boxplot statistics (Whitaker et al.), visualize, and save contours.

    Parameters:
        fields_dict (dict): year -> list of 2D np.array scalar fields.
        dates_dict (dict): year -> list of cftime objects.
        lats (1D np.array): Latitude values.
        lons (1D np.array): Longitude values (in [0, 360)).
        day_idx (int): Index of the date within each year to extract.
        threshold (float): Threshold value for contour boxplot.
        output_dir (str): Directory to store output JSON contour files.
    """
    samples = []
    years = list(sorted(fields_dict.keys()))

    for year in sorted(fields_dict.keys()):
        if day_idx < len(fields_dict[year]):
            field = fields_dict[year][day_idx]
            shifted_field, shifted_lons = shift_longitude(field, lons)
            cropped_field, clats, clons = crop_field_to_region(shifted_field, lats, shifted_lons, lat_range=(29, 76), lon_range=(-11, 41))
            # samples.append(shifted_field)
            samples.append(cropped_field)

    if not samples:
        raise ValueError(f"No valid samples found for day index {day_idx}.")

    data = np.stack(samples)

    # Step 1: compute masks and mismatch matrix
    masks = data >= threshold
    mismatch_matrix, _ = compute_mismatch_matrix(masks)

    # Step 2: tune epsilon and compute relaxed cBD scores
    eps_star = find_optimal_epsilon(mismatch_matrix)
    print(f"eps = {eps_star}")
    cbd_scores = compute_relaxed_cbd_eps(mismatch_matrix, eps_star)

    # Step 3: visualize and save
    # _, shifted_lons = shift_longitude(data[0], lons)
    # plot_contour_boxplot(data, cbd_scores, lats, shifted_lons, contour_level=threshold)

    # Step 4: save contours
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "cbd_scores.npz"), data=compute_pixelwise_frequency_map(masks), lats=clats, lons=clons)
    return extract_and_save_contour_boxplot_components(data, cbd_scores, years, clats, clons, threshold, output_dir)


def load_saved_contours(meta_file):
    """
    Load all contour components saved previously into memory.

    Parameters:
        meta_file (str): Path to the metadata JSON file (created by save_contours_to_tempfiles).

    Returns:
        dict: Dictionary with keys: 'median', 'mean', 'band_50', 'band_100', each mapping to list of contours.
    """
    return load_contours_from_tempfiles(meta_file)

def main(): 
    dset_str = None
    for arg in sys.argv:
        if "dataset" in arg or "dset" in arg:
            if "ukesm" in arg.lower():
                dset_str = "dataset_UKESM.json"
            elif "era5" in arg.lower():
                dset_str = "dataset_ERA5.json"
    if dset_str is None:
        print("No specified dataset name! Going with ERA5 dataset...")
        dset_str = "dataset_ERA5.json"
    print("Dataset file:", dset_str)
    
    with open(dset_str, "r") as dsetting:
        settings = json.load(dsetting)
    
    calendars = {
        "ERA5": "proleptic_gregorian",
        "UKESM": "360_day",
        "ERA5-normalize": "proleptic_gregorian",
        "UKESM-normalize": "360_day",
    }
    
    thresholds = {
        "ERA5": 130,
        "UKESM": 130,
        "ERA5-normalize": 1.2,
        "UKESM-normalize": 1.0,
    }
    
    overlap_size = 31
    
    data_root = settings["data_root"]
    dataset = settings["dataset"]
    if "UKESM" in dataset:
        start_month, start_day = 5, 27
        end_month, end_day = 9, 4
    else:
        start_month, start_day = 5, 28
        end_month, end_day = 9, 4
    
    fields_dict = {}
    dates_dict = {}
    lats = None
    lons = None
    
    years_to_dos = [settings["years"]] 
    
    for ei, years_to_do in enumerate(years_to_dos):
        for year in tqdm(years_to_do, desc="Loading years..."):
            dir_info = [data_root, dataset, str(year)]
            fields, files, dates, lats, lons = load_scalar_fields_from_npz(os.path.join(*dir_info), (start_month, start_day), (end_month, end_day), calendars[dataset])
            
            lons = np.where(lons > 180, lons - 360, lons)
            
            fields_dict[year] = fields
            dates_dict[year] = dates
        
        # after parameter tuning, we found that threshold=1.0 (or 1.2 for ERA5), overlap_size=31 is the optimal
        # threshold, overlap_size = region_overlap_tuning(settings, dates_dict, fields_dict, lats, lons, gt_dict)
        threshold = thresholds[dataset]
        test_years = years_to_do
        positive_days = {}
        positive_days_filename = f"positive_days_{dataset}.json" if len(years_to_dos) == 1 else f"positive_days_{dataset}_{ei}.json"
        
        if not os.path.exists(positive_days_filename):
            for year in tqdm(test_years, desc="Test years"):
                roi_comps_by_day = []
                dates = dates_dict[year]
                
                # Identify superlevel set component
                for idate, date in enumerate(dates_dict[year]):
                    roi_components = extract_superlevel_components(fields_dict[year][idate], lats, lons, threshold)
                    roi_comps_by_day.append(roi_components)
                
                # let's try region overlap
                valid_start_days = track_components_region_overlap(roi_comps_by_day, lats, min_overlap_pixels=overlap_size)
            
                # Compute the binary sequence of blocked days
                detected = set()
                for v_st in valid_start_days:
                    for vd in range(v_st, v_st + 5):
                        if vd >= len(dates_dict[year]):
                            continue
                        if (dates_dict[year][vd].month, dates_dict[year][vd].day) in detected:
                            continue
                        date = dates_dict[year][vd]
                        if date.year == year:
                            detected.add(vd)
                detected = list(detected)
                detected.sort()
                for vd in detected:
                    if vd not in positive_days:
                        positive_days[vd] = [year]
                    else:
                        positive_days[vd].append(year)
            
            with open(positive_days_filename, "w") as f:
                json.dump(positive_days, f)
        else:
            with open(positive_days_filename, "r") as f:
                positive_days = json.load(f)
            
        # output contour boxplot
        out_basedir = f"./{dataset}_contour_boxplot" if len(years_to_dos) == 1 else f"./{dataset}_contour_boxplot_{ei}"
        os.makedirs(out_basedir, exist_ok=True)
        positive_days_list = [(int(key), positive_days[key]) for key in positive_days]
        positive_days_list.sort(key=lambda x: len(x[1]), reverse=True)
        # This value is just a random number (but higher than the number of days in JJA)
        output_days = 300
        # For each day we compute contour boxplot
        for i in range(min(output_days, len(positive_days))):
            day_idx = positive_days_list[i][0]
            date_obj = dates_dict[test_years[0]][day_idx]
            # we obtain all years with positive events
            positive_years = positive_days_list[i][1]
            if len(positive_years) < 1:
                break
            
            out_day_dir = os.path.join(out_basedir, f"positive_{str(date_obj.month).zfill(2)}{str(date_obj.day).zfill(2)}")
            os.makedirs(out_day_dir, exist_ok=True)
            # extract the fields on days that are marked positive
            fields_dict_positive = {year: fields_dict[year] for year in positive_years}
            dates_dict_positive = {year: dates_dict[year] for year in positive_years}
            print("Day:", date_obj.month, date_obj.day)
            # compute the contour boxplot
            process_contour_boxplot_by_day(fields_dict_positive, dates_dict_positive, lats, lons, positive_days_list[i][0], threshold, out_day_dir)
        
            
if __name__ == "__main__":
    main()

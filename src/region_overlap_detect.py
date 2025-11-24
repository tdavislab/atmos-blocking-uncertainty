import os
import sys
import numpy as np
import json
import re
import csv
import cftime
import datetime
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from region_overlap_util import *
from region_overlap_boxplot import crop_field_to_region, shift_longitude

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
            try:
                data = np.load(os.path.join(directory, filename))
            except:
                print("FILE NOT EXIST???", directory, filename)
                raise FileExistsError
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
    gt_root = "C:/Users/mingz/Desktop/ResearchProjects/GWMT-Blocking/blocking-data/GTD/"
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


def plot_temporal_stats(result_category_data):
    # Convert (month, day) to dummy date objects for sorting
    dummy_year = 2000
    date_data = {
        datetime.date(dummy_year, m, d): counts
        for (m, d), counts in result_category_data.items()
    }

    df = pd.DataFrame(date_data).T
    df = df.sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = None
    # categories = ["TP", "FP", "TN", "FN"]
    categories = ["TN", "TP", "FP", "FN"]
    colors = [
        "#004D40",
        "#1E88E5",
        "#FFC107",
        "#D81B60",
    ]

    for cat, cl in zip(categories, colors):
        ax.bar(df.index, df[cat], bottom=bottom, color=cl, label=cat)
        bottom = df[cat] if bottom is None else bottom + df[cat]

    # Format x-axis with spaced ticks
    all_dates = df.index
    n_dates = len(all_dates)
    tick_spacing = max(n_dates // 15, 1)  # Show ~15 ticks max
    tick_positions = all_dates[::tick_spacing]
    tick_labels = [d.strftime("%m-%d") for d in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)

    # Labels and title
    ax.set_ylabel("Count")
    ax.set_xlabel("Date (MM-DD)")
    # ax.set_title("Stacked Bar Chart of TP, FP, TN, FN per Day")
    ax.legend(title="Category")

    plt.tight_layout()
    plt.show()


def main(): 
    dset_str = None
    method = "ours"
    for arg in sys.argv:
        if "dataset" in arg or "dset" in arg:
            if "ukesm" in arg.lower():
                dset_str = "dataset_UKESM.json"
            elif "era5" in arg.lower():
                dset_str = "dataset_ERA5.json"
        if "method" in arg:
            if "dg83" in arg.lower():
                method = "dg83"
    
    if dset_str is None:
        print("No specified dataset name! Going with ERA5 dataset...")
        dset_str = "dataset_ERA5.json"
    print("Dataset file:", dset_str)
    print("Detection method:", method)
        
    with open(dset_str, "r") as dsetting:
        settings = json.load(dsetting)
    
    calendars = {
        "ERA5": "proleptic_gregorian",
        "UKESM": "360_day",
        "ERA5-normalize": "proleptic_gregorian",
        "UKESM-normalize": "360_day",
    }
    
    # Loading basic data property settings
    data_root = settings["data_root"]
    dataset = settings["dataset"]
    ltds_file = settings["ltds_file"]
    day_dict_file = ltds_file.replace("ltds", "day_dict")
    if "UKESM" in dataset:
        start_month, start_day = 5, 27
        end_month, end_day = 9, 4
    else:
        start_month, start_day = 5, 28
        end_month, end_day = 9, 4
        
    ground_truth = load_ground_truth(dataset)
    
    fields_dict = {}
    dates_dict = {}
    gt_dict = {}
    lats = None
    lons = None
    # Shape: (lat, lon, day_of_the_year)
    ltds_load = np.load(ltds_file, allow_pickle=True)
    day_dict_load = np.load(day_dict_file, allow_pickle=True)
    ltds = ltds_load["ltds_smooth"]
    day_dict = day_dict_load["day_dict"].item()
    assert ltds.shape[2] == len(day_dict)
    
    in_season_indices = set()
    
    # We need to select the days corresponding to the GTD
    for year in tqdm(settings["years"], desc="Loading years..."):
        dir_info = [data_root, dataset, str(year)]
        fields, files, dates, lats, lons = load_scalar_fields_from_npz(os.path.join(*dir_info), (start_month, start_day), (end_month, end_day), calendars[dataset])
        
        lons = np.where(lons > 180, lons - 360, lons)
        
        gt = {}
        for yr, mo, da in ground_truth:
            if mo == start_month and da < start_day:
                continue
            if mo == end_month and da > end_day:
                continue
            in_season_indices.add(day_dict[mo, da])
            if yr == year:
                gt[(mo, da)] = ground_truth[(yr, mo, da)] 
        fields_dict[year] = fields
        dates_dict[year] = dates
        gt_dict[year] = gt
    
    # select the subset of data that are within the JJA season
    in_season_indices = list(in_season_indices)
    in_season_indices.sort()
    in_season_indices = np.asarray(in_season_indices, dtype=int)
    ltds_JJA = ltds[:, :, in_season_indices]
    ltds_JJA_means = np.average(ltds_JJA, axis=(0, 1))
    
    # We disabled parameter tuning for fixed parameters
    # If you want to replicate the parameter tuning results, go with "parameter_tuning=True"
    parameter_tuning = False
    if parameter_tuning:
        threshold, overlap_size = region_overlap_tuning(settings, dates_dict, fields_dict, lats, lons, gt_dict, ltds_JJA)
    else:
        if method == "ours":
            threshold = 1.0 if "UKESM" in dataset else 1.2
            overlap_size = 31
        elif method == "dg83":
            threshold = 1.5 
            overlap_size = 1
        else:
            raise ValueError("Unknown method:", method)
    if "normalize" not in dataset:
        ltds_threshold = np.maximum(100, threshold * ltds_JJA)
    
    # initializing the result accumulation
    test_years = settings["years"][:]
    frequency_map = np.zeros(fields_dict[test_years[0]][0].shape)
    result_category_data = {(date.month, date.day): {"TP": 0, "FP": 0, "TN": 0, "FN": 0} 
                            for date in dates_dict[test_years[0]]}
    
    # With all parameters determined, we start the detection process for every year
    test_detects = []
    worst_f1 = None
    for year in tqdm(test_years, desc="Test years"):
        roi_comps_by_day = []
        dates = dates_dict[year]
        # Extract superlevel set components
        for idate, date in enumerate(dates_dict[year]):
            if "normalize" not in dataset:
                roi_components = extract_superlevel_components(fields_dict[year][idate], lats, lons, ltds_threshold[:, :, idate])
            else:
                roi_components = extract_superlevel_components(fields_dict[year][idate], lats, lons, threshold)
            roi_comps_by_day.append(roi_components)
        
        # Find all the valid starting days using the region-overlap-based tracking
        valid_start_days = track_components_region_overlap(roi_comps_by_day, lats, min_overlap_pixels=overlap_size)
    
        # translate the valid_start_days results into a binary sequence for detection results
        detected = set()
        for v_st in valid_start_days:
            for vd in range(v_st, v_st + 5):
                if vd >= len(dates_dict[year]):
                    continue
                if (dates_dict[year][vd].month, dates_dict[year][vd].day) in detected:
                    continue
                frequency_map[fields_dict[year][vd] > threshold] += 1
                date = dates_dict[year][vd]
                if date.year == year:
                    detected.add((date.month, date.day))
        detected = list(detected)
        detected.sort()
            
        detect = Detection(detected, gt_dict[year], threshold, year, overlap_size)
        
        # compute the statistics
        detected_set = set(detected)
        tp_days = set()
        fp_days = set()
        fn_days = set()
        for mo, da in gt_dict[year]:
            val = gt_dict[year][(mo, da)]
            if int(val) > 0 and (mo, da) in detected_set:
                tp_days.add((mo, da))
                result_category_data[(mo, da)]["TP"] += 1
            elif int(val) < 1 and (mo, da) not in detected_set:
                result_category_data[(mo, da)]["TN"] += 1
            elif int(val) > 0 and (mo, da) not in detected_set:
                fn_days.add((mo, da))
                result_category_data[(mo, da)]["FN"] += 1
            else:
                fp_days.add((mo, da))
                result_category_data[(mo, da)]["FP"] += 1

        # Debug purpose only: output the stats for the year with the lowest testing F1-score for diagnose
        if worst_f1 is None or (not np.isnan(detect.f1) and detect.f1 < worst_f1):
            print("YEAR =", year)
            tp_days = list(tp_days)
            tp_days.sort()
            fp_days = list(fp_days)
            fp_days.sort()
            fn_days = list(fn_days)
            fn_days.sort()
            print("True Positive Days:", tp_days)
            print("False Positive Days:", fp_days)
            print("False Negative Days:", fn_days)
            worst_f1 = detect.f1
            print("F1-score:", detect.f1)
            
        test_detects.append(detect)
    
    # summary of testing stats
    test_stats = calc_stats_from_detects(test_detects)
    print("====================================")
    print("Testing Conclusion:")
    print(f"Accuracy: {test_stats[1]}, Precision: {test_stats[3]}, Recall: {test_stats[2]}, F1: {test_stats[0]}")
    
    # saving the blocking frequency heatmap
    shifted_frequency_map, shifted_lons = shift_longitude(frequency_map, lons)
    cropped_frequency_map, cropped_lats, cropped_lons = crop_field_to_region(shifted_frequency_map, lats, shifted_lons, lat_range=(29, 76), lon_range=(-11, 41))
    np.savez(f"frequency_map_{dataset}.npz", data=cropped_frequency_map, lats=cropped_lats, lons=cropped_lons)
    
    # saving the detection binary sequences (although organized in a different way)
    with open(f"{dataset}_detection_conclusion.json", "w") as outf:
        json.dump(str(result_category_data), outf)
        outf.close()

    # temporal stability analysis    
    plot_temporal_stats(result_category_data)
    
            
if __name__ == "__main__":
    main()

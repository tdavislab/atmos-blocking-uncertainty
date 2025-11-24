import os
import math
import numpy as np
from collections import defaultdict
from scipy import ndimage
from tqdm import tqdm

class Detection:
    def __init__(self, detected, gt, threshold, year, overlap_size):
        self.threshold = threshold
        self.year = year
        self.overlap_size = overlap_size
        # detected format: [(month, day)]
        self.detected = detected
        
        # gt format: {(month, day): gt_label}
        self.gt = gt
        true_positive = 0
        false_positive = 0
        self.total_dates = len(gt)
        gt_positive = np.sum([gt[each] for each in gt])
        
        for mo, da in detected:
            assert (mo, da) in gt
            if gt[(mo, da)] == 1:
                true_positive += 1
            else:
                false_positive += 1
        
        # true_positive, gt_positive, false_positive, total_dates,
        self.TP = true_positive
        self.gt_positive = gt_positive
        self.FP = false_positive
        
        self.FN = gt_positive - true_positive
        
        self.TN = self.total_dates - true_positive - false_positive - self.FN
        
        try:
            self.acc = (true_positive + self.TN) / self.total_dates
            self.recall = true_positive / gt_positive
            self.precision = true_positive / (true_positive + false_positive)
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        except ZeroDivisionError:
            if self.total_dates == 0:
                self.acc = np.nan
            if gt_positive == 0:
                self.recall = np.nan
            else:
                self.recall = true_positive / gt_positive
            if true_positive + false_positive == 0:
                self.precision = np.nan
            else:
                self.precision = true_positive / (true_positive + false_positive)
            if np.isnan(self.precision) or np.isnan(self.recall) or (self.precision + self.recall == 0):
                self.f1 = np.nan
            else:
                self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
                
    def safe_eq(self, x, y):
        if np.isnan(x) and np.isnan(y):
            return True
        return x == y

    def safe_lt(self, x, y):
        if np.isnan(x) and np.isnan(y):
            return False  # they are equal
        if np.isnan(x):
            return True   # treat nan as smaller
        if np.isnan(y):
            return False
        return x < y
    
    def safe_tuple_eq(self, t1, t2):
        return all(self.safe_eq(a, b) for a, b in zip(t1, t2))

    def safe_tuple_lt(self, t1, t2):
        for a, b in zip(t1, t2):
            if self.safe_eq(a, b):
                continue
            return self.safe_lt(a, b)
        return False  # all equal
    
    def __lt__(self, other):
        return self.safe_tuple_lt((self.f1, self.acc, self.recall, self.precision), (other.f1, other.acc, other.recall, other.precision))
    
    def __eq__(self, other):
        return self.safe_tuple_eq((self.f1, self.acc, self.recall, self.precision), (other.f1, other.acc, other.recall, other.precision))
    

def extract_superlevel_components(field, lats, lons, threshold,
                                  lat_range=(25, 70), lon_range=(-10, 40)):
    """
    Extract superlevel sets (connected components above threshold) within a bounding box,
    correctly handling periodic wraparound in longitude.

    Args:
        field: 2D numpy array of shape (lat, lon), longitude assumed periodic.
        lats: 1D array of latitude values.
        lons: 1D array of longitude values (assumed from 0 to 359).
        threshold: Scalar value for superlevel set.
        lat_range: Tuple (min_lat, max_lat) in degrees.
        lon_range: Tuple (min_lon, max_lon) in degrees.

    Returns:
        components: List of lists of (i, j) coordinates in the original grid.
    """
    def is_int_or_same_shape(a, b):
        return isinstance(a, (int, float, np.integer, np.float_, np.float32)) or (isinstance(a, np.ndarray) and np.shape(a) == np.shape(b))
    assert is_int_or_same_shape(threshold, field)
    
    mask = field > threshold
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    labeled, num_features = ndimage.label(mask, structure=structure)

    # Create mapping of connected labels at periodic boundary
    label_map = {}
    for i in range(field.shape[0]):
        label_left = labeled[i, 0]
        label_right = labeled[i, -1]
        if label_left and label_right and label_left != label_right:
            # Union labels
            root_left = label_map.get(label_left, label_left)
            root_right = label_map.get(label_right, label_right)
            new_root = min(root_left, root_right)
            for k, v in list(label_map.items()):
                if v == root_left or v == root_right:
                    label_map[k] = new_root
            label_map[label_left] = new_root
            label_map[label_right] = new_root

    # Remap labels using the map
    relabeled = labeled.copy()
    for i in range(relabeled.shape[0]):
        for j in range(relabeled.shape[1]):
            lbl = relabeled[i, j]
            if lbl in label_map:
                relabeled[i, j] = label_map[lbl]

    # Normalize longitude range to [0, 360)
    lon_min = lon_range[0] % 360
    lon_max = lon_range[1] % 360

    components = []
    for label in np.unique(relabeled):
        if label == 0:
            continue  # skip background

        coords = np.argwhere(relabeled == label)
        lat_lon_coords = []

        for i, j in coords:
            lat = lats[i]
            lon = lons[j] % 360

            lat_in_bounds = lat_range[0] <= lat <= lat_range[1]
            if lon_min <= lon_max:
                lon_in_bounds = lon_min <= lon <= lon_max
            else:
                lon_in_bounds = lon >= lon_min or lon <= lon_max

            if lat_in_bounds and lon_in_bounds:
                lat_lon_coords.append((i, j))

        if lat_lon_coords:
            components.append(lat_lon_coords)

    return components


def track_components_region_overlap(roi_comps_by_day, lats, min_length=5, min_overlap_pixels=10, min_feature_size=-1):
    """
    Track component trajectories across time using region overlap.

    Args:
        roi_comps_by_day: List of lists of components, where each component is a list of (lat, lon) tuples.
        min_length: Minimum trajectory length (in time steps) to be considered.
        min_overlap_ratio: Minimum fraction of overlapping pixels to establish a connection.

    Returns:
        valid_start_days: Set of day indices where a valid trajectory (length ≥ min_length) starts.
    """
    def weighted_pixel_count(intersection, lats):
        """
        intersection: iterable of (i, j) pixel indices (e.g., a set)
        lats: 1D array of latitude centers in degrees; lat of (i, j) is lats[i]

        Returns:
            Equator-equivalent pixel count (sum of per-pixel area weights).
            Each pixel is weighted by cos(lat[i]) (>=0), appropriate for
            equal-longitude-width grids.
        """
        if not intersection:
            return 0.0
        lats = np.asarray(lats, dtype=float)
        ii = np.fromiter((i for i, _ in intersection), dtype=int)
        w = np.cos(np.deg2rad(lats[ii]))
        w = np.clip(w, 0.0, None)  # guard tiny negatives at poles
        return float(w.sum())

    num_days = len(roi_comps_by_day)
    trajectories = []  # List of ongoing trajectories

    # Each component on day t is given a unique ID (tuple of day and component index)
    comp_to_pixels = {}  # Maps (day, comp_idx) -> set of pixels
    for t, comps in enumerate(roi_comps_by_day):
        for i, comp in enumerate(comps):
            comp_to_pixels[(t, i)] = set(comp)

    # Build a forward link graph based on overlaps
    forward_links = defaultdict(list)  # (t, i) -> list of (t+1, j)
    # for t in tqdm(range(num_days - 1), desc="Tracking components"):
    for t in range(num_days - 1):
        comps_t = roi_comps_by_day[t]
        comps_tp1 = roi_comps_by_day[t + 1]

        for i, comp_i in enumerate(comps_t):
            pixels_i = comp_to_pixels[(t, i)]
            if not pixels_i:
                continue
            if min_feature_size > 0 and weighted_pixel_count(pixels_i, lats) < min_feature_size:
                continue

            for j, comp_j in enumerate(comps_tp1):
                pixels_j = comp_to_pixels[(t + 1, j)]
                if not pixels_j:
                    continue
                if min_feature_size > 0 and weighted_pixel_count(pixels_j, lats) < min_feature_size:
                    continue

                intersection = pixels_i & pixels_j
                # overlap_pixels = len(intersection) 
                overlap_pixels = weighted_pixel_count(intersection, lats)

                if overlap_pixels >= min_overlap_pixels:
                    forward_links[(t, i)].append((t + 1, j))

    # Traverse trajectories starting from day 0 to num_days-1
    valid_start_days = set()

    for t in range(num_days):
        for i in range(len(roi_comps_by_day[t])):
            visited = set()
            stack = [((t, i), 1)]  # ((day, comp_idx), length_so_far)

            while stack:
                (curr_t, curr_i), length = stack.pop()
                visited.add((curr_t, curr_i))

                if length >= min_length:
                    valid_start_days.add(t)
                    break  # Found a long-enough trajectory from this start, no need to explore more

                for neighbor in forward_links.get((curr_t, curr_i), []):
                    if neighbor not in visited:
                        stack.append((neighbor, length + 1))

    return valid_start_days

def calc_stats_from_detects(detect_list: list[Detection]):
    samples = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    gt_positive = 0
    
    for detect in detect_list:
        if hasattr(detect, "total_dates"):
            samples += detect.total_dates
        else:
            samples += detect.TP + detect.TN + detect.FP + detect.FN
        TP += detect.TP
        TN += detect.TN
        FP += detect.FP
        FN += detect.FN
        gt_positive += detect.gt_positive
    
    acc = (TP + TN) / samples
    try:
        recall = TP / gt_positive
        precision = TP / (TP + FP)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if gt_positive:
            recall = TP / gt_positive
        else:
            recall = np.nan
        if (TP + FP):
            precision = TP / (TP + FP)
        else:
            precision = np.nan
        if (not np.isnan(precision)) and (not np.isnan(recall)) and (precision + recall):
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = np.nan
    
    print(f"Debug: #years={len(detect_list)}, TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    
    return f1, acc, recall, precision

def region_overlap_tuning(settings, dates_dict, fields_dict, lats, lons, gt_dict, ltds=None):
    dataset = settings["dataset"]
    training_size = min(len(settings["years"]) // 2, 40)
    training_years = settings["years"][:training_size]
    test_years = settings["years"][training_size:]
    validation_folds = 5
    validation_size = int(math.floor(training_size / validation_folds))    
    
    if "normalize" in dataset:
        thresholds = [1.0]
    elif ltds is not None:
        print("LTDS is not None. The threshold varies by LTDS at each pixel!")
        if "normalize" not in dataset:
            thresholds = np.arange(1.0, 2.01, 0.1)
        else:
            assert "Normalization should be arranged early."
    else:
        thresholds = np.arange(120, 141, 5)
    overlap_sizes = np.arange(27, 37, 1, dtype=int)
    
    detects = []
    
    training_output_fname = f"training_detect_by_folds_{dataset}_ltds_fields.npz"
    if not os.path.exists(training_output_fname):
        # Training with cross-validation 
        for i in tqdm(range(validation_folds), desc="Validation folds"):
            train_years = training_years[:i*validation_size] + training_years[(i+1)*validation_size:]
            
            detects_dict = {}
                
            for threshold in tqdm(thresholds, desc="Iterate thresholds", leave=False):
                if ltds is not None and 'normalize' not in dataset:
                    threshold_ltds = np.maximum(100, ltds * threshold)
                    
                for year in train_years:
                    roi_comps_by_day = []
                    dates = dates_dict[year]
                    if ltds is not None and 'normalize' not in dataset:
                        assert len(dates) == threshold_ltds.shape[2]
                        
                    for idate, date in enumerate(dates_dict[year]):
                        if ltds is None or 'normalize' in dataset:
                            roi_components = extract_superlevel_components(fields_dict[year][idate], lats, lons, threshold)
                        else:
                            roi_components = extract_superlevel_components(fields_dict[year][idate], lats, lons, threshold_ltds[:, :, idate])
                        roi_comps_by_day.append(roi_components)
                        
                    for overlap_size in overlap_sizes:
                        # let's try region overlap 
                        valid_start_days = track_components_region_overlap(roi_comps_by_day, lats, min_overlap_pixels=overlap_size)
                    
                        detected = set()
                        for v_st in valid_start_days:
                            for vd in range(v_st, v_st + 5):
                                if vd >= len(dates_dict[year]):
                                    continue
                                if (dates_dict[year][vd].month, dates_dict[year][vd].day) in detected:
                                    continue
                                date = dates_dict[year][vd]
                                if date.year == year:
                                    detected.add((date.month, date.day))
                        detected = list(detected)
                        detected.sort()
                        
                        detect = Detection(detected, gt_dict[year], threshold, year, overlap_size)
                        if (threshold, overlap_size) not in detects_dict:
                            detects_dict[(threshold, overlap_size)] = [detect]
                        else:
                            detects_dict[(threshold, overlap_size)].append(detect)
            
            detects.append(detects_dict)
        
        np.savez(training_output_fname, detects=np.array(detects, dtype=object))
    else:
        with np.load(training_output_fname, allow_pickle=True) as training_f:
            detects = training_f["detects"]
    
    # Now, let's do validation
    # detects contains a list of dictionaries of (threshold, overlap_size), of which the value is a list of Detection objects
    # For each fold, we compute a validation Detection object.
    # We choose the parameter from the fold that returns the highest (F1 score, Accuracy, Recall, Precision)
    best_val_settings = None
    best_val_scores = None
    best_val_diff_pr = None
    
    for i in tqdm(range(validation_folds), desc="Validating parameter choice"):
        detects_by_fold = detects[i]
        
        best_training_settings = None
        best_training_scores = None
        
        training_sorts = []
        for threshold in thresholds:
            for overlap_size in overlap_sizes:
                assert (threshold, overlap_size) in detects_by_fold
                detect_list = detects_by_fold[(threshold, overlap_size)]
                training_stats = calc_stats_from_detects(detect_list)
                
                
                if best_training_settings is None or best_training_scores is None:
                    best_training_settings = (threshold, overlap_size)
                    best_training_scores = training_stats
                elif training_stats > best_training_scores:
                    best_training_settings = (threshold, overlap_size)
                    best_training_scores = training_stats
                
                training_sorts.append((threshold, overlap_size, training_stats))
        
        acceptable_training_settings = None
        acceptable_training_scores = None
        
        # A small trick (not sure if it worked eventually):
        # For suboptimal validation results, if its f1-score is only marginally smaller than the best one, 
        # we will choose the suboptimal one if the difference between precision and recall is lower
        for thr, ovl_s, tr_stats in training_sorts:
            f1 = tr_stats[0]
            diff_pr = abs(tr_stats[2] - tr_stats[3])
            if best_training_scores[0] - f1 <= 0.005:
                if acceptable_training_settings is None or acceptable_training_scores is None:
                    acceptable_training_settings = (thr, ovl_s)
                    acceptable_training_scores = tr_stats
                elif diff_pr < abs(acceptable_training_scores[2] - acceptable_training_scores[3]):
                    acceptable_training_settings = (thr, ovl_s)
                    acceptable_training_scores = tr_stats
        
        print("====================================")
        print(f"Validation Fold {i}")
        print(f"Best Training Stats: Accuracy: {best_training_scores[1]}, Precision: {best_training_scores[3]}, Recall: {best_training_scores[2]}, F1: {best_training_scores[0]}")
        print(f"Parameters: threshold = {best_training_settings[0]}, overlap_size = {best_training_settings[1]}")
        
        best_training_settings = acceptable_training_settings
        best_training_scores = acceptable_training_scores
        
        print("====================================")
        print(f"Validation Fold {i}")
        print(f"Best Training Stats: Accuracy: {best_training_scores[1]}, Precision: {best_training_scores[3]}, Recall: {best_training_scores[2]}, F1: {best_training_scores[0]}")
        print(f"Parameters: threshold = {best_training_settings[0]}, overlap_size = {best_training_settings[1]}")
        
        # After getting the best parameter, we apply them to the validation dataset
        val_years = training_years[i * validation_size : (i+1) * validation_size]
        print("---------------------------")
        print("Validating on", val_years)
        
        val_detects = []
        
        for year in val_years:
            roi_comps_by_day = []
            dates = dates_dict[year]
            for idate, date in enumerate(dates_dict[year]):
                if ltds is None or "normalize" in dataset:
                    roi_components = extract_superlevel_components(fields_dict[year][idate], lats, lons, best_training_settings[0])
                else:
                    threshold_ltds = np.maximum(100, ltds * best_training_settings[0])
                    roi_components = extract_superlevel_components(fields_dict[year][idate], lats, lons, threshold_ltds[:, :, idate])
                roi_comps_by_day.append(roi_components)

            valid_start_days = track_components_region_overlap(roi_comps_by_day, lats, min_overlap_pixels=best_training_settings[1])
                
            detected = set()
            for v_st in valid_start_days:
                for vd in range(v_st, v_st + 5):
                    if vd >= len(dates_dict[year]):
                        continue
                    if (dates_dict[year][vd].month, dates_dict[year][vd].day) in detected:
                        continue
                    date = dates_dict[year][vd]
                    if date.year == year:
                        detected.add((date.month, date.day))
            detected = list(detected)
            detected.sort()
            
            detect = Detection(detected, gt_dict[year], best_training_settings[0], year, best_training_settings[1])
            val_detects.append(detect)
        
        val_stats = calc_stats_from_detects(val_detects)
        print("---------------------------")
        print(f"Validation stats: Accuracy: {val_stats[1]}, Precision: {val_stats[3]}, Recall: {val_stats[2]}, F1: {val_stats[0]}")
        
        # if best_val_scores is None or val_stats > best_val_scores:
        if best_val_scores is None or abs(val_stats[2] - val_stats[3]) < best_val_diff_pr:
            best_val_scores = val_stats
            best_val_settings = best_training_settings
            best_val_diff_pr = abs(val_stats[2] - val_stats[3])
    
    print("====================================")
    print("Validation Final Conclusion:")
    print(f"Best parameters: threshold = {best_val_settings[0]}, overlap_size = {best_val_settings[1]}")
    print(f"Best validation Stats: Accuracy: {best_val_scores[1]}, Precision: {best_val_scores[3]}, Recall: {best_val_scores[2]}, F1: {best_val_scores[0]}")
    
    return best_val_settings

# Parameters for DG83 were tuned by previous works
# We went through the same process, and found very similar parameter choices eventually
# Therefore, we omit this process in the paper
def dg83_tuning(settings, dates_dict, fields_dict, lats, lons, gt_dict, ltds=None):
    dataset = settings["dataset"]
    training_size = min(len(settings["years"]) // 2, 40)
    training_years = settings["years"][:training_size]
    test_years = settings["years"][training_size:]
    validation_folds = 5
    validation_size = int(math.floor(training_size / validation_folds))    
    
    thresholds = [1.5] if "UKESM" in dataset else [1.5 / 1.2]
    # A trick that helps improve DG83, but it seemed not that helpful
    # Idea: remove features that are too small in size
    feature_sizes = np.arange(15, 32, 1, dtype=int)
    
    detects = []
    
    training_output_fname = f"training_detect_by_folds_{dataset}_dg83.npz"
    if not os.path.exists(training_output_fname):
        # Training with cross-validation 
        for i in tqdm(range(validation_folds), desc="Validation folds"):
            train_years = training_years[:i*validation_size] + training_years[(i+1)*validation_size:]
            
            detects_dict = {}
                
            for threshold in tqdm(thresholds, desc="Iterate thresholds", leave=False):
                if ltds is not None and 'normalize' not in dataset:
                    threshold_ltds = np.maximum(100, ltds * threshold)
                    
                for year in train_years:
                    roi_comps_by_day = []
                    dates = dates_dict[year]
                    if ltds is not None and 'normalize' not in dataset:
                        assert len(dates) == threshold_ltds.shape[2]
                        
                    for idate, date in enumerate(dates_dict[year]):
                        if ltds is None or 'normalize' in dataset:
                            roi_components = extract_superlevel_components(fields_dict[year][idate], lats, lons, threshold)
                        else:
                            roi_components = extract_superlevel_components(fields_dict[year][idate], lats, lons, threshold_ltds[:, :, idate])
                        roi_comps_by_day.append(roi_components)
                        
                    for feature_size in feature_sizes:
                        # let's try region overlap with at least 1 pixel
                        valid_start_days = track_components_region_overlap(roi_comps_by_day, lats, min_overlap_pixels=1, min_feature_size=feature_size)
                    
                        detected = set()
                        for v_st in valid_start_days:
                            for vd in range(v_st, v_st + 5):
                                if vd >= len(dates_dict[year]):
                                    continue
                                if (dates_dict[year][vd].month, dates_dict[year][vd].day) in detected:
                                    continue
                                date = dates_dict[year][vd]
                                if date.year == year:
                                    detected.add((date.month, date.day))
                        detected = list(detected)
                        detected.sort()
                        
                        detect = Detection(detected, gt_dict[year], threshold, year, feature_size)
                        if (threshold, feature_size) not in detects_dict:
                            detects_dict[(threshold, feature_size)] = [detect]
                        else:
                            detects_dict[(threshold, feature_size)].append(detect)
            
            detects.append(detects_dict)
        
        np.savez(training_output_fname, detects=np.array(detects, dtype=object))
    else:
        with np.load(training_output_fname, allow_pickle=True) as training_f:
            detects = training_f["detects"]
    
    # Now, let's do validation
    # detects contains a list of dictionaries of (threshold, feature_size), of which the value is a list of Detection objects
    # For each fold, we compute a validation Detection object.
    # We choose the parameter from the fold that returns the highest (F1 score, Accuracy, Recall, Precision)
    best_val_settings = None
    best_val_scores = None
    best_val_diff_pr = None
    
    for i in tqdm(range(validation_folds), desc="Validating parameter choice"):
        detects_by_fold = detects[i]
        
        best_training_settings = None
        best_training_scores = None
        
        training_sorts = []
        for threshold in thresholds:
            for feature_size in feature_sizes:
                assert (threshold, feature_size) in detects_by_fold
                detect_list = detects_by_fold[(threshold, feature_size)]
                training_stats = calc_stats_from_detects(detect_list)
                
                
                if best_training_settings is None or best_training_scores is None:
                    best_training_settings = (threshold, feature_size)
                    best_training_scores = training_stats
                elif training_stats > best_training_scores:
                    best_training_settings = (threshold, feature_size)
                    best_training_scores = training_stats
                
                training_sorts.append((threshold, feature_size, training_stats))
        
        acceptable_training_settings = None
        acceptable_training_scores = None
        
        for thr, ovl_s, tr_stats in training_sorts:
            f1 = tr_stats[0]
            diff_pr = abs(tr_stats[2] - tr_stats[3])
            if best_training_scores[0] - f1 <= 0.005:
                if acceptable_training_settings is None or acceptable_training_scores is None:
                    acceptable_training_settings = (thr, ovl_s)
                    acceptable_training_scores = tr_stats
                elif diff_pr < abs(acceptable_training_scores[2] - acceptable_training_scores[3]):
                    acceptable_training_settings = (thr, ovl_s)
                    acceptable_training_scores = tr_stats
        
        print("====================================")
        print(f"Validation Fold {i}")
        print(f"Best Training Stats: Accuracy: {best_training_scores[1]}, Precision: {best_training_scores[3]}, Recall: {best_training_scores[2]}, F1: {best_training_scores[0]}")
        print(f"Parameters: threshold = {best_training_settings[0]}, feature_size = {best_training_settings[1]}")
        
        best_training_settings = acceptable_training_settings
        best_training_scores = acceptable_training_scores
        
        print("====================================")
        print(f"Validation Fold {i}")
        print(f"Best Training Stats: Accuracy: {best_training_scores[1]}, Precision: {best_training_scores[3]}, Recall: {best_training_scores[2]}, F1: {best_training_scores[0]}")
        print(f"Parameters: threshold = {best_training_settings[0]}, feature_size = {best_training_settings[1]}")
        
        # After getting the best parameter, we apply them to the validation dataset
        val_years = training_years[i * validation_size : (i+1) * validation_size]
        print("---------------------------")
        print("Validating on", val_years)
        
        val_detects = []
        
        for year in val_years:
            roi_comps_by_day = []
            dates = dates_dict[year]
            for idate, date in enumerate(dates_dict[year]):
                roi_components = extract_superlevel_components(fields_dict[year][idate], lats, lons, best_training_settings[0])
                roi_comps_by_day.append(roi_components)

            valid_start_days = track_components_region_overlap(roi_comps_by_day, lats, min_overlap_pixels=1, min_feature_size=best_training_settings[1])
                
            detected = set()
            for v_st in valid_start_days:
                for vd in range(v_st, v_st + 5):
                    if vd >= len(dates_dict[year]):
                        continue
                    if (dates_dict[year][vd].month, dates_dict[year][vd].day) in detected:
                        continue
                    date = dates_dict[year][vd]
                    if date.year == year:
                        detected.add((date.month, date.day))
            detected = list(detected)
            detected.sort()
            
            detect = Detection(detected, gt_dict[year], best_training_settings[0], year, best_training_settings[1])
            val_detects.append(detect)
        
        val_stats = calc_stats_from_detects(val_detects)
        print("---------------------------")
        print(f"Validation stats: Accuracy: {val_stats[1]}, Precision: {val_stats[3]}, Recall: {val_stats[2]}, F1: {val_stats[0]}")
        
        # if best_val_scores is None or val_stats > best_val_scores:
        if best_val_scores is None or abs(val_stats[2] - val_stats[3]) < best_val_diff_pr:
            best_val_scores = val_stats
            best_val_settings = best_training_settings
            best_val_diff_pr = abs(val_stats[2] - val_stats[3])
    
    print("====================================")
    print("Validation Final Conclusion:")
    print(f"Best parameters: threshold = {best_val_settings[0]}, feature_size = {best_val_settings[1]}")
    print(f"Best validation Stats: Accuracy: {best_val_scores[1]}, Precision: {best_val_scores[3]}, Recall: {best_val_scores[2]}, F1: {best_val_scores[0]}")
    
    return best_val_settings
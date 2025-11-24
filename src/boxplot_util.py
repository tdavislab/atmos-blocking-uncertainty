import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import measure
from scipy.ndimage import gaussian_filter

def add_smooth_noise(field, noise_scale=0.05, smoothing_sigma=2.0):
    noise = np.random.normal(scale=noise_scale, size=field.shape)
    smooth_noise = gaussian_filter(noise, sigma=smoothing_sigma)
    return field + smooth_noise

import itertools

def compute_mismatch_matrix(masks):
    n_samples = masks.shape[0]
    combinations = list(itertools.combinations(range(n_samples), 2))
    m = len(combinations)
    mismatch_matrix = np.zeros((m, n_samples))

    # for band_idx, (j, k) in enumerate(tqdm(combinations, desc="Building mismatch matrix")):
    for band_idx, (j, k) in enumerate(combinations):
        band_union = masks[j] | masks[k]
        band_intersection = masks[j] & masks[k]
        area_union = band_union.sum()
        area_intersection = band_intersection.sum()

        for i in range(n_samples):
            if (i == j) or (i == k):
                continue
            mask_i = masks[i]
            area_i = mask_i.sum()

            if area_i == 0:
                outside_band = 0.0
            else:
                outside_band = (mask_i & (~band_union)).sum() / area_i

            if area_intersection == 0:
                missing_band = 0.0
            else:
                missing_band = (band_intersection & (~mask_i)).sum() / area_intersection

            mismatch_matrix[band_idx, i] = max(outside_band, missing_band)

    return mismatch_matrix, combinations


def find_optimal_epsilon(mismatch_matrix, target_rate=1/6, tol=1e-3):
    low, high = 0.0, 1.0
    optimal_eps = None

    while high - low > tol:
        mid = (low + high) / 2
        match_matrix = (mismatch_matrix < mid).astype(int)
        match_rate = match_matrix.sum() / mismatch_matrix.size

        if match_rate < target_rate:
            low = mid
        else:
            high = mid
            optimal_eps = mid

    return optimal_eps


def compute_relaxed_cbd_eps(mismatch_matrix, epsilon):
    relaxed_mask = mismatch_matrix < epsilon
    cbd_scores = relaxed_mask.sum(axis=0) / relaxed_mask.shape[0]
    return cbd_scores

def longest_contour(contours):
    return max(contours, key=lambda c: c.shape[0]) if contours else None

def plot_original_contours(data, contour_level=0.0, max_plots=30):
    """
    Plot contours from the original scalar fields in `data`.

    Parameters:
        data (np.ndarray): Shape (n_samples, height, width), scalar field stack.
        contour_level (float): Level set to extract contours.
        max_plots (int): Maximum number of contours to plot.
    """
    n_samples = data.shape[0]
    max_plots = min(max_plots, n_samples)

    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(max_plots):
        contours = measure.find_contours(data[i], level=contour_level)
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color='blue', alpha=0.6, linewidth=1)

    # ax.set_title(f"Original Contours from {max_plots} Samples (level={contour_level})")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def compute_pixelwise_frequency_map(masks):
    """
    Compute the per-pixel coverage frequency: for each pixel, the fraction of
    input contours (masks) that include it.

    Parameters:
        masks (np.ndarray): Boolean array of shape (n_samples, H, W),
            True if the pixel is inside the contour for that sample.

    Returns:
        freq_map (np.ndarray): Array of shape (H, W) with values in [0, 1],
            equal to k(p)/n where k(p) is the number of masks covering pixel p.
    """
    if masks.ndim != 3:
        raise ValueError("masks must have shape (n_samples, H, W)")

    n_samples, H, W = masks.shape
    if n_samples == 0:
        return np.zeros((H, W), dtype=np.float32)

    if masks.dtype != np.bool_:
        masks = masks.astype(np.bool_, copy=False)

    counts = masks.sum(axis=0, dtype=np.float32)
    return counts

    
def compute_pixelwise_band_depth(masks):
    """
    Compute the continuous band depth (cBD) score for each pixel.
    
    Parameters:
        masks (np.ndarray): Boolean array of shape (n_samples, H, W), each element is True for superlevel set inclusion.
    
    Returns:
        depth_field (np.ndarray): Array of shape (H, W) with cBD scores in [0, 1].
    """
    n_samples = masks.shape[0]
    H, W = masks.shape[1], masks.shape[2]
    depth_field = np.zeros((H, W), dtype=np.float32)

    # All pairs (j, k) with j < k
    combinations = list(itertools.combinations(range(n_samples), 2))
    # for j, k in tqdm(combinations, desc="Computing pixelwise cBD"):
    for j, k in combinations:
        intersection = masks[j] & masks[k]
        depth_field += intersection.astype(np.float32)

    norm_factor = len(combinations)
    if norm_factor > 0:
        depth_field /= norm_factor

    return depth_field

def plot_contour_boxplot(data, cbd_scores, contour_level=0.0):
    n_samples = data.shape[0]
    sorted_indices = np.argsort(cbd_scores)[::-1]
    median_idx = sorted_indices[0]
    central_indices = sorted_indices[:int(0.5 * n_samples)]
    outlier_indices = np.where(cbd_scores == 0)[0]
    inlier_indices = [i for i in range(n_samples) if i not in outlier_indices]

    masks = data >= contour_level
    mean_mask = masks.sum(axis=0) > (n_samples // 2)

    # Band masks
    central_union = np.any(masks[central_indices], axis=0)
    central_intersection = np.all(masks[central_indices], axis=0)
    band_50_mask = central_union.astype(int) - central_intersection.astype(int)

    inlier_union = np.any(masks[inlier_indices], axis=0)
    inlier_intersection = np.all(masks[inlier_indices], axis=0)
    band_100_mask = inlier_union.astype(int) - inlier_intersection.astype(int)

    fig, ax = plt.subplots(figsize=(8, 8))
    X, Y = np.meshgrid(np.arange(data.shape[2]), np.arange(data.shape[1]))

    # 100% band using contourf
    ax.contourf(X, Y, band_100_mask.astype(float), levels=[0.5, 1.5], colors='lightgray', alpha=0.2)
    ax.contour(X, Y, band_100_mask.astype(float), levels=[0.5], colors='black', linewidths=1)

    # 50% band using contourf
    ax.contourf(X, Y, band_50_mask.astype(float), levels=[0.5, 1.5], colors='gray', alpha=0.35)
    ax.contour(X, Y, band_50_mask.astype(float), levels=[0.5], colors='black', linewidths=1.5)

    # Outliers (depth = 0)
    for idx in outlier_indices:
        contours = measure.find_contours(data[idx], level=contour_level)
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color='red', linestyle='--', linewidth=1)

    # Median (highest cBD) - longest contour
    # Median (all contours from most central sample)
    contours = measure.find_contours(data[median_idx], level=contour_level)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='gold', linewidth=2)

    # Mean contour (region in >50% samples) - longest contour
    contours_mean = measure.find_contours(mean_mask.astype(float), 0.5)
    for contour_mean in contours_mean:
        ax.plot(contour_mean[:, 1], contour_mean[:, 0], color='purple', linewidth=2)

    plt.ylim([26, 230])
    # ax.set_title("Contour Boxplot")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    # data: (n_samples, height, width) numpy array of scalar fields
    # contour_level: scalar threshold
    #
    # masks = data >= contour_level
    # Generate synthetic data
    np.random.seed(0)
    n_samples = 30  # reduced for tractability
    height, width = 100, 100
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    base_field = np.sin(X) + np.cos(Y)
    
    data = np.stack([
        add_smooth_noise(base_field, noise_scale=0.5, smoothing_sigma=2.0)
        for _ in range(n_samples)
    ])

    # Use superlevel sets for contour band depth masks
    contour_level = 0.0
    masks = data >= contour_level  # shape (n_samples, height, width)

    mismatch_matrix, _ = compute_mismatch_matrix(masks)
    eps_star = find_optimal_epsilon(mismatch_matrix)
    print(f"eps_star={eps_star}")
    cbd_eps_scores = compute_relaxed_cbd_eps(mismatch_matrix, eps_star)
    plot_contour_boxplot(data, cbd_eps_scores, contour_level)
    plot_original_contours(data, contour_level, max_plots=1)
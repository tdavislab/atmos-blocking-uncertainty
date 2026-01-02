# Spatiotemporal Atmospheric Blocking Detection and Uncertainty Visualization

This repository provides the source code for detecting atmospheric blocking events and summarizing their spatiotemporal behavior with uncertainty-aware visualizations.

The implementation accompanies the manuscript

> **Spatiotemporal Detection and Uncertainty Visualization of Atmospheric Blocking Events**  
> Mingzhe Li, Peer Nowack, Bei Wang  
> IEEE Pacific Visualization Symposium (PacificVis) TVCG Journal Track, accepted, 2026.  
> IEEE Transactions on Visualization and Computer Graphics, to appear, 2026  

The core pipeline is implemented in **Python**, with optional **ParaView** + **pvpython** scripts for 3D visual exploration.

---

## Dataset

The datasets used in this project (ERA5 and UKESM experiments) are hosted on Zenodo:

- **Zenodo record:** <https://zenodo.org/records/17674393>

To run the pipeline:

1. Download the files from the Zenodo record.
2. Place all downloaded files under:

```text
   data-preprocess/
```

You may choose a different folder layout, but then you **must** update the dataset configuration JSON files and any hard-coded paths (see below).

---

## Video demo

A video demo for interaction with 3D temporal visualizations is available: <https://youtu.be/9ioYrPCWTSM>

---

## Repository structure

At the top level:

```text
atmos-blocking-uncertainty/
├─ GTD/
│  ├─ ERA5-ground-truth.csv        # Ground-truth labels for ERA5 datasets
│  └─ UKESM-ground-truth.csv       # Ground-truth labels for UKESM datasets
├─ assets/
│  └─ ...                          # Coastlines and other VTK assets (*.vtp, *.vtu, etc.)
├─ data-preprocess/
│  ├─ data_utils.py                # Shared utilities for loading / preprocessing data
│  ├─ nc2mat.py                    # Preprocess NetCDF to 2D intermediate arrays
│  ├─ mat2vtk.py                   # (pvpython) Convert processed data to VTK volumes
│  └─ download_ERA5.py             # (optional) helper script for downloading ERA5
├─ src/
│  ├─ region_overlap_detect.py     # Core blocking detection + tracking pipeline
│  ├─ region_overlap_boxplot.py    # Summary + uncertainty visualization (e.g., contour boxplots)
│  ├─ region_overlap_util.py       # Utilities for region overlap / tracking logic
│  ├─ boxplot_util.py              # Helper functions for boxplot / summary statistics
│  ├─ contour2vtk.py               # (pvpython) Convert contours to VTK surfaces
│  ├─ volume_stack_3d.py           # (pvpython) Build 3D volume stacks for ParaView
│  ├─ dataset_ERA5.json            # Paths and configuration for ERA5 experiments
│  └─ dataset_UKESM.json           # Paths and configuration for UKESM experiments
├─ LICENSE
└─ README.md
```

Scripts labeled **(pvpython)** are intended to be run with ParaView’s Python interpreter (`pvpython`), not the standard system Python.

---

## Dependencies

### Python environment

The non-ParaView scripts use (directly or via imports):

* `xarray`
* `numpy`
* `dask` (and `dask.diagnostics.ProgressBar`)
* `gcsfs`
* `netCDF4`
* `cftime`
* `scipy`
* `matplotlib`
* `pandas`
* `tqdm`
* `scikit-image`
* plus standard library modules (`datetime`, `json`, `re`, `collections`, etc.)

A typical conda setup:

```bash
# Create and activate environment
conda create -n atmos-blocking python=3.11 -y
conda activate atmos-blocking

# Core scientific stack
conda install -c conda-forge \
  xarray dask gcsfs netcdf4 cftime scipy matplotlib pandas tqdm scikit-image -y
```

Install any additional tools (e.g., `jupyterlab`) as you like.

---

### ParaView and `pvpython` (optional)

ParaView is **only required** if you want to:

* Run `mat2vtk.py`, `contour2vtk.py`, or `volume_stack_3d.py`, and/or
* Explore results interactively in 3D.

1. Download and install ParaView (e.g., from the official ParaView website).
2. Ensure that the `pvpython` executable is available in your `PATH`:

   ```bash
   pvpython --version
   ```

This project was developed with **ParaView 5.13.3**. Other 5.13.x builds should also work, as the scripts only use standard functionality (basic readers, writers, and filters).

---

## Quickstart workflow

This is the recommended end-to-end workflow using the ERA5 or UKESM datasets.

### 1. Download and place the data

1. Download the dataset from the Zenodo record:
   [https://zenodo.org/records/17674393](https://zenodo.org/records/17674393)
2. Place the downloaded files in:

   ```text
   data-preprocess/
   ```

If you use a different folder naming scheme, adjust the JSON configs and any hard-coded paths accordingly.

---

### 2. Preprocess NetCDF → `.npz` (and intermediates)

From the repository root:

```bash
python data-preprocess/nc2mat.py
```

This script:

* Reads the NetCDF files from `data-preprocess/` (e.g., ERA5 / UKESM fields).
* Prepares the variables and grid used in the blocking detection pipeline.
* Writes `.npz` or other intermediate outputs used by later scripts.

Check the comments and configuration at the top of `data-preprocess/nc2mat.py` and `data-preprocess/data_utils.py` for:

* Expected file patterns (names, dimensions, variables).
* Output directories and file naming.

---

### 3. (Optional, ParaView) Convert `.npz` → VTK volumes

If you plan to visualize the full fields in ParaView, run:

```bash
pvpython data-preprocess/mat2vtk.py
```

This converts the processed data into VTK files (e.g., `.vtu`, `.vtp`) that you can load as background fields or reference volumes in ParaView.

---

### 4. Configure dataset JSONs and paths (important)

Before running the detection:

1. Edit the dataset configuration files in `src/`:

   * `src/dataset_ERA5.json`
   * `src/dataset_UKESM.json`

   Ensure that:

   * Input paths point to the preprocessed data you just generated.
   * Output directories exist and are writable.

2. There may be **additional hard-coded paths** in some scripts (e.g., absolute paths to local directories, scratch folders, or cached results).
   It is recommended to search the repo for such paths and update them for your system, for example:

   * Search for drive letters on Windows (`C:/`, etc.).

---

### 5. Run blocking detection

Core detection is performed by:

```bash
python src/region_overlap_detect.py --dataset=ERA5
# or
python src/region_overlap_detect.py --dataset=UKESM
```

This script:

* Loads the configured dataset (ERA5 or UKESM) using the corresponding JSON file.
* Identifies high-pressure regions in each time step.
* Tracks these regions using a **region overlap** criterion to define events.
* Enforces blocking conditions such as minimum duration (e.g., ≥ 5 days) and spatial constraints.

Outputs typically include:

* Blocking masks (per time and grid point).
* Event metadata (start & end times, event areas).
* Any intermediate debugging or diagnostic files, depending on configuration.

#### Optional: parameter tuning

There is code inside `src/region_overlap_detect.py` to perform **parameter tuning** (e.g., exploring different thresholds or tracking criteria). To enable this:

* Open `src/region_overlap_detect.py`.
* Locate the block that controls parameter sweeps / tuning (see comments in the script).
* Modify or enable this block as needed.
* Re-run:

  ```bash
  python src/region_overlap_detect.py --dataset=ERA5
  ```

This will generate additional outputs for each parameter combination and may take longer to run.

---

### 6. Summarize blocking and uncertainty

Once detection is complete, use:

```bash
python src/region_overlap_boxplot.py --dataset=ERA5
# or
python src/region_overlap_boxplot.py --dataset=UKESM
```

This script:

* Reads the detected blocking events and associated metadata.
* Aggregates events over time and across ensemble members / experiments.
* Computes summary statistics (e.g., frequency, persistence, spatial coverage).
* Produces **uncertainty-aware** contour summaries such as contour boxplots.

The output typically includes:

* JSON files for all contours of blocking events
* JSON files for median contours and bands at different depths
* NPZ files for frequency heatmaps 

See `src/region_overlap_boxplot.py` and `src/boxplot_util.py` for the exact output formats and paths.

---

### 7. (Optional, ParaView) Export contours and volume stacks

If ParaView is installed and you want to inspect the results in 3D:

```bash
pvpython src/contour2vtk.py ERA5-normalize
pvpython src/volume_stack_3d.py ERA5-normalize
```

These scripts:

* Convert contour information from the detection / summarization steps into `.vtp` surface files.
* Construct 3D volume stacks (e.g., with time as a third dimension) saved as `.vtu` files.

Check the script headers for:

* Expected input folders (where they look for detection/summary results).
* Output directories and file naming.

---

### 8. Visualize in ParaView

1. Open ParaView.

2. Load:

   * Background assets from `assets/` (e.g., coastlines, base fields) – `assets/*.vtu`, `assets/*.vtp`.
   * The generated contour and volume files from `contour2vtk.py` and `volume_stack_3d.py`.

3. Use ParaView’s tools to explore:

   * Time evolution of blocking events.
   * Overlays of blocking contours on base fields.
   * Slices, clips, and camera paths to illustrate spatiotemporal patterns.

This step is optional and purely for interactive exploration and presentation; all core analysis results are produced by the Python scripts.

---

## Ground truth utilities (`GTD/`)

The `GTD/` directory contains scripts and data for constructing / using ground-truth labels:

* `GTD/ERA5-ground-truth.csv` – labels for ERA5 dataset.
* `GTD/UKESM-ground-truth.csv` – labels for UKESM dataset.

These can be used to:

* Validate detections against reference labels.
* Compute evaluation metrics.
* Reproduce case-study comparisons in the manuscript.

---

## Reproducibility checklist

To reproduce a full run for ERA5 or UKESM:

1. Create and activate a Python environment, and install required packages.

2. (Optional) Install ParaView and ensure `pvpython` works.

3. Download the dataset from Zenodo and place it under `data-preprocess/`.

4. Run:

   ```bash
   python data-preprocess/nc2mat.py
   ```

5. (Optional) Run:

   ```bash
   pvpython data-preprocess/mat2vtk.py
   ```

6. Edit `src/dataset_ERA5.json` / `src/dataset_UKESM.json` and update any hard-coded paths.

7. Run detection:

   ```bash
   python src/region_overlap_detect.py --dataset=ERA5
   ```

8. Run summarization:

   ```bash
   python src/region_overlap_boxplot.py --dataset=ERA5
   ```

9. (Optional) Export VTK artefacts:

   ```bash
   pvpython src/contour2vtk.py ERA5-normalize
   pvpython src/volume_stack_3d.py ERA5-normalize
   ```

10. (Optional) Explore all generated assets in ParaView.

---

## Citation

If you use this code in your own work, please cite the underlying manuscript:

```bibtex
@article{Li2025AtmosBlocking,
  title  = {Spatiotemporal Detection and Uncertainty Visualization of Atmospheric Blocking Events},
  author = {Li, Mingzhe and Nowack, Peer and Wang, Bei},
  year   = {2026},
  note   = {to appear},
  journal = {IEEE Transactions on Visualization and Computer Graphics}
}
```

Update the bibliographic information once the paper is formally published.



import sys
import os
import re
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from paraview.simple import *
from tqdm import tqdm
import cftime
    
    
def save2DUnstructuredGridOnSphere(data, outFile, sphere_radius=10):
    scalarName = "zg"
    scalarfield = data["data"]  # shape: (nlats, nlons)
    lats = np.radians(data["lats"])  # in radians
    lons = np.radians(data["lons"])

    nlats, nlons = scalarfield.shape
    
    def coordID(i, j):
        return i * nlons + j

    # Convert lat/lon to Cartesian coordinates on sphere
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    x = sphere_radius * np.cos(lat_grid) * np.cos(lon_grid)
    y = sphere_radius * np.cos(lat_grid) * np.sin(lon_grid)
    z = sphere_radius * np.sin(lat_grid)

    points = vtk.vtkPoints()
    for i in range(nlats):
        for j in range(nlons):
            points.InsertNextPoint(x[i, j], y[i, j], z[i, j])

    # Create unstructured grid and add points
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)

    # Add quad cells with longitude wrap-around
    for i in range(nlats - 1):
        for j in range(nlons):
            jp1 = (j + 1) % nlons
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, coordID(i, j))
            quad.GetPointIds().SetId(1, coordID(i + 1, j))
            quad.GetPointIds().SetId(2, coordID(i + 1, jp1))
            quad.GetPointIds().SetId(3, coordID(i, jp1))
            ugrid.InsertNextCell(quad.GetCellType(), quad.GetPointIds())

    # Optional: Handle poles if your data includes them
    # Example: create triangles to connect last latitude row points to a pole point

    # Add scalar data
    scalar_array = numpy_to_vtk(scalarfield.ravel(order='C'))
    scalar_array.SetName(scalarName)
    ugrid.GetPointData().AddArray(scalar_array)

    # Add latitude and longitude as point data arrays (in degrees)
    lat_array = numpy_to_vtk(np.degrees(lat_grid).ravel(order='C'))
    lat_array.SetName("latitude")
    ugrid.GetPointData().AddArray(lat_array)

    lon_array = numpy_to_vtk(np.degrees(lon_grid).ravel(order='C'))
    lon_array.SetName("longitude")
    ugrid.GetPointData().AddArray(lon_array)
    
    lat_index_array = vtk.vtkIntArray()
    lat_index_array.SetName("lat_index")
    lat_index_array.SetNumberOfValues(nlats * nlons)  # N, M = grid size

    lon_index_array = vtk.vtkIntArray()
    lon_index_array.SetName("lon_index")
    lon_index_array.SetNumberOfValues(nlats * nlons)

    for i in range(nlats):  # latitude loop
        for j in range(nlons):  # longitude loop
            idx = coordID(i, j)
            lat_index_array.SetValue(idx, i)
            lon_index_array.SetValue(idx, j)

    ugrid.GetPointData().AddArray(lat_index_array)
    ugrid.GetPointData().AddArray(lon_index_array)
    
    time_array = vtk.vtkStringArray()
    time_array.SetName("time")
    time_array.SetNumberOfValues(1)
    time_array.SetValue(0, str(data["time"]).replace("=", ""))  # your time string

    # Add this string array to field data (global dataset metadata)
    ugrid.GetFieldData().AddArray(time_array)

    # Write to .vtu file (VTK unstructured grid format)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(outFile)
    writer.SetInputData(ugrid)
    writer.Write()
    

def save2DStructuredGridFromLatLon(data, outFile, Europe_only=False):
    """
    Save scalar field on a lat/lon grid into a 2D VTK structured grid (.vts file).
    Coordinates are mapped so that (lat, lon) = (0, 180) → (x, y) = (0, 0),
    and (lat, lon) = (0, 179) → (x, y) = (359, 0).
    """
    scalarName = "zg"
    scalarfield = data["data"]  # shape: (nlats, nlons)
    lats = data["lats"]         # in degrees
    lons = data["lons"]         # in degrees
    time_str = str(data["time"]).replace("=", "")
    
    # --- NEW: optional Europe cropping ---
    # Europe box: lon in [-10, 40] (degrees), lat in [30, 75]
    if Europe_only:
        # Normalize longitudes to [-180, 180] for easy comparison
        lons_pm180 = ((lons + 180.0) % 360.0) - 180.0
        lat_mask = (lats >= 30.0) & (lats <= 75.0)
        lon_mask = (lons_pm180 >= -10.0) & (lons_pm180 <= 40.0)

        # If masks are empty, keep behavior predictable (avoid slicing to empty)
        if np.any(lat_mask) and np.any(lon_mask):
            scalarfield = scalarfield[np.ix_(lat_mask, lon_mask)]
            lats = lats[lat_mask]
            lons = lons[lon_mask]
        else:
            # If nothing matches, return early or fall back without cropping.
            # Here we choose to fall back without cropping.
            pass
    # --- end NEW ---
    
    nlats, nlons = scalarfield.shape

    # Shift longitudes so 180E is first
    shift_idx = np.argmax(lons >= 180)
    lon_shifted = np.roll(lons, -shift_idx)
    scalarfield_shifted = np.roll(scalarfield, -shift_idx, axis=1)

    # Compute x = (lon - 180) % 360, y = lat
    lon_x = (lon_shifted - 180) % 360
    lat_y = lats

    # Use indexing='xy' to match (i, j) layout: (nlats, nlons)
    lon_grid, lat_grid = np.meshgrid(lon_x, lat_y, indexing='xy')
    x = lon_grid
    y = lat_grid
    z = np.zeros_like(x)

    # Build structured grid
    points = vtk.vtkPoints()
    for i in range(nlats):
        for j in range(nlons):
            points.InsertNextPoint(x[i, j], y[i, j], z[i, j])

    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(nlons, nlats, 1)
    sgrid.SetPoints(points)

    # Scalars and coordinate arrays (ravel in C order for structured grid)
    scalar_array = numpy_to_vtk(scalarfield_shifted.ravel(order='C'))
    scalar_array.SetName(scalarName)
    sgrid.GetPointData().AddArray(scalar_array)

    lat_array = numpy_to_vtk(lat_grid.ravel(order='C'))
    lat_array.SetName("latitude")
    sgrid.GetPointData().AddArray(lat_array)

    true_lon_grid, _ = np.meshgrid(lon_shifted, lat_y, indexing='xy')
    lon_array = numpy_to_vtk(true_lon_grid.ravel(order='C'))
    lon_array.SetName("longitude")
    sgrid.GetPointData().AddArray(lon_array)

    # Index arrays
    lat_index_array = vtk.vtkIntArray()
    lat_index_array.SetName("lat_index")
    lon_index_array = vtk.vtkIntArray()
    lon_index_array.SetName("lon_index")

    for i in range(nlats):
        for j in range(nlons):
            lat_index_array.InsertNextValue(i)
            lon_index_array.InsertNextValue(j)

    sgrid.GetPointData().AddArray(lat_index_array)
    sgrid.GetPointData().AddArray(lon_index_array)

    # Add time metadata
    time_array = vtk.vtkStringArray()
    time_array.SetName("time")
    time_array.InsertNextValue(time_str)
    sgrid.GetFieldData().AddArray(time_array)

    # Write VTS file
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(outFile)
    writer.SetInputData(sgrid)
    writer.Write()

    return outFile


def is_in_season(date, start_time, end_time):
    """Check if a cftime date is within a month-day range, ignoring year."""
    md = (date.month, date.day)
    
    if start_time <= end_time:
        return start_time <= md <= end_time
    else:
        # For ranges that span the new year (e.g., Nov–Feb)
        return md >= start_time or md <= end_time

def from_iso_to_cftime_manual(iso_str, calendar="360_day"):
    match = re.match(r'.*?(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})', iso_str)
    if not match:
        raise ValueError("Invalid ISO datetime format")
    y, m, d, H, M, S = map(int, match.groups())
    return cftime.datetime(y, m, d, H, M, S, calendar=calendar)
    
    
def load_data_anomaly(path, st_year, ed_year):
    years_list = os.listdir(path)
    out_years_list = []
    out_fname_list = []
    data_list_by_years = []
    for year_str in tqdm(years_list, desc="Loading data by years"):
        if st_year is not None:
            if int(year_str) < int(st_year):
                continue
        if ed_year is not None:
            if int(year_str) > int(ed_year):
                continue

        out_years_list.append(year_str)
        year_path = os.path.join(path, year_str)
        all_data_files = os.listdir(year_path)
        
        data_list = []
        fname_list = []
        # We load all data files within the year
        for filename in all_data_files:
            if filename.endswith(".npz"):
                try:
                    with np.load(os.path.join(year_path, filename)) as f:
                        if np.isnan(f['data']).all():
                            continue
                        data_temp = {key: f[key] for key in f.files}
                        data_list.append(data_temp)
                except:
                    print(year_path, filename)
                    exit()
                fname_list.append(filename)
        
        data_list_by_years.append(data_list)
        out_fname_list.append(fname_list)
    
    return out_years_list, out_fname_list, data_list_by_years


def main(argv):
    if len(argv) < 1:
        print("Usage: python mat2vtk.py data_fname [st_year] [ed_year]")
        return
    
    sphere_mesh = False
    Europe_only = True
    calendars = {
        "ERA5": "proleptic_gregorian",
        "UKESM": "360_day",
        "ERA5-normalize": "proleptic_gregorian",
        "UKESM-normalize": "360_day",
    }
    
    fname = argv[0]
    calendar = calendars[fname]
    if "UKESM" in fname:
        start_month, start_day = 5, 27
        end_month, end_day = 9, 4
    else:
        start_month, start_day = 5, 28
        end_month, end_day = 9, 4
    st_year = ed_year = None
    try:
        st_year = int(argv[1])
        ed_year = int(argv[2])
    except:
        print("Using default start and end year:", st_year, ed_year)
    print("fname:",fname)
    
    mat_data_root = fname # "UKESM" if "UKESM" in fname else "ERA5"
    years, fnames, data = load_data_anomaly(mat_data_root, st_year, ed_year)
    
    out_vtk_data_root = fname # "UKESM-2D" if "UKESM" in fname else "ERA5-2D"
    if not sphere_mesh:
        out_vtk_data_root += "-2D"    
    if Europe_only:
        out_vtk_data_root += "-Eu"
    out_vtk_data_root += "-VTK"
    os.makedirs(out_vtk_data_root, exist_ok=True)
    
    for i in tqdm(range(len(years)), desc="Generating VTK data by year"):
        year = years[i]
        fname_list = fnames[i]
        data_by_year = data[i]

        year_path = os.path.join(out_vtk_data_root, str(year))
        os.makedirs(year_path, exist_ok=True)

        # Use range(len(...)) and index inside to give tqdm length info
        for j in tqdm(range(len(fname_list)), 
                    desc=f"Processing days in {year}", 
                    leave=False):
            fname = fname_list[j]
            date_fname = from_iso_to_cftime_manual(fname.replace("=", ":"), calendar=calendar)
            if not is_in_season(date_fname, (start_month, start_day), (end_month, end_day)):
                continue
            data_by_day = data_by_year[j]
            fname_base = os.path.splitext(os.path.basename(fname))[0].replace("=", "").replace("-", "").replace("500zg", "zg").replace("T", "")
            if sphere_mesh:
                out_fname = os.path.join(year_path, fname_base + ".vtu")
                out_fname = out_fname.replace("0000.", ".")
                save2DUnstructuredGridOnSphere(data_by_day, out_fname)
            else:
                out_fname = os.path.join(year_path, fname_base + ".vts")
                out_fname = out_fname.replace("0000.", ".")
                save2DStructuredGridFromLatLon(data_by_day, out_fname, Europe_only)
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    # savePolyData(scalarfield, scalarName, output_filename, isTranspose)
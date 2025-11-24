import json
import os
import sys
from math import radians, cos, sin
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np

def load_contour_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)
    
def lonlat_to_hemisphere_xyz(lon, lat, radius=10.0):
    lon_rad = radians(lon)
    lat_rad = radians(lat)
    x = radius * cos(lat_rad) * cos(lon_rad)
    y = radius * cos(lat_rad) * sin(lon_rad)
    z = radius * sin(lat_rad)
    return x, y, z

def lonlat_to_2d_xyz(lon, lat):
    return lon + 180, lat, 0

def save2DUnstructuredGridOnSphere(data, outFile, scalarName="freq", sphere_radius=10):
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
        for j in range(nlons - 1):
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
    
    # time_array = vtk.vtkStringArray()
    # time_array.SetName("time")
    # time_array.SetNumberOfValues(1)
    # time_array.SetValue(0, str(data["time"]).replace("=", ""))  # your time string

    # Add this string array to field data (global dataset metadata)
    # ugrid.GetFieldData().AddArray(time_array)

    # Write to .vtu file (VTK unstructured grid format)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(outFile)
    writer.SetInputData(ugrid)
    writer.Write()
    
def save2DStructuredGridFromLatLon(data, outFile, scalarName="freq"):
    """
    Save scalar field on a lat/lon grid into a 2D VTK structured grid (.vts file).
    Coordinates are mapped so that (lat, lon) = (0, 180) → (x, y) = (0, 0),
    and (lat, lon) = (0, 179) → (x, y) = (359, 0).
    """
    scalarfield = data["data"]  # shape: (nlats, nlons)
    lats = data["lats"]
    lons = data["lons"]

    nlats, nlons = scalarfield.shape
    
    def coordID(i, j):
        return i * nlons + j
    
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

    # Write VTS file
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(outFile)
    writer.SetInputData(sgrid)
    writer.Write()


def build_polydata(contours, label, date_id, domain="sphere"):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    latitudes = []
    longitudes = []
    labels = []
    date_ids = []

    pt_id = 0
    for contour in contours:
        contour = np.array(contour)
        n_points = len(contour)
        if n_points < 2:
            continue

        ids = []
        for lon, lat in contour:
            if domain.lower() == "2d":
                x, y, z = lonlat_to_2d_xyz(lon, lat)
            else:
                x, y, z = lonlat_to_hemisphere_xyz(lon, lat)
            points.InsertNextPoint(x, y, z)
            latitudes.append(lat)
            longitudes.append(lon)
            ids.append(pt_id)
            date_ids.append(date_id)
            pt_id += 1

        lines.InsertNextCell(n_points)
        for i in ids:
            lines.InsertCellPoint(i)
        labels.append(label)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # Add metadata arrays
    arr_lat = numpy_to_vtk(np.array(latitudes, dtype=np.float32), deep=True)
    arr_lat.SetName("Latitude")
    polydata.GetPointData().AddArray(arr_lat)

    arr_lon = numpy_to_vtk(np.array(longitudes, dtype=np.float32), deep=True)
    arr_lon.SetName("Longitude")
    polydata.GetPointData().AddArray(arr_lon)
    
    arr_date_ids = numpy_to_vtk(np.array(date_ids, dtype=int), deep=True)
    arr_date_ids.SetName("Date_id")
    polydata.GetPointData().AddArray(arr_date_ids)

    arr_type = numpy_to_vtk(np.array([label] * lines.GetNumberOfCells(), dtype=int), deep=True)
    arr_type.SetName("Type")
    polydata.GetCellData().AddArray(arr_type)

    return polydata

def merge_polydata(polydata_list):
    append_filter = vtk.vtkAppendPolyData()
    for pd in polydata_list:
        append_filter.AddInputData(pd)
    append_filter.Update()
    return append_filter.GetOutput()

def write_vtp(polydata, output_file):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()

def main(meta_file, output_vtu, date_id, mode):
    with open(meta_file, 'r') as f:
        contour_files = json.load(f)

    label_mapping = {
        "median": 1,
        # "mean": 2,
        "band_50": 3,
        "band_100": 4
    }

    polydata_parts = []
    median_poly_parts = []
    for label_name in contour_files:
        if label_name in label_mapping:
            label_val = label_mapping[label_name]
        else:
            try:
                label_val = int(label_name)
            except:
                continue
        contours = load_contour_json(contour_files[label_name])
        pd = build_polydata(contours, label_val, date_id, mode)
        polydata_parts.append(pd)
        if label_val == 1:
            median_poly_parts.append(pd)

    merged = merge_polydata(polydata_parts)
    write_vtp(merged, output_vtu)
    return polydata_parts, median_poly_parts


if __name__ == "__main__":
    dataset = sys.argv[1]
    mode = "2d"
    if len(sys.argv) >= 2:
        for arg in sys.argv:
            if arg.lower() == "2d":
                mode = "2d"
                break
            
    print("Domain mode:", mode)
    root_dirs = [f"./{dataset}_contour_boxplot"] 
    for ei, root_dir in enumerate(root_dirs):
        subdirs = os.listdir(root_dir)
        subdirs.sort()
        
        # For each calendar date
        valid_date_counts = 0
        all_poly_parts = []
        for d in subdirs:
            if not os.path.isdir(os.path.join(root_dir, d)):
                continue
            if "archive" in d:
                continue
            file_nest = [root_dir, d, "contour_metadata.json"]
            out_nest = [root_dir, mode + "_" + d + ".vtp"]
            meta_path = os.path.join(*file_nest)
            output_path = os.path.join(*out_nest[:])
            
            # compute VTK objects from median contours
            poly_parts, median_poly_parts = main(meta_path, output_path, valid_date_counts, mode)
            all_poly_parts.append(median_poly_parts)
            valid_date_counts += 1
            
            freq_load = np.load(os.path.join(root_dir, d, "cBD_scores.npz"), allow_pickle=True)
            if mode == "sphere":
                save2DUnstructuredGridOnSphere(freq_load, output_path.replace(".vtp", ".vtu").replace("positive", f"freq_{mode}"))
            else:
                save2DStructuredGridFromLatLon(freq_load, output_path.replace(".vtp", ".vts").replace("positive", f"freq_{mode}"))
            
        all_poly_parts_nodaysep = []
        for each_part in all_poly_parts:
            all_poly_parts_nodaysep.extend(each_part)
        
        all_polys_merged = merge_polydata(all_poly_parts_nodaysep)
        write_vtp(all_polys_merged, f"{dataset}_all_contour_boxplots_{ei}.vtp")
    
    afterstr = ""
    freq_summary_load = np.load(os.path.join(".", f"frequency_map_{dataset}{afterstr}.npz"), allow_pickle=True)
    if mode == "sphere":
        save2DUnstructuredGridOnSphere(freq_summary_load, f"{dataset}_frequency_map_{mode}{afterstr}.vtu", "frequency")
    else:
        save2DStructuredGridFromLatLon(freq_summary_load, f"{dataset}_frequency_map_{mode}{afterstr}.vts", "frequency")
        
import numpy as np
import vtk
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import os
import sys

def create_volume_from_latlon_stack(data_stack, dz=1.0, scalar_name="binary"):
    nslices = len(data_stack)
    nlats, nlons = data_stack[0]["data"].shape

    # Assume uniform grid spacing in lat/lon
    lats = data_stack[0]["lats"]
    lons = data_stack[0]["lons"]
    dlat = lats[1] - lats[0]
    dlon = lons[1] - lons[0]

    # Pad coordinates to make consistent if needed
    shift_idx = np.argmax(lons >= 180)
    lons_shifted = np.roll(lons, -shift_idx)
    lon_x = (lons_shifted - 180) % 360
    lat_y = lats

    # Create full meshgrid (same for each slice)
    lon_grid, lat_grid = np.meshgrid(lon_x, lat_y, indexing="xy")  # shape (nlats, nlons)

    # Build vtkStructuredGrid
    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(nlons, nlats, nslices)

    points = vtk.vtkPoints()
    scalars = []

    for t, data in enumerate(data_stack):
        scalar2d = data["data"]
        scalar2d = np.roll(scalar2d, -shift_idx, axis=1)  # apply same shift
        z = t * dz

        for i in range(nlats):
            for j in range(nlons):
                x = lon_grid[i, j]
                y = lat_grid[i, j]
                points.InsertNextPoint(x, y, z)
                scalars.append(scalar2d[i, j])

    # Set points and scalar field
    sgrid.SetPoints(points)

    scalar_array = numpy_to_vtk(np.array(scalars), deep=True)
    scalar_array.SetName(scalar_name)
    sgrid.GetPointData().AddArray(scalar_array)
    sgrid.GetPointData().SetScalars(scalar_array)

    return sgrid

def export_structured_grid(sgrid, filename="volume.vts"):
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(sgrid)
    writer.Write()
    
if __name__ == "__main__":
    dataset = sys.argv[1]
    root_dirs = [f"./{dataset}_contour_boxplot"] 
    
    for ei, root_dir in enumerate(root_dirs):
        subdirs = os.listdir(root_dir)
        subdirs.sort()
        
        data_stack = []
        for d in subdirs:
            if not os.path.isdir(os.path.join(root_dir, d)):
                continue
            if "archive" in d:
                continue
            file_nest = [root_dir, d, "cbd_scores.npz"]
            meta_path = os.path.join(*file_nest)
            
            data_load = np.load(meta_path, allow_pickle=True)
            data_stack.append(data_load)
        
        volume = create_volume_from_latlon_stack(data_stack, dz=2, scalar_name="cBD")
        export_structured_grid(volume, os.path.join(root_dir, "cBD_volume.vts"))
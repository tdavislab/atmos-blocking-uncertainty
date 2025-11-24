- Run nc2mat.py after you download the ERA5 and UKESM dataset. 
  > Make sure they are in the same folder (or you change the file location in the code).
  > Usage: "python nc2mat.py data_fname [start_year] [end_year] [-normalize]"
  > "-normalize" is mandatory to replicate the results in the paper
  > data_fname choices: ERA5/UKESM

- Run mat2vtk.py if you need VTK files to visualize the field
  > Usage: "pvpython mat2vtk.py data_fname [start_year] [end_year]"
  > data_fname choices: ERA5/UKESM/ERA5-normalize/UKESM-normalize
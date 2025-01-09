# XSeasonsDetect

`XSeasonDetect` is a Python tool designed for the detection and analysis of meteorological and climatological seasons, designed to be compatible with [`xarray`](https://docs.xarray.dev/en/stable/index.html).

The tool is built upon a Machine Learning algorithm proposed by A. J Cannon in  the article [Defining climatological seasons using radially constrained clustering](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2005GL023410) (2005).


## Script mode
Create a new project with: 

`XSeas_newproj --name <project_name>`

In the data/raw/ERA5 create a file for each variable you wanto to include into the analysis. Fill each folder wit han .nc file named `final.nc`.




## Data Preprocessing Script for ERA5 NetCDF Files

Then run the script for the preprocessing of ERA5 data:

`XSeas_preprocessERA`

This script automates the preprocessing of ERA5 climate data stored in NetCDF format. Below is a step-by-step explanation of how the script works:

- Input Data Requirements:
    - Raw NetCDF files should be organized into subfolders within `data/raw/ERA5`.
    - A target grid file (`config/target_grid.txt`) must be present to define the spatial grid resolution.
    - A geographic boundary file (`data/raw/shapefiles/boundary.gpkg`) is required to clip the data spatially.

- Processing Workflow:
	1.	Folder Detection: The script scans `data/raw/ERA5` for subfolders containing raw NetCDF files.
	2.	Directory Setup: For each folder, it creates intermediate (`data/temp/ERA5`) and output (`data/preprocessed/ERA5`) directories if they don’t already exist.

- Preprocessing: The following operations are applied to each folder:
	1.	Regridding: Matches the spatial resolution defined in the target grid file.
	2.	Clipping: Restricts the data to the area defined in the boundary file.
	3.	Temporal Filtering: Keeps only the data within the specified time range (default: 1960–2020).
	4.	Overwrite Handling: If a preprocessed file (final.nc) already exists in the output directory, the user is prompted to overwrite or skip.

- Output:
    - Preprocessed data is saved as final.nc in the corresponding folder within data/preprocessed/ERA5.


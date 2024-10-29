# XSeasonsDetect

`XSeasonDetect` is a Python tool designed for the detection and analysis of meteorological and climatological seasons, designed to be compatible with [`xarray`](https://docs.xarray.dev/en/stable/index.html).

The tool is built upon a Machine Learning algorithm proposed by A. J Cannon in  the article [Defining climatological seasons using radially constrained clustering](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2005GL023410) (2005).


## Script mode
Create a new project with: 

`XSeas_newproj --name <project_name>`

In the data/raw/ERA5 create a file for each variable you wanto to include into the analysis. Fill each folder wit han .nc file named `final.nc`.

Then run the script for the preprocessing of ERA5 data:

`XSeas_preprocessERA`
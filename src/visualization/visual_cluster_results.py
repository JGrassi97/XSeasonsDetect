import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import cartopy.crs as ccrs


# -- TO MOVE IN UTILS

def get_ordinal_suffix(day):
    if 10 <= day <= 20:
        return 'th'
    else:
        last_digit = day % 10
        if last_digit == 1:
            return 'st'
        elif last_digit == 2:
            return 'nd'
        elif last_digit == 3:
            return 'rd'
        else:
            return 'th'

def day_of_year_to_date(day_of_year, year=None):
    if year is None:
        year = datetime.datetime.now().year
    
    try:
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
        day = date.day
        month = date.strftime("%b")
        ordinal_suffix = get_ordinal_suffix(day)
        return fr"{month} {day}{ordinal_suffix}"
    except ValueError:
        return "Invalid day of the year"
    


def plot_seasons_bk_results(result,
                            nrows=1, 
                            ncolumns=None, 
                            figsize=None, 
                            cmaps=None, 
                            titles=None, 
                            lims=None,
                            nlevs=6,
                            xlims=None,
                            ylims=None,
                            hspace=0.7,
                            wspace=0.1,
                            country_boundary=None, 
                            alpha = 1,
                            world_boundary=None,
                            contourf=False) -> plt.subplots:
    """
    Plots seasonal back-kernel (bk) results across clusters in a grid of subplots with custom configurations.

    Parameters:
    ----------
    result : xarray.DataArray
        The data to be plotted, with a `cluster` dimension to plot multiple clusters.
    nrows : int, optional
        Number of rows in the subplot grid (default is 1).
    ncolumns : int, optional
        Number of columns in the subplot grid. Defaults to the number of clusters.
    figsize : tuple, optional
        Figure size (width, height) in inches. Defaults to (5 * n_clusters, 5).
    cmaps : list of str, optional
        List of colormap names for each cluster plot (default is 'jet' for all).
    titles : list of str, optional
        Titles for each subplot (default is empty titles for all).
    lims : list of tuples, optional
        Colorbar limits for each plot in the form [(min, max), ...]. Defaults to the min/max of each cluster.
    nlevs : int, optional
        Number of contour levels (default is 6).
    xlims : tuple, optional
        x-axis limits in the form (xmin, xmax).
    ylims : tuple, optional
        y-axis limits in the form (ymin, ymax).
    hspace : float, optional
        Height space between subplots (default is 0.7).
    wspace : float, optional
        Width space between subplots (default is 0.1).
    country_boundary : any, optional
        Custom boundary for the country, if applicable.
    world_boundary : any, optional
        Custom boundary for the world, if applicable.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axs : numpy.ndarray
        Array of subplot axes.
    cbars : list of matplotlib.colorbar.Colorbar
        List of colorbars for each subplot.
    """
    
    # Determine number of clusters
    n_clusters = result.cluster.size

    # Set default number of columns to number of clusters if not specified
    if ncolumns is None:
        ncolumns = n_clusters

    # Set default figure size
    if figsize is None:
        figsize = (5 * n_clusters, 5)

    # Set default colormaps, titles, and limits
    if cmaps is None:
        cmaps = ['jet'] * n_clusters
    if titles is None:
        titles = [''] * n_clusters
    if lims is None:
        lims = [None] * n_clusters
    
    # Create subplot grid
    fig, axs = plt.subplots(nrows, ncolumns, figsize=figsize, gridspec_kw={'hspace': hspace, 'wspace': wspace})

    # List to hold colorbars for each plot
    cbars = []
    for j, cmap, title, lim, ax in zip(range(n_clusters), cmaps, titles, lims, axs.flatten()):

        # Select data for the current cluster
        to_plot = result.sel(cluster=j)

        # Define contour levels based on specified limits or data range
        if lim is None:
            lev = np.linspace(to_plot.min(), to_plot.max(), nlevs)
        else:
            lev = np.linspace(lim[0], lim[1], nlevs)
        
        # Plot contour fill for current cluster
        if contourf == True:
            plot = to_plot.plot.contourf(levels=lev, add_colorbar=False, ax=ax, cmap=cmap, alpha=alpha)
        
        else:
            plot = to_plot.plot(levels=lev, add_colorbar=False, ax=ax, cmap=cmap, alpha=alpha)    

        # Apply formatting and optional boundaries
        standard_format_single(plot,
                               country_boundary=country_boundary,
                               world_boundary=world_boundary,
                               custom_cbar=False,
                               xlims=xlims,
                               ylims=ylims)

        # Positioning the colorbar below each subplot
        pos = ax.get_position().bounds
        cbar_left = pos[0]
        cbar_bottom = pos[1] - 0.09
        cbar_width = pos[2]
        cbar_height = 0.015

        # Create horizontal colorbar with custom settings
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar = fig.colorbar(plot, cax=cbar_ax, orientation='horizontal', drawedges=True)
        cbar.set_ticks(lev)
        cbar.set_ticklabels([day_of_year_to_date(x) for x in lev])  # Custom tick labels for dates
        cbar.ax.tick_params(labelsize=9)

        # Customize colorbar outline and dividers
        cbar.outline.set_color('white')
        cbar.outline.set_linewidth(2)
        cbar.dividers.set_color('white')
        cbar.dividers.set_linewidth(2)

        # Append colorbar to list
        cbars.append(cbar)
        
        # Set subplot title
        ax.set_title(title)
    
    return fig, axs, cbars





def standard_format_single(plot, 
                           country_boundary=None, 
                           world_boundary=None, 
                           custom_cbar=True,
                           xlims=None,
                           ylims=None):

    fig = plot.figure  
    ax = plot.axes

    ax.set_ylabel('Latitude [°N]')
    ax.set_xlabel('Longitude [°E]')

    if country_boundary is not None:
        country_boundary.boundary.plot(ax=ax, color = 'k', linewidth = 0.6)
    
    if world_boundary is not None:
        world_boundary.boundary.plot(ax=ax, color = 'k', linewidth = 0.4)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(alpha=0.3)
    
    if custom_cbar:
        
        cbar = plt.colorbar(plot, orientation='horizontal', drawedges=True)
        cbar.outline.set_color('white')
        cbar.outline.set_linewidth(2)
        cbar.dividers.set_color('white')
        cbar.dividers.set_linewidth(2)

    if xlims is not None:
        ax.set_xlim(xlims)
    
    if ylims is not None:
        ax.set_ylim(ylims)

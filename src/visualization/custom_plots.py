import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

def standard_format_single(plot, country_boundary=None, world_boundary=None, custom_cbar=True):

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
        cbar.dividers.set_linewidth(1.5)
    
    ax.set_xlim(65, 99)
    ax.set_ylim(6, 36)


def standard_format(plot, country_boundary=None, world_boundary=None, custom_cbar=True):

    fig = plot.fig
          
    for ax in plot.axs.ravel():

        if ax.get_ylabel() == 'lat':
            ax.set_ylabel('Latitude [°N]')
        
        if ax.get_xlabel() == 'lon':
            ax.set_xlabel('Longitude [°E]')

        if country_boundary is not None:
            country_boundary.boundary.plot(ax=ax, color = 'k', linewidth = 0.6)
        
        if world_boundary is not None:
            world_boundary.boundary.plot(ax=ax, color = 'k', linewidth = 0.4)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(alpha=0.3)
    
    if custom_cbar:

        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        
        # Append axes to the right of ax, with the specified padding
        cax = fig.add_axes([0.2, -0.05, 0.6, 0.03])

        # Create colorbar in the new axes
        plot.add_colorbar(fraction=0.07, drawedges=True, cax=cax, orientation='horizontal')

        plot.cbar.outline.set_color('white')
        plot.cbar.outline.set_linewidth(2)
        plot.cbar.dividers.set_color('white')
        plot.cbar.dividers.set_linewidth(1.5)



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

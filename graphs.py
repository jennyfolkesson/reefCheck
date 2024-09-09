import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

# plotly.colors.sequential YlGnBu
COLOR_MAP = {
    2017: 'rgb(199,233,180)',
    2018: 'rgb(127,205,187)',
    2019: 'rgb(65,182,196)',
    2020: 'rgb(29,145,192)',
    2021: 'rgb(34,94,168)',
    2022: 'rgb(37,52,148)',
    2023: 'rgb(8,29,88)',
}


def view_change_points(few_days, change_points):
    fig = px.line(few_days, x=few_days.index, y=few_days.Temp)
    for cp in change_points[:-1]:
        fig.add_vline(x=few_days.index[cp], line_dash="dot")
    fig.update_layout(
        title="Change Point Detection",
        yaxis_title='Temperature (degrees C)',
        xaxis_title='Date',
    )
    fig.show()


def line_consecutive_years(temp_data, site_name):
    """
    Plot line graph with years/date on the x axis and temperature on y axis

    :param pd.DataFrame temp_data: Temperature data
    :param str site_name: Name of site where temperatures are recorded
    :return px.fig: Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=temp_data.index, y=temp_data["Temp"], name='Reef Check'))
    if 'SST' in list(temp_data):
        fig.add_trace(go.Scatter(
            x=temp_data.index, y=temp_data["SST"], name='Sea Surface Temp'))
    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        title="Water Temperatures at {}".format(site_name),
        yaxis_title='Temperature (degrees C)',
        xaxis_title='Date',
    )
    return fig


def line_overlaid_years(temp_data, site_name):
    """
    Plot data by month with years overlaid on top of each other on x-axis,
    temperature on y-axis.

    :param pd.DataFrame temp_data: Temperature data
    :param str site_name: Name of site where temperatures are recorded
    :return px.fig: Figure
    """
    temps = []
    for yr in temp_data.index.year.unique():
        yr_offset = 2000 - yr
        yr_data = temp_data.copy()
        yr_data = yr_data[yr_data.index.year == yr]
        yr_data['Year'] = yr_data.index.year
        yr_data["Date"] = yr_data.index + pd.offsets.DateOffset(years=yr_offset)
        temps.append(yr_data)

    yr_data = pd.concat(
        temps,
        axis=0,
    )

    fig = px.line(yr_data, x="Date", y="Temp", color='Year', color_discrete_map=COLOR_MAP)
    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        xaxis_tickformat="%B %d",
        title="Monthly water temperatures in {}".format(site_name),
        yaxis_title='Temperature (degrees C)',
        xaxis_title='Month',
    )
    fig.update_traces(connectgaps=False)
    return fig


def oisst_map(sst, sst_lat, sst_lon):

    fig = px.imshow(
        sst,
        color_continuous_scale='RdBu_r',
        origin='lower',
        x=sst_lon,
        y=sst_lat,
    )
    fig.update_layout(
        width=1000,
        height=600,
        xaxis_title="Longitude (degrees East)",
        yaxis_title="Latitude",
        coloraxis_colorbar=dict(
            title='Sea Surface Temp<br>Deg C',
        ),
    )
    return fig

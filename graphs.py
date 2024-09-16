import numpy as np
import os
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go


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
        x=temp_data.index,
        y=temp_data["Temp"],
        name='Reef Check',
        mode='lines',
    ))
    if 'SST' in list(temp_data):
        fig.add_trace(go.Scatter(
            x=temp_data.index,
            y=temp_data["SST"],
            name='Sea Surface Temp',
            mode='lines',
        ))
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


def coordinate_map(reef_meta):
    """
    Plots geographic coordinates for Reef Check sites as well as the closest matching
    OISST coordinates.

    :param pd.DataFrame reef_meta: Contains Reef Check and OISST coordinates for sites
    :return px.Fig: Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=reef_meta['Site Lat'],
        lon=reef_meta['Site Long'],
        mode='markers',
        marker=go.scattermap.Marker(
            size=10,
            color='rgb(0, 0, 255)',
            opacity=0.7
        ),
        text=reef_meta['Site'],
        hoverinfo='text'
    ))
    fig.add_trace(go.Scattermap(
        lat=reef_meta['sst_lat'],
        lon=reef_meta['sst_lon'],
        mode='markers',
        marker=go.scattermap.Marker(
            size=10,
            color='rgb(0, 255, 0)',
            opacity=0.7
        ),
        text=reef_meta['Site'],
        hoverinfo='text'
    ))
    fig.update_layout(
        title='Reef Check and OISST Coordinates',
        autosize=False,
        width=1100,
        height=1000,
        hovermode='closest',
        showlegend=False,
        map=dict(
            bearing=0,
            center=dict(
                lat=reef_meta['Site Lat'].mean(),
                lon=reef_meta['Site Long'].mean(),
            ),
            pitch=0,
            zoom=6,
            style='light'
        ),
    )
    return fig


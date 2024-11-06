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
        x=temp_data['Date'],
        y=temp_data["Temp"],
        name='Reef Check',
        mode='lines',
    ))
    if 'SST' in list(temp_data):
        fig.add_trace(go.Scatter(
            x=temp_data['Date'],
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
    for yr in temp_data['Date'].dt.year.unique():
        yr_offset = 2000 - yr
        yr_data = temp_data.copy()
        yr_data = yr_data[yr_data['Date'].dt.year == yr]
        yr_data['Year'] = yr_data['Date'].dt.year
        yr_data["Date"] = yr_data['Date'] + pd.offsets.DateOffset(years=yr_offset)
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
        xaxis_tickformat="%B",
        title="Water temperatures in {}".format(site_name),
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
        lat=reef_meta['Lat'],
        lon=reef_meta['Lon'],
        mode='markers',
        marker=go.scattermap.Marker(
            size=10,
            color='rgb(0, 0, 255)',
            opacity=0.5,
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
            color='rgb(255, 0, 0)',
            opacity=0.5,
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
                lat=reef_meta['Lat'].mean(),
                lon=reef_meta['Lon'].mean(),
            ),
            pitch=0,
            zoom=6,
            style='light'
        ),
    )
    return fig


def mean_temperature_lines(temp_data, freq='1D', title_txt=None):

    temp_mean = temp_data.copy()
    temp_mean = temp_mean.drop('Site', axis=1)
    if 'Region' in list(temp_mean):
        temp_mean = temp_mean.drop(['Region'], axis=1)
    if 'Code' in list(temp_mean):
        temp_mean = temp_mean.drop(['Code'], axis=1)
    temp_stats = temp_mean.groupby(
        pd.Grouper(key='Date', axis=0, freq=freq, sort=True),
    ).mean()
    temp_std = temp_mean.groupby(
        pd.Grouper(key='Date', axis=0, freq=freq, sort=True),
    ).std()
    temp_stats['Temp_std'] = temp_std['Temp']
    temp_stats['SST_std'] = temp_std['SST']
    temp_stats = temp_stats.dropna(subset=['Temp_std'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name='+std SST',
        x=temp_stats.index,
        y=temp_stats['SST'] + temp_stats['SST_std'],
        mode='lines',
        marker=dict(color='rgb(255, 0, 0)', opacity=0.75),
        line=dict(width=0),
        showlegend=False,
        opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        name='-std SST',
        x=temp_stats.index,
        y=temp_stats['SST'] - temp_stats['SST_std'],
        marker=dict(color='rgb(255, 0, 0)', opacity=0.75),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(255, 0, 0, .2)',  # '#EA8787',
        fill='tonexty',
        opacity=0.75,
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        name='+std Reef Check',
        x=temp_stats.index,
        y=temp_stats['Temp'] + temp_stats['Temp_std'],
        mode='lines',
        marker=dict(color='rgb(0, 0, 255)', opacity=0.75),
        line=dict(width=0),
        opacity=0.75,
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        name='-std Reef Check',
        x=temp_stats.index,
        y=temp_stats['Temp'] - temp_stats['Temp_std'],
        marker=dict(color='rgb(0, 0, 255)', opacity=0.75),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(0, 0, 255, .2)',  # '#87B8EA',
        fill='tonexty',
        opacity=0.75,
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        name='Sea Surface Temperature',
        x=temp_stats.index,
        y=temp_stats['SST'],
        mode='lines',
        line=dict(color='rgb(255, 0, 0)'),
    ))
    fig.add_trace(go.Scatter(
        name='Reef Check Temperature',
        x=temp_stats.index,
        y=temp_stats['Temp'],
        mode='lines',
        line=dict(color='rgb(0, 0, 255)'),
    ))
    if title_txt is None:
        title_txt = "Water temperature difference"
    fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
        xaxis_tickformat="%B %Y",
        title=title_txt,
        yaxis_title='Temperature (degrees C)',
        xaxis_title='Date',
    )
    return fig


def temperature_diff_graph(temperature_data, freq='1D', title_txt=None):

    temps = []
    for yr in temperature_data['Date'].dt.year.unique():
        yr_offset = 2000 - yr
        yr_data = temperature_data.copy()
        yr_data = yr_data[yr_data['Date'].dt.year == yr]
        yr_data['Year'] = yr_data['Date'].dt.year
        yr_data["Date"] = yr_data['Date'] + pd.offsets.DateOffset(years=yr_offset)
        temps.append(yr_data)
    yr_data = pd.concat(
        temps,
        axis=0,
    )
    yr_data['Diff'] = yr_data['SST'] - yr_data['Temp']
    yr_data = yr_data.drop('Site', axis=1)
    if 'Region' in list(yr_data):
        yr_data = yr_data.drop(['Region'], axis=1)
    if 'Code' in list(yr_data):
        yr_data = yr_data.drop(['Code'], axis=1)
    temp_stats = yr_data.groupby(
        pd.Grouper(key='Date', axis=0, freq=freq, sort=True),
    ).mean()
    temp_std = yr_data.groupby(
        pd.Grouper(key='Date', axis=0, freq=freq, sort=True),
    ).std()
    temp_stats['Diff_std'] = temp_std['Diff']
    temp_stats = temp_stats.dropna(subset=['Diff_std'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name='+std diff',
        x=temp_stats.index,
        y=temp_stats['Diff'] + temp_stats['Diff_std'],
        mode='lines',
        marker=dict(color='rgb(0, 0, 255)', opacity=0.75),
        line=dict(width=0),
        showlegend=False,
        opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        name='-std diff',
        x=temp_stats.index,
        y=temp_stats['Diff'] - temp_stats['Diff_std'],
        marker=dict(color='rgb(0, 0, 255)', opacity=0.75),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(0, 0, 255, .2)',  # '#87B8EA',
        fill='tonexty',
        opacity=0.75,
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        name='SST - Reef Check temperature difference',
        x=temp_stats.index,
        y=temp_stats['Diff'],
        mode='lines',
        line=dict(color='rgb(0, 0, 255)'),
    ))
    if title_txt is None:
        title_txt = "Water temperature difference"
    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        xaxis_tickformat="%B",
        title=title_txt,
        yaxis_title='Temperature (degrees C)',
        xaxis_title='Month',
        showlegend=False,
    )
    return fig

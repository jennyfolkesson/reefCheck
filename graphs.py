import datetime
import itertools
import numpy as np
import os
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def get_year_data(temperature_data):
    """
    Separate year from datetime column.

    :param pd.DataFrame temperature_data: Temperature data
    :return pd.DataFrame yr_data: Data where year is separated from date
    """
    temps = []
    for yr in temperature_data['Date'].dt.year.unique():
        yr_offset = 2000 - yr
        yr_data = temperature_data.copy()
        yr_data = yr_data[yr_data['Date'].dt.year == yr]
        yr_data['Year'] = yr_data['Date'].dt.year
        yr_data['Month'] = yr_data['Date'].dt.month
        yr_data["Date"] = yr_data['Date'] + pd.offsets.DateOffset(years=yr_offset)
        temps.append(yr_data)
    yr_data = pd.concat(
        temps,
        axis=0,
    )
    return yr_data


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
        y=temp_data["Temp Mean"],
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


def line_overlaid_years(temperature_data, site_name):
    """
    Plot data by month with years overlaid on top of each other on x-axis,
    temperature on y-axis.

    :param pd.DataFrame temperature_data: Temperature data
    :param str site_name: Name of site where temperatures are recorded
    :return px.fig: Figure
    """
    yr_data = get_year_data(temperature_data)
    fig = px.line(yr_data, x="Date", y="Temp Mean", color='Year', color_discrete_map=COLOR_MAP)
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


def _add_std_trace(fig, temp_stats, std_color, stat_name, tr_name, std_name):
    fig.add_trace(go.Scatter(
        name='+1 std {}'.format(tr_name),
        x=temp_stats.index,
        y=temp_stats[stat_name] + temp_stats[std_name],
        mode='lines',
        marker=dict(color=std_color, opacity=0.75),
        line=dict(width=0),
        showlegend=False,
        opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        name='-1 std {}'.format(tr_name),
        x=temp_stats.index,
        y=temp_stats[stat_name] - temp_stats[std_name],
        marker=dict(color=std_color, opacity=0.75),
        line=dict(width=0),
        mode='lines',
        fillcolor=std_color,
        fill='tonexty',
        opacity=0.75,
        showlegend=False,
    ))


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
    temp_stats['Temp_std'] = temp_std['Temp Mean']
    temp_stats['SST_std'] = temp_std['SST']
    temp_stats = temp_stats.dropna(subset=['Temp_std'])

    fig = go.Figure()
    _add_std_trace(fig, temp_stats, 'rgba(255, 0, 0, .2)', 'SST', 'SST', 'SST_std')
    _add_std_trace(fig, temp_stats, 'rgba(0, 0, 255, .2)', 'Temp Mean', 'Reef Check', 'Temp_std')
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
        y=temp_stats['Temp Mean'],
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
    """
    Show site temperature difference as a function of month aggregated
    across years.

    :param pd.DataFrame temperature_data: Temperature data
    :param str freq: Time frequency (default one day)
    :param str/None title_txt: Title text for graph
    :return go.Figure fig: Graph object
    """
    yr_data = get_year_data(temperature_data)
    yr_data['Diff'] = yr_data['SST'] - yr_data['Temp Mean']
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
    _add_std_trace(fig, temp_stats, 'rgba(0, 0, 255, .2)', 'Diff', 'Diff', 'Diff_std')
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
        title=title_txt,
        yaxis_title='Temperature (degrees C)',
        xaxis_title='Month',
        showlegend=False,
    )
    fig.update_xaxes(
        tickformat='%B',
        dtick='M1',
    )
    return fig


def temperature_depth_graph(temperature_data, reef_meta):
    """
    Plot circles for each site, where the size of the circle signify depth
    and the color of the circle is mean temperature in August.

    :param pd.DataFrame temperature_data: Temperature data
    :param pd.DataFrame reef_meta: Metadata including depth, lat, lon, etc
    :return go.Figure fig: Graph object
    """
    merged_data = temperature_data.merge(reef_meta, on='Site', how='left')
    yr_data = get_year_data(merged_data)
    temp_data = yr_data[['Temp Mean', 'Site', 'Month', 'depth (ft)', 'Lon', 'Lat']]
    temp_data = temp_data[temp_data['Month'] == 8]
    temp_data = temp_data.groupby('Site').mean(numeric_only=True).reset_index()
    temp_data = temp_data.dropna()
    temp_data['depth txt'] = temp_data['depth (ft)'].apply(lambda s: "{:.1f}".format(s))
    temp_data['temp txt'] = temp_data['Temp Mean'].apply(lambda s: "{:.1f}".format(s))

    hover_text = (temp_data['Site'] +
                  "<br> depth: " + temp_data['depth txt'] +
                  "<br> temp: " + temp_data['temp txt'])
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=temp_data['Lat'],
        lon=temp_data['Lon'],
        customdata=['Site', 'depth (ft)'],
        mode='markers',
        marker=go.scattermap.Marker(
            size=temp_data['depth (ft)'],
            color=temp_data['Temp Mean'],
            colorscale='thermal',
            showscale=True,
        ),
        text=hover_text,
        hoverinfo='text',
    ))

    fig.update_layout(
        title='Reef Check Sensor Depth',
        autosize=False,
        width=1100,
        height=1000,
        hovermode='closest',
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


def temp_depth_scatter(temperature_data, reef_meta, month=8):
    """
    Create a scatter plot of mean temperature for a specific month
    vs depth for different regions.

    :param pd.DataFrame temperature_data: Temperature data
    :param pd.DataFrame reef_meta: Reef Check metadata
    :param int month: Month (1-12)
    :return px.Figure: Scatter plot object
    """
    assert 1 <= month <= 12, "Month must be in the range [1,12]"
    merged_data = temperature_data.merge(reef_meta, on='Site', how='left')
    yr_data = get_year_data(merged_data)
    temp_data = yr_data[['Temp Mean', 'Site', 'Month', 'depth (ft)', 'Lon', 'Lat']]
    temp_data = temp_data[temp_data['Month'] == month]
    temp_data = temp_data.groupby('Site').mean(numeric_only=True).reset_index()
    temp_data = temp_data.dropna()
    merged_data = temp_data.merge(reef_meta[["Site", "Region"]], on='Site', how='left')

    fig_month = datetime.date(2000, month, 1).strftime('%B')
    fig_title = ("Mean Temperature vs Depth Across California "
                 "Regions in {}").format(fig_month)

    fig = px.scatter(
        merged_data,
        x="depth (ft)",
        y="Temp Mean",
        color="Region",
        hover_data=['Site'],
        trendline='ols',
    )
    fig.update_traces(marker_size=10)
    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        title=fig_title,
    )
    return fig


def temp_depth_annual_trendlines(temperature_data, reef_meta, region='Southern California'):
    """
    Create a scatter plot of mean temperature for a specific month
    vs depth for different regions.

    :param pd.DataFrame temperature_data: Temperature data
    :param pd.DataFrame reef_meta: Reef Check metadata
    :param str region: Region to analyze
    :return px.Figure: Scatter plot object
    """
    merged_data = temperature_data.merge(reef_meta, on='Site', how='left')
    yr_data = get_year_data(merged_data)
    yr_data = yr_data[yr_data['Region'] == region]
    temp_data = yr_data[['Temp Mean', 'Site', 'Month', 'depth (ft)', 'Lon', 'Lat']]

    temp_data = temp_data.groupby(['Site', 'Month']).mean(numeric_only=True).reset_index()
    temp_data = temp_data.dropna()
    temp_data = temp_data.sort_values(by=['depth (ft)'], ascending=False)

    fig = px.line(
        temp_data,
        x="depth (ft)",
        y="Temp Mean",
        color="Month",
    )
    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        title="Temperature vs depth for each month in Southern California",
    )
    return fig


def temp_depth_animation(temperature_data, reef_meta):
    # TODO: Some sites don't have all months which makes the animation look weird
    # Only include sites that span all months?
    merged_data = temperature_data.merge(reef_meta, on='Site', how='left')
    yr_data = get_year_data(merged_data)
    temp_data = yr_data[['Temp Mean', 'Site', 'Month', 'depth (ft)', 'Lon', 'Lat']]
    temp_data = temp_data.groupby(['Site', 'Month']).mean(numeric_only=True).reset_index()
    temp_data = temp_data.dropna()
    merged_data = temp_data.merge(reef_meta[["Site", "Region"]], on='Site', how='left')
    # merged_data
    fig = px.scatter(merged_data, x='depth (ft)', y='Temp Mean', animation_frame='Month',
               color='Region', range_y=[9, 22.5], hover_name='Site')
    return fig


def ecosystem_data_map(data_dir, file_name, class_code):
    # file_name = "Invert_California_Survey_means_2024.csv"
    file_path = os.path.join(data_dir, file_name)
    eco_data = pd.read_csv(file_path)

    lat_mid = (eco_data['Latitude'].min() +
               (eco_data['Latitude'].max() - eco_data['Latitude'].min()) / 2)
    lon_mid = (eco_data['Longitude'].min() +
               (eco_data['Longitude'].max() - eco_data['Longitude'].min()) / 2)

    df = eco_data[(eco_data['Classcode'] == class_code) &
                  (eco_data['MeanDens60m'] > 0)].copy()
    # (eco_data['Year'] == 2006) &

    df['Year txt'] = df['Year'].apply(lambda s: "{:d}".format(s))
    df['Dens txt'] = df['MeanDens60m'].apply(lambda s: "{:.1f}".format(s))
    hover_text = (df['Site'] +
                  "<br> Year: " + df['Year txt'] +
                  "<br> Density: " + df['Dens txt'])

    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=df['Latitude'],
        lon=df['Longitude'],
        # customdata=['Site', 'Year'],
        mode='markers',
        marker=go.scattermap.Marker(
            size=df['MeanDens60m']/50,
            color=df['Year'],
            colorscale='thermal',
            showscale=True,
        ),
        text=hover_text,
        hoverinfo='text',
    ))

    fig.update_layout(
        # title='Purple Urchin Ecosystem Map',
        autosize=False,
        width=1000,
        height=1200,
        hovermode='closest',
        map=dict(
            bearing=0,
            center=dict(
                lat=lat_mid,
                lon=lon_mid,
            ),
            pitch=0,
            zoom=6,
            style='light'
        ),
    )
    return fig


def ecosystem_subplots(data_dir, file_name, class_code, scale_down=50):
    # file_name = "Invert_California_Survey_means_2024.csv"
    file_path = os.path.join(data_dir, file_name)
    eco_data = pd.read_csv(file_path)

    lat_mid = (eco_data['Latitude'].min() +
               (eco_data['Latitude'].max() - eco_data['Latitude'].min()) / 2)
    lon_mid = (eco_data['Longitude'].min() +
               (eco_data['Longitude'].max() - eco_data['Longitude'].min()) / 2)

    df_class = eco_data[(eco_data['Classcode'] == class_code) &
                        (eco_data['MeanDens60m'] > 0)].copy()
    # Cap density at 1000 for visualization, change color instead for actual density
    df_class['DensCap'] = df_class['MeanDens60m'].clip(upper=1000)
    # Specify subplots
    yrs = eco_data.Year.unique()
    yrs.sort()
    year_txt = yrs[::3]
    year_txt = [str(s) for s in year_txt]
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=year_txt,
        start_cell="top-left",
        specs=[
            [{"type": "scattermap"}, {"type": "scattermap"}, {"type": "scattermap"}, {"type": "scattermap"}],
            [{"type": "scattermap"}, {"type": "scattermap"}, {"type": "scattermap"}, {"type": "scattermap"}]
        ],
    )

    grid_coords = list(itertools.product(list(range(1, 3)), list(range(1, 5))))
    # Visualize every 3 years
    for i, yr in enumerate(yrs[::3]):
        map_key = 'map' + str(i + 1) if i != 0 else 'map'
        df = df_class[df_class['Year'] == yr].copy()
        row_nbr, col_nbr = grid_coords[i]

        fig.add_trace(go.Scattermap(
                lat=df['Latitude'],
                lon=df['Longitude'],
                showlegend=False,
                name=str(yr),
                mode='markers',
                marker=go.scattermap.Marker(
                    size=df['DensCap']/scale_down,
                    color=df['MeanDens60m'],
                    colorscale='thermal',
                    showscale=False,
                ),
            ),
            row=row_nbr, col=col_nbr,
        )

        fig['layout'][map_key] = dict(
            domain=dict(
                x=[float(col_nbr - 1)/float(4), float(col_nbr)/float(4)],
                y=[1 - float(row_nbr)/float(2), 1 - float(row_nbr - 1)/float(2)]),
            bearing=0,
            center=dict(
                lat=lat_mid,
                lon=lon_mid,
            ),
            pitch=0,
            zoom=4.5,
            style='light',
        )

    fig.update_layout(
        autosize=False,
        width=1200,
        height=1200,
    )
    fig.update_annotations(yshift=20)
    return fig


def ecosystem_animation(data_dir, file_name, class_code):
    file_path = os.path.join(data_dir, file_name)
    eco_data = pd.read_csv(file_path)

    lat_mid = (eco_data['Latitude'].min() +
               (eco_data['Latitude'].max() - eco_data['Latitude'].min()) / 2)
    lon_mid = (eco_data['Longitude'].min() +
               (eco_data['Longitude'].max() - eco_data['Longitude'].min()) / 2)

    df_class = eco_data[(eco_data['Classcode'] == class_code) &
                        (eco_data['MeanDens60m'] > 0)].copy()
    df_class['DensCap'] = df_class['MeanDens60m'].clip(upper=1000)

    df_class = df_class.sort_values('Year')

    fig = px.scatter_mapbox(
        data_frame=df_class,
        lat="Latitude",
        lon="Longitude",
        animation_frame='Year',
        color_discrete_sequence='thermal',
        size='DensCap',
        color='MeanDens60m',
        # size_max=70,
        zoom=2,
        hover_name='Site',
        hover_data=['Site', 'MeanDens60m'],
    )

    map_bounds = {
        "west": df_class['Longitude'].min() - .5,
        "east": df_class['Longitude'].max() + .5,
        "south": df_class['Latitude'].min() - .5,
        "north": df_class['Latitude'].max() + .5,
    }

    fig.update_layout(
        # title='Purple Urchin Ecosystem Map',
        autosize=True,
        width=800,
        height=1000,
        hovermode='closest',
        mapbox_style="carto-positron",
        mapbox_bounds=map_bounds,
    )
    return fig

import argparse
import glob
import os
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import ruptures as rpt


def change_point_detection(temp_data, debug=False):
    """
    Searches for a change point in the first 10 days of time series and
    cuts off the rows before a detected change point.

    :param pd.DataFrame temp_data: Temperature time series data
    :return pd.DataFrame cropped_data: Temperature time series with initial temps removed
    """
    cropped_data = temp_data.copy()
    # Get median temperature for each hour
    hour_median = temp_data.resample('h', on='Datetime').median()
    # Do change point analysis, assume device is deployed in the first ten days
    first10 = hour_median.Temp[:240]
    algo = rpt.Dynp(model="l2", min_size=2)
    algo.fit(first10.values.reshape(-1, 1))
    # Only one breakpoint: when thermometer went into the water
    result = algo.predict(n_bkps=1)
    print(result)
    if len(result) == 2:
        # Change point detected
        change_point = first10.index[result[0]]
        # Debug plot of result
        if debug:
            first10 = hour_median.iloc[:240, :]
            fig = px.line(first10, x=first10.index, y=first10.Temp)
            fig.add_vline(x=change_point, line_dash="dot")
            fig.show()
        # Cut off temperature data before deployment
        cropped_data = cropped_data[cropped_data['Datetime'] > change_point]
    return cropped_data


def read_timeseries(data_dir):
    """
    Given path to data directory, find time series and read them.

    :param str data_dir: Data directory
    :return:
    """
    file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
    file_paths.sort()
    temps = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        year = int(file_name.split('_')[1])
        # Skip first row which only contains plot title
        temp_data = pd.read_csv(file_path, skiprows=1)
        # Remove # column (it's just an index)
        temp_data.drop(columns=['#'], inplace=True)
        assert temp_data.shape[1] == 2, "Data should have only Datetime and Temp"
        temp_data.columns = ['Datetime', 'Temp']
        # Convert to datetime format, all data seem to be formatted the same way
        temp_data['Datetime'] = pd.to_datetime(
            temp_data['Datetime'],
            format='%m/%d/%y %I:%M:%S %p',
            errors='coerce',
        )
        # Check for change point at the beginning of time series
        temp_data = change_point_detection(temp_data, debug=True)
        # Resample to day intervals, get median temperature
        temp_data = temp_data.resample('d', on='Datetime').median()
        temps.append(temp_data)

    # Concatenate all the temperature data files
    temp_data = pd.concat(
        temps,
        axis=0,
    )
    fig = px.line(temp_data, x=temp_data.index, y=temp_data.Temp)
    fig.update_layout(
        autosize=False,
        width=1000,
        height=700,
        title="Temperature ",
        yaxis_title='Temperature (degrees C)',
        xaxis_title='Date',
    )

    # Plot by month and day with years overlaid
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
    # TODO: maybe aggregate data more for better visual?
    fig = px.line(yr_data, x="Date", y="Temp", color='Year')
    fig.update_layout(xaxis_tickformat="%B-%d")

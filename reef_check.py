import argparse
import glob
import netCDF4
import numpy as np
import os
import pandas as pd
import ruptures as rpt
import subprocess

import graphs as graphs


OISST_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"


def parse_args():
    """
    Parse command line arguments

    :return args: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Path to directory containing temperature csv files',
    )
    parser.add_argument(
        '--resample',
        type=str,
        default='d',
        help='Resample rate of data. Default: d=day. Could do W=week',
    )
    return parser.parse_args()


def download_oisst(write_dir, years, months=None):
    """
    Downloads NOAA's OISST sea surface temperature data set for a given
    year and months within that year.

    :param str write_dir: Directory to write downloaded files to
    :param list[int] years: Years in range [1983, 2023]
    :param list[int] months: List of months as ints  in range[1, 12].
        Load all year if None.
    """
    # Create directory if not already there
    os.makedirs(write_dir, exist_ok=True)
    # Make sure date request is valid (we don't use 2024 data yet)
    if isinstance(years, int):
        years = [years]
    else:
        assert isinstance(years, list), "Years must be int or list of ints"
        years = [y for y in years if 1984 <= y <= 2023]
    if months is None:
        months = list(range(1, 13))
    elif isinstance(months, int):
        months = [months]
    else:
        assert isinstance(months, list), "Months must be int or list of ints"
        # Make sure elements are unique
        months = list(set(months))
        months = [m for m in months if 1 <= m <= 12]

    for year in years:
        for month in months:
            date = "{:4d}{:02d}".format(year, month)
            oisst_dir = os.path.join(OISST_URL, date)
            result = subprocess.run(
                ['wget', '-nc', '-r', '-l', '1', '-P', write_dir, oisst_dir],
            )


def get_geo_coords(site_name, write_dir):
    sst_path = os.path.join(
        write_dir,
        'oisst',
        OISST_URL.replace("https://", ""),
    )
    # Get only one directory
    one_dir = glob.glob(os.path.join(sst_path, '[!index]*'))[0]
    one_file = glob.glob(os.path.join(one_dir, 'oisst-avhrr*'))
    # Read CDF4 file
    oisst = netCDF4.Dataset(one_file)
    lat, lon = oisst.variables['lat'], oisst.variables['lon']
    # Sea surface temperature
    SST = oisst.variables['sst']
    sst = SST[:]

    # Read metadata file
    reef_path = os.path.join(write_dir,  "Reef Check_Current Hobo Deployments_2022.csv")
    reef_meta = pd.read_csv(reef_path)
    # TODO: get lat, lon for site and find nearest geograpich coordinate in OISST data

    idx_lat = (np.abs(lat - site_lat)).argmin()
    idy_lon = (np.abs(lon - site_lon)).argmin()


def change_point_detection(temp_data, at_start=True, nbr_days=14, penalty=60, debug=False):
    """
    Searches for a change point in the first 10 days of time series and
    cuts off the rows before a detected change point.

    :param pd.DataFrame temp_data: Temperature time series data
    :param bool at_start: Analyze beginning/end of time series
    :param int nbr_days: Number of days to use for change point detection
    :param int penalty: Penalty in change point detection algorithm
    :param bool debug: Generates debug plot if True
    :return pd.DataFrame cropped_data: Temperature time series with initial temps removed
    """
    cropped_data = temp_data.copy()
    # Get median temperature for each hour
    hour_mean = temp_data.resample('h', on='Datetime').mean()
    # Do change point analysis, assume device is deployed in the first couple days
    nbr_hours = nbr_days * 24
    if at_start:
        few_days = hour_mean.iloc[:nbr_hours]
    else:
        few_days = hour_mean.iloc[-nbr_hours:]
    # algo = rpt.Dynp(model="l2", min_size=5)
    # algo = rpt.Pelt(model="l2", min_size=5)
    # algo = rpt.Window(model="l2", width=25)
    algo = rpt.KernelCPD(kernel='linear', min_size=5)
    algo.fit(few_days.Temp.values.reshape(-1, 1))
    # There should  be only one or zero breakpoints:
    # when thermometer went into the water
    change_points = algo.predict(pen=penalty)
    if len(change_points) > 1:
        # Change point detected, use the first one found
        change_pos = -2
        if not at_start:
            change_pos = 0
        change_point = few_days.index[change_points[change_pos]]
        # Debug plot of result
        if debug:
            graphs.view_change_points(few_days, change_points)
        # Cut off temperature data before deployment
        if at_start:
            cropped_data = cropped_data[cropped_data['Datetime'] > change_point]
        else:
            cropped_data = cropped_data[cropped_data['Datetime'] < change_point]
    return cropped_data


def format_data(temp_data, dt_format='%m/%d/%y %I:%M:%S %p'):
    """
    Extract datetime and temperature columns from dataframe
    and format datetime column.

    :param pd.DataFrame temp_data: Temperature data
    :param str dt_format: Datetime format
    :return pd.DataFrame temp_data: Formatted dataframe
    """
    def find_name(col_names, search_str):
        found_col = [col for col in col_names if search_str in col.lower()]
        assert len(found_col) == 1, (
            "Ambiguous {} column: {}".format(search_str, found_col))
        return found_col

    col_names = temp_data.columns
    temp_col = find_name(col_names, 'temp')
    date_col = find_name(col_names, 'date')
    temp_data = temp_data[[date_col[0], temp_col[0]]].copy()
    temp_data.columns = ['Datetime', 'Temp']
    # Convert to datetime format, all data seem to be formatted the same way
    temp_data['Datetime'] = pd.to_datetime(
        temp_data['Datetime'],
        format=dt_format,
        errors='coerce',
    )
    temp_data.dropna(how='any', inplace=True)
    return temp_data


def read_timeseries(data_dir, resample_rate='d'):
    """
    Given path to data directory for a specific site,
    find time series and read them.

    :param str data_dir: Data directory
    :param str resample_rate: Resample time series (default d=day)
    :return:
    """

    site_name = os.path.split(os.path.dirname(data_dir))[-1]
    file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
    file_paths.sort()
    temps = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        year = int(file_name.split('_')[1])
        # Skip first row which only contains plot title
        temp_data = pd.read_csv(file_path, skiprows=1)
        temp_data = format_data(temp_data)
        # Check for change point at the beginning of time series
        temp_data = change_point_detection(temp_data, debug=False)
        temp_data = change_point_detection(temp_data, at_start=False, debug=False)
        # Resample to day intervals, get median temperature
        temp_data = temp_data.resample(resample_rate, on='Datetime').mean()
        temps.append(temp_data)

    # Concatenate all the temperature data files
    temp_data = pd.concat(
        temps,
        axis=0,
    )
    # Remove duplicate indices
    temp_data = temp_data[~temp_data.index.duplicated(keep='first')]
    # Fill missing dates with NaNs
    idx = pd.date_range(temp_data.index[0], temp_data.index[-1], freq=resample_rate)
    temp_data.index = pd.DatetimeIndex(temp_data.index)
    temp_data = temp_data.reindex(idx, fill_value=pd.NA)
    # # Plot temperatures in one line graph
    # fig = graphs.line_consecutive_years(temp_data, site_name)
    # fig.show()
    # # Plot monthly temperatures with years overlaid
    # fig = graphs.line_overlaid_years(temp_data, site_name)
    # fig.show()
    return temp_data, site_name


if __name__ == '__main__':
    args = parse_args()
    read_timeseries(args.dir, args.resample)

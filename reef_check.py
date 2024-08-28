import argparse
import glob
import os
import pandas as pd
import ruptures as rpt
import graphs as graphs


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
    return parser.parse_args()


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
    Given path to data directory, find time series and read them.

    :param str data_dir: Data directory
    :param str resample_rate: Resample time series (default day intervals)
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
    read_timeseries(args.dir)

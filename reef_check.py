import argparse
import datetime
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
        '--site',
        type=str,
        default=None,
        help='Site name. If None, collect temperatures and SST for all sites.',
    )
    parser.add_argument(
        '--debug',
        dest="debug",
        action="store_true",
        help='Debug status.',
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
            oisst_dir = _get_oisst_dir(year, month)
            result = subprocess.run(
                ['wget', '-nc', '-r', '-l', '1', '-P', write_dir, oisst_dir],
            )


def _get_oisst_dir(year, month, write_dir=None):
    """
    Return path to oisst data given year and month

    :param int year: Year
    :param int month: Month
    :param str/None write_dir: If files are stored locally within data_dir or to url
    :return str oisst_dir: Directory containing SST data for year and month
    """
    date = "{:4d}{:02d}".format(year, month)
    if write_dir is not None:
        oisst_dir = get_sst_path(write_dir)
        oisst_dir = os.path.join(oisst_dir, date)
    else:
        oisst_dir = os.path.join(OISST_URL, date)
    return oisst_dir


def read_oisst_file(file_path, mask_out=False):
    """
    Read CDF4 file containing one day of OISST data.
    OISST longitude coordinates are in degrees East, ranged [0, 360].
    Reef check longitude is in [-180, 180] interval.

    :param str file_path: Path to a daily OISST CDF4 file
    :param bool mask_out: Mask out land mass
    :return array sst: Sea surface temperature array (720, 1440)
    :return array sst_lat: Latitude coordinates (720,)
    :return array sst_lon: Longitude coordinates (1440,)
    """
    oisst = netCDF4.Dataset(file_path)
    sst_lat = oisst.variables['lat'][:].data
    sst_lon = oisst.variables['lon'][:].data

    # Sea surface temperature
    sst = np.squeeze(oisst.variables['sst'][:])
    sst_mask = sst.mask
    sst = sst.data
    if mask_out:
        # Mask out land masses
        sst[sst_mask] = np.nan

    return sst, (sst_lat, sst_lon)


def get_sst_path(write_dir):
    sst_path = os.path.join(
        write_dir,
        'oisst',
        OISST_URL.replace("https://", ""),
    )
    return sst_path


def _lon360to180(lon):
    """
    Convert longitudes from [-180,180] to [0,360]

    :param float lon: Longitude in [-180, 180] range
    :return float lon: Longitude in [0, 360] range
    """
    return ((lon - 180) % 360) - 180


def _lon180to360(lon):
    """
    Convert longitudes from [0, 360] to [-180, 180] interval

    :param float lon: Longitude in [0, 360] range
    :return float lon: Longitude in [-180, 180] range
    """
    return lon % 360


def match_coords(site_meta, sst, sst_coords):
    """
    Match coordinates of Reef Check site to coordinates of OISST data.

    :param pd.DataFrame site_meta: Reef Check metadata for site
    :param np.array sst: OISST data matrix
    :param list sst_coords: OISST nearest (lat, lon) coordinates for site
    #TODO: These paths could be separated
    :return int idx_lat: Index of OISST latitude with closest match to reef check
    :return int idx_lon: Index of OISST longitude with closest match to reef check
    """
    (sst_lat, sst_lon) = sst_coords
    # get lat, lon for site and find nearest point in the OISST data
    site_lat = site_meta['Site Lat'].item()
    # Convert longitude to match oisst system (which is in 360 East)
    site_lon = _lon180to360(site_meta['Site Long'].item())
    # Find the coordinates of the closest matching OISST data point
    # TODO: Upsample/interpolate OISST data for better match
    lon_mesh, lat_mesh = np.meshgrid(
    np.linspace(sst_lon[0], sst_lon[-1], sst_lon.shape[0]),
        np.linspace(sst_lat[0], sst_lat[-1], sst_lat.shape[0]),
    )
    # Compute distances from site coords to oisst coords
    dist = (lon_mesh - site_lon) ** 2 + (lat_mesh - site_lat) ** 2
    # Attribute an arbitraty high distance to land mass
    dist[sst != sst] = 1000000
    # Find nearest valid coordinates
    idx_lat, idx_lon = np.where(dist == dist.min())
    idx_lat = int(idx_lat[0])
    idx_lon = int(idx_lon[0])
    return idx_lat, idx_lon


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
    and format datetime column. Convert temperature to degrees Celcius
    if Fahrenheit.

    :param pd.DataFrame temp_data: Temperature data
    :param str dt_format: Datetime format
    :return pd.DataFrame temp_data: Formatted dataframe
    """
    def find_name(col_names, search_str):
        found_col = [col for col in col_names if search_str in col.lower()]
        assert len(found_col) == 1, (
            "Ambiguous {} column: {}".format(search_str, found_col))
        return found_col[0]

    col_names = temp_data.columns
    temp_col = find_name(col_names, 'temp')
    date_col = find_name(col_names, 'date')
    temp_data = temp_data[[date_col, temp_col]].copy()
    # Convert Fahrenheit to Celcius data
    if "Temp, °F" in temp_col:
        temp_data[temp_col] = (temp_data[temp_col] - 32) / 9 * 5
    temp_data.columns = ['Datetime', 'Temp']
    # Convert to datetime format, all data seem to be formatted the same way
    convert_to_datetime(temp_data, 'Datetime', dt_format)
    temp_data.dropna(how='any', inplace=True)
    return temp_data


def read_timeseries(site_dir, deploy_site=None, resample_rate='d', debug=False):
    """
    Given path to data directory for a specific site,
    find time series and read them.

    :param str site_dir: Data directory for specific site name
    :param pd.DataFrame/None deploy_site: Metadata with dates for deployed/retrieved
        for a specific site name. If None, do change point detection
    :param str resample_rate: Resample time series (default d=day)
    :param bool debug: Show debug plot if true
    :return pd.DataFrame temp_data: Temperature timeseries for site
    """
    temps = []
    if deploy_site is None:
        file_paths = glob.glob(os.path.join(site_dir, '*.csv'))
        file_paths.sort()
        for file_path in file_paths:
            temp_data = pd.read_csv(file_path, skiprows=1)
            temp_data = format_data(temp_data)
            # Check for change point at the beginning of time series
            temp_data = change_point_detection(temp_data, debug=False)
            temp_data = change_point_detection(temp_data, at_start=False, debug=False)
            # Resample to day intervals, get median temperature
            temp_data = temp_data.resample(resample_rate, on='Datetime').mean()
            temps.append(temp_data)
    else:
        # Loop through deployment metadata for site files
        for row in deploy_site.itertuples():
            if row.IsThereData == 'Y':
                search_str = row.Dep_ID + '*.csv'
                file_paths = glob.glob(os.path.join(site_dir, search_str))
                assert len(file_paths) == 1, "Ambiguous file paths for {}".format(row.Dep_ID)
                # Skip first row which only contains plot title
                temp_data = pd.read_csv(file_paths[0], skiprows=1)
                temp_data = format_data(temp_data)
                # Exclude times outside of deployment
                temp_data = temp_data[(temp_data.Datetime > row.Deployed) &
                                      (temp_data.Datetime <= row.Retrieved)]
                # Although using deployment data, some time series need cropping
                temp_data = change_point_detection(temp_data, nbr_days=7, debug=False)
                # Resample to day intervals, get median temperature
                temp_data = temp_data.resample(resample_rate, on='Datetime').mean()
                temps.append(temp_data)
    # Concatenate all the temperature dataframes
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
    return temp_data


def convert_to_datetime(dataframe, col_name, dt_format):
    dataframe[col_name] = pd.to_datetime(
        dataframe[col_name],
        format=dt_format,
        errors='coerce',
    )


def site_temperature(data_dir,
                     site_name,
                     deploy_meta,
                     reef_meta,
                     sst,
                     sst_coords,
                     resample_rate='d',
                     debug=False):
    """
    Read Reef Check temperature data for a given site name.
    Also reads OISST sea surface temperature data for the coordinates that are
    the closest to the Reef Check site coordinates not covered by land.

    :param str data_dir: Data directory
    :param str site_name: Reef Check site name
    :param pd.DataFrame deploy_meta: Reef Check deployment metadata
    :param pd.DataFrame reef_meta: Reef Check metadata for sites
    :param np.array sst: OISST sea surface temperature grid
    :param list sst_coords: OISST (lat, lon) coordinates closest to site
    :param str resample_rate: Resample rate for Reef Check data
    :param bool debug: If true, save a plot for site
    :return pd.DataFrame temp_data: Temperature data for site
    """
    print('Site name', site_name, '-----------------')
    deploy_site = deploy_meta[deploy_meta['Site_Name'] == site_name]
    # Sometimes deployment data hasn't been added
    if deploy_site.shape[0] == 0:
        return None
    # Get metadata for specific site
    site_meta = reef_meta[reef_meta['Site'] == site_name]
    region = site_meta['Region'].item()
    region = region.replace('CA', 'California')
    site_dir = os.path.join(data_dir, region, site_name)
    # Read all Reef check timeseries for site
    temp_data = read_timeseries(
        site_dir=site_dir,
        deploy_site=deploy_site,
        resample_rate=resample_rate,
    )
    # Get indices that match OISST coordinates best
    idx_lat, idx_lon = match_coords(site_meta, sst, sst_coords)
    # Add oisst coordinates to reef_meta
    sst_lat, sst_lon = sst_coords
    idx = reef_meta.index[reef_meta['Site'] == site_name]
    reef_meta.loc[idx, 'sst_lat'] = sst_lat[idx_lat]
    reef_meta.loc[idx, 'sst_lon'] = _lon360to180(sst_lon[idx_lon])
    # Read matching oisst data for site to compare sea surface temp with Reef Check
    temp_data['SST'] = pd.NA
    for site_date in temp_data.index:
        oisst_dir = _get_oisst_dir(
            site_date.year,
            site_date.month,
            write_dir=data_dir,
        )
        date_str = "{:4d}{:02d}{:02d}".format(
            site_date.year, site_date.month, site_date.day,
        )
        file_name = "oisst-avhrr-v02r01.{}.nc".format(date_str)
        file_path = os.path.join(oisst_dir, file_name)
        sst, _ = read_oisst_file(file_path)
        temp_data.loc[site_date, 'SST'] = sst[idx_lat, idx_lon]

    if debug:
        # Plot temperatures in one line graph
        fig = graphs.line_consecutive_years(temp_data, site_name)
        file_name = "ReefCheck_and_OISST_{}.png".format(site_name)
        file_path = os.path.join(data_dir, 'Graphs', file_name)
        fig.write_image(file_path, scale=5)

    return temp_data


def collect_temperature_data(data_dir,
                             site_name=None,
                             resample_rate='d',
                             debug=False):
    """
    Collect Reef Check temperature data and matching OISST sea surface
    temperature data for sites. Concatenate and save dataframe.

    :param str data_dir: Path to data directory (containing Reef Check and OISST)
    :param str/None site_name: Site name. None processed all sites
    :param str resample_rate: Resampling of data. Currently only 'd' works
    :param bool debug: Writes graph if true
    """

    # Read deployment metadata
    reef_path = os.path.join(data_dir, 'HOBO_deployment_metadata.csv')
    deploy_meta = pd.read_csv(reef_path)
    convert_to_datetime(
        deploy_meta,
        'Deployed',
        dt_format='%m/%d/%y %H:%M',
    )
    convert_to_datetime(
        deploy_meta,
        'Retrieved',
        dt_format='%m/%d/%y %H:%M',
    )
    # Read metadata file
    reef_path = os.path.join(
        data_dir,
        "Reef Check_Current Hobo Deployments_2022.csv",
    )
    reef_meta = pd.read_csv(reef_path)
    # Get one OISST file to match coordinates with
    sst_path = get_sst_path(data_dir)
    # Read one OISST file
    one_dir = glob.glob(os.path.join(sst_path, '[!index]*'))[0]
    one_file = glob.glob(os.path.join(one_dir, 'oisst-avhrr*'))[0]
    sst, sst_coords = read_oisst_file(one_file, mask_out=True)
    # Add oisst lat, lon to reef_meta
    reef_meta['sst_lat'] = pd.NA
    reef_meta['sst_lon'] = pd.NA
    # Make sure there's a directory for graphs if debug
    if debug:
        debug_dir = os.path.join(data_dir, 'Graphs')
        os.makedirs(debug_dir, exist_ok=True)
    # Get site names to analyze
    if isinstance(site_name, str):
        site_names = [site_name]
    elif site_name is None:
        site_names = list(reef_meta['Site'])
    # Loop through sites
    temperatures = []
    for site_name in site_names:
        temperature_data = site_temperature(
            data_dir,
            site_name,
            deploy_meta,
            reef_meta,
            sst,
            sst_coords,
            resample_rate=resample_rate,
            debug=True,
        )
        if temperature_data is not None:
            # Add site name since all data will be concatenated
            temperature_data['Site'] = site_name
            temperatures.append(temperature_data)
    # Concatenate all the temperature dataframes
    temperature_data = pd.concat(
        temperatures,
        axis=0,
    )
    temperature_data['Date'] = temperature_data.index
    temperature_data.to_csv(
        os.path.join(data_dir, "merged_reefcheck_oisst_data.csv"),
        index=False,
    )
    reef_meta.to_csv(
        os.path.join(data_dir, "reefcheck_and_oisst_coords.csv"),
        index=False,
    )


def read_data_and_coords(data_dir):
    """
    Check if csv file for merged data exists and reads if it does, creates if
    it doesn't. Also reads files containing coordinates.

    :param str data_dir: Path to data directory
    :return pd.DataFrame temp_data: Merged SOS data over all years
    :return pd.DataFrame col_config: Column info (name, sources, type,
        material, activity)
    """
    existing_file = glob.glob(os.path.join(data_dir, 'merged_reefcheck_oisst_data.csv'))
    if len(existing_file) == 0:
        collect_temperature_data(data_dir)
    # Make sure Date column is datetime
    temp_data = pd.read_csv(existing_file[0])
    temp_data['Date'] = pd.to_datetime(temp_data['Date'], errors='coerce')
    temp_data = temp_data.set_index('Date')
    # Read coordinates for all sites
    reef_path = os.path.join(data_dir, "reefcheck_and_oisst_coords.csv")
    reef_meta = pd.read_csv(reef_path)
    reef_meta = reef_meta.dropna(subset=['sst_lat'])

    return temp_data, reef_meta


if __name__ == '__main__':
    args = parse_args()
    collect_temperature_data(args.dir, args.site, args.resample, args.debug)

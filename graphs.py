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


def line_consecutive_years(temp_data):
    """
    Plot line graph with years/date on the x axis and temperature on y axis

    :param pd.DataFrame temp_data: Temperature data
    :return px.fig: Figure
    """
    fig = px.line(temp_data, x=temp_data.index, y=temp_data.Temp)
    fig.update_layout(
        title="Temperature ",
        yaxis_title='Temperature (degrees C)',
        xaxis_title='Date',
    )
    return fig


def line_overlaid_years(temp_data):
    """
    Plot data by month with years overlaid on top of each other on x axis,
    temperature on y axis.

    :param pd.DataFrame temp_data: Temperature data
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
        xaxis_tickformat="%B",
        title="Temperature ",
        yaxis_title='Temperature (degrees C)',
        xaxis_title='Month',
    )
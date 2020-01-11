import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


df = pd.read_csv('results/Forecasts/forecasts_2019-11-25.csv')

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    # Header H1
    html.H1(children='Nickel Price Forecasting Project'),
    html.Div(children='''by Chiemi Kato\n Tesla Battery Supply Chain'''),
    

    ])


dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),

)

########


if __name__ == '__main__':
    app.run_server(debug=True)
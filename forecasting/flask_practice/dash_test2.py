import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


df = pd.read_csv('results/Forecasts/forecasts_2019-11-25.csv')
df= df.iloc[1:, :]
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    # Header H1
    html.H1(children='Nickel Price Forecasting Project'),
    html.H2(children='''by Chiemi Kato'''),
    html.Div(children='''Tesla Battery Supply Chain'''),
    
    dcc.Graph(id='example-graph',
        figure={'data': [{'x':df['Dates'], 'y': df['Actual Price'] }],
        'layout': {'title': 'Nickel Price Forecasting'}
        }
        )
    ])

# TRY OUT MORE LAYOUTS HERE: https://www.youtube.com/watch?v=yPSbJSblrvw&t=311s

########


if __name__ == '__main__':
    app.run_server(debug=True)
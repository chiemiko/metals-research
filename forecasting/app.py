from flask import Flask, render_template, request #this has changed
import pandas as pd


import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json

# created object of flask 
app = Flask(__name__)

# Created a route which was for the homepage (using /)

@app.route('/')


# Method that has to return something OUTPUTTED TO THE BROWSER
def index():
	feature = 'Bar'
	df = pd.read_csv('results/Forecasts/forecasts_2019-11-25.csv')
	df= df.iloc[1:, :]
	bar = create_plot(feature)
	return render_template('index.html', plot=bar)

def create_plot(feature):
    if feature == 'Bar':
        N = 40
        x = np.linspace(0, 1, N)
        y = np.random.randn(N)
        df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
        data = [
            go.Bar(
                x=df['x'], # assign x as the dataframe column 'x'
                y=df['y']
            )
        ]
    else:
        N = 1000
        random_x = np.random.randn(N)
        random_y = np.random.randn(N)

        # Create a trace
        data = [go.Scatter(
            x = random_x,
            y = random_y,
            mode = 'markers'
        )]


    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


@app.route('/bar', methods=['GET', 'POST'])


def change_features():

    feature = request.args['selected']
    graphJSON= create_plot(feature)




    return graphJSON



if __name__ == '__main__':
	app.run(port=5000)
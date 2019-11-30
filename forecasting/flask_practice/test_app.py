from __future__ import unicode_literals
import json
import requests
import pandas as pd

from flask import Flask, request, Response, render_template, session, redirect, url_for

# https://github.com/satssehgal/scrapyAPI

app = Flask(__name__)

@app.route('/')
def scrape():
    df = pd.read_csv('../results/Forecasts/forecasts_2019-11-25.csv')
    df= df.iloc[1:, :]
    return render_template('simple.html',  tables=[df.to_html(classes='data',  index=False)], titles=df.columns.values)


if __name__ == '__main__':
    app.run(debug=True, port=1234)
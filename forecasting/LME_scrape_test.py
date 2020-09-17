from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from datetime import datetime

from requests import get
from requests.exceptions import RequestException
from contextlib import closing

from sqlalchemy import create_engine, MetaData, Table, Table, Column, String, Integer, Boolean

time_of_search = datetime.now().date()

import datetime

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    beautifulsoup object.
    """
    try:
        #with closing(get(url, stream=True)) as resp:
        chrome_options = webdriver.ChromeOptions()
        driver = webdriver.Chrome("chromedriver")
        driver.implicitly_wait(20)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        if type(driver) == webdriver.chrome.webdriver.WebDriver:
            return soup, driver
        else:
            return None

    except Exception:
        # MAIN ERROR?
        #log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        print('Error retrieving contents at {}'.format(url))
        return None

def run_scrape():
    url = f'https://www.lme.com/en-GB/Metals/Non-ferrous/Nickel#tabIndex=0'
    soup, driver = simple_get(url)

    soup = BeautifulSoup(driver.page_source, 'lxml')

    #h2 = soup.find('div', id="LME NICKEL OFFICIAL PRICES, US$ PER TONNE").h2

    html = soup.find_all('div', attrs={'id': 'module-63'})
    text_list = html[0].getText().strip().split()


    for i,item in enumerate(text_list):
        if item.lower() == 'cash':
            nickel_price = text_list[i+1]
    nickel_price2 = pd.DataFrame(pd.Series(nickel_price))
    nickel_price2.index = [time_of_search]
    nickel_price2.columns = ['0']
    nickel_price2

    return nickel_price2

if __name__ == '__main__':
    nickel_price = run_scrape()

    engine = create_engine('sqlite:///LME_nickel.sqlite')
    connection = engine.connect()

    df = pd.read_sql_query('SELECT * from daily_nickel_prices;', connection)
    df.index = pd.to_datetime(df['index'])
    df = df.iloc[:,1:]
    df = df.append(nickel_price)
    print(df)
    df.to_sql("daily_nickel_prices", con=engine, if_exists='replace', index = True)


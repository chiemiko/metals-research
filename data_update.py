
from sqlalchemy import create_engine
from sqlalchemy import(Table, Column, String, Integer, Boolean)
from sqlalchemy import MetaData, Table

import numpy as np

import datetime

import pandas as pd


# ## Step 1: SQL Connection to Database


# Connecting to Engine
engine = create_engine('sqlite:///metals_dashboard.sqlite')

connection = engine.connect()

def update_commodities():
    filename1 = 'raw_data_finals\MetalsDashboard(Aug)LATEST.xlsx'
    commodities = pd.read_excel(filename1, sheet_name="Commodities Data")


    # read census.csv into a dataframe to create pandas dataframe

    commodities.columns = commodities.iloc[0]

    #############
    commodities = commodities.iloc[6: ,:]
    commodities = commodities.iloc[:, 1:]
    commodities = commodities.rename(columns={"Column1": "Date"})

    commodities['Date'] = pd.to_datetime(commodities['Date'])
    commodities = commodities.set_index('Date')

    commodities = commodities.drop("Column2", axis=1)
    commodities['Fastmarkets - Cobalt High'] = commodities['Fastmarkets - Cobalt High'].replace('""', np.nan)
    commodities = commodities.astype(float)



    commodities.to_sql("commodities", con=engine, if_exists='replace', index = True)


def update_fastmarkets():
    # #### Reading and Cleaning Fastmarkets Data
    # 
    # Notes: 
    # Needed to add dates as indices and take out text columns of 'day of week' and 'actual vs forecast'

    # Fastmarkets table
    filename3 = 'raw_data_finals/CobaltManganese/Metals and FX Rate Master - Latest.xlsx'
    fastmarkets = pd.read_excel(filename3, sheet_name="LME - Daily")

    fastmarkets = fastmarkets.iloc[:, 1:len(fastmarkets.columns)]
    #############

    fastmarkets.columns = fastmarkets.iloc[0, :]
    fastmarkets = fastmarkets.iloc[1:, :]
    fastmarkets = fastmarkets.iloc[:, 0:len(fastmarkets.columns)-4]


    # Creating dummy dates column 

    fastmarkets['Date'] = pd.to_datetime('2000-01-01')

    for i in fastmarkets.index:
        fastmarkets.iloc[i-1, 14]= fastmarkets.iloc[i-1, 14]+ datetime.timedelta(days=i-1)

    fastmarkets.index = fastmarkets['Date']
    fastmarkets = fastmarkets.iloc[:, :len(fastmarkets.columns)-1]


    fastmarkets = fastmarkets.drop(columns=['Actual or Forecast', 'Day of the week' ])
    fastmarkets = fastmarkets.astype(float)

    # Conversion of date columns into datetime data types
    # fastmarkets['Date'] = pd.to_datetime(fastmarkets[['Year', 'Month']])

    fastmarkets.to_sql("fastmarkets", con=engine, if_exists='replace', index = True)



def update_battery_cap():
    # #### Reading and Cleaning Battery Capacity Data

    # Battery Capacity table
    #filename2 = 'raw_data_finals/BatteryCapacity/201908 Benchmark Minerals Megafactory data - August 2019.xlsx'
    #battery_cap = pd.read_excel(filename2, sheet_name="Sheet1")
    
    filename4 = 'raw_data_finals\metals_data_main.xlsx'
    battery_cap = pd.read_excel(filename4, sheet_name='battery_cap')


    battery_cap = battery_cap.iloc[:, 1:len(battery_cap.columns)]
    battery_cap =  battery_cap.iloc[6: ,:]
    battery_cap.columns = battery_cap.iloc[0,:]

    battery_cap = battery_cap.reset_index()
    battery_cap= battery_cap.iloc[1:,1:]

    # Creating dummy dates column 

    battery_cap['Date'] = pd.to_datetime('2000-01-01')

    for i in battery_cap.index:
        battery_cap.iloc[i-1, 5]= battery_cap.iloc[i-1, 5]+ datetime.timedelta(days=i-1)    

    battery_cap.index = battery_cap['Date']
    battery_cap = battery_cap.iloc[:, 0:len(battery_cap.columns)-1]

    battery_cap['Megafactory1'] = battery_cap["Megafactory"].str.split(",", n = 1, expand = True)[0]
    battery_cap['Location'] = battery_cap["Megafactory"].str.split(",", n = 1, expand = True)[1]
    battery_cap = battery_cap.drop("Megafactory", axis=1)

    battery_cap = battery_cap[['Megafactory1', 'Location', 'Region', 'GWh 2018', 'GWh 2023', 'GWh 2028']]
    battery_cap.rename(columns={"Megafactory1": "Megafactory"})


    battery_cap.to_sql("battery_cap", con=engine, if_exists='replace', index = True)


def update_usage():
    #### Reading and Cleaning Usage Rates 

    usage = pd.read_excel("raw_data_finals/usage.xlsx")

    usage.to_sql("usage", con=engine, if_exists='replace', index = True)


def update_EVSales():
        
    # #### Reading and Cleaning EV Sales (NOTE: The only way to "download" data file is to copy/paste data into excel spreadsheet from https://insideevs.com/news/368729/ev-sales-scorecard-august-2019/)
    # 
    # ## Notes About Structure
    # The following code merges two files. One is the excel file that is copy pasted, as instructed above. The second file **titled EV_kwh_references.xlsx** is a reference file containing 'look-up' information about each car model's KwH amount. 
    # 
    # ### In the case that new makes/companies are added to the data, the reference file needs to be updated with the model/make of the car and the kwh amount.


    model_lookup_kwh = 'raw_data_finals\EVSales\EVSalesKwHReferences.xlsx'
    model_lookup_data = pd.read_excel(model_lookup_kwh)

    for index, model in model_lookup_data['Model'].items():
        model_lookup_data.iloc[index, 0] = model.lower()

    kwh_dictionary = dict(zip(model_lookup_data['Model'], model_lookup_data['kwh'])) 


    #filename4 = 'raw_data_finals\EVSales\EVSales.xlsx'
    filename4 = 'raw_data_finals\metals_data_main.xlsx'
    EVSales = pd.read_excel(filename4, sheet_name='EVSales')


    EVSales.columns = EVSales.iloc[0]
    EVSales = EVSales.loc[:, EVSales.columns.notnull()]

    # Creating dummy dates column 
    for index, model in EVSales['2019 U.S. EV SALES'].items():
        EVSales.loc[index, '2019 U.S. EV SALES'] = model.lower()



    #############
    EVSales = EVSales.iloc[1:len(EVSales)-4, :len(EVSales.columns)]



    EVSales['kwh'] = EVSales['2019 U.S. EV SALES'].map(kwh_dictionary)

    Sales = EVSales.iloc[:, 1:]
    Sales = Sales.astype(float)
    EVSales = pd.concat([EVSales.iloc[:, 0], Sales], axis=1, sort=False)


    # ### Additional Data Transformations to Convert Sales to kwh units per make/model


    EVSales['JAN kwh'] = EVSales['JAN']*EVSales['kwh']
    EVSales['FEB kwh'] = EVSales['FEB']*EVSales['kwh']
    EVSales['MAR kwh'] = EVSales['MAR']*EVSales['kwh']
    EVSales['APR kwh'] = EVSales['APR']*EVSales['kwh']
    EVSales['MAY kwh'] = EVSales['MAY']*EVSales['kwh']
    EVSales['JUN kwh'] = EVSales['JUN']*EVSales['kwh']
    EVSales['JUL kwh'] = EVSales['JUL']*EVSales['kwh']
    EVSales['AUG kwh'] = EVSales['AUG']*EVSales['kwh']
    EVSales['SEP kwh'] = EVSales['SEP']*EVSales['kwh']
    EVSales['OCT kwh'] = EVSales['OCT']*EVSales['kwh']
    EVSales['NOV kwh'] = EVSales['NOV']*EVSales['kwh']
    EVSales['DEC kwh'] = EVSales['DEC']*EVSales['kwh']


    EVSales['Company'] = EVSales.loc[:, '2019 U.S. EV SALES'].str.split(' ', n=1, expand=True)[0]
    for key, item in EVSales['Company'].items():
        if item == 'bmwx5':
            EVSales['Company'][key] = 'bmw'


    EVSales = EVSales.groupby(EVSales['Company']).sum()


    EVSales = EVSales.reset_index()

    # Creating Dummy Dates for indices
    EVSales['Date'] = pd.to_datetime('2000-01-01')

    for i in EVSales.index:
        EVSales.iloc[i-1, len(EVSales.columns)-1]= EVSales.iloc[i-1, len(EVSales.columns)-1]+ datetime.timedelta(days=i-1)
        
    EVSales.index = EVSales['Date']
    EVSales = EVSales.iloc[:, 0:len(EVSales.columns)-1]

    EVSales.to_sql("EVSales", con=engine, if_exists='replace', index = True)



##### ADDING/CALCULATING BASELINES MATRICES PER COMMODITY
    
def update_baselines():
    df = pd.read_sql_query('SELECT * from commodities;', connection)
    df['month_year'] = pd.to_datetime(df['Date']).dt.to_period('M')

    baselines = df.groupby(['month_year']).mean()
    usage = pd.read_sql_query('SELECT * from usage;', connection)

    # #### Adding Lithium to Baselines Data for Further for Calculations of Usage

    LiOH = pd.read_excel('raw_data_finals/LiOH/03062019 LiOH Summary V1.xlsx', sheet_name="LiOH Supply")
    LiOH.index = LiOH['Volumes (MT)']
    LiOH = LiOH.iloc[:, 1:]
    LiOH.columns = LiOH.iloc[2, :]

    LiOH_col = LiOH.loc['Weighted average per kg price - all sources', :]
    LiOH_col = LiOH_col[4:]
    LiOH_col = LiOH_col*1000

    LiOH_col.index = pd.to_datetime(LiOH_col.index)
    LiOH_col.index = LiOH_col.index.to_period('M')


    baselines = pd.concat([baselines, LiOH_col], axis=1, sort=False)


    baselines = baselines.dropna(subset=['LME Ni cash price'])
    baselines = baselines.rename(columns={"Weighted average per kg price - all sources": "Lithium"})


    # Write baselines to sql datebase 
    '''
    baselines.to_csv('raw_data_finals/baselines.csv')

    baselines = pd.read_csv('raw_data_finals/baselines.csv')
    baselines['Unnamed: 0'] = pd.to_datetime(baselines['Unnamed: 0'])
    baselines.index = baselines['Unnamed: 0']
    baselines = baselines.iloc[:, 1:len(baselines.columns)]
    baselines.to_sql("baselines", con=engine, if_exists='replace', index = True)
    '''

    for key, model in usage['Model'].items():
        usage.iloc[key, 1]= model.replace('-', '')
    for key, model in usage['Model'].items():
        #print(model.replace('-', ''))
        vars()[str(model)] = baselines[['LME Ni cash price', 'LME Co cash price', 'LME Cu cash price', 'LME Al cash price', 'Lithium']]
        #print(vars()[str(model)])


    for key, model in usage['Model'].items():
        '''Split models by cells and vehicles'''
        # Cells usage calculations
        if model[0:4]=='cell':
            for metal in usage.columns[3:7]:
            #print('usage ', usage.loc[key, metal])

                if metal != 'Copper':
                    symbol = metal[:2]
                    col_name = 'LME ' + symbol + ' cash price'
                    new_col_name = symbol + ' usage'
                    '''Populate columns in dataframe/Calculation of new usage amounts'''
                    #print(str(model)[col_name]*usage.loc[key, metal])

                    vars()[str(model)].loc[:,new_col_name] = (vars()[str(model)].loc[:,col_name]*usage.loc[key, metal])/(usage.loc[key, 'cell_energy'])

                else:
                    symbol = 'Cu'
                    col_name = 'LME ' + symbol + ' cash price'
                    new_col_name = symbol + ' usage'
                    '''Populate columns in dataframe'''
                    #print(str(model)[col_name]*usage.loc[key, metal])
                    vars()[str(model)].loc[:,new_col_name] = (vars()[str(model)].loc[:,col_name]*usage.loc[key, metal])/(usage.loc[key, 'cell_energy'])

            # Separate calculation for lithium 
            col_name = 'Lithium'
            new_col_name = 'Li Usage'
            vars()[str(model)].loc[:,new_col_name] = (vars()[str(model)].loc[:,col_name]*usage.loc[key,'LiOH' ])/(usage.loc[key, 'cell_energy'])
            

        # Vehicles usage calculations
        else:
        
            for metal in usage.columns[3:7]:
                #print('usage ', usage.loc[key, metal])

                if metal != 'Copper':
                    symbol = metal[:2]
                    col_name = 'LME ' + symbol + ' cash price'
                    new_col_name = symbol + ' usage'
                    '''Populate columns in dataframe/Calculation of new usage amounts'''
                    #print(str(model)[col_name]*usage.loc[key, metal])

                    vars()[str(model)].loc[:,new_col_name] = (vars()[str(model)].loc[:,col_name]*usage.loc[key, metal])/(1000*usage.loc[key, 'cell_energy'])

                else:
                    symbol = 'Cu'
                    col_name = 'LME ' + symbol + ' cash price'
                    new_col_name = symbol + ' usage'
                    '''Populate columns in dataframe'''
                    #print(str(model)[col_name]*usage.loc[key, metal])
                    vars()[str(model)].loc[:,new_col_name] = (vars()[str(model)].loc[:,col_name]*usage.loc[key, metal])/(1000*usage.loc[key, 'cell_energy'])

            # Separate calculation for lithium 
            col_name = 'Lithium' 
            new_col_name = 'Li Usage'
            vars()[str(model)].loc[:,new_col_name] = (vars()[str(model)].loc[:,col_name]*usage.loc[key,'LiOH'])/(1000*usage.loc[key, 'cell_energy'])

        model_df_name = str(model) + '_df'
        csv_filename = 'raw_data_finals/usage_per_model/' + str(model) + '.csv'
        vars()[str(model)].to_csv(csv_filename)
        vars()[model_df_name] = pd.read_csv(csv_filename)
        vars()[model_df_name]['Unnamed: 0'] = pd.to_datetime(vars()[model_df_name]['Unnamed: 0'])
        vars()[model_df_name].index = vars()[model_df_name]['Unnamed: 0']
        vars()[model_df_name] = vars()[model_df_name].iloc[:, 1:len(vars()[model_df_name].columns)]
        vars()[model_df_name].to_sql(str(model), con=engine, if_exists='replace', index = True)


    baselines = baselines[['Lithium','LME Ni cash price', 'LME Co cash price', 'LME Cu cash price', 'LME Al cash price']]

    baselines = baselines.dropna(subset=['Lithium'])


    # #### !! Reset base if need be by changing date in base variable below

    base = baselines.loc['2018-01-01', :]


    for col in baselines.columns:
        baselines.loc[:, col] = (baselines.loc[:, col]*100)/base[col]

    baselines.to_csv('raw_data_finals/baselines.csv')
    baselines = pd.read_csv('raw_data_finals/baselines.csv')
    baselines.loc[:,'Unnamed: 0'] = pd.to_datetime(baselines.loc[:,'Unnamed: 0'])
    baselines.index = baselines['Unnamed: 0']
    baselines = baselines.iloc[:, 1:len(baselines.columns)]
    baselines.to_sql("baselines", con=engine, if_exists='replace', index = True)


def update_lithium():
    '''Updates Lithium prices from two data files for Lithium line plot'''
    baselines = pd.read_sql_query('SELECT * from baselines;', connection)

    Li_Benchmarks = pd.read_excel("raw_data_finals/LiOH/201907 Benchmark Lithium Prices - Jul 19.xlsx", sheet_name="Hydroxide")
    #Li_Fastmarkets = pd.read_excel("raw_data_finals/LiOH/20190729 Lithium Spot Prices.xlsx", sheet_name="Data")
    filename4 = 'raw_data_finals\metals_data_main.xlsx'
    Li_Fastmarkets = pd.read_excel(filename4, sheet_name='LiOH_Fastmarkets')

    Li_Benchmarks.index = Li_Benchmarks.iloc[:,0]
    Li_Benchmarks = Li_Benchmarks.iloc[:, 1:]
    Li_Benchmarks_col = Li_Benchmarks.iloc[:, 1]
    Li_Benchmarks_col.index = pd.to_datetime(Li_Benchmarks_col.index)

    Li_Fastmarkets = Li_Fastmarkets.dropna(subset=[ 'Lithium hydroxide monohydrate min 56.5% LiOH2O battery grade, spot prices CIF China, Japan & Korea, $/kg\r\nHigh (USD)'])
    Li_Fastmarkets.index = pd.to_datetime(Li_Fastmarkets['Unnamed: 0']).dt.to_period('M')
    Li_Fastmarkets_col = Li_Fastmarkets.loc[:, 'Lithium hydroxide monohydrate min 56.5% LiOH2O battery grade, spot prices CIF China, Japan & Korea, $/kg\r\nHigh (USD)']
    Li_Fastmarkets_col = Li_Fastmarkets_col.groupby(Li_Fastmarkets_col.index).mean()


    datelist = []
    for date in Li_Fastmarkets_col.index:
        date_add =  datetime.datetime(date.year, date.month, 1)
        datelist.append(date_add)


    Li_Fastmarkets_col.index = datelist




    Lithium = pd.concat([Li_Fastmarkets_col, Li_Benchmarks_col], axis=1, sort=False)

    #Lithium = Lithium.dropna(subset=[ 'Lithium hydroxide monohydrate min 56.5% LiOH2O battery grade, spot prices CIF China, Japan & Korea, $/kg\r\nHigh (USD)'])
    Lithium = Lithium.loc['2017-08-01':, :] 


    Lithium.to_csv('raw_data_finals/LiOH/lithium_benchmark_fastmarkets.csv')

    Lithium = pd.read_csv('raw_data_finals/LiOH/lithium_benchmark_fastmarkets.csv')
    Lithium['Unnamed: 0'] = pd.to_datetime(Lithium['Unnamed: 0'])
    Lithium.index = Lithium['Unnamed: 0']
    Lithium = Lithium.iloc[:, 1:len(Lithium.columns)]
    Lithium = Lithium.astype(float) 

    Lithium.to_sql("lithium_benchmark_fastmarkets", con=engine, if_exists='replace', index = True)





update_commodities()
update_fastmarkets()
update_battery_cap()
update_usage()
update_EVSales()

update_baselines()

update_lithium()

print()
print('Tables in the database are: ', engine.table_names())
print()

connection.close()



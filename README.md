# metals-research

### Fall/Winter 2019 
#### Goal statement: 
As the demand for lithium-ion batteries continues to grow, changes to global availability of the raw materials needed to make these batteries will inevitably occur. 
This project will focus on developing a predictive modeling tool that uses historic London Metals Exchange commodity price data to make forecasts for one and two years ahead of time. 

#### Results: 
The K Nearest Neighbors (supervised) algorithm ended up being the highest performing machine learning model in the univariate price data scenario. After transforming the data into stationary data with multiple lag features, I used MAE to evaluate results (most recent 5 years had mean absolute errors of $1700 and $2000). To interpret prediction results at original dollar/tonne scale, I wrote a stationarity reversal function.

#### Tools:
Python (Scikit-learn, Statsmodels, Pandas, Selenium, SQLAlchemy), Power BI

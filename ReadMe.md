# DSCI521: Final Project - WoRLD of World!

### Introduction:

Gross domestic product (GDP) is one of the most common indicators used to track the health of a nation's economy. It includes a number of different factors such as consumption and investment and it is also considered as the monetary value of all the finished goods and services produced within a country's borders in a specific time period and includes anything produced by the country's citizens and foreigners within its borders.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Project goal

In this project we are looking towards the World revenue longitudinal data and trying to find the best corelated variables which can be used to train our models and give the predicted value of GDP. Also under the analysis we are finding the corelation between GDP and GNI. As most of the attributes are in terms of GDP or in % of GDP so we are trying to predict and forecast GNI from model trained with those attributes.

Used Time Series calculation to find the forecasted value of GNI till 2040 with the attributes in terms of GDP with 95% confidence level.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Prerequisites

Following applications are required to run this project: 

Google chrome:

It is recommended to use chrome as a browser:
https://www.google.com/chrome/


Anaconda distribution to run Python Jupyter:

Download Anaconda from the link below and follow the instructions. 
https://www.anaconda.com/distribution

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Running the Project File

To run the project the file 'WoRLD of World.ipnyb' needs to be run from Jupyter notebook launched through Anaconda. 

1. Download the notebook 'WoRLD of World.ipnyb' to your system.

2. Run the notebook cell wise from the top. Instructions are provided for each cell in the notebook. The run can also be triggered for all cells (Go to 'cell'-> 'Run all'. Output corresponsding to each cell will be displayed below. 

3. To check the status of cell execution the 'status' bar on the bottom of the screen reflects the python kernel status as 'idle' or 'busy'. The notebook doesn't support concurrent cell execution.

4. When the acquisition block run finishes, a prediction of GNI will be calculated and shown with comparison to actual values. Also the array of forecasted value of GNI would be produced till 2040.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Project CONTENTS:

1.1 Introduction
1.2 Acknowledgements
1.3 Description
    1.3.1 Data Acquisition, Pre-processing and Cleaning
       1.3.1.1 Importing Modules
       1.3.1.2 Defining Funtions 
           1.3.1.2.1 Read_Data
           1.3.1.2.2 Country_Specific_Dataframe
           1.3.1.2.3 Read_country
           1.3.1.2.4 datetime_conv
       1.3.1.3 Working with Dataset
           1.3.1.3.1 Reading dataset
           1.3.1.3.2 Changing it into required format to be required in project
           1.3.1.3.3 Checking data for country list
           1.3.1.3.4 Prepare a Non Country List

2.0 Data modification and Final dataframe
    2.1 Reading data of 195 countries prepared above after removing non country list.
    2.2 Finding corelation between attributes
    2.3 Merging Attributes based on our Financial learning and IMF information
    2.4 Writing it to csv WoRLD_Data.csv to be used further in project.

3.0 Exploratory Data Analysis
    3.1 Visualization of attributes with top 10 countries for
    3.2 Scatter plots showing corelation between attributes selected
    
4.0 Modelling
    4.1 Single Linear Regression Model
    4.2 Multivariate Linear Regression
    4.3 Multivariate Linear Regression with LabelEncoder
    4.4 Random Forest
    4.5 Decision Tree Model
    4.6 KNeighbour Regression

5.0 GNI Prediction using TS
    5.1 Preparing TS dataframe
    5.2 Check for Stationarity
    5.3 Removing Stationarity, Seasonality and Trend
    5.4 Forecasting Values (Modelling)
        5.4.1 AR(Auto-Regressive Model)
	5.4.2 MA(Moving-Average Model)
	5.4.3 ARIMA(Auto-Regressive Integrated Moving Average Model)
	5.4.4 Rescaling
    5.5 Forecasting values

6.0 WoRLD of World.ipynb

7.0 CSV Files
    7.1 Raw_Data.csv 
    7.2 Intermed_Data.csv
    7.3 All_Data.csv
    7.4 All_New_Data.csv
    7.5 WoRLD_Data.csv

9.0 Conclusion
10.0 Limitation and Future Scope

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 1.1 Introduction:

World Bank databases are essential tools for supporting critical management decisions and providing key statistical information for Bank operational activities. The application of internationally accepted standards and norms results in a consistent, reliable source of information.

The International Monetary Fund (IMF) is an international organization headquartered in Washington, D.C., consisting of 189 countries working to foster global monetary cooperation, secure financial stability, facilitate international trade, promote high employment and sustainable economic growth, and reduce poverty around the world while periodically depending on the World Bank for its resources.

Attributes fetched in Raw data is as below: 

[
'Adjusted net national income (annual % growth)',

 'Adjusted net national income (current US$)',
 
 'Agricultural raw materials imports (% of merchandise imports)',
 
 'Agricultural raw materials exports (% of merchandise exports)',
 
 'Agriculture, forestry, and fishing, value added (% of GDP)',
 
 'Arms exports (SIPRI trend indicator values)',
 
 'Arms imports (SIPRI trend indicator values)',
 
 'Bank capital to assets ratio (%)',
 
 'Bank liquid reserves to bank assets ratio (%)',
 
 'Broad money (% of GDP)',
 
 'Central government debt, total (% of GDP)',
 
 'Claims on central government, etc. (% GDP)',
 
 'Claims on other sectors of the domestic economy (% of GDP)',
 
 'Coal rents (% of GDP)',
 
 'Commercial service exports (current US$)',
 
 'Commercial service imports (current US$)',
 
 'Current account balance (% of GDP)',
 
 'Current health expenditure (% of GDP)',
 
 'Domestic credit provided by financial sector (% of GDP)',
 
 'Domestic credit to private sector (% of GDP)',
 
 'Domestic credit to private sector by banks (% of GDP)',
 
 'Domestic general government health expenditure (% of GDP)',
 
 'Expense (% of GDP)',
 
 'Exports of goods and services (% of GDP)',
 
 'External balance on goods and services (% of GDP)',
 
 'Final consumption expenditure (% of GDP)',
 
 'Foreign direct investment, net inflows (% of GDP)',
 
 'Forest rents (% of GDP)',
 
 'General government final consumption expenditure (% of GDP)',
 
 'Government expenditure on education, total (% of GDP)',
 
 'Gross capital formation (% of GDP)',
 
 'Gross domestic savings (% of GDP)',
 
 'Gross fixed capital formation (% of GDP)',
 
 'Gross national expenditure (% of GDP)',
 
 'Gross savings (% of GDP)',
 
 'Imports of goods and services (% of GDP)',
 
 'Industry (including construction), value added (% of GDP)',
 
 'Inflation, GDP deflator (annual %)',
 
 'International tourism, receipts (% of total exports)',
 
 'Manufacturing, value added (% of GDP)',
 
 'Market capitalization of listed domestic companies (% of GDP)',
 
 'Merchandise trade (% of GDP)',
 
 'Military expenditure (% of GDP)',
 
 'Mineral rents (% of GDP)',
 
 'Natural gas rents (% of GDP)',
 
 'Net acquisition of financial assets (% of GDP)',
 
 'Net incurrence of liabilities, total (% of GDP)',
 
 'Net investment in nonfinancial assets (% of GDP)',
 
 'Net lending (+) / net borrowing (-) (% of GDP)',
 
 'Oil rents (% of GDP)',
 
 'Personal remittances, received (% of GDP)',
 
 'Revenue, excluding grants (% of GDP)',
 
 'Stocks traded, total value (% of GDP)',
 
 'Tax revenue (% of GDP)',
 
 'Total natural resources rents (% of GDP)',
 
 'Trade (% of GDP)',
 
 'Trade in services (% of GDP)']


#### Detailed project flow is scripted below with major steps.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##### 1.2 Acknowledgements:

Data is collected from World Bank databank after selecting the attributes learned from IMF website. We would have selected the data from IMF Databank as well but the attributes are minimal so we took it from World Bank. Websites to refer are as below:
 IMF : https://data.imf.org/?sk=77413F1D-1525-450A-A23A-47AEED40FE78
 World Bank : https://databank.worldbank.org/source/world-development-indicators

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##### 1.3 Description:

Project flow is divided into five major steps.

1.3.1 Data Acquisition, Pre-processing and Cleaning:
    1.3.1.1 Importing Modules
        import pandas as pd  
        import numpy as np  
        import csv
        import seaborn as sns
        import matplotlib.pyplot as plt  
        %matplotlib inline
        import seaborn as seabornInstance 
        from sklearn.model_selection import train_test_split 
        from sklearn.linear_model import LinearRegression
        from sklearn import metrics
        from sklearn.preprocessing import LabelEncoder
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_squared_log_error
        from sklearn import ensemble
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        from sklearn.neighbors import KNeighborsRegressor
        from datetime import datetime
        from pandas import read_csv
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.arima_model import ARIMA
        from statsmodels.tsa.stattools import acf
        from statsmodels.tsa.stattools import pacf
        from statsmodels.tsa.arima_model import ARIMA
        import statsmodels.api as sm
        from pandas import read_csv
        from statsmodels.tsa.stattools import adfuller
        from numpy import log
        
    1.3.1.2 Defining Funtions: 
        1.3.1.2.1 Read_Data : Function Read_data is defined to read csv and split the dataframe based on given columns list
        1.3.1.2.2 Country_Specific_Dataframe: Function Country_Specific_Dataframe is defined to fill NaN values with the mean data of the values from year 1990-2018 for specific attribute for a particular country.
        1.3.1.2.3 Read_country : Function Read_country is defined to read the dataframe for specific countries and return only the needed columns.
        1.3.1.2.4 datetime_conv : Function datetime_conv is defined to convert the string data present in Year column of dataframe to datetime type so that it can be used for Time Series analysis.

     1.3.1.3 Working with Dataset:

         1.3.1.3.1 Reading dataset
         1.3.1.3.2 Changing it into required format to be required in project
         1.3.1.3.3 Checking data for country list, to be sure that we have exactly 195 countries data.
         1.3.1.3.4 Prepare a Non Country List : While analysing the data its found that the Data contains some non-country data as well. So we created a list of non-country data to remove it from our Final Dataset.
            ARB Arab World
            CSS Caribbean small states
            CEB Central Europe and the Baltics
            EAR Early-demographic dividend
            EAS East Asia & Pacific
            EAP East Asia & Pacific (excluding high income)
            TEA East Asia & Pacific (IDA & IBRD countries)
            EMU Euro area
            ECS Europe & Central Asia
            ECA Europe & Central Asia (excluding high income)
            TEC Europe & Central Asia (IDA & IBRD countries)
            EUU European Union
            FCS Fragile and conflict affected situations
            HPC Heavily indebted poor countries (HIPC)
            HIC High income
            LTE Late-demographic dividend
            LCN Latin America & Caribbean
            LAC Latin America & Caribbean (excluding high income)
            TLA Latin America & the Caribbean (IDA & IBRD countries)
            LDC Least developed countries: UN classification
            LMY Low & middle income
            LIC Low income
            LMC Lower middle income
            MEA Middle East & North Africa
            MNA Middle East & North Africa (excluding high income)
            TMN Middle East & North Africa (IDA & IBRD countries)
            MIC Middle income
            NAC North America
            INX Not classified
            OED OECD members
            OSS Other small states
            PSS Pacific island small states
            PST Post-demographic dividend
            PRE Pre-demographic dividend
            SST Small states
            SAS South Asia
            TSA South Asia (IDA & IBRD)
            SSF Sub-Saharan Africa
            SSA Sub-Saharan Africa (excluding high income)
            TSS Sub-Saharan Africa (IDA & IBRD countries)
            UMC Upper middle income
            WLD World

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 2.0 Data modification and Final dataframe: 

    2.1 Reading data of 195 countries prepared above after removing non country list.
    2.2 Finding corelation between attributes
    2.3 Merging Attributes based on our Financial learning and IMF information
    2.4 Writing it to csv WoRLD_Data.csv to be used further in project.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
### 3.0 Exploratory Data Analysis:
-- Data descritption using describe() function.
-- Finding corelation with heatmap

    3.1 Visualization of attributes with top 10 countries for:
        GDP
        NNI
        Banks Assets Ratio
        Claims on Economy
        Current Expenditure
        Domestic Credit
        Current Rents
        Higher Revenue

    3.2 Scatter plots showing corelation between attributes selected
        GDP
        Exports of Goods and Services
        GNI
        Imports of Goods and Services
        Merchandise Exports
        Trade
        Received Revenue

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
### 4.0 Modelling:

    4.1 Single Linear Regression Model:
        GDP (current US$) vs GNI (current US$)

    4.2 Multivariate Linear Regression:
        [Adjusted net national income (current US$)
        Adjusted savings: consumption of fixed capital (current US$)
        GNI (current US$)
        International tourism, expenditures (current US$)
        Merchandise imports (current US$)]
        Vs
        GDP (current US$)

    4.3 Multivariate Linear Regression with LabelEncoder:
        [Adjusted net national income (current US$)
        Adjusted savings: consumption of fixed capital (current US$)
        Broad money (% of GDP)
        Exports of goods and services (% of GDP)
        Gross capital formation (% of GDP)
        GNI (current US$)
        Gross national expenditure (% of GDP)
        Imports of goods and services (% of GDP)
        International tourism, expenditures (current US$)
        Trade (% of GDP)
        Value Added (% of GDP)
        Bank assets ratio (% of GDP)
        Claims on economy (% of GDP)
        Current Expenditure (% of GDP)
        Recieved revenue (% of GDP)
        Merchandise imports (current US$)]
        Vs
        GDP (current US$)

    4.4 Random Forest:
        [Adjusted net national income (current US$)
        Adjusted savings: consumption of fixed capital (current US$)
        Broad money (% of GDP)
        GNI (current US$)
        International tourism, expenditures (current US$)
        Merchandise imports (current US$)]
        Vs
        [GDP (current US$)
        Exports of goods and services (% of GDP)
        Gross capital formation (% of GDP)
        Imports of goods and services (% of GDP)
        International tourism, expenditures (current US$)
        Trade (% of GDP)]
        
    4.5 Decision Tree Model:
        [Adjusted net national income (current US$)
        Adjusted savings: consumption of fixed capital (current US$)
        Broad money (% of GDP)
        GNI (current US$)
        International tourism, expenditures (current US$)
        Merchandise imports (current US$)]
        Vs
        [GDP (current US$)
        Exports of goods and services (% of GDP)
        Gross capital formation (% of GDP)
        Imports of goods and services (% of GDP)
        International tourism, expenditures (current US$)
        Trade (% of GDP)]
        
    4.6 KNeighbour Regression:
        [Adjusted net national income (current US$)
        Adjusted savings: consumption of fixed capital (current US$)
        Broad money (% of GDP)
        GNI (current US$)
        International tourism, expenditures (current US$)
        Merchandise imports (current US$)]
        Vs
        [GDP (current US$)
        Exports of goods and services (% of GDP)
        Gross capital formation (% of GDP)
        Imports of goods and services (% of GDP)
        International tourism, expenditures (current US$)
        Trade (% of GDP)]

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
### 5.0 GNI Prediction using TS:

    5.1 Preparing TS dataframe
    5.2 Check for Stationarity
    5.3 Removing Stationarity, Seasonality and Trend

    5.4 Forecasting Values (Modelling):
        ACF and PACF
        5.4.1 AR(Auto-Regressive Model)
        5.4.2 MA(Moving-Average Model)
        5.4.3 ARIMA(Auto-Regressive Integrated Moving Average Model)
        5.4.4 Rescaling

    5.5 Forecasting values

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
### 6.0 WoRLD of World.ipynb:

This is the jupyter notebook which has more thorough details consisting of all the code along with the markdowns description of what the code in each cell does.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 7.0 CSV Files:

    7.1 Raw_Data.csv:
        Consists of the raw data collected from IMF and World Bank.
    7.2 Intermed_Data.csv:
        Consists of the data in which modification of columns and conversion of integer values to float was done.
    7.3 All_Data.csv:
        The modified pandas dataframe was stored in this csv file.
    7.4 All_New_Data.csv:
        This dataset consists of the 195 countries.
    7.5 WoRLD_Data.csv:
	This is the final dataset which was created after clubbing and merging the related attributes after analyzing the correlation matrix and also by removing the attributes which consists of less 	correlation value and the ones which were already clubbed into one. 

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 9.0 Conclusion:

	1. Kneighbour Regression is best model to predict with higher accuracy and lowest Error.
	2. Second best model is Linear Regression model with Label Encoder.
	3. ARIMA TS model is best for predicting future values. Here we did it for next 20 years with 95% confidence.
	4. ARIMA model is used to predict GNI for the attributes highly corelated with GDP.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 10.0 Limitation and Future Scope:

	1. Currently, we are having the limited data which are made public from IMF and WorldBank so the analysis is done based on those data. Also, the data is not rich for all parameters or indicators 	   thus more rich data will give a better prediction. 
	2. Data for lower income group countries are not available so more rich dataset is needed.
	3. To have better Time-Series calculation data needs monthly level data instead of annual data.


## END of Project
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import pandas as pd

import itertools

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2021-10-29'

df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = config_tickers.DOW_30_TICKER).fetch_data()

print(df_raw.head())

#Preprocess data
fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature = False)

processed = fe.preprocess_data(df_raw)
list_ticker = processed["tic"].unique().tolist()  #get all unique stock tickers from the tic column
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str)) #create a complete date range from earliest to latest date in the data
combination = list(itertools.product(list_date,list_ticker)) # create every possible combination of dates and tickers. Results in a list of tuples like [(date1,ticker1), (date1,ticker2), ...]

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])] #ensure that only dates that existed in the original data are kept
processed_full = processed_full.sort_values(['date','tic']) #sort by date then by ticker

processed_full = processed_full.fillna(0) #replace all nan values with 0

print(processed_full.head())

#Save the data
train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
print(len(train))
print(len(trade))

#save data to csv file
train.to_csv('train_data.csv')
trade.to_csv('trade_data.csv')
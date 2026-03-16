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

print("Raw data head:")
print(df_raw.head())

# Split data first to prevent lookahead bias
train_raw = data_split(df_raw, TRAIN_START_DATE, TRAIN_END_DATE)
trade_raw = data_split(df_raw, TRADE_START_DATE, TRADE_END_DATE)


#Preprocess data
fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature = False)

print("Processing training data...")
processed_train = fe.preprocess_data(train_raw)
print("Processing trading data...")
processed_trade = fe.preprocess_data(trade_raw)


def process_and_fill_data(processed_df):
    """
    Perform data combination and filling steps.
    """
    list_ticker = processed_df["tic"].unique().tolist()
    # use date range from raw data to avoid issues with missing dates in processed data
    list_date = list(pd.date_range(processed_df['date'].min(), processed_df['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed_df, on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full['date'].isin(processed_df['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])
    processed_full = processed_full.fillna(0)
    return processed_full

processed_full_train = process_and_fill_data(processed_train)
processed_full_trade = process_and_fill_data(processed_trade)


print("Processed training data head:")
print(processed_full_train.head())

#Save the data
print(f"Training data length: {len(processed_full_train)}")
print(f"Trading data length: {len(processed_full_trade)}")

processed_full_train.to_csv('train_data.csv', index=False)
processed_full_trade.to_csv('trade_data.csv', index=False)
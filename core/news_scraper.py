import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import date, datetime

def scrape_and_analyze_finviz_news(ticker):
    """
    Scrapes news headlines for a given ticker from Finviz,
    analyzes sentiment using FinBERT, and returns a DataFrame.

    :param ticker: The stock ticker symbol (e.g., 'AAPL').
    :return: A pandas DataFrame with columns: 'datetime', 'headline', 'link', 
             'positive', 'negative', 'neutral'.
             Returns an empty DataFrame on error.
    """
    # --- Scraping Part ---
    if "USDT" in ticker.upper(): #Check if crypto
        url = f"https://finviz.com/crypto_charts.ashx?t={ticker}"
    else: #stocks
        url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')
    news_table = soup.find(id='news-table')
    
    if not news_table:
        print(f"No news table found for {ticker}.")
        return pd.DataFrame()

    news_list = []
    rows = news_table.find_all('tr')
    last_date = None

    for row in rows:
        if row.a:
            headline = row.a.text
            link = row.a['href']
            date_time_str = row.td.text.strip()
            
            date_part = None
            time_part = None

            if ' ' in date_time_str:
                parts = date_time_str.split()
                if parts[0] == 'Today':
                    date_part = date.today()
                else:
                    try:
                        # Finviz format is like 'Oct-12-25'
                        parsed_date = datetime.strptime(parts[0], '%b-%d-%y').date()
                        if parsed_date > date.today():
                            parsed_date = parsed_date.replace(year=parsed_date.year - 100)
                        date_part = parsed_date
                        last_date = date_part
                    except ValueError:
                        # This happens if the date is just a time
                        time_part = parts[0]
                        date_part = last_date

                if time_part is None:
                    time_part = parts[1]
            else:
                time_part = date_time_str
                date_part = last_date

            if date_part and time_part:
                # Combine date and time
                try:
                    time_obj = datetime.strptime(time_part, '%I:%M%p').time()
                    full_datetime = datetime.combine(date_part, time_obj)
                    
                    news_list.append({
                        'datetime': full_datetime,
                        'headline': headline,
                        'link': link
                    })
                except ValueError:
                    # Handle cases where time format might be different or invalid
                    print(f"Could not parse time: {time_part}")


    if not news_list:
        return pd.DataFrame()

    news_df = pd.DataFrame(news_list)
    # Sort by datetime in ascending order for merge_asof
    news_df.sort_values(by='datetime', ascending=True, inplace=True)

    # --- Sentiment Analysis Part ---
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    headlines = news_df['headline'].tolist()
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores = scores.detach().numpy()

    news_df['positive'] = scores[:, 0]
    news_df['negative'] = scores[:, 1]
    news_df['neutral'] = scores[:, 2]
    
    news_df = news_df[['datetime', 'headline', 'link', 'positive', 'negative', 'neutral']]

    return news_df

if __name__ == '__main__':
    # Example usage:
    ticker_symbol = 'TSLA'
    news_with_sentiment = scrape_and_analyze_finviz_news(ticker_symbol)
    
    if not news_with_sentiment.empty:
        print(news_with_sentiment.head())
    else:
        print(f"No news or analysis found for {ticker_symbol}")

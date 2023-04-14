# Need to run pip install yfinance numpy scikit-learn python-dateutil
import yfinance as yf
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression

def predict_future_price(past_prices, future_time_steps):
    # Convert the past prices to a 2D array
    past_prices = np.array(past_prices).reshape(-1, 1)

    # Create a linear regression model and fit it to the past prices
    model = LinearRegression()
    model.fit(np.arange(len(past_prices)).reshape(-1, 1), past_prices)

    # Use the model to predict future prices
    future_prices = model.predict(np.arange(len(past_prices), len(past_prices) + future_time_steps).reshape(-1, 1))

    return future_prices.flatten()

def main():
    ticker = input("Enter the ticker symbol you want to predict the future price of: ")
    years = input("How many years out do you want to predict? ")
    data_range = input("How many years of historical data do you want to use for the calculation? ")

    start_date = (datetime.datetime.today() - relativedelta(years=int(data_range))).strftime('%Y-%m-%d')
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start_date, end=end_date)
    monthly_close_prices = data.resample('M').last()['Close']
    future_price = predict_future_price(monthly_close_prices, int(years) * 12)[-1]
    
    print(f"Extrapolation via Linear Regression predicts that {ticker} will be worth ${future_price:.2f} in {years} years based on {data_range} years of historical data.")

if __name__ == '__main__':
  main()

import yfinance as yf
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def predict_future_price(past_prices, future_time_steps):
    # Convert the past prices to a 2D array
    past_prices = np.array(past_prices).reshape(-1, 1)

    # Create a linear regression model and fit it to the past prices
    model = LinearRegression()
    model.fit(np.arange(len(past_prices)).reshape(-1, 1), past_prices)

    # Use the model to predict future prices
    future_prices = model.predict(np.arange(len(past_prices), len(past_prices) + future_time_steps).reshape(-1, 1))

    return future_prices.flatten()

def predict_future_price_test(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    future_prices = model.predict(X_test)

    return future_prices

def main():
    ticker = input("Enter the ticker symbol you want to predict the future price of: ")
    years = input("How many years out do you want to predict? ")
    data_range = input("How many years of historical data do you want to use for the calculation? ")

    start_date = (datetime.datetime.today() - relativedelta(years=int(data_range))).strftime('%Y-%m-%d')
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start_date, end=end_date)
    monthly_close_prices = data.resample('M').last()['Close']

    X = np.arange(len(monthly_close_prices)).reshape(-1, 1)
    y = np.array(monthly_close_prices).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # future price calculations using split data
    future_prices = predict_future_price_test(X_train, y_train, X_test)

    mse = mean_squared_error(y_test, future_prices)
    r2 = r2_score(y_test, future_prices)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # future price calculation using all data
    future_price = predict_future_price(monthly_close_prices, int(years) * 12)[-1]

    print(f"Extrapolation via Linear Regression predicts that {ticker} will be worth ${future_price:.2f} in {years} years based on {data_range} years of historical data.")

if __name__ == '__main__':
    main()

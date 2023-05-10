import yfinance as yf
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def predict_future_price_linear_test(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_prices = model.predict(X_test)
    return future_prices

def predict_future_price_linear(past_prices, future_time_steps):
    past_prices = np.array(past_prices).reshape(-1, 1)
    model = LinearRegression()
    model.fit(np.arange(len(past_prices)).reshape(-1, 1), past_prices)
    future_prices = model.predict(np.arange(len(past_prices), len(past_prices) + future_time_steps).reshape(-1, 1))
    return future_prices.flatten()

def predict_future_price_poly_test(X_train, y_train, X_test, degree=2):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    future_prices = model.predict(X_test)
    return future_prices

def predict_future_price_poly(past_prices, future_time_steps, degree=2):
    past_prices = np.array(past_prices).reshape(-1, 1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(np.arange(len(past_prices)).reshape(-1, 1), past_prices)
    future_prices = model.predict(np.arange(len(past_prices), len(past_prices) + future_time_steps).reshape(-1, 1))
    return future_prices.flatten()

def main():
    while True:
        print("Select a method for stock price prediction:")
        print("1. Linear Regression")
        print("2. Polynomial Regression")
        print("3. Exit")
        choice = input("Enter the number corresponding to your choice: ")

        if choice == "3":
            break

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
        future_prices_test = []
        future_price = 0
        if choice == "1":
            future_prices_test = predict_future_price_linear_test(X_train, y_train, X_test)
            future_price = predict_future_price_linear(monthly_close_prices, int(years) * 12)[-1]
        elif choice == "2":
            degree = int(input("Enter the degree of the polynomial (e.g., 2 for quadratic, 3 for cubic): "))
            future_prices_test = predict_future_price_poly_test(X_train, y_train, X_test, degree)
            future_price = predict_future_price_poly(monthly_close_prices, int(years) * 12, degree)[-1]

        mse = mean_squared_error(y_test, future_prices_test)
        r2 = r2_score(y_test, future_prices_test)

        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")
        print(f"Extrapolation via {'Linear' if choice == '1' else 'Poly'} Regression predicts that {ticker} will be worth ${future_price:.2f}")


if __name__ == '__main__':
    main()

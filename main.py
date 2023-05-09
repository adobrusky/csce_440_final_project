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


def predict_future_price_linear(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_prices = model.predict(X_test)
    return future_prices


def predict_future_price_poly(X_train, y_train, X_test, degree=2):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    future_prices = model.predict(X_test)
    return future_prices


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
        future_prices = []
        if choice == "1":
            future_prices = predict_future_price_linear(X_train, y_train, X_test)
        elif choice == "2":
            degree = int(input("Enter the degree of the polynomial (e.g., 2 for quadratic, 3 for cubic): "))
            future_prices = predict_future_price_poly(X_train, y_train, X_test, degree)

        mse = mean_squared_error(y_test, future_prices)
        r2 = r2_score(y_test, future_prices)

        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")

        future_price = future_prices[-1]

        print(f"Extrapolation via {'Linear' if choice == '1' else 'Poly'} Regression predicts that {ticker} will be worth ${future_price.tolist()[0]:.2f}")


if __name__ == '__main__':
    main()

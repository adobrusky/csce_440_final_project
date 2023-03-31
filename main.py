import requests

def main():
    ticker = input("Enter the ticker symbol you want to predict future prices for: ")
    # currently this request hardcodes to use monthly data since september of 2020
    # we can modify this later to support using older data (if we want)
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/month/2020-09-04/2023-01-08?adjusted=true&sort=asc&limit=50000&apiKey=uJmCV27N49KaojqWMjLK68df4MDLBMtb'
    response = requests.get(url)
    candlestick_data = response.json()['results']
    # we should use the monthly close prices as our points for interpolation
    close_prices = [month['c'] for month in candlestick_data]
    print("Close prices to use for interpolation: " + str(close_prices))

if __name__ == '__main__':
  main()

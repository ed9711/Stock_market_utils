from bs4 import BeautifulSoup
import requests
import datetime
import time
import numpy as np
from scipy.stats import norm
import yfinance as yf
from ta.utils import dropna


def d1(S, K, T, r, sigma):
    return (np.log(S/K)+(r+(sigma**2)/2)*T)/sigma*np.sqrt(T)


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma)-sigma*np.sqrt(T)


def calls_formula(S, K, T, r, sigma):
    return S*norm.cdf(d1(S, K, T, r, sigma))-K*np.exp(-r*T)*norm.cdf(d2(S, K, T, r, sigma))


def puts_formulat(S, K, T, r, sigma):
    return K*np.exp(-r*T)-S+calls_formula(S, K, T, r, sigma)


time_period = datetime.datetime.today().strftime('%Y-%m-%d')
today = time.time()
one_year_ago = datetime.datetime.now()
one_year_ago = one_year_ago.replace(year=one_year_ago.year-1).strftime('%Y-%m-%d')
interest = round(yf.download("^TNX", start=time_period)["Close"][0], 2)
df = yf.download("SPY", start=one_year_ago, end=time_period)
df = dropna(df)
df["sigma"] = (df["Close"] - df["Close"].shift(1))/df["Close"].shift(1)
df = dropna(df)  # dropped one NA, maybe add one more day to data?
sigma = np.sqrt(252) * df["sigma"].std()
options_url = "https://finance.yahoo.com/quote/SPY/options?date="
# print(today)
date = datetime.datetime.fromtimestamp(int(today))
yy = date.year
mm = date.month
dd = date.day
dd += 1
options_day = datetime.datetime(yy, mm, dd, 16, 0, 0, 0)
# print(options_day.timetuple())
datestamp = int(time.mktime(options_day.timetuple()))
# print(datestamp)
# print(datetime.datetime.fromtimestamp(options_stamp))

dates_list = []

for i in range(0, 20):
    test_req = requests.get(options_url + str(datestamp)).content
    # print(options_url + str(datestamp))
    content = BeautifulSoup(test_req, "html.parser")
    # print(content)
    tables = content.find_all("table")
    if not tables == []:
        print(datestamp)
        dates_list.append(datestamp)
    dd += 1
    if dd > 30:
        dd = 1
        mm += 1
    options_day = datetime.datetime(yy, mm, dd, 16, 0, 0, 0)
    datestamp = int(time.mktime(options_day.timetuple()))

# print(dates_list)
result_dict = {}
for item in dates_list:
    data_url = options_url + str(item)
    print(data_url)
    data_html = requests.get(data_url).content
    content = BeautifulSoup(data_html, "html.parser")
    price = round(yf.download("SPY", start=time_period)["Close"][0], 2)
    options_tables = []
    tables = content.find_all("table")
    for i in range(0, len(content.find_all("table"))):
        options_tables.append(tables[i])

    calls = options_tables[0].find_all("tr")[1:]

    for call in calls:
        call_data = []
        for td in BeautifulSoup(str(call), "html.parser").find_all("td"):
            call_data.append(td.text)
        call_info = {'contract': call_data[0], 'strike': call_data[2], 'last': call_data[3], 'bid': call_data[4],
                     'ask': call_data[5], 'volume': call_data[8], 'iv': call_data[10]}
        expire_day = datetime.datetime.utcfromtimestamp(item).strftime('%Y-%m-%d')
        expire_day = datetime.datetime.strptime(expire_day, '%Y-%m-%d')
        expire = (expire_day - datetime.datetime.utcnow()).days / 365
        call_price = calls_formula(price, float(call_info["strike"]), expire, interest, sigma)

        if call_price > float(call_data[5]):
            if float(call_data[5]) > 0:
                print(call_info)
                print(call_price)
                result_dict[call_price//float(call_data[5])] = (call_info, call_price)

print("Tabulated final result: ")
for i in sorted(result_dict.keys()):
    print(i, result_dict[i])


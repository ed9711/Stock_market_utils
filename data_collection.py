import yfinance as yf
from ta.utils import dropna
from ta.trend import MACD
from ta.trend import EMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import StochasticOscillator
import datetime
import pandas as pd
import time


def get_data(name):
    past_date = datetime.datetime.today() - datetime.timedelta(70)
    time_period = (past_date.strftime('%Y-%m-%d'),
                   (datetime.datetime.today() + datetime.timedelta(1)).strftime('%Y-%m-%d'))
    df = yf.download(name, start=time_period[0], end=time_period[1])

    # print(df.loc[time_period[0]])
    df = dropna(df)

    macd_result = MACD(close=df["Close"])
    df["MACD"] = macd_result.macd()
    df["MACD_histo"] = macd_result.macd_diff()
    df["MACD_signal"] = macd_result.macd_signal()

    ema_result = EMAIndicator(df["Close"])
    df["EMA"] = ema_result.ema_indicator()

    obv_result = OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    df["OBV"] = obv_result.on_balance_volume()

    sto_result = StochasticOscillator(df["Close"], df["High"], df["Low"])
    df["STO"] = sto_result.stoch()
    df["STO_signal"] = sto_result.stoch_signal()
    df.drop(df.head(len(df)-5).index, inplace=True)
    del df["Open"]
    del df["High"]
    del df["Low"]
    return df


def get_names(files):
    names = []
    for file in files:
        df = pd.read_csv(file)
        names.extend(df[df.columns[0]].tolist())
    return names


if __name__ == "__main__":
    start = time.time()
    print("Getting stocks")
    overall = pd.DataFrame(columns=['Close', 'Adj Close', 'Volume', 'MACD', 'MACD_histo',
                                    'MACD_signal', 'EMA', 'OBV', 'STO', 'STO_signal'])
    names = get_names(["./nyse.csv", "./nasdaq.csv"])
    print("Accessing DATA...")

    for name in names:
        print(name)
        df = get_data(name)
        if len(df) == 5:
            overall = overall.append(df)
    overall.to_csv("./data.csv", index=False)
    end = time.time()
    print("Time taken:{}s".format(end-start))














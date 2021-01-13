import yfinance as yf
from ta.utils import dropna
from ta.trend import MACD
from ta.trend import EMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import StochasticOscillator
import datetime


if __name__ == "__main__":
    past_date = datetime.datetime.today()-datetime.timedelta(60)
    time_period = (past_date.strftime('%Y-%m-%d'), datetime.datetime.today().strftime('%Y-%m-%d'))

    df = yf.download("MSFT", start=time_period[0], end=time_period[1])
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

    df.to_csv("./data.csv", index=False)















import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import catboost
import pickle

from freqtrade.strategy import IStrategy
import technical.indicators as ftt
# --------------------------------
# Add your lib to import here
from freqtrade.strategy import CategoricalParameter, RealParameter, DecimalParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
# SSL Channels
def SSLChannels(dataframe, length = 7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']

class PredictionStrategy(IStrategy):
    INTERFACE_VERSION = 2
    
    # For the pkl file, it is at: https://filen.io/d/5728cb6b-2ec7-446d-9166-dd760614c366#!0BC2FH2jNxmDPDCae3bNCDORFU7mNWjb

    # Buy hyperspace params:
    buy_params = {
        "buy_val": 0.4281607787074923,
        "buy_trigger": "pred4",  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_val": 0.3494876619363504,
        "sell_trigger": "pred4",  # value loaded from strategy
    }

    # ROI table:
    minimal_roi = {
      "0": 0.24,
      "446": 0.218,
      "1154": 0.09,
      "2438": 0
    }


    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.9

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.248  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 240

    # HO
    buy_trigger = CategoricalParameter(['pred0', 'pred1', 'pred2', 'pred3', 'pred4'], default=buy_params['buy_trigger'],
                                       space='buy', optimize=False, load=True)
    sell_trigger = CategoricalParameter(['pred0', 'pred1', 'pred2', 'pred3', 'pred4'],
                                        default=sell_params['sell_trigger'],
                                        space='sell', optimize=False, load=True)

    buy_val = DecimalParameter(low=0.35, high=0.58, default=buy_params['buy_val'], space="buy", optimize=True, load=True)
    sell_val = DecimalParameter(low=0.10, high=0.35, default=sell_params['sell_val'], space="sell", optimize=True, load=True)

    # Open model
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        with open('model_1h_fgi_Acc_6c_val_05_08_v2.pkl', 'rb') as f:
            model = pickle.load(f)
        self.model = model[0]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        verbose = False
	col_use = [
            'volume', 'smadiff_3', 'smadiff_5', 'smadiff_8', 'smadiff_13',
            'smadiff_21', 'smadiff_34', 'smadiff_55', 'smadiff_89',
            'smadiff_120', 'smadiff_240', 'maxdiff_3', 'maxdiff_5', 'maxdiff_8',
            'maxdiff_13', 'maxdiff_21', 'maxdiff_34', 'maxdiff_55', 'maxdiff_89',
            'maxdiff_120', 'maxdiff_240', 'std_3', 'std_5', 'std_8', 'std_13',
            'std_21', 'std_34', 'std_55', 'std_89', 'std_120', 'std_240',
            'ma_3', 'ma_5', 'ma_8', 'ma_13', 'ma_21', 'ma_34', 'ma_55', 'ma_89',
            'ma_120', 'ma_240', 'z_score_120', 'time_hourmin', 'time_dayofweek', 'time_hour',
            'uo', 'cci', 'rsi', 'adx', 'sar', 'ao', 'ha_open', 'ha_close', 'fisher_rsi', 'fisher_rsi_norma']
        # Starting create features
        # sma diff
        for i in [3, 5, 8, 13, 21, 34, 55, 89, 120, 240]:
            dataframe[f"smadiff_{i}"] = (dataframe['close'].rolling(i).mean() - dataframe['close'])
        # max diff
        for i in [3, 5, 8, 13, 21, 34, 55, 89, 120, 240]:
            dataframe[f"maxdiff_{i}"] = (dataframe['close'].rolling(i).max() - dataframe['close'])
        # min diff
        for i in [3, 5, 8, 13, 21, 34, 55, 89, 120, 240]:
            dataframe[f"maxdiff_{i}"] = (dataframe['close'].rolling(i).min() - dataframe['close'])
        # volatiliy
        for i in [3, 5, 8, 13, 21, 34, 55, 89, 120, 240]:
            dataframe[f"std_{i}"] = dataframe['close'].rolling(i).std()

        # Return
        for i in [3, 5, 8, 13, 21, 34, 55, 89, 120, 240]:
            dataframe[f"ma_{i}"] = dataframe['close'].pct_change(i).rolling(i).mean()

        dataframe['z_score_120'] = ((dataframe.ma_13 - dataframe.ma_13.rolling(21).mean() + 1e-9)
                                    / (dataframe.ma_13.rolling(21).std() + 1e-9))

        dataframe["date"] = pd.to_datetime(dataframe["date"], unit='ms')
        dataframe['time_hourmin'] = dataframe.date.dt.hour * 60 + dataframe.date.dt.minute
        dataframe['time_dayofweek'] = dataframe.date.dt.dayofweek
        dataframe['time_hour'] = dataframe.date.dt.hour

        # Ultimate Oscillator
        dataframe['uo'] = ta.ULTOSC(dataframe)

        # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # SAR
        dataframe['sar'] = ta.SAR(dataframe)
        # Awesome Oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["ha_open"] = heikinashi["open"]
        dataframe["ha_close"] = heikinashi["close"]

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # Model predictions
        preds = pd.DataFrame(self.model.predict_proba(dataframe[col_use]))
        preds.columns = [f"pred{i}" for i in range(5)]
        dataframe = dataframe.reset_index(drop=True)
        dataframe = pd.concat([dataframe, preds], axis=1)
		
        ssl_down_1h, ssl_up_1h = SSLChannels(dataframe, 12)
        dataframe['ssl_down'] = ssl_down_1h
        dataframe['ssl_up'] = ssl_up_1h
		
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                    (dataframe[self.buy_trigger.value] > self.buy_val.value) &
                    # (dataframe["time_hour"].isin([23,2,5,8,11,14,17,20])) &
					(dataframe['ssl_up'] > dataframe['ssl_down']) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # if self.config['runmode'].value == 'hyperopt':

        dataframe.loc[
            (
                    (dataframe[self.sell_trigger.value] < self.sell_val.value)
                    # & (dataframe["time_hour"].isin([23,2,5,8,11,14,17,20]))
                    & (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe

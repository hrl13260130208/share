from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import pandas as pd
# from WindPy import w
# Import the backtrader platform
import backtrader as bt


class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        print("-------------")
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def next(self):

        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])
        if self.dataclose[0]<self.dataclose[-1]:
            if self.dataclose[-1]<self.dataclose[-2]:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()


if __name__ == '__main__':
    c=bt.Cerebro()
    c.broker.set_cash(1000000)
    c.addstrategy(TestStrategy)

    dataframe = pd.read_csv(r'D:\temp\dfqc.csv', index_col=0, parse_dates=True)
    dataframe['openinterest'] = 0
    data = bt.feeds.PandasData(dataname=dataframe,
                               fromdate=datetime.datetime(2015, 1, 1),
                               todate=datetime.datetime(2016, 12, 31)
                               )
    c.adddata(data)

    print("start:",c.broker.get_cash())
    c.run()
    print("end:", c.broker.get_cash())

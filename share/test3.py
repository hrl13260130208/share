from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import pandas as pd
import numpy as np
# from WindPy import w
# Import the backtrader platform
import backtrader as bt
import matplotlib.pyplot as plt

class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
        ('printlog', False),
    )

    def log(self, txt, dt=None,doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # print("-------------")

        self.dataclose = self.datas[0].close
        self.order=None

        self.buyprice = None
        self.buycomm = None
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

    def next(self):

        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # print(self.position)
        # print("sma:",repr(self.sma[0]))
        if not self.position:

            if self.dataclose[0] >self.sma[0]:

                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy()
        else:

            if self.dataclose[0]<self.sma[0]:

                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')


        self.order = None

    def notify_trade(self, trade):
        # print(type(trade),trade)
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), doprint=True)


def test1():
    c = bt.Cerebro()
    c.broker.set_cash(10000)
    # c.addstrategy(TestStrategy)
    c.optstrategy(TestStrategy, maperiod=range(10, 50))

    dataframe = pd.read_csv(r'D:\temp\dfqc.csv', index_col=0, parse_dates=True)
    dataframe['openinterest'] = 0
    data = bt.feeds.PandasData(dataname=dataframe,
                               fromdate=datetime.datetime(2015, 1, 1),
                               todate=datetime.datetime(2016, 12, 31)
                               )
    c.adddata(data)
    c.broker.setcommission(0.0001)

    c.addsizer(bt.sizers.FixedSize, stake=100)
    # c.addsizer(bt.sizers.AllInSizer)
    # print("start:",c.broker.get_cash())
    c.run()
    # print("end:", c.broker.get_cash())
    # c.plot()
if __name__ == '__main__':
    dataframe = pd.read_csv(r'D:\temp\dfqc.csv')

    y=dataframe["close"].values
    x=np.array([x+1 for x in range(len(y))])
    print(x)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(m,c)
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, m * x + c, 'r', label='Fitted line')
    plt.legend()

    # print(dataframe["time"])
    # plt.scatter(dataframe["time"].values, dataframe["close"].values, color='blue')
    # A = np.vstack([range(dataframe["close"].values), np.ones(len(x))]).T
    plt.show()

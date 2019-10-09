import datetime
import tensorflow as tf
import json
import numpy as np
import os
from share import data_format
from share import week_lstm
from share import lstm_no_tai
import logging
import sys



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

def get_main(num=10):
    p = week_lstm.week_lstm_predict()
    datas = p.predict_all()
    week_lstm_s1 = {}
    week_lstm_num=0

    items = sorted(datas.items(), key=lambda item: item[1][0][0][0], reverse=True)
    for item in items:

        if p.Data_Format.has_item(data_format.SHARE_TYPE_1, item[0]):
            print(item)
            week_lstm_s1[item[0]]=item[1]
            week_lstm_num+=1

        if week_lstm_num>num:
            break


    p = lstm_no_tai.lnt_predict()
    datas = p.predict_all()
    lnt_s1 ={}
    lnt_num=0
    items = sorted(datas.items(), key=lambda item: item[1][0][0][1], reverse=True)

    for item in items:
        if p.Data_Format.has_item(data_format.SHARE_TYPE_1, item[0]):
            print(item)
            lnt_s1[item[0]]=item[1]
            lnt_num+=1

        if lnt_num > num:
            break

    print(set(lnt_s1.keys())&set(week_lstm_s1))
    for key in set(lnt_s1.keys())&set(week_lstm_s1):
        print(key,week_lstm_s1[key],lnt_s1[key])


if __name__ == '__main__':
    get_main(num=100)
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

def recommend_week_lstm(num=100):
    p = week_lstm.week_lstm_predict()
    datas = p.predict_all()
    week_lstm_s1 = {}
    week_lstm_num = 0

    items = sorted(datas.items(), key=lambda item: item[1][0][0][0], reverse=True)
    for item in items:

        if p.Data_Format.has_item(data_format.SHARE_TYPE_1, item[0]):
            print(item)
            week_lstm_s1[item[0]] = item[1]
            week_lstm_num += 1

        if week_lstm_num > num:
            break
    for key in week_lstm_s1.keys():
        r,u,d=rise_days(key)
        print(r,u,d)

def rise_days(ts_code,num=7,min_days=6):
    df=data_format.Data_Format()
    true_num=0
    flase_num=0
    for item in df.get_real_data(ts_code,num):
        if item[1]==1:
            true_num+=1
        else:
            flase_num+=1
    return true_num>min_days,true_num,flase_num







if __name__ == '__main__':
    get_main(num=100)
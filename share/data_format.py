import tushare as ts
import numpy as np
import redis
import json
import datetime



REDIS_IP="10.3.1.99"
REDIS_PORT="6379"
REDIS_DB="12"
QUEUE_NAME="file_queue"
redis_ = redis.Redis(host=REDIS_IP, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

MU_NAME="mu"
SIGMA_NAME="sigma"
MAX_NAME="max"
MIN_NAME="min"

Z_Score_Normalization="Z"
Max_Min_Normalization="M"

OPEN_LINE_NAME="open"
HIGH_LINE_NAME="high"
LOW_LINE_NAME="low"
CLOSE_LINE_NAME="close"
VOL_LINE_NAME="vol"
AMONT_LINE_NAME="amont"

#存储每日数据的数据项目的项目名称
SOURCE_NAME="source_data"
MMSTD_NAME="mmstd_data"




def Z_ScoreNormalization(x ,mu ,sigma):
    x = (x - mu) / sigma
    return x

def test_z():
    np_data = np.load(r"C:\File\numpy_data\1.npy")
    print(np_data.shape)
    mu = np.mean(np_data[:, 0])
    sigma = np.std(np_data[:, 0])
    print(mu, sigma)
    print(np_data[1, 0])
    print(Z_ScoreNormalization(np_data[1, 0], mu, sigma))


def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x

def parser_MaxMinNormalization(x,Max,Min):
    return x*(Max-Min)+Min

def test_m():
    np_data = np.load(r"C:\File\numpy_data\1.npy")
    max = np.max(np_data[:, 5])
    min = np.min(np_data[:, 5])
    print(max, min)
    print(MaxMinNormalization(np_data[0, 5], max, min))

def save_data(np_data):
    '''
    存储数据标准化所需的数据（最大值、最小值、平均值、标准差）
    :return:
    '''
    ts_code=np_data[0,0]
    line1="open"
    save_line(ts_code,line1,np_data[:,2])
    line2="high"
    save_line(ts_code,line2,np_data[:,3])
    line3="low"
    save_line(ts_code,line3,np_data[:,4])
    line4="close"
    save_line(ts_code,line4,np_data[:,5])
    line5="pre_close"
    save_line(ts_code, line5, np_data[:, 6])
    line6="vol"
    save_line(ts_code,line6,np_data[:,9])
    line7="amont"
    save_line(ts_code,line7,np_data[:,10])



def save_line(ts_code,line_name,line_data):
    mu = np.mean(line_data)
    sigma = np.std(line_data)
    max = np.max(line_data)
    min = np.min(line_data)
    redis_.set(create_name(ts_code,line_name,MU_NAME),mu)
    redis_.set(create_name(ts_code,line_name,SIGMA_NAME),sigma)
    redis_.set(create_name(ts_code,line_name,MAX_NAME),max)
    redis_.set(create_name(ts_code,line_name,MIN_NAME),min)

def create_name(ts_code,line_name,option_name):
    return str(ts_code)+"_"+str(line_name)+"_"+str(option_name)

def normalization_line_data(ts_code,line_name,line_data,normalization_type=Max_Min_Normalization,one_data=True):
    '''
    对指定列进行标准化处理
    :param ts_code:
    :param line_name:
    :param line_data:
    :param normalization_type:
    :param one_data: True表示该列只有一个数据
    :return:
    '''
    if normalization_type==Max_Min_Normalization:
        max=float(redis_.get(create_name(ts_code,line_name,MAX_NAME)))
        min=float(redis_.get(create_name(ts_code,line_name,MIN_NAME)))
        if one_data:
            return MaxMinNormalization(line_data,max,min)
        else:
            return MaxMinNormalization(line_data,np.full(len(line_data),max),np.full(len(line_data),min))
    elif normalization_type==Z_Score_Normalization:
        mu = float(redis_.get(create_name(ts_code, line_name, MU_NAME)))
        sigma =float(redis_.get(create_name(ts_code, line_name, SIGMA_NAME)))
        print(mu,sigma)
        return Z_ScoreNormalization(line_data, np.full(len(line_data), mu), np.full(len(line_data), sigma))

def get_data():
    ts.set_token('9ba459eb7da2bca7e1827a240a5654bacdf481a1ea997210e049ea91')
    pro = ts.pro_api()
    data = pro.daily(ts_code='000005.SZ', start_date='20180701', end_date='20190718')
    data1=normalization_line_data(data.values[0, 0], OPEN_LINE_NAME, data.values[:, 2])
    data2=normalization_line_data(data.values[0, 0], HIGH_LINE_NAME, data.values[:, 3])
    data3=normalization_line_data(data.values[0, 0], LOW_LINE_NAME, data.values[:, 4])
    data4=normalization_line_data(data.values[0, 0], CLOSE_LINE_NAME, data.values[:, 5])
    data5=normalization_line_data(data.values[0, 0], VOL_LINE_NAME, data.values[:, 9])
    data6=normalization_line_data(data.values[0, 0], AMONT_LINE_NAME, data.values[:, 10])
    return np.c_[data.values[:, 1],data1,data2,data3,data4,data5,data6]

def parser_data(ts_code,line_name,data,normalization_type=Max_Min_Normalization):
    '''
    将标准化的数据转换为正常的数据
    :param ts_code:
    :param line_name:
    :param data:
    :param normalization_type:
    :return:
    '''
    if normalization_type==Max_Min_Normalization:
        max = float(redis_.get(create_name(ts_code, line_name, MAX_NAME)))
        min = float(redis_.get(create_name(ts_code, line_name, MIN_NAME)))
        return parser_MaxMinNormalization(data,max,min)
    else:
        pass

class Data_Format():
    def __init__(self):
        ts.set_token('9ba459eb7da2bca7e1827a240a5654bacdf481a1ea997210e049ea91')
        self.pro = ts.pro_api()

    def update_data(self):
        pass

    def get_share_list(self):
        data=self.pro.stock_basic()
        return data.values[:,0],data.values

    def init_data(self,ts_code):
        '''
        使用daily默认返回值的顺序
        :param ts_code:
        :return:
        '''
        data = self.pro.daily(ts_code=ts_code)
        np_data=data.values
        save_line(ts_code, OPEN_LINE_NAME, np_data[:, 2])
        save_line(ts_code, HIGH_LINE_NAME, np_data[:, 3])
        save_line(ts_code, LOW_LINE_NAME, np_data[:, 4])
        save_line(ts_code, CLOSE_LINE_NAME, np_data[:, 5])
        save_line(ts_code, VOL_LINE_NAME, np_data[:, 9])
        save_line(ts_code, AMONT_LINE_NAME, np_data[:, 10])
        for line in np_data:
            self.save_day_data(ts_code,line[1],line)


    def save_day_data(self,ts_code,day,data):
        mmstd_data=self.get_mmstad_data(ts_code,data)
        map_name=self.create_ts_map_name(ts_code)
        redis_.hset(map_name,day,json.dumps({SOURCE_NAME:data.tolist(),MMSTD_NAME:mmstd_data}))
    def get_day_data(self,ts_code,day):
        return redis_.hget(self.create_ts_map_name(ts_code),day)

    def create_ts_map_name(self,ts_code):
        return ts_code+"_days"

    def get_mmstad_data(self,ts_code,data):
        data1 = normalization_line_data(ts_code, OPEN_LINE_NAME, data[ 2])
        data2 = normalization_line_data(ts_code, HIGH_LINE_NAME, data[ 3])
        data3 = normalization_line_data(ts_code, LOW_LINE_NAME, data[ 4])
        data4 = normalization_line_data(ts_code, CLOSE_LINE_NAME, data[5])
        data5 = normalization_line_data(ts_code, VOL_LINE_NAME, data[9])
        data6 = normalization_line_data(ts_code, AMONT_LINE_NAME, data[ 10])
        return [ data1, data2, data3, data4, data5, data6]

    def get_train_and_result_data(self,ts_code,date):
        '''
        获取训练数据与对应结果
            训练数据：七天内的数据
            训练结果：0或1     1第二天的收盘价上涨，0下跌
        :param date:
        :return:
        '''
        train_date,result_date=self.get_used_date(date)
        train_data=[]
        close=None
        for i in range(7):
            # print(train_date[6-i])
            data=self.get_day_data(ts_code,train_date[6-i])
            # print(data)
            data=json.loads(data)
            train_data.append(data[MMSTD_NAME])
            if i==6:
                close=data[SOURCE_NAME][5]
        next_data=self.get_day_data(ts_code,result_date)
        next_data=json.loads(next_data)
        if next_data[SOURCE_NAME][5]>close:
            return train_data,1
        else:
            return train_data,0



    def get_used_date(self,date,auto_find=False):
        data=self.pro.trade_cal(start_date=date, end_date=date)
        if data.values[0,2]==1:
            train_list=self.get_train_date(date)
            result_date=self.get_result_date(date)
            #to do：判断result_date是否可用
            return train_list,result_date
        else:
            if auto_find:
                d = datetime.datetime.strptime(date, "%Y%m%d")
                d1 = d - datetime.timedelta(days=1)
                last_date = d1.date().strftime("%Y%m%d")
                return self.get_used_date(last_date,auto_find=True)
            else:
                raise ValueError("输入日期不是交易日！")

    def get_train_date(self,date,train_list=[]):
        if train_list.__len__()==7:
            return train_list
        d = datetime.datetime.strptime(date, "%Y%m%d")
        d1 = d - datetime.timedelta(days=1)
        last_date=d1.date().strftime("%Y%m%d")
        data = self.pro.trade_cal(start_date=date, end_date=date)
        if data.values[0, 2] == 1:
            train_list.append(date)
            return self.get_train_date(last_date,train_list)
        else:
            return self.get_train_date(last_date, train_list)

    def get_result_date(self,date):
        d = datetime.datetime.strptime(date, "%Y%m%d")
        d1 = d + datetime.timedelta(days=1)
        next_date = d1.date().strftime("%Y%m%d")
        data = self.pro.trade_cal(start_date=next_date, end_date=next_date)
        if data.values[0, 2] == 1:
            return next_date
        else:
            return self.get_result_date(next_date)



    def get_current_data(self,ts_code):
        '''
        获取当前最新的数据用以预测
        :return:
        '''
        d=datetime.datetime.now()
        date=d.date().strftime("%Y%m%d")
        train_date,result_date=self.get_used_date(date,auto_find=True)
        train_data = []

        for i in range(7):
            # print(train_date[6-i])
            data = self.get_day_data(ts_code, train_date[6 - i])
            # print(data)
            data = json.loads(data)
            train_data.append(data[MMSTD_NAME])

        return train_data








if __name__ == '__main__':

    # d=datetime.datetime.strptime("20190901","%Y%m%d")
    # d1=d - datetime.timedelta(days=1)
    # print(d1.date().strftime("%Y%m%d"))

    # Data_Format().init_data("000005.SZ")
    # list=Data_Format().get_train_date("20190910")
    # list=Data_Format().get_train_and_result_data("000005.SZ","20190906")
    list=Data_Format().get_current_data("000005.SZ")
    print(list)
    # d=np.array([1,2,3]).tolist()
    # print(type(d))
    # np.save(r"C:\File\numpy_data\1.npy", get_data())
    # redis_.hset("data","day1",json.dumps([0,1,2,3]))
    # d=redis_.hget("data","day1")
    # print(d,type(d))
    # print(json.loads(d)[0],type(json.loads(d)))

    # for key in redis_.keys("*"):
    #     # redis_.delete(key)
    #     # print(key ,redis_.type(key))
    #     if redis_.type(key) == "string":
    #         print(key, redis_.get(key))
    #     elif redis_.type(key) == "set":
    #         print(key, " : ", redis_.scard(key), " : ", redis_.smembers(key))
    #     elif redis_.type(key) == "list":
    #         print(key, " : ", redis_.llen(key), " : ")  # , redis_.lrange(key,0,100))
    #     elif redis_.type(key) == "hash":
    #         print(key, " : ", redis_.hscan(key))  # , redis_.lrange(key,0,100))


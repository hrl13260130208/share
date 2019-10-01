import tushare as ts
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import redis
import json
import datetime
import time
import logging
import sys
from apscheduler.schedulers.blocking import BlockingScheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)


REDIS_IP="localhost"
REDIS_PORT="6379"
REDIS_DB="12"
QUEUE_NAME="file_queue"
redis_ = redis.Redis(host=REDIS_IP, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


Z_Score_Normalization="Z"
Max_Min_Normalization="M"


'''
redis存储内容说明：
    1、数据标准化相关数据 类型:string 命名格式:ts_code+lineName+nName
    2、股票每日数据 类型：hash 命名格式：ts_code+固定字符串“days”  
            hash中key的命名格式：日期 例：20161103
            hash中value的值以json的形式存储 json中存储 
                source_data：原始数据 
                mmstd_data：标准化后的数据
                predict_data：模型预测的结果（初始化时不会写入）
                real_data：实际数据涨跌的结果（初始化时不会写入）
    3、数字id 类型：hash 命名格式：列名+固定字符串“id”  （这是为了将非数字类型的数据转换成数字，然后使用词向量将其加入到模型中）
            hash中key的命名格式：ts_code  例：000001.SZ
            hash中value的值：对应id号                             
    4、股票列表 类型：hash 命名格式：固定名称--share_list
             hash中key的命名格式：ts_code 例：000002.SZ
             hash中value的值以json的形式存储 json中存储 
             source_data：原始数据 
             ids_data:该数据对应的id （目前存储：ts_code、area、industry）

'''
#训练数据文件名
DAYS_FILE=r"D:\data\share\data\days"
CODES_FILE=r"D:\data\share\data\codes"
AREAS_FILE=r"D:\data\share\data\areas"
INDUSTRYS_FILE=r"D:\data\share\data\industrys"
RESULTS_FILE=r"D:\data\share\data\results"
DATA_FILE=r"D:\data\share\data\datas"


#lineName
OPEN_LINE_NAME="open"
HIGH_LINE_NAME="high"
LOW_LINE_NAME="low"
CLOSE_LINE_NAME="close"
VOL_LINE_NAME="vol"
AMONT_LINE_NAME="amont"

#columnNmae
TS_CODE_NAME="ts_code"
AREA_NAME="area"
INDUSTRY_NAME="industry"

#nName
MU_NAME="mu"
SIGMA_NAME="sigma"
MAX_NAME="max"
MIN_NAME="min"

#存储每日数据的数据项目的项目名称
SOURCE_NAME="source_data"
MMSTD_NAME="mmstd_data"
#存储每日的预测数据与实际数据用于统计
PREDICT_NAME="predict_data"
REAL_NAME="real_data"

IDS_NAME="ids_data"


SHARE_LIST_NAME="share_list"

DATE_FORMAT="%Y%m%d"

NONE_NAME="none_item"

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
    if Max-Min==0:
        return 0
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

class Redis_Name_Manager():
    def create_ts_map_name(self, ts_code):
        '''
        存储股票每日数据的hash的名称
        :param ts_code:
        :return:
        '''
        return ts_code + "_days"

    def create_current_day_name(self, ts_code):
        return ts_code + "_current_day"

    def create_id_hash_name(self,column_name):
        return column_name+"_id"


class Data_Format():
    '''
    数据存储结构：
        每日数据：
            存储在redis的hash中
            hash的名称由ts_code确定（使用create_ts_map_name方法）
            hash中的key为日期（格式为字符串如：20190911） value为json字符串（需要loads和jumps，目前json中存储了
                一份原始数据（key为SOURCE_NAME）和一份标准化后的数据（key为MMSTD_NAME））
        标准化数据：
            目前存储了最大最小标准化和Z_Score所需要的数据，数据存储在redis的string中
            string的key通过create_name方法获得

    '''
    def __init__(self):
        ts.set_token('9ba459eb7da2bca7e1827a240a5654bacdf481a1ea997210e049ea91')
        self.pro = ts.pro_api()
        self.redis_names=Redis_Name_Manager()

        self.logger=logging.getLogger("data_format")
        self.logger.setLevel(logging.INFO)

    def get_share_list(self):
        data=self.pro.stock_basic()
        return data.values[:,0],data.values

    def init_data(self,ts_code):
        '''
        使用daily默认返回值的顺序
        :param ts_code:
        :return:
        '''
        self.logger.info("存储股票每日价格数据，股票代码："+str(ts_code))
        data = self.pro.daily(ts_code=ts_code)
        np_data = data.values
        save_line(ts_code, OPEN_LINE_NAME, np_data[:, 2])
        save_line(ts_code, HIGH_LINE_NAME, np_data[:, 3])
        save_line(ts_code, LOW_LINE_NAME, np_data[:, 4])
        save_line(ts_code, CLOSE_LINE_NAME, np_data[:, 5])
        save_line(ts_code, VOL_LINE_NAME, np_data[:, 9])
        save_line(ts_code, AMONT_LINE_NAME, np_data[:, 10])
        for line in np_data:
            self.save_day_data_sm(ts_code, line[1], line)

    def init_all(self):
        self.logger.info("获取股票信息列表...")
        data = self.pro.stock_basic()

        self.logger.info("设置ids...")
        self.save_id_data(self.redis_names.create_id_hash_name(TS_CODE_NAME),set(data.values[:,0]))
        self.save_id_data(self.redis_names.create_id_hash_name(AREA_NAME),set(data.values[:,3]))
        self.save_id_data(self.redis_names.create_id_hash_name(INDUSTRY_NAME),set(data.values[:,4]))

        for line in data.values:
            try:
                self.save_share_info(line)
                self.init_data(line[0])
            except:
                time.sleep(30)
                self.save_share_info(line)
                self.init_data(line[0])


    def save_share_info(self,info):
        '''
        存储股票信息
        :param info:
        :return:
        '''
        ts_code=info[0]
        self.logger.info("存储股票信息，股票代码："+str(ts_code)+",股票名称："+str(info[2]))
        ts_code_id=self.get_id(self.redis_names.create_id_hash_name(TS_CODE_NAME),ts_code)
        if info[3]==None:
            area_id=self.get_id(self.redis_names.create_id_hash_name(AREA_NAME),NONE_NAME)
        else:
            area_id=self.get_id(self.redis_names.create_id_hash_name(AREA_NAME),info[3])
        if info[4]==None:
            industry_id = self.get_id(self.redis_names.create_id_hash_name(INDUSTRY_NAME), NONE_NAME)
        else:
            industry_id=self.get_id(self.redis_names.create_id_hash_name(INDUSTRY_NAME),info[4])
        self.save_share_data(ts_code,json.dumps({SOURCE_NAME:info.tolist(),IDS_NAME:[ts_code_id,area_id,industry_id]}))

    def save_share_data(self,ts_code,data):
        redis_.hset(SHARE_LIST_NAME,ts_code,data)

    def get_share_data(self,ts_code):
        return redis_.hget(SHARE_LIST_NAME,ts_code)

    def save_id_data(self,name,set):
        for id,item in enumerate(set):
            if item==None:
                redis_.hset(name,NONE_NAME,0)
            else:
                redis_.hset(name,item,id+1)

    def get_id(self,hash_name,item_name):
        return redis_.hget(hash_name,item_name)

    def save_day_data_sm(self,ts_code,day,data):
        '''
        存储每日的原始数据和标准化数据
        :param ts_code:
        :param day:
        :param data:
        :return:
        '''
        mmstd_data=self.get_mmstad_data(ts_code,data)
        self.save_day_data(ts_code,day,json.dumps({SOURCE_NAME:data.tolist(),MMSTD_NAME:mmstd_data}))
        self.set_current_day(ts_code,day)

    def save_day_data(self,ts_code,day,data):
        redis_.hset(self.redis_names.create_ts_map_name(ts_code),day,data)

    def get_day_data(self,ts_code,day):
        return redis_.hget(self.redis_names.create_ts_map_name(ts_code),day)

    def set_current_day(self,ts_code,day):
        item_name=self.redis_names.create_current_day_name(ts_code)

        current_date=datetime.datetime.strptime(day,DATE_FORMAT )
        old_date=redis_.get(item_name)
        if old_date==None:
            redis_.set(item_name,current_date.date().strftime(DATE_FORMAT))
        else:
            old_date=datetime.datetime.strptime(old_date,DATE_FORMAT )
            if old_date<current_date:
                redis_.set(item_name,current_date.date().strftime(DATE_FORMAT))

    def get_current_day(self,ts_code):
        item_name = self.redis_names.create_current_day_name(ts_code)
        return redis_.get(item_name)


    def get_mmstad_data(self,ts_code,data):
        data1 = normalization_line_data(ts_code, OPEN_LINE_NAME, data[2])
        data2 = normalization_line_data(ts_code, HIGH_LINE_NAME, data[3])
        data3 = normalization_line_data(ts_code, LOW_LINE_NAME, data[ 4])
        data4 = normalization_line_data(ts_code, CLOSE_LINE_NAME, data[5])
        data5 = normalization_line_data(ts_code, VOL_LINE_NAME, data[9])
        data6 = normalization_line_data(ts_code, AMONT_LINE_NAME, data[ 10])
        return [ data1, data2, data3, data4, data5, data6]

    def get_all_day_datas(self):
        '''
        生成训练数据,通过访问tushare查询到具体的股票开市日期，来获取数据
        :return:
        '''
        train = []
        ts_codes = []
        areas = []
        industrys = []
        result = []
        for key in redis_.hkeys(SHARE_LIST_NAME):
            data = self.pro.daily(ts_code=key)
            np_data = data.values
            share_info=self.get_share_data(key)
            share_info=json.loads(share_info)
            ids=share_info[IDS_NAME]
            for i in np_data[1:-7]:
                self.logger.info("获取训练数据，股票代码："+str(key)+" 日期："+str(i[1]))
                try:
                    t,r=self.get_train_and_result_data(key,i[1])

                    self.save_predict(key,i[1],real_data=r)
                    ts_codes.append(ids[0])
                    areas.append(ids[1])
                    industrys.append(ids[2])
                    train.append(t)
                    result.append(r)
                except:
                    self.logger.error("出错！", exc_info=True)
        return np.array(train),np.array(result),np.array(ts_codes),np.array(areas),np.array(industrys)

    def get_ts_code_datas_by_sort(self,ts_code):
        '''
        获取指定股票的训练数据
        :param ts_code:
        :return:
        '''
        train = []
        ts_codes = []
        areas = []
        industrys = []
        result = []


        share_info = self.get_share_data(ts_code)
        share_info = json.loads(share_info)
        ids = share_info[IDS_NAME]
        name = self.redis_names.create_ts_map_name(ts_code)
        dates = redis_.hkeys(name)
        dates.sort()
        for index, date in enumerate(dates[:-1]):
            if index < 6:
                continue
            self.logger.info("获取训练数据，股票代码：" + str(ts_code) + " 日期：" + str(date))
            try:
                t, r = self.get_train_and_result_data_by_dates(ts_code, dates[index - 6:index + 1], dates[index + 1])

                self.save_predict(ts_code, date, real_data=r)
                ts_codes.append(int(ids[0]))
                areas.append(int(ids[1]))
                industrys.append(int(ids[2]))
                train.append(t)
                result.append([r])
            except:
                self.logger.error("出错！", exc_info=True)
        return np.array(train), np.array(result), np.array(ts_codes), np.array(areas), np.array(industrys)


    def get_all_datas_by_sort(self):
        """
        通过对redis中存储的日期进行排序，然后输出数据
        :return:
        """
        for key in redis_.hkeys(SHARE_LIST_NAME):

            datafile=open(DATA_FILE,"a+",encoding="utf-8")
            time.sleep(10)
            share_info = self.get_share_data(key)
            share_info = json.loads(share_info)
            ids = share_info[IDS_NAME]
            name=self.redis_names.create_ts_map_name(key)
            dates=redis_.hkeys(name)
            dates.sort()
            for index,date in enumerate(dates[:-1]):
                if index<6:
                    continue
                self.logger.info("获取训练数据，股票代码：" + str(key) + " 日期：" + str(date))
                try:
                    # print(dates[index-6:index+1],"=======",dates[index+1])
                    t, r = self.get_train_and_result_data_by_dates(key,dates[index-6:index+1],dates[index+1])

                    self.save_predict(key, date, real_data=r)

                    #days,codes,areas,ins,res
                    datafile.write(json.dumps([t,ids[0],ids[1],ids[2],r])+"\n")
                except:
                    self.logger.error("出错！", exc_info=True)
            datafile.close()

    # def save_TFRecord(self,TFRecord_file=r"D:\data\share\data\datas"):
    #     writer = tf.python_io.TFRecordWriter(TFRecord_file)
    #     for key in redis_.hkeys(SHARE_LIST_NAME):
    #
    #         datafile = open(DATA_FILE, "a+", encoding="utf-8")
    #         time.sleep(10)
    #         share_info = self.get_share_data(key)
    #         share_info = json.loads(share_info)
    #         ids = share_info[IDS_NAME]
    #         name = self.redis_names.create_ts_map_name(key)
    #         dates = redis_.hkeys(name)
    #         dates.sort()
    #         for index, date in enumerate(dates[:-1]):
    #             if index < 6:
    #                 continue
    #             self.logger.info("获取训练数据，股票代码：" + str(key) + " 日期：" + str(date))
    #             try:
    #                 # print(dates[index-6:index+1],"=======",dates[index+1])
    #                 t, r = self.get_train_and_result_data_by_dates(key, dates[index - 6:index + 1], dates[index + 1])
    #
    #                 self.save_predict(key, date, real_data=r)
    #
    #                 # days,codes,areas,ins,res
    #                 datafile.write(json.dumps([t, ids[0], ids[1], ids[2], r]) + "\n")
    #             except:
    #                 self.logger.error("出错！", exc_info=True)
    #         datafile.close()

    def save_predict(self,ts_code,date,predict_data=None,real_data=None):
        '''
        存储预测数据
        :param ts_code:
        :param date:
        :param predict_data:
        :param real_data:
        :return:
        '''
        save_data=self.get_day_data(ts_code,date)
        save_data=json.loads(save_data)
        if predict_data!=None:
            save_data[PREDICT_NAME]=predict_data
        if real_data!=None:
            save_data[REAL_NAME]=real_data
        self.save_day_data(ts_code,date,json.dumps(save_data))


    def get_train_and_result_data(self,ts_code,date):
        '''
        获取训练数据与对应结果
            训练数据：七天内的数据
            训练结果：[0,1] 第二天的收盘价上涨，[1,0]下跌
        :param date:
        :return:
        '''
        train_date,result_date=self.get_used_date(date)
        train_date.sort()
        if result_date==None:
            raise ValueError("无可用的预测数据！")
        return self.get_train_and_result_data_by_dates(ts_code,train_date,result_date)

    def get_train_and_result_data_by_dates(self,ts_code,train_date,result_date):
        '''

        :param ts_code:
        :param train_date: 长度为7 的list
        :param result_date: string
        :return:
        '''
        train_data = []
        close = None
        c=None
        for i in range(7):
            data = self.get_day_data(ts_code, train_date[i])
            data = json.loads(data)
            train_data.append(data[MMSTD_NAME])
            if i == 6:
                close = data[SOURCE_NAME][5]
                c=data[SOURCE_NAME]

        next_data = self.get_day_data(ts_code, result_date)
        next_data = json.loads(next_data)
        print("courent:",c)
        print("next:",next_data)
        if next_data[SOURCE_NAME][5] > close:
            result_data = [0, 1]
        else:
            result_data = [1, 0]
        print("result:",result_data)
        return train_data, result_data

    def get_used_date(self,date,auto_find=False):
        '''
        获取训练数据及结果对应的日期
        :param date:
        :param auto_find:
        :return:
        '''
        data=self.pro.trade_cal(start_date=date, end_date=date)
        if data.values[0,2]==1:
            train_list=self.get_train_date(date)
            result_date=self.get_result_date(date)
            return train_list,result_date
        else:
            if auto_find:
                d = datetime.datetime.strptime(date, DATE_FORMAT)
                d1 = d - datetime.timedelta(days=1)
                last_date = d1.date().strftime(DATE_FORMAT)
                return self.get_used_date(last_date,auto_find=True)
            else:
                raise ValueError("输入日期不是交易日！")

    def get_train_date(self,date,train_list=None):
        '''
        获取训练数据对应的日期
        :param date:
        :param train_list:
        :return:
        '''
        if train_list==None:
            train_list=[]
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
        '''
        获取预测结果对应的日期
        :param date:
        :return:
        '''
        d = datetime.datetime.strptime(date, "%Y%m%d")
        d1 = d + datetime.timedelta(days=1)
        if d1>datetime.datetime.now():
            return None
        next_date = d1.date().strftime("%Y%m%d")
        data = self.pro.trade_cal(start_date=next_date, end_date=next_date)
        if data.values[0, 2] == 1:
            return next_date
        else:
            return self.get_result_date(next_date)

    def get_current_data(self,ts_code,date=None):
        '''
        获取当前最新的数据用以预测
        :return:
        '''
        if date==None:
            d=datetime.datetime.now()
            date=d.date().strftime("%Y%m%d")

        train_date,result_date=self.get_used_date(date,auto_find=True)
        train_data = []

        self.logger.info("当前数据日期列表为："+str(train_date))
        for i in range(7):
            # print(train_date[6-i])
            data = self.get_day_data(ts_code, train_date[6 - i])
            # print(data)
            if data==None:
                self.logger.info("当前数据中有空值，更新日期！")
                return self.get_current_data(ts_code,date=train_date[1])
            data = json.loads(data)
            train_data.append(data[MMSTD_NAME])
        return train_data,train_date[0]

    def update_data(self,ts_code):
        '''
        更新指定股票的数据
        :param ts_code:
        :return:
        '''
        self.logger.info("更新股票："+str(ts_code))
        old_date=self.get_current_day(ts_code)
        self.save_daily_datas(ts_code,start_date=old_date)

    def save_daily_datas(self, ts_code, start_date=None):
        '''
        存储每日数据
        :param ts_code:
        :return:
        '''
        data = self.pro.daily(ts_code=ts_code, start_date=start_date)
        # print(data)
        np_data = data.values
        dates=[]
        for line in np_data:
            if line[1]==start_date:
                continue
            self.logger.info("数据日期："+str(line[1]))
            self.save_day_data_sm(ts_code, line[1], line)
            dates.append(line[1])
        current_date=self.get_current_day(ts_code)
        for i in dates:
            if i==current_date:
                continue
            t,r=self.get_train_and_result_data(ts_code,i)
            self.save_predict(ts_code,i,real_data=r)


def all_update():
    df=Data_Format()
    for key in redis_.hkeys(SHARE_LIST_NAME):
        try:
            df.update_data(key)
        except:
            pass
def timing():
    scheduler = BlockingScheduler()
    scheduler.add_job(func=all_update, trigger="cron", day="*", hour="16")
    scheduler.start()




if __name__ == '__main__':

    ts_code="000001.SZ"
   
    # nm=Redis_Name_Manager()
    # name=nm.create_ts_map_name(ts_code)
    # keys=redis_.hkeys(name)
    # keys.sort()
    # print(keys)
    # for key in keys:
    #     print(redis_.hget(name,key))

    # df=Data_Format()
    # df.get_ts_code_datas_by_sort(ts_code)
    all_update()
    #
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


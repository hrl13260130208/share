import tushare as ts
import numpy as np
import redis



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

def normalization_line_data(ts_code,line_name,line_data,normalization_type=Max_Min_Normalization):
    if normalization_type==Max_Min_Normalization:
        max=float(redis_.get(create_name(ts_code,line_name,MAX_NAME)))
        min=float(redis_.get(create_name(ts_code,line_name,MIN_NAME)))
        print(max,min)
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



if __name__ == '__main__':

    np.save(r"C:\File\numpy_data\1.npy", get_data())
    # for key in redis_.keys("*"):
    #     print(key,redis_.get(key))


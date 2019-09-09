import tushare as ts
import numpy as np
import tensorflow as tf


ts.set_token('9ba459eb7da2bca7e1827a240a5654bacdf481a1ea997210e049ea91')
pro = ts.pro_api()

def test1():
    # data=pro.stock_basic()
    data=pro.daily(ts_code='000005.SZ', start_date='20180701', end_date='20190718')
    # data=pro.trade_cal(exchange='', start_date='20180101', end_date='20181231')
    # print(data)
    # print(data.values)
    # for line in data.values:
    #     for l in line:
    #         print(l,type(l))

    np_data=[]
    for index in range(len(data.values)):
        # print(line,index)
        print(data.values[len(data.values)-index-1])
        np_data.append(data.values[len(data.values)-index-1][2:])

    np_data=np.array(np_data)
    # np_data=np_data.reshape((256,9))
    print(np_data)
    print(np_data.shape)
    np.save(r"C:\File\numpy_data\1.npy",np_data)
    # np.save(r"C:\File\numpy_data\1.npy",np_data)
    # seq_length = 10
    #
    # # Create training examples / targets
    # val=tf.convert_to_tensor(np_data,dtype=tf.int32)
    # print(val)
    # chunks=tf.data.Dataset.from_tensor_slices(val).batch(20+1)
    # print(chunks)
    # data.to_excel(r"C:\temp\pandas_daily.xlsx")
    # print(type(data),data[data.is_open==1])
    # for line in data:
    #     print(line)

def test2():
    data=pro.stock_company(exchange='SZSE',fields='ts_code,exchange,chairman,manager,secretary,reg_capital,setup_date,province,city,office,employees,main_business,business_scope')
    print(data)
    data.to_excel(r"D:\temp\ts1.xlsx")


def test3():
    pro = ts.pro_api()
    data = pro.daily(ts_code='000001.SZ')
    data.to_excel(r"D:\temp\ts3.xlsx")

if __name__ == '__main__':
    test3()




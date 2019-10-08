import datetime
import tensorflow as tf
import json
import numpy as np
import os
from share import data_format



model_path="D:\data\share\model\lstm_no_tai"


def get_model():
    #输入
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.CuDNNLSTM(256,input_shape=(7,6)))
    model.add(tf.keras.layers.Reshape((1,256)))
    model.add(tf.keras.layers.Dense(2,activation="softmax"))

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy]
                  )
    return model


def test():
    file_index=1
    index=0
    newfile=open(r"D:\data\share\data\split_datas_"+str(file_index),"w+")
    for line in open(r"D:\data\share\data\datas", encoding="utf-8"):
        print(repr(line))
        newfile.write(line)
        index+=1
        if index>4000000:
            newfile.close()
            index=0
            file_index+=1
            newfile = open(r"D:\data\share\data\split_datas_" + str(file_index),"w+")


def train(data_path=r"D:\data\share\data\split_datas_1",model_path="D:\data\share\model\lstm_no_tai"):
    model = get_model()
    # if os.path.exists(model_path):
    #     model.load_weights(model_path)
    t = []
    r = []
    for line in open(data_path, encoding="utf-8"):
        item = json.loads(line)
        t.append(item[0])
        r.append([item[4]])

    t = np.array(t)
    r = np.array(r)

    model.fit(t,  r, batch_size=10000, epochs=1000)
    model.save(model_path)


class lnt_predict():
    def __init__(self):
        self.names=data_format.Redis_Name_Manager()
        self.Data_Format=data_format.Data_Format()
        self.model=self.get_model()

    def get_model(self):
        model = get_model()
        model.load_weights(model_path)
        return model

    def predict(self, ts_code, date=None):
        if date == None:
            train_data, date = self.Data_Format.get_current_data(ts_code,use_sort=True)
            # print(train_data,date)
            train_data = np.array(train_data).reshape((1, 7, 6))
            real_data = None
        else:
            train_data, real_data = self.Data_Format.get_train_and_result_data(ts_code, date)
            train_data = np.array(train_data).reshape((1, 7, 6))

        predict_data = self.model.predict(train_data)
        self.Data_Format.save_predict(ts_code, date, predict_data=predict_data.tolist())
        return predict_data, real_data

    def predict_all(self):
        datas={}
        for ts_code in self.Data_Format.get_share_list():
            predict_data,_=self.predict(ts_code)
            datas[ts_code]=predict_data
            # if predict_data[0][0][0]<predict_data[0][0][1]:
            #     print("涨",ts_code,predict_data)
            # else:
            #     print("跌", ts_code, predict_data)
        return datas

    def predict_share(self,ts_code, ues_predicted=False):
        true_num=0
        flase_num=0
        data_list=self.Data_Format.get_ts_code_datas_by_sort(ts_code,return_list=True)
        for i in data_list:
            print("预测时间："+str(i[0]))
            p_data=self.model.predict(i[1])
            true_data=abs(p_data[0][0][0]-i[2][0])+abs(p_data[0][0][1]-i[2][1])
            if true_data>1:
                flase_num+=1
            else:
                true_num+=1
            print("预测数据：",p_data,"真实数据：",i[2],"   ",true_data,true_data>1)

        print("准确率：",true_num/(true_num+flase_num),true_num,flase_num)

def get_sort_predict(file_path=r"D:\data\share\predict"):
    date=datetime.datetime.now().date()
    p=lnt_predict()
    datas = p.predict_all()
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    # print(datas.items())
    items = sorted(datas.items(), key=lambda item: item[1][0][0][1], reverse=True)
    with open(os.path.join(file_path,str(date)),"w+",encoding="utf-8") as file:
        for item in items:
            print(item[0],item[1])
            if p.Data_Format.has_item(data_format.SHARE_TYPE_1,item[0]):
               s1.append(item)
            elif p.Data_Format.has_item(data_format.SHARE_TYPE_2,item[0]):
                s2.append(item)
            elif p.Data_Format.has_item(data_format.SHARE_TYPE_3,item[0]):
                s3.append(item)
            elif p.Data_Format.has_item(data_format.SHARE_TYPE_4,item[0]):
                s4.append(item)

        file.write("主板：\n")
        for i in s1:
            file.write(str(i)+"\n")
        file.write("创业板：\n")
        for i in s2:
            file.write(str(i)+"\n")
        file.write("中小板：\n")
        for i in s3:
            file.write(str(i)+"\n")
        file.write("科创板：\n")
        for i in s4:
            file.write(str(i)+"\n")



if __name__ == '__main__':
    # train()
    # test()
    # print(lnt_predict().predict("000001.SZ",date="20190919"))
    lnt_predict().predict_share("603032.SH")
    # get_sort_predict()
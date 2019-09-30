
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
            train_data, date = self.Data_Format.get_current_data(ts_code)
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
        for ts_code in self.Data_Format.get_share_list():
            predict_data,_=self.predict(ts_code)
            if predict_data[0]<predict_data[1]:
                print("涨",ts_code,predict_data)
            else:
                print("跌", ts_code, predict_data)


if __name__ == '__main__':
    # train()
    # test()
    # print(lnt_predict().predict("000001.SZ",date="20190919"))
    lnt_predict().predict_all()
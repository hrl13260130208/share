import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from share import data_format
import json


model_path=r"D:\data\share\model\test"


def get_data():
    df=data_format.Data_Format()
    # df.get_train_and_result_data()
    # return get_train_data(0,new_data,7,1,[],[])



def get_train_data(num,data,input_days,out_days,train_list,result_list):
    if num + input_days+out_days>=len(data):
        return train_list,result_list
    else:
        i=data[num:num + input_days]
        if out_days==1:
            l = data[num + input_days + out_days]
        else:
            l=data[num + input_days:num+input_days+out_days]
        train_list.append(i)
        result_list.append(l)
        return get_train_data(num +1,data,input_days,out_days,train_list,result_list)

def get_model():
    #输入
    lstm_input = tf.keras.Input(shape=(7,6), name='lstm_input')
    ts_code_input = tf.keras.Input(shape=(1) ,name='ts_code_input')
    area_input = tf.keras.Input(shape=(1), name='area_input')
    industry_input = tf.keras.Input(shape=(1), name='industry_input')
    #与输入对接的lstm和embedding
    lstm=tf.keras.layers.LSTM(units=256, input_shape=(7, 6))(lstm_input)
    area_e=tf.keras.layers.Embedding(input_dim=6000,output_dim=256)(ts_code_input)
    ts_code_e=tf.keras.layers.Embedding(input_dim=6000,output_dim=256)(area_input)
    industry_e=tf.keras.layers.Embedding(input_dim=6000,output_dim=256)(industry_input)


    a=tf.keras.backend.concatenate([area_e,ts_code_e,industry_e])
    info_out=tf.keras.layers.Dense(units=256)(a)
    lstm_reshape=tf.keras.layers.Reshape((1,256))(lstm)
    b = tf.keras.backend.concatenate([lstm_reshape,info_out])
    out1 = tf.keras.layers.Dense(units=256)(b)
    out=tf.keras.layers.Dense(units=2,activation="softmax")(out1)

    print(out.shape)
    model=tf.keras.Model(inputs=[lstm_input,ts_code_input,area_input,industry_input],outputs=out)

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy]
                  )
    return model


def main():
    model=get_model()
    train_data=np.load(r"D:\data\share\train_0913.npy")
    result_data=np.load(r"D:\data\share\verify_0913.npy")
    train_data=np.array(train_data)
    result_data=np.array(result_data)
    # model.build((None,7,6))
    model.summary()
    model.fit(train_data,result_data,nb_epoch=100,batch_size=10,verbose=1)
    model.save(model_path)


def predict(train_data):
    model=get_model()
    model.load_weights(model_path)
    result=model.predict(train_data)
    return result


class Predict():
    def __init__(self):
        self.names=data_format.Redis_Name_Manager()
        self.Data_Format=data_format.Data_Format()



    def predict(self,ts_code,date=None):
        if date==None:
            train_data,date=self.Data_Format.get_current_data(ts_code)
            print(train_data,date)
            train_data=np.array(train_data).reshape((1,7,6))
            real_data=None
        else:
            train_data,real_data=self.Data_Format.get_train_and_result_data(ts_code,date)

        predict_data = predict(train_data)
        self.Data_Format.save_predict(ts_code,date,predict_data=predict_data.tolist())
        return predict_data,real_data

if __name__ == '__main__':
    # pass
    # get_data()
    # data=np.load(r"C:\File\numpy_data\1.npy")
    # print(np.array([data[len(data)-index-1] for index in range(data.__len__()) ]).shape)
    # main()
    # train_data = data_format.Data_Format().get_current_data("000005.SZ")
    # print(train_data)
    # train_data = np.array(train_data)
    # train_data = train_data.reshape((1, 7, 6))
    # print(predict(train_data))
    # get_data()
    # Predict().predict("000005.SZ",date="20190911")
    model=get_model()
    t, r, t1, a, i = data_format.Data_Format().get_all_datas_by_sort()
    model.fit({"lstm_input":t,"ts_code_input":t1,"area_input":a,"industry_input":i},r,batch_size=2000,epochs=1000)

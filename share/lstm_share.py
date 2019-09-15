import matplotlib.pyplot as plt
import tensorflow as tf
from share import utils
import numpy as np
from share import data_format

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
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(units=256, return_sequences=True, input_shape=(7,6)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=256, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=256, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=256))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=2,activation="softmax"))
    # model.compile(optimizer='adam', loss='softmax')

    # # input_shape = (time_step, 每个时间步的input_dim)
    # model.add(tf.keras.layers.LSTM(1024, input_shape=(7,6)))
    # model.add(tf.keras.layers.Dense(6, activation="softmax"))
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
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




if __name__ == '__main__':
    # pass
    # get_data()
    # data=np.load(r"C:\File\numpy_data\1.npy")
    # print(np.array([data[len(data)-index-1] for index in range(data.__len__()) ]).shape)
    # main()
    train_data = data_format.Data_Format().get_current_data("000005.SZ")
    print(train_data)
    train_data = np.array(train_data)
    train_data = train_data.reshape((1, 7, 6))
    print(predict(train_data))
    # get_data()


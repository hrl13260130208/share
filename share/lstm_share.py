import matplotlib.pyplot as plt
import tensorflow as tf
from share import utils
import numpy as np
from share import data_format

model_path=r"C:\data\rnn\share_lstm_test\model"


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
    model.add(tf.keras.layers.Dense(units=6))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # # input_shape = (time_step, 每个时间步的input_dim)
    # model.add(tf.keras.layers.LSTM(1024, input_shape=(7,6)))
    # model.add(tf.keras.layers.Dense(6, activation="softmax"))
    # model.compile(optimizer=tf.train.AdamOptimizer(0.01),
    #               loss='mse',
    #               metrics=['mae'])
    return model


def main():
    model=get_model()
    train_data,result_data=get_data()
    train_data=np.array(train_data)
    result_data=np.array(result_data)
    print(train_data.shape,"---------------",result_data[0])
    # model.build((None,7,6))
    model.summary()
    model.fit(train_data,result_data,nb_epoch=10,batch_size=10,verbose=1)
    model.save(model_path)


def predict():
    model=get_model()
    model.load_weights(model_path)
    train_data, result_data = get_data()
    train_data = np.array(train_data)
    result_data = np.array(result_data)

    results=[]
    for data in train_data:
        print(data.shape)
        result=model.predict(np.array([data]))
        results.append(result)
        print(utils.parser_data("000005.SZ",utils.OPEN_LINE_NAME,result[0][0]))

    print(result_data.shape)
    print(np.array(results).shape)
    plt.plot(result_data[:,5], color='black', label='source')
    plt.plot(np.array(results)[:,0,5], color='green', label='Predicted ')
    plt.title('TATA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('TATA Stock Price')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    pass
    # get_data()
    # data=np.load(r"C:\File\numpy_data\1.npy")
    # print(np.array([data[len(data)-index-1] for index in range(data.__len__()) ]).shape)
    # main()
    # predict()
    # get_data()


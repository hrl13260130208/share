import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


class Model_(tf.keras.Model):
    def __init__(self):
        super(Model_, self).__init__()
        self.out = tf.keras.layers.LSTM(10,dtype=tf.float32,return_state=True)
    def call(self, inputs, training=None, mask=None):
        return self.out(inputs)

data=np.array([1,2,3,4])
data=data.reshape((1,4,1))
data2=np.array([2,2,2])
data2=data2.reshape((1,3,1))
print(data)
inputs = tf.keras.layers.Input(shape=(4,1))
inputs2=tf.keras.layers.Input(shape=(3,1))

# out=tf.keras.layers.LSTM(10,return_state=True)(inputs)
# model = tf.keras.Model(inputs=inputs, outputs=out)

# out=tf.keras.layers.concatenate([inputs,inputs2],axis=0)
# model=tf.keras.Model(inputs=[inputs,inputs2],outputs=out)


model=Model_()
# model.build((None,3,1))
model.compile(optimizer='adam',
              loss='mse')
print(data)
print(model.predict(data))


# model = tf.keras.Sequential()
# model.add(tf.keras.layers.LSTM(10, input_shape=(3,1 ),return_sequences=True,return_state=True))
print(data)
print(model.predict(data))


# samples = tf.multinomial([[1., 1., 1.,2.,3.]], 1)
# print(samples[-1,0].numpy())

# file_path=r"C:\data\rnn\share_lstm_test"
#
#
# def get_data(num,data,input_days,out_days,list):
#     if num + input_days+out_days>=len(data):
#         return list
#     i=data[num:num + input_days]
#     if out_days==1:
#         l = data[num + input_days + out_days]
#     else:
#         l=data[num + input_days:num+input_days+out_days]
#     input_data.append(i)
#     labels.append(l)
#     list.append((i,l))
#     return get_data(num +1,data,input_days,out_days,list)
#
#
# input_data=[]
# labels=[]
# num_s = 0
# np_data =np.load(r"C:\File\numpy_data\1.npy")
# list=get_data(num_s,np_data,7,1,[])
#
#
# model = tf.keras.Sequential()
# input_data=np.array(input_data)
# labels=np.array(labels)
#
#
# # input_shape = (time_step, 每个时间步的input_dim)
# model.add(tf.keras.layers.BatchNormalization(axis=1))
# model.add(tf.keras.layers.LSTM(1024, input_shape=(7,9 )))
# model.add(tf.keras.layers.Dense(9, activation=tf.nn.relu))
# model.compile(optimizer=tf.train.AdamOptimizer(0.01),
#               loss='mse',
#               metrics=['mae'])
# model.fit(input_data,labels,nb_epoch=100, batch_size=100, verbose=1)
# model.save(file_path+"/test_model")
# model.load_weights(file_path+"/test_model")
# print(model.predict(input_data[0].reshape(1,7,9)))

# for data in list:
#     pass
    # print(data[0])
    # print(data[0].shape)
    # print(data[1])
    # print(data[1].shape)

    # model.fit(data[0], data[1], nb_epoch=100, batch_size=100, verbose=2)




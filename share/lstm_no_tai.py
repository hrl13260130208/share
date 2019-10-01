
import tensorflow as tf
import json
import numpy as np
import os
from share import predict



model_path="D:\data\share\model\lstm_no_tai"


def get_model():
    #输入
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.CuDNNLSTM(256,input_shape=(7,6)))
    model.add(tf.keras.layers.Reshape((1,256)))
    model.add(tf.keras.layers.Dense(2,activation="softmax"))

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.0002),
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
    if os.path.exists(model_path):
        model.load_weights(model_path)
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


class lnt_predict(predict.Predict):
    def get_model(self):
        model = get_model()
        model.load_weights(model_path)
        return model

if __name__ == '__main__':
    train()
    # test()
    # print(lnt_predict().predict("000001.SZ"))

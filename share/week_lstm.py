
import datetime
import tensorflow as tf
import json
import numpy as np
import os
from share import data_format



model_path="D:\data\share\model\week_lstm"

def get_model():
    #输入
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.CuDNNLSTM(256,input_shape=(30,6)))
    model.add(tf.keras.layers.Reshape((1,256)))
    model.add(tf.keras.layers.Dense(4,activation="softmax"))

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy]
                  )
    return model


def train(data_path=r"D:\data\share\data\week_split_datas_1",model_path=model_path):
    model = get_model()
    if os.path.exists(model_path):
        model.load_weights(model_path)
    t = []
    r = []
    for line in open(data_path, encoding="utf-8"):
        item = json.loads(line)
        t.append(item[0])
        r.append(item[4])

    t = np.array(t)
    r = np.array(r)

    model.fit(t,  r, batch_size=10000, epochs=1000)
    model.save(model_path)

if __name__ == '__main__':
    train()

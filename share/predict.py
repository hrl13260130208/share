
from share import data_format
import numpy as np

class Predict():
    def __init__(self):
        self.names=data_format.Redis_Name_Manager()
        self.Data_Format=data_format.Data_Format()
        self.model=self.get_model()

    def get_model(self):
        pass

    def predict(self,ts_code,date=None):
        if date==None:
            train_data,date=self.Data_Format.get_current_data(ts_code)
            # print(train_data,date)
            train_data=np.array(train_data).reshape((1,7,6))
            real_data=None
        else:
            train_data,real_data=self.Data_Format.get_train_and_result_data(ts_code,date)
            train_data = np.array(train_data).reshape((1, 7, 6))

        predict_data = self.model.predict(train_data)
        self.Data_Format.save_predict(ts_code,date,predict_data=predict_data.tolist())
        return predict_data,real_data
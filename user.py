import pickle
import pandas as pd
from Logistic_Regression.code.LoadLogisticRegressionmodel import get_trained_model as load_LR
from KNN.LoadKNNmodel import knn_model as load_knn
from XGBoost.LoadXGBoostmodel import loadXGBoost as load_xgboost
from LightGBM.LoadModel import loadmodel as load_lgb
from neural_network.LoadNNmodel import model4Prediction as load_nn


class User():

    def __init__(self, user_info) -> None:
        d = {}
        for key, value in user_info.items():
            d[key] = [value]
        self.user_info = d
            
    
    def predict(self, model_name):
        
        if (model_name == "lr"):
            model = load_LR()
        elif (model_name == "knn"):
            model = load_knn()
        elif (model_name == "rf"):
            pass
        elif (model_name == "xg"):
            model = load_xgboost()
        elif (model_name == "lgb"):
            model = load_lgb()
        elif (model_name == "nn"):
            model = load_nn()

        X_test = pd.DataFrame(self.user_info)
        y_pred = model.predict(X_test)
        return y_pred


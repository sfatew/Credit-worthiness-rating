import pickle
import os


def loadXGBoost():
    with open('XGBoost/xgboost_model.pkl', 'rb') as f:
        XGBoost_model = pickle.load(f)
    return XGBoost_model
import pickle


def loadXGBoost():
    with open('xgboost_model.pkl', 'rb') as f:
        XGBoost_model = pickle.load(f)
    return XGBoost_model

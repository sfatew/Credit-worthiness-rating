import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import numpy as np

def loadmodel():
    bst = lgb.Booster(model_file='LightGBM/lightgbm_model.bin')
    return bst

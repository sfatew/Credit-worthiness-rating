import keras
from keras.models import load_model
import numpy as np


def model4Prediction():
    model = load_model('neural_network/model/best_model_final.keras')
    return model

# model4Prediction()

import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
import keras.regularizers as regularizers


def model1(optimizer, metric, loss):
    model = Sequential()
    model.add(Input(shape=(20,)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate=0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = loss, optimizer = optimizer, metrics = metric)
    return model

def model2(optimizer, metric, loss):
    model = Sequential()
    model.add(Input(shape=(20,)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate=0.4))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate=0.4))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate=0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = loss, optimizer = optimizer, metrics = metric)
    return model


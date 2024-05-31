import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
import keras.regularizers as regularizers


def model1(optimizer, loss, metric = None):
    model = Sequential()
    model.add(Input(shape=(16,)))
    model.add(Dense(24, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate=0.4))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.compile(loss = loss, optimizer = optimizer, metrics = metric)
    return model

def model2(optimizer, loss, metric = None):
    model = Sequential()
    model.add(Input(shape=(16,)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate=0.4))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate=0.4))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.compile(loss = loss, optimizer = optimizer, metrics = metric)
    return model

# def model3(optimizer, metric, loss):
#     model = Sequential()
#     model.add(Input(shape=(16,)))
#     model.add(Dense(32, activation='tanh' , kernel_regularizer=regularizers.l2(0.01)))
#     model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
#     model.compile(loss = loss, optimizer = optimizer, metrics = metric)
#     return model

# def model4(optimizer, metric, loss):
#     model = Sequential()
#     model.add(Input(shape=(16,)))
#     model.add(Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
#     model.add(Dense(16, activation='tanh'))
#     model.add(Dropout(rate=0.5))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss = loss, optimizer = optimizer, metrics = metric)
#     return model

# def meta_model(optimizer, metric, loss, input):
#     model = Sequential()
#     model.add(Input(shape=(input,)))
#     model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#     model.add(Dropout(rate=0.4))
#     model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
#     model.compile(loss = loss, optimizer = optimizer, metrics = metric)
#     return model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import pandas as pd

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import ADASYN
import keras_costum_metric.Balanced_accuracy as ba
import keras_costum_metric.F1metric_keras as f1
import keras_custom_callback.Early_Stop_AccLoss as Stop_AL
import keras_custom_callback.Checkpoint_best_auc as Checkpoint_A


import Set_model as model
import plot


# Load the dataset
credit_df=pd.read_csv("..\\data\\new_german_credit.csv")
# pd.set_option('display.max_columns', None)

X = credit_df.drop(columns=['Creditability', 'Unnamed: 0'])
Y = credit_df['Creditability']


scaler = StandardScaler()
scaler.fit(X)
t_X = scaler.transform(X)

Y = np.array(Y)

X_train_split, X_val, y_train_split, y_val = train_test_split(t_X, Y, test_size=0.2, stratify=Y)


loss = keras.losses.BinaryCrossentropy()

adasyn = ADASYN(random_state=200)


adam1_5 = keras.optimizers.Adam(learning_rate=0.001)

model1_5 = model.model1(optimizer=adam1_5, loss=loss)

X_adasyn, y_adasyn = adasyn.fit_resample(X_train_split, y_train_split)

early_stopping = keras.callbacks.EarlyStopping(
monitor="val_loss",
min_delta=0.001,
patience=60,
verbose=1,
mode="min",
)

checkpoint = keras.callbacks.ModelCheckpoint(
filepath = 'model\\best_model_final.keras',
monitor= "val_loss",
verbose=1,
mode="min",
save_best_only=True,  # Save only the best model based on the monitored metric
save_weights_only=False,  # Save the full model (set to True to save only the weights)
save_freq='epoch'  # Save the model at the end of every epoch
)


history = model1_5.fit(X_adasyn, y_adasyn, epochs=600, validation_data=(X_val, y_val), callbacks = [early_stopping, checkpoint])

best_model_1 = load_model('model\\best_model_final.keras')
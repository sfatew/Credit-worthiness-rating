from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf


def plotAccuracy(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()

def plotBalanceAccuracy(history):
    plt.plot(history.history['balanced_accuracy'], label='Training balanced accuracy')
    plt.plot(history.history['val_balanced_accuracy'], label='Validation balanced accuracy')
    plt.title('Model balanced accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('balanced accuracy')
    plt.legend(loc='upper left')
    plt.show()


def plotAUC(history):
    plt.plot(history.history['auc'], label='Training auc')
    plt.plot(history.history['val_auc'], label='Validation auc')
    plt.title('Model auc')
    plt.xlabel('Epoch')
    plt.ylabel('auc')
    plt.legend(loc='upper left')
    plt.show()


def plotLoss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def plotConfusionMatrix(model, X_test, y_test):
    predicted = model.predict(X_test)
    predicted = tf.squeeze(predicted)
    predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
    actual = np.array(y_test)
    conf_mat = confusion_matrix(actual, predicted)
    displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    displ.plot()
    plt.show()

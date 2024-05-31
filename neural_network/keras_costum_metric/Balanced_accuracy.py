import numpy as np
from keras.metrics import Metric
from tensorflow.python.keras import backend as K

class BalancedAccuracy(Metric):
    def __init__(self, name='balanced_accuracy', **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.cast(y_pred >= 0.5, 'float32')  # Convert probabilities to binary predictions
        y_true = K.cast(y_true, y_pred.dtype)  # Cast y_true to the same dtype as y_pred
        
        true_positives = K.sum(K.cast(y_true * y_pred, 'float32'))  # Count true positives
        true_negatives = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float32'))  # Count true negatives
        false_positives = K.sum(K.cast((1 - y_true) * y_pred, 'float32'))  # Count false positives
        false_negatives = K.sum(K.cast(y_true * (1 - y_pred), 'float32'))  # Count false negatives

        self.true_positives.assign_add(true_positives)
        self.true_negatives.assign_add(true_negatives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        sensitivity = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        specificity = self.true_negatives / (self.true_negatives + self.false_positives + K.epsilon())
        return (sensitivity + specificity) / 2.0

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.true_negatives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)
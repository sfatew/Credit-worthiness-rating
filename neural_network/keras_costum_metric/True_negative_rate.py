import numpy as np
from keras.metrics import Metric
from keras import backend as K

class TrueNegativeRate(Metric):
    def __init__(self, name='true_negative_rate', **kwargs):
        super(TrueNegativeRate, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.cast(y_pred >= 0.5, 'float32')  # Convert probabilities to binary predictions
        true_negatives = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float32'))  # Count true negatives
        self.true_negatives.assign_add(true_negatives)

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_positives + K.epsilon())

    def reset_state(self):
        self.true_negatives.assign(0.0)
import keras
import numpy as np

class EarlyStopAccLoss(keras.callbacks.Callback):
    def __init__(self, monitor_loss='val_loss', monitor_auc='val_auc', min_delta=0.001, min_delta_loss = 0.001, patience_loss=10, patience_auc = 10, verbose=1, mode='max', restore_best_weights=False):
        super(EarlyStopAccLoss, self).__init__()
        self.monitor_loss = monitor_loss
        self.monitor_auc = monitor_auc
        self.min_delta = min_delta
        self.min_delta_loss = min_delta_loss
        self.patience_loss = patience_loss
        self.patience_auc = patience_auc
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_auc = -np.Inf if mode == 'max' else np.Inf
        self.best_weights = None
        self.loss_converged = False


    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor_loss)
        current_auc = logs.get(self.monitor_auc)

        # # Print available keys in logs to debug
        # if logs is not None:
        #     print("Available keys in logs:", logs.keys())

        if current_loss is None or current_auc is None:
            return  # Skip if the required logs are not available

        # Ensure at least two epochs have passed to compare loss values
        if not self.loss_converged and len(self.model.history.history.get(self.monitor_loss, [])) > 1:
            # Check if validation loss has stopped improving
            if epoch > 0 and abs(current_loss - self.model.history.history['val_loss'][-2]) < self.min_delta_loss:
                self.wait += 1
                if self.wait >= self.patience_loss:
                    self.loss_converged = True
                    self.wait = 0  # reset wait counter for AUC improvement
                    print(f'Validation loss has converged. Now monitoring AUC.')
            else:
                self.wait = 0  # reset wait counter if loss improves

        if self.loss_converged:
            if (self.mode == 'max' and current_auc > self.best_auc + self.min_delta) or \
               (self.mode == 'min' and current_auc < self.best_auc - self.min_delta):
                self.best_auc = current_auc
                self.best_weights = self.model.get_weights()  # Save the best weights
                self.wait = 0
            else:
                self.wait += 1

            if self.wait >= self.patience_auc:
                if self.verbose > 0:
                    print(f"Epoch {epoch + 1}: early stopping - no improvement in {self.monitor_auc} for {self.patience_auc} consecutive epochs after loss has converged.")
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print("Restoring model weights from the epoch with the best AUC.")
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True
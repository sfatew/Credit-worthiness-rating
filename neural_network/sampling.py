from sklearn.utils import resample
import numpy as np

def create_bootstrap_samples(X, y, n_samples):
    X_samples, y_samples = [], []
    for _ in range(n_samples):
        X_resampled, y_resampled = resample(X, y, replace=True, random_state=np.random.randint(10000))
        X_samples.append(X_resampled)
        y_samples.append(y_resampled)
    return X_samples, y_samples
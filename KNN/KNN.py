import pickle
import os

def knn_model():

    current_dir = os.path.dirname(__file__)
    
    pickle_file_path = os.path.join(current_dir, 'KNN_model.pkl')

    with open(pickle_file_path, 'rb') as file:
        model = pickle.load(file)
    return model


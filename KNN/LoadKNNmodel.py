import pickle

def knn_model():

    with open('KNN/KNN_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


default_data = {
    "labels": ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
    "datasets": [
        {
            "label": "Logistic Regression",
            "data": [65, 59, 90, 81, 56],
            "fill": True,
            "backgroundColor": "rgba(75, 192, 192, 0.2)",
            "borderColor": "rgb(75, 192, 192)",
            "pointBackgroundColor": "rgb(75, 192, 192)",
            "pointBorderColor": "#fff",
            "pointHoverBackgroundColor": "#fff",
            "pointHoverBorderColor": "rgb(75, 192, 192)",
            "matrix": [
                [50, 10, 5],
                [8, 45, 7],
                [4, 6, 60]
            ]
        },
        {
            "label": "Random Forest",
            "data": [70, 65, 85, 75, 60],
            "fill": True,
            "backgroundColor": "rgba(255, 99, 132, 0.2)",
            "borderColor": "rgb(255, 99, 132)",
            "pointBackgroundColor": "rgb(255, 99, 132)",
            "pointBorderColor": "#fff",
            "pointHoverBackgroundColor": "#fff",
            "pointHoverBorderColor": "rgb(255, 99, 132)",
            "matrix": [
                [50, 10, 5],
                [8, 45, 7],
                [4, 6, 60]
            ]
        },
        {
            "label": "kNN",
            "data": [75, 70, 80, 78, 65],
            "fill": True,
            "backgroundColor": "rgba(54, 162, 235, 0.2)",
            "borderColor": "rgb(54, 162, 235)",
            "pointBackgroundColor": "rgb(54, 162, 235)",
            "pointBorderColor": "#fff",
            "pointHoverBackgroundColor": "#fff",
            "pointHoverBorderColor": "rgb(54, 162, 235)",
            "matrix": [
                [50, 10, 5],
                [8, 45, 7],
                [4, 6, 60]
            ]
        },
        {
            "label": "XGBoost",
            "data": [72, 68, 82, 76, 62],
            "fill": True,
            "backgroundColor": "rgba(255, 205, 86, 0.2)",
            "borderColor": "rgb(255, 205, 86)",
            "pointBackgroundColor": "rgb(255, 205, 86)",
            "pointBorderColor": "#fff",
            "pointHoverBackgroundColor": "#fff",
            "pointHoverBorderColor": "rgb(255, 205, 86)",
            "matrix": [
                [50, 10, 5],
                [8, 45, 7],
                [4, 6, 60]
            ]
        },
        {
            "label": "LightGBM",
            "data": [78, 72, 88, 80, 70],
            "fill": True,
            "backgroundColor": "rgba(153, 102, 255, 0.2)",
            "borderColor": "rgb(153, 102, 255)",
            "pointBackgroundColor": "rgb(153, 102, 255)",
            "pointBorderColor": "#fff",
            "pointHoverBackgroundColor": "#fff",
            "pointHoverBorderColor": "rgb(153, 102, 255)",
            "matrix": [
                [50, 10, 5],
                [8, 45, 7],
                [4, 6, 60]
            ]
        },
        {
            "label": "Neural Network",
            "data": [80, 75, 85, 82, 75],
            "fill": True,
            "backgroundColor": "rgba(255, 159, 64, 0.2)",
            "borderColor": "rgb(255, 159, 64)",
            "pointBackgroundColor": "rgb(255, 159, 64)",
            "pointBorderColor": "#fff",
            "pointHoverBackgroundColor": "#fff",
            "pointHoverBorderColor": "rgb(255, 159, 64)",
            "matrix": [
                [50, 10, 5],
                [8, 45, 7],
                [4, 6, 60]
            ]
        }
    ]
}





import json
import joblib


class AlgorithmInfo:
    def __init__(self) -> None:
        self.data = default_data

    def loadLogisticRegression(self):
        with open('Logistic_Regression/LogisticRegression_result.json') as json_file:
            result = json.load(json_file)
        temp = self.data['datasets'][0] # respectively to default data
        temp['data'] = [x for x in result.values()]
        temp['data'].pop()

        temp['matrix'] = result['Confusion Matrix']

        self.data['datasets'][0] = temp

    def loadKNN(self):
        with open('KNN/KNN_evaluation.json') as json_file:
            result = json.load(json_file)
        temp = self.data['datasets'][1] # respectively to default data
        temp['data'] = [x for x in result.values()]
        temp['data'].pop()

        temp['matrix'] = result['Confusion Matrix']

        self.data['datasets'][1] = temp

    def loadRandomForest(self):
        pass

    def loadXGBoost(self):
        with open('XGBoost/XGBoost_evaluation.json') as json_file:
            result = json.load(json_file)
        temp = self.data['datasets'][3] # respectively to default data
        temp['data'] = [x for x in result.values()]
        temp['data'].pop()

        temp['matrix'] = result['Confusion Matrix']

        self.data['datasets'][3] = temp

    def loadLightGBM(self):
        with open('LightGBM/lightgbm_result.json') as json_file:
            result = json.load(json_file)
        temp = self.data['datasets'][4] # respectively to default data
        temp['data'] = [x for x in result.values()]
        temp['data'].pop()

        temp['matrix'] = result['Confusion Matrix']

        self.data['datasets'][4] = temp

    def loadNeuralNetwork(self):
        with open('neural_network/NeuralNetwork_result.json') as json_file:
            result = json.load(json_file)
        temp = self.data['datasets'][5] # respectively to default data
        temp['data'] = [x for x in result.values()]
        temp['data'].pop()

        temp['matrix'] = result['Confusion Matrix']

        self.data['datasets'][5] = temp

    def loadData(self):
        self.loadLogisticRegression()
        self.loadKNN()
        self.loadRandomForest()
        self.loadXGBoost()
        self.loadLightGBM()
        self.loadNeuralNetwork()
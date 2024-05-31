# using flask_restful 
from flask import Flask, jsonify, request 
from flask_restful import Resource, Api 
from flask_cors import CORS, cross_origin

from user import User
from algoinfo import AlgorithmInfo

import numpy as np

app = Flask(__name__)
api = Api(app)
cors = CORS(app)

class Predict(Resource): 

	def get(self):
		algorithm_info = AlgorithmInfo()
		algorithm_info.loadData()
		return jsonify(algorithm_info.data)

	def post(self): 
		data = request.get_json() # status code
        
		datamap = {
			'model': data['model'],
            'Account Balance': data['account_balance'],
            'Duration of Credit (month)': data['duration_of_credit'],
            'Payment Status of Previous Credit': data['payment_status'],
            'Purpose': data['purpose'],
            'Credit Amount': data['credit_amount'],
            'Value Savings/Stocks': data['value_savings_stocks'],
            'Length of current employment': data['length_current_employment'],
            'Instalment per cent': data['installments_percent'],
            'Sex & Marital Status': data['sex_marital_status'],
            'Most valuable available asset': data['most_valuable_asset'],
            'Age (years)': data['age_years'],
            'Concurrent Credits': data['concurrent_credits'],
            'Type of apartment': data['type_of_apartment'],
            'No of Credits at this Bank': data['num_credits'],
			'Occupation': data['occupation'],
			'Telephone': data['telephone']
        }
		model = datamap['model']
		del datamap['model']
		user = User(datamap)
		result = user.predict(datamap['model']).tolist()[0]
		print(result)
		return result



# adding the defined resources along with their corresponding urls
api.add_resource(Predict, '/') 


# driver function 
if __name__ == '__main__': 
	app.run(debug = True)

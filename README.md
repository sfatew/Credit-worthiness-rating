# Credit-worthiness-rating
 
 In code folder, we have 4 python program, which is for using and testing purpose
 This README will explain the meaning of each program and the order of running file if you want to try are in the present order respectively

 **First of all** : Run *find_out_best_model.py*
 * This is the program to file the best parameter to set up for the initial model before start training.
 * This program also provide us some measure unit score of the best model, which ar precision, recall, accuracy, f1 and auc (area under the curve)

 **Second** : Run the *TrainModel.py*
 * this program define and function which user can get the coefficient of the best model corresponding to user's data.
 * From the best parameter form the first program, we assign these into the initial model in this program.

### Free order after two program above
* **evaluate.py**: show the probability of the logit function with the given data 
* **ROC_AUC_valuation.py**: to visualize the ROC graph and give us the confusion matrix from evaluation with the K fold.
* **calling_LogisticRegression_model.py**: use the method get_trained_model to get the model by: 
    * import calling_LogisticRegression_model
    * model = get_trained_model(). 


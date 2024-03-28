import os
import pickle
import numpy as np
from utils.metrics import metric
for num in range(2, 7): 
    for file_val in os.listdir("results_" + str(num) + "_val"):
        if "preds"in file_val:
            continue
        with open("results_" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues = pickle.load(file_object) 
            file_object.close() 
            print("results_" + str(num) + "_val/" + file_val, np.shape(trues))
            print(trues[0])
        with open("results_" + str(num) + "_val/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds = pickle.load(file_object) 
            file_object.close() 
            print("results_" + str(num) + "_val/" + file_val.replace("trues", "preds"), np.shape(preds))
            print(preds[0])
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(mae, mse, rmse, mape, mspe)
        
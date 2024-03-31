import os
import pickle
import numpy as np
from utils.metrics import metric
for num in range(2, 3): 
    for file_val in os.listdir("results_all_" + str(num) + "_train"):
        if "preds"in file_val:
            continue
        with open("results_all_" + str(num) + "_train/" + file_val, 'rb') as file_object:
            trues = pickle.load(file_object) 
            file_object.close() 
            print("results_all_" + str(num) + "_train/" + file_val, np.shape(trues))
            print(trues[0]) 
        break
        
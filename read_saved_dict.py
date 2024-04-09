import os
import pickle
import numpy as np
import pandas as pd
from utils.metrics import metric_short

ndec = {"direction": 0, "time": 3, "speed": 0}

def transform_pd_file(pd_file):
    pd_file = pd_file.drop(labels = "date", axis = 1)
    pd_file = pd_file.to_numpy()
    return pd_file 

varnames = ["direction", "speed", "time", "longitude_no_abs", "latitude_no_abs"]
first_sufix = ["", "S_", "MS_"]
second_sufix = ["", "all_"]
types_used = ["train", "val", "test"]
first_sufix = ["S_", "MS_"]

for num in range(2, 7): 

    for varname in varnames: 

        print(varname, num)

        for s1 in first_sufix:
            for s2 in second_sufix:
                for t in types_used:
                    print(s1, s2, t)
 
                    pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_" + t.upper() + ".csv")
                    pd_file_val_transformed = transform_pd_file(pd_file_val)  

                    with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/pred_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
                        preds_val = pickle.load(file_object)  
                        file_object.close()

                    with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/true_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
                        trues_val = pickle.load(file_object)  
                        file_object.close()
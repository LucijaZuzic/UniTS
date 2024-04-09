import os
import pickle
import numpy as np
import pandas as pd
from utils.metrics import metric_short

ndec = {"direction": 0, "time": 3, "speed": 0}

def transform_pd_file(pd_file):
    pd_file = np.array(pd_file["OT"])
    return pd_file 

varnames = ["direction", "speed", "time", "longitude_no_abs", "latitude_no_abs"]
first_sufix = ["S_", "MS_"]
t = "val"
s2 = ""

chose_vals = dict()
for varname in varnames:
    min_combo = (varname, -1, -1, -1, -1)
    min_combo_val = 1000000
    for num in range(2, 7):
        for s1 in first_sufix:
                #print(varname, num, s1, s2, t)

                pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_" + t.upper() + ".csv")
                pd_file_val_transformed = transform_pd_file(pd_file_val)  

                with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/pred_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
                    trues_val = pickle.load(file_object)  
                    file_object.close()

                with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/true_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
                    preds_val = pickle.load(file_object)  
                    file_object.close()

                #mae, mse, rmse = metric_short(preds_val, trues_val)
                #print(mae, mse, rmse)
                mae, mse, rmse = metric_short(preds_val, pd_file_val_transformed)
                #print(mae, mse, rmse)

                if mse < min_combo_val:
                    min_combo_val = mse
                    min_combo = (varname, num, s1, s2, t)

    print(varname, min_combo, min_combo_val)
    chose_vals[varname] = (min_combo[1], min_combo[2])

t = "test"
s2 = "all_"
for varname in chose_vals:
    num, s1 = chose_vals[varname]

    pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_" + t.upper() + ".csv")
    pd_file_val_transformed = transform_pd_file(pd_file_val)  

    with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/pred_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
        trues_val = pickle.load(file_object)  
        file_object.close()

    with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/true_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
        preds_val = pickle.load(file_object)  
        file_object.close()

    #mae, mse, rmse = metric_short(preds_val, trues_val)
    #print(varname, mae, mse, rmse)
    mae, mse, rmse = metric_short(preds_val, pd_file_val_transformed)
    print(varname, mae, mse, rmse)

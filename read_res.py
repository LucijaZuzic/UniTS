import os
import pickle
import numpy as np
import pandas as pd
from utils.metrics import metric

for num in range(2, 3): 
    for file_val in os.listdir("results_all_" + str(num) + "_train"):

        if "preds"in file_val:
            continue

        varname = file_val.split("_")[1]

        with open("results_all_" + str(num) + "_train/" + file_val, 'rb') as file_object:
            trues_all_train = pickle.load(file_object) 
            file_object.close() 
            print("results_all_" + str(num) + "_train/" + file_val, np.shape(trues_all_train))
            print(trues_all_train[0]) 
        with open("results_all_" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_all_val = pickle.load(file_object) 
            file_object.close() 
            print("results_all_" + str(num) + "_val/" + file_val, np.shape(trues_all_val))
            print(trues_all_val[0]) 
        with open("results_all_" + str(num) + "_test/" + file_val, 'rb') as file_object:
            trues_all_test = pickle.load(file_object) 
            file_object.close() 
            print("results_all_" + str(num) + "_test/" + file_val, np.shape(trues_all_test))
            print(trues_all_test[0]) 
        
        with open("results_" + str(num) + "_train/" + file_val, 'rb') as file_object:
            trues_train = pickle.load(file_object) 
            file_object.close() 
            print("results_" + str(num) + "_train/" + file_val, np.shape(trues_train))
            print(trues_train[0]) 
        with open("results_" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_val = pickle.load(file_object) 
            file_object.close() 
            print("results_" + str(num) + "_val/" + file_val, np.shape(trues_val))
            print(trues_val[0]) 
        with open("results_" + str(num) + "_test/" + file_val, 'rb') as file_object:
            trues_test = pickle.load(file_object) 
            file_object.close() 
            print("results_" + str(num) + "_test/" + file_val, np.shape(trues_test))
            print(trues_test[0])

        with open("results_all_" + str(num) + "_train/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_all_train = pickle.load(file_object) 
            file_object.close() 
            print("results_all_" + str(num) + "_train/" + file_val.replace("trues", "preds"), np.shape(preds_all_train))
            print(preds_all_train[0]) 
        with open("results_all_" + str(num) + "_val/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_all_val = pickle.load(file_object) 
            file_object.close() 
            print("results_all_" + str(num) + "_val/" + file_val.replace("trues", "preds"), np.shape(preds_all_val))
            print(preds_all_val[0]) 
        with open("results_all_" + str(num) + "_test/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_all_test = pickle.load(file_object) 
            file_object.close() 
            print("results_all_" + str(num) + "_test/" + file_val.replace("trues", "preds"), np.shape(preds_all_test))
            print(preds_all_test[0]) 
        
        with open("results_" + str(num) + "_train/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_train = pickle.load(file_object) 
            file_object.close() 
            print("results_" + str(num) + "_train/" + file_val.replace("trues", "preds"), np.shape(preds_train))
            print(preds_train[0]) 
        with open("results_" + str(num) + "_val/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_val = pickle.load(file_object) 
            file_object.close() 
            print("results_" + str(num) + "_val/" + file_val.replace("trues", "preds"), np.shape(preds_val))
            print(preds_val[0]) 
        with open("results_" + str(num) + "_test/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_test = pickle.load(file_object) 
            file_object.close() 
            print("results_" + str(num) + "_test/" + file_val.replace("trues", "preds"), np.shape(preds_test))
            print(preds_test[0])

        pd_file_train = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_TRAIN.csv")
        pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_VAL.csv")
        pd_file_test = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_TEST.csv")
        
        for ix1 in range(len(pd_file_test["OT"])):
            for ix in range(num):
                print(pd_file_test[str(ix)][ix1], trues_all_test[ix1][0][ix], trues_test[ix1][0][ix])
            print(pd_file_test["OT"][ix1], trues_all_test[ix1][0][num], trues_test[ix1][0][num])
            break
        break
        
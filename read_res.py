import os
import pickle
import numpy as np
import pandas as pd
from utils.metrics import metric

for num in range(2, 7): 

    for file_val in os.listdir("results/" + str(num) + "_val"):

        if "transformed" not in file_val:
            continue

        if "preds" in file_val:
            continue

        varname = file_val.replace("trues_transformed_", "")

        pd_file_train = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_TRAIN.csv")
        pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_VAL.csv")
        pd_file_test = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_TEST.csv")

        print(varname, num)
        with open("results/" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_S_all_val = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_all_" + str(num) + "_val/" + file_val, np.shape(trues_S_all_val))
            #print(trues_S_all_val[0]) 
        print(len(pd_file_val["OT"]), np.shape(trues_S_all_val))
        with open("results/" + str(num) + "_val/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_S_all_val = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_all_" + str(num) + "_val/" + file_val.replace("trues", "preds"), np.shape(preds_S_all_val))
            #print(preds_S_all_val[0]) 
        print(len(pd_file_val["OT"]), np.shape(preds_S_all_val))
        continue

        for i in range(30):
            lv = []
            for j in range(num):
                lv.append(pd_file_val[str(j)][i])
            lv.append(pd_file_val["OT"][i])
            print(lv, trues_S_all_val[i], preds_S_all_val[i])
            break

        for i in range(len(pd_file_val["OT"])):
            lv = []
            for j in range(num):
                lv.append(pd_file_val[str(j)][i])
            lv.append(pd_file_val["OT"][i])
            if (lv[0] == 203 and lv[1] == 203):
                print(lv)

        with open("results/all_" + str(num) + "_train/" + file_val, 'rb') as file_object:
            trues_all_train = pickle.load(file_object) 
            file_object.close() 
            #print("results/all_" + str(num) + "_train/" + file_val, np.shape(trues_all_train))
            #print(trues_all_train[0]) 
        print(len(pd_file_train["OT"]), np.shape(trues_all_train))
        with open("results/all_" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_all_val = pickle.load(file_object) 
            file_object.close() 
            #print("results/all_" + str(num) + "_val/" + file_val, np.shape(trues_all_val))
            #print(trues_all_val[0])
        print(len(pd_file_val["OT"]), np.shape(trues_all_val))
        with open("results/all_" + str(num) + "_test/" + file_val, 'rb') as file_object:
            trues_all_test = pickle.load(file_object) 
            file_object.close() 
            #print("results/all_" + str(num) + "_test/" + file_val, np.shape(trues_all_test))
            #print(trues_all_test[0]) 
        print(len(pd_file_test["OT"]), np.shape(trues_all_test))
        
        with open("results/" + str(num) + "_train/" + file_val, 'rb') as file_object:
            trues_train = pickle.load(file_object) 
            file_object.close() 
            #print("results/" + str(num) + "_train/" + file_val, np.shape(trues_train))
            #print(trues_train[0]) 
        print(len(pd_file_train["OT"]), np.shape(trues_train))
        with open("results/" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_val = pickle.load(file_object) 
            file_object.close() 
            #print("results/" + str(num) + "_val/" + file_val, np.shape(trues_val))
            #print(trues_val[0]) 
        print(len(pd_file_val["OT"]), np.shape(trues_val))
        with open("results/" + str(num) + "_test/" + file_val, 'rb') as file_object:
            trues_test = pickle.load(file_object) 
            file_object.close() 
            #print("results/" + str(num) + "_test/" + file_val, np.shape(trues_test))
            #print(trues_test[0])
        print(len(pd_file_test["OT"]), np.shape(trues_test))

        with open("results/all_" + str(num) + "_train/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_all_train = pickle.load(file_object) 
            file_object.close() 
            #print("results/all_" + str(num) + "_train/" + file_val.replace("trues", "preds"), np.shape(preds_all_train))
            #print(preds_all_train[0]) 
        print(len(pd_file_train["OT"]), np.shape(preds_all_train))
        with open("results/all_" + str(num) + "_val/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_all_val = pickle.load(file_object) 
            file_object.close() 
            #print("results/all_" + str(num) + "_val/" + file_val.replace("trues", "preds"), np.shape(preds_all_val))
            #print(preds_all_val[0]) 
        print(len(pd_file_val["OT"]), np.shape(preds_all_val))
        with open("results/all_" + str(num) + "_test/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_all_test = pickle.load(file_object) 
            file_object.close() 
            #print("results/all_" + str(num) + "_test/" + file_val.replace("trues", "preds"), np.shape(preds_all_test))
            #print(preds_all_test[0]) 
        print(len(pd_file_test["OT"]), np.shape(preds_all_test))
        
        with open("results/" + str(num) + "_train/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_train = pickle.load(file_object) 
            file_object.close() 
            #print("results/" + str(num) + "_train/" + file_val.replace("trues", "preds"), np.shape(preds_train))
            #print(preds_train[0]) 
        print(len(pd_file_train["OT"]), np.shape(preds_train))
        with open("results/" + str(num) + "_val/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_val = pickle.load(file_object) 
            file_object.close() 
            #print("results/" + str(num) + "_val/" + file_val.replace("trues", "preds"), np.shape(preds_val))
            #print(preds_val[0]) 
        print(len(pd_file_val["OT"]), np.shape(preds_val))
        with open("results/" + str(num) + "_test/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_test = pickle.load(file_object) 
            file_object.close() 
            #print("results/" + str(num) + "_test/" + file_val.replace("trues", "preds"), np.shape(preds_test))
            #print(preds_test[0])
        print(len(pd_file_test["OT"]), np.shape(preds_test))
 
        with open("results/S_all_" + str(num) + "_train/" + file_val, 'rb') as file_object:
            trues_S_all_train = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_all_" + str(num) + "_train/" + file_val, np.shape(trues_S_all_train))
            #print(trues_S_all_train[0]) 
        print(len(pd_file_train["OT"]), np.shape(trues_S_all_train))
        with open("results/S_all_" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_S_all_val = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_all_" + str(num) + "_val/" + file_val, np.shape(trues_S_all_val))
            #print(trues_S_all_val[0]) 
        print(len(pd_file_val["OT"]), np.shape(trues_S_all_val))
        with open("results/S_all_" + str(num) + "_test/" + file_val, 'rb') as file_object:
            trues_S_all_test = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_all_" + str(num) + "_test/" + file_val, np.shape(trues_S_all_test))
            #print(trues_S_all_test[0]) 
        print(len(pd_file_test["OT"]), np.shape(trues_S_all_test))
        
        with open("results/S_" + str(num) + "_train/" + file_val, 'rb') as file_object:
            trues_S_train = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_" + str(num) + "_train/" + file_val, np.shape(trues_S_train))
            #print(trues_S_train[0]) 
        print(len(pd_file_train["OT"]), np.shape(trues_S_train))
        with open("results/S_" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_S_val = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_" + str(num) + "_val/" + file_val, np.shape(trues_S_val))
            #print(trues_S_val[0]) 
        print(len(pd_file_val["OT"]), np.shape(trues_S_val))
        with open("results/S_" + str(num) + "_test/" + file_val, 'rb') as file_object:
            trues_S_test = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_" + str(num) + "_test/" + file_val, np.shape(trues_S_test))
            #print(trues_S_test[0])
        print(len(pd_file_test["OT"]), np.shape(trues_S_test))

        with open("results/S_all_" + str(num) + "_train/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_S_all_train = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_all_" + str(num) + "_train/" + file_val.replace("trues", "preds"), np.shape(preds_S_all_train))
            #print(preds_S_all_train[0]) 
        print(len(pd_file_train["OT"]), np.shape(preds_S_all_train))
        with open("results/S_all_" + str(num) + "_val/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_S_all_val = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_all_" + str(num) + "_val/" + file_val.replace("trues", "preds"), np.shape(preds_S_all_val))
            #print(preds_S_all_val[0]) 
        print(len(pd_file_val["OT"]), np.shape(preds_S_all_val))
        with open("results/S_all_" + str(num) + "_test/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_S_all_test = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_all_" + str(num) + "_test/" + file_val.replace("trues", "preds"), np.shape(preds_S_all_test))
            #print(preds_S_all_test[0]) 
        print(len(pd_file_test["OT"]), np.shape(preds_S_all_test))
        
        with open("results/S_" + str(num) + "_train/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_S_train = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_" + str(num) + "_train/" + file_val.replace("trues", "preds"), np.shape(preds_S_train))
            #print(preds_S_train[0]) 
        print(len(pd_file_test["OT"]), np.shape(preds_S_train))
        with open("results/S_" + str(num) + "_val/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_S_val = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_" + str(num) + "_val/" + file_val.replace("trues", "preds"), np.shape(preds_S_val))
            #print(preds_S_val[0]) 
        print(len(pd_file_val["OT"]), np.shape(preds_S_val))
        with open("results/S_" + str(num) + "_test/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_S_test = pickle.load(file_object) 
            file_object.close() 
            #print("results/S_" + str(num) + "_test/" + file_val.replace("trues", "preds"), np.shape(preds_S_test))
            #print(preds_S_test[0])
        print(len(pd_file_test["OT"]), np.shape(preds_S_test))

        break
         
    break
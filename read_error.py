import os
import pickle
import numpy as np
import pandas as pd
from utils.metrics import metric

def transform_pd_file(pd_file):
    pd_file = pd_file.drop(labels = "date", axis = 1)
    pd_file = pd_file.to_numpy()
    return pd_file 

def transform_np_file(np_file):
    np_file = np.array(np_file).squeeze()
    return np_file 

def find_match(transformed_pd_file, transformed_np_file, type, use_all, nr, var_name):

    with open("results/" + use_all + str(nr) + "_" + type + "/trues_transformed_" + var_name + "_matches_np_pd_trues_" + use_all + type, 'rb') as file_object:
        matches_np_pd = pickle.load(file_object)

    with open("results/" + use_all + str(nr) + "_" + type + "/trues_transformed_" + var_name + "_matches_pd_np_trues_" + use_all + type, 'rb') as file_object:
        matches_pd_np = pickle.load(file_object)

    with open("results/" + use_all + str(nr) + "_" + type + "/preds_transformed_" + var_name, 'rb') as file_object:
        preds_use = pickle.load(file_object)
    preds_use = transform_np_file(preds_use)

    used_pd = sorted(list(set(matches_np_pd.values())))
    print(len(used_pd), len(matches_np_pd.values()))
    print(used_pd[0:10], used_pd[-10:-1])
    diffs = [used_pd[ix] - used_pd[ix - 1] for ix in range(1, len(used_pd))]
    print(min(diffs), max(diffs))
    print(min(used_pd), max(used_pd))

    maxdiff = -10000
    preds_paired = []

    for ix_np_entry in range(np.shape(transformed_np_file)[0]):

        ix_pd_entry = matches_np_pd[ix_np_entry]
        
        diff = sum([abs(transformed_np_file[ix_np_entry][ix2] - transformed_pd_file[ix_pd_entry][ix2]) for ix2 in range(len(transformed_np_file[ix_np_entry]))])

        maxdiff = max(maxdiff, diff)

    unpaired = 0
    for ix_pd_entry in range(np.shape(transformed_pd_file)[0]):

        ix_np_entry = matches_pd_np[ix_pd_entry]

        if ix_np_entry != -1:
            preds_paired.append(transformed_np_file[ix_np_entry])

            diff = sum([abs(transformed_np_file[ix_np_entry][ix2] - transformed_pd_file[ix_pd_entry][ix2]) for ix2 in range(len(transformed_np_file[ix_np_entry]))])

            maxdiff = max(maxdiff, diff)
        else:
            preds_paired.append(-1)
            unpaired += 1

    if maxdiff > 1:
        print(maxdiff, "error")
    if unpaired > 0:
        print(unpaired, "error2")
    return preds_paired

for num in range(2, 7): 

    for file_val in os.listdir("results/" + str(num) + "_val"):

        if "transformed" not in file_val:
            continue

        if "preds" in file_val:
            continue

        varname = file_val.replace("trues_transformed_", "")
        print(varname, num)

        pd_file_train = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_TRAIN.csv")
        pd_file_train_transformed = transform_pd_file(pd_file_train)
        pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_VAL.csv")
        pd_file_val_transformed = transform_pd_file(pd_file_val)
        pd_file_test = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_TEST.csv")
        pd_file_test_transformed = transform_pd_file(pd_file_test)
 
        with open("results/all_" + str(num) + "_train/" + file_val, 'rb') as file_object:
            trues_all_train = pickle.load(file_object) 
            file_object.close()
        trues_all_train = transform_np_file(trues_all_train)
        preds_paired = find_match(pd_file_train_transformed, trues_all_train, "train", "all_", num, varname)
           
        with open("results/all_" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_all_val = pickle.load(file_object) 
            file_object.close()
        trues_all_val = transform_np_file(trues_all_val)
        preds_paired = find_match(pd_file_val_transformed, trues_all_val, "val", "all_", num, varname)
         
        with open("results/all_" + str(num) + "_test/" + file_val, 'rb') as file_object:
            trues_all_test = pickle.load(file_object) 
            file_object.close()
        trues_all_test = transform_np_file(trues_all_test)
        preds_paired = find_match(pd_file_test_transformed, trues_all_test, "test", "all_", num, varname)

        break
  
        with open("results/" + str(num) + "_train/" + file_val, 'rb') as file_object:
            trues_train = pickle.load(file_object) 
            file_object.close()
        trues_train = transform_np_file(trues_train)
        preds_paired = find_match(pd_file_train_transformed, trues_train, "train", "", num, varname)
          
        with open("results/" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_val = pickle.load(file_object) 
            file_object.close()
        trues_val = transform_np_file(trues_val)
        preds_paired = find_match(pd_file_val_transformed, trues_val, "val", "", num, varname)
         
        with open("results/" + str(num) + "_test/" + file_val, 'rb') as file_object:
            trues_test = pickle.load(file_object) 
            file_object.close()
        trues_test = transform_np_file(trues_test)
        preds_paired = find_match(pd_file_test_transformed, trues_test, "test", "", num, varname)
 
        with open("results/all_" + str(num) + "_train/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_all_train = pickle.load(file_object) 
            file_object.close()
        preds_all_train = transform_np_file(preds_all_train)
        with open("results/all_" + str(num) + "_val/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_all_val = pickle.load(file_object) 
            file_object.close()
        preds_all_val = transform_np_file(preds_all_val)
        with open("results/all_" + str(num) + "_test/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_all_test = pickle.load(file_object) 
            file_object.close()
        preds_all_test = transform_np_file(preds_all_test)
        
        with open("results/" + str(num) + "_train/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_train = pickle.load(file_object) 
            file_object.close()
        preds_train = transform_np_file(preds_train)
        with open("results/" + str(num) + "_val/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_val = pickle.load(file_object) 
            file_object.close()
        preds_val = transform_np_file(preds_val)
        with open("results/" + str(num) + "_test/" + file_val.replace("trues", "preds"), 'rb') as file_object:
            preds_test = pickle.load(file_object) 
            file_object.close()
        preds_test = transform_np_file(preds_test)

    break
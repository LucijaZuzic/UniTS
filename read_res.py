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

def find_match(transformed_pd_file, transformed_np_file):

    matches_np_pd = dict()
    matches_pd_np = dict()

    for ix_np_entry in range(np.shape(transformed_np_file)[0]):
        matches_np_pd[ix_np_entry] = -1

    for ix_pd_entry in range(np.shape(transformed_pd_file)[0]):
        matches_pd_np[ix_pd_entry] = -1

    for ix_np_entry in range(np.shape(transformed_np_file)[0]):

        np_entry = transformed_np_file[ix_np_entry]
 
        new_pd_file = transformed_pd_file - np_entry
        new_pd_file = [sum([abs(new_pd_file[ix1][ix2]) for ix2 in range(len(new_pd_file[ix1]))]) for ix1 in range(len(new_pd_file))]
        min_pd_entry = min(new_pd_file)
        ix_pd_entry = new_pd_file.index(min_pd_entry)
        matches_np_pd[ix_np_entry] = ix_pd_entry
        matches_pd_np[ix_pd_entry] = ix_np_entry

    return matches_np_pd, matches_pd_np

for num in range(2, 7): 

    for file_val in os.listdir("results/" + str(num) + "_val"):

        if "transformed" not in file_val:
            continue

        if "preds" in file_val:
            continue

        varname = file_val.replace("trues_transformed_", "")

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
        matches_np_pd_trues_all_train, matches_pd_np_trues_all_train = find_match(pd_file_train_transformed, trues_all_train)
         
        with open("results/all_" + str(num) + "_train/" + file_val + "_matches_np_pd_trues_all_train", 'wb') as file_object:
            pickle.dump(matches_np_pd_trues_all_train, file_object) 
            file_object.close()
        with open("results/all_" + str(num) + "_train/" + file_val + "_matches_pd_np_trues_all_train", 'wb') as file_object:
            pickle.dump(matches_pd_np_trues_all_train, file_object) 
            file_object.close()

        with open("results/all_" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_all_val = pickle.load(file_object) 
            file_object.close()
        trues_all_val = transform_np_file(trues_all_val)
        matches_np_pd_trues_all_val, matches_pd_np_trues_all_val = find_match(pd_file_val_transformed, trues_all_val)
        
        with open("results/all_" + str(num) + "_val/" + file_val + "_matches_np_pd_trues_all_val", 'wb') as file_object:
            pickle.dump(matches_np_pd_trues_all_val, file_object) 
            file_object.close()
        with open("results/all_" + str(num) + "_val/" + file_val + "_matches_pd_np_trues_all_val", 'wb') as file_object:
            pickle.dump(matches_pd_np_trues_all_val, file_object) 
            file_object.close()

        with open("results/all_" + str(num) + "_test/" + file_val, 'rb') as file_object:
            trues_all_test = pickle.load(file_object) 
            file_object.close()
        trues_all_test = transform_np_file(trues_all_test)
        matches_np_pd_trues_all_test, matches_pd_np_trues_all_test = find_match(pd_file_test_transformed, trues_all_test)

        with open("results/all_" + str(num) + "_test/" + file_val + "_matches_np_pd_trues_all_test", 'wb') as file_object:
            pickle.dump(matches_np_pd_trues_all_test, file_object) 
            file_object.close()
        with open("results/all_" + str(num) + "_test/" + file_val + "_matches_pd_np_trues_all_test", 'wb') as file_object:
            pickle.dump(matches_pd_np_trues_all_test, file_object) 
            file_object.close()

        with open("results/" + str(num) + "_train/" + file_val, 'rb') as file_object:
            trues_train = pickle.load(file_object) 
            file_object.close()
        trues_train = transform_np_file(trues_train)
        matches_np_pd_trues_train, matches_pd_np_trues_train = find_match(pd_file_train_transformed, trues_train)
         
        with open("results/" + str(num) + "_train/" + file_val + "_matches_np_pd_trues_train", 'wb') as file_object:
            pickle.dump(matches_np_pd_trues_train, file_object) 
            file_object.close()
        with open("results/" + str(num) + "_train/" + file_val + "_matches_pd_np_trues_train", 'wb') as file_object:
            pickle.dump(matches_pd_np_trues_train, file_object) 
            file_object.close()

        with open("results/" + str(num) + "_val/" + file_val, 'rb') as file_object:
            trues_val = pickle.load(file_object) 
            file_object.close()
        trues_val = transform_np_file(trues_val)
        matches_np_pd_trues_val, matches_pd_np_trues_val = find_match(pd_file_val_transformed, trues_val)
        
        with open("results/" + str(num) + "_val/" + file_val + "_matches_np_pd_trues_val", 'wb') as file_object:
            pickle.dump(matches_np_pd_trues_val, file_object) 
            file_object.close()
        with open("results/" + str(num) + "_val/" + file_val + "_matches_pd_np_trues_val", 'wb') as file_object:
            pickle.dump(matches_pd_np_trues_val, file_object) 
            file_object.close()

        with open("results/" + str(num) + "_test/" + file_val, 'rb') as file_object:
            trues_test = pickle.load(file_object) 
            file_object.close()
        trues_test = transform_np_file(trues_test)
        matches_np_pd_trues_test, matches_pd_np_trues_test = find_match(pd_file_test_transformed, trues_test)

        with open("results/" + str(num) + "_test/" + file_val + "_matches_np_pd_trues_test", 'wb') as file_object:
            pickle.dump(matches_np_pd_trues_test, file_object) 
            file_object.close()
        with open("results/" + str(num) + "_test/" + file_val + "_matches_pd_np_trues_test", 'wb') as file_object:
            pickle.dump(matches_pd_np_trues_test, file_object) 
            file_object.close()

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
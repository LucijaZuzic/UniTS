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

def transform_np_file(np_file):
    np_file = np.array(np_file).squeeze()
    return np_file

def find_match_X(transformed_pd_file, transformed_xs_file, transformed_true_file, transformed_pred_file, var_name, suf1):
    print("X", var_name, suf1)
    dict_xs_pred_true = dict()
    use_ndec = 5
    if var_name in ndec:
        use_ndec = ndec[var_name]

    for ix_xs in range(len(transformed_xs_file)):
        dict_xs_pred_key = np.round(transformed_xs_file[ix_xs], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        dict_xs_true_val = np.round(transformed_true_file[ix_xs], use_ndec)
        dict_xs_pred_val = np.round(transformed_pred_file[ix_xs], use_ndec)
        if "M" in suf1:
            dict_xs_pred_key = dict_xs_pred_key[-1]
            dict_xs_true_val = dict_xs_true_val[-1]
            dict_xs_pred_val = dict_xs_pred_val[-1]
        if dict_xs_pred_key not in dict_xs_pred_true:
            dict_xs_pred_true[dict_xs_pred_key] = []
        dict_xs_pred_true[dict_xs_pred_key].append([dict_xs_true_val, dict_xs_pred_val])
    
    doubled = []
    for ix_xs in range(len(transformed_xs_file)):
        dict_xs_pred_key = np.round(transformed_xs_file[ix_xs], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        if "M" in suf1:
            dict_xs_pred_key = dict_xs_pred_key[-1]
        if len(dict_xs_pred_true[dict_xs_pred_key]) > 1:
            doubled.append(len(dict_xs_pred_true[dict_xs_pred_key]))

    #print("Double min max", min(doubled), max(doubled))
    #print("Double", len(doubled), len(transformed_true_file), np.round(len(doubled) / len(transformed_true_file) * 100, 2))

    notfound = 0
    maxgap = -1
    mingap = 1000000
    maxix = -1
    minix = 1000000
    lastix = 0
    newpred = []
    newtrue = []
    for ix_match in range(len(transformed_pd_file)):
        dict_xs_pred_key = np.round(transformed_pd_file[ix_match], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        else:
            dict_xs_pred_key = dict_xs_pred_key[-1]
        if dict_xs_pred_key not in dict_xs_pred_true:
            notfound += 1
            gap = ix_match - lastix
            mingap = min(gap, mingap)
            maxgap = max(gap, maxgap)

            if "S" not in suf1:
                current_key = np.array(dict_xs_pred_key)
                key_array = np.array([list(ck) for ck in dict_xs_pred_true.keys()])
            else:
                current_key = np.array([dict_xs_pred_key])
                key_array = np.array([[ck] for ck in dict_xs_pred_true.keys()])
            key_diff = [abs(kd) for kd in list(key_array - current_key)]
            if "S" not in suf1:
                key_diff = [sum(kd) for kd in key_diff]
            min_key_diff = min(key_diff)
            ix_min_key_diff = key_diff.index(min_key_diff)
            min_key = key_array[ix_min_key_diff]
            dict_xs_pred_key = min_key
            if "S" not in suf1:
                dict_xs_pred_key = tuple(min_key)
            else:
                dict_xs_pred_key = min_key[0]
        else:
            lastix = ix_match
            minix = min(lastix, minix)
            maxix = max(lastix, maxix)

        candidates = dict_xs_pred_true[dict_xs_pred_key]
        ix_choose = 0
        min_error = abs(candidates[ix_choose][0] - candidates[ix_choose][1])
        if "S" not in suf1:
            min_error = sum(min_error)
        for ix_c in range(len(candidates)):
            new_error = abs(candidates[ix_choose][0] - candidates[ix_choose][1])
            if "S" not in suf1:
                new_error = sum(new_error)
            if new_error < min_error:
                ix_choose = ix_c
                min_error = new_error
        newpred.append(candidates[ix_choose][0])
        newtrue.append(candidates[ix_choose][1])
    
    #print("Gap", mingap, maxgap)
    #print(minix, maxix, len(transformed_pd_file))
    #print("Empty", notfound, len(transformed_pd_file), np.round(notfound / len(transformed_pd_file) * 100, 2))
    newpred, newtrue = np.array(newpred), np.array(newtrue)
    mae, mse, rmse = metric_short(newpred, newtrue)
    print(mae, mse, rmse)

def find_match_Y(transformed_pd_file, transformed_true_file, transformed_pred_file, var_name, suf1):
    print("Y", var_name, suf1)
    dict_xs_pred_true = dict()
    use_ndec = 5
    if var_name in ndec:
        use_ndec = ndec[var_name]

    for ix_xs in range(len(transformed_true_file)):
        dict_xs_pred_key = np.round(transformed_true_file[ix_xs], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        dict_xs_pred_val = np.round(transformed_pred_file[ix_xs], use_ndec)
        if "M" in suf1:
            dict_xs_pred_key = dict_xs_pred_key[-1]
            dict_xs_pred_val = dict_xs_pred_val[-1]
        if dict_xs_pred_key not in dict_xs_pred_true:
            dict_xs_pred_true[dict_xs_pred_key] = []
        dict_xs_pred_true[dict_xs_pred_key].append(dict_xs_pred_val)
    
    doubled = []
    for ix_xs in range(len(transformed_true_file)):
        dict_xs_pred_key = np.round(transformed_true_file[ix_xs], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        if "M" in suf1:
            dict_xs_pred_key = dict_xs_pred_key[-1]
        if len(dict_xs_pred_true[dict_xs_pred_key]) > 1:
            doubled.append(len(dict_xs_pred_true[dict_xs_pred_key]))

    #print("Double min max", min(doubled), max(doubled))
    #print("Double", len(doubled), len(transformed_true_file), np.round(len(doubled) / len(transformed_true_file) * 100, 2))

    notfound = 0
    maxgap = -1
    mingap = 1000000
    maxix = -1
    minix = 1000000
    lastix = 0
    newpred = []
    newtrue = []
    for ix_match in range(len(transformed_pd_file)):
        dict_xs_pred_key = np.round(transformed_pd_file[ix_match], use_ndec)
        if "S" not in suf1:
            dict_xs_pred_key = tuple(dict_xs_pred_key)
        else:
            dict_xs_pred_key = dict_xs_pred_key[-1]
        if dict_xs_pred_key not in dict_xs_pred_true:
            notfound += 1
            gap = ix_match - lastix
            mingap = min(gap, mingap)
            maxgap = max(gap, maxgap)

            if "S" not in suf1:
                current_key = np.array(dict_xs_pred_key)
                key_array = np.array([list(ck) for ck in dict_xs_pred_true.keys()])
            else:
                current_key = np.array([dict_xs_pred_key])
                key_array = np.array([[ck] for ck in dict_xs_pred_true.keys()])
            key_diff = [abs(kd) for kd in list(key_array - current_key)]
            if "S" not in suf1:
                key_diff = [sum(kd) for kd in key_diff]
            min_key_diff = min(key_diff)
            ix_min_key_diff = key_diff.index(min_key_diff)
            min_key = key_array[ix_min_key_diff]
            dict_xs_pred_key = min_key
            if "S" not in suf1:
                dict_xs_pred_key = tuple(min_key)
            else:
                dict_xs_pred_key = min_key[0]
        else:
            lastix = ix_match
            minix = min(lastix, minix)
            maxix = max(lastix, maxix)

        candidates = dict_xs_pred_true[dict_xs_pred_key]
        ix_choose = 0
        min_error = abs(candidates[ix_choose] - dict_xs_pred_key)
        if "S" not in suf1:
            min_error = sum(min_error)
        for ix_c in range(len(candidates)):
            new_error = abs(candidates[ix_c] - dict_xs_pred_key)
            if "S" not in suf1:
                new_error = sum(new_error)
            if new_error < min_error:
                ix_choose = ix_c
                min_error = new_error
        newpred.append(dict_xs_pred_key)
        newtrue.append(candidates[ix_choose])
    
    #print("Gap", mingap, maxgap)
    #print(minix, maxix, len(transformed_pd_file))
    #print("Empty", notfound, len(transformed_pd_file), np.round(notfound / len(transformed_pd_file) * 100, 2))
    newpred, newtrue = np.array(newpred), np.array(newtrue)
    mae, mse, rmse = metric_short(newpred, newtrue)
    print(mae, mse, rmse)
    
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

                    with open("results/"+ s1 + s2 + str(num) + "_" + t + "/" + "xs_transformed_" + varname, 'rb') as file_object:
                        xs_val = pickle.load(file_object) 
                        file_object.close()
                    xs_val = transform_np_file(xs_val)

                    with open("results/"+ s1 + s2 + str(num) + "_" + t + "/" + "trues_transformed_" + varname, 'rb') as file_object:
                        trues_val = pickle.load(file_object) 
                        file_object.close()
                    trues_val = transform_np_file(trues_val) 
            
                    with open("results/"+ s1 + s2 + str(num) + "_" + t + "/" + "preds_transformed_" + varname, 'rb') as file_object:
                        preds_val = pickle.load(file_object) 
                        file_object.close()
                    preds_val = transform_np_file(preds_val)

                    #find_match_X(pd_file_val_transformed, xs_val, trues_val, preds_val, varname, s1)
                    find_match_Y(pd_file_val_transformed, trues_val, preds_val, varname, s1)
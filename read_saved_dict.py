import os
import pickle
import numpy as np
import pandas as pd
from utils.metrics import metric_short
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

username = "Admin"
username = "lzuzi"

def change_angle(angle, name_file):
    
    file_with_ride = pd.read_csv("C:/Users/" + username + "/Documents/GitHub/MarkovOtoTrak/" + name_file) 
    
    x_dir = list(file_with_ride["fields_longitude"])[0] < list(file_with_ride["fields_longitude"])[-1]
    y_dir = list(file_with_ride["fields_latitude"])[0] < list(file_with_ride["fields_latitude"])[-1]

    new_dir = (90 - angle + 360) % 360 
    if not x_dir: 
        new_dir = (180 - new_dir + 360) % 360
    if not y_dir: 
        new_dir = 360 - new_dir 

    return new_dir

def get_sides_from_angle(longest, angle):
    return longest * np.cos(angle / 180 * np.pi), longest * np.sin(angle / 180 * np.pi)
 
mode = "min"
ndec = {"direction": 0, "time": 3, "speed": 0}

def transform_pd_file(pd_file):
    pd_file = np.array(pd_file["OT"])
    return pd_file 

def get_XY(dat, time_steps, len_skip = -1, len_output = -1):
    X = []
    Y = [] 
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    for i in range(0, len(dat), len_skip):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        y_vals = dat[i + time_steps:i + time_steps + len_output]
        if len(x_vals) == time_steps and len(y_vals) == len_output:
            X.append(np.array(x_vals))
            Y.append(np.array(y_vals))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

varnames = ["direction", "speed", "time", "longitude_no_abs", "latitude_no_abs"]
first_sufix = ["S_", "MS_"]
t = "val"
s2 = ""

chose_vals = dict()
for varname in varnames:

    with open("actual/actual_" + varname, 'rb') as file_object:
        file_object_test = pickle.load(file_object)

    all_mine_flat = []
    for filename in file_object_test: 
        for val in file_object_test[filename]:
            all_mine_flat.append(val)

    min_combo = (varname, -1, -1, -1, -1)
    min_combo_val = 1000000
    for num in range(2, 7):
        for s1 in first_sufix:
                #print(varname, num, s1, s2, t)

                pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_" + t.upper() + ".csv")
                pd_file_val_transformed = transform_pd_file(pd_file_val)  

                with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/" + mode + "_pred_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
                    preds_val = pickle.load(file_object)  
                    file_object.close()

                with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/" + mode + "_true_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
                    trues_val = pickle.load(file_object)  
                    file_object.close()

                #mae, mse, rmse = metric_short(preds_val, trues_val)
                #print(mae, mse, rmse)
                mae, mse, rmse = metric_short(preds_val, pd_file_val_transformed)
                #print(mae, mse, rmse)

                final_train_MAE = mean_absolute_error(preds_val, pd_file_val_transformed)
                final_train_R2 = r2_score(preds_val, pd_file_val_transformed)
                final_train_RMSE = math.sqrt(mean_squared_error(preds_val, pd_file_val_transformed) / (max(all_mine_flat) - min(all_mine_flat)))
                #print(final_train_MAE, final_train_R2, final_train_RMSE)

                if final_train_RMSE < min_combo_val:
                    min_combo_val = final_train_RMSE
                    min_combo = (varname, num, s1, s2, t)

    print(varname, min_combo, min_combo_val)
    chose_vals[varname] = (min_combo[1], min_combo[2])

t = "test"
s2 = "all_"
predicted_all = dict()
y_test_all = dict()
ws_all = dict()
model_name = "UNITS"
for varname in chose_vals:
    predicted_all[varname] = dict()
    predicted_all[varname][model_name] = dict()
    y_test_all[varname] = dict()
    y_test_all[varname][model_name] = dict()
    ws_all[varname] = dict()
    ws_all[varname][model_name] = chose_vals[varname]
    num, s1 = chose_vals[varname]
    
    pd_file_val = pd.read_csv("dataset_new/" + str(num) + "/" + varname + "/newdata_" + t.upper() + ".csv")
    pd_file_val_transformed = transform_pd_file(pd_file_val)  

    with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/" + mode + "_pred_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
        preds_val = pickle.load(file_object)  
        file_object.close()

    with open("results_eval/" + varname + "/" + s1 + s2 + str(num) + "_" + t + "/Y/" + mode + "_true_Y_" + s1 + s2 + str(num) + "_" + t, 'rb') as file_object:
        trues_val = pickle.load(file_object)
        file_object.close()
 
    with open("actual/actual_" + varname, 'rb') as file_object:
        file_object_test = pickle.load(file_object)

    lens = []
        
    for k in file_object_test:

        x_test_part, y_test_part = get_XY(file_object_test[k], num, 1, 1)
        
        predicted_all[varname][model_name][k] = preds_val[sum(lens):sum(lens) + len(x_test_part)]
        y_test_all[varname][model_name][k] = trues_val[sum(lens):sum(lens) + len(x_test_part)]
        lens.append(len(x_test_part))

    #mae, mse, rmse = metric_short(preds_val, trues_val)
    #print(varname, mae, mse, rmse)
    mae, mse, rmse = metric_short(preds_val, pd_file_val_transformed)
    #print(varname, mae, mse, rmse)

    final_train_MAE = mean_absolute_error(preds_val, pd_file_val_transformed)
    final_train_R2 = r2_score(preds_val, pd_file_val_transformed)
    final_train_RMSE = math.sqrt(mean_squared_error(preds_val, pd_file_val_transformed) / (max(all_mine_flat) - min(all_mine_flat)))
    print(varname, final_train_MAE, final_train_R2, final_train_RMSE)
    
if not os.path.isdir("UNITS_result/"):
    os.makedirs("UNITS_result/")

with open("UNITS_result/predicted_all", 'wb') as file_object:
    pickle.dump(predicted_all, file_object)  
    file_object.close()

with open("UNITS_result/y_test_all", 'wb') as file_object:
    pickle.dump(y_test_all, file_object)  
    file_object.close()

with open("UNITS_result/ws_all", 'wb') as file_object:
    pickle.dump(ws_all, file_object)  
    file_object.close()

    predicted_long = dict()
predicted_lat = dict()

actual_long = dict()
actual_lat = dict()
 
for model_name in predicted_all["speed"]:
    
    print(model_name)
    print(ws_all["longitude_no_abs"][model_name], ws_all["latitude_no_abs"][model_name])
 
    actual_long[model_name] = dict()
    actual_lat[model_name] = dict()

    for k in y_test_all["longitude_no_abs"][model_name]:
        print(model_name, k, "actual")
        actual_long[model_name][k] = [0]
        actual_lat[model_name][k] = [0]
        
        max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name][0], ws_all["latitude_no_abs"][model_name][0])
        long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name][0]
        lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name][0]
        range_long = len(y_test_all["longitude_no_abs"][model_name][k]) - long_offset
        range_lat = len(y_test_all["latitude_no_abs"][model_name][k]) - lat_offset
        min_range_long_lat = min(range_long, range_lat)

        for ix in range(min_range_long_lat):
            actual_long[model_name][k].append(actual_long[model_name][k][-1] + y_test_all["longitude_no_abs"][model_name][k][ix + long_offset])
            actual_lat[model_name][k].append(actual_lat[model_name][k][-1] + y_test_all["latitude_no_abs"][model_name][k][ix + lat_offset])

    predicted_long[model_name] = dict()
    predicted_lat[model_name] = dict()
        
    predicted_long[model_name]["long no abs"] = dict()
    predicted_lat[model_name]["lat no abs"] = dict()

    for k in predicted_all["longitude_no_abs"][model_name]:
        print(model_name, k, "long no abs")
        predicted_long[model_name]["long no abs"][k] = [0]
        predicted_lat[model_name]["lat no abs"][k] = [0]
        
        max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name][0], ws_all["latitude_no_abs"][model_name][0])
        long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name][0]
        lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name][0]
        range_long = len(y_test_all["longitude_no_abs"][model_name][k]) - long_offset
        range_lat = len(y_test_all["latitude_no_abs"][model_name][k]) - lat_offset
        min_range_long_lat = min(range_long, range_lat)

        for ix in range(min_range_long_lat):
            predicted_long[model_name]["long no abs"][k].append(predicted_long[model_name]["long no abs"][k][-1] + predicted_all["longitude_no_abs"][model_name][k][ix + long_offset])
            predicted_lat[model_name]["lat no abs"][k].append(predicted_lat[model_name]["lat no abs"][k][-1] + predicted_all["latitude_no_abs"][model_name][k][ix + lat_offset])

    predicted_long[model_name]["long speed dir"] = dict()
    predicted_lat[model_name]["lat speed dir"] = dict()

    for k in predicted_all["speed"][model_name]:
        print(model_name, k, "long speed dir")
        predicted_long[model_name]["long speed dir"][k] = [0]
        predicted_lat[model_name]["lat speed dir"][k] = [0]
    
        max_offset_speed_dir_time = max(max(ws_all["speed"][model_name][0], ws_all["direction"][model_name][0]), ws_all["time"][model_name][0])
        speed_offset_time = max_offset_speed_dir_time - ws_all["speed"][model_name][0]
        dir_offset_time = max_offset_speed_dir_time - ws_all["direction"][model_name][0]
        time_offset_time = max_offset_speed_dir_time - ws_all["time"][model_name][0]
        range_speed_time = len(y_test_all["speed"][model_name][k]) - speed_offset_time
        range_dir_time = len(y_test_all["direction"][model_name][k]) - dir_offset_time
        range_time_time = len(y_test_all["time"][model_name][k]) - time_offset_time
        min_range_speed_dir_time = min(min(range_speed_time, range_dir_time), range_time_time)

        for ix in range(min_range_speed_dir_time):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][k][ix + speed_offset_time] / 111 / 0.1 / 3600 * predicted_all["time"][model_name][k][ix + time_offset_time], change_angle(predicted_all["direction"][model_name][k][ix + dir_offset_time], k))
            predicted_long[model_name]["long speed dir"][k].append(predicted_long[model_name]["long speed dir"][k][-1] + new_long)
            predicted_lat[model_name]["lat speed dir"][k].append(predicted_lat[model_name]["lat speed dir"][k][-1] + new_lat)
            
    predicted_long[model_name]["long speed ones dir"] = dict()
    predicted_lat[model_name]["lat speed ones dir"] = dict()

    for k in predicted_all["speed"][model_name]:
        print(model_name, k, "long speed ones dir")
        predicted_long[model_name]["long speed ones dir"][k] = [0]
        predicted_lat[model_name]["lat speed ones dir"][k] = [0]
    
        max_offset_speed_dir = max(ws_all["speed"][model_name][0], ws_all["direction"][model_name][0])
        speed_offset = max_offset_speed_dir - ws_all["speed"][model_name][0]
        dir_offset = max_offset_speed_dir - ws_all["direction"][model_name][0]
        range_speed = len(y_test_all["speed"][model_name][k]) - speed_offset
        range_dir = len(y_test_all["direction"][model_name][k]) - dir_offset
        min_range_speed_dir = min(range_speed, range_dir)

        for ix in range(min_range_speed_dir):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][k][ix + speed_offset] / 111 / 0.1 / 3600, change_angle(predicted_all["direction"][model_name][k][ix + dir_offset], k))
            predicted_long[model_name]["long speed ones dir"][k].append(predicted_long[model_name]["long speed ones dir"][k][-1] + new_long)
            predicted_lat[model_name]["lat speed ones dir"][k].append(predicted_lat[model_name]["lat speed ones dir"][k][-1] + new_lat)

if not os.path.isdir("UNITS_result"):
    os.makedirs("UNITS_result")

with open("UNITS_result/actual_long", 'wb') as file_object:
    pickle.dump(actual_long, file_object)  
    file_object.close()
with open("UNITS_result/actual_lat", 'wb') as file_object:
    pickle.dump(actual_lat, file_object)  
    file_object.close()
with open("UNITS_result/predicted_long", 'wb') as file_object:
    pickle.dump(predicted_long, file_object)  
    file_object.close()
with open("UNITS_result/predicted_lat", 'wb') as file_object:
    pickle.dump(predicted_lat, file_object)  
    file_object.close()
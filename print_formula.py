
alllinestrain = ""
alllinestest = ""

for ws in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 19, 20, 25, 29, 30]:
    for varname in ["speed", "longitude_no_abs", "latitude_no_abs", "time", "direction"]:
        alllinestrain += "python run_pretrain.py --is_training 1 --model_id UniTS_zeroshot_pretrain_x64_mine_all_" + varname + "_" + str(ws) + "_train --model UniTS_zeroshot  --prompt_num 10 --patch_len 1  --stride 1   --e_layers 3  --d_model 64 --des 'Exp' --acc_it 128  --batch_size 32  --learning_rate 5e-5  --min_lr 1e-4  --weight_decay 5e-6  --train_epochs 10  --warmup_epochs 0  --min_keep_ratio 0.5  --right_prob 0.5  --min_mask_ratio 0.7  --max_mask_ratio 0.8  --debug online --task_data_config_path data_provider/" + str(ws) + "/multi_task_pretrain_val_" + varname + ".yaml\n"
        alllinestest += "python run.py --is_training 0  --model_id UniTS_zeroshot_pretrain_x64_mine_all_" + varname + "_" + str(ws) + "_test --model UniTS_zeroshot --prompt_num 10  --patch_len 1 --stride 1  --e_layers 3 --d_model 64   --des 'Exp' --debug online   --project_name zeroshot_newdata_mine_all_" + varname + "_" + str(ws) + "_test  --pretrained_weight checkpoints/ALL_task_UniTS_zeroshot_pretrain_x64_mine_all_" + varname + "_" + str(ws) + "_train_UniTS_zeroshot_All_ftM_dm64_el3_Exp_0/pretrain_checkpoint.pth  --task_data_config_path data_provider/" + str(ws) + "/zeroshot_task_" + varname + ".yaml\n"

print(alllinestrain)
print(alllinestest)

from utilities import load_object
import numpy as np
f1 = load_object("results/all_30_test/preds_direction")
f2 = load_object("results/all_direction_30_test/preds_direction")
print(np.shape(f1))
print(np.shape(f2))
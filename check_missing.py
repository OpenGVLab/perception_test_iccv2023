import os
import json

# full_list = os.listdir('/mnt/petrelfs/share_data/chenguo/pt_data/features/tal_umt_large_sthv2_feature_s4/')
# cur_list = os.listdir('/mnt/petrelfs/share_data/chenguo/pt_data/features/tal_umt_large_sthv2_perception_test_ft1_feature_s2')

full_json = '/mnt/petrelfs/yujiashuo/pt/ckpt/perception_tsl_multi_train_2023-09-21 00:06:34/eval_results.json'
cur_json = '/mnt/petrelfs/yujiashuo/pt/ckpt/perception_tsl_multi_train_2023-09-21 11:06:11/eval_results.json'
with open(cur_json, 'r') as f:
    cur_list = json.load(f)
with open(full_json, 'r') as f:
    full_list = json.load(f)
print(len(cur_list), len(full_list))
cur_key = cur_list.keys()
full_key = full_list.keys()
for full in full_key:
    if full not in cur_key:
        cur_list[full] = full_list[full]
with open('./eval_results.json', 'w') as f:
    json.dump(cur_list, f)
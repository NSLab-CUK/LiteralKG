import os

lr_list = [0.0001, 0.001, 0.01, 0.1]
batch_size_list = [2048]
dropout_list = [ 0.1, 0.5]
n_layer_list = [2, 4, 8]
n_dim_list = [16, 32]
cmd_dict = {}

count = 1

for i in batch_size_list:
    for j in n_dim_list:
        for k in dropout_list:
            for v in lr_list:
                for x in n_layer_list:
                    cmd_dict[f"python main.py --aggregation_type gcn --n_conv_layers {x} --lr {v} --mess_dropout {k} --conv_dim {j} --pre_training_batch_size {i} --fine_tuning_batch_size {i}"] = count
                    count += 1


for cmd in cmd_dict:
    print(f"{cmd_dict[cmd]} - {cmd}")
    os.system(cmd)


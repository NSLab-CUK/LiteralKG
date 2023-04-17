import torch
from dataloader2 import DataLoader
import pandas as pd
import random
from argument_test import parse_args
import os


def generate_test_data(train2id, test_head_dict, head_ids, tail_ids, neg_rate):
    tail_ids = list(tail_ids)

    approved_example = {}

    for idx, h_id in enumerate(head_ids):
        test_pos_tail_list = []
        test_neg_tail_list = []

        if h_id in test_head_dict:
            test_pos_tail_list = test_head_dict[h_id]

        while True:
            if len(test_neg_tail_list) == len(test_pos_tail_list)*neg_rate:
                break

            neg_tail_id = random.choice(list(tail_ids))
            if neg_tail_id not in test_pos_tail_list and neg_tail_id not in test_neg_tail_list:
                test_neg_tail_list.append(neg_tail_id)

        for tail_id in test_pos_tail_list:
            approved_example[f"{h_id}\t{tail_id}\t1"] = 1
        
        for tail_id in test_neg_tail_list:
            approved_example[f"{h_id}\t{tail_id}\t0"] = 0

    data = {
        "train2id": train2id,
        "test_data": approved_example,
    }

    result_path = f"outputs"
    try:
        os.mkdir(result_path)
    except:
        print("Folder has already existed!")

    for result_file in data:
        result_file_name = result_file + ".txt"

        print("Save results in ", result_file_name)

        obj_list = data[result_file]

        if result_file == "approved_example":
            f = open(os.path.join(result_path, result_file_name), "w")
            for key in obj_list:
                f.write(str(key) + "\n")
        else:
            f = open(os.path.join(result_path, result_file_name), "w")
            f.write(str(len(obj_list)) + "\n")
            for key in obj_list:
                f.write(str(key) + "\n")
        f.close()

def evaluate(train, head_dict, tail_list, device, neg_rate):
    head_ids = list(head_dict.keys())
    batch_size = len(head_ids)

    head_ids_batches = [head_ids[i: i + batch_size]
                        for i in range(0, len(head_ids), batch_size)]
    head_ids_batches = [torch.LongTensor(d) for d in head_ids_batches]

    tail_ids = torch.LongTensor(tail_list).to(device)

    for batch_head_ids in head_ids_batches:
        batch_head_ids = batch_head_ids.to(device)

        generate_test_data(train, head_dict, batch_head_ids.cpu().numpy(), tail_ids.cpu().numpy(), neg_rate)

def test_model(args):
    device = torch.device("cpu")

    # load data
    data = DataLoader(args)

    evaluate(data.train_data , data.test_head_dict, data.prediction_tail_ids, device, neg_rate=1)


def main():
    args = parse_args()
    test_model(args)


if __name__ == '__main__':
    main()

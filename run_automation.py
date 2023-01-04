import pandas as pd
import os
from utils.model_utils import *

def main():
    output_path = "outputs"
    filename = "evaluation_gcn_248.xlsx"
    file_path = f'{output_path}/{filename}'
    device = "cuda:0"

    data = pd.read_excel(file_path)

    for index, row in data.iterrows():

        if row["Best Pretrain"] != -1:
            print(f"Skipping pre-training - {index}")
        else:
            run_pretraining(file_path, index, device)


        if row["Best Finetune"] != -1:
            print(f"Skipping fine tuning - {index}")
        else:
            run_finetuning(file_path, index, device)


        if row["Accuracy"] != 0:
            print(f"Skipping testing - {index}")
        else:
            run_testing(file_path, index, device)


if __name__ == '__main__':
    main()

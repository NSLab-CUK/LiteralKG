import pandas as pd
from utils.model_utils import *


def main():
    output_path = "outputs/final"
    file_list = ["evaluation_bi-interaction_248.xlsx", "evaluation_gin_248.xlsx", "evaluation_graphsage_248.xlsx", "evaluation_graphsage_1.xlsx", "evaluation_bi-interaction_1.xlsx", "evaluation_gin_1.xlsx"]

    device = "cuda:0"

    for filename in file_list:
        file_path = f'{output_path}/{filename}'
        data = pd.read_excel(file_path)
        for index, row in data.iterrows():
            # if index > 2:
            #     continue

            # if row["Best Pretrain"] != -1:
            #     print(f"Skipping pre-training - {index}")
            # else:
            #     run_pretraining(file_path, index, device)


            # if row["Best Finetune"] != -1:
            #     print(f"Skipping fine tuning - {index}")
            # else:
            #     run_finetuning(file_path, index, device)


            if row["Accuracy"] != 0:
                print(f"Skipping testing - {index}")
            else:
                run_testing(file_path, index, device)


if __name__ == '__main__':
    main()

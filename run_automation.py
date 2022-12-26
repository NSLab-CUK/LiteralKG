import pandas as pd
import os

def run_pretraining(filename, index, device):
    data= pd.read_excel(filename)

    cmd = f"python main_pretraining.py --aggregation_type {data['Aggregator'][index]} --n_conv_layers {data['Number Layers'][index]} --lr {data['Learning Rate'][index]} --mess_dropout {data['Dropout'][index]} --conv_dim {data['Convolutional Dim'][index]} --pre_training_batch_size {data['Batch Size'][index]} --evaluation_row {index} --device {device} --evaluation_file {filename}"

    print(f"Running pre training- {index} - {cmd}")

    os.system(cmd)


def run_finetuning(filename, index, device):
    data = pd.read_excel(filename)

    cmd = f"python main_finetuning.py --aggregation_type {data['Aggregator'][index]} --n_conv_layers {data['Number Layers'][index]} --lr {data['Learning Rate'][index]} --mess_dropout {data['Dropout'][index]} --conv_dim {data['Convolutional Dim'][index]} --fine_tuning_batch_size {data['Batch Size'][index]} --pretrain_epoch {int(data['Best Pretrain'][index])} --evaluation_row {index} --device {device} --evaluation_file {filename}"

    print(f"Running fine tuning - {index} - {cmd}")

    os.system(cmd)


def run_testing(filename, index, device):
    data = pd.read_excel(filename)

    cmd = f"python test.py --aggregation_type {data['Aggregator'][index]} --n_conv_layers {data['Number Layers'][index]} --lr {data['Learning Rate'][index]} --mess_dropout {data['Dropout'][index]} --conv_dim {data['Convolutional Dim'][index]} --model_epoch {data['Best Finetune'][index]} --evaluation_row {index} --device {device} --evaluation_file {filename}"

    print(f"Running test - {index} - {cmd}")

    os.system(cmd)


def main():
    output_path = "outputs"
    filename = "evaluation.xlsx"
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

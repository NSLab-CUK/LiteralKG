import pandas as pd
import os

def run_pretraining(data, index):
    cmd = f"python main_pretraining.py --aggregation_type {data['Aggregator'][index]} --n_conv_layers {data['Number Layers'][index]} --lr {data['Learning Rate'][index]} --mess_dropout {data['Dropout'][index]} --conv_dim {data['Convolutional Dim'][index]} --pre_training_batch_size {data['Batch Size'][index]} --evaluation_row {index}"
    
    print(f"Running pre training- {index} - {cmd}")

    os.system(cmd)

def run_finetuning(data, index):
    cmd = f"python main_finetuning.py --aggregation_type {data['Aggregator'][index]} --n_conv_layers {data['Number Layers'][index]} --lr {data['Learning Rate'][index]} --mess_dropout {data['Dropout'][index]} --conv_dim {data['Convolutional Dim'][index]} --fine_tuning_batch_size {data['Batch Size'][index]} --pretrain_epoch {int(data['Best Pretrain'][index])} --evaluation_row {index}"
        
    print(f"Running fine tuning - {index} - {cmd}")

    os.system(cmd)


def run_testing(data, index):
    cmd = f"python test.py --aggregation_type {data['Aggregator'][index]} --n_conv_layers {data['Number Layers'][index]} --lr {data['Learning Rate'][index]} --mess_dropout {data['Dropout'][index]} --conv_dim {data['Convolutional Dim'][index]} --model_epoch {data['Best Finetune'][index]} --evaluation_row {index}"
    
    print(f"Running test - {index} - {cmd}")

    os.system(cmd)


def main():
    output_path = "outputs"
    filename = "evaluation.xlsx"
    
    data = pd.read_excel(f'{output_path}/{filename}')

    for index, row in data.iterrows():

        data_updated = pd.read_excel(f'{output_path}/{filename}')

        if row["Best Pretrain"] != -1:
            print(f"Skipping pre-training - {index}")
        else:
            run_pretraining(data_updated, index)
        
        data_updated = pd.read_excel(f'{output_path}/{filename}')

        if row["Best Finetune"] != -1:
            print(f"Skipping fine tuning - {index}")
        else:
            run_finetuning(data_updated, index)

        data_updated = pd.read_excel(f'{output_path}/{filename}')

        if row["Accuracy"] != 0:
            print(f"Skipping testing - {index}")
        else:
            run_testing(data_updated, index)

if __name__ == '__main__':
    main()


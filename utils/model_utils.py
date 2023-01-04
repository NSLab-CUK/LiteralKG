import os
import torch
from utils.metric_utils import *
from tqdm import tqdm
import pandas as pd
import platform


def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None, name="training"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(
        model_dir, '{}_model_epoch{}.pth'.format(name, current_epoch))
    torch.save({'model_state_dict': model.state_dict(),
               'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(
            model_dir, '{}_model_epoch{}.pth'.format(name, last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.remove(old_model_state_file)


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate(model, head_dict, batch_size, tail_list, device, neg_rate):
    model.eval()
    head_ids = list(head_dict.keys())

    head_ids_batches = [head_ids[i: i + batch_size]
                        for i in range(0, len(head_ids), batch_size)]
    head_ids_batches = [torch.LongTensor(d) for d in head_ids_batches]

    tail_ids = torch.LongTensor(tail_list).to(device)

    prediction_scores = []
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metrics_dict = {m: [] for m in metric_names}

    with tqdm(total=len(head_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_head_ids in head_ids_batches:
            batch_head_ids = batch_head_ids.to(device)

            with torch.no_grad():
                # (n_batch_heads, n_tails)
                batch_scores = model(batch_head_ids, tail_ids, device=device, mode='predict')

            batch_scores = batch_scores.cpu()

            batch_metrics = calc_metrics(
                batch_scores, head_dict, batch_head_ids.cpu().numpy(), tail_ids.cpu().numpy(), neg_rate)

            # prediction_scores.append(batch_scores.numpy())
            for m in metric_names:
                metrics_dict[m].append(batch_metrics[m])
            pbar.update(1)
            torch.cuda.empty_cache()

    # prediction_scores = np.concatenate(prediction_scores, axis=0)
    for m in metric_names:
        metrics_dict[m] = np.array(metrics_dict[m]).mean()
    return prediction_scores, metrics_dict

def update_evaluation_value(file_path, colume, row, value):
    df = pd.read_excel(file_path)

    df[colume][row] = value

    df.to_excel(file_path, sheet_name='data', index=False)

def run_pretraining(filename, index, device):
    data= pd.read_excel(filename)

    cmd = f"python main_pretraining.py --aggregation_type {data['Aggregator'][index]} --n_conv_layers {data['Number Layers'][index]} --lr {data['Learning Rate'][index]} --mess_dropout {data['Dropout'][index]} --conv_dim {data['Convolutional Dim'][index]} --pre_training_batch_size {data['Batch Size'][index]} --fine_tuning_batch_size {data['Batch Size'][index]} --evaluation_row {index} --device {device} --evaluation_file {filename}"

    print(f"Running pre training- {index} - {cmd}")

    os.system(cmd)


def run_finetuning(filename, index, device):
    data = pd.read_excel(filename)

    cmd = f"python main_finetuning.py --aggregation_type {data['Aggregator'][index]} --n_conv_layers {data['Number Layers'][index]} --lr {data['Learning Rate'][index]} --mess_dropout {data['Dropout'][index]} --conv_dim {data['Convolutional Dim'][index]} --pre_training_batch_size {data['Batch Size'][index]} --fine_tuning_batch_size {data['Batch Size'][index]} --pretrain_epoch {int(data['Best Pretrain'][index])} --evaluation_row {index} --device {device} --evaluation_file {filename}"

    print(f"Running fine tuning - {index} - {cmd}")

    os.system(cmd)


def run_testing(filename, index, device):
    data = pd.read_excel(filename)

    cmd = f"python test.py --aggregation_type {data['Aggregator'][index]} --n_conv_layers {data['Number Layers'][index]} --lr {data['Learning Rate'][index]} --mess_dropout {data['Dropout'][index]} --conv_dim {data['Convolutional Dim'][index]} --pre_training_batch_size {data['Batch Size'][index]} --fine_tuning_batch_size {data['Batch Size'][index]} --model_epoch {data['Best Finetune'][index]} --evaluation_row {index} --device {device} --evaluation_file {filename}"

    print(f"Running test - {index} - {cmd}")

    os.system(cmd)


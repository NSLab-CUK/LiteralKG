import torch
import numpy as np
import random
from dataloader_bce import DataLoader
import torch.optim as optim
from tqdm import tqdm
from time import time
from model_bce import LiteralKG
import sys
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
from argument_finetuning import parse_args

from utils.log_utils import *
from utils.metric_utils import *
from utils.model_utils import *

def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(
        log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)

    # load data
    data = DataLoader(args, logging)
    torch.cuda.empty_cache()

    # construct model & optimizer
    model = LiteralKG(args, data.n_entities,
                      data.n_relations, data.A_in, data.num_embedding_table, data.text_embedding_table)

    logging.info(model)
    torch.autograd.set_detect_anomaly(True)

    fine_tuning_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pytorch_total_params = sum(p.numel()

                               for p in model.parameters() if p.requires_grad)
    print("Total parameters: {}".format(pytorch_total_params))

    writer = SummaryWriter(
        comment=f"_{args.aggregation_type}_{args.data_name}_lr{args.lr}_dropout{args.mess_dropout}-embed-dim{args.embed_dim}_relation-dim{args.relation_dim}_n-layers{args.n_conv_layers}_gat{args.scale_gat_dim}_conv{args.conv_dim}_bs{args.pre_training_batch_size}_num-dim{args.use_num_lit}_txt-dim{args.use_txt_lit}_fine_tuning_bce")

    logging.info("----- USE PRE-TRAINING MODEL -----")
    model = load_model(model, args.pretrain_model_path)

    ft_loss_list, ft_time_training = fine_tuning_train(model, data, fine_tuning_optimizer, device, args, writer)

    logging.info("FINALLY -------")

    logging.info("Fine tuning loss list {}".format(ft_loss_list))
    logging.info("Fine tuning time training {}".format(ft_time_training))


def fine_tuning_train(model, data, optimizer, device, args, writer):
    logging.info("-----Fine-turning model-----")

    if args.use_parallel_gpu:
        model = nn.DataParallel(model, device_ids=[2, 3])
        model.to(device)
    else:
        print("Device {}".format(device))
        model.to(device)
    # initialize metrics
    best_epoch_val = -1
    best_recall = 0

    # Ks = eval(args.Ks)
    # k_min = min(Ks)
    # k_max = max(Ks)

    epoch_list = []
    metrics_list = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    criterion = nn.BCELoss()

    # train
    ft_loss_list = []
    ft_time_training = []

    # Fine-tuning model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # train prediction
        prediction_total_loss = 0
        n_prediction_batch = len(data.train_data_heads) // data.fine_tuning_batch_size + 1

        prediction_batch_heads = torch.split(data.train_data_heads, data.fine_tuning_batch_size)
        prediction_batch_tails = torch.split(data.train_data_tails, data.fine_tuning_batch_size)
        prediction_batch_labels = torch.split(data.train_data_labels, data.fine_tuning_batch_size)

        for iter in tqdm(range(1, n_prediction_batch + 1), desc=f"EP:{epoch}_train"):
            time1 = time()
            idx = iter - 1
            prediction_batch_head = prediction_batch_heads[idx]
            prediction_batch_tail = prediction_batch_tails[idx]
            prediction_batch_label = prediction_batch_labels[idx]
            prediction_batch_label = prediction_batch_label.to(device)
            optimizer.zero_grad()

            #calculate output
            outputs = model(prediction_batch_head, prediction_batch_tail, device=device, mode='mlp').reshape(-1)

            #calculate loss
            prediction_batch_loss = criterion(outputs, prediction_batch_label)

            if np.isnan(prediction_batch_loss.cpu().detach().numpy()):
                logging.info(
                    'ERROR (Fine Tuning Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter,
                                                                                                          n_prediction_batch))
                sys.exit()

            prediction_batch_loss.backward()
            optimizer.step()

            prediction_total_loss += prediction_batch_loss.item()

            if iter % 50 == 0:
                torch.cuda.empty_cache()

            if (iter % args.fine_tuning_print_every) == 0:
                logging.info(
                    'Fine Tuning Training: Epoch {:04d}/{:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, args.n_epoch, iter, n_prediction_batch, time() - time1, prediction_batch_loss.item(),
                                                                       prediction_total_loss / iter))
        logging.info(
            'Fine Tuning Training: Epoch {:04d}/{:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
                epoch, args.n_epoch, n_prediction_batch, time() - time0, prediction_total_loss / n_prediction_batch))

        prediction_loss_value = prediction_total_loss / n_prediction_batch
        # if min_loss > prediction_loss_value:
        #     min_loss = prediction_loss_value
        #     save_model(bce_model, args.save_dir, epoch, best_epoch, name="fine-tuning")
        #     logging.info('Save pre-training model on epoch {:04d}!'.format(epoch))
        #     best_epoch = epoch

        ft_loss_list.append(prediction_loss_value)
        writer.add_scalar('Prediction Loss/train', prediction_loss_value, epoch)
        ft_time_training.append(time() - time0)

        torch.cuda.empty_cache()

        # evaluate prediction layer
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time2 = time()
            val_heads = data.val_data_heads
            val_tails = data.val_data_tails
            val_labels = data.val_data_labels

            _, metrics_dict = evaluate(model, val_heads, val_tails, val_labels, device)

            metrics_str = 'Fine Tuning Evaluation: Epoch {:04d} | Total Time {:.1f}s | Accuracy [{:.4f}], Precision [{:.4f}], Recall [{:.4f}], F1 [{:.4f}]'.format(
                epoch, time() - time2, metrics_dict['accuracy'], metrics_dict['precision'], metrics_dict['recall'],
                metrics_dict['f1'])

            writer.add_scalar('Accuracy Plot', metrics_dict['accuracy'], epoch)
            writer.add_scalar('Precision Plot', metrics_dict['precision'], epoch)
            writer.add_scalar('Recall Plot', metrics_dict['recall'], epoch)
            writer.add_scalar('F1 Score Plot', metrics_dict['f1'], epoch)

            logging.info(metrics_str)
            temp_metrics_df = pd.DataFrame(data=[{"metrics": metrics_str}])
            temp_metrics_df.to_csv(
                args.save_dir + '/metrics_{}.tsv'.format(epoch), sep='\t', index=False)

            epoch_list.append(epoch)
            for m in ['accuracy', 'precision', 'recall', 'f1']:
                metrics_list[m].append(metrics_dict[m])
            best_recall, should_stop = early_stopping(
                metrics_list['f1'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list['f1'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch_val,name="training")
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch_val = epoch

        # Logging every epoch
        logging.info("Fine tuning loss list {}".format(ft_loss_list))
        logging.info("Fine tuning time {}".format(ft_time_training))
    update_evaluation_value(args.evaluation_file, "Best Finetune", args.evaluation_row, best_epoch_val)
    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for m in ['accuracy', 'precision', 'recall', 'f1']:
        metrics_df.append(metrics_list[m])
        metrics_cols.append('{}'.format(m))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx']
                                  == best_epoch_val].iloc[0].to_dict()
    logging.info(
        'Best Prediction Layer Evaluation: Epoch {:04d} | Accuracy [{:.4f}], Precision [{:.4f}], Recall [{:.4f}], F1_Score [{:.4f}]'.format(
            int(best_metrics['epoch_idx']), best_metrics['accuracy'], best_metrics['precision'], best_metrics['recall'],
            best_metrics['f1']))

    return ft_loss_list, ft_time_training

def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()

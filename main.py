import torch
import numpy as np
import random
from dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm
from time import time
from time import sleep
from model import LiteralKG
import sys
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
from argument import parse_args

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

    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    logging.info(model)
    torch.autograd.set_detect_anomaly(True)

    fine_tuning_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    pre_training_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pytorch_total_params = sum(p.numel()

                               for p in model.parameters() if p.requires_grad)
    print("Total parameters: {}".format(pytorch_total_params))

    writer = SummaryWriter(
        comment=f"_{args.aggregation_type}_{args.data_name}_lr{args.lr}_dropout{args.mess_dropout}-embed-dim{args.embed_dim}_relation-dim{args.relation_dim}_n-layers{args.n_conv_layers}_gat{args.scale_gat_dim}_num-dim{args.use_num_lit}_txt-dim{args.use_txt_lit}")

    pt_loss_list = None

    if args.use_pretrain == 1:
        logging.info("----- USE PRE-TRAINING MODEL -----")
        model = load_model(model, args.pretrain_model_path)
    else:
        pt_loss_list, pt_time_training = pre_training_train(model, data, pre_training_optimizer, device, args, writer)

    ft_loss_list, ft_time_training = fine_tuning_train(model, data, fine_tuning_optimizer, device, args, writer)

    logging.info("FINALLY -------")
    if pt_loss_list is not None:
        logging.info("Pre-training loss list {}".format(pt_loss_list))
        logging.info("Pre training time training {}".format(pt_time_training))
    logging.info("Fine tuning loss list {}".format(ft_loss_list))
    logging.info("Fine tuning time training {}".format(ft_time_training))


def pre_training_train(model, data, optimizer, device, args, writer):
    logging.info("-----Pre-training model-----")
    if args.use_parallel_gpu:
        model = nn.DataParallel(model, device_ids=[2, 3])
        model.to(device)
    else:
        print("Device {}".format(device))
        model.to(device)
    # initialize metrics
    best_epoch = -1

    # train
    pt_loss_list = []

    pt_time_training = []

    min_loss = 100000

    # Pre-training model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()
        # pre training
        kg_total_loss = 0

        # Sampling data for each epoch
        n_data_samples = int(len(list(data.train_kg_dict)) * args.epoch_data_rate)
        epoch_sampling_data_list = random.sample(list(data.train_kg_dict), n_data_samples)
        epoch_sampling_data_dict = {k: data.train_kg_dict[k] for k in epoch_sampling_data_list}
        n_kg_batch = n_data_samples // data.pre_training_batch_size + 1

        for iter in tqdm(range(1, n_kg_batch + 1), desc=f"EP:{epoch}_train"):
            time1 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(
                epoch_sampling_data_dict, data.pre_training_batch_size, list(data.training_tails))
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            optimizer.zero_grad()
            kg_batch_loss = model(kg_batch_head, kg_batch_relation,
                                  kg_batch_pos_tail, kg_batch_neg_tail, device=device, mode='pre_training')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info(
                    'ERROR (Pre-training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter,
                                                                                                  n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            optimizer.step()
            kg_total_loss += kg_batch_loss.item()

            if iter % 50 == 0:
                torch.cuda.empty_cache()

            loss_value = kg_total_loss / n_kg_batch

            if (iter % args.kg_print_every) == 0:
                logging.info(
                    'Pre-training: Epoch {:04d}/{:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, args.n_epoch, iter, n_kg_batch, time() - time1, kg_batch_loss.item(),
                                                               kg_total_loss / iter))

        # update attention
        time2 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, device=device, mode='update_att')
        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(
            epoch, time() - time2))

        logging.info(
            'Pre-training: Epoch {:04d}/{:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
                epoch, args.n_epoch, n_kg_batch, time() - time0, loss_value))

        pt_loss_list.append(loss_value)
        pt_time_training.append(time() - time0)

        writer.add_scalar('Triplet Loss/train', loss_value, epoch)

        if min_loss > loss_value:
            min_loss = loss_value
            save_model(model, args.save_dir, epoch, best_epoch, name="pre-training")
            logging.info('Save pre-training model on epoch {:04d}!'.format(epoch))
            best_epoch = epoch

        torch.cuda.empty_cache()

        # Logging every epoch
        logging.info("Loss pre-training list {}".format(pt_loss_list))
        logging.info("Pre-training time {}".format(pt_time_training))

    return pt_loss_list, pt_time_training


def fine_tuning_train(model, data, optimizer, device, args, writer):
    logging.info("-----Fine-turning model-----")
    if args.use_parallel_gpu:
        model = nn.DataParallel(model, device_ids=[2, 3])
        model.to(device)
    else:
        print("Device {}".format(device))
        model.to(device)
    # initialize metrics
    best_epoch = -1
    best_epoch_val = -1
    best_recall = 0

    # Ks = eval(args.Ks)
    # k_min = min(Ks)
    # k_max = max(Ks)

    epoch_list = []
    metrics_list = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # train
    ft_loss_list = []
    ft_time_training = []

    min_loss = 100000

    # Fine-tuning model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # train prediction
        prediction_total_loss = 0
        n_prediction_batch = data.n_prediction_training // data.fine_tuning_batch_size + 1

        for iter in tqdm(range(1, n_prediction_batch + 1), desc=f"EP:{epoch}_train"):
            time1 = time()
            prediction_batch_head, prediction_batch_pos_tail, prediction_batch_neg_tail = data.generate_prediction_batch(
                data.train_head_dict, data.fine_tuning_batch_size)
            prediction_batch_head = prediction_batch_head.to(device)
            prediction_batch_pos_tail = prediction_batch_pos_tail.to(device)
            prediction_batch_neg_tail = prediction_batch_neg_tail.to(device)

            prediction_batch_loss = model(
                prediction_batch_head, prediction_batch_pos_tail, prediction_batch_neg_tail, device=device,
                mode='fine_tuning')

            if np.isnan(prediction_batch_loss.cpu().detach().numpy()):
                logging.info(
                    'ERROR (Fine Tuning Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter,
                                                                                                          n_prediction_batch))
                sys.exit()

            prediction_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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
        if min_loss > prediction_loss_value:
            min_loss = prediction_loss_value
            save_model(model, args.save_dir, epoch, best_epoch, name="fine-tuning")
            logging.info('Save pre-training model on epoch {:04d}!'.format(epoch))
            best_epoch = epoch

        ft_loss_list.append(prediction_loss_value)
        writer.add_scalar('Prediction Loss/train', prediction_loss_value, epoch)
        ft_time_training.append(time() - time0)

        torch.cuda.empty_cache()

        # evaluate prediction layer
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time2 = time()
            _, metrics_dict = evaluate(model, data.val_head_dict, data.test_batch_size, data.prediction_tail_ids, device, neg_rate=args.test_neg_rate)

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
                metrics_list['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch_val)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch_val = epoch

        # Logging every epoch
        logging.info("Fine tuning loss list {}".format(ft_loss_list))
        logging.info("Fine tuning time {}".format(ft_time_training))

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
                                  == best_epoch].iloc[0].to_dict()
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

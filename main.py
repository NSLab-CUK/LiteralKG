import torch
import numpy as np
import random
from dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm
from time import time
from model import LiteralKG
import sys
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
from argument import parse_args


from utils.log_utils import *
from utils.metric_utils import *
from utils.model_utils import *


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_head_dict = dataloader.train_head_dict
    test_head_dict = dataloader.test_head_dict

    model.eval()
    head_ids = list(test_head_dict.keys())

    head_ids_batches = [head_ids[i: i + test_batch_size]
                        for i in range(0, len(head_ids), test_batch_size)]
    head_ids_batches = [torch.LongTensor(d) for d in head_ids_batches]

    n_tails = dataloader.n_tails
    tail_ids = torch.LongTensor(dataloader.prediction_tail_ids).to(device)

    prediction_scores = []
    metric_names = ['precision', 'recall']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(head_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_head_ids in head_ids_batches:
            batch_head_ids = batch_head_ids.to(device)

            with torch.no_grad():
                # (n_batch_heads, n_tails)
                batch_scores = model(batch_head_ids, tail_ids, mode='predict')

            batch_scores = batch_scores.cpu()

            batch_metrics = calc_metrics_at_k(
                batch_scores, train_head_dict, test_head_dict, batch_head_ids.cpu().numpy(), tail_ids.cpu().numpy(), Ks)

            # prediction_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)
            torch.cuda.empty_cache()

    # prediction_scores = np.concatenate(prediction_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return prediction_scores, metrics_dict

def seperate_train_data(dataset):
    return dataset

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
    device = args.device

    # load data
    data = DataLoader(args, logging)

    # construct model & optimizer
    model = LiteralKG(args, data.n_entities,
                 data.n_relations, data.A_in, data.num_embedding_table, data.text_embedding_table)

    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    if args.use_parallel_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
        model = nn.DataParallel(model, output_device=1)
    else:
        print("Device {}".format(device))
        model.to(device)

    logging.info(model)
    torch.autograd.set_detect_anomaly(True)

    fine_tuning_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    pre_training_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Total parameters: {}".format(pytorch_total_params))


    writer = SummaryWriter()

    pt_loss_list, pt_time_training = pre_training_train(model, data, pre_training_optimizer, device, args, writer)

    ft_loss_list, ft_time_training = fine_tuning_train(model, data, fine_tuning_optimizer, device, args, writer)

    logging.info("FINALLL -------")
    # Logging every epoch
    logging.info("Pre-training loss list {}".format(pt_loss_list))
    logging.info("Pre training time training {}".format(pt_time_training))
    logging.info("Fine tuning loss list {}".format(ft_loss_list))
    logging.info("Fine tuning time training {}".format(ft_time_training))

def pre_training_train(model, data, optimizer, device, args, writer):
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
        epoch_sampling_data_dict = { k: data.train_kg_dict[k] for k in epoch_sampling_data_list}
        n_kg_batch = n_data_samples // data.pre_training_batch_size + 1

        for iter in tqdm(range(1, n_kg_batch + 1), desc=f"EP:{epoch}_train"):
            time1 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(
                epoch_sampling_data_dict, data.pre_training_batch_size, data.n_head_tail)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)
            
            optimizer.zero_grad()
            kg_batch_loss = model(kg_batch_head, kg_batch_relation,
                                  kg_batch_pos_tail, kg_batch_neg_tail, mode='pre_training')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info(
                    'ERROR (Pre-training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            optimizer.step()
            kg_total_loss += kg_batch_loss.item()

            if iter % 50 == 0:
                torch.cuda.empty_cache()

            loss_value = kg_total_loss / n_kg_batch

            if (iter % args.kg_print_every) == 0:
                logging.info('Pre-training: Epoch {:04d}/{:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                    epoch, args.n_epoch, iter, n_kg_batch, time() - time1, kg_batch_loss.item(), kg_total_loss / iter))

        # update attention
        time2 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')
        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(
            epoch, time() - time2))

        logging.info('Pre-training: Epoch {:04d}/{:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
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
    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': []} for k in Ks}

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

        for iter in range(1, n_prediction_batch + 1):
            time1 = time()
            prediction_batch_head, prediction_batch_pos_tail, prediction_batch_neg_tail = data.generate_prediction_batch(
                data.train_head_dict, data.fine_tuning_batch_size)
            prediction_batch_head = prediction_batch_head.to(device)
            prediction_batch_pos_tail = prediction_batch_pos_tail.to(device)
            prediction_batch_neg_tail = prediction_batch_neg_tail.to(device)

            prediction_batch_loss = model(
                prediction_batch_head, prediction_batch_pos_tail, prediction_batch_neg_tail, mode='fine_tuning')

            if np.isnan(prediction_batch_loss.cpu().detach().numpy()):
                logging.info(
                    'ERROR (Fine Tuning Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_prediction_batch))
                sys.exit()

            prediction_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            prediction_total_loss += prediction_batch_loss.item()

            if iter % 50 == 0:
                torch.cuda.empty_cache()

            if (iter % args.fine_tuning_print_every) == 0:
                logging.info('Fine Tuning Training: Epoch {:04d}/{:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                    epoch, args.n_epoch, iter, n_prediction_batch, time() - time1, prediction_batch_loss.item(), prediction_total_loss / iter))
        logging.info('Fine Tuning Training: Epoch {:04d}/{:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
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
            _, metrics_dict = evaluate(model, data, Ks, device)

            metrics_str = 'Fine Tuning Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}]'.format(
                epoch, time() - time2, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'])

            logging.info(metrics_str)
            temp_metrics_df = pd.DataFrame(data=[{"metrics": metrics_str}])
            temp_metrics_df.to_csv(
                args.save_dir + '/metrics_{}.tsv'.format(epoch), sep='\t', index=False)

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(
                metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

        # Logging every epoch
        logging.info("Fine tuning loss list {}".format(ft_loss_list))
        logging.info("Fine tuning time {}".format(ft_time_training))


    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx']
                                  == best_epoch].iloc[0].to_dict()
    logging.info('Best Prediction Layer Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}]'.format(
        int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)]))

    return ft_loss_list, ft_time_training

def predict(args):
    # GPU / CPU
    device = args.device

    # load data
    data = DataLoader(args, logging)

    # load model
    model = LiteralKG(args, data.n_entities, data.n_relations, data.numeric_embed, data.text_embed)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    prediction_scores, metrics_dict = evaluate(model, data, Ks, device)
    np.save(args.save_dir + 'prediction_scores.npy', prediction_scores)
    print('Fine Tuning Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall']))

def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()

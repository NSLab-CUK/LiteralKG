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
from argument_pretraining import parse_args

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

    pre_training_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pytorch_total_params = sum(p.numel()

                               for p in model.parameters() if p.requires_grad)
    print("Total parameters: {}".format(pytorch_total_params))

    writer = SummaryWriter(
        comment=f"_{args.aggregation_type}_{args.data_name}_lr{args.lr}_dropout{args.mess_dropout}-embed-dim{args.embed_dim}_relation-dim{args.relation_dim}_n-layers{args.n_conv_layers}_gat{args.scale_gat_dim}_conv{args.conv_dim}_bs{args.pre_training_batch_size}_num-dim{args.use_num_lit}_txt-dim{args.use_txt_lit}_pre_training")

    pt_loss_list = None

    pt_loss_list, pt_time_training = pre_training_train(model, data, pre_training_optimizer, device, args, writer)

    logging.info("FINALLY -------")
    if pt_loss_list is not None:
        logging.info("Pre-training loss list {}".format(pt_loss_list))
        logging.info("Pre training time training {}".format(pt_time_training))


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

    update_evaluation_value(args.evaluation_file, "Best Pretrain", args.evaluation_row, best_epoch)

    return pt_loss_list, pt_time_training

def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()

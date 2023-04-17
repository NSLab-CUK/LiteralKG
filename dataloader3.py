import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import random
import time
# from openke.module.model import TransR


class DataLoaderBase(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir
        self.device = args.device

        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'prediction_train.txt')
        self.test_file = os.path.join(self.data_dir, 'prediction_test.txt')
        self.val_file = os.path.join(self.data_dir, 'prediction_val.txt')

        self.prediction_dict_file = args.prediction_dict_file
        self.prediction_tail_ids = self.load_prediction_id_list()

        self.entity_dim = args.embed_dim

        self.relation_dim = args.relation_dim

        self.pre_training_neg_rate = args.pre_training_neg_rate
        self.fine_tuning_neg_rate = args.fine_tuning_neg_rate
        
        self.test_data_heads, self.test_data_tails, self.test_data_labels = self.load_prediction_data(self.test_file)
        self.analize_prediction()


    def load_prediction_id_list(self):
        file = open(os.path.join(
            self.data_dir, self.prediction_dict_file), 'rb')

        # dump information to that file
        data = pickle.load(file)

        return list(data)

    def load_prediction_data(self, filename):
        heads = []
        tails = []
        labels = []

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split("\t")]

            if len(inter) > 1:
                heads.append(inter[0])
                tails.append(inter[1])
                labels.append(inter[2])

        head_tensors = torch.LongTensor(heads)
        tail_tensors = torch.LongTensor(tails)
        label_tensors = torch.LongTensor(labels)

        return head_tensors, tail_tensors, label_tensors

    def analize_prediction(self):
        self.n_heads = max(max(self.prediction_train_data[0]), max(
            self.prediction_test_data[0])) + 1
        self.n_tails = max(max(self.prediction_train_data[1]), max(
            self.prediction_test_data[1])) + 1

        self.n_prediction_training = len(self.prediction_train_data[0])
        self.n_prediction_testing = len(self.prediction_test_data[0])

    def load_graph(self, filename):
        graph_data = pd.read_csv(filename, sep=' ', names=[
                              'h', 'r', 't'], engine='python')
        graph_data = graph_data.drop_duplicates()
        return graph_data

    def sample_pos_tails_for_head(self, head_dict, head_id, n_sample_pos_tails):
        pos_tails = head_dict[head_id]
        n_pos_tails = len(pos_tails)

        sample_pos_tails = []
        while True:
            if len(sample_pos_tails) == n_sample_pos_tails:
                break

            pos_tail_idx = np.random.randint(
                low=0, high=n_pos_tails, size=1)[0]
            pos_tail_id = pos_tails[pos_tail_idx]
            if pos_tail_id not in sample_pos_tails:
                sample_pos_tails.append(pos_tail_id)
        return sample_pos_tails

    def sample_neg_tails_for_head(self, head_dict, head_id, n_sample_neg_tails):
        pos_tails = head_dict[head_id]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_tails:
                break

            neg_tail_id = random.choice(list(self.prediction_tail_ids))
            if neg_tail_id not in pos_tails and neg_tail_id not in sample_neg_tails:
                sample_neg_tails.append(neg_tail_id)
        return sample_neg_tails

    def generate_prediction_batch(self, head_dict, batch_size):
        exist_heads = list(head_dict)

        batch_size = int(batch_size / self.fine_tuning_neg_rate)
        
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads)
                          for _ in range(batch_size)]

        batch_pos_tail, batch_neg_tail = [], []

        for u in batch_head:
            # Generate the positive samples for prediction
            batch_pos_tail += self.sample_pos_tails_for_head(head_dict, u, 1)

            # Generate the negative samples for prediction
            batch_neg_tail += self.sample_neg_tails_for_head(head_dict, u, self.fine_tuning_neg_rate)

        batch_head = self.generate_batch_by_neg_rate(batch_head, self.fine_tuning_neg_rate)
        batch_pos_tail = self.generate_batch_by_neg_rate(batch_pos_tail, self.fine_tuning_neg_rate)

        batch_head = torch.LongTensor(batch_head)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_pos_tail, batch_neg_tail

    def sample_pos_triples_for_head(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(
                low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_head(self, kg_dict, head, relation, n_sample_neg_triples, training_tails):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break
            try:
                tail = random.choice(training_tails)
            except:
                continue
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, kg_dict, batch_size, training_tails):
        exist_heads = kg_dict.keys()
        batch_size = int(batch_size / self.pre_training_neg_rate)

        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads)
                          for _ in range(batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []

        for h in batch_head:
            # Generate the positive samples
            relation, pos_tail = self.sample_pos_triples_for_head(
                kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            # Generate the negative samples
            neg_tail = self.sample_neg_triples_for_head(
                kg_dict, h, relation[0], self.pre_training_neg_rate, training_tails)

            batch_neg_tail += neg_tail

        batch_head = self.generate_batch_by_neg_rate(batch_head, self.pre_training_neg_rate)
        batch_relation = self.generate_batch_by_neg_rate(batch_relation, self.pre_training_neg_rate)
        batch_pos_tail = self.generate_batch_by_neg_rate(batch_pos_tail, self.pre_training_neg_rate)

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

    def generate_batch_by_neg_rate(self, batch, rate):
        zip_list = []
        results = []

        for i in range(rate):
            zip_list.append(batch)

        zip_list = list(zip(*zip_list))

        for x in zip_list:
            results += list(x)

        return results

class DataLoader(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.fine_tuning_batch_size = int(args.batch_size /  self.fine_tuning_neg_rate)
        self.pre_training_batch_size = int(args.batch_size / self.pre_training_neg_rate)
        self.test_batch_size = args.test_batch_size

        graph_data = self.load_graph(self.kg_file)
        self.construct_data(graph_data)
        self.training_tails = graph_data['t']
        self.print_info(logging)


    def construct_data(self, graph_data):
        # Removed addition of inverse

        # re-map head id
        # graph_data['r'] += 2
        self.n_relations = len(set(graph_data['r']))

        # add interactions to kg data

        self.pre_train_data = graph_data
        self.n_pre_training = len(self.pre_train_data)

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.pre_train_data.iterrows():
            h, r, t = row[1]

            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.n_heads = max(max(h_list) + 1, self.n_heads)
        self.n_tails = max(max(t_list) + 1, self.n_tails)

        self.n_entities = max(self.n_heads, self.n_tails)

        self.n_head_tail = self.n_entities

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def print_info(self, logging):
        logging.info('Total training heads:           %d' % self.n_heads)
        logging.info('Total training tails:           %d' % self.n_tails)
        logging.info('Total entities:        %d' % self.n_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_prediction_training:        %d' % self.n_prediction_training)
        logging.info('n_prediction_train:        %d' % len(self.train_head_dict))
        logging.info('n_prediction_validate:        %d' % len(self.val_head_dict))
        logging.info('n_prediction_testing:         %d' % self.n_prediction_testing)

        logging.info('n_pre_training:        %d' % self.n_pre_training)

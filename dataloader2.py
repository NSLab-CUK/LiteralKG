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


class DataLoaderBase(object):

    def __init__(self, args):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir
        self.device = "cpu"

        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'fine_tuning_train.txt')
        self.test_file = os.path.join(self.data_dir, 'fine_tuning_test.txt')
        self.kg_file = os.path.join(self.data_dir, "pre_training_train.txt")


        self.prediction_dict_file = args.prediction_dict_file
        self.prediction_tail_ids = self.load_prediction_id_list()

        self.entity_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.total_ent = args.total_ent
        self.total_rel = args.total_rel

        self.pre_training_neg_rate = args.pre_training_neg_rate
        self.fine_tuning_neg_rate = args.fine_tuning_neg_rate
        

        self.prediction_train_data, head_dict = self.load_prediction_data(
            self.train_file)
        self.train_head_dict = dict(list(head_dict.items())[:int(args.train_data_rate*len(head_dict))])
        self.val_head_dict = dict(list(head_dict.items())[int(args.train_data_rate*len(head_dict)):])
        self.prediction_test_data, self.test_head_dict = self.load_prediction_data(self.test_file)
        self.analize_prediction()

    def load_prediction_id_list(self):
        file = open(os.path.join(
            self.data_dir, self.prediction_dict_file), 'rb')

        # dump information to that file
        data = pickle.load(file)

        return list(data)

    def load_prediction_data(self, filename):
        head = []
        tail = []
        head_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                head_id, tail_ids = inter[0], inter[1:]
                tail_ids = list(set(tail_ids))

                for tail_id in tail_ids:
                    head.append(head_id)
                    tail.append(tail_id)
                head_dict[head_id] = tail_ids

        heads = np.array(head, dtype=np.int32)
        tails = np.array(tail, dtype=np.int32)
        return (heads, tails), head_dict

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

    def __init__(self, args):
        super().__init__(args)
        self.fine_tuning_batch_size = int(args.fine_tuning_batch_size /  self.fine_tuning_neg_rate)
        self.pre_training_batch_size = int(args.pre_training_batch_size / self.pre_training_neg_rate)
        self.test_batch_size = args.test_batch_size

        self.num_embedding_table = None
        self.text_embedding_table = None
        self.train_data = {}

        graph_data = self.load_graph(self.kg_file)
        self.construct_data(graph_data)
        self.training_tails = graph_data['t']
        

    def construct_data(self, graph_data):
        # Removed addition of inverse

        # re-map head id
        # graph_data['r'] += 2
        self.n_relations = len(set(graph_data['r']))

        # add interactions to kg data
        prediction_train_triples = pd.DataFrame(
            np.zeros((self.n_prediction_training, 3), dtype=np.int32), columns=['h', 'r', 't'])
        prediction_train_triples['h'] = self.prediction_train_data[0]
        prediction_train_triples['t'] = self.prediction_train_data[1]

        self.pre_train_data = pd.concat(
            [graph_data, prediction_train_triples], ignore_index=True)
        self.pre_train_data = graph_data


        for row in self.pre_train_data.iterrows():
            # try:
            #     self.train_data[f"{data} {self.pre_train_data['t'][index]} {self.pre_train_data['r'][index]}"] = index
            # except Exception as e:
            #     print(e)
            h, r, t = row[1]
            self.train_data[f"{h} {t} {r}"] = 1

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
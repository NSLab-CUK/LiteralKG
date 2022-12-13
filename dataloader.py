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
        self.train_file = os.path.join(self.data_dir, 'fine_tuning_train.txt')
        self.test_file = os.path.join(self.data_dir, 'fine_tuning_test.txt')
        self.kg_file = os.path.join(self.data_dir, "pre_training_train.txt")

        self.numeric_literal_files = [
            'age_dict.txt', 'weight_dict.txt']
        self.text_literal_files = ['cc_dict.pickle', 'disease_dict.pickle',
                                   'memo_dict.pickle', 'prescription_dict.pickle', 'treatment_dict.pickle']

        self.prediction_dict_file = args.prediction_dict_file
        self.prediction_tail_ids = self.load_prediction_id_list()

        self.embedding_path = os.path.join(self.data_dir, 'min_model_121.ckpt')

        self.text_dim = args.txt_lit_dim
        self.numeric_dim = args.num_lit_dim
        self.entity_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.total_ent = args.total_ent
        self.total_rel = args.total_rel

        self.pre_training_neg_rate = args.pre_training_neg_rate
        self.fine_tuning_neg_rate = args.fine_tuning_neg_rate

        # self.transr = TransR(
        #     ent_tot = self.total_ent,
        #     rel_tot = self.total_rel,
        #     dim_e = self.entity_dim,
        #     dim_r = self.relation_dim,
        #     p_norm = 1,
        #     norm_flag = True,
        #     rand_init = False)

        # self.load_transr()
        # self.load_entity_embedding()
        

        self.prediction_train_data, head_dict = self.load_prediction_data(
            self.train_file)
        self.train_head_dict = dict(list(head_dict.items())[:int(args.train_data_rate*len(head_dict))])
        self.val_head_dict = dict(list(head_dict.items())[int(args.train_data_rate*len(head_dict)):])
        self.prediction_test_data, self.test_head_dict = self.load_prediction_data(self.test_file)
        self.analize_prediction()

        self.numeric_embed = {}
        self.text_embed = {}
        self.load_attributes()

        # if self.use_pretrain == 1:
        #     self.load_pretrained_data()

    # def load_transr(self):
    #     self.transr.load_checkpoint(self.embedding_path)

    # def get_entity_embeddings(self, ent_id):
    #     return self.transr.get_parameters()['ent_embeddings.weight'][ent_id]

    # def load_entity_embedding(self):
    #     self.get_entity_embeddings(0)

    def load_prediction_id_list(self):
        file = open(os.path.join(
            self.data_dir, self.prediction_dict_file), 'rb')

        # dump information to that file
        data = pickle.load(file)

        return list(data)

    def load_attributes(self):
        count = 0
        for filename in self.numeric_literal_files:
            lines = open(os.path.join(
                self.data_dir, filename), 'r').readlines()
            max_value = 0
            dict_attr = {}
            for l in lines:
                
                data = l.split("\t")
                
                if (len(data) > 1):
                    value = float(data[1].strip("\n"))
                    dict_attr[int(data[0])] = value + 1
                    if max_value < value:
                        max_value = value

            for item in dict_attr:
                num_arr = np.zeros(self.numeric_dim)
                if(max_value != 0):
                    num_arr[count] = dict_attr[item] / max_value
                if self.args.use_num_lit:
                    self.numeric_embed[item] = num_arr
                if self.args.use_txt_lit:
                    self.text_embed[item] = np.zeros(self.text_dim)

            count += 1

        if self.args.use_txt_lit:
            for filename in self.text_literal_files:
                file = open(os.path.join(
                    self.data_dir, filename), 'rb')

                # dump information to that file
                data = pickle.load(file)

                for item in data:
                    if self.args.use_num_lit:
                        self.numeric_embed[item] = np.zeros(self.numeric_dim)
                        
                    if self.args.use_txt_lit:
                        self.text_embed[item] = data[item]

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

    def sample_neg_triples_for_head(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
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
                kg_dict, h, relation[0], self.pre_training_neg_rate, highest_neg_idx)

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

    # def load_pretrained_data(self):
    #     pre_model = 'mf'
    #     pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir,
    #                                       self.data_name, pre_model)
    #     pretrain_data = np.load(pretrain_path)
    #     self.head_pre_embed = pretrain_data['head_embed']
    #     self.tail_pre_embed = pretrain_data['tail_embed']

    #     assert self.head_pre_embed.shape[0] == self.n_heads
    #     assert self.tail_pre_embed.shape[0] == self.n_tails
    #     assert self.head_pre_embed.shape[1] == self.args.embed_dim
    #     assert self.tail_pre_embed.shape[1] == self.args.embed_dim

class DataLoader(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.fine_tuning_batch_size = int(args.fine_tuning_batch_size /  self.fine_tuning_neg_rate)
        self.pre_training_batch_size = int(args.pre_training_batch_size / self.pre_training_neg_rate)
        self.test_batch_size = args.test_batch_size

        self.num_embedding_table = None
        self.text_embedding_table = None

        graph_data = self.load_graph(self.kg_file)
        self.construct_data(graph_data)
        self.embed_num_literal()
        self.embed_txt_literal()
        self.print_info(logging)


        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()


    def construct_data(self, graph_data):
        # Removed addition of inverse

        # re-map head id
        # graph_data['r'] += 2
        self.n_relations = len(set(graph_data['r']))

        # add interactions to kg data
        # prediction_train_triples = pd.DataFrame(
        #     np.zeros((self.n_prediction_training, 3), dtype=np.int32), columns=['h', 'r', 't'])
        # prediction_train_triples['h'] = self.prediction_train_data[0]
        # prediction_train_triples['t'] = self.prediction_train_data[1]

        # self.pre_train_data = pd.concat(
        #     [graph_data, prediction_train_triples], ignore_index=True)

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
        if self.args.use_num_lit:
            self.n_entities = max(list(self.numeric_embed)) + 1
        elif self.args.use_txt_lit:
            self.n_entities = max(list(self.text_embed)) + 1
        else:
            self.n_entities = max(self.n_heads, self.n_tails) 
        self.n_head_tail = self.n_entities

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def embed_num_literal(self):
        if len((list(self.numeric_embed))) == 0:
            return
        self.num_embedding_table = torch.zeros((max(list(self.numeric_embed)) + 1, self.numeric_dim), device=self.device, dtype=torch.float32)
        for item in self.numeric_embed:
            self.num_embedding_table[item] = torch.tensor(self.numeric_embed[item], device=self.device, dtype=torch.float32)

    def embed_txt_literal(self):
        if len((list(self.text_embed))) == 0:
            
            return
        self.text_embedding_table = torch.zeros((max(list(self.text_embed)) + 1, self.text_dim), device=self.device, dtype=torch.float32)
        for item in self.text_embed:
            self.text_embedding_table[item] = torch.tensor(self.text_embed[item], device=self.device, dtype=torch.float32)


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        val_count = 0
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]

            vals = [1] * len(rows)
            val_count += len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(
                self.n_head_tail, self.n_head_tail))
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())

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

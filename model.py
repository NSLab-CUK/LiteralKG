import torch
import torch.nn as nn
import torch.nn.functional as F
from gate import Gate, GateMul
import math
import time

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type, use_residual=False, args=None, n_layers=3):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.use_residual = use_residual
        self.weight = nn.Parameter(torch.FloatTensor(self.in_dim,self.in_dim))
        self.reset_parameters()


        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(
                self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(
                self.in_dim * 2, self.out_dim)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(
                self.in_dim, self.out_dim)
            self.linear2 = nn.Linear(
                self.in_dim, self.out_dim) 
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        elif self.aggregator_type == 'gin':
            hidden_dim = args.gin_hidden_dim
            self.inp_linear = torch.nn.Linear(self.in_dim, hidden_dim)            
            self.out_linear = torch.nn.Linear(hidden_dim, self.out_dim)
            nn.init.xavier_uniform_(self.inp_linear.weight)
            nn.init.xavier_uniform_(self.out_linear.weight)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_dim)
        self.weight.data.uniform_(-stdv, stdv)

    def residual_connection(self, hi, h0, lamda, alpha, l):
       
        if self.use_residual:
            residual = (1-alpha)*hi+alpha*h0
            beta = math.log(lamda/l+1)
            identity_mapping = (1-beta) + beta*self.weight
            return torch.mm(residual, identity_mapping)
        else:
            return hi
            
    def forward(self, ego_embeddings, A_in, h0, lamda, alpha, l):
        """
        ego_embeddings:  (n_heads + n_tails, in_dim)
        A_in:            (n_heads + n_tails, n_heads + n_tails), torch.sparse.FloatTensor
        """
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            hi = ego_embeddings + side_embeddings
            embeddings = self.residual_connection(hi, h0, lamda, alpha, l)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            hi = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.residual_connection(hi, h0, lamda, alpha, l)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            hi_1 = ego_embeddings + side_embeddings
            sum_embeddings = self.residual_connection(hi_1, h0, lamda, alpha, l)
            sum_embeddings = self.activation(self.linear1(sum_embeddings))

            hi_2 = ego_embeddings * side_embeddings
            bi_embeddings = self.residual_connection(hi_2, h0, lamda, alpha, l)
            bi_embeddings = self.activation(self.linear2(bi_embeddings))
            embeddings = bi_embeddings + sum_embeddings
        elif self.aggregator_type == 'gin':
            hi = ego_embeddings + side_embeddings
            hi = self.residual_connection(hi, h0, lamda, alpha, l)
            h=self.inp_linear(ego_embeddings)
            layer_embeds = [h]
            X = self.inp_linear(hi)
            layer_embeds.append(X)

            X = torch.sum(torch.stack(layer_embeds), dim=0)

            embeddings = self.activation(self.out_linear(X))

        
        # (n_heads + n_tails, out_dim)
        embeddings = self.message_dropout(embeddings)
        return embeddings


class LiteralKG(nn.Module):

    def __init__(self, args, n_entities, n_relations, A_in=None, numerical_literals=None, text_literals=None):

        super(LiteralKG, self).__init__()
        self.use_pretrain = args.use_pretrain
        self.args = args

        self.device = args.device

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        # Use residual connection
        self.use_residual = args.use_residual
        self.alpha = args.alpha
        self.lamda = args.lamda


        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)

        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.prediction_l2loss_lambda = args.fine_tuning_l2loss_lambda

        self.pre_training_neg_rate = args.pre_training_neg_rate
        self.fine_tuning_neg_rate = args.fine_tuning_neg_rate

        # Num. Literal
        # num_ent x n_num_lit
        # self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = args.num_lit_dim
        # Txt. Literal
        # num_ent x n_txt_lit
        # self.text_literals = Variable(torch.from_numpy(text_literals)).cuda()
        self.n_txt_lit = args.txt_lit_dim

        self.entity_embed = nn.Embedding(
            self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        # self.trans_M = nn.Parameter(torch.Tensor(
        #     self.n_relations, self.embed_dim, self.relation_dim))
        self.gat_trans_M = nn.Parameter(torch.Tensor(
            self.n_relations, self.embed_dim + self.conv_dim_list[1] + self.conv_dim_list[2] + self.conv_dim_list[3], self.relation_dim))

        nn.init.xavier_uniform_(self.entity_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)
        # nn.init.xavier_uniform_(self.trans_M)
        nn.init.xavier_uniform_(self.gat_trans_M)

        self.aggregator_layers = nn.ModuleList()

        self.numerical_literals_embed = numerical_literals
        self.text_literals_embed = text_literals

        # LiteralE's g
        if self.args.use_num_lit and self.args.use_txt_lit:
            self.emb_mul_lit = GateMul(self.embed_dim, self.n_num_lit, self.n_txt_lit)
        elif self.args.use_num_lit:
            self.emb_num_lit = Gate(self.embed_dim, self.n_num_lit)
        elif self.args.use_txt_lit:
            self.emb_txt_lit = Gate(self.embed_dim, self.n_txt_lit)

        for k in range(self.n_layers):
            self.aggregator_layers.append(
                Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k],
                           self.aggregation_type, self.use_residual, args, k+1))

        self.A_in = nn.Parameter(
            torch.sparse.FloatTensor(self.n_entities, self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False

        self.milestone_score = args.milestone_score

    def gate_embeddings(self):
        ent_emb = self.entity_embed.weight
        self.numerical_literals_embed = self.numerical_literals_embed.to(self.device)
        self.text_literals_embed = self.text_literals_embed.to(self.device)

        if self.args.use_num_lit and self.args.use_txt_lit:
            return self.emb_mul_lit(ent_emb, self.numerical_literals_embed, self.text_literals_embed)
        elif self.args.use_num_lit:
            return self.emb_num_lit(ent_emb, self.numerical_literals_embed)
        elif self.args.use_txt_lit:
            return self.emb_txt_lit(ent_emb, self.text_literals_embed)
        return ent_emb

    def gate_embeddings_v2(self, e):
        ent_emb = self.entity_embed.weight

        ent_emb = ent_emb[e]

        if self.args.use_num_lit and self.args.use_txt_lit:
            num_emb = self.numerical_literals_embed[e]
            txt_emb = self.text_literals_embed[e]
            return self.emb_mul_lit(ent_emb, num_emb, txt_emb)
        elif self.args.use_num_lit:
            num_emb = self.numerical_literals_embed[e]
            return self.emb_num_lit(ent_emb, num_emb)
        elif self.args.use_txt_lit:
            txt_emb = self.text_literals_embed[e]
            return self.emb_txt_lit(ent_emb, txt_emb)
        return ent_emb

    def gat_embeddings(self):
        ent_lit_mul_r = self.gate_embeddings()

        all_embed = [ent_lit_mul_r]

        for idx, layer in enumerate(self.aggregator_layers):
            ent_lit_mul_r = layer(ent_lit_mul_r, self.A_in, ent_lit_mul_r[0], self.lamda, self.alpha, idx+1)
            norm_embed = F.normalize(ent_lit_mul_r, p=2, dim=1)
            all_embed.append(norm_embed)

        # (n_heads + n_tails, concat_dim)
        return torch.cat(all_embed, dim=1)

    def calculate_prediction_loss(self, head_ids, tail_pos_ids, tail_neg_ids):
        """
        head_ids:       (prediction_batch_size)
        tail_pos_ids:   (prediction_batch_size)
        tail_neg_ids:   (prediction_batch_size)
        """
        all_embed = self.gat_embeddings()  # (n_heads + n_tails, concat_dim)
        
        head_embed = all_embed[head_ids]  # (batch_size, concat_dim)
        tail_pos_embed = all_embed[tail_pos_ids]  # (batch_size, concat_dim)
        tail_neg_embed = all_embed[tail_neg_ids]  # (batch_size, concat_dim)

        # head_embed = self.gat_embeddings(head_ids)  # (batch_size, concat_dim)
        # tail_pos_embed = self.gat_embeddings(tail_pos_ids)  # (batch_size, concat_dim)
        # tail_neg_embed =self.gat_embeddings(tail_neg_ids)  # (batch_size, concat_dim)

        pos_score = torch.sum(head_embed * tail_pos_embed,
                              dim=1)  # (batch_size)
        print("Conpare pos_score neg_score")
        print(pos_score)
        neg_score = torch.sum(head_embed * tail_neg_embed,
                              dim=1)  # (batch_size)
        print(neg_score)

        # prediction_loss = F.softplus(neg_score - pos_score)
        prediction_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        prediction_loss = torch.mean(prediction_loss)

        l2_loss = _L2_loss_mean(
            head_embed) + _L2_loss_mean(tail_pos_embed) + _L2_loss_mean(tail_neg_embed)
        loss = prediction_loss + self.prediction_l2loss_lambda * l2_loss
        return loss

    # def embed_num_literal(self):
    #     embedding_table = torch.zeros((self.n_entities, self.n_num_lit), device='cuda:0', dtype=torch.long)
    #     for item in self.numerical_literals:
    #         embedding_table[item] = torch.tensor(self.numerical_literals[item])

    #     return embedding_table

    # def embed_txt_literal(self):
    #     embedding_table = torch.zeros((self.n_entities, self.n_txt_lit), device='cuda:0', dtype=torch.long)
    #     for item in self.text_literals:
    #         embedding_table[item] = torch.tensor(self.text_literals[item])

    #     return embedding_table

    def calc_triplet_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.gat_trans_M[r]  # (kg_batch_size, embed_dim, relation_dim)
        # h_embed = self.entity_embed(h)  # (kg_batch_size, embed_dim)
        # pos_t_embed = self.entity_embed(
        #     pos_t)  # (kg_batch_size, embed_dim)
        # neg_t_embed = self.entity_embed(
        #     neg_t)  # (kg_batch_size, embed_dim)

        # GAT embeddings
        all_embed = self.gat_embeddings()  # (n_heads + n_tails, concat_dim)

        head_embed = all_embed[h]  # (batch_size, concat_dim)
        tail_pos_embed = all_embed[pos_t]  # (batch_size, concat_dim)
        tail_neg_embed = all_embed[neg_t]  # (batch_size, concat_dim)

        # head_embed = self.gat_embeddings(h)  # (batch_size, concat_dim)
        # tail_pos_embed = self.gat_embeddings(pos_t)  # (batch_size, concat_dim)
        # tail_neg_embed = self.gat_embeddings(neg_t)  # (batch_size, concat_dim)

        r_mul_h = torch.bmm(head_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(tail_pos_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(tail_neg_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)

        # h_lit_embed = self.entity_literal_embed(h, h_embed)  # Gate(heads, head_embeddings)
        # pos_t_lit_embed = self.entity_literal_embed(
        #     pos_t, pos_t_embed)  # Gate(pos_tails, pos_tail_embeddings)
        # neg_t_lit_embed = self.entity_literal_embed(
        #     neg_t, neg_t_embed)  # Gate(neg_tails, neg_tail_embeddings)

        # r_mul_h = torch.bmm(h_lit_embed.unsqueeze(1), W_r).squeeze(
        #     1)  # (kg_batch_size, relation_dim)
        # r_mul_pos_t = torch.bmm(pos_t_lit_embed.unsqueeze(1), W_r).squeeze(
        #     1)  # (kg_batch_size, relation_dim)
        # r_mul_neg_t = torch.bmm(neg_t_lit_embed.unsqueeze(1), W_r).squeeze(
        #     1)  # (kg_batch_size, relation_dim)

        # Trans R

        # Equation (1)
        pos_score = torch.sum(
            torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        # triplet_loss = F.softplus(pos_score - neg_score)
        triplet_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        triplet_loss = torch.mean(triplet_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = triplet_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        # W_r = self.trans_M[r_idx]

        h_embed = self.entity_embed.weight[h_list]
        t_embed = self.entity_embed.weight[t_list]

        # Equation (4)
        # r_mul_h = torch.matmul(h_embed, W_r)
        # r_mul_t = torch.matmul(t_embed, W_r)
        # v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)

        v_list = torch.sum(t_embed * torch.tanh(h_embed + r_embed), dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(
                batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def calc_score(self, head_ids, tail_ids):
        all_embed = self.gat_embeddings()  # (n_heads + n_tails, concat_dim)
        head_embed = all_embed[head_ids]  # (n_heads, concat_dim)
        tail_embed = all_embed[tail_ids]  # (n_items, concat_dim)

        # head_embed = self.gat_embeddings(head_ids)  # (n_heads, concat_dim)
        # tail_embed = self.gat_embeddings(tail_ids)  # (n_items, concat_dim)

        prediction_score = torch.matmul(
            head_embed, tail_embed.transpose(0, 1))  # (n_heads, n_items)

        return prediction_score

    def predict_links(self, head_ids, tail_ids):
        scores = self.calc_score(head_ids, tail_ids)

        return (scores>self.milestone_score).int()

    def forward(self, *input, device, mode):
        self.device=device
        if mode == 'fine_tuning':
            return self.calculate_prediction_loss(*input)
        if mode == 'pre_training':
            return self.calc_triplet_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.predict_links(*input)

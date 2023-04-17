import torch
import torch.nn as nn
import torch.nn.functional as F
from gate import Gate, GateMul
import math


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type, use_residual=False, args=None):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.use_residual = use_residual
        self.weight = nn.Parameter(torch.FloatTensor(self.in_dim, self.in_dim))
        if use_residual:
            self.linear_h0 = nn.Linear(args.embed_dim, self.in_dim)
            nn.init.xavier_uniform_(self.linear_h0.weight)

        self.reset_parameters()

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.layer_normalize = nn.LayerNorm(self.out_dim)

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(
                self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            if self.use_residual:
                self.linear_h = nn.Linear(
                    self.in_dim * 2, self.in_dim)
                nn.init.xavier_uniform_(self.linear_h.weight)
                self.linear = nn.Linear(
                    self.in_dim, self.out_dim)
                nn.init.xavier_uniform_(self.linear.weight)
            else:
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
            hidden_dim = args.mlp_hidden_dim
            self.num_layers = args.n_mlp_layers
            self.weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))

            self.linear_h0 = nn.Linear(args.embed_dim, hidden_dim)
            nn.init.xavier_uniform_(self.linear_h0.weight)

            if self.num_layers == 1:
                # Linear model
                self.linear = nn.Linear(self.in_dim, self.out_dim)
            else:
                # Multi-layer model
                self.inp_linear = torch.nn.Linear(self.in_dim, hidden_dim)
                self.linears = torch.nn.ModuleList()
                self.mlp_layer_norms = torch.nn.ModuleList()

                for layer in range(self.num_layers - 1):
                    self.linears.append(nn.Linear(hidden_dim, hidden_dim))

                self.out_linear = nn.Linear(hidden_dim, self.out_dim)

                for layer in range(self.num_layers - 1):
                    self.mlp_layer_norms.append(nn.LayerNorm(hidden_dim))

        else:
            raise NotImplementedError

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_dim)
        self.weight.data.uniform_(-stdv, stdv)

    def residual_connection(self, hi, h0, lamda, alpha, l):

        if self.use_residual:
            h0 = self.linear_h0(h0)
            residual = (1 - alpha) * hi + alpha * h0
            beta = math.log(lamda / l + 1)
            identity_mapping = (1 - beta) + beta * self.weight
            return torch.mm(residual, identity_mapping)
        else:
            return hi

    def forward(self, ego_embeddings, A_in, all_layers, lamda, alpha, l):
        """
        ego_embeddings:  (n_heads + n_tails, in_dim)
        A_in:            (n_heads + n_tails, n_heads + n_tails), torch.sparse.FloatTensor
        """
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            hi = ego_embeddings + side_embeddings
            embeddings = self.residual_connection(hi, all_layers[0], lamda, alpha, l)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            hi = torch.cat([ego_embeddings, side_embeddings], dim=1)
            if self.use_residual:
                hi = self.linear_h(hi)
                embeddings = self.residual_connection(hi, all_layers[0], lamda, alpha, l)
            else:
                embeddings = hi
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            hi_1 = ego_embeddings + side_embeddings
            sum_embeddings = self.residual_connection(hi_1, all_layers[0], lamda, alpha, l)
            sum_embeddings =self.activation(self.linear1(sum_embeddings))

            hi_2 = ego_embeddings * side_embeddings
            bi_embeddings = self.residual_connection(hi_2, all_layers[0], lamda, alpha, l)
            bi_embeddings = self.activation(self.linear2(bi_embeddings))
            embeddings = bi_embeddings + sum_embeddings
        elif self.aggregator_type == 'gin':
            hi = ego_embeddings + side_embeddings
            h = self.inp_linear(ego_embeddings)
            layer_embeds = [h]
            if self.num_layers == 1:
                # Linear model
                hi = self.linear(hi)
                layer_embeds.append(hi)
            else:
                # If MLP
                h = self.inp_linear(hi)
                for layer in range(self.num_layers - 1):
                    h = self.mlp_layer_norms[layer](self.activation(self.linears[layer](h)))
                    layer_embeds.append(h)

            X = torch.sum(torch.stack(layer_embeds), dim=0)
            X = self.residual_connection(X, all_layers[0], lamda, alpha, l)

            embeddings = self.activation(self.out_linear(X))

            if len(all_layers) > 1:
                layer_embeds = [self.layer_normalize(embeddings)]

                for index, layer in enumerate(all_layers):
                    if index != 0:
                        layer_embeds.append(layer)

                embeddings = torch.sum(torch.stack(layer_embeds), dim=0)

        # (n_heads + n_tails, out_dim)
        embeddings = self.message_dropout(self.layer_normalize(embeddings))
        
        
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

        self.scale_gat_dim = args.scale_gat_dim

        # Use residual connection
        self.use_residual = args.use_residual
        self.alpha = args.alpha
        self.lamda = args.lamda

        self.aggregation_type = args.aggregation_type
        self.n_layers = args.n_conv_layers
        # self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.conv_dim_list = [args.embed_dim] + [args.conv_dim]*self.n_layers

        self.total_conv_dim = sum([self.conv_dim_list[i] for i in range(self.n_layers + 1)])

        #self.mess_dropout = eval(args.mess_dropout)
        self.mess_dropout = [args.mess_dropout]*self.n_layers

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

        if self.scale_gat_dim is not None:
            self.linear_gat = nn.Linear(self.total_conv_dim, self.scale_gat_dim)
            self.gat_activation = nn.LeakyReLU()
            nn.init.xavier_uniform_(self.linear_gat.weight)
            self.gat_trans_M = nn.Parameter(torch.Tensor(
                self.n_relations, self.scale_gat_dim, self.relation_dim))
        else:
            self.gat_trans_M = nn.Parameter(torch.Tensor(
                self.n_relations,
                self.total_conv_dim,
                self.relation_dim))

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
                           self.aggregation_type, self.use_residual, args))

        self.A_in = nn.Parameter(
            torch.sparse.FloatTensor(self.n_entities, self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False

        self.milestone_score = args.milestone_score

        if self.scale_gat_dim is not None:
            self.mlp_layer_1 = nn.Linear(self.scale_gat_dim, 128)
            self.mlp_layer_2 = nn.Linear(128, 64)
            self.mlp_layer_3 = nn.Linear(64, 1)
        else:
            self.mlp_layer_1 = nn.Linear(self.total_conv_dim, 128)
            self.mlp_layer_2 = nn.Linear(128, 64)
            self.mlp_layer_3 = nn.Linear(64, 1)

        nn.init.xavier_uniform_(self.mlp_layer_1.weight)
        nn.init.xavier_uniform_(self.mlp_layer_2.weight)
        nn.init.xavier_uniform_(self.mlp_layer_3.weight)

        self.mlp_activation = nn.LeakyReLU()
        self.output_activation = nn.LeakyReLU()

    def gate_embeddings(self):
        ent_emb = self.entity_embed.weight

        if self.args.use_num_lit and self.args.use_txt_lit:
            self.numerical_literals_embed = self.numerical_literals_embed.to(self.device)
            self.text_literals_embed = self.text_literals_embed.to(self.device)
            return self.emb_mul_lit(ent_emb, self.numerical_literals_embed, self.text_literals_embed)
        elif self.args.use_num_lit:
            self.numerical_literals_embed = self.numerical_literals_embed.to(self.device)
            return self.emb_num_lit(ent_emb, self.numerical_literals_embed)
        elif self.args.use_txt_lit:
            self.text_literals_embed = self.text_literals_embed.to(self.device)
            return self.emb_txt_lit(ent_emb, self.text_literals_embed)

        return ent_emb

    def gat_embeddings(self):
        ent_lit_mul_r = self.gate_embeddings()

        all_embed = [ent_lit_mul_r]

        for idx, layer in enumerate(self.aggregator_layers):
            ent_lit_mul_r = layer(ent_lit_mul_r, self.A_in, all_embed, self.lamda, self.alpha, idx + 1)
            norm_embed = F.normalize(ent_lit_mul_r, p=2, dim=1)
            all_embed.append(norm_embed)

        if self.scale_gat_dim is not None:
            gat_embed = self.linear_gat(torch.cat(all_embed, dim=1))
            gat_embed = self.gat_activation(gat_embed)
            return gat_embed
        else:
            # (n_heads + n_tails, concat_dim)
            return torch.cat(all_embed, dim=1)

    def calculate_prediction_score(self, head_ids, tail_pos_ids, tail_neg_ids):
        """
        head_ids:       (prediction_batch_size)
        tail_pos_ids:   (prediction_batch_size)
        tail_neg_ids:   (prediction_batch_size)
        """
        self.gat_embed = self.gat_embeddings()  # (n_heads + n_tails, concat_dim)

        head_embed = self.gat_embed[head_ids]  # (batch_size, concat_dim)
        tail_pos_embed = self.gat_embed[tail_pos_ids]  # (batch_size, concat_dim)
        tail_neg_embed = self.gat_embed[tail_neg_ids]  # (batch_size, concat_dim)

        x_pos = torch.cat((head_embed, tail_pos_embed), 1)
        x_neg = torch.cat((head_embed, tail_neg_embed), 1)
        x = torch.cat((x_pos, x_neg), 0)

        x = self.mlp_layer_1(x)
        x = self.mlp_activation(x)
        x = self.mlp_layer_2(x)
        x = self.mlp_activation(x)
        x = self.mlp_layer_3(x)
        out = self.output_activation(x)
        
        return out

    def calc_triplet_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.gat_trans_M[r]  # (kg_batch_size, embed_dim, relation_dim)

        # GAT embeddings
        self.gat_embed = self.gat_embeddings()  # (n_heads + n_tails, concat_dim)

        head_embed = self.gat_embed[h]  # (batch_size, concat_dim)
        tail_pos_embed = self.gat_embed[pos_t]  # (batch_size, concat_dim)
        tail_neg_embed = self.gat_embed[neg_t]  # (batch_size, concat_dim)

        r_mul_h = torch.bmm(head_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(tail_pos_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(tail_neg_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)

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

        h_embed = self.entity_embed.weight[h_list]
        t_embed = self.entity_embed.weight[t_list]

        # Equation
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
        self.gat_embed = self.gat_embeddings()  # (n_heads + n_tails, concat_dim)

        head_embed = self.gat_embed[head_ids]  # (batch_size, concat_dim)
        tail_embed = self.gat_embed[tail_ids]  # (batch_size, concat_dim)

        x = torch.cat((head_embed, tail_embed), 1)

        x = self.mlp_layer_1(x)
        x = self.mlp_activation(x)
        x = self.mlp_layer_2(x)
        x = self.mlp_activation(x)
        x = self.mlp_layer_3(x)
        out = self.output_activation(x)

        return out

    def predict_links(self, head_ids, tail_ids):
        scores = self.calc_score(head_ids, tail_ids)
        print(scores)
        return (scores > self.milestone_score).int()

    def forward(self, *input, device, mode):
        self.device = device
        if mode == 'fine_tuning':
            return self.calculate_prediction_score(*input)
        if mode == 'pre_training':
            return self.calc_triplet_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.predict_links(*input)

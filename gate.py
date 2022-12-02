import torch
import torch.nn as nn


class GateMul(nn.Module):

    def __init__(self, emb_size, num_lit_size, txt_lit_size, gate_activation=torch.sigmoid):
        super(GateMul, self).__init__()

        self.emb_size = emb_size
        self.num_lit_size = num_lit_size
        self.txt_lit_size = txt_lit_size

        self.gate_activation = gate_activation
        self.g = nn.Linear(emb_size+num_lit_size+txt_lit_size, emb_size)

        self.gate_ent = nn.Linear(emb_size, emb_size, bias=False)
        self.gate_num_lit = nn.Linear(num_lit_size, emb_size, bias=False)
        self.gate_txt_lit = nn.Linear(txt_lit_size, emb_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(emb_size))

    def forward(self, x_ent, x_lit_num, x_lit_txt):
        x = torch.cat([x_ent, x_lit_num, x_lit_txt], dim=1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.gate_ent(x_ent) + self.gate_num_lit(x_lit_num) + self.gate_txt_lit(x_lit_txt) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output

class Gate(nn.Module):

    def __init__(self, emb_size, lit_size, gate_activation=torch.sigmoid):
        super(Gate, self).__init__()

        self.emb_size = emb_size
        self.lit_size = lit_size

        self.gate_activation = gate_activation
        self.g = nn.Linear(emb_size+lit_size, emb_size)

        self.gate_ent = nn.Linear(emb_size, emb_size, bias=False)
        self.gate_lit = nn.Linear(lit_size, emb_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(emb_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], dim=1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.gate_ent(x_ent) + self.gate_lit(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output

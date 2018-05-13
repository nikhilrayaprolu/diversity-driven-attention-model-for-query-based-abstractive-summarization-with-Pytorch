import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class DiverseLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers=1, dropout=0.5):
        """"Constructor of the class"""
        super(DiverseLSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        ih, hh = [], []
        for i in range(nlayers):
            ih.append(nn.Linear(input_size, 4 * hidden_size))
            hh.append(nn.Linear(hidden_size, 4 * hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []

        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 2)
            i_gate = F.sigmoid(i_gate)
            f_gate = F.sigmoid(f_gate)
            c_gate = F.tanh(c_gate)
            o_gate = F.sigmoid(o_gate)
            ncx = (f_gate * cx) + (i_gate * c_gate)
            #TODO check sum dimension
            temp1 = torch.mul(ncx.view(-1,self.hidden_size), cx.view(-1,self.hidden_size)).sum(1)
            temp2 = torch.mul(cx.view(-1,self.hidden_size), cx.view(-1,self.hidden_size)).sum(1)
            fin_sub = torch.div(temp1, temp2)
            cxt = torch.t(cx.view(-1,self.hidden_size))
            fin = fin_sub.expand_as(cxt)
            ncx_diverse = ncx - torch.t(cxt*fin).contiguous().view(1,-1,self.hidden_size)

            nhx = o_gate * F.tanh(ncx_diverse)
            cy.append(ncx)
            hy.append(nhx)
            input = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
        return hy, cy

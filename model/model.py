import torch

import numpy as np
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import MessagePassing

class AggNet(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)
        self.parameter1 = torch.nn.Parameter(torch.FloatTensor(abs(np.random.rand(1)).reshape(1, 1)))
        self.parameter2 = torch.nn.Parameter(torch.FloatTensor(abs(np.random.rand(1)).reshape(1, 1)))
        self.bias = torch.nn.Parameter(torch.FloatTensor(np.zeros(1).reshape(1, 1)))

    def forward(self, x, edge_index, edge_weight):
        out = self.propagate(edge_index=edge_index, edge_weight=edge_weight, x=x)
        out += self.bias

        return F.leaky_relu(out, negative_slope=0.1)

    def message(self, x_i, x_j, edge_weight):
        return (abs(self.parameter1) * x_j + abs(self.parameter2) * x_i)

class scPRS(nn.Module):
    def __init__(self, dim_in, n_cell, n_gcn):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.FloatTensor(abs(np.random.rand(dim_in).reshape([dim_in, 1]))))
        self.bias1 = torch.nn.Parameter(torch.FloatTensor(np.zeros(1).reshape(1, 1)))
        self.agg = nn.ModuleList()
        for i in range(n_gcn):
            self.agg.append(AggNet())
        self.pred = nn.Linear(n_cell, 1)

    def forward(self, x, edge, edge_weight):
        x = (x@abs(self.parameter)/x.shape[-1]+self.bias1).transpose(0, 1).squeeze(-1)
        for net in self.agg:
            x = net(x=x, edge_index=edge, edge_weight=edge_weight)
        out = x.transpose(0, 1)
        out = self.pred(out)
        return out

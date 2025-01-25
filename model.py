import copy
import utils
import os.path as osp
import numpy as np
from torch_geometric.data import HeteroData
import torch
from dgllife.utils import EarlyStopping
from torch import nn
from torch_geometric.datasets import DBLP
from torch_geometric.nn import RGATConv
from prettytable import PrettyTable
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import random
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)


class RGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_nodes,
        num_relations,
        init_sizes,
        node_types,
    ):
        super(RGAT, self).__init__()
        self.conv1 = RGATConv(
            in_channels, hidden_channels, num_relations=num_relations, num_bases=30
        )
        self.conv2 = RGATConv(
            hidden_channels, hidden_channels, num_relations=num_relations, num_bases=30
        )
        self.num_nodes = num_nodes
        self.node_types = node_types
        self.lins = torch.nn.ModuleList()
        self.transform = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, out_channels),
        )
        for i in range(len(node_types)):
            lin = nn.Sequential(
                nn.Linear(init_sizes[i], in_channels),
                nn.ReLU(),
                nn.LayerNorm(in_channels),
                nn.Dropout(0.5),
            )
            self.lins.append(lin)

    def trans_dimensions(self, g):
        data = copy.deepcopy(g)
        for node_type, lin in zip(self.node_types, self.lins):
            data[node_type].feature = lin(data[node_type].feature)

        return data

    def forward(self, forward_type, data):
        data = self.trans_dimensions(data)
        homogeneous_data = data.to_homogeneous()
        edge_index, edge_type = homogeneous_data.edge_index, homogeneous_data.edge_type
        x = self.conv1(homogeneous_data.feature, edge_index, edge_type)
        x, atten = self.conv2(x, edge_index, edge_type, return_attention_weights=True)
        edge_of_atten = atten[0]
        num_of_atten = atten[1]
        if forward_type == "first" or forward_type == "test":
            save_tensor(
                edge_of_atten,
                True,
                "%d",
                "./atten/edge_of_atten_{}.csv".format(forward_type),
            )
            save_tensor(
                num_of_atten,
                False,
                "%.4f",
                "./atten/num_of_atten_{}.csv".format(forward_type),
            )
        x = x.squeeze()
        x = self.transform(x)
        x = x[: self.num_nodes]
        return x


def save_tensor(ten: torch.tensor, needT, fmt, name):
    array = ten.cpu().detach().numpy()
    if needT:
        array = array.T
    np.savetxt(name, array, fmt=fmt, delimiter=",")

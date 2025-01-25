import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义生成器
class GraphGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_edges, num_labels):
        super(GraphGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.edge_fc = nn.Linear(input_dim, num_edges)  # 生成边信息
        self.label_fc = nn.Linear(input_dim, num_labels)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        node_features = self.fc2(x)
        edges = torch.sigmoid(self.edge_fc(z))  # 邻接矩阵
        labels = torch.sigmoid(self.label_fc(z))
        return node_features, edges, labels


# 定义判别器
class GraphDiscriminator(nn.Module):
    def __init__(self, node_feature_length, hidden_dim, num_edges, num_labels):
        super(GraphDiscriminator, self).__init__()
        self.node_fc = nn.Linear(node_feature_length, hidden_dim)
        self.edge_fc = nn.Linear(num_edges, hidden_dim)
        self.label_fc = nn.Linear(num_labels, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 3, 1)

    def forward(self, node_features, edges, labels):
        node_output = torch.relu(self.node_fc(node_features))
        edge_output = torch.relu(self.edge_fc(edges))
        label_output = torch.relu(self.label_fc(labels))
        combined = torch.cat((node_output, edge_output, label_output), dim=1)
        return torch.sigmoid(self.fc(combined))
    
    
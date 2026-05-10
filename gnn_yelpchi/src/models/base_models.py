"""
Base GNN Models: MLP, GCN, SAGE, GAT, GRAPHCONV, CHEB, TAG, SG
YelpChi version - 3 relations (R-U-R, R-T-R, R-S-R)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphConv, ChebConv, TAGConv, SGConv


class MLPModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_indices=None, **kwargs):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.fc3(x)
        return logits, [], x


class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=3, dropout=0.3):
        super().__init__()
        self.num_relations = num_relations
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels) for _ in range(num_relations)])
        self.conv2 = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_relations)])
        self.conv3 = nn.ModuleList([GCNConv(hidden_channels, out_channels) for _ in range(num_relations)])
        self.dropout = dropout

    def forward(self, x, edge_indices, **kwargs):
        xs = []
        for i, conv in enumerate(self.convs):
            x_i = conv(x, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            x_i = self.conv2[i](x_i, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            xs.append(x_i)
        x = torch.stack(xs).mean(dim=0)
        logits = self.conv3[0](x, edge_indices[0])
        return logits, [], x


class SAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=3, dropout=0.3):
        super().__init__()
        self.num_relations = num_relations
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels) for _ in range(num_relations)])
        self.conv2 = nn.ModuleList([SAGEConv(hidden_channels, hidden_channels) for _ in range(num_relations)])
        self.conv3 = nn.ModuleList([SAGEConv(hidden_channels, out_channels) for _ in range(num_relations)])
        self.dropout = dropout

    def forward(self, x, edge_indices, **kwargs):
        xs = []
        for i, conv in enumerate(self.convs):
            x_i = conv(x, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            x_i = self.conv2[i](x_i, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            xs.append(x_i)
        x = torch.stack(xs).mean(dim=0)
        logits = self.conv3[0](x, edge_indices[0])
        return logits, [], x


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=3, dropout=0.3):
        super().__init__()
        self.num_relations = num_relations
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=4, dropout=dropout) for _ in range(num_relations)])
        self.conv2 = nn.ModuleList([GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout) for _ in range(num_relations)])
        self.conv3 = nn.ModuleList([GATConv(hidden_channels * 4, out_channels, heads=1, dropout=dropout) for _ in range(num_relations)])
        self.dropout = dropout

    def forward(self, x, edge_indices, **kwargs):
        xs = []
        for i, conv in enumerate(self.convs):
            x_i = conv(x, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            x_i = self.conv2[i](x_i, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            xs.append(x_i)
        x = torch.stack(xs).mean(dim=0)
        logits = self.conv3[0](x, edge_indices[0])
        return logits, [], x


class GraphConvModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=3, dropout=0.3):
        super().__init__()
        self.num_relations = num_relations
        self.convs = nn.ModuleList([GraphConv(in_channels, hidden_channels) for _ in range(num_relations)])
        self.conv2 = nn.ModuleList([GraphConv(hidden_channels, hidden_channels) for _ in range(num_relations)])
        self.conv3 = nn.ModuleList([GraphConv(hidden_channels, out_channels) for _ in range(num_relations)])
        self.dropout = dropout

    def forward(self, x, edge_indices, **kwargs):
        xs = []
        for i, conv in enumerate(self.convs):
            x_i = conv(x, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            x_i = self.conv2[i](x_i, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            xs.append(x_i)
        x = torch.stack(xs).mean(dim=0)
        logits = self.conv3[0](x, edge_indices[0])
        return logits, [], x


class ChebModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=3, dropout=0.3, K=3):
        super().__init__()
        self.num_relations = num_relations
        self.convs = nn.ModuleList([ChebConv(in_channels, hidden_channels, K) for _ in range(num_relations)])
        self.conv2 = nn.ModuleList([ChebConv(hidden_channels, hidden_channels, K) for _ in range(num_relations)])
        self.conv3 = nn.ModuleList([ChebConv(hidden_channels, out_channels, K) for _ in range(num_relations)])
        self.dropout = dropout

    def forward(self, x, edge_indices, edge_weights=None, **kwargs):
        xs = []
        for i, conv in enumerate(self.convs):
            ew = edge_weights[i] if edge_weights is not None else None
            x_i = conv(x, edge_indices[i], ew) if ew is not None else conv(x, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            x_i = self.conv2[i](x_i, edge_indices[i], ew) if ew is not None else self.conv2[i](x_i, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            xs.append(x_i)
        x = torch.stack(xs).mean(dim=0)
        ew = edge_weights[0] if edge_weights is not None else None
        logits = self.conv3[0](x, edge_indices[0], ew) if ew is not None else self.conv3[0](x, edge_indices[0])
        return logits, [], x


class TAGModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=3, dropout=0.3, K=3):
        super().__init__()
        self.num_relations = num_relations
        self.convs = nn.ModuleList([TAGConv(in_channels, hidden_channels, K=K) for _ in range(num_relations)])
        self.conv2 = nn.ModuleList([TAGConv(hidden_channels, hidden_channels, K=K) for _ in range(num_relations)])
        self.conv3 = nn.ModuleList([TAGConv(hidden_channels, out_channels, K=K) for _ in range(num_relations)])
        self.dropout = dropout

    def forward(self, x, edge_indices, **kwargs):
        xs = []
        for i, conv in enumerate(self.convs):
            x_i = conv(x, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            x_i = self.conv2[i](x_i, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            xs.append(x_i)
        x = torch.stack(xs).mean(dim=0)
        logits = self.conv3[0](x, edge_indices[0])
        return logits, [], x


class SGModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=3, dropout=0.3, K=2):
        super().__init__()
        self.num_relations = num_relations
        self.convs = nn.ModuleList([SGConv(in_channels, hidden_channels, K=K) for _ in range(num_relations)])
        self.conv2 = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(num_relations)])
        self.conv3 = nn.ModuleList([SGConv(hidden_channels, out_channels, K=K) for _ in range(num_relations)])
        self.dropout = dropout

    def forward(self, x, edge_indices, **kwargs):
        xs = []
        for i, conv in enumerate(self.convs):
            x_i = conv(x, edge_indices[i])
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            x_i = self.conv2[i](x_i)
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            xs.append(x_i)
        x = torch.stack(xs).mean(dim=0)
        logits = self.conv3[0](x, edge_indices[0])
        return logits, [], x

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index_dict):
        edge_index = edge_index_dict["rur"]

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x).squeeze(-1)

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_heads=8, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))

        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            )

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
            if i < len(self.convs) - 1:
                x = torch.relu(x)
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x).squeeze(-1)

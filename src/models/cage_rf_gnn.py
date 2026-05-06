import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class CAGERFGNNBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


class RelationGate(nn.Module):
    def __init__(self, hidden_dim, num_relations=6):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.num_relations = num_relations

    def forward(self, relation_embeddings):
        batch_size = relation_embeddings.shape[0]

        gate_logits = self.gate_mlp(relation_embeddings)

        gate_logits = gate_logits.view(batch_size, self.num_relations, 1)

        alpha = torch.softmax(gate_logits, dim=1)

        return alpha


class CAGERF_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3, use_gating=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_gating = use_gating

        self.rur_branch = CAGERFGNNBranch(input_dim, hidden_dim, num_layers, dropout)
        self.rtr_branch = CAGERFGNNBranch(input_dim, hidden_dim, num_layers, dropout)
        self.rsr_branch = CAGERFGNNBranch(input_dim, hidden_dim, num_layers, dropout)
        self.burst_branch = CAGERFGNNBranch(input_dim, hidden_dim, num_layers, dropout)
        self.semsim_branch = CAGERFGNNBranch(input_dim, hidden_dim, num_layers, dropout)
        self.behavior_branch = CAGERFGNNBranch(input_dim, hidden_dim, num_layers, dropout)

        if use_gating:
            self.gate = RelationGate(hidden_dim, num_relations=6)
        else:
            self.gate = None

        concat_dim = hidden_dim * 6
        projection_input_dim = hidden_dim if use_gating else concat_dim

        self.projection = nn.Sequential(
            nn.Linear(projection_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = dropout

    def forward(self, x, edge_index_dict):
        h_rur = self.rur_branch(x, edge_index_dict["rur"])
        h_rtr = self.rtr_branch(x, edge_index_dict["rtr"])
        h_rsr = self.rsr_branch(x, edge_index_dict["rsr"])
        h_burst = self.burst_branch(x, edge_index_dict["burst"])
        h_semsim = self.semsim_branch(x, edge_index_dict["semsim"])
        h_behavior = self.behavior_branch(x, edge_index_dict["behavior"])

        h_cat = torch.cat([h_rur, h_rtr, h_rsr, h_burst, h_semsim, h_behavior], dim=1)

        if self.use_gating:
            relation_stack = torch.stack([
                h_rur, h_rtr, h_rsr, h_burst, h_semsim, h_behavior
            ], dim=1)

            alpha = self.gate(relation_stack)

            h_fused = (alpha.squeeze(-1).unsqueeze(-1) * relation_stack).sum(dim=1)

            self.last_alpha = alpha
        else:
            h_fused = h_cat

        h_proj = self.projection(h_fused)

        logit = self.classifier(h_proj).squeeze(-1)

        return logit

    def get_relation_contribution(self):
        if hasattr(self, 'last_alpha'):
            return self.last_alpha.detach().cpu().numpy()
        return None

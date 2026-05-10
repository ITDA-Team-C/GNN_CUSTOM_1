"""
CAGE-RF CHEB v8: Skip Connection with Chebyshev Filter (Over-smoothing 해결)
YelpChi용 - 3 relations (R-U-R, R-T-R, R-S-R)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class CAGERFCHEBV8(nn.Module):
    """
    CAGE-RF with Skip Connections and Chebyshev Filter (YelpChi)

    Skip Connection이 각 레이어마다 적용되어 깊은 모델에서도 안정적
    Chebyshev 필터로 더 효과적인 특징 추출
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=3, dropout=0.3, K=3):
        super(CAGERFCHEBV8, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.dropout = dropout
        self.K = K

        # Branch Layer: 각 relation별 독립적 ChebConv
        self.relation_convs = nn.ModuleList([
            nn.ModuleList([
                ChebConv(in_channels, hidden_channels, K),
                ChebConv(hidden_channels, hidden_channels, K),
                ChebConv(hidden_channels, hidden_channels, K),
            ]) for _ in range(num_relations)
        ])

        # Skip connection 처리를 위한 projection
        self.skip_proj = nn.Linear(in_channels + hidden_channels, hidden_channels)

        # Gating Mechanism: 각 relation의 중요도 학습
        self.gating = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Softmax(dim=0)
        )

        # Auxiliary Loss를 위한 relation-specific classifiers
        self.aux_classifiers = nn.ModuleList([
            nn.Linear(hidden_channels, out_channels) for _ in range(num_relations)
        ])

        # Main Classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_indices, edge_weights=None, training=True):
        """
        x: Input features (N, in_channels)
        edge_indices: List of edge tensors for each relation
        edge_weights: Optional edge weights (for Chebyshev)
        training: bool, whether computing auxiliary loss
        """
        relation_embeddings = []
        aux_logits = []

        for rel_idx, conv_layers in enumerate(self.relation_convs):
            x_rel = x
            x_out = x

            if edge_weights is not None:
                edge_weight = edge_weights[rel_idx]
            else:
                edge_weight = None

            # 3 ChebConv layers with skip connections
            for layer_idx, conv in enumerate(conv_layers):
                if edge_weight is not None:
                    x_rel = conv(x_rel, edge_indices[rel_idx], edge_weight)
                else:
                    x_rel = conv(x_rel, edge_indices[rel_idx])

                x_rel = F.relu(x_rel)
                x_rel = F.dropout(x_rel, p=self.dropout, training=self.training)

                # Skip connection
                if layer_idx > 0:
                    x_combined = torch.cat([x_out, x_rel], dim=1)
                    x_rel = self.skip_proj(x_combined)

            relation_embeddings.append(x_rel)

            if training:
                aux_logits.append(self.aux_classifiers[rel_idx](x_rel))

        # Gating mechanism
        relation_embeddings = torch.stack(relation_embeddings)

        gate_weights = []
        for h in relation_embeddings:
            w = self.gating(h)
            gate_weights.append(w)

        gate_weights = torch.cat(gate_weights, dim=1)

        # Weighted fusion
        h_fused = torch.sum(
            relation_embeddings.transpose(0, 1) * gate_weights.unsqueeze(2),
            dim=1
        )

        # Main classifier
        logits = self.classifier(h_fused)

        return logits, aux_logits, h_fused

    def compute_loss(self, logits, aux_logits, y, criterion_main, criterion_aux, aux_weight=0.3):
        """
        Main Loss + Auxiliary Loss
        """
        main_loss = criterion_main(logits, y)

        aux_loss = torch.stack([
            criterion_aux(aux_logit, y) for aux_logit in aux_logits
        ]).mean()

        total_loss = main_loss + aux_weight * aux_loss

        return total_loss, main_loss, aux_loss

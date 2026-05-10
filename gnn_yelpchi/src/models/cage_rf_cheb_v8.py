"""
CAGE-RF CHEB v8: Skip Connection with Chebyshev Filter (Over-smoothing 해결)
입력 정보를 각 레이어 출력과 concatenate하여 over-smoothing 방지
Chebyshev 그래프 필터로 더 효과적인 신호 처리
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class CAGERFCHEBV8(nn.Module):
    """
    CAGE-RF with Skip Connections and Chebyshev Filter

    Skip Connection이 각 레이어마다 적용되어 깊은 모델에서도 안정적
    Chebyshev 필터로 더 효과적인 특징 추출
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=6, dropout=0.3, K=3):
        super(CAGERFCHEBV8, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.dropout = dropout
        self.K = K  # Chebyshev order

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

        # Gating Mechanism: 각 relation의 중요도 학습 (softmax over relations)
        self.gating = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
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
            x_out = x  # Skip connection source

            # Get lambda_max for this relation's graph
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

                # Skip connection: concatenate with input and project
                if layer_idx > 0:
                    x_combined = torch.cat([x_out, x_rel], dim=1)
                    x_rel = self.skip_proj(x_combined)

            relation_embeddings.append(x_rel)

            # Auxiliary loss computation
            if training:
                aux_logits.append(self.aux_classifiers[rel_idx](x_rel))

        # Gating mechanism: compute weights for each relation
        # (num_relations, N, hidden_channels) → (N, num_relations, hidden_channels)
        relation_stack = torch.stack(relation_embeddings, dim=1)  # (N, num_relations, hidden)

        # Compute gating logits per relation, then softmax over relations
        gate_logits = self.gating(relation_stack)  # (N, num_relations, 1)
        alpha = torch.softmax(gate_logits, dim=1)  # softmax over relations

        # Weighted fusion
        h_fused = (alpha * relation_stack).sum(dim=1)  # (N, hidden_channels)

        # Main classifier
        logits = self.classifier(h_fused)

        return logits, aux_logits, h_fused

    def compute_loss(self, logits, aux_logits, y, criterion_main, criterion_aux, aux_weight=0.3):
        """
        Main Loss + Auxiliary Loss (both use Focal Loss for binary classification)
        """
        # Main loss
        main_loss = criterion_main(logits, y)

        # Auxiliary loss (use same criterion as main for binary classification)
        if len(aux_logits) > 0:
            aux_loss = torch.stack([
                criterion_main(aux_logit.squeeze(-1) if aux_logit.shape[-1] == 1 else aux_logit, y)
                for aux_logit in aux_logits
            ]).mean()
        else:
            aux_loss = torch.tensor(0.0, device=logits.device)

        # Combined loss
        total_loss = main_loss + aux_weight * aux_loss

        return total_loss, main_loss, aux_loss

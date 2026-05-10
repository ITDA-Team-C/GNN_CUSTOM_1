"""
CAGE-RF CHEB v9: Two-Stage GNN with Chebyshev Filter (최고 성능)
Stage 1: Gating 가중치 학습
Stage 2: 학습한 가중치로 embedding 정제

최고 성능 달성 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class CAGERFCHEBV9(nn.Module):
    """
    CAGE-RF Two-Stage GNN with Chebyshev Filter

    Stage 1: Branch layers + Gating 가중치 학습
    Stage 2: 학습된 가중치로 adaptive embedding refinement
    Chebyshev 필터로 더 효과적인 신호 처리
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=6, dropout=0.3, K=3):
        super(CAGERFCHEBV9, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.dropout = dropout
        self.K = K  # Chebyshev order

        # ===== STAGE 1: Branch Layer =====
        self.relation_convs = nn.ModuleList([
            nn.ModuleList([
                ChebConv(in_channels, hidden_channels, K),
                ChebConv(hidden_channels, hidden_channels, K),
                ChebConv(hidden_channels, hidden_channels, K),
            ]) for _ in range(num_relations)
        ])

        # Gating Mechanism: 각 relation의 중요도 학습 (softmax over relations applied in forward)
        self.gating = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

        # ===== STAGE 2: Refinement Layer =====
        # Gating 가중치로 weighted embedding을 다시 정제
        self.refine_convs = nn.ModuleList([
            ChebConv(hidden_channels, hidden_channels, K)
            for _ in range(num_relations)
        ])

        # Refined gating (Stage 2) - softmax over relations applied in forward
        self.refined_gating = nn.Sequential(
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
        # ===== STAGE 1: Initial Embedding =====
        relation_embeddings = []
        aux_logits = []

        for rel_idx, conv_layers in enumerate(self.relation_convs):
            x_rel = x

            if edge_weights is not None:
                edge_weight = edge_weights[rel_idx]
            else:
                edge_weight = None

            # 3 ChebConv layers
            for conv in conv_layers:
                if edge_weight is not None:
                    x_rel = conv(x_rel, edge_indices[rel_idx], edge_weight)
                else:
                    x_rel = conv(x_rel, edge_indices[rel_idx])

                x_rel = F.relu(x_rel)
                x_rel = F.dropout(x_rel, p=self.dropout, training=self.training)

            relation_embeddings.append(x_rel)

            # Auxiliary loss computation (Stage 1)
            if training:
                aux_logits.append(self.aux_classifiers[rel_idx](x_rel))

        # Stage 1 Gating: softmax over relations
        relation_stack_s1 = torch.stack(relation_embeddings, dim=1)  # (N, num_relations, hidden)
        gate_logits_s1 = self.gating(relation_stack_s1)  # (N, num_relations, 1)
        alpha_s1 = torch.softmax(gate_logits_s1, dim=1)  # softmax over relations

        # Stage 1 Fusion
        h_fused_s1 = (alpha_s1 * relation_stack_s1).sum(dim=1)  # (N, hidden)

        # ===== STAGE 2: Adaptive Refinement =====
        relation_embeddings_s2 = []

        for rel_idx, refine_conv in enumerate(self.refine_convs):
            # Use Stage 1 fused embedding as input for refinement
            if edge_weights is not None:
                edge_weight = edge_weights[rel_idx]
            else:
                edge_weight = None

            if edge_weight is not None:
                x_s2 = refine_conv(h_fused_s1, edge_indices[rel_idx], edge_weight)
            else:
                x_s2 = refine_conv(h_fused_s1, edge_indices[rel_idx])

            x_s2 = F.relu(x_s2)
            x_s2 = F.dropout(x_s2, p=self.dropout, training=self.training)

            relation_embeddings_s2.append(x_s2)

        # Stage 2 Gating: softmax over relations
        relation_stack_s2 = torch.stack(relation_embeddings_s2, dim=1)  # (N, num_relations, hidden)
        gate_logits_s2 = self.refined_gating(relation_stack_s2)  # (N, num_relations, 1)
        alpha_s2 = torch.softmax(gate_logits_s2, dim=1)  # softmax over relations

        # Stage 2 Fusion (Final refined embedding)
        h_fused_s2 = (alpha_s2 * relation_stack_s2).sum(dim=1)  # (N, hidden)

        # Combine Stage 1 and Stage 2 (skip connection between stages)
        h_final = h_fused_s2 + h_fused_s1  # Residual connection between stages

        # Main classifier
        logits = self.classifier(h_final)

        return logits, aux_logits, h_final

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

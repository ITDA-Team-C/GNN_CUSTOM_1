"""
CAGE-RF CHEB v9: Two-Stage GNN with Chebyshev Filter (최고 성능)
YelpChi용 - 3 relations (R-U-R, R-T-R, R-S-R)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class CAGERFCHEBV9(nn.Module):
    """
    CAGE-RF Two-Stage GNN with Chebyshev Filter (YelpChi)

    Stage 1: Branch layers + Gating 가중치 학습
    Stage 2: 학습된 가중치로 adaptive embedding refinement
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=3, dropout=0.3, K=3):
        super(CAGERFCHEBV9, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.dropout = dropout
        self.K = K

        # ===== STAGE 1: Branch Layer =====
        self.relation_convs = nn.ModuleList([
            nn.ModuleList([
                ChebConv(in_channels, hidden_channels, K),
                ChebConv(hidden_channels, hidden_channels, K),
                ChebConv(hidden_channels, hidden_channels, K),
            ]) for _ in range(num_relations)
        ])

        # Gating Mechanism
        self.gating = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Softmax(dim=0)
        )

        # ===== STAGE 2: Refinement Layer =====
        self.refine_convs = nn.ModuleList([
            ChebConv(hidden_channels, hidden_channels, K)
            for _ in range(num_relations)
        ])

        # Refined gating (Stage 2)
        self.refined_gating = nn.Sequential(
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
        edge_weights: Optional edge weights
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

        # Stage 1 Gating weights
        relation_embeddings_s1 = torch.stack(relation_embeddings)

        gate_weights_s1 = []
        for h in relation_embeddings_s1:
            w = self.gating(h)
            gate_weights_s1.append(w)

        gate_weights_s1 = torch.cat(gate_weights_s1, dim=1)

        # Stage 1 Fusion
        h_fused_s1 = torch.sum(
            relation_embeddings_s1.transpose(0, 1) * gate_weights_s1.unsqueeze(2),
            dim=1
        )

        # ===== STAGE 2: Adaptive Refinement =====
        relation_embeddings_s2 = []

        for rel_idx, refine_conv in enumerate(self.refine_convs):
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

        # Stage 2 Gating weights (refined)
        relation_embeddings_s2 = torch.stack(relation_embeddings_s2)

        gate_weights_s2 = []
        for h in relation_embeddings_s2:
            w = self.refined_gating(h)
            gate_weights_s2.append(w)

        gate_weights_s2 = torch.cat(gate_weights_s2, dim=1)

        # Stage 2 Fusion
        h_fused_s2 = torch.sum(
            relation_embeddings_s2.transpose(0, 1) * gate_weights_s2.unsqueeze(2),
            dim=1
        )

        # Combine Stage 1 and Stage 2 (residual connection)
        h_final = h_fused_s2 + h_fused_s1

        # Main classifier
        logits = self.classifier(h_final)

        return logits, aux_logits, h_final

    def compute_loss(self, logits, aux_logits, y, criterion_main, criterion_aux, aux_weight=0.3):
        """
        Main Loss + Auxiliary Loss (both use Focal Loss for binary classification)
        """
        main_loss = criterion_main(logits, y)

        # Auxiliary loss (use same criterion as main for binary classification)
        if len(aux_logits) > 0:
            aux_loss = torch.stack([
                criterion_main(aux_logit.squeeze(-1) if aux_logit.shape[-1] == 1 else aux_logit, y)
                for aux_logit in aux_logits
            ]).mean()
        else:
            aux_loss = torch.tensor(0.0, device=logits.device)

        total_loss = main_loss + aux_weight * aux_loss

        return total_loss, main_loss, aux_loss

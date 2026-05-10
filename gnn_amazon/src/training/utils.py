"""
Training utilities: Focal Loss, metrics, full-batch training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        p_t = p * targets.float() + (1 - p) * (1 - targets.float())
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def compute_metrics(y_true, y_pred_proba):
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    thresholds = np.arange(0, 1, 0.01)
    f1_scores = []
    for th in thresholds:
        y_pred = (y_pred_proba >= th).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_scores.append(f1)

    best_f1 = max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]

    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'macro_f1': best_f1,
        'best_threshold': best_threshold,
    }


def train_epoch(model, optimizer, criterion, criterion_aux,
                features, labels, edge_indices, train_idx, aux_weight=0.3):
    """Full-batch training"""
    model.train()
    optimizer.zero_grad()

    logits, aux_logits, _ = model(features, edge_indices, training=True)
    logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits

    train_logits = logits[train_idx]
    train_labels = labels[train_idx].float()

    if hasattr(model, 'compute_loss') and aux_logits:
        train_aux_logits = [al[train_idx] if al.dim() > 1 and al.shape[0] == logits.shape[0] else al for al in aux_logits]
        loss, _, _ = model.compute_loss(train_logits, train_aux_logits, train_labels,
                                        criterion, criterion_aux, aux_weight)
    else:
        loss = criterion(train_logits, train_labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, features, labels, edge_indices, eval_idx):
    """Full-batch evaluation"""
    model.eval()

    logits, _, _ = model(features, edge_indices, training=False)
    logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits

    eval_logits = logits[eval_idx]
    eval_labels = labels[eval_idx]

    proba = torch.sigmoid(eval_logits).cpu().numpy()
    y_true = eval_labels.cpu().numpy()

    metrics = compute_metrics(y_true, proba)
    return metrics, y_true, proba

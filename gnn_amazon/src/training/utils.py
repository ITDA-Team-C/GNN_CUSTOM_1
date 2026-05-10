"""
Training utilities: Focal Loss, metrics, etc.
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


def train_epoch(model, optimizer, criterion, criterion_aux, train_loader, device, aux_weight=0.3):
    model.train()
    total_loss = 0.0
    batch_count = 0

    for x, y, edge_indices in train_loader:
        x = x.to(device)
        y = y.to(device).long()
        edge_indices = [ei.to(device) for ei in edge_indices]

        optimizer.zero_grad()

        logits, aux_logits, _ = model(x, edge_indices, training=True)
        logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits

        if hasattr(model, 'compute_loss'):
            loss, _, _ = model.compute_loss(logits, aux_logits, y, criterion, criterion_aux, aux_weight)
        else:
            loss = criterion(logits.view(-1), y.view(-1).float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / batch_count


@torch.no_grad()
def evaluate(model, eval_loader, device):
    model.eval()
    y_true = []
    y_pred_proba = []

    for x, y, edge_indices in eval_loader:
        x = x.to(device)
        y = y.to(device)
        edge_indices = [ei.to(device) for ei in edge_indices]

        logits, _, _ = model(x, edge_indices, training=False)
        logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits

        proba = torch.sigmoid(logits).cpu().numpy()

        y_true.extend(y.cpu().numpy())
        y_pred_proba.extend(proba)

    metrics = compute_metrics(y_true, y_pred_proba)
    return metrics, np.array(y_true), np.array(y_pred_proba)

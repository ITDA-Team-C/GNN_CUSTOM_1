"""
Complete training pipeline for all 10 Amazon GNN models (Full-batch GNN training)
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from src.training.utils import FocalLoss, train_epoch, evaluate
from src.models.base_models import (
    MLPModel, GCNModel, SAGEModel, GATModel,
    GraphConvModel, ChebModel, TAGModel, SGModel
)
from src.models.cage_rf_cheb_v8 import CAGERFCHEBV8
from src.models.cage_rf_cheb_v9 import CAGERFCHEBV9
from src.utils import set_seed

set_seed(42)


def load_data():
    """Load preprocessed Amazon data"""
    features = np.load("data/processed/features.npy")
    labels = np.load("data/processed/labels.npy")
    edge_data = torch.load("data/processed/edge_index_dict.pt")
    split_data = torch.load("data/processed/split_idx.pt")

    edge_indices = [edge_data['net_upu'], edge_data['net_usu'],
                    edge_data['net_uvu'], edge_data['homo']]

    train_idx = split_data['train_idx'].numpy()
    val_idx = split_data['val_idx'].numpy()
    test_idx = split_data['test_idx'].numpy()

    return features, labels, edge_indices, train_idx, val_idx, test_idx


def validate_edge_indices(edge_indices, num_nodes, features_shape):
    """Validate edge indices"""
    relation_names = ['net_upu', 'net_usu', 'net_uvu', 'homo']
    print(f"\n[Edge Index Validation]")
    print(f"  num_nodes={num_nodes}, features_shape={features_shape}")

    for i, (ei, name) in enumerate(zip(edge_indices, relation_names)):
        if ei.numel() == 0:
            print(f"  {name}: EMPTY!")
            continue

        max_src = ei[0].max().item() if ei.shape[1] > 0 else -1
        min_src = ei[0].min().item() if ei.shape[1] > 0 else -1
        max_dst = ei[1].max().item() if ei.shape[1] > 0 else -1
        min_dst = ei[1].min().item() if ei.shape[1] > 0 else -1

        print(f"  {name}: shape={ei.shape}, src=[{min_src},{max_src}], dst=[{min_dst},{max_dst}]")

        if max_src >= num_nodes or max_dst >= num_nodes or min_src < 0 or min_dst < 0:
            print(f"    ERROR: Out of bounds! Expected [0, {num_nodes-1}]")
            mask = (ei[0] >= 0) & (ei[0] < num_nodes) & (ei[1] >= 0) & (ei[1] < num_nodes)
            invalid_count = (~mask).sum().item()
            edge_indices[i] = ei[:, mask]
            print(f"    Fixed: removed {invalid_count} invalid edges, new shape={edge_indices[i].shape}")


def train_model(model, device, features, labels, edge_indices, train_idx, val_idx, test_idx,
                epochs=200, patience=20, aux_weight=0.3):
    """Train single model with full-batch"""
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    model.to(device)
    best_val_pr_auc = 0
    patience_counter = 0
    best_metrics = None
    best_state = None

    for epoch in range(epochs):
        avg_loss = train_epoch(model, optimizer, criterion, criterion,
                              features, labels, edge_indices, train_idx, aux_weight)

        val_metrics, _, _ = evaluate(model, features, labels, edge_indices, val_idx)
        val_pr_auc = val_metrics['pr_auc']

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            test_metrics, _, _ = evaluate(model, features, labels, edge_indices, test_idx)
            best_metrics = test_metrics
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val PR-AUC: {val_pr_auc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return best_metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    os.makedirs("outputs/output_amazon/models", exist_ok=True)
    os.makedirs("outputs/output_amazon/results", exist_ok=True)

    print("Loading data...")
    features, labels, edge_indices, train_idx, val_idx, test_idx = load_data()

    print(f"Features: {features.shape}, Labels: {labels.shape}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}\n")

    print("Validating edge indices...")
    num_nodes = len(labels)
    validate_edge_indices(edge_indices, num_nodes, features.shape)
    print()

    # Move all data to device once (full-batch GNN)
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).long().to(device)
    edge_indices = [ei.to(device) for ei in edge_indices]
    train_idx = torch.from_numpy(train_idx).long().to(device)
    val_idx = torch.from_numpy(val_idx).long().to(device)
    test_idx = torch.from_numpy(test_idx).long().to(device)

    in_channels = features.shape[1]
    hidden_channels = 64
    out_channels = 1
    num_relations = 4

    models = {
        'MLP': MLPModel(in_channels, hidden_channels, out_channels),
        'GCN': GCNModel(in_channels, hidden_channels, out_channels, num_relations),
        'SAGE': SAGEModel(in_channels, hidden_channels, out_channels, num_relations),
        'GAT': GATModel(in_channels, hidden_channels, out_channels, num_relations),
        'GRAPHCONV': GraphConvModel(in_channels, hidden_channels, out_channels, num_relations),
        'CHEB': ChebModel(in_channels, hidden_channels, out_channels, num_relations),
        'TAG': TAGModel(in_channels, hidden_channels, out_channels, num_relations),
        'SG': SGModel(in_channels, hidden_channels, out_channels, num_relations),
        'V8': CAGERFCHEBV8(in_channels, hidden_channels, out_channels, num_relations),
        'V9': CAGERFCHEBV9(in_channels, hidden_channels, out_channels, num_relations),
    }

    results = {}

    print("Training models...")
    for name, model in models.items():
        print(f"\n{name}...")
        metrics = train_model(model, device, features, labels, edge_indices, train_idx, val_idx, test_idx)
        results[name] = metrics
        torch.save(model.state_dict(), f"outputs/output_amazon/models/{name}.pt")
        print(f"  Test: PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"outputs/output_amazon/results/results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    for name in sorted(results.keys()):
        m = results[name]
        print(f"{name:15s} | PR-AUC: {m['pr_auc']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | Macro-F1: {m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

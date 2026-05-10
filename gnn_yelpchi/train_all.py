"""
Complete training pipeline for all 10 YelpChi GNN models
8 benchmarks (MLP, GCN, SAGE, GAT, GRAPHCONV, CHEB, TAG, SG) + v8 + v9
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from src.training.dataloader import create_data_loaders
from src.training.utils import FocalLoss, compute_metrics, train_epoch, evaluate
from src.models.base_models import (
    MLPModel, GCNModel, SAGEModel, GATModel,
    GraphConvModel, ChebModel, TAGModel, SGModel
)
from src.models.cage_rf_cheb_v8 import CAGERFCHEBV8
from src.models.cage_rf_cheb_v9 import CAGERFCHEBV9
from src.utils import set_seed

set_seed(42)


def load_preprocessed_data():
    """Load preprocessed YelpChi data"""
    features = np.load("data/processed/features.npy")
    labels = np.load("data/processed/labels.npy")

    edge_data = torch.load("data/processed/edge_index_dict.pt")
    edge_indices = [edge_data['rur'], edge_data['rtr'], edge_data['rsr']]

    split_data = torch.load("data/processed/split_idx.pt")
    train_idx = split_data['train_idx'].numpy()
    val_idx = split_data['val_idx'].numpy()
    test_idx = split_data['test_idx'].numpy()

    return features, labels, edge_indices, train_idx, val_idx, test_idx


def get_device():
    """Get GPU device if available"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train_model(model, device, train_loader, val_loader, test_loader,
                epochs=200, patience=20, aux_weight=0.3):
    """
    Train a single model with early stopping
    Returns: best metrics on test set
    """
    criterion_main = FocalLoss(alpha=0.75, gamma=2.0)
    criterion_aux = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    model.to(device)
    best_val_pr_auc = 0
    patience_counter = 0
    best_metrics = None
    best_model_state = None

    for epoch in range(epochs):
        avg_loss = train_epoch(model, optimizer, criterion_main, criterion_aux,
                              train_loader, device, aux_weight)

        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_pr_auc = val_metrics['pr_auc']

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            test_metrics, _, _ = evaluate(model, test_loader, device)
            best_metrics = test_metrics
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val PR-AUC: {val_pr_auc:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_metrics


def main():
    device = get_device()
    print(f"Using device: {device}\n")

    os.makedirs("outputs/output_yelpchi/models", exist_ok=True)
    os.makedirs("outputs/output_yelpchi/results", exist_ok=True)

    print("Loading preprocessed data...")
    features, labels, edge_indices, train_idx, val_idx, test_idx = load_preprocessed_data()

    in_channels = features.shape[1]
    hidden_channels = 64
    out_channels = 2
    num_relations = 3

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of relations: {num_relations}\n")

    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        features, labels, edge_indices, train_idx, val_idx, test_idx, batch_size=256
    )

    models_config = {
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*80)
    print("Training all models")
    print("="*80 + "\n")

    for model_name, model in models_config.items():
        print(f"Training {model_name}...")

        metrics = train_model(model, device, train_loader, val_loader, test_loader)
        results[model_name] = metrics

        torch.save(model.state_dict(), f"outputs/output_yelpchi/models/{model_name}.pt")
        print(f"  Test PR-AUC: {metrics['pr_auc']:.4f}, "
              f"ROC-AUC: {metrics['roc_auc']:.4f}, "
              f"Macro-F1: {metrics['macro_f1']:.4f}\n")

    results_path = f"outputs/output_yelpchi/results/results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "="*80)
    print("Test Set Performance Summary")
    print("="*80)
    for model_name in sorted(results.keys()):
        m = results[model_name]
        print(f"{model_name:15s} | PR-AUC: {m['pr_auc']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | Macro-F1: {m['macro_f1']:.4f}")

    best_pr_auc_model = max(results.items(), key=lambda x: x[1]['pr_auc'])
    best_roc_auc_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
    best_f1_model = max(results.items(), key=lambda x: x[1]['macro_f1'])

    print(f"\nBest PR-AUC: {best_pr_auc_model[0]} ({best_pr_auc_model[1]['pr_auc']:.4f})")
    print(f"Best ROC-AUC: {best_roc_auc_model[0]} ({best_roc_auc_model[1]['roc_auc']:.4f})")
    print(f"Best Macro-F1: {best_f1_model[0]} ({best_f1_model[1]['macro_f1']:.4f})")


if __name__ == "__main__":
    main()

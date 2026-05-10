"""
Test and compare all 10 Amazon GNN models (Full-batch)
"""
import os
import json
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.training.utils import evaluate
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

    expected_keys = ['net_upu', 'net_usu', 'net_uvu', 'homo']
    edge_indices = []
    for key in expected_keys:
        if key in edge_data:
            edge_indices.append(edge_data[key])
        else:
            raise KeyError(f"Edge key '{key}' not found. Available keys: {list(edge_data.keys())}")

    train_idx = split_data['train_idx'].numpy()
    val_idx = split_data['val_idx'].numpy()
    test_idx = split_data['test_idx'].numpy()

    return features, labels, edge_indices, train_idx, val_idx, test_idx


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    print("Loading data...")
    features, labels, edge_indices, train_idx, val_idx, test_idx = load_data()

    # Normalize features using train set statistics only
    scaler = StandardScaler()
    scaler.fit(features[train_idx])
    features = scaler.transform(features).astype(np.float32)

    # Move all data to device
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).long().to(device)
    edge_indices = [ei.to(device) for ei in edge_indices]
    test_idx = torch.from_numpy(test_idx).long().to(device)

    in_channels = features.shape[1]
    hidden_channels = 32
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

    print("Evaluating models...\n")
    for name, model in models.items():
        path = f"outputs/output_amazon/models/{name}.pt"
        if not os.path.exists(path):
            print(f"{name}: Model not found")
            continue

        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        metrics, _, _ = evaluate(model, features, labels, edge_indices, test_idx)
        results[name] = metrics
        print(f"{name:15s} | PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")

    os.makedirs("outputs/output_amazon/results", exist_ok=True)
    with open("outputs/output_amazon/results/test_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved!")


if __name__ == "__main__":
    main()

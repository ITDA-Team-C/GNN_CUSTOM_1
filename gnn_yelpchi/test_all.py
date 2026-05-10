"""
Test and compare all 10 trained YelpChi GNN models
"""
import os
import json
import numpy as np
import torch
from tabulate import tabulate

from src.training.dataloader import create_data_loaders
from src.training.utils import evaluate
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


def main():
    device = get_device()
    print(f"Using device: {device}\n")

    print("Loading preprocessed data...")
    features, labels, edge_indices, train_idx, val_idx, test_idx = load_preprocessed_data()

    in_channels = features.shape[1]
    hidden_channels = 64
    out_channels = 2
    num_relations = 3

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}\n")

    print("Creating data loaders...")
    _, _, test_loader = create_data_loaders(
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

    print("\n" + "="*80)
    print("Evaluating all models on test set")
    print("="*80 + "\n")

    for model_name, model in models_config.items():
        model_path = f"outputs/output_yelpchi/models/{model_name}.pt"

        if not os.path.exists(model_path):
            print(f"Warning: {model_name} model not found at {model_path}")
            continue

        print(f"Loading and evaluating {model_name}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        metrics, y_true, y_pred_proba = evaluate(model, test_loader, device)
        results[model_name] = metrics

        print(f"  PR-AUC: {metrics['pr_auc']:.4f}, "
              f"ROC-AUC: {metrics['roc_auc']:.4f}, "
              f"Macro-F1: {metrics['macro_f1']:.4f}\n")

    print("="*80)
    print("Test Set Performance Comparison")
    print("="*80 + "\n")

    table_data = []
    for model_name in sorted(results.keys()):
        m = results[model_name]
        table_data.append([
            model_name,
            f"{m['pr_auc']:.4f}",
            f"{m['roc_auc']:.4f}",
            f"{m['macro_f1']:.4f}",
            f"{m['best_threshold']:.4f}"
        ])

    headers = ["Model", "PR-AUC", "ROC-AUC", "Macro-F1", "Best Threshold"]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    print("\nRankings by Metric:")
    print("-" * 40)

    pr_auc_ranking = sorted(results.items(), key=lambda x: x[1]['pr_auc'], reverse=True)
    print("\nPR-AUC Ranking:")
    for rank, (name, m) in enumerate(pr_auc_ranking[:5], 1):
        print(f"  {rank}. {name}: {m['pr_auc']:.4f}")

    roc_auc_ranking = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    print("\nROC-AUC Ranking:")
    for rank, (name, m) in enumerate(roc_auc_ranking[:5], 1):
        print(f"  {rank}. {name}: {m['roc_auc']:.4f}")

    f1_ranking = sorted(results.items(), key=lambda x: x[1]['macro_f1'], reverse=True)
    print("\nMacro-F1 Ranking:")
    for rank, (name, m) in enumerate(f1_ranking[:5], 1):
        print(f"  {rank}. {name}: {m['macro_f1']:.4f}")

    os.makedirs("outputs/output_yelpchi/results", exist_ok=True)
    with open("outputs/output_yelpchi/results/test_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: outputs/output_yelpchi/results/test_comparison.json")


if __name__ == "__main__":
    main()

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import set_seed, load_config, save_object, save_json
from src.utils.metrics import calculate_metrics, find_best_threshold, print_metrics
from src.models.baseline_mlp import MLP
from src.models.baseline_gcn import GCN
from src.models.baseline_graphsage import GraphSAGE
from src.models.baseline_gat import GAT
from src.models.cage_rf_gnn import CAGERF_GNN
from src.models.losses import WeightedBCELoss

set_seed(42)


def load_graph_data(config):
    processed_dir = config["data_processed"]

    nodes_df = pd.read_csv(os.path.join(processed_dir, "node_samples.csv"))
    features = np.load(os.path.join(processed_dir, "features.npy"))

    edge_index_dict = torch.load(os.path.join(processed_dir, "edge_index_dict.pt"))

    x = torch.FloatTensor(features)
    y = torch.LongTensor(nodes_df["label"].values)

    train_mask = torch.BoolTensor(nodes_df["split"] == "train")
    valid_mask = torch.BoolTensor(nodes_df["split"] == "valid")
    test_mask = torch.BoolTensor(nodes_df["split"] == "test")

    print(f"[Data] 노드 수: {len(x)}")
    print(f"[Data] Feature 차원: {x.shape[1]}")
    print(f"[Data] Train: {train_mask.sum()} | Valid: {valid_mask.sum()} | Test: {test_mask.sum()}")
    print(f"[Data] Label 분포: {np.bincount(y.numpy())}")

    return x, y, edge_index_dict, train_mask, valid_mask, test_mask, nodes_df


def create_model(model_name, input_dim, config):
    if model_name == "mlp":
        return MLP(input_dim, hidden_dim=128, dropout=0.3)
    elif model_name == "gcn":
        return GCN(input_dim, hidden_dim=64, num_layers=2, dropout=0.3)
    elif model_name == "graphsage":
        return GraphSAGE(input_dim, hidden_dim=64, num_layers=2, dropout=0.3)
    elif model_name == "gat":
        return GAT(input_dim, hidden_dim=64, num_layers=2, num_heads=8, dropout=0.3)
    elif model_name == "cage_rf_gnn":
        return CAGERF_GNN(input_dim, hidden_dim=64, num_layers=2, dropout=0.3, use_gating=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def calculate_pos_weight(y, train_mask):
    pos_count = (y[train_mask] == 1).sum().item()
    neg_count = (y[train_mask] == 0).sum().item()
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    return torch.tensor(pos_weight)


def train_epoch(model, x, y, edge_index_dict, train_mask, optimizer, loss_fn, device):
    model.train()

    logits = model(x.to(device), {k: v.to(device) for k, v in edge_index_dict.items()})
    loss = loss_fn(logits[train_mask], y[train_mask].to(device))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, x, y, edge_index_dict, mask, device):
    model.eval()

    logits = model(x.to(device), {k: v.to(device) for k, v in edge_index_dict.items()})
    logits = logits.detach().cpu().numpy()

    y_score = 1 / (1 + np.exp(-logits))
    y_pred = (y_score >= 0.5).astype(int)

    y_true = y[mask].numpy()

    metrics = calculate_metrics(y_true, y_score[mask.numpy()], y_pred[mask.numpy()])

    return metrics, y_score, y_pred


def train(model_name, config_path):
    config = load_config(config_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA not available, falling back to CPU")

    print(f"[Train] 모델: {model_name.upper()}")
    print(f"[Device] {device}")

    x, y, edge_index_dict, train_mask, valid_mask, test_mask, nodes_df = load_graph_data(config)

    x = x.to(device)
    y = y.to(device)
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

    model = create_model(model_name, x.shape[1], config)
    model = model.to(device)

    pos_weight = calculate_pos_weight(y, train_mask)
    loss_fn = WeightedBCELoss(pos_weight=pos_weight.to(device))

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    print(f"\n[Training] 시작 (epochs={config['training']['num_epochs']})")

    best_valid_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config["training"]["num_epochs"]):
        loss = train_epoch(model, x, y, edge_index_dict, train_mask, optimizer, loss_fn, device)

        if (epoch + 1) % config["training"]["validation_interval"] == 0:
            valid_metrics, _, _ = evaluate(model, x, y, edge_index_dict, valid_mask, device)

            if valid_metrics["macro_f1"] > best_valid_f1:
                best_valid_f1 = valid_metrics["macro_f1"]
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(
                    f"  Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                    f"Valid F1: {valid_metrics['macro_f1']:.4f} | PR-AUC: {valid_metrics['pr_auc']:.4f} ✓"
                )
            else:
                patience_counter += 1
                print(
                    f"  Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                    f"Valid F1: {valid_metrics['macro_f1']:.4f} | PR-AUC: {valid_metrics['pr_auc']:.4f}"
                )

            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n[Evaluation]")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    valid_metrics, valid_scores, valid_preds = evaluate(
        model, x, y, edge_index_dict, valid_mask, device
    )
    test_metrics, test_scores, test_preds = evaluate(
        model, x, y, edge_index_dict, test_mask, device
    )

    print("\n[Valid Set]")
    print_metrics(valid_metrics, "Validation Metrics")

    best_threshold, _ = find_best_threshold(y[valid_mask].numpy(), valid_scores[valid_mask.numpy()])

    test_preds_thresholded = (test_scores[test_mask.numpy()] >= best_threshold).astype(int)
    test_metrics_thresholded = calculate_metrics(
        y[test_mask].numpy(), test_scores[test_mask.numpy()], test_preds_thresholded
    )

    print("\n[Test Set]")
    print_metrics(test_metrics_thresholded, "Test Metrics (with best threshold)")

    os.makedirs("outputs", exist_ok=True)

    model_path = os.path.join("outputs", f"best_model_{model_name}.pt")
    torch.save(best_model_state, model_path)
    print(f"\n[Save] {model_path}")

    metrics_dict = {
        "model": model_name,
        "best_threshold": best_threshold,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics_thresholded,
    }
    metrics_path = os.path.join("outputs", f"metrics_{model_name}.json")
    save_json(metrics_dict, metrics_path)
    print(f"[Save] {metrics_path}")

    return test_metrics_thresholded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cage_rf_gnn",
                        choices=["mlp", "gcn", "graphsage", "gat", "cage_rf_gnn"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    metrics = train(args.model, args.config)

#!/usr/bin/env python3
"""
평가 및 추론 파이프라인 (One-Shot)
저장된 모델 로드 → Test Set 평가 → 결과 시각화 및 저장
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import set_seed, load_config, save_json
from src.utils.metrics import calculate_metrics, find_best_threshold, print_metrics
from src.models.cage_rf_gnn import CAGERF_GNN
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve

set_seed(42)


def load_test_data(config):
    """Load test data and model inputs"""
    print("[Load] 테스트 데이터 로드 중...")

    processed_dir = config["data_processed"]

    nodes_df = pd.read_csv(os.path.join(processed_dir, "node_samples.csv"))
    features = np.load(os.path.join(processed_dir, "features.npy"))
    edge_index_dict = torch.load(os.path.join(processed_dir, "edge_index_dict.pt"))

    x = torch.FloatTensor(features)
    y = torch.LongTensor(nodes_df["label"].values)

    test_mask = torch.BoolTensor(nodes_df["split"] == "test")
    valid_mask = torch.BoolTensor(nodes_df["split"] == "valid")

    print(f"  총 노드: {len(x)}")
    print(f"  Test set: {test_mask.sum()}")
    print(f"  Valid set: {valid_mask.sum()}")

    return x, y, edge_index_dict, test_mask, valid_mask, nodes_df


def load_model_and_best_threshold(checkpoint_path, config, device, input_dim):
    """Load trained model and best threshold"""
    print(f"\n[Load] 모델 로드 중: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = CAGERF_GNN(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        use_gating=True
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"  ✅ 모델 로드 완료")

    # Load best threshold
    threshold_results_path = os.path.join("outputs", "threshold_search_results.json")
    if os.path.exists(threshold_results_path):
        with open(threshold_results_path, 'r') as f:
            threshold_data = json.load(f)
            best_threshold = threshold_data["best_threshold"]
            print(f"  ✅ Best Threshold 로드: {best_threshold:.3f}")
    else:
        best_threshold = 0.5
        print(f"  ⚠️  Best Threshold 파일 없음, 기본값 0.5 사용")

    return model, best_threshold


@torch.no_grad()
def predict(model, x, edge_index_dict, device):
    """Generate predictions"""
    print("\n[Predict] 추론 중...")

    x = x.to(device)
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

    logits = model(x, edge_index_dict)
    logits = logits.detach().cpu().numpy()

    y_score = 1 / (1 + np.exp(-logits))

    print(f"  ✅ 추론 완료")

    return y_score


def evaluate_test_set(y_true, y_score, y_pred):
    """Evaluate on test set"""
    print("\n[Evaluate] 평가 중...")

    metrics = calculate_metrics(y_true, y_score, y_pred)

    print_metrics(metrics, "Test Set Metrics")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(cm)

    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["정상", "사기"]))

    return metrics, cm


def save_predictions(nodes_df, y_score, y_pred, best_threshold, output_path="outputs/predictions.csv"):
    """Save predictions to CSV"""
    print(f"\n[Save] 예측값 저장 중: {output_path}")

    results_df = nodes_df.copy()
    results_df["fraud_score"] = y_score
    results_df["pred_label"] = y_pred

    results_df = results_df[[
        "review_id", "user_id", "prod_id", "date", "rating",
        "label", "fraud_score", "pred_label", "split"
    ]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"  ✅ 저장 완료: {len(results_df)} rows")

    # Top suspicious reviews
    top_fraud = results_df.nlargest(10, "fraud_score")
    print(f"\n🔴 Top 10 Suspicious Reviews:")
    print(top_fraud[["review_id", "user_id", "prod_id", "fraud_score", "true_label", "pred_label"]].to_string())

    return results_df


def generate_summary_report(metrics, cm, best_threshold, model_name, output_path="outputs/test_summary.json"):
    """Generate summary report"""
    print(f"\n[Report] 요약 보고서 생성 중: {output_path}")

    summary = {
        "model": model_name,
        "best_threshold": float(best_threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1])
        },
        "timestamp": pd.Timestamp.now().isoformat()
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(summary, output_path)

    print(f"  ✅ 저장 완료")

    return summary


def print_final_summary(summary):
    """Print final summary"""
    print("\n" + "="*80)
    print("📊 최종 평가 결과")
    print("="*80)

    print(f"\n[모델] {summary['model'].upper()}")
    print(f"[Threshold] {summary['best_threshold']:.3f}")

    print(f"\n[성능 지표]")
    print(f"  PR-AUC:    {summary['metrics']['pr_auc']:.4f}")
    print(f"  Macro F1:  {summary['metrics']['macro_f1']:.4f}")
    print(f"  ROC-AUC:   {summary['metrics']['roc_auc']:.4f}")
    print(f"  Precision: {summary['metrics']['precision']:.4f}")
    print(f"  Recall:    {summary['metrics']['recall']:.4f}")

    print(f"\n[Confusion Matrix]")
    cm = summary['confusion_matrix']
    print(f"  TN: {cm['tn']:6d}  FP: {cm['fp']:6d}")
    print(f"  FN: {cm['fn']:6d}  TP: {cm['tp']:6d}")

    print(f"\n[결과 저장]")
    print(f"  📁 outputs/predictions.csv")
    print(f"  📁 outputs/test_summary.json")
    print(f"  📁 outputs/metrics_{summary['model']}.json")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inference & Evaluation Pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/best_model_cage_rf_gnn.pt",
        help="Checkpoint path to load"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cage_rf_gnn",
        help="Model name for logging"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔍 GNN-based Fraud Detection - Inference & Evaluation Pipeline")
    print("="*80)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n[Device] CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"\n[Device] Using CPU")

    config = load_config(args.config)

    try:
        # Load data
        x, y, edge_index_dict, test_mask, valid_mask, nodes_df = load_test_data(config)

        # Load model and threshold with actual input dimension
        actual_input_dim = x.shape[1]
        model, best_threshold = load_model_and_best_threshold(args.checkpoint, config, device, actual_input_dim)

        # Predict
        y_score = predict(model, x, edge_index_dict, device)

        # Prepare predictions for test set
        y_pred = (y_score >= best_threshold).astype(int)
        y_test = y[test_mask].numpy()
        y_score_test = y_score[test_mask.numpy()]
        y_pred_test = y_pred[test_mask.numpy()]

        # Evaluate
        metrics, cm = evaluate_test_set(y_test, y_score_test, y_pred_test)

        # Save predictions
        results_df = save_predictions(nodes_df, y_score, y_pred, best_threshold)

        # Generate report
        summary = generate_summary_report(metrics, cm, best_threshold, args.model_name)

        # Print final summary
        print_final_summary(summary)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
전체 모델 비교 스크립트 (v2~v7 + Baselines)
outputs/ 디렉토리의 metrics_*.json 파일들을 읽어 성능 비교 표 출력
"""
import json
import os
from pathlib import Path
import pandas as pd


def load_metrics(model_name, file_path):
    """metrics_*.json 파일에서 test metrics 추출"""
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get("test_metrics", {})
    except Exception as e:
        print(f"[WARNING] Failed to load {file_path}: {e}")
        return None


def main():
    output_dir = Path("outputs")

    if not output_dir.exists():
        print("[ERROR] outputs/ directory not found")
        return

    models_to_compare = [
        # CAGE-RF versions (v2~v7)
        ("CAGE-RF v2", "metrics_v2.json"),
        ("CAGE-RF v3", "metrics_v3_ablation.json"),
        ("CAGE-RF v4", "metrics_v4_threshold_pr.json"),
        ("CAGE-RF v5", "metrics_v5_oversampling.json"),
        ("CAGE-RF v6", "metrics_v6_hard_mining.json"),
        ("CAGE-RF v7", "metrics_v7_ensemble.json"),
        # Baselines
        ("MLP", "metrics_mlp.json"),
        ("GCN", "metrics_gcn.json"),
        ("GraphSAGE", "metrics_graphsage.json"),
        ("GAT", "metrics_gat.json"),
    ]

    results = {}
    found_count = 0

    print("\n" + "=" * 130)
    print("📊 Loading metrics from outputs/")
    print("=" * 130)

    for display_name, file_name in models_to_compare:
        file_path = output_dir / file_name
        metrics = load_metrics(display_name, file_path)

        if metrics:
            results[display_name] = metrics
            found_count += 1
            status = "✅"
        else:
            status = "⏭️ "
        print(f"{status} {display_name:20s} | {file_name}")

    print("=" * 130)

    if not results:
        print("[ERROR] No metrics files found!")
        return

    print(f"\n✅ Loaded {found_count}/{len(models_to_compare)} models\n")

    # Create comparison DataFrame
    df = pd.DataFrame(results).T

    # Sort by PR-AUC (descending) if available
    if "pr_auc" in df.columns:
        df = df.sort_values("pr_auc", ascending=False)

    # Display comparison table
    print("=" * 130)
    print("🏆 Model Performance Comparison (Test Set)")
    print("=" * 130)

    # Format numeric columns
    pd.options.display.float_format = "{:.4f}".format
    print(df.to_string())

    print("\n" + "=" * 130)
    print("📈 Top 3 Models by PR-AUC")
    print("=" * 130)

    if "pr_auc" in df.columns:
        top3 = df.nlargest(3, "pr_auc")
        for rank, (model_name, row) in enumerate(top3.iterrows(), 1):
            pr_auc = row.get("pr_auc", "N/A")
            macro_f1 = row.get("macro_f1", "N/A")
            print(f"{rank}. {model_name:20s} | PR-AUC: {pr_auc:.4f} | Macro-F1: {macro_f1:.4f}")

    print("=" * 130 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
전체 모델 비교 스크립트 (v2~v7 + Baselines + GNN Benchmarks)
outputs/ 디렉토리의 metrics_*.json 파일들을 읽어 성능 비교 표 출력
라벨 검증 포함
"""
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# UTF-8 인코딩 설정 (Windows 호환성)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def validate_labels():
    """라벨 데이터 검증"""
    print("\n" + "=" * 130)
    print("🔍 LABEL VALIDATION CHECK")
    print("=" * 130)

    processed_dir = Path("data/processed")
    node_samples_path = processed_dir / "node_samples.csv"

    if not node_samples_path.exists():
        print("⚠️  [SKIP] node_samples.csv not found")
        return True

    try:
        df = pd.read_csv(node_samples_path)

        # 1. 라벨 값 확인
        unique_labels = set(df['label'].unique())
        expected_labels = {0, 1}

        print(f"\n✓ Label values check:")
        print(f"  Found: {sorted(unique_labels)}")
        print(f"  Expected: {sorted(expected_labels)}")

        if unique_labels != expected_labels:
            print(f"  ❌ FAILED: Expected {expected_labels}, got {unique_labels}")
            return False
        print(f"  ✅ PASSED")

        # 2. NaN 확인
        nan_count = df['label'].isna().sum()
        print(f"\n✓ NaN check:")
        print(f"  NaN count: {nan_count}")
        if nan_count > 0:
            print(f"  ❌ FAILED: Found {nan_count} NaN values")
            return False
        print(f"  ✅ PASSED")

        # 3. 각 split별 라벨 분포
        print(f"\n✓ Label distribution by split:")
        for split in ['train', 'valid', 'test']:
            mask = df['split'] == split
            split_df = df[mask]
            pos = (split_df['label'] == 1).sum()
            neg = (split_df['label'] == 0).sum()
            total = len(split_df)
            ratio = pos / total * 100 if total > 0 else 0
            print(f"  [{split:5s}] Total: {total:5d} | Fraud(1): {pos:5d} ({ratio:5.2f}%) | Normal(0): {neg:5d} ({100-ratio:5.2f}%)")

        # 4. 전체 라벨 분포
        print(f"\n✓ Overall label distribution:")
        pos_total = (df['label'] == 1).sum()
        neg_total = (df['label'] == 0).sum()
        ratio_total = pos_total / len(df) * 100
        print(f"  Fraud(1): {pos_total:5d} ({ratio_total:5.2f}%)")
        print(f"  Normal(0): {neg_total:5d} ({100-ratio_total:5.2f}%)")
        print(f"  Imbalance ratio: {neg_total / pos_total:.2f}:1")

        # 5. Stratification 확인
        print(f"\n✓ Stratification check:")
        for split in ['train', 'valid', 'test']:
            mask = df['split'] == split
            split_df = df[mask]
            if len(split_df) > 0:
                ratio = (split_df['label'] == 1).sum() / len(split_df) * 100
                print(f"  [{split:5s}] Fraud ratio: {ratio:.2f}%")

        print(f"\n✅ ALL LABEL CHECKS PASSED")
        return True

    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


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

    # 라벨 검증 먼저 수행
    validate_labels()

    models_to_compare = [
        # CAGE-RF versions (v2~v7) - outputs/cage_rf_gnn/
        ("CAGE-RF v2", output_dir / "cage_rf_gnn" / "metrics_v2.json"),
        ("CAGE-RF v3", output_dir / "cage_rf_gnn" / "metrics_v3_ablation.json"),
        ("CAGE-RF v4", output_dir / "cage_rf_gnn" / "metrics_v4_threshold_pr.json"),
        ("CAGE-RF v5", output_dir / "cage_rf_gnn" / "metrics_v5_oversampling.json"),
        ("CAGE-RF v6", output_dir / "cage_rf_gnn" / "metrics_v6_hard_mining.json"),
        ("CAGE-RF v7", output_dir / "cage_rf_gnn" / "metrics_v7_ensemble.json"),
        # Baselines - outputs/cage_rf_gnn/
        ("MLP", output_dir / "cage_rf_gnn" / "metrics_mlp.json"),
        ("GCN", output_dir / "cage_rf_gnn" / "metrics_gcn.json"),
        ("GraphSAGE", output_dir / "cage_rf_gnn" / "metrics_graphsage.json"),
        ("GAT", output_dir / "cage_rf_gnn" / "metrics_gat.json"),
        # GNN Layer Benchmarks (v7) - outputs/benchmark/{TYPE}/
        ("CAGE-RF SAGE v7", output_dir / "benchmark" / "SAGE" / "metrics_cage_rf_gnn_sage_v7_ensemble.json"),
        ("CAGE-RF GAT v7", output_dir / "benchmark" / "GAT" / "metrics_cage_rf_gnn_gat_v7_ensemble.json"),
        ("CAGE-RF GCN v7", output_dir / "benchmark" / "GCN" / "metrics_cage_rf_gnn_gcn_v7_ensemble.json"),
        ("CAGE-RF GraphConv v7", output_dir / "benchmark" / "GRAPHCONV" / "metrics_cage_rf_gnn_graphconv_v7_ensemble.json"),
        ("CAGE-RF Cheb v7", output_dir / "benchmark" / "CHEB" / "metrics_cage_rf_gnn_cheb_v7_ensemble.json"),
        ("CAGE-RF TAG v7", output_dir / "benchmark" / "TAG" / "metrics_cage_rf_gnn_tag_v7_ensemble.json"),
        ("CAGE-RF SG v7", output_dir / "benchmark" / "SG" / "metrics_cage_rf_gnn_sg_v7_ensemble.json"),
    ]

    results = {}
    found_count = 0

    print("\n" + "=" * 130)
    print("📊 Loading metrics from outputs/")
    print("=" * 130)

    for display_name, file_path in models_to_compare:
        metrics = load_metrics(display_name, file_path)

        if metrics:
            results[display_name] = metrics
            found_count += 1
            status = "✅"
        else:
            status = "⏭️ "
        print(f"{status} {display_name:25s} | {file_path.relative_to(output_dir)}")

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

    # Separate by model type
    print("\n" + "=" * 130)
    print("📊 GNN LAYER BENCHMARK (v7 Ensemble)")
    print("=" * 130)

    gnn_models = [idx for idx in df.index if 'CAGE-RF' in idx and 'v7' in idx and any(x in idx for x in ['SAGE', 'GAT', 'GCN', 'GraphConv', 'Cheb', 'TAG', 'SG'])]

    if gnn_models:
        gnn_df = df.loc[gnn_models].sort_values("pr_auc", ascending=False)
        print(gnn_df.to_string())
        print("\n✨ GNN Layer Rankings:")
        for rank, (model_name, row) in enumerate(gnn_df.iterrows(), 1):
            pr_auc = row.get("pr_auc", "N/A")
            macro_f1 = row.get("macro_f1", "N/A")
            model_type = model_name.split()[-2]  # Extract layer type (SAGE, GAT, etc.)
            print(f"  {rank}. {model_type:12s} | PR-AUC: {pr_auc:.4f} | Macro-F1: {macro_f1:.4f}")
    else:
        print("No GNN benchmark results found yet. Train models using:")
        print("  python run_all_gnn_benchmarks.py")

    print("\n" + "=" * 130)
    print("📈 Top 3 Models by PR-AUC (Overall)")
    print("=" * 130)

    if "pr_auc" in df.columns:
        top3 = df.nlargest(3, "pr_auc")
        for rank, (model_name, row) in enumerate(top3.iterrows(), 1):
            pr_auc = row.get("pr_auc", "N/A")
            macro_f1 = row.get("macro_f1", "N/A")
            print(f"{rank}. {model_name:30s} | PR-AUC: {pr_auc:.4f} | Macro-F1: {macro_f1:.4f}")

    print("=" * 130 + "\n")


if __name__ == "__main__":
    main()

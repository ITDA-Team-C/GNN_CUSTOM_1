#!/usr/bin/env python
"""
GNN 벤치마크 결과 비교 및 분석 스크립트
모든 GNN layer 버전의 성능을 비교합니다.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

models = [
    ("SAGE", "cage_rf_gnn_sage"),
    ("GAT", "cage_rf_gnn_gat"),
    ("GCN", "cage_rf_gnn_gcn"),
    ("GraphConv", "cage_rf_gnn_graphconv"),
    ("Cheb", "cage_rf_gnn_cheb"),
    ("TAG", "cage_rf_gnn_tag"),
    ("SG", "cage_rf_gnn_sg"),
]

def load_model_results(model_name):
    """모델 결과 로드"""
    result_path = f"outputs/{model_name}_v7_results.json"
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            return json.load(f)
    return None

def compare_results():
    """모든 모델 결과 비교"""
    print("\n" + "=" * 100)
    print("📊 GNN LAYER BENCHMARK RESULTS COMPARISON")
    print("=" * 100)

    results = []
    for display_name, model_name in models:
        data = load_model_results(model_name)
        if data:
            results.append({
                "Model": display_name,
                "Module": model_name,
                "PR-AUC": data.get("test_pr_auc", -1),
                "Macro-F1": data.get("test_macro_f1", -1),
                "Accuracy": data.get("test_accuracy", -1),
                "Threshold": data.get("best_threshold", -1),
            })
        else:
            print(f"⚠️  {display_name}: 결과 파일 미발견 ({result_path})")

    if not results:
        print("\n❌ 학습된 모델이 없습니다. 먼저 run_all_gnn_benchmarks.py를 실행하세요.")
        return

    df = pd.DataFrame(results)

    print("\n📈 Detailed Results:")
    print("-" * 100)
    print(df.to_string(index=False))
    print("-" * 100)

    print("\n🏆 Best Models:")
    print("-" * 100)
    if len(df) > 0:
        pr_auc_best = df.loc[df["PR-AUC"].idxmax()] if df["PR-AUC"].max() > 0 else None
        macro_f1_best = df.loc[df["Macro-F1"].idxmax()] if df["Macro-F1"].max() > 0 else None
        acc_best = df.loc[df["Accuracy"].idxmax()] if df["Accuracy"].max() > 0 else None

        if pr_auc_best is not None:
            print(f"✅ Best PR-AUC: {pr_auc_best['Model']:12s} ({pr_auc_best['PR-AUC']:.4f})")
        if macro_f1_best is not None:
            print(f"✅ Best Macro-F1: {macro_f1_best['Model']:12s} ({macro_f1_best['Macro-F1']:.4f})")
        if acc_best is not None:
            print(f"✅ Best Accuracy: {acc_best['Model']:12s} ({acc_best['Accuracy']:.4f})")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    compare_results()

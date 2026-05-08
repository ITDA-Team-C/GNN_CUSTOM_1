#!/usr/bin/env python3
"""
V6: Hard Negative Mining
정상인데 사기로 예측하는 샘플(hard negative)과 사기인데 정상으로 예측하는 샘플(hard positive)에 집중.
Precision과 Recall을 동시에 향상시켜 Macro-F1 & PR-AUC 개선.
"""

import os
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, load_config
from src.preprocessing.load_yelpzip import load_yelpzip, validate_columns, save_raw_eda, save_processed_data
from src.preprocessing.label_convert import label_convert
from src.preprocessing.sampling import product_user_time_hybrid_sampling, train_val_test_split, save_sampled_data
from src.preprocessing.feature_engineering import (
    extract_text_embedding, extract_numeric_features,
    normalize_features, concatenate_features, save_features
)
from src.graph.build_relations import build_all_relations, convert_to_undirected, print_statistics, save_relations, load_data as graph_load_data

import pandas as pd
import numpy as np
import torch

set_seed(42)


def step_1_load_data():
    """Step 1: Load & Validate Data"""
    print("\n" + "="*80)
    print("Step 1️⃣: Data Loading & Validation")
    print("="*80)

    df = load_yelpzip()
    df = validate_columns(df)
    save_raw_eda(df)
    df = save_processed_data(df)

    return df


def step_2_label_conversion():
    """Step 2: Label Conversion"""
    print("\n" + "="*80)
    print("Step 2️⃣: Label Conversion (-1→1, 1→0)")
    print("="*80)

    input_path = os.path.join("data/interim", "raw_data.csv")
    df = pd.read_csv(input_path)

    print(f"[Before] 라벨 분포: {df['label'].value_counts().to_dict()}")
    df["label"] = df["label"].map({-1: 1, 1: 0})

    assert set(df["label"].unique()).issubset({0, 1}), "라벨 변환 실패"

    print(f"[After] 라벨 분포: {df['label'].value_counts().to_dict()}")

    save_path = os.path.join("data/interim", "labeled_data.csv")
    df.to_csv(save_path, index=False)

    return df


def step_3_sampling(df):
    """Step 3: Sampling"""
    print("\n" + "="*80)
    print("Step 3️⃣: Product-User-Time Hybrid Dense Sampling")
    print("="*80)

    df = product_user_time_hybrid_sampling(df)
    df = train_val_test_split(df)
    save_sampled_data(df)

    return df


def step_4_feature_engineering(df):
    """Step 4: Feature Engineering"""
    print("\n" + "="*80)
    print("Step 4️⃣: Feature Engineering (TF-IDF + SVD + Numeric)")
    print("="*80)

    text_embeddings, vectorizer, svd = extract_text_embedding(df)
    numeric_features = extract_numeric_features(df)

    text_norm, numeric_norm, text_scaler, numeric_scaler = normalize_features(
        text_embeddings, numeric_features.values
    )

    combined_features = concatenate_features(text_norm, numeric_norm)
    combined_features = save_features(df, combined_features)

    return combined_features


def step_5_graph_construction(config):
    """Step 5: Graph Construction"""
    print("\n" + "="*80)
    print("Step 5️⃣: Multi-Relation Graph Construction")
    print("="*80)

    df, features = graph_load_data()

    edge_index_dict = build_all_relations(df, features)

    for relation_name in edge_index_dict:
        edge_index_dict[relation_name] = convert_to_undirected(edge_index_dict[relation_name])

    total_edges = print_statistics(edge_index_dict)
    save_relations(edge_index_dict, df)


def step_6_model_training(config_path):
    """Step 6: Model Training"""
    print("\n" + "="*80)
    print(f"Step 6️⃣: Model Training (CAGE-RF-GNN v6: Hard Negative Mining)")
    print("="*80)

    import subprocess

    cmd = f"python -m src.training.train --model cage_rf_gnn --config {config_path}"
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        raise RuntimeError("Model training failed: cage_rf_gnn v6")

    print(f"\n✅ CAGE-RF-GNN v6 모델 학습 완료")


def step_7_threshold_optimization():
    """Step 7: Threshold Optimization"""
    print("\n" + "="*80)
    print("Step 7️⃣: Threshold Optimization (Macro F1 기준)")
    print("="*80)

    import subprocess

    cmd = "python -m src.training.threshold"
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        raise RuntimeError("Threshold optimization failed")

    print(f"\n✅ Threshold 최적화 완료")


def step_8_final_evaluation():
    """Step 8: Final Evaluation"""
    print("\n" + "="*80)
    print("Step 8️⃣: Final Evaluation (Test Set)")
    print("="*80)

    import subprocess

    checkpoint = "outputs/best_model_cage_rf_gnn.pt"
    cmd = f"python -m src.training.evaluate --checkpoint {checkpoint}"
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        raise RuntimeError("Evaluation failed")

    print(f"\n✅ 평가 완료")


def main():
    parser = argparse.ArgumentParser(description="V6: Hard Negative Mining Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v6_hard_mining.yaml",
        help="Config file path (default: v6_hard_mining.yaml)"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing"
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip graph construction"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 V6: Hard Negative Mining - Challenging Sample Focus")
    print("="*80)

    start_time = time.time()
    config = load_config(args.config)

    try:
        if not args.skip_preprocessing:
            print("\n[Phase 1] 데이터 전처리 시작...")
            df = step_1_load_data()
            df = step_2_label_conversion()
            df = step_3_sampling(df)
            features = step_4_feature_engineering(df)
            print(f"\n✅ 데이터 전처리 완료 ({features.shape[0]} nodes, {features.shape[1]} features)")
        else:
            print("\n⏭️ 데이터 전처리 건너뜀")

        if not args.skip_graph:
            print("\n[Phase 2] 그래프 구성 시작...")
            step_5_graph_construction(config)
            print(f"\n✅ 그래프 구성 완료")
        else:
            print("\n⏭️ 그래프 구성 건너뜀")

        print(f"\n[Phase 3] 모델 학습 및 평가 시작...")
        step_6_model_training(args.config)
        step_7_threshold_optimization()
        step_8_final_evaluation()

        elapsed_time = time.time() - start_time

        print("\n" + "="*80)
        print("✅ V6 파이프라인 완료!")
        print("="*80)
        print(f"⏱️  실행 시간: {elapsed_time/60:.1f}분")
        print(f"📊 모델: CAGE-RF-GNN v6 (Hard Negative Mining)")
        print(f"📁 결과: outputs/ (best_model_cage_rf_gnn.pt, metrics_cage_rf_gnn.json)")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()

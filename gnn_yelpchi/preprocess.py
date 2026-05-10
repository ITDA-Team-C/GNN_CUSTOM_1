#!/usr/bin/env python3
"""
YelpChi 데이터 전처리 파이프라인
YelpZip 데이터 로드 → 라벨 변환 → 샘플링 → Feature Engineering → 그래프 구성
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import set_seed
from src.preprocessing import (
    load_yelpzip, validate_columns, save_raw_eda, save_processed_data,
    label_convert, product_user_time_hybrid_sampling, train_val_test_split, save_sampled_data,
    extract_text_embedding, extract_numeric_features, normalize_features, concatenate_features, save_features
)
from src.graph import build_all_relations, convert_to_undirected, print_statistics, save_relations, load_data

set_seed(42)


def main():
    print("\n" + "="*80)
    print("📊 YelpChi 데이터 전처리 시작")
    print("="*80)

    # Step 1: Load & Validate Data
    print("\n[Step 1/6] YelpZip 데이터 로드...")
    df = load_yelpzip()
    df = validate_columns(df)
    save_raw_eda(df)
    df = save_processed_data(df)
    print(f"✅ 데이터 로드 완료: {len(df)} rows")

    # Step 2: Label Conversion
    print("\n[Step 2/6] 라벨 변환 (-1→1, 1→0)...")
    df = label_convert(df)
    print(f"✅ 라벨 변환 완료")
    print(f"   - 정상(0): {(df['label'] == 0).sum()}")
    print(f"   - 사기(1): {(df['label'] == 1).sum()}")

    # Step 3: Sampling
    print("\n[Step 3/6] 샘플링 (25,000 nodes)...")
    sampled_df = product_user_time_hybrid_sampling(df)
    sampled_df = train_val_test_split(sampled_df)
    save_sampled_data(sampled_df)
    print(f"✅ 샘플링 및 분할 완료")

    # Step 4: Feature Engineering
    print("\n[Step 4/6] Feature Engineering...")
    print("   - 텍스트 임베딩 (TF-IDF → SVD 128D)...")
    text_features, vectorizer, svd = extract_text_embedding(sampled_df)

    print("   - 정형 특징 추출...")
    numeric_features = extract_numeric_features(sampled_df)

    print("   - 정규화...")
    text_features_norm, numeric_features_norm, text_scaler, numeric_scaler = normalize_features(
        text_features, numeric_features.values
    )

    features = concatenate_features(text_features_norm, numeric_features_norm)
    print(f"✅ Feature Engineering 완료: {features.shape}")

    # Step 5: Save Features
    print("\n[Step 5/6] Feature 저장...")
    combined_features = save_features(sampled_df, features)

    # Step 6: Graph Construction
    print("\n[Step 6/6] Multi-Relation 그래프 구성 중...")
    df_for_graph, features_for_graph = load_data()

    edge_index_dict = build_all_relations(df_for_graph, features_for_graph)

    for relation_name in edge_index_dict:
        edge_index_dict[relation_name] = convert_to_undirected(edge_index_dict[relation_name])

    total_edges = print_statistics(edge_index_dict)
    save_relations(edge_index_dict, df_for_graph)
    print("✅ 그래프 구성 완료!")

    print("\n" + "="*80)
    print("✨ YelpChi 전처리 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

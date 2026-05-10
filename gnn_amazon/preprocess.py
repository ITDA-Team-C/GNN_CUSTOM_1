#!/usr/bin/env python3
"""
Amazon 데이터 전처리 파이프라인
Amazon 데이터 로드 → 라벨 변환 → 사용자 샘플링 → Feature Engineering → 그래프 구성

주의: 원본 데이터(df)는 리뷰 단위이지만,
      사용자 특징은 사용자별로 집계됩니다.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import set_seed
from src.preprocessing import (
    load_amazon, validate_columns, save_raw_eda, save_processed_data,
    label_convert, user_based_sampling, train_val_test_split, save_sampled_data,
    aggregate_user_reviews, extract_text_embedding, extract_numeric_features,
    normalize_features, concatenate_features, save_features
)
from src.graph import build_all_relations, convert_to_undirected, print_statistics, save_relations, load_data

set_seed(42)


def main():
    print("\n" + "="*80)
    print("📊 Amazon 데이터 전처리 시작")
    print("="*80)

    # Step 1: Load & Validate Data
    print("\n[Step 1/6] Amazon 데이터 로드...")
    df = load_amazon()
    df = validate_columns(df)
    save_raw_eda(df)
    df = save_processed_data(df)
    print(f"✅ 데이터 로드 완료: {len(df)} reviews")

    # Step 2: Label Conversion
    print("\n[Step 2/6] 라벨 확인...")
    df = label_convert(df)
    print(f"✅ 라벨 처리 완료")
    print(f"   - 정상 사용자(0): {(df['label'] == 0).sum()}")
    print(f"   - 사기 사용자(1): {(df['label'] == 1).sum()}")

    # Step 3: Sampling (선택사항)
    print("\n[Step 3/6] 사용자 샘플링 (선택)...")
    sampled_df = user_based_sampling(df, target_users=None)
    sampled_df = train_val_test_split(sampled_df)
    save_sampled_data(sampled_df)
    print(f"✅ 샘플링 및 분할 완료")

    # Step 4: User-Level Aggregation & Feature Engineering
    print("\n[Step 4/6] Feature Engineering...")

    print("   - 사용자별 리뷰 집계...")
    user_df = aggregate_user_reviews(sampled_df)

    print("   - 텍스트 임베딩 (TF-IDF → SVD 128D)...")
    text_features, vectorizer, svd = extract_text_embedding(user_df)

    print("   - 정형 특징 추출...")
    numeric_features = extract_numeric_features(user_df)

    print("   - 정규화...")
    text_features_norm, numeric_features_norm, text_scaler, numeric_scaler = normalize_features(
        text_features, numeric_features.values
    )

    features = concatenate_features(text_features_norm, numeric_features_norm)
    print(f"✅ Feature Engineering 완료: {features.shape}")

    # Step 5: Save Features
    print("\n[Step 5/6] Feature 저장...")
    combined_features = save_features(user_df, features)

    # Step 6: Graph Construction (원본 df 사용)
    print("\n[Step 6/6] Multi-Relation 그래프 구성 중...")
    print("   (원본 리뷰 데이터 기반 사용자 관계 구성)")

    edge_index_dict = build_all_relations(sampled_df, features)

    for relation_name in edge_index_dict:
        edge_index_dict[relation_name] = convert_to_undirected(edge_index_dict[relation_name])

    total_edges = print_statistics(edge_index_dict)
    save_relations(edge_index_dict, sampled_df)
    print("✅ 그래프 구성 완료!")

    print("\n" + "="*80)
    print("✨ Amazon 전처리 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

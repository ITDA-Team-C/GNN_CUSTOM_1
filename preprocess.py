#!/usr/bin/env python3
"""
데이터 전처리 (전처리 전용)
YelpZip 데이터 로드 → 라벨 변환 → 샘플링 → Feature Engineering
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import set_seed
from src.preprocessing.load_yelpzip import load_yelpzip, validate_columns, save_raw_eda, save_processed_data
from src.preprocessing.label_convert import label_convert
from src.preprocessing.sampling import product_user_time_hybrid_sampling, train_val_test_split, save_sampled_data
from src.preprocessing.feature_engineering import (
    extract_text_embedding, extract_numeric_features,
    normalize_features, concatenate_features, save_features
)

set_seed(42)


def main():
    print("\n" + "="*80)
    print("📊 데이터 전처리 시작")
    print("="*80)

    # Step 1: Load & Validate Data
    print("\n[Step 1/5] YelpZip 데이터 로드...")
    df = load_yelpzip()
    df = validate_columns(df)
    save_raw_eda(df)
    df = save_processed_data(df)
    print(f"✅ 데이터 로드 완료: {len(df)} rows")

    # Step 2: Label Conversion
    print("\n[Step 2/5] 라벨 변환 (-1→1, 1→0)...")
    df = label_convert(df)
    print(f"✅ 라벨 변환 완료")
    print(f"   - 정상(0): {(df['label'] == 0).sum()}")
    print(f"   - 사기(1): {(df['label'] == 1).sum()}")

    # Step 3: Sampling
    print("\n[Step 3/5] 샘플링 (25,000 nodes)...")
    sampled_df = product_user_time_hybrid_sampling(df, n_samples=25000)
    train_df, valid_df, test_df = train_val_test_split(sampled_df)

    print(f"✅ 샘플링 완료")
    print(f"   - Train: {len(train_df)} ({len(train_df)/len(sampled_df)*100:.1f}%)")
    print(f"   - Valid: {len(valid_df)} ({len(valid_df)/len(sampled_df)*100:.1f}%)")
    print(f"   - Test:  {len(test_df)} ({len(test_df)/len(sampled_df)*100:.1f}%)")

    sampled_df = save_sampled_data(sampled_df, train_df, valid_df, test_df)

    # Step 4: Feature Engineering
    print("\n[Step 4/5] Feature Engineering...")

    # Text embedding (TF-IDF → SVD 128D)
    print("   - 텍스트 임베딩 (TF-IDF → SVD 128D)...")
    text_features = extract_text_embedding(sampled_df, n_components=128)

    # Numeric features
    print("   - 정형 특징 추출...")
    numeric_features = extract_numeric_features(sampled_df)

    # Normalize
    print("   - 정규화...")
    text_features_norm = normalize_features(text_features)
    numeric_features_norm = normalize_features(numeric_features)

    # Concatenate
    features = concatenate_features(text_features_norm, numeric_features_norm)
    print(f"✅ Feature Engineering 완료: {features.shape}")

    # Step 5: Save
    print("\n[Step 5/5] 데이터 저장...")
    save_features(features, sampled_df)
    print("✅ 전처리 완료! 데이터가 data/processed/에 저장되었습니다.")

    print("\n" + "="*80)
    print("✨ 이제 대시보드를 실행할 수 있습니다:")
    print("   streamlit run dashboard/app.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

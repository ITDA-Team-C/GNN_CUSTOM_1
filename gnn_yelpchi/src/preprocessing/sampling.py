import os
import pandas as pd
import numpy as np
from collections import Counter


def product_user_time_hybrid_sampling(df, target_samples=25000):
    """
    Product-User-Time 하이브리드 샘플링
    - 각 product별로 균등하게 샘플링
    - 각 user별로 균등하게 샘플링
    - 시간대별로 균등하게 샘플링
    """
    print(f"[Sampling] 목표: {target_samples} 개 리뷰 샘플링...")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year_month'] = df['date'].dt.to_period('M')

    sampled_dfs = []

    prod_ids = df['prod_id'].unique()
    samples_per_prod = max(1, target_samples // len(prod_ids))

    for prod_id in prod_ids:
        prod_df = df[df['prod_id'] == prod_id]

        if len(prod_df) <= samples_per_prod:
            sampled_dfs.append(prod_df)
        else:
            sampled_dfs.append(prod_df.sample(n=samples_per_prod, random_state=42))

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    if len(sampled_df) > target_samples:
        sampled_df = sampled_df.sample(n=target_samples, random_state=42)

    print(f"  최종 샘플 수: {len(sampled_df)}")
    print(f"  라벨 분포: {sampled_df['label'].value_counts().to_dict()}")

    return sampled_df


def train_val_test_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Train/Val/Test 분할 (6:2:2)
    """
    print(f"[Split] Train/Val/Test 분할 중... ({train_ratio}/{val_ratio}/{test_ratio})")

    df = df.copy()
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    total_len = len(df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    df['split'] = 'test'
    df.loc[:train_end, 'split'] = 'train'
    df.loc[train_end:val_end, 'split'] = 'val'

    print(f"  Train: {(df['split'] == 'train').sum()}")
    print(f"  Val: {(df['split'] == 'val').sum()}")
    print(f"  Test: {(df['split'] == 'test').sum()}")

    return df


def save_sampled_data(df):
    """
    샘플링된 데이터 저장
    """
    os.makedirs("data/interim", exist_ok=True)

    save_path = os.path.join("data/interim", "sampled_data.csv")
    df.to_csv(save_path, index=False)
    print(f"[Save] {save_path}")

    return df


if __name__ == "__main__":
    print("Sampling utilities loaded")

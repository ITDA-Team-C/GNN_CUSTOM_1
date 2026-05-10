import os
import pandas as pd
import numpy as np
from collections import Counter


def user_based_sampling(df, target_users=None):
    """
    사용자 기반 샘플링

    Parameters:
    -----------
    df : DataFrame
        원본 리뷰 데이터
    target_users : int
        목표 사용자 수 (None이면 모든 사용자 사용)

    Returns:
    --------
    sampled_df : DataFrame
        사용자별로 샘플링된 데이터
    """
    df = df.copy()

    unique_users = df['user_id'].unique()
    print(f"[Sampling] 전체 사용자 수: {len(unique_users)}")

    if target_users is None or target_users >= len(unique_users):
        print(f"[Sampling] 모든 사용자 사용")
        sampled_df = df
    else:
        print(f"[Sampling] 목표: {target_users} 명 사용자 샘플링...")

        # 라벨별로 균형있게 샘플링
        sampled_users = []
        for label in df['label'].unique():
            label_users = df[df['label'] == label]['user_id'].unique()
            sample_size = int(target_users * (len(label_users) / len(unique_users)))
            sampled = np.random.choice(label_users, size=min(sample_size, len(label_users)), replace=False)
            sampled_users.extend(sampled)

        sampled_df = df[df['user_id'].isin(sampled_users)]
        print(f"[Sampling] 최종 사용자 수: {len(sampled_df['user_id'].unique())}")

    print(f"  - 정상 사용자(0): {(sampled_df['label'] == 0).sum()}")
    print(f"  - 사기 사용자(1): {(sampled_df['label'] == 1).sum()}")

    return sampled_df


def train_val_test_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    사용자 단위로 Train/Val/Test 분할
    """
    print(f"[Split] Train/Val/Test 분할 중... ({train_ratio}/{val_ratio}/{test_ratio})")

    df = df.copy()
    unique_users = df['user_id'].unique()
    unique_users = np.random.permutation(unique_users)

    total_users = len(unique_users)
    train_end = int(total_users * train_ratio)
    val_end = int(total_users * (train_ratio + val_ratio))

    train_users = unique_users[:train_end]
    val_users = unique_users[train_end:val_end]
    test_users = unique_users[val_end:]

    df['split'] = 'test'
    df.loc[df['user_id'].isin(train_users), 'split'] = 'train'
    df.loc[df['user_id'].isin(val_users), 'split'] = 'val'

    print(f"  Train users: {len(train_users)}, reviews: {(df['split'] == 'train').sum()}")
    print(f"  Val users: {len(val_users)}, reviews: {(df['split'] == 'val').sum()}")
    print(f"  Test users: {len(test_users)}, reviews: {(df['split'] == 'test').sum()}")

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
    print("Amazon sampling utilities loaded")

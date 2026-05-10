import os
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy import sparse
from pathlib import Path
from src.utils import set_seed

set_seed(42)

CONFIG = {
    "raw_dir": "data/raw",
    "interim_dir": "data/interim",
    "filename": "amazon.mat",
}


def load_amazon():
    """
    Amazon 데이터 로드 (.mat 파일)

    구조:
    - homo: 그래프 행렬 (11944x11944)
    - net_upu: User-Product-User 그래프 (sparse)
    - net_usu: User-Star-User 그래프 (sparse)
    - net_uvu: User-Vocab-User 그래프 (sparse)
    - features: 노드 특징 (11944x25)
    - label: 노드 라벨 (1x11944)
    """
    raw_path = os.path.join(CONFIG["raw_dir"], CONFIG["filename"])

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Amazon file not found: {raw_path}\n"
            f"Place amazon.mat in data/raw/"
        )

    print(f"[Load] {raw_path} loading...")

    # .mat 파일 로드
    mat_data = sio.loadmat(raw_path)
    mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

    print(f"  Variables: {list(mat_data.keys())}")

    # 데이터 구조 확인
    # Label 처리
    label = mat_data['label']
    if label.ndim > 1:
        label = label.flatten()
    label = label.astype(int)

    # Features 처리
    features = mat_data['features']
    if sparse.issparse(features):
        features = features.toarray()

    num_nodes = len(label)

    result = {
        'num_nodes': num_nodes,
        'label': label,
        'features': features,
    }

    # 그래프 구조 - 다양한 키 이름 시도
    key_candidates = {
        'net_upu': ['net_upu', 'upu', 'NET_UPU', 'UPU', 'u_p_u'],
        'net_usu': ['net_usu', 'usu', 'NET_USU', 'USU', 'u_s_u'],
        'net_uvu': ['net_uvu', 'uvu', 'NET_UVU', 'UVU', 'u_v_u'],
        'homo': ['homo', 'HOMO', 'all', 'full'],
    }

    for std_key, candidates in key_candidates.items():
        found = False
        for cand in candidates:
            if cand in mat_data:
                adj = sparse.csr_matrix(mat_data[cand])
                if adj.shape[0] > num_nodes or adj.shape[1] > num_nodes:
                    print(f"  WARNING: {cand} shape {adj.shape} exceeds num_nodes {num_nodes}, clipping...")
                    adj = adj[:num_nodes, :num_nodes]
                result[std_key] = adj
                print(f"  Found {std_key} as '{cand}': {adj.shape}")
                found = True
                break

        if not found:
            print(f"  WARNING: {std_key} not found! Available keys: {list(mat_data.keys())}")

    print(f"  Num nodes: {result['num_nodes']}")
    print(f"  Feature shape: {result['features'].shape}")
    print(f"  Label distribution: {np.bincount(result['label'])}")

    return result




def validate_columns(df):
    """
    필수 컬럼 검증
    """
    required_cols = {"user_id", "prod_id", "rating", "review_text", "label"}
    existing_cols = set(df.columns)

    missing_cols = required_cols - existing_cols

    if missing_cols:
        print(f"[Warning] 누락된 컬럼: {missing_cols}")
        print(f"  필수 컬럼: {required_cols}")

    return df


def save_raw_eda(df):
    """
    원본 데이터 EDA 저장
    """
    os.makedirs(CONFIG["interim_dir"], exist_ok=True)

    eda_path = os.path.join(CONFIG["interim_dir"], "raw_eda.txt")

    with open(eda_path, "w", encoding="utf-8") as f:
        f.write("=== Amazon Raw Data EDA ===\n\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n\n")

        f.write("Column Info:\n")
        f.write(df.info(buf=None).__str__() + "\n\n")

        f.write("Numeric Columns:\n")
        f.write(df.describe().to_string() + "\n\n")

        f.write("Missing Values:\n")
        f.write(df.isnull().sum().to_string() + "\n\n")

        if "label" in df.columns:
            f.write("Label Distribution:\n")
            f.write(df["label"].value_counts().to_string() + "\n\n")

        if "user_id" in df.columns:
            f.write(f"Unique Users: {df['user_id'].nunique()}\n")

        if "prod_id" in df.columns:
            f.write(f"Unique Products: {df['prod_id'].nunique()}\n")

        if "rating" in df.columns:
            f.write(f"\nRating Distribution:\n")
            f.write(df["rating"].value_counts().sort_index().to_string() + "\n")

    print(f"[Save] EDA 저장됨: {eda_path}")


def save_processed_data(df):
    """
    처리된 데이터 저장
    """
    os.makedirs(CONFIG["interim_dir"], exist_ok=True)

    save_path = os.path.join(CONFIG["interim_dir"], "raw_data.csv")
    df.to_csv(save_path, index=False)
    print(f"[Save] {save_path}")

    return df


if __name__ == "__main__":
    df = load_amazon()
    df = validate_columns(df)
    save_raw_eda(df)
    df = save_processed_data(df)
    print("\n[Done] Amazon Data Loading 완료")

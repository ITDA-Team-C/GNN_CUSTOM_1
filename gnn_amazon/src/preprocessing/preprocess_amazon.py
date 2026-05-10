"""
Amazon 데이터 전처리 - 이미 구조화된 .mat 파일 처리
"""
import os
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from src.utils import set_seed
from src.preprocessing import load_amazon

set_seed(42)


def save_amazon_data(data_dict):
    """
    Amazon 데이터를 PyG 호환 형식으로 저장
    """
    os.makedirs("data/processed", exist_ok=True)

    # Features 저장
    features = data_dict['features']
    if sparse.issparse(features):
        features = features.toarray()

    np.save("data/processed/features.npy", features)
    print(f"[Save] Features: {features.shape}")

    # 라벨 저장
    labels = data_dict['label']
    np.save("data/processed/labels.npy", labels)
    print(f"[Save] Labels: {labels.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")

    # 그래프 구조 저장 (edge_index 형식으로 변환)
    edge_data = {}
    for net_name in ['net_upu', 'net_usu', 'net_uvu', 'homo']:
        if net_name in data_dict:
            adj = data_dict[net_name]
            if sparse.issparse(adj):
                edge_index = torch.from_numpy(
                    np.array(adj.nonzero())
                ).long()
            else:
                edge_index = torch.from_numpy(
                    np.array(adj.nonzero())
                ).long()

            edge_data[net_name] = edge_index
            print(f"[Save] {net_name}: {edge_index.shape}")

    torch.save(edge_data, "data/processed/edge_index_dict.pt")

    # 메타정보 저장
    meta = {
        'num_nodes': data_dict['num_nodes'],
        'num_features': features.shape[1],
        'num_classes': len(np.unique(labels)),
        'edge_types': list(edge_data.keys()),
    }

    import json
    with open("data/processed/meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Meta] Nodes: {meta['num_nodes']}, Features: {meta['num_features']}")

    return features, labels, edge_data


def create_train_val_test_split(labels, train_ratio=0.6, val_ratio=0.2):
    """
    라벨별 균형있게 분할
    """
    num_nodes = len(labels)
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    split_data = {
        'train_idx': torch.from_numpy(train_idx).long(),
        'val_idx': torch.from_numpy(val_idx).long(),
        'test_idx': torch.from_numpy(test_idx).long(),
    }

    torch.save(split_data, "data/processed/split_idx.pt")

    print(f"[Split] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return split_data


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Amazon Data Preprocessing")
    print("="*80)

    # Load
    print("\n[Step 1/2] Loading data...")
    data_dict = load_amazon()

    # Preprocess & Save
    print("\n[Step 2/2] Saving processed data...")
    features, labels, edge_data = save_amazon_data(data_dict)

    # Create splits
    print("\n[Step 3/2] Creating splits...")
    split_data = create_train_val_test_split(labels)

    print("\n" + "="*80)
    print("Amazon preprocessing complete!")
    print("="*80 + "\n")

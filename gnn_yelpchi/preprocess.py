#!/usr/bin/env python3
"""
YelpChi 데이터 전처리 파이프라인 - .mat 파일 직접 처리
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import set_seed
from src.preprocessing.load_yelpzip import load_yelpzip

set_seed(42)


def save_yelpchi_data(data_dict):
    """YelpChi 데이터를 PyG 호환 형식으로 저장"""
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
    num_nodes = len(labels)
    edge_data = {}
    for net_name in ['rur', 'rtr', 'rsr']:
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

            # Validate edge_index
            if edge_index.numel() > 0:
                max_node = edge_index.max().item()
                if max_node >= num_nodes:
                    print(f"  WARNING: {net_name} has node index {max_node} >= {num_nodes}, clipping...")
                    # Clip invalid indices
                    edge_index = edge_index[:, (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)]

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

    with open("data/processed/meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Meta] Nodes: {meta['num_nodes']}, Features: {meta['num_features']}")


def create_train_val_test_split(labels, train_ratio=0.6, val_ratio=0.2):
    """라벨별 균형있게 분할"""
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


def main():
    print("\n" + "="*80)
    print("YelpChi 데이터 전처리")
    print("="*80)

    # Step 1: Load
    print("\n[Step 1/2] YelpChi 데이터 로드...")
    data_dict = load_yelpzip()

    # Step 2: Preprocess & Save
    print("\n[Step 2/2] 전처리된 데이터 저장...")
    save_yelpchi_data(data_dict)

    # Step 3: Create splits
    print("\n[Step 3/3] 데이터 분할...")
    split_data = create_train_val_test_split(data_dict['label'])

    print("\n" + "="*80)
    print("YelpChi 전처리 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

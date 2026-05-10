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
from src.preprocessing.load_yelpchi import load_yelpchi

set_seed(42)


def sample_nodes(data_dict, sample_size=10000):
    """10,000개 노드 샘플링"""
    num_nodes = data_dict['num_nodes']
    labels = data_dict['label']

    print(f"\n[Sampling] Total nodes: {num_nodes} → Sample size: {sample_size}")

    if num_nodes <= sample_size:
        print(f"  No sampling needed (already {num_nodes} <= {sample_size})")
        return list(range(num_nodes))

    # 라벨 분포를 유지하면서 샘플링
    sampled_indices = []
    for label_val in np.unique(labels):
        label_indices = np.where(labels == label_val)[0]
        label_count = len(label_indices)
        ratio = label_count / num_nodes
        sample_count = int(sample_size * ratio)

        sampled = np.random.choice(label_indices, size=min(sample_count, label_count), replace=False)
        sampled_indices.extend(sampled)

    # 정확히 sample_size가 되도록 조정
    sampled_indices = np.array(sampled_indices)
    if len(sampled_indices) > sample_size:
        sampled_indices = sampled_indices[:sample_size]
    elif len(sampled_indices) < sample_size:
        remaining = np.array([i for i in range(num_nodes) if i not in sampled_indices])
        need = sample_size - len(sampled_indices)
        extra = np.random.choice(remaining, size=min(need, len(remaining)), replace=False)
        sampled_indices = np.concatenate([sampled_indices, extra])

    sampled_indices = np.sort(sampled_indices)

    print(f"  Sampled: {len(sampled_indices)} nodes")
    print(f"  Label distribution: {np.bincount(labels[sampled_indices])}")

    return sampled_indices


def save_yelpchi_data(data_dict, sample_size=30000):
    """YelpChi 데이터를 PyG 호환 형식으로 저장"""
    os.makedirs("data/processed", exist_ok=True)

    # 노드 샘플링
    sampled_indices = sample_nodes(data_dict, sample_size=30000)
    sampled_mask = np.zeros(data_dict['num_nodes'], dtype=bool)
    sampled_mask[sampled_indices] = True

    # Features 저장
    features = data_dict['features']
    if sparse.issparse(features):
        features = features.toarray()
    features = features[sampled_indices]

    np.save("data/processed/features.npy", features)
    print(f"[Save] Features: {features.shape}")

    # 라벨 저장
    labels = data_dict['label'][sampled_indices]
    np.save("data/processed/labels.npy", labels)
    print(f"[Save] Labels: {labels.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")

    # 그래프 구조 저장 (edge_index 형식으로 변환)
    num_sampled_nodes = len(sampled_indices)
    edge_data = {}

    # 샘플링된 노드 인덱스 매핑 (원본 idx → 새 idx)
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_indices)}

    for net_name in ['rur', 'rtr', 'rsr']:
        if net_name in data_dict:
            adj = data_dict[net_name]
            if sparse.issparse(adj):
                src, dst = adj.nonzero()
            else:
                src, dst = np.array(adj).nonzero()

            # 샘플링된 노드들만 포함하는 엣지 필터링
            edge_mask = sampled_mask[src] & sampled_mask[dst]
            src_filtered = src[edge_mask]
            dst_filtered = dst[edge_mask]

            # 노드 인덱스 재매핑
            src_remapped = np.array([node_mapping[s] for s in src_filtered])
            dst_remapped = np.array([node_mapping[d] for d in dst_filtered])

            edge_index = torch.from_numpy(
                np.array([src_remapped, dst_remapped])
            ).long()

            edge_data[net_name] = edge_index
            print(f"[Save] {net_name}: {edge_index.shape}")

    torch.save(edge_data, "data/processed/edge_index_dict.pt")

    # 메타정보 저장
    meta = {
        'num_nodes': len(sampled_indices),
        'num_features': features.shape[1],
        'num_classes': len(np.unique(labels)),
        'edge_types': list(edge_data.keys()),
    }

    with open("data/processed/meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Meta] Nodes: {meta['num_nodes']}, Features: {meta['num_features']}")

    return labels


def create_train_val_test_split(labels, train_ratio=0.64, val_ratio=0.16):
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
    data_dict = load_yelpchi()

    # Step 2: Preprocess & Save
    print("\n[Step 2/2] 전처리된 데이터 저장...")
    sampled_labels = save_yelpchi_data(data_dict)

    # Step 3: Create splits (샘플링된 라벨로 분할)
    print("\n[Step 3/3] 데이터 분할...")
    split_data = create_train_val_test_split(sampled_labels)

    print("\n" + "="*80)
    print("YelpChi 전처리 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

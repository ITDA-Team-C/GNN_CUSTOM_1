import os
import numpy as np
import pandas as pd
import torch
from src.graph.build_upu import build_upu
from src.graph.build_usu import build_usu
from src.graph.build_uvu import build_uvu

CONFIG = {
    "processed_dir": "data/processed",
    "nodes_file": "user_samples.csv",
    "features_file": "features.npy",
}


def load_data():
    """
    사용자 샘플과 특징 로드
    """
    nodes_path = os.path.join(CONFIG["processed_dir"], CONFIG["nodes_file"])
    features_path = os.path.join(CONFIG["processed_dir"], CONFIG["features_file"])

    df = pd.read_csv(nodes_path)
    features = np.load(features_path)

    print(f"[Load] 사용자 수: {len(df)}")
    print(f"[Load] Feature 형태: {features.shape}")

    return df, features


def build_all_relations(df, features):
    """
    모든 Relation 구성 (U-P-U, U-S-U, U-V-U)

    Parameters:
    -----------
    df : DataFrame
        원본 리뷰 데이터 (user_id, prod_id, rating, review_text 등 포함)
    features : ndarray
        사용자 특징 행렬
    """
    print("\n[Graph] 모든 Relation 구성 중...\n")

    edge_index_dict = {}

    print("=" * 60)
    edge_index_dict["upu"] = build_upu(df)
    print()

    print("=" * 60)
    edge_index_dict["usu"] = build_usu(df)
    print()

    print("=" * 60)
    edge_index_dict["uvu"] = build_uvu(df)
    print()

    return edge_index_dict


def convert_to_undirected(edge_index):
    """
    방향 그래프를 무방향 그래프로 변환
    """
    if edge_index.shape[1] == 0:
        return edge_index

    forward = edge_index
    backward = torch.stack([edge_index[1], edge_index[0]])

    undirected = torch.cat([forward, backward], dim=1)
    undirected = torch.unique(undirected, dim=1)

    return undirected


def print_statistics(edge_index_dict):
    """
    Relation 통계 출력
    """
    print("\n" + "=" * 60)
    print("[Summary] Relation 통계")
    print("=" * 60)

    total_edges = 0
    for relation_name, edge_index in edge_index_dict.items():
        num_edges = edge_index.shape[1]
        total_edges += num_edges
        print(f"{relation_name:15s}: {num_edges:8d} edges")

    print("-" * 60)
    print(f"{'Total':15s}: {total_edges:8d} edges")
    print("=" * 60)

    return total_edges


def save_relations(edge_index_dict, df):
    """
    Relation 저장
    """
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)

    import pickle
    relations_path = os.path.join(CONFIG["processed_dir"], "edge_index_dict.pkl")
    with open(relations_path, 'wb') as f:
        pickle.dump(edge_index_dict, f)

    print(f"[Save] {relations_path}")

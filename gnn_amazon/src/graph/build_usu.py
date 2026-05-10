import numpy as np
import pandas as pd
import torch
from collections import defaultdict


def build_usu(df):
    """
    U-S-U (User-Star-User): 동일한 제품에 같은 별점을 준 사용자들끼리 연결
    """
    print("[U-S-U] User-Star-User 관계 구성 중...")

    # (제품, 별점) 조합별로 사용자 목록 생성
    prod_rating_to_users = defaultdict(set)
    for user_id, prod_id, rating in zip(df["user_id"], df["prod_id"], df["rating"]):
        prod_rating_to_users[(prod_id, rating)].add(user_id)

    # 사용자 ID를 노드 인덱스로 매핑
    unique_users = sorted(df["user_id"].unique())
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}

    edges_src = []
    edges_dst = []
    k = 10  # 각 사용자당 최대 k개 이웃

    for (prod_id, rating), users in prod_rating_to_users.items():
        users_list = list(users)

        if len(users_list) > 1:
            for i, user_i in enumerate(users_list):
                candidates = [user_j for j, user_j in enumerate(users_list) if i != j]

                if len(candidates) > k:
                    candidates = np.random.choice(candidates, size=k, replace=False).tolist()

                for user_j in candidates:
                    idx_i = user_to_idx[user_i]
                    idx_j = user_to_idx[user_j]
                    edges_src.append(idx_i)
                    edges_dst.append(idx_j)

    if len(edges_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    edge_index = torch.unique(edge_index, dim=1)

    print(f"  엣지 수: {edge_index.shape[1]}")

    return edge_index

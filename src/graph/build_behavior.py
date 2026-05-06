import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def extract_user_behavior_features(df):
    print("  사용자 행동 feature 추출 중...")

    user_features = {}

    user_stats = df.groupby("user_id").agg({
        "review_id": "count",
        "rating": ["mean", "std"],
        "date": lambda x: (x.max() - x.min()).days + 1,
    }).fillna(0)

    user_stats.columns = ["review_count", "avg_rating", "rating_std", "active_days"]

    user_review_counts = df.groupby("user_id").size()
    user_product_diversity = df.groupby("user_id")["prod_id"].nunique()

    user_features = {}
    for user_id in df["user_id"].unique():
        features = [
            user_stats.loc[user_id, "review_count"] if user_id in user_stats.index else 0,
            user_stats.loc[user_id, "avg_rating"] if user_id in user_stats.index else 3.0,
            user_stats.loc[user_id, "rating_std"] if user_id in user_stats.index else 0,
            user_stats.loc[user_id, "active_days"] if user_id in user_stats.index else 1,
            user_product_diversity.get(user_id, 1),
        ]
        user_features[user_id] = features

    user_feature_matrix = np.array([user_features[uid] for uid in df["user_id"]])

    scaler = StandardScaler()
    user_feature_matrix = scaler.fit_transform(user_feature_matrix)

    return user_feature_matrix


def build_behavior(df):
    print("[R-UserBehavior-R] 사용자 행동 유사도 관계 구성 중...")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    user_feature_matrix = extract_user_behavior_features(df)

    similarity_matrix = cosine_similarity(user_feature_matrix)

    edges_src = []
    edges_dst = []

    k = 5

    for i in range(len(df)):
        sims = similarity_matrix[i]

        sorted_indices = np.argsort(-sims)[1:k+1]
        candidates = [j for j in sorted_indices if sims[j] > 0.3]

        for j in candidates:
            if i != j:
                edges_src.append(i)
                edges_dst.append(j)

    if len(edges_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    edge_index = torch.unique(edge_index, dim=1)

    print(f"  엣지 수: {edge_index.shape[1]}")

    return edge_index

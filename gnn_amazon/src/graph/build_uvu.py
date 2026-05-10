import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def build_uvu(df, top_percentile=95):
    """
    U-V-U (User-Vocab/TextSim-User): 리뷰 텍스트 유사도(TF-IDF 기준)가 높은 사용자들끼리 연결
    상위 top_percentile (기본 95%) 이내의 유사도를 가진 사용자들 연결
    """
    print(f"[U-V-U] User-TextSim-User 관계 구성 중... (상위 {top_percentile}% 유사도)")

    # 사용자별 리뷰 텍스트 집계
    user_reviews = df.groupby("user_id")["review_text"].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()
    user_reviews.columns = ["user_id", "aggregated_text"]

    # 사용자 ID를 노드 인덱스로 매핑
    unique_users = sorted(user_reviews["user_id"].unique())
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}

    # TF-IDF 벡터화
    print("  [TF-IDF] 사용자별 리뷰 텍스트 벡터화 중...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(user_reviews["aggregated_text"])

    # 코사인 유사도 계산
    print("  [Similarity] 코사인 유사도 계산 중...")
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 상위 percentile 임계값 계산
    similarity_flat = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    threshold = np.percentile(similarity_flat, top_percentile)
    print(f"    유사도 임계값 (상위 {100-top_percentile}%): {threshold:.4f}")

    # 엣지 생성
    edges_src = []
    edges_dst = []

    for i in range(len(unique_users)):
        for j in range(i + 1, len(unique_users)):
            similarity = similarity_matrix[i, j]

            if similarity >= threshold:
                edges_src.append(i)
                edges_dst.append(j)
                # 무방향 그래프를 위해 역방향도 추가
                edges_src.append(j)
                edges_dst.append(i)

    if len(edges_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    edge_index = torch.unique(edge_index, dim=1)

    print(f"  엣지 수: {edge_index.shape[1]}")

    return edge_index

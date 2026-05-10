import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def extract_text_embedding(df, n_components=128):
    """
    TF-IDF 벡터화 후 SVD로 차원 축소 (128D)
    """
    print("  [TF-IDF] 텍스트 벡터화 중...")

    if 'text' not in df.columns:
        raise ValueError("'text' 컬럼이 없습니다.")

    texts = df['text'].fillna('').astype(str)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    print(f"    TF-IDF 형태: {tfidf_matrix.shape}")

    print("  [SVD] 차원 축소 중...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    text_embeddings = svd.fit_transform(tfidf_matrix)

    print(f"    SVD 형태: {text_embeddings.shape}")
    print(f"    설명 분산: {svd.explained_variance_ratio_.sum():.4f}")

    return text_embeddings, vectorizer, svd


def extract_numeric_features(df):
    """
    정형 특징 추출: rating, user_review_count, prod_review_count 등
    """
    print("  [Numeric] 정형 특징 추출 중...")

    features_list = []

    if 'rating' in df.columns:
        features_list.append(df[['rating']])

    if 'user_id' in df.columns:
        user_review_count = df.groupby('user_id').size()
        df_temp = df.copy()
        df_temp['user_review_count'] = df['user_id'].map(user_review_count)
        features_list.append(df_temp[['user_review_count']])

    if 'prod_id' in df.columns:
        prod_review_count = df.groupby('prod_id').size()
        df_temp = df.copy()
        df_temp['prod_review_count'] = df['prod_id'].map(prod_review_count)
        features_list.append(df_temp[['prod_review_count']])

    numeric_features = pd.concat(features_list, axis=1)

    print(f"    정형 특징 형태: {numeric_features.shape}")

    return numeric_features


def normalize_features(text_features, numeric_features):
    """
    특징 정규화
    """
    print("  [Normalize] 특징 정규화 중...")

    text_scaler = StandardScaler()
    text_features_norm = text_scaler.fit_transform(text_features)

    numeric_scaler = StandardScaler()
    numeric_features_norm = numeric_scaler.fit_transform(numeric_features)

    return text_features_norm, numeric_features_norm, text_scaler, numeric_scaler


def concatenate_features(text_features, numeric_features):
    """
    텍스트 + 정형 특징 결합
    """
    features = np.concatenate([text_features, numeric_features], axis=1)
    print(f"  최종 특징 형태: {features.shape}")
    return features


def save_features(df, features):
    """
    특징 저장
    """
    os.makedirs("data/processed", exist_ok=True)

    features_path = os.path.join("data/processed", "features.npy")
    np.save(features_path, features)
    print(f"[Save] {features_path}")

    nodes_path = os.path.join("data/processed", "node_samples.csv")
    df.to_csv(nodes_path, index=False)
    print(f"[Save] {nodes_path}")

    return df


if __name__ == "__main__":
    print("Feature engineering utilities loaded")

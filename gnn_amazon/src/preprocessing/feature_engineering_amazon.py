import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def aggregate_user_reviews(df):
    """
    사용자별 리뷰 집계
    각 사용자의 여러 리뷰를 하나로 집계하여 사용자 레벨의 데이터 생성
    """
    print("  [Aggregate] 사용자별 리뷰 집계 중...")

    # 사용자별로 리뷰 텍스트 합치기
    user_reviews = df.groupby('user_id').agg({
        'review_text': lambda x: ' '.join(x.astype(str)),
        'rating': ['mean', 'std', 'min', 'max', 'count'],
        'label': 'first',  # 사용자별로 라벨은 동일하다고 가정
    }).reset_index()

    # 컬럼명 정렬
    user_reviews.columns = [
        'user_id', 'review_text',
        'rating_mean', 'rating_std', 'rating_min', 'rating_max', 'review_count',
        'label'
    ]

    # NaN 처리
    user_reviews['rating_std'] = user_reviews['rating_std'].fillna(0)

    print(f"    사용자 수: {len(user_reviews)}")

    return user_reviews


def extract_text_embedding(df, n_components=128):
    """
    TF-IDF 벡터화 후 SVD로 차원 축소 (128D)
    """
    print("  [TF-IDF] 텍스트 벡터화 중...")

    if 'review_text' not in df.columns:
        raise ValueError("'review_text' 컬럼이 없습니다.")

    texts = df['review_text'].fillna('').astype(str)

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
    정형 특징 추출: rating_mean, rating_std, review_count 등
    """
    print("  [Numeric] 정형 특징 추출 중...")

    features_list = []

    numeric_cols = ['rating_mean', 'rating_std', 'rating_min', 'rating_max', 'review_count']
    for col in numeric_cols:
        if col in df.columns:
            features_list.append(df[[col]])

    if features_list:
        numeric_features = pd.concat(features_list, axis=1)
    else:
        raise ValueError("추출할 정형 특징이 없습니다. 컬럼명을 확인하세요.")

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
    사용자 특징 저장
    """
    os.makedirs("data/processed", exist_ok=True)

    features_path = os.path.join("data/processed", "features.npy")
    np.save(features_path, features)
    print(f"[Save] {features_path}")

    nodes_path = os.path.join("data/processed", "user_samples.csv")
    df.to_csv(nodes_path, index=False)
    print(f"[Save] {nodes_path}")

    return df


if __name__ == "__main__":
    print("Amazon feature engineering utilities loaded")

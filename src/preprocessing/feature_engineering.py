import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from src.utils import set_seed, save_object, save_json

set_seed(42)

CONFIG = {
    "processed_dir": "data/processed",
    "input_file": "sampled_reviews.csv",
    "tfidf_max_features": 50000,
    "tfidf_min_df": 3,
    "tfidf_max_df": 0.9,
    "tfidf_ngram": (1, 2),
    "svd_components": 128,
}


def extract_text_embedding(df):
    print("[Feature] 텍스트 임베딩 추출 중... (TF-IDF + SVD)")

    texts = df["text"].fillna("").values

    print(f"  TF-IDF Vectorizer 학습 중... ({len(texts)} 문서)")
    vectorizer = TfidfVectorizer(
        max_features=CONFIG["tfidf_max_features"],
        min_df=CONFIG["tfidf_min_df"],
        max_df=CONFIG["tfidf_max_df"],
        ngram_range=CONFIG["tfidf_ngram"],
        lowercase=True,
        token_pattern=r"\b\w+\b",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  TF-IDF 형태: {tfidf_matrix.shape}")

    print(f"  TruncatedSVD 학습 중... ({CONFIG['svd_components']} 차원)")
    svd = TruncatedSVD(n_components=CONFIG["svd_components"], random_state=42)
    text_embeddings = svd.fit_transform(tfidf_matrix)

    print(f"  텍스트 임베딩 형태: {text_embeddings.shape}")
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

    return text_embeddings, vectorizer, svd


def extract_numeric_features(df):
    print("\n[Feature] 정형 Feature 추출 중...")

    features = {}

    features["rating_norm"] = (df["rating"] - 3.0) / 2.0

    features["review_length"] = df["text"].fillna("").str.len()
    features["review_length_log"] = np.log1p(features["review_length"])

    user_stats = df.groupby("user_id").agg({
        "review_id": "count",
        "rating": ["mean", "std"],
    }).fillna(0)
    user_stats.columns = ["user_review_count", "user_avg_rating", "user_rating_std"]

    features["user_review_count_log"] = np.log1p(
        df["user_id"].map(user_stats["user_review_count"])
    )
    features["user_avg_rating"] = df["user_id"].map(user_stats["user_avg_rating"])
    features["user_rating_std"] = df["user_id"].map(user_stats["user_rating_std"])

    product_stats = df.groupby("prod_id").agg({
        "review_id": "count",
        "rating": ["mean", "std"],
    }).fillna(0)
    product_stats.columns = ["product_review_count", "product_avg_rating", "product_rating_std"]

    features["product_review_count_log"] = np.log1p(
        df["prod_id"].map(product_stats["product_review_count"])
    )
    features["product_avg_rating"] = df["prod_id"].map(product_stats["product_avg_rating"])
    features["product_rating_std"] = df["prod_id"].map(product_stats["product_rating_std"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    min_date = df["date"].min()
    features["days_since_first_review"] = (df["date"] - min_date).dt.days

    month_num = df["date"].dt.month
    features["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    features["month_cos"] = np.cos(2 * np.pi * month_num / 12)

    feature_df = pd.DataFrame(features)
    feature_df = feature_df.fillna(0)

    print(f"  생성된 Feature 수: {feature_df.shape[1]}")
    print(f"  Feature 목록: {list(feature_df.columns)}")

    return feature_df


def normalize_features(text_embeddings, numeric_features):
    print("\n[Feature] 정규화 중...")

    text_scaler = StandardScaler()
    text_embeddings_normalized = text_scaler.fit_transform(text_embeddings)

    numeric_scaler = StandardScaler()
    numeric_features_normalized = numeric_scaler.fit_transform(numeric_features)

    print(f"  텍스트 임베딩 (정규화): {text_embeddings_normalized.shape}")
    print(f"  정형 Feature (정규화): {numeric_features_normalized.shape}")

    return text_embeddings_normalized, numeric_features_normalized, text_scaler, numeric_scaler


def concatenate_features(text_emb, numeric_feat):
    print("\n[Feature] Feature 결합 중...")

    combined = np.concatenate([text_emb, numeric_feat], axis=1)
    print(f"  최종 Feature 형태: {combined.shape}")

    return combined


def save_features(df, combined_features):
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)

    x_path = os.path.join(CONFIG["processed_dir"], "features.npy")
    np.save(x_path, combined_features)
    print(f"\n[Save] {x_path}")

    samples_path = os.path.join(CONFIG["processed_dir"], "node_samples.csv")
    df.to_csv(samples_path, index=False)
    print(f"[Save] {samples_path}")

    meta = {
        "num_nodes": combined_features.shape[0],
        "num_features": combined_features.shape[1],
        "text_embedding_dim": 128,
        "numeric_features_dim": combined_features.shape[1] - 128,
    }
    meta_path = os.path.join(CONFIG["processed_dir"], "feature_meta.json")
    save_json(meta, meta_path)
    print(f"[Save] {meta_path}")

    return combined_features


if __name__ == "__main__":
    input_path = os.path.join(CONFIG["processed_dir"], "sampled_reviews.csv")
    df = pd.read_csv(input_path)

    text_embeddings, vectorizer, svd = extract_text_embedding(df)

    numeric_features = extract_numeric_features(df)

    text_norm, numeric_norm, text_scaler, numeric_scaler = normalize_features(
        text_embeddings, numeric_features.values
    )

    combined_features = concatenate_features(text_norm, numeric_norm)

    combined_features = save_features(df, combined_features)

    print("\n[Done] Phase 2-4: Feature Engineering 완료")

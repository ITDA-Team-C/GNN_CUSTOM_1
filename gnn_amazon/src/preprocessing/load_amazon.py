import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import set_seed

set_seed(42)

CONFIG = {
    "raw_dir": "data/raw",
    "interim_dir": "data/interim",
    "filename": "amazon.csv",  # 사용자가 데이터 파일명 수정 가능
}


def load_amazon():
    """
    Amazon 데이터 로드

    필요한 컬럼:
    - user_id: 사용자 ID
    - prod_id: 제품 ID
    - rating: 별점 (1-5)
    - review_text: 리뷰 텍스트
    - label: 0 (정상 사용자), 1 (사기꾼 사용자)
    """
    raw_path = os.path.join(CONFIG["raw_dir"], CONFIG["filename"])

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Amazon 파일을 찾을 수 없습니다: {raw_path}\n"
            f"data/raw/ 폴더에 Amazon CSV 파일을 배치하세요.\n"
            f"필수 컬럼: user_id, prod_id, rating, review_text, label"
        )

    print(f"[Load] {raw_path} 로딩 중...")
    df = pd.read_csv(raw_path)
    print(f"  총 행 수: {len(df)}")
    print(f"  컬럼: {list(df.columns)}")

    return df


def validate_columns(df):
    """
    필수 컬럼 검증
    """
    required_cols = {"user_id", "prod_id", "rating", "review_text", "label"}
    existing_cols = set(df.columns)

    missing_cols = required_cols - existing_cols

    if missing_cols:
        print(f"[Warning] 누락된 컬럼: {missing_cols}")
        print(f"  필수 컬럼: {required_cols}")

    return df


def save_raw_eda(df):
    """
    원본 데이터 EDA 저장
    """
    os.makedirs(CONFIG["interim_dir"], exist_ok=True)

    eda_path = os.path.join(CONFIG["interim_dir"], "raw_eda.txt")

    with open(eda_path, "w", encoding="utf-8") as f:
        f.write("=== Amazon Raw Data EDA ===\n\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n\n")

        f.write("Column Info:\n")
        f.write(df.info(buf=None).__str__() + "\n\n")

        f.write("Numeric Columns:\n")
        f.write(df.describe().to_string() + "\n\n")

        f.write("Missing Values:\n")
        f.write(df.isnull().sum().to_string() + "\n\n")

        if "label" in df.columns:
            f.write("Label Distribution:\n")
            f.write(df["label"].value_counts().to_string() + "\n\n")

        if "user_id" in df.columns:
            f.write(f"Unique Users: {df['user_id'].nunique()}\n")

        if "prod_id" in df.columns:
            f.write(f"Unique Products: {df['prod_id'].nunique()}\n")

        if "rating" in df.columns:
            f.write(f"\nRating Distribution:\n")
            f.write(df["rating"].value_counts().sort_index().to_string() + "\n")

    print(f"[Save] EDA 저장됨: {eda_path}")


def save_processed_data(df):
    """
    처리된 데이터 저장
    """
    os.makedirs(CONFIG["interim_dir"], exist_ok=True)

    save_path = os.path.join(CONFIG["interim_dir"], "raw_data.csv")
    df.to_csv(save_path, index=False)
    print(f"[Save] {save_path}")

    return df


if __name__ == "__main__":
    df = load_amazon()
    df = validate_columns(df)
    save_raw_eda(df)
    df = save_processed_data(df)
    print("\n[Done] Amazon Data Loading 완료")

import os
import pandas as pd
import numpy as np
import scipy.io as sio
from pathlib import Path
from src.utils import set_seed

set_seed(42)

CONFIG = {
    "raw_dir": "data/raw",
    "interim_dir": "data/interim",
    "filename": "amazon.mat",  # .mat 파일
}


def load_amazon():
    """
    Amazon 데이터 로드 (.mat 파일)

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
            f"data/raw/ 폴더에 Amazon .mat 파일을 배치하세요.\n"
            f"필수 컬럼: user_id, prod_id, rating, review_text, label"
        )

    print(f"[Load] {raw_path} 로딩 중...")

    # .mat 파일 로드
    mat_data = sio.loadmat(raw_path)

    # 메타 정보 제거 (MATLAB은 메타 정보를 자동으로 추가)
    mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

    print(f"  .mat 파일 내 변수: {list(mat_data.keys())}")

    # DataFrame 변환
    df = _convert_mat_to_dataframe(mat_data)

    print(f"  총 행 수: {len(df)}")
    print(f"  컬럼: {list(df.columns)}")

    return df


def _convert_mat_to_dataframe(mat_data):
    """
    MATLAB 구조체를 DataFrame으로 변환
    필요에 따라 수정하세요
    """
    data_dict = {}

    # 일반적인 변수명들
    possible_keys = ['user_id', 'prod_id', 'product_id', 'rating', 'review_text', 'text', 'label']

    for key in possible_keys:
        if key in mat_data:
            data = mat_data[key]
            # numpy 배열을 리스트로 변환
            if isinstance(data, np.ndarray):
                if data.dtype.kind in ['U', 'S', 'O']:  # 문자열
                    data_dict[key] = [str(x).strip() for x in data.flatten()]
                else:  # 숫자
                    data_dict[key] = data.flatten().tolist()

    if not data_dict:
        raise ValueError(
            f"예상되는 변수를 찾을 수 없습니다.\n"
            f"사용 가능한 변수: {list(mat_data.keys())}\n"
            f"load_amazon.py의 _convert_mat_to_dataframe() 함수를 수정하세요."
        )

    return pd.DataFrame(data_dict)



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

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
    "filename": "yelpchi.mat",  # .mat 파일
}


def load_yelpchi():
    """
    YelpChi 데이터 로드 (.mat 파일)

    구조:
    - rur: Review-User-Review 그래프 (sparse)
    - rtr: Review-Tag-Review 그래프 (sparse)
    - rsr: Review-Store-Review 그래프 (sparse)
    - features: 노드 특징
    - label: 노드 라벨
    """
    raw_path = os.path.join(CONFIG["raw_dir"], CONFIG["filename"])

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"YelpChi 파일을 찾을 수 없습니다: {raw_path}\n"
            f"data/raw/ 폴더에 yelpchi.mat 파일을 배치하세요."
        )

    print(f"[Load] {raw_path} 로딩 중...")

    # .mat 파일 로드
    mat_data = sio.loadmat(raw_path)

    # 메타 정보 제거 (MATLAB은 메타 정보를 자동으로 추가)
    mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

    print(f"  .mat 파일 내 변수: {list(mat_data.keys())}")

    # 데이터 구조 확인
    from scipy import sparse

    # Label 처리
    label = mat_data['label']
    if label.ndim > 1:
        label = label.flatten()
    label = label.astype(int)

    # Features 처리
    features = mat_data['features']
    if sparse.issparse(features):
        features = features.toarray()

    num_nodes = len(label)

    result = {
        'num_nodes': num_nodes,
        'label': label,
        'features': features,
    }

    # 그래프 구조
    graph_keys = ['rur', 'rtr', 'rsr']
    for key in graph_keys:
        if key in mat_data:
            adj = sparse.csr_matrix(mat_data[key])
            # Clip adjacency matrix to num_nodes
            if adj.shape[0] > num_nodes or adj.shape[1] > num_nodes:
                print(f"  WARNING: {key} shape {adj.shape} exceeds num_nodes {num_nodes}, clipping...")
                adj = adj[:num_nodes, :num_nodes]
            result[key] = adj

    print(f"  Num nodes: {result['num_nodes']}")
    print(f"  Feature shape: {result['features'].shape}")
    print(f"  Label distribution: {np.bincount(result['label'])}")

    return result


def _convert_mat_to_dataframe(mat_data):
    """
    MATLAB 구조체를 DataFrame으로 변환
    필요에 따라 수정하세요
    """
    data_dict = {}

    # 일반적인 변수명들
    possible_keys = ['review_id', 'user_id', 'prod_id', 'text', 'date', 'rating', 'label']

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
            f"load_yelpchi.py의 _convert_mat_to_dataframe() 함수를 수정하세요."
        )

    return pd.DataFrame(data_dict)


def _convert_mat_array_to_dataframe(mat_data):
    """
    .mat 배열을 DataFrame으로 변환
    """
    # 가장 큰 배열 찾기
    largest_key = max(
        (k for k, v in mat_data.items() if isinstance(v, np.ndarray)),
        key=lambda k: mat_data[k].size
    )

    data = mat_data[largest_key]

    if data.ndim == 2:
        # 2D 배열: 컬럼명 필요
        columns = [f'col_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=columns)
    else:
        # 1D 배열: 단일 컬럼
        return pd.DataFrame({'data': data.flatten()})



def validate_columns(df):
    required_cols = {"review_id", "user_id", "prod_id", "text", "date", "rating", "label"}
    existing_cols = set(df.columns)

    missing_cols = required_cols - existing_cols

    if missing_cols:
        print(f"[Warning] 누락된 컬럼: {missing_cols}")

        if "review_id" not in existing_cols:
            print(f"  review_id 자동 생성 (index 기반)")
            df["review_id"] = np.arange(len(df))

    return df


def save_raw_eda(df):
    os.makedirs(CONFIG["interim_dir"], exist_ok=True)

    eda_path = os.path.join(CONFIG["interim_dir"], "raw_eda.txt")

    with open(eda_path, "w", encoding="utf-8") as f:
        f.write("=== YelpChi Raw Data EDA ===\n\n")
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
            f.write(df["label"].value_counts().to_string() + "\n")

    print(f"[Save] EDA 저장됨: {eda_path}")


def save_processed_data(df):
    os.makedirs(CONFIG["interim_dir"], exist_ok=True)

    save_path = os.path.join(CONFIG["interim_dir"], "raw_data.csv")
    df.to_csv(save_path, index=False)
    print(f"[Save] {save_path}")

    return df


if __name__ == "__main__":
    df = load_yelpchi()
    df = validate_columns(df)
    save_raw_eda(df)
    df = save_processed_data(df)
    print("\n[Done] Phase 2-1: Data Loading 완료")

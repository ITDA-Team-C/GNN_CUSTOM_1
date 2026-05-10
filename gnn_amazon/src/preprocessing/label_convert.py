import pandas as pd
import numpy as np


def label_convert(df):
    """
    Amazon 라벨 변환
    필요한 경우만 수정 (보통 0: 정상, 1: 사기꾼)
    """
    if 'label' not in df.columns:
        raise ValueError("'label' 컬럼이 없습니다.")

    df = df.copy()

    # 라벨 값 확인
    unique_labels = df['label'].unique()
    print(f"[Label] 고유 라벨 값: {unique_labels}")

    # 필요한 경우만 변환 (예: -1 → 1, 1 → 0 같은 매핑)
    # 기본적으로 Amazon은 0/1로 되어있다고 가정

    return df


if __name__ == "__main__":
    print("Label conversion utility loaded")

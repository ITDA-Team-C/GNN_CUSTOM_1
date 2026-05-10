import pandas as pd
import numpy as np


def label_convert(df):
    """
    YelpChi 라벨 변환: -1 (spam/bot) → 1, 1 (benign) → 0
    """
    if 'label' not in df.columns:
        raise ValueError("'label' 컬럼이 없습니다.")

    df = df.copy()
    df['label'] = df['label'].map({-1: 1, 1: 0})

    if df['label'].isnull().sum() > 0:
        print("[Warning] 예상치 못한 라벨 값이 있습니다. NaN으로 처리됨.")
        df = df.dropna(subset=['label'])

    return df


if __name__ == "__main__":
    print("Label conversion utility loaded")

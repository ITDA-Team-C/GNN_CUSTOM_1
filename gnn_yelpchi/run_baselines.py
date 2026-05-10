#!/usr/bin/env python3
"""
YelpChi 기본 모델 학습 스크립트
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import set_seed

set_seed(42)


def main():
    print("\n" + "="*80)
    print("🚀 YelpChi 기본 모델 학습 시작")
    print("="*80)

    print("\n[Config]")
    print("  Dataset: YelpChi")
    print("  Output: outputs/output_yelpchi/")
    print("  Data Path: data/processed/")

    print("\n[Notice] 모델 학습 코드는 기존 프로젝트에서 가져와 적용하세요.")
    print("  - GCN, GraphSAGE, GAT 등 모델 구현")
    print("  - Training loop 구현")
    print("  - 결과 저장: outputs/output_yelpchi/results/")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

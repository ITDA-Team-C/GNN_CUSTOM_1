#!/usr/bin/env python3
"""
Amazon 기본 모델 학습 스크립트
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import set_seed

set_seed(42)


def main():
    print("\n" + "="*80)
    print("🚀 Amazon 기본 모델 학습 시작")
    print("="*80)

    print("\n[Config]")
    print("  Dataset: Amazon (악기 카테고리)")
    print("  Node Type: User (사용자)")
    print("  Output: outputs/output_amazon/")
    print("  Data Path: data/processed/")

    print("\n[Graph Relations]")
    print("  - U-P-U: User-Product-User (동일 제품 리뷰 사용자)")
    print("  - U-S-U: User-Star-User (동일 제품, 동일 별점)")
    print("  - U-V-U: User-TextSim-User (리뷰 텍스트 유사도)")

    print("\n[Notice] 모델 학습 코드는 기존 프로젝트에서 가져와 적용하세요.")
    print("  - GCN, GraphSAGE, GAT 등 모델 구현")
    print("  - Training loop 구현 (사용자 노드 기반)")
    print("  - 결과 저장: outputs/output_amazon/results/")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

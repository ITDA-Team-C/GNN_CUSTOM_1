# YelpChi GNN 프로젝트

YelpChi 데이터셋을 사용한 그래프 신경망(GNN) 기반 사기/스팸 리뷰 탐지

## 📊 데이터셋 개요

- **노드**: Review (각 리뷰)
- **라벨**: 0 (정상 리뷰), 1 (스팸/봇 리뷰)
- **그래프 관계** (3가지 엣지):
  - R-U-R: Review-User-Review (같은 사용자의 리뷰들)
  - R-T-R: Review-Time-Review (같은 제품, 같은 월의 리뷰들)
  - R-S-R: Review-Star-Review (같은 제품, 같은 별점의 리뷰들)

## 🚀 사용 방법

### 1. 환경 설정
```bash
# 라이브러리 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
# data/raw/ 폴더에 yelp_zip.csv 파일 배치
# 필수 컬럼: review_id, user_id, prod_id, text, date, rating, label
```

### 3. 전처리 실행
```bash
cd gnn_yelpchi
python preprocess.py
```

**전처리 단계:**
1. 데이터 로드 및 검증
2. 라벨 변환 (-1→1, 1→0)
3. 리뷰 샘플링 (25,000개)
4. Train/Val/Test 분할 (6:2:2)
5. Feature Engineering (TF-IDF + SVD + 정형 특징)
6. 그래프 구성 (R-U-R, R-T-R, R-S-R)

**출력 파일:**
- `data/processed/node_samples.csv`: 샘플링된 리뷰 정보
- `data/processed/features.npy`: 노드 특징 (128 + α 차원)
- `data/processed/edge_index_dict.pkl`: 그래프 엣지

### 4. 모델 학습
```bash
python run_baselines.py
```

**결과 저장:**
- `outputs/output_yelpchi/models/`: 학습된 모델
- `outputs/output_yelpchi/results/`: 성능 지표 (PR-AUC, Macro-F1)

## 📂 폴더 구조

```
gnn_yelpchi/
├── src/
│   ├── preprocessing/    # 데이터 전처리
│   │   ├── load_yelpzip.py
│   │   ├── label_convert.py
│   │   ├── sampling.py
│   │   └── feature_engineering.py
│   ├── graph/           # 그래프 구성
│   │   ├── build_rur.py
│   │   ├── build_rtr.py
│   │   ├── build_rsr.py
│   │   └── build_relations.py
│   ├── models/          # GNN 모델들
│   ├── training/        # 학습 로직
│   └── utils/
├── data/
│   ├── raw/             # 원본 데이터
│   ├── interim/         # 중간 결과
│   └── processed/       # 최종 처리 데이터
├── outputs/
│   └── output_yelpchi/  # 학습 결과
├── preprocess.py        # 전처리 메인 스크립트
├── run_baselines.py     # 모델 학습 스크립트
└── requirements.txt
```

## 🔧 설정 수정

각 모듈의 CONFIG 딕셔너리를 수정하여 커스터마이징 가능:

- `load_yelpzip.py`: 데이터 파일명, 디렉토리
- `sampling.py`: 샘플 크기, 분할 비율
- `feature_engineering.py`: SVD 차원, TF-IDF 파라미터
- `build_*.py`: 엣지 k값 (이웃 수 제한)

## 📈 예상 성능

기본 Focal Loss + Auxiliary Loss 설정:
- PR-AUC: ~0.30
- Macro-F1: ~0.25 (조정 가능)

## 🤝 주의사항

1. 기존 프로젝트 코드와 완전히 독립적
2. YelpChi 데이터에 최적화됨
3. Amazon 데이터셋 사용 시 `gnn_amazon/` 폴더 사용

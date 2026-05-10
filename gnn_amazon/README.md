# Amazon GNN 프로젝트

Amazon 악기 카테고리 데이터셋을 사용한 그래프 신경망(GNN) 기반 사기꾼 사용자 탐지

## 📊 데이터셋 개요

- **노드**: User (각 사용자)
- **라벨**: 0 (정상 사용자), 1 (사기꾼 사용자)
- **그래프 관계** (3가지 엣지):
  - U-P-U: User-Product-User (동일 제품에 리뷰한 사용자들)
  - U-S-U: User-Star-User (동일 제품, 동일 별점을 준 사용자들)
  - U-V-U: User-Vocab/TextSim-User (리뷰 텍스트 유사도 상위 5%)

## 🚀 사용 방법

### 1. 환경 설정
```bash
# 라이브러리 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
# data/raw/ 폴더에 amazon.csv 파일 배치
# 필수 컬럼: user_id, prod_id, rating, review_text, label

# 파일명 변경 시 load_amazon.py의 CONFIG["filename"] 수정
```

**데이터 형식 예시:**
```
user_id,prod_id,rating,review_text,label
123,456,5,"Great product!",0
124,456,5,"Good quality",0
125,789,1,"Bad product",1
...
```

### 3. 전처리 실행
```bash
cd gnn_amazon
python preprocess.py
```

**전처리 단계:**
1. 데이터 로드 및 검증
2. 라벨 확인 (이미 0/1 형식이라 가정)
3. 사용자 샘플링 (선택사항, 기본: 모든 사용자)
4. Train/Val/Test 분할 (사용자 단위, 6:2:2)
5. **사용자별 리뷰 집계** (중요!)
6. Feature Engineering (TF-IDF + SVD + 정형 특징)
7. 그래프 구성 (U-P-U, U-S-U, U-V-U)

**출력 파일:**
- `data/processed/user_samples.csv`: 사용자 정보 (집계된 특징)
- `data/processed/features.npy`: 사용자 특징 (128 + α 차원)
- `data/processed/edge_index_dict.pkl`: 그래프 엣지

### 4. 모델 학습
```bash
python run_baselines.py
```

**결과 저장:**
- `outputs/output_amazon/models/`: 학습된 모델
- `outputs/output_amazon/results/`: 성능 지표 (PR-AUC, Macro-F1)

## 📂 폴더 구조

```
gnn_amazon/
├── src/
│   ├── preprocessing/           # 데이터 전처리
│   │   ├── load_amazon.py       # 데이터 로드
│   │   ├── label_convert.py     # 라벨 처리
│   │   ├── sampling_amazon.py   # 사용자 샘플링
│   │   └── feature_engineering_amazon.py
│   ├── graph/                   # 그래프 구성
│   │   ├── build_upu.py         # User-Product-User
│   │   ├── build_usu.py         # User-Star-User
│   │   ├── build_uvu.py         # User-TextSim-User
│   │   └── build_relations_amazon.py
│   ├── models/                  # GNN 모델들
│   ├── training/                # 학습 로직
│   └── utils/
├── data/
│   ├── raw/                     # 원본 데이터
│   ├── interim/                 # 중간 결과
│   └── processed/               # 최종 처리 데이터
├── outputs/
│   └── output_amazon/           # 학습 결과
├── preprocess.py                # 전처리 메인 스크립트
├── run_baselines.py             # 모델 학습 스크립트
└── requirements.txt
```

## 🔍 핵심 차이점 (YelpChi vs Amazon)

| 항목 | YelpChi | Amazon |
|------|---------|--------|
| 노드 단위 | Review | User |
| 샘플 크기 | 25,000 리뷰 | N 사용자 |
| 엣지 타입 | R-U-R, R-T-R, R-S-R | U-P-U, U-S-U, U-V-U |
| 특징 추출 | 리뷰 텍스트 | 사용자별 집계 텍스트 |
| 피처 수 | 128 + α | 128 + α (동일 구조) |

## 🔧 설정 수정

각 모듈의 CONFIG와 파라미터 커스터마이징:

- `load_amazon.py`: 데이터 파일명
- `sampling_amazon.py`: `target_users` (사용자 샘플 수)
- `feature_engineering_amazon.py`: SVD 차원, TF-IDF 파라미터
- `build_uvu.py`: `top_percentile` (유사도 임계값, 기본 95%)
- `build_upu.py`, `build_usu.py`: k값 (이웃 수 제한)

**파라미터 예시:**
```python
# preprocess.py에서
sampled_df = user_based_sampling(df, target_users=5000)  # 5000명 사용자 샘플링
```

## ⚙️ U-V-U 엣지 구성 상세

텍스트 유사도 기반 사용자 연결:
1. 각 사용자의 모든 리뷰를 합침
2. TF-IDF 벡터화
3. 코사인 유사도 계산
4. 상위 95% 유사도 이상인 사용자 쌍 연결

```python
# build_uvu.py 파라미터 조정 예시
edge_index_dict["uvu"] = build_uvu(df, top_percentile=90)  # 상위 10%만
```

## 📈 예상 성능

기본 설정 (Focal Loss + Auxiliary Loss):
- PR-AUC: 변동 (Amazon 데이터 특성에 따라)
- Macro-F1: 조정 가능

## 🤝 주의사항

1. **노드 레벨 차이**: YelpChi(리뷰) vs Amazon(사용자)
2. **Feature 집계**: `aggregate_user_reviews()`에서 사용자별 데이터로 변환
3. **그래프 구성**: 원본 리뷰 데이터 사용 (사용자 인덱싱)
4. 기존 프로젝트와 완전히 독립적
5. YelpChi와 병렬로 실험 가능

## 📝 데이터 로딩 커스터마이징

사용자가 직접 데이터를 로드하려면 `load_amazon.py` 수정:

```python
# 기본 (CSV 파일)
df = pd.read_csv('path/to/data.csv')

# 다른 형식 (Parquet, JSON 등)
df = pd.read_parquet('path/to/data.parquet')

# 필수: 컬럼명 확인
required_cols = {"user_id", "prod_id", "rating", "review_text", "label"}
```

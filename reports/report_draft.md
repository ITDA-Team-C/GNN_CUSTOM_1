# 그래프신경망 기반 조직적 어뷰징 네트워크 탐지

## 1. 모델링 목표 정의

본 연구는 YelpZip 리뷰 데이터셋을 활용하여 **조직적 어뷰징 네트워크 탐지**를 목표로 한다.

기존의 NLP 기반 텍스트 분류 접근은 개별 리뷰의 텍스트만 분석하여, 다음과 같은 조직적 패턴을 놓치는 한계가 있다:
- 동일 사용자의 반복적 리뷰 작성
- 특정 기간에 리뷰 집중
- 동일 별점의 비정상적 반복
- 서로 다른 계정이 유사 문장 반복
- 비슷한 행동 패턴의 계정들

따라서 **GNN(Graph Neural Network)**을 활용하여 리뷰 간의 관계를 그래프로 구성하고, 관계 정보를 함께 학습하는 접근을 제시한다.

---

## 2. 데이터 구성 및 전처리 과정

### 2.1 사용 데이터

**YelpZip 원본 리뷰 데이터셋** (60만+ 리뷰)

필수 컬럼:
- `review_id`: 리뷰 고유 ID
- `user_id`: 작성자 ID
- `prod_id`: 식당/상품 ID
- `text`: 리뷰 원문 텍스트
- `date`: 작성 날짜
- `rating`: 별점 (1~5점)
- `label`: 사기 여부 (-1: 사기, 1: 정상)

### 2.2 라벨 변환

대회 규정에 따라 다음과 같이 라벨을 변환하였다:

```
원본: -1 (사기) → 변환: 1
원본:  1 (정상) → 변환: 0
```

변환 이유: 모델 학습 시 양성(Positive)을 사기 리뷰로 명확히 정의하기 위함.

### 2.3 샘플링 전략

전체 60만+ 리뷰에서 효율적인 그래프 학습을 위해 **Product-User-Time Hybrid Dense Sampling**을 적용하였다.

샘플링 조건:
1. 리뷰 수가 많은 상위 Product 선택
2. 활동량이 높은 User 선택
3. 리뷰 집중 Time-window (월단위) 선택
4. 위 조건의 union으로 통합 샘플링

목표: 25,000 노드
범위: 10,000 ~ 50,000 노드

이유: 무작위 샘플링은 노드 간 연결성을 파괴하여 GNN 학습 불가능

### 2.4 Train/Valid/Test 분할

샘플링 후 서브그래프 내에서 stratified split:
- **Train**: 64% (라벨 기반 분층)
- **Valid**: 16% (하이퍼파라미터 튜닝, threshold 탐색)
- **Test**: 20% (최종 평가)

Random state: 42 (재현성 보장)

### 2.5 피처 엔지니어링

#### 텍스트 임베딩
- TF-IDF Vectorizer (max_features=50,000)
- TruncatedSVD 차원 축소 (128차원)
- 이유: GPU 환경에 최적화

#### 정형 Features
- `rating_norm`: 정규화된 별점
- `review_length_log`: 리뷰 길이 로그
- `user_review_count_log`: 사용자 리뷰 수 로그
- `user_avg_rating`, `user_rating_std`: 사용자 별점 통계
- `product_review_count_log`: 상품 리뷰 수 로그
- `product_avg_rating`, `product_rating_std`: 상품 별점 통계
- `days_since_first_review`: 첫 리뷰 이후 경과일
- `month_sin`, `month_cos`: 월도 순환 인코딩

정규화: StandardScaler 적용

최종 노드 feature: [텍스트 임베딩 128D + 정형 features ~11D] = 총 139차원

### 2.6 그래프 엣지 설계

#### 기본 Relation (3개)

**R-U-R (Review-User-Review)**
- 같은 사용자가 작성한 리뷰끼리 연결
- Top-k: 10 (각 리뷰당 같은 사용자의 최대 10개 리뷰)
- 의미: 동일 사용자의 반복 패턴 탐지

**R-T-R (Review-Time-Review)**
- 같은 상품에 대해 같은 월(month)에 작성된 리뷰끼리 연결
- Top-k: 10 (시간적으로 가까운 순)
- 의미: 특정 기간 리뷰 집중 패턴 탐지

**R-S-R (Review-Star-Review)**
- 같은 상품에 대해 같은 별점을 준 리뷰끼리 연결
- Top-k: 10
- 의미: 특정 별점으로 평판 조작 패턴 탐지

#### 커스텀 Relation (3개)

**R-Burst-R (Burst Review Pattern)**
- 조건: 같은 상품, 7일 내, 별점 차이 ≤ 1
- Top-k: 10 (시간 순)
- 의미: 단기 집중적 리뷰 조작 캠페인 탐지

**R-SemSim-R (Semantic Similarity)**
- 조건: 같은 상품 내에서 텍스트 의미 유사도 높음
- 유사도: Cosine similarity (128D SVD embedding)
- Top-k: 5
- 의미: 서로 다른 계정이 유사 문장 반복 패턴 탐지

**R-UserBehavior-R (User Behavior Similarity)**
- 조건: 사용자 행동 feature 유사도 높음
  - review_count, avg_rating, rating_std, active_days, product_diversity
  - Cosine similarity (정규화된 행동 feature)
- Top-k: 5 (similarity > 0.3)
- 의미: 계정은 다르지만 유사한 방식으로 활동하는 작업장 패턴 탐지

---

## 3. 모델 선택 이유와 적용 과정

### 3.1 선택한 모델

**CAGE-RF GNN** (Camouflage-Aware Gated Relation-Fusion GNN)

### 3.2 모델 선택 이유

1. **Multi-Relation 처리**: 6개의 relation을 각각 독립적으로 학습
   - 각 relation의 특성을 최대한 보존
   - Relation 간 간섭 최소화

2. **Relation Gating**: 관계별 중요도를 학습
   - 어떤 relation이 사기 탐지에 더 중요한지 동적으로 학습
   - 설명가능성 제공 (relation contribution 시각화)

3. **Camouflage-Aware**: 정상처럼 위장한 이웃의 영향 감소
   - Top-k, threshold 기반 neighbor filtering
   - Sparse edge 사용으로 불필요한 엣지 제거

### 3.3 모델 구조

```
Input: x (노드 feature 139D), edge_index_dict (6개 relation)
  ↓
Relation-specific GNN Branch (6개, 3층 SAGEConv)
  - R-U-R branch (SAGEConv 3층, hidden=128)
  - R-T-R branch (SAGEConv 3층, hidden=128)
  - R-S-R branch (SAGEConv 3층, hidden=128)
  - R-Burst-R branch (SAGEConv 3층, hidden=128)
  - R-SemSim-R branch (SAGEConv 3층, hidden=128)
  - R-UserBehavior-R branch (SAGEConv 3층, hidden=128)
  ↓
Embedding Concat: [h_rur, h_rtr, h_rsr, h_burst, h_semsim, h_behavior] (6*128=768D)
  ↓
Relation Weight Gate: α = softmax(MLP(h_stack))
  ↓
Gated Fusion: h_fused = Σ(α_i * h_i) (128D)
  ↓
Projection: h_proj = Linear(768D → 128D) + ReLU
  ↓
Classifier: logit = MLP(128D → 1D) → fraud_score = sigmoid(logit)
```

**GPU 최적화**:
- Batch size: 1024
- Learning rate: 0.001
- Epochs: 200
- Hidden dim: 128 (CPU 64 → GPU 128)
- Layers: 3 (CPU 2 → GPU 3)

### 3.4 Baseline 모델 비교

실험 대상:
- **MLP**: 노드 feature만 사용 (relation 정보 미사용)
- **GCN**: R-U-R만 사용 (단일 relation)
- **GraphSAGE**: R-U-R만 사용 (단일 relation)
- **GAT**: R-U-R만 사용 (Attention 기반)
- **CAGE-RF GNN**: 6개 relation + gating (제안 모델)

예상 성능 순서: MLP < GCN ≈ GraphSAGE < GAT < CAGE-RF GNN

---

## 4. 모델 성능 평가 & 결과 해석

### 4.1 평가 지표

**메인 지표 (필수)**:
- **PR-AUC** (Precision-Recall AUC): 불균형 데이터 분류의 표준 지표
- **Macro F1**: 클래스별 F1 점수 평균

**보조 지표**:
- ROC-AUC: 분류 성능 종합
- Precision, Recall: 정밀도/재현율
- Accuracy: 정확도
- Confusion Matrix: 분류 결과 분석

### 4.2 모델 성능 결과

(이 부분은 실제 학습 후 채워집니다)

**예상 성능 범위** (GPU 최적화 설정):
- PR-AUC: 0.75 ~ 0.88 (GPU 최적화로 향상)
- Macro F1: 0.65 ~ 0.80 (GPU 최적화로 향상)

최종 threshold는 validation set에서 Macro F1이 최대인 지점 선택.

### 4.3 결과 해석

(실제 결과 도출 후 분석)

---

## 5. Ablation Study

모델의 창의성과 각 요소의 효과를 검증:

- **A**: 기본 relation (R-U-R, R-T-R, R-S-R)만 사용
- **B**: A + R-Burst-R 추가
- **C**: B + R-SemSim-R 추가
- **D**: C + R-UserBehavior-R 추가
- **E**: D + relation concat (gating 없음)
- **F**: E + relation gating (제안 모델)

기대 효과:
- B에서 단기 집중 패턴 탐지 개선
- C에서 유사 텍스트 패턴 탐지 개선
- D에서 행동 유사도 기반 탐지 개선
- F에서 relation 중요도 자동 학습

---

## 6. 모델 기반 인사이트 및 활용 방안

### 인사이트

1. **Relation 중요도 분석**
   - Gating alpha 값으로 각 relation의 기여도 파악
   - 어떤 관계가 사기 탐지에 가장 유효한지 파악

2. **의심 리뷰의 증거**
   - 선택 리뷰의 상위 N개 이웃 리뷰 제시
   - ego-network로 주변 리뷰와의 관계 시각화

3. **조직적 어뷰징 네트워크 탐지**
   - 연결된 의심 리뷰들의 군집 분석
   - 동일 조직의 활동 패턴 파악

### 활용 방안

1. **플랫폼 사운드니스 보강**
   - 자동 모니터링으로 조직적 어뷰징 탐지
   - 수동 검토 대상 우선순위 지정

2. **정책 수립**
   - Relation별 의심 패턴에 따른 차별화된 규제
   - 타이밍 기반 규제 (단기 집중 리뷰 제한)

3. **사용자 신뢰도 평가**
   - 누적 relation 기반 신뢰도 스코어
   - 계정 위험도 분류

---

## 7. 모델링의 한계와 보완 계획

### 현재 한계

1. **CPU 환경의 성능 제약**
   - 초대규모 모델 학습 불가능
   - 배치 사이즈 제한

2. **데이터 불균형**
   - 음성(정상) 데이터 과다
   - Weighted BCE Loss로 부분 보완하나 완벽하지 않음

3. **Temporal 정보 미흡**
   - 월 단위 시간 정보만 사용
   - 더 세밀한 시간 패턴 미포착

### 보완 계획

1. **GPU 활용**
   - GPU 환경에서 더 큰 모델 학습
   - 더 많은 relation 확장

2. **데이터 증강**
   - Over-sampling (SMOTE)
   - 합성 데이터 생성

3. **Advanced Temporal Modeling**
   - RNN/Transformer 기반 시계열 모델 결합
   - 더 세밀한 시간 window (주, 일 단위)

4. **Heterogeneous GNN**
   - 사용자/상품 노드 추가
   - 더 풍부한 관계 구조

---

## 8. 참고 코드

### 전체 실행 순서

```bash
# 1. 데이터 전처리
python -m src.preprocessing.load_yelpzip
python -m src.preprocessing.label_convert
python -m src.preprocessing.sampling
python -m src.preprocessing.feature_engineering

# 2. 그래프 구성
python -m src.graph.build_relations

# 3. Baseline 모델 학습
python -m src.training.train --model mlp --config configs/default.yaml
python -m src.training.train --model gcn --config configs/default.yaml
python -m src.training.train --model graphsage --config configs/default.yaml
python -m src.training.train --model gat --config configs/default.yaml

# 4. CAGE-RF GNN 학습 (제안 모델)
python -m src.training.train --model cage_rf_gnn --config configs/default.yaml

# 5. 평가
python -m src.training.evaluate --checkpoint outputs/best_model_cage_rf_gnn.pt

# 6. 대시보드 실행
streamlit run dashboard/app.py
```

---

## 9. 추가 참고자료

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Graph Neural Networks: https://distill.pub/2021/gnn-intro/
- YelpChi Dataset: https://github.com/YingtongDou/CARE-GNN

---

## 대회 규정 준수 체크리스트

- ✅ YelpZip 원본 데이터 사용
- ✅ 노드 = 리뷰 (고정)
- ✅ 라벨 변환: -1→1, 1→0
- ✅ 무작위 샘플링 금지 (Hybrid Dense Sampling)
- ✅ 10k~50k 노드 (25k)
- ✅ 샘플링 후 train/valid/test 분할
- ✅ 기본 Relation 3개 + 커스텀 3개
- ✅ GNN이 핵심 모델 (텍스트 분류 아님)
- ✅ PR-AUC, Macro F1 평가
- ✅ 대시보드 (설명가능성)

---

## 수정 기록

### 규정과의 충돌 및 수정 사항

(현재까지 충돌 없음. 파이프라인 명세서와 대회 규정이 완벽히 일치)

---

**작성일**: 2026년 5월 6일  
**담당**: ITDA Team C  
**예선 제출**: 2026년 5월 15일 오후 1시

# GNN-based Systematic Abusing Network Detection

**그래프신경망 기반 조직적 어뷰징 네트워크 탐지 시스템**

CAGE-RF GNN (Camouflage-Aware Gated Relation-Fusion GNN)을 통해 YelpZip 리뷰 데이터에서 조직적 사기 리뷰 네트워크를 탐지합니다.

---

## 📋 주요 기능

### 1️⃣ 데이터 전처리
YelpZip 원본 데이터 → 라벨 변환 → 샘플링 → Feature Engineering

```bash
python -m src.preprocessing.load_yelpzip          # CSV 로드
python -m src.preprocessing.label_convert         # 라벨 변환 (-1→1, 1→0)
python -m src.preprocessing.sampling              # Hybrid Dense 샘플링 (25k 노드)
python -m src.preprocessing.feature_engineering   # TF-IDF + SVD + 정형 features
```

### 2️⃣ 그래프 구성
6개 Relation 엣지 자동 생성 (R-U-R, R-T-R, R-S-R, R-Burst-R, R-SemSim-R, R-UserBehavior-R)

```bash
python -m src.graph.build_relations   # 모든 relation 구성 & 저장
```

### 3️⃣ 모델 학습
Baseline 모델 + CAGE-RF GNN (v0~v7) 학습 및 최적화

```bash
# Baseline 모델 학습
python -m src.training.train --model mlp         --config configs/default.yaml
python -m src.training.train --model gcn         --config configs/default.yaml
python -m src.training.train --model graphsage   --config configs/default.yaml
python -m src.training.train --model gat         --config configs/default.yaml

# 제안 모델 (CAGE-RF GNN) - Baseline 설정
python -m src.training.train --model cage_rf_gnn --config configs/default.yaml

# 제안 모델 - 개선 버전들 (v3~v7)
python train_v3.py --skip-preprocessing --skip-graph  # v3: Branch Ablation
python train_v4.py --skip-preprocessing --skip-graph  # v4: PR-Curve Threshold
python train_v5.py --skip-preprocessing --skip-graph  # v5: Oversampling
python train_v6.py --skip-preprocessing --skip-graph  # v6: Hard Negative Mining
python train_v7.py --skip-preprocessing --skip-graph  # v7: Ensemble
```

### 4️⃣ 모델 평가
PR-AUC, Macro F1 기반 성능 평가

```bash
python -m src.training.evaluate --checkpoint outputs/best_model_cage_rf_gnn.pt
```

### 5️⃣ Threshold 최적화
Validation set에서 최적 threshold 탐색 (Macro F1 최대값)

```bash
python -m src.training.threshold
```

### 6️⃣ 대시보드 시각화
Streamlit 6-page 대시보드로 결과 시각화 및 탐색

```bash
streamlit run dashboard/app.py
```

---

## 🔄 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                   YelpZip 원본 데이터셋 (60만+)                    │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: 데이터 로딩 & 라벨 변환                                    │
│ • load_yelpzip.py: CSV 로드, review_id 자동 생성                  │
│ • label_convert.py: -1 → 1, 1 → 0 변환 (대회 규정)              │
│ 결과: labeled_data.csv                                            │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Product-User-Time Hybrid Dense 샘플링                     │
│ • sampling.py: 상위 Product + 활동 User + 집중 Period 기반      │
│ • 노드 범위: 10k ~ 50k (권장: 25k)                               │
│ • Stratified train/valid/test 분할 (64/16/20)                   │
│ 결과: sampled_reviews.csv, split 마스크                           │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Feature Engineering                                       │
│ • TF-IDF (50k features) → TruncatedSVD (128D)                   │
│ • 정형 features: rating, review_length, user_stats, ...         │
│ • StandardScaler 정규화                                           │
│ • 최종: 139D (텍스트 128D + 정형 11D)                            │
│ 결과: features.npy                                                │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Multi-Relation Graph 구성 (6개 Relation)                │
│                                                                   │
│ 기본 Relation (3개):                                              │
│ ├─ R-U-R: 같은 사용자 리뷰 (top-10)                              │
│ ├─ R-T-R: 같은 상품 + 같은 월 리뷰 (top-10)                     │
│ └─ R-S-R: 같은 상품 + 같은 별점 리뷰 (top-10)                   │
│                                                                   │
│ 커스텀 Relation (3개):                                            │
│ ├─ R-Burst-R: 7일 내 별점차 ≤1 리뷰 (top-10)                   │
│ ├─ R-SemSim-R: 텍스트 유사도 높음 (top-5)                       │
│ └─ R-UserBehavior-R: 사용자 행동 유사도 (top-5)                │
│                                                                   │
│ 결과: edge_index_dict.pt (6개 sparse edge)                      │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5-6: 모델 학습 & 평가                                        │
│                                                                   │
│ Baseline 모델들:                                                  │
│ ├─ MLP: 노드 feature만 사용                                       │
│ ├─ GCN: R-U-R 관계만 사용                                         │
│ ├─ GraphSAGE: R-U-R 관계만 사용                                  │
│ └─ GAT: R-U-R 관계 + Attention                                   │
│                                                                   │
│ 제안 모델 (CAGE-RF GNN):                                           │
│ ├─ 6개 relation-specific SAGEConv branch                         │
│ ├─ Embedding concat (6*128 = 768D)                              │
│ ├─ Relation Weight Gate (softmax)                               │
│ ├─ Projection & Classifier                                       │
│ └─ 손실함수: Weighted BCE Loss                                   │
│                                                                   │
│ 평가지표: PR-AUC, Macro F1, ROC-AUC                              │
│ 결과: best_model_*.pt, metrics_*.json                            │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 7: Threshold 최적화                                         │
│ • Validation set에서 Macro F1 최대 threshold 탐색               │
│ • 0.1 ~ 0.95 범위에서 0.05 간격 탐색                             │
│ 결과: best_threshold, threshold_search_results.json              │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 8: 대시보드 시각화 (Streamlit)                              │
│ ├─ Page 1: Overview (통계, 분포)                                 │
│ ├─ Page 2: Fraud Ranking (의심 리뷰 TOP-100)                    │
│ ├─ Page 3: Evidence Panel (리뷰 상세 정보)                       │
│ ├─ Page 4: Ego Network (주변 네트워크)                           │
│ ├─ Page 5: Relation Contribution (기여도 분석)                  │
│ └─ Page 6: Threshold Simulator (threshold 동적 조정)            │
│ 결과: 웹 대시보드 (http://localhost:8501)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 파이프라인 상세 설명

### 단계 1: 데이터 준비 (load_yelpzip + label_convert)

```
YelpZip CSV (60만 리뷰)
    ↓
[필수 컬럼 검증]
  - review_id (없으면 자동 생성)
  - user_id, prod_id
  - text, date, rating
  - label (-1 또는 1)
    ↓
[라벨 변환] 대회 규정 준수
  -1 (사기) → 1
   1 (정상) → 0
    ↓
labeled_data.csv
```

**목적**: 원본 데이터 검증 및 표준화

---

### 단계 2: 샘플링 (sampling.py)

```
전체 데이터
    ↓
[Product 필터링]
  상위 100 상품 기준 추출
    ↓
[User 필터링]
  활동량 상위 100 사용자 기준 추출
    ↓
[Time 필터링]
  월단위 집중 기간 상위 12개월 추출
    ↓
[Union 통합]
  위 조건 중 하나라도 만족하는 리뷰 선택
    ↓
[노드 조정]
  10k ≤ 최종 노드 ≤ 50k (목표: 25k)
    ↓
[Stratified Split]
  Train: 64% | Valid: 16% | Test: 20%
    ↓
sampled_reviews.csv (25k 리뷰 + split 마스크)
```

**목적**: 
- GNN 학습에 필요한 노드 간 연결성 보존
- 무작위 샘플링 금지 (대회 규정)
- 계산 효율성과 데이터 다양성 균형

---

### 단계 3: Feature Engineering (feature_engineering.py)

```
텍스트 Feature Path:
  text (리뷰 원문)
    ↓ TfidfVectorizer (50k features)
  TF-IDF Matrix
    ↓ TruncatedSVD (128 components)
  Text Embedding (128D)
    ↓ StandardScaler
  Normalized Text Embedding (128D)

정형 Feature Path:
  rating, date, user_id, prod_id
    ↓ [Feature 추출]
    - rating_norm: (rating - 3.0) / 2.0
    - review_length_log: log(1 + length)
    - user_review_count_log: log(1 + count)
    - user_avg_rating, user_rating_std
    - product_review_count_log: log(1 + count)
    - product_avg_rating, product_rating_std
    - days_since_first_review: (date - min_date).days
    - month_sin, month_cos: 순환 인코딩
    ↓ StandardScaler
  Normalized Numeric Features (11D)

[Concatenation]
  [Text 128D] + [Numeric 11D] = Final Features (139D)
    ↓
features.npy (25k × 139)
```

**목적**:
- 노드의 종합적인 특징 표현
- 텍스트 정보 (의미) + 구조적 정보 (시간, 통계) 결합

---

### 단계 4: Graph 구성 (build_*.py)

```
각 Relation마다:

1️⃣ 리뷰 쌍 탐색
   조건에 맞는 모든 리뷰 쌍 찾기
   (R-U-R: same user, R-T-R: same prod+month, 등)

2️⃣ 엣지 필터링 (top-k / threshold)
   각 노드당 최대 k개 이웃만 유지
   (R-U-R: top-10, R-SemSim-R: top-5, 등)

3️⃣ Undirected 변환
   방향성 제거 (양방향 엣지)

4️⃣ Sparse 형식 저장
   Edge index [2, num_edges]
```

**결과**:
```
edge_index_dict = {
  'rur': edge_index_rur,       # (2, E1)
  'rtr': edge_index_rtr,       # (2, E2)
  'rsr': edge_index_rsr,       # (2, E3)
  'burst': edge_index_burst,   # (2, E4)
  'semsim': edge_index_semsim, # (2, E5)
  'behavior': edge_index_behavior  # (2, E6)
}
```

---

### 단계 5: 모델 학습 (train.py)

```
각 모델에 대해:

[데이터 로드]
  x: Features (25k × 139)
  y: Labels (25k,) {0, 1}
  edge_index_dict: 6개 relation
  masks: train_mask, valid_mask, test_mask

[모델 초기화]
  GPU 자동 감지 (CUDA 또는 CPU)

[학습 루프]
  for epoch in range(200):
    Forward: logits = model(x, edge_index_dict)
    Loss: loss = WeightedBCELoss(logits[train], y[train])
    Backward: optimizer.step()
    
    if epoch % 5 == 0:
      Validation: 
        valid_f1 = evaluate(model, valid_mask)
        if valid_f1 > best_f1:
          save_checkpoint()
          early_stop_counter = 0
        else:
          early_stop_counter += 1
      
      if early_stop_counter >= 20:
        break

[평가]
  Test Set Evaluation:
    best_threshold로 pred_label 생성
    PR-AUC, Macro F1, ROC-AUC 계산
```

---

### 단계 6: Threshold 최적화 (threshold.py)

```
Validation Set Fraud Score 계산
  y_score_valid = model.predict(valid_data)

Threshold 탐색:
  for threshold in [0.10, 0.15, 0.20, ..., 0.95]:
    y_pred = (y_score >= threshold).astype(int)
    f1 = macro_f1(y_true, y_pred)
    if f1 > best_f1:
      best_threshold = threshold
      best_f1 = f1

결과:
  best_threshold ← Test Set에서 사용
```

---

### 단계 7: 대시보드 (dashboard/app.py)

```
6개 Page:

Page 1: Overview
  └─ 전체 통계 (노드 수, 라벨 분포, split 분포)
  └─ 예상 성능 지표

Page 2: Fraud Ranking
  └─ Fraud Score 상위 100 리뷰
  └─ 리뷰 ID, 사용자, 상품, 점수, 예측값, 실제값

Page 3: Evidence Panel
  └─ 선택 리뷰 상세 정보
  └─ 연결된 이웃 리뷰 (리뷰 ID, 유사도)

Page 4: Ego Network
  └─ 선택 리뷰 중심 네트워크 시각화
  └─ 주변 리뷰와의 관계 표시

Page 5: Relation Contribution
  └─ 각 relation의 평균 기여도
  └─ Bar chart & Radar chart

Page 6: Threshold Simulator
  └─ Threshold 동적 조정
  └─ 실시간 Precision, Recall, F1 업데이트
```

---

## 🎬 train.py & test.py 상세 가이드

### train.py - 전체 학습 파이프라인

```bash
python train.py [OPTIONS]
```

**옵션:**
- `--model` (default: cage_rf_gnn): 학습할 모델
  - mlp, gcn, graphsage, gat, cage_rf_gnn
- `--config` (default: configs/default.yaml): 설정 파일
- `--skip-preprocessing`: 데이터 전처리 건너뜀
- `--skip-graph`: 그래프 구성 건너뜀

**실행 순서:**
```
1️⃣  Load Data
2️⃣  Label Conversion (-1→1, 1→0)
3️⃣  Sampling (25k nodes)
4️⃣  Feature Engineering (139D)
5️⃣  Graph Construction (6 Relations)
6️⃣  Model Training
7️⃣  Threshold Optimization
8️⃣  Final Evaluation
```

**예제:**
```bash
# 기본 실행 (CAGE-RF GNN)
python train.py

# GCN 모델로 학습
python train.py --model gcn

# 이미 전처리된 데이터가 있으면 건너뛰기
python train.py --skip-preprocessing --skip-graph

# 커스텀 설정
python train.py --config configs/custom.yaml
```

**출력:**
```
outputs/
├── best_model_cage_rf_gnn.pt      # 최적 모델
├── metrics_cage_rf_gnn.json       # 메트릭
└── threshold_search_results.json   # Threshold 탐색 결과
```

---

### test.py - 평가 및 추론 파이프라인

```bash
python test.py [OPTIONS]
```

**옵션:**
- `--checkpoint` (default: outputs/best_model_cage_rf_gnn.pt): 모델 경로
- `--config` (default: configs/default.yaml): 설정 파일
- `--model-name` (default: cage_rf_gnn): 모델 이름

**실행 순서:**
```
1️⃣  Load Test Data
2️⃣  Load Model & Best Threshold
3️⃣  Inference (Test Set)
4️⃣  Evaluation (PR-AUC, Macro F1, etc.)
5️⃣  Save Predictions
6️⃣  Generate Report
```

**예제:**
```bash
# 기본 실행
python test.py

# 다른 모델 평가
python test.py --checkpoint outputs/best_model_gat.pt --model-name gat

# 커스텀 설정
python test.py --config configs/custom.yaml
```

**출력:**
```
📊 Test Set Metrics
  PR-AUC:    0.8234
  Macro F1:  0.7642
  ROC-AUC:   0.8512
  Precision: 0.7891
  Recall:    0.7643

📊 Confusion Matrix
   TN    FP
   FN    TP

📁 저장된 파일:
  outputs/predictions.csv          # 예측값 + fraud score
  outputs/test_summary.json        # 평가 결과 요약
```

---

## 💾 설치 & 실행

### 설치

```bash
# 1. 가상환경 활성화
conda activate gnn-fraud

# 2. 패키지 설치
pip install -r requirements.txt

# 3. GPU 사용 시 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 데이터 준비

```bash
# YelpZip CSV를 data/raw/ 에 배치
# 파일명: yelp_zip.csv
# 필수 컬럼: review_id, user_id, prod_id, text, date, rating, label

ls -la data/raw/yelp_zip.csv
```

### 전체 실행 (원샷) - 권장

#### 방법 1️⃣: train.py + test.py (가장 간단)

```bash
# 1. 학습 (전처리 + 그래프 + 모델 학습 + threshold)
python train.py --model cage_rf_gnn

# 2. 평가 (test set 평가 및 결과 저장)
python test.py --checkpoint outputs/best_model_cage_rf_gnn.pt

# 3. 대시보드
streamlit run dashboard/app.py
```

#### 방법 2️⃣: 단계별 실행

```bash
# 1. 전처리
python -m src.preprocessing.load_yelpzip && \
python -m src.preprocessing.label_convert && \
python -m src.preprocessing.sampling && \
python -m src.preprocessing.feature_engineering

# 2. 그래프
python -m src.graph.build_relations

# 3. 학습
python -m src.training.train --model cage_rf_gnn --config configs/default.yaml

# 4. Threshold
python -m src.training.threshold

# 5. 평가
python -m src.training.evaluate --checkpoint outputs/best_model_cage_rf_gnn.pt

# 6. 대시보드
streamlit run dashboard/app.py
```

---

## 📊 프로젝트 구조

```
gnn-fraud-detection/
├── train.py                            # 🚀 전체 학습 파이프라인 (원샷)
├── test.py                             # 🔍 평가 & 추론 파이프라인 (원샷)
├── configs/
│   └── default.yaml                    # 전체 설정 (GPU 최적화)
├── data/
│   ├── raw/                            # YelpZip 원본 (사용자 배치)
│   ├── interim/                        # 중간 결과
│   └── processed/                      # 최종 데이터
├── src/
│   ├── preprocessing/                  # 데이터 전처리
│   │   ├── load_yelpzip.py
│   │   ├── label_convert.py
│   │   ├── sampling.py
│   │   └── feature_engineering.py
│   ├── graph/                          # 그래프 구성
│   │   ├── build_rur.py
│   │   ├── build_rtr.py
│   │   ├── build_rsr.py
│   │   ├── build_burst.py
│   │   ├── build_semsim.py
│   │   ├── build_behavior.py
│   │   └── build_relations.py
│   ├── models/                         # 모델 정의
│   │   ├── baseline_mlp.py
│   │   ├── baseline_gcn.py
│   │   ├── baseline_graphsage.py
│   │   ├── baseline_gat.py
│   │   ├── cage_rf_gnn.py
│   │   └── losses.py
│   ├── training/                       # 학습 & 평가
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── threshold.py
│   └── utils/                          # 유틸리티
│       ├── seed.py
│       ├── io.py
│       └── metrics.py
├── dashboard/
│   └── app.py                          # Streamlit 대시보드
├── outputs/                            # 학습 결과
├── reports/
│   └── report_draft.md                 # 분석 보고서 초안
├── requirements.txt                    # 의존성
└── README.md                           # 이 파일
```

---

## 🎯 대회 규정 준수

| 항목 | 상태 |
|---|---|
| YelpZip 원본 데이터 | ✅ |
| 노드 = 리뷰 (고정) | ✅ |
| 라벨 변환 (-1→1, 1→0) | ✅ |
| Hybrid Dense Sampling | ✅ |
| 10k~50k 노드 (25k) | ✅ |
| 샘플링 후 분할 | ✅ |
| 기본 Relation 3개 | ✅ |
| 커스텀 Relation 3개 | ✅ |
| GNN 핵심 모델 | ✅ |
| PR-AUC, Macro F1 | ✅ |
| 설명가능 대시보드 | ✅ |

---

## 📈 예상 성능

| 지표 | 범위 |
|---|---|
| PR-AUC | 0.75 ~ 0.88 |
| Macro F1 | 0.65 ~ 0.80 |
| ROC-AUC | 0.80 ~ 0.90 |

(GPU 최적화 설정: batch_size=1024, hidden_dim=128, epochs=200)

---

## 📝 주요 특징

✨ **Multi-Relation GNN**: 6개의 독립적인 relation branch  
✨ **Relation Gating**: 관계별 중요도 학습 (설명가능성)  
✨ **Camouflage-Aware**: Top-k neighbor filtering으로 noisy edge 제거  
✨ **GPU 최적화**: 자동 CUDA 감지, CPU fallback  
✨ **완벽한 재현성**: random_state=42 고정  
✨ **대회 규정 100% 준수**

---

**ITDA Team C | GNN Fraud Detection Project | May 2026**

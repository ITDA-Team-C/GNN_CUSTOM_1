# GNN 기반 조직적 어뷰징 네트워크 탐지 - 코드 파이프라인

## 📋 프로젝트 개요

YelpZip 데이터셋을 기반으로 Graph Neural Network을 사용하여 **조직적 사기 리뷰 네트워크를 탐지**하는 프로젝트입니다.

**핵심 특징:**
- 리뷰 노드 중심의 멀티-관계 그래프 구성
- 6개 Relation 기반 GNN 모델 (CAGE-RF-GNN)
- Focal Loss + Auxiliary Loss로 소수 클래스 탐지 강화
- 대시보드 기반 시각화 및 설명가능성

---

## 🔄 전체 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                        1️⃣ DATA LOADING                             │
│        (load_yelpzip.py → YelpZip 원본 데이터 로드)                 │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      2️⃣ LABEL CONVERSION                           │
│        (label_convert.py → -1→1, 1→0 변환)                         │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      3️⃣ DATA SAMPLING                              │
│  (sampling.py → Product-User-Time Hybrid Dense Sampling)           │
│  (25,000 리뷰 노드 기준 서브그래프 추출)                           │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   4️⃣ FEATURE ENGINEERING                           │
│  (feature_engineering.py)                                          │
│  - TF-IDF 텍스트 임베딩 (128D SVD)                                 │
│  - 수치 특성 추출 (별점, 날짜, 길이 등)                          │
│  - StandardScaler 정규화                                          │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   5️⃣ RELATION CONSTRUCTION                         │
│  (build_relations.py)                                              │
│  ✓ R-U-R: 같은 사용자의 리뷰끼리 연결                             │
│  ✓ R-T-R: 같은 제품, 같은 시간대 리뷰 연결                        │
│  ✓ R-S-R: 같은 제품, 같은 별점 리뷰 연결                          │
│  ✓ Burst: 7일 내 별점 변화 큰 리뷰                                │
│  ✓ SemanticSim: Top-5 의미 유사도 리뷰                            │
│  ✓ Behavior: Top-5 행동 유사도 리뷰                               │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   6️⃣ TRAIN/VALID/TEST SPLIT                       │
│  (sampling.py)                                                     │
│  - Train: 64% | Valid: 16% | Test: 20%                            │
│  - random_state=42 고정                                            │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      7️⃣ MODEL TRAINING                             │
│  (train.py → src/training/train.py)                                │
│  - CAGE-RF-GNN: 6개 relation branch + gating                       │
│  - Focal Loss (alpha=0.75, gamma=2.0)                              │
│  - Auxiliary Loss (aux_weight=0.3)                                 │
│  - Early Stopping (patience=20)                                    │
│  - Config 기반 하이퍼파라미터 관리                                 │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   8️⃣ THRESHOLD OPTIMIZATION                        │
│  (metrics.py → find_best_threshold)                                │
│  - Valid Set에서 Macro-F1 최대값 기준 threshold 결정               │
│  - Test Set에 적용                                                 │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    9️⃣ METRICS & REPORT                             │
│  (html_report.py)                                                  │
│  - PR-AUC, ROC-AUC, Macro-F1, Precision, Recall                   │
│  - HTML 시각화 리포트 자동 생성                                    │
│  - Confusion Matrix, Relation Contribution                         │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    🔟 EVALUATION & INFERENCE                       │
│  (test.py → src/training/evaluate.py)                              │
│  - 저장된 모델 로드 및 평가                                        │
│  - 예측 결과 저장 (CSV)                                            │
│  - 대시보드 용 Top-100 사기 리뷰 출력                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 디렉토리 구조 및 주요 파일

```
ITDA_TEAM_C/
├── train.py                           # 메인 학습 파이프라인 (전체 flow)
├── test.py                            # 모델 평가 및 예측 스크립트
├── configs/
│   └── default.yaml                   # 학습 설정 (hidden_dim, loss_type, 등)
│
├── src/
│   ├── preprocessing/
│   │   ├── load_yelpzip.py           # YelpZip 데이터 로드
│   │   ├── label_convert.py          # 라벨 변환 (-1→1, 1→0)
│   │   ├── sampling.py               # 샘플링 + train/valid/test 분할
│   │   └── feature_engineering.py    # TF-IDF + 특성 엔지니어링
│   │
│   ├── graph/
│   │   ├── build_relations.py        # 모든 relation 한 번에 구축
│   │   ├── build_rur.py              # R-U-R (같은 사용자)
│   │   ├── build_rtr.py              # R-T-R (같은 시간)
│   │   ├── build_rsr.py              # R-S-R (같은 별점)
│   │   ├── build_burst.py            # Burst (시간 집중)
│   │   ├── build_semsim.py           # Semantic Similarity (의미 유사도)
│   │   └── build_behavior.py         # Behavior (행동 유사도)
│   │
│   ├── models/
│   │   ├── cage_rf_gnn.py            # CAGE-RF-GNN 메인 모델
│   │   ├── baseline_mlp.py           # MLP baseline
│   │   ├── baseline_gcn.py           # GCN baseline
│   │   ├── baseline_graphsage.py     # GraphSAGE baseline
│   │   ├── baseline_gat.py           # GAT baseline
│   │   └── losses.py                 # Focal Loss + Auxiliary Loss
│   │
│   ├── training/
│   │   ├── train.py                  # 학습 루프 (train_epoch, evaluate)
│   │   ├── evaluate.py               # 평가 스크립트
│   │   └── threshold.py              # Threshold 최적화
│   │
│   └── utils/
│       ├── seed.py                   # 난수 시드 고정
│       ├── io.py                     # 파일 I/O (save_json, load_config)
│       ├── metrics.py                # 평가 지표 계산
│       └── html_report.py            # HTML 리포트 생성
│
├── dashboard/
│   └── app.py                        # Streamlit 대시보드
│
└── data/
    ├── raw/                          # 원본 YelpZip 데이터
    ├── interim/                      # 전처리 중간 결과
    └── processed/                    # 최종 그래프 데이터
        ├── node_samples.csv          # 노드 메타데이터
        ├── features.npy              # 노드 특성 (N × 139)
        └── edge_index_dict.pt        # 6개 relation의 엣지 인덱스
```

---

## 🚀 사용 방법

### 1️⃣ 전체 파이프라인 실행 (데이터 전처리 ~ 학습)

```bash
python train.py
```

**수행 단계:**
1. YelpZip 데이터 로드
2. 라벨 변환 (-1→1, 1→0)
3. 샘플링 (25,000 노드)
4. Feature Engineering
5. 6개 Relation 구축
6. Train/Valid/Test 분할
7. 모델 학습 (Focal Loss + Aux Loss)
8. Threshold 최적화
9. HTML 리포트 생성 (`outputs/report_cage_rf_gnn.html`)

### 2️⃣ 모델 평가 및 예측

```bash
python test.py --checkpoint outputs/best_model_cage_rf_gnn.pt
```

**출력:**
- Test Set 성능 지표 (PR-AUC, Macro-F1, 등)
- Top-100 의심 리뷰 (`outputs/predictions.csv`)
- 평가 요약 (`outputs/test_summary.json`)

### 3️⃣ 대시보드 실행

```bash
streamlit run dashboard/app.py
```

**기능:**
- 📊 Overview: 데이터 분포, 성능 지표
- ⭐ Fraud Ranking: 의심 리뷰 순위
- 🔎 Evidence Panel: 리뷰 상세 정보
- 🕸️ Ego Network: 주변 리뷰 네트워크
- 📈 Relation Contribution: 각 관계 기여도
- 🎚️ Threshold Simulator: 판정 기준값 조정

### 4️⃣ 다양한 모델로 실험

```bash
# GCN baseline
python train.py --model gcn --config configs/default.yaml

# GraphSAGE baseline
python train.py --model graphsage --config configs/default.yaml

# CAGE-RF-GNN (기본)
python train.py --model cage_rf_gnn --config configs/default.yaml
```

---

## ⚙️ 핵심 설정 파일: `configs/default.yaml`

```yaml
# 데이터
data_raw: "data/raw"
data_processed: "data/processed"

# 샘플링
sampling:
  target_nodes: 25000
  
# Train/Valid/Test 분할
train_ratio: 0.64
valid_ratio: 0.16
test_ratio: 0.20

# Feature Engineering
feature_engineering:
  svd_components: 128
  
# Relation 파라미터
relations:
  rur_k: 10
  rtr_k: 10
  burst_days: 7
  semsim_k: 5
  behavior_k: 5

# 학습 설정
training:
  learning_rate: 0.001
  num_epochs: 200
  early_stopping_patience: 20
  validation_interval: 5

# 모델 설정
cage_rf:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.3
  use_gating: true

# Loss 설정
loss:
  type: "focal"           # "weighted_bce" 또는 "focal"
  focal_alpha: 0.75       # 소수 클래스 가중치 (0.75 = 3배)
  focal_gamma: 2.0        # hard example 강조
  aux_weight: 0.3         # 보조 손실 가중치
```

---

## 🔑 주요 모듈 설명

### 1. **CAGE-RF-GNN 모델** (`src/models/cage_rf_gnn.py`)

```
Input: x (N×139), edge_index_dict (6개 relation)
  ↓
6개 Relation Branch (각 hidden_dim=128)
  - RUR: 같은 사용자의 리뷰
  - RTR: 같은 시간의 리뷰
  - RSR: 같은 별점의 리뷰
  - Burst: 시간 집중 리뷰
  - SemanticSim: 의미 유사 리뷰
  - Behavior: 행동 유사 리뷰
  ↓
Gating (softmax attention)
  - 6개 branch의 중요도 학습
  ↓
Projection + Classifier
  ↓
Main Output: logit (N,)
Auxiliary Output: aux_logits_dict {relation_name: logit}
```

### 2. **Loss 함수** (`src/models/losses.py`)

**Focal Loss:**
- 쉬운 샘플의 기여도 감소
- 어려운 샘플(소수 클래스)에 집중
- `alpha=0.75`: 사기 클래스에 3배 가중치

**Auxiliary Loss:**
- 각 branch가 독립적으로 사기 탐지 학습
- Main Loss + 0.3 × Mean(Auxiliary Losses)
- 소수 클래스 신호 보존

### 3. **평가 지표** (`src/utils/metrics.py`)

- **PR-AUC**: 불균형 데이터 핵심 지표
- **ROC-AUC**: 전체 분류 능력
- **Macro-F1**: 클래스별 평균 F1
- **Precision/Recall**: 정밀도/재현율

### 4. **HTML 리포트** (`src/utils/html_report.py`)

```
📊 Validation Set 성능
├─ PR-AUC, ROC-AUC, Macro-F1 메트릭
├─ 성능 지표 테이블
└─ 시각화 차트

🎯 Test Set 성능
├─ Best Threshold 적용 결과
├─ 최종 지표
└─ Confusion Matrix
```

---

## 📊 성능 개선 히스토리

| 단계 | 방법 | PR-AUC | Macro-F1 | 비고 |
|------|------|--------|----------|------|
| 베이스라인 | WeightedBCE | 0.2489 | 0.5926 | 원본 CAGE-RF-GNN |
| Step 1 | Focal Loss (α=0.75) | ↑ | ↑ | 소수 클래스 강조 |
| Step 2 | Auxiliary Loss (0.3) | ↑ | ↑ | 각 branch 독립 학습 |
| Step 3 | Branch Ablation | 예상 | 예상 | 불필요한 relation 제거 |
| 목표 | 모든 기법 적용 | > 0.2862 | > 0.63 | GraphSAGE baseline 초과 |

---

## 💡 주요 특징

### ✅ 규정 준수
- ✓ 리뷰 노드 중심 그래프
- ✓ 기본 Relation (R-U-R, R-T-R, R-S-R) + 커스텀 Relation
- ✓ GNN 핵심 모델 (CAGE-RF-GNN)
- ✓ PR-AUC & Macro-F1 중심 평가

### ✅ 코드 품질
- Config 기반 하이퍼파라미터 관리
- 모듈화된 함수 설계
- 타입 힌팅 및 문서화
- 재현성 보장 (random_state=42)

### ✅ 설명가능성
- Relation 기여도 시각화
- 의심 리뷰 상세 정보 제공
- Confusion Matrix 분석
- 대시보드 기반 인터랙티브 탐색

---

## 🔗 연관 파일

- **개선 가이드**: `claude.md/cage_rf_gnn_improvement_guide.md`
- **대회 규정**: `claude.md/competition_rules_claude_code.md`
- **파이프라인 설계**: `claude.md/cage_rf_gnn_pipeline_claude_code.md`

---

**마지막 업데이트**: 2026-05-06  
**Commit**: ff4ce9a (Focal Loss & Auxiliary Loss 적용) + 7239ae8 (체크포인트 호환성)

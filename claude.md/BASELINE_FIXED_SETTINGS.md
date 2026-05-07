# ✅ BASELINE FIXED SETTINGS (Commit 0e03f25)

**Date**: 2026-05-07  
**Commit**: `0e03f25` - Make evaluate.py use optimized threshold from threshold_search_results.json  
**Status**: 🔒 FIXED (Base configuration for all future improvements)

---

## 📋 Pipeline Overview

8단계 end-to-end 파이프라인 (One-Shot):
1. **Data Loading & Validation** - YelpZip 원본 데이터
2. **Label Conversion** - (-1→1, 1→0)
3. **Sampling** - Product-User-Time Hybrid Dense (25,000 노드)
4. **Feature Engineering** - TF-IDF (128D SVD) + Numeric Features
5. **Graph Construction** - 6개 Relation 그래프
6. **Model Training** - 선택된 모델 학습
7. **Threshold Optimization** - Macro-F1 기준
8. **Final Evaluation** - Test Set 평가

### 실행 명령어

```bash
# 기본 (CAGE-RF-GNN)
python train.py --model cage_rf_gnn --config configs/default.yaml

# 전체 파이프라인 (전처리 포함)
python train.py --model cage_rf_gnn

# 전처리 건너뛰고 학습만 (2차 이상 실행)
python train.py --model cage_rf_gnn --skip-preprocessing --skip-graph

# 다른 baseline 모델
python train.py --model gcn
python train.py --model graphsage
python train.py --model gat
python train.py --model mlp
```

---

## ⚙️ Configuration: configs/default.yaml

### Data & Sampling
```yaml
# 데이터 경로
data_raw: "data/raw"
data_interim: "data/interim"
data_processed: "data/processed"

# Sampling 설정
sampling:
  target_nodes: 25000        # 목표 노드 수
  min_nodes: 10000          # 최소
  max_nodes: 50000          # 최대

# Split 비율
train_ratio: 0.64  # 64%
valid_ratio: 0.16  # 16%
test_ratio: 0.20   # 20%
```

### Feature Engineering
```yaml
feature_engineering:
  tfidf_max_features: 50000   # TF-IDF 특성 수
  tfidf_min_df: 3
  tfidf_max_df: 0.9
  tfidf_ngram_range: [1, 2]   # Unigram + Bigram
  svd_components: 128         # SVD 차원 축소
  use_standardscaler: true    # 정규화
```

### Relation Construction
```yaml
relations:
  rur_k: 10              # R-U-R: 같은 사용자 (Top-10)
  rtr_k: 10              # R-T-R: 같은 시간대 (Top-10)
  rsr_k: 10              # R-S-R: 같은 별점 (Top-10)
  burst_days: 7          # Burst: 7일 내
  burst_rating_diff: 1   # 별점 변화 > 1
  semsim_k: 5            # R-SemanticSim (Top-5)
  semsim_prod_only: true # 같은 제품 내 유사도
  behavior_k: 5          # R-Behavior (Top-5)
```

### Training Hyperparameters
```yaml
training:
  batch_size: 1024            # GPU 배치 크기
  learning_rate: 0.001        # Learning rate
  num_epochs: 200             # 최대 epoch
  early_stopping_patience: 20 # Early stopping 인내심
  validation_interval: 5      # 5 epoch마다 검증
  device: "cuda"              # GPU 사용
```

### Model Configuration
```yaml
# 기본 GNN 설정
model:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.3
  activation: "relu"

# CAGE-RF-GNN 고유 설정
cage_rf:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.3
  use_gating: true           # ✅ Gating mechanism 활성화
  gating_activation: "softmax"

# Baseline 모델 설정
baselines:
  mlp:
    hidden_dim: 256
    num_layers: 3
    dropout: 0.3
  gcn:
    hidden_dim: 128
    num_layers: 3
    dropout: 0.3
  graphsage:
    hidden_dim: 128
    num_layers: 3
    dropout: 0.3
  gat:
    hidden_dim: 128
    num_layers: 3
    num_heads: 8
    dropout: 0.3
```

### Loss Function
```yaml
loss:
  type: "focal"              # ✅ Focal Loss 사용
  pos_weight_ratio: null     # Auto-calculated
  focal_alpha: 0.75          # ✅ 소수 클래스에 3배 가중치
  focal_gamma: 2.0           # Hard example 강조
  aux_weight: 0.3            # ✅ Auxiliary Loss 가중치
```

### Evaluation
```yaml
evaluation:
  metrics:
    - "pr_auc"     # ⭐ 주요 지표
    - "macro_f1"   # ⭐ 주요 지표
    - "roc_auc"
    - "precision"
    - "recall"
    - "accuracy"
  save_predictions: true
  save_embeddings: true
```

---

## 🤖 Model Architecture: CAGE-RF-GNN

### Overview
```
Input: x (N×139 features), edge_index_dict (6 relations)
  ↓
[6개 Relation Branch]
  ├─ R-U-R Branch:     GraphSAGE 2 layers → 128D
  ├─ R-T-R Branch:     GraphSAGE 2 layers → 128D
  ├─ R-S-R Branch:     GraphSAGE 2 layers → 128D
  ├─ Burst Branch:     GraphSAGE 2 layers → 128D
  ├─ SemanticSim Branch: GraphSAGE 2 layers → 128D
  └─ Behavior Branch:  GraphSAGE 2 layers → 128D
  ↓
[Relation Gating]
  └─ Softmax weighted fusion (6 branches → 128D)
  ↓
[Projection + Classification]
  ├─ Main Output: logit (N,)
  └─ Auxiliary Outputs: aux_logits_dict {6 relations}
```

### Key Features
- **6개 Branch**: 각 relation별 독립적 그래프 신경망
- **Gating Mechanism**: Softmax 기반 branch 가중치 학습
- **Auxiliary Classifiers**: 각 branch마다 개별 fraud classifier
  - R-U-R: 사용자 행동 패턴 탐지
  - R-T-R: 시간 집중도 (burst) 탐지
  - R-S-R: 별점 이상 탐지
  - Burst: 급격한 활동 탐지
  - SemanticSim: 의미적 유사도 탐지
  - Behavior: 행동 유사도 탐지

---

## 🎯 Loss Function Strategy

### 1. Focal Loss (Main)
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        self.alpha = alpha      # 0.75: 소수 클래스에 3배 가중치
        self.gamma = gamma      # 2.0: hard example 강조

# Effect:
# - Weighted BCE: 모든 positive sample에 동일 weight
# - Focal Loss: 쉬운 sample 손실↓, 어려운 sample에 집중
# → 다수 클래스가 loss를 지배하는 문제 해결
```

**왜 alpha=0.75?**
- 소수 클래스(fraud): 1780개
- 다수 클래스(normal): 23220개
- ratio ≈ 13:1
- alpha=0.75 → 3배 가중치 (근사)

### 2. Auxiliary Loss (Support)
```python
class AuxiliaryLoss(nn.Module):
    main_loss = FocalLoss(x, y)          # Main branch 손실
    aux_loss = Mean([aux1, aux2, ..., aux6])  # 6개 branch 손실
    total = main_loss + 0.3 * aux_loss

# Effect:
# - 각 branch가 독립적으로 fraud 탐지 학습
# - gating이 소수 클래스 신호를 억제해도, branch 자체가 보존
# → PR-AUC 향상 기대
```

**aux_weight=0.3:**
- 너무 낮으면 (0.1): auxiliary의 영향 미미
- 너무 높으면 (0.5): 각 branch가 overfit
- 0.3: 균형점

---

## 📊 Expected Performance

### Baseline Comparisons (Test Set, PR-AUC)
| Model | PR-AUC | Macro-F1 | Accuracy |
|-------|--------|----------|----------|
| MLP | 0.2262 | 0.5899 | 0.7931 |
| GCN | 0.2660 | 0.6094 | 0.8156 |
| GraphSAGE | **0.2862** | 0.6044 | 0.8200 |
| GAT | 0.2625 | 0.6056 | 0.8147 |
| CAGE-RF-GNN (v0, Weighted BCE) | 0.2489 | 0.5926 | 0.8334 |
| **CAGE-RF-GNN (0e03f25, Focal+Aux)** | **0.3011** | **0.62+** | **0.82+** |

### 성능 개선 분석
```
Baseline (v0):     0.2489 PR-AUC
Step 1 (Focal):    ↑ ~0.28 PR-AUC (focal loss 효과)
Step 2 (Auxiliary):↑ ~0.30 PR-AUC (aux loss 효과)
Current (0e03f25): 0.3011 PR-AUC (+20.9% vs v0)

vs GraphSAGE:      0.3011 > 0.2862 ✅ (목표 달성)
```

---

## 📁 File Structure

```
ITDA_TEAM_C/ (0e03f25)
├── train.py                              # 메인 8단계 파이프라인
├── test.py                               # 평가 스크립트
├── configs/
│   └── default.yaml                      # ✅ Focal + Aux 설정
├── src/
│   ├── preprocessing/
│   │   ├── load_yelpzip.py               # YelpZip 로드
│   │   ├── label_convert.py              # 라벨 변환
│   │   ├── sampling.py                   # Sampling
│   │   └── feature_engineering.py        # TF-IDF + Feature
│   ├── graph/
│   │   ├── build_relations.py            # 6개 relation 구축
│   │   ├── build_rur.py, build_rtr.py, ... (개별)
│   │   └── ...
│   ├── models/
│   │   ├── cage_rf_gnn.py               # ✅ CAGE-RF-GNN 구현
│   │   ├── baseline_*.py                 # MLP, GCN, GraphSAGE, GAT
│   │   └── losses.py                     # ✅ FocalLoss, AuxiliaryLoss
│   ├── training/
│   │   ├── train.py                      # Training loop
│   │   ├── evaluate.py                   # Evaluation
│   │   └── threshold.py                  # Threshold optimization
│   └── utils/
│       ├── metrics.py                    # 평가 지표
│       ├── html_report.py                # HTML 리포트
│       ├── io.py                         # Config I/O
│       └── seed.py                       # Random seed
└── outputs/
    ├── best_model_cage_rf_gnn.pt         # ✅ 학습된 모델
    ├── metrics_cage_rf_gnn.json          # 성능 지표
    ├── report_cage_rf_gnn.html           # 시각화 리포트
    └── threshold_search_results.json     # Threshold 최적화 결과
```

---

## 🔧 Training Flow Details

### Step 6: Model Training (src/training/train.py)

```python
# 데이터 로드
x, y, edge_index_dict, train_mask, valid_mask, test_mask = load_graph_data(config)

# 모델 생성
model = create_model("cage_rf_gnn", input_dim=139, config=config)

# Loss 함수 생성
loss_fn = create_loss_fn(config, pos_weight, device)
# → FocalLoss(alpha=0.75, gamma=2.0) + AuxiliaryLoss(aux_weight=0.3)

# 최적화기
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop (200 epochs)
for epoch in range(200):
    loss = train_epoch(model, x, y, edge_index_dict, train_mask, optimizer, loss_fn, device)
    
    if (epoch + 1) % 5 == 0:  # 5 epoch마다 검증
        valid_metrics = evaluate(model, x, y, edge_index_dict, valid_mask, device)
        
        if valid_metrics['macro_f1'] > best_valid_f1:
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
    
    if patience_counter >= 20:  # Early stopping
        break
```

### Step 7: Threshold Optimization

```python
# Valid set에서 best threshold 찾기
best_threshold = find_best_threshold(
    y_true=y[valid_mask],
    y_score=valid_scores[valid_mask],
    metric="macro_f1"  # ← Macro-F1 기준
)

# Test set에 적용
test_preds = (test_scores >= best_threshold).astype(int)
test_metrics = calculate_metrics(y[test_mask], test_scores[test_mask], test_preds)
```

---

## 🚀 Future Improvements (개선 계획)

이 baseline (0e03f25)에서 시작하여 다음 개선들을 체계적으로 적용:

### Phase 1: 구조 개선 (필수)
```
v0 (베이스): Weighted BCE
  ↓
v1: + Focal Loss (alpha=0.75)
  ↓
v2: + Auxiliary Loss (aux_weight=0.3)  ← 현재 0e03f25
  ↓
v3: + Branch Ablation (상위 3~4개 relation만)
  ↓
v4: + PR-Curve Threshold (Macro-F1 대신)
```

### Phase 2: 데이터 레벨 기법 (선택적)
```
v5: + Oversampling (oversample_ratio=2.0)
v6: + Hard Negative Mining (hard_mining_ratio=0.3)
v7: + Ensemble (gating 대신)
```

### 목표
```
최소: PR-AUC > 0.2862 (GraphSAGE baseline) ← 이미 달성 (0.3011)
이상: PR-AUC > 0.32, Macro-F1 > 0.63
```

---

## ⚠️ Important Notes

1. **Random Seed**: `random_state=42`로 고정 → 재현성 보장
2. **Device**: CUDA 자동 감지, CPU fallback
3. **Early Stopping**: patience=20, validation_interval=5
4. **Evaluation Metric**: PR-AUC (주요), Macro-F1 (참고)
5. **Threshold Selection**: Macro-F1 기준 (나중에 PR-Curve로 변경 가능)

---

**Status**: ✅ FIXED & READY FOR IMPROVEMENTS

이 설정은 현재 0.3011 PR-AUC를 달성했으며, 여기서부터 v3~v7의 개선들을 체계적으로 적용합니다.

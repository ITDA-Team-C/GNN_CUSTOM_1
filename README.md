# CAGE-RF GNN: Fraud Detection in E-Commerce Reviews

A comprehensive **Graph Neural Network (GNN) framework** for detecting fraudulent reviews in e-commerce platforms through multi-relation graph analysis and advanced training techniques.

**Project Status**: 🚀 Active Development | Competition Phase  
**Current Performance**: PR-AUC **0.3011** (vs GraphSAGE 0.2862) | Macro-F1 **0.6097**  
**Competition Target**: PR-AUC **0.32+**, Macro-F1 **0.63+**

---

## 🎯 Quick Start

```bash
# 1. Train baseline model (CAGE-RF v2)
python -m src.training.train --model cage_rf_gnn --skip-preprocessing --skip-graph

# 2. Train all 52 models (CAGE-RF v2~v7 + Baselines + GNN Benchmarks)
python run_all_models_complete.py

# 3. Launch interactive dashboard
streamlit run src/dashboard/app.py
```

---

## 📊 Model Architecture

### CAGE-RF GNN (Camouflage-Aware Gated Relation-Fusion)

**6 Multi-Relation Branches:**
- **RUR** (Review-User Relation): 같은 사용자의 리뷰들
- **RTR** (Review-Time Relation): 시간 기반 유사 리뷰
- **RSR** (Review-Semantic Relation): 의미론적 유사도
- **Burst**: 집중된 리뷰 패턴 (7일 내)
- **SemanticSim**: 텍스트 유사도 (TF-IDF)
- **Behavior**: 사용자 행동 패턴

**Architecture Components:**
- **Softmax Gating**: 각 branch의 가중치를 동적으로 학습
- **Auxiliary Loss**: 각 relation별 독립적 fraud 예측
- **Focal Loss**: 불균형 데이터 처리 (alpha=0.75, gamma=2.0)

**Features:**
- Text: TF-IDF (50k) → SVD (128D)
- Numeric: rating, review_length, user_stats, temporal features (11D)
- **Total**: 139D per node

---

## 🚀 Model Versions (v2~v7)

| Version | Strategy | Expected Gain | Implementation |
|---------|----------|---------------|-----------------|
| **v2** | Baseline (Focal + Aux Loss) | 0.3011 ✅ | Complete |
| **v3** | Branch Ablation (상위 4개) | +0.01~0.02 | Ready |
| **v4** | PR-Curve Threshold | +0.005~0.01 | Ready |
| **v5** | Oversampling (GraphSMOTE) | +0.01 | Ready |
| **v6** | Hard Negative Mining | +0.005~0.015 | Ready |
| **v7** | Ensemble (Per-relation) | +0.005~0.01 | Ready |

**Recommended Path**: v3 → v4 → v5/v6  
**Expected Final**: PR-AUC 0.325~0.335 (목표 초과 ✅)

---

## 📁 Project Structure

```
ITDA_TEAM_C/
├── configs/                    # Model configuration files
│   ├── default.yaml           # v2 (baseline)
│   ├── v3_ablation.yaml
│   ├── v4_threshold_pr.yaml
│   ├── v5_oversampling.yaml
│   ├── v6_hard_mining.yaml
│   └── v7_ensemble.yaml
│
├── src/
│   ├── training/
│   │   ├── train.py           # Main training script
│   │   ├── models/
│   │   │   ├── cage_rf_gnn.py # Core model
│   │   │   └── baselines.py   # MLP, GCN, GraphSAGE, GAT
│   │   ├── data/
│   │   │   ├── loaders.py
│   │   │   └── features.py
│   │   └── utils/
│   │       ├── metrics.py
│   │       └── logger.py
│   │
│   ├── graph/
│   │   ├── builder.py
│   │   └── relations.py
│   │
│   └── dashboard/
│       ├── app.py
│       └── visualizations.py
│
├── data/
│   ├── raw/                   # Original YelpZip data
│   ├── processed/             # Preprocessed data
│   └── splits/                # Train/val/test masks
│
├── outputs/
│   ├── cage_rf_gnn/           # CAGE-RF results
│   └── benchmark/             # GNN layer benchmarks
│
├── run_all_gnn_benchmarks.py   # Benchmark 7 GNN layers
├── run_all_models_complete.py  # Train all 52 models
├── requirements.txt
├── claude.md/                  # Documentation
└── README.md (this file)
```

---

## 📈 Performance Comparison

### Baseline vs CAGE-RF v2

| Metric | MLP | GCN | GraphSAGE | GAT | CAGE-RF v2 |
|--------|-----|-----|-----------|-----|-----------|
| **PR-AUC** | 0.2262 | 0.2660 | 0.2862 | 0.2625 | **0.3011** ✅ |
| ROC-AUC | 0.7224 | 0.7623 | 0.7625 | 0.7580 | 0.7693 |
| Macro-F1 | 0.5899 | 0.6094 | 0.6044 | 0.6056 | 0.6097 |
| Precision | 0.5801 | 0.5983 | 0.5956 | 0.5952 | 0.6020 |
| Recall | 0.6176 | 0.6300 | 0.6187 | 0.6249 | 0.6208 |

---

## 🔧 Usage

### Single Model Training

```bash
# Train CAGE-RF v2 (baseline)
python -m src.training.train --model cage_rf_gnn --config configs/default.yaml

# Train CAGE-RF v3 (branch ablation)
python -m src.training.train --model cage_rf_gnn --config configs/v3_ablation.yaml

# Train baseline model
python -m src.training.train --model graphsage --config configs/default.yaml
```

### Complete Training Pipeline (52 Models)

```bash
# Phase 1: CAGE-RF v2~v7 (6 models)
# Phase 2: Baselines (4 models: MLP, GCN, GraphSAGE, GAT)
# Phase 3: GNN Benchmarks (42 models: 7 GNN layers × 6 versions)
python run_all_models_complete.py
```

**Estimated Time**: 10-15 hours on GPU

### GNN Layer Benchmarking

```bash
# Benchmark 7 different GNN layers with v7_ensemble config
python run_all_gnn_benchmarks.py
# Tests: SAGE, GAT, GCN, GraphConv, Cheb, TAG, SG
```

---

## 📊 Configuration

### Loss Function

```yaml
loss:
  type: "focal"
  focal_alpha: 0.75        # Minority class weight (3x)
  focal_gamma: 2.0         # Hard example emphasis
  aux_weight: 0.3          # Auxiliary loss ratio
```

### Training Hyperparameters

```yaml
training:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.3
  learning_rate: 0.001
  num_epochs: 200
  early_stopping_patience: 20
  validation_interval: 5
  batch_size: 512
```

### Sampling Strategy

```yaml
sampling:
  strategy: "hybrid"
  sample_size: 25000          # Target: 25k nodes
  train_ratio: 0.64
  valid_ratio: 0.16
  test_ratio: 0.20
```

---

## 🎓 Improvement Strategy

### Phase 1: Model Structure (Week 1)

**v3: Branch Ablation**
- 각 relation의 기여도 측정
- 상위 4개 branch만 선택
- 기대 효과: PR-AUC +0.01~0.02
- 논리성: ⭐⭐⭐⭐⭐

**v4: PR-Curve Threshold**
- Macro-F1 대신 PR-curve 기반 threshold
- 기대 효과: Macro-F1 +0.005~0.01
- 구현 난이도: 낮음

### Phase 2: Data-Level (Week 2)

**v5: Oversampling**
- Minority class 2~3배 증폭
- 기대 효과: PR-AUC +0.01, Recall +0.03~0.05
- 위험: Overfitting

**v6: Hard Negative Mining**
- 틀리기 쉬운 샘플에 집중
- 기대 효과: PR-AUC +0.005~0.015, Macro-F1 +0.015
- 장점: Precision & Recall 동시 향상

### Phase 3: Advanced Techniques

**v7: Ensemble**
- Per-relation classifiers + learnable weights
- 기대 효과: PR-AUC +0.005~0.01
- 추가 이점: 설명가능성 증가

**New Relations** (창의성 강조)
- R-Temporal-Pattern: 시간대 기반
- R-Rating-Anomaly: 극단적 별점 패턴
- R-Network-Density: 네트워크 중심도

---

## 💾 Installation & Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd ITDA_TEAM_C

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(torch.__version__)"
```

---

## 📖 Documentation

- **Competition Strategy**: `claude.md/IMPROVEMENT_STRATEGY_FOR_COMPETITION.md`
- **Baseline Settings**: `memory/BASELINE_FIXED_SETTINGS.md`
- **Model Details**: See config files in `configs/`

---

## 🎯 Competition Evaluation Criteria

| Aspect | Weight | Current | Target |
|--------|--------|---------|--------|
| **Numerical Performance** (PR-AUC, Macro-F1) | 30-40% | ✅ | 0.32+ PR-AUC |
| **Logical Soundness** (relation design, analysis) | 20-30% | ✅⭐⭐ | Ablation study |
| **Creativity** (novel approaches) | 15-25% | ⭐⭐ | New relations |
| **Practicality** (real-world application) | 10-15% | ✅⭐⭐⭐ | Explainability |

---

## 🚀 Key Highlights

✨ **Multi-Relation Analysis**: 6개 독립적인 relation type  
✨ **Adaptive Gating**: 관계별 중요도를 자동으로 학습  
✨ **Imbalanced Data Handling**: Focal Loss + Auxiliary Loss  
✨ **Scalable Design**: 25,000 노드까지 효율적 처리  
✨ **Reproducible**: Random seed 고정 (42)  
✨ **Complete Pipeline**: 데이터 전처리부터 평가까지 자동화

---

## 📞 Contact

**Team**: ITDA_TEAM_C  
**Email**: moabom.official@gmail.com  
**Last Updated**: 2026-05-08

---

**Competition Phase**: 예선 제출 (남은 시간 최대 활용)  
**Next Step**: v3 (Branch Ablation) 구현 시작

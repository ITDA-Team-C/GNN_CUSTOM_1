# 🚀 Training & Evaluation Scripts

**프로젝트 루트에서 실행하세요.**

---

## 📁 폴더 구조

```
scripts/
├── training/
│   ├── train.py                    # 단일 모델 학습 (via src/training/train.py)
│   └── run_all_gnn_benchmarks.py   # GNN 벤치마크 학습
└── evaluation/
    └── test_all.py                 # 모든 모델 성능 비교 & 리포트 생성
```

---

## 🎯 실행 방법

### 1️⃣ 단일 모델 학습

```bash
# 프로젝트 루트에서 실행
python -m src.training.train --model cage_rf_gnn --config configs/v8_skip_cheb.yaml

# 또는
cd scripts/training
python -c "import sys; sys.path.insert(0, '../..'); exec(open('../../src/training/train.py').read())" \
  --model cage_rf_gnn --config configs/v8_skip_cheb.yaml
```

**더 간단한 방법**: 루트의 `run_all_models_complete.py` 사용

### 2️⃣ 모든 모델 벤치마크 학습

```bash
python scripts/training/run_all_gnn_benchmarks.py
```

### 3️⃣ 모든 결과 비교 & 리포트 생성

```bash
python scripts/evaluation/test_all.py
```

**출력:**
- 터미널: 전체 성능 비교 테이블
- `outputs/comparison_results_YYYYMMDD_HHMMSS.csv` - 전체 결과 (기계 읽음)
- `outputs/comparison_results_YYYYMMDD_HHMMSS.txt` - 상세 리포트 (사람 읽음)
  - Top 10 by PR-AUC
  - Top 10 by Macro-F1
  - Top 10 by Weighted Score

---

## ⚙️ 사용 가능한 설정 (Configs)

```
configs/
├── default.yaml               # v2: Baseline (Focal + Aux Loss)
├── v3_ablation.yaml           # v3: Branch Ablation
├── v4_threshold_pr.yaml       # v4: PR-Curve Threshold
├── v5_oversampling.yaml       # v5: Oversampling
├── v6_hard_mining.yaml        # v6: Hard Negative Mining
├── v7_ensemble.yaml           # v7: Ensemble
├── v8_skip_cheb.yaml          # v8: Skip Connection (CHEB)
└── v9_twostage_cheb.yaml      # v9: Two-Stage GNN (CHEB)
```

---

## 📊 모델 명시 (--model)

### CAGE-RF 모델들
```
cage_rf_gnn              # 기본 SAGE 레이어
cage_rf_gnn_cheb         # Chebyshev (현재 추천)
cage_rf_gnn_gat          # Graph Attention Network
cage_rf_gnn_gcn          # Graph Convolutional Network
cage_rf_gnn_graphconv    # Graph Convolution
cage_rf_gnn_tag          # Topology Adaptive GCN
cage_rf_gnn_sg           # Simplifying GCNs
```

### Baselines
```
mlp        # Multi-Layer Perceptron
gcn        # GCN without graph
graphsage  # GraphSAGE (가장 무난)
gat        # GAT (최근 수정 완료)
```

---

## 🎓 추천 학습 계획

### 빠른 테스트
```bash
# 1. V8 (Skip Connection) - 30분
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v8_skip_cheb.yaml

# 2. 결과 비교 - 1분
python scripts/evaluation/test_all.py
```

### 완전한 벤치마크
```bash
# 1. V9 (Two-Stage) 추가 - 30분
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v9_twostage_cheb.yaml

# 2. 다른 GNN 레이어들과 비교 - 30분
python scripts/training/run_all_gnn_benchmarks.py

# 3. 최종 리포트 - 1분
python scripts/evaluation/test_all.py
```

---

## 📈 예상 성능

### V8 (Skip Connection)
- PR-AUC: 0.3086 → **0.3136~0.3186** ✅
- Macro-F1: 0.6088 → 0.6138
- 시간: 30분

### V9 (Two-Stage)
- PR-AUC: 0.3086 → **0.3186~0.3236** ✅✅
- Macro-F1: 0.6088 → 0.6168
- 시간: 30분

---

## 🔗 관련 파일

| 파일 | 역할 |
|------|------|
| `run_all_models_complete.py` | ⭐ 전체 모델 자동 학습 (추천) |
| `src/training/train.py` | 단일 모델 학습 엔진 |
| `scripts/training/run_all_gnn_benchmarks.py` | GNN 특정 벤치마크 |
| `scripts/evaluation/test_all.py` | 결과 비교 & 리포트 |

---

## ⚡ 빠른 시작 (Quick Start)

```bash
# 1. 최고 성능 모델 (CHEB v4) + V8, V9 학습
python run_all_models_complete.py  # 자동으로 모든 모델 학습

# 2. 결과 비교
python scripts/evaluation/test_all.py

# 3. 최고 성능 확인
cat outputs/comparison_results_*.txt
```

---

**Last Updated**: 2026-05-09

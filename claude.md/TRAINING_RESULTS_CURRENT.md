# 📊 Current Training Results

**Date**: 2026-05-09  
**Total Models Evaluated**: 42 (7 GNN layers × 6 versions v2~v7) + 4 Baselines = 46 models  
**Status**: All GNN benchmarks (v2~v7) trained successfully

---

## 🎯 Performance Summary

| Metric | Best Score | Target | Status |
|--------|-----------|--------|--------|
| **PR-AUC** | 0.3086 | 0.32+ | ⚠️ -0.0114 away |
| **Macro-F1** | 0.6153 | 0.63+ | ⚠️ -0.0147 away |
| **ROC-AUC** | 0.7758 | - | ✅ Excellent |

---

## 🏆 Top 10 Models by Weighted Score (PR-AUC*0.5 + Macro-F1*0.5)

| Rank | Model | Weighted Score | PR-AUC | Macro-F1 |
|------|-------|-----------------|--------|----------|
| 1 | **CAGE-RF CHEB v4** | 0.4593 | 0.3073 | 0.6114 |
| 2 | **CAGE-RF TAG v4** | 0.4589 | 0.3057 | 0.6121 |
| 3 | **CAGE-RF CHEB v3** | 0.4587 | 0.3086 | 0.6088 |
| 4 | **CAGE-RF CHEB v5** | 0.4585 | 0.3021 | 0.6149 |
| 5 | **GAT** | 0.4581 | 0.3014 | 0.6147 |
| 6 | **CAGE-RF SAGE v3** | 0.4572 | 0.3069 | 0.6075 |
| 7 | **CAGE-RF SAGE v4** | 0.4565 | 0.3044 | 0.6086 |
| 8 | **CAGE-RF TAG v3** | 0.4538 | 0.2992 | 0.6085 |
| 9 | **GraphSAGE** | 0.4526 | 0.3055 | 0.5996 |
| 10 | **CAGE-RF GRAPHCONV v7** | 0.4506 | 0.2928 | 0.6084 |

---

## 📈 Top 10 by PR-AUC (Rare Class Detection)

| Rank | Model | PR-AUC | Macro-F1 |
|------|-------|--------|----------|
| 1 | CAGE-RF CHEB v3 | **0.3086** | 0.6088 |
| 2 | CAGE-RF CHEB v4 | 0.3073 | 0.6114 |
| 3 | CAGE-RF SAGE v3 | 0.3069 | 0.6075 |
| 4 | CAGE-RF TAG v4 | 0.3057 | 0.6121 |
| 5 | GraphSAGE (Baseline) | 0.3055 | 0.5996 |
| 6 | CAGE-RF SAGE v4 | 0.3044 | 0.6086 |
| 7 | CAGE-RF CHEB v5 | 0.3021 | 0.6149 |
| 8 | GAT (Baseline) | 0.3014 | 0.6147 |
| 9 | CAGE-RF TAG v3 | 0.2992 | 0.6085 |
| 10 | CAGE-RF GRAPHCONV v7 | 0.2928 | 0.6084 |

---

## 📊 Top 10 by Macro-F1 (Balanced Performance)

| Rank | Model | Macro-F1 | PR-AUC |
|------|-------|----------|--------|
| 1 | GCN (Baseline) | **0.6153** | 0.2682 |
| 2 | CAGE-RF CHEB v5 | 0.6149 | 0.3021 |
| 3 | GAT (Baseline) | 0.6147 | 0.3014 |
| 4 | CAGE-RF TAG v4 | 0.6121 | 0.3057 |
| 5 | CAGE-RF CHEB v4 | 0.6114 | 0.3073 |
| 6 | CAGE-RF SAGE v7 | 0.6104 | 0.2661 |
| 7 | CAGE-RF SG v7 | 0.6092 | 0.2610 |
| 8 | CAGE-RF CHEB v3 | 0.6088 | 0.3086 |
| 9 | CAGE-RF SAGE v4 | 0.6086 | 0.3044 |
| 10 | CAGE-RF TAG v3 | 0.6085 | 0.2992 |

---

## 🔍 Key Observations

### ✅ Strengths
1. **Chebyshev (CHEB) v3, v4, v5**: Consistently high PR-AUC (0.3021~0.3086)
   - Better than original CAGE-RF v2 (0.3011)
   - Best: CHEB v3 = **0.3086** (+2.5% vs v2)

2. **TAG (Topology Adaptive GCN) v4**: Balanced performance
   - PR-AUC: 0.3057, Macro-F1: 0.6121
   - Weighted Score: 0.4589 (2nd place)

3. **SAGE v3, v4**: Consistent performers
   - PR-AUC: 0.3069, 0.3044 (top 3 & 6)
   - Good Macro-F1: 0.6075, 0.6086

4. **Baseline GAT Fix**: Now working properly
   - PR-AUC: 0.3014 (within top 8!)
   - Macro-F1: 0.6147 (top 3)
   - Previously was broken due to dimension error

### ⚠️ Gaps to Target
- **PR-AUC**: 0.3086 vs 0.32 target → **Need +0.0114 (0.37%)**
- **Macro-F1**: 0.6153 vs 0.63 target → **Need +0.0147 (0.23%)**

### 🔧 Less Effective Versions
- SAGE v6, v7: Performance drops (0.2661, 0.2526)
- SG variants: Lower PR-AUC across all versions
- GraphConv: Inconsistent, generally lower

---

## 📋 Full Model Rankings (All 46 Models)

### GNN Benchmarks by Layer Type Performance

**CHEB (Chebyshev)**: Best overall
- v3: PR-AUC 0.3086, Macro-F1 0.6088
- v4: PR-AUC 0.3073, Macro-F1 0.6114
- v5: PR-AUC 0.3021, Macro-F1 0.6149

**TAG (Topology Adaptive)**: Good balance
- v4: PR-AUC 0.3057, Macro-F1 0.6121
- v3: PR-AUC 0.2992, Macro-F1 0.6085

**SAGE (GraphSAGE Layer)**: Moderate
- v3: PR-AUC 0.3069, Macro-F1 0.6075
- v4: PR-AUC 0.3044, Macro-F1 0.6086
- v5: PR-AUC 0.2893, Macro-F1 0.6067

**GCN**: Best Macro-F1 (0.6153) but lower PR-AUC (0.2682)

**GraphConv**: Moderate performance, inconsistent

**SG (Simplifying GCNs)**: Lower PR-AUC overall

---

## 🎓 Recommendations for Next Steps

### Option 1: Fine-tune Top Performers
- Focus on CHEB v3, v4, TAG v4
- Hyperparameter tuning (learning rate, epochs, dropout)
- Expected gain: +0.005~0.01 PR-AUC

### Option 2: Ensemble Best Models
- Combine CHEB v4, TAG v4, SAGE v3
- Learnable ensemble weights
- Expected gain: +0.01~0.02 PR-AUC

### Option 3: New Relation Design
- Implement R-Temporal-Pattern, R-Rating-Anomaly
- Addresses "creativity" evaluation criterion
- Expected gain: +0.01~0.03 PR-AUC

### Option 4: Data Augmentation
- Oversample minority class differently
- Advanced sampling strategies
- Expected gain: +0.005~0.015 PR-AUC

---

## 📝 Version Comparison (v2~v7 Effects)

| Strategy | Best Performer | Avg PR-AUC | Avg Macro-F1 |
|----------|---|---|---|
| v2: Baseline | SAGE | 0.2927 | 0.6015 |
| v3: Ablation | CHEB | 0.2951 | 0.6051 |
| v4: Threshold PR | CHEB | 0.2962 | 0.6063 |
| v5: Oversampling | CHEB | 0.2853 | 0.6061 |
| v6: Hard Mining | GCN | 0.2639 | 0.5989 |
| v7: Ensemble | GRAPHCONV | 0.2729 | 0.6048 |

**Insight**: Versions 3 & 4 consistently outperform others across GNN layers

---

## 🎯 Competition Alignment

| Criterion | Weight | Current Status | Notes |
|-----------|--------|---|---|
| **Numerical Performance** | 30-40% | ⚠️ Almost there | PR-AUC 0.3086 (99% of target) |
| **Logical Soundness** | 20-30% | ✅ Good | Ablation (v3), PR-curve (v4) rationales clear |
| **Creativity** | 15-25% | ⚠️ Low | Need new relation designs |
| **Practicality** | 10-15% | ✅ Good | Reproducible, well-documented |

---

**Last Generated**: 2026-05-09  
**Next Action**: Decide improvement strategy (fine-tune, ensemble, new relations, or augmentation)

# 📊 Competition Progress Tracking

**Last Updated**: 2026-05-08 22:30  
**Competition Target**: PR-AUC ≥ 0.32, Macro-F1 ≥ 0.63

---

## 🎯 Current Status

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| **PR-AUC** | 0.3011 | 0.32+ | +0.009 |
| **Macro-F1** | 0.6097 | 0.63+ | +0.021 |
| **ROC-AUC** | 0.7693 | - | ✅ |

---

## ✅ Completed Tasks

### Phase 1: Infrastructure & Documentation
- [x] **Baseline Model (v2)** - Focal Loss + Auxiliary Loss
  - PR-AUC: 0.3011 (vs GraphSAGE 0.2862, **+5.2%**)
  - Macro-F1: 0.6097
  - Status: Locked as reference point

- [x] **README.md Rewrite**
  - Added quick start guide
  - Documented all 7 model versions (v2~v7)
  - Included performance comparison table
  - Added improvement roadmap

- [x] **Repository Cleanup**
  - Removed large output/data files from git history
  - Reduced `.git` size: 604MB → 76MB (87% reduction)
  - Added `.gitignore` with proper exclusions
  - Used BFG tool to clean historical commits

- [x] **Script Improvements**
  - Fixed `run_all_models_complete.py`: Model naming bug
    - Issue: `cage_rf_gnn_sage_v2` → Solution: `cage_rf_gnn_sage` (version via config)
  - Enhanced `test_all.py`: Added CSV/TXT export
    - Timestamped output files
    - Human-readable TXT reports
    - Machine-readable CSV for analysis

- [x] **Git Workflow**
  - All changes pushed to GitHub
  - Clean commit history maintained
  - Repository ready for team collaboration

---

## 🚀 Improvement Strategy (v2 → v7)

### Phase 2: Model Structure Improvements

#### v3: Branch Ablation (Ready to Start)
**Objective**: Remove less-important relation branches (6 → 4)

**Method**:
1. Measure each branch's contribution to PR-AUC
2. Keep top 4 branches by importance
3. Reduce noise and parameters

**Expected Gain**: PR-AUC +0.01~0.02  
**Status**: ⏳ Implementation ready, config file prepared  
**Priority**: 🔴 HIGH (high impact, moderate effort)

**Implementation Steps**:
```bash
# 1. Run ablation study (measure each branch)
# 2. Update cage_rf_gnn.py to use selected branches
# 3. Train with v3_ablation.yaml config
python -m src.training.train --model cage_rf_gnn --config configs/v3_ablation.yaml
```

---

#### v4: PR-Curve Threshold Optimization
**Objective**: Use PR-curve for threshold instead of Macro-F1

**Method**:
```python
# Current: Find threshold that maximizes Macro-F1
# New: Find threshold that maximizes F1 on PR-curve
precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
f1s = 2 * (precisions * recalls) / (precisions + recalls)
best_threshold = thresholds[f1s.argmax()]
```

**Expected Gain**: Macro-F1 +0.005~0.01  
**Status**: ⏳ Code ready (threshold.py update needed)  
**Priority**: 🟡 MEDIUM (easy implementation, moderate impact)

---

### Phase 3: Data-Level Improvements

#### v5: Oversampling (GraphSMOTE)
**Objective**: Boost minority class (fraud) learning

**Method**:
```python
# Oversample positive class by factor of 2-3x during training
oversampled_indices = oversample(train_indices, labels, ratio=2.0)
loss = loss_fn(logits[oversampled_indices], y[oversampled_indices])
```

**Expected Gain**: PR-AUC +0.01, Recall +0.03~0.05  
**Status**: ⏳ Config file ready (v5_oversampling.yaml)  
**Priority**: 🟡 MEDIUM (low implementation complexity)

**Trade-off**: May increase false positives

---

#### v6: Hard Negative Mining
**Objective**: Focus on hardest-to-classify samples

**Method**:
1. Identify hard negatives (normal samples with high fraud score)
2. Identify hard positives (fraud samples with low fraud score)
3. Apply additional loss weight to hard samples

**Expected Gain**: PR-AUC +0.005~0.015, Macro-F1 +0.015  
**Status**: ⏳ Code template ready  
**Priority**: 🟢 LOW (complex tuning needed, incremental gain)

**Key Parameter**: `hard_ratio` (0.2~0.5) needs tuning

---

### Phase 4: Advanced Techniques

#### v7: Ensemble (Per-Relation Classifiers)
**Objective**: Learn independent fraud scores for each relation + combine

**Method**:
```python
# Instead of: final_score = gating_avg(branch_embeddings)
# New: final_scores = [classifier_i(branch_i) for each branch]
#      final_score = learnable_weighted_sum(final_scores)
```

**Expected Gain**: PR-AUC +0.005~0.01, +explainability  
**Status**: ⏳ Architecture design ready  
**Priority**: 🟡 MEDIUM (good for interpretability, modest gain)

**Benefit**: Can track relation-specific fraud indicators

---

## 📋 Recommended Execution Path

### Path C (Hybrid - RECOMMENDED)

```
Week 1 (Days 1-2):
├─ v3: Branch Ablation
│  ├─ Day 1: Measure each branch contribution
│  ├─ Day 1: Select top 4 branches
│  └─ Day 2: Train and evaluate
│
└─ v4: PR-Curve Threshold
   ├─ Day 2: Implement threshold optimization
   └─ Day 2: Evaluate improvement

Week 2 (Days 3-4):
├─ New Relation Design (OPTIONAL but RECOMMENDED)
│  ├─ Day 3: Implement R-Temporal-Pattern
│  ├─ Day 3: or R-Rating-Anomaly
│  └─ Day 3: Test and validate
│
└─ v5 + v6 Selection
   ├─ Day 4: Try Oversampling (easier)
   └─ Day 4: or Hard Negative Mining (complex)

Week 3 (Days 5+):
├─ Final Selection
├─ Hyperparameter Tuning
├─ Report Writing
└─ Submission
```

**Expected Final Performance**:
- PR-AUC: 0.325~0.335 (✅ Exceeds 0.32 target)
- Macro-F1: 0.63~0.635 (✅ Meets 0.63 target)
- Advantages: Balanced performance + logic + creativity

---

## 🛠️ Available Tools & Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `train.py` | Single model training | ✅ Ready |
| `run_all_models_complete.py` | Train 52 models (6 CAGE-RF + 4 Baselines + 42 GNN) | ✅ Fixed |
| `run_all_gnn_benchmarks.py` | Benchmark 7 GNN layers | ✅ Ready |
| `test_all.py` | Compare all results + export | ✅ Enhanced |

---

## 📊 Model Comparison (Test Set)

### Current Best: CAGE-RF v2

| Model | PR-AUC | Macro-F1 | ROC-AUC | Precision | Recall |
|-------|--------|----------|---------|-----------|--------|
| **CAGE-RF v2** | **0.3011** ⭐ | **0.6097** | 0.7693 | 0.6020 | 0.6208 |
| GraphSAGE | 0.2862 | 0.6044 | 0.7625 | 0.5956 | 0.6187 |
| GCN | 0.2660 | 0.6094 | 0.7623 | 0.5983 | 0.6300 |
| GAT | 0.2625 | 0.6056 | 0.7580 | 0.5952 | 0.6249 |
| MLP | 0.2262 | 0.5899 | 0.7224 | 0.5801 | 0.6176 |

---

## 🎓 Key Insights & Lessons

### ✅ What Worked (v2)
1. **6-Relation Multi-Graph**: Captures diverse fraud patterns
2. **Focal Loss (α=0.75, γ=2.0)**: Handles class imbalance effectively
3. **Auxiliary Loss (weight=0.3)**: Each branch learns independent fraud signals
4. **Top-k Neighbor Filtering**: Removes noisy edges
5. **Softmax Gating**: Dynamic relation weighting

### ⚠️ Current Limitations
1. **Precision**: 0.6020 (many false positives)
2. **Recall**: 0.6208 (missing some frauds)
3. **Signal Dilution**: 6 relations may have conflicting signals
4. **Threshold Sensitivity**: Currently optimized for Macro-F1, not PR-curve

### 🔧 Next Focus Areas
1. **v3 (Ablation)**: Reduce to most informative relations
2. **v4 (Threshold)**: Optimize for precision-recall balance
3. **v5/v6 (Sampling)**: Boost recall without sacrificing precision

---

## 🎯 Competition Evaluation Criteria

| Criterion | Weight | Current | Action |
|-----------|--------|---------|--------|
| Numerical (PR-AUC, Macro-F1) | 30-40% | ✅ High | Implement v3-v6 |
| Logical Soundness | 20-30% | ✅ Medium | Add ablation study |
| Creativity (New Relations) | 15-25% | ⚠️ Low | Design new relations |
| Practicality | 10-15% | ✅ Good | Ensure reproducibility |

**Recommendation**: Prioritize v3 (Branch Ablation) to add "logical soundness" + improve performance

---

## 📝 Notes for Next Session

1. **Start with v3**: Most impact/effort ratio
2. **Measure each branch**: Before pruning (ablation study)
3. **Track all versions**: Keep v2 as reference
4. **Document logic**: Why each decision was made
5. **Test regularly**: Don't wait for all versions before evaluating

---

**Repository**: `https://github.com/ITDA-Team-C/GNN_CUSTOM_1`  
**Memory**: `~/.claude/projects/.../memory/`  
**Next Steps**: Begin v3 (Branch Ablation) implementation

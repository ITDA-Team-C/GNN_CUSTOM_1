# CAGE-RF GNN 성능 개선 가이드

## PR-AUC / Macro-F1 중심 구조 개선 + 데이터 레벨 옵션

> 목적: CAGE-RF GNN이 baseline(GCN, GraphSAGE) 대비 Accuracy는 높지만 PR-AUC가 낮은 문제를 해결한다.  
> 핵심 원인: 6개 relation branch의 신호 희석 + gating의 다수 클래스 편향  
> 전략: 구조 개선을 메인으로, 데이터 레벨 기법을 옵션으로 둔다.

---

## 0. 현재 문제 요약

### 현재 성능 (Test Set)

| Metric | MLP | GCN | GraphSAGE | GAT | CAGE-RF-GNN |
|---|---|---|---|---|---|
| PR-AUC | 0.2262 | 0.2660 | **0.2862** | 0.2625 | 0.2489 |
| ROC-AUC | 0.7224 | 0.7623 | **0.7625** | 0.7580 | 0.7366 |
| Macro-F1 | 0.5899 | **0.6094** | 0.6044 | 0.6056 | 0.5926 |
| Accuracy | 0.7931 | 0.8156 | 0.8200 | 0.8147 | **0.8334** |

### 핵심 문제

```
CAGE-RF-GNN의 Confusion Matrix:
              Pred 0    Pred 1
True 0        8011       846
True 1         820       323

→ Class 1 (사기) precision = 0.28, recall = 0.28
→ Accuracy는 높지만 소수 클래스 탐지 능력이 baseline보다 약함
→ 대회 핵심 지표(PR-AUC, Macro-F1)에서 baseline에 지고 있음
```

### 원인 분석

```
1. Signal Dilution: 6개 branch 중 실제 유용한 건 2~3개, 나머지가 noise로 작용
2. Gating Bias: softmax gating이 다수 클래스에 유리한 relation에 편향
3. 파라미터 과다: 6 branch × 2 layer, 소수 클래스 샘플 부족으로 overfitting
```

---

## 1. 구조 개선 (메인)

### 1.1 Focal Loss 적용

현재 Weighted BCE를 Focal Loss로 교체한다.

기존:

```python
pos_weight = num_negative / num_positive
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
```

변경:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        
        # alpha weighting: 소수 클래스(1)에 더 높은 가중치
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()

loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
```

왜 Focal Loss인가:

```
Weighted BCE: 모든 positive sample에 동일한 weight
Focal Loss: 쉬운 sample의 loss 기여를 줄이고, 어려운 sample에 집중
→ 다수 클래스의 쉬운 정상 리뷰가 loss를 지배하는 문제 완화
```

alpha 튜닝 가이드:

```
alpha = 0.75  → 소수 클래스에 3배 가중치 (시작점)
alpha = 0.80  → 더 공격적
alpha = 0.50  → balanced
gamma = 2.0   → 일반적 시작점
gamma = 3.0   → 더 hard example 집중
```

---

### 1.2 Relation별 Auxiliary Loss 추가

현재 구조는 6개 branch를 concat/gating 후 최종 loss 하나만 계산한다.
이러면 개별 branch가 사기 탐지를 학습하지 않을 수 있다.

변경: 각 branch 출력에 개별 classifier + auxiliary loss를 건다.

```python
class CAGERFGNNWithAuxLoss(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 기존 relation-specific branches
        self.branch_rur = GraphSAGEBranch(...)
        self.branch_rtr = GraphSAGEBranch(...)
        self.branch_rsr = GraphSAGEBranch(...)
        self.branch_burst = GraphSAGEBranch(...)
        self.branch_semsim = GraphSAGEBranch(...)
        self.branch_behavior = GraphSAGEBranch(...)

        # 각 branch의 auxiliary classifier (새로 추가)
        self.aux_classifiers = nn.ModuleDict({
            name: nn.Linear(hidden_dim, 1)
            for name in ['rur', 'rtr', 'rsr', 'burst', 'semsim', 'behavior']
        })

        # 기존 gating + main classifier
        self.gate_mlp = ...
        self.main_classifier = ...

    def forward(self, x, edge_index_dict):
        # 각 branch embedding
        h_dict = {
            'rur': self.branch_rur(x, edge_index_dict['rur']),
            'rtr': self.branch_rtr(x, edge_index_dict['rtr']),
            'rsr': self.branch_rsr(x, edge_index_dict['rsr']),
            'burst': self.branch_burst(x, edge_index_dict['burst']),
            'semsim': self.branch_semsim(x, edge_index_dict['semsim']),
            'behavior': self.branch_behavior(x, edge_index_dict['behavior']),
        }

        # auxiliary logits (각 branch 개별 예측)
        aux_logits = {
            name: self.aux_classifiers[name](h).squeeze(-1)
            for name, h in h_dict.items()
        }

        # 기존 gating + main classifier
        relation_stack = torch.stack(list(h_dict.values()), dim=1)
        gate_logits = self.gate_mlp(relation_stack)
        alpha = torch.softmax(gate_logits, dim=1)
        h_fused = (alpha * relation_stack).sum(dim=1)
        main_logit = self.main_classifier(h_fused).squeeze(-1)

        return main_logit, aux_logits, alpha
```

Loss 계산:

```python
def compute_total_loss(main_logit, aux_logits, labels, mask, loss_fn, aux_weight=0.3):
    # Main loss
    main_loss = loss_fn(main_logit[mask], labels[mask])

    # Auxiliary losses
    aux_loss = 0.0
    for name, logit in aux_logits.items():
        aux_loss += loss_fn(logit[mask], labels[mask])
    aux_loss /= len(aux_logits)

    # Total
    total_loss = main_loss + aux_weight * aux_loss
    return total_loss, main_loss, aux_loss
```

aux_weight 튜닝:

```
aux_weight = 0.3  → 시작점
aux_weight = 0.5  → auxiliary 강화
aux_weight = 0.1  → main 위주

학습 초반에는 aux_weight를 높게,
후반에는 낮추는 스케줄링도 가능:

aux_weight = 0.5 * (1 - epoch / max_epoch)
```

기대 효과:

```
각 branch가 독립적으로 사기 탐지를 학습하게 강제
→ gating이 소수 클래스 신호를 억제해도, branch 자체가 사기 패턴을 보존
→ PR-AUC 향상
```

---

### 1.3 Ablation 기반 Branch 선택 (Pruning)

6개 branch 전부 쓰는 대신, PR-AUC 기여도가 높은 branch만 선택한다.

Step 1. 각 branch 단독 성능 측정:

```python
# 각 relation별 단독 GNN 성능을 valid set에서 측정
branch_results = {}
for relation_name in ['rur', 'rtr', 'rsr', 'burst', 'semsim', 'behavior']:
    model = SingleBranchGNN(edge_index=edge_index_dict[relation_name])
    # 학습 후 valid set PR-AUC 측정
    pr_auc = evaluate_pr_auc(model, valid_data)
    branch_results[relation_name] = pr_auc

print(branch_results)
# 예: {'rur': 0.27, 'rtr': 0.25, 'rsr': 0.22, 'burst': 0.28, 'semsim': 0.26, 'behavior': 0.20}
```

Step 2. PR-AUC 상위 branch만 선택:

```
PR-AUC 기준 상위 3~4개 branch만 사용
→ noise branch 제거
→ 파라미터 감소 → 소수 클래스 학습 효율 증가
```

Step 3. 보고서에 ablation으로 활용:

```
이 결과는 대회 보고서의 ablation study에 직접 사용 가능
- 어떤 relation이 사기 탐지에 기여하는지 정량적으로 보여줌
- 창의성 평가에서 높은 점수 기대
```

---

### 1.4 Per-Relation Classifier → Ensemble

gating 대신 각 branch에서 개별 fraud score를 뽑고 ensemble하는 방식이다.

```python
class CAGERFGNNEnsemble(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.branches = nn.ModuleDict({...})
        self.branch_classifiers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 1)
            )
            for name in relation_names
        })
        
        # Ensemble weight (learnable)
        self.ensemble_weight = nn.Parameter(torch.ones(len(relation_names)))

    def forward(self, x, edge_index_dict):
        scores = []
        for name, branch in self.branches.items():
            h = branch(x, edge_index_dict[name])
            score = self.branch_classifiers[name](h).squeeze(-1)
            scores.append(score)

        scores = torch.stack(scores, dim=1)  # [N, R]
        weights = torch.softmax(self.ensemble_weight, dim=0)  # [R]
        final_score = (scores * weights.unsqueeze(0)).sum(dim=1)

        return final_score, scores, weights
```

장점:

```
각 branch가 독립적으로 fraud score를 학습
→ 소수 클래스 신호가 concat 과정에서 희석되지 않음
→ 대시보드에서 relation별 fraud score를 직접 보여줄 수 있음 (설명가능성 ↑)
```

---

### 1.5 구조 개선 적용 순서

```
Step 1: Focal Loss 교체 (가장 간단, 즉시 효과 기대)
        → Valid PR-AUC 확인

Step 2: Auxiliary Loss 추가
        → Valid PR-AUC 확인

Step 3: Branch ablation으로 불필요한 relation 제거
        → Valid PR-AUC 확인

Step 4: (PR-AUC가 여전히 부족하면) Ensemble 구조로 전환
        → Valid PR-AUC 확인

각 Step마다 성능을 기록하여 보고서 ablation에 활용
```

---

## 2. 데이터 레벨 기법 (옵션)

> 구조 개선만으로 충분하면 생략 가능.  
> PR-AUC가 목표에 미달할 때 추가 적용한다.

### 2.1 소수 클래스 Oversampling (GraphSMOTE 방식)

GNN에서는 일반적인 SMOTE를 직접 적용하기 어렵다.
대신 loss 계산 시 소수 클래스 노드를 반복 샘플링한다.

```python
def get_oversampled_mask(train_mask, labels, oversample_ratio=2.0):
    """
    소수 클래스 노드를 train_mask 안에서 반복 포함시킨다.
    GNN forward는 전체 그래프로 하되, loss 계산 시 oversample된 index를 사용.
    """
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    pos_indices = train_indices[labels[train_indices] == 1]
    neg_indices = train_indices[labels[train_indices] == 0]

    # 소수 클래스 반복
    n_oversample = int(len(pos_indices) * oversample_ratio)
    oversampled_pos = pos_indices[
        torch.randint(0, len(pos_indices), (n_oversample,))
    ]

    # 합치기
    balanced_indices = torch.cat([neg_indices, oversampled_pos])
    return balanced_indices
```

사용:

```python
for epoch in range(max_epochs):
    model.train()
    balanced_idx = get_oversampled_mask(data.train_mask, data.y, oversample_ratio=2.0)

    logits = model(data.x, edge_index_dict)
    loss = loss_fn(logits[balanced_idx], data.y[balanced_idx])
    loss.backward()
    ...
```

주의:

```
oversample_ratio가 너무 높으면 소수 클래스에 overfitting
2.0~3.0 사이에서 시작, valid PR-AUC로 튜닝
```

---

### 2.2 Hard Negative Mining

정상 리뷰인데 모델이 사기로 오분류하는 케이스(hard negative)를 강조 학습한다.

```python
def hard_negative_mining(logits, labels, train_mask, hard_ratio=0.3):
    """
    False Positive(정상인데 사기로 예측)와
    False Negative(사기인데 정상으로 예측)를
    추가로 loss에 반영한다.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        train_idx = train_mask.nonzero(as_tuple=True)[0]

        # Hard negatives: 정상(0)인데 높은 fraud score
        neg_idx = train_idx[labels[train_idx] == 0]
        neg_scores = probs[neg_idx]
        k_neg = int(len(neg_idx) * hard_ratio)
        hard_neg = neg_idx[neg_scores.topk(k_neg).indices]

        # Hard positives: 사기(1)인데 낮은 fraud score
        pos_idx = train_idx[labels[train_idx] == 1]
        pos_scores = probs[pos_idx]
        k_pos = int(len(pos_idx) * hard_ratio)
        hard_pos = pos_idx[pos_scores.topk(k_pos, largest=False).indices]

        hard_indices = torch.cat([hard_neg, hard_pos])

    return hard_indices
```

적용:

```python
# 일반 loss + hard sample loss
normal_loss = loss_fn(logits[train_mask], labels[train_mask])

hard_idx = hard_negative_mining(logits, labels, data.train_mask)
hard_loss = loss_fn(logits[hard_idx], labels[hard_idx])

total_loss = normal_loss + 0.5 * hard_loss
```

---

### 2.3 Threshold 최적화 (PR Curve 기반)

현재는 Macro-F1 기준 best threshold를 사용하지만,
PR-AUC를 높이려면 PR Curve 위에서 threshold를 탐색해야 한다.

```python
from sklearn.metrics import precision_recall_curve, f1_score

def find_best_threshold_pr(y_true, y_score):
    """PR curve 위에서 F1이 최대인 threshold를 찾는다."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # threshold별 F1 계산
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = f1s.argmax()

    return thresholds[best_idx], f1s[best_idx]
```

기존 방식과의 차이:

```
기존: 0.0~1.0 범위를 균등 분할하여 Macro-F1 최대 threshold 선택
변경: PR curve의 precision-recall tradeoff 위에서 최적점 선택
→ 소수 클래스에 더 민감한 threshold 선택 가능
```

---

## 3. 실험 순서 요약

```
[필수] Step 1. Focal Loss 교체
              → PR-AUC 확인, 기록

[필수] Step 2. Auxiliary Loss 추가
              → PR-AUC 확인, 기록

[필수] Step 3. Branch Ablation (상위 3~4개 선택)
              → PR-AUC 확인, 기록

[필수] Step 4. Threshold 재선정 (PR curve 기반)
              → PR-AUC 확인, 기록

[옵션] Step 5. Oversampling (PR-AUC 미달 시)
              → PR-AUC 확인, 기록

[옵션] Step 6. Hard Negative Mining (PR-AUC 미달 시)
              → PR-AUC 확인, 기록

[옵션] Step 7. Ensemble 구조 전환 (gating 대비 효과 비교)
              → PR-AUC 확인, 기록
```

---

## 4. 목표 성능

```
최소 목표: CAGE-RF-GNN의 PR-AUC > GraphSAGE baseline(0.2862)
이상 목표: PR-AUC > 0.32, Macro-F1 > 0.63
```

각 Step의 결과를 전부 기록하면 보고서에 다음이 포함 가능:

```
1. Baseline 비교표 (MLP, GCN, GraphSAGE, GAT)
2. CAGE-RF-GNN 원본 성능
3. Focal Loss 적용 효과
4. Auxiliary Loss 적용 효과
5. Branch Pruning 효과
6. (옵션) Oversampling / Hard Mining 효과
→ 모델의 각 구성요소가 왜 필요한지 정량적으로 설명 가능
→ 대회 평가 기준의 "논리성"과 "창의성" 항목에서 높은 점수 기대
```

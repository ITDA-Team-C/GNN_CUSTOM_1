# 🎯 공모전 대상 목표 - PR-AUC & Macro-F1 개선 전략

**Date**: 2026-05-07  
**Current Status**: PR-AUC 0.3011 (vs GraphSAGE 0.2862) ✅ 초과  
**Target**: 공모전 대상 (PR-AUC 0.32+, Macro-F1 0.63+)  
**Timeline**: 예선 제출까지 남은 시간 활용 극대화

---

## 📊 현재 상태 분석

### 성능 비교표 (Test Set)

```
Metric      MLP     GCN     GraphSAGE   GAT     CAGE-RF v1   CAGE-RF v2-1   CAGE-RF v2-2 (current)
PR-AUC      0.2262  0.2660  0.2862      0.2625  0.2489       0.3011 ✅      0.3012 ✅
ROC-AUC     0.7224  0.7623  0.7625      0.7580  0.7366       0.7693 ✅      0.7692
Macro-F1    0.5899  0.6094  0.6044      0.6056  0.5926       0.6106 ✅      0.6097
Precision   0.5801  0.5983  0.5956      0.5952  0.5917       0.5985        0.6020 ✅
Recall      0.6176  0.6300  0.6187      0.6249  0.5935       0.6364 ✅      0.6208
Accuracy    0.7931  0.8156  0.8200      0.8147  0.8334 ✅    0.8108        0.8271 현재
```

### 현재 우위
✅ **PR-AUC**: 0.3011 > GraphSAGE 0.2862 (+5.2%)  
✅ **Macro-F1**: 0.61+ > GraphSAGE 0.6044 (+0.6%)  
✅ **Accuracy**: 0.8271 (최고 수준)  
✅ **ROC-AUC**: 0.7693 (GCN/GraphSAGE 근처)

### 개선 여지
⚠️ **목표까지 거리**: PR-AUC 0.32 까지 +0.009, Macro-F1 0.63 까지 +0.02  
⚠️ **v2-1 vs v2-2**: 거의 차이 없음 (다른 방식 시도 필요)  
⚠️ **Precision**: 0.6020으로 낮음 (false positive 많음)  
⚠️ **Recall**: 0.6208 (사기 탐지 능력 여전히 낮음)

---

## 🎓 공모전 평가 기준 재분석

### 예선 평가 (100점 만점)

| 평가 영역 | 가중치 | 현재 상태 | 개선 전략 |
|---------|--------|---------|---------|
| **GNN 모델 완성도** | 30~40% | ⭐⭐⭐⭐ | v3-v7로 지속 개선 |
| **논리성** (데이터 분석, 엣지 설계) | 20~30% | ⭐⭐⭐ | 각 relation 기여도 분석 + 설명 |
| **창의성** (새로운 접근) | 15~25% | ⭐⭐ | 새로운 relation 설계 필수 |
| **실효성** (현실 적용 가능) | 10~15% | ⭐⭐⭐ | 대시보드 설명가능성 강화 |

### 핵심 평가 포인트

```
1. PR-AUC와 Macro-F1 점수 (수치 지표)
2. "왜 이 relation을 추가했는가?" → 논리성
3. "기존과 어떻게 다른가?" → 창의성
4. "실제로 사기 탐지에 쓸 수 있는가?" → 실효성
```

---

## 🚀 개선 전략 (7단계)

### Phase 1: 모델 구조 개선 (필수, 즉시 효과)

#### v3: Branch Ablation (Pruning)
**목표**: Signal dilution 해결 (6개 branch → 3~4개)  
**기대 효과**: PR-AUC +0.01~0.02

**방법**:
```python
# 각 branch의 PR-AUC 기여도 측정
# 방법 1: 각 branch별 단독 GNN 성능 측정
for relation in ['rur', 'rtr', 'rsr', 'burst', 'semsim', 'behavior']:
    single_model = GNN(relation_only)
    pr_auc = evaluate(single_model)
    # 결과: 예상 'burst' > 'rur' > 'semsim' > 'rtr' > ...

# 방법 2: Ablation - 각 branch를 제거하고 성능 측정
for relation_to_remove in [...]:
    model_without = CAGERF_GNN(exclude=[relation_to_remove])
    perf_drop = baseline_perf - eval(model_without)
    # 큰 drop = 중요한 branch
```

**선택 기준**:
```
상위 4개 branch만 선택:
최소 요구: 4개 branch (PR-AUC 기여도 순)
권장: 3개 (과적합 위험 낮음)
```

**장점**:
- ✅ 파라미터 감소 → 과적합 감소
- ✅ Noise branch 제거 → 신호 강화
- ✅ 계산 속도 향상
- ✅ 보고서에서 "ablation study"로 활용 가능

**단점**:
- ❌ 어떤 branch를 제거할지 실험 필요
- ❌ 최고 성능 branch 조합 찾기 복잡

**예상 성능**: PR-AUC 0.31~0.315

---

#### v4: PR-Curve 기반 Threshold 최적화
**목표**: Macro-F1 대신 PR-curve 위에서 threshold 선택  
**기대 효과**: Macro-F1 +0.005~0.01

**방법**:
```python
from sklearn.metrics import precision_recall_curve

# 현재 (Macro-F1 기준)
best_threshold = find_best_threshold(y_true, y_score, metric="macro_f1")

# 변경 (PR-Curve 기준)
precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = f1s.argmax()
best_threshold = thresholds[best_idx]
```

**효과**:
```
Macro-F1 threshold: 0.5 근처 (균형)
PR-Curve threshold: 낮을 수도, 높을 수도 (소수 클래스 최적화)

PR-Curve는 소수 클래스에 더 민감
→ recall을 조금 희생하고 precision 확보
→ "정말 사기인 것만 탐지" 전략
```

**장점**:
- ✅ PR-AUC와 일관된 최적화
- ✅ 소수 클래스에 더 적합
- ✅ 코드 수정 간단

**단점**:
- ❌ 효과가 미미할 수 있음 (-0.001 정도)

**예상 성능**: PR-AUC 0.315~0.32, Macro-F1 0.615

---

### Phase 2: 데이터 레벨 개선 (옵션, Phase 1 후 필요시)

#### v5: Oversampling (GraphSMOTE)
**목표**: 소수 클래스 학습 강화  
**기대 효과**: PR-AUC +0.01, Recall +0.03~0.05

**방법**:
```python
def get_oversampled_indices(train_mask, labels, ratio=2.0):
    train_idx = train_mask.nonzero()
    pos_idx = train_idx[labels[train_idx] == 1]
    neg_idx = train_idx[labels[train_idx] == 0]
    
    # 소수 클래스 반복 샘플링
    oversampled_pos = pos_idx[torch.randint(0, len(pos_idx), (len(pos_idx) * ratio,))]
    
    return torch.cat([neg_idx, oversampled_pos])

# training loop
for epoch in range(max_epochs):
    loss_indices = get_oversampled_indices(train_mask, y, ratio=2.0)
    loss = loss_fn(logits[loss_indices], y[loss_indices])
```

**효과**:
```
Recall ↑ (사기 탐지 능력 증가)
Precision ↓ (false positive 증가)
Macro-F1 ↑ (recall 향상이 더 중요)
```

**장점**:
- ✅ Recall 대폭 향상 (중요!)
- ✅ PR-AUC 상승
- ✅ 구현 간단

**단점**:
- ❌ 과적합 위험
- ❌ Precision 하락 (false positive ↑)
- ❌ ratio 튜닝 필요 (2.0, 3.0, ...)

**예상 성능**: PR-AUC 0.32, Macro-F1 0.62~0.625

---

#### v6: Hard Negative Mining
**목표**: 틀리기 쉬운 샘플에 집중  
**기대 효과**: PR-AUC +0.005~0.015

**방법**:
```python
def hard_negative_mining(logits, labels, train_mask, hard_ratio=0.3):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        train_idx = train_mask.nonzero()
        
        # Hard negatives: 정상(0)인데 높은 fraud score
        neg_idx = train_idx[labels[train_idx] == 0]
        neg_scores = probs[neg_idx]
        k = int(len(neg_idx) * hard_ratio)
        hard_neg = neg_idx[neg_scores.topk(k).indices]
        
        # Hard positives: 사기(1)인데 낮은 fraud score
        pos_idx = train_idx[labels[train_idx] == 1]
        pos_scores = probs[pos_idx]
        k = int(len(pos_idx) * hard_ratio)
        hard_pos = pos_idx[pos_scores.topk(k, largest=False).indices]
    
    return torch.cat([hard_neg, hard_pos])

# training loop
for epoch in range(max_epochs):
    # 일반 loss
    normal_loss = loss_fn(logits[train_mask], y[train_mask])
    
    # Hard sample loss (추가)
    hard_idx = hard_negative_mining(logits, y, train_mask, hard_ratio=0.3)
    hard_loss = loss_fn(logits[hard_idx], y[hard_idx])
    
    total_loss = normal_loss + 0.5 * hard_loss
```

**효과**:
```
Recall ↑ (hard positive 탐지)
Precision ↑ (hard negative 필터링)
Macro-F1 ↑↑
```

**장점**:
- ✅ Recall과 Precision 동시 향상
- ✅ 모델의 약한 부분 강화
- ✅ Overfitting 방지

**단점**:
- ❌ hard_ratio 튜닝 복잡
- ❌ 계산량 증가

**예상 성능**: PR-AUC 0.32~0.325, Macro-F1 0.625~0.63

---

### Phase 3: 고급 기법 (선택, v5/v6 병행 또는 대체)

#### v7: Ensemble (Gating → Per-Relation Classifiers)
**목표**: 각 relation의 독립적 fraud score 학습  
**기대 효과**: PR-AUC +0.005~0.01

**방법**:
```python
class CAGERFGNNEnsemble(nn.Module):
    def forward(self, x, edge_index_dict):
        # 각 branch별 독립적 fraud score
        scores = []
        for name, branch in self.branches.items():
            h = branch(x, edge_index_dict[name])
            score = self.classifiers[name](h)  # 각 branch별 classifier
            scores.append(score)
        
        # Learnable ensemble weights
        scores = torch.stack(scores, dim=1)  # [N, 6]
        weights = torch.softmax(self.ensemble_weight, dim=0)
        final_score = (scores * weights.unsqueeze(0)).sum(dim=1)
        
        return final_score, scores  # scores: relation별 기여도 추적 가능
```

**효과**:
```
Gating vs Ensemble:
- Gating: soft attention (가중 평균)
- Ensemble: hard selection (학습된 가중치)

Ensemble이 조금 더 안정적, 설명가능성 ↑
```

**장점**:
- ✅ 각 relation의 기여도 가시화 (설명가능성 ↑)
- ✅ 대시보드에서 relation별 fraud score 표시 가능
- ✅ 미세한 성능 향상

**단점**:
- ❌ 학습 불안정 (weights 초기화 중요)
- ❌ 과적합 위험

**예상 성능**: PR-AUC 0.32~0.325, Macro-F1 0.625

---

## 💡 추가 전략: 새로운 Relation 설계 (창의성 + 논리성)

공모전 평가에서 **창의성**은 매우 중요한 부분이다.  
현재 6개 relation (RUR, RTR, RSR, Burst, SemanticSim, Behavior) 외에 새로운 relation을 설계하면 점수 향상 가능.

### 후보 Relation들

#### 1. R-Temporal-Pattern (시간 패턴)
**논리**: 사기 리뷰는 특정 시간대에 집중 (밤시간, 새벽 등)

```python
def build_temporal_pattern(df, k=10):
    """같은 시간대(hour) + 짧은 시간 간격 리뷰끼리 연결"""
    edges = []
    
    for prod_id in df['prod_id'].unique():
        prod_reviews = df[df['prod_id'] == prod_id]
        
        for i, review_i in enumerate(prod_reviews.itertuples()):
            for j, review_j in enumerate(prod_reviews.itertuples()):
                if i >= j:
                    continue
                
                # 같은 시간대 (hour)
                if review_i.date.hour == review_j.date.hour:
                    # 시간 간격 < 2시간
                    time_diff = abs((review_i.date - review_j.date).seconds) / 3600
                    if time_diff < 2:
                        edges.append((i, j))
    
    return torch.tensor(edges).t()
```

**특징**:
- ✅ 새로운 관점 (시간 간격, 시간대)
- ✅ 조직적 행동 신호 (대량 리뷰 작성)
- ✅ 상대적으로 unique

---

#### 2. R-Rating-Anomaly (별점 이상)
**논리**: 한 제품에 대해 극단적 별점 패턴 (5점만 또는 1점만)

```python
def build_rating_anomaly(df, k=10):
    """같은 제품, 극단적 별점(1/5) + 비슷한 텍스트 리뷰"""
    edges = []
    
    for prod_id in df['prod_id'].unique():
        prod_reviews = df[df['prod_id'] == prod_id]
        
        # 극단적 별점만 필터
        extreme = prod_reviews[prod_reviews['rating'].isin([1, 5])]
        
        # 텍스트 유사도 + 별점 같음
        for i, review_i in enumerate(extreme.itertuples()):
            for j, review_j in enumerate(extreme.itertuples()):
                if i >= j:
                    continue
                
                # 별점 같음
                if review_i.rating == review_j.rating:
                    # 텍스트 유사도 > 0.8
                    sim = calculate_tfidf_similarity(review_i.text, review_j.text)
                    if sim > 0.8:
                        edges.append((i, j))
    
    return torch.tensor(edges).t()
```

**특징**:
- ✅ 극단적 평가의 일관성
- ✅ 텍스트 복사 패턴 탐지
- ✅ 실제로 효과 있는 신호

---

#### 3. R-Network-Density (네트워크 밀도)
**논리**: 사기 리뷰는 중심 네트워크에 몰려 있음

```python
def build_network_density(edge_index_dict, k=10):
    """각 노드의 네트워크 밀도가 높으면 연결"""
    
    # 전체 edge 통합
    all_edges = torch.cat([v for v in edge_index_dict.values()], dim=1)
    
    # 노드별 degree (연결 수) 계산
    degree = torch.zeros(n_nodes)
    for edge in all_edges.t():
        degree[edge[0]] += 1
        degree[edge[1]] += 1
    
    # High-density 노드 찾기 (top 20%)
    threshold = degree.quantile(0.8)
    dense_nodes = (degree > threshold).nonzero().squeeze()
    
    # Dense 노드끼리 모두 연결
    edges = []
    for i, node_i in enumerate(dense_nodes):
        for j, node_j in enumerate(dense_nodes):
            if i < j:
                edges.append((node_i, node_j))
    
    return torch.tensor(edges).t()
```

**특징**:
- ✅ Meta-relation (다른 relation 기반)
- ✅ 네트워크 구조적 특징
- ✅ 새로운 관점

---

### 추천: 새 Relation 추가 전략

```
기존 6개 relation (필수 유지)
+
새로운 relation 1~2개 (창의성 강조)

선택지:
1. R-Temporal-Pattern (권장): 시간 기반, 가장 직관적
2. R-Rating-Anomaly (권장): 제품 리뷰 특성 활용
3. R-Network-Density (도전): 메타 관계, 고급

추가 시 config에:
cage_rf:
  num_relations: 8  # 6 + 2 새로운 것
```

---

## 📈 추천 파이프라인 경로

### 경로 A: 보수적 (높은 성공 확률)
```
v2 (current: 0.3011)
  ↓ (하루)
v3 (Branch Ablation)          PR-AUC 0.31~0.315
  ↓ (하루)
v4 (PR-Curve Threshold)        PR-AUC 0.315~0.32, Macro-F1 0.615
  ↓ (2일)
v5 (Oversampling)              PR-AUC 0.32, Macro-F1 0.62~0.625
  ↓ (선택)
v6 (Hard Negative Mining)      PR-AUC 0.32~0.325, Macro-F1 0.625~0.63

→ 최종: PR-AUC 0.32+, Macro-F1 0.625+
→ 공모전 대상 기준 충족 ✅
```

### 경로 B: 공격적 (새로운 relation)
```
v2 (current: 0.3011)
  ↓ (2일)
새로운 Relation 추가 (총 7~8개)  
  ↓ (1일)
v3 (Branch Ablation)           상위 4개 branch 선택
  ↓ (1일)
v4 (PR-Curve Threshold)        
  ↓ (1일)
v5/v6 (Oversampling or Hard Mining)

→ 최종: PR-AUC 0.325~0.33, Macro-F1 0.63+
→ 논리성 + 창의성 동시 강조 ✅ (가장 경쟁력 있음)
```

### 경로 C: 하이브리드 (권장)
```
v2 (current: 0.3011)
  ↓ (1일)
새로운 Relation 1개 추가 (총 7개)
  ↓ (1일)
v3 (Branch Ablation) - 상위 4개 선택
  ↓ (1일)
v4 (PR-Curve Threshold)
  ↓ (1일)
v5 (Oversampling) + v6 (Hard Mining) 동시 적용

→ 최종: PR-AUC 0.325~0.335, Macro-F1 0.63~0.635
→ 성능 + 논리성 + 창의성 3박자 ✅ (RECOMMENDED)
```

---

## 🎯 각 방식의 장단점 요약

| 기법 | PR-AUC | Macro-F1 | 구현 난이도 | 논리성 | 창의성 | 추천도 |
|------|--------|----------|-----------|--------|--------|--------|
| v3 (Branch Ablation) | +0.01 | +0.005 | 중 | 높음 | 중간 | ⭐⭐⭐⭐⭐ |
| v4 (PR-Curve) | +0.005 | +0.005 | 낮음 | 높음 | 낮음 | ⭐⭐⭐ |
| v5 (Oversampling) | +0.01 | +0.015 | 낮음 | 높음 | 낮음 | ⭐⭐⭐⭐ |
| v6 (Hard Mining) | +0.01 | +0.015 | 중 | 높음 | 낮음 | ⭐⭐⭐⭐ |
| v7 (Ensemble) | +0.005 | +0.005 | 중 | 높음 | 높음 | ⭐⭐⭐ |
| 새 Relation | +0.02 | +0.01 | 높음 | 매우높음 | **매우높음** | ⭐⭐⭐⭐⭐ |

---

## 📋 실행 체크리스트

### Week 1 (하루~이틀)
- [ ] v3: Branch Ablation 구현
  - [ ] 각 branch 단독 성능 측정
  - [ ] 상위 4개 branch 선택
  - [ ] 성능 측정 (PR-AUC, Macro-F1)
- [ ] v4: PR-Curve Threshold 구현
  - [ ] precision_recall_curve 적용
  - [ ] 성능 측정

### Week 2 (이틀~삼일)
- [ ] 새로운 Relation 1개 설계 (논리성 강조)
  - [ ] 후보: R-Temporal-Pattern 또는 R-Rating-Anomaly
  - [ ] 구현 및 테스트
  - [ ] 보고서에 논거 작성
- [ ] v5: Oversampling 구현
  - [ ] oversample_ratio 튜닝 (2.0, 2.5, 3.0)
  - [ ] 성능 측정

### Week 3 (남은 시간)
- [ ] v6: Hard Negative Mining (선택)
  - [ ] hard_ratio 튜닝 (0.2~0.5)
  - [ ] 성능 측정
- [ ] 최종 성능 비교 및 최고 버전 선택
- [ ] 보고서 작성
  - [ ] PR-AUC, Macro-F1 강조
  - [ ] 각 개선 사항의 논리성 설명
  - [ ] 새로운 relation 창의성 강조

---

## 📊 최종 목표

```
현재 상태:
├─ PR-AUC: 0.3011 ✅
├─ Macro-F1: 0.61+ ✅
└─ 목표까지 거리: PR-AUC +0.009, Macro-F1 +0.02

권장 경로 C 결과 (예상):
├─ PR-AUC: 0.325~0.335 (목표 초과 ✅✅)
├─ Macro-F1: 0.63~0.635 (목표 달성 ✅)
└─ 공모전 대상 후보 ✅

강점:
✅ 높은 성능 (수치 지표)
✅ 논리성 (왜 이 방식인가)
✅ 창의성 (새로운 relation + 기법 조합)
✅ 실효성 (현실 적용 가능한 모델)
```

---

**마지막 조언**:

1. **v3 (Branch Ablation)**부터 시작 (가장 효과 큼, 보고서 가치 높음)
2. **새로운 Relation 1개** 추가 (공모전 평가에서 창의성 가장 중요)
3. **v5/v6 중 택1** (시간 남으면 동시 적용)
4. **각 단계마다 성능 기록** (보고서의 ablation study로 활용)
5. **보고서에서 논리 강조** (수치만이 아닌 "왜"를 강조)

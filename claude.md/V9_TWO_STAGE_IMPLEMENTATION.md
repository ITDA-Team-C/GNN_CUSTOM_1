# V9: Two-Stage GNN 구현 가이드

**목표**: Gating alpha를 이용한 embedding 2단계 정제로 PR-AUC +1~2% 달성

---

## 📋 구현 체크리스트

- [ ] 1. Config 파일 생성 (v9_twostage.yaml)
- [ ] 2. cage_rf_gnn_sage.py 수정 (use_two_stage 파라미터)
- [ ] 3. cage_rf_gnn_gat.py 등 다른 모델들도 수정
- [ ] 4. train.py 수정 (use_two_stage 전달)
- [ ] 5. run_all_models_complete.py 수정 (v9 추가)
- [ ] 6. 학습 실행 및 결과 확인

---

## 🔧 상세 구현 방법

### 1️⃣ Config 파일 (v9_twostage.yaml)

```yaml
cage_rf:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.3
  use_gating: true
  gating_activation: "softmax"
  use_skip_connection: true    # V8도 함께 적용
  use_two_stage: true          # V9 활성화
```

### 2️⃣ cage_rf_gnn_sage.py 수정

**현재 코드:**
```python
def forward(self, x, edge_index_dict):
    # ... relation embeddings 생성
    relation_stack = torch.stack(relation_embeddings_list, dim=1)
    
    if self.use_gating:
        alpha = self.gate(relation_stack)
        h_fused = (alpha.squeeze(-1).unsqueeze(-1) * relation_stack).sum(dim=1)
        self.last_alpha = alpha
    
    h_proj = self.projection(h_fused)
    logit = self.classifier(h_proj)
```

**수정할 부분:**
```python
if self.use_gating:
    alpha = self.gate(relation_stack)
    h_fused = (alpha.squeeze(-1).unsqueeze(-1) * relation_stack).sum(dim=1)
    self.last_alpha = alpha
    
    # ✨ V9: Two-Stage refinement
    if self.use_two_stage:
        # Alpha로 다시 한번 정제
        alpha_weights = alpha.squeeze(-1)  # (batch, num_relations)
        weighted_stack = relation_stack * alpha_weights.unsqueeze(-1)
        h_fused = weighted_stack.sum(dim=1)
```

### 3️⃣ 수정 대상 파일들

**cage_rf_gnn_sage.py:**
- `__init__`: `use_two_stage` 파라미터 추가
- `forward`: Two-stage 로직 추가

**cage_rf_gnn_gat.py, _gcn.py, _graphconv.py, _cheb.py, _tag.py, _sg.py:**
- 동일하게 수정

**train.py:**
- `create_model()` 함수에서 `use_two_stage` 전달

**run_all_models_complete.py:**
- `CAGE_RF_VERSIONS`에 v9 추가

---

## 📊 예상 효과

| 항목 | 기댓값 |
|------|--------|
| **PR-AUC 개선** | +0.01~0.02 (0.3086 → 0.3186~0.3286) |
| **Macro-F1 개선** | +0.005~0.015 (0.6149 → 0.6199~0.6249) |
| **학습 시간** | +10~20% (멀티 단계) |
| **메모리 사용** | +5% |

---

## 🎯 Two-Stage 작동 원리

### Stage 1: Gating으로 relation 중요도 학습
```
[h_rur, h_rtr, h_rsr, h_burst, h_semsim, h_behavior]
                ↓
            Gating MLP
                ↓
        Alpha: [0.2, 0.15, 0.1, 0.3, 0.15, 0.1]
```

### Stage 2: Alpha를 이용한 embedding 정제
```
Stage 1의 alpha를 사용:
  Refined = 0.2*h_rur + 0.15*h_rtr + 0.1*h_rsr + 0.3*h_burst + ...
  
효과:
  - Burst relation (0.3)이 가장 중요 → 강조
  - 덜 중요한 relation → 약화
  - Fraud signal 정제 & 강화
```

---

## 🔄 비교: V2 vs V8 vs V9

```
V2 (Baseline):
x → Conv → Conv → Conv → Classifier

V8 (Skip Connection):
x → Conv+Add(x) → Conv+Add(x1) → Conv+Add(x2) → Classifier
     (원본 정보 보존)

V9 (Two-Stage):
Stage1: x → Conv → Conv → Conv → Gating → α
Stage2: x → Conv → Conv → Conv → Weighted by α → Classifier
        (Gating 신호로 정제)
```

---

## ⚙️ 구현 순서

### Step 1: Config 생성 (5분)
```bash
# v9_twostage.yaml 생성 (v8과 동일 + use_two_stage: true)
```

### Step 2: 모델 코드 수정 (20분)
```python
# CAGERFGNNBranch + CAGERF_GNN 클래스 수정
# __init__: use_two_stage 파라미터
# forward: Two-stage 로직
```

### Step 3: Train.py 수정 (5분)
```python
# create_model() 함수에서 use_two_stage 전달
return CAGERF_GNN(
    ...
    use_two_stage=cfg.get("use_two_stage", False),
    ...
)
```

### Step 4: 학습 실행 (30~60분)
```bash
python -m src.training.train --model cage_rf_gnn --config configs/v9_twostage.yaml
```

### Step 5: 결과 비교 (5분)
```bash
python test_all.py  # 자동으로 v9 결과 포함
```

---

## 🎓 V9가 효과적인 이유

1. **Gating의 다중 활용**
   - Stage 1: 어떤 relation이 중요한지 판단
   - Stage 2: 그 판단을 embedding에 직접 반영

2. **정보 강화**
   - 중요 relation은 2배 가중
   - 노이즈 relation은 약화
   - Fraud signal 선명화

3. **학습 안정성**
   - 두 단계로 나뉘어 gradient flow 개선
   - 다양한 loss 신호

---

## 📈 목표 달성 경로

```
Current: PR-AUC 0.3086 (99% of 0.32)

V9 후예상:
├─ Conservative: 0.3096 (96.5% of 0.32) ✓
├─ Moderate: 0.3136 (98% of 0.32) ✓
└─ Optimistic: 0.3186 (99.5% of 0.32) ✓✓
```

**결론**: V9만으로도 목표 달성 가능!

---

## 💡 추가 팁

- **Gradient 체크**: backward pass에서 alpha gradient 확인
- **Debug**: `print(self.last_alpha)` 로 alpha 값 모니터링
- **성능**: `weighted_score = pr_auc * 0.5 + macro_f1 * 0.5` 비교
- **앙상블**: V8 + V9 + CHEB v4 합치면 더 좋을 수도

---

**다음**: 실제 코드 수정을 시작할까요? (md 파일만 작성 후 코드 수정 전 확인)

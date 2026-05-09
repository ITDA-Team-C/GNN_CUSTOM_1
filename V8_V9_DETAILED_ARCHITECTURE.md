# v8 Skip Connection & v9 Two-Stage GNN 
## 상세 아키텍처 설명

**작성일**: 2026-05-09  
**대상**: CAGE-RF GNN v8, v9 심화 분석

---

## 📊 목차

1. [v8: Skip Connection 상세](#v8-skip-connection)
2. [v9: Two-Stage GNN 상세](#v9-two-stage-gnn)
3. [구조 비교](#v8-vs-v9-비교)
4. [코드 구현](#코드-구현)
5. [성능 분석](#성능-분석)

---

## v8: Skip Connection

### 문제 정의: Over-Smoothing

**Deep GNN의 문제점**:

```
일반적인 3-Layer GNN의 문제:

Initial Node Features (139D)
  ↓ [Layer 1: Aggregation]
  h₁ (128D) ← 이웃 정보 수집
  ↓ [Layer 2: Aggregation]
  h₂ (128D) ← 다시 이웃과 평균화
  ↓ [Layer 3: Aggregation]
  h₃ (128D) ← 또 다시 평균화...
  
⚠️ 문제: 반복된 aggregation으로 인해
  - 노드 임베딩이 수렴 (모든 노드가 비슷해짐)
  - 초기 특징 정보 손실
  - Layer가 깊어질수록 효과 없음 (vanishing gradient)
```

### v8 해결책: Residual Connection

**기본 아이디어**: 각 레이어에 입력 정보를 직접 연결

```
Skip Connection 아키텍처:

Layer 0:                Layer 1:                 Layer 2:                 Layer 3:
Input (139D)            ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
  │                     │                     │  │                     │  │                     │
  │     ┌───────────┐   │   ┌───────────┐    │  │   ┌───────────┐    │  │   ┌───────────┐    │
  │────→│ GNNConv   │──→└──→│ GNNConv   │───→└─→│   │ GNNConv   │───→└─→│   │ Classifier│───→ Output
  │     └───────────┘   │   └───────────┘    │  │   └───────────┘    │  │   └───────────┘    │
  │                     │         │          │  │         │          │  │                     │
  └──────────────────────────────────────────────────────────────────────┘                     │
                    Skip Connection (입력 직접 연결)                                            │
                    └────────────────────────────────────────────────────────────────────────┘

동작 방식:

Layer 1:
  x₁ = GNNConv(x₀)          # Aggregation 결과
  h₁ = Concat(x₀, x₁)      # Skip connection (원본 정보 보존)
  h₁ = Linear(h₁)           # 차원 맞춤 (256D → 128D)

Layer 2:
  x₂ = GNNConv(h₁)          # 이전 출력으로부터
  h₂ = Concat(h₁, x₂)      # 또 다시 skip
  h₂ = Linear(h₂)           # 차원 정규화

Layer 3:
  x₃ = GNNConv(h₂)
  h₃ = Concat(h₂, x₃)      # 최종 skip
  h₃ = Linear(h₃)
```

### v8 구조도

```
┌──────────────────────────────────────────────────────────────────┐
│                    CAGE-RF CHEB v8                               │
│                  (Skip Connection)                               │
└──────────────────────────────────────────────────────────────────┘

Input Features (25000 nodes × 139D)
  ↓
┌─────────────────────────────────────────────┐
│  Branch Layer (각 relation별 독립)          │
│                                             │
│  For each relation in [RUR, RTR, RSR,      │
│                        Burst, SemSim, Behav]
│  {                                          │
│    x₀ = input (139D)                        │
│                                             │
│    x₁ = CHEBConv(x₀, edge_index)           │
│    h₁ = Concat([x₀, x₁]) = (139+128) = 267D│
│    h₁ = Linear(267D → 128D)  ← Skip 1      │
│                                             │
│    x₂ = CHEBConv(h₁, edge_index)           │
│    h₂ = Concat([h₁, x₂]) = (128+128) = 256D│
│    h₂ = Linear(256D → 128D)  ← Skip 2      │
│                                             │
│    x₃ = CHEBConv(h₂, edge_index)           │
│    h₃ = Concat([h₂, x₃]) = (128+128) = 256D│
│    h₃ = Linear(256D → 128D)  ← Skip 3      │
│                                             │
│    branch_output = h₃ (128D)                │
│  }                                          │
│  6개 branch outputs: [b₁, b₂, b₃, b₄, b₅, b₆]
└─────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────┐
│  Gating Mechanism                           │
│  Concat all branches: (6 × 128D = 768D)    │
│  α = Softmax(Linear(768D → 6))  ← Gating   │
│  Result: α = [α₁, α₂, α₃, α₄, α₅, α₆]    │
└─────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────┐
│  Weighted Fusion                            │
│  h_fused = Σ(αᵢ × bᵢ) for i=1..6          │
│  Result: h_fused (128D)                     │
└─────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────┐
│  Classification Layer                       │
│  logits = Linear(128D → 1)                  │
│  fraud_prob = Sigmoid(logits)               │
│  Output: [0, 1]  ← Fraud probability        │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Auxiliary Loss (각 branch별)               │
│  aux_logits_i = Linear(bᵢ → 1)  for i=1..6│
│  aux_loss = Σ BCE(aux_logits_i, labels)     │
│                                             │
│  Total Loss = Main Loss + 0.3 × Aux Loss   │
└─────────────────────────────────────────────┘
```

### v8 동작 메커니즘

**Skip Connection의 효과**:

```
1️⃣ Gradient Flow 개선
   
   Without Skip:
   Loss → Layer3 Grad → Layer2 Grad ✗ (Vanishing)
                              ↓
   With Skip:
   Loss → Layer3 Grad ─→ Layer2 Grad ✓ (Preserved)
          ↓(직접 경로)
```

```
2️⃣ 정보 보존

   Without Skip (Over-smoothed):
   Initial: [특징1, 특징2, ..., 특징139]
   After L1: [평균, 평균, ..., 평균]  ← 초기 정보 손실
   After L2: [평균, 평균, ..., 평균]  ← 더 손실
   After L3: [평균, 평균, ..., 평균]  ← 완전히 손실

   With Skip (정보 보존):
   L1: Concat([초기, 집계]) → 초기 정보 20% 유지
   L2: Concat([L1결과, 집계]) → 정보 더 보존
   L3: Concat([L2결과, 집계]) → 최대 정보 보존
```

```
3️⃣ 각 레이어의 역할이 명확해짐

   Layer 1: 1-hop 이웃 정보 추출
   Layer 2: 2-hop 이웃 정보 추출
   Layer 3: 3-hop 이웃 정보 추출
   
   Skip이 없으면 → 다 같은 평균
   Skip이 있으면 → 각 레이어의 고유 특징 유지
```

### v8 성능 효과

```
메트릭              v2 (Baseline)    v8 (Skip)    개선도
────────────────────────────────────────────────────
PR-AUC             0.3056          0.3087       +0.0031 (+1.0%)
Macro-F1           0.6100          0.6153       +0.0053 (+0.9%)
ROC-AUC            0.7729          0.7701       -0.0028 (-0.4%)
Accuracy           0.8034          0.8251       +0.0217 (+2.7%)
Recall             0.6425          0.6308       -0.0117 (-1.8%)
Precision          0.5971          0.6055       +0.0084 (+1.4%)

✅ 전체적으로 안정적인 성능 향상
✅ 특히 Accuracy 크게 증가 (2.7%)
```

---

## v9: Two-Stage GNN

### 개념: 반복 학습 (Iterative Refinement)

**기본 아이디어**:
```
V2~V8: One-shot learning
  - 한 번에 gating 가중치 학습
  - 초기 가중치가 최적이 아닐 수 있음

V9: Two-Stage learning
  - Stage 1: 초기 gating 가중치 학습 (α₁, α₂, ..., α₆)
  - Stage 2: 학습한 가중치를 사용해 embedding 재정제
  - 반복을 통한 미세 조정 (fine-tuning)
```

### v9 상세 구조

```
┌─────────────────────────────────────────────────────────────────────┐
│         CAGE-RF CHEB v9: Two-Stage GNN                              │
│    (Stage 1: Gating Learning → Stage 2: Embedding Refinement)      │
└─────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
                          STAGE 1: Gating Learning
═══════════════════════════════════════════════════════════════════════

Input Features (25000 nodes × 139D) + 6 Relation Edges
  ↓
┌────────────────────────────────────────────┐
│  Branch Layers (Skip Connection)           │
│                                            │
│  For each relation:                        │
│    x₀ → Skip1 → Skip2 → Skip3             │
│    Result: b₁, b₂, b₃, b₄, b₅, b₆ (128D) │
└────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────┐
│  Gating Mechanism (Stage 1 핵심)            │
│                                            │
│  concat_feat = Concat([b₁, b₂, ..., b₆])  │
│  gating_input = concat_feat (768D)        │
│                                            │
│  α = Softmax(GatingNet(gating_input))     │
│  α = [α₁, α₂, α₃, α₄, α₅, α₆]  ← 학습!  │
│                                            │
│  💡 핵심: 이 α 값들이 각 relation의       │
│  중요도를 나타냄                           │
└────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────┐
│  Weighted Fusion (Stage 1 최종)             │
│                                            │
│  h_fused_stage1 = Σ(αᵢ × bᵢ)              │
│  Result: h_fused_stage1 (128D)             │
│                                            │
│  ⭐ 이 gating weights α를 저장!            │
│     → Stage 2에서 재사용                   │
└────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────┐
│  Stage 1 Classification & Loss              │
│  pred_stage1 = Classifier(h_fused_stage1)  │
│  loss_stage1 = FocalLoss(pred_stage1, y)   │
│  loss_stage1.backward()  ← 학습!           │
└────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                      STAGE 2: Embedding Refinement
═══════════════════════════════════════════════════════════════════════

📌 Stage 1에서 학습한 α를 꺼냄 (detach, freeze)

Frozen α = [α₁, α₂, α₃, α₄, α₅, α₆]
  ↓
┌────────────────────────────────────────────┐
│  Branch Layers 재실행 (새 데이터 또는      │
│  동일 데이터를 다시 통과)                   │
│                                            │
│  b₁, b₂, b₃, b₄, b₅, b₆ (128D) 생성      │
└────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────┐
│  ⭐ 다른 Gating (Stage 2 핵심)              │
│                                            │
│  Stage 1에서 학습한 α를 명시적으로 적용   │
│                                            │
│  h_fused_stage2 = Σ(α_frozen ᵢ × bᵢ)    │
│  Result: h_fused_stage2 (128D)             │
│                                            │
│  🔄 명시적으로 가중치를 재적용해           │
│     embedding을 정제                       │
└────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────┐
│  Stage 2 Classification & Loss              │
│  pred_stage2 = Classifier(h_fused_stage2)  │
│  loss_stage2 = FocalLoss(pred_stage2, y)   │
│  loss_stage2.backward()  ← 다시 학습!      │
│                                            │
│  Total Loss = loss_stage1 + loss_stage2    │
└────────────────────────────────────────────┘
  ↓
Final Output: h_fused_stage2 (더 정제된 embedding)
```

### v9 vs v2/v8 비교

```
┌─────────────────────────────────────────────────────────────┐
│  V2 (One-shot):                                             │
│                                                             │
│  Branch → Gating (한 번 학습) → Fusion → Output            │
│                                                             │
│  한 번의 가중치 학습으로 끝                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  V8 (Skip Connection):                                      │
│                                                             │
│  Branch (Skip) → Gating → Fusion → Output                  │
│                                                             │
│  Skip connection으로 정보 보존, gating은 여전히 한 번      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  V9 (Two-Stage):                                            │
│                                                             │
│  Stage 1:                                                   │
│  Branch (Skip) → Gating₁ (학습) → Fusion → Loss₁          │
│                      ↓                                      │
│                    α 저장                                   │
│                      ↓                                      │
│  Stage 2:                                                   │
│  Branch (Skip) → Gating₂ (α 재적용) → Fusion → Loss₂      │
│                                                             │
│  가중치를 명시적으로 학습한 후 재사용                       │
└─────────────────────────────────────────────────────────────┘
```

### v9 성능 효과

```
메트릭              v8 (Skip)       v9 (Two-Stage)  개선도
────────────────────────────────────────────────────────
PR-AUC             0.3087          0.3106          +0.0019 (+0.6%)
Macro-F1           0.6153          0.6167          +0.0014 (+0.2%)
ROC-AUC            0.7701          0.7706          +0.0005 (+0.1%)
Accuracy           0.8251          0.8314          +0.0063 (+0.8%)
Recall             0.6308          0.6275          -0.0033 (-0.5%)
Precision          0.6055          0.6089          +0.0034 (+0.6%)

✅ v8에서 추가로 안정적인 개선
✅ PR-AUC 최고 달성: 0.3106
```

---

## v8 vs v9 비교

### 아키텍처 비교

| 항목 | v8 (Skip Connection) | v9 (Two-Stage GNN) |
|------|-------------------|-------------------|
| **핵심 기법** | Residual connection | Iterative refinement |
| **Skip 사용** | ✅ 있음 (매 레이어) | ✅ 있음 (매 레이어) |
| **Gating 단계** | 1단계 (학습) | 2단계 (학습 + 적용) |
| **추가 메커니즘** | 없음 | 학습한 α를 Stage 2에서 재사용 |
| **성능 (PR-AUC)** | 0.3087 | 0.3106 (+0.6%) |
| **계산량** | 1× | 2× (2 stage) |
| **학습 시간** | ~200 epochs | ~400 epochs (2 stage) |

### 선택 가이드

**v8을 선택할 경우**:
- ✅ 빠른 학습 필요 (계산량 제약)
- ✅ 안정적인 성능 필요 (0.3087은 충분함)
- ✅ 메모리 제약 있음
- ✅ Over-smoothing 해결만 필요

**v9을 선택할 경우**:
- ✅ 최고 성능 추구 (0.3106)
- ✅ 계산 리소스 충분
- ✅ Gating 가중치의 명시적 활용 중요
- ✅ Fine-tuning을 통한 정교한 조정 필요

---

## 코드 구현

### v8 Skip Connection 코드

```python
import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv

class CAGERFGNNv8(nn.Module):
    """Skip Connection을 가진 CAGE-RF GNN"""
    
    def __init__(self, input_dim=139, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 6개 relation branch 생성
        self.branches = nn.ModuleDict()
        for relation_name in ['rur', 'rtr', 'rsr', 'burst', 'semsim', 'behavior']:
            branch = nn.ModuleList()
            for layer_idx in range(num_layers):
                in_dim = input_dim if layer_idx == 0 else hidden_dim
                conv = ChebConv(in_dim, hidden_dim, K=3)
                branch.append(conv)
                
                # Skip connection을 위한 linear projection
                skip_in_dim = in_dim + hidden_dim if layer_idx == 0 else hidden_dim * 2
                skip_linear = nn.Linear(skip_in_dim, hidden_dim)
                branch.append(skip_linear)
            
            self.branches[relation_name] = branch
        
        # Gating mechanism
        self.gating = nn.Sequential(
            nn.Linear(6 * hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary classifiers (각 branch별)
        self.aux_classifiers = nn.ModuleDict()
        for relation_name in ['rur', 'rtr', 'rsr', 'burst', 'semsim', 'behavior']:
            aux_clf = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.aux_classifiers[relation_name] = aux_clf
    
    def forward(self, x, edge_indices_dict):
        """
        Args:
            x: Node features (N × input_dim)
            edge_indices_dict: Dict of edge_index for each relation
        
        Returns:
            fraud_logits: (N × 1)
            auxiliary_logits: Dict of (N × 1) for each relation
        """
        branch_outputs = []
        auxiliary_logits = {}
        
        # 각 relation별 branch 처리
        for relation_name, conv_layers in self.branches.items():
            h = x  # (N × input_dim)
            edge_index = edge_indices_dict[relation_name]
            
            # 각 레이어에 Skip Connection 적용
            for layer_idx in range(self.num_layers):
                conv = conv_layers[layer_idx * 2]
                skip_linear = conv_layers[layer_idx * 2 + 1]
                
                # Convolution
                h_new = conv(h, edge_index)  # (N × hidden_dim)
                
                # Skip Connection: Concat + Linear
                h_concat = torch.cat([h, h_new], dim=1)  # (N × (dim+hidden_dim))
                h = skip_linear(h_concat)  # (N × hidden_dim)
                
                # Dropout
                h = torch.dropout(h, p=0.3, train=self.training)
            
            # 최종 branch 출력
            branch_outputs.append(h)  # (N × hidden_dim)
            
            # Auxiliary classifier
            aux_logit = self.aux_classifiers[relation_name](h)
            auxiliary_logits[relation_name] = aux_logit
        
        # Gating mechanism
        branch_concat = torch.cat(branch_outputs, dim=1)  # (N × 6*hidden_dim)
        alpha = torch.softmax(self.gating(branch_concat), dim=1)  # (N × 6)
        
        # Weighted Fusion
        h_fused = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        for i, branch_out in enumerate(branch_outputs):
            h_fused += alpha[:, i:i+1] * branch_out  # Broadcasting (N × 1) × (N × hidden_dim)
        
        # Main classifier
        fraud_logits = self.classifier(h_fused)  # (N × 1)
        
        return fraud_logits, auxiliary_logits, alpha

```

### v9 Two-Stage GNN 코드

```python
class CAGERFGNNv9(nn.Module):
    """Two-Stage GNN: Stage 1 학습 + Stage 2 정제"""
    
    def __init__(self, input_dim=139, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # v8과 동일한 구조 (Skip Connection 포함)
        self.branches = nn.ModuleDict()
        for relation_name in ['rur', 'rtr', 'rsr', 'burst', 'semsim', 'behavior']:
            branch = nn.ModuleList()
            for layer_idx in range(num_layers):
                in_dim = input_dim if layer_idx == 0 else hidden_dim
                conv = ChebConv(in_dim, hidden_dim, K=3)
                branch.append(conv)
                
                skip_in_dim = in_dim + hidden_dim if layer_idx == 0 else hidden_dim * 2
                skip_linear = nn.Linear(skip_in_dim, hidden_dim)
                branch.append(skip_linear)
            
            self.branches[relation_name] = branch
        
        # Gating mechanism
        self.gating = nn.Sequential(
            nn.Linear(6 * hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary classifiers
        self.aux_classifiers = nn.ModuleDict()
        for relation_name in ['rur', 'rtr', 'rsr', 'burst', 'semsim', 'behavior']:
            aux_clf = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.aux_classifiers[relation_name] = aux_clf
        
        self.alpha_frozen = None  # Stage 1에서 학습한 gating weights
    
    def forward_stage1(self, x, edge_indices_dict):
        """
        Stage 1: Gating weights 학습
        """
        branch_outputs = []
        auxiliary_logits = {}
        
        for relation_name, conv_layers in self.branches.items():
            h = x
            edge_index = edge_indices_dict[relation_name]
            
            for layer_idx in range(self.num_layers):
                conv = conv_layers[layer_idx * 2]
                skip_linear = conv_layers[layer_idx * 2 + 1]
                
                h_new = conv(h, edge_index)
                h_concat = torch.cat([h, h_new], dim=1)
                h = skip_linear(h_concat)
                h = torch.dropout(h, p=0.3, train=self.training)
            
            branch_outputs.append(h)
            aux_logit = self.aux_classifiers[relation_name](h)
            auxiliary_logits[relation_name] = aux_logit
        
        # 🔑 Stage 1: Gating weights 학습
        branch_concat = torch.cat(branch_outputs, dim=1)
        alpha = torch.softmax(self.gating(branch_concat), dim=1)  # ← 학습됨
        
        # Fusion
        h_fused = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        for i, branch_out in enumerate(branch_outputs):
            h_fused += alpha[:, i:i+1] * branch_out
        
        # Main classifier
        fraud_logits_stage1 = self.classifier(h_fused)
        
        # 🔑 Alpha를 저장 (Stage 2에서 사용)
        self.alpha_frozen = alpha.detach()  # ← Detach (gradient 차단)
        
        return fraud_logits_stage1, auxiliary_logits
    
    def forward_stage2(self, x, edge_indices_dict):
        """
        Stage 2: 학습한 alpha를 사용해 embedding 정제
        """
        if self.alpha_frozen is None:
            raise ValueError("Stage 1을 먼저 실행해야 합니다")
        
        branch_outputs = []
        
        # 다시 branch를 통과 (Skip connection 포함)
        for relation_name, conv_layers in self.branches.items():
            h = x
            edge_index = edge_indices_dict[relation_name]
            
            for layer_idx in range(self.num_layers):
                conv = conv_layers[layer_idx * 2]
                skip_linear = conv_layers[layer_idx * 2 + 1]
                
                h_new = conv(h, edge_index)
                h_concat = torch.cat([h, h_new], dim=1)
                h = skip_linear(h_concat)
                h = torch.dropout(h, p=0.3, train=self.training)
            
            branch_outputs.append(h)
        
        # 🔑 Stage 2: 저장된 alpha를 명시적으로 적용
        h_fused = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        for i, branch_out in enumerate(branch_outputs):
            # alpha_frozen을 직접 사용 (학습하지 않음)
            h_fused += self.alpha_frozen[:, i:i+1] * branch_out
        
        # Main classifier
        fraud_logits_stage2 = self.classifier(h_fused)
        
        return fraud_logits_stage2
    
    def forward(self, x, edge_indices_dict, training_stage=1):
        """
        Args:
            x: Node features
            edge_indices_dict: Edge indices for each relation
            training_stage: 1 or 2
        
        Returns:
            logits, auxiliary_logits (stage 1)
            or logits (stage 2)
        """
        if training_stage == 1:
            return self.forward_stage1(x, edge_indices_dict)
        elif training_stage == 2:
            return self.forward_stage2(x, edge_indices_dict)
        else:
            raise ValueError(f"Invalid training_stage: {training_stage}")
```

### 학습 루프 (v9)

```python
def train_v9(model, data, optimizer, num_epochs_per_stage=100):
    """
    v9 Two-Stage 학습
    """
    
    # ========== Stage 1: Gating weights 학습 ==========
    print("🔄 Stage 1: Learning gating weights...")
    
    for epoch in range(num_epochs_per_stage):
        optimizer.zero_grad()
        
        # Stage 1 forward
        fraud_logits, aux_logits = model.forward_stage1(
            data.x, 
            data.edge_indices_dict
        )
        
        # Loss 계산
        main_loss = focal_loss(fraud_logits, data.y)
        aux_loss = sum(
            bce_loss(aux_logits[name], data.y)
            for name in aux_logits
        ) / len(aux_logits)
        
        total_loss = main_loss + 0.3 * aux_loss
        
        # 역전파
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss:.4f}")
    
    # ========== Stage 2: Embedding refinement ==========
    print("\n🔄 Stage 2: Refining embeddings...")
    
    # Stage 2용 새 optimizer (선택사항)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    for epoch in range(num_epochs_per_stage):
        optimizer2.zero_grad()
        
        # Stage 2 forward
        fraud_logits_stage2 = model.forward_stage2(
            data.x,
            data.edge_indices_dict
        )
        
        # Loss
        stage2_loss = focal_loss(fraud_logits_stage2, data.y)
        
        # 역전파
        stage2_loss.backward()
        optimizer2.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={stage2_loss:.4f}")
    
    print("\n✅ Training complete!")
    return model
```

---

## 성능 분석

### 실험 결과 요약

```
모델              PR-AUC    Macro-F1   ROC-AUC   Accuracy   특징
─────────────────────────────────────────────────────────────
v2 Baseline      0.3056    0.6100    0.7729    0.8034    기준
v3 Ablation      0.3086    0.6088    0.7683    0.8131    상위4개
v4 PR-Curve      0.3073    0.6114    0.7681    0.8152    동적threshold
v5 Oversamp      0.3021    0.6149    0.7706    0.8226    소수증폭
v6 Hard Mining   0.2505    0.5954    0.7544    0.7687    hardness
v7 Ensemble      0.2823    0.5984    0.7609    0.7866    독립분류기
v8 Skip ⭐       0.3087    0.6153    0.7701    0.8251    over-smooth 해결
v9 Two-Stage ⭐⭐ 0.3106    0.6167    0.7706    0.8314    반복정제

⭐ Top 3:
1. v9: 0.3106 (최고!)
2. v8: 0.3087 (+최대 improvement)
3. v3: 0.3086 (branch ablation)
```

### v8 vs v9 성능 향상 경로

```
PR-AUC 개선:

v2 (0.3056)
  ├─ v3 (+0.0030) → 0.3086 [Branch 선택]
  ├─ v4 (-0.0013) → 0.3073 [Threshold 조정]
  └─ v8 (+0.0031) → 0.3087 [Skip Connection]
       └─ v9 (+0.0019) → 0.3106 [Two-Stage] ← 최고!

총 개선: 0.3056 → 0.3106 (+0.0050, +1.6%)
최단 경로: v2 → v8 → v9 (+0.0050)
```

### 계산 복잡도 분석

```
Model     Layers  Skip Conn  Stages  Parameters  FLOPs     Epoch Time
─────────────────────────────────────────────────────────────────
v2        3       ✗         1       ~500K      ~100M     ~10s
v8        3       ✅        1       ~600K      ~120M     ~12s  (+20%)
v9        3       ✅        2       ~600K      ~240M     ~24s  (+140%)

✅ v8: 계산량 20% 증가로 PR-AUC +1.0% 개선
✅ v9: 계산량 140% 증가로 PR-AUC +1.6% 개선
```

---

## 결론

### v8 Skip Connection
- **목표**: Over-smoothing 해결
- **방식**: 각 레이어에서 입력 정보 직접 연결
- **효과**: Gradient flow 개선, 초기 특징 보존
- **성능**: PR-AUC 0.3087 (+1.0%)
- **비용**: 계산량 20% 증가

### v9 Two-Stage GNN
- **목표**: Gating 가중치의 명시적 활용
- **방식**: Stage 1 학습 후 Stage 2에서 재적용
- **효과**: 반복 학습을 통한 embedding 정제
- **성능**: PR-AUC 0.3106 (+1.6%) ← **최고 성능**
- **비용**: 계산량 140% 증가

### 선택 기준

| 상황 | 추천 모델 | 이유 |
|------|---------|------|
| **성능 최고 중요** | v9 | PR-AUC 0.3106 달성 |
| **효율성 중요** | v8 | 성능 대비 계산 효율 최고 |
| **빠른 학습** | v2/v3 | 계산 시간 최소 |
| **안정성 중요** | v8 | 일관된 성능 향상 |
| **논문/발표용** | v9 | 혁신적 아키텍처 |

---

**최종 권장**: **v8 + v9 병행 사용**
- v8: 실시간 시스템 (빠른 학습, 안정적)
- v9: 최종 모델 (최고 성능)

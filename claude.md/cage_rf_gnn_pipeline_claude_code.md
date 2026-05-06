# Claude Code용 CAGE-RF GNN 파이프라인 구현 명세서

## Camouflage-Aware Gated Relation-Fusion GNN for YelpZip Review Fraud Detection

> 목적: 이 문서는 YelpZip 기반 조직적 어뷰징 네트워크 탐지 공모전에서 수상 가능성을 높이기 위한 GNN 파이프라인을 구현 지시서 형태로 정리한 것이다.  
> 핵심 방향: 리뷰를 노드로 고정하고, 여러 relation별 GNN embedding을 생성한 뒤 concat + gating으로 융합하여 사기 리뷰 점수를 예측한다.  
> 대시보드에서는 모델 결과를 단순 수치가 아니라 **왜 해당 리뷰가 의심되는지** 설명 가능한 형태로 보여준다.

---

# 0. 제안 모델명

## CAGE-RF GNN

**Camouflage-Aware Gated Relation-Fusion GNN**

한글명:

```text
위장 관계를 고려한 게이트 기반 관계 융합 GNN
```

프로젝트명:

```text
관계 융합형 GNN 기반 조직적 사기 리뷰 네트워크 탐지 대시보드
```

---

# 1. 핵심 아이디어

기존 텍스트 분류 모델은 리뷰 하나의 문장만 보고 사기 여부를 판단한다.

하지만 사기 리뷰는 대개 다음과 같은 조직적 패턴으로 나타난다.

```text
같은 사용자가 여러 리뷰 작성
특정 상품에 짧은 기간 동안 리뷰 집중
같은 별점이 비정상적으로 반복
서로 다른 계정이 비슷한 문장을 작성
비슷한 행동 패턴을 보이는 계정들이 분산되어 활동
```

따라서 본 모델은 리뷰를 노드로 보고, 리뷰 간 관계를 여러 relation으로 나누어 그래프를 만든다.

```text
Review Node
+
Multi-Relation Edges
+
Relation-specific GNN Branches
+
Embedding Concat
+
Relation Gating
=
Fraud Score
```

---

# 2. 전체 파이프라인

```text
YelpZip Original Dataset
        ↓
Label Conversion
        ↓
Product-User-Time Hybrid Dense Sampling
        ↓
Train / Valid / Test Split
        ↓
Node Feature Engineering
        ↓
Multi-Relation Graph Construction
        ↓
Camouflage-Aware Neighbor Filtering
        ↓
Relation-Specific GNN Branches
        ↓
Embedding Concat
        ↓
Relation Gating / Attention Fusion
        ↓
Projection Layer
        ↓
Classifier Head
        ↓
Fraud Score Prediction
        ↓
Evaluation
        ↓
Explainable Dashboard
```

---

# 3. 데이터 입력

## 3.1 사용 데이터

메인 데이터는 YelpZip 원본 데이터셋이다.

필수 컬럼:

```text
review_id
user_id
prod_id
text
date
rating
label
```

`review_id`가 없다면 index를 기반으로 생성한다.

```python
df["review_id"] = np.arange(len(df))
```

---

# 4. 라벨 변환

원본 YelpZip 라벨은 다음과 같다.

```text
-1 = 사기 리뷰
 1 = 정상 리뷰
```

모델 학습용 라벨은 다음과 같이 변환한다.

```text
-1 → 1
 1 → 0
```

구현:

```python
df["label"] = df["label"].map({-1: 1, 1: 0})
```

라벨 변환 후 반드시 검증한다.

```python
assert set(df["label"].unique()).issubset({0, 1})
```

---

# 5. 샘플링 전략

## 5.1 무작위 샘플링 금지

단순 무작위 샘플링은 사용하지 않는다.  
노드 간 연결성을 파괴하여 GNN 학습에 불리하기 때문이다.

## 5.2 Product-User-Time Hybrid Dense Sampling

본 프로젝트에서는 다음 세 기준을 결합한 샘플링을 사용한다.

```text
1. 리뷰가 많은 상위 Product 중심 추출
2. 활동량이 높은 User 중심 추출
3. 특정 기간에 리뷰가 집중된 Time-window 중심 추출
```

## 5.3 구체 절차

```text
1. prod_id별 review_count 계산
2. 상위 product 후보 추출
3. user_id별 activity_count 계산
4. 상위 active user 후보 추출
5. 월 단위 date bucket 생성
6. 리뷰가 집중된 상위 month 후보 추출
7. product 조건, user 조건, time 조건의 union으로 후보 리뷰 추출
8. relation 생성 후 largest connected component 또는 dense component 추출
9. 최종 노드 수 10k~50k 사이로 조정
```

권장 CPU 설정:

```text
target_nodes = 25000
min_nodes = 10000
max_nodes = 50000
random_state = 42
```

---

# 6. 데이터 분할

반드시 샘플링이 완료된 서브그래프 내부에서 분할한다.

추천 분할:

```text
train: 64%
valid: 16%
test: 20%
```

구현:

```python
train_idx, temp_idx = train_test_split(
    node_indices,
    test_size=0.36,
    stratify=labels,
    random_state=42
)

valid_idx, test_idx = train_test_split(
    temp_idx,
    test_size=20/36,
    stratify=labels[temp_idx],
    random_state=42
)
```

마스크 생성:

```python
data.train_mask
data.val_mask
data.test_mask
```

---

# 7. Node Feature Engineering

## 7.1 기본 feature

각 리뷰 노드에 아래 feature를 부여한다.

```text
text_embedding
rating
date/time feature
user activity feature
product activity feature
product burst feature
review length
tag feature, if available
```

---

## 7.2 텍스트 임베딩

CPU 환경을 고려해 1차 구현은 다음을 사용한다.

```text
TF-IDF + TruncatedSVD 128차원
```

구현:

```python
vectorizer = TfidfVectorizer(
    max_features=50000,
    min_df=3,
    max_df=0.9,
    ngram_range=(1, 2)
)

tfidf = vectorizer.fit_transform(df["text"].fillna(""))
svd = TruncatedSVD(n_components=128, random_state=42)
text_emb = svd.fit_transform(tfidf)
```

고도화 실험으로 다음을 추가할 수 있다.

```text
sentence-transformers/all-MiniLM-L6-v2
```

단, GNN 학습 중에 BERT를 같이 돌리지 않는다.  
텍스트 임베딩은 반드시 사전 계산 후 저장한다.

---

## 7.3 정형 feature

추가 feature 예시:

```text
rating_norm
review_length_log
user_review_count_log
product_review_count_log
user_avg_rating
user_rating_std
product_avg_rating
product_rating_std
days_since_first_review
month_sin
month_cos
burst_score
```

정규화:

```python
StandardScaler()
```

최종 노드 feature:

```python
x = concat([
    text_embedding,
    numeric_features
])
```

---

# 8. Multi-Relation Graph Construction

대회 규정상 기본 relation 1개 이상과 커스텀 relation 1개 이상이 필요하다.  
본 모델은 기본 relation 3개와 커스텀 relation 3개를 사용한다.

---

# 8.1 기본 Relation

## R-U-R: Same User Relation

같은 사용자가 작성한 리뷰끼리 연결한다.

```text
Review_i.user_id == Review_j.user_id
```

의미:

```text
동일 사용자의 반복 작성 패턴 탐지
```

Edge 제한:

```text
각 리뷰당 같은 user의 다른 리뷰 top-k 연결
k = 10
```

---

## R-T-R: Same Time Window Relation

같은 상품에 대해 같은 기간에 작성된 리뷰끼리 연결한다.

```text
Review_i.prod_id == Review_j.prod_id
and same month
```

의미:

```text
특정 상품/식당에 같은 기간 리뷰가 몰리는 패턴 탐지
```

Edge 제한:

```text
같은 prod_id + same month 내부에서 날짜가 가까운 top-k 연결
k = 10
```

---

## R-S-R: Same Rating Relation

같은 상품에 대해 같은 별점을 준 리뷰끼리 연결한다.

```text
Review_i.prod_id == Review_j.prod_id
and Review_i.rating == Review_j.rating
```

의미:

```text
특정 별점으로 평판을 조작하는 패턴 탐지
```

Edge 제한:

```text
같은 prod_id + rating 그룹 내부에서 top-k 연결
k = 10
```

---

# 8.2 커스텀 Relation

## R-Burst-R: Burst Review Pattern Relation

같은 상품에 대해 짧은 기간 안에 비슷한 별점의 리뷰가 몰린 경우 연결한다.

```text
same prod_id
and date difference <= 7 days
and rating difference <= 1
```

또는 강한 조건:

```text
same prod_id
and date difference <= 3 days
and same rating
```

의미:

```text
단기간 집중적으로 발생한 리뷰 조작 캠페인 탐지
```

Edge 제한:

```text
각 노드당 시간적으로 가까운 top-k 연결
k = 10
```

---

## R-SemSim-R: Semantic Similarity Relation

리뷰 텍스트 의미가 유사한 리뷰끼리 연결한다.

```text
cosine_similarity(text_embedding_i, text_embedding_j) >= threshold
```

추천 설정:

```text
같은 prod_id 내부에서만 비교
top-k nearest neighbors
k = 5
```

의미:

```text
서로 다른 계정이 유사한 문장 구조나 의미를 반복하는 패턴 탐지
```

---

## R-UserBehavior-R: User Behavior Similarity Relation

서로 다른 사용자가 작성했지만 행동 패턴이 유사한 리뷰끼리 연결한다.

사용자 행동 feature:

```text
user_review_count
user_avg_rating
user_rating_std
user_active_days
user_burstiness
user_product_diversity
```

연결 조건:

```text
cosine_similarity(user_behavior_i, user_behavior_j) >= threshold
```

또는:

```text
top-k behavior similar neighbors
k = 5
```

의미:

```text
계정은 다르지만 비슷한 방식으로 움직이는 작업장 패턴 탐지
```

---

# 9. Edge 생성 시 CPU 최적화

절대 dense adjacency matrix를 만들지 않는다.

금지:

```python
adj = np.zeros((N, N))
```

사용:

```python
edge_index = torch.tensor([src, dst], dtype=torch.long)
```

각 relation별 edge_index를 따로 저장한다.

```python
edge_index_dict = {
    "rur": edge_index_rur,
    "rtr": edge_index_rtr,
    "rsr": edge_index_rsr,
    "burst": edge_index_burst,
    "semsim": edge_index_semsim,
    "behavior": edge_index_behavior,
}
```

각 relation은 undirected graph로 만든다.

```python
edge_index = to_undirected(edge_index)
```

---

# 10. Camouflage-Aware Neighbor Filtering

일반 GNN은 모든 이웃을 섞는다.  
하지만 사기 리뷰 탐지에서는 정상처럼 위장한 이웃이 섞이면 사기 신호가 희석될 수 있다.

따라서 relation별로 top-k 또는 threshold 기반 neighbor filtering을 사용한다.

## 10.1 Filtering 원칙

```text
R-U-R: 같은 사용자 리뷰가 너무 많으면 최신/가까운 리뷰 top-k
R-T-R: 같은 기간 리뷰가 너무 많으면 날짜 가까운 top-k
R-S-R: 같은 별점 리뷰가 너무 많으면 같은 상품 내 top-k
R-Burst-R: 시간 차이가 작은 리뷰 우선
R-SemSim-R: cosine similarity 상위 top-k
R-UserBehavior-R: behavior similarity 상위 top-k
```

## 10.2 기대 효과

```text
1. 불필요한 noisy edge 감소
2. 사기 신호 희석 완화
3. CPU 학습 효율 개선
4. 대시보드에서 설명 가능한 관계 제공
```

---

# 11. 모델 구조

## 11.1 전체 구조

```text
x, edge_index_dict
        ↓
Relation-specific GNN Branches
        ↓
h_RUR, h_RTR, h_RSR, h_Burst, h_SemSim, h_Behavior
        ↓
Embedding Concat
        ↓
Relation Gating / Attention Fusion
        ↓
Projection Layer
        ↓
Classifier Head
        ↓
Fraud Score
```

---

# 11.2 Relation-specific GNN Branch

각 relation마다 별도의 GNN branch를 둔다.

권장 branch:

```text
GraphSAGEConv 2층
```

이유:

```text
CPU에서 비교적 안정적
대규모 그래프에 적합
이웃 정보 aggregation에 적합
```

각 branch는 동일한 node feature x를 입력받되, relation별 edge_index를 다르게 받는다.

```python
h_rur = gnn_rur(x, edge_index_rur)
h_rtr = gnn_rtr(x, edge_index_rtr)
h_rsr = gnn_rsr(x, edge_index_rsr)
h_burst = gnn_burst(x, edge_index_burst)
h_semsim = gnn_semsim(x, edge_index_semsim)
h_behavior = gnn_behavior(x, edge_index_behavior)
```

---

# 11.3 Embedding Concat

각 relation branch의 output을 concat한다.

```python
h_cat = torch.cat([
    h_rur,
    h_rtr,
    h_rsr,
    h_burst,
    h_semsim,
    h_behavior
], dim=-1)
```

예:

```text
hidden_dim = 64
relation_count = 6
h_cat_dim = 64 * 6 = 384
```

---

# 11.4 Relation Gating

단순 concat 후 바로 classifier로 보내지 않고, relation 중요도를 학습한다.

## 방식 A: Relation Weight Gate

각 relation embedding에 대해 중요도 score를 계산한다.

```python
relation_stack = torch.stack([
    h_rur,
    h_rtr,
    h_rsr,
    h_burst,
    h_semsim,
    h_behavior
], dim=1)  # [N, R, H]

gate_logits = gate_mlp(relation_stack)  # [N, R, 1]
alpha = torch.softmax(gate_logits, dim=1)

h_fused = (alpha * relation_stack).sum(dim=1)
```

이 방식은 대시보드에서 relation contribution으로 활용하기 좋다.

## 방식 B: Concat + Projection

```python
h_cat = torch.cat([...], dim=-1)
h_proj = projection(h_cat)
```

CPU 구현 난이도를 낮추려면 B부터 구현하고, 이후 A를 추가한다.

추천:

```text
최종 제출 모델에는 Relation Weight Gate 포함
```

---

# 11.5 Classifier Head

```python
classifier = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_dim, 1)
)
```

출력:

```python
logit = classifier(h_fused).squeeze(-1)
fraud_score = torch.sigmoid(logit)
```

---

# 12. Loss Function

클래스 불균형을 고려한다.

## 12.1 Weighted BCE Loss

```python
pos_weight = num_negative / num_positive
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
```

## 12.2 Focal Loss 선택 가능

고도화 옵션:

```text
Focal Loss
```

다만 우선 구현은 Weighted BCE로 한다.

---

# 13. 평가 지표

대회 필수 지표:

```text
PR-AUC
Macro F1
```

추가 권장 지표:

```text
ROC-AUC
Precision@K
Recall@K
Confusion Matrix
Threshold별 Precision/Recall/F1
```

구현:

```python
average_precision_score(y_true, y_score)  # PR-AUC
f1_score(y_true, y_pred, average="macro")
```

Threshold는 validation set에서 Macro F1이 가장 높은 값을 선택한다.

```python
best_threshold = argmax_valid_macro_f1
```

test set 평가는 이 threshold를 사용한다.

---

# 14. Baseline 실험

수상 가능성을 높이기 위해 반드시 비교 실험을 포함한다.

## 14.1 Baseline 목록

```text
1. MLP only
2. GCN
3. GraphSAGE
4. GAT
5. R-GCN or Relation-aware GNN
6. Proposed CAGE-RF GNN
```

## 14.2 실험 목적

```text
MLP only:
관계 정보를 쓰지 않을 때의 한계 확인

GCN:
기본 GNN 성능 확인

GraphSAGE:
이웃 aggregation 기반 baseline

GAT:
attention 기반 이웃 중요도 학습 비교

R-GCN:
relation-aware 구조와 비교

CAGE-RF GNN:
relation 분리 + filtering + gating의 효과 확인
```

---

# 15. Ablation Study

모델의 창의성과 논리성을 보이기 위해 ablation을 수행한다.

## 15.1 Ablation 목록

```text
A. 기본 relation만 사용
B. 기본 relation + R-Burst-R
C. 기본 relation + R-Burst-R + R-SemSim-R
D. 모든 relation 사용, gating 없음
E. 모든 relation 사용, gating 있음
F. 모든 relation 사용, gating + neighbor filtering 있음
```

## 15.2 기대 설명

```text
R-Burst-R 추가 시 단기 집중 리뷰 패턴 탐지 개선
R-SemSim-R 추가 시 유사 문장 기반 작업장 리뷰 탐지 개선
Relation gating 추가 시 relation별 중요도 반영 가능
Neighbor filtering 추가 시 noisy edge와 camouflage 완화
```

---

# 16. 대시보드 설계

대시보드는 Streamlit을 추천한다.

## 16.1 Dashboard Page 1: Overview

표시 항목:

```text
전체 리뷰 수
샘플링된 리뷰 수
사기 라벨 비율
예측된 의심 리뷰 수
PR-AUC
Macro F1
Precision@K
Recall@K
```

---

## 16.2 Dashboard Page 2: Fraud Ranking

Fraud Score 상위 리뷰 목록.

컬럼:

```text
rank
review_id
fraud_score
true_label
pred_label
user_id
prod_id
rating
date
top_relation
```

---

## 16.3 Dashboard Page 3: Evidence Panel

선택한 리뷰에 대한 증거 패널.

표시:

```text
리뷰 원문 text
user_id
prod_id
rating
date
fraud_score
true_label
pred_label
연결된 의심 리뷰 Top 5
가장 유사한 리뷰 text 비교
```

---

## 16.4 Dashboard Page 4: Ego Network View

선택 리뷰 중심 ego-network를 보여준다.

노드 색상:

```text
파랑: 정상 예측
빨강: 사기 예측
노란색 테두리: 선택한 중심 리뷰
```

엣지 색상:

```text
R-U-R: blue
R-T-R: orange
R-S-R: green
R-Burst-R: red
R-SemSim-R: purple
R-UserBehavior-R: gray
```

---

## 16.5 Dashboard Page 5: Relation Contribution

Relation gating의 alpha 값을 사용한다.

```text
R-U-R contribution
R-T-R contribution
R-S-R contribution
R-Burst-R contribution
R-SemSim-R contribution
R-UserBehavior-R contribution
```

시각화:

```text
bar chart
radar chart
```

---

## 16.6 Dashboard Page 6: Threshold Simulator

사용자가 threshold를 조정하면 다음 값을 업데이트한다.

```text
탐지된 리뷰 수
Precision
Recall
F1
Confusion Matrix
```

---

# 17. 저장해야 할 결과 파일

학습 이후 대시보드에서 사용할 파일을 저장한다.

```text
outputs/
├── predictions.csv
├── metrics.json
├── node_embeddings.npy
├── relation_contributions.npy
├── sampled_reviews.csv
├── edge_rur.csv
├── edge_rtr.csv
├── edge_rsr.csv
├── edge_burst.csv
├── edge_semsim.csv
├── edge_behavior.csv
└── config.json
```

## predictions.csv 컬럼

```text
review_id
user_id
prod_id
date
rating
text
true_label
fraud_score
pred_label
top_relation
```

## metrics.json

```json
{
  "pr_auc": 0.0,
  "macro_f1": 0.0,
  "roc_auc": 0.0,
  "precision_at_k": 0.0,
  "recall_at_k": 0.0,
  "best_threshold": 0.0
}
```

---

# 18. 구현 폴더 구조

```text
gnn-fraud-detection/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   ├── preprocessing/
│   │   ├── load_yelpzip.py
│   │   ├── label_convert.py
│   │   ├── sampling.py
│   │   └── feature_engineering.py
│   ├── graph/
│   │   ├── build_relations.py
│   │   ├── build_rur.py
│   │   ├── build_rtr.py
│   │   ├── build_rsr.py
│   │   ├── build_burst.py
│   │   ├── build_semsim.py
│   │   └── build_behavior.py
│   ├── models/
│   │   ├── baseline_mlp.py
│   │   ├── baseline_gcn.py
│   │   ├── baseline_gat.py
│   │   ├── baseline_graphsage.py
│   │   ├── cage_rf_gnn.py
│   │   └── losses.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── threshold.py
│   ├── explain/
│   │   ├── relation_contribution.py
│   │   └── ego_network.py
│   └── utils/
│       ├── seed.py
│       ├── io.py
│       └── metrics.py
├── dashboard/
│   └── app.py
├── outputs/
├── reports/
└── notebooks/
```

---

# 19. 실행 순서

## 19.1 전처리

```bash
python -m src.preprocessing.load_yelpzip
python -m src.preprocessing.label_convert
python -m src.preprocessing.sampling
python -m src.preprocessing.feature_engineering
```

## 19.2 그래프 생성

```bash
python -m src.graph.build_relations
```

## 19.3 학습

```bash
python -m src.training.train --model cage_rf_gnn --config configs/default.yaml
```

## 19.4 평가

```bash
python -m src.training.evaluate --checkpoint outputs/best_model.pt
```

## 19.5 대시보드

```bash
streamlit run dashboard/app.py
```

---

# 20. Claude Code 구현 우선순위

## Phase 1. 데이터 로딩 및 라벨 변환

- YelpZip CSV 로드
- label 변환
- 기본 EDA 저장
- processed CSV 저장

## Phase 2. 샘플링

- Product-User-Time Hybrid Dense Sampling 구현
- 10k~50k 사이 노드 확보
- sampled_reviews.csv 저장

## Phase 3. Feature Engineering

- TF-IDF + SVD text embedding
- rating/date/user/product feature
- StandardScaler
- feature matrix 저장

## Phase 4. Relation Edge 생성

- R-U-R
- R-T-R
- R-S-R
- R-Burst-R
- R-SemSim-R
- R-UserBehavior-R
- 각 relation edge csv 저장

## Phase 5. Baseline 모델

- MLP
- GCN
- GraphSAGE
- GAT

## Phase 6. CAGE-RF GNN

- relation-specific branch
- concat
- gating
- classifier
- Weighted BCE

## Phase 7. 평가

- PR-AUC
- Macro F1
- Precision@K
- threshold tuning
- ablation study

## Phase 8. Dashboard

- overview
- fraud ranking
- evidence panel
- ego network
- relation contribution
- threshold simulator

## Phase 9. 보고서/발표 자료용 산출물

- 성능표
- ablation 표
- 파이프라인 그림
- relation 설계 설명
- 대시보드 캡처

---

# 21. 대회 규정과의 적합성 점검

## 21.1 적합한 부분

| 대회 조건 | 우리 설계 |
|---|---|
| YelpZip 원본 데이터셋 활용 | YelpZip 원본 데이터 사용 |
| 노드 리뷰 단위 고정 | Node = Review |
| 라벨 변환 필수 | -1→1, 1→0 적용 |
| 무작위 샘플링 금지 | Product-User-Time Hybrid Dense Sampling |
| 1만~5만 노드 서브그래프 | 25k 권장 |
| 샘플링 후 분할 | 샘플링 후 64/16/20 분할 |
| 기본 Relation 1개 이상 | R-U-R, R-T-R, R-S-R 모두 사용 |
| 커스텀 Relation 1개 이상 | R-Burst-R, R-SemSim-R, R-UserBehavior-R |
| GNN 핵심 사용 | Relation-specific GNN branch |
| PR-AUC 포함 | 평가 지표에 포함 |
| Macro F1 포함 | 평가 지표에 포함 |
| 대시보드 구현 | Streamlit 대시보드 |
| GNN 설명가능성 | Relation contribution, ego network, evidence panel |

---

## 21.2 어긋날 수 있는 위험 요소와 수정 방향

### 위험 1. Amazon 데이터셋을 메인으로 사용

문제:

```text
Amazon 참고 데이터는 사용자 노드 기반이라, 이번 대회의 리뷰 노드 고정 조건과 충돌할 수 있음
```

수정:

```text
Amazon은 참고 또는 확장 가능성으로만 언급한다.
메인 구현은 YelpZip 원본 리뷰 노드 기반으로 한다.
```

---

### 위험 2. 커스텀 Relation이 YelpChi relation과 너무 비슷함

문제:

```text
R-U-R, R-T-R, R-S-R만 사용하면 창의성이 약함
```

수정:

```text
R-Burst-R, R-SemSim-R, R-UserBehavior-R을 추가한다.
특히 R-Burst-R은 조직적 어뷰징 탐지와 직접 연결된다.
```

---

### 위험 3. 텍스트 임베딩 모델이 주인공처럼 보이는 문제

문제:

```text
BERT 기반 텍스트 분류처럼 보이면 GNN 핵심 조건이 약해짐
```

수정:

```text
텍스트 임베딩은 node feature 생성용 보조 정보로만 사용한다.
최종 예측은 multi-relation GNN이 수행한다.
```

---

### 위험 4. Edge 수 폭발

문제:

```text
같은 상품/같은 기간 리뷰를 모두 연결하면 edge가 지나치게 많아짐
CPU 학습이 어려워짐
```

수정:

```text
각 relation별 top-k 또는 threshold 기반 neighbor filtering을 적용한다.
edge_index sparse format을 사용한다.
```

---

### 위험 5. Train/test 분할 순서 오류

문제:

```text
전체 데이터에서 먼저 train/test를 나누고 샘플링하면 규정 위반
```

수정:

```text
반드시 서브그래프 샘플링 후 train/valid/test를 분할한다.
```

---

### 위험 6. Accuracy 중심 평가

문제:

```text
공지 필수 지표는 PR-AUC와 Macro F1이다.
Accuracy만 제시하면 감점 가능성 있음
```

수정:

```text
PR-AUC와 Macro F1을 메인 지표로 사용한다.
Accuracy는 보조 지표로만 둔다.
```

---

# 22. 최종 결론

이 파이프라인은 대회 규정과 잘 맞는다.

특히 강점은 다음과 같다.

```text
1. YelpZip 원본 데이터를 메인으로 사용
2. 리뷰 노드 고정 조건 충족
3. 기본 relation 3개와 커스텀 relation 3개 사용
4. GNN이 핵심인 relation-specific branch 구조
5. relation embedding concat과 gating으로 모델 창의성 확보
6. camouflage-aware filtering으로 GNN 단점 보완
7. PR-AUC와 Macro F1 중심 평가
8. 대시보드에서 relation contribution과 evidence panel 제공
```

따라서 이 설계는 규정을 어기지 않으면서도, 대학생 공모전 수준에서 충분히 어렵고 창의적이며 수상 가능성이 높은 방향이다.

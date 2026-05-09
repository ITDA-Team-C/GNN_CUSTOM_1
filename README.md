# 🔍 CAGE-RF: 조직적 어뷰징 네트워크 탐지 (GNN 기반)

**그래프신경망(GNN) 기반 조직적 어뷰징 네트워크 탐지 시스템**

- 📊 **[Live Dashboard](https://gnncustom1-faxbbqmcprwrdfciudb7vh.streamlit.app/)**
- 🏆 **팀**: ITDA Team C
- 📅 **마지막 업데이트**: 2026-05-09

---

## 🎯 프로젝트 개요

CAGE-RF는 YelpZip 데이터셋의 리뷰 네트워크에서 **조직적 어뷰징(Organized Abuse)을 탐지**하는 고급 GNN 기반 시스템입니다.

- **목표**: 사기성 리뷰 집단(organized review fraud)을 고정확도로 탐지
- **방식**: 다중 관계 그래프 + GNN 기반 분류
- **성능**: PR-AUC 0.30+ 달성 (기본 베이스라인 대비 개선)

---

## 🚀 주요 특징

### 1️⃣ CAGE-RF 모델 버전 (v2~v9)

| 버전 | 이름 | 주요 개선사항 |
|------|------|------------|
| **v2** | Default | 기본 CAGE-RF 아키텍처 |
| **v3** | Branch Ablation | 상위 4개 relation만 사용 (signal clarity) |
| **v4** | PR-Curve Threshold | PR 곡선 기반 임계값 최적화 |
| **v5** | Oversampling | 소수 클래스 2배 증폭 (recall 향상) |
| **v6** | Hard Mining | Hard negative/positive 샘플 집중 학습 |
| **v7** | Ensemble | Relation별 독립 classifier + learnable weights |
| **v8** | Skip Connection | Input을 layer output과 연결 (over-smoothing 해결) |
| **v9** | Two-Stage GNN | Stage 1에서 gating 신호 학습 후 embedding 정제 |

### 2️⃣ 7개 GNN 레이어 벤치마크

모든 v2~v9 버전이 다음 GNN 레이어들과 호환:
- **SAGE** (GraphSAGE): 샘플링 기반, 확장성 우수
- **GAT** (Graph Attention Network): 어텐션 메커니즘, 해석가능성 높음
- **GCN** (Graph Convolutional Network): 안정적이고 견고함
- **GraphConv** (Graph Convolution): 일반적인 그래프 합성곱
- **CHEB** (Chebyshev): 다항식 기반, 고차 정보 활용
- **TAG** (Topology Adaptive GCN): 위상 정보 적응
- **SG** (Simplifying GCNs): 단순화된 버전, 효율성 최고

**총 모델 수**: 56개 GNN 벤치마크 + 9개 Base Models = **65개**

### 3️⃣ 다중 Relation 지원

6가지 관계 유형으로 사기 패턴 포착:
- **R-U-R** (Review-User-Review): 같은 사용자의 반복적 행동
- **R-T-R** (Review-Time-Review): 시간적 집중도
- **R-S-R** (Review-Star-Review): 별점 조작 패턴
- **Burst**: 단시간 내 리뷰 폭증
- **SemanticSim**: 텍스트 유사도 기반
- **Behavior**: 행동 패턴 유사도

### 4️⃣ Focal Loss + Auxiliary Loss

- Focal Loss로 불균형 데이터 처리
- Auxiliary Loss로 relation 기여도 학습
- PR-AUC와 Macro-F1 동시 최적화

---

## 📊 대시보드 (9개 페이지)

### 🔗 [Live Dashboard](https://gnncustom1-faxbbqmcprwrdfciudb7vh.streamlit.app/)

| 페이지 | 설명 | 모델 수 |
|--------|------|--------|
| **1️⃣ 벤치마크 성능비교** | 순수 GNN Layers (v2 기본 설정만) | 7 |
| **2️⃣ v2~v7 + 벤치마크** | CAGE-RF 6개 버전 × 7 GNN 레이어 | 42 |
| **3️⃣ v8,v9 포함 전체** | 모든 GNN 벤치마크 + CAGE-RF v2~v9 + MLP | 65 |
| **4️⃣ 라벨 검증** | 데이터 품질 검증 (Label distribution, Stratification) | - |
| **5️⃣ 데이터 개요** | YelpZip 샘플링 결과 및 통계 | - |
| **6️⃣ Fraud Score 분석** | 최고 성능 모델의 점수 분포 + Threshold Simulator | - |
| **7️⃣ Relation 기여도** | 각 Relation이 판단에 미치는 영향도 분석 | - |
| **8️⃣ 네트워크 탐지** | 의심 사례 분석 + Ego network 시각화 | - |
| **9️⃣ 결과 & 인사이트** | 주요 발견, 실무 활용 방안, 개선 방향 | - |

---

## 📈 데이터 및 성능

### 데이터셋
- **소스**: YelpZip 리뷰 네트워크
- **샘플 수**: 25,000 노드 (동적 샘플링)
- **라벨 분포**: 불균형 13:1 (정상:사기)
- **분할**: Train 64%, Valid 16%, Test 20% (Stratification 적용)

### 주요 성능 지표
- **PR-AUC**: 0.30+ (베이스라인 기준 개선)
- **Macro-F1**: 높은 정확도와 재현율 동시 달성
- **모델 안정성**: 모든 GNN 레이어에서 일관된 성능

---

## 🛠️ 프로젝트 구조

```
ITDA_TEAM_C/
├── dashboard/
│   └── app.py                    # Streamlit 대시보드 (9페이지)
├── src/
│   ├── training/
│   │   ├── train.py             # 모델 학습 메인 스크립트
│   │   └── __pycache__/
│   ├── models/
│   │   └── cage_rf_gnn.py       # CAGE-RF GNN 모델
│   └── data/
│       └── graph_processor.py   # 그래프 처리
├── configs/
│   ├── default.yaml             # v2 기본 설정
│   ├── v3_ablation.yaml         # v3 설정
│   ├── v4_threshold_pr.yaml     # v4 설정
│   ├── v5_oversampling.yaml     # v5 설정
│   ├── v6_hard_mining.yaml      # v6 설정
│   ├── v7_ensemble.yaml         # v7 설정
│   ├── v8_skip.yaml             # v8 설정
│   └── v9_twostage.yaml         # v9 설정
├── scripts/
│   ├── run_all_models_complete.py   # 전체 모델 학습
│   ├── run_all_gnn_benchmarks.py    # GNN 벤치마크 학습
│   └── evaluation/
│       └── test_all.py          # 성능 비교 평가
├── data/
│   ├── raw/                     # 원본 데이터
│   ├── interim/                 # 전처리 중간 데이터
│   └── processed/               # 최종 처리 데이터
│       ├── node_samples.csv
│       ├── features.npy
│       └── metadata
├── outputs/
│   ├── cage_rf_gnn/             # Base 모델들 (v2~v9)
│   │   └── metrics_*.json
│   └── benchmark/               # GNN 벤치마크 (7 layers × 8 versions)
│       ├── SAGE/
│       ├── GAT/
│       ├── GCN/
│       ├── GRAPHCONV/
│       ├── CHEB/
│       ├── TAG/
│       └── SG/
├── requirements.txt             # 의존성 패키지
├── .gitignore
├── PROGRESS.md                  # 진행 상황 추적
├── BASELINE_FIXED_SETTINGS.md   # 기본 설정 문서
└── README.md                    # 이 파일
```

---

## 🚀 배포 (Streamlit Cloud)

### 현재 배포 상태

- **배포 URL**: [https://gnncustom1-faxbbqmcprwrdfciudb7vh.streamlit.app/](https://gnncustom1-faxbbqmcprwrdfciudb7vh.streamlit.app/)
- **배포 날짜**: 2026-05-09
- **상태**: ✅ Live

### 배포 내용

- ✅ 완전한 대시보드 (9개 페이지)
- ✅ 모든 모델 메트릭 (65개 모델)
- ✅ 처리된 데이터셋
- ✅ 실시간 상호작용 (Threshold Simulator 등)

---

## 💻 로컬 실행

### 설치

```bash
# 저장소 클론
git clone https://github.com/ITDA-Team-C/GNN_CUSTOM_1.git
cd ITDA_TEAM_C

# 의존성 설치
pip install -r requirements.txt
```

### 대시보드 실행

```bash
streamlit run dashboard/app.py
```

브라우저에서 `http://localhost:8501` 접속

### 모델 학습

```bash
# 단일 모델 학습
python src/training/train.py --config configs/default.yaml

# 모든 모델 학습
python scripts/run_all_models_complete.py

# GNN 벤치마크 학습
python scripts/run_all_gnn_benchmarks.py
```

### 성능 평가

```bash
python scripts/evaluation/test_all.py
```

CSV/TXT 형식의 성능 비교 결과 생성

---

## 📋 설정 파일 (YAML)

모든 설정은 `configs/` 디렉토리의 YAML 파일로 관리합니다.

---

## 🔄 진행 상황

### ✅ 완료된 작업

1. **v8 (Skip Connection) 구현**
   - Input을 각 layer output과 연결
   - Over-smoothing 문제 해결
   - 모든 7개 GNN 레이어 호환

2. **v9 (Two-Stage GNN) 구현**
   - Stage 1: Gating 신호로 relation weights 학습
   - Stage 2: Gating 신호로 embedding 정제 후 재학습
   - 추가 0.1~0.2% 성능 개선

3. **완전 벤치마크**
   - v2~v9 버전 × 7 GNN 레이어 = 56개 모델 학습
   - 모든 모델 메트릭 저장 (JSON)

4. **Streamlit 대시보드 (9페이지)**
   - Page 1: 순수 GNN v2 (7개 모델)
   - Page 2: CAGE-RF v2~v7 + 벤치마크 (42개 모델)
   - Page 3: 전체 v2~v9 + MLP (65개 모델)
   - Pages 4-9: 분석, 검증, 인사이트

5. **GitHub 저장소**
   - 모든 코드 및 설정 커밋
   - Data/outputs 포함 (총 185MB)
   - requirements.txt 정리 (호환성)

6. **Streamlit Cloud 배포**
   - 공개 대시보드 배포
   - URL: https://gnncustom1-faxbbqmcprwrdfciudb7vh.streamlit.app/

---

## 📊 성능 요약

### 모델 수
- **GNN Benchmarks**: 56개 (v2~v9 × 7 layers)
- **Base Models**: 8개 (CAGE-RF v2~v9)
- **Baselines**: 1개 (MLP)
- **총 65개**

### 평가 지표
- PR-AUC (Precision-Recall Area Under Curve)
- Macro-F1 (Imbalanced 데이터 최적화)
- ROC-AUC
- Accuracy, Precision, Recall

---

## 🎓 주요 기술

- **Graph Neural Networks (GNN)**: PyTorch Geometric
- **Deep Learning**: PyTorch 2.11.0
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly (Interactive)
- **Dashboard**: Streamlit 1.57.0
- **Configuration**: YAML

---

## 👥 팀 정보

- **팀명**: ITDA Team C
- **프로젝트**: 그래프신경망 기반 조직적 어뷰징 네트워크 탐지
- **대회**: 2026 AI/ML 경진대회

---

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

---

**마지막 수정**: 2026-05-09
**대시보드**: [https://gnncustom1-faxbbqmcprwrdfciudb7vh.streamlit.app/](https://gnncustom1-faxbbqmcprwrdfciudb7vh.streamlit.app/)


# 프로젝트 전체 구조 가이드

## 📁 루트 디렉토리 개요

```
ITDA_TEAM_C/
├── [기존 코드들...]                 # 원래 프로젝트 코드 (변경 없음)
├── gnn_yelpchi/                    # YelpChi 전용 독립 프로젝트
├── gnn_amazon/                     # Amazon 전용 독립 프로젝트
└── PROJECT_STRUCTURE.md            # 이 파일
```

---

## 🎯 두 프로젝트의 역할

### **gnn_yelpchi/** - 리뷰(Review) 레벨 분석
- **노드**: 각 리뷰 (25,000개 샘플)
- **태스크**: 스팸/봇 리뷰 탐지
- **엣지**: R-U-R, R-T-R, R-S-R (3가지)
- **상태**: 기존 코드 기반 (최적화됨)

### **gnn_amazon/** - 사용자(User) 레벨 분석
- **노드**: 각 사용자 (N명)
- **태스크**: 사기꾼 사용자 탐지
- **엣지**: U-P-U, U-S-U, U-V-U (3가지)
- **상태**: 새로 구현 (모듈식)

---

## 🚀 빠른 시작 가이드

### **YelpChi 학습**
```bash
cd gnn_yelpchi
python preprocess.py        # 전처리
python run_baselines.py     # 모델 학습
```

### **Amazon 학습**
```bash
cd gnn_amazon
python preprocess.py        # 전처리
python run_baselines.py     # 모델 학습
```

---

## 📊 데이터 흐름 비교

### **YelpChi Pipeline**
```
yelp_zip.csv
    ↓
[load_yelpzip.py] 로드
    ↓
[label_convert.py] -1→1, 1→0 변환
    ↓
[sampling.py] 25,000 리뷰 샘플링
    ↓
[feature_engineering.py] TF-IDF + SVD (128D)
    ↓
[build_rur/rtr/rsr.py] 3가지 엣지 구성
    ↓
outputs/output_yelpchi/
├── models/
├── results/
└── logs/
```

### **Amazon Pipeline**
```
amazon.csv (리뷰 단위)
    ↓
[load_amazon.py] 로드
    ↓
[label_convert.py] 라벨 확인
    ↓
[sampling_amazon.py] 사용자 샘플링
    ↓
[aggregate_user_reviews.py] 사용자별 집계 ⭐ (핵심!)
    ↓
[feature_engineering_amazon.py] TF-IDF + SVD (128D)
    ↓
[build_upu/usu/uvu.py] 3가지 엣지 구성
    ↓
outputs/output_amazon/
├── models/
├── results/
└── logs/
```

---

## 🔑 핵심 파일 설명

### **YelpChi**
| 파일 | 역할 |
|------|------|
| `preprocess.py` | 전체 전처리 오케스트레이션 |
| `load_yelpzip.py` | CSV 데이터 로드 |
| `sampling.py` | 25,000 리뷰 샘플링 |
| `feature_engineering.py` | 리뷰 텍스트 → 벡터 |
| `build_rur.py` | 같은 유저 리뷰 연결 |
| `build_rtr.py` | 같은 제품, 같은 달 리뷰 연결 |
| `build_rsr.py` | 같은 제품, 같은 별점 리뷰 연결 |

### **Amazon**
| 파일 | 역할 |
|------|------|
| `preprocess.py` | 전체 전처리 오케스트레이션 |
| `load_amazon.py` | CSV 데이터 로드 (사용자 정의 가능) |
| `sampling_amazon.py` | 사용자 샘플링 |
| `aggregate_user_reviews.py` | 사용자별 리뷰 집계 ⭐ |
| `feature_engineering_amazon.py` | 사용자 텍스트 → 벡터 |
| `build_upu.py` | 동일 제품 리뷰 사용자 연결 |
| `build_usu.py` | 동일 제품, 동일 별점 사용자 연결 |
| `build_uvu.py` | 텍스트 유사도 사용자 연결 |

---

## 💾 출력 구조

### **outputs/output_yelpchi/**
```
output_yelpchi/
├── models/
│   ├── baseline_gcn_epoch_50.pt
│   ├── cage_rf_gcn_v3_best.pt
│   └── ...
├── results/
│   ├── pr_auc_scores.json
│   ├── macro_f1_scores.json
│   ├── confusion_matrix.png
│   └── ...
└── logs/
    └── training.log
```

### **outputs/output_amazon/**
```
output_amazon/
├── models/
│   ├── baseline_gcn_epoch_50.pt
│   ├── cage_rf_gcn_v3_best.pt
│   └── ...
├── results/
│   ├── pr_auc_scores.json
│   ├── macro_f1_scores.json
│   ├── confusion_matrix.png
│   └── ...
└── logs/
    └── training.log
```

---

## 🔍 중요 배경 지식

### **YelpChi의 R-U-R, R-T-R, R-S-R**
```
R-U-R: 같은 사용자가 여러 리뷰 작성 → 리뷰들을 서로 연결
R-T-R: 같은 제품의 같은 달 리뷰들 → 시간적 패턴 캡처
R-S-R: 같은 제품의 같은 별점 리뷰 → 사용자 행동 패턴
```

### **Amazon의 U-P-U, U-S-U, U-V-U**
```
U-P-U: 동일 제품 리뷰한 사용자들 → 공통 관심사 사용자
U-S-U: 동일 제품, 동일 별점 → 유사한 평가 사용자
U-V-U: 리뷰 텍스트 유사도 → 스타일/언어 유사 사용자
```

### **핵심 차이**
- **YelpChi**: 리뷰 노드 → 리뷰 간 관계
- **Amazon**: 사용자 노드 → 사용자 간 관계

---

## 🎓 학습 모델 공유

두 프로젝트 모두 동일한 GNN 모델 사용 가능:
- GCN (Graph Convolutional Network)
- GraphSAGE
- GAT (Graph Attention Network)
- 기타 모든 GNN 변형

**노드 수가 다르므로 입력 레이어만 조정하면 됨**

---

## 📝 체크리스트

### **YelpChi 시작하기**
- [ ] `gnn_yelpchi/data/raw/` 에 `yelp_zip.csv` 배치
- [ ] `cd gnn_yelpchi && python preprocess.py` 실행
- [ ] `outputs/output_yelpchi/` 에서 결과 확인
- [ ] 모델 학습 코드 추가 (기존 프로젝트에서 적용)

### **Amazon 시작하기**
- [ ] `gnn_amazon/data/raw/` 에 `amazon.csv` 배치
- [ ] `load_amazon.py` 에서 필요시 컬럼명 확인
- [ ] `cd gnn_amazon && python preprocess.py` 실행
- [ ] `outputs/output_amazon/` 에서 결과 확인
- [ ] 모델 학습 코드 추가 (기존 프로젝트에서 적용)

---

## 🤔 FAQ

**Q: 두 프로젝트를 동시에 실행할 수 있나?**  
A: 네! 완전히 독립적이므로 병렬 실행 가능. 별도의 터미널에서 실행하세요.

**Q: 기존 코드를 어떻게 통합하나?**  
A: 모델 학습 로직을 각 폴더의 `run_baselines.py`에 추가하세요.

**Q: 커스텀 엣지를 추가할 수 있나?**  
A: 네! 새로운 `build_*.py` 파일을 만들고 `build_relations*.py`에 등록하세요.

**Q: 컬럼명이 다르면?**  
A: 각 `load_*.py`의 `CONFIG` 와 컬럼명을 수정하세요.

---

## 📞 문제 해결

| 문제 | 해결책 |
|------|--------|
| 데이터 파일 못 찾음 | 올바른 경로에 파일 배치, 파일명 확인 |
| 임포트 에러 | `sys.path.insert(0, ...)` 또는 PYTHONPATH 설정 |
| 메모리 부족 | `sampling.py`에서 샘플 수 감소 |
| 느린 그래프 구성 | `build_*.py`의 k값 감소 |

---

**생성일**: 2026-05-10  
**마지막 수정**: 2026-05-10

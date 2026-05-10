# Amazon & YelpChi Preprocessing

Amazon, YelpChi 참고 데이터셋의 전처리 과정 정리.

---

## 1. 입력 데이터

| 데이터셋 | 파일 | 원본 노드 | 노드 단위 |
|---------|------|----------|----------|
| Amazon | `amazon.mat` | 11,944 | 사용자 (User) |
| YelpChi | `yelpchi.mat` | ~45,000 | 리뷰 (Review) |

---

## 2. 전처리 흐름 (공통 5단계)

```
1. .mat 로드          → features, labels, adjacency 추출
2. 샘플링             → 라벨 분포 유지하며 노드 추출
3. 노드 인덱스 재매핑  → 0부터 시작하는 새 인덱스로 변환
4. Edge 필터링        → 샘플된 노드끼리 연결된 엣지만 유지
5. Train/Val/Test 분할 → 64% / 16% / 20%
```

---

## 3. 데이터셋별 차이

| 항목 | Amazon | YelpChi |
|------|--------|---------|
| Relation 수 | 4개 | 3개 |
| Relation 종류 | net_upu, net_usu, net_uvu, homo | rur, rtr, rsr |
| 샘플 크기 | 10,000 | 30,000 |

---

## 4. 출력 파일 (`data/processed/`)

| 파일 | 내용 |
|------|------|
| `features.npy` | 노드 feature 행렬 |
| `labels.npy` | 노드 라벨 (0=정상, 1=사기) |
| `edge_index_dict.pt` | 관계별 edge index dict |
| `split_idx.pt` | train/val/test 인덱스 |
| `meta.json` | 메타정보 (노드 수, feature 차원 등) |

---

## 5. 실행

```bash
# Amazon
cd gnn_amazon
python preprocess.py

# YelpChi
cd gnn_yelpchi
python preprocess.py
```

---

## 6. 핵심 포인트

- **샘플링 후 분할**: 대회 규정에 맞게 샘플링을 먼저 한 뒤 분할 수행
- **라벨 분포 유지**: 정상/사기 비율을 유지한 stratified 샘플링
- **노드 인덱스 재매핑**: 샘플된 노드들이 0부터 연속된 인덱스를 갖도록 변환 (CUDA 인덱스 오류 방지)
- **랜덤 시드 고정**: `random_state=42`로 재현성 확보

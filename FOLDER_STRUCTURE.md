# 프로젝트 Outputs 폴더 구조

## 📁 전체 구조

```
outputs/
├── cage_rf_gnn/              # CAGE-RF 메인 모델 및 Baseline
│   ├── metrics_v2.json
│   ├── metrics_v3_ablation.json
│   ├── metrics_v4_threshold_pr.json
│   ├── metrics_v5_oversampling.json
│   ├── metrics_v6_hard_mining.json
│   ├── metrics_v7_ensemble.json
│   ├── metrics_mlp.json      # Baseline MLP
│   ├── metrics_gcn.json      # Baseline GCN
│   ├── metrics_graphsage.json # Baseline GraphSAGE
│   ├── metrics_gat.json      # Baseline GAT
│   ├── best_model_cage_rf_gnn.pt
│   ├── best_model_mlp.pt
│   ├── best_model_gcn.pt
│   ├── best_model_graphsage.pt
│   ├── best_model_gat.pt
│   ├── report_v*.html
│   └── report_*.html
│
└── benchmark/                 # GNN Layer 벤치마크 (v7 Ensemble)
    ├── SAGE/
    │   ├── metrics_cage_rf_gnn_sage_v7_ensemble.json
    │   ├── best_model_cage_rf_gnn_sage.pt
    │   └── report_cage_rf_gnn_sage_v7_ensemble.html
    │
    ├── GAT/
    │   ├── metrics_cage_rf_gnn_gat_v7_ensemble.json
    │   ├── best_model_cage_rf_gnn_gat.pt
    │   └── report_cage_rf_gnn_gat_v7_ensemble.html
    │
    ├── GCN/
    │   ├── metrics_cage_rf_gnn_gcn_v7_ensemble.json
    │   ├── best_model_cage_rf_gnn_gcn.pt
    │   └── report_cage_rf_gnn_gcn_v7_ensemble.html
    │
    ├── GRAPHCONV/
    │   ├── metrics_cage_rf_gnn_graphconv_v7_ensemble.json
    │   ├── best_model_cage_rf_gnn_graphconv.pt
    │   └── report_cage_rf_gnn_graphconv_v7_ensemble.html
    │
    ├── CHEB/
    │   ├── metrics_cage_rf_gnn_cheb_v7_ensemble.json
    │   ├── best_model_cage_rf_gnn_cheb.pt
    │   └── report_cage_rf_gnn_cheb_v7_ensemble.html
    │
    ├── TAG/
    │   ├── metrics_cage_rf_gnn_tag_v7_ensemble.json
    │   ├── best_model_cage_rf_gnn_tag.pt
    │   └── report_cage_rf_gnn_tag_v7_ensemble.html
    │
    └── SG/
        ├── metrics_cage_rf_gnn_sg_v7_ensemble.json
        ├── best_model_cage_rf_gnn_sg.pt
        └── report_cage_rf_gnn_sg_v7_ensemble.html
```

## 📊 폴더별 내용

### `outputs/cage_rf_gnn/`
- **용도**: CAGE-RF 메인 모델(v2~v7)과 Baseline 모델들의 결과
- **파일 구성**:
  - `metrics_*.json`: 모델 성능 메트릭 (PR-AUC, Macro-F1, Accuracy 등)
  - `best_model_*.pt`: 최고 성능 모델 가중치
  - `report_*.html`: 상세 분석 리포트

### `outputs/benchmark/{GNN_TYPE}/`
- **용도**: GNN Layer 벤치마크 모델들 (v7 Ensemble 기준)
- **GNN 타입**: SAGE, GAT, GCN, GRAPHCONV, CHEB, TAG, SG
- **파일 구성**: cage_rf_gnn/과 동일한 구조

## 🔄 학습 흐름에 따른 파일 생성

### 1. CAGE-RF 메인 모델 학습
```bash
python src/training/train.py --model cage_rf_gnn --config configs/v7_ensemble.yaml
```
→ `outputs/cage_rf_gnn/metrics_v7_ensemble.json` 생성

### 2. Baseline 모델 학습
```bash
python src/training/train.py --model mlp --config configs/default.yaml
```
→ `outputs/cage_rf_gnn/metrics_mlp.json` 생성

### 3. GNN Layer 벤치마크
```bash
python src/training/train.py --model cage_rf_gnn_gat --config configs/v7_ensemble.yaml
```
→ `outputs/benchmark/GAT/metrics_cage_rf_gnn_gat_v7_ensemble.json` 생성

## 📈 결과 비교

### 기본 비교 (모든 모델)
```bash
python test_all.py
```

이 스크립트는:
1. ✅ 라벨 검증
2. ✅ 모든 모델 메트릭 로드
3. ✅ 성능 비교 테이블 출력
4. ✅ Top 3 모델 순위 표시
5. ✅ GNN Layer 벤치마크 결과 분리 표시

### 개별 비교
```bash
# GNN 벤치마크만 비교
python compare_gnn_results.py

# 특정 모델 결과 확인
cat outputs/cage_rf_gnn/metrics_v7_ensemble.json
cat outputs/benchmark/GAT/metrics_cage_rf_gnn_gat_v7_ensemble.json
```

## 💾 저장 규칙

| 모델 타입 | 메트릭 파일 경로 | 모델 파일 경로 |
|---------|-----------------|---|
| CAGE-RF v2~v7 | `cage_rf_gnn/metrics_v*.json` | `cage_rf_gnn/best_model_cage_rf_gnn.pt` |
| Baseline (MLP, GCN 등) | `cage_rf_gnn/metrics_mlp.json` | `cage_rf_gnn/best_model_mlp.pt` |
| GNN Benchmark (SAGE) | `benchmark/SAGE/metrics_cage_rf_gnn_sage_v7_ensemble.json` | `benchmark/SAGE/best_model_cage_rf_gnn_sage.pt` |
| GNN Benchmark (GAT) | `benchmark/GAT/metrics_cage_rf_gnn_gat_v7_ensemble.json` | `benchmark/GAT/best_model_cage_rf_gnn_gat.pt` |
| GNN Benchmark (GCN) | `benchmark/GCN/metrics_cage_rf_gnn_gcn_v7_ensemble.json` | `benchmark/GCN/best_model_cage_rf_gnn_gcn.pt` |

## 🔍 파일 형식

### metrics_*.json
```json
{
  "model": "cage_rf_gnn",
  "best_threshold": 0.5,
  "valid_metrics": {
    "pr_auc": 0.3012,
    "roc_auc": 0.7659,
    "macro_f1": 0.6089,
    ...
  },
  "test_metrics": {
    "pr_auc": 0.2670,
    "roc_auc": 0.7559,
    "macro_f1": 0.6102,
    ...
  }
}
```

## 🗑️ 정리하기

특정 모델 결과만 삭제:
```bash
# GAT 벤치마크 결과 삭제
rm -rf outputs/benchmark/GAT/

# v7만 유지하고 v2~v6 삭제
rm outputs/cage_rf_gnn/metrics_v{2,3,4,5,6}.json
```

전체 정리:
```bash
rm -rf outputs/
```

## 📌 주의사항

1. **폴더 수동 생성 불필요**: train.py가 자동으로 폴더를 생성합니다
2. **기존 파일**: 학습 시 기존 파일을 덮어씁니다 (백업 권장)
3. **모델 파일 크기**: 각 `.pt` 파일은 100MB~2GB 범위입니다
4. **리포트**: HTML 리포트는 별도로 생성되며 브라우저에서 확인 가능합니다

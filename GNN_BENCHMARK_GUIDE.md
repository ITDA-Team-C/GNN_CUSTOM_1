# CAGE-RF GNN Layer Benchmark Guide

여러 GNN layer들의 성능을 비교하기 위한 벤치마킹 가이드입니다.

## 📊 지원되는 GNN Layer들

| 이름 | 모델 파일 | 스크립트 | 설명 |
|------|---------|--------|------|
| SAGE | `cage_rf_gnn_sage.py` | `run_cage_rf_versions_sage.py` | GraphSAGE - 샘플링 기반 |
| GAT | `cage_rf_gnn_gat.py` | `run_cage_rf_versions_gat.py` | Graph Attention Network - 어텐션 기반 |
| GCN | `cage_rf_gnn_gcn.py` | `run_cage_rf_versions_gcn.py` | Graph Convolutional Network |
| GraphConv | `cage_rf_gnn_graphconv.py` | `run_cage_rf_versions_graphconv.py` | 일반 Graph Convolution |
| Cheb | `cage_rf_gnn_cheb.py` | `run_cage_rf_versions_cheb.py` | Chebyshev 다항식 기반 |
| TAG | `cage_rf_gnn_tag.py` | `run_cage_rf_versions_tag.py` | Topology Adaptive GCN |
| SG | `cage_rf_gnn_sg.py` | `run_cage_rf_versions_sg.py` | Simplifying GCNs |

## 🚀 사용 방법

### 1. 개별 GNN Layer 학습

특정 GNN layer만 학습하고 싶다면:

```bash
# SAGE 버전 (기본)
python run_cage_rf_versions_sage.py

# GAT 버전
python run_cage_rf_versions_gat.py

# GCN 버전
python run_cage_rf_versions_gcn.py

# GraphConv 버전
python run_cage_rf_versions_graphconv.py

# Cheb 버전
python run_cage_rf_versions_cheb.py

# TAG 버전
python run_cage_rf_versions_tag.py

# SG 버전
python run_cage_rf_versions_sg.py
```

### 2. 모든 GNN Layer 한 번에 벤치마킹

```bash
python run_all_gnn_benchmarks.py
```

이 스크립트는 모든 GNN layer 버전을 순서대로 학습하고 결과를 비교합니다.

### 3. 결과 비교

```bash
python compare_gnn_results.py
```

모든 모델의 성능을 PR-AUC, Macro-F1, Accuracy 기준으로 비교합니다.

## 🔧 커스텀 설정

각 모델별 하이퍼파라미터는 `configs/v7_ensemble.yaml`에서 수정할 수 있습니다:

```yaml
cage_rf:
  hidden_dim: 128        # 숨겨진 차원
  num_layers: 3          # 레이어 수
  dropout: 0.3           # 드롭아웃 확률
  use_gating: true       # Gating 사용
  use_ensemble: false    # Ensemble 사용
  heads: 8              # GAT 헤드 수 (GAT만 사용)
  K: 3                  # Chebyshev/TAG의 order (Cheb, TAG만 사용)
```

## 📈 기대되는 성능 비교

각 GNN layer의 특성:

- **SAGE**: 빠르고 안정적, 샘플링 기반의 효율성
- **GAT**: 어텐션 메커니즘으로 중요 이웃에 가중치 부여
- **GCN**: 고전적이고 이해하기 쉬운 아키텍처
- **GraphConv**: 일반적인 그래프 합성곱
- **Cheb**: Chebyshev 다항식으로 더 높은 차수의 이웃 정보 활용
- **TAG**: 위상 정보에 적응적인 구조
- **SG**: GCN을 단순화한 버전, 계산 효율성 증대

## 📊 출력 파일

각 모델 학습 후 다음 파일들이 생성됩니다:

- `outputs/cage_rf_gnn_{layer}_v7_results.json` - 모델 성능 메트릭
- `outputs/best_model_cage_rf_gnn_{layer}.pt` - 최고 성능 모델 가중치
- `outputs/cage_rf_gnn_{layer}_v7_report.html` - 상세 리포트

## 🔄 기본 동작

모든 버전은 다음을 공통으로 수행합니다:

1. v7 Ensemble 설정으로 학습
2. 6개 relation (rur, rtr, rsr, burst, semsim, behavior) 모두 사용
3. Gating 메커니즘으로 relation 가중치 학습
4. Focal Loss + Auxiliary Loss 조합
5. 동일한 전처리 및 그래프 구조 활용

## 💡 권장사항

1. **첫 실행**: SAGE 버전부터 시작 (가장 안정적)
2. **성능 비교**: 모든 버전을 실행 후 비교
3. **세밀 튜닝**: 최고 성능 모델의 하이퍼파라미터 조정
4. **배포**: 최고 성능 모델 선택 및 배포

## 🐛 트러블슈팅

**모듈을 찾을 수 없다는 오류:**
```
ImportError: cannot import name 'CAGERF_GNN' from 'src.models.cage_rf_gnn_xxx'
```
→ 모델 파일이 제대로 생성되었는지 확인: `src/models/cage_rf_gnn_*.py`

**설정 파일을 찾을 수 없다는 오류:**
```
FileNotFoundError: configs/v7_ensemble.yaml
```
→ 설정 파일 경로 확인

**GPU 메모리 부족:**
→ `configs/v7_ensemble.yaml`에서 batch_size 감소

## 📝 참고

- 기본 cage_rf_gnn.py는 cage_rf_gnn_sage.py의 래퍼로 작동합니다 (호환성 유지)
- 새로운 코드에서는 구체적인 GNN layer 버전 (cage_rf_gnn_sage 등)을 사용하세요
- 모든 모델은 동일한 학습 데이터와 그래프를 사용합니다

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모든 모델 통합 학습 스크립트
Baseline (4) + GNN Benchmark v2~v9 (7 GNN × 8 versions = 56) = 60개 모델
순서대로 학습 및 결과 비교

버전별 설명:
  v2~v7: 기존 버전들
  v8: Skip Connection (Over-smoothing 해결)
  v9: Two-Stage GNN (Skip + Gating 신호로 정제)

폴더 구조:
  outputs/
  ├── cage_rf_gnn/           (Baselines: MLP, GCN, GraphSAGE, GAT)
  └── benchmark/{GNN_TYPE}/  (7 GNN layers × 8 versions v2~v9)
"""
import subprocess
import sys
import time
import os

# UTF-8 인코딩 설정 (Windows 호환성)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 학습할 모델 그룹
CAGE_RF_VERSIONS = [
    ("v2", "configs/default.yaml"),
    ("v3", "configs/v3_ablation.yaml"),
    ("v4", "configs/v4_threshold_pr.yaml"),
    ("v5", "configs/v5_oversampling.yaml"),
    ("v6", "configs/v6_hard_mining.yaml"),
    ("v7", "configs/v7_ensemble.yaml"),
]

BASELINES = [
    ("mlp", "configs/default.yaml"),
    ("gcn", "configs/default.yaml"),
    ("graphsage", "configs/default.yaml"),
    ("gat", "configs/default.yaml"),
]

GNN_BENCHMARKS = []
GNN_LAYERS = [
    ("SAGE", "cage_rf_gnn_sage", "GraphSAGE"),
    ("GAT", "cage_rf_gnn_gat", "Graph Attention Network"),
    ("GCN", "cage_rf_gnn_gcn", "Graph Convolutional Network"),
    ("GraphConv", "cage_rf_gnn_graphconv", "Graph Convolution"),
    ("Cheb", "cage_rf_gnn_cheb", "Chebyshev"),
    ("TAG", "cage_rf_gnn_tag", "Topology Adaptive GCN"),
    ("SG", "cage_rf_gnn_sg", "Simplifying GCNs"),
]

GNN_VERSIONS = [
    ("v2", "configs/default.yaml"),
    ("v3", "configs/v3_ablation.yaml"),
    ("v4", "configs/v4_threshold_pr.yaml"),
    ("v5", "configs/v5_oversampling.yaml"),
    ("v6", "configs/v6_hard_mining.yaml"),
    ("v7", "configs/v7_ensemble.yaml"),
    ("v8", "configs/v8_skip.yaml"),
    ("v9", "configs/v9_twostage.yaml"),
]

for name, base_model, desc in GNN_LAYERS:
    for version, config in GNN_VERSIONS:
        GNN_BENCHMARKS.append((f"{name}", base_model, f"{desc} {version}", config))

def run_training(cmd, model_name, timeout=3600):
    """학습 실행 헬퍼 함수"""
    try:
        result = subprocess.run(cmd, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⏱️  {model_name} timeout after {timeout}s")
        return False
    except KeyboardInterrupt:
        print(f"⛔ {model_name} interrupted")
        return False
    except Exception as e:
        print(f"❌ {model_name} error: {e}")
        return False

print("\n" + "=" * 100)
print("🚀 COMPLETE MODEL TRAINING PIPELINE")
print("=" * 100)
total_models = len(BASELINES) + len(GNN_BENCHMARKS)
print(f"Total Models: {total_models}")
print(f"  - Baselines: {len(BASELINES)}")
print(f"  - GNN Benchmarks (v2~v9): {len(GNN_BENCHMARKS)} ({len(GNN_LAYERS)} GNN × 8 versions)")
print(f"    • v2~v7: Standard versions")
print(f"    • v8: Skip Connection (Over-smoothing 해결)")
print(f"    • v9: Two-Stage GNN (Skip + Gating 정제)")
print("=" * 100)

start_time = time.time()
all_results = []

# ============================================================================
# 1. Baseline 모델 학습
# ============================================================================
print("\n" + "=" * 100)
print("[Phase 1/2] Baseline Models Training")
print("-" * 100)

baseline_results = []
for i, (model, config) in enumerate(BASELINES, 1):
    print(f"\n[1-{i}/4] {model.upper()} with {config}")
    print("⏱️  Starting at", time.strftime('%H:%M:%S'))

    cmd = [
        sys.executable, "-m", "src.training.train",
        "--model", model,
        "--config", config,
        "--skip-preprocessing",
        "--skip-graph"
    ]

    model_start = time.time()
    success = run_training(cmd, model.upper())
    elapsed = time.time() - model_start

    if success:
        print(f"✅ {model.upper()} completed in {elapsed:.1f}s")
        baseline_results.append((model.upper(), "SUCCESS", elapsed))
        all_results.append((model.upper(), "SUCCESS", elapsed))
    else:
        print(f"❌ {model.upper()} failed")
        baseline_results.append((model.upper(), "FAILED", elapsed))
        all_results.append((model.upper(), "FAILED", elapsed))

# ============================================================================
# 2. GNN Layer Benchmark 학습
# ============================================================================
print("\n" + "=" * 100)
print("[Phase 2/2] GNN Layer Benchmark (v2~v9 All Versions)")
print("-" * 100)

gnn_results = []
for i, (name, model, desc, config) in enumerate(GNN_BENCHMARKS, 1):
    print(f"\n[2-{i}/{len(GNN_BENCHMARKS)}] {name} - {desc}")
    print("⏱️  Starting at", time.strftime('%H:%M:%S'))

    cmd = [
        sys.executable, "-m", "src.training.train",
        "--model", model,
        "--config", config,
        "--skip-preprocessing",
        "--skip-graph"
    ]

    model_start = time.time()
    success = run_training(cmd, f"CAGE-RF {name}")
    elapsed = time.time() - model_start

    if success:
        print(f"✅ {name} - {desc} completed in {elapsed:.1f}s")
        gnn_results.append((f"CAGE-RF {name} {desc.split()[-1]}", "SUCCESS", elapsed))
        all_results.append((f"CAGE-RF {name} {desc.split()[-1]}", "SUCCESS", elapsed))
    else:
        print(f"❌ {name} - {desc} failed")
        gnn_results.append((f"CAGE-RF {name} {desc.split()[-1]}", "FAILED", elapsed))
        all_results.append((f"CAGE-RF {name} {desc.split()[-1]}", "FAILED", elapsed))

# ============================================================================
# 최종 결과 요약
# ============================================================================
print("\n" + "=" * 100)
print("📊 COMPLETE TRAINING SUMMARY")
print("=" * 100)

print("\n[Phase 1] Baseline Models")
print("-" * 100)
for name, status, elapsed in baseline_results:
    icon = "✅" if status == "SUCCESS" else "❌"
    print(f"{icon} {name:20s} | {status:8s} | {elapsed:7.1f}s")

print("\n[Phase 2] GNN Layer Benchmark (v2~v9)")
print("-" * 100)
for name, status, elapsed in gnn_results:
    icon = "✅" if status == "SUCCESS" else "❌"
    print(f"{icon} {name:30s} | {status:8s} | {elapsed:7.1f}s")

# 통계
print("\n" + "=" * 100)
print("📈 STATISTICS")
print("=" * 100)
success_count = sum(1 for _, status, _ in all_results if status == "SUCCESS")
total_count = len(all_results)
total_time = time.time() - start_time

print(f"✅ Success: {success_count}/{total_count} models")
print(f"❌ Failed: {total_count - success_count}/{total_count} models")
print(f"⏱️  Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
print(f"📊 Average time per model: {total_time/total_count:.1f}s")
print("=" * 100)

# 최종 상태 확인
if success_count == total_count:
    print(f"\n🎉 SUCCESS! All {total_count} models trained successfully!")
    sys.exit(0)
else:
    print(f"\n⚠️  WARNING: {total_count - success_count} model(s) failed. Check results above.")
    sys.exit(1)

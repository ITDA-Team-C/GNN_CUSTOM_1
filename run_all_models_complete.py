#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모든 모델 통합 학습 스크립트
CAGE-RF v2~v7 (6) + Baseline (4) + GNN Benchmark (7) = 17개 모델
순서대로 학습 및 결과 비교

폴더 구조:
  outputs/
  ├── cage_rf_gnn/           (v2~v7 + Baselines)
  └── benchmark/{GNN_TYPE}/  (SAGE, GAT, GCN, GraphConv, Cheb, TAG, SG)
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

GNN_BENCHMARKS = [
    ("SAGE", "cage_rf_gnn_sage", "GraphSAGE", "configs/v7_ensemble.yaml"),
    ("GAT", "cage_rf_gnn_gat", "Graph Attention Network", "configs/v7_ensemble.yaml"),
    ("GCN", "cage_rf_gnn_gcn", "Graph Convolutional Network", "configs/v7_ensemble.yaml"),
    ("GraphConv", "cage_rf_gnn_graphconv", "Graph Convolution", "configs/v7_ensemble.yaml"),
    ("Cheb", "cage_rf_gnn_cheb", "Chebyshev", "configs/v7_ensemble.yaml"),
    ("TAG", "cage_rf_gnn_tag", "Topology Adaptive GCN", "configs/v7_ensemble.yaml"),
    ("SG", "cage_rf_gnn_sg", "Simplifying GCNs", "configs/v7_ensemble.yaml"),
]

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
print(f"Total Models: {len(CAGE_RF_VERSIONS) + len(BASELINES) + len(GNN_BENCHMARKS)} (CAGE-RF v2~v7: 6 + Baselines: 4 + GNN Benchmarks: 7)")
print("=" * 100)

start_time = time.time()
all_results = []

# ============================================================================
# 1. CAGE-RF v2~v7 학습
# ============================================================================
print("\n[Phase 1/3] CAGE-RF v2~v7 Training")
print("-" * 100)

cage_rf_results = []
for i, (version, config) in enumerate(CAGE_RF_VERSIONS, 1):
    print(f"\n[1-{i}/6] CAGE-RF {version} with {config}")
    print("⏱️  Starting at", time.strftime('%H:%M:%S'))

    cmd = [
        sys.executable, "-m", "src.training.train",
        "--model", "cage_rf_gnn",
        "--config", config,
        "--skip-preprocessing",
        "--skip-graph"
    ]

    model_start = time.time()
    success = run_training(cmd, f"CAGE-RF {version}")
    elapsed = time.time() - model_start

    if success:
        print(f"✅ CAGE-RF {version} completed in {elapsed:.1f}s")
        cage_rf_results.append((f"CAGE-RF {version}", "SUCCESS", elapsed))
        all_results.append((f"CAGE-RF {version}", "SUCCESS", elapsed))
    else:
        print(f"❌ CAGE-RF {version} failed")
        cage_rf_results.append((f"CAGE-RF {version}", "FAILED", elapsed))
        all_results.append((f"CAGE-RF {version}", "FAILED", elapsed))

# ============================================================================
# 2. Baseline 모델 학습
# ============================================================================
print("\n" + "=" * 100)
print("[Phase 2/3] Baseline Models Training")
print("-" * 100)

baseline_results = []
for i, (model, config) in enumerate(BASELINES, 1):
    print(f"\n[2-{i}/4] {model.upper()} with {config}")
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
# 3. GNN Layer Benchmark 학습
# ============================================================================
print("\n" + "=" * 100)
print("[Phase 3/3] GNN Layer Benchmark (v7 Ensemble)")
print("-" * 100)

gnn_results = []
for i, (name, model, desc, config) in enumerate(GNN_BENCHMARKS, 1):
    print(f"\n[3-{i}/7] {name} - {desc}")
    print("⏱️  Starting at", time.strftime('%H:%M:%S'))

    cmd = [
        sys.executable, "-m", "src.training.train",
        "--model", model,
        "--config", config,
        "--skip-preprocessing",
        "--skip-graph"
    ]

    model_start = time.time()
    success = run_training(cmd, f"GNN {name}")
    elapsed = time.time() - model_start

    if success:
        print(f"✅ {name} completed in {elapsed:.1f}s")
        gnn_results.append((f"GNN {name} v7", "SUCCESS", elapsed))
        all_results.append((f"GNN {name} v7", "SUCCESS", elapsed))
    else:
        print(f"❌ {name} failed")
        gnn_results.append((f"GNN {name} v7", "FAILED", elapsed))
        all_results.append((f"GNN {name} v7", "FAILED", elapsed))

# ============================================================================
# 최종 결과 요약
# ============================================================================
print("\n" + "=" * 100)
print("📊 COMPLETE TRAINING SUMMARY")
print("=" * 100)

print("\n[Phase 1] CAGE-RF v2~v7")
print("-" * 100)
for name, status, elapsed in cage_rf_results:
    icon = "✅" if status == "SUCCESS" else "❌"
    print(f"{icon} {name:20s} | {status:8s} | {elapsed:7.1f}s")

print("\n[Phase 2] Baseline Models")
print("-" * 100)
for name, status, elapsed in baseline_results:
    icon = "✅" if status == "SUCCESS" else "❌"
    print(f"{icon} {name:20s} | {status:8s} | {elapsed:7.1f}s")

print("\n[Phase 3] GNN Layer Benchmark")
print("-" * 100)
for name, status, elapsed in gnn_results:
    icon = "✅" if status == "SUCCESS" else "❌"
    print(f"{icon} {name:20s} | {status:8s} | {elapsed:7.1f}s")

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모든 GNN layer 벤치마킹 스크립트
SAGE, GAT, GCN, GraphConv, Cheb, TAG, SG 버전을 순서대로 학습

폴더 구조:
  outputs/benchmark/
    ├── SAGE/
    │   ├── metrics_cage_rf_gnn_sage_v7_ensemble.json
    │   ├── best_model_cage_rf_gnn_sage.pt
    │   └── report_cage_rf_gnn_sage_v7_ensemble.html
    ├── GAT/
    ├── GCN/
    ├── GRAPHCONV/
    ├── CHEB/
    ├── TAG/
    └── SG/
"""
import subprocess
import sys
import time
import os

# UTF-8 인코딩 설정 (Windows 호환성)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

models = [
    ("SAGE", "cage_rf_gnn_sage", "GraphSAGE"),
    ("GAT", "cage_rf_gnn_gat", "Graph Attention Network"),
    ("GCN", "cage_rf_gnn_gcn", "Graph Convolutional Network"),
    ("GraphConv", "cage_rf_gnn_graphconv", "Graph Convolution"),
    ("Cheb", "cage_rf_gnn_cheb", "Chebyshev"),
    ("TAG", "cage_rf_gnn_tag", "Topology Adaptive GCN"),
    ("SG", "cage_rf_gnn_sg", "Simplifying GCNs"),
]

print("\n" + "=" * 80)
print("🚀 CAGE-RF GNN v7 - ALL GNN LAYERS BENCHMARK")
print("=" * 80)

start_time = time.time()
results = []

for i, (name, model, desc) in enumerate(models, 1):
    print(f"\n[{i}/7] {name.upper()} - {desc}")
    print("-" * 80)
    print(f"⏱️  Starting at {time.strftime('%H:%M:%S')}")

    cmd = [
        sys.executable, "-m", "src.training.train",
        "--model", model,
        "--config", "configs/v7_ensemble.yaml",
        "--skip-preprocessing",
        "--skip-graph"
    ]

    version_start = time.time()
    try:
        result = subprocess.run(cmd, timeout=3600)  # 1시간 타임아웃
        version_time = time.time() - version_start

        if result.returncode != 0:
            print(f"❌ {name} training failed with code {result.returncode}")
            results.append((name, "FAILED", version_time))
        else:
            print(f"✅ {name} training completed in {version_time:.1f}s")
            results.append((name, "SUCCESS", version_time))
    except subprocess.TimeoutExpired:
        version_time = time.time() - version_start
        print(f"⏱️  {name} training timeout after {version_time:.1f}s")
        results.append((name, "TIMEOUT", version_time))
    except KeyboardInterrupt:
        version_time = time.time() - version_start
        print(f"⛔ {name} training interrupted after {version_time:.1f}s")
        results.append((name, "INTERRUPTED", version_time))
        break
    except Exception as e:
        version_time = time.time() - version_start
        print(f"❌ {name} training error: {e}")
        results.append((name, "ERROR", version_time))

print("\n" + "=" * 80)
print("📊 GNN LAYER BENCHMARK SUMMARY")
print("=" * 80)
for name, status, elapsed in results:
    status_icon = "✅" if status == "SUCCESS" else "❌"
    print(f"{status_icon} {name:12s} | {status:8s} | {elapsed:7.1f}s")

total_time = time.time() - start_time
print("-" * 80)
print(f"Total time: {total_time/60:.1f} minutes")
print("=" * 80)

success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
print(f"\n✅ {success_count}/{len(results)} models trained successfully")

#!/usr/bin/env python
"""
CAGE-RF GNN v7 GAT 버전 학습 스크립트
Graph Attention Network layer를 사용한 v7(Ensemble) 모델 학습
"""
import subprocess
import sys
import time

print("\n" + "=" * 70)
print("🚀 CAGE-RF GNN v7 GAT (Graph Attention Network) Training")
print("=" * 70)

start_time = time.time()

print("\n[GAT Training] with configs/v7_ensemble.yaml")
print("-" * 70)

cmd = [
    sys.executable, "-m", "src.training.train",
    "--model", "cage_rf_gnn_gat",
    "--config", "configs/v7_ensemble.yaml",
    "--skip-preprocessing",
    "--skip-graph"
]

result = subprocess.run(cmd)
elapsed = time.time() - start_time

if result.returncode != 0:
    print(f"❌ GAT v7 training failed with code {result.returncode}")
else:
    print(f"✅ GAT v7 training completed in {elapsed:.1f}s")

print("\n" + "=" * 70)
print(f"Total time: {elapsed/60:.1f} minutes")
print("=" * 70)

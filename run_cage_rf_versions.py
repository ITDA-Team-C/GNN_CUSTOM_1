#!/usr/bin/env python
"""
CAGE-RF GNN v2~v7 버전 일괄 학습 스크립트
v2 (Baseline) ~ v7 (Ensemble) 까지 순서대로 학습
"""
import subprocess
import sys
import time

versions = [
    ("v2", "configs/default.yaml"),
    ("v3", "configs/v3_ablation.yaml"),
    ("v4", "configs/v4_threshold_pr.yaml"),
    ("v5", "configs/v5_oversampling.yaml"),
    ("v6", "configs/v6_hard_mining.yaml"),
    ("v7", "configs/v7_ensemble.yaml"),
]

print("\n" + "=" * 70)
print("🚀 CAGE-RF GNN Versions Training Pipeline (v2~v7)")
print("=" * 70)

start_time = time.time()
results = []

for i, (version, config) in enumerate(versions, 1):
    print(f"\n[{i}/6] {version.upper()} Training with {config}")
    print("-" * 70)

    cmd = [
        sys.executable, "-m", "src.training.train",
        "--model", "cage_rf_gnn",
        "--config", config
    ]

    version_start = time.time()
    result = subprocess.run(cmd)
    version_time = time.time() - version_start

    if result.returncode != 0:
        print(f"❌ {version} training failed with code {result.returncode}")
        results.append((version, "FAILED", version_time))
    else:
        print(f"✅ {version} training completed in {version_time:.1f}s")
        results.append((version, "SUCCESS", version_time))

print("\n" + "=" * 70)
print("📋 Training Summary")
print("=" * 70)
for version, status, elapsed in results:
    status_icon = "✅" if status == "SUCCESS" else "❌"
    print(f"{status_icon} {version:5s} | {status:8s} | {elapsed:7.1f}s")

total_time = time.time() - start_time
print("-" * 70)
print(f"Total time: {total_time/60:.1f} minutes")
print("=" * 70)

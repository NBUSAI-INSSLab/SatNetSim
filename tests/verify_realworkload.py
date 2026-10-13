"""真实验证 realworkload：真实负载标定 + 真实任务流 + 星载耗时对照。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satnetsim.realworkload import (  # noqa: E402
    DEFAULT_SPECS, evaluate_real_compute, workload_available,
)

print("=== 负载可用性 ===")
for k, s in DEFAULT_SPECS.items():
    print("  %-22s %-10s %s" % (k, s.category, "可用" if workload_available(k) else "缺依赖"))

print()
rep, tasks = evaluate_real_compute(
    n_tasks=120, n_gs=6, duration_s=600.0, sat_capacity_tops=8.0,
    repeats=3, progress=lambda m: print("  ·", m, flush=True))

print()
print("=" * 78)
print("%-30s %10s %16s %10s %10s" % ("真实负载", "实测(ms)", "吞吐", "有效GFLOPS", "真实产出"))
print("-" * 78)
for a, b, c, d, e in rep.summary_rows():
    print("%-30s %10s %16s %10s %10s" % (a, b, c, d, e))
print("=" * 78)
print("%-30s %10s %16s %10s %10s" % ("负载", "实测(ms)", "吞吐", "有效GFLOPS", "真实产出"))
print("-" * 78)
for k, v in rep.headline():
    print("  %-22s %s" % (k, v))
print("-" * 78)
print("任务画像:", rep.mix_text())
print("首个任务:", tasks[0] if tasks else None)
for w in rep.warnings:
    print("WARN:", w)

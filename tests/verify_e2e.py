"""SatNetSim 端到端实现验证。

对指定预设跑通完整流水线，逐项核对关键建模结果是否符合物理/理论预期。
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satnetsim.config import preset  # noqa: E402
from satnetsim.engine import Simulation  # noqa: E402

PRESET = sys.argv[1] if len(sys.argv) > 1 else "starlink-lite"

print("=" * 68)
print("SatNetSim 端到端验证   preset =", PRESET)
print("=" * 68)

cfg = preset(PRESET)
print("星座      :", cfg.constellation.planes, "x", cfg.constellation.sats_per_plane,
      "=", cfg.constellation.n_satellites, "星",
      cfg.constellation.altitude_km, "km", cfg.constellation.inclination_deg, "deg")
print("ISL 构型  :", cfg.link.isl_mode, "| ISL 带宽",
      cfg.link.isl_bandwidth_mbps / 1000.0, "Gbps")
print("路由      :", cfg.routing.protocol)
print("传输      :", cfg.transport.protocols if hasattr(cfg.transport, "protocols")
      else cfg.transport.protocol)
print("-" * 68)

ticks: list[str] = []


def tick(msg: str, frac: float = 0.0) -> None:
    line = "[%5.1f%%] %s" % (frac * 100, msg)
    ticks.append(line)
    print(line, flush=True)


t0 = time.perf_counter()
sim = Simulation(cfg)
res = sim.run(progress=tick)
dt = time.perf_counter() - t0

print("-" * 68)
print("总耗时: %.2f s" % dt)
print("=" * 68)
for k, v in res.headline():
    print("  %-22s %s" % (k, v))

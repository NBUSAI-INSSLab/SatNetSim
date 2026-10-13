"""真实验证 reallink：真实 TCP + 真实包级可靠传输 + 解析对照。"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satnetsim.reallink import (  # noqa: E402
    LinkProfile, evaluate_transfer, measure_rtt_ms, packet_loss_from_rtt_loss,
)

# 目标：每 RTT 有效丢包 0.5%（解析模型口径）→ 换算成每包丢包
OWD, BW, RTT_LOSS = 39.6, 500.0, 0.005
_bdp_seg = BW * 1e6 / 8 * (2 * OWD) / 1000 / 1200
prof = LinkProfile(name="LEO 代表路径", owd_ms=OWD, bandwidth_mbps=BW,
                   loss_rate=packet_loss_from_rtt_loss(RTT_LOSS, _bdp_seg),
                   jitter_ms=2.0)
print("链路:", prof.describe())
print("BDP : %.0f KiB / %.0f 段" % (prof.bdp_bytes / 1024, _bdp_seg))
print("-" * 68)

t0 = time.perf_counter()
rtt = measure_rtt_ms(prof, trials=7)
print("实测 RTT: %.2f ms (理论 %.2f ms)  [%.2f s]"
      % (rtt, prof.rtt_ms, time.perf_counter() - t0))
print("-" * 68)

rep = evaluate_transfer(prof, payload_mbit=16.0, tcp_trials=2, packet_trials=1,
                        packet_algos=("reno", "cubic", "hybla"),
                        progress=lambda m: print("  ·", m, flush=True))

print("-" * 68)
print("%-28s %11s %13s %7s %s" % ("方法", "完成(s)", "吞吐(Mbps)", "重传", "完整性"))
for a, b, c, d, e in rep.summary_rows():
    print("%-28s %11s %13s %7s %s" % (a, b, c, d, e))
print("-" * 68)
print("墙钟 %.1f s" % rep.wall_s)
print("包级链路:", rep.packet_profile.describe() if rep.packet_profile else "-")
for w in rep.warnings:
    print("WARN:", w)

"""liveviz 冒烟：构建场景 → 离屏渲染几帧 → 导出单文件 HTML。"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from satnetsim.config import preset
from satnetsim.liveviz import LiveScene, LiveView, build_frame, export_html
from satnetsim.orbit import Constellation, unix_from_iso
from satnetsim.realdatastream import simulate_real_data
from satnetsim.topology import TopologyBuilder

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "verify")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    cfg = preset("starlink-lite")
    con = Constellation(cfg.constellation)
    times = np.arange(0.0, cfg.duration_s + 1e-9, cfg.snapshot_interval_s)
    u = unix_from_iso(cfg.start_utc)
    con.propagate(times); con.to_ecef(u)
    snaps = TopologyBuilder(cfg.link, cfg.ground_stations).build(
        con, times, u, cfg.snapshot_interval_s)

    rd = cfg.real_data
    rep = simulate_real_data(rd, con, snaps, cfg.ground_stations,
                             snapshot_interval_s=cfg.snapshot_interval_s,
                             sim_duration_s=cfg.duration_s, seed=cfg.seed)

    t0 = time.perf_counter()
    scene = LiveScene.build(con, snaps, cfg.ground_stations, rep, times)
    print(f"场景构建 {time.perf_counter()-t0:.2f}s；分段 {scene.seg_t0.size}，"
          f"事件 {scene.ev_t.size}，流 {len(scene.flows)}")

    tmax = scene.duration_s
    for frac in (0.15, 0.45, 0.8):
        t = tmax * frac
        a = time.perf_counter()
        fm = build_frame(scene, t, 1280, 760, 0.55, 0.42)
        print(f"  t={t:6.2f}s  彗星 {len(fm.comets):3d}  脉冲 {len(fm.pulses):3d}  "
              f"链路 {fm.links.shape[0]:3d}  用时 {(time.perf_counter()-a)*1000:.0f} ms")

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    view = LiveView()
    view.resize(1280, 760)
    view.set_scene(scene)
    for frac in (0.18, 0.5, 0.85):
        img = view.snap_image(tmax * frac, 1280, 760)
        p = os.path.join(OUT, f"live_frame_{int(frac*100):02d}.png")
        img.save(p)
        print("  截图 →", p)

    html = export_html(scene, os.path.join(OUT, "satnetsim_live.html"))
    print("HTML →", html, f"{os.path.getsize(html)/1024:.0f} KiB")
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

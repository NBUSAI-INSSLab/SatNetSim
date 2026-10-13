"""可视化层验证：把全套图表真实渲染到 PNG（按 viz 的真实签名调用）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from satnetsim import viz  # noqa: E402
from satnetsim.config import DEFAULT_GS, preset  # noqa: E402
from satnetsim.engine import Simulation  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "verify")
os.makedirs(OUT, exist_ok=True)

cfg = preset("starlink-lite")
res = Simulation(cfg).run(progress=lambda *a, **k: None)

# 真实评测图表需要一个带真实评测结果的结果对象（真的跑 TCP 与算子）
from satnetsim.config import real_eval_preset  # noqa: E402

_real_cfg = real_eval_preset("starlink-lite", payload_mbit=8.0, tcp_trials=1,
                             packet_trials=1, real_task_count=30, repeats=2)
_real_cfg.duration_s = 120.0
_real_cfg.compute.task_count = 40
res_real = Simulation(_real_cfg).run(
                                     progress=lambda *a, **k: None)

gs_lon = [g.lon_deg for g in DEFAULT_GS]
gs_lat = [g.lat_deg for g in DEFAULT_GS]


def fig1():
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    viz.plot_constellation_3d(ax, res.constellation, snapshot_index=0,
                              ground_stations=DEFAULT_GS)
    return fig


def fig2():
    fig, ax = plt.subplots(figsize=(9, 5))
    viz.plot_ground_tracks(ax, res.constellation, ground_stations=DEFAULT_GS)
    return fig


def fig3():
    fig, ax = plt.subplots(figsize=(8, 5))
    viz.plot_route_metrics(ax, res.routes)
    return fig


def fig4():
    fig, ax = plt.subplots(figsize=(8, 5))
    viz.plot_route_bars(ax, res.routes)
    return fig


def fig5():
    fig, ax = plt.subplots(figsize=(8, 5))
    viz.plot_transport_compare(ax, res.transport)
    return fig


def fig6():
    fig, ax = plt.subplots(figsize=(9, 5))
    viz.plot_compute_compare(ax, res.compute)
    return fig


def fig7():
    caps, curves = res.contention
    fig, ax = plt.subplots(figsize=(8, 5))
    viz.plot_contention_sweep(ax, caps, curves)
    return fig


def fig8():
    fig, ax = plt.subplots(figsize=(9, 5))
    viz.plot_real_transfer_compare(ax, res_real.real_transfer)
    return fig


def fig9():
    fig, ax = plt.subplots(figsize=(9, 5))
    viz.plot_real_cwnd(ax, res_real.real_transfer)
    return fig


def fig10():
    fig, ax = plt.subplots(figsize=(9, 5))
    viz.plot_real_workload_time(ax, res_real.real_compute)
    return fig


def fig11():
    fig, ax = plt.subplots(figsize=(8, 5))
    viz.plot_real_compute_split(ax, res_real.real_compute)
    return fig


jobs = [
    ("01_constellation_3d", fig1),
    ("02_ground_tracks", fig2),
    ("03_route_metrics", fig3),
    ("04_route_bars", fig4),
    ("05_transport_compare", fig5),
    ("06_compute_compare", fig6),
    ("07_contention_sweep", fig7),
    ("08_real_transfer_vs_analytic", fig8),
    ("09_real_cwnd", fig9),
    ("10_real_workload_time", fig10),
    ("11_real_compute_split", fig11),
]

ok = 0
for name, fn in jobs:
    try:
        fig = fn()
        path = os.path.join(OUT, name + ".png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        ok += 1
        print("  [OK]   %-24s -> %8d bytes" % (name, os.path.getsize(path)))
    except Exception as exc:  # noqa: BLE001
        print("  [FAIL] %-24s %s: %s" % (name, type(exc).__name__, exc))

print("-" * 62)
print("渲染成功 %d / %d" % (ok, len(jobs)))

txt = viz.text_report(res)
with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt)
print("文本报告 -> %d 字" % len(txt))

txt_real = viz.text_report(res_real)
with open(os.path.join(OUT, "report_real_eval.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt_real)
print("真实评测报告 -> %d 字" % len(txt_real))

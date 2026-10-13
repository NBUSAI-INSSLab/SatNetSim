"""可视化工具：把仿真结果画成图。

全部基于 matplotlib（与 PySide6 通过 QtAgg 后端内嵌），不依赖任何在线服务
（这一点与 Hypatia 的 Cesium 可视化路线不同：牺牲了三维地球质感，换来了
离线可用、零 token、可脚本化导出）。

图表清单
--------
plot_constellation_3d   星座三维构型（ECI/ECEF 散点）
plot_ground_tracks      星下点轨迹（含地面站位置）
plot_route_3d           一条端到端路径的逐跳三维示意
plot_route_metrics      各协议的时延/跳数随快照变化
plot_transport_compare  传输协议完成时间/吞吐对比
plot_cwnd_trace         拥塞窗口随轮次演化
plot_compute_compare    星载计算架构对比（时延/下行量）
plot_contention_sweep   算力-时延曲线上各调度策略的分化
plot_real_transfer_compare  真实传输（内核 TCP / 包级可靠传输）vs 解析模型
plot_real_cwnd          真实包级传输的拥塞窗口演化（含孔径参考线）
plot_real_workload_time 真实负载在本机的实测耗时与吞吐
plot_real_compute_split 真实任务流：地面串行 vs 星载折算耗时
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.font_manager as _fm
import numpy as np


CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",      # Windows
    "SimHei",               # Windows
    "PingFang SC",          # macOS
    "Hiragino Sans GB",     # macOS
    "Noto Sans CJK SC",     # Linux
    "Source Han Sans SC",   # Linux
    "WenQuanYi Micro Hei",  # Linux
]

_CJK_CHOSEN: List[str] = []


def _setup_cjk_font() -> None:
    """选择首个可用的中文字体，保证任何入口下图表中文都不出现方框。

    此前该配置只存在于 gui.py，导致 CLI 导出或外部脚本直接调用本模块时
    中文标签退化为 tofu（DejaVu Sans 无 CJK 字形）；现统一下沉到可视化层。

    这里同时把命中的字体名暴露为 ``CJK_FONT_CANDIDATES`` 中的实际集合，
    供需要显式指定 ``family=`` 的调用方复用（例如等宽风格的文字面板——
    若写死 Consolas，面板里的中文会再次变成方框）。
    """
    installed = {f.name for f in _fm.fontManager.ttflist}
    chosen = [n for n in CJK_FONT_CANDIDATES if n in installed]
    matplotlib.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    _CJK_CHOSEN[:] = chosen


def cjk_font_family() -> str:
    """返回首个可用的中文字体名；无可用中文字体时退回 DejaVu Sans。"""
    return _CJK_CHOSEN[0] if _CJK_CHOSEN else "DejaVu Sans"


_setup_cjk_font()


# 统一配色（与桌面端薄荷绿主题协调，同时保证浅色底可读）
C_PRIMARY = "#0E7C86"     # 主色：深青
C_ACCENT = "#D97706"      # 强调：琥珀
C_ALT = "#2563EB"         # 次色：蓝
C_WARN = "#B91C1C"        # 警示：红
C_MUTED = "#6B7280"       # 次要文字
C_GRID = "#D1D5DB"        # 网格
PALETTE = ["#0E7C86", "#D97706", "#2563EB", "#7C3AED", "#B91C1C", "#059669", "#4B5563"]


def _style(ax) -> None:
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.6)
    ax.set_facecolor("#FBFDFD")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8)


# --------------------------------------------------------------------------- #
# 星座几何
# --------------------------------------------------------------------------- #
def plot_constellation_3d(ax, constellation, snapshot_index: int = 0,
                          ground_stations: Optional[Sequence] = None,
                          highlight_path: Optional[Sequence[int]] = None) -> None:
    """三维星座构型（ECEF，单位 km）。"""
    if not hasattr(ax, "plot_surface"):  # 传进来的不是 3D 轴
        return
    pos = constellation.positions_ecef
    if pos is None:
        return
    gi = min(snapshot_index, pos.shape[1] - 1)
    p = pos[:, gi, :]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=6, c=C_PRIMARY, alpha=0.75,
               depthshade=False, label=f"卫星 ×{p.shape[0]}")

    # 地球参考球
    from .orbit import EARTH_RADIUS_KM

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    ys = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    zs = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#CFE8EA", alpha=0.35, linewidth=0,
                    antialiased=True, shade=False)

    if ground_stations:
        from .orbit import lla_to_ecef

        gs = np.array([lla_to_ecef(g.lat_deg, g.lon_deg, g.alt_km)
                       for g in ground_stations])
        ax.scatter(gs[:, 0], gs[:, 1], gs[:, 2], s=30, c=C_ACCENT,
                   marker="^", depthshade=False, label="地面站")

    if highlight_path:
        node_pos = {}
        for i in highlight_path:
            if i < p.shape[0]:
                node_pos[i] = p[i]
        if len(node_pos) >= 2:
            pts = np.array([node_pos[i] for i in highlight_path if i in node_pos])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c=C_WARN, lw=1.6,
                    marker="o", ms=3, label="代表路径")

    ax.set_xlabel("X (km)", fontsize=8)
    ax.set_ylabel("Y (km)", fontsize=8)
    ax.set_zlabel("Z (km)", fontsize=8)
    ax.tick_params(labelsize=7, colors=C_MUTED)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("星座三维构型（ECEF）", fontsize=10, color="#111827")


def plot_ground_tracks(ax, constellation, ground_stations=None,
                       max_tracks: int = 12) -> None:
    """星下点轨迹 + 地面站位置。"""
    lat, lon = constellation.lat_deg, constellation.lon_deg
    if lat is None:
        return
    n = min(max_tracks, lat.shape[0])
    idx = np.linspace(0, lat.shape[0] - 1, n).astype(int)
    for k, i in enumerate(idx):
        ax.plot(lon[i], lat[i], lw=1.0, alpha=0.8, color=PALETTE[k % len(PALETTE)])
    if ground_stations:
        for g in ground_stations:
            ax.plot([g.lon_deg], [g.lat_deg], marker="^", ms=9, color=C_ACCENT,
                    mec="#7C2D12", mew=0.6, ls="none")
            ax.annotate(g.name, (g.lon_deg, g.lat_deg), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="#7C2D12")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("经度 (°)", fontsize=8)
    ax.set_ylabel("纬度 (°)", fontsize=8)
    ax.set_title(f"星下点轨迹（显示 {n} 条）", fontsize=10, color="#111827")
    _style(ax)


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
def plot_route_metrics(ax, routes: Dict[str, object]) -> None:
    """各路由协议的端到端时延随快照变化。"""
    for k, (name, ra) in enumerate(routes.items()):
        d = np.asarray(ra.delays, dtype=float)
        x = np.arange(d.size) * 1.0
        ax.plot(x, d, lw=1.4, color=PALETTE[k % len(PALETTE)],
                label=f"{ra.label} (μ={np.nanmean(d):.1f} ms)")
    ax.set_xlabel("快照序号", fontsize=8)
    ax.set_ylabel("端到端单向时延 (ms)", fontsize=8)
    ax.set_title("路由协议时延对比", fontsize=10, color="#111827")
    ax.legend(fontsize=7)
    _style(ax)


def plot_route_bars(ax, routes: Dict[str, object]) -> None:
    """时延 / 跳数 / 路径切换次数三指标柱状对比。"""
    names = [ra.label for ra in routes.values()]
    delays = [ra.mean_delay_ms for ra in routes.values()]
    hops = [ra.mean_hops for ra in routes.values()]
    switches = [ra.switch_count for ra in routes.values()]

    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, delays, w, color=C_PRIMARY, label="平均时延 (ms)")
    ax2 = ax.twinx()
    ax2.bar(x, hops, w, color=C_ACCENT, label="平均跳数")
    ax2.bar(x + w, switches, w, color=C_ALT, alpha=0.85, label="路径切换次数")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=12)
    ax.set_ylabel("时延 (ms)", fontsize=8, color=C_PRIMARY)
    ax2.set_ylabel("跳数 / 切换次数", fontsize=8, color=C_ACCENT)
    ax.set_title("路由协议多指标对比", fontsize=10, color="#111827")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    _style(ax)
    ax2.tick_params(labelsize=7, colors=C_MUTED)


# --------------------------------------------------------------------------- #
# 传输
# --------------------------------------------------------------------------- #
def plot_transport_compare(ax, flows: Sequence) -> None:
    """传输协议：完成时间柱状 + 吞吐折线。"""
    labels = [f.label for f in flows]
    comp = [f.completion_s * 1000 for f in flows]
    good = [f.goodput_mbps for f in flows]

    x = np.arange(len(labels))
    ax.bar(x, comp, 0.55, color=C_PRIMARY, alpha=0.9, label="完成时间 (ms)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=20)
    ax.set_ylabel("完成时间 (ms，对数轴)", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, good, marker="o", ms=4, color=C_ACCENT, lw=1.6, label="有效吞吐 (Mbps)")
    ax2.set_ylabel("有效吞吐 (Mbps)", fontsize=8, color=C_ACCENT)
    ax.set_title("传输协议端到端性能对比（500 Mbit 传输）", fontsize=10, color="#111827")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    _style(ax)
    ax2.tick_params(labelsize=7, colors=C_MUTED)


def plot_cwnd_trace(ax, flows: Sequence) -> None:
    """拥塞窗口随 RTT 轮次的演化。"""
    for k, f in enumerate(flows):
        if not f.cwnd_trace:
            continue
        ax.plot(range(len(f.cwnd_trace)), f.cwnd_trace, lw=1.4,
                color=PALETTE[k % len(PALETTE)], label=f.label)
    ax.set_xlabel("RTT 轮次", fontsize=8)
    ax.set_ylabel("拥塞窗口 (段)", fontsize=8)
    ax.set_title("拥塞窗口演化", fontsize=10, color="#111827")
    ax.legend(fontsize=7)
    _style(ax)


# --------------------------------------------------------------------------- #
# 星载计算
# --------------------------------------------------------------------------- #
def plot_compute_compare(ax, reports: Dict[str, object]) -> None:
    """架构对比：平均时延 vs 下行数据量。"""
    labels = list(reports.keys())
    lat = [reports[k].mean_latency_s for k in labels]
    dl = [reports[k].downlink_mbit for k in labels]

    x = np.arange(len(labels))
    ax.bar(x, lat, 0.55, color=C_PRIMARY, label="平均端到端时延 (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=12)
    ax.set_ylabel("平均时延 (s)", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, dl, marker="s", ms=4, color=C_WARN, lw=1.4, label="星地下行总量 (Mbit)")
    ax2.set_yscale("log")
    ax2.set_ylabel("下行数据量 (Mbit，对数轴)", fontsize=8, color=C_WARN)
    ax.set_title("弯管 vs 星上边缘计算（OEC）", fontsize=10, color="#111827")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    _style(ax)
    ax2.tick_params(labelsize=7, colors=C_MUTED)


def plot_contention_sweep(ax, caps: Sequence[float], curves: Dict[str, List[float]]) -> None:
    """单星算力 → 平均时延：各调度策略的分化区间。"""
    if not len(caps):
        ax.text(0.5, 0.5, "未运行拥塞扫描", ha="center", va="center",
                fontsize=9, color=C_MUTED, transform=ax.transAxes)
        _style(ax)
        return
    for k, (name, vals) in enumerate(curves.items()):
        ax.plot(caps, vals, marker="o", ms=4, lw=1.4,
                color=PALETTE[k % len(PALETTE)], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("单星算力 (TOPS，对数轴)", fontsize=8)
    ax.set_ylabel("平均端到端时延 (s)", fontsize=8)
    ax.set_title("算力-时延曲线：调度策略何时开始起作用", fontsize=10, color="#111827")
    ax.legend(fontsize=7)
    _style(ax)


# --------------------------------------------------------------------------- #
# 真实评测（真实数据传输 / 真实计算任务）
# --------------------------------------------------------------------------- #
def _placeholder(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9,
            color=C_MUTED, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax)


def plot_real_transfer_compare(ax, report) -> None:
    """真实传输 vs 解析模型：有效吞吐对比（对数横轴）。

    三类方法分色：真实 TCP（内核协议栈）、真实包级（用户态可靠传输 + 拥塞
    控制状态机）、解析模型（窗口级积分）。三者同口径（同一链路剖面/载荷/
    丢包率）才有可比性。
    """
    if report is None:
        _placeholder(ax, "未启用真实数据传输评测\n请在左侧勾选「真实评测 → 真实数据传输」后重新运行")
        return
    names: List[str] = []
    good: List[float] = []
    colors: List[str] = []
    for t in report.tcp:
        names.append(f"真实 TCP（内核栈）· {t.integrity}")
        good.append(max(t.goodput_mbps, 1e-3))
        colors.append(C_PRIMARY)
    for t in report.packet:
        names.append(f"真实包级 {t.label}")
        good.append(max(t.goodput_mbps, 1e-3))
        colors.append(C_ACCENT)
    for k, v in sorted(report.analytic.items(), key=lambda kv: kv[1]):
        names.append(f"解析模型 {k}")
        good.append(max(report.analytic_goodput.get(k, 0.0), 1e-3))
        colors.append(C_ALT)
    if not names:
        _placeholder(ax, "无可用传输结果")
        return
    y = np.arange(len(names))[::-1]
    ax.barh(y, good, 0.6, color=colors, alpha=0.9)
    for yy, g in zip(y, good):
        ax.text(g * 1.08, yy, f"{g:.1f} Mbps", va="center",
                fontsize=7, color="#111827")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("有效吞吐 (Mbps，对数轴)", fontsize=8)
    ax.set_title("真实传输 vs 解析模型（同口径有效吞吐）", fontsize=10, color="#111827")
    ax.axvline(report.harness_cap_mbps, color=C_WARN, lw=0.9, ls=":",
               label=f"用户态包级测试床可靠容量 ~{report.harness_cap_mbps:.0f} Mbps")
    ax.legend(fontsize=6.5, loc="lower right")
    _style(ax)


def plot_real_cwnd(ax, report) -> None:
    """真实包级传输的拥塞窗口演化（KB，含 BDP 参考线）。"""
    if report is None or not report.packet:
        _placeholder(ax, "未启用真实数据传输评测\n请在左侧「真实评测」中勾选后重新运行")
        return
    for k, t in enumerate(report.packet):
        if not t.cwnd_trace:
            continue
        xs = [p[0] for p in t.cwnd_trace]
        ys = [p[1] / 1024.0 for p in t.cwnd_trace]
        ax.plot(xs, ys, lw=1.3, color=PALETTE[k % len(PALETTE)],
                label=f"{t.label}（重传 {t.retransmits}）")
    prof = report.packet_profile
    if prof is not None:
        ax.axhline(prof.bdp_bytes / 1024.0, color=C_WARN, lw=0.9, ls=":",
                   label=f"链路 BDP {prof.bdp_bytes/1024:.0f} KB")
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("拥塞窗口 (KB)", fontsize=8)
    ax.set_title("真实包级传输：拥塞窗口演化", fontsize=10, color="#111827")
    ax.legend(fontsize=6.5)
    _style(ax)


def plot_real_workload_time(ax, report) -> None:
    """真实负载在本机的实测耗时（对数横轴）+ 吞吐标注。"""
    if report is None or not report.calibration.runs:
        _placeholder(ax, "未启用真实计算评测\n请在左侧「真实评测」中勾选后重新运行")
        return
    runs = report.calibration.runs
    labels, vals, colors = [], [], []
    for r in runs:
        labels.append(r.label + ("" if r.ok else "（失败）"))
        vals.append(max(r.elapsed_s * 1000.0, 1e-3))
        colors.append(C_PRIMARY if r.ok else C_MUTED)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, 0.62, color=colors, alpha=0.9)
    for rr, yy in zip(runs, y):
        txt = f"{rr.elapsed_s*1000:.2f} ms"
        if rr.ok and rr.gflops_effective > 0:
            txt += f" / {rr.gflops_effective:.1f} GFLOPS"
        ax.text(max(rr.elapsed_s * 1000.0, 1e-3) * 1.1, yy, txt,
                va="center", fontsize=6.5, color="#111827")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("本机实测耗时 (ms，对数轴)", fontsize=8)
    ax.set_title("真实计算负载：本机实测耗时（真实算子）", fontsize=10, color="#111827")
    _style(ax)


def plot_real_compute_split(ax, report) -> None:
    """真实任务流：地面串行耗时 vs 星载折算耗时。"""
    if report is None:
        _placeholder(ax, "未启用真实计算评测\n请在左侧「真实评测」中勾选后重新运行")
        return
    vals = [max(report.ground_total_s, 1e-6), max(report.on_orbit_total_s, 1e-6)]
    names = [f"地面本机串行\n（{report.task_count} 个真实任务实测）",
             "星载折算\n（按单星有效算力）"]
    x = np.arange(2)
    ax.bar(x, vals, 0.5, color=[C_ACCENT, C_PRIMARY], alpha=0.9)
    for xx, v in zip(x, vals):
        ax.text(xx, v * 1.15, f"{v:.3f} s", ha="center", fontsize=8, color="#111827")
    ax.set_yscale("log")
    # 顶部留白：给数值标注与任务画像文字让位（对数轴下不影响比值可读性）
    ax.set_ylim(top=max(vals) * 40)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("串行耗时 (s，对数轴)", fontsize=8)
    ax.set_title(f"真实任务流耗时对照（加速比 {report.speedup:.1f}×）",
                 fontsize=10, color="#111827")
    # 任务画像放右上角（右柱很短，上方空间大），手工折行避免压到柱体
    mix = f"任务画像：{report.mix_text()}"
    wrapped = "\n".join(textwrap.wrap(mix, 26))
    ax.text(0.97, 0.95, wrapped, transform=ax.transAxes, fontsize=5.8,
            color=C_MUTED, va="top", ha="right", linespacing=1.4)
    _style(ax)


# --------------------------------------------------------------------------- #
# 报告式汇总文字
# --------------------------------------------------------------------------- #
def plot_tcp_compare(ax, tcp_flows: Sequence, analytic: Sequence = (),
                     capacity: Optional[Dict[str, float]] = None) -> None:
    """真实包级 TCP vs 解析模型：有效吞吐横向对比（对数横轴）。"""
    if not tcp_flows:
        _placeholder(ax, "未启用真实 TCP 包级仿真\n请在左侧「传输仿真引擎」选择真实包级")
        return
    names: List[str] = []
    good: List[float] = []
    colors: List[str] = []
    for k, f in enumerate(tcp_flows):
        names.append(f"{f.label}")
        good.append(max(f.goodput_mbps, 1e-3))
        colors.append(PALETTE[k % len(PALETTE)])
    for f in analytic:
        names.append(f"解析模型 {f.label}")
        good.append(max(f.goodput_mbps, 1e-3))
        colors.append(C_MUTED)
    y = np.arange(len(names))[::-1]
    ax.barh(y, good, 0.62, color=colors, alpha=0.9)
    for yy, g in zip(y, good):
        ax.text(g * 1.06, yy, f"{g:.1f}", va="center", fontsize=6.5, color="#111827")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("有效吞吐 (Mbps，对数轴)", fontsize=8)
    ax.set_title("真实包级 TCP 状态机 vs 解析模型", fontsize=10, color="#111827")
    if capacity:
        wl = capacity.get("window_limited_mbps", 0.0)
        if wl and wl > 0:
            ax.axvline(wl, color=C_WARN, lw=0.9, ls=":",
                       label=f"窗口上限 {wl:.0f} Mbps")
            ax.legend(fontsize=6.5, loc="lower right")
    _style(ax)


def plot_tcp_cwnd(ax, flow) -> None:
    """真实 TCP 拥塞窗口演化：cwnd / ssthresh / RTT 与相位标记。"""
    if flow is None or not getattr(flow, "samples", None):
        _placeholder(ax, "无真实 TCP 采样")
        return
    ts = [s.t for s in flow.samples]
    cw = [s.cwnd / 1024.0 for s in flow.samples]
    ss = [s.ssthresh / 1024.0 if np.isfinite(s.ssthresh) else np.nan
          for s in flow.samples]
    ax.plot(ts, cw, lw=1.5, color=C_PRIMARY, label="cwnd")
    ax.plot(ts, ss, lw=1.0, color=C_ACCENT, ls="--", label="ssthresh")
    bdp = getattr(flow, "mean_bdp_bytes", 0.0) / 1024.0
    if bdp > 0:
        ax.axhline(bdp, color=C_WARN, lw=0.9, ls=":",
                   label=f"平均 BDP {bdp:.0f} KB")
    # 相位着色：恢复期（快速重传/超时）标红带
    for s in flow.samples:
        if s.phase == "rec":
            ax.axvspan(s.t, s.t + 0.05, color=C_ALT, alpha=0.25, lw=0)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("窗口 (KB)", fontsize=8)
    ax.set_title(f"真实 TCP：{flow.label} 窗口演化", fontsize=10, color="#111827")
    ax.legend(fontsize=6.5)
    _style(ax)


def plot_tcp_profile(ax, profile) -> None:
    """时变路径剖面：RTT 与瓶颈带宽，路径阶跃处标红。"""
    if profile is None or not len(profile.t):
        _placeholder(ax, "无路径剖面")
        return
    t = np.asarray(profile.t)
    ax.plot(t, profile.rtt_ms, lw=1.6, color=C_PRIMARY, marker="o", ms=3,
            label="往返时延 RTT")
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("RTT (ms)", fontsize=8, color=C_PRIMARY)
    ax2 = ax.twinx()
    ax2.plot(t, profile.bw_mbps, lw=1.2, color=C_ACCENT, ls="-", alpha=0.85,
             label="瓶颈带宽")
    ax2.set_ylabel("瓶颈带宽 (Mbps)", fontsize=8, color=C_ACCENT)
    # 路径阶跃（拓扑切换）标记
    if t.size > 1:
        jumps = np.where(np.abs(np.diff(profile.owd_ms)) > 0.2)[0]
        for j in jumps:
            ax.axvline(t[j + 1], color=C_WARN, lw=0.8, ls=":", alpha=0.9)
    ax.set_title(f"路径剖面：{profile.describe()}", fontsize=9, color="#111827")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper left")
    _style(ax)
    ax2.tick_params(labelsize=7, colors=C_MUTED)


def tcp_table(ax, tcp_flows: Sequence, capacity: Optional[Dict[str, float]] = None) -> None:
    """真实 TCP 评测明细表。"""
    ax.axis("off")
    if not tcp_flows:
        ax.set_title("未启用真实 TCP 包级仿真", fontsize=10, color=C_MUTED)
        return
    rows = []
    for f in tcp_flows:
        rows.append([
            f.protocol.upper(),
            f"{f.completion_s:.3f}" if np.isfinite(f.completion_s) else "未完成",
            f"{f.goodput_mbps:.1f}",
            f"{f.utilization*100:.0f}%",
            f"{f.retransmits}({f.retransmit_ratio*100:.0f}%)",
            f"{int(f.timeouts)}/{int(f.fast_retransmits)}",
            f"{f.rtt_min_ms:.0f}/{f.rtt_p95_ms:.0f}",
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["算法", "完成(s)", "吞吐(Mbps)", "效率", "重传(占比)", "超时/快重", "RTT min/p95"],
        colWidths=[0.13, 0.13, 0.15, 0.09, 0.17, 0.15, 0.18],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    table.scale(1.0, 1.35)
    note = ""
    if capacity:
        note = (f"\n平均 BDP {capacity.get('mean_bdp_bytes', 0)/1024:.0f} KB"
                f"（{capacity.get('mean_bdp_segments', 0):.0f} 段）"
                f"，窗口上限 {capacity.get('window_limited_mbps', 0):.0f} Mbps"
                f"，路径阶跃 {int(capacity.get('path_changes', 0))} 次")
    ax.set_title("真实 TCP 包级评测（真协议状态机）" + note,
                 fontsize=8.5, color="#111827")


def route_eval_table(ax, routes: Dict[str, object]) -> None:
    """路由算法效果评测明细表。"""
    ax.axis("off")
    if not routes:
        ax.set_title("无路由评测结果", fontsize=10, color=C_MUTED)
        return
    rows = []
    for ra in routes.values():
        rows.append([
            ra.label,
            f"{ra.mean_delay_ms:.1f}",
            f"{ra.p95_delay_ms:.1f}",
            f"{getattr(ra, 'jitter_ms', 0.0):.1f}",
            f"{ra.mean_hops:.1f}",
            f"{ra.success_rate*100:.0f}%",
            f"{ra.switch_count}",
            f"{ra.switch_rate:.2f}",
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["路由算法", "均时延(ms)", "P95", "抖动", "跳数", "可达率",
                   "切换次数", "切换/秒"],
        colWidths=[0.20, 0.12, 0.09, 0.09, 0.08, 0.10, 0.11, 0.11],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    table.scale(1.0, 1.35)
    ax.set_title("路由算法效果评测（时延 / 稳定性 / 可达性）",
                 fontsize=9, color="#111827")


def plot_route_eval(ax, routes: Dict[str, object]) -> None:
    """路由效果：可达率 / 路径切换率 / 时延抖动 三指标对比。"""
    if not routes:
        _placeholder(ax, "无路由评测结果")
        return
    names = [ra.label for ra in routes.values()]
    succ = [ra.success_rate * 100.0 for ra in routes.values()]
    swr = [ra.switch_rate for ra in routes.values()]
    jit = [getattr(ra, "jitter_ms", 0.0) for ra in routes.values()]
    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, succ, w, color=C_PRIMARY, label="可达率 (%)")
    ax.bar(x, swr, w, color=C_ACCENT, label="路径切换 (/s)")
    ax.bar(x + w, jit, w, color=C_ALT, alpha=0.85, label="时延抖动 (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=12)
    ax.set_title("路由稳定性与可达性", fontsize=10, color="#111827")
    ax.legend(fontsize=6.5, ncol=3, loc="upper center")
    _style(ax)


def plot_route_cdf(ax, routes: Dict[str, object]) -> None:
    """各路由协议的端到端时延 CDF（比均值更能说明尾部行为）。"""
    drew = False
    for k, ra in enumerate(routes.values()):
        d = np.asarray(ra.delays, dtype=float)
        d = d[np.isfinite(d)]
        if not d.size:
            continue
        d = np.sort(d)
        cdf = np.arange(1, d.size + 1) / d.size
        ax.plot(d, cdf, lw=1.5, color=PALETTE[k % len(PALETTE)],
                label=f"{ra.label}（P95 {ra.p95_delay_ms:.0f} ms）")
        drew = True
    if not drew:
        _placeholder(ax, "无有效时延样本")
        return
    ax.set_xlabel("端到端单向时延 (ms)", fontsize=8)
    ax.set_ylabel("累积概率", fontsize=8)
    ax.set_title("时延分布 CDF", fontsize=10, color="#111827")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5)
    _style(ax)


def plot_route_switch_timeline(ax, routes: Dict[str, object]) -> None:
    """路径切换事件时间线：直观展示各算法的路径稳定性差异。"""
    drew = False
    for k, ra in enumerate(routes.values()):
        paths = getattr(ra, "paths", [])
        if len(paths) < 2:
            continue
        times = np.arange(len(paths), dtype=float) * max(
            getattr(ra, "sample_interval_s", 10.0), 1e-6)
        changes = [times[i + 1] for i in range(len(paths) - 1)
                   if list(paths[i]) != list(paths[i + 1])]
        if not changes:
            continue
        ax.scatter(changes, np.full(len(changes), k), s=16,
                   color=PALETTE[k % len(PALETTE)], marker="|", lw=1.6)
        drew = True
    if not drew:
        _placeholder(ax, "评测窗口内无路径切换（路由稳定）")
        return
    ax.set_yticks(range(len(routes)))
    ax.set_yticklabels([ra.label for ra in routes.values()], fontsize=7)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_title("路径切换事件时间线（越少越稳定）", fontsize=10, color="#111827")
    _style(ax)


def plot_route_transport(ax, routes: Dict[str, object]) -> None:
    """路由 × 真实传输：各路由算法所选路径上的真实 TCP 端到端吞吐。"""
    items = []
    for ra in routes.values():
        g = float(getattr(ra, "tcp_goodput_mbps", float("nan")))
        if np.isfinite(g):
            items.append((ra.label, g, int(getattr(ra, "tcp_retransmits", 0)),
                          float(getattr(ra, "tcp_completion_s", float("nan")))))
    if not items:
        _placeholder(ax, "未启用路由 × 传输联合评测")
        return
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    x = np.arange(len(names))
    ax.bar(x, vals, 0.55, color=C_PRIMARY, alpha=0.9, label="真实 TCP 吞吐")
    for xi, (_n, v, rt, _c) in zip(x, items):
        ax.text(xi, v * 1.03, f"{v:.1f}\n重传 {rt}", ha="center", va="bottom",
                fontsize=6, color="#111827")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5, rotation=14)
    ax.set_ylabel("真实 TCP 吞吐 (Mbps)", fontsize=8)
    ax.set_title("路由 × 真实传输联合评测（同载荷同算法）",
                 fontsize=9, color="#111827")
    _style(ax)


def plot_tcp_loss_sweep(ax, sweep: Dict[str, List[float]],
                        points: Sequence[float]) -> None:
    """链路丢包率 × 真实 TCP 吞吐：丢包型与模型型拥塞控制的分化。"""
    if not sweep or not len(points):
        _placeholder(ax, "未运行丢包率扫描")
        return
    xs = np.asarray(list(points), dtype=float) * 100.0
    drew = False
    for k, algo in enumerate(sorted(sweep)):
        ys = np.asarray(sweep[algo], dtype=float)
        if ys.size != xs.size:
            continue
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5,
                color=PALETTE[k % len(PALETTE)], label=algo.upper())
        drew = True
    if not drew:
        _placeholder(ax, "扫描数据不完整")
        return
    ax.set_xlabel("链路丢包率 (%/包)", fontsize=8)
    ax.set_ylabel("有效吞吐 (Mbps)", fontsize=8)
    ax.set_title("真实 TCP 丢包敏感性（LEO 残余误码）",
                 fontsize=9, color="#111827")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25, lw=0.5)
    _style(ax)


# --------------------------------------------------------------------------- #
# ⑫ 全局离散事件内核
# --------------------------------------------------------------------------- #
_DES_KIND_COLOR = {
    "sim:start": C_MUTED, "topo:refresh": C_PRIMARY,
    "route:recompute": C_ALT, "path:switch": C_WARN,
    "flow:arrive": "#7C3AED", "flow:done": "#059669",
    "flow:switch": "#DC2626", "flow:timeout": C_WARN,
    "task:arrive": C_ACCENT, "task:done": "#B45309",
    "metric:sample": "#9CA3AF",
}


def plot_des_timeline(ax, des) -> None:
    """事件时间线甘特图：每类实体一条泳道，事件画成刻度点。

    直接消费 ``des.trace.records``；长仿真记录被环形缓冲截断时仍如实标注。
    """
    ax.axis("off")
    if des is None or not des.trace.records:
        _placeholder(ax, "未启用事件内核（勾选「离散事件编排」后运行）")
        return
    kinds = [k for k in ("topo:refresh", "route:recompute", "path:switch",
                         "flow:arrive", "flow:switch", "flow:done", "task:arrive",
                         "task:done", "metric:sample") if k in des.trace.counts]
    height = max(len(kinds), 1)
    for row, kind in enumerate(kinds):
        y = height - row
        ax.hlines(y, 0, des.sim_end_s, color="#E5E7EB", lw=3, zorder=1)
        ts = [r.t for r in des.trace.records if r.kind == kind]
        if ts:
            ax.plot(ts, [y] * len(ts), linestyle="none", marker="|", ms=7,
                    mew=1.6, color=_DES_KIND_COLOR.get(kind, C_PRIMARY),
                    zorder=2)
    ax.set_yticks([height - i for i in range(len(kinds))])
    ax.set_yticklabels(kinds, fontsize=6.5, fontfamily=cjk_font_family())
    ax.set_xlim(0, max(des.sim_end_s, 1.0))
    ax.set_ylim(0.3, height + 0.7)
    ax.set_xlabel("仿真时间 (s)", fontsize=8)
    n_total = int(sum(des.trace.counts.values()))
    ax.set_title(f"事件时间线（执行 {des.n_executed} 个事件，"
                 f"轨迹记录 {n_total} 条）", fontsize=9, color="#111827")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def plot_des_events(ax, des) -> None:
    """事件类型构成（对数柱状图）。"""
    if des is None or not des.trace.counts:
        _placeholder(ax, "无事件统计")
        return
    items = des.event_kind_table()
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [_DES_KIND_COLOR.get(k, C_PRIMARY) for k in labels]
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=6,
                       fontfamily=cjk_font_family())
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                ha="center", va="bottom", fontsize=6, color=C_MUTED)
    ax.set_ylabel("事件数（对数）", fontsize=8)
    ax.set_title("事件类型构成", fontsize=9, color="#111827")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    _style(ax)


def plot_des_queue(ax, des) -> None:
    """计算任务流：累计到达 vs 累计完成（阶梯线）+ 队列深度（右轴）。

    星载算力相对任务规模通常富余，队列深度长期为 0——单独画深度没有
    信息量，叠加"到达/完成"累计曲线才能看出任务流的真实节奏。
    """
    if des is None or not des.tasks:
        _placeholder(ax, "无任务事件")
        return
    arr = np.asarray([t.arrival_s for t in des.tasks])
    fin = np.asarray([t.finish_s for t in des.tasks])
    xs = np.linspace(0.0, max(des.sim_end_s, fin.max() if fin.size else 1.0), 200)
    ax.plot(xs, np.searchsorted(np.sort(arr), xs), color=C_ACCENT, lw=1.4,
            label="累计到达")
    ax.plot(xs, np.searchsorted(np.sort(fin), xs), color=C_PRIMARY, lw=1.4,
            label="累计完成")
    ax.set_xlabel("仿真时间 (s)", fontsize=8)
    ax.set_ylabel("任务数（累计）", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    _style(ax)
    if des.queue_series and max(d for _, d in des.queue_series) > 0:
        ax2 = ax.twinx()
        qx = [t for t, _ in des.queue_series]
        qy = [d for _, d in des.queue_series]
        ax2.plot(qx, qy, color=C_WARN, lw=1.0, alpha=0.75,
                 drawstyle="steps-post", label="队列深度")
        ax2.set_ylabel("排队任务数", fontsize=8, color=C_WARN)
        ax2.tick_params(axis="y", labelsize=6.5, colors=C_WARN)
        ax2.spines["right"].set_color(C_WARN)
    wait_p95 = des.task_p95_wait_s
    ax.set_title(f"事件级任务流（等待均值 {des.task_mean_wait_s:.3f} s / "
                 f"P95 {wait_p95:.3f} s）", fontsize=9, color="#111827")


def des_table(ax, des) -> None:
    """事件内核摘要表：流、任务、拓扑事件三类口径。"""
    ax.axis("off")
    if des is None:
        _placeholder(ax, "未启用事件内核")
        return
    rows = [[k, v] for k, v in des.headline()]
    table = ax.table(cellText=rows, colLabels=["指标", "数值"],
                     colWidths=[0.30, 0.70], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.30)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E6F4F3")
            cell.set_text_props(weight="bold", color="#0E615F")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 else "#F6FAFA")
    ax.set_title("离散事件内核摘要", fontsize=9, color="#111827")


def text_report(result) -> str:
    """把仿真结果整理成一段可直接粘进报告的文字。"""
    lines: List[str] = []
    lines.append("SatNetSim 仿真结果摘要")
    lines.append("=" * 52)
    for k, v in result.headline():
        lines.append(f"{k:<12}{v}")
    lines.append("")
    if result.routes:
        lines.append("路由协议")
        lines.append(f"{'协议':<12}{'均时延(ms)':>12}{'P95':>10}{'跳数':>8}{'切换':>8}{'可达率':>9}")
        for ra in result.routes.values():
            lines.append(
                f"{ra.label:<12}{ra.mean_delay_ms:>12.2f}{ra.p95_delay_ms:>10.2f}"
                f"{ra.mean_hops:>8.1f}{ra.switch_count:>8d}{ra.success_rate*100:>8.1f}%"
            )
        lines.append("")
        lines.append("路由算法效果评测")
        lines.append(f"{'算法':<12}{'中位(ms)':>10}{'抖动(ms)':>11}{'P95-P5':>9}"
                     f"{'跳数σ':>8}{'切换/秒':>10}{'可达快照':>10}{'真实TCP':>12}")
        for ra in result.routes.values():
            tcp_g = getattr(ra, "tcp_goodput_mbps", float("nan"))
            tcp_txt = f"{tcp_g:.1f} Mbps" if np.isfinite(tcp_g) else "—"
            lines.append(
                f"{ra.label:<12}{getattr(ra,'median_delay_ms',float('nan')):>10.1f}"
                f"{getattr(ra,'jitter_ms',0.0):>11.2f}"
                f"{getattr(ra,'delay_spread_ms',0.0):>9.2f}"
                f"{getattr(ra,'hop_std',0.0):>8.2f}"
                f"{ra.switch_rate:>10.3f}{getattr(ra,'reachable_snapshots',0):>10d}"
                f"{tcp_txt:>12}")
        lines.append("  抖动 = 端到端时延标准差；切换/秒 = 拓扑时变下路由重算频度；")
        lines.append("  可达率 < 100% 表示该快照下源目的不可达（覆盖缺口或拓扑断开）；")
        lines.append("  真实TCP = 该路由算法所选路径上跑真实包级 TCP 的端到端吞吐（同载荷同算法）。")
        lines.append("")
    tcp_flows = getattr(result, "tcp_flows", None) or []
    if tcp_flows:
        cap = getattr(result, "tcp_capacity", {}) or {}
        prof = getattr(result, "tcp_profile", None)
        lines.append("真实包级 TCP（完整状态机：慢启动 / 快速重传 / RTO 退避 / SACK / 流控）")
        if prof is not None:
            lines.append("  路径剖面：" + prof.describe())
        if cap:
            lines.append(
                f"  平均 BDP {cap.get('mean_bdp_bytes', 0)/1024:.0f} KB"
                f"（{cap.get('mean_bdp_segments', 0):.0f} 段），"
                f"窗口受限吞吐上限 {cap.get('window_limited_mbps', 0):.0f} Mbps")
        lines.append(f"{'算法':<10}{'完成(s)':>10}{'吞吐(Mbps)':>12}{'效率':>8}"
                     f"{'重传率':>9}{'超时':>7}{'快重':>7}{'RTTmin':>9}{'RTTp95':>9}")
        for f in tcp_flows:
            comp = f"{f.completion_s:.3f}" if np.isfinite(f.completion_s) else "未完成"
            lines.append(
                f"{f.protocol.upper():<10}{comp:>10}{f.goodput_mbps:>12.2f}"
                f"{f.utilization*100:>7.0f}%{f.retransmit_ratio*100:>8.1f}%"
                f"{f.timeouts:>7d}{f.fast_retransmits:>7d}"
                f"{f.rtt_min_ms:>9.0f}{f.rtt_p95_ms:>9.0f}")
        lines.append("")
        sweep = getattr(result, "tcp_sweep", None) or {}
        pts = getattr(result, "tcp_sweep_points", None) or []
        if sweep and pts:
            lines.append("真实 TCP 丢包率敏感性（每包丢包率 → 有效吞吐 Mbps）")
            lines.append("  丢包率%   " + "".join(f"{p*100:>10.2f}" for p in pts))
            for algo in sorted(sweep):
                ys = sweep[algo]
                if len(ys) != len(pts):
                    continue
                lines.append(f"  {algo.upper():<9}" + "".join(
                    (f"{y:>10.1f}" if np.isfinite(y) else f"{'未完成':>10}") for y in ys))
            lines.append("  说明：丢包型（CUBIC/HYBLA）会把误码误判为拥塞而大幅降窗，")
            lines.append("        模型型（BBR）不受影响 —— 这正是 BBR 类算法在卫星链路上的机理优势。")
            lines.append("")
    if result.transport:
        lines.append("解析模型（RTT 轮次积分，作为同口径对照）")
        lines.append(f"{'协议':<12}{'完成(ms)':>12}{'吞吐(Mbps)':>12}{'效率':>9}")
        for f in result.transport:
            lines.append(
                f"{f.label:<12}{f.completion_s*1000:>12.2f}"
                f"{f.goodput_mbps:>12.2f}{f.efficiency*100:>8.1f}%"
            )
        lines.append("")
    if result.compute:
        lines.append("星载计算架构")
        lines.append(f"{'策略':<16}{'均时延(s)':>11}{'P95(s)':>10}{'超期率':>9}"
                     f"{'卸载率':>9}{'下行(Mbit)':>14}")
        for k, rep in result.compute.items():
            s = rep.summary_row()
            lines.append(
                f"{k:<16}{s['mean_latency_s']:>11.3f}{s['p95_latency_s']:>10.3f}"
                f"{s['deadline_miss_pct']:>8.1f}%{s['offload_pct']:>8.0f}%"
                f"{s['downlink_mbit']:>14.1f}"
            )
        lines.append("")
    rt = getattr(result, "real_transfer", None)
    if rt is not None:
        lines.append("")
        lines.append("真实数据传输评测（对照解析模型）")
        lines.append("  " + rt.profile.describe())
        lines.append(f"  实测 RTT {rt.measured_rtt_ms:.1f} ms（理论 "
                     f"{rt.profile.rtt_ms:.1f} ms）；载荷 {rt.payload_mbit:.1f} Mbit")
        if rt.packet_profile is not None:
            lines.append("  " + rt.packet_profile.describe())
            lines.append(f"  包级载荷 {rt.packet_payload_mbit:.0f} Mbit，"
                         f"用户态测试床可靠容量 ~{rt.harness_cap_mbps:.0f} Mbps")
        lines.append(f"  {'方法':<26}{'完成(s)':>10}{'吞吐(Mbps)':>12}"
                     f"{'重传':>7}  完整性")
        for name, comp, tput, retx, integ in rt.summary_rows():
            lines.append(f"  {name:<26}{comp:>10}{tput:>12}{retx:>7}  {integ}")
    rc = getattr(result, "real_compute", None)
    if rc is not None:
        lines.append("")
        lines.append("真实计算任务评测")
        for k, v in rc.headline():
            lines.append(f"  {k:<14}{v}")
        lines.append(f"  任务画像：{rc.mix_text()}")
        lines.append(f"  {'真实负载':<28}{'耗时(ms)':>10}{'吞吐':>16}"
                     f"{'GFLOPS':>10}  真实产出")
        for name, ms, tput, gf, out in rc.summary_rows():
            lines.append(f"  {name:<28}{ms:>10}{tput:>16}{gf:>10}  {out}")
    rd = getattr(result, "real_data", None)
    if rd is not None:
        lines.append("")
        lines.append("真实数据流仿真（真字节 × 真拓扑 × 逐包多跳）")
        for k, v in rd.headline():
            lines.append(f"  {k:<14}{v}")
        lines.append(f"  载荷详情：{rd.payload_mime}，MSS {rd.mss_bytes} B，"
                     f"{rd.n_segments} 分片")
        for f in rd.flows:
            lines.append(f"    流{f.flow_id}: {f.src_label} → {f.dst_label}"
                         f"   {f.rate_mbps:.2f} Mbps")
        lines.append(f"  链路负载 Top5（含丢包）：")
        for s in rd.link_stats[:5]:
            lines.append(f"    {s.edge[0]:>4}→{s.edge[1]:<4} 包 {s.packets:>6}  "
                         f"{s.bytes/1e6:>8.2f} Mbit  丢 {s.drops:>4}  "
                         f"峰值队列 {s.peak_queue_bytes/1024:>7.0f} KiB")
    des = getattr(result, "des", None)
    if des is not None:
        lines.append("")
        lines.append("全局离散事件内核（NS-3 式统一事件时间轴）")
        for k, v in des.headline():
            lines.append(f"  {k:<14}{v}")
        kinds = des.event_kind_table()
        if kinds:
            lines.append("  事件构成：" + "，".join(f"{k}×{n}" for k, n in kinds))
        if des.flows:
            lines.append(f"  {'流':<6}{'到达(s)':>9}{'传输时长(s)':>12}"
                         f"{'吞吐(Mbps)':>12}{'重传':>7}")
            for f in des.flows:
                comp = (f.completion_s
                        if f.done and np.isfinite(f.completion_s) else float("nan"))
                lines.append(
                    f"  流{f.idx:<4}{f.arrival_s:>9.1f}{comp:>12.2f}"
                    f"{f.goodput_mbps:>12.1f}{f.retransmits:>7d}")
        if des.node_utilization:
            util_txt = "，".join(f"{n} {u*100:.0f}%" for n, u in
                                 sorted(des.node_utilization.items())[:6])
            lines.append(f"  节点利用率：{util_txt}")
    if result.warnings:
        lines.append("")
        lines.append("告警：" + "；".join(result.warnings))
    return "\n".join(lines)

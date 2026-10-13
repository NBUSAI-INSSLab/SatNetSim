"""仿真引擎：把各层插件按统一时间轴串起来。

执行流水线
----------
    场景配置
      ↓ ① 星座生成（Walker-Delta / TLE）
      ↓ ② 轨道传播（解析 J2 / SGP4）→ ECI → ECEF → LLA
      ↓ ③ 拓扑快照（ISL 构型 + 星地可见性 + 链路时延/容量）
      ↓ ④ 路由分析（多协议 × 全快照 → 时延、跳数、路径切换率）
      ↓ ⑤ 传输分析（沿代表路径做多协议流级对比，**解析模型**）
      ↓ ⑥ 星载计算（任务卸载 + 调度 + 能耗，含弯管基线）
      ↓ ⑦ 真实评测【可选】真实数据传输（内核 TCP + 用户态逐包可靠传输）
                         + 真实计算任务（真跑算子）+ 解析模型同口径对照
      ↓ 结果对象（供 GUI / 报告 / 批量扫参使用）

第 ③ 步之后的每一层都是**纯函数式**的：给定快照即可复算，因此天然支持
"同一拓扑 × 多协议" 的矩阵式对比，也便于并行化（进程池按快照切分）。

第 ⑦ 步是唯一**非纯函数**的一步：它真的把字节推过本机内核协议栈、真的执行
算子，因此结果与本机硬件/系统状态相关，且耗时远高于前六步（数十秒量级）。
它默认关闭（`cfg.real_eval.enabled=False`），作用是把第 ⑤/⑥ 步的解析假设
放到真实数据上当面对照——真实评测与解析模型的排序一致，解析模型才可信。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .computing import ComputeReport, Task, compare_architectures
from .config import SimConfig
from .dessim import DesReport, run_des
from .orbit import Constellation, sun_direction_eci, sun_eclipse_fraction, unix_from_iso
from .reallink import (
    LinkProfile,
    RealTransferReport,
    evaluate_transfer,
    make_profile,
)
from .realdatastream import DataFlowReport, simulate_real_data
from .realworkload import RealComputeReport, evaluate_real_compute
from .routing import (
    RouteResult,
    ROUTERS,
    make_router,
    path_switch_count,
)
from .topology import Snapshot, TopologyBuilder, topology_stats
from .tcpstack import (
    PathProfile,
    TcpFlowResult,
    compare_flows,
    default_tcp_config,
    path_capacity_summary,
    profile_from_route,
    simulate_flow,
)
from .transport import FlowResult, TransportSimulator


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass
class RouteAnalysis:
    protocol: str
    label: str
    mean_delay_ms: float
    p95_delay_ms: float
    mean_hops: float
    max_hops: int
    switch_count: int
    switch_rate: float  # 每秒路径切换次数
    success_rate: float
    # ---- 效果评测补充指标 ----
    median_delay_ms: float = float("nan")
    jitter_ms: float = 0.0          # 端到端时延标准差（抖动）
    delay_spread_ms: float = 0.0    # P95 − P5
    hop_std: float = 0.0
    reachable_snapshots: int = 0
    sample_interval_s: float = 10.0
    # ---- 路由 × 真实传输 联合评测 ----
    tcp_goodput_mbps: float = float("nan")
    tcp_completion_s: float = float("nan")
    tcp_retransmits: int = 0
    tcp_path_note: str = ""
    delays: List[float] = field(default_factory=list)
    hops: List[int] = field(default_factory=list)
    paths: List[List[int]] = field(default_factory=list)
    all_paths: List[List[int]] = field(default_factory=list)
    examples: List[List[int]] = field(default_factory=list)


@dataclass
class SimulationResult:
    config: SimConfig
    constellation: Constellation
    snapshots: List[Snapshot]
    topo_stats: Dict[str, float]
    routes: Dict[str, RouteAnalysis]
    transport: List[FlowResult]
    compute: Dict[str, ComputeReport]
    contention: Tuple[List[float], Dict[str, List[float]]]
    representative_path: List[int]
    path_owd_ms: float
    path_bottleneck_mbps: float
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # ---------------- ⑦ 真实评测（可选） ---------------- #
    real_transfer: Optional[RealTransferReport] = None
    real_compute: Optional[RealComputeReport] = None
    real_tasks: List[Task] = field(default_factory=list)
    real_profile: Optional[LinkProfile] = None

    # ---------------- ⑧ 真实数据流（可选） ---------------- #
    real_data: Optional["DataFlowReport"] = None

    # ---------------- ⑤ 真实传输（包级 TCP 状态机） ---------------- #
    tcp_flows: List[TcpFlowResult] = field(default_factory=list)
    tcp_profile: Optional[PathProfile] = None
    tcp_capacity: Dict[str, float] = field(default_factory=dict)
    route_cross: Dict[str, TcpFlowResult] = field(default_factory=dict)
    tcp_sweep: Dict[str, List[float]] = field(default_factory=dict)
    tcp_sweep_points: List[float] = field(default_factory=list)
    transport_engine: str = "analytic"

    # ---------------- ⑫ 全局离散事件编排（可选） ---------------- #
    des: Optional[DesReport] = None

    @property
    def has_des(self) -> bool:
        return self.des is not None

    @property
    def has_real_eval(self) -> bool:
        return self.real_transfer is not None or self.real_compute is not None

    @property
    def has_real_data(self) -> bool:
        return self.real_data is not None

    # ---------------- 便捷视图 ---------------- #
    def headline(self) -> List[Tuple[str, str]]:
        c = self.config
        rows = [
            ("星座", f"{c.constellation.name}  {c.constellation.planes}×"
                     f"{c.constellation.sats_per_plane} = {c.constellation.n_satellites} 星"),
            ("轨道", f"{c.constellation.altitude_km:.0f} km / "
                     f"{c.constellation.inclination_deg:.1f}°"),
            ("ISL 构型", f"{c.link.isl_mode}  "
                        f"{c.link.isl_bandwidth_mbps/1000:.1f} Gbps"),
            ("快照数", str(len(self.snapshots))),
        ]
        rows.append(("平均 ISL 度", f"{self.topo_stats.get('mean_isl_degree', 0):.2f}"))
        if self.routes:
            best = min(self.routes.values(), key=lambda r: r.mean_delay_ms)
            rows.append(("最优路由", f"{best.label}  {best.mean_delay_ms:.1f} ms"))
        rows.append(("代表路径 OWD", f"{self.path_owd_ms:.1f} ms "
                                      f"(瓶颈 {self.path_bottleneck_mbps:.0f} Mbps)"))
        if self.transport:
            best_t = min(self.transport, key=lambda t: t.completion_s)
            rows.append(("最快传输", f"{best_t.label}  {best_t.completion_s:.2f} s "
                                     f"({best_t.goodput_mbps:.0f} Mbps)"))
        # ---- ⑤ 真实包级 TCP 摘要 ----
        if self.tcp_flows:
            ok = [f for f in self.tcp_flows if np.isfinite(f.completion_s)]
            if ok:
                best_c = max(ok, key=lambda f: f.goodput_mbps)
                rows.append(("真实 TCP 最优", f"{best_c.label}  "
                                             f"{best_c.goodput_mbps:.1f} Mbps / "
                                             f"{best_c.completion_s:.2f} s "
                                             f"(效率 {best_c.utilization*100:.0f}%)"))
            if self.tcp_capacity:
                rows.append(("路径 BDP", f"{self.tcp_capacity.get('mean_bdp_bytes', 0)/1024:.0f} KB"
                                        f"（{self.tcp_capacity.get('mean_bdp_segments', 0):.0f} 段）"
                                        f"，窗口上限 "
                                        f"{self.tcp_capacity.get('window_limited_mbps', 0):.0f} Mbps"))
        if self.routes:
            st = min(self.routes.values(), key=lambda r: r.switch_rate)
            rows.append(("最稳路由", f"{st.label}  切换 {st.switch_count} 次 "
                                     f"({st.switch_rate:.2f}/s)，抖动 "
                                     f"{getattr(st, 'jitter_ms', 0.0):.1f} ms"))
        if "value_density" in self.compute:
            cr = self.compute["value_density"]
            rows.append(("星载调度", f"均值时延 {cr.mean_latency_s:.2f} s, "
                                     f"超期率 {cr.deadline_miss_rate*100:.1f}%, "
                                     f"卸载率 {cr.offload_ratio*100:.0f}%"))
        # ---- 真实评测摘要 ----
        rt = self.real_transfer
        if rt is not None:
            if rt.tcp:
                t = max(rt.tcp, key=lambda x: x.goodput_mbps)
                rows.append(("真实 TCP 实测", f"{t.goodput_mbps:.0f} Mbps, "
                                             f"{t.completion_s:.3f} s, {t.integrity}"))
            bp = rt.best_packet()
            if bp is not None:
                rows.append(("真实包级最优", f"{bp.label}  {bp.goodput_mbps:.1f} Mbps, "
                                            f"重传 {bp.retransmits}, {bp.integrity}"))
            if rt.analytic:
                k = min(rt.analytic, key=lambda x: rt.analytic[x])
                rows.append(("解析模型最优", f"{k}  {rt.analytic[k]:.3f} s / "
                                            f"{rt.analytic_goodput.get(k, 0):.0f} Mbps"))
        rc = self.real_compute
        if rc is not None:
            rows.append(("真实计算负载", f"{rc.task_count} 个真实任务 / "
                                         f"{rc.total_gflops:.1f} GFLOP / "
                                         f"本机峰值 {rc.calibration.peak_gflops():.0f} GFLOPS"))
            rows.append(("地面/星载耗时", f"{rc.ground_total_s:.3f} s vs "
                                         f"{rc.on_orbit_total_s:.3f} s "
                                         f"({rc.speedup:.1f}×)"))
        rd = self.real_data
        if rd is not None:
            rows.append(("真实数据流", f"{rd.payload_label} {rd.payload_bytes/1024:.0f} KiB → "
                                       f"{rd.n_delivered}/{rd.n_packets} 分组"))
            rows.append(("端到端时延", f"均值 {rd.e2e_mean_s*1000:.1f} ms / "
                                       f"P95 {rd.e2e_p95_s*1000:.1f} ms / "
                                       f"{rd.mean_hops:.1f} 跳"))
            rows.append(("流完整性", f"{rd.integrity}（吞吐 {rd.goodput_mbps:.2f} Mbps）"))
        # ---- ⑫ 全局离散事件编排摘要 ----
        if self.des is not None:
            rows.append(("事件内核", f"{self.des.n_executed} 个事件 / "
                                     f"时间轴 {self.des.sim_end_s:.0f} s / "
                                     f"{len(self.des.trace.counts)} 类"))
            rows.append(("事件级 TCP 流", f"{self.des.flows_done}/{len(self.des.flows)} 条完成"
                                          f"（均值 {self.des.flow_goodput_mean_mbps:.1f} Mbps）"))
            rows.append(("事件级任务", f"{len(self.des.tasks)} 个到达，均值等待 "
                                       f"{self.des.task_mean_wait_s:.3f} s，"
                                       f"时限达标 {self.des.task_deadline_rate*100:.0f}%"))
        return rows


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class Simulation:
    """一体化仿真器。使用方式::

        cfg = preset("starlink-lite")
        result = Simulation(cfg).run(progress=print)
    """

    ALL_PROTOCOLS = ("shortest_delay", "shortest_hop", "load_balanced",
                     "virtual_node", "geographic")

    def __init__(self, config: SimConfig):
        self.cfg = config
        self.warnings: List[str] = []

    # ------------------------------------------------------------------ #
    def run(
        self,
        progress: Optional[Callable[[str, float], None]] = None,
        protocols: Optional[Sequence[str]] = None,
        run_compute: bool = True,
        run_real: Optional[bool] = None,
        run_real_data: Optional[bool] = None,
    ) -> SimulationResult:
        """跑完整流水线。

        run_real       : None → 跟随 ``cfg.real_eval.enabled``；True/False 强制开关
                         第 ⑦ 步真实评测（真实数据传输 + 真实计算任务）。
        run_real_data  : None → 跟随 ``cfg.real_data.enabled``；True/False 强制开关
                         第 ⑧ 步真实数据流（真字节逐包多跳转发，供实时回放）。
        """
        cfg = self.cfg
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}

        def tick(msg: str, frac: float) -> None:
            if progress:
                progress(msg, frac)

        # ---------------- ① 星座 ---------------- #
        tick("生成星座…", 0.02)
        constellation = Constellation(cfg.constellation)

        # ---------------- ② 轨道传播 ---------------- #
        tick("轨道传播…", 0.10)
        t0 = time.perf_counter()
        dt = max(1.0, cfg.snapshot_interval_s / 2.0)
        times = np.arange(0.0, cfg.duration_s + 1e-9, dt)
        if times.size < 2:
            times = np.array([0.0, cfg.duration_s])
        start_unix = unix_from_iso(cfg.start_utc)
        constellation.propagate(times)
        constellation.to_ecef(start_unix)
        timings["propagate"] = time.perf_counter() - t0

        # 太阳方向 → 日照比例（供星载能量预算）
        sun = sun_direction_eci(start_unix + cfg.duration_s / 2.0)
        solar_fraction = sun_eclipse_fraction(constellation.positions_eci, sun)

        # ---------------- ③ 拓扑快照 ---------------- #
        tick("构建时变拓扑快照…", 0.25)
        t0 = time.perf_counter()
        builder = TopologyBuilder(cfg.link, cfg.ground_stations)
        snapshots = builder.build(
            constellation, times, start_unix, cfg.snapshot_interval_s
        )
        stats = topology_stats(snapshots, constellation.n_sat)
        timings["topology"] = time.perf_counter() - t0
        if stats.get("mean_isl_degree", 0) < 1.0:
            self.warnings.append(
                "平均 ISL 度数 < 1，星座可能未形成连通网络（检查 ISL 最大距离/构型）"
            )

        n_gs = len(cfg.ground_stations)
        if n_gs < 2:
            self.warnings.append("地面站少于 2 个，无法建立端到端路径，路由分析将跳过")

        # ---------------- ④ 路由分析 ---------------- #
        routes: Dict[str, RouteAnalysis] = {}
        rep_path: List[int] = []
        rep_owd = 0.0
        rep_bw = 0.0
        src_node = dst_node = -1
        src_label, dst_label = "—", "—"
        rep_paths: List[List[int]] = []
        vadj = np.zeros((0, 2), dtype=int)
        if n_gs >= 2:
            tick("路由协议对比…", 0.45)
            t0 = time.perf_counter()
            proto_list = list(protocols or self._default_protocols())
            # 选一对地面站做端到端分析：默认取球面距离最远的一对（最能体现 ISL 多跳）
            src_gs, dst_gs = self._farthest_gs_pair(cfg.ground_stations)
            src_node = constellation.n_sat + src_gs
            dst_node = constellation.n_sat + dst_gs
            src_label = getattr(cfg.ground_stations[src_gs], "name", f"GS{src_gs}")
            dst_label = getattr(cfg.ground_stations[dst_gs], "name", f"GS{dst_gs}")
            vadj = builder.virtual_topology(constellation)
            positions = constellation.positions_ecef  # (n_sat, n_t, 3)
            grid_dt = dt
            for proto in proto_list:
                router = make_router(proto, cfg.routing)
                ra = self._analyse_route(
                    router, snapshots, constellation, src_node, dst_node,
                    vadj, cfg.duration_s, positions, grid_dt,
                )
                routes[proto] = ra
            timings["routing"] = time.perf_counter() - t0

            # 逐快照路径（供第 ⑤ 步真实 TCP 栈构造时变路径剖面）
            rep_router = make_router(cfg.routing.protocol, cfg.routing)
            for k, snap in enumerate(snapshots):
                gi = int(np.clip(round(k * cfg.snapshot_interval_s / dt),
                                 0, positions.shape[1] - 1))
                rr = rep_router.route(
                    snap, src_node, dst_node,
                    {"positions": positions[:, gi, :], "virtual_adj": vadj})
                rep_paths.append(list(rr.path) if rr.ok else [])

            # 代表路径：时延最短路在第一个快照上的结果
            base = routes.get("shortest_delay") or next(iter(routes.values()))
            if base.paths:
                rep_path = base.paths[0]
                # 用平均快照重算该路径的 OWD 与瓶颈带宽
                snap = snapshots[len(snapshots) // 2]
                delay, hops, bw = self._path_metrics(snap, rep_path)
                rep_owd = delay / 2.0 if np.isfinite(delay) else base.mean_delay_ms / 2.0
                rep_bw = bw if np.isfinite(bw) and bw > 0 else cfg.link.gsl_bandwidth_mbps

        # ---------------- ⑤ 传输分析 ---------------- #
        tick("传输协议对比…", 0.60)
        t0 = time.perf_counter()
        owd = rep_owd if rep_owd > 0 else 30.0
        bn = rep_bw if rep_bw > 0 else min(cfg.link.gsl_bandwidth_mbps,
                                           cfg.link.isl_bandwidth_mbps)
        ts = TransportSimulator(cfg.transport)
        size_mbit = cfg.flows[0].get("size_mbit", 500.0) if cfg.flows else 500.0
        n_hops = max(len(rep_path) - 1, 1) if rep_path else 1
        engine_mode = str(getattr(cfg.transport, "engine", "real")).lower()
        tcp_flows: List[TcpFlowResult] = []
        tcp_profile: Optional[PathProfile] = None
        tcp_capacity: Dict[str, float] = {}
        route_cross: Dict[str, TcpFlowResult] = {}
        tcfg = default_tcp_config(cfg.transport)

        # ---- ⑤.1 真实包级 TCP：在**时变路径**上跑完整协议状态机 ----
        if engine_mode in ("real", "both") and rep_paths:
            tick("真实 TCP 包级仿真（慢启动/快速重传/RTO/SACK 状态机）…", 0.63)
            tcp_profile = profile_from_route(
                snapshots, paths=rep_paths, times=list(times),
                loss_prob=cfg.transport.packet_loss_prob,
                src_label=src_label, dst_label=dst_label,
                nominal_rtt_ms=2.0 * owd,
            )
            tcp_capacity = path_capacity_summary(tcp_profile, cfg.transport.mss_bytes)
            tcp_capacity["window_limited_mbps"] = (
                float(tcfg.rcvbuf_bytes) * 8.0
                / max(float(np.mean(tcp_profile.rtt_ms)) / 1000.0, 1e-6) / 1e6)
            algos = tuple(str(a).lower() for a in
                          (cfg.transport.tcp_algos or ["reno", "cubic", "bbr", "hybla"]))
            try:
                tcp_flows = compare_flows(
                    tcp_profile, float(cfg.transport.app_payload_mbit),
                    algos=algos, cfg=tcfg, seed=cfg.seed,
                    progress=lambda m: tick(m, 0.66),
                )
            except Exception as exc:  # pragma: no cover
                self.warnings.append(f"真实 TCP 包级仿真失败：{exc}")
                tcp_flows = []

        # ---- ⑤.2 真实 TCP 丢包率扫描：展示拥塞控制在 LEO 残余误码下的分化 ----
        tcp_sweep: Dict[str, List[float]] = {}
        tcp_sweep_points: List[float] = []
        if getattr(cfg.transport, "loss_sweep", True) and rep_paths:
            tick("真实 TCP 丢包率扫描…", 0.69)
            tcp_sweep_points = [float(x) for x in cfg.transport.loss_sweep_points]
            algos_s = [str(a).lower() for a in cfg.transport.loss_sweep_algos]
            tcp_sweep = {a: [] for a in algos_s}
            for i, pt in enumerate(tcp_sweep_points):
                prof3 = profile_from_route(
                    snapshots, paths=rep_paths, times=list(times),
                    loss_prob=pt, src_label=src_label, dst_label=dst_label,
                    nominal_rtt_ms=2.0 * owd)
                for a in algos_s:
                    try:
                        fl = simulate_flow(
                            prof3, float(cfg.transport.loss_sweep_mbit),
                            algo=a, cfg=tcfg, seed=cfg.seed + 7 * i)
                        tcp_sweep[a].append(
                            fl.goodput_mbps if np.isfinite(fl.completion_s)
                            else float("nan"))
                    except Exception:  # pragma: no cover
                        tcp_sweep[a].append(float("nan"))

        # ---- ⑤.3 路由 × 真实传输联合评测：每种路由算法各跑一次真实 TCP ----
        if getattr(cfg.transport, "route_sweep", True) and routes:
            tick("路由 × 真实传输联合评测…", 0.70)
            algo0 = str(cfg.transport.protocol).lower()
            if algo0 not in ("reno", "cubic", "bbr", "hybla", "vegas"):
                algo0 = "cubic"
            for _proto, ra in routes.items():
                ap = getattr(ra, "all_paths", None)
                if not ap:
                    continue
                try:
                    prof2 = profile_from_route(
                        snapshots, paths=ap, times=list(times),
                        loss_prob=cfg.transport.packet_loss_prob,
                        src_label=src_label, dst_label=dst_label)
                    fl = simulate_flow(
                        prof2, float(cfg.transport.app_payload_mbit),
                        algo=algo0, cfg=tcfg, seed=cfg.seed)
                    ra.tcp_goodput_mbps = fl.goodput_mbps
                    ra.tcp_completion_s = fl.completion_s
                    ra.tcp_retransmits = fl.retransmits
                    ra.tcp_path_note = prof2.describe()
                    route_cross[ra.label] = fl
                except Exception as exc:  # pragma: no cover
                    self.warnings.append(
                        f"路由「{ra.label}」的传输联合评测失败：{exc}")

        # ---- ⑤.2 解析模型（RTT 轮次积分）：保留作为快速对照 ----
        if engine_mode in ("analytic", "both") or not tcp_flows:
            transport = ts.compare(
                owd, min(bn, 2000.0), float(size_mbit),
                protocols=("reno", "cubic", "bbr", "hybla", "quic", "pep", "dtn"),
                n_trials=3, seed=cfg.seed, n_hops=n_hops,
            )
        else:
            transport = ts.compare(
                owd, min(bn, 2000.0), float(cfg.transport.app_payload_mbit),
                protocols=("quic", "pep", "dtn"),
                n_trials=2, seed=cfg.seed, n_hops=n_hops,
            )
        timings["transport"] = time.perf_counter() - t0

        # ---------------- ⑥ 星载计算 ---------------- #
        compute: Dict[str, ComputeReport] = {}
        contention: Tuple[List[float], Dict[str, List[float]]] = ([], {})
        if run_compute:
            tick("星载计算调度仿真…", 0.72)
            t0 = time.perf_counter()

            def visibility_fn(gs_local: int, t_s: float) -> np.ndarray:
                idx = int(np.clip(t_s // max(cfg.snapshot_interval_s, 1e-9),
                                  0, len(snapshots) - 1))
                return snapshots[idx].gs_visible.get(gs_local, np.zeros(0, dtype=int))

            gs_antennas = max(
                (int(g.antennas) for g in cfg.ground_stations), default=4
            )
            compute = compare_architectures(
                cfg.compute,
                n_sat=constellation.n_sat,
                n_gs=max(n_gs, 1),
                duration_s=cfg.duration_s,
                gsl_bandwidth_mbps=cfg.link.gsl_bandwidth_mbps,
                seed=cfg.seed,
                visibility_fn=visibility_fn,
                gs_antennas=gs_antennas,
            )
            timings["compute"] = time.perf_counter() - t0

            # 拥塞扫描：找出"调度策略开始起作用"的算力区间（可关闭以省时间）
            if cfg.compute.task_count <= 400:
                tick("星载算力拥塞扫描…", 0.80)
                from .computing import sweep_sat_capacity

                contention = sweep_sat_capacity(
                    cfg.compute,
                    n_sat=constellation.n_sat,
                    n_gs=max(n_gs, 1),
                    duration_s=cfg.duration_s,
                    gsl_bandwidth_mbps=cfg.link.gsl_bandwidth_mbps,
                    seed=cfg.seed,
                    visibility_fn=visibility_fn,
                    gs_antennas=gs_antennas,
                )
                timings["contention_sweep"] = time.perf_counter() - t0

        # ---------------- ⑦ 真实评测（真实传输 + 真实计算） ---------------- #
        real_cfg = cfg.real_eval
        do_real = real_cfg.enabled if run_real is None else bool(run_real)
        real_transfer: Optional[RealTransferReport] = None
        real_compute: Optional[RealComputeReport] = None
        real_tasks: List[Task] = []
        real_profile: Optional[LinkProfile] = None

        if do_real:
            t0 = time.perf_counter()

            # ---- ⑦.1 真实数据传输 ----
            if real_cfg.transfer:
                tick("真实数据传输评测（内核 TCP + 包级可靠传输）…", 0.90)
                t0t = time.perf_counter()
                try:
                    real_profile = self._real_link_profile(rep_owd, rep_bw, n_gs)
                    real_transfer = evaluate_transfer(
                        real_profile,
                        payload_mbit=real_cfg.payload_mbit,
                        tcp_trials=real_cfg.tcp_trials,
                        packet_trials=real_cfg.packet_trials,
                        packet_algos=tuple(real_cfg.packet_algos),
                        packet_bandwidth_mbps=real_cfg.packet_bandwidth_mbps,
                        packet_payload_mbit=real_cfg.packet_payload_mbit,
                        compare_analytic=real_cfg.compare_analytic,
                        rtt_ref_ms=cfg.transport.rtt_ref_ms,
                        progress=lambda m: tick(m, 0.90),
                    )
                    for w in real_transfer.warnings:
                        self.warnings.append("[真实传输] " + w)
                except Exception as exc:  # noqa: BLE001
                    self.warnings.append(
                        f"真实数据传输评测失败：{type(exc).__name__}: {exc}")
                timings["real_transfer"] = time.perf_counter() - t0t

            # ---- ⑦.2 真实计算任务 ----
            if real_cfg.compute:
                t0c = time.perf_counter()
                tick("真实计算任务评测（标定本机算力 + 真实算子）…", 0.94)
                try:
                    real_compute, real_tasks = evaluate_real_compute(
                        n_tasks=int(real_cfg.real_task_count),
                        n_gs=max(n_gs, 1),
                        duration_s=cfg.duration_s,
                        sat_capacity_tops=cfg.compute.sat_capacity_tops,
                        radiation_derate=cfg.compute.radiation_derate,
                        repeats=int(real_cfg.repeats),
                        seed=cfg.seed,
                        progress=lambda m: tick(m, 0.94),
                    )
                    for w in real_compute.warnings:
                        self.warnings.append("[真实计算] " + w)
                except Exception as exc:  # noqa: BLE001
                    self.warnings.append(
                        f"真实计算任务评测失败：{type(exc).__name__}: {exc}")
                timings["real_compute"] = time.perf_counter() - t0c

            timings["real_eval"] = time.perf_counter() - t0

        # ---------------- ⑧ 真实数据流（真字节 × 真拓扑 × 逐包多跳） ---------------- #
        real_data: Optional[DataFlowReport] = None
        do_rdata = cfg.real_data.enabled if run_real_data is None else bool(run_real_data)
        if do_rdata and n_gs >= 2:
            tick("真实数据流仿真（真载荷逐包多跳转发）…", 0.97)
            t0d = time.perf_counter()
            try:
                real_data = simulate_real_data(
                    cfg.real_data, constellation, snapshots, cfg.ground_stations,
                    snapshot_interval_s=cfg.snapshot_interval_s,
                    sim_duration_s=cfg.duration_s,
                    seed=cfg.seed,
                    progress=lambda m: tick(m, 0.97),
                )
                for w in real_data.warnings:
                    self.warnings.append("[真实数据流] " + w)
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(
                    f"真实数据流仿真失败：{type(exc).__name__}: {exc}")
            timings["real_data"] = time.perf_counter() - t0d

        tick("完成", 1.0)
        timings["total"] = time.perf_counter() - t_start

        result = SimulationResult(
            config=cfg,
            constellation=constellation,
            snapshots=snapshots,
            topo_stats=stats,
            routes=routes,
            transport=transport,
            compute=compute,
            contention=contention,
            representative_path=rep_path,
            path_owd_ms=rep_owd,
            path_bottleneck_mbps=rep_bw,
            timings=timings,
            warnings=list(self.warnings),
            real_transfer=real_transfer,
            real_compute=real_compute,
            real_tasks=real_tasks,
            real_profile=real_profile,
            real_data=real_data,
            tcp_flows=tcp_flows,
            tcp_profile=tcp_profile,
            tcp_capacity=tcp_capacity,
            route_cross=route_cross,
            tcp_sweep=tcp_sweep,
            tcp_sweep_points=tcp_sweep_points,
            transport_engine=engine_mode,
        )

        # ---------------- ⑫ 全局离散事件编排（可选叠加） ---------------- #
        if cfg.des.enabled:
            t0e = time.perf_counter()
            try:
                result.des = run_des(
                    duration_s=cfg.duration_s,
                    snapshot_interval_s=cfg.snapshot_interval_s,
                    tcp_profile=tcp_profile,
                    compute_cfg=cfg.compute,
                    des_cfg=cfg.des,
                    tcp_cfg=default_tcp_config(cfg.transport),
                    progress=lambda m: tick(f"事件内核：{m}", 0.995),
                )
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(
                    f"离散事件编排失败：{type(exc).__name__}: {exc}")
            timings["des"] = time.perf_counter() - t0e

        return result

    # ------------------------------------------------------------------ #
    def _real_link_profile(self, rep_owd: float, rep_bw: float,
                           n_gs: int) -> LinkProfile:
        """由代表路径构造真实评测用的链路剖面。

        代表路径来自"球面距离最远的一对地面站"的时延最短路。若拓扑退化
        （地面站 < 2 或路径不可达），按 ``real_eval.fallback_profile`` 决定
        是否退回标称链路剖面，否则抛错由调用方记为告警。
        """
        cfg = self.cfg
        mtu = int(getattr(cfg.transport, "mss_bytes", 1200) or 1200)
        if rep_owd > 0 and rep_bw > 0:
            name = "代表路径（最远地面站对·时延最短路）"
        else:
            if not cfg.real_eval.fallback_profile:
                raise RuntimeError("无可用的端到端路径，无法构造链路剖面")
            # 标称剖面：时延按链路几何量级取，带宽取星地/星间瓶颈
            rep_owd = 25.0
            rep_bw = min(cfg.link.gsl_bandwidth_mbps, cfg.link.isl_bandwidth_mbps)
            name = "标称链路（无可用端到端路径）"
            self.warnings.append(
                "[真实传输] 无可用的端到端代表路径，改用标称链路剖面"
                f"（OWD {rep_owd:.0f} ms / {rep_bw:.0f} Mbps）")
        return make_profile(rep_owd, rep_bw, cfg.transport.loss_prob,
                            mtu_bytes=mtu, name=name)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _farthest_gs_pair(stations) -> Tuple[int, int]:
        """返回球面距离最远的地面站索引对（用于端到端路由分析）。"""
        import math

        if len(stations) < 2:
            return 0, max(len(stations) - 1, 0)
        best = (0, 1)
        best_d = -1.0
        for i in range(len(stations)):
            for j in range(i + 1, len(stations)):
                a, b = stations[i], stations[j]
                la1, lo1 = math.radians(a.lat_deg), math.radians(a.lon_deg)
                la2, lo2 = math.radians(b.lat_deg), math.radians(b.lon_deg)
                cosang = (
                    math.sin(la1) * math.sin(la2)
                    + math.cos(la1) * math.cos(la2) * math.cos(lo2 - lo1)
                )
                d = math.acos(max(min(cosang, 1.0), -1.0))
                if d > best_d:
                    best_d, best = d, (i, j)
        return best

    # ------------------------------------------------------------------ #
    def _default_protocols(self) -> Sequence[str]:
        want = self.cfg.routing.protocol
        base = ["shortest_delay", "shortest_hop", "virtual_node", "geographic"]
        if want not in base and want in ROUTERS:
            base.append(want)
        return base

    # ------------------------------------------------------------------ #
    @staticmethod
    def _path_metrics(snap: Snapshot, path: Sequence[int]) -> Tuple[float, int, float]:
        edge_of: Dict[Tuple[int, int], int] = {}
        for i in range(snap.n_edges):
            a, b = int(snap.src[i]), int(snap.dst[i])
            edge_of[(a, b)] = i
            edge_of[(b, a)] = i
        total, bn = 0.0, float("inf")
        for u, v in zip(path[:-1], path[1:]):
            i = edge_of.get((int(u), int(v)))
            if i is None:
                return float("inf"), len(path) - 1, 0.0
            total += float(snap.delay_ms[i])
            bn = min(bn, float(snap.capacity_mbps[i]))
        return total, len(path) - 1, bn

    # ------------------------------------------------------------------ #
    def _analyse_route(
        self,
        router,
        snapshots: Sequence[Snapshot],
        constellation: Constellation,
        src: int,
        dst: int,
        vadj: np.ndarray,
        duration_s: float,
        positions: np.ndarray,
        grid_dt: float,
        sample_paths: int = 24,
    ) -> RouteAnalysis:
        delays: List[float] = []
        hops: List[int] = []
        paths: List[List[int]] = []
        all_paths: List[List[int]] = []
        examples: List[List[int]] = []
        ok = 0
        n_t = positions.shape[1]
        snap_dt = max(self.cfg.snapshot_interval_s, 1e-9)

        step = max(1, len(snapshots) // sample_paths)
        for k, snap in enumerate(snapshots):
            gi = int(np.clip(round(k * snap_dt / grid_dt), 0, n_t - 1))
            ctx = {"positions": positions[:, gi, :], "virtual_adj": vadj}
            res: RouteResult = router.route(snap, src, dst, ctx)
            if res.ok and np.isfinite(res.delay_ms):
                ok += 1
                delays.append(res.delay_ms)
                hops.append(res.hops)
                all_paths.append(list(res.path))
                if k % step == 0:
                    paths.append(list(res.path))
                    if len(examples) < 6:
                        examples.append(list(res.path))
            else:
                delays.append(float("nan"))
                hops.append(0)
                all_paths.append([])

        finite = [d for d in delays if np.isfinite(d)]
        d_arr = np.asarray(finite, dtype=float) if finite else np.zeros(0)
        h_arr = np.asarray([h for h in hops if h > 0], dtype=float)
        p5 = float(np.percentile(d_arr, 5)) if d_arr.size else float("nan")
        p95 = float(np.percentile(d_arr, 95)) if d_arr.size else float("nan")
        return RouteAnalysis(
            protocol=router.name,
            label=router.label,
            mean_delay_ms=float(d_arr.mean()) if d_arr.size else float("nan"),
            p95_delay_ms=p95,
            mean_hops=float(h_arr.mean()) if h_arr.size else 0.0,
            max_hops=int(max(hops)) if hops else 0,
            switch_count=path_switch_count(paths) if paths else 0,
            switch_rate=(path_switch_count(paths) / duration_s) if paths and duration_s > 0 else 0.0,
            success_rate=ok / max(len(snapshots), 1),
            median_delay_ms=float(np.median(d_arr)) if d_arr.size else float("nan"),
            jitter_ms=float(d_arr.std()) if d_arr.size else 0.0,
            delay_spread_ms=float(p95 - p5) if d_arr.size else 0.0,
            hop_std=float(h_arr.std()) if h_arr.size else 0.0,
            reachable_snapshots=int(ok),
            sample_interval_s=snap_dt,
            delays=delays,
            hops=hops,
            paths=paths,
            all_paths=all_paths,
            examples=examples,
        )

"""时变拓扑：可见性判定、星间链路（ISL）构型、快照序列。

核心抽象是**时变图的时间离散化**。由于星座运动是确定性的，采用"快照 + 回放"
策略（Hypatia / Stardust / Celestial 的共同思路）：把仿真时长切成若干快照，
在每个快照内认为拓扑与链路时延不变，快照间的变化用离散事件推进。

快照间隔 Δt 的选取准则
--------------------
设相对角速度上限为 ω_max（同轨道面相邻星）、卫星高度 h，则相邻星距离变化率
上限约为 v_rel = 2 v_orb sin(Δθ/2) ≈ 7.6 km/s（LEO）。若要求单跳时延误差
不超过 ε，则

    Δt ≤ ε · c / v_rel

取 ε = 1 ms → Δt ≤ 300 km / 7.6 km/s ≈ 39 s。默认 10 s 已留足余量。

ISL 构型（connection-vector 形式）
--------------------------------
参照 arXiv:2508.15307 的轨道-相位坐标（OPCS）写法，用连接向量 φ(ṅ, ṁ) 描述
同一星座内任意两颗卫星的相对连接关系（ṅ 为轨道面差，ṁ 为面内槽位差）：

    +Grid   : {(0,±1), (±1,0)}                 同面 2 条 + 邻面 2 条
    *Grid   : {(0,±1), (±1,+1), (±1,-1)}       同面 2 条 + 跨面 4 条
    Grid-V  : {(±1,0), (±1,+1), (±1,-1)}       舍弃轨内链路，以跨邻轨为主

链路可用性需同时满足：① 几何距离 < 上限；② 两点连线不与地球（含大气余量）
相交；③ 极轨反向缝（counter-rotating seam）默认不建链。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import GroundStationConfig, LinkConfig
from .orbit import (
    C_KM_S,
    EARTH_RADIUS_KM,
    Constellation,
    elevation_deg,
    lla_to_ecef,
)

# 预置连接向量
ISL_PATTERNS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "+Grid": ((0, 1), (0, -1), (1, 0), (-1, 0)),
    "*Grid": ((0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)),
    "Grid-V": ((1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)),
    "none": (),
}


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #
def segment_clearance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """线段 AB 到地心的最小距离（km）。

    a、b 形状 (E, 3)，返回 (E,)。这是"地球遮挡"判据的向量化实现：
    当最小距离小于 Re + h_atm 时链路被地球挡住。
    """
    d = b - a
    dd = np.einsum("ij,ij->i", d, d)
    t = -np.einsum("ij,ij->i", a, d) / np.maximum(dd, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * d
    return np.linalg.norm(closest, axis=1)


def shannon_capacity_mbps(
    distance_km: np.ndarray,
    bandwidth_mhz: float,
    carrier_ghz: float,
    tx_power_dbw: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    noise_temp_k: float,
    atm_loss_db: float,
) -> np.ndarray:
    """自由空间 + 香农容量的射频链路预算（可选定容模型）。

    接收功率:  P_rx = P_tx + G_tx + G_rx - FSPL - L_atm
    自由空间损耗: FSPL[dB] = 92.45 + 20log10(f[GHz]) + 20log10(d[km])
    噪声功率:  N = k T B,  k = 1.380649e-23 J/K
    香农容量:  C = B log2(1 + P_rx/N)
    """
    d = np.maximum(np.asarray(distance_km, dtype=float), 1.0)
    fspl = 92.45 + 20.0 * math.log10(carrier_ghz) + 20.0 * np.log10(d)
    p_rx_dbm = (
        10.0 * (tx_power_dbw + 3.0)  # dBW → dBm
        + tx_gain_dbi
        + rx_gain_dbi
        - fspl
        - atm_loss_db
    )
    p_rx_w = 10.0 ** ((p_rx_dbm - 30.0) / 10.0)
    b_hz = bandwidth_mhz * 1e6
    noise_w = 1.380649e-23 * noise_temp_k * b_hz
    snr = p_rx_w / max(noise_w, 1e-30)
    snr = np.maximum(snr, 1e-6)
    c_bps = b_hz * np.log2(1.0 + snr)
    return c_bps / 1e6


# --------------------------------------------------------------------------- #
# 快照
# --------------------------------------------------------------------------- #
@dataclass
class Snapshot:
    """单个时间片内的静态加权图。"""

    index: int
    t_s: float
    unix: float
    n_sat: int
    n_gs: int
    # 边（无向图，每条边只存一次）
    src: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    dst: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    delay_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    capacity_mbps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))  # 0=ISL 1=GSL
    # 地面站可见卫星：gs 局部索引 -> 卫星索引数组
    gs_visible: Dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.n_sat + self.n_gs

    @property
    def n_edges(self) -> int:
        return int(self.src.size)

    def to_networkx(self):
        """转成 networkx.Graph，边权 `delay`（ms）、`capacity`（Mbps）。"""
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(self.n_nodes))
        for i in range(self.n_edges):
            g.add_edge(
                int(self.src[i]),
                int(self.dst[i]),
                delay=float(self.delay_ms[i]),
                capacity=float(self.capacity_mbps[i]),
                kind=int(self.kind[i]),
            )
        return g

    def adjacency(self, weighted: bool = True):
        """返回 CSR 风格的邻接表，供自研路由快速使用。

        returns: (indptr, indices, weights) —— 与 scipy.sparse.csr_matrix 兼容。
        """
        n = self.n_nodes
        deg = np.bincount(self.src, minlength=n) + np.bincount(self.dst, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(deg, out=indptr[1:])
        # 边在两个方向各放一次
        order = np.argsort(np.concatenate([self.src, self.dst]), kind="stable")
        indices = np.concatenate([self.dst, self.src])[order]
        w = np.concatenate([self.delay_ms, self.delay_ms])[order]
        return indptr, indices, w


# --------------------------------------------------------------------------- #
# 拓扑构建器
# --------------------------------------------------------------------------- #
class TopologyBuilder:
    """由星座 + 地面站 + 链路配置生成快照序列。"""

    def __init__(
        self,
        link_cfg: LinkConfig,
        ground_stations: Sequence[GroundStationConfig],
    ):
        self.link = link_cfg
        self.gs = list(ground_stations)

    # ------------------------------------------------------------------ #
    def _sat_edge_index(self, constellation: Constellation) -> Tuple[np.ndarray, np.ndarray]:
        """一次性构造卫星-卫星的候选边端点索引（不判可见性）。"""
        cfg = constellation.cfg
        P, S = int(cfg.planes), int(cfg.sats_per_plane)
        pattern = ISL_PATTERNS.get(self.link.isl_mode, ISL_PATTERNS["+Grid"])

        src: List[int] = []
        dst: List[int] = []
        for p in range(P):
            for s in range(S):
                i = p * S + s
                for dn, dm in pattern:
                    pn = (p + dn) % P
                    sn = (s + dm) % S
                    # 反向缝：极轨相邻面的相对运动方向相反，默认不建链
                    if dn != 0 and not self.link.isl_seam_links:
                        if {p, pn} == {0, P - 1}:
                            continue
                    j = pn * S + sn
                    if j <= i:
                        continue
                    src.append(i)
                    dst.append(j)
        # 去重（同一条边可能被两个方向生成）
        pairs = np.array(list(dict.fromkeys(zip(src, dst))), dtype=np.int64)
        if pairs.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return pairs[:, 0].copy(), pairs[:, 1].copy()

    # ------------------------------------------------------------------ #
    def build(
        self,
        constellation: Constellation,
        times_s: np.ndarray,
        start_unix: float,
        snapshot_interval_s: float,
    ) -> List[Snapshot]:
        if constellation.positions_ecef is None:
            raise RuntimeError("请先对星座调用 propagate() 与 to_ecef()")

        n_sat = constellation.n_sat
        n_gs = len(self.gs)
        pos = constellation.positions_ecef  # (n_sat, n_t, 3)
        times = np.asarray(times_s, dtype=float)

        idx_src, idx_dst = self._sat_edge_index(constellation)
        gs_ecef = [lla_to_ecef(g.lat_deg, g.lon_deg, g.alt_km) for g in self.gs]

        # 快照时间点（含端点）
        snap_times = np.arange(0.0, times[-1] + 1e-9, snapshot_interval_s)
        # 映射到传播网格上的最近索引
        grid_idx = np.clip(
            np.searchsorted(times, snap_times, side="left"), 0, times.size - 1
        )

        snapshots: List[Snapshot] = []
        for k, (ti, gi) in enumerate(zip(snap_times, grid_idx)):
            p = pos[:, gi, :]  # (n_sat, 3)
            snap = Snapshot(
                index=k,
                t_s=float(ti),
                unix=float(start_unix + ti),
                n_sat=n_sat,
                n_gs=n_gs,
            )

            # ---------------- 星间链路 ---------------- #
            if idx_src.size:
                a = p[idx_src]
                b = p[idx_dst]
                dist = np.linalg.norm(b - a, axis=1)
                clear = segment_clearance(a, b)
                ok = (dist <= self.link.isl_max_range_km) & (
                    clear >= EARTH_RADIUS_KM + self.link.min_clearance_km
                )
                if np.any(ok):
                    s = idx_src[ok]
                    d = idx_dst[ok]
                    dd = dist[ok]
                    delay = dd / C_KM_S * 1000.0 + self.link.isl_latency_ms_fixed
                    if self.link.use_physics_budget:
                        cap = shannon_capacity_mbps(
                            dd,
                            bandwidth_mhz=min(self.link.isl_bandwidth_mbps, 2000.0),
                            carrier_ghz=self.link.rf_carrier_ghz,
                            tx_power_dbw=self.link.tx_power_dbw,
                            tx_gain_dbi=self.link.tx_gain_dbi,
                            rx_gain_dbi=self.link.rx_gain_dbi,
                            noise_temp_k=self.link.noise_temp_k,
                            atm_loss_db=self.link.atm_loss_db,
                        )
                    else:
                        cap = np.full(dd.shape, self.link.isl_bandwidth_mbps)
                    snap.src = np.concatenate([snap.src, s])
                    snap.dst = np.concatenate([snap.dst, d])
                    snap.delay_ms = np.concatenate([snap.delay_ms, delay])
                    snap.capacity_mbps = np.concatenate([snap.capacity_mbps, cap])
                    snap.kind = np.concatenate(
                        [snap.kind, np.zeros(s.size, dtype=int)]
                    )

            # ---------------- 星地链路 ---------------- #
            for g_local, (g_cfg, r_gs) in enumerate(zip(self.gs, gs_ecef)):
                rho = p - r_gs
                rng = np.linalg.norm(rho, axis=1)
                el, _ = elevation_deg(g_cfg.lat_deg, g_cfg.lon_deg, g_cfg.alt_km, p)
                ok = (el >= g_cfg.min_elevation_deg) & (
                    rng <= self.link.gsl_max_range_km
                )
                idx = np.flatnonzero(ok)
                if idx.size:
                    # 只保留仰角最高（几何上最有利）的若干颗，模拟有限天线数
                    keep = idx[np.argsort(-el[idx])][: max(1, g_cfg.antennas)]
                    snap.gs_visible[g_local] = keep
                    g_node = n_sat + g_local
                    dly = rng[keep] / C_KM_S * 1000.0 + self.link.gsl_latency_ms_fixed
                    snap.src = np.concatenate(
                        [snap.src, np.full(keep.size, g_node, dtype=int)]
                    )
                    snap.dst = np.concatenate([snap.dst, keep])
                    snap.delay_ms = np.concatenate([snap.delay_ms, dly])
                    snap.capacity_mbps = np.concatenate(
                        [snap.capacity_mbps,
                         np.full(keep.size, self.link.gsl_bandwidth_mbps)]
                    )
                    snap.kind = np.concatenate(
                        [snap.kind, np.ones(keep.size, dtype=int)]
                    )

            snapshots.append(snap)
        return snapshots

    # ------------------------------------------------------------------ #
    def virtual_topology(self, constellation: Constellation):
        """基于"平均位置"构造静态虚拟拓扑，用于虚拟节点（VP）路由。

        思路：把时变物理图投影到一张**变化很慢甚至不变**的虚拟图上，
        在虚拟图上算路，再映射回物理下一跳。这样路由表更新频率大幅下降。
        """
        mean_ecef = constellation.mean_positions_ecef()  # (n_sat, 3)
        idx_src, idx_dst = self._sat_edge_index(constellation)
        if idx_src.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        dist = np.linalg.norm(mean_ecef[idx_dst] - mean_ecef[idx_src], axis=1)
        # 用轨道面/槽位差作为"虚拟坐标"，只保留几何上稳定的逻辑邻接
        return np.stack([idx_src, idx_dst], axis=1)[dist < 2.0 * self.link.isl_max_range_km]


# --------------------------------------------------------------------------- #
# 拓扑统计
# --------------------------------------------------------------------------- #
def topology_stats(snapshots: Sequence[Snapshot], n_sat: int) -> Dict[str, float]:
    """快照序列的宏观统计量，用于校验星座构型是否合理。"""
    if not snapshots:
        return {}
    deg_sat, deg_gs, delays, isl_cnt, gsl_cnt = [], [], [], [], []
    for s in snapshots:
        cover = np.bincount(s.src, minlength=s.n_nodes) + np.bincount(
            s.dst, minlength=s.n_nodes
        )
        deg_sat.append(cover[:n_sat].mean())
        if s.n_gs:
            deg_gs.append(cover[n_sat:].mean())
        isl = s.kind == 0
        gsl = s.kind == 1
        isl_cnt.append(int(isl.sum()))
        gsl_cnt.append(int(gsl.sum()))
        if np.any(isl):
            delays.append(float(np.mean(s.delay_ms[isl])))
    return {
        "snapshots": len(snapshots),
        "mean_isl_degree": float(np.mean(deg_sat)),
        "mean_gs_degree": float(np.mean(deg_gs)) if deg_gs else 0.0,
        "mean_isl_count": float(np.mean(isl_cnt)),
        "mean_gsl_count": float(np.mean(gsl_cnt)),
        "mean_isl_delay_ms": float(np.mean(delays)) if delays else 0.0,
    }

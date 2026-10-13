"""实时仿真过程可视化：把真实数据流的每一跳回放成"光点穿过星座"的实时画面。

设计要点
--------
* **场景层** `LiveScene`：把 `DataFlowReport` 的分组轨迹整理成可 O(1) 查询的
  分段索引（按开始时刻分桶 + 最大时长回看），并缓存星座几何与快照链路，
  使得任意时刻 t 的"活动分段 / 可见链路 / 节点坐标"都能即时取出。
* **渲染层** `LiveView`：纯 QPainter 手绘的深空指挥舱视图——
  地球经纬网 + 卫星光点 + 星间链路 + 地面站脉冲 + 分组彗星拖尾 + HUD。
* **播放层** `LivePlayer`：播放/暂停/倍速/进度拖拽 + 时间轴缩略图。
* **导出层** `export_html`：自包含单文件 HTML（Canvas 2D，几何以 base64
  Float32 内嵌），可直接在浏览器里放给他人看。

坐标与投影
----------
节点坐标取 ECEF(km)；相机为方位角 az + 俯仰角 el 的正交投影，
地球半径 6371 km。可见性判据：深度 >= 0（面向相机）或屏幕半径超出地球圆盘。
"""

from __future__ import annotations

import base64
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RE_KM = 6371.0

# 深空指挥舱配色
BG_TOP = (6, 10, 22)
BG_BOT = (2, 3, 8)
C_EARTH = (10, 26, 40)
C_EARTH_EDGE = (46, 140, 172)
C_GRAT = (30, 78, 96)
C_SAT = (120, 235, 255)
C_SAT_DIM = (60, 120, 145)
C_ISL = (30, 96, 118)
C_GS = (255, 186, 80)
C_HUD = (150, 230, 245)
C_HUD_DIM = (95, 140, 160)
C_OK = (80, 240, 170)
C_WARN = (255, 120, 90)

# 业务流配色（亮色，深底上醒目）
FLOW_COLORS = [
    (0, 235, 255),     # 青
    (255, 176, 60),    # 琥珀
    (255, 96, 200),    # 品红
    (150, 255, 120),   # 亮绿
    (170, 150, 255),   # 紫
    (255, 240, 120),   # 亮黄
]


# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #
@dataclass
class LiveScene:
    """为实时回放准备好的静态场景 + 时序索引。"""

    n_sat: int
    n_gs: int
    duration_s: float
    # 节点坐标查表：node_pos(node, t) -> np.ndarray(3,) ECEF km
    _pos_sat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3)))
    _pos_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _pos_gs: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    gs_labels: List[str] = field(default_factory=list)
    sat_labels: List[str] = field(default_factory=list)

    # 分段（每一跳）
    seg_t0: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    seg_t1: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    seg_u: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    seg_v: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    seg_pid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    seg_flow: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    seg_size: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    _pkt_seg0: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    _pkt_seg1: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    _bin_width: float = 0.05
    _bins: List[np.ndarray] = field(default_factory=list)
    _seg_order: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    # 快照链路（按快照索引）
    link_src: List[np.ndarray] = field(default_factory=list)
    link_dst: List[np.ndarray] = field(default_factory=list)
    link_kind: List[np.ndarray] = field(default_factory=list)
    link_idx_t: List[float] = field(default_factory=list)
    snapshot_interval_s: float = 10.0

    # 事件：分组创建/交付/丢弃（用于闪烁效果）
    ev_t: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    ev_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    ev_node: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    ev_flow: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    _ev_bins: List[np.ndarray] = field(default_factory=list)
    _bin_width_ev: float = 0.05

    # 时间序列
    series_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    series_inflight: np.ndarray = field(default_factory=lambda: np.zeros(0))
    series_goodput: np.ndarray = field(default_factory=lambda: np.zeros(0))
    series_delivered: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # 链路热力（累计流量）
    heat: Dict[Tuple[int, int], float] = field(default_factory=dict)
    heat_max: float = 1.0

    flows: List[dict] = field(default_factory=list)
    title: str = ""
    payload_label: str = ""
    payload_sha: str = ""
    integrity_ok: bool = False

    # ---------------- 构造 ---------------- #
    @classmethod
    def build(cls, constellation, snapshots: Sequence, ground_stations: Sequence,
              report, times_s: np.ndarray) -> "LiveScene":
        pos = constellation.positions_ecef
        n_sat = constellation.n_sat
        n_gs = len(ground_stations)
        from .orbit import lla_to_ecef
        gs_ecef = (np.array([lla_to_ecef(g.lat_deg, g.lon_deg, g.alt_km)
                             for g in ground_stations])
                   if n_gs else np.zeros((0, 3)))

        sc = cls(
            n_sat=n_sat, n_gs=n_gs,
            duration_s=float(report.duration_s),
            _pos_sat=np.asarray(pos), _pos_t=np.asarray(times_s, dtype=float),
            _pos_gs=gs_ecef,
            gs_labels=[getattr(g, "name", f"GS-{i}") for i, g in enumerate(ground_stations)],
            sat_labels=[f"SAT-{i:04d}" for i in range(n_sat)],
            snapshot_interval_s=float(report.snapshot_interval_s),
            payload_label=report.payload_label,
            payload_sha=report.payload_sha,
            integrity_ok=report.integrity_ok,
        )

        # ---- 分段索引（按分组顺序追加 → 每个分组的分段在数组里连续） ----
        t0, t1, u, v, pid, fl, sz = [], [], [], [], [], [], []
        pkt_seg0 = [0]
        for p in report.packets:
            for (a, b, nu, nv) in p.segs:
                t0.append(a); t1.append(b); u.append(nu); v.append(nv)
                pid.append(p.pid); fl.append(p.flow_id); sz.append(p.size_bytes)
            pkt_seg0.append(len(t0))
        sc._pkt_seg0 = np.asarray(pkt_seg0[:-1], dtype=np.int32)
        sc._pkt_seg1 = np.asarray(pkt_seg0[1:], dtype=np.int32)
        sc.seg_t0 = np.asarray(t0, dtype=np.float32)
        sc.seg_t1 = np.asarray(t1, dtype=np.float32)
        sc.seg_u = np.asarray(u, dtype=np.int32)
        sc.seg_v = np.asarray(v, dtype=np.int32)
        sc.seg_pid = np.asarray(pid, dtype=np.int32)
        sc.seg_flow = np.asarray(fl, dtype=np.int32)
        sc.seg_size = np.asarray(sz, dtype=np.int32)
        if sc.seg_t0.size:
            dur = float(np.max(sc.seg_t1 - sc.seg_t0))
            sc._bin_width = max(dur, 1e-4)
            nb = int(math.ceil(sc.duration_s / sc._bin_width)) + 2
            sc._bins = [[] for _ in range(max(nb, 1))]
            for i in range(sc.seg_t0.size):
                k = int(sc.seg_t0[i] / sc._bin_width)
                k = min(max(k, 0), len(sc._bins) - 1)
                sc._bins[k].append(i)
            sc._bins = [np.asarray(b, dtype=np.int32) for b in sc._bins]

        # ---- 事件（创建 / 交付 / 丢弃）----
        et, ek, en, ef = [], [], [], []
        for p in report.packets:
            et.append(p.created_s); ek.append(0); en.append(p.src_node); ef.append(p.flow_id)
            if p.delivered_s is not None:
                et.append(p.delivered_s); ek.append(1); en.append(p.dst_node)
                ef.append(p.flow_id)
            elif p.end_s is not None:
                et.append(p.end_s); ek.append(2)
                en.append(p.drop_node if p.drop_node is not None else p.src_node)
                ef.append(p.flow_id)
        order = np.argsort(np.asarray(et, dtype=float)) if et else np.zeros(0, dtype=int)
        sc.ev_t = np.asarray(et, dtype=np.float32)[order] if et else np.zeros(0, np.float32)
        sc.ev_kind = np.asarray(ek, dtype=np.int8)[order] if et else np.zeros(0, np.int8)
        sc.ev_node = np.asarray(en, dtype=np.int32)[order] if et else np.zeros(0, np.int32)
        sc.ev_flow = np.asarray(ef, dtype=np.int8)[order] if et else np.zeros(0, np.int8)
        if sc.ev_t.size:
            evb = max(sc.duration_s / 600.0, 0.01)
            sc._bin_width_ev = evb
            nb = int(math.ceil(sc.duration_s / evb)) + 2
            sc._ev_bins = [[] for _ in range(max(nb, 1))]
            for i in range(sc.ev_t.size):
                k = min(int(sc.ev_t[i] / evb), len(sc._ev_bins) - 1)
                sc._ev_bins[k].append(i)
            sc._ev_bins = [np.asarray(b, dtype=np.int32) for b in sc._ev_bins]

        # ---- 快照链路 ----
        for snap in snapshots:
            m = snap.kind == 0            # 只画 ISL，GSL 只在端点站示意
            sc.link_src.append(snap.src[m].astype(np.int32))
            sc.link_dst.append(snap.dst[m].astype(np.int32))
            sc.link_kind.append(snap.kind[m].astype(np.int8))
            sc.link_idx_t.append(float(snap.t_s))

        # ---- 序列 ----
        sc.series_t = np.asarray(report.series_t, dtype=float)
        sc.series_inflight = np.asarray(report.series_inflight, dtype=float)
        sc.series_goodput = np.asarray(report.series_goodput, dtype=float)
        sc.series_delivered = np.asarray(report.series_delivered, dtype=float)

        # ---- 链路热力 ----
        for s in report.link_stats:
            sc.heat[s.edge] = float(s.bytes)
        sc.heat_max = max(sc.heat.values(), default=1.0) or 1.0

        # ---- 流信息 ----
        sc.flows = [
            {
                "id": f.flow_id,
                "src": f.src_node, "dst": f.dst_node,
                "src_label": f.src_label, "dst_label": f.dst_label,
                "rate_mbps": f.rate_mbps,
                "color": FLOW_COLORS[f.flow_id % len(FLOW_COLORS)],
            }
            for f in report.flows
        ]
        sc.title = f"真实数据流实时仿真 · {report.payload_label}"
        return sc

    # ---------------- 查询 ---------------- #
    def node_pos(self, node: int, t_s: float) -> np.ndarray:
        if node < self.n_sat:
            n_t = self._pos_t.size
            if n_t == 0:
                return np.zeros(3)
            t0, t1 = float(self._pos_t[0]), float(self._pos_t[-1])
            frac = (t_s - t0) / max(t1 - t0, 1e-9)
            k = int(np.clip(round(frac * (n_t - 1)), 0, n_t - 1))
            return self._pos_sat[node, k]
        j = node - self.n_sat
        if 0 <= j < self._pos_gs.shape[0]:
            return self._pos_gs[j]
        return np.zeros(3)

    def nodes_pos_array(self, nodes: np.ndarray, t_s: float) -> np.ndarray:
        """批量取节点坐标，返回 (len(nodes), 3)。"""
        out = np.zeros((nodes.size, 3), dtype=float)
        sat_mask = nodes < self.n_sat
        if sat_mask.any():
            n_t = self._pos_t.size
            t0, t1 = float(self._pos_t[0]), float(self._pos_t[-1])
            k = int(np.clip(round((t_s - t0) / max(t1 - t0, 1e-9) * (n_t - 1)),
                            0, n_t - 1)) if n_t else 0
            out[sat_mask] = self._pos_sat[nodes[sat_mask], k]
        if (~sat_mask).any():
            idx = nodes[~sat_mask] - self.n_sat
            ok = (idx >= 0) & (idx < self._pos_gs.shape[0])
            if ok.any():
                out[np.flatnonzero(~sat_mask)[ok]] = self._pos_gs[idx[ok]]
        return out

    def active_segments(self, t_s: float) -> np.ndarray:
        """返回时刻 t 正在传输中的分段索引。"""
        if not self._bins:
            return np.zeros(0, dtype=np.int32)
        k = int(t_s / self._bin_width)
        out: List[np.ndarray] = []
        for kk in (k - 1, k):
            if 0 <= kk < len(self._bins):
                idx = self._bins[kk]
                if idx.size:
                    m = (self.seg_t0[idx] <= t_s) & (t_s <= self.seg_t1[idx])
                    if m.any():
                        out.append(idx[m])
        if not out:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(out)

    def recent_events(self, t_s: float) -> np.ndarray:
        if not self._ev_bins:
            return np.zeros(0, dtype=np.int32)
        w = self._bin_width_ev
        k = int(t_s / w)
        out: List[np.ndarray] = []
        for kk in (k - 1, k, k + 1):
            if 0 <= kk < len(self._ev_bins):
                idx = self._ev_bins[kk]
                if idx.size:
                    m = (t_s - 0.25 <= self.ev_t[idx]) & (self.ev_t[idx] <= t_s)
                    if m.any():
                        out.append(idx[m])
        if not out:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(out)

    def snapshot_index(self, t_s: float) -> int:
        if not self.link_src:
            return -1
        iv = max(self.snapshot_interval_s, 1e-9)
        return int(min(max(int(t_s // iv), 0), len(self.link_src) - 1))

    def interp_scalar(self, series: np.ndarray, t_s: float) -> float:
        if series.size == 0 or self.series_t.size == 0:
            return 0.0
        return float(np.interp(t_s, self.series_t, series))


# --------------------------------------------------------------------------- #
# 投影
# --------------------------------------------------------------------------- #
def project(P: np.ndarray, az: float, el: float, r_px: float
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF(km) → 屏幕坐标 + 深度。相机沿 +y 看，正交投影。"""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    ca, sa = math.cos(az), math.sin(az)
    x = P[:, 0] * ca + P[:, 1] * sa
    y = -P[:, 0] * sa + P[:, 1] * ca
    z = P[:, 2]
    ce, se = math.cos(el), math.sin(el)
    y2 = y * ce - z * se
    z2 = y * se + z * ce
    s = r_px / RE_KM
    return x * s, -z2 * s, y2


def front_mask(sx: np.ndarray, sy: np.ndarray, depth: np.ndarray,
               r_px: float) -> np.ndarray:
    """可见性：朝向相机，或投影落在地球圆盘之外。"""
    outside = (sx * sx + sy * sy) >= r_px * r_px
    return (depth >= 0.0) | outside


# --------------------------------------------------------------------------- #
# 供 Qt / HTML 共用的绘制数学（把"某一帧要画什么"算出来）
# --------------------------------------------------------------------------- #
@dataclass
class FrameModel:
    """一帧的全部几何（屏幕坐标），Qt 与 HTML 两套后端都消费它。"""

    width: int
    height: int
    cx: float
    cy: float
    r_px: float
    az: float
    el: float
    t_s: float

    sats: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    sats_vis: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    gs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    gs_vis: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    links: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    link_heat: np.ndarray = field(default_factory=lambda: np.zeros(0))
    comets: List[dict] = field(default_factory=list)
    pulses: List[dict] = field(default_factory=list)
    flow_ends: List[dict] = field(default_factory=list)


def build_frame(scene: LiveScene, t_s: float, width: int, height: int,
                az: float, el: float, *, max_comets: int = 400,
                trail_hops: int = 3, link_alpha_min: float = 0.0) -> FrameModel:
    """计算时刻 t 的一帧几何。"""
    cx, cy = width * 0.5, height * 0.5
    r_px = min(width, height) * 0.30
    fm = FrameModel(width=width, height=height, cx=cx, cy=cy, r_px=r_px,
                    az=az, el=el, t_s=t_s)

    # 卫星
    if scene.n_sat:
        P = scene.nodes_pos_array(np.arange(scene.n_sat), t_s)
        sx, sy, dz = project(P, az, el, r_px)
        vis = front_mask(sx, sy, dz, r_px)
        fm.sats = np.stack([cx + sx, cy + sy], axis=1)
        fm.sats_vis = vis

    # 地面站
    if scene.n_gs:
        idx = scene.n_sat + np.arange(scene.n_gs)
        P = scene.nodes_pos_array(idx, t_s)
        sx, sy, dz = project(P, az, el, r_px)
        vis = front_mask(sx, sy, dz, r_px)
        fm.gs = np.stack([cx + sx, cy + sy], axis=1)
        fm.gs_vis = vis

    # ISL 链路（当前快照）
    si = scene.snapshot_index(t_s)
    if si >= 0:
        src, dst = scene.link_src[si], scene.link_dst[si]
        if src.size:
            nodes = np.concatenate([src, dst])
            P = scene.nodes_pos_array(nodes, t_s)
            sx, sy, dz = project(P, az, el, r_px)
            n = src.size
            x1 = cx + sx[:n]; y1 = cy + sy[:n]
            x2 = cx + sx[n:]; y2 = cy + sy[n:]
            v1 = front_mask(sx[:n], sy[:n], dz[:n], r_px)
            v2 = front_mask(sx[n:], sy[n:], dz[n:], r_px)
            m = v1 | v2
            if m.any():
                fm.links = np.stack([x1[m], y1[m], x2[m], y2[m]], axis=1)
                heat = np.array([scene.heat.get((int(a), int(b)), 0.0)
                                 for a, b in zip(src[m], dst[m])])
                fm.link_heat = heat / max(scene.heat_max, 1e-9)

    # 传输中的分组 → 彗星（含多跳拖尾）
    seg = scene.active_segments(t_s)
    if seg.size:
        if seg.size > max_comets:
            seg = seg[np.linspace(0, seg.size - 1, max_comets).astype(int)]
        t0 = scene.seg_t0[seg].astype(float)
        t1 = scene.seg_t1[seg].astype(float)
        frac = np.clip((t_s - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
        A = scene.nodes_pos_array(scene.seg_u[seg], t_s)
        B = scene.nodes_pos_array(scene.seg_v[seg], t_s)
        sxA, syA, dzA = project(A, az, el, r_px)
        sxB, syB, dzB = project(B, az, el, r_px)
        # 拖尾所需的更早分段节点（批量投影）
        trail_coords: Dict[int, List[Tuple[float, float]]] = {}
        if trail_hops > 0:
            need_nodes: List[int] = []
            need_meta: List[Tuple[int, int]] = []      # (seg_index, 节点在列表中的位置)
            for ii in range(seg.size):
                si_ = int(seg[ii])
                p0 = int(scene._pkt_seg0[scene.seg_pid[si_]])
                lo = max(p0, si_ - trail_hops)
                for k in range(lo, si_):
                    need_meta.append((ii, len(need_nodes)))
                    need_nodes.append(int(scene.seg_u[k]))
                    need_meta.append((ii, len(need_nodes)))
                    need_nodes.append(int(scene.seg_v[k]))
            if need_nodes:
                PN = scene.nodes_pos_array(np.asarray(need_nodes), t_s)
                sxn, syn, dzn = project(PN, az, el, r_px)
                acc: Dict[int, List[Tuple[int, float, float]]] = {}
                for (ii, pos), k in zip(need_meta, range(len(need_nodes))):
                    acc.setdefault(ii, []).append(
                        (pos, cx + sxn[k], cy + syn[k]))
                for ii, lst in acc.items():
                    lst.sort()
                    trail_coords[ii] = [(x, y) for _p, x, y in lst]
        for i in range(seg.size):
            f = float(frac[i])
            x1 = cx + sxA[i]; y1 = cy + syA[i]
            x2 = cx + sxB[i]; y2 = cy + syB[i]
            visA = dzA[i] >= 0 or (sxA[i] ** 2 + syA[i] ** 2) >= r_px ** 2
            visB = dzB[i] >= 0 or (sxB[i] ** 2 + syB[i] ** 2) >= r_px ** 2
            if not (visA or visB):
                continue
            hx = x1 + (x2 - x1) * f
            hy = y1 + (y2 - y1) * f
            fm.comets.append({
                "x": hx, "y": hy, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "f": f, "flow": int(scene.seg_flow[seg[i]]),
                "pid": int(scene.seg_pid[seg[i]]),
                "size": int(scene.seg_size[seg[i]]),
                "trail": trail_coords.get(i, []) + [(x1, y1)],
            })

    # 事件脉冲（创建/交付/丢弃 的闪烁）
    ev = scene.recent_events(t_s)
    if ev.size:
        if ev.size > 160:
            ev = ev[np.linspace(0, ev.size - 1, 160).astype(int)]
        P = scene.nodes_pos_array(scene.ev_node[ev], t_s)
        sx, sy, dz = project(P, az, el, r_px)
        for i in range(ev.size):
            if not (dz[i] >= 0 or (sx[i] ** 2 + sy[i] ** 2) >= r_px ** 2):
                continue
            fm.pulses.append({
                "x": cx + sx[i], "y": cy + sy[i],
                "age": float(t_s - scene.ev_t[ev[i]]),
                "kind": int(scene.ev_kind[ev[i]]),
                "flow": int(scene.ev_flow[ev[i]]),
            })

    # 业务流端点（高亮 + 标注）
    for f in scene.flows:
        for role, nd in (("src", f["src"]), ("dst", f["dst"])):
            P = scene.nodes_pos_array(np.array([nd]), t_s)
            sx, sy, dz = project(P, az, el, r_px)
            if not (dz[0] >= 0 or (sx[0] ** 2 + sy[0] ** 2) >= r_px ** 2):
                continue
            fm.flow_ends.append({
                "x": cx + sx[0], "y": cy + sy[0],
                "label": f["src_label"] if role == "src" else f["dst_label"],
                "role": role, "color": f["color"], "flow": f["id"],
            })
    return fm


# --------------------------------------------------------------------------- #
# Qt 渲染后端
# --------------------------------------------------------------------------- #
try:  # PySide6 只在 GUI 场景需要；CLI 导出 HTML 时不强依赖
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter,
                               QPainterPath, QPen, QRadialGradient)
    from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                                   QSlider, QVBoxLayout, QWidget)
    QT_AVAILABLE = True
except Exception:  # noqa: BLE001
    QT_AVAILABLE = False

    class QWidget:  # type: ignore
        pass


def _c(rgb, a: int = 255) -> "QColor":
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(a))


if QT_AVAILABLE:

    class LiveView(QWidget):
        """深空指挥舱视图：手绘地球/卫星/链路/分组彗星/HUD。"""

        clicked_time = Signal(float)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.scene: Optional[LiveScene] = None
            self.t_s = 0.0
            self.az = 0.55
            self.el = 0.42
            self.auto_rotate = True
            self.show_heat = True
            self.show_links = True
            self.trail_len = 3          # 彗星尾迹保留的跳数
            self._stars: Optional[np.ndarray] = None
            self.setMinimumSize(520, 420)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._last_mouse = None

        # ---------------- 交互 ---------------- #
        def set_scene(self, scene: Optional[LiveScene]) -> None:
            self.scene = scene
            self.t_s = 0.0
            self.update()

        def set_time(self, t_s: float, rotate_dt: float = 0.0) -> None:
            if self.scene is None:
                return
            self.t_s = float(min(max(t_s, 0.0), self.scene.duration_s))
            if self.auto_rotate:
                self.az += rotate_dt * 0.06
            self.update()

        def snap_image(self, t_s: float, w: int = 1280, h: int = 760):
            """离屏渲染某一时刻的画面（用于截图 / 报告）。"""
            self.resize(w, h)
            self.t_s = float(t_s)
            return self.grab()

        def mousePressEvent(self, ev):  # noqa: N802
            self._last_mouse = ev.position()

        def mouseMoveEvent(self, ev):  # noqa: N802
            if self._last_mouse is None:
                return
            d = ev.position() - self._last_mouse
            self._last_mouse = ev.position()
            self.az -= d.x() * 0.006
            self.el = float(np.clip(self.el + d.y() * 0.005, -1.45, 1.45))
            self.update()

        def mouseReleaseEvent(self, ev):  # noqa: N802
            self._last_mouse = None

        def wheelEvent(self, ev):  # noqa: N802
            self.az += 0.12 if ev.angleDelta().y() > 0 else -0.12
            self.update()

        # ---------------- 绘制 ---------------- #
        def _ensure_stars(self, w: int, h: int) -> None:
            if self._stars is None or self._stars.shape[0] == 0:
                rng = np.random.default_rng(7)
                n = 420
                self._stars = np.stack([
                    rng.uniform(0, 1, n), rng.uniform(0, 1, n),
                    rng.uniform(0.25, 1.0, n)], axis=1)

        def paintEvent(self, ev):  # noqa: N802
            w, h = max(self.width(), 10), max(self.height(), 10)
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            self._paint_background(p, w, h)
            if self.scene is None:
                p.setPen(_c(C_HUD_DIM))
                p.setFont(QFont("Microsoft YaHei", 11))
                p.drawText(self.rect(), Qt.AlignCenter,
                           "尚未运行真实数据流仿真\n在左侧勾选「真实数据流仿真」后点击开始仿真")
                p.end()
                return

            fm = build_frame(self.scene, self.t_s, w, h, self.az, self.el)
            self._paint_earth(p, fm)
            if self.show_links:
                self._paint_links(p, fm)
            self._paint_sats(p, fm)
            self._paint_streams(p, fm)
            self._paint_ground(p, fm)
            self._paint_hud(p, fm)
            p.end()

        def _paint_background(self, p, w: int, h: int) -> None:
            g = QLinearGradient(0, 0, 0, h)
            g.setColorAt(0.0, _c(BG_TOP))
            g.setColorAt(1.0, _c(BG_BOT))
            p.fillRect(0, 0, w, h, QBrush(g))
            self._ensure_stars(w, h)
            st = self._stars
            p.setPen(Qt.NoPen)
            for i in range(st.shape[0]):
                col = int(120 + 135 * st[i, 2])
                p.setBrush(_c((col, col, min(255, col + 20)), int(40 + 150 * st[i, 2])))
                r = 0.6 + 1.5 * st[i, 2]
                p.drawEllipse(int(st[i, 0] * w), int(st[i, 1] * h), int(r * 2), int(r * 2))

        def _paint_earth(self, p, fm: FrameModel) -> None:
            # 辉光
            glow = QRadialGradient(fm.cx, fm.cy, fm.r_px * 1.9)
            glow.setColorAt(0.0, _c((20, 90, 120), 70))
            glow.setColorAt(0.45, _c((14, 60, 90), 40))
            glow.setColorAt(1.0, _c((5, 15, 30), 0))
            p.setBrush(QBrush(glow))
            p.setPen(Qt.NoPen)
            p.drawEllipse(int(fm.cx - fm.r_px * 1.9), int(fm.cy - fm.r_px * 1.9),
                          int(fm.r_px * 3.8), int(fm.r_px * 3.8))
            # 球体
            body = QRadialGradient(fm.cx - fm.r_px * 0.35, fm.cy - fm.r_px * 0.35,
                                   fm.r_px * 1.5)
            body.setColorAt(0.0, _c((24, 62, 86)))
            body.setColorAt(0.7, _c(C_EARTH))
            body.setColorAt(1.0, _c((4, 10, 18)))
            p.setBrush(QBrush(body))
            p.setPen(QPen(_c(C_EARTH_EDGE, 190), 1.4))
            p.drawEllipse(int(fm.cx - fm.r_px), int(fm.cy - fm.r_px),
                          int(fm.r_px * 2), int(fm.r_px * 2))
            # 经纬网
            from PySide6.QtGui import QPolygon
            p.setPen(QPen(_c(C_GRAT, 165), 0.8))
            for lat in range(-60, 61, 30):
                for pts in _grat_circle(fm, lat, True):
                    p.drawPolyline(QPolygon(pts))
            for lon in range(0, 180, 30):
                for pts in _grat_circle(fm, lon, False):
                    p.drawPolyline(QPolygon(pts))

        def _paint_links(self, p, fm: FrameModel) -> None:
            if fm.links.shape[0] == 0:
                return
            # 底图：所有 ISL 画一条极淡的线（呈现星座结构）
            p.setPen(QPen(_c(C_ISL, 105), 0.8))
            for i in range(fm.links.shape[0]):
                x1, y1, x2, y2 = fm.links[i]
                p.drawLine(int(x1), int(y1), int(x2), int(y2))
            # 热力叠加：有流量的链路加亮
            if self.show_heat:
                for i in range(fm.links.shape[0]):
                    v = float(np.clip(fm.link_heat[i] ** 0.5, 0.0, 1.0))
                    if v <= 0.02:
                        continue
                    col = QColor(int(40 + 130 * v), int(140 + 110 * v),
                                 int(170 + 80 * v), int(60 + 150 * v))
                    p.setPen(QPen(col, 0.6 + 2.0 * v))
                    x1, y1, x2, y2 = fm.links[i]
                    p.drawLine(int(x1), int(y1), int(x2), int(y2))

        def _paint_sats(self, p, fm: FrameModel) -> None:
            p.setPen(Qt.NoPen)
            for i in range(fm.sats.shape[0]):
                x, y = fm.sats[i]
                if fm.sats_vis[i]:
                    p.setBrush(_c(C_SAT, 210))
                    p.drawEllipse(int(x - 1.5), int(y - 1.5), 3, 3)
                else:
                    p.setBrush(_c(C_SAT_DIM, 70))
                    p.drawEllipse(int(x - 1.0), int(y - 1.0), 2, 2)

        def _paint_ground(self, p, fm: FrameModel) -> None:
            for i in range(fm.gs.shape[0]):
                x, y = fm.gs[i]
                if not fm.gs_vis[i]:
                    continue
                g = QRadialGradient(x, y, 11)
                g.setColorAt(0.0, _c(C_GS, 200))
                g.setColorAt(0.5, _c(C_GS, 70))
                g.setColorAt(1.0, _c(C_GS, 0))
                p.setBrush(QBrush(g)); p.setPen(Qt.NoPen)
                p.drawEllipse(int(x - 11), int(y - 11), 22, 22)
                p.setBrush(_c(C_GS)); p.drawEllipse(int(x - 3), int(y - 3), 6, 6)

        def _paint_streams(self, p, fm: FrameModel) -> None:
            # 端点高亮 + 标注
            p.setFont(QFont("Microsoft YaHei", 7))
            for fe in fm.flow_ends:
                col = fe["color"]
                col = QColor(*[int(v) for v in col])
                x, y = fe["x"], fe["y"]
                g = QRadialGradient(x, y, 16)
                g.setColorAt(0.0, QColor(col.red(), col.green(), col.blue(), 220))
                g.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
                p.setBrush(QBrush(g)); p.setPen(Qt.NoPen)
                p.drawEllipse(int(x - 16), int(y - 16), 32, 32)
                p.setPen(QPen(col, 1.4))
                p.drawEllipse(int(x - 5), int(y - 5), 10, 10)
                p.setPen(QPen(col, 1.0))
                p.drawText(int(x + 9), int(y + 3), fe["label"])

            # 彗星（含多跳拖尾）
            p.setPen(Qt.NoPen)
            for cm in fm.comets:
                col = FLOW_COLORS[cm["flow"] % len(FLOW_COLORS)]
                qcol = QColor(int(col[0]), int(col[1]), int(col[2]))
                trail = cm.get("trail") or [(cm["x1"], cm["y1"])]
                pts = trail + [(cm["x"], cm["y"])]
                n = len(pts)
                for k in range(n - 1):
                    a = 1.0 - k / max(n - 1, 1)          # 越靠前越淡
                    x1_, y1_ = pts[k]
                    x2_, y2_ = pts[k + 1]
                    pen = QPen(QColor(qcol.red(), qcol.green(), qcol.blue(),
                                      int(28 + 155 * a * a)), 0.8 + 1.5 * a)
                    pen.setCapStyle(Qt.RoundCap)
                    p.setPen(pen)
                    p.drawLine(int(x1_), int(y1_), int(x2_), int(y2_))
                p.setPen(Qt.NoPen)
                hx, hy = cm["x"], cm["y"]
                g = QRadialGradient(hx, hy, 11)
                g.setColorAt(0.0, QColor(255, 255, 255, 245))
                g.setColorAt(0.25, QColor(qcol.red(), qcol.green(), qcol.blue(), 215))
                g.setColorAt(1.0, QColor(qcol.red(), qcol.green(), qcol.blue(), 0))
                p.setBrush(QBrush(g))
                p.drawEllipse(int(hx - 11), int(hy - 11), 22, 22)
                p.setBrush(QColor(255, 255, 255, 240))
                p.drawEllipse(int(hx - 1.8), int(hy - 1.8), 5, 5)

            # 事件脉冲
            for pu in fm.pulses:
                age = float(np.clip(pu["age"] / 0.25, 0.0, 1.0))
                r = 4 + 22 * age
                if pu["kind"] == 2:
                    col = C_WARN
                elif pu["kind"] == 1:
                    col = C_OK
                else:
                    col = FLOW_COLORS[pu["flow"] % len(FLOW_COLORS)]
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(int(col[0]), int(col[1]), int(col[2]),
                                     int(180 * (1 - age))), 1.6))
                p.drawEllipse(int(pu["x"] - r), int(pu["y"] - r),
                              int(r * 2), int(r * 2))

        def _paint_hud(self, p, fm: FrameModel) -> None:
            sc = self.scene
            w, h = fm.width, fm.height
            p.setBrush(Qt.NoBrush)                  # drawRect 会用当前画刷填充
            p.setPen(QPen(_c(C_HUD, 40), 1))
            p.drawRect(9, 9, w - 18, h - 18)

            # 左上角：LIVE + 载荷
            pulse = 0.5 + 0.5 * math.sin(self.t_s * 6.0 + time_phase())
            p.setBrush(_c((255, 70, 70), int(120 + 135 * pulse)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(22, 22, 9, 9)
            p.setPen(QPen(_c(C_HUD)))
            p.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
            p.drawText(38, 32, "LIVE · 真实数据流实时仿真")
            p.setFont(QFont("Microsoft YaHei", 8))
            p.setPen(QPen(_c(C_HUD_DIM)))
            p.drawText(22, 50, f"载荷：{sc.payload_label}")
            p.drawText(22, 64, f"SHA-256：{sc.payload_sha[:24]}…  "
                               f"{'完整性一致' if sc.integrity_ok else '完整性异常'}")

            # 右上角：大字指标
            t = self.t_s
            inflight = sc.interp_scalar(sc.series_inflight, t)
            gp = sc.interp_scalar(sc.series_goodput, t)
            dl = sc.interp_scalar(sc.series_delivered, t)
            x0 = w - 250
            p.setFont(QFont("Consolas", 17, QFont.Bold))
            p.setPen(QPen(_c((120, 245, 255))))
            p.drawText(x0, 40, f"{t:7.2f} s")
            p.setFont(QFont("Microsoft YaHei", 8))
            p.setPen(QPen(_c(C_HUD_DIM)))
            p.drawText(x0 + 132, 40, f"/ {sc.duration_s:.0f} s")
            rows = [
                ("在途分组", f"{inflight:,.0f}", (120, 245, 255)),
                ("实时吞吐", f"{gp:,.2f} Mbps", (255, 200, 90)),
                ("已送达", f"{dl:,.1f} Mbit", (120, 255, 190)),
            ]
            for i, (k, v, col) in enumerate(rows):
                yy = 66 + i * 20
                p.setPen(QPen(_c(C_HUD_DIM)))
                p.setFont(QFont("Microsoft YaHei", 8))
                p.drawText(x0, yy, k)
                p.setPen(QPen(_c(col)))
                p.setFont(QFont("Consolas", 11, QFont.Bold))
                p.drawText(x0 + 70, yy, v)

            # 左下角：业务流图例
            p.setFont(QFont("Microsoft YaHei", 8))
            for i, f in enumerate(sc.flows):
                yy = h - 108 + i * 17
                col = QColor(*[int(v) for v in f["color"]])
                p.setBrush(col); p.setPen(Qt.NoPen)
                p.drawEllipse(24, yy - 7, 8, 8)
                p.setPen(QPen(_c(C_HUD)))
                p.drawText(38, yy, f"{f['src_label']} → {f['dst_label']}"
                                   f"   {f['rate_mbps']:.2f} Mbps")

            # 底部时间轴
            self._paint_timeline(p, fm)

        def _paint_timeline(self, p, fm: FrameModel) -> None:
            sc = self.scene
            w, h = fm.width, fm.height
            bx, by, bw, bh = 24, h - 34, w - 48, 12
            p.setPen(Qt.NoPen)
            p.setBrush(_c((18, 34, 46), 220))
            p.drawRoundedRect(bx, by, bw, bh, 5, 5)
            # 缩略：已送达曲线
            if sc.series_delivered.size > 1 and sc.series_delivered.max() > 0:
                path = QPainterPath()
                n = sc.series_delivered.size
                mx = max(float(sc.series_delivered.max()), 1e-9)
                path.moveTo(bx, by + bh)
                for i in range(n):
                    xx = bx + bw * i / (n - 1)
                    yy = by + bh - bh * 0.85 * (sc.series_delivered[i] / mx)
                    path.lineTo(xx, yy)
                p.setPen(QPen(_c((90, 220, 220), 180), 1.2))
                p.drawPath(path)
            # 游标
            pos = bx + bw * (self.t_s / max(sc.duration_s, 1e-9))
            p.setBrush(_c((255, 90, 90), 230))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(int(pos - 2), by - 3, 4, bh + 6, 2, 2)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(_c(C_HUD, 90), 0.8))
            p.drawRoundedRect(bx, by, bw, bh, 5, 5)

    class LivePlayer(QWidget):
        """播放控制 + LiveView 的组合控件。"""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.scene: Optional[LiveScene] = None
            self.playing = False
            self.stretch = 6.0                # 时间拉伸：越小越快
            self.speed_cap = 6.0
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(4)
            self.view = LiveView(self)
            lay.addWidget(self.view, 1)

            bar = QHBoxLayout()
            self.btn_play = QPushButton("▶ 播放")
            self.btn_play.setObjectName("ghost")
            self.btn_play.clicked.connect(self.toggle)
            self.btn_reset = QPushButton("⟲ 重播")
            self.btn_reset.setObjectName("ghost")
            self.btn_reset.clicked.connect(self.reset)
            self.sld = QSlider(Qt.Horizontal)
            self.sld.setRange(0, 1000)
            self.sld.sliderMoved.connect(self._on_scrub)
            self.lab_speed = QLabel("倍速 ×1.0")
            self.lab_speed.setMinimumWidth(84)
            self.cmb_speed = QSlider(Qt.Horizontal)
            self.cmb_speed.setRange(0, 40)      # 0.25x … 10x
            self.cmb_speed.setValue(4)
            self.cmb_speed.setMaximumWidth(120)
            self.cmb_speed.valueChanged.connect(self._on_speed)
            for wdg in (self.btn_play, self.btn_reset, self.sld, self.lab_speed,
                        self.cmb_speed):
                bar.addWidget(wdg)
            lay.addLayout(bar)

            self.timer = QTimer(self)
            self.timer.setInterval(33)
            self.timer.timeout.connect(self._tick)
            self._on_speed(4)

        def set_scene(self, scene: Optional[LiveScene]) -> None:
            self.scene = scene
            self.view.set_scene(scene)
            self.timer.stop()
            self.playing = False
            self.btn_play.setText("▶ 播放")
            self.sld.setValue(0)

        def _on_speed(self, v: int) -> None:
            v = max(0, min(40, v))
            mult = 0.25 * (1.125 ** v)
            self.speed = mult
            self.lab_speed.setText(f"倍速 ×{mult:.2f}")
            # 实时推进 = 每帧 33 ms × 倍速；时间拉伸越大越慢
            self.dt = 0.033 * mult

        def _on_scrub(self, v: int) -> None:
            if self.scene is None:
                return
            self.view.t_s = self.scene.duration_s * v / 1000.0
            self.view.update()

        def toggle(self) -> None:
            if self.scene is None:
                return
            self.playing = not self.playing
            self.btn_play.setText("⏸ 暂停" if self.playing else "▶ 播放")
            if self.playing:
                self.timer.start()
            else:
                self.timer.stop()

        def reset(self) -> None:
            if self.scene is None:
                return
            self.view.t_s = 0.0
            self.view.update()
            self.sld.setValue(0)

        def _tick(self) -> None:
            if self.scene is None:
                return
            t = self.view.t_s + self.dt
            if t >= self.scene.duration_s:
                t = self.scene.duration_s
                self.playing = False
                self.timer.stop()
                self.btn_play.setText("▶ 播放")
            self.view.set_time(t, rotate_dt=self.dt)
            self.sld.setValue(int(1000 * t / max(self.scene.duration_s, 1e-9)))

        def snap_image(self, t_s: float, w: int = 1280, h: int = 760):
            """离屏渲染某一时刻的画面（用于截图/报告）。"""
            self.view.resize(w, h)
            self.view.t_s = float(t_s)
            return self.view.grab()


def _grat_circle(fm: FrameModel, param: float, is_lat: bool):
    """经纬网的一条线，返回 1~2 段 QPoint 列表（被地球遮挡处断开）。"""
    from PySide6.QtCore import QPoint
    n = 72
    segs: List[List["QPoint"]] = []
    cur: List["QPoint"] = []
    for i in range(n + 1):
        a = 2.0 * math.pi * i / n
        if is_lat:
            lat = math.radians(param)
            lon = a
        else:
            lat = math.asin(math.sin(a) * math.cos(math.radians(param)))
            lon = math.atan2(math.sin(a) * math.sin(math.radians(param)),
                             math.cos(a)) + math.radians(param)
        x = RE_KM * math.cos(lat) * math.cos(lon)
        y = RE_KM * math.cos(lat) * math.sin(lon)
        z = RE_KM * math.sin(lat)
        sx, sy, dz = project(np.array([[x, y, z]]), fm.az, fm.el, fm.r_px)
        hidden = dz[0] < 0 and (sx[0] ** 2 + sy[0] ** 2) < fm.r_px ** 2
        if hidden:
            if len(cur) > 1:
                segs.append(cur)
            cur = []
            continue
        cur.append(QPoint(int(fm.cx + sx[0]), int(fm.cy + sy[0])))
    if len(cur) > 1:
        segs.append(cur)
    return segs


def time_phase() -> float:
    import time as _t
    return (_t.perf_counter() % 10.0)


# --------------------------------------------------------------------------- #
# 单文件 HTML 导出
# --------------------------------------------------------------------------- #
def _b64_f32(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode("ascii")


def _b64_i32(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<i4").tobytes()).decode("ascii")


def export_html(scene: LiveScene, path: str, *, title: str = "",
                pos_bins: int = 150, payload: Optional[dict] = None) -> str:
    """导出**自包含单文件 HTML**（Canvas 2D 动画），可直接分享。"""
    T = np.linspace(0.0, scene.duration_s, max(8, pos_bins))
    n_sat = scene.n_sat
    # 卫星坐标按时间采样 → (n_t, n_sat, 3)
    pos_blob = np.zeros((T.size, max(n_sat, 1), 3), dtype=np.float32)
    for k, t in enumerate(T):
        if n_sat:
            pos_blob[k] = scene.nodes_pos_array(np.arange(n_sat), float(t))
    gs_blob = scene._pos_gs.astype(np.float32) if scene._pos_gs.size else np.zeros((0, 3), np.float32)

    flows = [{"id": f["id"], "src": f["src"], "dst": f["dst"],
              "srcLabel": f["src_label"], "dstLabel": f["dst_label"],
              "rate": f["rate_mbps"], "color": list(f["color"])}
             for f in scene.flows]

    data = {
        "duration": scene.duration_s,
        "nSat": n_sat,
        "nGs": scene.n_gs,
        "snapIv": scene.snapshot_interval_s,
        "posT": T.astype(float).tolist(),
        "posB64": _b64_f32(pos_blob),
        "gsB64": _b64_f32(gs_blob),
        "gsLabels": scene.gs_labels,
        "segT0": _b64_f32(scene.seg_t0),
        "segT1": _b64_f32(scene.seg_t1),
        "segU": _b64_i32(scene.seg_u),
        "segV": _b64_i32(scene.seg_v),
        "segFlow": _b64_i32(scene.seg_flow),
        "segSize": _b64_i32(scene.seg_size),
        "seriesT": scene.series_t.tolist(),
        "seriesInflight": scene.series_inflight.tolist(),
        "seriesGoodput": scene.series_goodput.tolist(),
        "seriesDelivered": scene.series_delivered.tolist(),
        "flows": flows,
        "title": title or scene.title,
        "payload": scene.payload_label,
        "sha": scene.payload_sha,
        "integrity": bool(scene.integrity_ok),
        "nSeg": int(scene.seg_t0.size),
    }
    html = _HTML_TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>SatNetSim · 真实数据流实时仿真</title>
<style>
  html,body{margin:0;height:100%;background:#03060c;color:#9fd9e8;
    font-family:"Microsoft YaHei",system-ui,sans-serif;overflow:hidden}
  #wrap{position:relative;width:100%;height:100%}
  canvas{display:block;width:100%;height:100%}
  #bar{position:absolute;left:0;right:0;bottom:0;padding:8px 14px;
    display:flex;gap:12px;align-items:center;background:rgba(4,10,18,.72);
    border-top:1px solid rgba(90,200,230,.18)}
  button{background:rgba(20,60,80,.6);color:#bfeaf6;border:1px solid rgba(90,200,230,.35);
    border-radius:6px;padding:5px 14px;font-size:13px;cursor:pointer}
  button:hover{background:rgba(30,100,130,.7)}
  input[type=range]{flex:1;accent-color:#38d6ff}
  #lab{font-size:12px;color:#7fb4c6;min-width:150px;text-align:right}
</style>
</head>
<body>
<div id="wrap"><canvas id="cv"></canvas>
<div id="bar">
  <button id="play">▶ 播放</button><button id="reset">⟲ 重播</button>
  <input id="scrub" type="range" min="0" max="1000" value="0"/>
  <div id="lab"></div>
</div></div>
<script>
const D = /*__DATA__*/;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let W=0,H=0,CX=0,CY=0,R=0;
function b64f32(s){const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
function b64i32(s){const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Int32Array(u.buffer);}
const posB = b64f32(D.posB64), gsB = b64f32(D.gsB64);
const sT0=b64f32(D.segT0), sT1=b64f32(D.segT1), sU=b64i32(D.segU),
      sV=b64i32(D.segV), sF=b64i32(D.segFlow);
const RE=6371, NT=D.posT.length;
const stars=[];for(let i=0;i<420;i++)stars.push([Math.random(),Math.random(),0.25+Math.random()*0.75]);
let az=0.55, el=0.42, t=0, playing=false, speed=1;
function resize(){W=cv.width=cv.clientWidth*devicePixelRatio;
  H=cv.height=cv.clientHeight*devicePixelRatio;
  CX=W/2;CY=H/2;R=Math.min(W,H)*0.30;}
window.addEventListener('resize',resize);
function proj(x,y,z){
  const ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el);
  const xx=x*ca+y*sa, yy=-x*sa+y*ca;
  const y2=yy*ce-z*se, z2=yy*se+z*ce;
  const s=R/RE; return [CX+xx*s, CY-z2*s, y2];
}
function satPos(i,ti){
  const k=Math.max(0,Math.min(NT-1,Math.round(ti/(D.duration||1)*(NT-1))));
  const o=(k*D.nSat+i)*3; return [posB[o],posB[o+1],posB[o+2]];
}
function gsPos(j){
  const o=j*3; return [gsB[o],gsB[o+1],gsB[o+2]];
}
function nodePos(n,ti){ return n<D.nSat ? satPos(n,ti) : gsPos(n-D.nSat); }
function visible(sx,sy,dz){return dz>=0 || (sx-CX)*(sx-CX)+(sy-CY)*(sy-CY)>=R*R;}
function draw(){
  const g=ctx.createLinearGradient(0,0,0,H);
  g.addColorStop(0,'#060a16');g.addColorStop(1,'#020308');
  ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
  for(const s of stars){const c=120+135*s[2];ctx.fillStyle='rgba('+c+','+c+','+c+','+(0.15+0.6*s[2])+')';
    ctx.beginPath();ctx.arc(s[0]*W,s[1]*H,1.4*s[2]+0.4,0,6.283);ctx.fill();}
  ctx.save();
  const gg=ctx.createRadialGradient(CX,CY,R*0.6,CX,CY,R*1.9);
  gg.addColorStop(0,'rgba(20,90,120,0.28)');gg.addColorStop(1,'rgba(5,15,30,0)');
  ctx.fillStyle=gg;ctx.beginPath();ctx.arc(CX,CY,R*1.9,0,6.283);ctx.fill();
  const bg=ctx.createRadialGradient(CX-R*0.35,CY-R*0.35,R*0.2,CX,CY,R*1.05);
  bg.addColorStop(0,'#18445c');bg.addColorStop(0.7,'#0a1a28');bg.addColorStop(1,'#040a12');
  ctx.fillStyle=bg;ctx.beginPath();ctx.arc(CX,CY,R,0,6.283);ctx.fill();
  ctx.strokeStyle='rgba(32,110,140,0.75)';ctx.lineWidth=1.4;ctx.stroke();
  ctx.strokeStyle='rgba(30,78,96,0.45)';ctx.lineWidth=0.7;
  for(let la=-60;la<=60;la+=30){ctx.beginPath();
    for(let i=0;i<=72;i++){const a=2*Math.PI*i/72,lat=la*Math.PI/180;
      const x=RE*Math.cos(lat)*Math.cos(a),y=RE*Math.cos(lat)*Math.sin(a),z=RE*Math.sin(lat);
      const p=proj(x,y,z); if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);}
    ctx.stroke();}
  ctx.restore();
  // 卫星
  for(let i=0;i<D.nSat;i++){const P=satPos(i,t),p=proj(P[0],P[1],P[2]);
    if(visible(p[0],p[1],p[2])){ctx.fillStyle='rgba(120,235,255,0.85)';
      ctx.beginPath();ctx.arc(p[0],p[1],1.5*devicePixelRatio,0,6.283);ctx.fill();}
    else{ctx.fillStyle='rgba(60,120,145,0.28)';ctx.beginPath();ctx.arc(p[0],p[1],1*devicePixelRatio,0,6.283);ctx.fill();}}
  // 地面站
  for(let j=0;j<D.nGs;j++){const P=gsPos(j),p=proj(P[0],P[1],P[2]);
    if(!visible(p[0],p[1],p[2]))continue;
    const rg=ctx.createRadialGradient(p[0],p[1],0,p[0],p[1],11*devicePixelRatio);
    rg.addColorStop(0,'rgba(255,186,80,0.8)');rg.addColorStop(1,'rgba(255,186,80,0)');
    ctx.fillStyle=rg;ctx.beginPath();ctx.arc(p[0],p[1],11*devicePixelRatio,0,6.283);ctx.fill();
    ctx.fillStyle='#ffba50';ctx.beginPath();ctx.arc(p[0],p[1],3*devicePixelRatio,0,6.283);ctx.fill();}
  // 彗星
  let cnt=0;
  for(let i=0;i<sT0.length;i++){
    if(sT0[i]<=t&&t<=sT1[i]){
      if(cnt++>900)break;
      const f=(t-sT0[i])/Math.max(sT1[i]-sT0[i],1e-9);
      const A=nodePos(sU[i],t),B=nodePos(sV[i],t);
      const pa=proj(A[0],A[1],A[2]),pb=proj(B[0],B[1],B[2]);
      if(!visible(pa[0],pa[1],pa[2])&&!visible(pb[0],pb[1],pb[2]))continue;
      const hx=pa[0]+(pb[0]-pa[0])*f, hy=pa[1]+(pb[1]-pa[1])*f;
      const c=D.flows[sF[i]%D.flows.length].color;
      ctx.strokeStyle='rgba('+c[0]+','+c[1]+','+c[2]+',0.28)';
      ctx.lineWidth=1.2*devicePixelRatio;
      ctx.beginPath();ctx.moveTo(hx,hy);ctx.lineTo(pa[0]+(hx-pa[0])*0.55,pa[1]+(hy-pa[1])*0.55);ctx.stroke();
      const rg=ctx.createRadialGradient(hx,hy,0,hx,hy,9*devicePixelRatio);
      rg.addColorStop(0,'rgba(255,255,255,0.95)');
      rg.addColorStop(0.3,'rgba('+c[0]+','+c[1]+','+c[2]+',0.8)');
      rg.addColorStop(1,'rgba('+c[0]+','+c[1]+','+c[2]+',0)');
      ctx.fillStyle=rg;ctx.beginPath();ctx.arc(hx,hy,9*devicePixelRatio,0,6.283);ctx.fill();
    }
  }
  // HUD
  const S=devicePixelRatio;
  ctx.fillStyle='rgba(150,230,245,0.9)';
  ctx.font=(13*S)+'px "Microsoft YaHei"';
  ctx.fillText('LIVE · 真实数据流实时仿真',22*S,30*S);
  ctx.fillStyle='rgba(95,140,160,0.95)';ctx.font=(10*S)+'px "Microsoft YaHei"';
  ctx.fillText('载荷：'+D.payload,22*S,50*S);
  ctx.fillText('SHA-256：'+D.sha.slice(0,24)+'…  '+(D.integrity?'完整性一致':'完整性异常'),22*S,66*S);
  const infl=interp(D.seriesT,D.seriesInflight,t), gp=interp(D.seriesT,D.seriesGoodput,t),
        dl=interp(D.seriesT,D.seriesDelivered,t);
  const X=W-260*S;
  ctx.fillStyle='#78f5ff';ctx.font='bold '+(20*S)+'px Consolas,monospace';
  ctx.fillText(t.toFixed(2)+' s',X,34*S);
  ctx.fillStyle='rgba(95,140,160,0.95)';ctx.font=(10*S)+'px "Microsoft YaHei"';
  ctx.fillText('/ '+D.duration.toFixed(0)+' s',X+120*S,34*S);
  const rows=[['在途分组',infl.toFixed(0),'#78f5ff'],['实时吞吐',gp.toFixed(2)+' Mbps','#ffc85a'],
              ['已送达',dl.toFixed(1)+' Mbit','#78ffbe']];
  rows.forEach((r,i)=>{const y=(64+i*20)*S;
    ctx.fillStyle='rgba(95,140,160,0.95)';ctx.font=(10*S)+'px "Microsoft YaHei"';ctx.fillText(r[0],X,y);
    ctx.fillStyle=r[2];ctx.font='bold '+(13*S)+'px Consolas,monospace';ctx.fillText(r[1],X+72*S,y);});
  D.flows.forEach((f,i)=>{const y=H-(120-i*18)*S;
    ctx.fillStyle='rgb('+f.color+')';ctx.beginPath();ctx.arc(28*S,y-4*S,4*S,0,6.283);ctx.fill();
    ctx.fillStyle='rgba(150,230,245,0.9)';ctx.font=(10*S)+'px "Microsoft YaHei"';
    ctx.fillText(f.srcLabel+' → '+f.dstLabel+'   '+f.rate.toFixed(2)+' Mbps',40*S,y);});
  // 时间轴
  const bx=24*S,by=H-36*S,bw=W-48*S,bh=12*S;
  ctx.fillStyle='rgba(18,34,46,0.85)';ctx.fillRect(bx,by,bw,bh);
  ctx.strokeStyle='rgba(90,220,220,0.5)';ctx.lineWidth=1*S;
  let mx=1e-9;for(const v of D.seriesDelivered)mx=Math.max(mx,v);
  ctx.beginPath();
  D.seriesDelivered.forEach((v,i)=>{const xx=bx+bw*i/(D.seriesDelivered.length-1),
    yy=by+bh-bh*0.85*v/mx; i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);});
  ctx.stroke();
  const pos=bx+bw*(t/D.duration);
  ctx.fillStyle='rgba(255,90,90,0.9)';ctx.fillRect(pos-2*S,by-3*S,4*S,bh+6*S);
}
function interp(xs,ys,x){ if(!xs.length)return 0;
  if(x<=xs[0])return ys[0]; if(x>=xs[xs.length-1])return ys[ys.length-1];
  let lo=0,hi=xs.length-1;
  while(hi-lo>1){const m=(lo+hi)>>1; if(xs[m]<=x)lo=m;else hi=m;}
  const f=(x-xs[lo])/(xs[hi]-xs[lo]); return ys[lo]+(ys[hi]-ys[lo])*f;}
function loop(){ if(playing){ t+=0.033*speed; if(t>=D.duration){t=D.duration;playing=false;
    document.getElementById('play').textContent='▶ 播放';}
    document.getElementById('scrub').value=1000*t/D.duration; }
  az+=0.0009; draw(); requestAnimationFrame(loop);}
document.getElementById('play').onclick=function(){playing=!playing;
  this.textContent=playing?'⏸ 暂停':'▶ 播放';};
document.getElementById('reset').onclick=function(){t=0;};
document.getElementById('scrub').oninput=function(){t=D.duration*this.value/1000;};
document.getElementById('lab').textContent='分组 '+D.nSeg+' 段 · 拖动可旋转视角';
resize(); loop();
</script>
</body></html>
"""


__all__ = [
    "LiveScene", "LiveView", "LivePlayer", "FrameModel", "build_frame",
    "project", "export_html", "FLOW_COLORS", "QT_AVAILABLE",
]


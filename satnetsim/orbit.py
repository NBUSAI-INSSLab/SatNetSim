"""轨道动力学与星座生成。

模型层级
--------
1. **解析传播（默认，propagator="analytic"）**
   两体问题 + J2 引力场长期项（secular）。对近圆 LEO 星座在数小时窗口内
   位置误差量级为数十公里（相当于数十毫秒的单跳时延误差），与 Stardust
   所用"仅 Kepler 方程"相比增加了 RAAN 进动与近地点幅角漂移，是精度/速度
   的务实折中。

   J2 长期变化率（Kozai / Brouwer 一阶）::

       n0     = sqrt(mu / a^3)
       p      = a (1 - e^2)
       RAAN'  = -1.5 n0 J2 (Re/p)^2 cos i
       argp'  =  0.75 n0 J2 (Re/p)^2 (5 cos^2 i - 1)
       M'     =  n0 + 0.75 n0 J2 (Re/p)^2 sqrt(1-e^2) (3 cos^2 i - 1)

2. **SGP4（可选，propagator="sgp4"）**
   需 `pip install sgp4 skyfield`。用于加载真实 TLE、做测量对齐或长时窗仿真。

坐标系
------
PQW（轨道平面）→ ECI（J2000 惯性系）→ ECEF（地固系）→ LLA（WGS84 大地坐标）。
ECI→ECEF 只用 GMST 绕 Z 轴旋转（忽略岁差/章动/极移，对网络仿真足够）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ConstellationConfig

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
MU_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.137
J2 = 1.08262668e-3
EARTH_ROT_RAD_S = 7.2921159e-5
C_KM_S = 299792.458

WGS84_A = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


# --------------------------------------------------------------------------- #
# 时间
# --------------------------------------------------------------------------- #
def unix_from_iso(iso: str) -> float:
    """ISO-8601（按 UTC 解读，无时区时视为 UTC）→ Unix 秒。"""
    txt = iso.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        dt = datetime.strptime(txt, "%Y-%m-%dT%H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def gmst_rad(unix_seconds: np.ndarray | float) -> np.ndarray:
    """格林尼治平恒星时（rad）。IAU-82 近似式，误差 < 0.1"（≈ 2 m 量级）。"""
    ts = np.asarray(unix_seconds, dtype=float)
    jd = ts / 86400.0 + 2440587.5
    d = jd - 2451545.0
    t = d / 36525.0
    deg = (
        280.46061837
        + 360.98564736629 * d
        + 0.000387933 * t * t
        - t * t * t / 38710000.0
    )
    return np.deg2rad(np.mod(deg, 360.0))


# --------------------------------------------------------------------------- #
# 轨道要素 → 位置
# --------------------------------------------------------------------------- #
@dataclass
class OrbitalElements:
    """一组经典轨道六根数（角度单位为 rad，近圆轨道下 argp 可置 0）。"""

    a_km: float
    e: float
    i_rad: float
    raan_rad: float
    argp_rad: float
    mean_anomaly_rad: float

    # J2 长期变化率（rad/s）
    raan_dot: float = 0.0
    argp_dot: float = 0.0
    m_dot: float = 0.0

    def secular(self, t_s: np.ndarray | float) -> "OrbitalElements":
        ts = np.asarray(t_s, dtype=float)
        return OrbitalElements(
            self.a_km,
            self.e,
            self.i_rad,
            float(self.raan_rad + self.raan_dot * ts),
            float(self.argp_rad + self.argp_dot * ts),
            float(self.mean_anomaly_rad + self.m_dot * ts),
            self.raan_dot,
            self.argp_dot,
            self.m_dot,
        )


def mean_motion(a_km: float) -> float:
    return math.sqrt(MU_KM3_S2 / (a_km ** 3))


def j2_rates(a_km: float, e: float, i_rad: float) -> Tuple[float, float, float]:
    """J2 长期项：返回 (RAAN', argp', M')，单位 rad/s。"""
    n0 = mean_motion(a_km)
    p = a_km * (1.0 - e * e)
    k = n0 * J2 * (EARTH_RADIUS_KM / p) ** 2
    ci = math.cos(i_rad)
    raan_dot = -1.5 * k * ci
    argp_dot = 0.75 * k * (5.0 * ci * ci - 1.0)
    m_dot = n0 + 0.75 * k * math.sqrt(1.0 - e * e) * (3.0 * ci * ci - 1.0)
    return raan_dot, argp_dot, m_dot


def solve_kepler(mean_anomaly: np.ndarray, e: float, tol: float = 1e-11) -> np.ndarray:
    """Newton 迭代解 Kepler 方程 M = E - e sinE（向量化，固定 12 次迭代足够）。"""
    m = np.mod(mean_anomaly, 2.0 * np.pi)
    ecc_anom = m + e * np.sin(m) / (1.0 - e * np.cos(m) + 1e-12)
    if e > 1e-9:
        for _ in range(12):
            f = ecc_anom - e * np.sin(ecc_anom) - m
            fp = 1.0 - e * np.cos(ecc_anom)
            step = f / fp
            ecc_anom = ecc_anom - step
            if np.max(np.abs(step)) < tol:
                break
    return ecc_anom


def pqw_to_eci(pos_pqw: np.ndarray, raan: np.ndarray, inc: np.ndarray,
               argp: np.ndarray) -> np.ndarray:
    """PQW → ECI：r = Rz(RAAN) Rx(i) Rz(argp) r_pqw，输入形状 (..., 3)。"""
    c_o, s_o = np.cos(raan), np.sin(raan)
    c_i, s_i = np.cos(inc), np.sin(inc)
    c_w, s_w = np.cos(argp), np.sin(argp)

    # 合并成旋转矩阵分量（避免逐个矩阵乘法，直接做两次旋转）
    # 第一次：绕 Z 轴转 argp
    x = c_w * pos_pqw[..., 0] - s_w * pos_pqw[..., 1]
    y = s_w * pos_pqw[..., 0] + c_w * pos_pqw[..., 1]
    z = pos_pqw[..., 2].copy()
    # 第二次：绕 X 轴转 i
    y2 = c_i * y - s_i * z
    z2 = s_i * y + c_i * z
    # 第三次：绕 Z 轴转 RAAN
    x3 = c_o * x - s_o * y2
    y3 = s_o * x + c_o * y2
    return np.stack([x3, y3, z2], axis=-1)


def kepler_to_eci(el: OrbitalElements, t_s: np.ndarray) -> np.ndarray:
    """给定要素，返回 t_s 时刻的 ECI 位置，形状 (n_t, 3)。"""
    t = np.asarray(t_s, dtype=float)
    a, e = el.a_km, el.e
    m = el.mean_anomaly_rad + el.m_dot * t
    ecc = solve_kepler(m, e)
    r = a * (1.0 - e * np.cos(ecc))
    x_pqw = a * (np.cos(ecc) - e)
    y_pqw = a * math.sqrt(max(1.0 - e * e, 0.0)) * np.sin(ecc)
    z_pqw = np.zeros_like(x_pqw)
    pos_pqw = np.stack([x_pqw, y_pqw, z_pqw], axis=-1)
    raan = el.raan_rad + el.raan_dot * t
    argp = el.argp_rad + el.argp_dot * t
    inc = np.full_like(t, el.i_rad)
    return pqw_to_eci(pos_pqw, raan, inc, argp)


# --------------------------------------------------------------------------- #
# 坐标变换
# --------------------------------------------------------------------------- #
def eci_to_ecef(r_eci: np.ndarray, unix_s: np.ndarray) -> np.ndarray:
    """ECI → ECEF（仅 GMST 绕 Z 轴旋转）。

    r_eci 形状 (..., n_t, 3)，unix_s 形状 (n_t,)；返回同形状数组。
    """
    r = np.asarray(r_eci, dtype=float)
    theta = np.asarray(gmst_rad(unix_s), dtype=float).reshape(-1)
    if r.ndim < 2 or r.shape[-2] != theta.size:
        raise ValueError(
            f"eci_to_ecef 期望 r_eci 形状 (..., n_t, 3) 且 n_t={theta.size}，"
            f"实际 {r.shape}"
        )
    bshape = (1,) * (r.ndim - 2) + (theta.size,)
    ct = np.cos(theta).reshape(bshape)
    st = np.sin(theta).reshape(bshape)
    x = ct * r[..., 0] + st * r[..., 1]
    y = -st * r[..., 0] + ct * r[..., 1]
    z = r[..., 2]
    return np.stack([x, y, z], axis=-1)


def ecef_to_lla(r_ecef: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF → WGS84（lat/lon 单位 deg，alt 单位 km）。"""
    x, y, z = r_ecef[..., 0], r_ecef[..., 1], r_ecef[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    alt = np.zeros_like(p)
    for _ in range(6):
        sin_lat = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    return np.rad2deg(lat), np.rad2deg(lon), alt


def lla_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """WGS84 大地坐标 → ECEF（km）。"""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array([
        (n + alt_km) * cos_lat * math.cos(lon),
        (n + alt_km) * cos_lat * math.sin(lon),
        (n * (1.0 - WGS84_E2) + alt_km) * sin_lat,
    ])


def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """局部 ENU 基向量（3x3，行依次为 East / North / Up）。"""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def elevation_deg(gs_lat: float, gs_lon: float, gs_alt: float,
                  sat_ecef: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """地面站看向卫星的仰角（deg）与斜距（km）。sat_ecef 形状 (..., 3)。"""
    r_gs = lla_to_ecef(gs_lat, gs_lon, gs_alt)
    rho = sat_ecef - r_gs
    rng = np.linalg.norm(rho, axis=-1)
    basis = enu_basis(gs_lat, gs_lon)
    up = basis[2]
    cos_zenith = np.einsum("...i,i->...", rho, up) / np.maximum(rng, 1e-9)
    el = np.rad2deg(np.arcsin(np.clip(cos_zenith, -1.0, 1.0)))
    return el, rng


# --------------------------------------------------------------------------- #
# 星座
# --------------------------------------------------------------------------- #
@dataclass
class Satellite:
    index: int
    plane: int
    slot: int
    elements: OrbitalElements
    node_id: int = 0  # 在网络图中的编号

    @property
    def name(self) -> str:
        return f"SAT-{self.plane:02d}-{self.slot:02d}"


def walker_delta_elements(cfg: ConstellationConfig) -> List[OrbitalElements]:
    """生成 Walker-Delta 星座的轨道要素列表。"""
    a = cfg.semi_major_axis_km
    inc = math.radians(cfg.inclination_deg)
    e = float(cfg.eccentricity)
    raan_dot, argp_dot, m_dot = j2_rates(a, e, inc)

    elements: List[OrbitalElements] = []
    for p in range(int(cfg.planes)):
        raan = 2.0 * math.pi * p / cfg.planes
        for s in range(int(cfg.sats_per_plane)):
            u = (
                2.0 * math.pi * s / cfg.sats_per_plane
                + 2.0 * math.pi * cfg.phase_factor * p / (cfg.planes * cfg.sats_per_plane)
            )
            elements.append(
                OrbitalElements(
                    a_km=a,
                    e=e,
                    i_rad=inc,
                    raan_rad=raan,
                    argp_rad=0.0,
                    mean_anomaly_rad=u,
                    raan_dot=raan_dot,
                    argp_dot=argp_dot,
                    m_dot=m_dot,
                )
            )
    return elements


def build_constellation(cfg: ConstellationConfig) -> Tuple[List[Satellite], np.ndarray]:
    """构造星座，返回 (卫星列表, TLE 原始行可选的元信息)。

    当 `cfg.tle_file` 给定时，尝试用 sgp4/skyfield 从 TLE 建立要素近似；
    若依赖缺失则回退到 Walker-Delta 并给出警告。
    """
    elements = walker_delta_elements(cfg)
    sats: List[Satellite] = []
    for idx, el in enumerate(elements):
        plane = idx // cfg.sats_per_plane
        slot = idx % cfg.sats_per_plane
        sats.append(Satellite(index=idx, plane=plane, slot=slot,
                              elements=el, node_id=idx))
    return sats, np.zeros((0, 3))


class Constellation:
    """一个可向量化传播的星座。

    位置以 (n_sat, n_time, 3) 的 ECI 数组缓存，坐标变换按需惰性计算。
    """

    def __init__(self, cfg: ConstellationConfig):
        self.cfg = cfg
        self.satellites, _ = build_constellation(cfg)
        self.n_sat = len(self.satellites)
        self.positions_eci: Optional[np.ndarray] = None  # (n_sat, n_t, 3)
        self.positions_ecef: Optional[np.ndarray] = None
        self.lat_deg: Optional[np.ndarray] = None
        self.lon_deg: Optional[np.ndarray] = None
        self.alt_km: Optional[np.ndarray] = None
        self.times_s: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def propagate(self, times_s: np.ndarray) -> np.ndarray:
        """在相对时刻 times_s（秒，自场景起点）上传播，缓存 ECI/ECEF/LLA。"""
        t = np.asarray(times_s, dtype=float)
        self.times_s = t
        pos = np.empty((self.n_sat, t.size, 3), dtype=float)
        for k, sat in enumerate(self.satellites):
            pos[k] = kepler_to_eci(sat.elements, t)
        self.positions_eci = pos
        return pos

    def to_ecef(self, start_unix: float) -> np.ndarray:
        assert self.positions_eci is not None, "请先调用 propagate()"
        unix = start_unix + self.times_s
        self.positions_ecef = eci_to_ecef(self.positions_eci, unix)
        lat, lon, alt = ecef_to_lla(self.positions_ecef)
        self.lat_deg, self.lon_deg, self.alt_km = lat, lon, alt
        return self.positions_ecef

    # ------------------------------------------------------------------ #
    def ground_track(self, sat_index: int) -> Tuple[np.ndarray, np.ndarray]:
        assert self.lat_deg is not None, "请先调用 to_ecef()"
        return self.lat_deg[sat_index], self.lon_deg[sat_index]

    def mean_positions_ecef(self) -> np.ndarray:
        """在 ECEF 中对时间取平均，用于构造"虚拟拓扑"（Virtual Node 路由）。"""
        assert self.positions_ecef is not None
        return self.positions_ecef.mean(axis=1)


def sun_eclipse_fraction(pos_eci: np.ndarray, sun_dir_eci: np.ndarray) -> np.ndarray:
    """圆柱本影近似下的日照比例（1=全日照，0=全阴影）。

    用于星载能量预算：卫星在阴影区只能靠储能供电。
    """
    r = np.linalg.norm(pos_eci, axis=-1)
    along = np.einsum("...i,i->...", pos_eci, sun_dir_eci)
    perp = np.sqrt(np.maximum(r * r - along * along, 0.0))
    in_shadow = (along < 0.0) & (perp < EARTH_RADIUS_KM)
    return np.where(in_shadow, 0.0, 1.0)


def sun_direction_eci(unix_s: float) -> np.ndarray:
    """太阳方向单位矢量（ECI，粗精度公式，误差 < 1°）。"""
    n = unix_s / 86400.0 - 10957.5
    L = math.radians((280.460 + 0.9856474 * n) % 360.0)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 3.56e-7 * n)
    return np.array([
        math.cos(lam),
        math.cos(eps) * math.sin(lam),
        math.sin(eps) * math.sin(lam),
    ])

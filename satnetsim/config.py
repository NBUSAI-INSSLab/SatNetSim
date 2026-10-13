"""场景配置：全部仿真参数的数据模型与出厂预设。

所有配置均为 `dataclass`，可 `to_dict()` / `from_dict()` 与 YAML/JSON 互转，
便于 GUI 保存场景、批量扫参（sweep）与结果复现。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# 星座 / 轨道
# --------------------------------------------------------------------------- #
@dataclass
class ConstellationConfig:
    """Walker-Delta 星座参数（也支持直接给定 TLE 文件）。

    Walker-Delta 记号 ``i: P/S/F``：倾角 i，轨道面数 P，每面卫星数 S，相位因子 F。
    第 p 面第 s 颗卫星：
        RAAN = 2*pi*p/P
        u    = 2*pi*s/S + 2*pi*F*p/(P*S)      （纬度幅角，近圆轨道下即 argp+nu）
    """

    name: str = "Walker-550"
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    planes: int = 6
    sats_per_plane: int = 11
    phase_factor: int = 1
    eccentricity: float = 0.0

    # 可选：直接给出 TLE 文件（每行三元素，或标准两行格式），为空则用 Walker 生成
    tle_file: Optional[str] = None

    # 传播器选择："analytic"（两体+J2 长期项，内置）| "sgp4"（需安装 sgp4/skyfield）
    propagator: str = "analytic"

    @property
    def n_satellites(self) -> int:
        return int(self.planes) * int(self.sats_per_plane)

    @property
    def semi_major_axis_km(self) -> float:
        from .orbit import EARTH_RADIUS_KM

        return EARTH_RADIUS_KM + float(self.altitude_km)


# --------------------------------------------------------------------------- #
# 地面站 / 用户
# --------------------------------------------------------------------------- #
@dataclass
class GroundStationConfig:
    """地面关口站或用户终端（简化：每站有若干可并行天线）。"""

    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.05
    antennas: int = 4
    min_elevation_deg: float = 20.0
    is_gateway: bool = True  # False 表示用户终端（只做接入，不出口）

    @property
    def lla(self) -> Tuple[float, float, float]:
        return (self.lat_deg, self.lon_deg, self.alt_km)


# --------------------------------------------------------------------------- #
# 链路
# --------------------------------------------------------------------------- #
@dataclass
class LinkConfig:
    """链路几何与物理层参数。"""

    # 星间链路
    isl_mode: str = "+Grid"  # +Grid | *Grid | Grid-V | none
    isl_max_range_km: float = 5000.0
    isl_bandwidth_mbps: float = 100_000.0  # 激光链路：100 Gbps（Starlink 在轨量级）
    isl_latency_ms_fixed: float = 0.5  # 星上光交换/转发固定时延
    isl_seam_links: bool = False  # 是否在反向缝处建立跨面链路

    # 星地链路
    gsl_bandwidth_mbps: float = 500.0
    gsl_max_range_km: float = 2200.0
    gsl_latency_ms_fixed: float = 1.0

    # 地球遮挡：两点连线到地心的最小距离须大于此值
    min_clearance_km: float = 100.0

    # 链路预算（仅当 use_physics_budget=True 时生效，用于按香农容量动态定容）
    use_physics_budget: bool = False
    rf_carrier_ghz: float = 20.0
    tx_power_dbw: float = 3.0
    tx_gain_dbi: float = 38.0
    rx_gain_dbi: float = 38.0
    noise_temp_k: float = 500.0
    atm_loss_db: float = 1.0


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
@dataclass
class RoutingConfig:
    protocol: str = "shortest_delay"
    # 可选：shortest_delay | shortest_hop | virtual_node | geographic | load_balanced
    recompute_interval_s: float = 10.0  # 路由重算周期（≈快照间隔）
    max_hops: int = 64
    load_balance_alpha: float = 0.5  # load_balanced 的拥塞惩罚权重
    vn_lambda: float = 0.5  # 虚拟节点路由：虚拟拓扑相对物理拓扑的压缩比


# --------------------------------------------------------------------------- #
# 传输
# --------------------------------------------------------------------------- #
@dataclass
class TransportConfig:
    protocol: str = "cubic"  # cubic | bbr | hybla | reno | quic | dtn | pep
    quic_cc: str = "bbr"  # QUIC 底层的拥塞控制
    initial_window_seg: int = 10
    mss_bytes: int = 1460
    buffer_bytes: int = 512 * 1024  # 瓶颈链路缓冲
    rtt_ref_ms: float = 25.0  # Hybla 参考 RTT
    loss_prob: float = 0.002  # 【解析模型口径】每 RTT 有效丢包率（有损场景）
    packet_loss_prob: float = 0.0002  # 【真实包级栈口径】每包丢包率（LEO 残余 BER 折算，0.02%）
    dtn_bundle_overhead: float = 0.05  # DTN 包头开销比例
    pep_split_gain: float = 0.7  # PEP 分割连接对 RTT 敏感度的折减
    dtn_contact_period_s: float = 20.0  # DTN 每跳等待下一个接触机会的平均周期
    quic_migration: bool = True  # QUIC 是否启用连接迁移（路径切换不重握手）

    # ---- 真实协议栈（第 ⑤ 步传输分析改用包级状态机，而非 RTT 轮次解析模型）----
    engine: str = "real"           # real（真实包级 TCP 状态机）| analytic（旧解析模型）| both
    sack: bool = True              # 选择性确认（丢包后只重传缺口）
    delayed_ack: bool = True       # 接收端延迟 ACK（每 2 段或 40 ms 确认一次）
    rcvbuf_bytes: int = 4 * 1024 * 1024   # 接收窗口上限（真实卫星链路需窗口缩放，BDP 可达数 MB）
    app_payload_mbit: float = 256.0 # 业务载荷（Mbit），真实栈实际要传的字节数
    app_rtt_loss: float = 0.0      # 应用层附加丢包率（0 → 沿用 loss_prob）
    path_switch_penalty: bool = True  # 路径切换时是否触发 RTO（真实 TCP 会遇到）
    tcp_algos: List[str] = field(
        default_factory=lambda: ["reno", "cubic", "bbr", "hybla"])
    route_sweep: bool = True       # 「路由 × 真实传输」联合评测（每种路由各跑一次真实 TCP）
    loss_sweep: bool = True        # 真实 TCP 的链路丢包率扫描（展示拥塞控制算法分化）
    loss_sweep_points: List[float] = field(
        default_factory=lambda: [0.0, 0.0002, 0.001, 0.01])  # 每包丢包率
    loss_sweep_algos: List[str] = field(
        default_factory=lambda: ["cubic", "bbr", "hybla"])
    loss_sweep_mbit: float = 256.0 # 扫描使用的载荷（比主评测小以控制耗时）


# --------------------------------------------------------------------------- #
# 星载计算
# --------------------------------------------------------------------------- #
@dataclass
class ComputeConfig:
    # 单星算力平台（默认参考星载智算计算机量级）
    sat_capacity_tops: float = 8.0
    sat_power_idle_w: float = 20.0
    sat_power_peak_w: float = 120.0
    sat_energy_budget_wh: float = 400.0  # 单圈可用电能（含储能）
    # 地面云
    cloud_capacity_tops: float = 200.0
    # 任务流
    task_count: int = 200
    task_size_mbit: Tuple[float, float] = (50.0, 800.0)
    task_cycles_gops: Tuple[float, float] = (20.0, 400.0)
    task_deadline_s: float = 60.0
    offload_ratio: float = 0.5  # 愿意卸载到卫星执行的任务比例
    scheduler: str = "value_density"  # earliest_finish | value_density | random | local_only
    radiation_derate: float = 0.15  # 辐射导致的有效算力折减
    aging: float = 2.0  # 价值密度调度的优先级老化系数（抑制长任务饥饿）


# --------------------------------------------------------------------------- #
# 真实评测（用真实数据传输 + 真实计算任务做端到端验证）
# --------------------------------------------------------------------------- #
@dataclass
class RealEvalConfig:
    """把"解析模型假设的数据传输"与"合成生成的计算任务"换成真实量的评测开关。

    与前述各层的区别：前面的传输/计算层是**解析模型**（窗口级积分、解析 FLOP
    统计），本模块**真的把字节推过内核协议栈**、**真的在本机执行算子**，因此：
      - 传输侧：真实 TCP（内核栈）+ 用户态逐包可靠传输（cwnd 状态机）；
      - 计算侧：真实算子（ResNet-18 推理、FFT、GEMM、OpenCV、SHA-256…）。
    结果作为"真实 vs 解析"的对照，用于校验解析模型的可信度。
    """

    enabled: bool = False          # 总开关；False 时引擎完全不进入真实评测
    transfer: bool = True          # 真实数据传输评测
    compute: bool = True           # 真实计算任务评测

    # ---- 传输侧 ----
    payload_mbit: float = 16.0     # 真实 TCP 评测载荷（Mbit）
    tcp_trials: int = 2            # 真实 TCP 重复次数
    packet_trials: int = 1         # 每种协议的真实包级重复次数
    packet_algos: List[str] = field(
        default_factory=lambda: ["reno", "cubic", "hybla"])
    packet_bandwidth_mbps: Optional[float] = None  # None → min(链路带宽, 20)
    packet_payload_mbit: Optional[float] = None    # None → 按包级 BDP 自适应
    compare_analytic: bool = True  # 同口径跑解析模型做对照

    # ---- 计算侧 ----
    real_task_count: int = 120     # 真实负载任务流的任务数
    repeats: int = 3               # 每个真实负载的标定重复次数

    # 无端到端路径时（地面站 < 2 / 拓扑不连通）是否退回标称链路剖面
    fallback_profile: bool = True


# --------------------------------------------------------------------------- #
# 真实数据流仿真（真字节 × 真拓扑 × 逐包多跳转发）
# --------------------------------------------------------------------------- #
@dataclass
class DesConfig:
    """全局离散事件内核（NS-3 式统一时间轴）编排的开关与规模。

    与默认步进式流水线的关系：步进式负责"各层把结论算出来"；本编排把
    拓扑刷新 / 路由重算 / 路径切换 / TCP 流到达与完成 / 计算任务到达与
    服务 / 指标采样放进**同一条事件时间轴**，回答"事件到底如何交错"。
    """

    enabled: bool = False          # 默认关闭（在步进结果之上叠加一次事件级编排）
    trace: bool = True             # 记录事件轨迹（GUI ⑫ 页签甘特图 / JSON 导出）
    n_flows: int = 3               # 时间轴上的 TCP 流条数（泊松到达）
    flow_mbit: float = 32.0        # 每流载荷（Mbit）
    tcp_algo: str = "bbr"          # 流使用的拥塞控制算法
    flow_rate: float = 0.0         # 流到达率（条/秒）；0 → 自动均匀铺满窗口
    n_tasks: int = 200             # 计算任务到达数（泊松过程）
    task_rate: float = 0.0         # 任务到达率（个/秒）；0 → n_tasks/duration
    n_nodes: int = 8               # 参与排队的星载节点数（负载均衡分发）
    sample_every_s: float = 5.0    # 指标采样间隔
    seed: int = 7


@dataclass
class RealDataConfig:
    """把**真实业务数据**送进**真实时变拓扑**逐包转发的开关。

    与 `RealEvalConfig` 的分工：
      - `RealEvalConfig` 在**回环**上标定"真实 TCP/算子能跑多快"（协议实现保真）；
      - 本配置在**星座图**上跑"真字节能不能、多久、多完整地穿过网络"
        （拓扑与排队保真），并把每一跳的时空轨迹留作实时回放。
    """

    enabled: bool = True           # 是否在引擎里执行真实数据流仿真
    payload_kind: str = "image"    # image | text | telemetry | file | mixed
    payload_mbit: float = 64.0     # 单轮真实载荷大小（Mbit）；时长更长时按轮复用
    payload_file: str = ""         # payload_kind=file 时的本地文件路径
    mss_bytes: int = 1200          # 分组最大段长（分片大小）

    rate_mbps: float = 0.0         # 业务速率；<=0 → 自动（链路瓶颈 & 铺满整个窗口）
    n_flows: int = 3               # 并发业务流路数（各自独立选路）
    duration_s: float = 0.0        # 业务仿真窗口；0 → 跟随 SimConfig.duration_s（不再截断）
    buffer_mult: float = 2.0       # 每条有向链路缓冲 = mult × BDP

    # ---- 传输层保真 ----
    reliable: bool = True          # True → 端到端真实可靠传输（TCP 状态机：cwnd/RTO/重传/SACK）
    tx_window_mult: float = 1.0    # 链路并发发射倍数（可靠模式下按 cwnd 节流）

    # ---- 规模预算（长窗口下控制事件数与回放数据量）----
    max_packets: int = 30000       # 全窗口分组总数上限，超出则按比例下调业务速率

    src_gs: int = -1               # -1 → 自动取球面距离最远的一对
    dst_gs: int = -1

    sample_bins: int = 240         # 时间序列分箱数（供实时图表）
    trace_limit: int = 4000        # 单文件 HTML 导出时保留的最大分组轨迹数


# --------------------------------------------------------------------------- #
# 全局仿真
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    constellation: ConstellationConfig = field(default_factory=ConstellationConfig)
    ground_stations: List[GroundStationConfig] = field(default_factory=list)
    link: LinkConfig = field(default_factory=LinkConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    real_eval: RealEvalConfig = field(default_factory=RealEvalConfig)
    real_data: RealDataConfig = field(default_factory=RealDataConfig)
    des: DesConfig = field(default_factory=DesConfig)

    duration_s: float = 600.0
    snapshot_interval_s: float = 10.0
    start_utc: str = "2026-03-20T00:00:00"
    seed: int = 2026
    # 抽样业务流：{src_gs, dst_gs, size_mbit, start_s}
    flows: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------ 序列化 ------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimConfig":
        return cls(
            constellation=ConstellationConfig(**data.get("constellation", {})),
            ground_stations=[
                GroundStationConfig(**g) for g in data.get("ground_stations", [])
            ],
            link=LinkConfig(**data.get("link", {})),
            routing=RoutingConfig(**data.get("routing", {})),
            transport=TransportConfig(**data.get("transport", {})),
            compute=ComputeConfig(**data.get("compute", {})),
            real_eval=RealEvalConfig(**data.get("real_eval", {})),
            real_data=RealDataConfig(**data.get("real_data", {})),
            des=DesConfig(**data.get("des", {})),
            duration_s=data.get("duration_s", 600.0),
            snapshot_interval_s=data.get("snapshot_interval_s", 10.0),
            start_utc=data.get("start_utc", "2026-03-20T00:00:00"),
            seed=data.get("seed", 2026),
            flows=data.get("flows", []),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SimConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# 出厂预设
# --------------------------------------------------------------------------- #
def _gs(name: str, lat: float, lon: float, **kw: Any) -> GroundStationConfig:
    return GroundStationConfig(name=name, lat_deg=lat, lon_deg=lon, **kw)


# 六个跨洲关口站（参考 Starlink/OneWeb 主要落地点的量级）
DEFAULT_GS: List[GroundStationConfig] = [
    _gs("北京", 39.90, 116.41, antennas=8),
    _gs("上海", 31.23, 121.47, antennas=6),
    _gs("法兰克福", 50.11, 8.68, antennas=6),
    _gs("洛杉矶", 34.05, -118.24, antennas=6),
    _gs("圣保罗", -23.55, -46.63, antennas=4),
    _gs("悉尼", -33.87, 151.21, antennas=4),
]


def _make_preset(
    name: str,
    planes: int,
    sats_per_plane: int,
    incl: float,
    alt: float,
    phase: int = 1,
    **overrides: Any,
) -> SimConfig:
    cfg = SimConfig(
        constellation=ConstellationConfig(
            name=name,
            altitude_km=alt,
            inclination_deg=incl,
            planes=planes,
            sats_per_plane=sats_per_plane,
            phase_factor=phase,
        ),
        ground_stations=[GroundStationConfig(**vars(g)) for g in DEFAULT_GS],
    )
    for key, value in overrides.items():
        section, _, attr = key.partition(".")
        if attr:
            setattr(getattr(cfg, section), attr, value)
        else:
            setattr(cfg, section, value)
    return cfg


PRESETS: Dict[str, SimConfig] = {
    # 简化教学星座：66 星（对标 Iridium 规模）
    "iridium-66": _make_preset("Iridium-66", 6, 11, 86.4, 780.0, 1),
    # 单壳 Starlink Shell-1 量级（1584 星，550 km / 53°）
    "starlink-shell1": _make_preset("Starlink-Shell1", 72, 22, 53.0, 550.0, 1,
                                    routing__protocol="shortest_delay"),
    # 缩小版 Starlink（便于笔记本上交互式运行）
    "starlink-lite": _make_preset("Starlink-Lite", 16, 18, 53.0, 550.0, 1),
    # Walker 极轨（含反向缝，用于验证 seam 处理）
    "walker-polar": _make_preset("Walker-Polar", 8, 16, 87.9, 1000.0, 2),
    # 计算星座（对标"三体计算星座"首发规模的算力参数）
    "compute-constellation": _make_preset("Compute-Constellation", 6, 12, 97.4, 500.0, 1),
}


def preset(name: str) -> SimConfig:
    """取一份预设配置的深拷贝（避免调用方误改全局预设）。"""
    if name not in PRESETS:
        raise KeyError(f"未知预设 {name!r}，可选：{sorted(PRESETS)}")
    return SimConfig.from_dict(PRESETS[name].to_dict())


def real_eval_preset(name: str = "starlink-lite",
                     **real_kw: Any) -> SimConfig:
    """在预设基础上打开真实评测（真实数据传输 + 真实计算任务）。

    真实评测会真的跑内核 TCP 与真实算子，耗时远高于解析层（数十秒量级），
    因此默认不随预设开启，需要显式调用本函数或设置 ``cfg.real_eval.enabled``。
    """
    cfg = preset(name)
    cfg.real_eval.enabled = True
    for k, v in real_kw.items():
        if not hasattr(cfg.real_eval, k):
            raise KeyError(f"RealEvalConfig 无字段 {k!r}")
        setattr(cfg.real_eval, k, v)
    return cfg

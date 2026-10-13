"""SatNetSim —— 面向"通信-感知-计算"一体的卫星互联网仿真器。

设计定位
--------
现有工具各有取舍：
  * Hypatia / StarPerf  —— 网络层保真度高，但只做"网"，没有星上算力；
  * StarryNet / OpenSN / RHONE / Celestial —— 容器化仿真/仿真床，资源开销大、单机规模受限；
  * Stardust —— 面向 3D Continuum 的编排仿真，但计算模型是黑盒插件；
  * ASTRA-sim —— 只做地面数据中心内部互联与集合通信。

SatNetSim 的目标是把 **时变拓扑 + 路由 + 传输 + 星载计算**
放进同一个 Python 时间轴内，形成可解释、可复现、可扩展的一体化仿真内核。

分层（自底向上）
----------------
    orbit       轨道动力学与星座生成（Kepler / Walker-Delta / 可选 SGP4）
    topology    可见性与链路几何，星间链路（ISL）模式，时变图快照
    routing     路由协议插件（时延最短路 / 虚拟节点 / 地理贪婪 / 负载均衡）
    transport   传输协议插件（CUBIC / BBR / Hybla / QUIC / DTN / PEP）
    computing   星载计算（任务模型、算力-能耗-热约束、调度器）
    engine      仿真引擎（快照驱动 + 事件推进 + 指标采集）
    gui         PySide6 图形界面

单位约定
--------
除特别注明外：距离 km、时间 s、带宽 Mbps、算力 TOPS、功率 W、能量 Wh。
"""

from .config import (
    ConstellationConfig,
    ComputeConfig,
    GroundStationConfig,
    LinkConfig,
    RoutingConfig,
    SimConfig,
    TransportConfig,
    PRESETS,
)

__version__ = "0.1.0"

__all__ = [
    "ConstellationConfig",
    "GroundStationConfig",
    "LinkConfig",
    "RoutingConfig",
    "TransportConfig",
    "ComputeConfig",
    "SimConfig",
    "PRESETS",
    "__version__",
]

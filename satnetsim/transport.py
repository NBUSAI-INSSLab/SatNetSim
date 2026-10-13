"""传输协议插件层：面向"高 BDP + 时变路径"的流级建模。

为何不用包级仿真
----------------
包级（packet-level）仿真最精确，但代价是 ns-3 那样的事件数爆炸。SatNetSim 的
定位是"系统级设计空间探索"，因此在传输层采用**窗口级 / RTT 轮次（round）模型**：
在每个 RTT 内认为 cwnd 恒定，按轮次积分，丢弃/重传按概率期望折算。这一层次
足以刻画不同拥塞控制在高 BDP 链路上的本质差异（收敛速度、是否依赖丢包、
是否填满缓冲），而计算量比包级低 3–4 个数量级。

链路本质：LEO 单跳 RTT ≈ 2×(500/7.5)+固定值 ≈ 5–10 ms，
端到端（含洋际多跳）RTT 可达 40–100 ms，BDP 在 10 Gbps 链路上为
10 Gbps × 60 ms = 75 MB —— 远超任何地面接入链路的量级。这正是
"卫星传输"与"地面传输"的分水岭，也是 BBR 类模型型算法在 LEO 上表现更好的原因。

建模要点
--------
* **拥塞控制**：Reno / CUBIC（三次函数窗口增长）/ BBR（基于 BtlBW 与 RTT_min 的
  速率控制）/ Hybla（按 ρ = RTT/RTT_ref 放大窗口增长率，补偿长时延）。
* **缓冲与排队**：窗口超过 BDP + buffer 时判定为队列溢出丢包。
* **握手开销**：TCP = 1 RTT（三次握手）+ 1 RTT（TLS1.3）；QUIC = 1 RTT（或 0-RTT
  复用）；QUIC 连接迁移使路径切换不触发重握手。
* **PEP**：分割连接把"广域 RTT"从拥塞控制回路上摘除，等效 RTT 按 ρ 折减。
* **DTN（BPv7）**：存储转发，每跳需等待下一个接触机会，吞吐不由 cwnd 决定而由
  接触窗口与队列决定。

文献锚点：ESA QUICoS / QUICOPTSAT（QUIC over satellite，IETF "Careful Resume"
draft-ietf-tsvwg-careful-resume）；QUIC-BBR 在 LEO 5G IAB 视频流场景下综合最优
（arXiv:2604.16634）；Bujari et al., Computer Communications 225 (2024) 239–249.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import TransportConfig

# 协议元信息：握手 RTT 数、是否依赖丢包、是否支持路径迁移
PROTOCOL_META: Dict[str, Dict[str, object]] = {
    "reno":   {"label": "TCP Reno",     "handshake_rtt_rtts": 2, "loss_based": True,  "migrate": False},
    "cubic":  {"label": "TCP CUBIC",    "handshake_rtt_rtts": 2, "loss_based": True,  "migrate": False},
    "bbr":    {"label": "TCP BBRv3",    "handshake_rtt_rtts": 2, "loss_based": False, "migrate": False},
    "hybla":  {"label": "TCP Hybla",    "handshake_rtt_rtts": 2, "loss_based": True,  "migrate": False},
    "quic":   {"label": "QUIC/HTTP3",   "handshake_rtt_rtts": 1, "loss_based": False, "migrate": True},
    "pep":    {"label": "TCP + PEP",    "handshake_rtt_rtts": 2, "loss_based": True,  "migrate": False},
    "dtn":    {"label": "DTN (BPv7)",   "handshake_rtt_rtts": 0, "loss_based": False, "migrate": True},
}


@dataclass
class FlowResult:
    protocol: str
    label: str
    size_mbit: float
    path_owd_ms: float
    rtt_ms: float
    bottleneck_mbps: float
    handshake_s: float
    transfer_s: float
    completion_s: float
    goodput_mbps: float
    losses: int
    rounds: int
    cwnd_trace: List[float] = field(default_factory=list)   # 单位：段
    rate_trace: List[float] = field(default_factory=list)   # 单位：Mbps

    @property
    def efficiency(self) -> float:
        """相对瓶颈带宽的链路利用率。"""
        if self.bottleneck_mbps <= 0:
            return 0.0
        return self.goodput_mbps / self.bottleneck_mbps


class TransportSimulator:
    """滑动窗口 / RTT 轮次传输模型。"""

    def __init__(self, cfg: TransportConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    def simulate(
        self,
        path_owd_ms: float,
        bottleneck_mbps: float,
        size_mbit: float,
        protocol: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        n_hops: int = 1,
        path_is_dtn_friendly: bool = False,
    ) -> FlowResult:
        proto = (protocol or self.cfg.protocol).lower()
        meta = PROTOCOL_META.get(proto, PROTOCOL_META["cubic"])
        rng = rng or np.random.default_rng(0)

        owd = max(float(path_owd_ms), 0.05) / 1000.0  # 秒
        bw = max(float(bottleneck_mbps), 0.01)
        size_bytes = max(float(size_mbit), 1e-6) * 1e6 / 8.0
        mss = float(self.cfg.mss_bytes)
        b = bw * 1e6 / 8.0  # bytes/s

        # 有效 RTT（PEP 把广域 RTT 从拥塞控制回路上摘除）
        if proto == "pep":
            rtt = 2.0 * owd * (1.0 - self.cfg.pep_split_gain) + 2.0 * self.cfg.rtt_ref_ms / 1000.0
        else:
            rtt = 2.0 * owd
        rtt = max(rtt, 1e-4)

        # 握手：TCP 1 RTT + TLS1.3 1 RTT；QUIC 1 RTT（或 0-RTT 复用）；DTN 无
        handshake = float(meta["handshake_rtt_rtts"]) * rtt * 0.5
        if proto == "quic" and not self.cfg.quic_migration:
            handshake = rtt * 0.5

        # DTN 走独立的接触窗口模型
        if proto == "dtn":
            return self._simulate_dtn(
                size_bytes, b, owd, rtt, size_mbit, meta, rng, max(int(n_hops), 1)
            )

        bdp = b * rtt  # 填满管道所需字节
        cap_per_round = b * rtt  # 单个 RTT 最多能交付的字节
        cwnd = self.cfg.initial_window_seg * mss
        ssthresh = float("inf")
        w_max = cwnd
        cubic_epoch = 0.0
        rtt_ref = max(self.cfg.rtt_ref_ms, 1.0) / 1000.0

        delivered = 0.0
        losses = 0
        rounds = 0
        cwnd_trace: List[float] = []
        rate_trace: List[float] = []
        max_rounds = 400_000

        while delivered < size_bytes and rounds < max_rounds:
            rounds += 1
            # 实际在途窗口（受瓶颈管道容量 + 缓冲限制）
            win = min(cwnd, bdp + self.cfg.buffer_bytes)
            x = min(win, cap_per_round, size_bytes - delivered)

            # 丢包判定
            p = float(self.cfg.loss_prob)
            congestion_loss = 0
            if bool(meta["loss_based"]) and cwnd > bdp + self.cfg.buffer_bytes:
                congestion_loss = 1
            elif proto == "quic" and cwnd > bdp + 0.5 * self.cfg.buffer_bytes:
                congestion_loss = 1
            if congestion_loss or (p > 0 and rng.random() < p):
                losses += 1
                lost = True
            else:
                lost = False

            # 重传占用：丢包时按"一个窗口的 1/4 被浪费"近似
            waste = 0.25 * x if lost else 0.0
            delivered += max(x - waste, 0.0)

            rtt_now = rtt
            rate_trace.append((x - waste) * 8.0 / rtt_now / 1e6)
            cwnd_trace.append(cwnd / mss)

            # ---------------- 拥塞控制状态更新 ---------------- #
            if proto == "reno":
                if lost:
                    ssthresh = max(cwnd / 2.0, 2 * mss)
                    cwnd = ssthresh
                elif cwnd < ssthresh:
                    cwnd = min(cwnd * 2.0, bdp + self.cfg.buffer_bytes)
                else:
                    cwnd += mss
            elif proto == "cubic":
                if lost:
                    w_max = cwnd
                    cwnd = max(cwnd * 0.7, 2 * mss)
                    cubic_epoch = 0.0
                else:
                    c = 0.4
                    beta = 0.7
                    k = ((w_max * (1.0 - beta)) / c / mss) ** (1.0 / 3.0) if w_max > mss else 0.0
                    cubic_epoch += rtt
                    target = c * ((cubic_epoch - k) ** 3) * mss + w_max
                    if cwnd < ssthresh:  # 慢启动阶段仍按指数增长
                        cwnd = min(cwnd * 2.0, bdp + self.cfg.buffer_bytes)
                    else:
                        cwnd = max(cwnd, min(target, bdp + self.cfg.buffer_bytes))
            elif proto == "hybla":
                rho = rtt / rtt_ref
                if lost:
                    ssthresh = max(cwnd / 2.0, 2 * mss)
                    cwnd = ssthresh
                elif cwnd < ssthresh:
                    cwnd = min(cwnd * (1.0 + rho * rho), bdp + self.cfg.buffer_bytes)
                else:
                    cwnd += mss * rho * rho / max(cwnd / bdp, 1.0)
            elif proto in ("bbr", "quic"):
                # 模型型：向 BDP 收敛后保持，不靠丢包探测上限
                if lost:
                    cwnd = max(cwnd * 0.9, 2 * mss)
                elif cwnd < bdp:
                    cwnd = min(cwnd * 2.0, bdp * 1.05)
                else:
                    cwnd = bdp * 1.05  # 稳态：不制造常驻队列
            elif proto == "pep":
                # 分割连接：局部回环 RTT 很小，窗口增长接近地面网络
                if lost:
                    cwnd = max(cwnd * 0.7, 2 * mss)
                elif cwnd < bdp:
                    cwnd = min(cwnd * 2.0, bdp + self.cfg.buffer_bytes)
                else:
                    cwnd += mss
            cwnd = max(cwnd, mss)

        transfer = rounds * rtt
        completion = handshake + transfer
        goodput = size_mbit / max(completion, 1e-9)
        return FlowResult(
            protocol=proto,
            label=str(meta["label"]),
            size_mbit=float(size_mbit),
            path_owd_ms=owd,
            rtt_ms=rtt * 1000.0,
            bottleneck_mbps=bw,
            handshake_s=handshake,
            transfer_s=transfer,
            completion_s=completion,
            goodput_mbps=goodput,
            losses=losses,
            rounds=rounds,
            cwnd_trace=cwnd_trace,
            rate_trace=rate_trace,
        )

    # ------------------------------------------------------------------ #
    def _simulate_dtn(
        self,
        size_bytes: float,
        b: float,
        owd: float,
        rtt: float,
        size_mbit: float,
        meta: Dict[str, object],
        rng: np.random.Generator,
        n_hops: int = 1,
    ) -> FlowResult:
        """DTN：存储转发。吞吐由接触机会密度与每跳存储队列决定。

        假设每个 bundle 1 MB；每跳平均等待半个接触周期（contact period），
        随后按该跳带宽串行推送。端到端 ≈ 首包穿越时延 + (n-1) × 出队间隔。
        """
        overhead = 1.0 + self.cfg.dtn_bundle_overhead
        bundle_bytes = 1.0e6
        n_bundle = max(int(math.ceil(size_bytes * overhead / bundle_bytes)), 1)
        per_hop_wait = self.cfg.dtn_contact_period_s * 0.5
        hops = max(int(n_hops), 1)
        transmit = bundle_bytes / max(b, 1e-9)
        first = hops * (per_hop_wait + transmit) + 2.0 * owd
        interval = transmit + per_hop_wait * 0.1
        transfer = first + (n_bundle - 1) * interval
        completion = transfer
        return FlowResult(
            protocol="dtn",
            label=str(meta["label"]),
            size_mbit=float(size_mbit),
            path_owd_ms=owd * 1000.0,
            rtt_ms=rtt * 1000.0,
            bottleneck_mbps=b * 8.0 / 1e6,
            handshake_s=0.0,
            transfer_s=transfer,
            completion_s=completion,
            goodput_mbps=size_mbit / max(completion, 1e-9),
            losses=0,
            rounds=n_bundle,
            cwnd_trace=[],
            rate_trace=[],
        )

    # ------------------------------------------------------------------ #
    def compare(
        self,
        path_owd_ms: float,
        bottleneck_mbps: float,
        size_mbit: float,
        protocols: Sequence[str] = ("reno", "cubic", "bbr", "hybla", "quic", "pep"),
        n_trials: int = 5,
        seed: int = 7,
        n_hops: int = 1,
    ) -> List[FlowResult]:
        """多协议对比（多次试验取均值，抑制丢包随机性）。"""
        out: List[FlowResult] = []
        for proto in protocols:
            trials = [
                self.simulate(
                    path_owd_ms, bottleneck_mbps, size_mbit,
                    protocol=proto, rng=np.random.default_rng(seed + k),
                    n_hops=n_hops,
                )
                for k in range(max(1, n_trials))
            ]
            mean = FlowResult(
                protocol=proto,
                label=trials[0].label,
                size_mbit=size_mbit,
                path_owd_ms=path_owd_ms,
                rtt_ms=trials[0].rtt_ms,
                bottleneck_mbps=bottleneck_mbps,
                handshake_s=float(np.mean([t.handshake_s for t in trials])),
                transfer_s=float(np.mean([t.transfer_s for t in trials])),
                completion_s=float(np.mean([t.completion_s for t in trials])),
                goodput_mbps=float(np.mean([t.goodput_mbps for t in trials])),
                losses=int(np.mean([t.losses for t in trials])),
                rounds=int(np.mean([t.rounds for t in trials])),
                cwnd_trace=trials[0].cwnd_trace,
                rate_trace=trials[0].rate_trace,
            )
            out.append(mean)
        return out


# --------------------------------------------------------------------------- #
# 辅助分析
# --------------------------------------------------------------------------- #
def bdp_bytes(bandwidth_mbps: float, rtt_ms: float) -> float:
    """带宽时延积（字节）——判断"窗口是否成为瓶颈"的第一指标。"""
    return bandwidth_mbps * 1e6 / 8.0 * rtt_ms / 1000.0


def sweep_loss(
    cfg: TransportConfig,
    path_owd_ms: float,
    bottleneck_mbps: float,
    size_mbit: float,
    losses=(0.0, 0.001, 0.005, 0.01, 0.02, 0.05),
    protocols: Sequence[str] = ("cubic", "bbr", "hybla", "quic"),
    n_trials: int = 5,
    seed: int = 11,
) -> Dict[str, List[float]]:
    """扫描链路丢包率，观察"丢包型"与"模型型"拥塞控制的性能分化。

    这是 LEO 传输的核心实验：星地链路的残余误码率随仰角/天气变化，
    丢包型算法（CUBIC）会把误码误判为拥塞而大幅降窗，模型型算法（BBR）
    则不受影响 —— 这正是 BBR 类算法在卫星链路上更有优势的机理。
    """
    sim = TransportSimulator(cfg)
    out: Dict[str, List[float]] = {p: [] for p in protocols}
    base = cfg.loss_prob
    for p_loss in losses:
        cfg.loss_prob = float(p_loss)
        res = sim.compare(
            path_owd_ms, bottleneck_mbps, size_mbit,
            protocols=protocols, n_trials=n_trials, seed=seed,
        )
        for r in res:
            out[r.protocol].append(r.goodput_mbps)
    cfg.loss_prob = base
    return out


def path_capacity_analysis(
    path_owd_ms: float, link_bandwidths_mbps: Sequence[float], cfg: TransportConfig
) -> Dict[str, float]:
    """给出固定窗口下的理论上限（窗口受限 vs 带宽受限的分界）。"""
    bn = float(min(link_bandwidths_mbps)) if len(link_bandwidths_mbps) else 0.0
    rtt = 2.0 * path_owd_ms
    bdp = bdp_bytes(bn, rtt)
    max_win = 65535.0 * 64  # 放大窗口上限的示意值
    window_limited = max_win / max(rtt / 1000.0, 1e-6) * 8.0 / 1e6
    return {
        "bottleneck_mbps": bn,
        "rtt_ms": rtt,
        "bdp_mb": bdp / 1e6,
        "buffer_mb": cfg.buffer_bytes / 1e6,
        "window_limited_mbps": window_limited,
    }

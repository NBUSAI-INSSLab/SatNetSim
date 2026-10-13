"""真实 TCP 协议栈：包级状态机（拥塞控制 / RTO / 快速重传 / SACK / 流控）。

与 ``transport.py`` 的分工
--------------------------
``transport.py`` 是**窗口级 / RTT 轮次解析模型**：每个 RTT 内认为 cwnd 恒定，
按轮次积分，适合做几十种参数的快速扫描，但看不到"段级"行为（RTO 退避、乱序、
SACK 缺口、延迟 ACK、窗口探测）。

本模块是**真实的包级 TCP 状态机**：每个段都有发送时刻、到达时刻与确认时刻，
接收端做累积确认 + SACK，发送端按下列 RFC 演化窗口与时延估计：

* RFC 5681 —— 慢启动 / 拥塞避免 / 快速重传 / 快速恢复（Reno）
* RFC 8312 —— CUBIC 的三次窗口增长
* RFC 6298 —— SRTT / RTTVAR / RTO 计算，Karn 算法（重传段不采样）
* RFC 6675 —— 基于 SACK 的丢失检测与恢复
* RFC 1122 —— 延迟 ACK（每 2 段或 40 ms）

它跑在 :class:`PathProfile`（由星座快照推导的**时变** RTT / 带宽 / 丢包剖面）上，
因此能如实反映低轨星座的两个本质挑战：

1. **高 BDP**：窗口收敛慢，慢启动过冲的直接代价是链路队列溢出丢包；
2. **路径时变**：卫星与链路切换造成 RTT 阶跃、乱序与 RTO 误判 ——
   解析模型完全看不到这一层。

已实现算法：``reno`` / ``cubic`` / ``bbr`` / ``hybla``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .des import Simulator

INF = float("inf")

# 事件类型（作为调度 priority 与 trace 标签使用；定序由内核 (t, priority, uid) 保证）
_EV_SEG_ARRIVE = 0   # 段到达接收端
_EV_ACK_ARRIVE = 1   # ACK 到达发送端
_EV_RTO = 2          # 重传超时
_EV_KIND_NAME = {0: "tcp:seg", 1: "tcp:ack", 2: "tcp:rto"}

ALGO_LABELS: Dict[str, str] = {
    "reno": "TCP Reno（RFC 5681）",
    "cubic": "TCP CUBIC（RFC 8312）",
    "bbr": "TCP BBRv3（模型型）",
    "hybla": "TCP Hybla（长时延补偿）",
    "vegas": "TCP Vegas（时延型）",
}


# --------------------------------------------------------------------------- #
# 1. 时变路径剖面
# --------------------------------------------------------------------------- #
@dataclass
class PathProfile:
    """一条端到端路径的**时变**剖面（由拓扑快照逐点推导）。

    每个采样点 k 对应一个拓扑快照，给出该时刻这条路径的往返时延、瓶颈带宽、
    每包丢包概率与跳数。段在发送时刻取用剖面值，因此"路径切换"会真实地
    体现为后续段的 RTT 阶跃与乱序。
    """

    t: np.ndarray            # (K,) 时刻 (s)
    owd_ms: np.ndarray       # (K,) 单向传播时延（含各跳）
    rtt_ms: np.ndarray       # (K,) 往返时延
    bw_mbps: np.ndarray      # (K,) 端到端瓶颈带宽
    loss: np.ndarray         # (K,) 每包丢包概率
    hops: np.ndarray         # (K,) 跳数
    queue_bytes: np.ndarray  # (K,) 瓶颈链路缓冲上限
    loss_is_per_rtt: bool = False  # loss 是"每 RTT 丢包率"还是"每包丢包率"（默认每包，与文献口径一致）
    src_label: str = "源"
    dst_label: str = "目的"

    def __post_init__(self) -> None:
        for name in ("t", "owd_ms", "rtt_ms", "bw_mbps", "loss", "hops", "queue_bytes"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))

    @property
    def duration_s(self) -> float:
        return float(self.t[-1]) if self.t.size else 0.0

    @property
    def mean_rtt_ms(self) -> float:
        return float(np.mean(self.rtt_ms)) if self.rtt_ms.size else 0.0

    @property
    def min_rtt_ms(self) -> float:
        return float(np.min(self.rtt_ms)) if self.rtt_ms.size else 0.0

    @property
    def max_rtt_ms(self) -> float:
        return float(np.max(self.rtt_ms)) if self.rtt_ms.size else 0.0

    @property
    def rtt_swing_ms(self) -> float:
        """RTT 波动幅度（LEO 路径切换强度的直接度量）。"""
        return self.max_rtt_ms - self.min_rtt_ms

    @property
    def path_changes(self) -> int:
        """相邻快照跳数/时延发生阶跃的次数。"""
        if self.t.size < 2:
            return 0
        d_owd = np.abs(np.diff(self.owd_ms))
        d_hop = np.abs(np.diff(self.hops))
        return int(np.sum((d_owd > 0.2) | (d_hop > 0.4)))

    def index_at(self, t_s: float) -> int:
        if self.t.size == 0:
            return 0
        return int(np.clip(np.searchsorted(self.t, t_s, side="right") - 1,
                           0, self.t.size - 1))

    def at(self, t_s: float) -> Tuple[float, float, float, int, float]:
        """返回 (owd_s, bw_mbps, loss, hops, queue_bytes)。"""
        k = self.index_at(t_s)
        return (float(self.owd_ms[k]) / 1000.0, float(self.bw_mbps[k]),
                float(self.loss[k]), int(self.hops[k]),
                float(self.queue_bytes[k]))

    def describe(self) -> str:
        return (f"RTT {self.min_rtt_ms:.1f}–{self.max_rtt_ms:.1f} ms"
                f"（摆动 {self.rtt_swing_ms:.1f} ms，阶跃 {self.path_changes} 次），"
                f"瓶颈 {np.min(self.bw_mbps):.0f}–{np.max(self.bw_mbps):.0f} Mbps，"
                f"跳数 {int(np.min(self.hops))}–{int(np.max(self.hops))}")


def profile_from_route(
    snapshots: Sequence,
    *,
    paths: Optional[Sequence[Sequence[int]]] = None,
    route_fn: Optional[Callable[[object], Sequence[int]]] = None,
    times: Optional[Sequence[float]] = None,
    loss_prob: float = 0.0,
    queue_mult: float = 2.0,
    src_label: str = "源",
    dst_label: str = "目的",
    nominal_rtt_ms: float = 0.0,
) -> PathProfile:
    """用逐快照路由结果构建时变路径剖面。

    传入 ``paths``（与 snapshots 等长的节点序列列表）最直接；也可传 ``route_fn``
    由本函数逐快照调用。路径不可达的采样点沿用上一次可用结果
    （真实网络里这是"路由尚未收敛"，而不是瞬间断链）。
    """
    times = list(times) if times is not None else None
    loss_prob = max(float(loss_prob), 0.0)

    # 预计算边索引（每个快照一份）
    def _edge_map(snap):
        m = {}
        for i in range(snap.n_edges):
            a, b = int(snap.src[i]), int(snap.dst[i])
            m[(a, b)] = i
            m[(b, a)] = i
        return m

    ts: List[float] = []
    owds: List[float] = []
    bws: List[float] = []
    losses: List[float] = []
    hops_l: List[int] = []
    queues: List[float] = []
    last: Optional[Tuple[float, float, int]] = None

    for k, snap in enumerate(snapshots):
        if paths is not None:
            path = list(paths[k]) if k < len(paths) else []
        elif route_fn is not None:
            path = list(route_fn(snap))
        else:
            path = []
        if len(path) >= 2:
            try:
                em = _edge_map(snap)
                owd = 0.0
                bw = INF
                ok = True
                for u, v in zip(path[:-1], path[1:]):
                    ei = em.get((int(u), int(v)))
                    if ei is None:
                        ok = False
                        break
                    owd += float(snap.delay_ms[ei])
                    bw = min(bw, float(snap.capacity_mbps[ei]))
                if ok and np.isfinite(bw):
                    last = (owd, bw, len(path) - 1)
            except Exception:
                pass
        if last is None:
            continue
        owd, bw, hops_n = last
        ts.append(float(times[k]) if times and k < len(times)
                  else float(k) * 10.0)
        owds.append(owd)
        bws.append(bw)
        losses.append(loss_prob)
        hops_l.append(hops_n)
        queues.append(max(queue_mult * bw * 1e6 / 8.0 * max(owd / 1000.0, 1e-4), 16384.0))

    if not ts:
        # 无可用路由：退回标称剖面（保证仍有可评测的传输行为）
        rtt = max(nominal_rtt_ms, 1.0)
        q = max(queue_mult * 50.0 * 1e6 / 8.0 * (rtt / 2000.0), 16384.0)
        ts, owds, bws = [0.0, 1.0], [rtt / 2.0, rtt / 2.0], [50.0, 50.0]
        losses, hops_l, queues = [loss_prob, loss_prob], [1, 1], [q, q]

    return PathProfile(
        t=np.asarray(ts), owd_ms=np.asarray(owds),
        rtt_ms=np.asarray(owds) * 2.0, bw_mbps=np.asarray(bws),
        loss=np.asarray(losses), hops=np.asarray(hops_l),
        queue_bytes=np.asarray(queues),
        src_label=src_label, dst_label=dst_label,
    )


# --------------------------------------------------------------------------- #
# 2. 结果容器
# --------------------------------------------------------------------------- #
@dataclass
class TcpSample:
    """一个采样时刻的连接状态（供曲线绘制与机理分析）。"""

    t: float
    cwnd: float          # 字节
    ssthresh: float      # 字节
    inflight: int        # 字节
    delivered: int       # 已确认字节
    rtt_ms: float        # 平滑 RTT（srtt）
    rto_ms: float        # 当前 RTO
    phase: str = "ss"    # ss（慢启动）| ca（拥塞避免）| rec（恢复）| to（超时）


@dataclass
class TcpFlowResult:
    """一条真实 TCP 连接的完整结果。"""

    protocol: str
    label: str
    bytes_total: int
    bytes_delivered: int
    mss: int  # noqa: E501

    completion_s: float = float("nan")
    goodput_mbps: float = 0.0
    utilization: float = 0.0

    segments_sent: int = 0
    retransmits: int = 0
    fast_retransmits: int = 0
    timeouts: int = 0
    dupacks: int = 0
    pkts_lost: int = 0
    reordering_events: int = 0

    rtt_min_ms: float = float("nan")
    rtt_mean_ms: float = float("nan")
    rtt_p95_ms: float = float("nan")
    rttvar_ms: float = 0.0
    final_rto_ms: float = 0.0
    cwnd_max_bytes: float = 0.0
    ssthresh_final: float = INF
    mean_inflight_bytes: float = 0.0
    mean_bdp_bytes: float = 0.0
    window_limited_mbps: float = 0.0   # 接收窗口 × RTT 给出的吞吐上限
    rcvbuf_bytes: int = 0

    samples: List[TcpSample] = field(default_factory=list)
    phase_time: Dict[str, float] = field(default_factory=dict)
    path_note: str = ""

    @property
    def size_mbit(self) -> float:
        return self.bytes_total * 8.0 / 1e6

    @property
    def retransmit_ratio(self) -> float:
        """重传率 = 重传段数 ÷ 已发送段数（含超时后的窗口重发）。"""
        return self.retransmits / max(self.segments_sent, 1)

    @property
    def cwnd_trace(self) -> List[float]:
        """兼容旧绘图接口：拥塞窗口序列（段数）。"""
        return [s.cwnd / max(self.mss, 1) for s in self.samples]

    @property
    def efficiency(self) -> float:
        return self.utilization

    def describe(self) -> str:
        return (f"{self.label}：完成 {self.completion_s:.3f} s，"
                f"吞吐 {self.goodput_mbps:.2f} Mbps（链路利用率 "
                f"{self.utilization*100:.1f}%），重传 {self.retransmits}"
                f"（快速 {self.fast_retransmits} / 超时 {self.timeouts}），"
                f"RTT {self.rtt_min_ms:.1f}–{self.rtt_p95_ms:.1f} ms")


# --------------------------------------------------------------------------- #
# 3. TCP 连接状态机
# --------------------------------------------------------------------------- #
@dataclass
class TcpConfig:
    """真实 TCP 行为参数（默认取 Linux 常见值）。"""

    mss: int = 1460
    initial_window_seg: int = 10
    rcvbuf_bytes: int = 262_144
    delayed_ack: bool = True
    sack: bool = True
    min_rto_s: float = 0.2
    max_rto_s: float = 60.0
    initial_rto_s: float = 1.0
    rtt_ref_ms: float = 25.0        # Hybla 参考 RTT
    bbr_gain_cycle: bool = True     # BBR 是否周期性 ProbeBW
    sample_every_s: float = 0.05    # 状态采样间隔
    max_time_s: float = 120.0       # 单连接最长仿真时长（超过记为"未完成"）
    # 路径切换（LEO 卫星切换 / 星间链路重路由）代价：切换瞬间旧路径在途包作废，
    # 发射静默 cost 秒；penalty 开启时窗口收缩到初始窗（等价于一次无重传风暴的 RTO）
    path_switch_penalty: bool = True
    path_switch_cost_s: float = 0.2


class TcpConnection:
    """单条 TCP 连接的包级状态机（发送端 + 接收端）。"""

    # ------------------------------------------------------------------ #
    def __init__(self, profile: PathProfile, cfg: TcpConfig, algo: str = "cubic",
                 seed: int = 0, progress: Optional[Callable[[float], None]] = None,
                 sim: Optional[Simulator] = None):
        self.p = profile
        self.cfg = cfg
        self.algo = algo.lower()
        self.rng = np.random.default_rng(seed)
        self.progress = progress
        # 事件内核：外部可注入（星座级全局编排），否则 run() 时自建
        self.sim = sim
        self._on_done: Optional[Callable[["TcpConnection"], None]] = None

        # ---- 发送端状态 ----
        self.mss = int(cfg.mss)
        bdp_max = (float(np.max(profile.bw_mbps) * 1e6 / 8.0
                         * np.max(profile.rtt_ms) / 1000.0)
                   if profile.bw_mbps.size else 0.0)
        # 窗口硬上限：管道容量的 2.5 倍与接收窗口 2 倍取大者
        self._cwnd_cap = max(2.5 * bdp_max, float(cfg.rcvbuf_bytes) * 2.0,
                             4.0 * self.mss)
        # 丢包率换算：链路参数按"每 RTT 有效丢包率"给出 → 折算到每包
        self.loss_scale = 1.0
        if profile.loss_is_per_rtt and profile.bw_mbps.size:
            segs_per_rtt = ((float(np.mean(profile.bw_mbps)) * 1e6 / 8.0)
                            * (float(np.mean(profile.rtt_ms)) / 1000.0)
                            / max(self.mss, 1))
            self.loss_scale = 1.0 / max(segs_per_rtt, 1.0)
        # 窗口受限吞吐上限（流控瓶颈，LEO 高 BDP 下往往先于带宽饱和）
        _rtt_mean = (float(np.mean(profile.rtt_ms)) / 1000.0
                     if profile.rtt_ms.size else 0.05)
        self.window_limited_mbps = (float(cfg.rcvbuf_bytes) * 8.0
                                    / max(_rtt_mean, 1e-6) / 1e6)
        self.cwnd = float(cfg.initial_window_seg * self.mss)
        self.ssthresh = INF
        self.snd_una = 0
        self.snd_nxt = 0
        self.rwnd = float(cfg.rcvbuf_bytes)
        self.srtt: Optional[float] = None
        self.rttvar: Optional[float] = None
        self.rto = float(cfg.initial_rto_s)
        self.dupacks = 0
        self.recover = 0
        self.in_fast_recovery = False
        self.rto_backoff = 0
        self._recovery_retx: set = set()
        self._rto_seq = 0

        # ---- CUBIC / BBR 状态 ----
        self._w_max = self.cwnd
        self._t_epoch = 0.0
        self._k = 0.0
        self._bbr_btlbw = 0.0
        self._bbr_rtprop = INF
        self._bbr_rtprop_stamp = -INF
        self._bbr_phase = "startup"
        self._bbr_round = 0

        # ---- 接收端状态 ----
        self.rcv_expect = 0
        self.rcv_ooo: Dict[int, int] = {}
        self._ack_pending = 0

        # ---- 统计 ----
        self.seg_state: Dict[int, List] = {}   # seq -> [size, tx_t, retx, acked]
        self.sent_count = 0
        self.retransmits = 0
        self.fast_retx = 0
        self.timeouts = 0
        self.pkts_lost = 0
        self.reorder_events = 0
        self.rtt_samples: List[float] = []
        self.samples: List[TcpSample] = []
        self.inflight_bytes_accum = 0.0
        self.inflight_samples = 0

        # ---- 链路/事件 ----
        self.link_free = 0.0
        self.queue_bytes = 0.0
        self.t = 0.0
        self._done = False          # 未装载/已终止的连接统一按此判定（kill / 切换注入）
        self._next_sample = 0.0
        self._phase_time: Dict[str, float] = {}
        self._last_phase_t = 0.0

    # ------------------------------------------------------------------ #
    # 事件调度（des.Simulator 承载，替换原先的私有 heapq 主循环）
    # ------------------------------------------------------------------ #
    def _push(self, t: float, kind: int, payload: tuple) -> None:
        """调度一条协议事件到内核。priority 恒为 0：同刻事件按插入序（FIFO）
        执行，与旧实现 ``(t, tie, kind)`` 的定序完全一致。"""
        self.sim.schedule_at(float(t), self._dispatch, kind, payload,
                             name=_EV_KIND_NAME.get(kind, "tcp"))
    def _phase(self) -> str:
        if self.in_fast_recovery:
            return "rec"
        return "ss" if self.cwnd < self.ssthresh else "ca"

    def _track_phase(self, t: float) -> None:
        ph = self._phase()
        self._phase_time[ph] = self._phase_time.get(ph, 0.0) + max(t - self._last_phase_t, 0.0)
        self._last_phase_t = t

    # ------------------------------------------------------------------ #
    # 发送
    # ------------------------------------------------------------------ #
    def _effective_window(self) -> float:
        return max(min(self.cwnd, self.rwnd), float(self.mss))

    def _send_segment(self, seq: int, size: int, t: float) -> None:
        owd, bw, loss, _hops, q_limit = self.p.at(t)
        bw = max(bw, 1e-3)
        tx = size * 8.0 / (bw * 1e6)
        t_start = max(t, self.link_free)
        t_end = t_start + tx
        # 链路排队：忙期剩余时间换算成在途字节；超过缓冲即尾丢弃（拥塞信号）
        queued = max(0.0, self.link_free - t) * bw * 1e6 / 8.0
        drop = (bool(self.rng.random() < loss * self.loss_scale)
                or (queued + size > q_limit))
        self.link_free = t_end
        if drop:
            self.pkts_lost += 1
            return
        self.seg_state[seq][1] = t           # 记录发送时刻
        self.sent_count += 1
        self._push(t_end + owd, _EV_SEG_ARRIVE, (seq, size, t))

    def _flush_send(self, total_bytes: int) -> None:
        while True:
            flight = self.snd_nxt - self.snd_una
            if flight >= self._effective_window():
                return
            if self.snd_nxt >= total_bytes:
                return
            size = min(self.mss, total_bytes - self.snd_nxt)
            seq = self.snd_nxt
            st = self.seg_state.get(seq)
            if st is None:
                self.seg_state[seq] = [size, self.t, 0, False]
            else:
                # 窗口回退后重发：按重传计数（Karn 算法要求不采样其 RTT）
                st[2] += 1
                self.retransmits += 1
            self._send_segment(seq, size, self.t)
            self.snd_nxt += size

    # ------------------------------------------------------------------ #
    # 接收端
    # ------------------------------------------------------------------ #
    def _on_segment(self, t: float, seq: int, size: int, sent_t: float) -> None:
        if seq == self.rcv_expect:
            self.rcv_expect += size
            advanced = True
            while advanced:
                advanced = False
                for s in list(self.rcv_ooo):
                    if s == self.rcv_expect:
                        self.rcv_expect += self.rcv_ooo.pop(s)
                        advanced = True
            self._schedule_ack(t)
        elif seq > self.rcv_expect:
            if seq not in self.rcv_ooo:
                self.rcv_ooo[seq] = size
                self.reorder_events += 1
            self._schedule_ack(t, force=True)      # 乱序 → 立即 ACK（含 SACK）
        else:
            self._schedule_ack(t, force=True)

    def _sack_blocks(self) -> List[Tuple[int, int]]:
        if not self.cfg.sack or not self.rcv_ooo:
            return []
        starts = sorted(self.rcv_ooo)
        blocks: List[Tuple[int, int]] = []
        s0 = prev = starts[0]
        for s in starts[1:]:
            if s == prev + self.rcv_ooo[prev]:
                prev = s
            else:
                blocks.append((s0, prev + self.rcv_ooo[prev]))
                s0 = prev = s
        blocks.append((s0, prev + self.rcv_ooo[prev]))
        return blocks

    def _schedule_ack(self, t: float, force: bool = False) -> None:
        self._ack_pending += 1
        if not force and self.cfg.delayed_ack and self._ack_pending < 2:
            return
        self._ack_pending = 0
        owd, _bw, _l, _h, _q = self.p.at(t)
        sack = self._sack_blocks()
        # 接收窗口：缓冲被乱序数据占用后收窄（流控）
        used = sum(self.rcv_ooo.values())
        rwnd = max(int(self.cfg.rcvbuf_bytes) - used, self.mss)
        self._push(t + owd, _EV_ACK_ARRIVE,
                   (self.rcv_expect, tuple(sack), rwnd, t))

    # ------------------------------------------------------------------ #
    # 发送端：ACK 处理
    # ------------------------------------------------------------------ #
    def _on_ack(self, t: float, ack: int, sack: Tuple, rwnd: int, ack_sent_t: float) -> None:
        self.rwnd = float(max(rwnd, self.mss))
        advanced = ack > self.snd_una
        if advanced:
            newly = ack - self.snd_una
            # 释放被确认的段，并做 RTT 采样（Karn：重传过的段不采样）
            s = self.snd_una
            while s < ack:
                st = self.seg_state.get(s)
                if st is None:
                    break
                size, sent_t, retx, _acked = st
                if retx == 0:
                    self._update_rtt(t - sent_t)
                st[3] = True
                s += size
            self.snd_una = ack
            self.dupacks = 0
            self.rto_backoff = 0
            if self.in_fast_recovery:
                self._recovery_retx.discard(ack)
                if ack >= self.recover:
                    self.in_fast_recovery = False
                    self.cwnd = max(self.ssthresh, 2 * self.mss)
                else:
                    # 部分 ACK（NewReno）：继续修复下一个缺口
                    self.cwnd = max(self.ssthresh + 3 * self.mss, 2 * self.mss)
                    self._retransmit_gap(t, ack, sack)
            self._grow_window(newly)
        else:
            # 重复 ACK：计数并考虑快速重传
            self.dupacks += 1
            new_sack = self.cfg.sack and bool(sack)
            if not self.in_fast_recovery and (self.dupacks == 3 or (new_sack and self.dupacks >= 2)):
                self._enter_recovery(t, sack)
            elif self.in_fast_recovery:
                self.cwnd += self.mss                       # 快速恢复窗口膨胀
                if new_sack:                                # 仅在 SACK 给出新缺口时补洞
                    self._retransmit_gap(t, ack, sack)
        # 重排 RTO 定时器（旧的会因 snd_una 变化而作废）
        self._arm_rto(t)

    def _update_rtt(self, r: float) -> None:
        if r <= 0 or not np.isfinite(r):
            return
        self.rtt_samples.append(r * 1000.0)
        if self.srtt is None:
            self.srtt = r
            self.rttvar = r / 2.0
        else:
            self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - r)
            self.srtt = 0.875 * self.srtt + 0.125 * r
        self.rto = float(np.clip(self.srtt + 4.0 * self.rttvar,
                                 self.cfg.min_rto_s, self.cfg.max_rto_s))
        # BBR：RTprop 最小值（10 s 窗口）
        if r < self._bbr_rtprop:
            self._bbr_rtprop = r
        self._bbr_rtprop_stamp = self.t

    def _arm_rto(self, t: float) -> None:
        if self.snd_nxt > self.snd_una:
            self._rto_seq += 1
            self._push(t + self.rto * (2 ** self.rto_backoff), _EV_RTO,
                       (self.snd_una, self._rto_seq))

    # ------------------------------------------------------------------ #
    # 拥塞控制
    # ------------------------------------------------------------------ #
    def _grow_window(self, acked_bytes: int) -> None:
        acked_bytes = max(int(acked_bytes), 0)
        if acked_bytes == 0:
            return
        a = self.algo
        if a == "hybla":
            rho = min(max((self.srtt or 0.025) * 1000.0 / max(self.cfg.rtt_ref_ms, 1e-6),
                          1e-3), 64.0)
            if self.cwnd < self.ssthresh:
                self.cwnd += (2.0 ** rho) * acked_bytes * 0.25
            else:
                self.cwnd += rho * rho * self.mss * acked_bytes / max(self.cwnd, 1.0)
        elif a == "reno":
            if self.cwnd < self.ssthresh:
                self.cwnd += acked_bytes
            else:
                self.cwnd += self.mss * acked_bytes / max(self.cwnd, 1.0)
        elif a == "vegas":
            rtt_min = min(self.rtt_samples[-8:]) / 1000.0 if self.rtt_samples else 0.05
            expected = self.cwnd / max(self.srtt or rtt_min, 1e-6) * acked_bytes
            actual = acked_bytes / max(rtt_min, 1e-6)
            diff = (expected - actual) * max(self.srtt or rtt_min, 1e-6) / max(self.mss, 1)
            if diff < 1:
                self.cwnd += self.mss
            elif diff > 3:
                self.cwnd = max(self.cwnd * 0.875, 2 * self.mss)
        elif a == "bbr":
            self._bbr_grow(acked_bytes)
        else:  # cubic
            if self.cwnd < self.ssthresh:
                self.cwnd += acked_bytes * 0.6      # 慢启动（受限版，抑制过冲）
            else:
                srtt = max(self.srtt or 0.05, 1e-3)
                wmax_pkt = self._w_max / max(self.mss, 1)
                if self._k <= 0.0:
                    self._k = (((wmax_pkt * (1.0 - 0.7) / 0.4) ** (1.0 / 3.0))
                               if wmax_pkt > 0 else 0.0)
                dt_rtt = (self.t - self._t_epoch) / srtt
                target_pkt = 0.4 * max(dt_rtt - self._k, 0.0) ** 3 + wmax_pkt
                target = target_pkt * self.mss
                if target > self.cwnd:
                    self.cwnd += (target - self.cwnd) * acked_bytes / max(self.cwnd, 1.0)
                else:
                    self.cwnd += self.mss * acked_bytes / max(self.cwnd, 1.0)
        # 窗口硬上限：真实 TCP 不会把窗口撑到管道容量的数倍（接收窗口 + 管道余量）
        self.cwnd = min(max(self.cwnd, float(self.mss)), self._cwnd_cap)

    def _bbr_grow(self, acked_bytes: int) -> None:
        """简化 BBR：以 BtlBW×RTprop 估算管道，窗口向 2×BDP 收敛（不靠丢包探顶）。"""
        rtprop = self._bbr_rtprop if self._bbr_rtprop < INF else max(self.srtt or 0.05, 1e-3)
        if self.srtt and self.srtt > 0:
            self._bbr_btlbw = max(self._bbr_btlbw, acked_bytes / self.srtt)
        bdp = max(self._bbr_btlbw, 1.0) * rtprop
        target = 2.0 * bdp
        if self.cwnd < target:
            self.cwnd += acked_bytes                 # 上升段：接近倍增
        else:
            self.cwnd = max(self.cwnd * 0.97, target)  # 收敛段：不制造常驻队列
        self._bbr_round += 1
        if self._bbr_round % 200 == 0:               # 周期性 ProbeBW 抖动
            self._bbr_btlbw *= 1.02

    # ------------------------------------------------------------------ #
    # 丢包恢复
    # ------------------------------------------------------------------ #
    def _enter_recovery(self, t: float, sack: Tuple) -> None:
        flight = self.snd_nxt - self.snd_una
        self.ssthresh = max(flight / 2.0, 2 * self.mss)
        if self.algo == "cubic":
            self._w_max = self.cwnd
            self._t_epoch = t
            self._k = 0.0
            self.cwnd = max(self.cwnd * 0.7, 2 * self.mss)
        elif self.algo == "bbr":
            # BBR 是模型型：不把丢包当作拥塞信号（这正是它在有损高 BDP 链路上的优势）
            self._bbr_btlbw *= 0.98
            self.cwnd = max(self.cwnd, 2 * self.mss)
            self._bbr_phase = "probe_bw"
        else:
            self.cwnd = self.ssthresh
        self.fast_retx += 1
        self.in_fast_recovery = True
        self.recover = self.snd_nxt
        self._recovery_retx.clear()
        self._retransmit_gap(t, self.snd_una, sack)

    def _retransmit_gap(self, t: float, ack: int, sack: Tuple) -> None:
        """重传最早未确认的段；同一次恢复期内同一段只重传一次（防重传放大）。"""
        if ack in self._recovery_retx:
            return
        st = self.seg_state.get(ack)
        if st is None:
            return
        self._recovery_retx.add(ack)
        self.seg_state[ack][2] += 1
        self.retransmits += 1
        self._send_segment(ack, st[0], t)

    def _on_rto(self, t: float, expect_una: int) -> None:
        if expect_una != self.snd_una or self.snd_nxt <= self.snd_una:
            return                        # ACK 已推进，旧定时器作废
        self.timeouts += 1
        self.in_fast_recovery = False
        flight = self.snd_nxt - self.snd_una
        self.ssthresh = max(flight / 2.0, 2 * self.mss)
        self.cwnd = float(self.mss)
        if self.algo == "cubic":
            self._w_max = self.ssthresh
            self._t_epoch = t
            self._k = 0.0
        if self.algo == "bbr":
            self._bbr_btlbw *= 0.7
            self._bbr_phase = "startup"
        self.rto_backoff = min(self.rto_backoff + 1, 6)
        self._recovery_retx.clear()
        # 发送指针回退到最早未确认段：按 cwnd = 1 MSS 重新推送窗口
        # （真实 TCP 超时后正是这么做的；不回退会造成"窗口已发完却收不到 ACK"的死锁）
        self.snd_nxt = self.snd_una
        st = self.seg_state.get(self.snd_una)
        if st is not None:
            st[2] += 1
            self.retransmits += 1
            self._send_segment(self.snd_una, st[0], t)
            self.snd_nxt = self.snd_una + st[0]
        self._arm_rto(t)

    # ------------------------------------------------------------------ #
    # 采样
    # ------------------------------------------------------------------ #
    def _sample(self, t: float) -> None:
        if t < self._next_sample:
            return
        flight = self.snd_nxt - self.snd_una
        self.inflight_bytes_accum += flight
        self.inflight_samples += 1
        self.samples.append(TcpSample(
            t=float(t), cwnd=float(self.cwnd), ssthresh=float(self.ssthresh),
            inflight=int(flight), delivered=int(self.snd_una),
            rtt_ms=float((self.srtt or 0.0) * 1000.0),
            rto_ms=float(self.rto * 1000.0), phase=self._phase()))
        self._next_sample = t + max(self.cfg.sample_every_s, 1e-3)

    # ------------------------------------------------------------------ #
    # 事件分发（由 des.Simulator 主循环回调）
    # ------------------------------------------------------------------ #
    def _start_flow(self) -> None:
        """流起点事件：完成三次握手后的首个动作——灌满发送窗口 + 装 RTO 定时器。"""
        self._track_phase(self.t)
        self._flush_send(self._total_bytes)
        self._arm_rto(self.t)

    def _dispatch(self, kind: int, payload: tuple) -> None:
        if self._done:
            return
        total_bytes = self._total_bytes
        t = self.sim.now()
        if kind == _EV_RTO:
            expect_una, vseq = payload
            if vseq != self._rto_seq:
                return                 # 已被更新的 RTO 定时器取代（惰性取消）
            self._track_phase(t)
            self.t = t
            self._sample(t)
            self._on_rto(t, expect_una)
            self._flush_send(total_bytes)
        else:
            self._track_phase(t)
            self.t = t
            self._sample(t)
            if kind == _EV_SEG_ARRIVE:
                seq, size, sent_t = payload
                self._on_segment(t, seq, size, sent_t)
            else:  # _EV_ACK_ARRIVE
                ack, sack, rwnd, ack_sent_t = payload
                self._on_ack(t, ack, sack, rwnd, ack_sent_t)
            self._flush_send(total_bytes)
        if self.progress and self.samples and len(self.samples) % 200 == 0:
            self.progress(min(self.snd_una / total_bytes, 1.0))
        if self.snd_una >= total_bytes:
            self._done = True
            if self._owns_sim:
                self.sim.stop()        # 独立运行：立即终止主循环（与旧版 break 等价）
            if self._on_done is not None:
                cb = self._on_done
                self._on_done = None
                cb(self)               # 共享内核：通知编排层流已完成

    # ------------------------------------------------------------------ #
    # 流生命周期：arm（装载）→ run（独立阻塞）/ start（共享内核非阻塞）
    # ------------------------------------------------------------------ #
    def _arm(self, total_bytes: int, handshake_s: float) -> float:
        """装载流状态并在内核上调度起点事件，返回本流的时限 t_end。

        连接时钟对齐内核：共享内核上流在 t>0 才到达，此后 ``self.t`` 始终
        跟随 ``sim.now()``——否则 RTT 采样（t − sent_t）会混入到达前的
        空窗时间，SRTT/RTO 全部失真。
        """
        handshake = max(float(handshake_s), 0.0)
        base = float(self.sim.now()) if self.sim is not None else 0.0
        self._done = False
        self.t = base + handshake
        self._next_sample = self.t
        owd0, bw0, _l0, _h0, q0 = self.p.at(self.t)
        self._bbr_rtprop = max(owd0 * 2, 1e-3)
        self._bbr_btlbw = bw0 * 1e6 / 8.0
        self.queue_bytes = 0.0
        t_end = max(self.p.duration_s, self.t) + self.cfg.max_time_s
        t_end = min(t_end, self.t + self.cfg.max_time_s)
        self._last_phase_t = self.t
        self._t0 = base           # 流到达时刻（完成时长 = 结束 − 到达，含握手，与独立运行口径一致）
        self.sim.schedule_at(self.t, self._start_flow, name="tcp:start")
        return t_end

    def _finish(self, total_bytes: int) -> TcpFlowResult:
        completion = (self.t - getattr(self, "_t0", 0.0)
                      if self.snd_una >= total_bytes else float("nan"))
        delivered = int(min(self.snd_una, total_bytes))
        samples = np.asarray(self.rtt_samples, dtype=float)
        inflight_mean = (self.inflight_bytes_accum / self.inflight_samples
                         if self.inflight_samples else 0.0)
        bdp_samples = self.p.bw_mbps * 1e6 / 8.0 * self.p.rtt_ms / 1000.0
        res = TcpFlowResult(
            protocol=self.algo,
            label=ALGO_LABELS.get(self.algo, self.algo.upper()),
            bytes_total=total_bytes,
            bytes_delivered=delivered,
            mss=self.mss,
            completion_s=completion,
            goodput_mbps=(total_bytes * 8.0 / 1e6 / completion
                          if np.isfinite(completion) and completion > 0 else 0.0),
            utilization=0.0,
            segments_sent=self.sent_count,
            retransmits=self.retransmits,
            fast_retransmits=self.fast_retx,
            timeouts=self.timeouts,
            dupacks=0,
            pkts_lost=self.pkts_lost,
            reordering_events=self.reorder_events,
            rtt_min_ms=float(samples.min()) if samples.size else float("nan"),
            rtt_mean_ms=float(samples.mean()) if samples.size else float("nan"),
            rtt_p95_ms=float(np.percentile(samples, 95)) if samples.size else float("nan"),
            rttvar_ms=float((self.rttvar or 0.0) * 1000.0),
            final_rto_ms=float(self.rto * 1000.0),
            cwnd_max_bytes=float(max((s.cwnd for s in self.samples), default=0.0)),
            ssthresh_final=float(self.ssthresh),
            mean_inflight_bytes=float(inflight_mean),
            mean_bdp_bytes=float(np.mean(bdp_samples)) if bdp_samples.size else 0.0,
            window_limited_mbps=float(self.window_limited_mbps),
            rcvbuf_bytes=int(self.cfg.rcvbuf_bytes),
            samples=self.samples,
            phase_time=dict(self._phase_time),
            path_note=self.p.describe(),
        )
        # 可达效率：完成时间内交付字节 ÷ min(链路瓶颈, 窗口上限) —— LEO 高 BDP 下
        # 后者常常先饱和，只报"链路利用率"会把窗口瓶颈误读成链路没喂满
        if np.isfinite(completion) and completion > 0:
            cap_mbps = min(float(np.mean(self.p.bw_mbps)),
                           self.window_limited_mbps or INF)
            capacity_bytes = cap_mbps * 1e6 / 8.0 * completion
            res.utilization = float(min(total_bytes / max(capacity_bytes, 1.0), 1.0))
        return res

    def run(self, total_bytes: int, handshake_s: float = 0.0) -> TcpFlowResult:
        """独立运行整段流（阻塞）：无注入内核时自建一个并跑完。"""
        total_bytes = int(max(total_bytes, 1))
        self._total_bytes = total_bytes
        self._owns_sim = self.sim is None
        if self._owns_sim:
            self.sim = Simulator()
        t_end = self._arm(total_bytes, handshake_s)
        self.sim.run(until=t_end)
        return self._finish(total_bytes)

    def start(self, total_bytes: int, handshake_s: float = 0.0,
              on_done: Optional[Callable[["TcpConnection"], None]] = None) -> None:
        """在**已注入的共享内核**上非阻塞启动（全局离散事件编排用）。

        与 :meth:`run` 的区别：不驱动主循环、完成时不 stop 内核，只把流的
        事件挂到共享时间轴上；完成时回调 ``on_done(self)``（仍在内核事件
        上下文内执行）。须在构造时传 ``sim=``。
        """
        if self.sim is None:
            raise RuntimeError("start() 需要共享内核：构造时传入 sim=Simulator(...)")
        total_bytes = int(max(total_bytes, 1))
        self._total_bytes = total_bytes
        self._owns_sim = False
        self._on_done = on_done
        self._arm(total_bytes, handshake_s)

    def kill(self) -> None:
        """终止本流（全局编排的看门狗用）：不再响应任何事件。"""
        self._done = True

    def on_path_switch(self, t: Optional[float] = None) -> bool:
        """路径切换事件（离散事件编排注入）：真实 TCP 的切换代价。

        LEO 切换瞬间旧路径的在途包作废，真实 TCP 的表现是：
        1. 发射静默 ``path_switch_cost_s`` 秒（星上重寻址 / 新路径建立）；
        2. 拥塞窗口收缩回初始窗、退出快速恢复（等价于一次 RTO 的后果，
           但不制造重传风暴——在途段本就没到达，无需逐段补发）；
        3. 旧 RTO 定时器作废，由后续 ACK / 重发事件重新武装。

        返回是否生效（流已完成或未开启惩罚时返回 False）。
        """
        if self._done:
            return False
        t = float(self.sim.now() if t is None else t)
        # 发射静默：切换中断期（链路重建）不发送任何新段
        self.link_free = max(self.link_free, t + max(self.cfg.path_switch_cost_s, 0.0))
        if not self.cfg.path_switch_penalty:
            return False
        self._track_phase(t)
        self.t = t
        self._sample(t)
        flight = self.snd_nxt - self.snd_una
        self.ssthresh = max(flight / 2.0, 2 * self.mss)
        self.cwnd = float(self.cfg.initial_window_seg * self.mss)
        if self.algo == "cubic":
            self._w_max = max(self.cwnd, self.mss)   # CUBIC 从新窗口重新起涨
            self._t_epoch = t
            self._k = 0.0
        elif self.algo == "bbr":
            self._bbr_btlbw = max(self._bbr_btlbw * 0.5, 1.0)   # 保守重估瓶颈
        if self.in_fast_recovery:
            self.in_fast_recovery = False
            self._recovery_retx.clear()
        self.dupacks = 0
        self._rto_seq += 1          # 作废旧 RTO 定时器（惰性取消）
        self._arm_rto(t)
        return True

    def handshake_seconds(self, handshake_rtts: float = 2.0) -> float:
        """握手占用时长（TCP 1 RTT + TLS1.3 1 RTT = 2 RTT）。"""
        rtt0 = (self.p.rtt_ms[0] / 1000.0
                if self.p.rtt_ms.size else 0.05)
        return max(float(handshake_rtts), 0.0) * max(rtt0, 1e-3)


# --------------------------------------------------------------------------- #
# 4. 便捷入口
# --------------------------------------------------------------------------- #
def default_tcp_config(transport_cfg=None, **over) -> TcpConfig:
    """由 TransportConfig 生成本模块的配置（两套参数保持语义一致）。"""
    cfg = TcpConfig()
    if transport_cfg is not None:
        cfg.mss = int(getattr(transport_cfg, "mss_bytes", cfg.mss))
        cfg.initial_window_seg = int(getattr(transport_cfg, "initial_window_seg",
                                             cfg.initial_window_seg))
        cfg.rcvbuf_bytes = int(getattr(transport_cfg, "rcvbuf_bytes", cfg.rcvbuf_bytes))
        cfg.sack = bool(getattr(transport_cfg, "sack", cfg.sack))
        cfg.delayed_ack = bool(getattr(transport_cfg, "delayed_ack", cfg.delayed_ack))
        cfg.rtt_ref_ms = float(getattr(transport_cfg, "rtt_ref_ms", cfg.rtt_ref_ms))
        # 路径切换代价：与 TransportConfig 的同名语义保持一致
        cfg.path_switch_penalty = bool(getattr(transport_cfg, "path_switch_penalty",
                                               cfg.path_switch_penalty))
    for k, v in over.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def simulate_flow(profile: PathProfile, payload_mbit: float, algo: str = "cubic",
                  cfg: Optional[TcpConfig] = None, seed: int = 0,
                  handshake_rtts: float = 2.0,
                  progress: Optional[Callable[[float], None]] = None) -> TcpFlowResult:
    """在给定路径剖面上跑一条真实 TCP 连接。

    handshake_rtts：握手占用的 RTT 数（TCP 三次握手 1 RTT + TLS1.3 1 RTT = 2）。
    """
    cfg = cfg or TcpConfig()
    rtt0 = max(profile.rtt_ms[0] / 1000.0, 1e-3) if profile.rtt_ms.size else 0.05
    conn = TcpConnection(profile, cfg, algo=algo, seed=seed, progress=progress)
    return conn.run(int(payload_mbit * 1e6 / 8.0), handshake_s=handshake_rtts * rtt0)


def compare_flows(profile: PathProfile, payload_mbit: float,
                  algos: Sequence[str] = ("reno", "cubic", "bbr", "hybla"),
                  cfg: Optional[TcpConfig] = None, seed: int = 7,
                  progress: Optional[Callable[[str], None]] = None) -> List[TcpFlowResult]:
    """多算法在同一时变路径上的真实包级对比。"""
    out: List[TcpFlowResult] = []
    for i, a in enumerate(algos):
        if progress:
            progress(f"真实 TCP：{ALGO_LABELS.get(a, a)} …")
        out.append(simulate_flow(profile, payload_mbit, algo=a, cfg=cfg,
                                 seed=seed + 13 * i))
    return out


def path_capacity_summary(profile: PathProfile, mss: int = 1460) -> Dict[str, float]:
    """路径容量画像：BDP、窗口需求、时变强度。"""
    bdp = profile.bw_mbps * 1e6 / 8.0 * profile.rtt_ms / 1000.0
    return {
        "mean_bdp_bytes": float(np.mean(bdp)) if bdp.size else 0.0,
        "max_bdp_bytes": float(np.max(bdp)) if bdp.size else 0.0,
        "mean_bdp_segments": float(np.mean(bdp) / max(mss, 1)) if bdp.size else 0.0,
        "rtt_swing_ms": profile.rtt_swing_ms,
        "path_changes": float(profile.path_changes),
        "mean_bw_mbps": float(np.mean(profile.bw_mbps)) if profile.bw_mbps.size else 0.0,
        "mean_hops": float(np.mean(profile.hops)) if profile.hops.size else 0.0,
    }


__all__ = [
    "PathProfile", "profile_from_route", "TcpConfig", "TcpConnection",
    "TcpFlowResult", "TcpSample", "simulate_flow", "compare_flows",
    "default_tcp_config", "path_capacity_summary", "ALGO_LABELS",
]

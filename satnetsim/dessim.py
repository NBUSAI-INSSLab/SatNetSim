"""全局离散事件编排——把七层流水线放进 NS-3 式统一事件时间轴。

为什么需要它
------------
步进式引擎（``engine.run``）按快照推进：每个快照上"轨道 → 拓扑 → 路由 →
各层分析"依次执行。它的结论都是**层内统计**（平均时延、吞吐、完成率），
回答不了**事件级**的问题：

1. 一次卫星切换与一个计算任务到达同时发生时，谁先谁后？
2. 一条 TCP 流从到达、握手、慢启动到完成，与拓扑刷新、路径切换如何交错？
3. 任务到达-排队-服务这条队列链，在事件粒度上的等待时间分布是什么？

本模块用 :class:`satnetsim.des.Simulator` 把以下六类事件放进同一条时间轴：

* ``topo:refresh`` / ``route:recompute`` —— 每个快照时刻（对应引擎逐快照流水线）；
* ``path:switch`` —— 代表路径的 RTT/跳数阶跃时刻（由 PathProfile 推导）；
* ``flow:arrive`` / ``flow:done`` —— 真实包级 TCP 流（``tcpstack.TcpConnection``
  以非阻塞 ``start()`` 模式挂到共享内核上，事件级并发，而非逐流串行）；
* ``task:arrive`` / ``task:done`` —— 计算任务泊松到达 + 星载节点 FIFO 排队
  （ join-shortest-queue 分发，服务速率取 ComputeConfig 的星载 TOPS）；
* ``metric:sample`` —— 周期采样（队列深度时间序列，供 GUI 画曲线）。

建模口径（诚实声明）
--------------------
* TCP 各流**各自**对着 PathProfile 取样，流间共享瓶颈的相互挤压不在本层
  （那属于 ``realdatastream`` 的逐包多跳转发）；本层回答的是"事件交错"。
* 路由重算事件只做记账（引擎已逐快照预计算），不重复跑 Dijkstra。
* 计算排队是事件驱动的 M/G/1-approximation（FIFO + JSQ），与
  ``computing.py`` 的批量放置（到达序 + 价值密度）是**两种口径**，可对照。

参考：ns-3 Model Library "Simulator"；离散事件仿真标准教材（Banks et al.,
*Discrete-Event System Simulation*）的事件表 / 忙期 / 队列长度统计口径。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import ComputeConfig, DesConfig
from .tcpstack import ALGO_LABELS, PathProfile, TcpConfig, TcpConnection
from .des import Simulator, Trace

INF = float("inf")


# --------------------------------------------------------------------------- #
# 1. 记录与报告容器
# --------------------------------------------------------------------------- #
@dataclass
class DesTaskRecord:
    """一个计算任务在事件时间轴上的完整生命周期。"""

    tid: int
    arrival_s: float
    start_s: float
    finish_s: float
    node: str
    gflops: float
    deadline_s: float

    @property
    def wait_s(self) -> float:
        return self.start_s - self.arrival_s

    @property
    def sojourn_s(self) -> float:
        return self.finish_s - self.arrival_s

    @property
    def service_s(self) -> float:
        return self.finish_s - self.start_s

    @property
    def met(self) -> bool:
        return self.sojourn_s <= self.deadline_s


@dataclass
class DesFlowRecord:
    """一条 TCP 流在共享时间轴上的到达与完成。"""

    idx: int
    arrival_s: float
    algo: str
    done: bool = False
    completion_s: float = float("nan")
    goodput_mbps: float = 0.0
    retransmits: int = 0


@dataclass
class DesReport:
    """全局离散事件编排的结果。"""

    duration_s: float
    n_executed: int = 0
    n_cancelled: int = 0
    scheduler: str = "HeapScheduler"
    sim_end_s: float = 0.0

    trace: Trace = field(default_factory=Trace)
    flows: List[DesFlowRecord] = field(default_factory=list)
    tasks: List[DesTaskRecord] = field(default_factory=list)
    node_names: List[str] = field(default_factory=list)
    queue_series: List[Tuple[float, int]] = field(default_factory=list)

    topo_events: int = 0
    route_events: int = 0
    path_switches: int = 0
    flow_switches: int = 0      # 波及在途 TCP 流的切换次数（切换代价已生效）
    notes: List[str] = field(default_factory=list)

    # ---------------- 统计 ---------------- #
    @property
    def flows_done(self) -> int:
        return sum(1 for f in self.flows if f.done)

    @property
    def flow_goodput_mean_mbps(self) -> float:
        ok = [f for f in self.flows if f.done and f.completion_s > 0]
        if not ok:
            return float("nan")
        return float(np.mean([f.goodput_mbps for f in ok]))

    @property
    def task_mean_wait_s(self) -> float:
        return float(np.mean([t.wait_s for t in self.tasks])) if self.tasks else float("nan")

    @property
    def task_p95_wait_s(self) -> float:
        return (float(np.percentile([t.wait_s for t in self.tasks], 95))
                if self.tasks else float("nan"))

    @property
    def task_mean_sojourn_s(self) -> float:
        return float(np.mean([t.sojourn_s for t in self.tasks])) if self.tasks else float("nan")

    @property
    def task_deadline_rate(self) -> float:
        return float(np.mean([t.met for t in self.tasks])) if self.tasks else float("nan")

    @property
    def node_utilization(self) -> Dict[str, float]:
        """每节点服务时间占比（busy 时间 ÷ 仿真时长，事件粒度累计）。"""
        out: Dict[str, float] = {}
        for name in self.node_names:
            busy = sum(t.service_s for t in self.tasks if t.node == name)
            out[name] = busy / max(self.duration_s, 1e-9)
        return out

    # ---------------- 展示 ---------------- #
    def headline(self) -> List[Tuple[str, str]]:
        rows = [
            ("事件执行数", f"{self.n_executed}（取消 {self.n_cancelled}，"
                          f"调度器 {self.scheduler}）"),
            ("时间轴跨度", f"{self.sim_end_s:.1f} s（配置 {self.duration_s:.0f} s）"),
            ("拓扑/路由事件", f"{self.topo_events} 次刷新 / {self.route_events} 次重算"),
            ("路径切换", f"{self.path_switches} 次（事件时刻触发）"
                        + (f"，波及在途流 {self.flow_switches} 次" if self.flows else "")),
        ]
        if self.flows:
            ok = [f for f in self.flows if f.done]
            rows.append(("TCP 流", f"{len(ok)}/{len(self.flows)} 条完成，"
                                   f"均值吞吐 {self.flow_goodput_mean_mbps:.1f} Mbps"))
        if self.tasks:
            rows.append(("计算任务", f"{len(self.tasks)} 个到达，"
                                     f"均值等待 {self.task_mean_wait_s:.3f} s"
                                     f"（P95 {self.task_p95_wait_s:.3f} s），"
                                     f"时限达标 {self.task_deadline_rate*100:.0f}%"))
            util = [u for u in self.node_utilization.values()]
            if util:
                rows.append(("节点利用率", f"均值 {float(np.mean(util))*100:.1f}% / "
                                          f"峰值 {float(np.max(util))*100:.1f}%"
                                          f"（{len(util)} 节点，服务时间占比）"))
        return rows

    def summary_rows(self) -> List[Tuple[str, str]]:
        return self.headline()

    def event_kind_table(self) -> List[Tuple[str, int]]:
        return sorted(self.trace.counts.items(), key=lambda kv: -kv[1])

    def to_dict(self, max_trace: int = 2000) -> Dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "sim_end_s": self.sim_end_s,
            "n_executed": self.n_executed,
            "n_cancelled": self.n_cancelled,
            "scheduler": self.scheduler,
            "event_kinds": dict(self.trace.counts),
            "topo_events": self.topo_events,
            "route_events": self.route_events,
            "path_switches": self.path_switches,
            "flow_switches": self.flow_switches,
            "flows": [f.__dict__ | {"algo": f.algo} for f in self.flows],
            "tasks": {
                "n": len(self.tasks),
                "mean_wait_s": self.task_mean_wait_s,
                "p95_wait_s": self.task_p95_wait_s,
                "mean_sojourn_s": self.task_mean_sojourn_s,
                "deadline_rate": self.task_deadline_rate,
                "node_utilization": self.node_utilization,
            },
            "queue_series": self.queue_series,
            "trace": {
                "counts": dict(self.trace.counts),
                "total": int(sum(self.trace.counts.values())),
                "kept": len(self.trace.records),
                "records": [r.as_tuple() for r in self.trace.records[:max_trace]],
            },
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# 2. 星载节点（事件驱动 FIFO 队列）
# --------------------------------------------------------------------------- #
class _DesNode:
    """事件驱动的单服务器队列（FIFO）。

    与 ``computing.py`` 的区别：那里是"批量任务一次放置"（busy_until 推进）；
    这里每个到达/离开都是内核事件，等待时间由队列真实演化给出。
    """

    __slots__ = ("name", "tops", "queue", "busy", "n_done", "busy_accum")

    def __init__(self, name: str, tops: float) -> None:
        self.name = name
        self.tops = float(tops)
        self.queue: deque = deque()
        self.busy = False
        self.n_done = 0
        self.busy_accum = 0.0

    @property
    def depth(self) -> int:
        return len(self.queue) + (1 if self.busy else 0)


# --------------------------------------------------------------------------- #
# 3. 主编排
# --------------------------------------------------------------------------- #
def run_des(*, duration_s: float, snapshot_interval_s: float,
            tcp_profile: Optional[PathProfile] = None,
            compute_cfg: Optional[ComputeConfig] = None,
            des_cfg: Optional[DesConfig] = None,
            tcp_cfg: Optional[TcpConfig] = None,
            progress: Optional[Callable[[str], None]] = None) -> DesReport:
    """在一条全局事件时间轴上重放/编排整个星座仿真场景。

    参数
    ----
    duration_s / snapshot_interval_s：仿真时长与快照间隔（与 SimConfig 一致）。
    tcp_profile：时变路径剖面（引擎第 ⑤ 步产物）；缺失则跳过 TCP 流事件。
    compute_cfg：算力配置（星载 TOPS、任务规模分布、时限）。
    des_cfg：编排自身的开关与规模。
    tcp_cfg：TCP 流参数（引擎应传 ``default_tcp_config(cfg.transport)``
      以保持与第 ⑤ 步一致；缺省用 TcpConfig 出厂值）。
    """
    des_cfg = des_cfg or DesConfig()
    compute_cfg = compute_cfg or ComputeConfig()
    duration_s = max(float(duration_s), 1.0)
    snap_dt = max(float(snapshot_interval_s), 1e-9)

    sim = Simulator(seed=des_cfg.seed, trace_limit=8000 if des_cfg.trace else 1)
    rep = DesReport(duration_s=duration_s, scheduler=type(sim.scheduler).__name__,
                    trace=sim.trace)
    tr = sim.trace
    active_conns: List[TcpConnection] = []      # 在途 TCP 流（路径切换事件波及它们）

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    # ---------------- ① 起点事件 ---------------- #
    def on_start() -> None:
        tr.add(sim.now(), "sim:start", "core", f"时长 {duration_s:.0f}s")

    sim.schedule_at(0.0, on_start, name="des:start")

    # ---------------- ② 拓扑刷新 / 路由重算（每快照） ---------------- #
    n_snaps = int(np.floor(duration_s / snap_dt)) + 1

    def on_topo(k: int) -> None:
        rep.topo_events += 1
        tr.add(sim.now(), "topo:refresh", "orbit", f"快照 #{k}")

    def on_route(k: int) -> None:
        rep.route_events += 1
        tr.add(sim.now(), "route:recompute", "routing", f"快照 #{k}")

    for k in range(n_snaps):
        t = k * snap_dt
        sim.schedule_at(t, on_topo, k, name="topo:refresh", priority=5)
        sim.schedule_at(t + 1e-6, on_route, k, name="route:recompute", priority=4)

    # ---------------- ③ 路径切换事件（由剖面阶跃推导） ---------------- #
    if tcp_profile is not None and tcp_profile.t.size >= 2:
        d_owd = np.abs(np.diff(tcp_profile.owd_ms))
        d_hop = np.abs(np.diff(tcp_profile.hops))
        idxs = np.where((d_owd > 0.2) | (d_hop > 0.4))[0]

        def on_switch(i: int) -> None:
            rep.path_switches += 1
            tr.add(sim.now(), "path:switch", "isl",
                   f"跳数 {int(tcp_profile.hops[i])}→{int(tcp_profile.hops[i + 1])}")
            # 波及在途流：真实 TCP 的切换代价（发射静默 + 窗口收缩）
            n_hit = 0
            for conn in active_conns:
                if conn.on_path_switch(sim.now()):
                    n_hit += 1
            if n_hit:
                rep.flow_switches += n_hit
                tr.add(sim.now(), "flow:switch", "tcp",
                       f"{n_hit} 条在途流承受切换代价")

        for i in idxs:
            sim.schedule_at(float(tcp_profile.t[i + 1]), on_switch, int(i),
                            name="path:switch", priority=3)

    # ---------------- ④ TCP 流（共享内核非阻塞并发） ---------------- #
    rng = sim.rng.get("des:flow")
    if tcp_profile is not None and des_cfg.n_flows > 0:
        # 到达时刻：rate 给定则泊松；否则在窗口内均匀铺开 + 抖动
        if des_cfg.flow_rate > 0:
            ts_f: List[float] = []
            t_cur = 0.05 * duration_s
            while t_cur < 0.95 * duration_s and len(ts_f) < 4 * max(des_cfg.n_flows, 1):
                ts_f.append(t_cur)
                t_cur += rng.exponential(1.0 / max(des_cfg.flow_rate, 1e-9))
        else:
            base = np.linspace(0.05, 0.90, max(des_cfg.n_flows, 1)) * duration_s
            ts_f = [float(b + rng.uniform(-0.02, 0.02) * duration_s) for b in base]
        ts_f = ts_f[:max(des_cfg.n_flows, 1)]

        tcp_cfg = tcp_cfg or TcpConfig()
        flow_bytes = int(max(des_cfg.flow_mbit, 0.001) * 1e6 / 8.0)

        for i, ta in enumerate(ts_f):
            rec = DesFlowRecord(idx=i, arrival_s=float(ta),
                                algo=des_cfg.tcp_algo)

            def on_flow_arrive(i=i, rec=rec) -> None:
                tr.add(sim.now(), "flow:arrive", f"flow{i}",
                       f"{ALGO_LABELS.get(rec.algo, rec.algo)}")
                conn = TcpConnection(tcp_profile, tcp_cfg, algo=rec.algo,
                                     seed=des_cfg.seed + 97 * (i + 1), sim=sim)
                active_conns.append(conn)
                conn.start(flow_bytes, handshake_s=conn.handshake_seconds(),
                           on_done=lambda c, rec=rec: on_flow_done(c, rec))
                # 看门狗：流超过 max_time 仍未完成则终止（防长尾事件撑爆时间轴）
                sim.schedule_at(sim.now() + tcp_cfg.max_time_s + 1.0,
                                lambda rec=rec: on_flow_timeout(rec),
                                name=f"flow{i}:watchdog")

            def on_flow_done(conn: TcpConnection, rec: DesFlowRecord) -> None:
                res = conn._finish(conn._total_bytes)
                rec.done = True
                rec.completion_s = res.completion_s
                rec.goodput_mbps = res.goodput_mbps
                rec.retransmits = res.retransmits
                tr.add(sim.now(), "flow:done", f"flow{rec.idx}",
                       f"{res.goodput_mbps:.1f} Mbps")

            def on_flow_timeout(rec: DesFlowRecord) -> None:
                if not rec.done:
                    rep.notes.append(f"流 {rec.idx} 超时未完成（看门狗终止）")
                    tr.add(sim.now(), "flow:timeout", f"flow{rec.idx}", "")

            sim.schedule_at(float(ta), on_flow_arrive, name=f"flow{i}:arrive",
                            priority=2)
            rep.flows.append(rec)

    # ---------------- ⑤ 计算任务（泊松到达 + JSQ 分发 + FIFO 服务） ---------------- #
    tops_eff = max(compute_cfg.sat_capacity_tops * (1.0 - compute_cfg.radiation_derate),
                   1e-6)
    nodes = [_DesNode(f"sat{j:02d}", tops_eff) for j in range(max(des_cfg.n_nodes, 1))]
    rep.node_names = [n.name for n in nodes]
    rng_t = sim.rng.get("des:task")
    task_rate = (des_cfg.task_rate if des_cfg.task_rate > 0
                 else des_cfg.n_tasks / duration_s)

    def on_task_arrive() -> None:
        gf = float(rng_t.uniform(*compute_cfg.task_cycles_gops))
        size = float(rng_t.uniform(*compute_cfg.task_size_mbit))
        # JSQ：加入当前最短的队列（含在服务的 1 个）
        node = min(nodes, key=lambda n: n.depth)
        tid = len(rep.tasks)
        deadline = compute_cfg.task_deadline_s
        rec = DesTaskRecord(tid=tid, arrival_s=sim.now(), start_s=sim.now(),
                            finish_s=sim.now(), node=node.name, gflops=gf,
                            deadline_s=deadline)
        rep.tasks.append(rec)
        tr.add(sim.now(), "task:arrive", node.name, f"#{tid} {gf:.0f} GOPS")
        node.queue.append((rec, size))
        if not node.busy:
            _serve_next(node)

    def _serve_next(node: _DesNode) -> None:
        if not node.queue:
            node.busy = False
            return
        node.busy = True
        rec, _size = node.queue.popleft()
        # 与 computing.py 同一口径：work_tops_s = cycles_gops/1000，服务 = work/capacity
        service = rec.gflops / 1000.0 / node.tops
        rec.start_s = sim.now()

        def on_task_done(node=node, rec=rec) -> None:
            rec.finish_s = sim.now()
            node.n_done += 1
            node.busy_accum += rec.service_s
            tr.add(sim.now(), "task:done", node.name,
                   f"#{rec.tid} 等待 {rec.wait_s:.3f}s")
            _serve_next(node)

        sim.schedule_at(sim.now() + service, on_task_done,
                        name=f"{node.name}:serve", priority=1)

    # 到达事件：泊松过程逐个调度
    t_a = 0.0
    n_arr = 0
    while n_arr < des_cfg.n_tasks:
        t_a += rng_t.exponential(1.0 / max(task_rate, 1e-9))
        if t_a >= duration_s * 0.98:
            break
        sim.schedule_at(t_a, on_task_arrive, name="task:arrive", priority=2)
        n_arr += 1

    # ---------------- ⑥ 指标采样 ---------------- #
    def on_sample() -> None:
        depth = sum(n.depth for n in nodes)
        rep.queue_series.append((float(sim.now()), int(depth)))
        tr.add(sim.now(), "metric:sample", "core", f"队列深度 {depth}")
        say(f"事件时间轴 {sim.now():.0f}/{duration_s:.0f} s"
            f"（队列 {depth}，已完成流 {rep.flows_done}/{len(rep.flows)}）")

    t_s = des_cfg.sample_every_s
    while t_s < duration_s:
        sim.schedule_at(t_s, on_sample, name="metric:sample")
        t_s += max(des_cfg.sample_every_s, 0.5)

    # ---------------- 运行 ---------------- #
    say("全局离散事件编排 …")
    sim.run(until=duration_s)
    rep.n_executed = sim.n_executed
    rep.n_cancelled = sim.n_cancelled
    rep.sim_end_s = sim.now()
    if rep.flows and rep.flows_done == 0:
        rep.notes.append("所有流均未完成：检查 PathProfile 是否覆盖时间轴 / 载荷是否过大")
    if rep.queue_series and max(d for _, d in rep.queue_series) == 0:
        util = [u for u in rep.node_utilization.values()]
        rep.notes.append(
            "队列全程无排队（负载较轻）：任务到达率远低于总算力，"
            "如需观察排队/等待时延，请提高 des.n_tasks / task_rate 或减小 n_nodes。")
        if util and max(util) < 0.05:
            rep.notes.append(
                f"节点利用率峰值仅 {max(util)*100:.1f}%：星载算力相对任务规模富余，"
                "与 computing.py 的批量放置口径一致。")
    return rep

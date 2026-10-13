"""星载计算层：任务模型、算力-能耗约束、卸载与调度。

背景与建模依据
--------------
传统"弯管（bent-pipe）"架构把星上原始数据全部下传、在地面算，随星座规模增长
会撞上下行带宽与落地站容量的墙。Orbital Edge Computing（OEC）把计算推到星上，
只下传"有价值的结果"：

* Denby & Lucia, *Orbital Edge Computing: Nanosatellite Constellations as a New
  Class of Computer System*, ASPLOS 2020 —— 提出计算纳米卫星流水线（CNP），
  报告可把地面基础设施规模降低 24×、边缘处理时延降低 617×。
* Denby et al., *Kodan: Addressing the Computational Bottleneck in Space*,
  ASPLOS 2023, pp.392–403 —— 在"下行饱和 + 星上算力刚性"双重约束下按
  **数据价值密度**调度，价值密度提升 89%–97%。
* Xing et al., *Deciphering the Enigma of Satellite Computing with COTS Devices*,
  ACM MobiCom 2024 —— COTS 星载计算的实测特征（功耗/热/辐射降级）。
* RHONE（USENIX ATC 2025）—— 用遥测数据 + 硬件在环做功耗/热/轨道建模，
  单机 700 星，功耗与计算模型误差 < 5%，热模型误差 1.3–2.5 ℃。

本模块的抽象
------------
* **任务**：数据量、计算量、时延期限、价值。
* **节点**：算力（TOPS，含辐射降级）、功耗曲线、能量预算。
* **调度**：把任务映射到"星上算"或"地面算"，并决定执行顺序。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ComputeConfig


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    """一个可被卸载的感知/推理任务。"""

    tid: int
    origin_gs: int
    arrival_s: float
    # 星上处理所需输入数据（若在星上算，只需上传指令+必要上下文）
    input_mbit: float
    # 弯管架构下需要下传的原始数据量（通常远大于 input）
    raw_data_mbit: float
    cycles_gops: float  # 计算量，单位 GFLOPs 量级
    deadline_s: float
    value: float  # 任务价值（0–1），用于价值密度调度
    result_mbit: float

    @property
    def work_tops_s(self) -> float:
        """把 GFLOPs 折算成 TOPS·s。"""
        return self.cycles_gops / 1000.0


@dataclass
class ComputeNode:
    id: int
    name: str
    capacity_tops: float
    power_idle_w: float
    power_peak_w: float
    energy_budget_wh: float
    is_satellite: bool
    # 运行时状态
    busy_until_s: float = 0.0
    energy_used_wh: float = 0.0
    planned_energy_wh: float = 0.0
    tasks_done: int = 0
    reject_count: int = 0

    @property
    def energy_left_wh(self) -> float:
        if self.energy_budget_wh <= 0:
            return float("inf")
        return self.energy_budget_wh - self.energy_used_wh

    def exhausted(self, extra_wh: float = 0.0) -> bool:
        return self.energy_left_wh - extra_wh <= 0.0


@dataclass
class TaskRecord:
    tid: int
    node: str
    is_satellite: bool
    arrival_s: float
    start_s: float
    finish_s: float
    latency_s: float
    deadline_met: bool
    energy_wh: float
    compute_s: float
    transfer_s: float
    policy: str
    value: float = 0.0
    downlink_s: float = 0.0


@dataclass
class ComputeReport:
    policy: str
    n_tasks: int
    records: List[TaskRecord] = field(default_factory=list)
    energy_wh: Dict[str, float] = field(default_factory=dict)
    node_load: Dict[str, float] = field(default_factory=dict)
    throughput_data_mbit: float = 0.0
    downlink_mbit: float = 0.0

    # ---- 汇总指标 ---- #
    @property
    def mean_latency_s(self) -> float:
        return float(np.mean([r.latency_s for r in self.records])) if self.records else 0.0

    @property
    def p95_latency_s(self) -> float:
        return float(np.percentile([r.latency_s for r in self.records], 95)) if self.records else 0.0

    @property
    def deadline_miss_rate(self) -> float:
        if not self.records:
            return 0.0
        return float(np.mean([0.0 if r.deadline_met else 1.0 for r in self.records]))

    @property
    def offload_ratio(self) -> float:
        if not self.records:
            return 0.0
        return float(np.mean([1.0 if r.is_satellite else 0.0 for r in self.records]))

    @property
    def total_sat_energy_wh(self) -> float:
        return float(sum(v for k, v in self.energy_wh.items() if not k.startswith("cloud")))

    @property
    def load_imbalance(self) -> float:
        """节点负载的变异系数（越小越均衡）。"""
        vals = np.array([v for v in self.node_load.values() if v > 0], dtype=float)
        if vals.size == 0 or vals.mean() <= 0:
            return 0.0
        return float(vals.std() / vals.mean())

    @property
    def value_throughput(self) -> float:
        """单位时间交付的**有效价值**（v/s）：超期任务不计入。

        这是价值密度调度的目标函数；与 `mean_latency_s` 构成一对张力指标。
        """
        if not self.records:
            return 0.0
        horizon = self.horizon_s
        if horizon <= 0:
            return 0.0
        return float(
            sum(r.value for r in self.records if r.deadline_met) / horizon
        )

    @property
    def horizon_s(self) -> float:
        if not self.records:
            return 0.0
        return max(max(r.arrival_s for r in self.records), 1e-9)

    def high_value_miss_rate(self, threshold: float = 0.7) -> float:
        """高价值任务（value ≥ threshold）的超期比例。"""
        hv = [r for r in self.records if r.value >= threshold]
        if not hv:
            return 0.0
        return float(np.mean([0.0 if r.deadline_met else 1.0 for r in hv]))

    def summary_row(self) -> Dict[str, float]:
        return {
            "policy": self.policy,
            "n_tasks": self.n_tasks,
            "mean_latency_s": self.mean_latency_s,
            "p95_latency_s": self.p95_latency_s,
            "deadline_miss_pct": self.deadline_miss_rate * 100.0,
            "high_value_miss_pct": self.high_value_miss_rate() * 100.0,
            "offload_pct": self.offload_ratio * 100.0,
            "sat_energy_wh": self.total_sat_energy_wh,
            "load_imbalance_cv": self.load_imbalance,
            "downlink_mbit": self.downlink_mbit,
            "value_throughput": self.value_throughput,
        }


# --------------------------------------------------------------------------- #
# 任务生成
# --------------------------------------------------------------------------- #
def generate_tasks(
    cfg: ComputeConfig,
    n_gs: int,
    duration_s: float,
    seed: int = 0,
    poisson: bool = True,
) -> List[Task]:
    """生成泊松到达的感知/推理任务流（默认速率 = 任务数 / 仿真时长）。"""
    rng = np.random.default_rng(seed)
    lo_sz, hi_sz = cfg.task_size_mbit
    lo_c, hi_c = cfg.task_cycles_gops

    if poisson:
        rate = cfg.task_count / max(duration_s, 1.0)
        arrivals = np.cumsum(rng.exponential(1.0 / max(rate, 1e-9), cfg.task_count))
        arrivals = arrivals[arrivals <= duration_s]
    else:
        arrivals = np.linspace(0.0, duration_s, cfg.task_count, endpoint=False)

    tasks: List[Task] = []
    for i, t0 in enumerate(arrivals):
        raw = float(rng.uniform(lo_sz, hi_sz))
        # 星上过滤后上行的数据量：默认压缩/裁剪到 10%–40%
        input_mbit = raw * float(rng.uniform(0.10, 0.40))
        tasks.append(
            Task(
                tid=i,
                origin_gs=int(rng.integers(0, max(n_gs, 1))),
                arrival_s=float(t0),
                input_mbit=input_mbit,
                raw_data_mbit=raw,
                cycles_gops=float(rng.uniform(lo_c, hi_c)),
                deadline_s=cfg.task_deadline_s,
                value=float(rng.uniform(0.2, 1.0)),
                result_mbit=float(rng.uniform(0.5, 8.0)),
            )
        )
    return tasks


# --------------------------------------------------------------------------- #
# 调度器
# --------------------------------------------------------------------------- #
class ComputeSimulator:
    """把任务流映射到"星上/地面"并推进时间轴。

    visibility_fn(gs_local, t_s) -> 该时刻可见的卫星索引数组（可选）。
    不提供时认为任意卫星均可服务。
    """

    def __init__(self, cfg: ComputeConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    def build_nodes(
        self,
        n_sat: int,
        gsl_bandwidth_mbps: float,
        satellite_solar_fraction: Optional[np.ndarray] = None,
    ) -> Tuple[List[ComputeNode], ComputeNode]:
        derate = 1.0 - float(np.clip(self.cfg.radiation_derate, 0.0, 0.95))
        solar = (
            1.0
            if satellite_solar_fraction is None
            else float(np.clip(np.mean(satellite_solar_fraction), 0.05, 1.0))
        )
        sats: List[ComputeNode] = []
        for i in range(n_sat):
            sats.append(
                ComputeNode(
                    id=i,
                    name=f"SAT-{i}",
                    capacity_tops=self.cfg.sat_capacity_tops * derate,
                    power_idle_w=self.cfg.sat_power_idle_w,
                    power_peak_w=self.cfg.sat_power_peak_w,
                    energy_budget_wh=self.cfg.sat_energy_budget_wh * solar,
                    is_satellite=True,
                )
            )
        cloud = ComputeNode(
            id=n_sat,
            name="GROUND-CLOUD",
            capacity_tops=self.cfg.cloud_capacity_tops,
            power_idle_w=0.0,
            power_peak_w=1500.0,
            energy_budget_wh=0.0,  # 0 表示不限制
            is_satellite=False,
        )
        return sats, cloud

    # ------------------------------------------------------------------ #
    def run(
        self,
        tasks: Sequence[Task],
        n_sat: int,
        gsl_bandwidth_mbps: float,
        satellite_solar_fraction: Optional[np.ndarray] = None,
        visibility_fn: Optional[Callable[[int, float], np.ndarray]] = None,
        policy: Optional[str] = None,
        gs_antennas: int = 4,
    ) -> ComputeReport:
        """两阶段调度：

        阶段一（放置）：按**到达序**遍历任务，为每个任务挑节点。到达序保证
        先到先分配，避免"后面的任务反过来抬高前面任务的排队时间"这一经典建模陷阱。
        阶段二（排队）：在每个节点内部按调度策略排序，按非抢占优先级队列串行推进，
        且任何任务不得早于 `到达时刻 + 上行传输` 开始执行。
        """
        policy = policy or self.cfg.scheduler
        sats, cloud = self.build_nodes(n_sat, gsl_bandwidth_mbps, satellite_solar_fraction)
        report = ComputeReport(policy=policy, n_tasks=len(tasks))
        self._gs_antennas = max(int(gs_antennas), 1)

        for node in list(sats) + [cloud]:
            report.energy_wh[node.name] = 0.0
            report.node_load[node.name] = 0.0
            node.planned_energy_wh = 0.0

        # ---------------- 阶段一：放置 ---------------- #
        buckets: Dict[str, List[Task]] = {n.name: [] for n in list(sats) + [cloud]}
        for task in sorted(tasks, key=lambda t: t.arrival_s):
            cand = self._candidates(task, sats, cloud, visibility_fn, policy)
            chosen = self._select(task, cand, gsl_bandwidth_mbps, policy)
            buckets[chosen.name].append(task)
            need = self._energy_of(
                task.work_tops_s / max(chosen.capacity_tops, 1e-6), chosen
            )
            chosen.planned_energy_wh += need

        # ---------------- 阶段二：节点内非抢占优先级队列（事件驱动） ---------------- #
        for node in list(sats) + [cloud]:
            self._drain_queue(
                node, buckets[node.name], gsl_bandwidth_mbps, policy, report
            )

        # ---------------- 阶段三：共享下行队列 ---------------- #
        # 这是"弯管 vs OEC"的分水岭：地面站的落地带宽是**全星座共享**的稀缺资源。
        # 弯管架构下每个任务都要下传整幅原始数据，下行队列迅速饱和；OEC 只下传
        # 结果（本例中约为原始数据的 1%–3%），同样的落地带宽能支撑两个数量级的
        # 任务量。这一步把该效应显式建模出来。
        n_gs = max((int(t.origin_gs) for t in tasks), default=0) + 1
        self._downlink_queues(report, tasks, n_gs, gsl_bandwidth_mbps)

        report.records.sort(key=lambda r: r.tid)
        return report

    # ------------------------------------------------------------------ #
    def _downlink_queues(
        self,
        report: ComputeReport,
        tasks: Sequence[Task],
        n_gs: int,
        gsl_bandwidth_mbps: float,
    ) -> None:
        """按地面站对落地数据做 FIFO 排队，修正任务端到端时延。

        每个地面站的落地能力 = 天线数 × 单站带宽（简化：忽略可见窗口的空档，
        因为任务已经被约束为"由该站可见的卫星服务"）。
        """
        by_tid = {t.tid: t for t in tasks}
        gs_capacity = max(float(gsl_bandwidth_mbps), 1e-6) * max(
            int(getattr(self, "_gs_antennas", 4)), 1
        )
        busy: Dict[int, float] = {}
        # 按"计算完成时刻"排序，模拟数据就绪后的落地竞争
        for rec in sorted(report.records, key=lambda r: r.finish_s):
            task = by_tid.get(rec.tid)
            if task is None:
                continue
            data_mbit = task.result_mbit if rec.is_satellite else task.raw_data_mbit
            gs = int(task.origin_gs) % max(n_gs, 1)
            start = max(rec.finish_s, busy.get(gs, 0.0))
            duration = data_mbit / gs_capacity
            finish = start + duration
            busy[gs] = finish
            rec.downlink_s = duration
            rec.finish_s = finish
            rec.latency_s = finish - rec.arrival_s
            rec.deadline_met = rec.latency_s <= task.deadline_s

    # ------------------------------------------------------------------ #
    def _drain_queue(
        self,
        node: ComputeNode,
        tasks: Sequence[Task],
        gsl_bw: float,
        policy: str,
        report: ComputeReport,
    ) -> None:
        """在单个节点上推进非抢占优先级队列。

        关键点：**后来到达的任务绝不能拖延先到的任务**。因此必须按事件推进 ——
        只有"release ≤ 当前忙时刻"的任务才进入待服务池，服务器空闲时若池内无任务，
        就把时钟直接跳到最近一个 release，而不是让 busy 被未来任务推高。
        """
        if not tasks:
            return

        def release(t: Task) -> float:
            return t.arrival_s + self._transfer_time(t, node, gsl_bw)

        rng = np.random.default_rng(20260320)
        rand_key = {t.tid: float(rng.random()) for t in tasks}
        ordered = sorted(tasks, key=release)
        horizon = max((t.arrival_s for t in tasks), default=1.0) or 1.0
        aging = float(max(self.cfg.aging, 0.0))

        busy = 0.0
        ready: List[Task] = []
        i = 0
        guard = 0
        while (i < len(ordered) or ready) and guard < 10 * len(tasks) + 1000:
            guard += 1
            if not ready:
                busy = max(busy, release(ordered[i]))
            while i < len(ordered) and release(ordered[i]) <= busy + 1e-12:
                ready.append(ordered[i])
                i += 1

            # 选下一个执行的任务
            if policy == "value_density":
                def key(t: Task) -> float:
                    wait = max(busy - t.arrival_s, 0.0)
                    return -(t.value / max(t.work_tops_s, 1e-6)) * (
                        1.0 + aging * wait / horizon
                    )
            elif policy == "random":
                def key(t: Task) -> float:
                    return rand_key[t.tid]
            else:  # FIFO
                def key(t: Task) -> float:
                    return release(t)

            idx = min(range(len(ready)), key=lambda k: key(ready[k]))
            task = ready.pop(idx)

            tr = self._transfer_time(task, node, gsl_bw)
            comp = task.work_tops_s / max(node.capacity_tops, 1e-6)
            e = self._energy_of(comp, node)
            start = max(busy, release(task))
            finish = start + comp
            busy = finish
            node.busy_until_s = finish
            node.energy_used_wh += e
            node.tasks_done += 1

            latency = finish - task.arrival_s
            report.records.append(
                TaskRecord(
                    tid=task.tid,
                    node=node.name,
                    is_satellite=node.is_satellite,
                    arrival_s=task.arrival_s,
                    start_s=start,
                    finish_s=finish,
                    latency_s=latency,
                    deadline_met=latency <= task.deadline_s,
                    energy_wh=e,
                    compute_s=comp,
                    transfer_s=tr,
                    policy=policy,
                    value=task.value,
                )
            )
            report.energy_wh[node.name] += e
            report.node_load[node.name] += comp
            # 跨星地链路的数据量：星上算只下传结果，地面算要下传整幅原始数据
            if node.is_satellite:
                report.downlink_mbit += task.result_mbit
                report.throughput_data_mbit += task.result_mbit
            else:
                report.downlink_mbit += task.raw_data_mbit
                report.throughput_data_mbit += task.result_mbit

    # ------------------------------------------------------------------ #
    def _candidates(
        self,
        task: Task,
        sats: Sequence[ComputeNode],
        cloud: ComputeNode,
        visibility_fn: Optional[Callable[[int, float], np.ndarray]],
        policy: str,
    ) -> List[ComputeNode]:
        if policy == "local_only":
            return [cloud]
        if policy == "random":
            rng = np.random.default_rng((task.tid + 1) * 7919)
            idx = int(rng.integers(0, max(len(sats), 1)))
            pick = sats[idx] if sats else cloud
            if pick.is_satellite and pick.energy_left_wh <= 1.0:
                return [cloud]
            return [pick]

        visible: Optional[np.ndarray] = None
        if visibility_fn is not None and sats:
            try:
                visible = np.asarray(visibility_fn(task.origin_gs, task.arrival_s))
            except Exception:
                visible = None
        pool: List[ComputeNode] = list(sats)
        if visible is not None and visible.size:
            pool = [sats[int(i)] for i in visible if 0 <= int(i) < len(sats)]
        if not pool:
            pool = list(sats)

        need = task.work_tops_s / max(self.cfg.sat_capacity_tops or 1.0, 1e-6)
        avail = [
            n for n in pool
            if n.capacity_tops > 1e-6
            and n.energy_left_wh - n.planned_energy_wh - need > 0.0
        ]
        if not avail:
            return [cloud]
        scored = sorted(
            avail,
            key=lambda n: n.busy_until_s
            + task.work_tops_s / max(n.capacity_tops, 1e-6),
        )[:8]
        return scored + [cloud]

    # ------------------------------------------------------------------ #
    def _transfer_time(self, task: Task, node: ComputeNode, gsl_bw: float) -> float:
        """上行（+ 若在地面算则下行原始数据）的传输时间。"""
        if node.is_satellite:
            return task.input_mbit / max(gsl_bw, 1e-9)
        return task.raw_data_mbit / max(gsl_bw, 1e-9)

    def _select(
        self,
        task: Task,
        cand: Sequence[ComputeNode],
        gsl_bw: float,
        policy: str,
    ) -> ComputeNode:
        """节点选择：在所有候选中取"预计完成时刻"最小者。

        注意价值不参与节点选择（对同一任务而言价值是公共因子，不影响排序），
        它只决定节点内部的排队优先级 —— 这正是 Kodan"按价值密度筛选"的语义。
        """
        best, best_finish = None, float("inf")
        for n in cand:
            tr = self._transfer_time(task, n, gsl_bw)
            comp = task.work_tops_s / max(n.capacity_tops, 1e-6)
            start = max(n.busy_until_s, task.arrival_s + tr)
            finish = start + comp
            if n.is_satellite and n.energy_left_wh - n.planned_energy_wh - self._energy_of(
                comp, n
            ) <= 0.0:
                continue
            if finish < best_finish:
                best, best_finish = n, finish
        return best or cand[-1]

    def _energy_of(self, compute_s: float, node: ComputeNode) -> float:
        """能耗（Wh）= 峰值功耗 × 计算时长 / 3600。"""
        return node.power_peak_w * compute_s / 3600.0


# --------------------------------------------------------------------------- #
# 架构对比
# --------------------------------------------------------------------------- #
def compare_architectures(
    cfg: ComputeConfig,
    n_sat: int,
    n_gs: int,
    duration_s: float,
    gsl_bandwidth_mbps: float,
    seed: int = 0,
    visibility_fn: Optional[Callable[[int, float], np.ndarray]] = None,
    gs_antennas: int = 4,
) -> Dict[str, ComputeReport]:
    """对比四种架构/策略。

    local_only      —— 弯管架构：星上不处理，原始数据全部下传后在地面算
    earliest_finish —— OEC + FIFO 调度（最小化平均时延）
    value_density   —— OEC + 价值密度优先 + 优先级老化（Kodan 思路，最大化交付价值）
    random          —— 随机卸载（消融基线）

    三阶段：放置 → 节点内优先级队列 → 地面站共享下行队列。
    """
    tasks = generate_tasks(cfg, n_gs, duration_s, seed=seed)
    sim = ComputeSimulator(cfg)
    out: Dict[str, ComputeReport] = {}
    for policy in ("local_only", "earliest_finish", "value_density", "random"):
        out[policy] = sim.run(
            tasks, n_sat, gsl_bandwidth_mbps,
            visibility_fn=visibility_fn, policy=policy,
            gs_antennas=gs_antennas,
        )
    return out


def sweep_sat_capacity(
    cfg: ComputeConfig,
    n_sat: int,
    n_gs: int,
    duration_s: float,
    gsl_bandwidth_mbps: float,
    capacities_tops: Sequence[float] = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    seed: int = 0,
    visibility_fn: Optional[Callable[[int, float], np.ndarray]] = None,
    gs_antennas: int = 4,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """扫描单星算力，找出"调度策略开始起作用"的拥塞区间。

    返回 (算力档位, {策略: 平均时延列表})。当单星算力高到一定程度后，
    各策略曲线会收敛（星上算力过剩）；反之在低算力区曲线分离最明显。
    """
    tasks = generate_tasks(cfg, n_gs, duration_s, seed=seed)
    curves: Dict[str, List[float]] = {p: [] for p in
                                      ("local_only", "earliest_finish",
                                       "value_density", "random")}
    for cap in capacities_tops:
        cfg.sat_capacity_tops = float(cap)
        sim = ComputeSimulator(cfg)
        for pol in curves:
            rep = sim.run(tasks, n_sat, gsl_bandwidth_mbps,
                          visibility_fn=visibility_fn, policy=pol,
                          gs_antennas=gs_antennas)
            curves[pol].append(rep.mean_latency_s)
    return list(capacities_tops), curves

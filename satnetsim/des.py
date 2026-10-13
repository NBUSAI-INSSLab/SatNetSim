"""离散事件仿真内核（Discrete-Event Simulation kernel）——对标 ns-3 的 ``Simulator``。

为什么需要它
------------
原先的引擎是**步进式**的：先按固定步长推进轨道与拓扑，再依次调用各层分析函数，
每层内部各自维护自己的小循环（``tcpstack`` 自己有一个 heapq，``realdatastream``
自己有一个 heapq）。这种写法有三个代价：

1. **时间语义不统一**：不同层里的 "t" 含义不同（快照序号 / 轮次 / 秒），
   跨层耦合只能靠"上层跑完传参数给下层"，无法表达"A 层的事件触发 B 层的事件"；
2. **无法处理同时刻多源事件**：卫星切换、链路中断、任务到达、定时器超时
   在同一时刻竞争时，没有统一的定序规则（ns-3 里这是 Scheduler 的职责）；
3. **不可复现**：各层各自的随机数流相互独立，参数一改就整条链路漂移。

本模块提供 ns-3 风格的统一时间轴：

* :class:`Scheduler` —— 事件队列（Heap / List 两种实现，语义一致，可对比性能）；
* :class:`Simulator` —— ``Schedule`` / ``Cancel`` / ``Run`` / ``Stop`` / ``Now``，
  与 ``ns3::Simulator`` 一一对应；
* :class:`EventId` —— 事件句柄，支持取消与过期查询（``ns3::EventId``）；
* :class:`Process` —— 生成器协程进程（``Yield`` / ``WaitUntil``），
  把"协议状态机"写成**顺序代码**而不是回调地狱；
* :class:`Trace` —— 事件追踪（环形缓冲 + 分类统计 + 可选流式回调），
  供 GUI 画事件时间线；
* :class:`RandStream` —— 命名确定性随机流，保证同种子同结果。

定序规则
--------
同一时刻的事件按 ``(时间, 优先级降序, 插入序)`` 出队——与 ns-3 一致：
时间相同则优先级高者先执行，再相同则先插入者先执行（FIFO 稳定序）。
**绝不使用对象地址或字典序做 tie-break**，否则不同进程/不同运行会给出不同结果。

用法
----
::

    from satnetsim.des import Simulator

    sim = Simulator(seed=1)
    def hello(who):
        print(f"{sim.now():.3f}s hello {who}")
        sim.schedule(1.0, hello, "again")     # 相对延时
    sim.schedule(0.5, hello, "world")
    sim.run(until=2.5)

与 ns-3 的对应关系
------------------
============================  ==========================  ==========================
本模块                          ns-3                       说明
============================  ==========================  ==========================
``Simulator.schedule``          ``Simulator::Schedule``     相对延时调度
``Simulator.schedule_at``       ``Simulator::ScheduleWithContext`` / 绝对时刻
``Simulator.cancel``            ``Simulator::Cancel``       取消未执行事件
``Simulator.stop``              ``Simulator::Stop``         停止（可再 run 续跑）
``Simulator.run``               ``Simulator::Run``          主循环
``HeapScheduler`` / ``ListScheduler``  ``HeapScheduler`` / ``ListScheduler``  队列实现可换
``EventId``                     ``EventId``                 句柄
``Process``                     （无直接对应）               生成器协程，等价于 SimPy 的 Process
============================  ==========================  ==========================

参考：ns-3 Model Library, "Simulator" 一节（Scheduler / EventId / MakeEvent /
Simulator::Schedule / Run / Stop / Destroy）。
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EventId", "Event", "Scheduler", "HeapScheduler", "ListScheduler",
    "Simulator", "Process", "Trace", "TraceRecord", "RandStream",
    "Yield", "WaitUntil", "StopProcess",
    "des_self_test",
]


# --------------------------------------------------------------------------- #
# 1. 事件与句柄
# --------------------------------------------------------------------------- #
class EventId:
    """事件句柄（不可变）。``ns3::EventId`` 的对等物。

    ``Cancel()`` 是**惰性**的：只把事件标记为已取消，真正的清除发生在它出队时，
    这样取消操作是 O(1) 的，不需要在堆里定位。
    """

    __slots__ = ("uid", "t", "priority", "name", "_cancelled")

    def __init__(self, uid: int, t: float, priority: int, name: str) -> None:
        self.uid = uid
        self.t = float(t)
        self.priority = int(priority)
        self.name = name
        self._cancelled = False

    def Cancel(self) -> None:  # noqa: N802  (对齐 ns-3 命名)
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def __repr__(self) -> str:  # pragma: no cover
        tag = " (cancelled)" if self._cancelled else ""
        return f"<EventId #{self.uid} t={self.t:g} {self.name!r}{tag}>"


@dataclass
class Event:
    """队列里的一条事件。用户一般不直接构造，用 ``Simulator.schedule*``。"""

    t: float
    priority: int
    uid: int
    fn: Callable[..., Any]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    eid: EventId
    name: str = ""

    def run(self) -> Any:
        return self.fn(*self.args, **self.kwargs)


# --------------------------------------------------------------------------- #
# 2. 调度器（队列实现）
# --------------------------------------------------------------------------- #
class Scheduler:
    """事件队列抽象基类：只暴露 ns-3 Scheduler 的四个接口。

    ``Remove`` 是惰性实现（打标记），因此所有实现都只需保证"出队有序"。
    """

    def Insert(self, ev: Event) -> None:  # noqa: N802
        raise NotImplementedError

    def PeekNext(self) -> Optional[Event]:  # noqa: N802
        raise NotImplementedError

    def RemoveNext(self) -> Optional[Event]:  # noqa: N802
        raise NotImplementedError

    def IsEmpty(self) -> bool:  # noqa: N802
        raise NotImplementedError

    @property
    def depth(self) -> int:
        raise NotImplementedError


class HeapScheduler(Scheduler):
    """二叉堆实现（ns-3 默认）。插入 O(log n)，取最小 O(log n)。

    排序键 ``(t, -priority, uid)``：时刻最早者先；同刻高优先级先；再同则先入队者先。
    """

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, int, Event]] = []

    def Insert(self, ev: Event) -> None:  # noqa: N802
        heapq.heappush(self._heap, (ev.t, -ev.priority, ev.uid, ev))

    def PeekNext(self) -> Optional[Event]:  # noqa: N802
        return self._heap[0][3] if self._heap else None

    def RemoveNext(self) -> Optional[Event]:  # noqa: N802
        return heapq.heappop(self._heap)[3] if self._heap else None

    def IsEmpty(self) -> bool:  # noqa: N802
        return not self._heap

    @property
    def depth(self) -> int:
        return len(self._heap)


class ListScheduler(Scheduler):
    """有序链表实现（ns-3 的 ListScheduler）。插入 O(n)，取最小 O(1)。

    只在小规模或需要逐条可视化事件流的场景下更直观；本模块保留它是为了
    **对照验证**两种实现的语义完全一致（``tests/verify_des.py`` 会断言）。
    """

    def __init__(self) -> None:
        self._items: List[Event] = []

    @staticmethod
    def _key(ev: Event) -> Tuple[float, int, int]:
        return (ev.t, -ev.priority, ev.uid)

    def Insert(self, ev: Event) -> None:  # noqa: N802
        import bisect

        bisect.insort(self._items, ev, key=self._key)

    def PeekNext(self) -> Optional[Event]:  # noqa: N802
        return self._items[0] if self._items else None

    def RemoveNext(self) -> Optional[Event]:  # noqa: N802
        return self._items.pop(0) if self._items else None

    def IsEmpty(self) -> bool:  # noqa: N802
        return not self._items

    @property
    def depth(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------- #
# 3. 事件追踪
# --------------------------------------------------------------------------- #
@dataclass
class TraceRecord:
    t: float
    kind: str
    entity: str
    detail: str = ""
    priority: int = 0

    def as_tuple(self) -> Tuple[float, str, str, str]:
        return (self.t, self.kind, self.entity, self.detail)


class Trace:
    """事件追踪器：环形缓冲 + 分类计数 + 可选流式回调。

    设计目的有两个：
      * **诊断**：仿真时间轴上到底发生了什么、谁触发了谁；
      * **可视化**：GUI 的「事件内核」页签直接消费 ``records`` 画时间线甘特图。

    ``max_records`` 满后按 FIFO 丢弃最旧的记录（长仿真下事件数可达百万级，
    但可视化只需要最近的一批），同时 ``counts`` 仍然统计**全部**事件。
    """

    def __init__(self, max_records: int = 20000,
                 callback: Optional[Callable[[TraceRecord], None]] = None) -> None:
        self.max_records = int(max_records)
        self.records: List[TraceRecord] = []
        self.counts: Dict[str, int] = {}
        self.dropped = 0
        self.enabled = True
        self._callback = callback

    def add(self, t: float, kind: str, entity: str, detail: str = "",
            priority: int = 0) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if not self.enabled:
            return
        rec = TraceRecord(float(t), str(kind), str(entity), str(detail), int(priority))
        if len(self.records) >= self.max_records:
            self.records.pop(0)
            self.dropped += 1
        self.records.append(rec)
        if self._callback is not None:
            self._callback(rec)

    # ------------------------------------------------------------------ #
    def kinds(self) -> List[str]:
        return sorted(self.counts)

    def span(self) -> Tuple[float, float]:
        if not self.records:
            return (0.0, 0.0)
        return (self.records[0].t, self.records[-1].t)

    def timeline(self, kinds: Optional[Sequence[str]] = None,
                 max_items: int = 400) -> List[Tuple[str, float, float, str]]:
        """把事件压成"活动条"：相邻同类同实体事件合并为一段 ``(标签, 起, 止, 详情)``。

        用于画甘特图——直接画 2 万个点没有意义，压成活动条后一眼能看出
        "哪一层在什么时间段活跃"。
        """
        sel = set(kinds) if kinds else None
        bars: List[Tuple[str, float, float, str]] = []
        open_bar: Dict[Tuple[str, str], List[Any]] = {}
        for r in self.records:
            if sel is not None and r.kind not in sel:
                continue
            key = (r.kind, r.entity)
            if key in open_bar:
                open_bar[key][1] = r.t
                open_bar[key][2] = r.detail
            else:
                open_bar[key] = [r.t, r.t, r.detail]
        for (kind, ent), (t0, t1, det) in open_bar.items():
            bars.append((f"{kind}·{ent}" if ent else kind, t0, t1, det))
        bars.sort(key=lambda b: (b[1], b[2]))
        if len(bars) > max_items:                      # 太碎就按时间等分抽样
            step = max(1, len(bars) // max_items)
            bars = bars[::step]
        return bars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "total": int(sum(self.counts.values())),
            "dropped": int(self.dropped),
            "kept": len(self.records),
            "span_s": list(self.span()),
            "records": [r.as_tuple() for r in self.records],
        }


# --------------------------------------------------------------------------- #
# 4. 确定性随机流
# --------------------------------------------------------------------------- #
class RandStream:
    """命名随机流：``sim.rng("tcp:reno")`` 得到可复现的独立流。

    与"全局 random 模块"相比，按名字分流的好处是：改动某一层的事件数量
    不会扰动其他层的随机序列，实验可对比性大幅提升（ns-3 的 RngSeedManager
    用 stream index 解决同一问题，这里用命名流，更直观）。
    """

    def __init__(self, *, seed: int = 0, run: int = 0, stream_seed: Optional[int] = None) -> None:
        # 与经典 RNG 一致的 seed/run 组合语义
        self._base = (0x5DEECE66D if stream_seed is None else int(stream_seed))
        self._seed = int(seed)
        self._run = int(run)
        self._streams: Dict[str, np.random.Generator] = {}
        self._next_stream = 0

    def get(self, name: str) -> np.random.Generator:
        gen = self._streams.get(name)
        if gen is None:
            if name in ("", "auto"):
                idx = self._next_stream
                self._next_stream += 1
            else:                                       # 名字 → 稳定索引
                idx = (abs(hash(name)) % 100_000) if False else _stable_hash(name)
            gen = np.random.default_rng(
                [(self._seed + 1) * 7919 + idx, (self._run + 1) * 104_729 + idx])
            self._streams[name] = gen
        return gen


def _stable_hash(text: str) -> int:
    """跨进程稳定的字符串散列（不用内置 hash —— 它带随机盐，破坏可复现性）。"""
    h = 2166136261
    for ch in text.encode("utf-8"):
        h = (h ^ ch) * 16777619 & 0xFFFFFFFF
    return h


# --------------------------------------------------------------------------- #
# 5. 协程进程
# --------------------------------------------------------------------------- #
class Yield:
    """进程内 ``yield Yield(delay)``：挂起 delay 秒后继续。"""

    __slots__ = ("delay",)

    def __init__(self, delay: float) -> None:
        self.delay = float(delay)


class WaitUntil:
    """进程内 ``yield WaitUntil(t_abs)``：挂起到绝对时刻。"""

    __slots__ = ("t",)

    def __init__(self, t: float) -> None:
        self.t = float(t)


class StopProcess(Exception):
    """在进程内抛出即可静默结束该进程。"""


class Process:
    """生成器协程进程（SimPy 风格，ns-3 中通常用回调链表达）。

    把协议/调度逻辑写成顺序代码::

        def server(sim, name):
            while True:
                yield Yield(1.0)          # 每秒处理一个任务
                print(name, sim.now())

        Process(sim, server(sim, "gs0"))

    与"回调套回调"相比，进程写法在表达"等待-处理-等待"这类状态机时
    可读性高得多，且不影响内核语义（进程只是把回调拆成多个事件）。
    """

    __slots__ = ("sim", "_gen", "finished", "result", "_eid", "name")

    def __init__(self, sim: "Simulator", gen, name: str = "process") -> None:
        if not hasattr(gen, "send"):
            raise TypeError("Process 需要生成器对象")
        self.sim = sim
        self._gen = gen
        self.finished = False
        self.result: Any = None
        self.name = name
        self._eid: Optional[EventId] = None
        self._step(None)

    def _step(self, value: Any) -> None:
        try:
            cmd = self._gen.send(value)
        except StopIteration as exc:
            self.finished = True
            self.result = exc.value
            return
        except StopProcess:
            self.finished = True
            return
        if isinstance(cmd, Yield):
            if cmd.delay <= 0:
                self._eid = self.sim.schedule_now(self._step, None, name=self.name)
            else:
                self._eid = self.sim.schedule(cmd.delay, self._step, None, name=self.name)
        elif isinstance(cmd, WaitUntil):
            self._eid = self.sim.schedule_at(cmd.t, self._step, None, name=self.name)
        elif cmd is None:
            self._eid = self.sim.schedule_now(self._step, None, name=self.name)
        else:
            raise TypeError(f"进程 yield 了不支持的对象：{type(cmd)!r}")

    def cancel(self) -> None:
        if self._eid is not None:
            self._eid.Cancel()
        self.finished = True


# --------------------------------------------------------------------------- #
# 6. 仿真器
# --------------------------------------------------------------------------- #
class Simulator:
    """离散事件仿真器（``ns3::Simulator`` 的对等物，但是**实例化**的）。

    实例化而非全局单例，是为了让"一条 TCP 连接 = 一个独立仿真器"成为可能
    （见 :func:`satnetsim.tcpstack.simulate_flow`）：既能独立跑，也能被
    整体的星座级仿真嵌套调度。
    """

    def __init__(self, *, seed: int = 0, run: int = 0,
                 scheduler: str = "heap",
                 trace_limit: int = 20000,
                 trace_callback: Optional[Callable[[TraceRecord], None]] = None) -> None:
        self._now = 0.0
        self._uid = itertools.count(1)
        self._finished = False
        self._stop_t: Optional[float] = None
        sched = (scheduler or "heap").lower()
        self.scheduler: Scheduler = ListScheduler() if sched == "list" else HeapScheduler()
        self.rng = RandStream(seed=seed, run=run)
        self.trace = Trace(max_records=trace_limit, callback=trace_callback)
        # 统计
        self.n_executed = 0
        self.n_cancelled = 0
        self.queue_series: List[Tuple[float, int]] = []   # (t, 队列深度) 采样
        self.queue_limit = 20000                          # 采样点上限（满则等比抽稀）
        self._q_stride = 1
        self._q_count = 0

    # ---------------- 时间 ---------------- #
    def now(self) -> float:
        return self._now

    def peek_time(self) -> Optional[float]:
        ev = self.scheduler.PeekNext()
        return ev.t if ev is not None else None

    def n_pending(self) -> int:
        return self.scheduler.depth

    # ---------------- 调度 ---------------- #
    def schedule(self, delay: float, fn: Callable[..., Any], *args,
                 priority: int = 0, name: str = "", **kwargs) -> EventId:
        """相对延时调度（``ns3::Simulator::Schedule``）。``delay`` 以秒计。"""
        return self.schedule_at(self._now + max(float(delay), 0.0), fn, *args,
                                priority=priority, name=name, **kwargs)

    def schedule_at(self, t_abs: float, fn: Callable[..., Any], *args,
                    priority: int = 0, name: str = "", **kwargs) -> EventId:
        """绝对时刻调度。``t_abs`` 早于当前时刻时按"立即"处理（不回溯时间）。"""
        t = max(float(t_abs), self._now)
        uid = next(self._uid)
        label = name or getattr(fn, "__name__", "event")
        eid = EventId(uid, t, priority, label)
        ev = Event(t, int(priority), uid, fn, args, kwargs, eid, label)
        self.scheduler.Insert(ev)
        return eid

    def schedule_now(self, fn: Callable[..., Any], *args,
                     priority: int = 0, name: str = "", **kwargs) -> EventId:
        """在当前时刻立刻入队（同刻按优先级/插入序执行）。"""
        return self.schedule_at(self._now, fn, *args, priority=priority,
                               name=name, **kwargs)

    def cancel(self, eid: Optional[EventId]) -> None:
        if eid is not None:
            eid.Cancel()

    # ---------------- 主循环 ---------------- #
    def run(self, until: Optional[float] = None) -> float:
        """执行事件直到队列空 / 到达 ``until`` / 被 :meth:`stop` 打断。

        返回值是仿真结束时刻。**可反复调用**：``run(10)`` 之后 ``run(20)``
        会从 10 s 继续（与 ns-3 相同）。
        """
        if self._stop_t is not None:
            until = self._stop_t if until is None else min(until, self._stop_t)
            self._stop_t = None
        self._finished = False
        last_t = self._now
        while True:
            ev = self.scheduler.RemoveNext()
            if ev is None:
                break
            if self._finished:                       # stop() 之后不再推进
                break
            if until is not None and ev.t > until:
                # 事件超出本次时限：塞回队列，等下次 run 继续
                self.scheduler.Insert(ev)
                break
            if ev.eid.cancelled:
                self.n_cancelled += 1
                continue
            if ev.t != last_t:
                if self.queue_series and self.queue_series[-1][1] == 0:
                    self.queue_series.pop()
                self._q_count += 1
                if self._q_count % self._q_stride == 0:
                    self.queue_series.append((float(ev.t), self.scheduler.depth + 1))
                    if len(self.queue_series) >= self.queue_limit:
                        self.queue_series = self.queue_series[::2]
                        self._q_stride *= 2
                last_t = ev.t
            self._now = ev.t
            self.n_executed += 1
            ev.run()
        return self._now

    def stop(self, t: Optional[float] = None) -> None:
        """停止推进；``t`` 给出则下次 ``run()`` 最多跑到 ``t``（``ns3::Simulator::Stop(time)``）。"""
        if t is None:
            self._finished = True
        else:
            self._stop_t = float(t)

    def is_finished(self) -> bool:
        return self._finished

    def destroy(self) -> None:
        """清空队列与追踪（``ns3::Simulator::Destroy``）。"""
        self.scheduler = HeapScheduler() if isinstance(self.scheduler, HeapScheduler) else ListScheduler()
        self._now = 0.0
        self._finished = False
        self._stop_t = None
        self.trace.records.clear()
        self.queue_series.clear()
        self._q_stride = 1
        self._q_count = 0
        self.n_executed = 0
        self.n_cancelled = 0

    def reset_clock(self) -> None:
        self._now = 0.0

    # ---------------- 便捷：观察 ---------------- #
    def seconds_left(self) -> int:
        """剩余待执行事件数（用于进度条）。"""
        return self.n_pending()

    def summary(self) -> Dict[str, Any]:
        return {
            "now_s": self._now,
            "executed": self.n_executed,
            "cancelled": self.n_cancelled,
            "pending": self.n_pending(),
            "scheduler": type(self.scheduler).__name__,
            "trace_total": int(sum(self.trace.counts.values())),
            "trace_kinds": len(self.trace.counts),
        }


# --------------------------------------------------------------------------- #
# 7. 自检
# --------------------------------------------------------------------------- #
def des_self_test(verbose: bool = False) -> Tuple[bool, List[str]]:
    """内核语义自检：定序 / 取消 / 分段运行 / 可复现 / 两种调度器等价。

    返回 ``(全部通过?, 日志行)``。``tests/verify_des.py`` 直接消费本函数。
    """
    log: List[str] = []
    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        log.append(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  —— {extra}" if extra else ""))

    # -- 1) 定序：时间优先、同刻优先级优先、再同刻 FIFO --
    order: List[Tuple[str, float]] = []
    sim = Simulator(seed=1)
    sim.schedule(1.0, lambda: order.append(("a", sim.now())))
    sim.schedule(0.5, lambda: order.append(("b", sim.now())))
    sim.schedule(1.0, lambda: order.append(("c", sim.now())))
    sim.schedule(1.0, lambda: order.append(("hi", sim.now())), priority=5)
    sim.run()
    names = [n for n, _ in order]
    check("时间优先（0.5 早于 1.0）", names[0] == "b", str(names))
    check("同刻高优先级先执行", names[1] == "hi", str(names))
    check("同刻同优先级保持 FIFO", names[2:] == ["a", "c"], str(names))
    check("仿真时钟停在最后一个事件时刻", abs(sim.now() - 1.0) < 1e-12, f"{sim.now():.6f}")

    # -- 2) 取消语义 --
    hits: List[str] = []
    sim2 = Simulator(seed=1)
    sim2.schedule(1.0, lambda: hits.append("keep"))
    eid = sim2.schedule(1.0, lambda: hits.append("drop"))
    sim2.schedule(0.5, lambda: sim2.cancel(eid))
    sim2.run()
    check("取消的事件不执行", hits == ["keep"], str(hits))
    check("取消计数被记录", sim2.n_cancelled == 1, f"n_cancelled={sim2.n_cancelled}")

    # -- 3) 分段运行（run 可续跑）--
    marks: List[float] = []
    sim3 = Simulator(seed=1)
    for t in (1.0, 2.0, 3.0):
        sim3.schedule(t, lambda: marks.append(sim3.now()))
    sim3.run(until=1.5)
    check("run(until) 到点即停", marks == [1.0] and abs(sim3.now() - 1.0) < 1e-12, str(marks))
    sim3.run()
    check("续跑执行剩余事件", marks == [1.0, 2.0, 3.0], str(marks))

    # -- 4) stop(t) 语义 --
    sim4 = Simulator(seed=1)
    cnt = {"n": 0}
    for t in (1.0, 2.0, 3.0):
        sim4.schedule(t, lambda: cnt.__setitem__("n", cnt["n"] + 1))
    sim4.stop(2.5)
    sim4.run()
    check("stop(t) 在 t 之前停止", cnt["n"] == 2, f"executed={cnt['n']}")

    # -- 5) 可复现性：同种子两次运行的事件序列完全一致 --
    def build(seed: int) -> List[Tuple[float, str]]:
        s = Simulator(seed=seed)
        seq: List[Tuple[float, str]] = []
        def tick(tag: str, gen: int):
            seq.append((round(s.now(), 9), tag))
            if gen < 5:                          # 有限链：每条只繁衍 5 代
                s.schedule(0.7, tick, f"{tag}*", gen + 1)
        s.schedule(0.3, tick, "x", 0)
        s.schedule(0.3, tick, "y", 0)
        s.schedule(0.5, tick, "z", 0)
        s.run()
        return seq
    check("同种子事件序列可复现", build(7) == build(7))
    check("不同种子不改变确定性时序", [t for t, _ in build(7)] == [t for t, _ in build(9)])

    # -- 6) 两种调度器语义等价 --
    def with_sched(kind: str) -> List[Tuple[float, str]]:
        s = Simulator(seed=3, scheduler=kind)
        seq: List[Tuple[float, str]] = []
        for i, d in enumerate([1.0, 1.0, 0.5, 0.5, 2.0, 0.25]):
            s.schedule(d, lambda i=i: seq.append((s.now(), f"e{i}")))
        s.run()
        return seq
    check("HeapScheduler 与 ListScheduler 输出一致",
          with_sched("heap") == with_sched("list"), str(with_sched("heap")))

    # -- 7) 协程进程 --
    logp: List[Tuple[float, str]] = []
    sim7 = Simulator(seed=1)

    def worker(tag: str):
        for _ in range(3):
            yield Yield(1.0)
            logp.append((sim7.now(), tag))
        yield WaitUntil(10.0)
        logp.append((sim7.now(), tag + ":final"))

    Process(sim7, worker("w1"))
    Process(sim7, worker("w2"))
    sim7.run()
    w1 = [t for t, tag in logp if tag == "w1" or tag == "w1:final"]
    check("进程 Yield 定时正确", w1 == [1.0, 2.0, 3.0, 10.0], str(w1))
    check("多进程交错推进", len(logp) == 8, str(len(logp)))

    # -- 8) 随机流确定性 --
    a = Simulator(seed=42).rng.get("tcp").random(5)
    b = Simulator(seed=42).rng.get("tcp").random(5)
    c = Simulator(seed=43).rng.get("tcp").random(5)
    check("命名随机流可复现", np.allclose(a, b))
    check("不同种子产生不同序列", not np.allclose(a, c))
    d = Simulator(seed=42).rng.get("other").random(5)
    check("不同名字的流互相独立", not np.allclose(a, d))

    # -- 9) 追踪 --
    sim9 = Simulator(seed=1, trace_limit=100)
    for i in range(10):
        ev = sim9.schedule(i * 0.1, lambda: None)
        sim9.trace.add(ev.t, "demo", "obj", f"#{i}")
    sim9.run()
    check("追踪计数覆盖全部事件", sim9.trace.counts.get("demo") == 10,
          str(sim9.trace.counts))
    bars = sim9.trace.timeline()
    check("追踪可压成时间线条", bool(bars), f"{len(bars)} 条")
    check("队列深度序列已采样", len(sim9.queue_series) > 0, f"{len(sim9.queue_series)} 点")

    # -- 10) 环形缓冲不丢计数 --
    simA = Simulator(seed=1, trace_limit=5)
    for i in range(20):
        simA.trace.add(i * 0.1, "k", "e", "")
    check("环形缓冲丢弃旧记录但保留计数",
          len(simA.trace.records) == 5 and simA.trace.counts["k"] == 20,
          f"kept={len(simA.trace.records)} total={simA.trace.counts['k']}")

    if verbose:
        for line in log:
            print(line)
    return ok, log


if __name__ == "__main__":  # pragma: no cover
    ok_, lines = des_self_test(verbose=True)
    print("\n内核自检:", "全部通过" if ok_ else "存在失败项")
    raise SystemExit(0 if ok_ else 1)

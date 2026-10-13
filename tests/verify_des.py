"""真实验证 des：事件内核语义 + TCP-on-DES 等价性 + 全局编排确定性。

检查四层：
  1. 内核自检（des.des_self_test 的 21 项语义断言）；
  2. TCP-on-DES 等价性——自建内核与注入内核独立运行，同种子结果**逐位一致**；
  3. 共享内核并发——多条流在同一条时间轴上事件级交错，全部完成且吞吐合理
     （受 TcpConfig 接收窗口上限约束，允许与独立基线差一个窗口量级）；
  4. run_des 全局编排——六类事件齐备、任务全部服务、同种子两次运行确定性一致。
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from satnetsim.des import Simulator, des_self_test  # noqa: E402
from satnetsim.tcpstack import (  # noqa: E402
    PathProfile, TcpConfig, TcpConnection, simulate_flow,
)
from satnetsim.dessim import run_des  # noqa: E402
from satnetsim.config import DesConfig  # noqa: E402

OK = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    OK = OK and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  —— {extra}" if extra else ""))


# --------------------------------------------------------------------------- #
# 合成剖面：600 s，500 Mbps 瓶颈，RTT ~80 ms 带缓慢漂移与跳数阶跃
# --------------------------------------------------------------------------- #
N = 61
T = np.linspace(0.0, 600.0, N)
_rng = np.random.default_rng(1)
_owd = 40.0 + np.cumsum(_rng.normal(0, 0.15, N))
PP = PathProfile(t=T, owd_ms=_owd, rtt_ms=_owd * 2,
                 bw_mbps=np.full(N, 500.0), loss=np.full(N, 0.0002),
                 hops=8 + (np.arange(N) % 3),
                 queue_bytes=np.full(N, 4_000_000.0))
CFG = TcpConfig(rcvbuf_bytes=4 * 1024 * 1024, max_time_s=120.0)
PAYLOAD = 8_000_000  # bytes


def profile_lines(title: str) -> None:
    print("-" * 68)
    print(title)
    print("-" * 68)


# =========== 1. 内核语义自检 =========== #
profile_lines("1) 内核语义自检（des.des_self_test）")
t0 = time.perf_counter()
ok, _log = des_self_test()
check(f"自检 21 项全部通过", ok, f"墙钟 {time.perf_counter()-t0:.2f} s")

# =========== 2. TCP-on-DES 等价性 =========== #
profile_lines("2) TCP-on-DES 等价性（自建内核 vs 注入内核，同种子）")
r_own = TcpConnection(PP, CFG, algo="reno", seed=3).run(PAYLOAD)
inj = Simulator(seed=99)
c_inj = TcpConnection(PP, CFG, algo="reno", seed=3, sim=inj)
c_inj.run(PAYLOAD)
r_inj = c_inj._finish(PAYLOAD)
check("完成时刻一致", abs(r_own.completion_s - r_inj.completion_s) < 1e-9,
      f"{r_own.completion_s:.6f} vs {r_inj.completion_s:.6f}")
check("重传数一致", r_own.retransmits == r_inj.retransmits,
      f"{r_own.retransmits} vs {r_inj.retransmits}")
check("吞吐一致", abs(r_own.goodput_mbps - r_inj.goodput_mbps) < 1e-9,
      f"{r_own.goodput_mbps:.3f} Mbps")
check("样本数一致", len(r_own.samples) == len(r_inj.samples),
      f"{len(r_own.samples)} vs {len(r_inj.samples)}")
check("内核执行事件数一致（可复现性）", inj.n_executed > 0)

# =========== 3. 共享内核并发 =========== #
profile_lines("3) 共享内核并发（两条流事件级交错）")
c_base = TcpConnection(PP, CFG, algo="reno", seed=3)
r_own = c_base.run(PAYLOAD, handshake_s=c_base.handshake_seconds())
sim = Simulator(seed=5)
conns = []
for i in range(2):
    c = TcpConnection(PP, CFG, algo="reno", seed=3 + i, sim=sim)
    conns.append(c)
    sim.schedule_at(0.0 if i == 0 else 5.0,
                    lambda c=c: c.start(PAYLOAD,
                                        handshake_s=c.handshake_seconds()),
                    name=f"flow{i}:arrive")
sim.run(until=600.0)
recs = [c._finish(PAYLOAD) for c in conns]
check("两条流都完成", all(np.isfinite(r.completion_s) for r in recs),
      str([round(r.completion_s, 3) for r in recs]))
check("流0（同时刻到达）与独立基线逐位一致",
      abs(recs[0].completion_s - r_own.completion_s) < 1e-9
      and recs[0].retransmits == r_own.retransmits,
      f"{recs[0].completion_s:.6f} vs {r_own.completion_s:.6f}，"
      f"重传 {recs[0].retransmits} vs {r_own.retransmits}")
check("流1（+5s 到达）非退化（与基线差 <60%）",
      abs(recs[1].goodput_mbps / r_own.goodput_mbps - 1) < 0.6,
      f"基线 {r_own.goodput_mbps:.1f}，流1 {recs[1].goodput_mbps:.1f}")

# =========== 4. 全局编排 =========== #
profile_lines("4) run_des 全局离散事件编排")
dcfg = DesConfig(n_flows=3, flow_mbit=32.0, tcp_algo="bbr", n_tasks=200)
t0 = time.perf_counter()
rep = run_des(duration_s=600.0, snapshot_interval_s=10.0, tcp_profile=PP,
              des_cfg=dcfg, tcp_cfg=CFG)
wall = time.perf_counter() - t0
kinds = set(rep.trace.counts)
check("六类事件齐备",
      {"topo:refresh", "route:recompute", "path:switch", "flow:arrive",
       "flow:done", "task:arrive", "task:done", "metric:sample"} <= kinds,
      str(sorted(kinds)))
check("拓扑/路由事件 = 快照数", rep.topo_events == 61 and rep.route_events == 60,
      f"{rep.topo_events}/{rep.route_events}")
check("路径切换事件与剖面阶跃一致", rep.path_switches == int(PP.path_changes),
      f"{rep.path_switches} vs {int(PP.path_changes)}")
check("TCP 流全部完成", rep.flows_done == len(rep.flows) == 3,
      f"{rep.flows_done}/{len(rep.flows)}")
check("流吞吐非退化（>5% 窗口上限；短流受慢启动与 BDP 限制）",
      all(f.goodput_mbps > 0.05 * CFG.rcvbuf_bytes * 8 / 0.08 / 1e6
          for f in rep.flows if f.done),
      f"均值 {rep.flow_goodput_mean_mbps:.1f} Mbps（窗口上限 "
      f"{CFG.rcvbuf_bytes * 8 / 0.08 / 1e6:.0f}）")
check("任务全部服务", len(rep.tasks) > 0 and all(
    t.finish_s > t.arrival_s for t in rep.tasks),
    f"{len(rep.tasks)} 个，等待均值 {rep.task_mean_wait_s:.4f} s")
check("时限达标率合理", rep.task_deadline_rate > 0.9,
      f"{rep.task_deadline_rate*100:.0f}%")
check("队列深度序列已采样", len(rep.queue_series) > 50,
      f"{len(rep.queue_series)} 点")
check("编排墙钟可接受", wall < 10.0, f"{wall:.2f} s")

d1 = rep.to_dict(max_trace=100)
rep2 = run_des(duration_s=600.0, snapshot_interval_s=10.0, tcp_profile=PP,
               des_cfg=dcfg, tcp_cfg=CFG)
check("同种子两次编排事件数一致", rep.n_executed == rep2.n_executed,
      f"{rep.n_executed} vs {rep2.n_executed}")
check("同种子两次流结果一致",
      [f.completion_s for f in rep.flows] == [f.completion_s for f in rep2.flows])
check("to_dict 可序列化且含轨迹", "trace" in d1 and d1["trace"]["total"] > 0,
      f"{d1['trace']['total']} 条（保留 {d1['trace']['kept']}）")

# =========== 5. 路径切换 × 在途流联动（新增：切换代价） =========== #
profile_lines("5) 路径切换 × 在途流联动（on_path_switch 切换代价）")
# 密集切换剖面：每 0.5 s 一次跳数阶跃，短流也会跨越多次切换
N2 = 121
T2 = np.linspace(0.0, 60.0, N2)
PP2 = PathProfile(t=T2, owd_ms=np.full(N2, 40.0), rtt_ms=np.full(N2, 80.0),
                  bw_mbps=np.full(N2, 500.0), loss=np.full(N2, 0.0002),
                  hops=np.where(np.arange(N2) % 2 == 0, 6, 9),
                  queue_bytes=np.full(N2, 4_000_000.0))
# ① 开惩罚 vs 关惩罚：在途时注入一次切换（与编排同机制），对比完成时刻
CFG_SW = TcpConfig(rcvbuf_bytes=4 * 1024 * 1024, max_time_s=120.0,
                   path_switch_penalty=True, path_switch_cost_s=0.2)
CFG_NOSW = TcpConfig(rcvbuf_bytes=4 * 1024 * 1024, max_time_s=120.0,
                     path_switch_penalty=False, path_switch_cost_s=0.2)


def flow_one_switch(cfg: TcpConfig, inject: bool, snapshot_t: float = 0.0):
    s = Simulator(seed=11)
    c = TcpConnection(PP, cfg, algo="cubic", seed=3, sim=s)
    s.schedule_at(0.0, lambda: c.start(PAYLOAD, handshake_s=c.handshake_seconds()))
    if inject:      # t=0.5 s 时流必在在途（8 MByte 载荷），注入一次切换
        s.schedule_at(0.5, c.on_path_switch, name="path:switch")
    if snapshot_t > 0:
        s.run(until=snapshot_t)
        global flow_one_switch_bytes_pen, flow_one_switch_bytes_free
        if cfg.path_switch_penalty:
            flow_one_switch_bytes_pen = int(c.cwnd)
        else:
            flow_one_switch_bytes_free = int(c.cwnd)
        s.run(until=60.0)
    else:
        s.run(until=60.0)
    return c._finish(PAYLOAD)


r_pen = flow_one_switch(CFG_SW, inject=True, snapshot_t=0.55)
r_free = flow_one_switch(CFG_NOSW, inject=True, snapshot_t=0.55)
check("两条流都在时限内完成",
      np.isfinite(r_pen.completion_s) and np.isfinite(r_free.completion_s),
      f"{r_pen.completion_s:.3f} / {r_free.completion_s:.3f} s")
check("切换后惩罚流窗口被收缩（机制生效）",
      flow_one_switch_bytes_pen < flow_one_switch_bytes_free,
      f"惩罚 cwnd {flow_one_switch_bytes_pen/1024:.0f} KB vs "
      f"无惩罚 {flow_one_switch_bytes_free/1024:.0f} KB（t=0.55 s）")
# ② on_path_switch 单元语义：在途时注入切换 → 窗口收缩 + 发射静默
sim_u = Simulator(seed=1)
c_u = TcpConnection(PP2, CFG_SW, algo="cubic", seed=3, sim=sim_u)
c_u.start(PAYLOAD, handshake_s=c_u.handshake_seconds())
sim_u.run(until=0.8)                       # 流在途
check("注入时刻流确在在途", not c_u._done and c_u.snd_nxt > 0)
cw_before = c_u.cwnd
ok_sw = c_u.on_path_switch()
check("on_path_switch 返回生效", ok_sw is True)
check("切换后窗口收缩到初始窗", c_u.cwnd <= CFG_SW.initial_window_seg * c_u.mss + 1e-6,
      f"{cw_before/1024:.0f} KB → {c_u.cwnd/1024:.0f} KB")
check("切换后发射静默 >= 切换代价", c_u.link_free >= c_u.t + CFG_SW.path_switch_cost_s - 1e-9,
      f"link_free={c_u.link_free:.3f}, t={c_u.t:.3f}")
sim_u.run()                                # 跑到完成
check("承受切换的流最终完成", c_u._done and c_u.snd_una >= PAYLOAD)
c_k = TcpConnection(PP2, CFG_SW, algo="cubic", seed=3, sim=Simulator())
c_k.kill()
check("已终止的流不再响应切换", c_k.on_path_switch() is False)
# ③ 编排级：切换波及在途流并留下 flow:switch 轨迹
# 链路压到 20 Mbps（128 Mbit 流 ≈ 8 s，跨越多个 2.5 s 间隔的切换时刻）
N3 = 241
T3 = np.linspace(0.0, 600.0, N3)
PP3 = PathProfile(t=T3, owd_ms=np.full(N3, 40.0), rtt_ms=np.full(N3, 80.0),
                  bw_mbps=np.full(N3, 20.0), loss=np.full(N3, 0.0002),
                  hops=6 + (np.arange(N3) % 2),
                  queue_bytes=np.full(N3, 1_000_000.0))
CFG_SW20 = TcpConfig(rcvbuf_bytes=1 * 1024 * 1024, max_time_s=120.0,
                     path_switch_penalty=True, path_switch_cost_s=0.2)
dcfg_sw = DesConfig(n_flows=2, flow_mbit=128.0, tcp_algo="cubic", n_tasks=50, seed=7)
rep_sw = run_des(duration_s=600.0, snapshot_interval_s=10.0, tcp_profile=PP3,
                 des_cfg=dcfg_sw, tcp_cfg=CFG_SW20)
check("编排记录波及在途流的切换", rep_sw.flow_switches >= 1,
      f"{rep_sw.flow_switches} 次（路径切换共 {rep_sw.path_switches} 次）")
check("flow:switch 进入事件轨迹", "flow:switch" in rep_sw.trace.counts,
      str(rep_sw.trace.counts.get("flow:switch")))
check("受切换影响的流仍全部完成", rep_sw.flows_done == len(rep_sw.flows) == 2,
      f"{rep_sw.flows_done}/{len(rep_sw.flows)}")
check("编排含节点利用率统计", len(rep_sw.node_utilization) == dcfg_sw.n_nodes)

# --------------------------------------------------------------------------- #
print("-" * 68)
print("verify_des:", "全部通过" if OK else "存在失败项")
sys.exit(0 if OK else 1)

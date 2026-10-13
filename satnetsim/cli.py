"""命令行批量仿真入口（无 GUI，便于脚本化 / 扫参 / CI）。

用法::

    python -m satnetsim run starlink-lite --duration 600 --tasks 300
    python -m satnetsim run starlink-lite --real-all
    python -m satnetsim run starlink-lite --real-transfer --real-payload-mbit 32
    python app.py --preset starlink-shell1 --out report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .config import PRESETS, preset
from .engine import Simulation
from .viz import text_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="satnetsim run",
                                description="以指定预设跑一次完整仿真并输出摘要")
    p.add_argument("preset", nargs="?", default="starlink-lite",
                   help=f"预设名，可选 {sorted(PRESETS)}")
    p.add_argument("--duration", type=float, help="仿真时长 (s)")
    p.add_argument("--snapshot", type=float, help="快照间隔 (s)")
    p.add_argument("--tasks", type=int, help="星载计算任务数")
    p.add_argument("--isl-bandwidth-gbps", type=float, help="ISL 带宽")
    p.add_argument("--gsl-bandwidth-mbps", type=float, help="星地带宽")
    p.add_argument("--sat-tops", type=float, help="单星算力 (TOPS)")
    p.add_argument("--real-transfer", action="store_true",
                   help="启用真实数据传输评测（内核 TCP + 用户态包级可靠传输）")
    p.add_argument("--real-compute", action="store_true",
                   help="启用真实计算任务评测（真跑 ResNet/FFT/GEMM/SHA-256 等算子）")
    p.add_argument("--real-all", action="store_true",
                   help="等价于同时打开 --real-transfer 与 --real-compute")
    p.add_argument("--real-payload-mbit", type=float,
                   help="真实 TCP 评测载荷 (Mbit)")
    p.add_argument("--real-tasks", type=int, help="真实负载任务流的任务数")
    p.add_argument("--real-repeats", type=int, help="每个真实负载的标定重复次数")
    p.add_argument("--real-algos", type=str,
                   help="真实包级传输对比的拥塞控制算法，逗号分隔（默认 reno,cubic,hybla）")
    p.add_argument("--no-analytic-compare", action="store_true",
                   help="真实传输评测中不做解析模型对照")
    p.add_argument("--trans-engine", type=str,
                   help="传输仿真引擎：real（真实包级 TCP 状态机）| analytic（解析轮次模型）| both")
    p.add_argument("--tcp-mbit", type=float,
                   help="真实 TCP 业务载荷 (Mbit)")
    p.add_argument("--tcp-rcvbuf-kb", type=float,
                   help="真实 TCP 接收窗口上限 (KB)")
    p.add_argument("--tcp-algos", type=str,
                   help="真实 TCP 对比的拥塞控制算法，逗号分隔（默认 reno,cubic,bbr,hybla）")
    p.add_argument("--no-route-sweep", action="store_true",
                   help="不做「路由 × 真实传输」联合评测")
    p.add_argument("--real-data", action="store_true",
                   help="启用真实数据流仿真（真字节逐包多跳转发）")
    p.add_argument("--no-real-data", action="store_true",
                   help="关闭真实数据流仿真")
    p.add_argument("--rdata-kind", type=str,
                   help="真实载荷类型：image | text | telemetry | file")
    p.add_argument("--rdata-mbit", type=float, help="单轮真实载荷大小 (Mbit)")
    p.add_argument("--rdata-flows", type=int, help="并发业务流数")
    p.add_argument("--rdata-window", type=float,
                   help="数据仿真窗口 (s)，0 表示跟随仿真时长")
    p.add_argument("--rdata-budget", type=int,
                   help="全窗口分组总数上限（控制规模与回放数据量）")
    p.add_argument("--rdata-file", type=str,
                   help="payload_kind=file 时的本地文件路径")
    p.add_argument("--des", action="store_true",
                   help="启用全局离散事件内核编排（NS-3 式统一事件时间轴）")
    p.add_argument("--no-des-trace", action="store_true",
                   help="不记录事件轨迹（更快；默认记录）")
    p.add_argument("--des-flows", type=int, help="事件时间轴上的 TCP 流条数")
    p.add_argument("--des-tasks", type=int, help="事件时间轴上的计算任务到达数")
    p.add_argument("--des-algo", type=str, help="事件流使用的拥塞控制算法")
    p.add_argument("--des-mbit", type=float, help="事件流单流载荷 (Mbit)")
    p.add_argument("--no-switch-penalty", action="store_true",
                   help="真实 TCP 不承受路径切换代价（默认切换时窗口收缩+发射静默）")
    p.add_argument("--live-html", type=str, default=None,
                   help="把真实数据流导出为自包含单文件 HTML 动画")
    p.add_argument("--duration-sweep", type=str, default=None,
                   help="时长扫描，逗号分隔（如 300,600,1200）")
    p.add_argument("--out", type=str, default=None, help="把摘要写到文件")
    p.add_argument("--json", type=str, default=None, help="把结构化结果写到 JSON")
    p.add_argument("--quiet", action="store_true", help="不打印进度")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preset not in PRESETS:
        print(f"未知预设 {args.preset!r}，可选：{sorted(PRESETS)}", file=sys.stderr)
        return 2

    cfg = preset(args.preset)
    if args.duration is not None:
        cfg.duration_s = args.duration
    if args.snapshot is not None:
        cfg.snapshot_interval_s = args.snapshot
    if args.tasks is not None:
        cfg.compute.task_count = args.tasks
    if args.isl_bandwidth_gbps is not None:
        cfg.link.isl_bandwidth_mbps = args.isl_bandwidth_gbps * 1000.0
    if args.gsl_bandwidth_mbps is not None:
        cfg.link.gsl_bandwidth_mbps = args.gsl_bandwidth_mbps
    if args.sat_tops is not None:
        cfg.compute.sat_capacity_tops = args.sat_tops

    # ---- 真实评测开关 ----
    if args.real_all:
        args.real_transfer = True
        args.real_compute = True
    if args.real_transfer or args.real_compute:
        cfg.real_eval.enabled = True
        cfg.real_eval.transfer = bool(args.real_transfer)
        cfg.real_eval.compute = bool(args.real_compute)
    if args.real_payload_mbit is not None:
        cfg.real_eval.payload_mbit = args.real_payload_mbit
    if args.real_tasks is not None:
        cfg.real_eval.real_task_count = args.real_tasks
    if args.real_repeats is not None:
        cfg.real_eval.repeats = args.real_repeats
    if args.real_algos:
        cfg.real_eval.packet_algos = [
            a.strip().lower() for a in args.real_algos.split(",") if a.strip()]
    if args.no_analytic_compare:
        cfg.real_eval.compare_analytic = False

    # ---- 真实数据流开关 ----
    if args.real_data:
        cfg.real_data.enabled = True
    if args.no_real_data:
        cfg.real_data.enabled = False
    if args.trans_engine:
        cfg.transport.engine = args.trans_engine.strip().lower()
    if args.tcp_mbit is not None:
        cfg.transport.app_payload_mbit = args.tcp_mbit
    if args.tcp_rcvbuf_kb is not None:
        cfg.transport.rcvbuf_bytes = int(args.tcp_rcvbuf_kb * 1024)
    if args.tcp_algos:
        cfg.transport.tcp_algos = [
            a.strip().lower() for a in args.tcp_algos.split(",") if a.strip()]
    if args.no_route_sweep:
        cfg.transport.route_sweep = False
    if args.rdata_kind:
        cfg.real_data.payload_kind = args.rdata_kind
    if args.rdata_mbit is not None:
        cfg.real_data.payload_mbit = args.rdata_mbit
    if args.rdata_flows is not None:
        cfg.real_data.n_flows = args.rdata_flows
    if args.rdata_budget is not None:
        cfg.real_data.max_packets = int(args.rdata_budget)
    if args.rdata_window is not None:
        cfg.real_data.duration_s = args.rdata_window
    if args.rdata_file:
        cfg.real_data.payload_kind = "file"
        cfg.real_data.payload_file = args.rdata_file

    # ---- ⑫ 全局离散事件编排 ----
    if args.des:
        cfg.des.enabled = True
    if args.no_des_trace:
        cfg.des.trace = False
    if args.des_flows is not None:
        cfg.des.n_flows = int(args.des_flows)
    if args.des_tasks is not None:
        cfg.des.n_tasks = int(args.des_tasks)
    if args.des_algo:
        cfg.des.tcp_algo = args.des_algo.strip().lower()
    if args.des_mbit is not None:
        cfg.des.flow_mbit = float(args.des_mbit)
    if args.no_switch_penalty:
        cfg.transport.path_switch_penalty = False

    durations = (
        [float(x) for x in args.duration_sweep.split(",")]
        if args.duration_sweep
        else [cfg.duration_s]
    )

    chunks: List[str] = []
    structured = []
    results = []
    for dur in durations:
        cfg.duration_s = dur
        progress = None if args.quiet else (
            lambda m, f: sys.stdout.write(f"\r  [{f*100:5.1f}%] {m:<28s}")
        )
        res = Simulation(cfg).run(progress=progress)
        results.append(res)
        if progress:
            sys.stdout.write("\r" + " " * 60 + "\r")
        header = f"\n########## 仿真时长 {dur:.0f} s ##########\n"
        chunks.append(header + text_report(res))
        if res.des is not None:
            d = res.des
            lines = ["", "—— ⑫ 全局离散事件内核（统一事件时间轴） ——"]
            lines += [f"  {k:<12} {v}" for k, v in d.headline()]
            if d.notes:
                lines += [f"  [备注] {n}" for n in d.notes]
            chunks.append("\n".join(lines))
        rt = res.real_transfer
        rc = res.real_compute
        structured.append({
            "duration_s": dur,
            "timings": res.timings,
            "topology": res.topo_stats,
            "routes": {
                k: {
                    "label": v.label,
                    "mean_delay_ms": v.mean_delay_ms,
                    "p95_delay_ms": v.p95_delay_ms,
                    "mean_hops": v.mean_hops,
                    "switch_count": v.switch_count,
                    "success_rate": v.success_rate,
                }
                for k, v in res.routes.items()
            },
            "transport": [
                {
                    "protocol": f.protocol,
                    "label": f.label,
                    "completion_ms": f.completion_s * 1e3,
                    "goodput_mbps": f.goodput_mbps,
                    "efficiency": f.efficiency,
                }
                for f in res.transport
            ],
            "compute": {k: v.summary_row() for k, v in res.compute.items()},
            "real_transfer": None if rt is None else {
                "profile": rt.profile.describe(),
                "measured_rtt_ms": rt.measured_rtt_ms,
                "theoretical_rtt_ms": rt.profile.rtt_ms,
                "payload_mbit": rt.payload_mbit,
                "packet_profile": (rt.packet_profile.describe()
                                   if rt.packet_profile else None),
                "packet_payload_mbit": rt.packet_payload_mbit,
                "harness_cap_mbps": rt.harness_cap_mbps,
                "wall_s": rt.wall_s,
                "trials": [
                    {
                        "label": t.label, "family": fam, "proto": t.proto,
                        "completion_s": t.completion_s,
                        "goodput_mbps": t.goodput_mbps,
                        "bytes_received": t.bytes_received,
                        "sha_match": t.sha_match,
                        "retransmits": t.retransmits,
                    }
                    for fam, group in (("tcp", rt.tcp), ("packet", rt.packet))
                    for t in group
                ],
                "analytic": rt.analytic,
                "analytic_goodput": rt.analytic_goodput,
            },
            "real_compute": None if rc is None else {
                "device": rc.calibration.device,
                "cpu_count": rc.calibration.cpu_count,
                "peak_gflops": rc.calibration.peak_gflops(),
                "task_count": rc.task_count,
                "total_gflops": rc.total_gflops,
                "total_input_mbit": rc.total_input_mbit,
                "total_raw_mbit": rc.total_raw_mbit,
                "ground_total_s": rc.ground_total_s,
                "on_orbit_total_s": rc.on_orbit_total_s,
                "speedup": rc.speedup,
                "workload_mix": dict(rc.workload_mix),
                "wall_s": rc.wall_s,
                "workloads": [
                    {
                        "key": r.key, "label": r.label, "category": r.category,
                        "elapsed_ms": r.elapsed_s * 1e3,
                        "gflops_effective": r.gflops_effective,
                        "payload_bytes": r.payload_bytes, "ok": r.ok,
                    }
                    for r in rc.calibration.runs
                ],
            },
            "warnings": res.warnings,
        })
        if res.real_data is not None:
            structured[-1]["real_data"] = res.real_data.to_dict(
                trace_limit=0 if not args.live_html else cfg.real_data.trace_limit)
        if res.des is not None:
            structured[-1]["des"] = res.des.to_dict(max_trace=1000)

    if args.live_html and results and results[-1].real_data is not None:
        from .liveviz import LiveScene, export_html

        last = results[-1]
        scene = LiveScene.build(last.constellation, last.snapshots,
                                last.config.ground_stations,
                                last.real_data, last.constellation.times_s)
        p = export_html(scene, args.live_html)
        print(f"实时回放 HTML 已导出：{p}")

    text = "\n".join(chunks)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"\n摘要已写入 {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(structured, fh, ensure_ascii=False, indent=2)
        print(f"结构化结果已写入 {args.json}")
    return 0

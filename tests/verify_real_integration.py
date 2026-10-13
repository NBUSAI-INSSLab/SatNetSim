"""端到端集成验证：真实评测走完整引擎流水线（第 ⑦ 步）。

验证目标
--------
1. `cfg.real_eval.enabled=False` 时引擎不进入真实评测（零额外耗时）；
2. `enabled=True` 时引擎跑出 real_transfer / real_compute，字段自洽；
3. 真实评测不影响前七步结果（同一拓扑/路由/传输/计算结论保持一致）；
4. 真实 vs 解析的**排序一致性**（Hybla 在两侧都应优于 Reno/CUBIC）；
5. 图表函数在真实评测结果上不报错；
6. 文本报告包含真实评测章节。

运行::

    python tests/verify_real_integration.py            # 快速档（小星座）
    python tests/verify_real_integration.py --full      # 含 GUI 离屏渲染
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from satnetsim.config import SimConfig, preset, real_eval_preset  # noqa: E402
from satnetsim.engine import Simulation  # noqa: E402
from satnetsim import viz  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def main() -> int:
    full = "--full" in sys.argv

    # ---------------- 1) 关闭时零开销 ----------------
    print("\n=== 1) 真实评测关闭：引擎不应进入第 ⑦ 步 ===")
    cfg_off = preset("starlink-lite")
    cfg_off.duration_s = 120.0
    cfg_off.compute.task_count = 60
    t0 = time.perf_counter()
    res_off = Simulation(cfg_off).run()
    t_off = time.perf_counter() - t0
    check("real_eval 默认关闭", cfg_off.real_eval.enabled is False)
    check("结果无 real_transfer", res_off.real_transfer is None)
    check("结果无 real_compute", res_off.real_compute is None)
    check("has_real_eval=False", res_off.has_real_eval is False)
    check("无 real_eval 分项计时", "real_eval" not in res_off.timings,
          f"timings={sorted(res_off.timings)}")
    print(f"  基线耗时 {t_off:.2f} s（{cfg_off.constellation.n_satellites} 星）")

    # ---------------- 2) 打开时跑通第 ⑦ 步 ----------------
    print("\n=== 2) 真实评测开启：引擎应跑通第 ⑦ 步 ===")
    cfg_on = real_eval_preset(
        "starlink-lite",
        payload_mbit=8.0, tcp_trials=1, packet_trials=1,
        packet_algos=["reno", "cubic", "hybla"],
        real_task_count=40, repeats=2, compare_analytic=True,
    )
    cfg_on.duration_s = cfg_off.duration_s
    cfg_on.compute.task_count = cfg_off.compute.task_count
    t0 = time.perf_counter()
    res_on = Simulation(cfg_on).run()
    t_on = time.perf_counter() - t0
    check("has_real_eval=True", res_on.has_real_eval is True)
    check("real_eval 分项计时存在", "real_eval" in res_on.timings,
          f"{res_on.timings.get('real_eval', 0):.2f} s")
    print(f"  真实评测耗时 {t_on - t_off:.2f} s（总 {t_on:.2f} s）")

    # ---------------- 3) 传输侧自洽性 ----------------
    print("\n=== 3) 真实数据传输评测自洽性 ===")
    rt = res_on.real_transfer
    check("real_transfer 非空", rt is not None)
    if rt is not None:
        check("链路剖面来自代表路径",
              rt.profile.owd_ms > 0 and rt.profile.bandwidth_mbps > 0,
              rt.profile.describe())
        check("实测 RTT 与理论同量级",
              0.5 <= rt.measured_rtt_ms / max(rt.profile.rtt_ms, 1e-9) <= 2.0,
              f"实测 {rt.measured_rtt_ms:.1f} ms / 理论 {rt.profile.rtt_ms:.1f} ms")
        check("真实 TCP 至少成功一次",
              any(t.ok and t.sha_match for t in rt.tcp),
              f"{len(rt.tcp)} 次试验")
        check("真实 TCP 字节级 SHA-256 一致",
              all(t.sha_match for t in rt.tcp if t.ok))
        check("真实包级至少成功一次",
              any(t.ok for t in rt.packet), f"{len(rt.packet)} 次试验")
        check("真实包级 SHA-256 一致",
              all(t.sha_match for t in rt.packet if t.ok))
        check("包级链路已按测试床容量收敛",
              rt.packet_profile is not None
              and rt.packet_profile.bandwidth_mbps <= max(rt.harness_cap_mbps, 1e-6)
              + 1e-6,
              f"{rt.packet_profile.bandwidth_mbps:.0f} Mbps ≤ "
              f"{rt.harness_cap_mbps:.0f} Mbps"
              if rt.packet_profile else "无包级剖面")
        check("解析模型对照已产出", len(rt.analytic) >= 3,
              f"解析：{ {k: round(v,3) for k,v in rt.analytic.items()} }")
        # 排序一致性：Hybla 应优于 Reno / CUBIC（卫星长时延链路）
        pk = rt.goodput_by_proto("packet")
        if {"hybla", "reno", "cubic"} <= set(pk):
            real_order = pk["hybla"] >= max(pk["reno"], pk["cubic"])
            an = rt.analytic_goodput
            an_order = (an.get("hybla", 0) >= max(an.get("reno", 0),
                                                  an.get("cubic", 0)))
            check("真实侧排序：Hybla ≥ Reno/CUBIC", real_order,
                  f"hybla {pk['hybla']:.1f} / reno {pk['reno']:.1f} / "
                  f"cubic {pk['cubic']:.1f} Mbps")
            check("解析侧排序：Hybla ≥ Reno/CUBIC", an_order,
                  f"hybla {an.get('hybla',0):.0f} / reno {an.get('reno',0):.0f} / "
                  f"cubic {an.get('cubic',0):.0f} Mbps")
            check("真实与解析排序一致", real_order == an_order)
        else:
            check("包级三协议齐备（Reno/CUBIC/Hybla）", False, f"实得 {sorted(pk)}")

    # ---------------- 4) 计算侧自洽性 ----------------
    print("\n=== 4) 真实计算任务评测自洽性 ===")
    rc = res_on.real_compute
    check("real_compute 非空", rc is not None)
    check("真实任务流非空", len(res_on.real_tasks) > 0,
          f"{len(res_on.real_tasks)} 个任务")
    if rc is not None:
        from satnetsim.realworkload import DEFAULT_SPECS

        ok_runs = [r for r in rc.calibration.runs if r.ok]
        check("真实负载无失败项", len(ok_runs) == len(rc.calibration.runs),
              f"{len(ok_runs)}/{len(rc.calibration.runs)} 成功，"
              f"{len(rc.calibration.skipped)} 项缺依赖")
        check("本机峰值算力自洽（不超过 ~500 GFLOPS 量级）",
              0 < rc.calibration.peak_gflops() < 500,
              f"{rc.calibration.peak_gflops():.1f} GFLOPS")
        check("至少有一个真实负载真正跑通", len(ok_runs) >= 1,
              ", ".join(r.key for r in ok_runs))
        if rc.calibration.skipped:
            check("缺依赖负载已显式告警",
                  any("缺少依赖" in w for w in res_on.warnings),
                  f"跳过 {[k for k, _ in rc.calibration.skipped]}")
        check("真实任务计数一致", rc.task_count == len(res_on.real_tasks))
        check("计算总量与任务流一致",
              abs(rc.total_gflops - sum(t.cycles_gops for t in res_on.real_tasks))
              < 1e-6 * max(rc.total_gflops, 1.0),
              f"{rc.total_gflops:.1f} GFLOP")
        check("地面串行耗时 > 0（真实执行）", rc.ground_total_s > 0,
              f"{rc.ground_total_s:.3f} s")
        check("星载折算耗时 > 0", rc.on_orbit_total_s > 0,
              f"{rc.on_orbit_total_s:.3f} s")
        check("加速比自洽", abs(rc.speedup - rc.ground_total_s /
                                rc.on_orbit_total_s) < 1e-6)
        check("任务画像覆盖多个负载", len(rc.workload_mix) >= 4,
              rc.mix_text())
        # 任务必须来自已定义的真实负载画像（不缺依赖时还必须能追溯到实测）
        check("任务来源均为已定义真实负载",
              all(getattr(t, "workload_key", "") in DEFAULT_SPECS
                  for t in res_on.real_tasks))
        measured_keys = {r.key for r in ok_runs}
        tasks_measured = [t for t in res_on.real_tasks
                          if getattr(t, "workload_key", "") in measured_keys]
        check("有实测记录的负载在任务流中被真实引用",
              len(tasks_measured) > 0 or not measured_keys,
              f"{len(tasks_measured)}/{len(res_on.real_tasks)} 个任务有实测来源")

    # ---------------- 5) 真实性不影响前七步结论 ----------------
    print("\n=== 5) 真实评测不改变前七步结论 ===")
    check("星数一致",
          res_on.constellation.n_sat == res_off.constellation.n_sat)
    check("快照数一致", len(res_on.snapshots) == len(res_off.snapshots))
    d_on = min(r.mean_delay_ms for r in res_on.routes.values())
    d_off = min(r.mean_delay_ms for r in res_off.routes.values())
    check("路由时延一致（同 seed 确定性）", abs(d_on - d_off) < 1e-9,
          f"{d_off:.3f} ms")
    f_on = min(f.completion_s for f in res_on.transport)
    f_off = min(f.completion_s for f in res_off.transport)
    check("解析传输结论一致", abs(f_on - f_off) < 1e-9, f"{f_off:.3f} s")

    # ---------------- 6) 图表与报告 ----------------
    print("\n=== 6) 图表与文本报告 ===")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 8), dpi=90)
    axs = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    try:
        viz.plot_real_transfer_compare(axs[0], rt)
        viz.plot_real_cwnd(axs[1], rt)
        viz.plot_real_workload_time(axs[2], rc)
        viz.plot_real_compute_split(axs[3], rc)
        fig.tight_layout()
        out = os.path.join("docs", "verify", "real_integration.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, facecolor="#FFFFFF")
        check("四个真实评测图表渲染成功", os.path.getsize(out) > 20_000,
              f"{out} ({os.path.getsize(out)//1024} KB)")
    except Exception as exc:  # noqa: BLE001
        check("四个真实评测图表渲染成功", False, f"{type(exc).__name__}: {exc}")

    txt = viz.text_report(res_on)
    check("报告含真实数据传输章节", "真实数据传输评测" in txt)
    check("报告含真实计算任务章节", "真实计算任务评测" in txt)
    check("报告含 SHA-256 完整性结论", "SHA-256" in txt)
    check("概览含真实评测行",
          any("真实" in k for k, _ in res_on.headline()),
          str([k for k, _ in res_on.headline() if "真实" in k]))

    # ---------------- 7) GUI 离屏（可选） ----------------
    if full:
        print("\n=== 7) GUI 离屏渲染（含两个新页签） ===")
        try:
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance() or QApplication([])
            from satnetsim.gui import MainWindow

            win = MainWindow()
            tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
            check("页签数 = 11", win.tabs.count() == 11, str(tabs))
            check("含『传输任务』", any("传输任务" in t for t in tabs))
            check("含『计算任务』", any("计算任务" in t for t in tabs))
            check("日志页签存在", any("日志" in t for t in tabs))
            check("参数面板有真实评测开关", hasattr(win, "chk_real")
                  and hasattr(win, "chk_real_transfer"))
            # 字体回归守卫：真实评测页签里有中文文字面板，若写死 Consolas
            # 之类无 CJK 字形的等宽字体，会静默退化为方框
            import warnings as _w

            with _w.catch_warnings(record=True) as rec:
                _w.simplefilter("always")
                win._render(res_on)
            glyph = [str(x.message) for x in rec
                     if "missing from font" in str(x.message)]
            check("GUI 渲染无缺字形告警（CJK 字体来源统一）", not glyph,
                  (glyph[:3] if glyph else "无"))
            win._render(res_on)
            check("GUI 渲染真实评测结果不报错", True)
            del win, app
        except Exception as exc:  # noqa: BLE001
            check("GUI 离屏渲染", False, f"{type(exc).__name__}: {exc}")

    # ---------------- 8) JSON 往返 ----------------
    print("\n=== 8) 配置 JSON 往返（real_eval 段） ===")
    cfg = real_eval_preset("iridium-66", real_task_count=77, payload_mbit=32.0)
    cfg.real_eval.packet_algos = ["cubic", "hybla"]
    back = SimConfig.from_dict(cfg.to_dict())
    check("real_eval 往返保真",
          back.real_eval.enabled and back.real_eval.real_task_count == 77
          and abs(back.real_eval.payload_mbit - 32.0) < 1e-9
          and back.real_eval.packet_algos == ["cubic", "hybla"],
          f"{back.real_eval}")

    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

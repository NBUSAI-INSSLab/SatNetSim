"""真实数据流仿真 + 实时回放 验证脚本。

用法::
    python tests/verify_realdata.py            # 快速（小星座）
    python tests/verify_realdata.py --full     # 追加 starlink-lite 全量 + GUI 离屏
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from satnetsim.config import preset
from satnetsim.engine import Simulation
from satnetsim.orbit import Constellation, unix_from_iso
from satnetsim.realdatastream import (build_payload, reassemble, segment_payload,
                                      simulate_real_data)
from satnetsim.topology import TopologyBuilder

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  —— {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  —— {detail}" if detail else ""))


def build_scene(name: str):
    cfg = preset(name)
    con = Constellation(cfg.constellation)
    times = np.arange(0.0, cfg.duration_s + 1e-9, cfg.snapshot_interval_s)
    u = unix_from_iso(cfg.start_utc)
    con.propagate(times)
    con.to_ecef(u)
    snaps = TopologyBuilder(cfg.link, cfg.ground_stations).build(
        con, times, u, cfg.snapshot_interval_s)
    return cfg, con, snaps, times


def main() -> int:
    full = "--full" in sys.argv
    print("=" * 68)
    print("① 载荷与分片（真实字节）")
    print("=" * 68)
    p_img = build_payload("image", 512 * 1024)
    check("影像载荷为真 JPEG/MJPEG 字节",
          p_img.data[:3] == b"\xff\xd8\xff" or p_img.data[:8] == b"\x89PNG\r\n\x1a\n",
          f"{p_img.mime} {p_img.n_bytes/1024:.0f} KiB ×{p_img.frames} 帧")
    check("影像载荷尺寸贴近目标", 0.9 * 512 * 1024 <= p_img.n_bytes <= 1.1 * 512 * 1024,
          f"{p_img.n_bytes/1024:.0f} / 512 KiB")
    p_txt = build_payload("text", 200_000)
    check("文本载荷为真 UTF-8 且按字符边界截断",
          p_txt.data.decode("utf-8", errors="ignore").encode("utf-8") == p_txt.data,
          f"{p_txt.n_bytes} B")
    p_tel = build_payload("telemetry", 100_000)
    import struct as _struct
    _rec = _struct.calcsize(">dfffff")          # t(double) + 5×float = 28 B
    check("遥测载荷 struct 定长记录（不截断半条）",
          len(p_tel.data) % _rec == 0 and len(p_tel.data) > 0,
          f"{len(p_tel.data)} B / 记长 {_rec} B")
    segs = segment_payload(p_img, 1200)
    check("分片数 = ceil(N/MSS)", len(segs) == (p_img.n_bytes + 1199) // 1200,
          f"{len(segs)} 片")
    back, ok = reassemble({s.seq: s.data for s in segs}, len(segs))
    check("重组逐字节一致", ok and back == p_img.data)

    print()
    print("=" * 68)
    print("② 逐包多跳转发（离散事件内核）")
    print("=" * 68)
    cfg, con, snaps, times = build_scene("walker-polar" if not full else "starlink-lite")
    rd = cfg.real_data
    rd.payload_mbit = 16.0
    rd.n_flows = 2
    rd.duration_s = 8.0
    rep = simulate_real_data(rd, con, snaps, cfg.ground_stations,
                             snapshot_interval_s=cfg.snapshot_interval_s,
                             sim_duration_s=cfg.duration_s, seed=cfg.seed)
    check("分组数 ≥ 分片数（至少完整一轮）", rep.n_packets >= rep.n_segments,
          f"{rep.n_packets} 分组 / {rep.n_segments} 分片，循环 {rep.n_rounds} 轮")
    check("分组数在预算内", rep.n_packets <= max(int(rd.max_packets), 1) * rd.n_flows,
          f"{rep.n_packets} ≤ 预算 {rd.max_packets}×{rd.n_flows}")
    check("仿真窗口 = 配置窗口", abs(rep.duration_s - rd.duration_s) < 1e-6,
          f"{rep.duration_s:.1f} s")
    check("载荷字节数与分片一致",
          rep.n_segments * rep.mss_bytes >= rep.payload_bytes
          and (rep.n_segments - 1) * rep.mss_bytes < rep.payload_bytes,
          f"{rep.payload_bytes/1024:.0f} KiB / MSS {rep.mss_bytes}")
    check("全部送达（覆盖缺口自动避让）",
          rep.n_delivered == rep.n_packets,
          f"{rep.n_delivered}/{rep.n_packets}")
    check("SHA-256 一致", rep.integrity_ok and rep.digest_src == rep.digest_dst,
          f"{rep.digest_src[:16]}")
    check("收到的字节数 = 载荷", rep.received_bytes == rep.payload_bytes,
          f"{rep.received_bytes/1024:.0f} KiB")
    e2e = [p.e2e_s for p in rep.packets if p.ok]
    check("端到端时延 ≥ 传播下界（单跳 ≥ 2 ms）", min(e2e) >= 0.002,
          f"min {min(e2e)*1000:.1f} ms / mean {rep.e2e_mean_s*1000:.1f} ms")
    check("时延均值 = 各分组均值",
          abs(rep.e2e_mean_s - float(np.mean(e2e))) < 1e-9)
    hops = [p.hops for p in rep.packets]
    check("平均跳数与记录一致",
          abs(rep.mean_hops - float(np.mean(hops))) < 1e-9,
          f"{rep.mean_hops:.2f} 跳（最多 {rep.max_hops}）")
    # 守恒：每条链路的发送字节数 = 沿途经过该链路的分组字节数之和
    link_bytes = sum(s.bytes for s in rep.link_stats)
    seg_bytes = sum(len(p.segs) * p.size_bytes for p in rep.packets)
    check("链路字节守恒（逐跳累计 = 分组×跳数）", link_bytes == seg_bytes,
          f"{link_bytes/1e6:.3f} vs {seg_bytes/1e6:.3f} Mbit")
    # 链路缓冲不超限
    over = [s for s in rep.link_stats if s.peak_queue_bytes > 1 << 30]
    check("无链路队列异常膨胀", not over)
    # 序列
    check("时间序列长度一致",
          len(rep.series_t) == len(rep.series_inflight)
          == len(rep.series_goodput) == len(rep.series_delivered),
          f"{len(rep.series_t)} 箱")
    check("在途数单调性合理（峰值 ≤ 并发分组数）",
          max(rep.series_inflight) <= rep.n_packets,
          f"峰值 {max(rep.series_inflight):.0f}")
    check("累计送达量单调不减",
          all(a <= b + 1e-9 for a, b in
              zip(rep.series_delivered, rep.series_delivered[1:])),
          f"终值 {rep.series_delivered[-1]:.1f} Mbit")
    check("JSON 可序列化（含轨迹）",
          isinstance(rep.to_dict(trace_limit=100)["trace"], list))

    if full:
        print()
        print("=" * 68)
        print("③ 引擎集成（第 ⑨ 步）")
        print("=" * 68)
        cfg2 = preset("iridium-66")
        cfg2.duration_s = 120.0
        cfg2.compute.task_count = 40
        res_on = Simulation(cfg2).run(run_real_data=True, run_real=False)
        check("SimulationResult.real_data 已产出", res_on.real_data is not None)
        check("headline 含真实数据流行",
              any("真实数据流" in k for k, _ in res_on.headline()))
        res_off = Simulation(cfg2).run(run_real_data=False, run_real=False)
        check("关闭时 real_data 为 None", res_off.real_data is None)
        check("关闭时不产生 [真实数据流] 告警",
              not any("真实数据流" in w for w in res_off.warnings))
        check("文本报告含真实数据流章节",
              "真实数据流仿真" in __import__("satnetsim.viz", fromlist=["text_report"])
              .text_report(res_on))

    print()
    print("=" * 68)
    print("④ 实时回放（LiveScene / 帧模型 / HTML）")
    print("=" * 68)
    from satnetsim.liveviz import LiveScene, build_frame, export_html

    scene = LiveScene.build(con, snaps, cfg.ground_stations, rep, times)
    check("分段索引覆盖所有跳", scene.seg_t0.size == sum(len(p.segs) for p in rep.packets),
          f"{scene.seg_t0.size} 段")
    check("事件数 = 创建 + 终结", scene.ev_t.size == 2 * rep.n_packets,
          f"{scene.ev_t.size} 事件")
    check("分组建段连续（拖尾可回溯）",
          int(scene._pkt_seg1[-1]) == scene.seg_t0.size)
    cnts = []
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        fm = build_frame(scene, scene.duration_s * f, 1280, 760, 0.55, 0.42)
        cnts.append(len(fm.comets))
    check("任意时刻都有彗星在飞（画面持续有流量）", min(cnts) > 0,
          f"彗星数 {cnts}")
    check("链路网格已生成", fm.links.shape[0] > 0, f"{fm.links.shape[0]} 条")
    check("地面站投影齐全", fm.gs.shape[0] == len(cfg.ground_stations))
    # 端点高亮受视角遮挡影响：换 8 个方位角，必须能让每条流的端点都露出来
    seen_flows = set()
    for k in range(8):
        fmk = build_frame(scene, scene.duration_s * 0.5, 1280, 760,
                          0.785 * k, 0.42)
        for fe in fmk.flow_ends:
            seen_flows.add(fe["flow"])
    check("每个流的端点都能在某些视角下高亮",
          seen_flows == {f["id"] for f in scene.flows},
          f"覆盖 {sorted(seen_flows)} / {len(scene.flows)} 流")
    check("帧几何无 NaN", np.isfinite(fm.sats).all() and np.isfinite(fm.links).all())
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "verify", "realdata_live.html")
    export_html(scene, out)
    check("单文件 HTML 导出成功", os.path.getsize(out) > 100_000,
          f"{os.path.getsize(out)/1024:.0f} KiB")

    if full:
        print()
        print("=" * 68)
        print("⑤ GUI 离屏（仿真展示页签）")
        print("=" * 68)
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from satnetsim.gui import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        win = MainWindow()
        check("页签数 = 11", win.tabs.count() == 11, f"{win.tabs.count()}")
        idx_live = next(i for i in range(win.tabs.count())
                        if "仿真展示" in win.tabs.tabText(i))
        check("页签含「仿真展示」",
              "仿真展示" in win.tabs.tabText(idx_live),
              win.tabs.tabText(idx_live))
        check("参数面板有真实数据流控件",
              hasattr(win, "chk_rdata") and hasattr(win, "sp_rdata_flows"))
        win._render(res_on)
        check("GUI 渲染真实数据流结果不报错", win.live.scene is not None,
              f"分段 {win.live.scene.seg_t0.size}")
        img = win.live.view.snap_image(0.35 * win.live.scene.duration_s,
                                       1280, 760)
        png = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "docs", "verify", "gui_tab10_realdata.png")
        img.save(png)
        check("离屏截图已保存", os.path.getsize(png) > 50_000, png)
        del win, app

    print()
    print("=" * 68)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 68)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

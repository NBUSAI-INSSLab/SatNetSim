"""截取 ④路由分析 / ⑤传输分析 / ⑧真实数据传输 / ⑨真实计算任务 四个页签。

用于人工目检"这四个页签确实有数据"（本轮修复的四个问题）：
  - ⑧⑨ 之前只显示"未运行"占位符；
  - ④ 之前只有基础路由表，没有效果评测；
  - ⑤ 之前只有解析轮次模型，没有真实包级 TCP。

默认离屏（QT_QPA_PLATFORM=offscreen）。传 --show 在桌面真实开窗截取。

运行::  python tests/_shot_final_tabs.py [--show]
输出::  docs/verify/gui_tab4_route.png 等
"""
from __future__ import annotations

import os
import sys

if "--show" not in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")

from satnetsim.config import preset  # noqa: E402
from satnetsim.engine import Simulation  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "verify")
os.makedirs(OUT, exist_ok=True)


def build_cfg():
    cfg = preset("starlink-lite")
    cfg.duration_s = 300.0
    cfg.compute.task_count = 200
    # 真实评测（⑧⑨）——GUI 默认也是打开的
    cfg.real_eval.enabled = True
    cfg.real_eval.transfer = True
    cfg.real_eval.compute = True
    cfg.real_eval.payload_mbit = 16.0
    cfg.real_eval.tcp_trials = 1
    cfg.real_eval.packet_trials = 1
    cfg.real_eval.real_task_count = 60
    cfg.real_eval.repeats = 2
    # 传输分析引擎：真实包级 TCP + 解析对照
    cfg.transport.engine = "both"
    cfg.transport.app_payload_mbit = 64.0
    # 真实数据流（⑩）
    cfg.real_data.enabled = True
    cfg.real_data.duration_s = 0.0        # 跟随仿真时长
    cfg.real_data.n_flows = 2
    return cfg


cfg = build_cfg()
res = Simulation(cfg).run(progress=lambda *a, **k: None)

from PySide6.QtWidgets import QApplication  # noqa: E402
from satnetsim.gui import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1600, 980)
win.show()
app.processEvents()
win._render(res)
win.resize(1600, 980)
app.processEvents()

shots = [
    (3, "gui_tab4_route"),
    (4, "gui_tab5_transport"),
    (7, "gui_tab8_real_transfer"),
    (8, "gui_tab9_real_compute"),
]
for idx, name in shots:
    win.tabs.setCurrentIndex(idx)
    app.processEvents()
    pix = win.grab()
    path = os.path.join(OUT, name + ".png")
    pix.save(path)
    print(f"[OK] {name} -> {os.path.getsize(path)} bytes")

print("窗口尺寸", win.width(), "x", win.height())
print("真实评测耗时", round(res.timings.get("real_eval", 0), 2), "s")
print("真实数据流耗时", round(res.timings.get("real_data", 0), 2), "s")
print("has_real_eval =", res.has_real_eval, " has_real_data =", res.has_real_data)
del win, app

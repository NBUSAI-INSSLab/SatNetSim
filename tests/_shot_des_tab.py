"""截取 ⑫事件内核 页签并做 GUI 冒烟（离散事件轮新增）。

离屏运行：真实评测 + 真实数据流 + 离散事件编排全开，渲染后截取：
  - ⑫ 事件内核（时间线甘特图 / 事件构成 / 队列深度 / 摘要表）
  - ① 概览（结果摘要卡片行）

运行::  python tests/_shot_des_tab.py
输出::  docs/verify/gui_tab12_des.png、gui_overview_cards.png
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")

from satnetsim.config import preset  # noqa: E402
from satnetsim.engine import Simulation  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "verify")
os.makedirs(OUT, exist_ok=True)

cfg = preset("starlink-lite")
cfg.duration_s = 300.0
cfg.real_eval.enabled = True
cfg.real_eval.transfer = True
cfg.real_eval.compute = True
cfg.real_eval.payload_mbit = 16.0
cfg.real_eval.tcp_trials = 1
cfg.real_eval.packet_trials = 1
cfg.real_eval.real_task_count = 60
cfg.real_eval.repeats = 2
cfg.transport.engine = "both"
cfg.transport.app_payload_mbit = 64.0
cfg.real_data.enabled = True
cfg.real_data.duration_s = 0.0
cfg.real_data.n_flows = 2
# ⑫ 离散事件编排
cfg.des.enabled = True
cfg.des.n_flows = 3
cfg.des.flow_mbit = 32.0
cfg.des.tcp_algo = "bbr"
cfg.des.n_tasks = 200
cfg.des.trace = True

t0 = __import__("time").perf_counter()
res = Simulation(cfg).run(progress=lambda *a, **k: None)
print(f"仿真耗时 {__import__('time').perf_counter() - t0:.1f} s")
print("has_des =", res.has_des)
assert res.des is not None, "离散事件编排未产出 DesReport"

from PySide6.QtWidgets import QApplication  # noqa: E402
from satnetsim.gui import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1600, 980)
win.show()
app.processEvents()
win._render(res)
win._fill_cards(res)
app.processEvents()

idx = win.tabs.indexOf(win.p_des)
win.tabs.setCurrentIndex(idx)
app.processEvents()
pix = win.grab()
p1 = os.path.join(OUT, "gui_tab12_des.png")
pix.save(p1)
print(f"[OK] gui_tab12_des -> {os.path.getsize(p1)} bytes")

win.tabs.setCurrentIndex(0)
app.processEvents()
pix = win.grab()
p2 = os.path.join(OUT, "gui_overview_cards.png")
pix.save(p2)
print(f"[OK] gui_overview_cards -> {os.path.getsize(p2)} bytes")

print("页签数 =", win.tabs.count(), [win.tabs.tabText(i) for i in range(win.tabs.count())])
print("事件执行数 =", res.des.n_executed, " 流完成 =",
      f"{res.des.flows_done}/{len(res.des.flows)}")
del win, app

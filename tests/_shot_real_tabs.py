"""截取 GUI 的⑧真实数据传输 / ⑨真实计算任务 页签，用于人工目检布局。

默认离屏（QT_QPA_PLATFORM=offscreen；Qt 侧字体回退在离屏平台可能失效，
截图里的中文方框只说明离屏字体枚举受限，不代表桌面端显示）。
传 --show 则在桌面真实开窗截取（与用户所见一致）。

运行::  python tests/_shot_real_tabs.py [--show]
输出::  docs/verify/gui_tab*.png
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

from satnetsim.config import real_eval_preset  # noqa: E402
from satnetsim.engine import Simulation  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "verify")
os.makedirs(OUT, exist_ok=True)

cfg = real_eval_preset("starlink-lite", payload_mbit=16.0, tcp_trials=1,
                       packet_trials=1, real_task_count=60, repeats=2)
cfg.duration_s = 300.0
cfg.compute.task_count = 200
res = Simulation(cfg).run(progress=lambda *a, **k: None)

from PySide6.QtWidgets import QApplication  # noqa: E402
from satnetsim.gui import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1500, 940)
win.show()
app.processEvents()
win._render(res)
win.resize(1500, 940)
app.processEvents()

shots = [(0, "gui_tab1_overview"), (7, "gui_tab8_real_transfer"),
         (8, "gui_tab9_real_compute")]
for idx, name in shots:
    win.tabs.setCurrentIndex(idx)
    app.processEvents()
    win.pbar_dummy = None
    pix = win.grab()
    path = os.path.join(OUT, name + ".png")
    pix.save(path)
    print(f"[OK] {name} -> {os.path.getsize(path)} bytes")

print("窗口尺寸", win.width(), "x", win.height())
print("真实评测耗时", round(res.timings.get("real_eval", 0), 2), "s")
del win, app

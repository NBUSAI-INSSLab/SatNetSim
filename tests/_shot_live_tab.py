"""截取 GUI 的⑩实时数据流页签（桌面开窗，动画播放中），用于人工目检。

运行::  python tests/_shot_live_tab.py
输出::  docs/verify/gui_tab10_realdata_desktop.png
"""
from __future__ import annotations

import os
import sys

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
cfg.compute.task_count = 120
cfg.real_eval.enabled = False          # 只跑真实数据流，节省时间
res = Simulation(cfg).run(progress=lambda *a, **k: None)
assert res.real_data is not None

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from satnetsim.gui import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1500, 940)
win.show()
app.processEvents()
win._render(res)
win.tabs.setCurrentIndex(9)            # ⑩ 实时数据流
win.live.toggle()                      # 开始播放

shots = []


def snap() -> None:
    p = os.path.join(OUT, f"gui_tab10_realdata_desktop_{len(shots)}.png")
    win.live.view.grab().save(p)
    shots.append(p)
    print("截图 →", p)
    if len(shots) >= 3:
        app.quit()


timer = QTimer()
timer.timeout.connect(snap)
timer.start(2500)
app.exec()
print("完成，共", len(shots), "张")

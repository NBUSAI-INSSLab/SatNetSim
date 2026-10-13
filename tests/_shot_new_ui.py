"""离屏截取新 UI：导航栏 + 概览页 + 事件内核页，供人工目检。

运行::  python tests/_shot_new_ui.py
输出::  docs/verify/gui_newui_*.png
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")

from satnetsim.gui import MainWindow  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "verify")
os.makedirs(OUT, exist_ok=True)

app = __import__("PySide6.QtWidgets", fromlist=["QApplication"]).QApplication.instance()
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1500, 940)
win.show()
app.processEvents()


def shot(name: str) -> None:
    pix = win.grab()
    path = os.path.join(OUT, name + ".png")
    pix.save(path)
    print("[OK]", path, os.path.getsize(path), "bytes")


shot("gui_newui_overview")

# 切到 ⑫ 事件内核页（先填点假数据看渲染不炸即可：仅页面切换）
idx = win.tabs.indexOf(win.p_des)
win.tabs.setCurrentIndex(idx)
app.processEvents()
shot("gui_newui_des_tab")

# 切到 ⑤ 传输分析
win.tabs.setCurrentIndex(win.tabs.indexOf(win.p_trans))
app.processEvents()
shot("gui_newui_trans_tab")

win.close()
print("DONE")

"""GUI 离屏冒烟测试：真实构建 MainWindow 并核对导航页结构。"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QListWidget, QGroupBox  # noqa: E402

from satnetsim.gui import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()

print("窗口标题 :", win.windowTitle())

navs = [n for n in win.findChildren(QListWidget) if n.objectName() == "navList"]
print("导航列表数量 :", len(navs))
if navs:
    n = navs[0]
    pages = [i for i in range(n.count())
             if n.item(i).flags() & Qt.ItemIsSelectable]
    print("导航行数 :", n.count(), "（分组标题", n.count() - len(pages), "个）")
    for i in range(n.count()):
        mark = " " if i in pages else "·"
        print("   %s[%d] %s" % (mark, i, n.item(i).text()))
    assert len(pages) == 11, f"可选页应为 11 个，实际 {len(pages)}"
    # 页栈与导航一一对应
    assert win.tabs.stack.count() == 11
    assert win.tabs.currentWidget() is win.tabs.stack.widget(0)

    # 分组结构：仿真分析 / 真实任务 / 实时动态（3 个不可选分组头）
    sections = [n.item(i).text() for i in range(n.count())
                if not (n.item(i).flags() & Qt.ItemIsSelectable)]
    print("分组标题 :", sections)
    assert sections == ["仿真分析", "真实任务", "实时动态"], sections
    labels = [n.item(i).text() for i in range(n.count())
              if n.item(i).flags() & Qt.ItemIsSelectable]
    assert any("结果概览" in t for t in labels), "缺少「结果概览」"
    assert not any("仿真概览" in t for t in labels), "「仿真概览」应已改名"
    # 真实任务分组的两个页签
    i_sec = [i for i in range(n.count()) if n.item(i).text() == "真实任务"][0]
    assert "传输任务" in n.item(i_sec + 1).text(), n.item(i_sec + 1).text()
    assert "计算任务" in n.item(i_sec + 2).text(), n.item(i_sec + 2).text()

groups = win.findChildren(QGroupBox)
print("参数分组数 :", len(groups))
for g in groups:
    print("   -", g.title())

# 收集一次配置，确认 UI → config 的映射没坏
cfg = win._collect_config()
print("收集配置 :", cfg.constellation.planes, "x", cfg.constellation.sats_per_plane,
      "| ISL", cfg.link.isl_bandwidth_mbps / 1000.0, "Gbps")

# JSON 按钮反馈回归：应用到场景 / 从场景刷新 必须给出可见反馈
from PySide6.QtWidgets import QMessageBox as _MB  # noqa: E402
_warns = []
_MB.warning = staticmethod(lambda *a, **k: _warns.append(a[1:3]))
d = json.loads(win.txt_json.toPlainText())
d["constellation"]["planes"] = 8
win.txt_json.setPlainText(json.dumps(d, ensure_ascii=False, indent=2))
win.btn_json_apply.click()
assert win.cfg.constellation.planes == 8, "应用到场景未生效"
assert not win.lab_json_hint.isHidden() and "✓" in win.lab_json_hint.text(), \
    f"应用到场景缺少反馈：{win.lab_json_hint.text()!r}"
print("应用到场景反馈 :", win.lab_json_hint.text())
win.btn_json_fill.click()
assert "✓" in win.lab_json_hint.text(), "从场景刷新缺少反馈"
print("从场景刷新反馈 :", win.lab_json_hint.text())

win.close()
print("GUI SMOKE OK")

"""截取当前桌面上的 SatNetSim 主窗口（整屏抓取后裁剪到窗口区域）。

用于人工目检"运行中的软件"实际长什么样（含真实评测页签数据）。
需要在桌面会话下运行；离屏平台不可用。

运行::  python tests/_shot_desktop.py [输出名]
输出::  docs/verify/<输出名>.png（默认 gui_desktop_now.png）
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = os.path.join(_ROOT, "docs", "verify")
os.makedirs(OUT, exist_ok=True)
name = sys.argv[1] if len(sys.argv) > 1 else "gui_desktop_now"

user32 = ctypes.windll.user32
try:
    user32.SetProcessDPIAware()
except Exception:
    pass

TITLE_BUF = ctypes.create_unicode_buffer(512)
target: list[tuple] = []


@ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
def _cb(hwnd, _lparam):
    if not user32.IsWindowVisible(hwnd):
        return True
    user32.GetWindowTextW(hwnd, TITLE_BUF, 512)
    title = TITLE_BUF.value
    if "SatNetSim" not in title:
        return True
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w > 800 and h > 500:
        target.append((hwnd, title, rect.left, rect.top, w, h))
    return True


user32.EnumWindows(_cb, 0)

if not target:
    print("[FAIL] 未找到可见的 SatNetSim 主窗口（GUI 是否在运行？）")
    raise SystemExit(1)

hwnd, title, x, y, w, h = target[0]
print(f"[OK] 窗口: {title!r}  位置 ({x},{y})  尺寸 {w}x{h}")

user32.ShowWindow(hwnd, 9)          # SW_RESTORE
user32.SetForegroundWindow(hwnd)

app = QApplication.instance() or QApplication([])
pix = app.primaryScreen().grabWindow(0)   # 整屏
shot = pix.copy(x, y, w, h)
path = os.path.join(OUT, name + ".png")
shot.save(path)
print(f"[OK] 截图 -> {path} ({os.path.getsize(path)} bytes)")

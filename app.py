#!/usr/bin/env python
"""SatNetSim 桌面入口。

    python app.py                       # 启动图形界面
    python app.py run starlink-lite     # 命令行跑一次仿真
    python app.py list                  # 列出内置场景预设
"""

from __future__ import annotations

import os
import sys

# 允许直接以脚本方式运行（无需 pip install）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from satnetsim.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

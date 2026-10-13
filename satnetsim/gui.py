"""PySide6 图形界面。

设计取向
--------
1. **零门槛运行**：打开即有一个可直接跑的默认场景（starlink-lite），点"开始仿真"
   就能在几秒内得到 7 个页面的结果。
2. **参数即场景**：左侧所有控件都直连 `SimConfig` 数据类；另有 JSON 高级页可直接
   粘贴/导出完整场景，便于批量扫参与结果复现。
3. **不阻塞 UI**：仿真跑在 QThread 里，通过信号回传进度；绘图统一在结果返回后
   一次性完成。
4. **离线可视化**：matplotlib 内嵌（QtAgg），不依赖 Cesium 等在线服务。
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
# 中文字体配置统一由 satnetsim.viz 负责（本模块下方 from . import viz 时生效）

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from PySide6.QtCore import Qt, QThread, Signal, QTimer  # noqa: E402
from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QAction, QFont, QColor, QIcon  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import viz  # noqa: E402
from .config import (  # noqa: E402
    PRESETS,
    GroundStationConfig,
    SimConfig,
    preset,
)
from .engine import Simulation, SimulationResult  # noqa: E402
from .routing import ROUTER_LABELS  # noqa: E402
from .transport import PROTOCOL_META  # noqa: E402

# 薄荷绿主题（浅色底，2026 版：侧边导航 + 卡片化布局）
QSS = """
* { outline: none; }
QWidget { background: #F4F9F9; color: #10302F; font-size: 12px;
          font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
QToolTip {
    background: #0E615F; color: #FFFFFF; border: none; padding: 4px 8px;
    border-radius: 4px; font-size: 11px;
}

/* ---------- 左侧导航（浅色，与主界面同色系） ---------- */
#navPanel { background: #FBFDFD; border-right: 1px solid #D8ECEA; }
#navLogo { background: transparent; }
#navBrand {
    background: transparent; color: #0E7C86; font-size: 16px; font-weight: 800;
    padding: 0;
}
#navBrandSub {
    background: transparent; color: #7FA8A5; font-size: 10px; padding-top: 1px;
}
#navList {
    background: transparent; border: none; color: #2F5B58; font-size: 12.5px;
    padding: 6px 8px 10px 8px;
}
#navList::item {
    background: transparent; border-radius: 8px; padding: 7px 10px;
    margin: 1px 0; color: #35605D;
}
#navList::item:hover { background: #E6F4F3; color: #0E7C86; }
#navList::item:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #128892, stop:1 #0E7C86);
    color: #FFFFFF; font-weight: 600;
}

/* ---------- 参数分组卡片 ---------- */
QGroupBox {
    border: 1px solid #CBE4E2; border-radius: 10px; margin-top: 15px;
    padding: 12px 9px 9px 9px; background: #FFFFFF;
}
QGroupBox:hover { border-color: #7FBFBC; }
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: #0E7C86; font-weight: 600;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #128892, stop:1 #0E7C86);
    color: #FFFFFF; border: none; border-radius: 6px;
    padding: 6px 16px; font-weight: 600;
}
QPushButton:hover { background: #0B6B74; }
QPushButton:pressed { background: #095A62; }
QPushButton:disabled { background: #A8C7C6; color: #EAF4F3; }
QPushButton#ghost {
    background: #FFFFFF; color: #0E7C86; border: 1px solid #8EC7C4;
}
QPushButton#ghost:hover { background: #E6F4F3; border-color: #0E7C86; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #FFFFFF; border: 1px solid #C4E0DE; border-radius: 5px;
    padding: 3px 6px; selection-background-color: #CDE9E7;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #8EC7C4; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1.5px solid #0E7C86; background: #FBFEFE;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: #FFFFFF; border: 1px solid #8EC7C4; border-radius: 6px;
    selection-background-color: #CDE9E7; selection-color: #0E615F;
}
QTableWidget {
    gridline-color: #DCEBEA; selection-background-color: #CDE9E7;
    border: 1px solid #DCEBEA; border-radius: 6px;
    alternate-background-color: #F7FCFB;
}
QHeaderView::section {
    background: #E6F4F3; border: none; border-right: 1px solid #CBDFDE;
    padding: 6px; font-weight: 600; color: #0E615F;
}
QProgressBar {
    border: 1px solid #B8DBD9; border-radius: 6px; background: #FFFFFF;
    text-align: center; color: #0E615F; height: 16px; font-weight: 600;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #35A8A3, stop:1 #0E7C86);
    border-radius: 5px;
}
QTextEdit, QPlainTextEdit {
    background: #FFFFFF; border: 1px solid #B8DBD9; border-radius: 6px;
    font-family: Consolas, "Microsoft YaHei"; font-size: 11px;
}
QTextEdit:focus, QPlainTextEdit:focus { border: 1.5px solid #0E7C86; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: transparent; width: 9px; border-radius: 4px; margin: 2px;
}
QScrollBar::handle:vertical {
    background: #9CCFCB; border-radius: 4px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #0E7C86; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: transparent; height: 9px; border-radius: 4px; margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #9CCFCB; border-radius: 4px; min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background: #0E7C86; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QStatusBar { background: #EDF6F5; color: #35605D; }
QStatusBar::item { border: none; }
QSplitter::handle { background: #D8ECEA; width: 3px; }
QSplitter::handle:hover { background: #0E7C86; }
QCheckBox::indicator, QGroupBox QCheckBox::indicator {
    width: 15px; height: 15px; border: 1.5px solid #8EC7C4; border-radius: 4px;
    background: #FFFFFF;
}
QCheckBox::indicator:checked { background: #0E7C86; border-color: #0E7C86; }
QToolBar { background: #EDF6F5; border-bottom: 1px solid #CBDFDE;
           spacing: 5px; padding: 5px 10px; }
QToolBar::separator { background: #C4E0DE; width: 1px; margin: 6px 8px; }
QToolBar QToolButton {
    padding: 5px 10px; border-radius: 6px; color: #35605D; font-size: 12.5px;
}
QToolBar QToolButton:hover { background: #D8ECEA; color: #0E7C86; }
QToolBar QToolButton:disabled { color: #9CC3C0; background: transparent; }
QPushButton#primaryRun {
    font-size: 13px; font-weight: 700; padding: 7px 24px; border-radius: 8px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #149BA5, stop:1 #0E7C86);
    color: #FFFFFF;
}
QPushButton#primaryRun:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #0E7C86, stop:1 #0B6B74);
}
QPushButton#primaryRun:pressed { background: #095A62; }
QPushButton#primaryRun:disabled { background: #A8C7C6; color: #EAF4F3; }
QLabel#progressMsg { color: #35605D; font-size: 11px; }
QLabel#etaLabel { color: #6B7280; font-size: 11px; }
QLabel#jsonHint { padding-left: 2px; }
QLabel#sectionTitle {
    color: #0E7C86; font-size: 13px; font-weight: 700;
    padding: 4px 2px 0 2px;
}
QFrame#reportCard {
    background: #FFFFFF; border: 1px solid #CBE4E2; border-radius: 10px;
}
QLabel#resultCard {
    background: #FFFFFF; border: 1px solid #CBE4E2; border-radius: 9px;
}
QLabel#resultCard:hover { border-color: #7FBFBC; }
QTableWidget::item { padding: 4px 6px; }
"""


# --------------------------------------------------------------------------- #
# 现代化页签容器：浅色侧边导航（logo 品牌区 + 分组条目 + 选中胶囊） + 右侧页栈
# --------------------------------------------------------------------------- #
class NavStack(QWidget):
    """侧边导航页签容器。

    视觉：浅色导航面板（与主界面同色系）——logo 品牌区、分组小标题、
    emoji 条目、薄荷绿选中胶囊。
    语义：与 QTabWidget 兼容（addTab/count/tabText/indexOf/currentWidget/
    setCurrentIndex/setCurrentIndex），因此既有渲染与截图代码零改动。
    分组小标题（addSection）占一个导航行但不可选中，行→页通过 ``_map`` 映射。
    """

    LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "resources", "logo.svg")

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ---- 导航面板（品牌区 + 条目列表） ----
        panel = QWidget()
        panel.setObjectName("navPanel")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(14, 12, 8, 0)
        brand_row.setSpacing(9)
        self.lab_logo = QLabel()
        self.lab_logo.setObjectName("navLogo")
        pm = QIcon(self.LOGO_PATH).pixmap(34, 34)
        if not pm.isNull():
            self.lab_logo.setPixmap(pm)
        brand_row.addWidget(self.lab_logo)
        bt = QVBoxLayout()
        bt.setSpacing(0)
        brand = QLabel("SatNetSim")
        brand.setObjectName("navBrand")
        brand_sub = QLabel("卫星互联网一体化仿真器")
        brand_sub.setObjectName("navBrandSub")
        bt.addWidget(brand)
        bt.addWidget(brand_sub)
        brand_row.addLayout(bt, 1)
        brand_row.addStretch(1)
        pv.addLayout(brand_row)

        self.nav = QListWidget()
        self.nav.setObjectName("navList")
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.nav.currentRowChanged.connect(self._on_row)
        pv.addWidget(self.nav, 1)
        lay.addWidget(panel)

        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self._panel = panel
        self._map: List[Optional[int]] = []       # 导航行 → 页索引 / None(分组头)
        self._labels: List[str] = []              # 页索引 → 标题
        panel.setFixedWidth(212)

    def _on_row(self, row: int) -> None:
        if 0 <= row < len(self._map) and self._map[row] is not None:
            self.stack.setCurrentIndex(int(self._map[row]))

    def addSection(self, label: str) -> None:
        """插入不可选中的分组小标题。"""
        it = QListWidgetItem(label)
        it.setFlags(Qt.NoItemFlags)
        it.setForeground(QColor("#5E9E9A"))
        f = QFont()
        f.setPointSize(8)
        f.setBold(True)
        it.setFont(f)
        it.setSizeHint(QSize(0, 30))
        self.nav.addItem(it)
        self._map.append(None)

    # ---------------- QTabWidget 兼容 API ---------------- #
    def addTab(self, w: QWidget, label: str) -> int:
        self._labels.append(label)
        self._map.append(self.stack.count())
        self.nav.addItem(QListWidgetItem(label))
        self.stack.addWidget(w)
        return self.stack.count() - 1

    def count(self) -> int:
        return self.stack.count()

    def tabText(self, i: int) -> str:
        return self._labels[i] if 0 <= i < len(self._labels) else ""

    def indexOf(self, w: QWidget) -> int:
        return self.stack.indexOf(w)

    def currentWidget(self) -> QWidget:
        return self.stack.currentWidget()

    def setCurrentIndex(self, i: int) -> None:
        if i in self._map:
            self.nav.setCurrentRow(self._map.index(i))

    def setCurrentWidget(self, w: QWidget) -> None:
        self.setCurrentIndex(self.stack.indexOf(w))

    def widget(self, i: int) -> QWidget:
        return self.stack.widget(i)


# --------------------------------------------------------------------------- #
# 后台仿真线程
# --------------------------------------------------------------------------- #
class SimWorker(QThread):
    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: SimConfig):
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:  # noqa: D102
        try:
            sim = Simulation(self.cfg)
            res = sim.run(progress=lambda m, f: self.progress.emit(m, f))
            self.done.emit(res)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------------- #
# 绘图容器
# --------------------------------------------------------------------------- #
class PlotCanvas(QWidget):
    """一个带工具栏风格标题的 matplotlib 画布容器。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 4.6), dpi=100, facecolor="#FFFFFF")
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def clear(self):
        self.figure.clear()

    def draw(self):
        self.figure.tight_layout()
        self.canvas.draw_idle()


class MultiPlot(QWidget):
    """2×2 布局的四联图。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(9, 6.6), dpi=100, facecolor="#FFFFFF")
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def draw(self):
        self.figure.tight_layout()
        self.canvas.draw_idle()


def _table(headers: List[str], rows: List[List[str]]) -> QTableWidget:
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            item = QTableWidgetItem(str(val))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            t.setItem(i, j, item)
    t.resizeColumnsToContents()
    t.verticalHeader().setVisible(False)
    return t


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SatNetSim —— 卫星互联网一体化仿真器")
        self.resize(1500, 940)
        self.cfg: SimConfig = preset("starlink-lite")
        # 桌面版默认打开真实评测：⑧ 真实数据传输 / ⑨ 真实计算任务 页签开箱即有数据
        self.cfg.real_eval.enabled = True
        self.result: Optional[SimulationResult] = None
        self.worker: Optional[SimWorker] = None

        self._build_toolbar()
        self._build_ui()
        self.setStyleSheet(QSS)
        self._sync_from_config()
        self.log("欢迎使用 SatNetSim。默认已载入场景「starlink-lite」，点击“开始仿真”即可。")

    # ------------------------------------------------------------------ #
    def _build_toolbar(self) -> None:
        """主工具栏：主行动大按钮 + 分组图标按钮 + 右侧进度区。"""
        tb = self.addToolBar("主工具栏")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        tb.setObjectName("mainToolbar")

        # ---- 主行动：开始/停止（大按钮，薄荷绿主视觉） ----
        self.btn_tb_run = QPushButton("▶  开始仿真")
        self.btn_tb_run.setObjectName("primaryRun")
        self.btn_tb_run.setToolTip("运行完整仿真流水线（Ctrl+R）")
        self.btn_tb_run.clicked.connect(self.run_simulation)
        tb.addWidget(self.btn_tb_run)
        self.act_run = QAction(self)                 # 仅承载快捷键
        self.act_run.setShortcut("Ctrl+R")
        self.act_run.triggered.connect(self.run_simulation)
        self.addAction(self.act_run)
        self.act_stop = QAction("⏹  停止", self)
        self.act_stop.setToolTip("终止当前仿真")
        self.act_stop.triggered.connect(self.stop_simulation)
        self.act_stop.setEnabled(False)
        tb.addAction(self.act_stop)
        tb.addSeparator()

        # ---- 场景文件组 ----
        self.act_save = QAction("💾  保存场景", self)
        self.act_save.setShortcut("Ctrl+S")
        self.act_save.setToolTip("把当前参数保存为场景 JSON（Ctrl+S）")
        self.act_save.triggered.connect(self.save_scene)
        self.act_load = QAction("📂  载入场景", self)
        self.act_load.setShortcut("Ctrl+O")
        self.act_load.setToolTip("从场景文件恢复参数（Ctrl+O）")
        self.act_load.triggered.connect(self.load_scene)
        for a in (self.act_save, self.act_load):
            tb.addAction(a)
        tb.addSeparator()

        # ---- 结果导出组 ----
        self.act_report = QAction("📄  导出报告", self)
        self.act_report.setShortcut("Ctrl+E")
        self.act_report.setToolTip("把文本报告写到文件（Ctrl+E）")
        self.act_report.triggered.connect(self.export_report)
        self.act_tab_png = QAction("🖼  导出图表", self)
        self.act_tab_png.setShortcut("Ctrl+P")
        self.act_tab_png.setToolTip("把当前页签的图表另存为 PNG（Ctrl+P）")
        self.act_tab_png.triggered.connect(self.export_current_tab)
        for a in (self.act_report, self.act_tab_png):
            tb.addAction(a)
        tb.addSeparator()

        # ---- 视图组 ----
        self.act_toggle_side = QAction("📑  参数面板", self)
        self.act_toggle_side.setShortcut("Ctrl+L")
        self.act_toggle_side.setToolTip("折叠/展开左侧参数面板（Ctrl+L）")
        self.act_toggle_side.triggered.connect(self.toggle_sidebar)
        tb.addAction(self.act_toggle_side)

        # ---- 右侧进度区（弹簧推到最右，宽度固定不跳动） ----
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.lab_progress_msg = QLabel("")
        self.lab_progress_msg.setObjectName("progressMsg")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(190)
        self.progress.setRange(0, 100)
        self.progress.setFormat("%p%")
        self.lab_eta = QLabel("ETA —")
        self.lab_eta.setObjectName("etaLabel")
        for w in (self.lab_progress_msg, self.progress, self.lab_eta):
            tb.addWidget(w)

    # ------------------------------------------------------------------ #
    def toggle_sidebar(self) -> None:
        """折叠/展开左侧参数面板（结果页随之占满窗口）。"""
        w = self._left_widget
        visible = not w.isVisible()
        w.setVisible(visible)
        self.act_toggle_side.setText("📑  参数面板" if visible else "📑  展开面板")

    def export_current_tab(self) -> None:
        """把当前页签的图表导出为 PNG（表格/日志页签提示不可导出）。"""
        if self.result is None:
            QMessageBox.information(self, "提示", "请先运行一次仿真。")
            return
        w = self.tabs.currentWidget()
        fig = getattr(w, "figure", None)
        if fig is None:
            QMessageBox.information(self, "提示", "当前页签没有可导出的图表。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出页签图像",
                                              "tab.png", "PNG (*.png)")
        if not path:
            return
        try:
            fig.savefig(path, dpi=180, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            self.log(f"页签图像已导出：{path}")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        split = QSplitter(Qt.Horizontal)

        # ---------------- 左侧参数面板 ---------------- #
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        hv = QVBoxLayout(holder)
        hv.setContentsMargins(2, 2, 6, 2)
        hv.setSpacing(10)

        # 预设
        gb0 = QGroupBox("场景预设")
        f0 = QFormLayout(gb0)
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(PRESETS.keys()))
        self.cmb_preset.setCurrentText("starlink-lite")
        self.cmb_preset.currentTextChanged.connect(self.apply_preset)
        f0.addRow("预设", self.cmb_preset)
        hv.addWidget(gb0)

        # 星座
        gb1 = QGroupBox("星座与轨道")
        f1 = QFormLayout(gb1)
        self.sp_alt = QDoubleSpinBox(); self.sp_alt.setRange(200, 2000); self.sp_alt.setSuffix(" km")
        self.sp_inc = QDoubleSpinBox(); self.sp_inc.setRange(0, 180); self.sp_inc.setSuffix(" °")
        self.sp_planes = QSpinBox(); self.sp_planes.setRange(1, 200)
        self.sp_sats = QSpinBox(); self.sp_sats.setRange(1, 200)
        self.sp_phase = QSpinBox(); self.sp_phase.setRange(0, 200)
        f1.addRow("轨道高度", self.sp_alt)
        f1.addRow("倾角", self.sp_inc)
        f1.addRow("轨道面数", self.sp_planes)
        f1.addRow("每面卫星数", self.sp_sats)
        f1.addRow("相位因子 F", self.sp_phase)
        self.lab_nsat = QLabel("—")
        f1.addRow("卫星总数", self.lab_nsat)
        hv.addWidget(gb1)

        # 仿真时长
        gb2 = QGroupBox("仿真时间")
        f2 = QFormLayout(gb2)
        self.sp_dur = QDoubleSpinBox(); self.sp_dur.setRange(30, 7200); self.sp_dur.setSuffix(" s")
        self.sp_snap = QDoubleSpinBox(); self.sp_snap.setRange(1, 600); self.sp_snap.setSuffix(" s")
        f2.addRow("仿真时长", self.sp_dur)
        f2.addRow("快照间隔 Δt", self.sp_snap)
        hv.addWidget(gb2)

        # 链路
        gb3 = QGroupBox("链路与 ISL 构型")
        f3 = QFormLayout(gb3)
        self.cmb_isl = QComboBox(); self.cmb_isl.addItems(["+Grid", "*Grid", "Grid-V", "none"])
        self.sp_islbw = QDoubleSpinBox(); self.sp_islbw.setRange(0.1, 1000); self.sp_islbw.setSuffix(" Gbps")
        self.sp_islrange = QDoubleSpinBox(); self.sp_islrange.setRange(500, 20000); self.sp_islrange.setSuffix(" km")
        self.sp_gslbw = QDoubleSpinBox(); self.sp_gslbw.setRange(1, 20000); self.sp_gslbw.setSuffix(" Mbps")
        self.sp_elev = QDoubleSpinBox(); self.sp_elev.setRange(0, 80); self.sp_elev.setSuffix(" °")
        self.chk_seam = QCheckBox("在反向缝建立跨面链路")
        f3.addRow("ISL 构型", self.cmb_isl)
        f3.addRow("ISL 带宽", self.sp_islbw)
        f3.addRow("ISL 最大距离", self.sp_islrange)
        f3.addRow("星地带宽", self.sp_gslbw)
        f3.addRow("最小仰角", self.sp_elev)
        f3.addRow("", self.chk_seam)
        hv.addWidget(gb3)

        # 路由 / 传输
        gb4 = QGroupBox("协议选择")
        f4 = QFormLayout(gb4)
        self.cmb_route = QComboBox()
        for k, v in ROUTER_LABELS.items():
            self.cmb_route.addItem(v, k)
        self.cmb_trans = QComboBox()
        for k, meta in PROTOCOL_META.items():
            self.cmb_trans.addItem(str(meta["label"]), k)
        self.cmb_trans_engine = QComboBox()
        for k, v in (("real", "真实包级 TCP 状态机"), ("analytic", "解析模型（RTT 轮次）"),
                     ("both", "两者对照")):
            self.cmb_trans_engine.addItem(v, k)
        self.sp_loss = QDoubleSpinBox(); self.sp_loss.setRange(0, 0.2); self.sp_loss.setDecimals(4)
        f4.addRow("默认路由", self.cmb_route)
        f4.addRow("默认传输", self.cmb_trans)
        f4.addRow("传输仿真引擎", self.cmb_trans_engine)
        f4.addRow("链路丢包率", self.sp_loss)
        hv.addWidget(gb4)

        # 星载计算
        gb5 = QGroupBox("星载计算")
        f5 = QFormLayout(gb5)
        self.sp_satcap = QDoubleSpinBox(); self.sp_satcap.setRange(0.05, 2000); self.sp_satcap.setSuffix(" TOPS")
        self.sp_tasks = QSpinBox(); self.sp_tasks.setRange(10, 5000)
        self.sp_deadline = QDoubleSpinBox(); self.sp_deadline.setRange(0.1, 3600); self.sp_deadline.setSuffix(" s")
        self.cmb_sched = QComboBox()
        for k, v in (("value_density", "价值密度优先"), ("earliest_finish", "最早完成"),
                     ("random", "随机卸载"), ("local_only", "弯管（全地面算）")):
            self.cmb_sched.addItem(v, k)
        f5.addRow("单星算力", self.sp_satcap)
        f5.addRow("任务数", self.sp_tasks)
        f5.addRow("任务期限", self.sp_deadline)
        f5.addRow("默认调度", self.cmb_sched)
        hv.addWidget(gb5)

        # 真实评测
        gb6b = QGroupBox("真实评测（真跑，非解析模型）")
        f6b = QFormLayout(gb6b)
        self.chk_real = QCheckBox("启用真实评测（真跑内核 TCP 与算子，约 10–40 s）")
        self.chk_real_transfer = QCheckBox("真实数据传输（内核 TCP + 包级可靠传输）")
        self.chk_real_compute = QCheckBox("真实计算任务（真跑算子）")
        self.sp_real_payload = QDoubleSpinBox(); self.sp_real_payload.setRange(1, 500)
        self.sp_real_payload.setSuffix(" Mbit")
        self.sp_real_tasks = QSpinBox(); self.sp_real_tasks.setRange(10, 2000)
        self.sp_real_repeats = QSpinBox(); self.sp_real_repeats.setRange(1, 9)
        self.chk_real_analytic = QCheckBox("与解析模型同口径对照")
        f6b.addRow("", self.chk_real)
        f6b.addRow("", self.chk_real_transfer)
        f6b.addRow("", self.chk_real_compute)
        f6b.addRow("TCP 评测载荷", self.sp_real_payload)
        f6b.addRow("真实任务数", self.sp_real_tasks)
        f6b.addRow("负载标定重复", self.sp_real_repeats)
        f6b.addRow("", self.chk_real_analytic)
        hv.addWidget(gb6b)
        # 勾选任一子项即自动打开总开关（否则会出现"勾了子项却没有数据"）
        self.chk_real_transfer.toggled.connect(self._auto_enable_real)
        self.chk_real_compute.toggled.connect(self._auto_enable_real)

        # 真实数据流
        gb6c = QGroupBox("真实数据流仿真（真字节逐包多跳 + 实时回放）")
        f6c = QFormLayout(gb6c)
        self.chk_rdata = QCheckBox("启用真实数据流仿真")
        self.cmb_rdata_kind = QComboBox()
        self.cmb_rdata_kind.addItems(["image", "text", "telemetry", "file"])
        self.sp_rdata_mbit = QDoubleSpinBox(); self.sp_rdata_mbit.setRange(1, 1024)
        self.sp_rdata_mbit.setSuffix(" Mbit")
        self.sp_rdata_flows = QSpinBox(); self.sp_rdata_flows.setRange(1, 6)
        self.sp_rdata_win = QDoubleSpinBox(); self.sp_rdata_win.setRange(0, 3600)
        self.sp_rdata_win.setSuffix(" s"); self.sp_rdata_win.setSpecialValueText("自动")
        self.sp_rdata_mss = QSpinBox(); self.sp_rdata_mss.setRange(256, 8000)
        self.sp_rdata_mss.setSuffix(" B")
        self.chk_rdata_reliable = QCheckBox("端到端可靠传输（真实 TCP 状态机：cwnd/RTO/SACK）")
        f6c.addRow("", self.chk_rdata)
        f6c.addRow("载荷类型", self.cmb_rdata_kind)
        f6c.addRow("载荷大小", self.sp_rdata_mbit)
        f6c.addRow("并发流数", self.sp_rdata_flows)
        f6c.addRow("数据窗口(0=仿真时长)", self.sp_rdata_win)
        f6c.addRow("分片 MSS", self.sp_rdata_mss)
        f6c.addRow("", self.chk_rdata_reliable)
        hv.addWidget(gb6c)

        # 离散事件内核
        gb6d = QGroupBox("离散事件内核（NS-3 式统一事件时间轴）")
        f6d = QFormLayout(gb6d)
        self.chk_des = QCheckBox("启用全局事件编排（叠加在流水线之上，约 1–3 s）")
        self.sp_des_flows = QSpinBox(); self.sp_des_flows.setRange(0, 20)
        self.sp_des_mbit = QDoubleSpinBox(); self.sp_des_mbit.setRange(0.1, 2048)
        self.sp_des_mbit.setSuffix(" Mbit")
        self.cmb_des_algo = QComboBox()
        for k in ("bbr", "cubic", "reno", "hybla", "vegas"):
            self.cmb_des_algo.addItem(k.upper(), k)
        self.sp_des_tasks = QSpinBox(); self.sp_des_tasks.setRange(0, 20000)
        self.chk_des_trace = QCheckBox("记录事件轨迹（⑫ 页签时间线 / JSON 导出）")
        f6d.addRow("", self.chk_des)
        f6d.addRow("TCP 流条数", self.sp_des_flows)
        f6d.addRow("流载荷", self.sp_des_mbit)
        f6d.addRow("流拥塞算法", self.cmb_des_algo)
        f6d.addRow("计算任务数", self.sp_des_tasks)
        f6d.addRow("", self.chk_des_trace)
        hv.addWidget(gb6d)

        # JSON 高级
        gb7 = QGroupBox("高级：完整场景 JSON")
        f7 = QVBoxLayout(gb7)
        self.txt_json = QPlainTextEdit()
        self.txt_json.setMinimumHeight(160)
        self.lab_json_hint = QLabel("")
        self.lab_json_hint.setObjectName("jsonHint")
        self.lab_json_hint.setVisible(False)
        btn_row = QHBoxLayout()
        self.btn_json_apply = QPushButton("应用到场景")
        self.btn_json_apply.setObjectName("ghost")
        self.btn_json_apply.setToolTip("把上方 JSON 解析为场景配置，并同步到左侧各参数")
        self.btn_json_apply.clicked.connect(self.apply_json)
        self.btn_json_fill = QPushButton("从场景刷新")
        self.btn_json_fill.setObjectName("ghost")
        self.btn_json_fill.setToolTip("用当前场景配置重新生成上方 JSON")
        self.btn_json_fill.clicked.connect(self.fill_json_from_scene)
        btn_row.addWidget(self.btn_json_apply); btn_row.addWidget(self.btn_json_fill)
        f7.addWidget(self.txt_json)
        f7.addLayout(btn_row)
        f7.addWidget(self.lab_json_hint)
        hv.addWidget(gb7)
        hv.addStretch(1)

        scroll.setWidget(holder)
        lv.addWidget(scroll, 1)

        run_row = QHBoxLayout()
        run_row.setContentsMargins(2, 4, 2, 2)
        self.btn_run = QPushButton("▶  开始仿真")
        self.btn_run.setObjectName("primaryRun")
        self.btn_run.setToolTip("运行完整仿真流水线（Ctrl+R）")
        self.btn_run.clicked.connect(self.run_simulation)
        self.btn_stop = QPushButton("⏹  停止")
        self.btn_stop.setObjectName("ghost")
        self.btn_stop.setToolTip("终止当前仿真")
        self.btn_stop.clicked.connect(self.stop_simulation)
        self.btn_stop.setEnabled(False)
        run_row.addWidget(self.btn_run, 2)
        run_row.addWidget(self.btn_stop, 1)
        lv.addLayout(run_row)

        # 统一表单呼吸感：所有参数分组的行距/列距
        for gb in self.findChildren(QGroupBox):
            fl = gb.layout()
            if isinstance(fl, QFormLayout):
                fl.setVerticalSpacing(8)
                fl.setHorizontalSpacing(14)

        self._left_widget = left
        split.addWidget(left)

        # ---------------- 右侧结果页 ---------------- #
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)

        # 结果摘要卡片行（运行后填充）
        self.card_row = QHBoxLayout()
        self.card_row.setSpacing(8)
        self._cards: List[QLabel] = []
        card_holder = QWidget()
        card_holder.setLayout(self.card_row)
        card_holder.setMaximumHeight(84)
        rv.addWidget(card_holder)

        self.tabs = NavStack()
        rv.addWidget(self.tabs, 1)

        self.tab_overview = QWidget()
        ov = QVBoxLayout(self.tab_overview)
        ov.setContentsMargins(10, 10, 10, 10)
        ov.setSpacing(8)

        lab_kpi = QLabel("关键指标")
        lab_kpi.setObjectName("sectionTitle")
        ov.addWidget(lab_kpi)
        self.tbl_head = _table(["指标", "数值"], [])
        self.tbl_head.setFrameShape(QFrame.NoFrame)
        ov.addWidget(self.tbl_head)

        lab_rep = QLabel("详细报告")
        lab_rep.setObjectName("sectionTitle")
        ov.addWidget(lab_rep)
        rep_frame = QFrame()
        rep_frame.setObjectName("reportCard")
        rv2 = QVBoxLayout(rep_frame)
        rv2.setContentsMargins(2, 2, 2, 2)
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setFont(QFont("Consolas", 9))
        self.txt_report.setFrameShape(QFrame.NoFrame)
        rv2.addWidget(self.txt_report)
        ov.addWidget(rep_frame, 1)

        # ---- 导航结构：分组小标题 + emoji 条目（无 ①-⑫ 圆圈数字） ----
        self.tabs.addSection("仿真分析")
        self.tabs.addTab(self.tab_overview, "📊  结果概览")

        self.p_const = PlotCanvas()
        self.tabs.addTab(self.p_const, "🌍  三维星座")

        self.p_track = PlotCanvas()
        self.tabs.addTab(self.p_track, "🛰  卫星轨迹")

        self.p_route = MultiPlot()
        self.tabs.addTab(self.p_route, "🔀  路由分析")

        self.p_trans = MultiPlot()
        self.tabs.addTab(self.p_trans, "📶  传输分析")

        self.p_comp = MultiPlot()
        self.tabs.addTab(self.p_comp, "🧮  星载计算")

        # 实时数据流（手绘动画，非 matplotlib）
        from .liveviz import LivePlayer
        self.live = LivePlayer()
        self.tabs.addTab(self.live, "🎬  仿真展示")

        self.tabs.addSection("真实任务")
        self.p_real_trans = MultiPlot()
        self.tabs.addTab(self.p_real_trans, "⚡  传输任务")

        self.p_real_comp = MultiPlot()
        self.tabs.addTab(self.p_real_comp, "🧪  计算任务")

        self.tabs.addSection("实时动态")
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 9))
        self.tabs.addTab(self.txt_log, "📜  运行日志")

        # 全局离散事件内核
        self.p_des = MultiPlot()
        self.tabs.addTab(self.p_des, "⏱  事件内核")
        self.tabs.setCurrentIndex(0)

        split.addWidget(right)
        split.setSizes([400, 1100])
        self.setCentralWidget(split)
        self.setStatusBar(QStatusBar())
        self._status_scale = QLabel("—")
        self.statusBar().addPermanentWidget(self._status_scale)

    # ------------------------------------------------------------------ #
    # 配置同步
    # ------------------------------------------------------------------ #
    def _sync_from_config(self) -> None:
        c = self.cfg
        k = c.constellation
        self.sp_alt.setValue(k.altitude_km)
        self.sp_inc.setValue(k.inclination_deg)
        self.sp_planes.setValue(k.planes)
        self.sp_sats.setValue(k.sats_per_plane)
        self.sp_phase.setValue(k.phase_factor)
        self.lab_nsat.setText(str(k.n_satellites))
        self.sp_dur.setValue(c.duration_s)
        self.sp_snap.setValue(c.snapshot_interval_s)
        self.cmb_isl.setCurrentText(c.link.isl_mode)
        self.sp_islbw.setValue(c.link.isl_bandwidth_mbps / 1000.0)
        self.sp_islrange.setValue(c.link.isl_max_range_km)
        self.sp_gslbw.setValue(c.link.gsl_bandwidth_mbps)
        self.sp_elev.setValue(max((g.min_elevation_deg for g in c.ground_stations),
                                  default=20.0))
        self.chk_seam.setChecked(c.link.isl_seam_links)
        idx = self.cmb_route.findData(c.routing.protocol)
        if idx >= 0:
            self.cmb_route.setCurrentIndex(idx)
        idx = self.cmb_trans.findData(c.transport.protocol)
        if idx >= 0:
            self.cmb_trans.setCurrentIndex(idx)
        idx = self.cmb_trans_engine.findData(c.transport.engine)
        if idx >= 0:
            self.cmb_trans_engine.setCurrentIndex(idx)
        self.sp_loss.setValue(c.transport.loss_prob)
        self.sp_satcap.setValue(c.compute.sat_capacity_tops)
        self.sp_tasks.setValue(c.compute.task_count)
        self.sp_deadline.setValue(c.compute.task_deadline_s)
        idx = self.cmb_sched.findData(c.compute.scheduler)
        if idx >= 0:
            self.cmb_sched.setCurrentIndex(idx)
        re_ = c.real_eval
        self.chk_real.setChecked(re_.enabled)
        self.chk_real_transfer.setChecked(re_.transfer)
        self.chk_real_compute.setChecked(re_.compute)
        self.sp_real_payload.setValue(re_.payload_mbit)
        self.sp_real_tasks.setValue(re_.real_task_count)
        self.sp_real_repeats.setValue(re_.repeats)
        self.chk_real_analytic.setChecked(re_.compare_analytic)
        rd_ = c.real_data
        self.chk_rdata.setChecked(rd_.enabled)
        self.cmb_rdata_kind.setCurrentText(rd_.payload_kind)
        self.sp_rdata_mbit.setValue(rd_.payload_mbit)
        self.sp_rdata_flows.setValue(rd_.n_flows)
        self.sp_rdata_win.setValue(rd_.duration_s)
        self.sp_rdata_mss.setValue(rd_.mss_bytes)
        self.chk_rdata_reliable.setChecked(rd_.reliable)
        de_ = c.des
        self.chk_des.setChecked(de_.enabled)
        self.sp_des_flows.setValue(de_.n_flows)
        self.sp_des_mbit.setValue(de_.flow_mbit)
        idx = self.cmb_des_algo.findData(de_.tcp_algo)
        if idx >= 0:
            self.cmb_des_algo.setCurrentIndex(idx)
        self.sp_des_tasks.setValue(de_.n_tasks)
        self.chk_des_trace.setChecked(de_.trace)
        self.txt_json.setPlainText(json.dumps(c.to_dict(), ensure_ascii=False, indent=2))

    def _auto_enable_real(self, checked: bool) -> None:
        """勾选任一真实评测子项时自动打开总开关，消除"子项已勾却无数据"的困惑。"""
        if checked and not self.chk_real.isChecked():
            self.chk_real.setChecked(True)
            self.log("检测到真实评测子项被勾选，已自动打开「启用真实评测」总开关。")

    def _collect_config(self) -> SimConfig:
        c = self.cfg
        k = c.constellation
        k.altitude_km = self.sp_alt.value()
        k.inclination_deg = self.sp_inc.value()
        k.planes = self.sp_planes.value()
        k.sats_per_plane = self.sp_sats.value()
        k.phase_factor = self.sp_phase.value()
        c.duration_s = self.sp_dur.value()
        c.snapshot_interval_s = self.sp_snap.value()
        c.link.isl_mode = self.cmb_isl.currentText()
        c.link.isl_bandwidth_mbps = self.sp_islbw.value() * 1000.0
        c.link.isl_max_range_km = self.sp_islrange.value()
        c.link.gsl_bandwidth_mbps = self.sp_gslbw.value()
        c.link.isl_seam_links = self.chk_seam.isChecked()
        for g in c.ground_stations:
            g.min_elevation_deg = self.sp_elev.value()
        c.routing.protocol = self.cmb_route.currentData()
        c.transport.protocol = self.cmb_trans.currentData()
        c.transport.engine = self.cmb_trans_engine.currentData()
        c.transport.loss_prob = self.sp_loss.value()
        c.compute.sat_capacity_tops = self.sp_satcap.value()
        c.compute.task_count = self.sp_tasks.value()
        c.compute.task_deadline_s = self.sp_deadline.value()
        c.compute.scheduler = self.cmb_sched.currentData()
        re_ = c.real_eval
        re_.enabled = self.chk_real.isChecked()
        re_.transfer = self.chk_real_transfer.isChecked()
        re_.compute = self.chk_real_compute.isChecked()
        re_.payload_mbit = self.sp_real_payload.value()
        re_.real_task_count = self.sp_real_tasks.value()
        re_.repeats = self.sp_real_repeats.value()
        re_.compare_analytic = self.chk_real_analytic.isChecked()
        rd_ = c.real_data
        rd_.enabled = self.chk_rdata.isChecked()
        rd_.payload_kind = self.cmb_rdata_kind.currentText()
        rd_.payload_mbit = self.sp_rdata_mbit.value()
        rd_.n_flows = self.sp_rdata_flows.value()
        rd_.duration_s = self.sp_rdata_win.value()
        rd_.mss_bytes = self.sp_rdata_mss.value()
        rd_.reliable = self.chk_rdata_reliable.isChecked()
        de_ = c.des
        de_.enabled = self.chk_des.isChecked()
        de_.n_flows = self.sp_des_flows.value()
        de_.flow_mbit = self.sp_des_mbit.value()
        de_.tcp_algo = self.cmb_des_algo.currentData()
        de_.n_tasks = self.sp_des_tasks.value()
        de_.trace = self.chk_des_trace.isChecked()
        self.lab_nsat.setText(str(k.n_satellites))
        self._status_scale.setText(
            f"{k.planes}×{k.sats_per_plane} = {k.n_satellites} 星 · "
            f"{c.duration_s:.0f} s · Δt {c.snapshot_interval_s:.0f} s")
        return c

    # ------------------------------------------------------------------ #
    # 动作
    # ------------------------------------------------------------------ #
    def apply_preset(self, name: str) -> None:
        try:
            self.cfg = preset(name)
            self.cfg.real_eval.enabled = True   # 桌面版默认开启真实评测
            self._sync_from_config()
            self.log(f"已载入预设「{name}」：{self.cfg.constellation.planes}×"
                     f"{self.cfg.constellation.sats_per_plane} = "
                     f"{self.cfg.constellation.n_satellites} 星")
        except Exception as exc:
            self.log(f"[错误] 载入预设失败：{exc}")

    def _json_hint(self, text: str, ok: bool = True) -> None:
        """在 JSON 分组内显示一条即时反馈（数秒后自动淡出）。"""
        color = "#0E7C86" if ok else "#C0392B"
        self.lab_json_hint.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.lab_json_hint.setText(text)
        self.lab_json_hint.setVisible(True)
        QTimer.singleShot(4000, lambda: self.lab_json_hint.setVisible(False))
        # 工具栏状态区也同步一份，避免用户停留在其它页时看不到反馈
        if hasattr(self, "lab_progress_msg"):
            self.lab_progress_msg.setText(text)

    def apply_json(self) -> None:
        raw = self.txt_json.toPlainText().strip()
        if not raw:
            self._json_hint("✗ JSON 为空，无法应用到场景", ok=False)
            QMessageBox.warning(self, "JSON 为空", "请先填入或载入 JSON 场景配置。")
            return
        try:
            data = json.loads(raw)
            cfg = SimConfig.from_dict(data)
            self.cfg = cfg
            self._sync_from_config()
            c = cfg.constellation
            msg = f"✓ 已应用到场景（{c.planes}×{c.sats_per_plane} = {c.n_satellites} 星）"
            self._json_hint(msg, ok=True)
            self.log(msg)
        except Exception as exc:
            self._json_hint(f"✗ 应用失败：{exc}", ok=False)
            QMessageBox.warning(self, "JSON 解析失败", str(exc))

    def fill_json_from_scene(self) -> None:
        try:
            self.txt_json.setPlainText(
                json.dumps(self.cfg.to_dict(), ensure_ascii=False, indent=2))
            self._json_hint("✓ 已从场景刷新 JSON", ok=True)
        except Exception as exc:
            self._json_hint(f"✗ 刷新失败：{exc}", ok=False)

    def save_scene(self) -> None:
        cfg = self._collect_config()
        path, _ = QFileDialog.getSaveFileName(self, "导出场景", "scene.json", "JSON (*.json)")
        if path:
            cfg.to_json(path)
            self.log(f"场景已导出：{path}")

    def load_scene(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "载入场景", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.cfg = SimConfig.from_json(path)
            self._sync_from_config()
            self.log(f"场景已载入：{path}")
        except Exception as exc:
            QMessageBox.warning(self, "载入失败", str(exc))

    def export_report(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "提示", "请先运行一次仿真。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出报告", "report.txt",
                                              "文本 (*.txt);;Markdown (*.md)")
        if not path:
            return
        from .viz import text_report

        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text_report(self.result))
            self.log(f"结果报告已导出：{path}")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    # ------------------------------------------------------------------ #
    def run_simulation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        cfg = self._collect_config()
        self.log("=" * 60)
        self.log(f"开始仿真：{cfg.constellation.n_satellites} 星 / "
                 f"{cfg.duration_s:.0f} s / Δt={cfg.snapshot_interval_s:.0f} s")
        if cfg.real_eval.enabled:
            self.log(f"    真实评测已启用：传输={cfg.real_eval.transfer}（载荷 "
                     f"{cfg.real_eval.payload_mbit:.0f} Mbit）、计算="
                     f"{cfg.real_eval.compute}（{cfg.real_eval.real_task_count} 个真实任务）")
        if cfg.real_data.enabled:
            self.log(f"    真实数据流已启用：{cfg.real_data.payload_kind} "
                     f"{cfg.real_data.payload_mbit:.0f} Mbit × "
                     f"{cfg.real_data.n_flows} 流")
        if cfg.des.enabled:
            self.log(f"    离散事件编排已启用：{cfg.des.n_flows} 条 "
                     f"{cfg.des.tcp_algo.upper()} 流 × "
                     f"{cfg.des.flow_mbit:.0f} Mbit，{cfg.des.n_tasks} 个计算任务"
                     f"{'（含事件轨迹）' if cfg.des.trace else ''}")
        self.progress.setValue(0)
        self.lab_eta.setText("ETA —")
        self.lab_progress_msg.setText("正在仿真…")
        self._t_start_run = time.monotonic()
        self.btn_tb_run.setEnabled(False); self.btn_run.setEnabled(False)
        self.act_stop.setEnabled(True); self.btn_stop.setEnabled(True)

        self.worker = SimWorker(cfg)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def stop_simulation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(1500)
            self.log("[中断] 用户终止了仿真。")
            self._set_idle()

    def _set_idle(self) -> None:
        self.btn_tb_run.setEnabled(True); self.btn_run.setEnabled(True)
        self.act_stop.setEnabled(False); self.btn_stop.setEnabled(False)
        self.lab_progress_msg.setText("")

    def on_progress(self, msg: str, frac: float) -> None:
        self.progress.setValue(int(frac * 100))
        self.statusBar().showMessage(msg)
        if frac > 0.02 and getattr(self, "_t_start_run", None) is not None:
            elapsed = time.monotonic() - self._t_start_run
            eta = elapsed / frac * (1.0 - frac)
            self.lab_eta.setText(f"  已用 {elapsed:.0f}s · ETA {eta:.0f}s")
        self.log(f"  [{frac*100:5.1f}%] {msg}")

    def on_failed(self, text: str) -> None:
        self._set_idle()
        self.log("[失败] " + text)
        QMessageBox.critical(self, "仿真失败", text.split("\n")[0])

    # ------------------------------------------------------------------ #
    def on_done(self, res: SimulationResult) -> None:
        self._set_idle()
        self.result = res
        self.progress.setValue(100)
        self.statusBar().showMessage(f"仿真完成，用时 {res.timings.get('total', 0):.2f} s")
        self.log(f"仿真完成，总用时 {res.timings.get('total', 0):.3f} s；"
                 f"分项 {({k: round(v, 3) for k, v in res.timings.items()})}")
        for w in res.warnings:
            self.log(f"[告警] {w}")
        if res.real_transfer is not None:
            rt = res.real_transfer
            self.log(f"真实数据传输：实测 RTT {rt.measured_rtt_ms:.1f} ms，"
                     f"载荷 {rt.payload_mbit:.1f} Mbit")
            for name, comp, tput, retx, integ in rt.summary_rows():
                self.log(f"    {name:<24}{comp:>9} s{tput:>9} Mbps"
                         f"{('  重传 ' + retx) if retx != '—' else ''}  {integ}")
        if res.real_compute is not None:
            rc = res.real_compute
            self.log(f"真实计算任务：{rc.task_count} 个真实任务，"
                     f"{rc.total_gflops:.1f} GFLOP，"
                     f"本机峰值 {rc.calibration.peak_gflops():.1f} GFLOPS，"
                     f"地面/星载 {rc.speedup:.1f}×")
            self.log(f"    任务画像：{rc.mix_text()}")
        if res.real_data is not None:
            rd = res.real_data
            self.log(f"真实数据流：{rd.payload_label}（{rd.payload_mime}）"
                     f"{rd.payload_bytes/1024:.0f} KiB / {rd.n_segments} 分片")
            self.log(f"    送达 {rd.n_delivered}/{rd.n_packets} 分组，"
                     f"端到端 {rd.e2e_mean_s*1000:.1f} ms（P95 "
                     f"{rd.e2e_p95_s*1000:.1f} ms），{rd.mean_hops:.1f} 跳，"
                     f"{rd.integrity}")
            self.log("    已生成实时回放（⑩ 实时数据流页签，可播放/暂停/拖动时间轴）")
        if res.des is not None:
            d = res.des
            self.log(f"事件内核：{d.n_executed} 个事件 / 时间轴 {d.sim_end_s:.0f} s，"
                     f"流 {d.flows_done}/{len(d.flows)} 完成"
                     f"（均值 {d.flow_goodput_mean_mbps:.1f} Mbps），"
                     f"任务 {len(d.tasks)} 个 / 时限达标 {d.task_deadline_rate*100:.0f}%")
            for n in d.notes:
                self.log(f"    [事件内核] {n}")
        self._fill_cards(res)
        try:
            self._render(res)
            self.log("图表渲染完成。")
        except Exception as exc:  # pragma: no cover
            self.log("[渲染失败] " + str(exc) + "\n" + traceback.format_exc())

    # ------------------------------------------------------------------ #
    # 结果摘要卡片
    # ------------------------------------------------------------------ #
    def _fill_cards(self, res: SimulationResult) -> None:
        cards: List[Tuple[str, str, str]] = []
        best_route = (min(res.routes.values(),
                          key=lambda r: r.mean_delay_ms) if res.routes else None)
        if best_route is not None:
            cards.append(("最优路由", f"{best_route.mean_delay_ms:.1f} ms",
                          best_route.label))
        tcp_ok = [f for f in res.tcp_flows if np.isfinite(f.completion_s)]
        if tcp_ok:
            b = max(tcp_ok, key=lambda f: f.goodput_mbps)
            cards.append(("真实 TCP 最优", f"{b.goodput_mbps:.1f} Mbps",
                          b.protocol.upper()))
        if res.real_data is not None:
            rd = res.real_data
            cards.append(("数据流送达",
                          f"{rd.n_delivered}/{rd.n_packets}",
                          f"{rd.e2e_mean_s*1000:.0f} ms 均时延"))
        if res.des is not None:
            cards.append(("事件内核", f"{res.des.n_executed}",
                          "个事件已执行"))
        if res.compute:
            cr = res.compute.get("value_density")
            if cr is not None:
                cards.append(("星载调度", f"{cr.mean_latency_s:.2f} s",
                              f"超期 {cr.deadline_miss_rate*100:.0f}%"))
        # 重建卡片
        while self.card_row.count():
            it = self.card_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for title, value, note in cards[:6]:
            lab = QLabel(
                f"<div style='padding:8px 14px 7px 14px;'>"
                f"<span style='color:#8A9BA0;font-size:10px;letter-spacing:1px;'>"
                f"{title}</span><br>"
                f"<span style='color:#0E7C86;font-size:20px;font-weight:700;'>"
                f"{value}</span>"
                f"<span style='color:#8A9BA0;font-size:10px;'>　{note}</span>"
                f"</div>")
            lab.setObjectName("resultCard")
            lab.setMinimumHeight(64)
            self.card_row.addWidget(lab, 1)   # 等分铺满整行

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def _render(self, res: SimulationResult) -> None:
        # ① 概览
        rows = [[k, v] for k, v in res.headline()]
        rows.append(["拓扑统计", f"平均 ISL 度 {res.topo_stats.get('mean_isl_degree', 0):.2f}，"
                                f"平均星地度 {res.topo_stats.get('mean_gs_degree', 0):.2f}，"
                                f"平均 ISL 时延 {res.topo_stats.get('mean_isl_delay_ms', 0):.2f} ms"])
        self.tbl_head.setRowCount(len(rows))
        self.tbl_head.setColumnCount(2)
        self.tbl_head.setHorizontalHeaderLabels(["指标", "数值"])
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                it = QTableWidgetItem(str(v))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.tbl_head.setItem(i, j, it)
        self.tbl_head.resizeColumnsToContents()
        self.txt_report.setPlainText(viz.text_report(res))

        # ⑩ 实时数据流（手绘动画）
        try:
            from .liveviz import LiveScene
            if res.real_data is not None:
                times = res.constellation.times_s
                scene = LiveScene.build(res.constellation, res.snapshots,
                                        res.config.ground_stations,
                                        res.real_data, times)
                self.live.set_scene(scene)
            else:
                self.live.set_scene(None)
        except Exception as exc:  # pragma: no cover
            self.log(f"[实时回放] 构建场景失败：{exc}")

        # ② 星座三维
        self.p_const.clear()
        ax = self.p_const.figure.add_subplot(111, projection="3d")
        viz.plot_constellation_3d(ax, res.constellation,
                                 snapshot_index=len(res.snapshots) // 2,
                                 ground_stations=res.config.ground_stations,
                                 highlight_path=res.representative_path)
        self.p_const.draw()

        # ③ 星下点轨迹
        self.p_track.clear()
        ax = self.p_track.figure.add_subplot(111)
        viz.plot_ground_tracks(ax, res.constellation, res.config.ground_stations)
        self.p_track.draw()

        # ④ 路由分析（协议对比 + 效果评测 + 路由×传输）
        self.p_route.figure.clear()
        for i, fn in enumerate((
            lambda a: viz.plot_route_metrics(a, res.routes),
            lambda a: viz.plot_route_eval(a, res.routes),
            lambda a: viz.route_eval_table(a, res.routes),
            lambda a: viz.plot_route_cdf(a, res.routes),
            lambda a: viz.plot_route_switch_timeline(a, res.routes),
            lambda a: viz.plot_route_transport(a, res.routes),
            lambda a: self._plot_hops(a, res),
            lambda a: self._plot_degree(a, res),
        )):
            fn(self.p_route.figure.add_subplot(2, 4, i + 1))
        self.p_route.figure.set_size_inches(11.0, 5.4)
        self.p_route.draw()

        # ⑤ 传输分析（真实包级 TCP 状态机 + 解析模型对照）
        self.p_trans.figure.clear()
        pick = res.tcp_flows[1] if len(res.tcp_flows) > 1 else (
            res.tcp_flows[0] if res.tcp_flows else None)
        for i, fn in enumerate((
            lambda a: viz.plot_tcp_compare(a, res.tcp_flows, res.transport, res.tcp_capacity),
            lambda a: viz.plot_tcp_cwnd(a, pick),
            lambda a: viz.plot_tcp_profile(a, res.tcp_profile),
            lambda a: viz.tcp_table(a, res.tcp_flows, res.tcp_capacity),
            lambda a: viz.plot_tcp_loss_sweep(a, res.tcp_sweep, res.tcp_sweep_points),
            lambda a: viz.plot_transport_compare(a, res.transport),
        )):
            fn(self.p_trans.figure.add_subplot(2, 3, i + 1))
        self.p_trans.draw()

        # ⑥ 星载计算
        self.p_comp.figure.clear()
        ax1 = self.p_comp.figure.add_subplot(221)
        viz.plot_compute_compare(ax1, res.compute)
        ax2 = self.p_comp.figure.add_subplot(222)
        if res.contention[0]:
            viz.plot_contention_sweep(ax2, res.contention[0], res.contention[1])
        else:
            viz.plot_contention_sweep(ax2, [], {})
        ax3 = self.p_comp.figure.add_subplot(223)
        self._plot_latency_cdf(ax3, res)
        ax4 = self.p_comp.figure.add_subplot(224)
        self._plot_downlink(ax4, res)
        self.p_comp.draw()

        # 传输任务（真实链路标定）
        self.p_real_trans.figure.clear()
        rt = res.real_transfer
        ax1 = self.p_real_trans.figure.add_subplot(221)
        viz.plot_real_transfer_compare(ax1, rt)
        ax2 = self.p_real_trans.figure.add_subplot(222)
        viz.plot_real_cwnd(ax2, rt)
        ax3 = self.p_real_trans.figure.add_subplot(223)
        self._plot_real_transfer_table(ax3, rt)
        ax4 = self.p_real_trans.figure.add_subplot(224)
        self._plot_real_profile(ax4, rt)
        self.p_real_trans.draw()

        # 计算任务（真实算子标定）
        self.p_real_comp.figure.clear()
        rc = res.real_compute
        ax1 = self.p_real_comp.figure.add_subplot(221)
        viz.plot_real_workload_time(ax1, rc)
        ax2 = self.p_real_comp.figure.add_subplot(222)
        viz.plot_real_compute_split(ax2, rc)
        ax3 = self.p_real_comp.figure.add_subplot(223)
        self._plot_real_workload_table(ax3, rc)
        ax4 = self.p_real_comp.figure.add_subplot(224)
        self._plot_real_volume(ax4, rc)
        self.p_real_comp.draw()

        # 全局离散事件内核
        self.p_des.figure.clear()
        des = res.des
        for i, fn in enumerate((
            lambda a: viz.plot_des_timeline(a, des),
            lambda a: viz.plot_des_events(a, des),
            lambda a: viz.plot_des_queue(a, des),
            lambda a: viz.des_table(a, des),
        )):
            fn(self.p_des.figure.add_subplot(2, 2, i + 1))
        self.p_des.draw()

    # ---------------- 真实评测附加子图 ---------------- #
    def _plot_real_transfer_table(self, ax, rt) -> None:
        ax.axis("off")
        if rt is None:
            ax.set_title("未启用真实数据传输评测", fontsize=10, color=viz.C_MUTED)
            return
        rows = [[a, b, c, d, e] for a, b, c, d, e in rt.summary_rows()]
        table = ax.table(
            cellText=rows or [["—"] * 5],
            colLabels=["方法", "完成(s)", "吞吐(Mbps)", "重传", "完整性"],
            colWidths=[0.30, 0.13, 0.17, 0.09, 0.31],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1.0, 1.3)
        ax.set_title("真实传输 vs 解析模型（同口径）", fontsize=10, color="#111827")

    def _plot_real_profile(self, ax, rt) -> None:
        ax.axis("off")
        if rt is None:
            ax.set_title("未启用真实数据传输评测", fontsize=10, color=viz.C_MUTED)
            return
        # describe() 一行较长，折行避免溢出面板右缘
        import textwrap

        lines: List[str] = []
        for seg in rt.profile.describe().split("\n"):
            wrapped = textwrap.wrap(seg, 46,
                                    subsequent_indent="    ") or [""]
            lines.extend(wrapped)
        lines.append("")
        lines.append(f"实测 RTT（探针多次取中位）: {rt.measured_rtt_ms:.2f} ms")
        lines.append(f"理论 RTT（2×OWD）        : {rt.profile.rtt_ms:.2f} ms")
        lines.append(f"评测载荷                 : {rt.payload_mbit:.1f} Mbit（真实字节）")
        if rt.packet_profile is not None:
            lines.append(f"包级链路                 : "
                         f"{rt.packet_profile.bandwidth_mbps:.0f} Mbps / "
                         f"BDP {rt.packet_profile.bdp_bytes/1024:.0f} KB")
            lines.append(f"包级载荷                 : {rt.packet_payload_mbit:.0f} Mbit")
        lines.append(f"测试床可靠容量上界       : ~{rt.harness_cap_mbps:.0f} Mbps")
        lines.append(f"评测墙钟                 : {rt.wall_s:.1f} s")
        if rt.packet:
            best = rt.best_packet()
            if best is not None:
                lines.append("")
                lines.append(f"最优包级：{best.label} {best.goodput_mbps:.1f} Mbps / "
                             f"重传 {best.retransmits} / {best.integrity}")
        ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=6.8, va="top", ha="left",
                family=viz.cjk_font_family(),   # 不能用 Consolas：无 CJK 字形
                color="#111827", linespacing=1.5)
        ax.set_title("链路剖面与实测口径", fontsize=10, color="#111827")

    def _plot_real_workload_table(self, ax, rc) -> None:
        ax.axis("off")
        if rc is None:
            ax.set_title("未启用真实计算评测", fontsize=10, color=viz.C_MUTED)
            return
        rows = [[a, b, c, d, e] for a, b, c, d, e in rc.summary_rows()]
        table = ax.table(
            cellText=rows or [["—"] * 5],
            colLabels=["真实负载", "耗时(ms)", "吞吐", "GFLOPS", "真实产出"],
            colWidths=[0.30, 0.14, 0.22, 0.14, 0.20],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.0)
        table.scale(1.0, 1.25)
        ax.set_title("真实负载实测（本机真实执行）", fontsize=10, color="#111827")

    def _plot_real_volume(self, ax, rc) -> None:
        if rc is None:
            viz._placeholder(ax, "未运行真实计算评测")
            return
        names = ["计算量\n(GFLOP)", "真实输入\n(Mbit)", "弯管原始\n(Mbit)"]
        vals = [max(rc.total_gflops, 1e-3),
                max(rc.total_input_mbit, 1e-3),
                max(rc.total_raw_mbit, 1e-3)]
        x = np.arange(3)
        ax.bar(x, vals, 0.55, color=[viz.C_PRIMARY, viz.C_ACCENT, viz.C_WARN],
               alpha=0.9)
        for xx, v in zip(x, vals):
            ax.text(xx, v * 1.15, f"{v:,.1f}", ha="center", fontsize=7.5,
                    color="#111827")
        ax.set_yscale("log")
        ax.set_ylim(top=max(vals) * 6)   # 顶部留白，避免标注顶到标题
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("量（对数轴）", fontsize=8)
        ax.set_title(f"真实任务流总量（{rc.task_count} 个任务）",
                     fontsize=10, color="#111827")
        viz._style(ax)

    # ---------------- 附加子图 ---------------- #
    def _plot_hops(self, ax, res) -> None:
        for k, ra in enumerate(res.routes.values()):
            ax.plot(range(len(ra.hops)), ra.hops, lw=1.3,
                    color=viz.PALETTE[k % len(viz.PALETTE)], label=ra.label)
        ax.set_xlabel("快照序号", fontsize=8)
        ax.set_ylabel("跳数", fontsize=8)
        ax.set_title("路径跳数变化", fontsize=10, color="#111827")
        ax.legend(fontsize=7)
        viz._style(ax)

    def _plot_degree(self, ax, res) -> None:
        degs = []
        for s in res.snapshots:
            cover = np.bincount(s.src, minlength=s.n_nodes) + np.bincount(
                s.dst, minlength=s.n_nodes)
            degs.append(cover[: res.constellation.n_sat])
        arr = np.concatenate(degs) if degs else np.zeros(1)
        ax.hist(arr, bins=np.arange(-0.5, arr.max() + 1.5, 1.0),
                color=viz.C_PRIMARY, alpha=0.85)
        ax.set_xlabel("节点度（ISL 邻居数）", fontsize=8)
        ax.set_ylabel("计数", fontsize=8)
        ax.set_title("卫星节点度分布", fontsize=10, color="#111827")
        viz._style(ax)

    def _plot_latency_cdf(self, ax, res) -> None:
        for k, rep in enumerate(res.compute.values()):
            lat = np.sort([r.latency_s for r in rep.records])
            if lat.size:
                ax.plot(lat, np.linspace(0, 1, lat.size), lw=1.4,
                        color=viz.PALETTE[k % len(viz.PALETTE)], label=rep.policy)
        ax.set_xlabel("端到端时延 (s)", fontsize=8)
        ax.set_ylabel("累积概率", fontsize=8)
        ax.set_title("任务时延 CDF", fontsize=10, color="#111827")
        ax.legend(fontsize=7)
        viz._style(ax)

    def _plot_downlink(self, ax, res) -> None:
        labels = list(res.compute.keys())
        dl = [res.compute[k].downlink_mbit for k in labels]
        tt = [res.compute[k].throughput_data_mbit for k in labels]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, dl, 0.4, color=viz.C_WARN, label="星地下行总量")
        ax.bar(x + 0.2, tt, 0.4, color=viz.C_PRIMARY, label="有效输出数据")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=12)
        ax.set_ylabel("数据量 (Mbit，对数轴)", fontsize=8)
        ax.set_title("下行数据量对比（弯管 vs OEC）", fontsize=10, color="#111827")
        ax.legend(fontsize=7)
        viz._style(ax)

    # ------------------------------------------------------------------ #
    def log(self, msg: str) -> None:
        self.txt_log.append(msg)
        self.txt_log.verticalScrollBar().setValue(
            self.txt_log.verticalScrollBar().maximum()
        )


# --------------------------------------------------------------------------- #
def app_logo() -> QIcon:
    """应用 logo（窗口图标 / 任务栏图标）。"""
    return QIcon(NavStack.LOGO_PATH)


def main() -> int:
    app = QApplication.instance() or QApplication([])
    app.setWindowIcon(app_logo())
    win = MainWindow()
    win.setWindowIcon(app_logo())
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

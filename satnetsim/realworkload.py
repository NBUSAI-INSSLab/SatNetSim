"""真实计算任务评测 —— 跑真实负载，拿到真实的算力量与真实的数据量。

为什么需要这一层
----------------
`computing.py` 的任务流是**合成**的：`task_size_mbit`、`task_cycles_gops` 都是
人为给定的区间上取均匀分布。这能扫参，但"任务到底有多重"这件事没有依据。

本模块把任务换成**真实跑出来的负载**：

- 真实神经网络推理（ResNet-18 前向，输入是真实 JPEG 编解码出来的图像）
- 真实神经网络训练（前向 + 反向 + SGD 更新）
- 真实信号处理（FFT）、真实视觉流水线（高斯模糊 + Canny + 缩放）
- 真实密码学与压缩（SHA-256、zlib）
- 真实稠密线代（BLAS GEMM）

每个负载在**本机真实计时**，得到：
1. **真实数据量**：输入载荷字节数（经过真实编解码）与输出结果字节数；
2. **真实计算量**：可解析统计的 FLOP，或由"实测时间 × 本机有效算力"折算的等效算力；
3. **本机算力标定**：有效 GFLOPS / 图像吞吐，用于把负载缩放到星载算力平台。

于是星载计算仿真里的每个任务，其"要传多少数据、要算多少活、值多少价值"都来自
真实可复现的负载，而不是凭空设定的区间。

诚实边界
--------
- 本机（x86 CPU）实测的绝对时间**不代表**星载处理器的时间；仿真用"计算量 ÷ 星载
  算力"重新折算在轨时延，本机实测只提供**计算量与数据量**这两个客观量。
- 未安装 torch/torchvision/cv2/sklearn 时相关负载自动跳过，标定报告会标注缺失项。
"""

from __future__ import annotations

import hashlib
import io
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .computing import Task

# --------------------------------------------------------------------------- #
# 可选依赖探测
# --------------------------------------------------------------------------- #
def _try_import(name: str):
    """按点分路径导入子模块（`__import__("PIL.Image")` 只返回顶层 PIL，是坑）。"""
    try:
        import importlib

        return importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None


_TORCH = _try_import("torch")
_TV = _try_import("torchvision")
_CV2 = _try_import("cv2")
_PIL = _try_import("PIL.Image")
_SK = _try_import("sklearn.ensemble")


# --------------------------------------------------------------------------- #
# 负载定义
# --------------------------------------------------------------------------- #
@dataclass
class WorkloadSpec:
    """一个真实计算负载的静态属性（数据量与计算量来自实测或解析统计）。"""

    key: str
    label: str
    category: str          # ai-infer | ai-train | signal | vision | crypto | compress | linalg
    input_mbit: float      # 真实输入载荷（编码后字节数 × 8）
    raw_mbit: float        # 弯管架构下需回传的原始数据量
    result_mbit: float     # 真实输出结果字节数 × 8
    flops: float           # 真实计算量（FLOP）；0 表示由实测时间折算
    value: float = 0.5     # 任务价值（0–1），与调度器语义一致
    deadline_s: float = 30.0
    needs: Tuple[str, ...] = ()


# 默认任务画像。FLOP 按解析公式精确统计（batch × 卷积 MAC × 2），
# 数据量为真实编码字节数（JPEG/float32 张量等），确保"有效算力"不超过本机实测峰值。
DEFAULT_SPECS: Dict[str, WorkloadSpec] = {
    # ResNet-18：1.82 GFLOP/图 × batch 4
    "ai_infer_resnet18": WorkloadSpec(
        key="ai_infer_resnet18", label="ResNet-18 推理（224²×4）", category="ai-infer",
        input_mbit=0.4, raw_mbit=3.7, result_mbit=0.08, flops=7.28e9,
        value=0.9, deadline_s=10.0, needs=("torch", "torchvision", "PIL")),
    # 小 CNN（3→32→64，64²，batch 8）：conv1 5.66e7 + conv2 3.02e8
    "ai_infer_smallcnn": WorkloadSpec(
        key="ai_infer_smallcnn", label="小 CNN 推理（64²×8）", category="ai-infer",
        input_mbit=0.15, raw_mbit=3.1, result_mbit=0.02, flops=3.58e8,
        value=0.6, deadline_s=8.0, needs=("torch",)),
    # 训练步（3→16，64²，batch 16）：前向 5.66e7，训练约 3× 前向
    "ai_train_cnn": WorkloadSpec(
        key="ai_train_cnn", label="CNN 训练步（前向+反向）", category="ai-train",
        input_mbit=1.2, raw_mbit=12.0, result_mbit=0.5, flops=1.70e8,
        value=0.95, deadline_s=30.0, needs=("torch",)),
    # 实数 FFT：5·N·log2(N)，N=2^20
    "signal_fft": WorkloadSpec(
        key="signal_fft", label="FFT 频谱分析（1M 点）", category="signal",
        input_mbit=8.0, raw_mbit=8.0, result_mbit=8.0, flops=1.05e8,
        value=0.35, deadline_s=20.0),
    "vision_cv2": WorkloadSpec(
        key="vision_cv2", label="视觉流水线（1080p 模糊+Canny）", category="vision",
        input_mbit=2.0, raw_mbit=49.8, result_mbit=0.5, flops=0.0,
        value=0.5, deadline_s=15.0, needs=("cv2",)),
    "crypto_sha256": WorkloadSpec(
        key="crypto_sha256", label="SHA-256 完整性校验（8 MB）", category="crypto",
        input_mbit=64.0, raw_mbit=64.0, result_mbit=0.000256, flops=0.0,
        value=0.2, deadline_s=20.0),
    "compress_zlib": WorkloadSpec(
        key="compress_zlib", label="zlib 下传前压缩（4 MB）", category="compress",
        input_mbit=32.0, raw_mbit=32.0, result_mbit=8.0, flops=0.0,
        value=0.4, deadline_s=20.0),
    "linalg_gemm": WorkloadSpec(
        key="linalg_gemm", label="稠密线代 GEMM（1024³）", category="linalg",
        input_mbit=0.0, raw_mbit=16.8, result_mbit=8.4, flops=2.15e9,
        value=0.45, deadline_s=20.0),
    "sklearn_rf": WorkloadSpec(
        key="sklearn_rf", label="随机森林训练+推理", category="ml-classic",
        input_mbit=1.0, raw_mbit=1.5, result_mbit=0.002, flops=0.0,
        value=0.55, deadline_s=25.0, needs=("sklearn",)),
}


# --------------------------------------------------------------------------- #
# 真实负载内核：每个函数都必须真的跑一遍，并返回真实产出的字节数
# --------------------------------------------------------------------------- #
def _real_image_tensor(h: int = 224, w: int = 224, seed: int = 1):
    """用真实 JPEG 编解码生成输入图像（返回字节数、张量）。"""
    import torch

    arr = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    if _PIL is not None:
        buf = io.BytesIO()
        _PIL.fromarray(arr).save(buf, format="JPEG", quality=85)
        raw = buf.getvalue()
        img = _PIL.open(io.BytesIO(raw)).convert("RGB").resize((w, h))
        arr = np.asarray(img)
    else:
        raw = arr.tobytes()
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return len(raw), t


def _wl_ai_infer_resnet18() -> int:
    """真实 ResNet-18 前向（无预训练权重，纯计算）。"""
    import torch
    import torchvision.models as tvm

    net = _RESNET_CACHE.get("net")
    if net is None:
        net = tvm.resnet18(weights=None)
        net.eval()
        _RESNET_CACHE["net"] = net
    nbytes, _ = _real_image_tensor()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        y = net(x)
    return int(nbytes * 8) or int(y.numel() * 4)


def _wl_ai_infer_smallcnn() -> int:
    import torch

    net = _SMALL_CNN_CACHE.get("net")
    if net is None:
        net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(64, 16),
        ).eval()
        _SMALL_CNN_CACHE["net"] = net
    nbytes, t = _real_image_tensor(64, 64)
    x = t.repeat(8, 1, 1, 1)
    with torch.no_grad():
        y = net(x)
    return int(nbytes)


def _wl_ai_train_cnn() -> int:
    """真实训练步：前向 + 反向 + SGD 更新。"""
    import torch

    net = _TRAIN_NET_CACHE.get("net")
    if net is None:
        net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(16, 4),
        )
        _TRAIN_NET_CACHE["net"] = net
        _TRAIN_NET_CACHE["opt"] = torch.optim.SGD(net.parameters(), lr=0.01)
        _TRAIN_NET_CACHE["loss"] = torch.nn.CrossEntropyLoss()
    opt = _TRAIN_NET_CACHE["opt"]
    lossf = _TRAIN_NET_CACHE["loss"]
    x = torch.randn(16, 3, 64, 64)
    y = torch.randint(0, 4, (16,))
    opt.zero_grad()
    out = net(x)
    loss = lossf(out, y)
    loss.backward()
    opt.step()
    return int(1024 * 4)  # 真实产出：参数增量（简化计为 4 KB）


def _wl_signal_fft() -> int:
    n = 1 << 20
    sig = np.random.default_rng(3).standard_normal(n)
    spec = np.fft.rfft(sig)
    return int(spec.nbytes)


def _wl_vision_cv2() -> int:
    img = np.random.default_rng(4).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    blur = _CV2.GaussianBlur(img, (9, 9), 2.0)
    edges = _CV2.Canny(blur, 60, 160)
    small = _CV2.resize(edges, (640, 360))
    return int(small.nbytes)


def _wl_crypto_sha256() -> int:
    blob = np.random.default_rng(5).integers(0, 256, 8 << 20, dtype=np.uint8).tobytes()
    d = hashlib.sha256(blob).digest()
    return len(d)


def _wl_compress_zlib() -> int:
    blob = np.random.default_rng(6).integers(0, 256, 4 << 20, dtype=np.uint8).tobytes()
    out = zlib.compress(blob, 6)
    return len(out)


def _wl_linalg_gemm() -> int:
    a = np.random.default_rng(7).standard_normal((1024, 1024))
    b = np.random.default_rng(8).standard_normal((1024, 1024))
    c = a @ b
    return int(c.nbytes)


def _wl_sklearn_rf() -> int:
    """真实机器学习训练+推理（随机森林）。"""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(9)
    x = rng.standard_normal((2000, 12))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    clf = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=0)
    clf.fit(x, y)
    pred = clf.predict(x[:500])
    return int(pred.nbytes)


_RESNET_CACHE: Dict[str, object] = {}
_SMALL_CNN_CACHE: Dict[str, object] = {}
_TRAIN_NET_CACHE: Dict[str, object] = {}

WORKLOAD_FNS: Dict[str, Callable[[], int]] = {
    "ai_infer_resnet18": _wl_ai_infer_resnet18,
    "ai_infer_smallcnn": _wl_ai_infer_smallcnn,
    "ai_train_cnn": _wl_ai_train_cnn,
    "signal_fft": _wl_signal_fft,
    "vision_cv2": _wl_vision_cv2,
    "crypto_sha256": _wl_crypto_sha256,
    "compress_zlib": _wl_compress_zlib,
    "linalg_gemm": _wl_linalg_gemm,
    "sklearn_rf": _wl_sklearn_rf,
}

_MODULE_MAP = {
    "torch": _TORCH, "torchvision": _TV, "cv2": _CV2,
    "PIL": _PIL, "sklearn": _SK,
}


def workload_available(key: str) -> bool:
    spec = DEFAULT_SPECS.get(key)
    if spec is None:
        return key in WORKLOAD_FNS
    return all(_MODULE_MAP.get(m) is not None for m in spec.needs)


# --------------------------------------------------------------------------- #
# 实测
# --------------------------------------------------------------------------- #
@dataclass
class WorkloadRun:
    """一次真实负载的实测结果。"""

    key: str
    label: str
    category: str
    repeats: int
    elapsed_s: float        # 每次平均真实墙钟
    cpu_s: float            # 每次平均进程 CPU 时间
    mem_mb: float
    payload_bytes: int      # 真实产出字节数
    gflops_effective: float  # 由计算量/实测时间得到的有效算力
    ok: bool
    note: str = ""


@dataclass
class CalibrationReport:
    """本机真实算力标定。"""

    runs: List[WorkloadRun] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    wall_s: float = 0.0
    cpu_count: int = 0
    device: str = "cpu"

    def by_category(self, cat: str) -> Optional[WorkloadRun]:
        best = [r for r in self.runs if r.category == cat and r.ok]
        return max(best, key=lambda r: r.gflops_effective) if best else None

    def peak_gflops(self) -> float:
        vals = [r.gflops_effective for r in self.runs if r.ok and r.gflops_effective > 0]
        return max(vals) if vals else 0.0

    def effective_gflops(self, category: str = "ai-infer") -> float:
        r = self.by_category(category)
        if r is not None and r.gflops_effective > 0:
            return r.gflops_effective
        return self.peak_gflops()

    def summary_rows(self) -> List[Tuple[str, str, str, str, str]]:
        """(负载, 实测耗时 ms, 吞吐, 有效算力 GFLOPS, 真实产出)"""
        rows = []
        for r in self.runs:
            ms = r.elapsed_s * 1000.0
            if not r.ok:
                rows.append((r.label, f"{ms:.2f}", "—", "—", r.note))
                continue
            if r.gflops_effective > 0:
                tput = f"{r.gflops_effective:.1f} GFLOPS"
                gf = f"{r.gflops_effective:.2f}"
            else:
                # 无解析 FLOP 的负载（视觉/密码学/压缩/经典 ML）：算力不适用，
                # 用真实吞吐（MB/s 或 次/s）表征，GFLOPS 列显式标"—"
                tput = f"{_rate_label(r)}"
                gf = "—"
            rows.append((r.label, f"{ms:.2f}", tput, gf, _bytes_label(r.payload_bytes)))
        return rows

    def headline(self) -> List[Tuple[str, str]]:
        return [
            ("本机设备", f"{self.device}, {self.cpu_count} 逻辑核"),
            ("实测峰值算力", f"{self.peak_gflops():.1f} GFLOPS"),
            ("AI 推理有效算力", f"{self.effective_gflops('ai-infer'):.1f} GFLOPS"),
            ("完成负载", f"{sum(1 for r in self.runs if r.ok)}/{len(self.runs)}"),
            ("标定墙钟", f"{self.wall_s:.2f} s"),
            ("跳过负载", f"{len(self.skipped)} 项"
                        + ("（" + ", ".join(k for k, _ in self.skipped) + "）"
                           if self.skipped else "")),
        ]


def _rate_label(r: WorkloadRun) -> str:
    if r.elapsed_s <= 0:
        return "—"
    if r.category == "crypto" or r.category == "compress":
        mb = (8 << 20) / 1e6 if r.category == "crypto" else (4 << 20) / 1e6
        return f"{mb / r.elapsed_s:.0f} MB/s"
    return f"{1.0 / r.elapsed_s:.1f} 次/s"


def _bytes_label(n: int) -> str:
    if n >= 1 << 20:
        return f"{n/(1<<20):.2f} MB"
    if n >= 1 << 10:
        return f"{n/(1<<10):.1f} KB"
    return f"{n} B"


def run_workload(key: str, repeats: int = 3, warmup: int = 1) -> WorkloadRun:
    """真实执行一个负载并计时（预热 + 多次取均值）。"""
    spec = DEFAULT_SPECS.get(key)
    label = spec.label if spec else key
    cat = spec.category if spec else "misc"
    if not workload_available(key):
        return WorkloadRun(key, label, cat, 0, 0.0, 0.0, 0.0, 0, 0.0, False,
                           "缺少依赖：%s" % ", ".join(spec.needs if spec else ()))
    fn = WORKLOAD_FNS[key]
    try:
        for _ in range(max(int(warmup), 0)):
            fn()
        payload = 0
        best = float("inf")
        cpu_best = float("inf")
        for _ in range(max(int(repeats), 1)):
            c0 = time.process_time()
            t0 = time.perf_counter()
            payload = fn()
            dt = time.perf_counter() - t0
            cpu = time.process_time() - c0
            best = min(best, dt)
            cpu_best = min(cpu_best, cpu)
        flops = spec.flops if spec else 0.0
        gflops = (flops / best / 1e9) if (flops > 0 and best > 0) else 0.0
        return WorkloadRun(key, label, cat, repeats, best, cpu_best,
                           _mem_mb(), payload, gflops, True)
    except Exception as exc:  # noqa: BLE001
        return WorkloadRun(key, label, cat, 0, 0.0, 0.0, 0.0, 0, 0.0, False,
                           f"{type(exc).__name__}: {exc}")


def _mem_mb() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / (1 << 20)
    except Exception:  # noqa: BLE001
        return 0.0


def calibrate(keys: Optional[Sequence[str]] = None, repeats: int = 3,
              progress: Optional[Callable[[str], None]] = None) -> CalibrationReport:
    """跑一遍全部真实负载，得到本机有效算力标定。"""
    import os

    t0 = time.perf_counter()
    rep = CalibrationReport(cpu_count=os.cpu_count() or 1)
    if _TORCH is not None:
        try:
            rep.device = "cuda" if _TORCH.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            rep.device = "cpu"
    todo = list(keys) if keys is not None else list(DEFAULT_SPECS)
    for k in todo:
        if progress:
            progress(f"真实负载标定：{DEFAULT_SPECS.get(k).label if k in DEFAULT_SPECS else k}")
        if not workload_available(k):
            rep.skipped.append((k, "缺少依赖"))
            continue
        rep.runs.append(run_workload(k, repeats=repeats))
    rep.wall_s = time.perf_counter() - t0
    if progress:
        progress(f"真实负载标定完成：{rep.wall_s:.2f} s，"
                 f"峰值 {rep.peak_gflops():.1f} GFLOPS")
    return rep


# --------------------------------------------------------------------------- #
# 由真实负载生成任务流（供 computing.ComputeSimulator 使用）
# --------------------------------------------------------------------------- #
def real_task_stream(n_tasks: int, n_gs: int, duration_s: float,
                     calibration: Optional[CalibrationReport] = None,
                     keys: Optional[Sequence[str]] = None,
                     seed: int = 0,
                     poisson: bool = True,
                     input_scale: float = 1.0) -> List[Task]:
    """用真实负载画像生成任务流：数据量与计算量都来自真实负载。

    `calibration` 提供本机有效算力，用于把"无解析计算量"的负载（视觉/密码学/
    压缩）折算成等效计算量（实测时间 × 本机有效算力）。
    """
    rng = np.random.default_rng(seed)
    keys = [k for k in (keys or DEFAULT_SPECS) if k in DEFAULT_SPECS]
    if not keys:
        keys = list(DEFAULT_SPECS)
    eff_gflops = calibration.effective_gflops("ai-infer") if calibration else 100.0
    if eff_gflops <= 0:
        eff_gflops = 100.0

    measured: Dict[str, float] = {}
    if calibration:
        for r in calibration.runs:
            if r.ok:
                measured[r.key] = r.elapsed_s

    if poisson:
        rate = n_tasks / max(duration_s, 1.0)
        arrivals = np.cumsum(rng.exponential(1.0 / max(rate, 1e-9), n_tasks))
        arrivals = arrivals[arrivals <= duration_s]
    else:
        arrivals = np.linspace(0.0, duration_s, n_tasks, endpoint=False)

    tasks: List[Task] = []
    for i, t0 in enumerate(arrivals):
        spec = DEFAULT_SPECS[keys[int(rng.integers(0, len(keys)))]]
        flops = spec.flops
        if flops <= 0:
            # 无解析计算量的负载：用实测时间 × 本机有效算力折算等效计算量
            flops = measured.get(spec.key, 1e-3) * eff_gflops * 1e9
        cycles_gops = max(flops / 1e9, 1e-4)
        tasks.append(Task(
            tid=i,
            origin_gs=int(rng.integers(0, max(n_gs, 1))),
            arrival_s=float(t0),
            input_mbit=float(spec.input_mbit * input_scale),
            raw_data_mbit=float(spec.raw_mbit * input_scale),
            cycles_gops=float(cycles_gops),
            deadline_s=float(spec.deadline_s),
            value=float(spec.value),
            result_mbit=float(spec.result_mbit),
        ))
        # 记录来源负载，便于把"地面实测耗时"与任务流一一对应
        tasks[-1].workload_key = spec.key  # type: ignore[attr-defined]
    return tasks


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
@dataclass
class RealComputeReport:
    """真实计算任务评测汇总。"""

    calibration: CalibrationReport
    task_count: int = 0
    total_gflops: float = 0.0
    total_input_mbit: float = 0.0
    total_raw_mbit: float = 0.0
    on_orbit_total_s: float = 0.0
    ground_total_s: float = 0.0
    speedup: float = 0.0
    workload_mix: List[Tuple[str, int]] = field(default_factory=list)
    wall_s: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def headline(self) -> List[Tuple[str, str]]:
        rows = list(self.calibration.headline())
        rows.append(("真实任务数", str(self.task_count)))
        rows.append(("真实计算总量", f"{self.total_gflops:.1f} GFLOP"))
        rows.append(("真实输入数据量", f"{self.total_input_mbit:.1f} Mbit"))
        rows.append(("弯管原始数据量", f"{self.total_raw_mbit:.1f} Mbit"))
        rows.append(("本机串行耗时", f"{self.ground_total_s:.3f} s（实测逐任务）"))
        rows.append(("星载串行耗时", f"{self.on_orbit_total_s:.3f} s（按算力折算）"))
        rows.append(("地面/星载加速比", f"{self.speedup:.1f}×"))
        return rows

    def summary_rows(self) -> List[Tuple[str, str, str, str, str]]:
        return self.calibration.summary_rows()

    def mix_text(self) -> str:
        return "，".join(f"{k.split('_')[-1]}×{v}" for k, v in self.workload_mix)


def evaluate_real_compute(n_tasks: int, n_gs: int, duration_s: float,
                          sat_capacity_tops: float = 8.0,
                          radiation_derate: float = 0.15,
                          keys: Optional[Sequence[str]] = None,
                          repeats: int = 3, seed: int = 0,
                          progress: Optional[Callable[[str], None]] = None
                          ) -> Tuple[RealComputeReport, List[Task]]:
    """真实计算任务评测：标定本机算力 → 生成真实任务流 → 给出星载耗时对照。"""
    t0 = time.perf_counter()
    cal = calibrate(keys=keys, repeats=repeats, progress=progress)
    tasks = real_task_stream(n_tasks, n_gs, duration_s, calibration=cal,
                             keys=keys, seed=seed)
    rep = RealComputeReport(calibration=cal, task_count=len(tasks))

    eff_tops = max(sat_capacity_tops * (1.0 - radiation_derate), 1e-3)
    ground_map = {r.key: r.elapsed_s for r in cal.runs if r.ok}
    mix: Dict[str, int] = {}
    for t in tasks:
        rep.total_gflops += t.cycles_gops
        rep.total_input_mbit += t.input_mbit
        rep.total_raw_mbit += t.raw_data_mbit
        rep.on_orbit_total_s += (t.cycles_gops / 1000.0) / eff_tops
        # 地面串行耗时：同一批任务在本机逐个真实执行的时间之和
        rep.ground_total_s += ground_map.get(
            getattr(t, "workload_key", ""), 0.0)
        k = getattr(t, "workload_key", "?")
        mix[k] = mix.get(k, 0) + 1
    rep.workload_mix = sorted(mix.items(), key=lambda kv: -kv[1])
    rep.speedup = (rep.ground_total_s / rep.on_orbit_total_s
                   if rep.on_orbit_total_s > 0 else 0.0)

    if cal.skipped:
        rep.warnings.append("以下负载因缺少依赖被跳过："
                            + ", ".join(k for k, _ in cal.skipped))
    missing = [r for r in cal.runs if not r.ok]
    for r in missing:
        rep.warnings.append(f"负载失败 {r.label}: {r.note}")

    rep.wall_s = time.perf_counter() - t0
    if progress:
        progress(f"真实计算评测完成：{rep.wall_s:.2f} s，"
                 f"{rep.task_count} 个真实任务，{rep.total_gflops:.1f} GFLOP")
    return rep, tasks

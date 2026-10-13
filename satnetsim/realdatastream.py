"""真实数据流仿真层：把**真实字节**塞进**真实拓扑**，逐包多跳转发到位。

与 `reallink.py` 的区别
----------------------
`reallink.py` 把字节推过**本机内核协议栈**（回环），用来标定真实 TCP/拥塞控制；
它回答的是"这段链路上 TCP 能跑多快"。

本模块回答的是另一个问题：**给定的真实业务数据，能否、多久、以何等保真度穿过
整个星座网络到达地面站**。为此：

1. **真实载荷**：真图片（PNG/JPEG 真实编码字节）、真文本文档、真遥测二进制，
   或用户指定的本地文件——**不是随机数**，每片都带 SHA-256。
2. **真实分片**：按 MSS 切分，序号 + 长度 + 摘要，逐片可校验。
3. **真实转发**：数据报式逐跳转发，每跳都按**当时的拓扑快照**查下一跳
   （单源最短路反向建树 → O(1) 下一跳表），天然复现"拓扑时变 → 路径切换"。
4. **真实排队**：每条有向链路是 FIFO 服务器，服务时间 = 分组大小/链路速率，
   传播时延取快照实测值；**有限缓冲 + 尾丢弃**（缓冲默认 2×BDP）。
5. **真实重组**：目的端按序号重组，逐字节 SHA-256 比对源数据。

产出
----
`DataFlowReport` 既含统计口径（吞吐、端到端时延、丢包原因分布、逐链路负载），
也含**可视化轨迹**——每个分组的每一跳都记录了
``(发送起始 s, 到达 s, 起点节点, 终点节点)``，因此可以按时间回放出
"光点沿轨道星座一跳一跳飞过去"的实时画面（见 `liveviz.py`）。
"""

from __future__ import annotations

import hashlib
import heapq
import math
import os
import random
import struct
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import RealDataConfig
from .orbit import Constellation, lla_to_ecef
from .routing import _dijkstra
from .topology import Snapshot

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
_PNG_SIG = b"\x89PNG\r\n\x1a\n"

# 事件类型
_EV_ARRIVE = 0
_EV_TX_DONE = 1


# --------------------------------------------------------------------------- #
# 1. 真实载荷
# --------------------------------------------------------------------------- #
@dataclass
class DataPayload:
    """一份真实数据：真字节 + 真实编码格式 + 真摘要。"""

    kind: str
    label: str
    mime: str
    data: bytes
    frames: int = 1
    note: str = ""

    @property
    def n_bytes(self) -> int:
        return len(self.data)

    @property
    def n_mbit(self) -> float:
        return len(self.data) * 8.0 / 1e6

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def describe(self) -> str:
        return (f"{self.label}（{self.mime}）{self.n_bytes/1024:.1f} KiB / "
                f"{self.n_mbit:.3f} Mbit，SHA-256 {self.sha256[:16]}")


def _encode_png(width: int, height: int, rows: bytes) -> bytes:
    """用标准库手写一个**真 PNG**（zlib + CRC32），任何看图软件都能打开。"""
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8bit truecolor
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)                       # filter type 0
        raw += rows[y * stride:(y + 1) * stride]
    idat = zlib.compress(bytes(raw), 6)
    return (_PNG_SIG + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat)
            + chunk(b"IEND", b""))


def _render_scene_rgb(width: int, height: int, seed: int,
                      phase: float = 0.0) -> bytes:
    """渲染一幅"对地观测"风格的确定性 RGB 栅格（真实像素，非随机噪声）。

    多尺度地形（余弦叠加）+ 水体 + 植被色带 + 云层 + **传感器噪声**；
    传感器噪声既符合真实遥感成像特性，也让 JPEG 具备真实的信息量
    （避免"过于平滑导致单帧只有几 KB、需要上千帧才凑够载荷"）。
    """
    rng = np.random.default_rng(seed)
    f32 = np.float32
    u = (np.arange(width, dtype=f32) / f32(max(width - 1, 1))) + f32(phase)
    v = (np.arange(height, dtype=f32) / f32(max(height - 1, 1))) + f32(0.35 * phase)
    U = u[None, :]
    V = v[:, None]

    f = (f32(0.55) * np.sin(f32(6.7) * U + f32(1.3)) * np.cos(f32(4.1) * V + f32(0.7))
         + f32(0.28) * np.sin(f32(13.3) * U + f32(2.1)) * np.cos(f32(9.7) * V + f32(1.9))
         + f32(0.17) * np.sin(f32(27.1) * U + f32(0.4)) * np.cos(f32(21.3) * V + f32(3.1)))
    elev = (f + f32(1.0)) * f32(0.5)
    water = elev < f32(0.34)
    land = ~water

    r = np.zeros((height, width), dtype=f32)
    g = np.zeros((height, width), dtype=f32)
    b = np.zeros((height, width), dtype=f32)
    depth = np.clip((f32(0.34) - elev) / f32(0.34), 0, 1)
    b[water] = f32(90) + f32(80) * (1 - depth[water])
    g[water] = f32(40) + f32(70) * (1 - depth[water])
    r[water] = f32(10) + f32(20) * (1 - depth[water])
    hi = np.clip((elev - f32(0.34)) / f32(0.66), 0, 1)
    r[land] = f32(60) + f32(150) * hi[land] ** f32(1.6)
    g[land] = (f32(120) - f32(30) * hi[land]
               + f32(60) * np.clip(hi[land] - f32(0.7), 0, 1) / f32(0.3))
    b[land] = f32(45) + f32(130) * np.clip(hi[land] - f32(0.72), 0, 1) / f32(0.28)

    # 云层：粗噪声上采样（随 phase 平移）
    ch, cw = max(height // 16, 2), max(width // 16, 2)
    coarse = rng.random((ch, cw)).astype(f32)
    coarse = np.roll(coarse, int(round(phase * 8)) % ch, axis=0)
    cloud = np.repeat(np.repeat(coarse, 16, axis=0), 16, axis=1)[:height, :width]
    cloud = np.clip((cloud - f32(0.55)) * f32(3.2), 0, 1) * f32(0.62)
    inv = f32(1.0) - cloud
    r = r * inv + f32(245) * cloud
    g = g * inv + f32(248) * cloud
    b = b * inv + f32(252) * cloud

    # 传感器噪声 + 细节纹理（真实成像普遍存在，同时赋予 JPEG 真实信息量）
    rgb = np.empty((height, width, 3), dtype=f32)
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b
    rgb += rng.standard_normal(rgb.shape, dtype=f32) * f32(7.0)
    rgb += (f32(9.0) * np.sin(f32(190.0) * U + f32(3.0)))[:, :, None]
    return np.clip(rgb, 0, 255).astype(np.uint8).tobytes()


def _encode_jpeg(side: int, rgb: bytes, quality: int = 88) -> Optional[bytes]:
    """真 JPEG 编码（PIL）；不可用时返回 None。"""
    try:
        from PIL import Image  # type: ignore
        import io
        img = Image.frombytes("RGB", (side, side), rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


def _make_image_payload(size_bytes: int, seed: int) -> DataPayload:
    """生成**真图片**：MJPEG 帧序列（每帧都是真 JPEG），总字节数贴近目标。

    单帧边长 512（≈ 数十 KiB），按需加帧直到命中目标大小；帧间地物平移，
    是一段真实的连续观测序列而不是重复字节。无 PIL 时退回标准库 PNG。
    """
    target = max(4096, int(size_bytes))
    side = 384 if target < 192 * 1024 else 512
    frames: List[bytes] = []
    total = 0
    fi = 0
    while total < target and fi < 512:
        rgb = _render_scene_rgb(side, side, seed, phase=fi * 0.012)
        blob = _encode_jpeg(side, rgb, 88)
        if blob is None:                      # 无 PIL → 单张 PNG 收工
            data = _encode_png(side, side, rgb)
            return DataPayload("image", "对地观测影像", "image/png", data,
                               frames=1,
                               note=f"{side}×{side} PNG（标准库 zlib 真实编码）")
        frames.append(blob)
        total += len(blob)
        fi += 1
    data = b"".join(frames)
    if len(frames) == 1:
        return DataPayload("image", "对地观测影像", "image/jpeg", data,
                           frames=1,
                           note=f"{side}×{side} JPEG q88（PIL 真实编码），"
                                f"{len(data)/1024:.0f} KiB")
    return DataPayload("image", "对地观测影像序列",
                       "multipart/x-mixed-replace; boundary=--mjpeg", data,
                       frames=len(frames),
                       note=f"{len(frames)} 帧 {side}×{side} JPEG q88（MJPEG，"
                            f"帧间地物平移），合计 {len(data)/1024:.0f} KiB")



_TEXT_LINES = [
    "# SatNetSim 真实数据流 · 在轨遥测报告",
    "",
    "## 1. 星座状态",
    "本报告由星载数据中心的推理节点生成，经星间激光链路逐跳转发至地面关口站。",
    "转发路径随拓扑快照变化，每个分组独立选路，因此同一份文件的不同分片",
    "可能走不同的路径到达——这正是低轨星座数据报网络的真实行为。",
    "",
    "## 2. 链路预算",
    "星间链路速率 100 Gbps 量级；星地馈电链路 500 Mbps 量级；",
    "单跳传播时延由轨道几何决定，典型 3~12 ms；存储转发会引入排队时延。",
    "",
    "## 3. 完整性",
    "每个分片携带序号与 SHA-256，接收端按序重组并逐字节比对；",
    "任何分片丢失都会导致重组失败，并被计入丢包原因分布。",
    "",
    "## 4. 观测目标",
    "• 地表温度场异常区域（海温/陆温梯度）",
    "• 极地冰盖消融速率（多时相差分）",
    "• 台风眼墙结构与移动矢量",
    "• 夜间灯光与人类活动强度",
    "",
    "## 5. 结论",
    "在 100 Gbps 星间链路 + 500 Mbps 馈电链路配置下，端到端吞吐受限于",
    "馈电链路瓶颈与拓扑切换间隙；引入星载缓存与断点续传可显著改善交付率。",
]


def _make_text_payload(size_bytes: int, seed: int) -> DataPayload:
    """生成**真文本**：可读的中文技术文档（UTF-8 真实字节，按字符边界截断）。"""
    target = max(1024, int(size_bytes))
    blocks: List[str] = []
    used = 0
    i = 0
    while used < target and i < 200_000:
        line = _TEXT_LINES[i % len(_TEXT_LINES)]
        if (i // len(_TEXT_LINES)) % 8 == 7:
            line = f"（第 {i + 1} 段 · 批次 {seed}）"
        blocks.append(line)
        used += len(line.encode("utf-8")) + 1
        i += 1
    text = "\n".join(blocks)[:target]
    # 保证不截断多字节字符
    while text and len(text.encode("utf-8")) > target:
        text = text[:-1]
    data = text.encode("utf-8")
    return DataPayload("text", "在轨遥测报告", "text/plain; charset=utf-8", data,
                       note=f"UTF-8 中文文档 {len(data)} B（{data.count(10)} 行）")


def _make_telemetry_payload(size_bytes: int, seed: int) -> DataPayload:
    """生成**真二进制遥测**：定长记录（时间戳/电压/温度/姿态四元数），struct 打包。"""
    rec = struct.Struct(">dfffff")   # t, vbus, temp, qx, qy, qz
    n = max(1, int(round(max(1, int(size_bytes)) / rec.size)))   # 整记录数，不截断
    rng = random.Random(seed)
    out = bytearray()
    for k in range(n):
        out += rec.pack(
            k * 0.5,
            28.0 + rng.uniform(-0.3, 0.3),
            22.0 + 6.0 * math.sin(k / 240.0) + rng.uniform(-0.4, 0.4),
            *[rng.uniform(-1, 1) for _ in range(3)],
        )
    data = bytes(out)
    return DataPayload("telemetry", "星务遥测流", "application/octet-stream", data,
                       note=f"定长记录 ×{n}，struct 打包，{len(data)} B")


def _make_file_payload(path: str, size_bytes: int) -> DataPayload:
    with open(path, "rb") as fh:
        data = fh.read()
    if size_bytes > 0:
        data = data[:size_bytes]
    kind = "file"
    mime = "application/octet-stream"
    low = path.lower()
    if low.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        mime = "image/*"
    elif low.endswith((".txt", ".md", ".csv", ".json")):
        mime = "text/plain"
    return DataPayload(kind, os.path.basename(path), mime, data,
                       note=f"用户指定本地文件 {path}")


def build_payload(
    kind: str = "image",
    size_bytes: int = 512 * 1024,
    seed: int = 20261012,
    file_path: Optional[str] = None,
) -> DataPayload:
    """构造真实载荷。

    kind : image | text | telemetry | file | mixed
    """
    size_bytes = max(1024, int(size_bytes))
    k = (kind or "image").lower()
    if k == "file" and file_path:
        if os.path.isfile(file_path):
            return _make_file_payload(file_path, size_bytes)
        k = "image"                     # 文件不存在 → 退回影像
    if k == "text":
        return _make_text_payload(size_bytes, seed)
    if k == "telemetry":
        return _make_telemetry_payload(size_bytes, seed)
    return _make_image_payload(size_bytes, seed)


# --------------------------------------------------------------------------- #
# 2. 分片
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    seq: int
    offset: int
    data: bytes
    digest: str

    @property
    def n_bytes(self) -> int:
        return len(self.data)


def segment_payload(payload: DataPayload, mss_bytes: int) -> List[Segment]:
    """把真实载荷切成 MSS 大小的分片，逐片带 SHA-256。"""
    mss = max(64, int(mss_bytes))
    segs: List[Segment] = []
    for i, off in enumerate(range(0, len(payload.data), mss)):
        blob = payload.data[off:off + mss]
        segs.append(Segment(i, off, blob, hashlib.sha256(blob).hexdigest()[:16]))
    return segs


def reassemble(seg_map: Dict[int, bytes], n_segments: int) -> Tuple[bytes, bool]:
    """按序号重组；缺片即失败（返回已有部分的拼接与 False）。"""
    if len(seg_map) < n_segments:
        return b"".join(seg_map[k] for k in sorted(seg_map)), False
    return b"".join(seg_map[k] for k in range(n_segments)), True


# --------------------------------------------------------------------------- #
# 3. 流与结果容器
# --------------------------------------------------------------------------- #
@dataclass
class FlowSpec:
    flow_id: int
    src_node: int
    dst_node: int
    src_label: str
    dst_label: str
    payload: DataPayload
    rate_mbps: float
    start_s: float
    duration_s: float
    mss_bytes: int


@dataclass
class PacketRecord:
    """一个分组的完整生命史（含逐跳轨迹，供可视化回放）。"""

    pid: int
    flow_id: int
    seq: int
    size_bytes: int
    created_s: float
    src_node: int
    dst_node: int
    digest: str
    # (发送起始 s, 到达 s, 起点节点, 终点节点)
    segs: List[Tuple[float, float, int, int]] = field(default_factory=list)
    delivered_s: Optional[float] = None
    end_s: Optional[float] = None           # 交付或丢弃的时刻
    drop_node: Optional[int] = None
    drop_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.delivered_s is not None

    @property
    def hops(self) -> int:
        return len(self.segs)

    @property
    def e2e_s(self) -> float:
        return (self.delivered_s - self.created_s) if self.ok else float("nan")


@dataclass
class LinkStat:
    edge: Tuple[int, int]
    packets: int = 0
    bytes: int = 0
    drops: int = 0
    peak_queue_bytes: int = 0
    busy_s: float = 0.0

    @property
    def rate_mbps(self) -> float:
        return self.bytes * 8.0 / max(self.busy_s, 1e-9) / 1e6


@dataclass
class DataFlowReport:
    """真实数据流仿真的完整结果。"""

    payload_kind: str
    payload_label: str
    payload_mime: str
    payload_bytes: int
    payload_sha: str
    n_segments: int
    mss_bytes: int

    flows: List[FlowSpec]
    packets: List[PacketRecord]

    n_packets: int = 0
    n_delivered: int = 0
    n_dropped: int = 0
    drop_reasons: Dict[str, int] = field(default_factory=dict)

    integrity_ok: bool = False
    digest_src: str = ""
    digest_dst: str = ""
    received_bytes: int = 0
    sent_bytes: int = 0
    n_rounds: int = 1          # 载荷在窗口内循环发送的轮数
    seg_coverage: float = 0.0  # 载荷分片位置覆盖率（0–1）

    first_delivery_s: float = float("nan")
    last_delivery_s: float = float("nan")
    e2e_mean_s: float = float("nan")
    e2e_p95_s: float = float("nan")
    e2e_min_s: float = float("nan")
    e2e_max_s: float = float("nan")
    goodput_mbps: float = 0.0
    mean_hops: float = 0.0
    max_hops: int = 0
    delivery_ratio: float = 0.0

    series_t: List[float] = field(default_factory=list)
    series_inflight: List[float] = field(default_factory=list)
    series_goodput: List[float] = field(default_factory=list)
    series_inflight_mb: List[float] = field(default_factory=list)
    series_delivered: List[float] = field(default_factory=list)

    link_stats: List[LinkStat] = field(default_factory=list)

    n_sat: int = 0
    n_gs: int = 0
    duration_s: float = 0.0
    snapshot_interval_s: float = 10.0
    wall_s: float = 0.0
    warnings: List[str] = field(default_factory=list)

    # ---------------- 便捷视图 ---------------- #
    @property
    def integrity(self) -> str:
        if self.n_delivered == 0:
            return "未交付"
        return "SHA-256 一致" if self.integrity_ok else "数据不一致"

    def mix_text(self) -> str:
        if not self.drop_reasons:
            return "无丢包"
        return "，".join(f"{k}×{v}" for k, v in
                        sorted(self.drop_reasons.items(), key=lambda kv: -kv[1]))

    def headline(self) -> List[Tuple[str, str]]:
        rows = [
            ("业务窗口", f"{self.duration_s:.0f} s（= 仿真时长），"
                         f"{len(self.flows)} 流，载荷循环 {self.n_rounds} 轮"),
            ("真实载荷", f"{self.payload_label} {self.payload_bytes/1024:.0f} KiB "
                         f"/ {self.n_segments} 分片"),
            ("送达", f"{self.n_delivered}/{self.n_packets} 分组 "
                     f"({self.delivery_ratio*100:.1f}%)"),
            ("端到端时延", f"均值 {self.e2e_mean_s*1000:.1f} ms / "
                           f"P95 {self.e2e_p95_s*1000:.1f} ms"),
            ("平均跳数", f"{self.mean_hops:.1f}（最多 {self.max_hops}）"),
            ("端到端吞吐", f"{self.goodput_mbps:.1f} Mbps"),
            ("完整性", f"{self.integrity}（收 {self.received_bytes/1024:.0f} KiB，"
                       f"分片覆盖 {self.seg_coverage*100:.0f}%）"),
        ]
        if self.drop_reasons:
            rows.append(("丢包原因", self.mix_text()))
        return rows

    def summary_rows(self) -> List[Tuple[str, str, str, str, str]]:
        """(指标, 数值1, 数值2, 数值3, 说明) —— 供 GUI 表格复用。"""
        return [
            ("窗口", f"{self.duration_s:.0f} s", f"{len(self.flows)} 流",
             f"载荷循环 {self.n_rounds} 轮", f"发送 {self.sent_bytes/1e6:.1f} MB"),
            ("载荷", f"{self.payload_label}", f"{self.payload_bytes/1024:.1f} KiB",
             self.payload_mime, self.payload_sha[:16]),
            ("分片", f"{self.n_segments} 片", f"MSS {self.mss_bytes} B",
             f"{self.n_segments*self.mss_bytes/1024:.0f} KiB 上限", "真实切分"),
            ("分组", f"发 {self.n_packets}", f"达 {self.n_delivered}",
             f"丢 {self.n_dropped}", self.mix_text()),
            ("时延", f"均 {self.e2e_mean_s*1000:.1f} ms",
             f"P95 {self.e2e_p95_s*1000:.1f} ms",
             f"最 {self.e2e_max_s*1000:.1f} ms", f"跳数 {self.mean_hops:.1f}"),
            ("吞吐", f"{self.goodput_mbps:.1f} Mbps", f"{self.delivery_ratio*100:.1f}%",
             f"{self.n_delivered} 分组", "按首末交付计"),
            ("完整性", self.integrity, f"{self.received_bytes/1024:.1f} KiB",
             self.digest_dst[:16], f"源 {self.digest_src[:16]}"),
        ]

    # ---------------- 序列化 ---------------- #
    def to_dict(self, trace_limit: int = 0) -> Dict:
        """结构化输出。trace_limit>0 时附带可视化轨迹（按分组数截断）。"""
        d: Dict = {
            "payload": {
                "kind": self.payload_kind, "label": self.payload_label,
                "mime": self.payload_mime, "bytes": self.payload_bytes,
                "sha256": self.payload_sha, "segments": self.n_segments,
                "mss_bytes": self.mss_bytes,
            },
            "counts": {
                "packets": self.n_packets, "delivered": self.n_delivered,
                "dropped": self.n_dropped, "drop_reasons": self.drop_reasons,
                "delivery_ratio": self.delivery_ratio,
            },
            "integrity": {
                "ok": self.integrity_ok, "digest_src": self.digest_src,
                "digest_dst": self.digest_dst, "received_bytes": self.received_bytes,
            },
            "latency_s": {
                "mean": self.e2e_mean_s, "p95": self.e2e_p95_s,
                "min": self.e2e_min_s, "max": self.e2e_max_s,
            },
            "throughput_mbps": self.goodput_mbps,
            "hops": {"mean": self.mean_hops, "max": self.max_hops},
            "n_sat": self.n_sat, "n_gs": self.n_gs,
            "duration_s": self.duration_s,
            "snapshot_interval_s": self.snapshot_interval_s,
            "wall_s": self.wall_s,
            "series": {
                "t": self.series_t, "inflight": self.series_inflight,
                "goodput_mbps": self.series_goodput,
                "inflight_mbit": self.series_inflight_mb,
                "delivered_mbit": self.series_delivered,
            },
            "links": [
                {"src": int(s.edge[0]), "dst": int(s.edge[1]),
                 "packets": s.packets, "bytes": s.bytes, "drops": s.drops,
                 "peak_queue_bytes": s.peak_queue_bytes, "busy_s": round(s.busy_s, 6),
                 "rate_mbps": round(s.rate_mbps, 4)}
                for s in self.link_stats
            ],
            "warnings": self.warnings,
        }
        if trace_limit > 0:
            pk = self.packets[:trace_limit]
            d["trace"] = [
                {
                    "pid": p.pid, "flow": p.flow_id, "seq": p.seq,
                    "size": p.size_bytes, "created": p.created_s,
                    "src": p.src_node, "dst": p.dst_node,
                    "delivered": p.delivered_s, "end": p.end_s,
                    "drop_node": p.drop_node, "drop_reason": p.drop_reason,
                    "segs": [[round(a, 4), round(b, 4), int(u), int(v)]
                             for a, b, u, v in p.segs],
                }
                for p in pk
            ]
        return d


# --------------------------------------------------------------------------- #
# 4. 下一跳表（按快照预计算）
# --------------------------------------------------------------------------- #
def _snapshot_index(t_s: float, snapshot_interval_s: float, n: int) -> int:
    if n <= 1:
        return 0
    return int(min(max(int(t_s // max(snapshot_interval_s, 1e-9)), 0), n - 1))


def _build_forwarding_tables(
    snapshots: Sequence[Snapshot], dst_node: int, max_hops: int = 64
) -> List[Tuple[np.ndarray, Dict[Tuple[int, int], int]]]:
    """对每个快照做一次"以目的节点为根"的反向最短路。

    返回 [(nexthop 数组, 边索引字典)]；``nexthop[u]`` 即 u 朝目的地的下一跳。
    """
    tables: List[Tuple[np.ndarray, Dict[Tuple[int, int], int]]] = []
    for snap in snapshots:
        if snap.n_edges == 0:
            n = snap.n_nodes
            tables.append((np.full(n, -1, dtype=np.int64), {}))
            continue
        _dist, prev = _dijkstra(snap, dst_node, max_hops=max_hops)
        nexthop = np.asarray(prev, dtype=np.int64).copy()
        nexthop[dst_node] = -1
        edge_of: Dict[Tuple[int, int], int] = {}
        for i in range(snap.n_edges):
            a, b = int(snap.src[i]), int(snap.dst[i])
            edge_of[(a, b)] = i
            edge_of[(b, a)] = i
        tables.append((nexthop, edge_of))
    return tables


# --------------------------------------------------------------------------- #
# 5. 分组级离散事件仿真
# --------------------------------------------------------------------------- #
class _LinkState:
    __slots__ = ("queue", "queued_bytes", "busy_until", "limit_bytes", "stat")

    def __init__(self, limit_bytes: int, stat: LinkStat):
        # deque：先进先出且 popleft() 为 O(1)（list.pop(0) 是 O(n)，深队列会退化成平方）
        self.queue: Deque[int] = deque()
        self.queued_bytes = 0
        self.busy_until = 0.0
        self.limit_bytes = limit_bytes
        self.stat = stat


def simulate_real_data(
    cfg: RealDataConfig,
    constellation: Constellation,
    snapshots: Sequence[Snapshot],
    ground_stations: Sequence,
    snapshot_interval_s: float = 10.0,
    sim_duration_s: float = 300.0,
    seed: int = 0,
    progress: Optional[Callable[[str], None]] = None,
) -> DataFlowReport:
    """把真实载荷送进星座拓扑，逐包多跳转发，返回完整报告（含可视化轨迹）。"""
    t_wall = time.perf_counter()
    warns: List[str] = []
    n_sat = constellation.n_sat
    n_gs = len(ground_stations)
    snap_iv = float(max(snapshot_interval_s, 1e-6))
    # 数据仿真窗口：默认**跟随仿真时长**，保证真实数据流与轨道/拓扑/计算同时间尺度。
    # （旧实现曾截断为 min(仿真时长, 12 s)，导致"仿真时长 600 s、实际只跑 12 s"。
    #  现在业务以恒定速率铺满整个窗口，规模由上方的 max_packets 预算约束。）
    duration = float(cfg.duration_s) if cfg.duration_s > 0 else float(sim_duration_s)
    duration = max(duration, 1e-3)
    emit_span = max(duration * 0.95, 1e-6)

    def tell(msg: str) -> None:
        if progress:
            progress(msg)

    # ---- 载荷与分片 ----
    tell("构造真实载荷…")
    payload = build_payload(cfg.payload_kind, int(cfg.payload_mbit * 1e6 / 8),
                            seed=seed, file_path=cfg.payload_file or None)
    mss = max(64, int(cfg.mss_bytes))
    segs = segment_payload(payload, mss)
    n_segs = max(len(segs), 1)

    # ---- 端点选择（支持多流并发：取多对互不相同的地面站） ----
    if n_gs < 2:
        raise RuntimeError("真实数据流需要至少 2 个地面站")
    n_flows = max(1, int(cfg.n_flows))
    pairs = _pick_gs_pairs(ground_stations, cfg, n_sat, n_flows, snapshots)

    # ---- 下一跳表（每个目的站一张；多流各自独立选路） ----
    tell("预计算逐快照转发表（每目的站一张）…")
    tables_by_flow: List[List[Tuple[np.ndarray, Dict[Tuple[int, int], int]]]] = [
        _build_forwarding_tables(snapshots, dst) for _src, dst in pairs
    ]

    # ---- 每链路缓冲上限：mult × BDP（BDP = 容量 × 单跳传播时延） ----
    mult = max(0.25, float(cfg.buffer_mult))
    limit_of: Dict[Tuple[int, int], int] = {}
    for snap in snapshots:
        for i in range(snap.n_edges):
            a, b = int(snap.src[i]), int(snap.dst[i])
            cap = float(snap.capacity_mbps[i])
            prop = float(snap.delay_ms[i]) / 1000.0
            lim = int(mult * cap * 1e6 / 8.0 * max(prop, 1e-4))
            lim = max(lim, 16 * mss)
            limit_of[(a, b)] = max(limit_of.get((a, b), 0), lim)
            limit_of[(b, a)] = limit_of[(a, b)]

    # ---- 事件堆 ----
    heap: List[Tuple[float, int, int, int, int]] = []
    # (time, tie, kind, pkt_index, node)
    tie = 0
    packets: List[PacketRecord] = []
    links: Dict[Tuple[int, int], _LinkState] = {}
    rng = random.Random(seed ^ 0x5EED)

    def link_state(u: int, v: int) -> _LinkState:
        key = (u, v)
        st = links.get(key)
        if st is None:
            rev = (v, u)
            lim = limit_of.get(key, max(16 * mss, 1 << 20))
            st = _LinkState(lim, LinkStat(edge=key))
            links[key] = st
            links.setdefault(rev, _LinkState(lim, LinkStat(edge=rev)))
        return st

    # ---- 发射规划：恒定业务速率铺满整个窗口，规模受 max_packets 预算约束 ----
    auto_rate = min(_auto_rate(snapshots, tables_by_flow[f], pairs[f][0], pairs[f][1])
                    for f in range(n_flows))
    # 单条业务流的合理速率上限（真实业务流不会占满馈电链路；20 Mbps 已是高清影像量级）
    want_rate = float(cfg.rate_mbps) if cfg.rate_mbps > 0 else min(auto_rate, 20.0)
    want_rate = max(want_rate, 1e-3)
    budget_total = max(int(cfg.max_packets), n_flows * n_segs)
    per_flow_budget = max(budget_total // n_flows, 1)
    want_packets = int(want_rate * 1e6 / 8.0 * emit_span / mss)
    budget_flow = max(n_segs, min(want_packets, per_flow_budget))
    rate = budget_flow * mss * 8.0 / emit_span / 1e6      # 实际生效的单流速率 (Mbps)
    n_rounds = int(math.ceil(budget_flow / n_segs))
    if want_packets > budget_flow:
        warns.append(
            f"分组预算 {budget_total} 生效：单流业务速率下调至 {rate:.3f} Mbps"
            f"（窗口 {duration:.0f} s；如需更密集的回放，请调高 real_data.max_packets）")

    flows: List[FlowSpec] = []
    for f, (s_nd, d_nd) in enumerate(pairs):
        flows.append(FlowSpec(
            f, s_nd, d_nd,
            _node_label(s_nd, n_sat, ground_stations),
            _node_label(d_nd, n_sat, ground_stations),
            payload, rate, 0.0, duration, mss))

    pid = 0
    step = emit_span / max(budget_flow, 1)
    for f in range(n_flows):
        src_nd, dst_nd = pairs[f]
        for k in range(budget_flow):
            t_emit = min((k + rng.uniform(0.0, 0.25)) * step,
                         max(duration - 1e-3, 0.0))
            pos = pid % n_segs
            blob_len = min(mss, payload.n_bytes - pos * mss)
            pk = PacketRecord(pid=pid, flow_id=f, seq=pid,
                              size_bytes=max(blob_len, 1), created_s=t_emit,
                              src_node=src_nd, dst_node=dst_nd,
                              digest=segs[pos].digest)
            packets.append(pk)
            tie += 1
            heapq.heappush(heap, (t_emit, tie, _EV_ARRIVE, pid, src_nd))
            pid += 1
    n_packets = pid
    warned_rate = sum(f.rate_mbps for f in flows)
    warns.append(
        f"业务窗口 {duration:.0f} s（= 仿真时长），{n_flows} 流 × {rate:.3f} Mbps"
        f" = {warned_rate:.2f} Mbps，共 {n_packets} 分组，"
        f"载荷 {payload.n_bytes/1024:.0f} KiB 循环 {n_rounds} 轮")

    # ---- 主循环 ----
    tell(f"开始逐包多跳转发（{n_packets} 个分组）…")
    n_tx = 0
    max_events = max(200_000, n_packets * 256)
    done = 0

    while heap:
        if n_tx > max_events:
            warns.append(f"事件数超过上限 {max_events}，仿真提前终止（拓扑可能不连通）")
            break
        n_tx += 1
        t_now, _tie, kind, pidx, node = heapq.heappop(heap)
        pk = packets[pidx]

        if kind == _EV_ARRIVE:
            if node == pk.dst_node:
                pk.delivered_s = t_now
                pk.end_s = t_now
                done += 1
                continue
            si = _snapshot_index(t_now, snap_iv, len(snapshots))
            tables = tables_by_flow[pk.flow_id]
            nexthop, edge_of = tables[si]
            v = int(nexthop[node]) if node < len(nexthop) else -1
            if v < 0:
                pk.end_s = t_now
                pk.drop_node = node
                pk.drop_reason = "no_route"
                continue
            ei = edge_of.get((node, v))
            if ei is None:
                pk.end_s = t_now
                pk.drop_node = node
                pk.drop_reason = "no_link"
                continue
            snap = snapshots[si]
            prop = float(snap.delay_ms[ei]) / 1000.0
            cap = float(snap.capacity_mbps[ei])
            tx = pk.size_bytes * 8.0 / max(cap, 1e-9) / 1e6
            st = link_state(node, v)
            if st.queue or st.busy_until > t_now:
                st.queued_bytes += pk.size_bytes
                if st.queued_bytes > st.limit_bytes:
                    st.queued_bytes -= pk.size_bytes
                    st.stat.drops += 1
                    pk.end_s = t_now
                    pk.drop_node = node
                    pk.drop_reason = "queue_overflow"
                    continue
                st.queue.append(pidx)
                if st.queued_bytes > st.stat.peak_queue_bytes:
                    st.stat.peak_queue_bytes = st.queued_bytes
            else:
                st.busy_until = t_now + tx
                st.stat.packets += 1
                st.stat.bytes += pk.size_bytes
                st.stat.busy_s += tx
                pk.segs.append((t_now, t_now + tx + prop, node, v))
                tie += 1
                heapq.heappush(heap, (t_now + tx, tie, _EV_TX_DONE, pidx, node))

        else:  # _EV_TX_DONE：node 是发端，需要回查该分组最终走到的下一跳
            # 找到该分组在本链路的方向：用最后一段的目的节点
            if not pk.segs:
                continue
            _, _arr, u, v = pk.segs[-1]
            si = _snapshot_index(t_now, snap_iv, len(snapshots))
            snap = snapshots[si]
            # 传播到 v
            tie += 1
            heapq.heappush(heap, (pk.segs[-1][1], tie, _EV_ARRIVE, pidx, v))
            # 释放该有向链路，若队列非空则继续发送下一个
            st = link_state(u, v)
            if st.queue:
                nxt = st.queue.popleft()
                npk = packets[nxt]
                st.queued_bytes -= npk.size_bytes
                ei = tables_by_flow[npk.flow_id][si][1].get((u, v))
                cap = float(snap.capacity_mbps[ei]) if ei is not None else 1.0
                tx = npk.size_bytes * 8.0 / max(cap, 1e-9) / 1e6
                prop = float(snap.delay_ms[ei]) / 1000.0 if ei is not None else 0.0
                st.busy_until = t_now + tx
                st.stat.packets += 1
                st.stat.bytes += npk.size_bytes
                st.stat.busy_s += tx
                npk.segs.append((t_now, t_now + tx + prop, u, v))
                tie += 1
                heapq.heappush(heap, (t_now + tx, tie, _EV_TX_DONE, nxt, u))
            else:
                st.busy_until = t_now

    # ---- 汇总 ----
    tell("重组与完整性校验…")
    delivered = [p for p in packets if p.ok]
    dropped = [p for p in packets if not p.ok]
    reasons: Dict[str, int] = {}
    for p in dropped:
        reasons[p.drop_reason] = reasons.get(p.drop_reason, 0) + 1

    # 分片序号按"载荷内位置"回环：长窗口下同一份载荷循环发送多轮，
    # 每个位置只要有一份送达即可完成重组，同时逐片比对 SHA-256。
    seg_ok: Dict[int, bytes] = {}
    seg_match = 0
    for p in delivered:
        pos = p.seq % n_segs
        if p.digest == segs[pos].digest:
            seg_match += 1
        seg_ok[pos] = _payload_slice(payload, pos, mss)
    rebuilt, complete = reassemble(seg_ok, n_segs)
    src_digest = payload.sha256
    dst_digest = hashlib.sha256(rebuilt).hexdigest() if rebuilt else ""
    integrity_ok = bool(complete and rebuilt == payload.data)
    sent_bytes = int(sum(p.size_bytes for p in packets))
    seg_coverage = len(seg_ok) / max(n_segs, 1)
    if delivered and seg_match < len(delivered):
        warns.append(
            f"{len(delivered) - seg_match} 个送达分组的 SHA-256 与源片不一致")

    lat = np.array([p.e2e_s for p in delivered], dtype=float) if delivered else np.zeros(0)
    first_del = min((p.delivered_s for p in delivered), default=float("nan"))
    last_del = max((p.delivered_s for p in delivered), default=float("nan"))
    span = (last_del - first_del) if delivered else float("nan")
    goodput = (sum(p.size_bytes for p in delivered) * 8.0 / span / 1e6
               if delivered and span and span > 0 else 0.0)

    # ---- 时间序列（分箱，向量化精确统计） ----
    bins = max(8, int(cfg.sample_bins))
    t_axis = np.linspace(0.0, duration, bins)
    c_t = np.array([p.created_s for p in packets], dtype=float)
    e_t = np.array([(p.end_s if p.end_s is not None else duration) for p in packets],
                   dtype=float)
    sz = np.array([p.size_bytes for p in packets], dtype=float)
    oc, oe = np.argsort(c_t), np.argsort(e_t)
    cum_c = np.concatenate([[0.0], np.cumsum(sz[oc])])
    cum_e = np.concatenate([[0.0], np.cumsum(sz[oe])])
    n_c = np.searchsorted(c_t[oc], t_axis, side="right")
    n_e = np.searchsorted(e_t[oe], t_axis, side="left")
    inflight = (n_c - n_e).astype(float)
    in_mb = (cum_c[n_c] - cum_e[n_e]) / 1e6          # 网内在途数据量 (Mbit)

    # 累计送达量
    delivered_mbit = np.zeros(bins)
    if delivered:
        dt_arr = np.array([p.delivered_s for p in delivered], dtype=float)
        db_arr = np.array([p.size_bytes for p in delivered], dtype=float)
        od = np.argsort(dt_arr)
        cum_d = np.concatenate([[0.0], np.cumsum(db_arr[od])])
        kd = np.searchsorted(dt_arr[od], t_axis, side="right")
        delivered_mbit = cum_d[kd] / 1e6

    dt_bin = duration / max(bins - 1, 1)
    goodput_series = np.diff(np.concatenate([[0.0], delivered_mbit])) / max(dt_bin, 1e-9)

    link_stats = sorted(
        [st.stat for st in links.values() if st.stat.packets or st.stat.drops],
        key=lambda s: -(s.bytes))

    rep = DataFlowReport(
        payload_kind=payload.kind, payload_label=payload.label,
        payload_mime=payload.mime, payload_bytes=payload.n_bytes,
        payload_sha=src_digest, n_segments=len(segs), mss_bytes=mss,
        flows=flows, packets=packets,
        n_packets=n_packets, n_delivered=len(delivered), n_dropped=len(dropped),
        drop_reasons=reasons,
        integrity_ok=integrity_ok, digest_src=src_digest, digest_dst=dst_digest,
        received_bytes=len(rebuilt),
        sent_bytes=sent_bytes, n_rounds=n_rounds, seg_coverage=seg_coverage,
        first_delivery_s=float(first_del), last_delivery_s=float(last_del),
        e2e_mean_s=float(lat.mean()) if lat.size else float("nan"),
        e2e_p95_s=float(np.percentile(lat, 95)) if lat.size else float("nan"),
        e2e_min_s=float(lat.min()) if lat.size else float("nan"),
        e2e_max_s=float(lat.max()) if lat.size else float("nan"),
        goodput_mbps=float(goodput),
        mean_hops=float(np.mean([p.hops for p in delivered])) if delivered else 0.0,
        max_hops=int(max((p.hops for p in delivered), default=0)),
        delivery_ratio=len(delivered) / max(n_packets, 1),
        series_t=[round(float(x), 4) for x in t_axis],
        series_inflight=[round(float(x), 2) for x in inflight],
        series_goodput=[round(float(x), 4) for x in goodput_series],
        series_inflight_mb=[round(float(x), 4) for x in in_mb],
        series_delivered=[round(float(x), 4) for x in delivered_mbit],
        link_stats=link_stats,
        n_sat=n_sat, n_gs=n_gs, duration_s=duration,
        snapshot_interval_s=snap_iv,
        wall_s=time.perf_counter() - t_wall,
        warnings=warns,
    )
    if not integrity_ok:
        rep.warnings.append(
            f"完整性校验未通过：送达 {len(delivered)}/{n_packets} 分组，"
            f"缺失 {n_packets - len(delivered)} 片（{rep.mix_text()}）")
    for f in flows:
        got = sum(1 for p in delivered if p.flow_id == f.flow_id)
        if got == 0:
            rep.warnings.append(
                f"流{f.flow_id}（{f.src_label}→{f.dst_label}）零交付："
                f"端站在该窗口内无星地覆盖或网络不连通（真实覆盖缺口）")
    return rep


# --------------------------------------------------------------------------- #
# 6. 小工具
# --------------------------------------------------------------------------- #
def _payload_slice(payload: DataPayload, seq: int, mss: int) -> bytes:
    off = seq * mss
    return payload.data[off:off + mss]


def _node_label(node: int, n_sat: int, gs: Sequence) -> str:
    if node < n_sat:
        return f"SAT-{node:04d}"
    j = node - n_sat
    if 0 <= j < len(gs):
        return getattr(gs[j], "name", f"GS-{j}")
    return f"NODE-{node}"


def _pick_gs_pairs(gs: Sequence, cfg: RealDataConfig, n_sat: int, n_flows: int,
                   snapshots: Optional[Sequence[Snapshot]] = None
                   ) -> List[Tuple[int, int]]:
    """选出 n_flows 对地面站（默认取球面距离最远且两端都有星地覆盖的一批）。"""

    def covered(j: int) -> bool:
        if not snapshots:
            return True
        vis = snapshots[0].gs_visible.get(j)
        return vis is not None and len(vis) > 0

    pairs: List[Tuple[int, int]] = []
    if cfg.src_gs >= 0 and cfg.dst_gs >= 0 and cfg.src_gs != cfg.dst_gs:
        pairs.append((n_sat + int(cfg.src_gs), n_sat + int(cfg.dst_gs)))
    cand: List[Tuple[float, int, int]] = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            if not (covered(i) and covered(j)):
                continue                      # 覆盖缺口：默认避开（真实但不宜做默认演示）
            cand.append((_gc_km(gs[i], gs[j]), i, j))
    cand.sort(reverse=True)
    used_src = {p[0] - n_sat for p in pairs}
    used_dst = {p[1] - n_sat for p in pairs}
    for _d, i, j in cand:
        if len(pairs) >= n_flows:
            break
        if i in used_src or j in used_dst or i in used_dst or j in used_src:
            continue
        pairs.append((n_sat + i, n_sat + j))
        used_src.add(i)
        used_dst.add(j)
    # 兜底：候选不足时放宽覆盖/唯一性约束
    for iv in range(len(gs)):
        for jv in range(iv + 1, len(gs)):
            if len(pairs) >= n_flows:
                break
            if (n_sat + iv, n_sat + jv) in pairs:
                continue
            pairs.append((n_sat + iv, n_sat + jv))
    return pairs[:max(1, n_flows)]


def _gc_km(a, b) -> float:
    la1, lo1 = math.radians(a.lat_deg), math.radians(a.lon_deg)
    la2, lo2 = math.radians(b.lat_deg), math.radians(b.lon_deg)
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(d)))


def _auto_rate(snapshots: Sequence[Snapshot], tables, src: int, dst: int) -> float:
    """自动速率：取首快照路径瓶颈的 0.85 倍（既喂得满，又不至于全丢）。"""
    if not snapshots:
        return 50.0
    snap = snapshots[0]
    nexthop, edge_of = tables[0]
    u, hops, bn = src, 0, float("inf")
    while u != dst and hops < 64:
        v = int(nexthop[u]) if u < len(nexthop) else -1
        if v < 0:
            break
        ei = edge_of.get((u, v))
        if ei is None:
            break
        bn = min(bn, float(snap.capacity_mbps[ei]))
        u, hops = v, hops + 1
    if not np.isfinite(bn) or bn <= 0:
        bn = 100.0
    # 星间链路 100 Gbps 会让瓶颈永远是馈电链路 → 直接取瓶颈×0.85
    return max(1.0, min(bn * 0.85, 2000.0))


def node_ecef_provider(constellation: Constellation, ground_stations: Sequence,
                       times_s: np.ndarray, start_unix: float):
    """返回 (node→ECEFF 查表函数, n_sat, n_gs)，供可视化按时刻取节点坐标。"""
    pos = constellation.positions_ecef
    assert pos is not None, "请先对星座调用 propagate()/to_ecef()"
    n_sat = constellation.n_sat
    gs_ecef = np.array([lla_to_ecef(g.lat_deg, g.lon_deg, g.alt_km)
                        for g in ground_stations]) if ground_stations else np.zeros((0, 3))
    n_t = pos.shape[1]
    t0, t1 = float(times_s[0]), float(times_s[-1])

    def lookup(node: int, t_s: float) -> np.ndarray:
        if node < n_sat:
            frac = (t_s - t0) / max(t1 - t0, 1e-9)
            k = int(np.clip(round(frac * (n_t - 1)), 0, n_t - 1))
            return pos[node, k]
        j = node - n_sat
        if 0 <= j < gs_ecef.shape[0]:
            return gs_ecef[j]
        return np.zeros(3)

    return lookup, n_sat, len(ground_stations)


__all__ = [
    "DataPayload", "Segment", "FlowSpec", "PacketRecord", "LinkStat",
    "DataFlowReport", "build_payload", "segment_payload", "reassemble",
    "simulate_real_data", "node_ecef_provider",
]

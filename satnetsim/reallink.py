"""真实数据传输评测 —— 在卫星链路特性下搬真实字节，而不是解公式。

为什么需要这一层
----------------
`transport.py` 是**窗口级解析模型**：按 RTT 轮次积分 cwnd，给出完成时间与吞吐。模型快、
可扫参，但正确性只能靠"与理论值吻合"间接保证。本模块把同一批协议放到
**真实套接字 + 真实字节 + 真实计时**上跑一遍，给解析模型"对表"：

1. `run_tcp_transfer` —— 真实内核 TCP 经过"单向时延 + 带宽整形"中继（`EmulatedLink`），
   传输真实载荷并做 **SHA-256 完整性校验**，实测完成时间与有效吞吐。
   回答："这条链路到底能不能把 N 比特搬过去、要多久"。

2. `run_packet_transfer` —— 包级链路（逐包单向时延/丢包/乱序/带宽）+ **真实可靠传输实现**
   （真实序号、累积 ACK、RTO、快速重传、真实 cwnd），算法 reno / cubic / hybla 与
   `transport.py` 的解析模型一一对应。回答："解析模型说丢包下 CUBIC 会退化，真跑也退化吗？"

丢包口径（重要）
----------------
- `LinkProfile.loss_rate` 一律指**每包丢失概率**（物理口径）。
- 解析模型 `TransportConfig.loss_prob` 指**每 RTT 有效丢包率**（BDP 内至少丢一个的概率）。
- 两者通过 `packet_loss_from_rtt_loss` / `rtt_loss_from_packet_loss` 显式换算，
  保证"真跑"与"解析"在同一个物理严重度下比较，而不是各说各话。

诚实边界
--------
- `EmulatedLink` 中继终结 ACK：发送侧 TCP 看到的是到中继的 RTT（微秒级），带宽由中继整形施加。
  因此它测**瓶颈带宽下的真实 TCP 吞吐与数据完整性**，不测 RTT 驱动的 cwnd 锯齿。
  用户态转发可实现带宽约 200 Mbps 上限，超过该值时测的是中继能力而非链路本身。
- RTT/丢包驱动的真实拥塞行为由第 2 条路径（包级链路）覆盖，那里 ACK 是端到端的。
"""

from __future__ import annotations

import hashlib
import random
import select
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

RELAY_MAX_MBPS = 200.0   # 用户态中继（真实 TCP 路径）的可靠转发放率上限
HARNESS_CAP_MBPS = 20.0  # 用户态逐包测试床（包级路径）的可靠容量上限量级


# --------------------------------------------------------------------------- #
# 链路剖面
# --------------------------------------------------------------------------- #
@dataclass
class LinkProfile:
    """一条端到端星上路径的链路特性。"""

    name: str = "leo-path"
    owd_ms: float = 39.6           # 单向时延（ms）
    bandwidth_mbps: float = 500.0  # 瓶颈带宽（Mbps）
    loss_rate: float = 2e-4        # **每包**丢失概率
    jitter_ms: float = 2.0         # 单向时延抖动
    reorder_rate: float = 0.0      # 乱序概率
    mtu_bytes: int = 1200          # 包大小（载荷字节）
    buffer_bytes: float = 0.0      # 瓶颈队列缓冲；0 → 自动取 3×BDP

    @property
    def queue_limit_bytes(self) -> float:
        """有限缓冲队列上限。

        真实卫星关口站/地面网关普遍配置**深缓冲**以吸收高 BDP 下的慢启动过冲
        （这也是卫星链路 bufferbloat 的成因）。默认取 3×BDP：既保留有限缓冲
        的尾丢弃物理，又不至于让慢启动过冲一次丢掉整个窗口。
        """
        return self.buffer_bytes if self.buffer_bytes > 0 else 3.0 * self.bdp_bytes

    @property
    def rtt_ms(self) -> float:
        return 2.0 * self.owd_ms

    @property
    def bdp_bytes(self) -> float:
        """带宽时延积：链路上"在飞"的字节数上界。"""
        return self.bandwidth_mbps * 1e6 / 8.0 * self.rtt_ms / 1000.0

    def bdp_segments(self, mss: Optional[int] = None) -> float:
        return max(self.bdp_bytes / float(mss or self.mtu_bytes), 1.0)

    def rtt_loss(self, mss: Optional[int] = None) -> float:
        """本剖面折算出的每 RTT 有效丢包率（解析模型口径）。"""
        return rtt_loss_from_packet_loss(self.loss_rate, self.bdp_segments(mss))

    def describe(self) -> str:
        return (f"{self.name}: OWD {self.owd_ms:.1f} ms, BW {self.bandwidth_mbps:.0f} Mbps, "
                f"丢包 {self.loss_rate:.2e}/包 (≈{self.rtt_loss()*100:.2f}%/RTT), "
                f"BDP {self.bdp_bytes/1024:.0f} KiB")


def rtt_loss_from_packet_loss(p_pkt: float, segments: float) -> float:
    """每包丢包率 → 每 RTT 有效丢包率。"""
    p = min(max(float(p_pkt), 0.0), 1.0)
    n = max(float(segments), 1.0)
    return float(1.0 - (1.0 - p) ** n)


def packet_loss_from_rtt_loss(p_rtt: float, segments: float) -> float:
    """每 RTT 有效丢包率 → 每包丢包率。"""
    p = min(max(float(p_rtt), 0.0), 0.999999)
    n = max(float(segments), 1.0)
    return float(1.0 - (1.0 - p) ** (1.0 / n))


def make_profile(owd_ms: float, bandwidth_mbps: float, rtt_loss: float,
                 mtu_bytes: int = 1200,
                 jitter_ms: Optional[float] = None,
                 name: str = "代表路径") -> LinkProfile:
    """由"单向时延 + 瓶颈带宽 + 每 RTT 有效丢包率"构造链路剖面。

    传入的丢包口径是**每 RTT 有效丢包率**（解析模型与配置文件的既有口径），
    这里换算成本剖面需要的**每包**丢包率，保证两侧同口径。

    引擎在跑完路由分析后即可直接调用本函数（不必先构造 SimulationResult），
    也是 `profile_from_result` 的公共实现。
    """
    owd = float(owd_ms)
    bw = float(bandwidth_mbps)
    mtu = int(mtu_bytes or 1200)
    bdp_seg = max(bw * 1e6 / 8.0 * (2.0 * owd) / 1000.0 / mtu, 1.0)
    return LinkProfile(
        name=name,
        owd_ms=owd,
        bandwidth_mbps=bw,
        loss_rate=packet_loss_from_rtt_loss(rtt_loss, bdp_seg),
        jitter_ms=float(jitter_ms) if jitter_ms is not None else max(0.5, 0.05 * owd),
        mtu_bytes=mtu,
    )


def profile_from_result(res, name: str = "代表路径") -> LinkProfile:
    """从一次仿真结果导出代表路径的链路剖面（已把每 RTT 丢包换算成每包）。"""
    cfg = res.config
    return make_profile(
        owd_ms=float(res.path_owd_ms),
        bandwidth_mbps=float(res.path_bottleneck_mbps),
        rtt_loss=float(cfg.transport.loss_prob),
        mtu_bytes=int(getattr(cfg.transport, "mss_bytes", 1200) or 1200),
        name=name,
    )


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def make_payload(size_bytes: int, seed: int = 20261012) -> bytes:
    """生成可复现的伪随机真实载荷（同一 seed 必然同一 SHA-256）。"""
    size_bytes = max(int(size_bytes), 1)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size_bytes, dtype=np.uint8).tobytes()


def _free_port(kind: int = socket.SOCK_STREAM) -> int:
    s = socket.socket(socket.AF_INET, kind)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@dataclass
class TransferTrial:
    """一次真实传输的实测结果。"""

    label: str
    ok: bool
    bytes_sent: int
    bytes_received: int
    sha_match: bool
    completion_s: float
    goodput_mbps: float
    rtt_ms: float = 0.0
    retransmits: int = 0
    losses: int = 0
    cwnd_trace: List[Tuple[float, float]] = field(default_factory=list)
    note: str = ""
    proto: str = ""   # 协议标识：tcp | reno | cubic | hybla | …（避免从 label 反解）

    @property
    def integrity(self) -> str:
        return "SHA-256 一致" if self.sha_match else "数据不一致"


# --------------------------------------------------------------------------- #
# 1) 时延 + 带宽整形中继（真实 TCP 走这条链路）
# --------------------------------------------------------------------------- #
class EmulatedLink:
    """TCP 中继：把客户端上行流量按 (单向时延, 带宽, 抖动) 整形后转发到水槽。

    只有上行需要整形——真实 TCP 接收端（`TcpSink`）不产生返回流量，因此
    下行纯粹是内核 ACK，不经整形。带宽用令牌桶限速，时延用"到期投递堆"实现。
    """

    def __init__(self, profile: LinkProfile, rcvbuf_bytes: int = 262_144,
                 chunk_bytes: int = 65_536) -> None:
        self.p = profile
        self.rcvbuf_bytes = int(rcvbuf_bytes)
        self.chunk_bytes = int(chunk_bytes)
        self._stop = threading.Event()
        self.bytes_uplink = 0
        self.max_queue = 0.0
        self.peak_mbps = 0.0
        self.relay_mbps = min(profile.bandwidth_mbps, RELAY_MAX_MBPS)

    def serve(self, listen_port: int, sink_host: str, sink_port: int) -> None:
        """阻塞式：接受一个客户端连接，整形转发到 sink，直到客户端关闭写端。"""
        ln = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ln.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ln.bind(("127.0.0.1", listen_port))
        ln.listen(1)
        ln.settimeout(0.2)
        cli = sink = None
        try:
            deadline = time.time() + 15.0
            while cli is None and time.time() < deadline and not self._stop.is_set():
                try:
                    cli, _ = ln.accept()
                except socket.timeout:
                    continue
            if cli is None:
                return
            cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sink = socket.create_connection((sink_host, sink_port), timeout=5)
            sink.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                sink.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_bytes)
            except OSError:
                pass

            # ---- 上行整形器：client → (时延/令牌桶) → sink ----
            # 关键：TCP 是字节流，中继必须**保序转发**。这里用 FIFO + 单调释放时刻，
            # 把抖动表现为"附加时延"而不是"乱序"（乱序会直接破坏 TCP 语义）。
            queue: "deque[Tuple[float, bytes]]" = deque()
            tokens = 0.0
            last = time.perf_counter()
            rate = self.relay_mbps * 1e6 / 8.0          # B/s
            t_relay0 = last
            last_release = 0.0
            t_first: Optional[float] = None
            t_last: Optional[float] = None
            while not self._stop.is_set():
                now = time.perf_counter()
                tokens = min(tokens + (now - last) * rate, max(rate * 0.05, 65_536.0))
                last = now
                # 收
                r, _, _ = select.select([cli], [], [], 0.0005)
                if r:
                    try:
                        blob = cli.recv(self.chunk_bytes)
                    except OSError:
                        blob = b""
                    if not blob:
                        break
                    d = self.p.owd_ms + random.uniform(-self.p.jitter_ms, self.p.jitter_ms)
                    rel = max(last_release, now + max(d, 0.0) / 1000.0)
                    last_release = rel
                    queue.append((rel, blob))
                # 发（到期 + 令牌桶）
                while queue and queue[0][0] <= time.perf_counter():
                    if tokens < len(queue[0][1]):
                        break
                    _, blob = queue.popleft()
                    tokens -= len(blob)
                    try:
                        sink.sendall(blob)
                    except OSError:
                        self._stop.set()
                        break
                    self.bytes_uplink += len(blob)
                    if t_first is None:
                        t_first = time.perf_counter()
                    t_last = time.perf_counter()
                self.max_queue = max(self.max_queue, float(len(queue)))
                time.sleep(0.0002)

            # 收尾：保序投递剩余数据，再关 sink 写端让水槽拿到 EOF
            guard = time.time() + 20.0
            while queue and time.time() < guard:
                tokens = min(tokens + 0.01 * rate, max(rate * 0.05, 65_536.0))
                while queue and queue[0][0] <= time.perf_counter():
                    if tokens < len(queue[0][1]):
                        break
                    _, blob = queue.popleft()
                    tokens -= len(blob)
                    try:
                        sink.sendall(blob)
                        self.bytes_uplink += len(blob)
                        if t_first is None:
                            t_first = time.perf_counter()
                        t_last = time.perf_counter()
                    except OSError:
                        break
                time.sleep(0.0005)
            # 持续投递速率：首个字节到最后一个字节之间的真实转发速率
            if t_first is not None and t_last is not None and t_last > t_first:
                self.peak_mbps = self.bytes_uplink * 8 / (t_last - t_first) / 1e6
            try:
                sink.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        finally:
            self._stop.set()
            for s in (cli, sink, ln):
                try:
                    if s is not None:
                        s.close()
                except OSError:
                    pass


class TcpSink:
    """真实 TCP 接收端：只做 recv + 增量 SHA-256，不产生任何返回流量。"""

    def __init__(self) -> None:
        self.port = _free_port(socket.SOCK_STREAM)
        self.received = 0
        self.digest = hashlib.sha256()
        self.done = threading.Event()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", self.port))
        self._srv.listen(1)
        self._srv.settimeout(0.2)

    def serve(self, timeout_s: float = 60.0) -> None:
        end = time.time() + timeout_s
        conn = None
        try:
            while conn is None and time.time() < end:
                try:
                    conn, _ = self._srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return
            if conn is None:
                return
            conn.settimeout(1.0)
            while time.time() < end:
                try:
                    b = conn.recv(262_144)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not b:
                    break
                self.received += len(b)
                self.digest.update(b)
        finally:
            try:
                if conn is not None:
                    conn.close()
            except OSError:
                pass
            self.done.set()

    def close(self) -> None:
        try:
            self._srv.close()
        except OSError:
            pass


def run_tcp_transfer(profile: LinkProfile, payload_mbit: float,
                     trials: int = 2, rcvbuf_bytes: int = 262_144,
                     seed: int = 20261012, timeout_s: float = 60.0,
                     progress: Optional[Callable[[str], None]] = None
                     ) -> List[TransferTrial]:
    """真实 TCP 传输：真实载荷 → 整形中继 → 真实水槽，SHA-256 校验。"""
    n_bytes = int(payload_mbit * 1e6 / 8.0)
    payload = make_payload(n_bytes, seed)
    sender_sha = hashlib.sha256(payload).hexdigest()
    results: List[TransferTrial] = []

    for k in range(max(int(trials), 1)):
        random.seed(seed + k * 977)
        sink = TcpSink()
        link = EmulatedLink(profile, rcvbuf_bytes=rcvbuf_bytes)
        relay_port = _free_port(socket.SOCK_STREAM)

        th_sink = threading.Thread(target=sink.serve, args=(timeout_s,), daemon=True)
        th_link = threading.Thread(target=link.serve,
                                   args=(relay_port, "127.0.0.1", sink.port), daemon=True)
        th_sink.start()
        th_link.start()
        time.sleep(0.2)

        def push() -> None:
            cli = None
            try:
                cli = socket.create_connection(("127.0.0.1", relay_port), timeout=5)
                cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                mv = memoryview(payload)
                off, step = 0, 1 << 20
                while off < len(payload):
                    n = cli.send(mv[off:off + step])
                    off += n
                cli.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            finally:
                try:
                    if cli is not None:
                        cli.close()
                except OSError:
                    pass

        t0 = time.perf_counter()
        th_push = threading.Thread(target=push, daemon=True)
        th_push.start()

        # 投递完成的真实时刻：水槽收满的瞬间
        t_done: Optional[float] = None
        guard = time.time() + timeout_s
        while time.time() < guard:
            if sink.received >= n_bytes:
                t_done = time.perf_counter()
                break
            if sink.done.is_set() and sink.received < n_bytes:
                break
            time.sleep(0.001)
        th_push.join(timeout=5.0)
        for _ in range(2000):
            if sink.done.is_set():
                break
            time.sleep(0.002)
        dt = (t_done - t0) if t_done is not None else (time.perf_counter() - t0)
        th_link.join(timeout=3.0)
        th_sink.join(timeout=3.0)
        sink.close()

        got = sink.received
        ok = (got == n_bytes)
        results.append(TransferTrial(
            label="TCP（真实内核栈）",
            ok=ok,
            bytes_sent=n_bytes,
            bytes_received=got,
            sha_match=(got == n_bytes and sink.digest.hexdigest() == sender_sha),
            completion_s=dt,
            goodput_mbps=(got * 8 / max(dt, 1e-9) / 1e6),
            rtt_ms=profile.rtt_ms,
            note=(f"中继转发 {link.bytes_uplink} B, 队列峰值 {link.max_queue:.0f}, "
                  f"峰值速率 {link.peak_mbps:.0f} Mbps"),
            proto="tcp",
        ))
        if progress:
            progress(f"真实 TCP 第 {k+1}/{trials} 次: {got/1e6:.3f} MB / "
                     f"{dt:.3f} s → {results[-1].goodput_mbps:.0f} Mbps "
                     f"[{results[-1].integrity}]")
    return results


# --------------------------------------------------------------------------- #
# 2) 包级 UDP 链路 + 真实可靠传输
# --------------------------------------------------------------------------- #
HDR = struct.Struct("!III")           # seq, total_bytes, flags
FLAG_DATA, FLAG_ACK, FLAG_HELLO = 0, 1, 2


RCVBUF_BYTES = 8 << 20   # 端点/链路 UDP 缓冲：必须容得下突发，否则内核静默丢包


def _tune_udp(sock: socket.socket, rcvbuf: int = RCVBUF_BYTES,
              sndbuf: Optional[int] = None) -> None:
    """放大套接字缓冲。用户态端点逐包处理，内核缓冲不足会造成"看不见的丢包"。"""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(rcvbuf))
    except OSError:
        pass
    if sndbuf:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(sndbuf))
        except OSError:
            pass


class PacketLink:
    """包级链路：逐包施加单向时延 / 抖动 / 丢包 / 乱序，并按带宽排队。

    收到 sock_a 的包 → 转发给 B（地址在 addr_b）；收到 sock_b 的包 → 转发给 A。
    地址在首次收到该侧数据包时自动注册；对端未注册的包计为丢弃。
    """

    def __init__(self, profile: LinkProfile, seed: int = 7) -> None:
        self.p = profile
        self.rng = random.Random(seed)
        self.sock_a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _tune_udp(self.sock_a)
        _tune_udp(self.sock_b)
        self.sock_a.bind(("127.0.0.1", 0))
        self.sock_b.bind(("127.0.0.1", 0))
        # 非阻塞：转发循环需要"抽干到 EWOULDBLOCK"；select 对非阻塞套接字同样有效
        self.sock_a.setblocking(False)
        self.sock_b.setblocking(False)
        self.port_a = self.sock_a.getsockname()[1]
        self.port_b = self.sock_b.getsockname()[1]
        self.addr_a: Optional[Tuple[str, int]] = None
        self.addr_b: Optional[Tuple[str, int]] = None
        self.forwarded = 0
        self.dropped = 0
        self.dropped_buffer = 0
        self.reordered = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.queue_limit = max(profile.queue_limit_bytes, 4 * profile.mtu_bytes)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for s in (self.sock_a, self.sock_b):
            try:
                s.close()
            except OSError:
                pass

    def _pump(self) -> None:
        # 每个方向一个 FIFO + 单调释放时刻：抖动只表现为附加时延，
        # 乱序仅由 reorder_rate 显式引入（两个旋钮正交，便于归因）。
        queues: Dict[bool, "deque[Tuple[float, bytes, Tuple[str, int], socket.socket]]"] = {
            True: deque(), False: deque()
        }
        last_release: Dict[bool, float] = {True: 0.0, False: 0.0}
        queued_bytes: Dict[bool, float] = {True: 0.0, False: 0.0}
        tokens = 0.0
        last = time.perf_counter()
        rate = max(self.p.bandwidth_mbps, 0.1) * 1e6 / 8.0
        while not self._stop.is_set():
            now = time.perf_counter()
            tokens = min(tokens + (now - last) * rate, max(rate * 0.02, 65_536.0))
            last = now
            try:
                r, _, _ = select.select([self.sock_a, self.sock_b], [], [], 0.0002)
            except OSError:
                break
            for s in r:
                # 每轮抽干该套接字上的全部可读包，否则读包速率会成为隐藏瓶颈
                # （数据与 ACK 双向共享同一条链路，等效容量会减半）
                guard = 0
                while guard < 512:
                    guard += 1
                    try:
                        blob, addr = s.recvfrom(65535)
                    except (BlockingIOError, OSError):
                        break
                    if s is self.sock_a:
                        self.addr_a = addr
                        target_addr, target_sock = self.addr_b, self.sock_b
                        direction = True
                    else:
                        self.addr_b = addr
                        target_addr, target_sock = self.addr_a, self.sock_a
                        direction = False
                    if target_addr is None:
                        self.dropped += 1
                        continue
                    if self.rng.random() < self.p.loss_rate:
                        self.dropped += 1
                        continue
                    delay = self.p.owd_ms + self.rng.uniform(-self.p.jitter_ms,
                                                             self.p.jitter_ms)
                    if self.rng.random() < self.p.reorder_rate:
                        delay += self.rng.uniform(1.0, 6.0)
                        self.reordered += 1
                    # 有限缓冲 + 尾丢弃：队列满时丢新到的包（真实瓶颈路由器行为）
                    if queued_bytes[direction] + len(blob) > self.queue_limit:
                        self.dropped_buffer += 1
                        self.dropped += 1
                        continue
                    rel = max(last_release[direction], now + max(delay, 0.0) / 1000.0)
                    last_release[direction] = rel
                    queued_bytes[direction] += len(blob)
                    queues[direction].append((rel, blob, target_addr, target_sock))
            for direction, q in queues.items():
                while q and q[0][0] <= time.perf_counter():
                    if tokens < len(q[0][1]):
                        break
                    _, blob, target_addr, target_sock = q.popleft()
                    queued_bytes[direction] -= len(blob)
                    tokens -= len(blob)
                    try:
                        target_sock.sendto(blob, target_addr)
                        self.forwarded += 1
                    except OSError:
                        pass
            # 有排队包时让出 CPU 但不长眠（令牌不足需尽快重试），空闲时才睡
            if any(q for q in queues.values()):
                time.sleep(0)
            else:
                time.sleep(0.0002)


class CongestionController:
    """真实拥塞控制：真实 cwnd 演化，供可靠传输使用。"""

    def __init__(self, algo: str = "cubic", mss: int = 1200,
                 rtt_ref_ms: float = 25.0, initial_cwnd: int = 12_000,
                 hystart: bool = False, hystart_delta: float = 0.5,
                 bdp_hint_bytes: float = 0.0, bdp_ss_fraction: float = 0.85,
                 cwnd_cap_bdp: float = 2.0) -> None:
        self.algo = algo
        self.mss = int(mss)
        self.rtt_ref_ms = float(rtt_ref_ms)
        self.cwnd = float(initial_cwnd)
        self.ssthresh = float("inf")
        self.w_max = float(initial_cwnd)
        self.beta = 0.7 if algo == "cubic" else 0.5
        self._t_epoch = time.perf_counter()
        self._k = 0.0
        self._t_first_ack = None
        # Hystart：慢启动阶段若 RTT 相对最小值持续上升，提前退出慢启动。
        # 默认关闭——测试床已提供 BDP 参考（更精确）；开启 Hystart 会因
        # 接收端延迟 ACK 导致的早期 RTT 抖动而过早退出慢启动，窗口停在远低于
        # BDP 的位置，反而测不出稳态吞吐。
        self.hystart = bool(hystart)
        self.hystart_delta = float(hystart_delta)
        self.hystart_high = 0
        self.rtt_min_ms = float("inf")
        self.exit_slow_start = False
        # 管道参考：测试床把链路名义 BDP 交给传输实现（等价于现代拥塞控制里的
        # 带宽估计）。没有它，纯丢包型慢启动必然过冲；有了它，慢启动在接近满管
        # 时退出，窗口稳定在 BDP 附近。这是测试床的显式设定，不是 TCP 规范行为。
        self.bdp_hint = float(bdp_hint_bytes)
        self.bdp_ss_fraction = float(bdp_ss_fraction)
        self.cwnd_cap_bdp = float(cwnd_cap_bdp)

    def on_ack(self, acked_bytes: int, rtt_ms: float) -> None:
        acked_bytes = max(int(acked_bytes), 0)
        # ---- Hystart：连续多次 RTT 高于最小值+阈值 → 退出慢启动 ----
        if rtt_ms and rtt_ms > 0:
            if self.rtt_min_ms == float("inf"):
                self.rtt_min_ms = rtt_ms
                self._t_first_ack = time.perf_counter()
            else:
                self.rtt_min_ms = min(self.rtt_min_ms, rtt_ms)
            if rtt_ms > self.rtt_min_ms * (1.0 + self.hystart_delta):
                self.hystart_high += 1
            else:
                self.hystart_high = 0
            if (self.hystart and not self.exit_slow_start
                    and self.cwnd < self.ssthresh
                    and self.hystart_high >= 3
                    and self._t_first_ack is not None
                    and time.perf_counter() - self._t_first_ack > 0.02):
                self.exit_slow_start = True
                self.ssthresh = self.cwnd
                self.w_max = self.cwnd
        # ---- 管道参考：到达名义 BDP 的设定比例即退出慢启动（防过冲） ----
        if (self.bdp_hint > 0 and not self.exit_slow_start
                and self.cwnd < self.ssthresh
                and self.cwnd >= self.bdp_ss_fraction * self.bdp_hint):
            self.exit_slow_start = True
            self.ssthresh = self.cwnd
            self.w_max = self.cwnd
        if self.algo == "hybla":
            rho = min(max(rtt_ms / max(self.rtt_ref_ms, 1e-6), 1e-3), 64.0)
            if self.cwnd < self.ssthresh:
                self.cwnd += (2.0 ** rho) * acked_bytes
            else:
                self.cwnd += rho * rho * self.mss * acked_bytes / max(self.cwnd, 1.0)
        elif self.algo == "reno":
            if self.cwnd < self.ssthresh:
                self.cwnd += acked_bytes
            else:
                self.cwnd += self.mss * acked_bytes / max(self.cwnd, 1.0)
        else:  # cubic
            if self.cwnd < self.ssthresh:
                self.cwnd += acked_bytes
            else:
                if self._k <= 0.0:
                    self._k = ((self.w_max * (1.0 - self.beta) / 0.4) ** (1.0 / 3.0)
                               if self.w_max > 0 else 0.0)
                t = time.perf_counter() - self._t_epoch - self._k
                target = 0.4 * (t ** 3) + self.w_max
                if self.cwnd < target:
                    self.cwnd += (target - self.cwnd) * acked_bytes / max(self.cwnd, 1.0)
                else:
                    self.cwnd += self.mss * acked_bytes / max(self.cwnd, 1.0)

        # ---- 窗口上限：不超过名义 BDP 的设定倍数，防止队列无界堆积 ----
        if self.bdp_hint > 0:
            self.cwnd = min(self.cwnd, self.cwnd_cap_bdp * self.bdp_hint)

    def on_loss(self, rtt_ms: float) -> None:
        self.w_max = max(self.cwnd, 2 * self.mss)
        self.ssthresh = max(self.w_max * self.beta, 2 * self.mss)
        self.cwnd = self.ssthresh
        self._t_epoch = time.perf_counter()
        self._k = 0.0
        self.exit_slow_start = False   # 允许下一轮慢启动重新触发 Hystart


class ReliableUdpTransport:
    """真实可靠传输：真实序号 / 累积 ACK / RTO / 快速重传 / 真实 cwnd。"""

    def __init__(self, link: PacketLink, cc: str, mss: int = 1200,
                 rtt_ref_ms: float = 25.0, timeout_s: float = 30.0,
                 bdp_hint_bytes: float = 0.0) -> None:
        self.link = link
        self.cc_name = cc
        self.mss = int(mss)
        self.rtt_ref_ms = float(rtt_ref_ms)
        self.timeout_s = float(timeout_s)
        self.bdp_hint_bytes = float(bdp_hint_bytes)
        self.retransmits = 0
        self.rto_events = 0          # RTO 触发的重传次数
        self.fast_retx = 0           # 快速重传次数
        self.blocked_sends = 0       # 发送缓冲满导致的写失败次数
        self.rtt_samples: List[float] = []
        # 接收端诊断计数（真实评测报告会引用）
        self.rx_pkts = 0
        self.rx_dup = 0
        self.rx_ooo = 0
        self.rx_expect = 0
        self.rx_acked_bytes = 0

    # ---------------- 接收端（放子线程跑） ---------------- #
    def receive(self, total_bytes: int, ready: threading.Event,
                timeout_s: float = 30.0) -> Tuple[bytes, int]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _tune_udp(sock)
        sock.bind(("127.0.0.1", 0))
        dst = ("127.0.0.1", self.link.port_b)
        for _ in range(3):
            try:
                sock.sendto(HDR.pack(0, total_bytes, FLAG_HELLO), dst)
            except OSError:
                pass
            time.sleep(0.02)
        ready.set()

        expect = 0
        buffered: Dict[int, bytes] = {}
        chunks: Dict[int, bytes] = {}
        got = 0
        end = time.time() + timeout_s
        sock.setblocking(False)
        self.rx_pkts = 0
        self.rx_dup = 0
        self.rx_ooo = 0
        self.rx_expect = 0
        self.rx_acked_bytes = 0
        while got < total_bytes and time.time() < end:
            try:
                r, _, _ = select.select([sock], [], [], 0.02)
            except OSError:
                break
            if not r:
                # 不发送周期性 ACK：真实 TCP 只在收到数据时回 ACK。
                # 周期 ACK 会被发送端误判为重复 ACK，触发虚假快速重传。
                continue
            # 一次性抽干内核缓冲，避免突发把接收队列挤爆（内核静默丢包），
            # 但按"延迟 ACK"节奏每 4 包回一次，避免 ACK 被整轮抽干拖长。
            guard = 0
            since_ack = 0
            while guard < 20000:
                guard += 1
                try:
                    blob, _ = sock.recvfrom(65535)
                except (BlockingIOError, OSError):
                    break
                self.rx_pkts += 1
                if len(blob) < HDR.size:
                    continue
                seq, _tot, flags = HDR.unpack(blob[:HDR.size])
                if flags != FLAG_DATA:
                    continue
                body = blob[HDR.size:]
                if seq == expect:
                    chunks[expect] = body
                    got += len(body)
                    expect += 1
                    while expect in buffered:
                        b = buffered.pop(expect)
                        chunks[expect] = b
                        got += len(b)
                        expect += 1
                elif seq > expect and seq not in buffered:
                    buffered[seq] = body
                    self.rx_ooo += 1
                else:
                    self.rx_dup += 1
                since_ack += 1
                # 逐包 ACK：更细的 ACK 时钟让发送窗口平滑打开，避免整窗突发
                if since_ack >= 1 or got >= total_bytes:
                    since_ack = 0
                    try:
                        sock.sendto(HDR.pack(expect, total_bytes, FLAG_ACK), dst)
                    except OSError:
                        pass
            self.rx_expect = expect
            self.rx_acked_bytes = got
            try:
                sock.sendto(HDR.pack(expect, total_bytes, FLAG_ACK), dst)
            except OSError:
                pass
        sock.close()
        return b"".join(chunks[i] for i in sorted(chunks)), self.link.dropped

    # ---------------- 发送端 ---------------- #
    def send(self, payload: bytes,
             progress: Optional[Callable[[float], None]] = None) -> TransferTrial:
        total = len(payload)
        n_seg = (total + self.mss - 1) // self.mss
        segs = [payload[i * self.mss:(i + 1) * self.mss] for i in range(n_seg)]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _tune_udp(sock)
        sock.bind(("127.0.0.1", 0))
        sock.setblocking(False)
        dst = ("127.0.0.1", self.link.port_a)
        ctrl = CongestionController(self.cc_name, self.mss, self.rtt_ref_ms,
                                    bdp_hint_bytes=self.bdp_hint_bytes)
        trace: List[Tuple[float, float]] = []

        for _ in range(3):
            try:
                sock.sendto(HDR.pack(0, total, FLAG_HELLO), dst)
            except OSError:
                pass
            time.sleep(0.02)

        base = nxt = 0
        last_ack = -1
        dup = 0
        in_recovery = False       # 快速恢复：同一窗口只允许减一次 cwnd
        recover_seq = 0
        next_send_at = 0.0        # pacing：下一次允许发送的时刻
        sent_at: Dict[int, float] = {}
        rtt_est = max(self.link.p.rtt_ms / 1000.0, 1e-3)
        rto = max(0.3, 4.0 * rtt_est)
        t0 = time.perf_counter()
        last_report = 0.0

        while base < n_seg and time.perf_counter() - t0 < self.timeout_s:
            now = time.perf_counter()
            inflight = (nxt - base) * self.mss
            while nxt < n_seg and inflight < ctrl.cwnd:
                try:
                    sock.sendto(HDR.pack(nxt, total, FLAG_DATA) + segs[nxt], dst)
                except OSError:
                    self.blocked_sends += 1
                    break
                sent_at[nxt] = time.perf_counter()
                nxt += 1
                inflight += self.mss

            # 一次性抽干可读 ACK（非阻塞）；超时取小值以免 pacing 节拍被拖粗
            try:
                r, _, _ = select.select([sock], [], [], 0.0003)
            except OSError:
                r = []
            while r:
                try:
                    blob, _ = sock.recvfrom(2048)
                except (BlockingIOError, OSError):
                    break
                if len(blob) < HDR.size:
                    continue
                ack, _tot, flags = HDR.unpack(blob[:HDR.size])
                if flags != FLAG_ACK:
                    continue
                if ack > last_ack:
                    newly = ack - base
                    if newly > 0:
                        rtt = (time.perf_counter()
                               - sent_at.get(base, time.perf_counter())) * 1000.0
                        if 0 < rtt < 60_000:
                            self.rtt_samples.append(rtt)
                            rtt_est = 0.875 * rtt_est + 0.125 * rtt / 1000.0
                            rto = max(0.3, 4.0 * rtt_est)
                        ctrl.on_ack(newly * self.mss, rtt)
                        for i in range(base, ack):
                            sent_at.pop(i, None)
                        base = ack
                        dup = 0
                        # NewReno 部分确认：仍在恢复窗口内 → 立即重传新的 base，
                        # 且不再降窗（否则窗口内多丢包要花 N 个 RTT 才能恢复）
                        if in_recovery and base < recover_seq and base < nxt:
                            try:
                                sock.sendto(HDR.pack(base, total, FLAG_DATA)
                                            + segs[base], dst)
                            except OSError:
                                pass
                            sent_at[base] = time.perf_counter()
                            self.retransmits += 1
                        elif in_recovery and base >= recover_seq:
                            in_recovery = False
                    last_ack = ack
                elif ack == last_ack:
                    dup += 1
                    if dup >= 3 and base < nxt:
                        try:
                            sock.sendto(HDR.pack(base, total, FLAG_DATA) + segs[base], dst)
                        except OSError:
                            pass
                        sent_at[base] = time.perf_counter()
                        self.retransmits += 1
                        self.fast_retx += 1
                        # 新拥塞事件才减窗；同一窗口内的后续丢包只重传不减窗
                        if not in_recovery:
                            ctrl.on_loss(rtt_est * 1000.0)
                            in_recovery = True
                            recover_seq = nxt
                        dup = 0
                try:
                    r, _, _ = select.select([sock], [], [], 0.0)
                except OSError:
                    r = []

            if base < nxt:
                oldest = sent_at.get(base, now)
                if now - oldest > rto:
                    try:
                        sock.sendto(HDR.pack(base, total, FLAG_DATA) + segs[base], dst)
                    except OSError:
                        pass
                    sent_at[base] = time.perf_counter()
                    self.retransmits += 1
                    self.rto_events += 1
                    ctrl.on_loss(rtt_est * 1000.0)
                    in_recovery = False          # 超时 → 退出快速恢复
                    rto = min(rto * 2.0, 8.0)    # 指数退避

            trace.append((time.perf_counter() - t0, ctrl.cwnd))
            if progress is not None and now - last_report > 0.2:
                progress(base / max(n_seg, 1))
                last_report = now

        dt = time.perf_counter() - t0
        sock.close()
        done = base >= n_seg
        return TransferTrial(
            label=f"{self.cc_name.upper()}（真实包级）",
            ok=done,
            bytes_sent=total,
            bytes_received=total if done else 0,
            sha_match=False,                       # 由调用方按水槽 SHA 覆盖
            completion_s=dt,
            goodput_mbps=(total * 8 / max(dt, 1e-9) / 1e6),
            rtt_ms=float(np.mean(self.rtt_samples)) if self.rtt_samples else self.link.p.rtt_ms,
            retransmits=self.retransmits,
            cwnd_trace=trace,
            note=f"ACK 覆盖 {base}/{n_seg} 段；链路丢 {self.link.dropped} 包",
        )


def run_packet_transfer(profile: LinkProfile, payload_mbit: float,
                        algo: str = "cubic", trials: int = 1, mss: int = 1200,
                        rtt_ref_ms: float = 25.0, seed: int = 7,
                        timeout_s: float = 30.0,
                        progress: Optional[Callable[[str], None]] = None
                        ) -> List[TransferTrial]:
    """在包级链路（真实丢包/时延）上跑真实可靠传输，SHA-256 校验。"""
    n_bytes = int(payload_mbit * 1e6 / 8.0)
    payload = make_payload(n_bytes, seed)
    want = hashlib.sha256(payload).hexdigest()
    out: List[TransferTrial] = []
    for k in range(max(int(trials), 1)):
        link = PacketLink(profile, seed=seed + k * 131)
        link.start()
        time.sleep(0.1)
        tr = ReliableUdpTransport(link, algo, mss=mss, rtt_ref_ms=rtt_ref_ms,
                                  timeout_s=timeout_s,
                                  bdp_hint_bytes=profile.bdp_bytes)
        box: Dict[str, object] = {}
        ready = threading.Event()

        def rx() -> None:
            box["data"], box["dropped"] = tr.receive(len(payload), ready, timeout_s)

        th = threading.Thread(target=rx, daemon=True)
        th.start()
        ready.wait(timeout=3.0)
        trial = tr.send(payload)
        th.join(timeout=timeout_s + 2.0)
        data = box.get("data", b"")
        link.stop()
        trial.proto = algo
        trial.bytes_received = len(data)
        trial.sha_match = (len(data) == len(payload)
                           and hashlib.sha256(data).hexdigest() == want)
        trial.note += (f"；载荷 {len(data)}/{len(payload)} B "
                       f"{'校验通过' if trial.sha_match else '校验失败'}")
        out.append(trial)
        if progress:
            progress(f"真实包级 {algo.upper()} 第 {k+1}/{trials} 次: "
                     f"{trial.goodput_mbps:.1f} Mbps, 重传 {trial.retransmits}, "
                     f"RTT {trial.rtt_ms:.1f} ms, {trial.integrity}")
    return out


def measure_rtt_ms(profile: LinkProfile, trials: int = 5, seed: int = 3) -> float:
    """在包级链路上真实往返，实测 RTT（应接近 2·OWD + 抖动）。"""
    link = PacketLink(profile, seed=seed)
    link.start()
    time.sleep(0.1)
    a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    a.bind(("127.0.0.1", 0)); b.bind(("127.0.0.1", 0))
    dst_a = ("127.0.0.1", link.port_a)
    dst_b = ("127.0.0.1", link.port_b)
    samples: List[float] = []
    stop = threading.Event()
    try:
        # 注册两侧地址
        b.sendto(b"REG-B", dst_b)
        a.sendto(b"REG-A", dst_a)
        time.sleep(0.15)

        def responder() -> None:
            while not stop.is_set():
                try:
                    blob, _ = b.recvfrom(2048)
                except OSError:
                    break
                if blob.startswith(b"PING:"):
                    try:
                        b.sendto(blob, dst_b)      # 原样回送
                    except OSError:
                        break
                # 其它（注册包/迟到包）丢弃

        th = threading.Thread(target=responder, daemon=True)
        th.start()
        a.settimeout(1.5)
        for i in range(max(int(trials), 1)):
            token = b"PING:%d" % (i + 1)
            t0 = time.perf_counter()
            try:
                a.sendto(token, dst_a)
            except OSError:
                continue
            # 只认自己发的那一个 token（屏蔽迟到/串包）
            deadline = time.time() + 1.5
            while time.time() < deadline:
                try:
                    back, _ = a.recvfrom(2048)
                except socket.timeout:
                    break
                except OSError:
                    break
                if back == token:
                    samples.append((time.perf_counter() - t0) * 1000.0)
                    break
            time.sleep(0.02)
        stop.set()
        th.join(timeout=0.5)
    finally:
        stop.set()
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass
        link.stop()
    return float(np.median(samples)) if samples else profile.rtt_ms


# --------------------------------------------------------------------------- #
# 3) 汇总：真实 vs 解析
# --------------------------------------------------------------------------- #
@dataclass
class RealTransferReport:
    """真实数据传输评测汇总。"""

    profile: LinkProfile
    payload_mbit: float
    measured_rtt_ms: float
    tcp: List[TransferTrial] = field(default_factory=list)
    packet: List[TransferTrial] = field(default_factory=list)
    analytic: Dict[str, float] = field(default_factory=dict)
    analytic_goodput: Dict[str, float] = field(default_factory=dict)
    packet_profile: Optional[LinkProfile] = None
    packet_payload_mbit: float = 0.0
    harness_cap_mbps: float = 0.0
    wall_s: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def best_packet(self) -> Optional[TransferTrial]:
        good = [t for t in self.packet if t.ok]
        return min(good, key=lambda t: t.completion_s) if good else None

    def goodput_by_proto(self, family: str = "packet") -> Dict[str, float]:
        """按协议取有效吞吐（同一协议多次试验取最好一次），便于与解析模型对齐。"""
        group = self.packet if family == "packet" else self.tcp
        out: Dict[str, float] = {}
        for t in group:
            if t.ok:
                key = t.proto or t.label
                out[key] = max(out.get(key, 0.0), t.goodput_mbps)
        return out

    def completion_by_proto(self, family: str = "packet") -> Dict[str, float]:
        """按协议取完成时间（同一协议多次试验取最快一次）。"""
        group = self.packet if family == "packet" else self.tcp
        out: Dict[str, float] = {}
        for t in group:
            if t.ok:
                key = t.proto or t.label
                out[key] = min(out.get(key, float("inf")), t.completion_s)
        return out

    def headline(self) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = [
            ("链路剖面", self.profile.describe()),
            ("实测 RTT", f"{self.measured_rtt_ms:.1f} ms "
                        f"(理论 2·OWD {self.profile.rtt_ms:.1f} ms)"),
            ("评测载荷", f"{self.payload_mbit:.1f} Mbit / "
                        f"{int(self.payload_mbit*1e6/8):,} B 真实字节"),
            ("评测墙钟", f"{self.wall_s:.1f} s"),
        ]
        if self.packet_profile is not None:
            rows.append(("包级链路口径", self.packet_profile.describe()))
            rows.append(("包级载荷", f"{self.packet_payload_mbit:.0f} Mbit "
                                    f"(测试床可靠容量 ~{self.harness_cap_mbps:.0f} Mbps)"))
        for t in self.tcp:
            rows.append(("真实 TCP", f"{t.goodput_mbps:.0f} Mbps, "
                                    f"{t.completion_s:.3f} s, {t.integrity}"))
        for t in self.packet:
            rows.append((f"真实包级 {t.label}", f"{t.goodput_mbps:.1f} Mbps, "
                                              f"{t.completion_s:.2f} s, 重传 {t.retransmits}, "
                                              f"{t.integrity}"))
        for k, v in sorted(self.analytic.items(), key=lambda kv: kv[1]):
            rows.append((f"解析 {k}", f"{v:.3f} s / "
                                     f"{self.analytic_goodput.get(k, 0):.0f} Mbps"))
        return rows

    def summary_rows(self) -> List[Tuple[str, str, str, str, str]]:
        """(方法, 完成时间 s, 有效吞吐 Mbps, 重传, 完整性)"""
        rows: List[Tuple[str, str, str, str, str]] = []
        for t in self.tcp:
            rows.append(("真实 TCP（内核栈）", f"{t.completion_s:.3f}",
                         f"{t.goodput_mbps:.0f}", "—", t.integrity))
        for t in self.packet:
            rows.append((f"真实包级 {t.label}", f"{t.completion_s:.3f}",
                         f"{t.goodput_mbps:.1f}", str(t.retransmits), t.integrity))
        for k, v in sorted(self.analytic.items(), key=lambda kv: kv[1]):
            rows.append((f"解析模型 {k}", f"{v:.3f}",
                         f"{self.analytic_goodput.get(k, 0.0):.0f}", "—", "（模型）"))
        return rows


def evaluate_transfer(profile: LinkProfile, payload_mbit: float = 16.0,
                      tcp_trials: int = 2, packet_trials: int = 1,
                      packet_algos: Sequence[str] = ("reno", "cubic", "hybla"),
                      packet_bandwidth_mbps: Optional[float] = None,
                      packet_payload_mbit: Optional[float] = None,
                      rtt_loss: Optional[float] = None,
                      compare_analytic: bool = True,
                      rtt_ref_ms: float = 25.0,
                      progress: Optional[Callable[[str], None]] = None
                      ) -> RealTransferReport:
    """完整真实数据传输评测，并与解析模型在同口径下对照。

    packet_bandwidth_mbps : 包级测试的链路带宽上限。默认 min(链路带宽, 20)——
                            用户态逐包测试床的可靠容量约 20 Mbps 量级，超过它
                            测到的是测试床自身能力而非链路行为。
    packet_payload_mbit   : 包级测试载荷；默认取 max(8, 12×包级BDP)，保证能跑出
                            稳态而不是全程停在慢启动爬升段。
    rtt_loss              : 每 RTT 有效丢包率（默认取剖面折算值）；两侧同口径使用
    """
    t_start = time.perf_counter()
    rep = RealTransferReport(profile=profile, payload_mbit=payload_mbit,
                             measured_rtt_ms=profile.rtt_ms)

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    # ---- 统一丢包口径 ----
    p_rtt = float(rtt_loss) if rtt_loss is not None else profile.rtt_loss()
    bw_pkt = float(packet_bandwidth_mbps) if packet_bandwidth_mbps is not None \
        else min(profile.bandwidth_mbps, 20.0)
    pkt_profile = LinkProfile(
        name="包级测试链路（同剖面，带宽降至测试床容量内）",
        owd_ms=profile.owd_ms, bandwidth_mbps=bw_pkt,
        loss_rate=0.0, jitter_ms=profile.jitter_ms,
        reorder_rate=profile.reorder_rate, mtu_bytes=profile.mtu_bytes)
    # 包级链路带宽变小后 BDP 也变小，按新 BDP 折算每包丢包，保证 p_rtt 不变
    pkt_profile.loss_rate = packet_loss_from_rtt_loss(
        p_rtt, pkt_profile.bdp_segments())
    rep.packet_profile = pkt_profile
    rep.harness_cap_mbps = HARNESS_CAP_MBPS
    rep.packet_payload_mbit = 0.0
    if bw_pkt > HARNESS_CAP_MBPS:
        rep.warnings.append(
            f"包级链路带宽 {bw_pkt:.0f} Mbps 高于用户态测试床可靠容量 "
            f"~{HARNESS_CAP_MBPS:.0f} Mbps，实测吞吐反映测试床能力而非链路行为")
    if packet_payload_mbit is not None:
        pkt_mbit = float(packet_payload_mbit)
    else:
        pkt_mbit = float(min(max(8.0, 12.0 * pkt_profile.bdp_bytes * 8 / 1e6), 32.0))
    rep.packet_payload_mbit = pkt_mbit

    say(f"实测链路 RTT（{profile.owd_ms:.1f} ms 单向）…")
    try:
        rep.measured_rtt_ms = measure_rtt_ms(profile, trials=5)
        dev = abs(rep.measured_rtt_ms - profile.rtt_ms) / max(profile.rtt_ms, 1e-6)
        if dev > 0.35:
            rep.warnings.append(
                f"实测 RTT {rep.measured_rtt_ms:.1f} ms 与理论 {profile.rtt_ms:.1f} ms "
                f"偏差 {dev*100:.0f}%（链路抖动/排队导致）")
    except Exception as exc:  # noqa: BLE001
        rep.warnings.append(f"RTT 实测失败：{type(exc).__name__}: {exc}")

    say(f"真实 TCP 传输 {payload_mbit:.1f} Mbit ×{tcp_trials}…")
    try:
        rep.tcp = run_tcp_transfer(profile, payload_mbit, trials=tcp_trials, progress=say)
    except Exception as exc:  # noqa: BLE001
        rep.warnings.append(f"真实 TCP 评测失败：{type(exc).__name__}: {exc}")

    for algo in packet_algos:
        say(f"真实包级传输 {algo.upper()} ×{packet_trials}（{pkt_mbit:.0f} Mbit）…")
        try:
            rep.packet.extend(run_packet_transfer(
                pkt_profile, pkt_mbit, algo=algo, trials=packet_trials,
                mss=pkt_profile.mtu_bytes, rtt_ref_ms=rtt_ref_ms, progress=say))
        except Exception as exc:  # noqa: BLE001
            rep.warnings.append(f"真实包级 {algo} 评测失败：{type(exc).__name__}: {exc}")

    if compare_analytic:
        try:
            from .config import TransportConfig
            from .transport import TransportSimulator

            tcfg = TransportConfig()
            tcfg.rtt_ref_ms = rtt_ref_ms
            tcfg.loss_prob = p_rtt               # 同口径：每 RTT 有效丢包率
            # 解析模型按**包级测试**的链路与载荷跑，保证与真实包级结果一一对照
            sim = TransportSimulator(tcfg)
            for algo in ("reno", "cubic", "hybla"):
                flows = sim.compare(
                    path_owd_ms=pkt_profile.owd_ms,
                    bottleneck_mbps=pkt_profile.bandwidth_mbps,
                    size_mbit=pkt_mbit,
                    protocols=[algo],
                )
                if flows:
                    rep.analytic[algo] = float(flows[0].completion_s)
                    rep.analytic_goodput[algo] = float(flows[0].goodput_mbps)
        except Exception as exc:  # noqa: BLE001
            rep.warnings.append(f"解析模型对照失败：{type(exc).__name__}: {exc}")

    rep.wall_s = time.perf_counter() - t_start
    say(f"真实传输评测完成，用时 {rep.wall_s:.1f} s")
    return rep

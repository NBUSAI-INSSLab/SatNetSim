"""真实评测能力探测：回环套接字、可用负载依赖、CPU/磁盘基准。"""
from __future__ import annotations

import importlib
import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Python:", sys.version.split()[0], "| pid", os.getpid())
print("CPU 逻辑核:", os.cpu_count())
print("-" * 60)

# ---- 1) 回环 TCP 能力 ----
try:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    got: list[bytes] = []

    def serve() -> None:
        c, _ = srv.accept()
        while True:
            b = c.recv(65536)
            if not b:
                break
            got.append(b)
        c.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    cli = socket.create_connection(("127.0.0.1", port), timeout=3)
    payload = b"x" * (256 * 1024)
    t0 = time.perf_counter()
    cli.sendall(payload)
    cli.shutdown(socket.SHUT_WR)
    cli.close()
    th.join(timeout=5)
    srv.close()
    dt = time.perf_counter() - t0
    total = sum(len(b) for b in got)
    print("[OK]   回环 TCP: 发送 %d B, 收到 %d B, 用时 %.4f s, 速率 %.0f Mbps"
          % (len(payload), total, dt, total * 8 / max(dt, 1e-9) / 1e6))
except Exception as exc:  # noqa: BLE001
    print("[FAIL] 回环 TCP: %s: %s" % (type(exc).__name__, exc))

# ---- 2) 回环 UDP 能力 ----
try:
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.bind(("127.0.0.1", 0))
    up = u.getsockname()[1]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(2.0)
    n = 0
    for i in range(50):
        s.sendto(b"y" * 1000, ("127.0.0.1", up))
    while True:
        try:
            d, _ = u.recvfrom(4096)
            n += 1
        except socket.timeout:
            break
    s.close()
    u.close()
    print("[OK]   回环 UDP: 发送 50 包, 收到 %d 包" % n)
except Exception as exc:  # noqa: BLE001
    print("[FAIL] 回环 UDP: %s: %s" % (type(exc).__name__, exc))

# ---- 3) TCP_CONGESTION 可设置性（Linux 专有）----
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cur = s.getsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, 64)
    print("[OK]   TCP_CONGESTION 可读: %r (Linux)" % cur)
    s.close()
except Exception as exc:  # noqa: BLE001
    print("[--]   TCP_CONGESTION 不可用 (%s) → 用用户态可靠传输做包级评测" % type(exc).__name__)

# ---- 4) 真实负载依赖 ----
print("-" * 60)
for name in ["numpy", "scipy", "PIL", "torch", "torchvision", "psutil",
             "aioquic", "cv2", "sklearn", "zstandard", "lz4"]:
    try:
        m = importlib.import_module(name)
        print("  [可] %-14s %s" % (name, getattr(m, "__version__", "?")))
    except Exception:  # noqa: BLE001
        print("  [--] %-14s 未安装" % name)

# ---- 5) 真实算力基准（GEMM GFLOPS）----
print("-" * 60)
try:
    import numpy as np

    rng = np.random.default_rng(7)
    n = 1200
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    for _ in range(2):  # 预热
        a @ b
    reps = 8
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    dt = time.perf_counter() - t0
    flops = 2 * n ** 3 * reps
    print("[OK]   真实 GEMM %dx%d x%d: %.4f s → %.2f GFLOPS (单核/多核视 BLAS 而定)"
          % (n, n, reps, dt, flops / dt / 1e9))
except Exception as exc:  # noqa: BLE001
    print("[FAIL] GEMM 基准: %s: %s" % (type(exc).__name__, exc))

# ---- 6) 真实磁盘带宽 ----
try:
    import numpy as np

    blob = np.random.default_rng(1).integers(0, 255, 32 * 1024 * 1024, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bench.bin")
        t0 = time.perf_counter()
        with open(p, "wb") as fh:
            fh.write(blob.tobytes())
        fh_ok = True
        tw = time.perf_counter() - t0
        t0 = time.perf_counter()
        with open(p, "rb") as fh:
            data = fh.read()
        tr = time.perf_counter() - t0
        ok = len(data) == blob.nbytes
        print("[OK]   磁盘写入 %.0f MB/s, 读取 %.0f MB/s, 校验=%s"
              % (blob.nbytes / tw / 1e6, blob.nbytes / tr / 1e6, ok))
except Exception as exc:  # noqa: BLE001
    print("[FAIL] 磁盘基准: %s: %s" % (type(exc).__name__, exc))

print("-" * 60)
print("探测完成")

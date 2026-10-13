import socket
import threading
import time

print("create server", flush=True)
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", 0))
srv.listen(1)
port = srv.getsockname()[1]
print("listening on", port, flush=True)

got = []


def serve():
    c, _ = srv.accept()
    print("accepted", flush=True)
    while True:
        b = c.recv(65536)
        if not b:
            break
        got.append(b)
    c.close()
    print("server done", flush=True)


th = threading.Thread(target=serve, daemon=True)
th.start()
print("connecting", flush=True)
cli = socket.create_connection(("127.0.0.1", port), timeout=3)
print("connected", flush=True)
payload = b"x" * (256 * 1024)
t0 = time.perf_counter()
cli.sendall(payload)
cli.shutdown(socket.SHUT_WR)
cli.close()
th.join(timeout=5)
srv.close()
dt = time.perf_counter() - t0
total = sum(len(b) for b in got)
print("sent %d recv %d in %.4f s -> %.0f Mbps" % (len(payload), total, dt, total * 8 / max(dt, 1e-9) / 1e6), flush=True)
print("OK", flush=True)

"""路由协议插件层。

所有路由算法实现统一的 `Router` 接口，作用于**单个快照**（Snapshot）之上，
返回节点序列。这样做的意义在于：路由与拓扑解耦后，同一套协议可以在任意
星座、任意 ISL 构型上复用，也方便做"协议 × 星座"的矩阵式对比实验。

内置协议
--------
shortest_delay    时延最短路（Dijkstra，权重 = 传播+处理时延）
shortest_hop      最小跳数（权重 = 1）
virtual_node      虚拟节点路由：在静态虚拟拓扑上算路，再贪心映射回物理下一跳
geographic        地理贪婪转发（每跳选"离目的星最近"的邻居），环检测后回退最短路
load_balanced     时延 + 拥塞惩罚（w = delay · (1 + α·utilization)）

对照文献
--------
* 时延最短路 / 最小跳数：Hypatia(IMC'20)、StarPerf(ICNP'20) 的默认策略
* 虚拟节点（VP）路由：经典 LEO 路由思路，用逻辑坐标替代物理坐标，降低路由更新频率
* 分布式/去中心化路由：DisCoRoute / PolarDisCo（ASMS-SPSC'25）针对极区覆盖的改进
* 学习型路由：Zhu et al., Ad Hoc Networks 178 (2025) 103995 综述了 ML/RL 路线的设计空间
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .topology import Snapshot

INF = float("inf")


# --------------------------------------------------------------------------- #
# 路由此结果
# --------------------------------------------------------------------------- #
@dataclass
class RouteResult:
    path: List[int]
    delay_ms: float
    hops: int
    bottleneck_mbps: float
    stretch: float = 1.0  # 相对几何最短距离的拉伸比（仅卫星段有意义）

    @property
    def ok(self) -> bool:
        return len(self.path) >= 2


# --------------------------------------------------------------------------- #
# Dijkstra
# --------------------------------------------------------------------------- #
try:  # 优先使用 scipy 的 C 实现（比纯 Python 快 1–2 个数量级）
    from scipy.sparse import csr_matrix as _csr
    from scipy.sparse.csgraph import dijkstra as _sp_dijkstra

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _dijkstra(
    snap: Snapshot,
    src: int,
    weight: Optional[np.ndarray] = None,
    max_hops: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """单源最短路，返回 (dist, prev)。

    weight 缺省用边时延（ms）。实现优先走 scipy.sparse.csgraph，退化时用
    纯 numpy 的 O(n^2) 版本（无需任何可选依赖）。
    """
    n = snap.n_nodes
    if snap.n_edges == 0:
        return np.full(n, INF), np.full(n, -1, dtype=np.int64)

    if weight is None:
        w = np.concatenate([snap.delay_ms, snap.delay_ms]).astype(float)
    else:
        w = np.concatenate([weight, weight]).astype(float)

    if _HAVE_SCIPY:
        rows = np.concatenate([snap.src, snap.dst])
        cols = np.concatenate([snap.dst, snap.src])
        mat = _csr((w, (rows, cols)), shape=(n, n))
        dist, prev = _sp_dijkstra(
            mat, directed=False, indices=[src], return_predecessors=True
        )
        d = np.asarray(dist[0], dtype=float)
        p = np.asarray(prev[0], dtype=np.int64)
        # scipy 用 -9999 表示不可达
        p = np.where(p < 0, -1, p)
        p[src] = -1
        return d, p

    return _dijkstra_py(snap, src, w, max_hops)


def _dijkstra_py(
    snap: Snapshot, src: int, w_all: np.ndarray, max_hops: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """纯 numpy 的稠密 Dijkstra（O(n²)），作为无 scipy 环境的兜底。"""
    n = snap.n_nodes
    indptr, indices, _ = snap.adjacency()
    order = np.argsort(np.concatenate([snap.src, snap.dst]), kind="stable")
    w_sorted = w_all[order]

    dist = np.full(n, INF)
    prev = np.full(n, -1, dtype=np.int64)
    hops = np.zeros(n, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    dist[src] = 0.0

    for _ in range(n):
        cand = np.where(visited, INF, dist)
        u = int(np.argmin(cand))
        if not np.isfinite(cand[u]):
            break
        visited[u] = True
        if hops[u] >= max_hops:
            continue
        lo, hi = indptr[u], indptr[u + 1]
        if hi <= lo:
            continue
        nb = indices[lo:hi]
        nd = dist[u] + w_sorted[lo:hi]
        better = (~visited[nb]) & (nd < dist[nb])
        if np.any(better):
            dist[nb[better]] = nd[better]
            prev[nb[better]] = u
            hops[nb[better]] = hops[u] + 1
    return dist, prev


def _reconstruct(prev: np.ndarray, src: int, dst: int,
                 max_hops: Optional[int] = None) -> List[int]:
    if src == dst:
        return [src]
    if prev[dst] < 0:
        return []
    path = [int(dst)]
    cur = int(dst)
    guard = 0
    while cur != src and guard < 4096:
        cur = int(prev[cur])
        if cur < 0:
            return []
        path.append(cur)
        guard += 1
    path.reverse()
    if max_hops is not None and len(path) - 1 > max_hops:
        return []
    return path


def _path_metrics(snap: Snapshot, path: Sequence[int]) -> Tuple[float, int, float]:
    """沿节点序列累加时延，并取瓶颈带宽。"""
    if len(path) < 2:
        return 0.0, 0, INF
    # 建立 (u,v) -> 边索引 的查询
    edge_of: Dict[Tuple[int, int], int] = {}
    for i in range(snap.n_edges):
        a, b = int(snap.src[i]), int(snap.dst[i])
        edge_of[(a, b)] = i
        edge_of[(b, a)] = i
    total, bn = 0.0, INF
    for u, v in zip(path[:-1], path[1:]):
        i = edge_of.get((int(u), int(v)))
        if i is None:
            return INF, len(path) - 1, 0.0
        total += float(snap.delay_ms[i])
        bn = min(bn, float(snap.capacity_mbps[i]))
    return total, len(path) - 1, bn


# --------------------------------------------------------------------------- #
# 路由器基类与实现
# --------------------------------------------------------------------------- #
class Router:
    name = "base"
    label = "基类"

    def __init__(self, cfg):
        self.cfg = cfg  # RoutingConfig

    # 子类实现：返回节点序列
    def _find_path(self, snap: Snapshot, src: int, dst: int,
                   ctx: Optional[dict] = None) -> List[int]:
        raise NotImplementedError

    def route(self, snap: Snapshot, src: int, dst: int,
              ctx: Optional[dict] = None) -> RouteResult:
        path = self._find_path(snap, src, dst, ctx)
        if not path:
            return RouteResult([], INF, 0, 0.0)
        delay, hops, bn = _path_metrics(snap, path)
        return RouteResult(path, delay, hops, bn)


class ShortestDelayRouter(Router):
    name = "shortest_delay"
    label = "时延最短路"

    def _find_path(self, snap, src, dst, ctx=None):
        _, prev = _dijkstra(snap, src, max_hops=self.cfg.max_hops)
        return _reconstruct(prev, src, dst, self.cfg.max_hops)


class ShortestHopRouter(Router):
    name = "shortest_hop"
    label = "最小跳数"

    def _find_path(self, snap, src, dst, ctx=None):
        w = np.ones(snap.n_edges)
        _, prev = _dijkstra(snap, src, weight=w, max_hops=self.cfg.max_hops)
        return _reconstruct(prev, src, dst, self.cfg.max_hops)


class LoadBalancedRouter(Router):
    """时延 + 拥塞惩罚：w = delay · (1 + α·utilization)。"""

    name = "load_balanced"
    label = "负载均衡"

    def _find_path(self, snap, src, dst, ctx=None):
        util = (ctx or {}).get("edge_util")
        if util is None:
            util = np.zeros(snap.n_edges)
        util = np.asarray(util, dtype=float)
        w = snap.delay_ms * (1.0 + self.cfg.load_balance_alpha * np.clip(util, 0, 1))
        _, prev = _dijkstra(snap, src, weight=w, max_hops=self.cfg.max_hops)
        return _reconstruct(prev, src, dst, self.cfg.max_hops)


class VirtualNodeRouter(Router):
    """虚拟节点路由：静态虚拟拓扑算路 + 物理层贪心映射。

    `ctx["virtual_adj"]` 提供 (n_sat, n_sat) 逻辑邻接的边对数组；
    `ctx["positions"]` 提供当前快照的 ECEF 位置用于贪心映射。
    """

    name = "virtual_node"
    label = "虚拟节点路由"

    def _find_path(self, snap, src, dst, ctx=None):
        ctx = ctx or {}
        vadj = ctx.get("virtual_adj")
        if vadj is None or vadj.size == 0 or src >= snap.n_sat or dst >= snap.n_sat:
            # 退化到物理时延最短路
            _, prev = _dijkstra(snap, src, max_hops=self.cfg.max_hops)
            return _reconstruct(prev, src, dst, self.cfg.max_hops)

        v_path = _virtual_shortest(vadj, snap.n_sat, src, dst)
        if not v_path:
            _, prev = _dijkstra(snap, src, max_hops=self.cfg.max_hops)
            return _reconstruct(prev, src, dst, self.cfg.max_hops)

        positions = ctx.get("positions")
        phys = self._map_to_physical(snap, v_path, positions)
        if not phys:
            _, prev = _dijkstra(snap, src, max_hops=self.cfg.max_hops)
            return _reconstruct(prev, src, dst, self.cfg.max_hops)
        return phys

    # 把逻辑路径逐跳映射为物理可达的下一跳
    def _map_to_physical(self, snap: Snapshot, vpath: Sequence[int],
                         positions: Optional[np.ndarray]) -> List[int]:
        nbr = _neighbor_sets(snap)
        phys = [int(vpath[0])]
        cur = int(vpath[0])
        for target in vpath[1:]:
            if target in nbr.get(cur, set()):
                phys.append(int(target))
                cur = int(target)
                continue
            # 当前物理邻居中，选离逻辑目标最近的
            best, best_d = None, INF
            if positions is not None and target < positions.shape[0]:
                tp = positions[target]
                for nb in nbr.get(cur, ()):  # type: ignore[union-attr]
                    d = float(np.linalg.norm(positions[nb] - tp))
                    if d < best_d:
                        best, best_d = nb, d
            if best is None:
                # 无处可去：以物理最短路补齐剩余段
                _, prev = _dijkstra(snap, cur, max_hops=self.cfg.max_hops)
                rest = _reconstruct(prev, cur, target, self.cfg.max_hops)
                if len(rest) <= 1:
                    return phys
                phys.extend(rest[1:])
                cur = int(target)
            else:
                phys.append(int(best))
                cur = int(best)
        return phys


class GeographicRouter(Router):
    """地理贪婪转发 + 环检测回退（类 GPSR 的两级策略）。"""

    name = "geographic"
    label = "地理贪婪路由"

    def _find_path(self, snap, src, dst, ctx=None):
        ctx = ctx or {}
        positions = ctx.get("positions")
        if positions is None or dst >= snap.n_sat or src >= snap.n_sat:
            _, prev = _dijkstra(snap, src, max_hops=self.cfg.max_hops)
            return _reconstruct(prev, src, dst, self.cfg.max_hops)

        nbr = _neighbor_sets(snap)
        target = positions[dst]
        path = [int(src)]
        cur = int(src)
        visited = {cur}
        for _ in range(self.cfg.max_hops):
            if cur == dst:
                return path
            best, best_d = None, float(np.linalg.norm(positions[cur] - target))
            for nb in nbr.get(cur, ()):
                if nb in visited or nb >= positions.shape[0]:
                    continue
                d = float(np.linalg.norm(positions[nb] - target))
                if d < best_d:
                    best, best_d = nb, d
            if best is None:
                # 贪心陷入局部极小：用最短路修补剩余路径
                _, prev = _dijkstra(snap, cur, max_hops=self.cfg.max_hops)
                rest = _reconstruct(prev, cur, dst, self.cfg.max_hops)
                if len(rest) <= 1:
                    return path
                return path + rest[1:]
            path.append(int(best))
            visited.add(int(best))
            cur = int(best)
        return path


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _virtual_shortest(vadj: np.ndarray, n_sat: int, src: int, dst: int) -> List[int]:
    """在虚拟拓扑上跑 BFS（虚拟图通常规则且同构，BFS 足够）。"""
    adj: List[List[int]] = [[] for _ in range(n_sat)]
    for u, v in vadj:
        u, v = int(u), int(v)
        if 0 <= u < n_sat and 0 <= v < n_sat:
            adj[u].append(v)
            adj[v].append(u)
    prev = [-1] * n_sat
    seen = [False] * n_sat
    from collections import deque

    q = deque([src])
    seen[src] = True
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                prev[v] = u
                q.append(v)
    if not seen[dst]:
        return []
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
        if path[-1] < 0:
            return []
    path.reverse()
    return path


def _neighbor_sets(snap: Snapshot) -> Dict[int, set]:
    nbr: Dict[int, set] = {}
    for i in range(snap.n_edges):
        a, b = int(snap.src[i]), int(snap.dst[i])
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    return nbr


ROUTERS: Dict[str, type] = {
    ShortestDelayRouter.name: ShortestDelayRouter,
    ShortestHopRouter.name: ShortestHopRouter,
    LoadBalancedRouter.name: LoadBalancedRouter,
    VirtualNodeRouter.name: VirtualNodeRouter,
    GeographicRouter.name: GeographicRouter,
}

ROUTER_LABELS: Dict[str, str] = {k: v.label for k, v in ROUTERS.items()}


def make_router(name: str, cfg) -> Router:
    cls = ROUTERS.get(name, ShortestDelayRouter)
    return cls(cfg)


# --------------------------------------------------------------------------- #
# 路径稳定性分析
# --------------------------------------------------------------------------- #
def path_switch_count(paths: Sequence[Sequence[int]]) -> int:
    """统计相邻快照间路径发生改变的次数（LEO 网络的关键代价指标）。"""
    changes = 0
    for a, b in zip(paths[:-1], paths[1:]):
        if list(a) != list(b):
            changes += 1
    return changes


def route_stretch(path_geo_km: float, direct_km: float) -> float:
    if direct_km <= 1e-9:
        return 1.0
    return path_geo_km / direct_km

# SatNetSim —— 卫星互联网一体化仿真器

> 面向 **路由协议 + 传输协议 + 星载计算** 三类研究的一体化仿真工具。
> 纯 Python 实现，PySide6 图形界面，离线可用，单文件即可跑通 1584 星规模。
> 内置**真实评测**通道：解析模型之外，还能把字节真的推过内核协议栈、把算子真的跑一遍。

---

## 1. 快速开始

```bash
# 安装依赖（torch / opencv / scikit-learn 为真实计算评测的可选依赖）
pip install -r requirements.txt

# 启动图形界面
python app.py
# 或
python -m satnetsim

# 命令行跑一次完整仿真（输出文本摘要）
python app.py run starlink-lite --duration 600 --tasks 300
python app.py list          # 列出内置场景预设

# 打开真实评测（真实数据传输 + 真实计算任务）
python app.py run starlink-lite --real-all
python app.py run starlink-lite --real-transfer --real-payload-mbit 32
python app.py run starlink-lite --real-compute --real-tasks 200

# 真实数据流仿真 + 实时回放（真字节逐包多跳转发，导出可分享的 HTML 动画）
python app.py run starlink-lite --real-data --rdata-flows 3 --rdata-mbit 64
python app.py run starlink-lite --real-data --live-html docs/verify/live.html
python app.py run starlink-lite --real-data --rdata-kind file --rdata-file 报告.docx

# 数据窗口跟随仿真时长（不再截断到 12 s）+ 控制回放规模
python app.py run starlink-lite --duration 600 --real-data --rdata-window 0 --rdata-budget 30000

# 传输分析引擎选择：真实包级 TCP 状态机 / 解析轮次模型 / 两者对照
python app.py run starlink-lite --trans-engine both --tcp-mbit 256 --tcp-algos reno,cubic,bbr,hybla
python app.py run starlink-lite --no-route-sweep      # 跳过「路由 × 真实传输」联合评测

# 全局离散事件内核（NS-3 式统一事件时间轴）：拓扑刷新/路由重算/路径切换/
# TCP 流到达完成/计算任务到达服务/指标采样 全部放进同一条时间轴
python app.py run starlink-lite --des --des-flows 3 --des-tasks 200 --des-algo bbr
```

打开 GUI 后默认载入 `starlink-lite`（16 面 × 18 星 = 288 颗，550 km / 53°），点击
**▶ 开始仿真**，得到 11 个页面的结果：结果概览、三维星座、卫星轨迹、路由分析、
传输分析、星载计算、**仿真展示**、**传输任务**、**计算任务**、运行日志、
**事件内核**。
2026 版界面采用**浅色侧边导航**（logo 品牌区 + 分组小标题「仿真分析 / 真实任务 / 实时动态」+ emoji 条目 +
薄荷绿选中胶囊）替代传统顶部页签，右侧结果区随导航切换，参数面板卡片化，
窗口与任务栏图标为内置 SVG logo（`satnetsim/resources/logo.svg`：轨道弧 +
星座节点 + 星间链路）。仿真展示页是实时回放——把真实载荷的每一跳画成"光点沿星座飞行"
的动画，可播放/暂停/倍速/拖动时间轴；事件内核页是离散事件时间线（甘特图）、
事件构成、任务流与内核摘要。

**操作效率**：`Ctrl+R` 运行 / `Ctrl+S` 导出场景 / `Ctrl+O` 载入场景 /
`Ctrl+E` 导出报告 / `Ctrl+P` 导出当前页签图像 / `Ctrl+L` 折叠参数面板；
工具栏进度条带 ETA，运行完成后顶部有一排结果摘要卡片。

真实评测（⑧⑨，会真的做网络传输与算子计算）在 GUI 里**默认开启**，因此一次点击
约 10–40 s（取决于本机算力与载荷）。若只想快速看解析模型结论，在左侧
「真实评测」分组里取消勾选即可——**取消总开关会同时收起两个子项**，
反之勾选任一子项会自动打开总开关。

## 2. 内置场景预设

| 预设名 | 星座 | 规模 | 轨道 | 说明 |
|---|---|---|---|---|
| `iridium-66` | Walker | 6×11 = 66 | 780 km / 86.4° | 极轨铱星规模 |
| `starlink-lite` | Walker | 16×18 = 288 | 550 km / 53° | 交互式运行推荐 |
| `starlink-shell1` | Walker | 72×22 = 1584 | 550 km / 53° | Starlink Shell-1 量级 |
| `walker-polar` | Walker | 8×16 = 128 | 1000 km / 87.9° | 含反向缝场景 |
| `compute-constellation` | Walker | 6×12 = 72 | 500 km / 97.4° | 计算星座（对标三体计算星座） |

## 3. 架构总览

```
┌──────────────────────────── GUI (PySide6) ────────────────────────────┐
│ 场景配置 │ 侧边分组导航：结果概览/三维星座/卫星轨迹/路由/传输/计算/展示    │
│   面板   │ 传输任务/计算任务（真实任务）+ 运行日志/事件内核（实时动态）     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ SimulationResult
┌───────────────────────────────▼───────────────────────────────────────┐
│  engine  仿真引擎（快照驱动 + 事件推进 + 指标采集）                    │
├──────────────────────────────────────────────────────────────────────┤
│  des        离散事件内核（Simulator/Scheduler/Process/Trace，ns-3 式）│
│  dessim     全局事件编排（⑫：拓扑/路由/切换/流/任务/采样一条时间轴）  │
│  orbit       轨道动力学  →  topology  时变拓扑  →  routing  路由插件  │
│                                              →  transport 传输插件   │
│                                              →  tcpstack 真实包级TCP │
│                                              →  computing 星载计算   │
│                         ⑦ real eval  real_link 真实传输 / real_workload│
└──────────────────────────────────────────────────────────────────────┘
```

**分层可插拔**：路由、传输、调度、ISL 构型都是注册表驱动（`ROUTERS` /
`PROTOCOL_META` / `ISL_PATTERNS`），新增一个协议只需实现一个类。

**解析模型 vs 真实评测**：第 ①–⑥ 步全是解析/合成模型（窗口级 RTT 积分、解析
FLOP 统计、合成任务区间），优点是秒级跑完、可扫参；第 ⑦ 步是**真实执行**层，
用来给解析模型做地面真值校验。两者不是替代关系，而是在同一口径下互相对照。

## 4. 各层建模要点

| 层 | 模型 | 关键公式 / 机制 |
|---|---|---|
| 轨道 | 两体 + J2 长期项（可选 SGP4） | RAAN′ = −1.5 n₀ J₂ (Re/p)² cos i |
| 拓扑 | 时变图快照序列 | Δt ≤ ε·c/v_rel，默认 10 s |
| ISL | 连接向量 φ(ṅ,ṁ) | `+Grid` / `*Grid` / `Grid-V`，反向缝默认断开 |
| 链路 | 距离 + 地球遮挡 + 仰角 | 线段到地心最小距离 ≥ Re + h_atm |
| 路由 | 单快照 Dijkstra（scipy） | 时延最短路 / 最小跳 / VP / 地理贪婪 / 负载均衡 |
| 传输 | 解析：窗口级 RTT 轮次模型；**真实：包级 TCP 状态机** | 解析 Reno / CUBIC / BBR / Hybla / QUIC / PEP / DTN；真实 RFC 5681 慢启动·拥塞避免·快速重传·快速恢复 + RFC 8312 CUBIC + RFC 6298 SRTT/RTTVAR/RTO(Karn) + RFC 6675 SACK + RFC 1122 延迟 ACK |
| **路由效果评测**（④） | 逐快照路径 → 时延/抖动/切换统计 × 真实 TCP 联合评测 | 中位时延、抖动 σ、P95−P5、跳数 σ、切换/秒、可达快照数；同一时变路径下各路由的真实 TCP 吞吐对照 |
| 星载计算 | 三阶段：放置 → 节点内优先级队列 → 地面站共享下行队列 | 价值密度 + 优先级老化；能量-辐射-日照约束 |
| **真实传输**（⑦） | 用户态整形中继（真实内核 TCP）+ 逐包链路 + 可靠传输状态机 | 令牌桶限速；单调释放保序；有限缓冲尾丢弃；慢启动/拥塞避免/快速重传/快速恢复/RTO 退避 |
| **真实包级 TCP**（⑤） | `tcpstack.py`：逐段发送/ACK/RTO 事件驱动的完整 TCP 连接模型，跑在由逐快照路由结果推导的**时变路径剖面**上 | 窗口受限吞吐上限 `rcvbuf×8/RTT`；高 BDP（LEO 场景 ~10 MB）下常先于带宽饱和；BBR 不把丢包当拥塞信号（`_enter_recovery` 仅 0.98 折减），在丢包型 vs 模型型算法对比中体现机理差异 |
| **真实计算**（⑦） | 本机真实跑算子，实测墙钟与真实字节产出 | 有效算力 = FLOP ÷ 实测时间；无解析 FLOP 的负载按实测时间折算等效计算量 |
| **真实数据流**（⑧） | 真载荷（MJPEG 影像序列 / UTF-8 文档 / struct 遥测 / 本地文件）分片后，沿时变拓扑**逐包多跳**转发 | 每跳按当时快照查下一跳（单源最短路反向建树，O(1) 转发表）；每条有向链路 FIFO + 有限缓冲尾丢弃（2×BDP）；到端按序重组 + 逐字节 SHA-256 |
| **实时回放**（⑨） | QPainter 手绘深空指挥舱视图（非 matplotlib） | 正交投影 + 经纬网地球 + 卫星/ISL 网格 + 分组彗星多跳拖尾 + 事件脉冲 + HUD；可导出**自包含单文件 HTML** 动画 |

### 4.1 真实评测的两个工程口径

开真实评测时最容易踩的是"测到的是测试床而不是链路"。本项目把两个边界显式化：

- **每包丢包率 ↔ 每 RTT 丢包率**：配置文件里的 `transport.loss_prob` 是每 RTT 口径，
  而包级链路按每包丢包处理。两侧必须换算到同一口径才可比
  （`reallink.rtt_loss_from_packet_loss` / `packet_loss_from_rtt_loss`）。
- **用户态测试床容量**
  - 包级逐包路径（Python 用户态收发）：可靠容量约 **20 Mbps**（`HARNESS_CAP_MBPS`），
    因此包级评测的定位是**协议间相对比较与丢包响应验证**；
  - 真实 TCP 中继路径：转发放率上限约 **200 Mbps**（`RELAY_MAX_MBPS`），
    对**绝对吞吐**给出可信结果。
  超出容量上界的配置会在结果里打告警，不会假装测到了链路极限。

### 4.2 真实数据流与实时回放

第 ⑨ 层回答的问题与第 ⑧ 层不同：`reallink` 在**回环**上标定"这段链路 TCP 能跑多快"，
`realdatastream` 在**星座图**上回答"这份真实文件能不能、多久、多完整地穿过网络"。

- **真字节，不是随机数**：默认载荷是一段**真 JPEG 帧序列**（MJPEG，帧间地物平移），
  由传感器噪声 + 多尺度地形真实编码而成；也支持 UTF-8 中文文档、struct 定长遥测、
  或用户指定的本地文件。每个分片带序号与 SHA-256，到端重组后逐字节比对。
- **数据报式逐跳转发**：分组每到一个节点，都按**当时的拓扑快照**查下一跳
  （以目的站为根的单源最短路反向建树），天然复现"拓扑时变 → 路径切换"。
- **覆盖缺口是真实行为**：若某端站在窗口内没有可见卫星（例如 `iridium-66` 预设下
  悉尼在某些时刻可见卫星数为 0），对应流会整流 `no_route` 丢弃并在报告里告警——
  自动选对时会优先避开覆盖缺口，需要研究中断场景时可用
  `--rdata-*`/GUI 显式指定端站。
- **窗口跟随仿真时长，不截断**：数据仿真窗口默认等于 `SimConfig.duration_s`
  （`--rdata-window 0`，即"跟随仿真时长"）。窗口越短在途分组越密、回放画面越热闹
  （12 s / 64 Mbit / 3 流 ≈ 30 个彗星同时在飞）；长窗口（如 600 s）则展示**长时段行为**——
  路径切换、覆盖缺口、吞吐随几何变化。同一份载荷按 `pid % n_segs` 循环复用覆盖
  全部分片位置，因此长窗口不会造成事件数膨胀；事件总量由 `--rdata-budget`
  （`RealDataConfig.max_packets`，默认 30000）封顶，超出则按比例下调业务速率。
- **可视化是手绘的，不是截图**：`liveviz.LiveView` 用 QPainter 画深空指挥舱——
  星空 + 径向渐变地球 + 经纬网 + ISL 网格（含按累计流量加亮的热力）+ 地面站脉冲
  + 分组彗星（多跳拖尾）+ 事件脉冲环 + HUD 大字指标 + 可拖动时间轴。
  另可 `--live-html out.html` 导出**零依赖单文件 HTML**（Canvas 2D，几何 base64 内嵌），
  直接在浏览器里给别人看。

### 4.3 传输分析：真实包级 TCP 与路由效果评测

第 ⑤ 步的传输分析有两种引擎，用 `transport.engine`（CLI `--trans-engine`）切换：

| 引擎 | 实现 | 适用 |
|---|---|---|
| `real`（默认） | `tcpstack.py` 离散事件状态机：逐段发送/ACK/RTO 事件驱动，跑在由逐快照路由结果推导的**时变路径剖面**（`PathProfile`）上 | 定量结论、丢包率敏感性、算法间机理对比 |
| `analytic` | `transport.py` 窗口级 RTT 轮次积分 | 秒级扫参、与旧结论对齐 |
| `both` | 两者都跑，同一口径并列展示 | 交叉验证（⑤ 页签的"传输引擎对照"子图） |

`PathProfile` 由路由结果推导，因此天然承接"拓扑时变"：RTT 阶跃、瓶颈带宽变化、
跳数变化都在剖面里，`_on_rto` 与 `path_switch_penalty` 会复现真实 TCP 在路径切换
时的超时退避。

**丢包口径**（最容易踩的坑）：配置里有两个丢包字段，不要混用——

- `transport.loss_prob`（默认 0.002）是**每 RTT 口径**，供解析模型使用；
- `transport.packet_loss_prob`（默认 0.0002）是**每包口径**，供真实包级栈使用。

**路由效果评测**（第 ④ 步）：对每种路由，逐快照取路径并统计
中位时延、抖动 σ、P95−P5、跳数 σ、切换/秒、可达快照数；随后把**该路由的时变路径**
喂给真实 TCP 状态机做「路由 × 传输」联合评测（`--no-route-sweep` 可跳过）。
这一步把"路由的时延指标"与"路由对传输协议的连带影响"接在同一口径上——
在 LEO 里，路由换了 1 跳就可能改掉整条路径的 RTT 与窗口上限。

### 4.4 离散事件内核与全局编排（对标 ns-3）

`des.py` 提供 ns-3 风格的事件内核：`Simulator`（Schedule/Cancel/Run/Stop/Now）、
`EventId`（惰性取消）、`HeapScheduler`/`ListScheduler`（语义等价可对照）、
`Process` 生成器协程、`Trace` 环形缓冲事件追踪、`RandStream` 命名确定性随机流。
定序规则与 ns-3 一致：`(时间, 优先级降序, 插入序)`，绝不用对象地址做 tie-break，
同种子跨进程可复现。

真实包级 TCP（`tcpstack.py`）**直接跑在这个内核上**：连接不再自带私有主循环，
段到达 / ACK 到达 / RTO 都是内核事件；`TcpConnection.start()` 支持把一条流
非阻塞地挂到**共享内核**上（完成时回调），因此多条流可以事件级并发。

`dessim.py` 在此之上做**全局编排**（`cfg.des` / CLI `--des`，GUI ⑫ 页签）：
拓扑刷新、路由重算、路径切换、TCP 流到达与完成、计算任务泊松到达 + JSQ
分发 + FIFO 服务、指标采样，全部放进同一条时间轴。它回答的是**事件级**问题
（"切换与任务到达同时发生时谁先谁后"、"流与拓扑刷新如何交错"），与步进式
流水线的**层内统计**互补。建模口径的诚实声明：各 TCP 流各自对剖面取样（流间
共享瓶颈的挤压由真实数据流层体现）；路由重算事件只做记账（引擎已预计算）；
任务排队是事件驱动的 FIFO+JSQ，与 `computing.py` 的批量放置是两种口径可对照。

**路径切换 × 在途流联动**（`TcpConnection.on_path_switch`）：切换事件不再只
记账——每个阶跃时刻会波及所有在途 TCP 流，按真实 TCP 的行为建模切换代价：
发射静默 `path_switch_cost_s` 秒（星上重寻址/新路径建立）+ 拥塞窗口收缩回
初始窗 + 退出快速恢复 + 旧 RTO 定时器作废（CUBIC 重置拐点、BBR 保守重估
瓶颈带宽）。受波及的流留下 `flow:switch` 轨迹事件（⑫ 页签时间线红点），
`DesReport.flow_switches` 汇总次数；`--no-switch-penalty` 可关闭做对照。
报告同时给出事件粒度的**节点利用率**（服务时间占比）与轻负载提示
（队列全程无排队时明说"算力富余"，并指出如何加大负载观察排队）。

## 5. 代码结构

```
SatNetSim/
├── app.py                  桌面入口（GUI / CLI）
├── requirements.txt
├── satnetsim/
│   ├── __init__.py
│   ├── config.py           场景配置 dataclass + 出厂预设
│   ├── orbit.py            Kepler / Walker-Delta / 坐标变换 / 日照
│   ├── topology.py         可见性、ISL 构型、快照序列
│   ├── routing.py          路由插件（5 种内置协议）
│   ├── transport.py        传输插件（7 种解析协议 + 丢包率扫描）
│   ├── des.py              **离散事件内核**（Simulator/EventId/Process/Trace/RandStream，ns-3 式）
│   ├── dessim.py           全局离散事件编排（⑫：一条时间轴编排六类事件）
│   ├── tcpstack.py         **真实包级 TCP 状态机**（RFC 5681/8312/6298/6675/1122，跑在 des 内核上）+ 时变路径剖面
│   ├── computing.py        星载任务、算力节点、三阶段调度器、容量扫描
│   ├── reallink.py         真实数据传输：整形中继、包级链路、可靠传输状态机
│   ├── realworkload.py     真实计算任务：真实算子负载 + 本机算力标定
│   ├── realdatastream.py   真实数据流：真载荷分片 + 逐包多跳 DES + 逐跳轨迹
│   ├── liveviz.py          实时回放：QPainter 深空指挥舱视图 + 单文件 HTML 导出
│   ├── engine.py           一体化引擎与结果容器（⑦ 真实评测 / ⑧ 真实数据流）
│   ├── viz.py              matplotlib 图表与文本报告（内置 CJK 字体配置）
│   ├── gui.py              PySide6 主窗口
│   ├── cli.py              命令行批量仿真
│   └── __main__.py
├── tests/                  实现验证脚本（环境 / 端到端 / GUI / 可视化 / 真实评测 / 实时回放）
└── docs/screenshots/       GUI 图表示例
```

### 验证脚本用法

```bash
python tests/check_env.py        # 依赖自检
python tests/verify_des.py       # 事件内核语义 + TCP-on-DES 等价性 + 编排确定性
python tests/verify_e2e.py starlink-lite   # 端到端仿真 + 关键指标核对
python tests/verify_gui.py       # GUI 离屏构建（11 页签 / 11 参数分组）
python tests/verify_viz.py       # 全套图表离线渲染到 docs/verify/
python tests/probe_real_eval.py  # 真实评测环境能力探测
python tests/verify_reallink.py  # 真实传输：TCP + 包级 + 解析对照
python tests/verify_realworkload.py  # 真实计算：9 个负载的实测与标定
python tests/verify_real_integration.py [--full]  # 真实评测走完整引擎流水线
python tests/verify_realdata.py [--full]  # 真实数据流 + 实时回放（含 GUI 离屏）
```

## 6. 二次开发指南

**新增一种路由协议**

```python
from satnetsim.routing import Router, ROUTERS

class MyRouter(Router):
    name, label = "my_router", "我的路由"
    def _find_path(self, snap, src, dst, ctx=None):
        # snap.src / snap.dst / snap.delay_ms / snap.capacity_mbps 是本快照的边集
        ...
        return [src, ..., dst]     # 返回节点序列

ROUTERS["my_router"] = MyRouter
```

**新增一种传输协议**：在 `transport.PROTOCOL_META` 注册元信息，并在
`TransportSimulator.simulate` 的拥塞控制分支中加入你的窗口更新规则。

**批量扫参**：

```python
from satnetsim.config import preset
from satnetsim.engine import Simulation

for alt in (500, 800, 1100):
    cfg = preset("starlink-lite")
    cfg.constellation.altitude_km = alt
    res = Simulation(cfg).run()
    print(alt, res.routes["shortest_delay"].mean_delay_ms)
```

**打开真实评测（用真实数据校验解析模型）**：

```python
from satnetsim.config import real_eval_preset
from satnetsim.engine import Simulation

cfg = real_eval_preset("starlink-lite", payload_mbit=16.0,
                       real_task_count=200, packet_algos=["reno", "cubic", "hybla"])
res = Simulation(cfg).run(progress=lambda m, f: print(f"{f*100:5.1f}% {m}"))

# 真实 vs 解析（同口径：同一链路剖面 / 载荷 / 每 RTT 丢包率）
print(res.real_transfer.summary_rows())
print(res.real_transfer.goodput_by_proto("packet"))   # {'reno': 7.3, 'cubic': 7.7, ...}
print(res.real_compute.headline())
```

也可以只在某一步打开：`Simulation(cfg).run(run_real=True)`；或部分关闭：
`cfg.real_eval.transfer = False`。

## 7. 验证情况（本仓库内置基准）

在 `starlink-lite`（288 星，600 s，Δt = 10 s）上：

- **轨道**：单星速度 ≈ 7.59 km/s，周期 ≈ 95.6 min，星下点轨迹连续闭合；
- **拓扑**：`+Grid` 构型平均 ISL 度 ≈ 3.93（理论 4），节点度分布集中在 3–4；
- **路由**：北京 ↔ 圣保罗（球面最远站对）平均单向时延 ≈ 79 ms，≈ 10.5 跳，
  600 s 内路径切换 5 次，可达率 100%；
- **路由算法效果评测**（④ 页签，61 个快照）：

| 算法 | 中位时延 | 抖动 σ | P95−P5 | 跳数 σ | 切换/秒 | 可达快照 | 真实 TCP |
|---|---|---|---|---|---|---|---|
| 时延最短路 | 81.3 ms | 2.50 ms | 5.94 ms | 0.50 | 0.008 | 61/61 | 56.5 Mbps |
| 最小跳数 | 83.1 ms | 2.09 ms | 6.81 ms | 0.50 | 0.008 | 61/61 | 54.7 Mbps |
| 虚拟节点路由 | 81.3 ms | 2.50 ms | 5.94 ms | 0.50 | 0.008 | 61/61 | 56.5 Mbps |
| 地理贪婪路由 | 81.3 ms | 2.50 ms | 5.94 ms | 0.50 | 0.008 | 61/61 | 56.5 Mbps |

  最后一列是**路由 × 真实传输联合评测**：把该路由逐快照选出的**时变路径**喂给真实
  包级 TCP 状态机，得到"这条路由下的端到端吞吐"。它把"路由的时延指标"与
  "路由对传输协议的影响"接在一个口径上——最小跳数路由时延并非最差但路径更长、
  RTT 摆动更大，真实 TCP 吞吐比时延最短路低约 3%。

- **传输（真实包级 TCP，⑤ 页签）**：路径剖面 RTT 152.5–165.1 ms（摆动 12.6 ms，
  阶跃 12 次）、瓶颈 500 Mbps、平均 BDP 9673 KB（6785 段）、**窗口受限吞吐上限 212 Mbps**。
  同一载荷（256 Mbit）下：

| 算法 | 完成 | 吞吐 | 效率 | 重传率 | 超时 | 快重传 |
|---|---|---|---|---|---|---|
| RENO | 4.52 s | 56.6 Mbps | 27% | 17.3% | 1 | 3 |
| CUBIC | 6.19 s | 41.4 Mbps | 20% | 20.5% | 1 | 7 |
| BBR | 4.36 s | **58.7 Mbps** | 28% | 21.8% | 2 | 6 |
| HYBLA | 7.80 s | 32.8 Mbps | 16% | 20.8% | 2 | 12 |

  **丢包率敏感性**（每包丢包率 → 有效吞吐 Mbps）：

| 算法 | 0.00% | 0.02% | 0.10% | 1.00% |
|---|---|---|---|---|
| BBR | 96.1 | 56.8 | 31.2 | 5.1 |
| CUBIC | 81.7 | 46.0 | 17.7 | 3.2 |
| HYBLA | 136.3 | 28.5 | 30.6 | 14.4 |

  丢包型算法（CUBIC/HYBLA）把误码误判为拥塞而大幅降窗，模型型（BBR）不受影响——
  这正是 BBR 类算法在卫星链路上的机理优势。**LEO 高 BDP 下瓶颈往往先出现在接收窗口
  （212 Mbps）而不是链路带宽（500 Mbps）**，窗口缩放是绕不开的工程前提。

- **传输（解析轮次模型，同口径对照）**：500 Mbit 在 500 Mbps 瓶颈、79 ms RTT 下，
  Reno/CUBIC/BBR ≈ 1.74 s（效率 57%），Hybla 1.35 s（74%），QUIC 因少一次握手再快
  40 ms，DTN 需 166 s（受接触机会主导）；
- **星载计算**：弯管架构下行 71 787 Mbit vs OEC 732 Mbit —— **约 98×** 的下行量
  差异，同时端到端时延从 0.96 s 降到 0.25 s。

### 7.1 真实评测基线（同一场景，真实执行）

**真实数据传输**（代表路径 OWD 39.6 ms / 500 Mbps 瓶颈 / 每 RTT 0.2% 丢包）：

| 方法 | 完成时间 | 有效吞吐 | 重传 | 完整性 |
|---|---|---|---|---|
| 真实 TCP（内核协议栈，16 Mbit 载荷） | 0.075–0.099 s | 161–212 Mbps | — | SHA-256 一致 |
| 真实包级 Reno（用户态状态机） | 2.6 s | 7.3 Mbps | 0 | SHA-256 一致 |
| 真实包级 CUBIC | 2.5 s | 7.7 Mbps | 0 | SHA-256 一致 |
| 真实包级 Hybla | 1.6 s | 12.2 Mbps | 0 | SHA-256 一致 |
| 解析模型 Reno / CUBIC | 1.233 s | 15 Mbps | — | （模型） |
| 解析模型 Hybla | 1.188 s | 16 Mbps | — | （模型） |

实测 RTT 86.4 ms（理论 79.2 ms，链路抖动导致）。**关键结论**：Hybla 优于
Reno/CUBIC 这一排序在真实侧与解析侧**一致**——解析模型的定性结论可信；
绝对吞吐差异来自用户态逐包测试床的容量上界（~20 Mbps），不构成链路本身的
吞吐能力结论。（第 ⑧ 层的包级测试床是"协议保真、容量受限"的对照台；
第 ⑤ 层的 `tcpstack` 包级状态机跑在真实路径剖面上，不受该容量上界约束，
两者互补。）

**真实数据流**（⑨ 层，`starlink-lite` 600 s 窗口、2 流）：

| 指标 | 结果 |
|---|---|
| 业务窗口 | 600 s（= 仿真时长，不再截断到 12 s） |
| 真实载荷 | 对地观测影像序列（MJPEG）7826 KiB / 6679 分片 |
| 送达 | 30000 / 30000 分组（100%） |
| 端到端时延 | 均值 74.5 ms / P95 82.0 ms |
| 平均跳数 | 9.9（最多 23） |
| 完整性 | SHA-256 一致，分片覆盖 100%，载荷循环 3 轮 |

长窗口下业务速率被 30000 分组预算自动压到 0.25 Mbps/流——这是**回放数据量的
工程取舍**，不是链路能力结论：把 `--rdata-budget` 调高即可得到更密集的彗星画面。

**真实计算任务**（9 个真实负载全部跑通）：

| 负载 | 实测耗时 | 真实吞吐 |
|---|---|---|
| ResNet-18 推理（224²×4，真实 JPEG 输入） | 45.20 ms | 40.3 GFLOPS |
| 小 CNN 推理（64²×8） | 3.26 ms | 109.8 GFLOPS |
| CNN 训练步（前向+反向+SGD） | 3.50 ms | 48.6 GFLOPS |
| FFT 频谱分析（2²⁰ 点） | 26.82 ms | 3.9 GFLOPS |
| 视觉流水线（1080p 模糊+Canny+缩放） | 17.35 ms | 57.3 次/s |
| SHA-256 完整性校验（8 MB） | 26.48 ms | 317 MB/s |
| zlib 压缩（4 MB） | 114.31 ms | 37 MB/s |
| 稠密 GEMM（1024³） | 27.07 ms | 79.4 GFLOPS |
| 随机森林训练+推理 | 67.25 ms | 14.9 次/s |

本机实测峰值算力 **161.2 GFLOPS**。用这批真实负载画像生成 101 个任务：
真实计算总量 586.5 GFLOP，地面本机串行 3.919 s（逐任务实测之和）vs 星载按
单星有效算力折算 0.086 s，得到 **45.4×** 的架构加速比。

## 8. 局限与路线图

当前版本刻意把保真度控制在"系统级设计空间探索"层次。尚未包含：

1. **真实评测的容量上界**：包级逐包路径受 Python 用户态收发限制（~20 Mbps），
   只适合协议间相对比较；需要绝对吞吐时以真实 TCP 中继路径为准（~200 Mbps）。
   真正的高速率包级内核需要 C++/DPDK 或复用 ns-3 后端；
   ——注意第 ⑤ 层的 `tcpstack` 是**离散事件状态机**（不是真收发包），
   不受该上界约束，它给出的是"该路径剖面上 TCP 的理论可达吞吐"；
2. **`tcpstack` 尚未覆盖的机制**：ECN、Pacing（BBR 的发送节拍）、
   多流公平性与 AQM（CoDel/FQ-CoDel）、显式拥塞反馈；
3. **物理层信道**：雨衰（ITU-R P.618）、多普勒、多波束调度；
4. **真实 TLE 接入**：代码已预留 `sgp4` / `skyfield` 后端接口（`propagator="sgp4"`）；
5. **分布式执行**：快照天然可并行，可按快照切分到进程池 / 多机；
6. **学习型路由 / 调度**：预留 `Router` 插件接口，可加载 PyTorch 策略。

## 9. 主要参考

- Kassing et al. *Exploring the "Internet from Space" with Hypatia.* IMC 2020.
- Lai et al. *StarPerf: Characterizing Network Performance for Emerging
  Mega-Constellations.* ICNP 2020.
- Lai et al. *StarryNet: Empowering Researchers to Evaluate Futuristic
  Integrated Space and Terrestrial Networks.* USENIX NSDI 2023.
- Lu et al. *OpenSN: An Open Source Library for Emulating LEO Satellite
  Networks.* IEEE TPDS 36(8), 2025.
- Pusztai et al. *Stardust: A Scalable and Extensible Simulator for the 3D
  Continuum.* IEEE EDGE 2025 (arXiv:2506.01513).
- Pfandzelter & Bermbach. *Celestial: Virtual Software System Testbeds for the
  LEO Edge.* ACM Middleware 2022.
- Wang et al. *Emulating Space Computing Networks with RHONE.* USENIX ATC 2025.
- Denby & Lucia. *Orbital Edge Computing.* ASPLOS 2020.
- Denby et al. *Kodan: Addressing the Computational Bottleneck in Space.*
  ASPLOS 2023.
- Won et al. *ASTRA-sim 2.0.* ISPASS 2023 (arXiv:2303.14006).

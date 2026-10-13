"""`python -m satnetsim` 入口：默认启动图形界面。"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # 无参数 → GUI；带 --headless → 命令行批处理
    if argv and argv[0] in ("-h", "--help"):
        print(
            "SatNetSim —— 卫星互联网一体化仿真器\n\n"
            "  python -m satnetsim                 启动 PySide6 图形界面\n"
            "  python -m satnetsim run <preset>    以预设场景跑一次命令行仿真\n"
            "  python -m satnetsim list            列出内置预设\n"
        )
        return 0
    if argv and argv[0] == "list":
        from .config import PRESETS

        for name, cfg in PRESETS.items():
            c = cfg.constellation
            print(f"  {name:24s} {c.planes}×{c.sats_per_plane}"
                  f"={c.n_satellites:5d} 星  {c.altitude_km:.0f} km  "
                  f"{c.inclination_deg:.1f}°")
        return 0
    if argv and argv[0] == "run":
        from .cli import main as cli_main

        return cli_main(argv[1:])

    from .gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())

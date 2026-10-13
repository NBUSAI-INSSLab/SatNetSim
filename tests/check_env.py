"""环境自检：确认 SatNetSim 依赖是否齐备。"""
import importlib
import sys

REQUIRED = ["numpy", "scipy", "matplotlib", "networkx", "PySide6", "pandas"]
OPTIONAL = ["pyqtgraph", "yaml", "sgp4", "skyfield"]

print("Python:", sys.version.split()[0])
print("exe   :", sys.executable)
print("-" * 52)

missing = []
for name in REQUIRED:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print("  [OK]   %-12s %s" % (name, ver))
    except Exception as exc:  # noqa: BLE001
        missing.append(name)
        print("  [MISS] %-12s %s" % (name, type(exc).__name__))

print("-" * 52)
for name in OPTIONAL:
    try:
        mod = importlib.import_module(name)
        print("  [opt]  %-12s %s" % (name, getattr(mod, "__version__", "?")))
    except Exception:  # noqa: BLE001
        print("  [opt]  %-12s (未安装)" % name)

print("-" * 52)
print("RESULT:", "ALL REQUIRED PRESENT" if not missing else "MISSING -> %s" % missing)

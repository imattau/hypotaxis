# PyInstaller configuration for the native desktop build.
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


root = Path(SPECPATH)
hiddenimports = (
    collect_submodules("studio")
    + collect_submodules("webview")
    + collect_submodules("qtpy")
)
datas = [
    (str(root / "studio" / "static"), "studio/static"),
    (str(root / "stories"), "stories"),
    (str(root / "registry"), "registry"),
    (str(root / "data"), "data"),
]

a = Analysis(
    [str(root / "run_desktop.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="hypotaxis",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

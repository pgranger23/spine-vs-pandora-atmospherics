"""
Centralised environment setup for the SPINE vs Pandora analysis scripts.

Every script used to repeat the same block: locate the locally-built
duneanaobj / SRProxy reader, preload its shared libraries, and hard-code the
input CAF filename. That block lived at the top of ~30 files with a hard-coded
platform path (`duneanaobj/build/Linux/...`), macOS-only library names
(`.dylib`), and a hard-coded data filename. This module replaces all of it, so
you configure the environment ONCE via two optional environment variables:

    SPINE_CAF         Path to the input CAF ROOT file.
                      Default: ./spine_merged_reprocess.root

    DUNEANAOBJ_BUILD  Path to the duneanaobj build directory containing
                      `lib/` and `python/<X.Y>/`.
                      Default: ./duneanaobj/build/<System>  (System = Linux or
                      Darwin, auto-detected; if that does not exist, falls back
                      to any ./duneanaobj/build/* that contains a lib/ dir).

The library extension (.so on Linux, .dylib on macOS) and the Python-bindings
version folder are detected automatically, so the same checkout runs on both
Linux (e.g. FNAL) and macOS without edits.

Usage
-----
Scripts that read via SRProxy:

    import caf_env
    pySRProxy = caf_env.setup_pysrproxy()
    prox = pySRProxy.StandardRecordFileReader("recTree", caf_env.CAF_FILE)
    # or simply: prox = caf_env.open_proxy()

Scripts that read via ROOT.TFile:

    import caf_env
    ROOT = caf_env.setup_root()
    f = ROOT.TFile.Open(caf_env.CAF_FILE)
"""

import os
import sys
import ctypes
import glob
import platform

# --- input CAF --------------------------------------------------------------
CAF_FILE = os.environ.get("SPINE_CAF", "spine_merged_reprocess.root")

# --- duneanaobj build directory ---------------------------------------------
_SYSTEM = platform.system()                       # 'Linux' or 'Darwin'
_LIB_EXT = "dylib" if _SYSTEM == "Darwin" else "so"

_STDREC_LIBS = [
    "libSRProxy_BasicTypes",
    "libduneanaobj_StandardRecord",
    "libduneanaobj_StandardRecordProxy",
]


def _default_build_dir():
    base = os.path.join(os.getcwd(), "duneanaobj", "build")
    cand = os.path.join(base, _SYSTEM)
    if os.path.isdir(os.path.join(cand, "lib")):
        return cand
    # fall back to any build/<X>/ that actually contains a lib/ directory
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(os.path.join(d, "lib")):
            return d
    return cand   # may not exist yet; setup_* will surface a clear error


BUILD_DIR = os.environ.get("DUNEANAOBJ_BUILD", _default_build_dir())
LIB_PATH = os.path.join(BUILD_DIR, "lib")


def _python_bindings_dir():
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    cand = os.path.join(BUILD_DIR, "python", pyver)
    if os.path.isdir(cand):
        return cand
    others = sorted(glob.glob(os.path.join(BUILD_DIR, "python", "*")))
    return others[-1] if others else cand


def setup_pysrproxy():
    """Configure sys.path, preload the StandardRecord libs, import pySRProxy.

    Returns the imported ``pySRProxy`` module.
    """
    py_path = _python_bindings_dir()
    if py_path not in sys.path:
        sys.path.insert(0, py_path)
    for _lib in _STDREC_LIBS:
        full = os.path.join(LIB_PATH, f"{_lib}.{_LIB_EXT}")
        try:
            ctypes.CDLL(full)
        except OSError as e:
            # RPATH may still resolve the libraries at import time; the preload
            # is best-effort, so warn but keep going.
            print(f"  WARNING: could not preload {os.path.basename(full)}: {e}")
    import pySRProxy
    return pySRProxy


def setup_root():
    """Import ROOT and load the StandardRecord dictionary libraries.

    Returns the imported ``ROOT`` module.
    """
    try:
        import subprocess
        libdir = subprocess.check_output(
            ["root-config", "--libdir"]).decode().strip()
        if libdir not in sys.path:
            sys.path.insert(0, libdir)
    except Exception:
        pass
    import ROOT
    for _lib in _STDREC_LIBS:
        ROOT.gSystem.Load(os.path.join(LIB_PATH, _lib))
    return ROOT


def open_proxy(tree="recTree", path=None):
    """Convenience wrapper: return a StandardRecordFileReader over the CAF."""
    pySRProxy = setup_pysrproxy()
    return pySRProxy.StandardRecordFileReader(tree, path or CAF_FILE)

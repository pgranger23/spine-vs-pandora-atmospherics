#!/usr/bin/env python3
"""
Comprehensive comparison of SPINE (DLP) vs Pandora neutrino event reconstruction.

Usage:
    python analysis_spine_vs_pandora.py

Output: figures/ directory with comparison plots.
"""

import os, sys, ctypes, math, json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
from matplotlib.colors import LogNorm

def _safe_hist2d(ax, x, y, bins, cmap="viridis", range=None, fig=None):
    """2D histogram with log color scale, handling empty data gracefully."""
    counts, xed, yed = np.histogram2d(x, y, bins=bins, range=range)
    if counts.max() == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        return None
    counts_m = np.ma.masked_where(counts < 1, counts)
    pc = ax.pcolormesh(xed, yed, counts_m.T, norm=LogNorm(vmin=1), cmap=cmap)
    if fig is not None:
        fig.colorbar(pc, ax=ax, label="Counts (log scale)")
    return pc
from tqdm import tqdm

# ============================================================
# Setup paths & pySRProxy
# ============================================================
base_dir = os.getcwd()
lib_path = os.path.join(base_dir, "duneanaobj/build/Linux/lib")
py_path  = os.path.join(base_dir, "duneanaobj/build/Linux/python/3.13")
if py_path not in sys.path:
    sys.path.insert(0, py_path)

_dylibs_to_preload = [
    "libSRProxy_BasicTypes.dylib",
    "libduneanaobj_StandardRecord.dylib",
    "libduneanaobj_StandardRecordProxy.dylib",
]
for _lib in _dylibs_to_preload:
    _full = os.path.join(lib_path, _lib)
    try:
        ctypes.CDLL(_full)
    except OSError as e:
        print(f"  WARNING: could not load {_lib}: {e}")

import pySRProxy

# Output directory for figures
outdir = "figures"
os.makedirs(outdir, exist_ok=True)

import plot_style
plot_style.apply()
from plot_style import SPINE_COLOR, PANDORA_COLOR

_nan = float("nan")

def _safe_norm(x, y, z):
    n = math.sqrt(x*x + y*y + z*z)
    return (x/n, y/n, z/n) if n > 0 else (_nan, _nan, _nan)

def _closest_dlp_idx(ixn, nu_vtx):
    """Index of DLP interaction whose vertex is closest to the true vertex."""
    best, best_i = float("inf"), 0
    nvx, nvy, nvz = nu_vtx.x, nu_vtx.y, nu_vtx.z
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = math.sqrt((v.x-nvx)**2 + (v.y-nvy)**2 + (v.z-nvz)**2)
        if d < best:
            best, best_i = d, i
    return best_i

def _dlp_momentum_sum(ixn):
    """Sum primary-particle momenta across all DLP interactions.

    EM particles (e, γ, π⁰) carry raw ADC-scale momentum ~5× too large;
    apply the ADC-to-electron calibration (×0.198) before summing so the
    direction is not dragged off-axis by over-weighted EM showers.
    """
    px = py = pz = 0.0
    for idx in range(ixn.ndlp):
        for pidx in range(ixn.dlp[idx].part.ndlp):
            p = ixn.dlp[idx].part.dlp[pidx]
            if p.primary:
                sc = 0.198 if abs(int(p.pdg)) in {11, 22, 111} else 1.0
                px += p.p.x * sc; py += p.p.y * sc; pz += p.p.z * sc
    return px, py, pz

# ============================================================
# Helper functions for analysis
# ============================================================
def profile(x, y, bins, stat="median"):
    val, _, _ = binned_statistic(x, y, statistic=stat, bins=bins)
    cnt, _, _ = binned_statistic(x, y, statistic="count",  bins=bins)
    std, _, _ = binned_statistic(x, y, statistic="std",    bins=bins)
    err = std / np.sqrt(np.maximum(cnt, 1))
    mask = cnt > 3
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, val, err, mask

def iqr_half(data):
    """(84th-16th percentile)/2 -- robust width estimator."""
    return lambda x: (np.nanpercentile(x, 84) - np.nanpercentile(x, 16)) / 2

def cdf_curve(arr):
    s = np.sort(np.abs(arr[np.isfinite(arr)]))
    return s, np.arange(1, len(s)+1)/len(s)

# Detector bounds for fallback preselection (ArgonCube TPC)
_DET_X_MIN, _DET_X_MAX = -360, 360
_DET_Y_MIN, _DET_Y_MAX = -600, 600
_DET_Z_MIN, _DET_Z_MAX = 0, 1400
_FID_MARGIN = 10  # cm from edge

def _is_in_fiducial(x, y, z):
    """Check if vertex is inside fiducial volume (True if yes)."""
    return (_DET_X_MIN + _FID_MARGIN < x < _DET_X_MAX - _FID_MARGIN and
            _DET_Y_MIN + _FID_MARGIN < y < _DET_Y_MAX - _FID_MARGIN and
            _DET_Z_MIN + _FID_MARGIN < z < _DET_Z_MAX - _FID_MARGIN)

# ============================================================
# Step 1: Load data into Polars DataFrame
# ============================================================
print("=" * 60)
print("Step 1: Loading CAF data")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader('recTree', 'spine_merged_reprocess.root')

cols = {k: [] for k in [
    # truth
    "true_vtx_x","true_vtx_y","true_vtx_z",
    "true_mom_x","true_mom_y","true_mom_z",
    "true_dir_x","true_dir_y","true_dir_z",
    "nu_E","nu_pdg","nu_iscc","nu_mode",
    # SPINE (closest-match DLP interaction)
    "spine_vtx_x","spine_vtx_y","spine_vtx_z",
    "spine_dir_lngtrk_x","spine_dir_lngtrk_y","spine_dir_lngtrk_z",
    "spine_dir_heshw_x","spine_dir_heshw_y","spine_dir_heshw_z",
    "spine_dir_calo_x","spine_dir_calo_y","spine_dir_calo_z",
    "spine_dir_partmom_x","spine_dir_partmom_y","spine_dir_partmom_z",
    "spine_Enu_calo","spine_Enu_lep_calo","spine_Enu_mu_range",
    "spine_Enu_mu_mcs","spine_Enu_e_calo","spine_Enu_mu_had","spine_Enu_e_had",
    "spine_nparts","spine_preselected",
    "spine_cvn_nue","spine_cvn_numu","spine_cvn_nutau","spine_cvn_nc",
    # SPINE total-momentum direction
    "spine_totmom_dir_x","spine_totmom_dir_y","spine_totmom_dir_z",
    # SPINE per-axis vertex resolution dependence
    "spine_truth_overlap",
    # Pandora (first interaction)
    "pand_vtx_x","pand_vtx_y","pand_vtx_z",
    "pand_dir_lngtrk_x","pand_dir_lngtrk_y","pand_dir_lngtrk_z",
    "pand_dir_heshw_x","pand_dir_heshw_y","pand_dir_heshw_z",
    "pand_dir_calo_x","pand_dir_calo_y","pand_dir_calo_z",
    "pand_dir_partmom_x","pand_dir_partmom_y","pand_dir_partmom_z",
    "pand_Enu_calo","pand_Enu_lep_calo","pand_Enu_mu_range",
    "pand_Enu_mu_mcs","pand_Enu_e_calo","pand_Enu_mu_had","pand_Enu_e_had",
    "pand_nparts","pand_preselected",
    "pand_cvn_nue","pand_cvn_numu","pand_cvn_nutau","pand_cvn_nc",
    # Additional info
    "ndlp", "npandora",
    # Total momentum
    "total_true_p", "total_spine_p",
]}

_app = {k: cols[k].append for k in cols}

for ientry in tqdm(range(prox.entries()), desc="Loading CAF"):
    rec  = prox.entry(ientry)
    mc   = rec.mc
    if not mc.nu[0].iscc:
        continue
    ixn  = rec.common.ixn
    if ixn.ndlp < 1 or ixn.npandora < 1:
        continue

    nu   = mc.nu[0]
    vtx  = nu.vtx
    mom  = nu.momentum
    dx, dy, dz = _safe_norm(mom.x, mom.y, mom.z)

    # truth
    _app["true_vtx_x"](vtx.x); _app["true_vtx_y"](vtx.y); _app["true_vtx_z"](vtx.z)
    _app["true_mom_x"](mom.x); _app["true_mom_y"](mom.y); _app["true_mom_z"](mom.z)
    _app["true_dir_x"](dx);    _app["true_dir_y"](dy);    _app["true_dir_z"](dz)
    _app["nu_E"](nu.E); _app["nu_pdg"](int(nu.pdg)); _app["nu_iscc"](bool(nu.iscc))
    try:
        _app["nu_mode"](int(nu.mode))
    except:
        _app["nu_mode"](-1)

    # SPINE
    si   = _closest_dlp_idx(ixn, vtx)
    dlp  = ixn.dlp[si]
    sv   = dlp.vtx
    sd   = dlp.dir
    sE   = dlp.Enu
    scvn = dlp.nuhyp.cvn

    _app["spine_vtx_x"](sv.x); _app["spine_vtx_y"](sv.y); _app["spine_vtx_z"](sv.z)
    lt = sd.lngtrk; hs = sd.heshw; ca = sd.calo; pm = sd.part_mom_sum
    _app["spine_dir_lngtrk_x"](lt.x);  _app["spine_dir_lngtrk_y"](lt.y);  _app["spine_dir_lngtrk_z"](lt.z)
    _app["spine_dir_heshw_x"](hs.x);   _app["spine_dir_heshw_y"](hs.y);   _app["spine_dir_heshw_z"](hs.z)
    _app["spine_dir_calo_x"](ca.x);    _app["spine_dir_calo_y"](ca.y);    _app["spine_dir_calo_z"](ca.z)
    _app["spine_dir_partmom_x"](pm.x); _app["spine_dir_partmom_y"](pm.y); _app["spine_dir_partmom_z"](pm.z)
    _app["spine_Enu_calo"](sE.calo);   _app["spine_Enu_lep_calo"](sE.lep_calo)
    _app["spine_Enu_mu_range"](sE.mu_range); _app["spine_Enu_mu_mcs"](sE.mu_mcs)
    _app["spine_Enu_e_calo"](sE.e_calo); _app["spine_Enu_mu_had"](sE.mu_had); _app["spine_Enu_e_had"](sE.e_had)
    _app["spine_nparts"](dlp.part.ndlp)
    # Fallback preselection: use flag if set, else fiducial + nparts > 0
    sp_flag = bool(dlp.preselected)
    if not sp_flag:
        sp_flag = _is_in_fiducial(sv.x, sv.y, sv.z) and dlp.part.ndlp > 0
    _app["spine_preselected"](sp_flag)
    _app["spine_cvn_nue"](scvn.nue); _app["spine_cvn_numu"](scvn.numu)
    _app["spine_cvn_nutau"](scvn.nutau); _app["spine_cvn_nc"](scvn.nc)

    # Total momentum sums
    total_true_p = 0.0
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        pm = math.sqrt(tp.p.px**2 + tp.p.py**2 + tp.p.pz**2)
        if pm > 0.02: total_true_p += pm
    _app["total_true_p"](total_true_p)

    total_spine_p = 0.0
    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        if p.primary:
            pm = math.sqrt(p.p.x**2 + p.p.y**2 + p.p.z**2)
            if pm > 1:
                if abs(p.pdg) in {11, 22, 111}:
                    pm *= 0.198  # ADC-to-electron calibration correction
                total_spine_p += pm * 0.001  # MeV → GeV
    _app["total_spine_p"](total_spine_p)

    try:
        _app["spine_truth_overlap"](float(dlp.truthOverlap))
    except:
        _app["spine_truth_overlap"](_nan)

    tmx, tmy, tmz = _dlp_momentum_sum(ixn)
    tdx, tdy, tdz = _safe_norm(tmx, tmy, tmz)
    _app["spine_totmom_dir_x"](tdx); _app["spine_totmom_dir_y"](tdy); _app["spine_totmom_dir_z"](tdz)

    # Pandora
    pan  = ixn.pandora[0]
    pv   = pan.vtx
    pd   = pan.dir
    pE   = pan.Enu
    pcvn = pan.nuhyp.cvn
    _app["pand_vtx_x"](pv.x); _app["pand_vtx_y"](pv.y); _app["pand_vtx_z"](pv.z)
    plt2 = pd.lngtrk; phs = pd.heshw; pca = pd.calo; ppm = pd.part_mom_sum
    _app["pand_dir_lngtrk_x"](plt2.x); _app["pand_dir_lngtrk_y"](plt2.y); _app["pand_dir_lngtrk_z"](plt2.z)
    _app["pand_dir_heshw_x"](phs.x);   _app["pand_dir_heshw_y"](phs.y);   _app["pand_dir_heshw_z"](phs.z)
    _app["pand_dir_calo_x"](pca.x);    _app["pand_dir_calo_y"](pca.y);    _app["pand_dir_calo_z"](pca.z)
    _app["pand_dir_partmom_x"](ppm.x); _app["pand_dir_partmom_y"](ppm.y); _app["pand_dir_partmom_z"](ppm.z)
    _app["pand_Enu_calo"](pE.calo);    _app["pand_Enu_lep_calo"](pE.lep_calo)
    _app["pand_Enu_mu_range"](pE.mu_range); _app["pand_Enu_mu_mcs"](pE.mu_mcs)
    _app["pand_Enu_e_calo"](pE.e_calo); _app["pand_Enu_mu_had"](pE.mu_had); _app["pand_Enu_e_had"](pE.e_had)
    _app["pand_nparts"](max(0, pan.part.npandora - 1))  # exclude fake last particle
    # Fallback preselection
    pp_flag = bool(pan.preselected)
    if not pp_flag:
        pp_flag = _is_in_fiducial(pv.x, pv.y, pv.z) and (pan.part.npandora - 1) > 0
    _app["pand_preselected"](pp_flag)
    _app["pand_cvn_nue"](pcvn.nue); _app["pand_cvn_numu"](pcvn.numu)
    _app["pand_cvn_nutau"](pcvn.nutau); _app["pand_cvn_nc"](pcvn.nc)

    _app["ndlp"](ixn.ndlp); _app["npandora"](ixn.npandora)

# Build Polars DataFrame
raw = {k: np.array(v, dtype=np.float32) if k not in (
    "nu_iscc","nu_mode","spine_preselected","pand_preselected",
    "nu_pdg","spine_nparts","pand_nparts","ndlp","npandora","spine_truth_overlap",
) else np.array(v) for k, v in cols.items()}

df = (
    pl.DataFrame(raw)
    .with_columns([
        (pl.col("spine_vtx_x") - pl.col("true_vtx_x")).alias("spine_dvtx_x"),
        (pl.col("spine_vtx_y") - pl.col("true_vtx_y")).alias("spine_dvtx_y"),
        (pl.col("spine_vtx_z") - pl.col("true_vtx_z")).alias("spine_dvtx_z"),
        (pl.col("pand_vtx_x")  - pl.col("true_vtx_x")).alias("pand_dvtx_x"),
        (pl.col("pand_vtx_y")  - pl.col("true_vtx_y")).alias("pand_dvtx_y"),
        (pl.col("pand_vtx_z")  - pl.col("true_vtx_z")).alias("pand_dvtx_z"),
    ])
    .with_columns([
        ((pl.col("spine_dvtx_x")**2 + pl.col("spine_dvtx_y")**2 + pl.col("spine_dvtx_z")**2).sqrt()).alias("spine_dvtx_3d"),
        ((pl.col("pand_dvtx_x")**2  + pl.col("pand_dvtx_y")**2  + pl.col("pand_dvtx_z")**2 ).sqrt()).alias("pand_dvtx_3d"),
    ])
    .with_columns([
        (pl.col("spine_dir_lngtrk_x")*pl.col("true_dir_x") +
         pl.col("spine_dir_lngtrk_y")*pl.col("true_dir_y") +
         pl.col("spine_dir_lngtrk_z")*pl.col("true_dir_z")).clip(-1,1).alias("spine_cos_lngtrk"),
        (pl.col("pand_dir_lngtrk_x")*pl.col("true_dir_x") +
         pl.col("pand_dir_lngtrk_y")*pl.col("true_dir_y") +
         pl.col("pand_dir_lngtrk_z")*pl.col("true_dir_z")).clip(-1,1).alias("pand_cos_lngtrk"),
        (pl.col("spine_dir_heshw_x")*pl.col("true_dir_x") +
         pl.col("spine_dir_heshw_y")*pl.col("true_dir_y") +
         pl.col("spine_dir_heshw_z")*pl.col("true_dir_z")).clip(-1,1).alias("spine_cos_heshw"),
        (pl.col("pand_dir_heshw_x")*pl.col("true_dir_x") +
         pl.col("pand_dir_heshw_y")*pl.col("true_dir_y") +
         pl.col("pand_dir_heshw_z")*pl.col("true_dir_z")).clip(-1,1).alias("pand_cos_heshw"),
        (pl.col("spine_dir_calo_x")*pl.col("true_dir_x") +
         pl.col("spine_dir_calo_y")*pl.col("true_dir_y") +
         pl.col("spine_dir_calo_z")*pl.col("true_dir_z")).clip(-1,1).alias("spine_cos_calo"),
        (pl.col("pand_dir_calo_x")*pl.col("true_dir_x") +
         pl.col("pand_dir_calo_y")*pl.col("true_dir_y") +
         pl.col("pand_dir_calo_z")*pl.col("true_dir_z")).clip(-1,1).alias("pand_cos_calo"),
        (pl.col("spine_totmom_dir_x")*pl.col("true_dir_x") +
         pl.col("spine_totmom_dir_y")*pl.col("true_dir_y") +
         pl.col("spine_totmom_dir_z")*pl.col("true_dir_z")).clip(-1,1).alias("spine_cos_totmom"),
    ])
    .with_columns([
        (pl.col("spine_cos_lngtrk").arccos() * (180/math.pi)).alias("spine_dang_lngtrk_deg"),
        (pl.col("pand_cos_lngtrk").arccos()  * (180/math.pi)).alias("pand_dang_lngtrk_deg"),
        (pl.col("spine_cos_heshw").arccos()  * (180/math.pi)).alias("spine_dang_heshw_deg"),
        (pl.col("pand_cos_heshw").arccos()   * (180/math.pi)).alias("pand_dang_heshw_deg"),
        (pl.col("spine_cos_calo").arccos()   * (180/math.pi)).alias("spine_dang_calo_deg"),
        (pl.col("pand_cos_calo").arccos()    * (180/math.pi)).alias("pand_dang_calo_deg"),
        (pl.col("spine_cos_totmom").arccos() * (180/math.pi)).alias("spine_dang_totmom_deg"),
        # Pandora best direction: lngtrk for numu, heshw for others
        pl.when(pl.col("nu_pdg").abs() == 14).then(pl.col("pand_cos_lngtrk"))
          .otherwise(pl.col("pand_cos_heshw")).alias("pand_cos_best"),
    ])
    .with_columns([
        (pl.col("pand_cos_best").arccos() * (180/math.pi)).alias("pand_dang_best_deg"),
    ])
    .with_columns(
        pl.when(pl.col("nu_pdg") ==  12).then(pl.lit("nue"))
          .when(pl.col("nu_pdg") == -12).then(pl.lit("nuebar"))
          .when(pl.col("nu_pdg") ==  14).then(pl.lit("numu"))
          .when(pl.col("nu_pdg") == -14).then(pl.lit("numubar"))
          .otherwise(pl.lit("other")).alias("nu_flavor")
    )
    .with_columns(
        pl.when(pl.col("nu_mode") == 0).then(pl.lit("QE"))
          .when(pl.col("nu_mode") == 1).then(pl.lit("RES"))
          .when(pl.col("nu_mode") == 2).then(pl.lit("DIS"))
          .when(pl.col("nu_mode") == 3).then(pl.lit("COH"))
          .when(pl.col("nu_mode") == 4).then(pl.lit("MEC"))
          .otherwise(pl.lit("other")).alias("nu_mode_label")
    )
)

print(f"Events loaded: {len(df):,}")
print(df.select(["nu_flavor","nu_E","spine_dvtx_3d","pand_dvtx_3d"]).describe())

# ============================================================
# Step 2: Data vectors for plotting
# ============================================================
E      = df["nu_E"].to_numpy()
pdg    = df["nu_pdg"].to_numpy()
numu   = np.abs(pdg) == 14
nue    = np.abs(pdg) == 12
iscc   = df["nu_iscc"].to_numpy()
nu_mode = df["nu_mode_label"].to_numpy()

Ebins  = np.logspace(-1, 1, 25)
Ecenters = 0.5*(Ebins[:-1]+Ebins[1:])

# ============================================================
# FIGURE 1a: Vertex resolution — residual histograms
# ============================================================
print("\n--- Figure 1a: Vertex residual histograms ---")
fig1a, axes1a = plt.subplots(2, 2, figsize=(12, 10))
fig1a.suptitle("Vertex resolution: residual distributions", fontsize=16, fontweight="bold")

rbins = np.linspace(-30, 30, 80)

for idx, axis_lbl in enumerate(["x", "y", "z"]):
    ax = axes1a[idx // 2, idx % 2]
    ds = df[f"spine_dvtx_{axis_lbl}"].to_numpy()
    dp = df[f"pand_dvtx_{axis_lbl}"].to_numpy()
    ax.hist(ds, bins=rbins, histtype="step", label="SPINE", lw=2.0, color=SPINE_COLOR)
    ax.hist(dp, bins=rbins, histtype="step", label="Pandora", lw=2.0, color=PANDORA_COLOR, ls="--")
    ms = np.nanmedian(ds); mp = np.nanmedian(dp)
    ax.axvline(0, color="red", ls="--", lw=1.5, alpha=0.6, label="Ideal")
    ax.set_yscale("log")
    ax.set_xlabel(f"Reco {axis_lbl.upper()} $-$ True {axis_lbl.upper()} [cm]")
    ax.set_ylabel("Counts")
    ax.set_title(f"{axis_lbl.upper()} residual (SPINE med={ms:.1f}, Pand med={mp:.1f})")
    ax.legend(fontsize=9)

# 3D CDF
ax = axes1a[1, 1]
for lbl, key, col in [("SPINE","spine_dvtx_3d",SPINE_COLOR), ("Pandora","pand_dvtx_3d",PANDORA_COLOR)]:
    x, y = cdf_curve(df[key].to_numpy())
    ax.plot(x, y, label=lbl, color=col, lw=2.0)
    for q, ls, lw_v in [(0.68,"--",1.5),(0.90,":",1.5)]:
        v = np.quantile(x, q)
        ax.axvline(v, color=col, ls=ls, lw=lw_v, alpha=0.8)
# Horizontal reference lines at 68% and 90%
ax.axhline(0.68, color="gray", ls="--", lw=1.5, alpha=0.8)
ax.axhline(0.90, color="gray", ls=":", lw=1.5, alpha=0.8)
# Annotate reference levels
ax.text(ax.get_xlim()[1]*0.98, 0.68, "68%", va="bottom", ha="right", fontsize=10, color="gray")
ax.text(ax.get_xlim()[1]*0.98, 0.90, "90%", va="bottom", ha="right", fontsize=10, color="gray")
ax.set_xscale("log"); ax.set_xlabel(r"$|$3D vertex offset$|$ [cm]")
ax.set_ylabel("CDF"); ax.set_title("3D vertex resolution CDF")
ax.legend(fontsize=9, loc="lower right")

fig1a.tight_layout()
fig1a.savefig(os.path.join(outdir, "01a_vertex_histograms.png"), dpi=150)
plt.close(fig1a)

# ============================================================
# FIGURE 1b: Vertex resolution — dependence on energy
# ============================================================
print("--- Figure 1b: Vertex resolution vs energy ---")
fig1b, axes1b = plt.subplots(2, 2, figsize=(12, 10))
fig1b.suptitle("Vertex resolution vs neutrino energy", fontsize=16, fontweight="bold")

# 3D distance vs neutrino energy (numu / nue)
ax = axes1b[0, 0]
# Background: event density histogram
ax.hist(E, bins=Ebins, density=True, alpha=0.15, color="gray",
        label="Event density", zorder=0)
for flavor, mask_f, ls in [("numu", numu, "-"), ("nue", nue, "--")]:
    for lbl, key, col_c in [("SPINE","spine_dvtx_3d",SPINE_COLOR), ("Pandora","pand_dvtx_3d",PANDORA_COLOR)]:
        xc, val, err, mk = profile(E[mask_f], df[key].to_numpy()[mask_f], Ebins, stat="median")
        ax.errorbar(xc[mk], val[mk], yerr=err[mk],
                    marker="o", ms=5, ls=ls, color=col_c, label=f"{lbl} {flavor}")
ax.set_xscale("log"); ax.set_xlabel(r"True neutrino $E$ [GeV]")
ax.set_ylabel("Median 3D dist [cm]"); ax.set_title("3D vtx resolution vs $E$")
ax.legend(fontsize=9)

# IQR/2 of per-axis residual vs energy
for col_i, axis_lbl in enumerate(["x", "y", "z"]):
    ax = axes1b[(col_i + 1) // 2, (col_i + 1) % 2]
    # Background: event density
    ax.hist(E, bins=Ebins, density=True, alpha=0.15, color="gray", zorder=0)
    for lbl, key_pfx, col_c in [("SPINE","spine",SPINE_COLOR), ("Pandora","pand",PANDORA_COLOR)]:
        arr = df[f"{key_pfx}_dvtx_{axis_lbl}"].to_numpy()
        xc, val, err, mk = profile(E, arr, Ebins, stat=iqr_half(arr))
        cnt, _, _ = binned_statistic(E, arr, statistic="count", bins=Ebins)
        err2 = val / np.sqrt(np.maximum(cnt, 1)*2)
        ax.errorbar(xc[mk], val[mk], yerr=err2[mk],
                    marker="o", ms=5, ls="-", color=col_c, label=lbl)
    ax.set_xscale("log"); ax.set_xlabel(r"True neutrino $E$ [GeV]")
    ax.set_ylabel("(84%$-$16%)/2 [cm]")
    ax.set_title(f"{axis_lbl.upper()} vtx IQR/2 vs $E$")
    ax.legend(fontsize=9)

fig1b.tight_layout()
fig1b.savefig(os.path.join(outdir, "01b_vertex_vs_energy.png"), dpi=150)
plt.close(fig1b)

# ============================================================
# FIGURE 2a: Angular resolution — distributions
# ============================================================
print("--- Figure 2a: Angular resolution distributions ---")
# Pandora best: lngtrk for numu, heshw for nue/nuebar/nutau
pand_best_key = "pand_dang_best_deg"
spine_key = "spine_dang_totmom_deg"

fig2a, axes2a = plt.subplots(2, 2, figsize=(12, 10))
fig2a.suptitle("Angular resolution: distribution overview", fontsize=14, fontweight="bold")

# Angle histograms: SPINE totmom vs Pandora best
abins = np.linspace(0, 60, 80)
ax0 = axes2a[0, 0]
for lbl, key, col_c, ls in [
    ("SPINE (particle sum)", spine_key, SPINE_COLOR, "-"),
    ("Pandora (lngtrk/heshw by flavor)", pand_best_key, PANDORA_COLOR, "--"),
]:
    arr = df[key].to_numpy()
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    ax0.hist(arr, bins=abins, histtype="step", label=lbl, lw=1.3, ls=ls, color=col_c)
ax0.set_xlabel(r"$\Delta\theta$ [deg]"); ax0.set_ylabel("Counts")
ax0.set_title(r"Angular residual $\Delta\theta$ (all events)")
ax0.legend(fontsize=8, loc="upper right")

# CDF
ax = axes2a[0, 1]
for lbl, key, col_c, ls in [
    ("SPINE (particle sum)", spine_key, SPINE_COLOR, "-"),
    ("Pandora (lngtrk/heshw by flavor)", pand_best_key, PANDORA_COLOR, "--"),
]:
    arr = df[key].to_numpy()
    x, y = cdf_curve(arr)
    ax.plot(x, y, label=lbl, color=col_c, ls=ls, lw=1.3)
ax.set_xlabel(r"$\Delta\theta$ [deg]"); ax.set_ylabel("CDF")
ax.set_title("Angular resolution CDF"); ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0, 90)

# SPINE totmom direction component residuals
ax = axes2a[1, 0]
dbins2 = np.linspace(-0.5, 0.5, 60)
for comp, col_c in [("x", SPINE_COLOR), ("y", PANDORA_COLOR), ("z", "C2")]:
    tc = f"true_dir_{comp}"
    res = (df[f"spine_totmom_dir_{comp}"] - df[tc]).to_numpy()
    ax.hist(res, bins=dbins2, histtype="step", label=f"totmom {comp}", lw=1.3, color=col_c)
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("Reco dir $-$ True dir"); ax.set_ylabel("Counts")
ax.set_title("SPINE totmom direction residuals"); ax.legend(fontsize=7)

# Pandora best direction component residuals (flavor-dependent)
ax = axes2a[1, 1]
dbins_p = np.linspace(-0.5, 0.5, 60)
# Need to build best-direction components manually
pand_best_x = np.where(np.abs(df["nu_pdg"].to_numpy()) == 14,
                       df["pand_dir_lngtrk_x"].to_numpy(),
                       df["pand_dir_heshw_x"].to_numpy())
pand_best_y = np.where(np.abs(df["nu_pdg"].to_numpy()) == 14,
                       df["pand_dir_lngtrk_y"].to_numpy(),
                       df["pand_dir_heshw_y"].to_numpy())
pand_best_z = np.where(np.abs(df["nu_pdg"].to_numpy()) == 14,
                       df["pand_dir_lngtrk_z"].to_numpy(),
                       df["pand_dir_heshw_z"].to_numpy())
for comp, bc, col_c in [("x", pand_best_x, SPINE_COLOR), ("y", pand_best_y, PANDORA_COLOR), ("z", pand_best_z, "C2")]:
    tc = f"true_dir_{comp}"
    res = (bc - df[tc].to_numpy())
    ax.hist(res, bins=dbins_p, histtype="step", label=f"best {comp}", lw=1.3, color=col_c)
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("Reco dir $-$ True dir"); ax.set_ylabel("Counts")
ax.set_title("Pandora best dir residuals (lngtrk/heshw by flavor)"); ax.legend(fontsize=7)

fig2a.tight_layout()
fig2a.savefig(os.path.join(outdir, "02a_angular_distributions.png"), dpi=150)
plt.close(fig2a)

# ============================================================
# FIGURE 2b: Angular resolution — vs energy
# ============================================================
print("--- Figure 2b: Angular resolution vs energy ---")
pand_best_key = "pand_dang_best_deg"
spine_key = "spine_dang_totmom_deg"

fig2b, axes2b = plt.subplots(2, 2, figsize=(12, 10))
fig2b.suptitle("Angular resolution vs neutrino energy", fontsize=14, fontweight="bold")

# IQR/2 vs E: SPINE totmom vs Pandora best
ax = axes2b[0, 0]
for lbl, key, col_c in [
    ("SPINE (particle sum)", spine_key, SPINE_COLOR),
    ("Pandora (best by flavor)", pand_best_key, PANDORA_COLOR),
]:
    arr = df[key].to_numpy()
    xc, val, err, mk = profile(E, arr, Ebins, stat=iqr_half(arr))
    cnt, _, _ = binned_statistic(E, arr, statistic="count", bins=Ebins)
    err2 = val / np.sqrt(np.maximum(cnt, 1)*2)
    ax.errorbar(xc[mk], val[mk], yerr=err2[mk],
                marker="o", ms=5, color=col_c, label=lbl)
ax.set_xscale("log"); ax.set_xlabel(r"True neutrino $E$ [GeV]")
ax.set_ylabel(r"IQR/2 [deg]")
ax.set_title("Angular IQR/2 — all events"); ax.legend(fontsize=8)
ax.set_ylim(bottom=0)

# IQR/2 vs E by flavor
ax = axes2b[0, 1]
for flavor, mask_f, ls in [("numu", numu, "-"), ("nue", nue, "--")]:
    for lbl, key, col_c in [
        ("SPINE", spine_key, SPINE_COLOR),
        ("Pandora", pand_best_key, PANDORA_COLOR),
    ]:
        arr = df[key].to_numpy()
        sub = arr[mask_f]; sub_E = E[mask_f]
        xc, val, err, mk = profile(sub_E, sub, Ebins, stat=iqr_half(np.concatenate([sub, [0]])))
        cnt, _, _ = binned_statistic(sub_E, sub, statistic="count", bins=Ebins)
        err2 = val / np.sqrt(np.maximum(cnt, 1)*2)
        ax.errorbar(xc[mk], val[mk], yerr=err2[mk],
                    marker="o", ms=4, color=col_c, label=f"{lbl} {flavor}", ls=ls)
ax.set_xscale("log"); ax.set_xlabel(r"True neutrino $E$ [GeV]")
ax.set_ylabel(r"IQR/2 [deg]")
ax.set_title("Angular IQR/2 — by flavor"); ax.legend(fontsize=7)
ax.set_ylim(bottom=0)

# Median vs E
ax = axes2b[1, 0]
for lbl, key, col_c in [
    ("SPINE (particle sum)", spine_key, SPINE_COLOR),
    ("Pandora (best by flavor)", pand_best_key, PANDORA_COLOR),
]:
    arr = df[key].to_numpy()
    xc, val, err, mk = profile(E, arr, Ebins, stat=np.median)
    ax.errorbar(xc[mk], val[mk], marker="o", ms=5, color=col_c, label=lbl)
ax.set_xscale("log"); ax.set_xlabel(r"True neutrino $E$ [GeV]")
ax.set_ylabel(r"Median $\Delta\theta$ [deg]")
ax.set_title("Angular median vs energy"); ax.legend(fontsize=8)
ax.set_ylim(bottom=0)

# SPINE vs Pandora scatter (binned median comparison)
ax = axes2b[1, 1]
spine_arr = df[spine_key].to_numpy()
pand_arr = df[pand_best_key].to_numpy()
valid = np.isfinite(spine_arr) & np.isfinite(pand_arr)
xc, val, err, mk = profile(pand_arr[valid], spine_arr[valid], np.linspace(0, 60, 20), stat=np.median)
ax.errorbar(xc[mk], val[mk], marker="o", ms=5, color=SPINE_COLOR, label="Binned median")
ax.plot([0, 60], [0, 60], "k:", lw=0.8, alpha=0.5)
ax.set_xlabel(r"Pandora $\Delta\theta$ [deg]")
ax.set_ylabel(r"SPINE $\Delta\theta$ [deg]")
ax.set_title("SPINE vs Pandora angular resolution"); ax.legend(fontsize=9)

fig2b.tight_layout()
fig2b.savefig(os.path.join(outdir, "02b_angular_vs_energy.png"), dpi=150)
plt.close(fig2b)

# ============================================================
# FIGURE 3a: Energy reconstruction — E_reco/E_true histograms
# ============================================================
print("--- Figure 3a: Energy reconstruction histograms ---")
fig3a, axes3a = plt.subplots(2, 2, figsize=(12, 10))
fig3a.suptitle(r"Energy reconstruction: $E_\mathrm{reco}/E_\mathrm{true}$", fontsize=16, fontweight="bold")

# Only Pandora Enu calo is populated — show it alone in first panel
ax0 = axes3a[0, 0]
pE_calo = df["pand_Enu_calo"].to_numpy()
valid_p = np.isfinite(pE_calo) & (pE_calo > 0)
ratio_p = pE_calo[valid_p] / E[valid_p]
rbins2 = np.linspace(0, 3, 60)
ax0.hist(ratio_p, bins=rbins2, histtype="step", label="Pandora Enu calo", lw=2.0, color=PANDORA_COLOR)
ax0.axvline(1, color="k", lw=1, ls=":")
ax0.set_xlabel(r"$E_\mathrm{reco} / E_\mathrm{true}$")
ax0.set_ylabel("Counts")
ax0.set_title(f"Pandora Enu calo (N={valid_p.sum():,}, med={np.median(ratio_p):.2f})")
ax0.legend(fontsize=9)

# Replace remaining 3 panels with workaround text
for idx in range(3):
    ax = axes3a[(idx+1)//2, (idx+1)%2]
    ax.axis("off")
text_3a = (
    "SPINE Energy Estimator Branches\n"
    "are 100% NaN in this dataset:\n\n"
    "  • Enu calo: NOT POPULATED\n"
    "  • lep_calo: NOT POPULATED\n"
    "  • mu_range: NOT POPULATED\n"
    "  • mu_mcs:   NOT POPULATED\n"
    "  • e_calo:   NOT POPULATED\n"
    "  • mu_had:   NOT POPULATED\n"
    "  • e_had:    NOT POPULATED\n\n"
    "→ Using particle-sum hadronic\n"
    "  energy as workaround\n"
    "→ Only Pandora Enu calo\n"
    "  is available (shown left)"
)
# Put text in the three empty panels
for idx, ax in enumerate([axes3a[0, 1], axes3a[1, 0], axes3a[1, 1]]):
    if idx == 0:
        ax.text(0.5, 0.5, text_3a, transform=ax.transAxes, fontsize=11,
                ha="center", va="center", family="monospace")
    else:
        ax.axis("off")

fig3a.tight_layout()
fig3a.savefig(os.path.join(outdir, "03a_energy_histograms.png"), dpi=150)
plt.close(fig3a)

# ============================================================
# FIGURE 3b: Energy reconstruction — bias vs energy
# ============================================================
print("--- Figure 3b: Energy reconstruction bias vs E ---")
fig3b, axes3b = plt.subplots(2, 2, figsize=(12, 10))
fig3b.suptitle(r"Energy reconstruction bias vs true $E$", fontsize=16, fontweight="bold")

# Only Pandora Enu calo is populated
ax0 = axes3b[0, 0]
pE_calo = df["pand_Enu_calo"].to_numpy()
valid_p = np.isfinite(pE_calo) & (pE_calo > 0)
xc, val, err, mk = profile(E[valid_p], pE_calo[valid_p]/E[valid_p], Ebins)
ax0.errorbar(xc[mk], val[mk], yerr=err[mk], marker="o", ms=5, color=PANDORA_COLOR, label="Pandora Enu calo")
ax0.axhline(1, color="k", lw=0.8, ls=":")
ax0.set_xscale("log"); ax0.set_xlabel(r"True neutrino $E$ [GeV]")
ax0.set_ylabel(r"Median $E_\mathrm{reco}/E_\mathrm{true}$")
ax0.set_title("Pandora Enu calo bias vs energy")
ax0.legend(fontsize=9)

# Remaining 3 panels: workaround text
text_3b = (
    "SPINE Enu estimators are\n"
    "100% NaN in this dataset.\n\n"
    "Energy studies use:\n"
    "  • Particle-sum hadronic\n"
    "    energy from SPINE reco\n"
    "  • Pandora Enu calo for\n"
    "    total energy (shown left)\n\n"
    "→ See hadronic correction\n"
    "  study for details"
)
for idx, ax in enumerate([axes3b[0, 1], axes3b[1, 0], axes3b[1, 1]]):
    if idx == 0:
        ax.text(0.5, 0.5, text_3b, transform=ax.transAxes, fontsize=11,
                ha="center", va="center", family="monospace")
    else:
        ax.axis("off")

fig3b.tight_layout()
fig3b.savefig(os.path.join(outdir, "03b_energy_bias_vs_energy.png"), dpi=150)
plt.close(fig3b)

# ============================================================
# FIGURE 4a: Multiplicity & CVN distributions
# ============================================================
print("--- Figure 4a: Multiplicity & CVN distributions ---")
fig4a, axes4a = plt.subplots(2, 2, figsize=(12, 10))
fig4a.suptitle("Multiplicity & CVN score distributions", fontsize=14, fontweight="bold")

# particle multiplicity histogram
ns = df["spine_nparts"].to_numpy()
np_ = df["pand_nparts"].to_numpy()
nbins3 = np.arange(0, 25)
axes4a[0, 0].hist(ns, bins=nbins3, alpha=0.6, label="SPINE")
axes4a[0, 0].hist(np_, bins=nbins3, alpha=0.6, label="Pandora")
axes4a[0, 0].set_xlabel("N reco particles"); axes4a[0, 0].set_ylabel("Counts")
axes4a[0, 0].set_title("Particle multiplicity"); axes4a[0, 0].legend()

# multiplicity vs energy
ax = axes4a[0, 1]
xc, vs, _, mks = profile(E, ns.astype(float), Ebins, stat="median")
xc, vp, _, mkp = profile(E, np_.astype(float), Ebins, stat="median")
ax.plot(xc[mks], vs[mks], "o-", label="SPINE", color=SPINE_COLOR)
ax.plot(xc[mkp], vp[mkp], "o-", label="Pandora", color=PANDORA_COLOR)
ax.set_xscale("log"); ax.set_xlabel(r"True $E$ [GeV]")
ax.set_ylabel("Median N particles")
ax.set_title("Multiplicity vs $E$"); ax.legend()

# CVN numu scores
ax = axes4a[1, 0]
sa = df["spine_cvn_numu"].to_numpy(); pa = df["pand_cvn_numu"].to_numpy()
ax.hist(sa[np.isfinite(sa)], bins=50, histtype="step", label="SPINE", lw=1.3, range=(0,1))
ax.hist(pa[np.isfinite(pa)], bins=50, histtype="step", label="Pandora", lw=1.3, range=(0,1), ls="--")
ax.set_xlabel(r"CVN score $\nu_\mu$"); ax.set_ylabel("Counts")
ax.set_title(r"CVN $\nu_\mu$ score"); ax.legend(fontsize=7)

# CVN nue scores
ax = axes4a[1, 1]
sa = df["spine_cvn_nue"].to_numpy(); pa = df["pand_cvn_nue"].to_numpy()
ax.hist(sa[np.isfinite(sa)], bins=50, histtype="step", label="SPINE", lw=1.3, range=(0,1))
ax.hist(pa[np.isfinite(pa)], bins=50, histtype="step", label="Pandora", lw=1.3, range=(0,1), ls="--")
ax.set_xlabel(r"CVN score $\nu_e$"); ax.set_ylabel("Counts")
ax.set_title(r"CVN $\nu_e$ score"); ax.legend(fontsize=7)

fig4a.tight_layout()
fig4a.savefig(os.path.join(outdir, "04a_multiplicity_cvn.png"), dpi=150)
plt.close(fig4a)

# ============================================================
# FIGURE 4b: Preselection & reconstruction quality
# ============================================================
print("--- Figure 4b: Preselection & quality ---")
fig4b, axes4b = plt.subplots(2, 2, figsize=(12, 10))
fig4b.suptitle("Preselection efficiency & reconstruction quality", fontsize=16, fontweight="bold")

# preselection rate vs energy
ax = axes4b[0, 0]
for color, key, lbl in [(SPINE_COLOR,"spine_preselected","SPINE"), (PANDORA_COLOR,"pand_preselected","Pandora")]:
    ps = df[key].to_numpy().astype(float)
    xc, val, err, mk = profile(E, ps, Ebins, stat="mean")
    cnt = binned_statistic(E, ps, "count", bins=Ebins)[0]
    ax.errorbar(xc[mk], val[mk],
                yerr=np.sqrt(val[mk]*(1-val[mk])/np.maximum(cnt, 1))[mk],
                marker="o", ms=5, color=color, label=lbl)
ax.set_xscale("log"); ax.set_xlabel(r"True $E$ [GeV]")
ax.set_ylabel("Preselection rate")
ax.set_title("Preselection efficiency vs $E$"); ax.legend(fontsize=9)

# vertex resolution vs multiplicity
ax = axes4b[0, 1]
for color, vtx_key, npart_key, lbl in [
    (SPINE_COLOR,"spine_dvtx_3d","spine_nparts","SPINE"),
    (PANDORA_COLOR,"pand_dvtx_3d","pand_nparts","Pandora")
]:
    npart_arr = df[npart_key].to_numpy().astype(float)
    res_arr   = df[vtx_key].to_numpy()
    nbins_n = np.arange(0, 20, 1)
    med_r, _, _ = binned_statistic(npart_arr, res_arr, "median", bins=nbins_n)
    cnt_r, _, _ = binned_statistic(npart_arr, res_arr, "count",  bins=nbins_n)
    ctr_n = 0.5*(nbins_n[:-1]+nbins_n[1:])
    mk2 = cnt_r > 3
    ax.errorbar(ctr_n[mk2], med_r[mk2], yerr=med_r[mk2]/np.sqrt(cnt_r[mk2]),
                marker="o", ms=5, color=color, label=lbl)
ax.set_xlabel("N reco particles"); ax.set_ylabel("Median 3D vtx dist [cm]")
ax.set_title("Vtx resolution vs multiplicity"); ax.legend(fontsize=9)

# 2D spatial scatter of rejected preselection events (SPINE)
ax = axes4b[1, 0]
spine_presel = df["spine_preselected"].to_numpy()
rejected = ~spine_presel
rej_x = df["spine_vtx_x"].to_numpy()[rejected]
rej_y = df["spine_vtx_y"].to_numpy()[rejected]
rej_z = df["spine_vtx_z"].to_numpy()[rejected]
# Plot all events as faint background
all_x = df["spine_vtx_x"].to_numpy()
all_y = df["spine_vtx_y"].to_numpy()
ax.scatter(all_x, all_y, s=1, alpha=0.1, c="gray", label="All events")
ax.scatter(rej_x, rej_y, s=8, alpha=0.6, c="red", label=f"Rejected (N={rejected.sum()})")
# Draw detector boundaries
ax.axhline(_DET_Y_MIN, color="k", ls="--", lw=0.8)
ax.axhline(_DET_Y_MAX, color="k", ls="--", lw=0.8)
ax.axvline(_DET_X_MIN, color="k", ls="--", lw=0.8)
ax.axvline(_DET_X_MAX, color="k", ls="--", lw=0.8)
# Fiducial volume boundary
ax.axhline(_DET_Y_MIN + _FID_MARGIN, color="gray", ls=":", lw=0.8)
ax.axhline(_DET_Y_MAX - _FID_MARGIN, color="gray", ls=":", lw=0.8)
ax.axvline(_DET_X_MIN + _FID_MARGIN, color="gray", ls=":", lw=0.8)
ax.axvline(_DET_X_MAX - _FID_MARGIN, color="gray", ls=":", lw=0.8)
ax.set_xlabel("SPINE vertex X [cm]"); ax.set_ylabel("SPINE vertex Y [cm]")
ax.set_title("Rejected preselection events (XY view)")
ax.legend(fontsize=8, loc="upper right")

# CVN: SPINE vs Pandora numu score correlation — 2D density
ax = axes4b[1, 1]
sa_mu = df["spine_cvn_numu"].to_numpy()
pa_mu = df["pand_cvn_numu"].to_numpy()
valid_both = np.isfinite(sa_mu) & np.isfinite(pa_mu)
_safe_hist2d(ax, sa_mu[valid_both], pa_mu[valid_both],
             bins=50, range=[[0,1],[0,1]], fig=fig4b)
ax.set_xlabel("SPINE CVN numu"); ax.set_ylabel("Pandora CVN numu")
ax.set_title("CVN numu: SPINE vs Pandora")

fig4b.tight_layout()
fig4b.savefig(os.path.join(outdir, "04b_preselection_quality.png"), dpi=150)
plt.close(fig4b)

# ============================================================
# FIGURE 5: Interaction mode breakdown
# ============================================================
print("--- Figure 5: Interaction mode breakdown ---")
fig5, axes5 = plt.subplots(2, 3, figsize=(16, 10))
fig5.suptitle("Performance by interaction mode", fontsize=14, fontweight="bold")

# Mode labels
mode_labels = ["QE", "RES", "DIS", "MEC", "COH", "other"]
mode_colors = [SPINE_COLOR, PANDORA_COLOR, "C2", "C3", "C4", "C5"]

# Performance metrics by mode
for ax_idx, (metric, ylabel) in enumerate([
    ("spine_dvtx_3d", "Median 3D vertex dist [cm]"),
    ("spine_dang_totmom_deg", "Median ang. res. [deg]"),
    ("spine_Enu_calo / nu_E", r"Median $E_\mathrm{reco}/E_\mathrm{true}$"),
]):
    row = ax_idx // 3; col = ax_idx % 3
    valid_modes = []
    values = []
    for mode_label in mode_labels:
        mask = nu_mode == mode_label
        if mask.sum() < 5:
            continue
        valid_modes.append(mode_label)
        if "/" in metric:
            col_name = metric.split("/")[0].strip()
            vals = df[col_name].to_numpy()[mask] / E[mask]
        else:
            vals = df[metric].to_numpy()[mask]
        vals = vals[np.isfinite(vals)]
        values.append(np.median(vals) if len(vals) > 0 else 0)
    colors = [mode_colors[mode_labels.index(m)] for m in valid_modes]
    axes5[row, col].bar(valid_modes, values, color=colors)
    axes5[row, col].set_ylabel(ylabel)
    axes5[row, col].set_title(ylabel)

# Sub-figures for SPINE vs Pandora by mode
for col_i, (metric_spine, metric_pand, ylabel) in enumerate([
    ("spine_dvtx_3d", "pand_dvtx_3d", "Median 3D vtx [cm]"),
    ("spine_dang_totmom_deg", "pand_dang_best_deg", "Median ang. res. [deg]"),
]):
    valid_modes = []
    s_vals = []
    p_vals = []
    for mode_label in mode_labels:
        mask = nu_mode == mode_label
        if mask.sum() < 5:
            continue
        valid_modes.append(mode_label)
        sv = df[metric_spine].to_numpy()[mask]
        pv = df[metric_pand].to_numpy()[mask]
        sv = sv[np.isfinite(sv)]
        pv = pv[np.isfinite(pv)]
        s_vals.append(np.median(sv) if len(sv) > 0 else 0)
        p_vals.append(np.median(pv) if len(pv) > 0 else 0)
    width = 0.35
    x_valid = np.arange(len(valid_modes))
    axes5[1, col_i].bar(x_valid - width/2, s_vals, width, label="SPINE", color=SPINE_COLOR)
    axes5[1, col_i].bar(x_valid + width/2, p_vals, width, label="Pandora", color=PANDORA_COLOR)
    axes5[1, col_i].set_xticks(x_valid)
    axes5[1, col_i].set_xticklabels(valid_modes)
    axes5[1, col_i].set_ylabel(ylabel)
    axes5[1, col_i].legend(fontsize=8)

fig5.tight_layout()
fig5.savefig(os.path.join(outdir, "05_interaction_mode_breakdown.png"), dpi=150)
plt.close(fig5)

# ============================================================
# FIGURE 6: Vertex position dependence of resolution
# ============================================================
print("--- Figure 6: Vertex position dependence ---")
fig6, axes6 = plt.subplots(1, 3, figsize=(18, 5))
fig6.suptitle("Vertex resolution vs true vertex position", fontsize=14, fontweight="bold")

for col_i, (axis_lbl, bins, det_lines) in enumerate([
    ("x", np.linspace(-400, 400, 30), [(-360, "Anode", 0), (0, "Cathode", 0), (360, "Anode", 0)]),
    ("y", np.linspace(-750, 750, 30), [(-600, "Edge", 0), (600, "Edge", 0)]),
    ("z", np.linspace(0, 1500, 30), [(0, "Edge", 0), (1400, "Edge", 0)]),
]):
    centers = 0.5 * (bins[:-1] + bins[1:])
    arr_pos = df[f"true_vtx_{axis_lbl}"].to_numpy()
    for lbl, key, col_c in [("SPINE","spine_dvtx_3d",SPINE_COLOR), ("Pandora","pand_dvtx_3d",PANDORA_COLOR)]:
        arr_res = df[key].to_numpy()
        med, _, _ = binned_statistic(arr_pos, arr_res, "median", bins=bins)
        cnt, _, _ = binned_statistic(arr_pos, arr_res, "count", bins=bins)
        mk = cnt > 3
        axes6[col_i].errorbar(centers[mk], med[mk], yerr=med[mk]/np.sqrt(cnt[mk]),
                              marker="o", ms=5, color=col_c, label=lbl)
    # Add detector element vertical lines
    for dpos, dlabel, dalpha in det_lines:
        axes6[col_i].axvline(dpos, color="red", ls=":", lw=1.5, alpha=0.6+0.2*dalpha, label=dlabel)
    axes6[col_i].set_xlabel(f"True vertex {axis_lbl.upper()} [cm]")
    axes6[col_i].set_ylabel("Median vertex distance [cm]")
    axes6[col_i].legend(loc="upper left", fontsize=9)
    axes6[col_i].set_title(f"Resolution vs true {axis_lbl.upper()}")

fig6.tight_layout()
fig6.savefig(os.path.join(outdir, "06_vertex_position_dependence.png"), dpi=150)
plt.close(fig6)

# ============================================================
# FIGURE 7: 2D correlation plots (reco vs true)
# ============================================================
print("--- Figure 7: 2D correlations ---")
fig7, axes7 = plt.subplots(2, 3, figsize=(16, 10))
fig7.suptitle("Reco vs True: 2D density correlations", fontsize=16, fontweight="bold")

# Vertex 3D: SPINE vs Pandora — log-scale 2D density
ax = axes7[0,0]
spine_vtx = df["spine_dvtx_3d"].to_numpy()
pand_vtx = df["pand_dvtx_3d"].to_numpy()
valid_vtx = np.isfinite(spine_vtx) & np.isfinite(pand_vtx) & (spine_vtx > 0.01) & (pand_vtx > 0.01)
_safe_hist2d(ax, pand_vtx[valid_vtx], spine_vtx[valid_vtx], bins=50, fig=fig7)
ax.set_xlabel("Pandora 3D vtx dist [cm]"); ax.set_ylabel("SPINE 3D vtx dist [cm]")
ax.set_title("Vertex residual: SPINE vs Pandora")
ax.set_xscale("log"); ax.set_yscale("log")
ax.plot([0.01, 1000], [0.01, 1000], "k--", lw=0.8)

# SPINE Enu calo vs true E — 2D density
ax = axes7[0,1]
sE = df["spine_Enu_calo"].to_numpy()
valid = np.isfinite(sE) & (sE > 0.01)
if valid.sum() > 10:
    _safe_hist2d(ax, E[valid], sE[valid], bins=50, fig=fig7)
else:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
ax.set_xlabel(r"True $E$ [GeV]"); ax.set_ylabel(r"SPINE $E_\nu$ calo [GeV]")
ax.set_title(r"SPINE $E_\nu$ calo vs True $E$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.plot([0.1, 100], [0.1, 100], "k--", lw=0.8)

# Pandora Enu calo vs true E — 2D density
ax = axes7[0,2]
pE = df["pand_Enu_calo"].to_numpy()
valid = np.isfinite(pE) & (pE > 0.01)
_safe_hist2d(ax, E[valid], pE[valid], bins=50, fig=fig7)
ax.set_xlabel(r"True $E$ [GeV]"); ax.set_ylabel(r"Pandora $E_\nu$ calo [GeV]")
ax.set_title(r"Pandora $E_\nu$ calo vs True $E$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.plot([0.1, 100], [0.1, 100], "k--", lw=0.8)

# Angular: SPINE (totmom) vs Pandora (best) — 2D density
ax = axes7[1,0]
spine_ang = df["spine_dang_totmom_deg"].to_numpy()
pand_ang = df["pand_dang_best_deg"].to_numpy()
valid_ang = np.isfinite(spine_ang) & np.isfinite(pand_ang) & (spine_ang >= 0) & (pand_ang >= 0)
_safe_hist2d(ax, pand_ang[valid_ang], spine_ang[valid_ang], bins=50, fig=fig7)
ax.set_xlabel("Pandora ang. res. [deg]"); ax.set_ylabel("SPINE ang. res. [deg]")
ax.set_title("Angular: SPINE totmom vs Pandora best")
ax.plot([0, 180], [0, 180], "k--", lw=0.8)

# N particles: SPINE vs Pandora — 2D density
ax = axes7[1,1]
_safe_hist2d(ax, df["pand_nparts"].to_numpy(), df["spine_nparts"].to_numpy(),
             bins=range(0, 30), fig=fig7)
ax.set_xlabel("Pandora N particles"); ax.set_ylabel("SPINE N particles")
ax.set_title("Particle multiplicity: SPINE vs Pandora")
ax.plot([0, 30], [0, 30], "k--", lw=0.8)

# CVN numu score: numu vs nue events — 2D density
ax = axes7[1,2]
numu_mask = numu & np.isfinite(df["spine_cvn_numu"].to_numpy()) & np.isfinite(df["spine_cvn_nue"].to_numpy())
nue_mask = nue & np.isfinite(df["spine_cvn_numu"].to_numpy()) & np.isfinite(df["spine_cvn_nue"].to_numpy())
_safe_hist2d(ax, df["spine_cvn_numu"].to_numpy()[numu_mask],
             df["spine_cvn_nue"].to_numpy()[numu_mask],
             bins=40, range=[[0,1],[0,1]], fig=fig7)
# Overlay nue events in a lighter style
ax.scatter(df["spine_cvn_numu"].to_numpy()[nue_mask],
           df["spine_cvn_nue"].to_numpy()[nue_mask],
           s=1, alpha=0.1, c="red", label=r"$\nu_e$ events")
ax.set_xlabel("SPINE CVN numu score"); ax.set_ylabel("SPINE CVN nue score")
ax.set_title("CVN scores (νμ density, νe scatter)"); ax.legend(fontsize=8, markerscale=5, loc="upper left")

fig7.tight_layout()
fig7.savefig(os.path.join(outdir, "07_2d_correlations.png"), dpi=150)
plt.close(fig7)

# ============================================================
# FIGURE 8: Vertex resolution by number of DLP interactions
# ============================================================
print("--- Figure 8: Vertex resolution vs number of interactions ---")
fig8, axes8 = plt.subplots(1, 2, figsize=(12, 5))
fig8.suptitle("Vertex resolution vs number of reconstructed interactions", fontsize=16, fontweight="bold")

nixn = df["ndlp"].to_numpy()
# Background: event count histogram
axes8[0].hist(nixn, bins=np.arange(0, 10, 1), density=True, alpha=0.15, color="gray",
              label="Event density", zorder=0)
for lbl, key, col_c in [("SPINE", "spine_dvtx_3d", SPINE_COLOR), ("Pandora", "pand_dvtx_3d", PANDORA_COLOR)]:
    arr = df[key].to_numpy()
    nbins_i = np.arange(0, 10, 1)
    med, _, _ = binned_statistic(nixn.astype(float), arr, "median", bins=nbins_i)
    cnt, _, _ = binned_statistic(nixn.astype(float), arr, "count", bins=nbins_i)
    ctr = 0.5 * (nbins_i[:-1] + nbins_i[1:])
    mk = cnt > 3
    axes8[0].errorbar(ctr[mk], med[mk], yerr=med[mk]/np.sqrt(cnt[mk]),
                      marker="o", ms=6, color=col_c, label=lbl)
axes8[0].set_xlabel("N DLP interactions"); axes8[0].set_ylabel("Median 3D vtx dist [cm]")
axes8[0].set_title("Vtx resolution vs N interactions"); axes8[0].legend(fontsize=9)

# Count distribution of interactions
axes8[1].hist(nixn, bins=np.arange(0, 10, 1), histtype="step", lw=2.0, color=SPINE_COLOR)
axes8[1].set_xlabel("N DLP interactions"); axes8[1].set_ylabel("Counts")
axes8[1].set_title("Distribution of number of interactions")

fig8.tight_layout()
fig8.savefig(os.path.join(outdir, "08_vertex_res_vs_ndlp.png"), dpi=150)
plt.close(fig8)

# ============================================================
# FIGURE 9: Energy estimator comparison (skip empty branches)
# ============================================================
print("--- Figure 9: Energy estimator workaround summary ---")
# Create a single-page text summary replacing the empty estimator panels
fig9, ax9 = plt.subplots(figsize=(12, 8))
ax9.axis("off")

# Check which estimators have data (for reporting)
estimator_status = []
for name, key in [
    ("Enu calo    ", "spine_Enu_calo"),
    ("lep_calo    ", "spine_Enu_lep_calo"),
    ("mu_range    ", "spine_Enu_mu_range"),
    ("mu_mcs      ", "spine_Enu_mu_mcs"),
    ("e_calo      ", "spine_Enu_e_calo"),
    ("mu_had      ", "spine_Enu_mu_had"),
    ("e_had       ", "spine_Enu_e_had"),
]:
    vals = df[key].to_numpy()
    valid = np.isfinite(vals) & (vals > 0)
    status = "POPULATED" if valid.sum() > 0 else "100% NaN"
    estimator_status.append((name, status, valid.sum()))

pand_calo_valid = np.isfinite(df["pand_Enu_calo"].to_numpy()) & (df["pand_Enu_calo"].to_numpy() > 0)
pand_med = np.median(df["pand_Enu_calo"].to_numpy()[pand_calo_valid] / E[pand_calo_valid]) if pand_calo_valid.sum() > 0 else 0

# Build text
lines = [
    "═" * 58,
    "  ENERGY ESTIMATOR WORKAROUND SUMMARY",
    "═" * 58,
    "",
    "  Problem:",
    "  All SPINE Enu estimator branches are 100% NaN",
    "  in this data production (6 estimators).",
    "",
    "  SPINE Estimator Status:",
]
for name, status, n in estimator_status:
    lines.append(f"    {name}   {status}")
lines += [
    "",
    "  Workaround:",
    "  • Use SPINE particle-sum hadronic energy",
    "    (summing KE of non-lepton primaries)",
    "  • Particle-sum median / truth = ~0.40",
    "  • Power-law correction fitted to recover",
    "    true hadronic energy",
    "",
    "  Pandora Enu calo (only working estimator):",
    f"    Populated: {pand_calo_valid.sum():,} / {len(E):,} events",
    f"    Median E_reco/E_true: {pand_med:.2f}",
    "",
    "  Recommendation:",
    "  Fix Enu estimator branches in future",
    "  SPINE data productions so that Enu_calo,",
    "  e_calo, mu_had, etc. are populated.",
]
ax9.text(0.05, 0.95, "\n".join(lines), transform=ax9.transAxes, fontsize=11,
         verticalalignment="top", family="monospace")
ax9.set_title("Energy Estimator Status & Workaround", fontsize=16, fontweight="bold")
fig9.tight_layout()
fig9.savefig(os.path.join(outdir, "09_energy_estimator_workaround.png"), dpi=150)
plt.close(fig9)
print("Saved 09_energy_estimator_workaround.png")

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 60)
print("Summary Tables")
print("=" * 60)

print("\n=== Vertex resolution summary ===")
print(f"{'':>16} {'68% [cm]':>10} {'90% [cm]':>10} {'median [cm]':>13}")
for lbl, key in [("SPINE 3D","spine_dvtx_3d"),("Pandora 3D","pand_dvtx_3d"),
                  ("SPINE X","spine_dvtx_x"),("Pandora X","pand_dvtx_x"),
                  ("SPINE Y","spine_dvtx_y"),("Pandora Y","pand_dvtx_y"),
                  ("SPINE Z","spine_dvtx_z"),("Pandora Z","pand_dvtx_z")]:
    arr = np.abs(df[key].to_numpy())
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        print(f"{lbl:>16} {'N/A':>10} {'N/A':>10} {'N/A':>13}")
    else:
        print(f"{lbl:>16} {np.quantile(arr,0.68):>10.2f} {np.quantile(arr,0.90):>10.2f} {np.median(arr):>13.2f}")

print("\n=== Angular resolution ===")
print(f"{'':>16} {'68% [deg]':>10} {'90% [deg]':>10} {'median [deg]':>13}")
for lbl, key in [("SPINE totmom","spine_dang_totmom_deg"),
                  ("Pandora best","pand_dang_best_deg")]:
    arr = df[key].to_numpy(); arr = arr[np.isfinite(arr) & (arr>=0)]
    if len(arr) == 0:
        print(f"{lbl:>16} {'N/A':>10} {'N/A':>10} {'N/A':>13}")
    else:
        print(f"{lbl:>16} {np.quantile(arr,0.68):>10.2f} {np.quantile(arr,0.90):>10.2f} {np.median(arr):>13.2f}")

print("\n=== Energy resolution (calo method) ===")
print(f"{'':>16} {'median':>10} {'IQR/2':>10} {'RMSE':>10}")
for lbl, key in [("SPINE","spine_Enu_calo"),("Pandora","pand_Enu_calo")]:
    arr = df[key].to_numpy()
    valid = np.isfinite(arr) & (arr > 0)
    ratio = arr[valid] / E[valid]
    print(f"{lbl:>16} {np.median(ratio):>10.3f} {iqr_half(ratio)(ratio):>10.3f} "
          f"{np.sqrt(np.mean((ratio-1)**2)):>10.3f}")

print("\n=== Multiplicity ===")
print(f"{'':>16} {'mean':>8} {'median':>8} {'max':>8}")
for lbl, key in [("SPINE","spine_nparts"),("Pandora","pand_nparts")]:
    arr = df[key].to_numpy()
    print(f"{lbl:>16} {np.mean(arr):>8.1f} {np.median(arr):>8.1f} {np.max(arr):>8.0f}")

print("\n=== Preselection rate ===")
for lbl, key in [("SPINE","spine_preselected"),("Pandora","pand_preselected")]:
    arr = df[key].to_numpy()
    print(f"  {lbl:>10}: {arr.mean()*100:.1f}% ({arr.sum()}/{len(arr)})")

print("\n=== Event counts by flavor ===")
flavor_counts = df["nu_flavor"].value_counts()
for row in flavor_counts.iter_rows():
    print(f"  {row[0]:>10}: {row[1]:>6}")

# Save summary as JSON
def safe_quantile(arr, q):
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) > 0 else None

s_ad = df["spine_dang_totmom_deg"].to_numpy()
p_ad = df["pand_dang_best_deg"].to_numpy()
summary = {
    "vertex_68pct": {
        "SPINE": safe_quantile(np.abs(df["spine_dvtx_3d"].to_numpy()), 0.68),
        "Pandora": safe_quantile(np.abs(df["pand_dvtx_3d"].to_numpy()), 0.68),
    },
    "vertex_90pct": {
        "SPINE": safe_quantile(np.abs(df["spine_dvtx_3d"].to_numpy()), 0.90),
        "Pandora": safe_quantile(np.abs(df["pand_dvtx_3d"].to_numpy()), 0.90),
    },
    "angular_68pct": {
        "SPINE": safe_quantile(s_ad[(np.isfinite(s_ad)) & (s_ad >= 0)], 0.68),
        "Pandora": safe_quantile(p_ad[(np.isfinite(p_ad)) & (p_ad >= 0)], 0.68),
    },
    "n_events": len(df),
    "flavor_counts": {str(row[0]): int(row[1]) for row in flavor_counts.iter_rows()},
}
with open(os.path.join(outdir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ============================================================
# FIGURE 18: Total SPINE momentum vs truth by flavor
# ============================================================
print("\n--- Figure 18: Total SPINE momentum vs truth by flavor ---")
fig18, axes18 = plt.subplots(2, 3, figsize=(18, 11))
fig18.suptitle(r"Total SPINE primary momentum vs truth by flavor", fontsize=14, fontweight="bold")

nu_pdg_arr = df["nu_pdg"].to_numpy()
total_tp = df["total_true_p"].to_numpy()
total_sp = df["total_spine_p"].to_numpy()
valid_total = (total_tp > 0) & (total_sp > 0) & np.isfinite(total_tp) & np.isfinite(total_sp)
ratio_all = total_sp[valid_total] / total_tp[valid_total]

flavor_config = [
    (12, "νe CC", "#1f77b4"), (-12, "ν̄e CC", "#ff7f0e"),
    (14, "νμ CC", "#2ca02c"), (-14, "ν̄μ CC", "#d62728"),
]

# 1: SPINE vs truth scatter (log-log) by flavor
ax = axes18[0, 0]
for pdg_v, lbl, col in flavor_config:
    mask = (nu_pdg_arr == pdg_v) & valid_total
    if mask.sum() < 10: continue
    ax.scatter(total_tp[mask], total_sp[mask], s=1, alpha=0.2, color=col, label=lbl, rasterized=True)
lims = [0.03, 300]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"Total truth $p$ [GeV/$c$]"); ax.set_ylabel(r"Total SPINE $p$ [GeV/$c$]")
ax.set_title("SPINE vs truth total momentum"); ax.legend(fontsize=7, markerscale=5, loc="upper left")

# 2: Ratio distribution by flavor
ax = axes18[0, 1]
rbins = np.logspace(np.log10(0.05), np.log10(30), 80)
for pdg_v, lbl, col in flavor_config:
    mask = (nu_pdg_arr == pdg_v) & valid_total
    if mask.sum() < 10: continue
    r = total_sp[mask] / total_tp[mask]
    ax.hist(r, bins=rbins, histtype="step", color=col, label=f"{lbl} (med={np.median(r):.2f})", lw=1.3, density=True)
ax.axvline(1, color="k", ls=":", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"SPINE total $p$ / truth total $p$"); ax.set_ylabel("Density")
ax.set_title("Total momentum ratio by flavor"); ax.legend(fontsize=7)

# 3: Ratio vs true momentum by flavor
ax = axes18[0, 2]
tp_bins = np.logspace(-1, 2, 20)
for pdg_v, lbl, col in flavor_config:
    mask = (nu_pdg_arr == pdg_v) & valid_total
    if mask.sum() < 10: continue
    r = total_sp[mask] / total_tp[mask]
    med, _, _ = binned_statistic(total_tp[mask], r, "median", bins=tp_bins)
    cnt, _, _ = binned_statistic(total_tp[mask], r, "count", bins=tp_bins)
    centers = 0.5 * (tp_bins[:-1] + tp_bins[1:])
    mk = cnt > 3
    ax.errorbar(centers[mk], med[mk], yerr=None,
                marker="o", ms=4, color=col, label=lbl, lw=1)
ax.axhline(1, color="k", ls=":", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"Total truth momentum [GeV/$c$]"); ax.set_ylabel("Median ratio")
ax.set_title("Ratio vs truth momentum"); ax.legend(fontsize=7)

# 4: Ratio distribution (all flavors combined, wider view)
ax = axes18[1, 0]
for pdg_v, lbl, col in flavor_config:
    mask = (nu_pdg_arr == pdg_v) & valid_total
    if mask.sum() < 10: continue
    r = total_sp[mask] / total_tp[mask]
    r_cut = r[(r > 0.01) & (r < 100)]
    ax.hist(np.log10(r_cut), bins=60, histtype="step", color=col,
            label=f"{lbl} 10%tile={np.percentile(r,10):.2f}  90%tile={np.percentile(r,90):.2f}", lw=1.3, density=True)
ax.axvline(0, color="k", ls=":", lw=0.8)
ax.text(0.02, 0.95, "Physical limit\n(EM calib. brings\nratio→1.0)", transform=ax.transAxes,
        fontsize=7, va="top", ha="left", color="gray", style="italic")
ax.set_xlabel(r"$\log_{10}$(SPINE total $p$ / truth total $p$)"); ax.set_ylabel("Density")
ax.set_title("Log ratio distribution"); ax.legend(fontsize=6.5)

# 5: Ratio vs nspine
ax = axes18[1, 1]
for pdg_v, lbl, col in flavor_config:
    mask = (nu_pdg_arr == pdg_v) & valid_total
    if mask.sum() < 10: continue
    nsp = df["spine_nparts"].to_numpy()[mask].astype(float)
    r = total_sp[mask] / total_tp[mask]
    n_bins = np.arange(0, 15, 1)
    med, _, _ = binned_statistic(nsp, r, "median", bins=n_bins)
    cnt, _, _ = binned_statistic(nsp, r, "count", bins=n_bins)
    ctr = 0.5 * (n_bins[:-1] + n_bins[1:])
    mk = cnt > 3
    ax.errorbar(ctr[mk], med[mk], yerr=med[mk]/np.sqrt(cnt[mk]),
                marker="o", ms=4, color=col, label=lbl, lw=1)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xlabel("N SPINE primaries"); ax.set_ylabel("Median ratio")
    ax.set_title("Ratio vs SPINE multiplicity"); ax.legend(fontsize=7)

# 6: Total true and SPINE p histogram
ax = axes18[1, 2]
tp_bins_h = np.logspace(np.log10(0.05), np.log10(300), 60)
for pdg_v, lbl, col in flavor_config:
    mask = (nu_pdg_arr == pdg_v) & valid_total
    if mask.sum() < 10: continue
    ax.hist(total_tp[mask], bins=tp_bins_h, histtype="step", color=col,
            label=f"{lbl} true", lw=1, ls="-", density=True)
    ax.hist(total_sp[mask], bins=tp_bins_h, histtype="step", color=col,
            label=f"{lbl} SPINE", lw=1, ls="--", density=True)
ax.set_xscale("log"); ax.set_xlabel(r"Momentum [GeV/$c$]"); ax.set_ylabel("Density")
ax.set_title(r"True vs SPINE $p$ distribution"); ax.legend(fontsize=6)

fig18.tight_layout()
fig18.savefig(os.path.join(outdir, "18_total_momentum_by_flavor.png"), dpi=150)
plt.close(fig18)
print("Saved 18_total_momentum_by_flavor.png")

# ============================================================
# FIGURE 19: Per-particle performance summary
# ============================================================
print("--- Figure 19: Per-particle performance summary ---")
pp_path = os.path.join(outdir, "per_particle_summary.json")
if os.path.exists(pp_path):
    with open(pp_path) as f:
        pp_data = json.load(f)
    fig19, axes19 = plt.subplots(2, 3, figsize=(16, 10))
    fig19.suptitle("Per-particle performance summary (SPINE)", fontsize=14, fontweight="bold")

    ptypes = ["e", "mu", "gamma", "pi0", "pi", "K", "p", "n", "other"]
    ptypes_present = [p for p in ptypes if pp_data.get(f"spine_{p}_N", 0) >= 5]
    x = np.arange(len(ptypes_present))

    # 1: Momentum ratio by type
    ax = axes19[0, 0]
    prats = [pp_data.get(f"spine_{p}_p_ratio_median", 0) for p in ptypes_present]
    colors = ["#e41a1c" if p == "e" else "#377eb8" if p == "mu" else "#4daf4a" if p == "p"
              else "#984ea3" if p == "n" else "#ff7f00" if p == "gamma" else "#a65628"
              for p in ptypes_present]
    ax.bar(x, prats, color=colors, alpha=0.8)
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ptypes_present)
    ax.set_ylabel("p_reco / p_true"); ax.set_title("Momentum ratio by type")

    # 2: Angular resolution by type
    ax = axes19[0, 1]
    angs = [pp_data.get(f"spine_{p}_ang_median_deg", 0) for p in ptypes_present]
    iqrs = [pp_data.get(f"spine_{p}_ang_iqr_deg", 0) for p in ptypes_present]
    ax.bar(x, angs, color=colors, alpha=0.8, yerr=iqrs, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(ptypes_present)
    ax.set_ylabel("Median Δθ [deg]"); ax.set_title("Angular resolution by type")

    # 3: N matched by type
    ax = axes19[0, 2]
    ns = [pp_data.get(f"spine_{p}_N", 0) for p in ptypes_present]
    ax.bar(x, ns, color=colors, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ptypes_present)
    ax.set_ylabel("N matched pairs"); ax.set_title("Sample size by type")

    # 4: Completeness & purity
    ax = axes19[1, 0]
    metrics = ["Completeness", "Purity"]
    vals = [pp_data.get("completeness_mean", 0) * 100, pp_data.get("purity_mean", 0) * 100]
    ax.bar(metrics, vals, color=["#2ca02c", "#1f77b4"], alpha=0.8)
    ax.set_ylabel("Percent [%]"); ax.set_ylim(0, 100)
    ax.set_title(f"Mean completeness & purity\nGhosts/ev: {pp_data.get('ghosts_mean', 0):.2f}, Splits/ev: {pp_data.get('split_mean', 0):.2f}")

    # 5: p_ratio vs angular resolution scatter (linear x to avoid log(0) issues)
    ax = axes19[1, 1]
    ax.scatter(angs, prats, c=colors, s=80, alpha=0.8)
    for i, p in enumerate(ptypes_present):
        ax.annotate(p, (angs[i], prats[i]), fontsize=9, ha="center", va="bottom")
    ax.set_xlabel("Median angular resolution [deg]"); ax.set_ylabel("p_reco / p_true")
    ax.set_title("p_ratio vs angular resolution"); ax.axhline(1, color="k", ls=":", lw=0.5)

    # 6: Text summary of key findings
    ax = axes19[1, 2]
    ax.axis("off")
    n_matched = pp_data.get("spine_n_matched_pairs", 0)
    n_ev = pp_data.get("n_events", 0)
    text = (
        f"Key findings:\n"
        f"• {n_matched:,} matched pairs\n"
        f"  in {n_ev} events\n\n"
        f"• e⁻: p ×{pp_data.get('e_p_ratio_median',0):.2f}\n"
        f"  (after ×0.198 calibration)\n\n"
        f"• μ: p ×{pp_data.get('mu_p_ratio_median',0):.2f}\n"
        f"  (well-reconstructed)\n\n"
        f"• p: p ×{pp_data.get('p_p_ratio_median',0):.2f}\n"
        f"  but PID 71% → π mis-ID\n\n"
        f"• Hadronic system:\n"
        f"  ~40% of truth in both νe, νμ\n\n"
        f"• SPINE total p / truth:\n"
        f"  νe: 0.59×  ν̄e: 0.49×\n"
        f"  νμ: 0.51×  ν̄μ: 0.38×"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", family="monospace")

    fig19.tight_layout()
    fig19.savefig(os.path.join(outdir, "19_per_particle_summary.png"), dpi=150)
    plt.close(fig19)
    print("Saved 19_per_particle_summary.png")

# ============================================================
# Per-particle summary text (from analysis_per_particle.py)
# ============================================================
pp_path = os.path.join(outdir, "per_particle_summary.json")
if os.path.exists(pp_path):
    with open(pp_path) as f:
        pp = json.load(f)
    print("\n" + "=" * 60)
    print("Per-particle performance (SPINE, 5000 events)")
    print("=" * 60)
    print(f"  Matched pairs: {pp.get('spine_n_matched_pairs', 'N/A'):,}")
    print(f"  Completeness: {pp.get('completeness_mean', 0)*100:.1f}% (mean)")
    print(f"  Purity:       {pp.get('purity_mean', 0)*100:.1f}% (mean)")
    print(f"  Ghosts/event: {pp.get('ghosts_mean', 0):.2f}")
    print(f"  Splits/event: {pp.get('split_mean', 0):.2f}")
    ptypes = ["e", "mu", "gamma", "pi0", "pi", "K", "p", "n", "other"]
    print(f"\n  {'type':>6} {'N':>6} {'Δθ med':>8} {'Δθ IQR/2':>10} {'p_reco/p_true':>13}")
    for pt in ptypes:
        n = pp.get(f"spine_{pt}_N", 0)
        if n < 5: continue
        ang  = pp.get(f"spine_{pt}_ang_median_deg", 0)
        iqr  = pp.get(f"spine_{pt}_ang_iqr_deg", 0)
        prat = pp.get(f"spine_{pt}_p_ratio_median", 0)
        print(f"  {pt:>6} {n:>6} {ang:>8.1f}° {iqr:>10.2f}° {prat:>13.2f}")
    summary["per_particle"] = pp
    print(f"\n  → Key finding: Electron momentum over-estimated ×{pp.get('spine_e_p_ratio_median', 0):.1f}")
    print(f"  → Key finding: Muon momentum well-reconstructed ×{pp.get('spine_mu_p_ratio_median', 0):.2f}")
    print(f"  → Key finding: SPINE p.E stores kinetic energy (converted to total E in analysis)")
    # Flavor-dependent total momentum ratio
    for pdg_val, lbl in [(12, "νe"), (-12, "ν̄e"), (14, "νμ"), (-14, "ν̄μ")]:
        mask = df["nu_pdg"].to_numpy() == pdg_val
        if mask.sum() < 10: continue
        total_tp = df["total_true_p"].to_numpy()[mask]
        total_sp = df["total_spine_p"].to_numpy()[mask]
        valid = (total_tp > 0) & (total_sp > 0) & np.isfinite(total_tp) & np.isfinite(total_sp)
        if valid.sum() < 10: continue
        ratio = total_sp[valid] / total_tp[valid]
        med_r = float(np.median(ratio))
        print(f"    {lbl:>4} CC (N={mask.sum():>5}): total SPINE/truth = {med_r:.2f} ×")
        summary.setdefault("total_p_by_flavor", {})[f"nu_{pdg_val}"] = {
            "n_events": int(mask.sum()), "median_ratio": med_r,
        }

    print(f"\n    → Hadronic system under-estimated to ~40% of truth in both flavors")
    print(f"    → Shower cone analysis: bremsstrahlung adds only ~4% to momentum in 10° cone")
    print(f"    → Electron ×4.8 is NOT from shower merging → genuine EM calibration bias in SPINE")

# Save updated summary
with open(os.path.join(outdir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll figures saved to {outdir}/")
print("Done!")

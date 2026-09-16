#!/usr/bin/env python3
"""
Investigate SPINE vertex resolution degradation at detector boundaries.

Studies:
1. Vertex resolution vs proximity to detector edges (fiducial distance)
2. Per-axis resolution profile vs position in each coordinate
3. Comparison of near-boundary vs central regions
4. Asymmetry between cathode/anode (Y) and upstream/downstream (Z)

Output: figures/vertex_boundary_*.png
"""

import os, sys, ctypes, math, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from tqdm import tqdm

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

import polars as pl

# Detector bounds (ArgonCube TPC)
DET_X_MIN, DET_X_MAX = -360, 360
DET_Y_MIN, DET_Y_MAX = -600, 600
DET_Z_MIN, DET_Z_MAX = 0, 1400

def _safe_norm(x, y, z):
    n = math.sqrt(x*x + y*y + z*z)
    return (x/n, y/n, z/n) if n > 0 else (float("nan"),)*3

def _closest_dlp_idx(ixn, nu_vtx):
    best, best_i = float("inf"), 0
    nvx, nvy, nvz = nu_vtx.x, nu_vtx.y, nu_vtx.z
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = math.sqrt((v.x-nvx)**2 + (v.y-nvy)**2 + (v.z-nvz)**2)
        if d < best:
            best, best_i = d, i
    return best_i

outdir = "figures"
os.makedirs(outdir, exist_ok=True)

import plot_style
plot_style.apply()

# Load data
print("=" * 60)
print("Loading CAF data for vertex boundary study")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader('recTree', 'spine_merged_reprocess.root')

cols = {k: [] for k in [
    "true_vtx_x","true_vtx_y","true_vtx_z",
    "spine_vtx_x","spine_vtx_y","spine_vtx_z",
    "pand_vtx_x","pand_vtx_y","pand_vtx_z",
    "nu_E","nu_pdg",
]}

_app = {k: cols[k].append for k in cols}

for ientry in tqdm(range(prox.entries()), desc="Loading"):
    rec = prox.entry(ientry)
    mc = rec.mc
    if not mc.nu[0].iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1 or ixn.npandora < 1:
        continue

    nu = mc.nu[0]
    vtx = nu.vtx

    _app["true_vtx_x"](vtx.x); _app["true_vtx_y"](vtx.y); _app["true_vtx_z"](vtx.z)
    _app["nu_E"](nu.E); _app["nu_pdg"](int(nu.pdg))

    si = _closest_dlp_idx(ixn, vtx)
    sv = ixn.dlp[si].vtx
    _app["spine_vtx_x"](sv.x); _app["spine_vtx_y"](sv.y); _app["spine_vtx_z"](sv.z)

    pv = ixn.pandora[0].vtx
    _app["pand_vtx_x"](pv.x); _app["pand_vtx_y"](pv.y); _app["pand_vtx_z"](pv.z)

raw = {k: np.array(v, dtype=np.float32) for k, v in cols.items()}
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
        ((pl.col("spine_dvtx_x")**2 + pl.col("spine_dvtx_y")**2 +
          pl.col("spine_dvtx_z")**2).sqrt()).alias("spine_dvtx_3d"),
        ((pl.col("pand_dvtx_x")**2  + pl.col("pand_dvtx_y")**2 +
          pl.col("pand_dvtx_z")**2 ).sqrt()).alias("pand_dvtx_3d"),
    ])
    .with_columns([
        # Distance to nearest boundary in each axis
        (pl.col("true_vtx_x") - DET_X_MIN).alias("dist_x_min"),
        (DET_X_MAX - pl.col("true_vtx_x")).alias("dist_x_max"),
        (pl.col("true_vtx_y") - DET_Y_MIN).alias("dist_y_min"),
        (DET_Y_MAX - pl.col("true_vtx_y")).alias("dist_y_max"),
        (pl.col("true_vtx_z") - DET_Z_MIN).alias("dist_z_min"),
        (DET_Z_MAX - pl.col("true_vtx_z")).alias("dist_z_max"),
    ])
    .with_columns([
        pl.min_horizontal(["dist_x_min", "dist_x_max"]).alias("dist_to_x_edge"),
        pl.min_horizontal(["dist_y_min", "dist_y_max"]).alias("dist_to_y_edge"),
        pl.min_horizontal(["dist_z_min", "dist_z_max"]).alias("dist_to_z_edge"),
    ])
    .with_columns(
        pl.min_horizontal(["dist_to_x_edge", "dist_to_y_edge", "dist_to_z_edge"]
                          ).alias("dist_to_nearest_edge")
    )
)

print(f"  Events loaded: {len(df):,}")

E = df["nu_E"].to_numpy()

# ---------------------------------------------------------------
# Figure 1: 3D vertex resolution vs distance to nearest edge
# ---------------------------------------------------------------
print("\n--- Figure 1: Resolution vs distance to detector edge ---")
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
fig1.suptitle("Vertex resolution vs proximity to detector boundaries",
              fontsize=14, fontweight="bold")

edge_labels = [
    ("dist_to_x_edge", "Distance to X edge [cm]"),
    ("dist_to_y_edge", "Distance to Y edge [cm]"),
    ("dist_to_z_edge", "Distance to Z edge [cm]"),
]

for idx, (edge_col, xlabel) in enumerate(edge_labels):
    ax = axes1[idx // 2, idx % 2]
    edge_arr = df[edge_col].to_numpy()
    for lbl, key, col_c in [("SPINE","spine_dvtx_3d","C0"), ("Pandora","pand_dvtx_3d","C1")]:
        res_arr = df[key].to_numpy()
        bins = np.linspace(0, 400, 20)
        med, _, _ = binned_statistic(edge_arr, res_arr, "median", bins=bins)
        cnt, _, _ = binned_statistic(edge_arr, res_arr, "count", bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        mk = cnt > 3
        ax.errorbar(centers[mk], med[mk], yerr=None,
                    marker="o", ms=5, color=col_c, label=lbl)
    ax.set_xlabel(xlabel); ax.set_ylabel("Median 3D vtx dist [cm]")
    ax.set_title(f"3D resolution vs {edge_col}")
    ax.legend(fontsize=8)

# Nearest edge (combined)
ax = axes1[1, 1]
edge_all = df["dist_to_nearest_edge"].to_numpy()
for lbl, key, col_c in [("SPINE","spine_dvtx_3d","C0"), ("Pandora","pand_dvtx_3d","C1")]:
    res_arr = df[key].to_numpy()
    bins = np.linspace(0, 300, 20)
    med, _, _ = binned_statistic(edge_all, res_arr, "median", bins=bins)
    cnt, _, _ = binned_statistic(edge_all, res_arr, "count", bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mk = cnt > 3
    ax.errorbar(centers[mk], med[mk], yerr=None,
                marker="o", ms=5, color=col_c, label=lbl)
ax.set_xlabel("Distance to nearest detector edge [cm]")
ax.set_ylabel("Median 3D vtx dist [cm]")
ax.set_title("3D resolution vs nearest edge distance")
ax.legend(fontsize=8)

fig1.tight_layout()
fig1.savefig(os.path.join(outdir, "vertex_boundary_vs_edge_distance.png"), dpi=150)
plt.close(fig1)

# ---------------------------------------------------------------
# Figure 2: Per-axis residuals vs position (cathode/anode analysis)
# ---------------------------------------------------------------
print("\n--- Figure 2: Per-axis resolution vs position ---")
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle("Per-axis vertex residual vs true position", fontsize=14, fontweight="bold")

axis_config = [
    ("x", np.linspace(-380, 380, 30), [-360, 360],
     ["Cathode", "Anode"]),
    ("y", np.linspace(-650, 650, 30), [-600, 600],
     ["Bottom", "Top"]),
    ("z", np.linspace(-50, 1450, 30), [0, 1400],
     ["Upstream", "Downstream"]),
]

for col_i, (axis_lbl, pos_bins, det_edges, edge_names) in enumerate(axis_config):
    pos_arr = df[f"true_vtx_{axis_lbl}"].to_numpy()

    # Row 0: Median abs residual vs position
    ax = axes2[0, col_i]
    for lbl, key_pfx, col_c in [("SPINE","spine","C0"), ("Pandora","pand","C1")]:
        res_arr = np.abs(df[f"{key_pfx}_dvtx_{axis_lbl}"].to_numpy())
        med, _, _ = binned_statistic(pos_arr, res_arr, "median", bins=pos_bins)
        cnt, _, _ = binned_statistic(pos_arr, res_arr, "count", bins=pos_bins)
        centers = 0.5 * (pos_bins[:-1] + pos_bins[1:])
        mk = cnt > 3
        ax.plot(centers[mk], med[mk], "o-", ms=4, color=col_c, label=lbl)
    for edge in det_edges:
        ax.axvline(edge, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(f"True {axis_lbl.upper()} [cm]")
    ax.set_ylabel(f"Median |{axis_lbl.upper()} residual| [cm]")
    ax.set_title(f"{axis_lbl.upper()} median abs residual vs position")
    ax.legend(fontsize=8)

    # Row 1: Signed residual (bias) vs position
    ax = axes2[1, col_i]
    for lbl, key_pfx, col_c in [("SPINE","spine","C0"), ("Pandora","pand","C1")]:
        res_arr = df[f"{key_pfx}_dvtx_{axis_lbl}"].to_numpy()
        med, _, _ = binned_statistic(pos_arr, res_arr, "median", bins=pos_bins)
        cnt, _, _ = binned_statistic(pos_arr, res_arr, "count", bins=pos_bins)
        centers = 0.5 * (pos_bins[:-1] + pos_bins[1:])
        mk = cnt > 3
        yerr_val = np.abs(med[mk]) / np.sqrt(cnt[mk]) * 1.5
        ax.errorbar(centers[mk], med[mk], yerr=yerr_val,
                    marker="o", ms=4, color=col_c, label=lbl, capsize=2)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    for edge in det_edges:
        ax.axvline(edge, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(f"True {axis_lbl.upper()} [cm]")
    ax.set_ylabel(f"Median {axis_lbl.upper()} residual [cm]")
    ax.set_title(f"{axis_lbl.upper()} signed residual (bias) vs position")
    ax.legend(fontsize=8)

fig2.tight_layout()
fig2.savefig(os.path.join(outdir, "vertex_boundary_per_axis_position.png"), dpi=150)
plt.close(fig2)

# ---------------------------------------------------------------
# Figure 3: Boundary region comparison (near vs far from edges)
# ---------------------------------------------------------------
print("\n--- Figure 3: Near-boundary vs central region comparison ---")
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
fig3.suptitle("Vertex resolution: boundary vs central detector regions",
              fontsize=14, fontweight="bold")

# Define regions
fid_margin = 30  # cm
near_x = (np.abs(df["true_vtx_x"].to_numpy()) > (DET_X_MAX - fid_margin))
near_y = (np.abs(df["true_vtx_y"].to_numpy()) > (DET_Y_MAX - fid_margin))
near_z = ((df["true_vtx_z"].to_numpy() < (DET_Z_MIN + fid_margin)) |
          (df["true_vtx_z"].to_numpy() > (DET_Z_MAX - fid_margin)))
near_any = near_x | near_y | near_z
far = ~near_any

print(f"  Near boundary: {near_any.sum()} events, Central (far): {far.sum()} events")

# CDF comparison
ax = axes3[0, 0]
for lbl, key, col_c in [("SPINE boundary","spine_dvtx_3d","C0"),
                          ("SPINE central","spine_dvtx_3d","C1"),
                          ("Pandora boundary","pand_dvtx_3d","C2"),
                          ("Pandora central","pand_dvtx_3d","C3")]:
    is_near = "boundary" in lbl
    is_spine = "SPINE" in lbl
    mask = (near_any if is_near else far)
    key_v = key
    arr = df[key_v].to_numpy()[mask]
    s_arr = np.sort(np.abs(arr[np.isfinite(arr)]))
    cdf = np.arange(1, len(s_arr)+1)/len(s_arr)
    ax.plot(s_arr, cdf, label=lbl, color=col_c, lw=1.3)
    for q, ls in [(0.68,"--"), (0.90,":")]:
        v = np.quantile(s_arr, q)
        ax.axvline(v, color=col_c, ls=ls, lw=1, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("|3D vertex offset| [cm]")
ax.set_ylabel("CDF")
ax.set_title("CDF: boundary vs central regions")
ax.legend(fontsize=7)

# 3D resolution by axis - which axis boundary causes worst degradation
ax = axes3[0, 1]
regions = [
    ("All events", slice(None)),
    ("Near X edge", near_x),
    ("Near Y edge", near_y),
    ("Near Z edge", near_z),
    ("Central (all far)", far),
]
x_pos = np.arange(len(regions))
width = 0.35
for idx, (lbl, key, col_c) in enumerate([("SPINE","spine_dvtx_3d","C0"),
                                           ("Pandora","pand_dvtx_3d","C1")]):
    vals = []
    for reg_lbl, mask in regions:
        if isinstance(mask, slice):
            arr = df[key].to_numpy()
        else:
            arr = df[key].to_numpy()[mask]
        vals.append(np.nanmedian(arr))
    ax.bar(x_pos + (idx-0.5)*width, vals, width, color=col_c, label=lbl)
ax.set_xticks(x_pos)
ax.set_xticklabels([r[0] for r in regions], fontsize=8)
ax.set_ylabel("Median 3D vtx dist [cm]")
ax.set_title("Resolution by boundary region")
ax.legend(fontsize=8)

# Per-axis 68% quantile by region
metrics = [
    ("spine_dvtx_x", "SPINE X"),
    ("spine_dvtx_y", "SPINE Y"),
    ("spine_dvtx_z", "SPINE Z"),
    ("pand_dvtx_x", "Pandora X"),
    ("pand_dvtx_y", "Pandora Y"),
    ("pand_dvtx_z", "Pandora Z"),
]

ax = axes3[1, 0]
x_pos2 = np.arange(len(regions))
width2 = 0.12
for idx, (key, lbl) in enumerate(metrics):
    vals = []
    for reg_lbl, mask in regions:
        if isinstance(mask, slice):
            arr = np.abs(df[key].to_numpy())
        else:
            arr = np.abs(df[key].to_numpy()[mask])
        vals.append(np.nanquantile(arr, 0.68))
    ax.bar(x_pos2 + (idx - len(metrics)/2 + 0.5)*width2, vals, width2, label=lbl)
ax.set_xticks(x_pos2)
ax.set_xticklabels([r[0] for r in regions], fontsize=7, rotation=15)
ax.set_ylabel("68% quantile [cm]")
ax.set_title("Per-axis 68% residual by region")
ax.legend(fontsize=6)

# Summary table in text
ax = axes3[1, 1]
ax.axis("off")
text_lines = ["Vertex resolution by region (68% quantile [cm]):"]
text_lines.append(f"{'Region':>20} {'SPINE X':>8} {'SPINE Y':>8} {'SPINE Z':>8}"
                  f"  {'Pand X':>8} {'Pand Y':>8} {'Pand Z':>8}")
for reg_lbl, mask in regions:
    s_x = np.nanquantile(np.abs(df["spine_dvtx_x"].to_numpy()), 0.68)
    s_y = np.nanquantile(np.abs(df["spine_dvtx_y"].to_numpy()), 0.68)
    s_z = np.nanquantile(np.abs(df["spine_dvtx_z"].to_numpy()), 0.68)
    p_x = np.nanquantile(np.abs(df["pand_dvtx_x"].to_numpy()), 0.68)
    p_y = np.nanquantile(np.abs(df["pand_dvtx_y"].to_numpy()), 0.68)
    p_z = np.nanquantile(np.abs(df["pand_dvtx_z"].to_numpy()), 0.68)
    if not isinstance(mask, slice):
        s_x = np.nanquantile(np.abs(df["spine_dvtx_x"].to_numpy()[mask]), 0.68)
        s_y = np.nanquantile(np.abs(df["spine_dvtx_y"].to_numpy()[mask]), 0.68)
        s_z = np.nanquantile(np.abs(df["spine_dvtx_z"].to_numpy()[mask]), 0.68)
        p_x = np.nanquantile(np.abs(df["pand_dvtx_x"].to_numpy()[mask]), 0.68)
        p_y = np.nanquantile(np.abs(df["pand_dvtx_y"].to_numpy()[mask]), 0.68)
        p_z = np.nanquantile(np.abs(df["pand_dvtx_z"].to_numpy()[mask]), 0.68)
    text_lines.append(f"{reg_lbl:>20} {s_x:>8.2f} {s_y:>8.2f} {s_z:>8.2f}"
                      f"  {p_x:>8.2f} {p_y:>8.2f} {p_z:>8.2f}")
ax.text(0.05, 0.95, "\n".join(text_lines), transform=ax.transAxes,
        fontsize=8, verticalalignment="top", family="monospace")

fig3.tight_layout()
fig3.savefig(os.path.join(outdir, "vertex_boundary_regions.png"), dpi=150)
plt.close(fig3)

# ---------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Vertex Boundary Degradation Summary")
print("=" * 60)
print(f"\nDetector bounds: X [{DET_X_MIN}, {DET_X_MAX}], "
      f"Y [{DET_Y_MIN}, {DET_Y_MAX}], Z [{DET_Z_MIN}, {DET_Z_MAX}]")
print(f"Fiducial margin: {fid_margin} cm")
print(f"Boundary events: {near_any.sum()}/{len(df)} ({near_any.sum()/len(df)*100:.1f}%)")
print(f"Central events:  {far.sum()}/{len(df)} ({far.sum()/len(df)*100:.1f}%)")

# Global medians
for lbl, key in [("SPINE 3D","spine_dvtx_3d"), ("Pandora 3D","pand_dvtx_3d")]:
    arr = df[key].to_numpy()
    print(f"\n{lbl}:")
    for reg_lbl, mask in [("All", slice(None)), ("Boundary", near_any), ("Central", far)]:
        if isinstance(mask, slice):
            a = arr
        else:
            a = arr[mask]
        print(f"  {reg_lbl:>10}: median={np.nanmedian(a):.2f} cm, "
              f"68%={np.nanquantile(np.abs(a), 0.68):.2f} cm, "
              f"N={len(a):,}")

print("\nDone!")

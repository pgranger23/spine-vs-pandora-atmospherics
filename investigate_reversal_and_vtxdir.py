#!/usr/bin/env python3
"""
Two targeted diagnostics on CC neutrino direction reconstruction:

1. Reversal rate: fraction of events where the estimated direction is
   flipped (θ > 90°) relative to the true neutrino direction.
   Computed for SPINE all-sum and Pandora lngtrk/heshw vs E_ν.

2. Vertex–direction correlation: within fixed E_ν slices, does a better
   vertex resolution (smaller Δr_vtx) correlate with better direction?
   Shown as 2D binned-median plots and Pearson correlations.

CC events, full sample.
Output: figures/reversal_and_vtxdir.png, figures/reversal_vtxdir.json
"""

import os, sys, ctypes, math, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, pearsonr
from tqdm import tqdm

base_dir = os.getcwd()
lib_path = os.path.join(base_dir, "duneanaobj/build/Linux/lib")
py_path  = os.path.join(base_dir, "duneanaobj/build/Linux/python/3.13")
if py_path not in sys.path:
    sys.path.insert(0, py_path)

for _lib in ["libSRProxy_BasicTypes.dylib", "libduneanaobj_StandardRecord.dylib",
             "libduneanaobj_StandardRecordProxy.dylib"]:
    try:
        ctypes.CDLL(os.path.join(lib_path, _lib))
    except OSError:
        pass

import pySRProxy
import plot_style
plot_style.apply()

outdir = "figures"
os.makedirs(outdir, exist_ok=True)
_nan = float("nan")

E_EDGES  = [0.0, 0.5, 1.0, 2.0, 1e9]
E_LABELS = ["E<0.5 GeV\n(QE-like)", "0.5–1 GeV\n(RES-like)",
             "1–2 GeV\n(MEC/DIS)", ">2 GeV\n(DIS-like)"]
E_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

def _e_bin(e):
    for i in range(len(E_EDGES) - 1):
        if E_EDGES[i] <= e < E_EDGES[i + 1]:
            return i
    return len(E_EDGES) - 2

def _vmag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def _cos(ax, ay, az, bx, by, bz):
    na = _vmag(ax, ay, az)
    nb = _vmag(bx, by, bz)
    if not (math.isfinite(na) and na > 1e-9) or not (math.isfinite(nb) and nb > 1e-9):
        return _nan
    return max(-1.0, min(1.0, (ax*bx + ay*by + az*bz) / (na * nb)))

def _ang(c):
    return math.degrees(math.acos(c)) if math.isfinite(c) else _nan

def _closest(items, n, vtx):
    best, bi = float("inf"), 0
    for i in range(n):
        v = items[i].vtx
        d = _vmag(v.x - vtx.x, v.y - vtx.y, v.z - vtx.z)
        if d < best:
            best, bi = d, i
    return bi

# ---------------------------------------------------------------
print("=" * 60)
print("Reversal rate + vertex–direction correlation")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader("recTree", "spine_merged_reprocess.root")

records = []

for ientry in tqdm(range(prox.entries()), desc="Scanning"):
    rec = prox.entry(ientry)
    mc  = rec.mc
    if mc.nnu < 1 or not mc.nu[0].iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu    = mc.nu[0]
    nu_E  = float(nu.E)
    nu_pdg = int(nu.pdg)
    tvx, tvy, tvz = nu.vtx.x, nu.vtx.y, nu.vtx.z

    tm = nu.momentum
    tm_mag = _vmag(tm.x, tm.y, tm.z)
    if not (math.isfinite(tm_mag) and tm_mag > 1e-9):
        continue
    tdx, tdy, tdz = tm.x/tm_mag, tm.y/tm_mag, tm.z/tm_mag

    # SPINE
    si  = _closest(ixn.dlp, ixn.ndlp, nu.vtx)
    dlp = ixn.dlp[si]
    sv  = dlp.vtx
    spine_dvtx = _vmag(sv.x-tvx, sv.y-tvy, sv.z-tvz) * 0.1  # mm→cm

    spx = spy = spz = 0.0
    for pi in range(dlp.part.ndlp):
        _p = dlp.part.dlp[pi]
        pp = _p.p
        # EM particles (e, γ, π⁰): apply ADC-to-electron calibration (×0.198)
        _sc = 0.198 if abs(int(_p.pdg)) in (11, 22, 111) else 1.0
        spx += pp.x * _sc; spy += pp.y * _sc; spz += pp.z * _sc
    spine_cos = _cos(spx, spy, spz, tdx, tdy, tdz)
    spine_ang = _ang(spine_cos)

    # Pandora
    pand_dvtx = _nan
    pand_ang  = _nan
    if ixn.npandora > 0:
        pi2 = _closest(ixn.pandora, ixn.npandora, nu.vtx)
        pan = ixn.pandora[pi2]
        pv  = pan.vtx
        pand_dvtx = _vmag(pv.x-tvx, pv.y-tvy, pv.z-tvz) * 0.1
        pd = pan.dir
        if abs(nu_pdg) == 14:
            px2, py2, pz2 = pd.lngtrk.x, pd.lngtrk.y, pd.lngtrk.z
        else:
            px2, py2, pz2 = pd.heshw.x, pd.heshw.y, pd.heshw.z
        pand_ang = _ang(_cos(px2, py2, pz2, tdx, tdy, tdz))

    records.append({
        "nu_E": nu_E, "e_bin": _e_bin(nu_E),
        "spine_ang": spine_ang, "spine_dvtx": spine_dvtx,
        "pand_ang":  pand_ang,  "pand_dvtx":  pand_dvtx,
        "spine_cos": spine_cos,
    })

print(f"  CC events: {len(records):,}")

nu_E   = np.array([r["nu_E"]    for r in records])
e_bin  = np.array([r["e_bin"]   for r in records], dtype=int)
s_ang  = np.array([r["spine_ang"]  for r in records])
p_ang  = np.array([r["pand_ang"]   for r in records])
s_dvtx = np.array([r["spine_dvtx"] for r in records])
p_dvtx = np.array([r["pand_dvtx"]  for r in records])

# ---------------------------------------------------------------
# 1. Reversal rate
# ---------------------------------------------------------------
print("\n--- Reversal rate (θ > 90°) ---")

E_BINS_FINE = np.linspace(0, 4.0, 25)
centers_E   = 0.5 * (E_BINS_FINE[:-1] + E_BINS_FINE[1:])

def _rev_rate(ang_arr, e_arr, e_bins):
    is_valid = np.isfinite(ang_arr) & np.isfinite(e_arr) & (e_arr < 4.0)
    is_rev   = is_valid & (ang_arr > 90.0)
    rate, _, _ = binned_statistic(e_arr[is_valid], is_rev[is_valid].astype(float),
                                  "mean", bins=e_bins)
    cnt,  _, _ = binned_statistic(e_arr[is_valid], e_arr[is_valid], "count", bins=e_bins)
    return rate, cnt

sr, sc = _rev_rate(s_ang, nu_E, E_BINS_FINE)
pr, pc = _rev_rate(p_ang, nu_E, E_BINS_FINE)

print("Reversal rates per E_ν bin:")
for bi, lb in enumerate(E_LABELS):
    bm_s = (e_bin == bi) & np.isfinite(s_ang)
    bm_p = (e_bin == bi) & np.isfinite(p_ang)
    rs   = float((s_ang[bm_s] > 90).mean()) if bm_s.sum() > 0 else _nan
    rp   = float((p_ang[bm_p] > 90).mean()) if bm_p.sum() > 0 else _nan
    print(f"  {lb.split(chr(10))[0]:14s}: SPINE={rs:.3f} ({rs*100:.1f}%)  "
          f"Pandora={rp:.3f} ({rp*100:.1f}%)  N={bm_s.sum():,}")

# ---------------------------------------------------------------
# 2. Vertex–direction correlation per E_ν slice
# ---------------------------------------------------------------
print("\n--- Vertex–direction Pearson correlation per E_ν slice ---")

VTX_BINS = np.linspace(0, 10, 20)  # cm

corr_summary = {}
for bi, (lb, col) in enumerate(zip(E_LABELS, E_COLORS)):
    bm  = e_bin == bi
    bms = bm & np.isfinite(s_ang) & np.isfinite(s_dvtx)
    bmp = bm & np.isfinite(p_ang) & np.isfinite(p_dvtx)
    rs = pearsonr(s_dvtx[bms], s_ang[bms])[0]  if bms.sum() > 10 else _nan
    rp = pearsonr(p_dvtx[bmp], p_ang[bmp])[0]  if bmp.sum() > 10 else _nan
    print(f"  {lb.split(chr(10))[0]:14s}: SPINE r={rs:.3f}  Pandora r={rp:.3f}")
    corr_summary[lb.replace('\n', ' ')] = {"spine_pearson_r": round(rs, 4),
                                            "pandora_pearson_r": round(rp, 4),
                                            "n_spine": int(bms.sum()),
                                            "n_pandora": int(bmp.sum())}

# ---------------------------------------------------------------
# Figure: 3×2 layout
#   Row 0: reversal rate (SPINE) | reversal rate (both, same axes)
#   Rows 1-2: vertex–direction per E_ν slice (4 panels, one per bin)
# ---------------------------------------------------------------
fig = plt.figure(figsize=(16, 15))
fig.suptitle("Reversal Rate and Vertex–Direction Correlation — SPINE vs Pandora\n(CC events, full sample)",
             fontsize=12, fontweight="bold")

# Top row: reversal rate
ax_rev_spine  = fig.add_subplot(3, 2, 1)
ax_rev_both   = fig.add_subplot(3, 2, 2)

for ax in [ax_rev_spine, ax_rev_both]:
    ok_s = sc > 10
    ok_p = pc > 10
    ax.plot(centers_E[ok_s], sr[ok_s] * 100, color="#1f77b4", lw=2, marker="o", ms=5,
            label="SPINE all-sum")
    if ax is ax_rev_both:
        ax.plot(centers_E[ok_p], pr[ok_p] * 100, color="#ff7f0e", lw=2, marker="D", ms=5,
                label="Pandora lngtrk/heshw")
    for edge in E_EDGES[1:-1]:
        ax.axvline(edge, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"True $E_\nu$ [GeV]")
    ax.set_ylabel("Reversal rate [%]  (θ > 90°)")
    ax.set_xlim(0, 4)
    ax.set_ylim(-1, None)
    ax.legend(fontsize=9)

ax_rev_spine.set_title("SPINE all-sum reversal rate vs $E_\\nu$")
ax_rev_both.set_title("SPINE vs Pandora reversal rate")

# Add E_ν bin labels
for ax in [ax_rev_spine, ax_rev_both]:
    for i, (lb, col) in enumerate(zip(E_LABELS, E_COLORS)):
        xmid = 0.5 * (E_EDGES[i] + min(E_EDGES[i+1], 4.0))
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 30
        ax.text(xmid, ymax * 0.97, lb.split("\n")[1],
                ha="center", va="top", fontsize=7, color=col, alpha=0.7)

# Bottom 4 panels: vertex–direction correlation per E_ν slice
VTX_CLIP = 10.0   # cm, clip for display
ANG_CLIP = 180.0

for bi, (lb, col) in enumerate(zip(E_LABELS, E_COLORS)):
    ax = fig.add_subplot(3, 2, 3 + bi)
    bm = e_bin == bi

    for key_ang, key_vtx, col2, lbl in [
        ("spine_ang", "spine_dvtx", "#1f77b4", "SPINE"),
        ("pand_ang",  "pand_dvtx",  "#ff7f0e", "Pandora"),
    ]:
        ang_arr  = s_ang  if key_ang  == "spine_ang" else p_ang
        dvtx_arr = s_dvtx if key_vtx  == "spine_dvtx" else p_dvtx
        mk = bm & np.isfinite(ang_arr) & np.isfinite(dvtx_arr) & (dvtx_arr < VTX_CLIP)
        if mk.sum() < 20:
            continue

        # Scatter (thinned)
        idx = np.where(mk)[0]
        if len(idx) > 3000:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, 3000, replace=False)
        ax.scatter(dvtx_arr[idx], ang_arr[idx],
                   s=0.6, alpha=0.06, color=col2, rasterized=True)

        # Binned median
        med_d, _, _ = binned_statistic(dvtx_arr[mk], ang_arr[mk],
                                       "median", bins=VTX_BINS)
        q25, _, _   = binned_statistic(dvtx_arr[mk], ang_arr[mk],
                                       lambda x: np.percentile(x, 25) if len(x) else _nan,
                                       bins=VTX_BINS)
        q75, _, _   = binned_statistic(dvtx_arr[mk], ang_arr[mk],
                                       lambda x: np.percentile(x, 75) if len(x) else _nan,
                                       bins=VTX_BINS)
        cnt_v, _, _ = binned_statistic(dvtx_arr[mk], dvtx_arr[mk], "count", bins=VTX_BINS)
        cv = 0.5 * (VTX_BINS[:-1] + VTX_BINS[1:])
        ok = cnt_v > 10
        ax.plot(cv[ok], med_d[ok], color=col2, lw=2, marker="o", ms=4,
                label=f"{lbl}  r={corr_summary[lb.replace(chr(10),' ')][f'{lbl.lower()}_pearson_r']:.2f}")
        ax.fill_between(cv[ok],
                        np.where(np.isfinite(q25[ok]), q25[ok], med_d[ok]),
                        np.where(np.isfinite(q75[ok]), q75[ok], med_d[ok]),
                        color=col2, alpha=0.12)

    ax.set_xlabel(r"Vertex $\Delta r$ [cm]")
    ax.set_ylabel(r"Direction $\Delta\theta$ [deg]")
    ax.set_title(f"{lb.replace(chr(10), ' ')}  (N={bm.sum():,})", color=col, fontweight="bold")
    ax.axhline(90, color="gray", ls="--", lw=0.8, alpha=0.5, label="90° (reversal)")
    ax.set_xlim(0, VTX_CLIP); ax.set_ylim(0, ANG_CLIP)
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(outdir, "reversal_and_vtxdir.png"), dpi=150)
plt.close(fig)
print("  Saved reversal_and_vtxdir.png")

# ---------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------
summary = {"n_cc": len(records)}

rev = {}
for bi, lb in enumerate(E_LABELS):
    key = lb.replace("\n", " ")
    bms = (e_bin == bi) & np.isfinite(s_ang)
    bmp = (e_bin == bi) & np.isfinite(p_ang)
    rev[key] = {
        "spine_reversal_rate":   round(float((s_ang[bms] > 90).mean()), 4) if bms.sum() > 0 else None,
        "pandora_reversal_rate": round(float((p_ang[bmp] > 90).mean()), 4) if bmp.sum() > 0 else None,
        "n": int(bms.sum()),
    }
summary["reversal_rates"] = rev
summary["overall_reversal"] = {
    "spine":   round(float((s_ang[np.isfinite(s_ang)] > 90).mean()), 4),
    "pandora": round(float((p_ang[np.isfinite(p_ang)] > 90).mean()), 4),
}
summary["vtx_dir_pearson"] = corr_summary

with open(os.path.join(outdir, "reversal_vtxdir.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved reversal_vtxdir.json")
print("\nDone!")

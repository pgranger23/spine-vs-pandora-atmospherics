#!/usr/bin/env python3
"""
Study M — Direction-resolution error budget.

Goal: decide WHAT to improve to sharpen the SPINE neutrino-direction estimate
(the EM-calibrated all-particle momentum sum).  Starting from the reconstructed
direction, we use the CAF's stored truth matching (sp.truth / sp.truthOverlap)
to replace ONE reconstructed ingredient at a time with truth and measure how much
the direction resolution improves.  The substitution that recovers the most
resolution is the one worth investing in.

Per CC event, the direction is  V = Σ_i  w_i · p̂_i , with w_i = scale_i·|p_i|
(scale = 0.198 for EM particles).  Scenarios (each vs the TRUE ν direction):

  base        reconstructed as-is (current SPINE method)
  +mag        matched particles use TRUE |p| (fixes energy/momentum magnitude,
              incl. EM calibration & hadronic loss)
  +dir        matched particles use TRUE direction (fixes per-particle angle)
  +PID        EM×0.198 scale decided by the TRUE species (fixes PID mis-scaling)
  −ghosts     drop reco particles with no truth match
  +missing    add TRUE momenta of reconstructible primaries reco MISSED (efficiency)
  perfect-vis Σ of ALL true visible primaries = perfect reco of the visible system
              (the irreducible Fermi-motion / invisible-neutron floor)

Metrics per Eν bin: median 3D opening angle, and (P84−P16)/2 of the zenith
residual Δθ_y (the paper metric).  Reads via PyROOT.

Output: figures/direction_budget.png, figures/direction_budget.json
"""

import os, sys, math, json, subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic as _bstat
from tqdm import tqdm

base_dir = os.getcwd()
try:
    _libdir = subprocess.check_output(["root-config", "--libdir"]).decode().strip()
    if _libdir not in sys.path:
        sys.path.insert(0, _libdir)
except Exception:
    pass
import ROOT
for _lib in ["libSRProxy_BasicTypes", "libduneanaobj_StandardRecord",
             "libduneanaobj_StandardRecordProxy"]:
    ROOT.gSystem.Load(os.path.join(base_dir, "duneanaobj/build/Linux/lib", _lib))

import plot_style
plot_style.apply()
outdir = "figures"
os.makedirs(outdir, exist_ok=True)
_nan = float("nan")

K_PRIMARY = 1
EM_PDG    = (11, 22, 111)
EM_CORR   = 0.198
VIS       = (11, 13, 22, 211, 321, 2212)   # visible (ionising) species
P_MIN_MEV = 0.5
P_VIS_MIN = 0.02   # GeV/c — min true momentum to enter the visible/missing sums

def vmag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def closest(coll, n, vtx):
    best, bi = 1e18, 0
    for i in range(n):
        v = coll[i].vtx
        d = vmag(v.x - vtx.x, v.y - vtx.y, v.z - vtx.z)
        if d < best:
            best, bi = d, i
    return bi

def ang3d(vx, vy, vz, tx, ty, tz):
    m = vmag(vx, vy, vz)
    if m < 1e-12:
        return _nan
    return math.degrees(math.acos(max(-1.0, min(1.0, (vx*tx+vy*ty+vz*tz)/m))))

def zen_res(vx, vy, vz, tdy):
    """Δθ_y = arccos(d_y^reco) − arccos(d_y^true), degrees."""
    m = vmag(vx, vy, vz)
    if m < 1e-12:
        return _nan
    return math.degrees(math.acos(max(-1.,min(1.,vy/m))) - math.acos(max(-1.,min(1.,tdy))))

def best_match(sp):
    n = sp.truth.size()
    if n == 0:
        return None
    bo, bk = -1.0, 0
    for k in range(sp.truthOverlap.size()):
        if sp.truthOverlap[k] > bo:
            bo, bk = sp.truthOverlap[k], k
    tid = sp.truth[bk]
    return (bo, int(tid.ixn), int(tid.part), int(tid.type))

SCEN = ["base", "+mag", "+dir", "+PID", "-ghosts", "+missing", "perfect-vis"]

f = ROOT.TFile.Open("spine_merged_reprocess.root")
t = f.Get("recTree")
rows = []

for ie in tqdm(range(t.GetEntries()), desc="Scanning"):
    t.GetEntry(ie)
    rec = t.rec
    if rec.mc.nnu < 1:
        continue
    nu = rec.mc.nu[0]
    if not nu.iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue
    tm = nu.momentum
    tmag = vmag(tm.x, tm.y, tm.z)
    if tmag < 1e-9:
        continue
    tdx, tdy, tdz = tm.x/tmag, tm.y/tmag, tm.z/tmag
    E = float(nu.E)

    dlp = ixn.dlp[closest(ixn.dlp, ixn.ndlp, nu.vtx)]

    # gather reco primaries + their truth match
    parts = []            # dict per reco primary
    matched_prim = set()  # true-primary indices that got a match
    for pi in range(dlp.part.ndlp):
        sp = dlp.part.dlp[pi]
        if not sp.primary:
            continue
        pp = sp.p
        pm = vmag(pp.x, pp.y, pp.z)
        if not (math.isfinite(pm) and pm > P_MIN_MEV):
            continue
        rpdg = abs(int(sp.pdg))
        dr = (pp.x/pm, pp.y/pm, pp.z/pm)
        pmag_g = pm * 0.001
        sc = EM_CORR if rpdg in EM_PDG else 1.0
        bm = best_match(sp)
        tinfo = None
        if bm is not None:
            _, ii, part_i, typ = bm
            try:
                nut = rec.mc.nu[ii]
                tp = nut.prim[part_i] if typ == K_PRIMARY else nut.sec[part_i]
                tvx, tvy, tvz = tp.p.px, tp.p.py, tp.p.pz
                tpm = vmag(tvx, tvy, tvz)
                tinfo = {"pdg": abs(int(tp.pdg)), "pmag": tpm,
                         "dir": (tvx/tpm, tvy/tpm, tvz/tpm) if tpm > 1e-9 else dr}
                if typ == K_PRIMARY and ii == 0:
                    matched_prim.add(part_i)
            except Exception:
                tinfo = None
        parts.append({"dir": dr, "pmag": pmag_g, "sc": sc, "rpdg": rpdg, "t": tinfo})

    if not parts:
        continue

    def accum(getw, getd, subset=None):
        vx = vy = vz = 0.0
        for p in (parts if subset is None else subset):
            w = getw(p); d = getd(p)
            vx += w*d[0]; vy += w*d[1]; vz += w*d[2]
        return vx, vy, vz

    V = {}
    # base
    V["base"] = accum(lambda p: p["sc"]*p["pmag"], lambda p: p["dir"])
    # +mag : matched -> true |p| (physical, no EM scale); ghosts keep reco
    V["+mag"] = accum(lambda p: (p["t"]["pmag"] if p["t"] else p["sc"]*p["pmag"]),
                      lambda p: p["dir"])
    # +dir : matched -> true direction
    V["+dir"] = accum(lambda p: p["sc"]*p["pmag"],
                      lambda p: (p["t"]["dir"] if p["t"] else p["dir"]))
    # +PID : EM scale from TRUE species
    def w_pid(p):
        if p["t"] is not None:
            sc = EM_CORR if p["t"]["pdg"] in EM_PDG else 1.0
        else:
            sc = p["sc"]
        return sc*p["pmag"]
    V["+PID"] = accum(w_pid, lambda p: p["dir"])
    # -ghosts : matched only
    V["-ghosts"] = accum(lambda p: p["sc"]*p["pmag"], lambda p: p["dir"],
                         subset=[p for p in parts if p["t"] is not None])
    # +missing : base + true momenta of reconstructible primaries reco missed
    bx, by, bz = V["base"]
    for i in range(nu.nprim):
        if i in matched_prim:
            continue
        tp = nu.prim[i]
        if abs(int(tp.pdg)) not in VIS:
            continue
        pmm = vmag(tp.p.px, tp.p.py, tp.p.pz)
        if pmm < P_VIS_MIN:
            continue
        bx += tp.p.px; by += tp.p.py; bz += tp.p.pz
    V["+missing"] = (bx, by, bz)
    # perfect-vis : all true visible primaries
    px = py = pz = 0.0
    for i in range(nu.nprim):
        tp = nu.prim[i]
        if abs(int(tp.pdg)) not in VIS:
            continue
        pmm = vmag(tp.p.px, tp.p.py, tp.p.pz)
        if pmm < P_VIS_MIN:
            continue
        px += tp.p.px; py += tp.p.py; pz += tp.p.pz
    V["perfect-vis"] = (px, py, pz)

    row = {"E": E}
    for s in SCEN:
        vx, vy, vz = V[s]
        row[f"a3d_{s}"] = ang3d(vx, vy, vz, tdx, tdy, tdz)
        row[f"zen_{s}"] = zen_res(vx, vy, vz, tdy)
    rows.append(row)

print(f"  CC events: {len(rows):,}")
A = {k: np.array([r.get(k, _nan) for r in rows], dtype=float) for k in rows[0]}

nu_bins = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
nu_ctr  = 0.5 * (nu_bins[:-1] + nu_bins[1:])

def binned(x, y, fn, bins=nu_bins):
    m = np.isfinite(x) & np.isfinite(y)
    v, _, _ = _bstat(x[m], y[m], fn, bins=bins)
    c, _, _ = _bstat(x[m], y[m], "count", bins=bins)
    return np.where(c >= 10, v, _nan)

med   = lambda x, y: binned(x, y, "median")
rhalf = lambda x, y: binned(x, y, lambda v: 0.5*(np.percentile(v,84)-np.percentile(v,16)))

COL = {"base":"#000000", "+mag":"#1f77b4", "+dir":"#ff7f0e", "+PID":"#2ca02c",
       "-ghosts":"#9467bd", "+missing":"#8c564b", "perfect-vis":"#d62728"}

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Study M — Direction-resolution error budget (single-fix substitutions, CC)", fontsize=13)

x_all = A["E"]
for s in SCEN:
    ls = "-" if s in ("base", "perfect-vis") else "--"
    lw = 2.6 if s in ("base", "perfect-vis") else 1.8
    ax[0].plot(nu_ctr, med(x_all, A[f"a3d_{s}"]), color=COL[s], ls=ls, lw=lw, marker="o", ms=3, label=s)
ax[0].set_xlabel(r"True $E_\nu$ [GeV]"); ax[0].set_ylabel("Median 3D direction error [°]")
ax[0].set_title("Median 3D opening angle vs energy"); ax[0].set_xlim(0.1,4.0); ax[0].set_ylim(0,None)
ax[0].legend(fontsize=8, ncol=2)

# bar chart: low-E (E<0.5) median 3D error per scenario
lowE = x_all < 0.5
vals = [np.nanmedian(A[f"a3d_{s}"][lowE]) for s in SCEN]
ax[1].barh(range(len(SCEN)), vals, color=[COL[s] for s in SCEN])
ax[1].set_yticks(range(len(SCEN))); ax[1].set_yticklabels(SCEN)
ax[1].invert_yaxis()
for i, v in enumerate(vals):
    ax[1].text(v, i, f" {v:.1f}°", va="center", fontsize=9)
ax[1].set_xlabel("Median 3D direction error [°]")
ax[1].set_title("E < 0.5 GeV: effect of each single fix")
fig.tight_layout()
out = os.path.join(outdir, "direction_budget.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

# ── table + JSON ──────────────────────────────────────────────────────────────
E_CUTS = [(0,0.5),(0.5,1),(1,2),(2,1e9)]; E_LABS=["E<0.5","0.5-1","1-2",">2"]
summary = {}
print(f"\nMedian 3D direction error [°] by scenario and Eν:")
print(f"{'scenario':<13}" + "".join(f"{lb:>9}" for lb in E_LABS))
for s in SCEN:
    r = {}
    line = f"{s:<13}"
    for (lo,hi),lb in zip(E_CUTS,E_LABS):
        m = (x_all>=lo)&(x_all<hi)
        v = float(np.nanmedian(A[f"a3d_{s}"][m]))
        r[lb] = v; line += f"{v:>8.1f}°"
    summary[s] = r
    print(line)
with open(os.path.join(outdir,"direction_budget.json"),"w") as fh:
    json.dump(summary, fh, indent=1)
print(f"Saved: {os.path.join(outdir,'direction_budget.json')}")

#!/usr/bin/env python3
"""
Study N — Which missed particles matter for the direction (breakdown of Study M's
"+missing" lever, by species and by momentum).

Study M showed that at low energy the dominant improvable direction error comes
from reconstructible primaries SPINE MISSES.  Here we attribute that improvement:
starting from the reco direction, we add back the true momenta of missed
reconstructible primaries restricted to ONE species (or ONE momentum band) at a
time, and measure the direction recovered.  We also show the census/spectrum of
what is actually missed.

"Missed" = a true primary of a visible species (e, μ, γ, π±, p), true |p| > 20
MeV/c, with NO reco particle matched to it (stored truth matching).

Metric: median 3D direction error [°].  Reads via PyROOT.
Output: figures/missing_breakdown.png, figures/missing_breakdown.json
"""

import os, sys, math, json, subprocess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import caf_env
ROOT = caf_env.setup_root()

import plot_style
plot_style.apply()
outdir = "figures"
os.makedirs(outdir, exist_ok=True)
_nan = float("nan")

K_PRIMARY = 1
EM_PDG    = (11, 22, 111)
EM_CORR   = 0.198
SPECIES   = [2212, 211, 13, 11, 22]
SP_LABEL  = {2212: "p", 211: r"$\pi^\pm$", 13: r"$\mu$", 11: "e", 22: r"$\gamma$"}
SP_COL    = {2212: "#1f77b4", 211: "#ff7f0e", 13: "#2ca02c", 11: "#d62728", 22: "#9467bd"}
P_MIN_MEV = 0.5
P_VIS_MIN = 0.02   # GeV/c
PBANDS    = [(0.02, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1e9)]
PB_LABEL  = ["0.02–0.25", "0.25–0.5", "0.5–1.0", ">1.0"]

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

def best_match(sp):
    n = sp.truth.size()
    if n == 0:
        return None
    bo, bk = -1.0, 0
    for k in range(sp.truthOverlap.size()):
        if sp.truthOverlap[k] > bo:
            bo, bk = sp.truthOverlap[k], k
    tid = sp.truth[bk]
    return (int(tid.ixn), int(tid.part), int(tid.type))

f = ROOT.TFile.Open(caf_env.CAF_FILE)
t = f.Get("recTree")
rows = []
miss_spectrum = {s: [] for s in SPECIES}   # (E, pmag) of missed particles
miss_count    = {s: [] for s in SPECIES}   # per-event (E, n_missed_of_species)

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

    # base reco sum + which true primaries got matched
    bx = by = bz = 0.0
    matched_prim = set()
    for pi in range(dlp.part.ndlp):
        sp = dlp.part.dlp[pi]
        if not sp.primary:
            continue
        pp = sp.p
        pm = vmag(pp.x, pp.y, pp.z)
        if not (math.isfinite(pm) and pm > P_MIN_MEV):
            continue
        sc = EM_CORR if abs(int(sp.pdg)) in EM_PDG else 1.0
        g = pm * 0.001
        bx += sc*g*pp.x/pm; by += sc*g*pp.y/pm; bz += sc*g*pp.z/pm
        bm = best_match(sp)
        if bm is not None and bm[2] == K_PRIMARY and bm[0] == 0:
            matched_prim.add(bm[1])

    # missed reconstructible primaries by species / momentum
    missed = []   # (species, pmag, px, py, pz)
    per_sp_count = {s: 0 for s in SPECIES}
    for i in range(nu.nprim):
        if i in matched_prim:
            continue
        tp = nu.prim[i]
        s = abs(int(tp.pdg))
        if s not in SPECIES:
            continue
        pmm = vmag(tp.p.px, tp.p.py, tp.p.pz)
        if pmm < P_VIS_MIN:
            continue
        missed.append((s, pmm, tp.p.px, tp.p.py, tp.p.pz))
        miss_spectrum[s].append((E, pmm))
        per_sp_count[s] += 1
    for s in SPECIES:
        miss_count[s].append((E, per_sp_count[s]))

    row = {"E": E, "base": ang3d(bx, by, bz, tdx, tdy, tdz)}
    # add-back by species
    for s in SPECIES:
        ax = bx; ay = by; az = bz
        for (ss, pm, px, py, pz) in missed:
            if ss == s:
                ax += px; ay += py; az += pz
        row[f"sp_{s}"] = ang3d(ax, ay, az, tdx, tdy, tdz)
    # add-back by momentum band
    for bi, (lo, hi) in enumerate(PBANDS):
        ax = bx; ay = by; az = bz
        for (ss, pm, px, py, pz) in missed:
            if lo <= pm < hi:
                ax += px; ay += py; az += pz
        row[f"pb_{bi}"] = ang3d(ax, ay, az, tdx, tdy, tdz)
    # all missed
    ax = bx; ay = by; az = bz
    for (ss, pm, px, py, pz) in missed:
        ax += px; ay += py; az += pz
    row["all"] = ang3d(ax, ay, az, tdx, tdy, tdz)
    rows.append(row)

print(f"  CC events: {len(rows):,}")
A = {k: np.array([r.get(k, _nan) for r in rows], dtype=float) for k in rows[0]}

def med_lowE(key, emax=0.5):
    m = (A["E"] < emax) & np.isfinite(A[key])
    return float(np.median(A[key][m]))

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Study N — Which missed particles matter for the direction (E < 0.5 GeV unless noted)", fontsize=13)

# (a) add-back by species — bar chart
ax = axes[0, 0]
base_v = med_lowE("base")
names  = ["base"] + [f"+{SP_LABEL[s]}" for s in SPECIES] + ["+all"]
vals   = [base_v] + [med_lowE(f"sp_{s}") for s in SPECIES] + [med_lowE("all")]
cols   = ["#000000"] + [SP_COL[s] for s in SPECIES] + ["#8c564b"]
ax.barh(range(len(names)), vals, color=cols)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v, i, f" {v:.1f}° ({v-base_v:+.1f})" if i > 0 else f" {v:.1f}°", va="center", fontsize=8)
ax.set_xlabel("Median 3D direction error [°]")
ax.set_title("Add back missed particles of ONE species")

# (b) momentum spectrum of missed particles (E<0.5)
ax = axes[0, 1]
bins = np.linspace(0, 1.5, 31)
for s in SPECIES:
    p = [pm for (E, pm) in miss_spectrum[s] if E < 0.5]
    ax.hist(p, bins=bins, histtype="step", lw=2.0, color=SP_COL[s], label=SP_LABEL[s])
ax.set_xlabel("Missed particle true |p| [GeV/c]"); ax.set_ylabel("Count (E<0.5 GeV)")
ax.set_title("Spectrum of missed reconstructible primaries"); ax.legend(fontsize=9)

# (c) add-back by momentum band — bar chart
ax = axes[1, 0]
names2 = ["base"] + [f"+[{lb}]" for lb in PB_LABEL] + ["+all"]
vals2  = [base_v] + [med_lowE(f"pb_{bi}") for bi in range(len(PBANDS))] + [med_lowE("all")]
cols2  = ["#000000", "#c7e9c0", "#74c476", "#31a354", "#006d2c", "#8c564b"]
ax.barh(range(len(names2)), vals2, color=cols2)
ax.set_yticks(range(len(names2))); ax.set_yticklabels(names2); ax.invert_yaxis()
for i, v in enumerate(vals2):
    ax.text(v, i, f" {v:.1f}° ({v-base_v:+.1f})" if i > 0 else f" {v:.1f}°", va="center", fontsize=8)
ax.set_xlabel("Median 3D direction error [°]")
ax.set_title("Add back missed particles in ONE |p| band [GeV/c]")

# (d) mean number of missed particles per event by species vs energy
ax = axes[1, 1]
nu_bins = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
nu_ctr  = 0.5*(nu_bins[:-1]+nu_bins[1:])
from scipy.stats import binned_statistic as _bstat
for s in SPECIES:
    E = np.array([e for (e, n) in miss_count[s]])
    N = np.array([n for (e, n) in miss_count[s]], dtype=float)
    v, _, _ = _bstat(E, N, "mean", bins=nu_bins)
    ax.plot(nu_ctr, v, color=SP_COL[s], lw=2.0, marker="o", ms=3, label=SP_LABEL[s])
ax.set_xlabel(r"True $E_\nu$ [GeV]"); ax.set_ylabel("Mean # missed reconstructible / event")
ax.set_title("How many are missed, by species vs energy"); ax.set_xlim(0.1, 4.0); ax.legend(fontsize=9)

fig.tight_layout()
out = os.path.join(outdir, "missing_breakdown.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\nE<0.5 GeV median 3D error: base = {base_v:.1f}°")
print("  add back missed species:")
for s in SPECIES:
    print(f"    +{SP_LABEL[s]:<7} {med_lowE(f'sp_{s}'):5.1f}°  ({med_lowE(f'sp_{s}')-base_v:+.1f})")
print(f"    +all      {med_lowE('all'):5.1f}°  ({med_lowE('all')-base_v:+.1f})")
print("  add back missed by |p| band:")
for bi, lb in enumerate(PB_LABEL):
    print(f"    [{lb:<9}] {med_lowE(f'pb_{bi}'):5.1f}°  ({med_lowE(f'pb_{bi}')-base_v:+.1f})")
# missed census
print("\n  mean missed reconstructible primaries / event (E<0.5):")
for s in SPECIES:
    n = np.mean([n for (e, n) in miss_count[s] if e < 0.5])
    print(f"    {SP_LABEL[s]:<7} {n:.3f}")

summary = {"base_lowE": base_v,
           "add_species": {int(s): med_lowE(f"sp_{s}") for s in SPECIES},
           "add_pband": {PB_LABEL[bi]: med_lowE(f"pb_{bi}") for bi in range(len(PBANDS))},
           "add_all": med_lowE("all"),
           "mean_missed_lowE": {int(s): float(np.mean([n for (e,n) in miss_count[s] if e<0.5])) for s in SPECIES}}
with open(os.path.join(outdir, "missing_breakdown.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"Saved: {os.path.join(outdir,'missing_breakdown.json')}")

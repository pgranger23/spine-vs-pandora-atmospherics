#!/usr/bin/env python3
"""
Study P — Is the PID topology dependence (Study O) real, or just a reflection of
the true-particle energy spectrum differing by mode?

Study O showed PID efficiency vs NEUTRINO energy depends on mode. But at fixed Eν
a DIS event shares its energy among many soft particles while a QE event has two
hard ones, so the mode dependence could simply be a per-particle-energy effect.
The decisive test is PID efficiency vs the TRUE PARTICLE kinetic energy, split by
mode: if the curves overlap, PID is governed by the particle's own energy and the
"topology bias" is really a spectrum effect; if they separate, topology genuinely
matters beyond the particle energy.

Also shows the proton fate (reco as proton / pion / muon / not reconstructed) vs
true proton KE — the mechanism behind the composition bias (fake pions, etc.).

Uses stored truth matching.  Reads via PyROOT.
Output: figures/pid_particle_energy.png, figures/pid_proton_fate.png, figures/pid_particle_energy.json
"""

import os, sys, math, json, subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic as _bstat
from tqdm import tqdm

import caf_env
ROOT = caf_env.setup_root()

import plot_style
plot_style.apply()
outdir = "figures"
os.makedirs(outdir, exist_ok=True)
_nan = float("nan")

K_PRIMARY = 1
SPECIES   = [13, 2212, 211, 11]
SP_LABEL  = {13: r"$\mu$", 2212: "p", 211: r"$\pi^\pm$", 11: "e"}
MASS      = {13: 0.105658, 2212: 0.938272, 211: 0.139571, 11: 0.000511, 22: 0.0}
MODE_LABEL = {0: "QE", 1: "RES", 2: "DIS", 10: "MEC"}
MODE_COL   = {"QE": "#1f77b4", "RES": "#2ca02c", "DIS": "#d62728", "MEC": "#ff7f0e"}
OV_MIN = 0.10

def cat(pdg):
    a = abs(int(pdg))
    return a if a in (11, 13, 22, 211, 2212) else None

def vmag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def true_ke(pdg, p):
    m = MASS.get(abs(int(pdg)), 0.0)
    return math.sqrt(p*p + m*m) - m

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

def closest(coll, n, vtx):
    best, bi = 1e18, 0
    for i in range(n):
        v = coll[i].vtx
        d = vmag(v.x - vtx.x, v.y - vtx.y, v.z - vtx.z)
        if d < best:
            best, bi = d, i
    return bi

f = ROOT.TFile.Open(caf_env.CAF_FILE)
t = f.Get("recTree")

# eff_ke[species][mode] -> list of (KE, correct)
eff_ke = {s: {m: [] for m in MODE_LABEL.values()} for s in SPECIES}
# proton fate: list of (KE, reco_category or 'none')
p_fate = []

for ie in tqdm(range(t.GetEntries()), desc="Scanning"):
    t.GetEntry(ie)
    rec = t.rec
    if rec.mc.nnu < 1:
        continue
    nu = rec.mc.nu[0]
    if not nu.iscc:
        continue
    mode = MODE_LABEL.get(int(nu.mode))
    if mode is None:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue
    dlp = ixn.dlp[closest(ixn.dlp, ixn.ndlp, nu.vtx)]

    # true primary index -> best (overlap, reco_cat)
    prim_match = {}
    for pi in range(dlp.part.ndlp):
        sp = dlp.part.dlp[pi]
        if not sp.primary:
            continue
        rc = cat(sp.pdg)
        bm = best_match(sp)
        if bm is None:
            continue
        ov, ii, part, typ = bm
        if ii == 0 and typ == K_PRIMARY and ov >= OV_MIN:
            cur = prim_match.get(part)
            if cur is None or ov > cur[0]:
                prim_match[part] = (ov, rc)

    for i in range(nu.nprim):
        tp = nu.prim[i]
        s = cat(tp.pdg)
        if s is None or s == 22:
            continue
        ke = true_ke(tp.pdg, vmag(tp.p.px, tp.p.py, tp.p.pz))
        m = prim_match.get(i)
        rc = m[1] if m is not None else None
        eff_ke[s][mode].append((ke, 1.0 if rc == s else 0.0))
        if s == 2212:
            p_fate.append((ke, rc))

print("  done scanning")

ke_bins = np.logspace(np.log10(0.01), np.log10(2.0), 13)
ke_ctr  = np.sqrt(ke_bins[:-1]*ke_bins[1:])
def eff_curve(pairs):
    if len(pairs) < 20:
        return np.full(len(ke_bins)-1, _nan)
    x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    v, _, _ = _bstat(x, y, "mean", bins=ke_bins)
    c, _, _ = _bstat(x, y, "count", bins=ke_bins)
    return np.where(c >= 20, v, _nan)

# ── Figure 1: PID efficiency vs true particle KE, split by mode ───────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
fig.suptitle("Study P — PID efficiency vs TRUE PARTICLE kinetic energy, split by interaction mode", fontsize=13)
for c, s in enumerate(SPECIES):
    ax = axes[c]
    for m in ["QE", "RES", "DIS", "MEC"]:
        ax.plot(ke_ctr, eff_curve(eff_ke[s][m]), color=MODE_COL[m], lw=2.0, marker="o", ms=3, label=m)
    ax.set_title(SP_LABEL[s], fontsize=13)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.set_xlabel("True particle KE [GeV]")
    if c == 0:
        ax.set_ylabel("PID efficiency"); ax.legend(fontsize=9, title="mode")
fig.tight_layout()
out1 = os.path.join(outdir, "pid_particle_energy.png")
fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out1}")

# ── Figure 2: proton fate vs true proton KE ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
fates = [(2212, "reco proton", "#1f77b4"), (211, "reco pion", "#ff7f0e"),
         (13, "reco muon", "#2ca02c"), (11, "reco e", "#d62728"),
         (None, "not reconstructed", "#7f7f7f")]
KE = np.array([k for (k, rc) in p_fate])
for fc, lab, col in fates:
    hit = np.array([1.0 if rc == fc else 0.0 for (k, rc) in p_fate])
    v, _, _ = _bstat(KE, hit, "mean", bins=ke_bins)
    c, _, _ = _bstat(KE, hit, "count", bins=ke_bins)
    ax.plot(ke_ctr, np.where(c >= 20, v, _nan), color=col, lw=2.0, marker="o", ms=3, label=lab)
ax.axvline(0.25, color="k", ls=":", lw=1.0)  # p reconstructibility mom threshold ~ KE 33 MeV -> actually mark 0.25 GeV/c ~ KE 0.033
ax.set_xscale("log"); ax.set_xlabel("True proton KE [GeV]"); ax.set_ylabel("Fraction of true protons")
ax.set_ylim(0, 1.02)
ax.set_title("Proton fate vs true proton KE (all CC modes)")
ax.legend(fontsize=9)
fig.tight_layout()
out2 = os.path.join(outdir, "pid_proton_fate.png")
fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out2}")

# ── summary: efficiency at matched KE across modes (spread) ───────────────────
E_CUTS = [(0.05, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 1.5)]
E_LABS = ["KE 0.05-0.15", "0.15-0.3", "0.3-0.6", "0.6-1.5"]
summary = {}
for s in SPECIES:
    print(f"\nPID eff of {SP_LABEL[s]} vs true KE, by mode (tests topology-vs-energy confound):")
    print(f"  {'KE band':<14}" + "".join(f"{m:>7}" for m in ['QE','RES','DIS','MEC']) + "   spread")
    summary[int(s)] = {}
    for (lo, hi), lb in zip(E_CUTS, E_LABS):
        vals = {}
        for m in ['QE','RES','DIS','MEC']:
            arr = [c for (k, c) in eff_ke[s][m] if lo <= k < hi]
            vals[m] = float(np.mean(arr)) if len(arr) >= 20 else _nan
        finite = [v for v in vals.values() if math.isfinite(v)]
        spread = (max(finite) - min(finite)) if len(finite) >= 2 else _nan
        summary[int(s)][lb] = {**vals, "spread": spread}
        print(f"  {lb:<14}" + "".join(f"{vals[m]:>7.2f}" if math.isfinite(vals[m]) else f"{'--':>7}" for m in ['QE','RES','DIS','MEC']) + f"   {spread:.2f}")

with open(os.path.join(outdir, "pid_particle_energy.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"Saved: {os.path.join(outdir,'pid_particle_energy.json')}")

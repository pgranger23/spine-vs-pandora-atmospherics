#!/usr/bin/env python3
"""
Investigate why SPINE electron momentum is over-estimated by ~4.76x.

Key finding from investigate_em_bias.py: p_mag ≈ p_E for electrons (ratio=1.000),
and p.E = sqrt(p²+m²)-m holds for ALL particles. So p.p IS genuine momentum
(in MeV/c) and p.E is kinetic energy derived from it.

The question: why is p.p 4.76x too large for EM but correct for tracks?
Hypotheses:
1. SPINE uses track-range to infer momentum (wrong for showers)
2. SPINE sums "MIP-equivalent" track lengths (wrong for showers)
3. Network output for EM particles is in different units
4. Training data bias for EM showers
"""

import os, sys, ctypes, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from tqdm import tqdm

import caf_env
pySRProxy = caf_env.setup_pysrproxy()

prox = pySRProxy.StandardRecordFileReader("recTree", caf_env.CAF_FILE)

_MASS = {11: 0.000511, 13: 0.10566, 22: 0.0, 211: 0.13957, 321: 0.49368,
         2212: 0.93827, 2112: 0.93957, 111: 0.13498, 310: 0.49761, 311: 0.49761}

def mag(x, y, z): return math.sqrt(x*x + y*y + z*z)
def best_dlp_idx(ixn, vtx):
    best_d, best_i = float("inf"), 0
    for i in range(min(ixn.ndlp, 100)):
        v = ixn.dlp[i].vtx
        d = mag(v.x - vtx.x, v.y - vtx.y, v.z - vtx.z)
        if d < best_d: best_d, best_i = d, i
    return best_i

# Collect per-particle matched data for electrons only
electrons_reco = []  # SPINE reco electrons
electrons_truth = [] # truth electrons (matched)
other_for_check = [] # muons, protons

MAX_EVENTS = 5000
for entry in tqdm(range(min(MAX_EVENTS, prox.entries())), desc="Scanning"):
    rec = prox.entry(entry); mc = rec.mc
    if not (mc.nnu > 0 and mc.nu[0].iscc): continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1: continue
    nu = mc.nu[0]
    si = best_dlp_idx(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    # Truth primaries
    truths = []
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        pm = mag(px, py, pz)
        if pm < 20: continue
        dx, dy, dz = (px/pm, py/pm, pz/pm)
        if math.isnan(dx): continue
        mass = _MASS.get(abs(tp.pdg), 0) * 1000
        truths.append({"pdg": tp.pdg, "p": pm, "E": math.sqrt(pm*pm + mass*mass),
                       "dx": dx, "dy": dy, "dz": dz})
    if not truths: continue

    # SPINE reco primaries
    spines = []
    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        if not p.primary: continue
        px, py, pz = p.p.x, p.p.y, p.p.z
        pm = mag(px, py, pz)
        if pm < 1: continue
        dx, dy, dz = (px/pm, py/pm, pz/pm)
        if math.isnan(dx): continue
        mass = _MASS.get(abs(p.pdg), 0) * 1000
        spines.append({"idx": pi, "pdg": p.pdg, "p": pm, "E": p.E + mass,
                       "E_kin": p.E,
                       "dx": dx, "dy": dy, "dz": dz})
    if not spines: continue

    # Match: best-per-truth, greedy
    used_truth = set()
    used_reco = set()
    for ti, tp in enumerate(truths):
        best_cos, best_ri = -2, -1
        for ri, rp in enumerate(spines):
            if ri in used_reco: continue
            cos = tp["dx"]*rp["dx"] + tp["dy"]*rp["dy"] + tp["dz"]*rp["dz"]
            if cos > best_cos: best_cos, best_ri = cos, ri
        if best_cos > 0.5 and best_ri >= 0:
            tp_ = truths[ti]
            rp_ = spines[best_ri]
            ang = math.degrees(math.acos(max(-1, min(1, best_cos))))
            d = {"true_pdg": tp_["pdg"], "true_p": tp_["p"], "true_E": tp_["E"],
                 "reco_p": rp_["p"], "reco_pdg": rp_["pdg"],
                 "reco_E": rp_["E"], "reco_Ekin": rp_["E_kin"],
                 "ang_deg": ang, "cos": best_cos}

            apdg = abs(tp_["pdg"])
            if apdg == 11:
                electrons_reco.append(d)
            elif apdg in (13, 2212):
                other_for_check.append(d)

            used_truth.add(ti)
            used_reco.add(best_ri)

print(f"\nMatched electrons: {len(electrons_reco)}")
print(f"Matched muons+protons: {len(other_for_check)}")

if len(electrons_reco) < 10:
    print("Too few electrons. Exiting.")
    sys.exit(0)

# Convert to arrays
ep = {k: np.array([d[k] for d in electrons_reco]) for k in electrons_reco[0].keys()}
op = {k: np.array([d[k] for d in other_for_check]) for k in other_for_check[0].keys()}

# =============================================================
# 1. Check the relationship: what is SPINE p_mag proportional to?
# =============================================================
print("\n" + "="*70)
print("ELECTRON MOMENTUM ANALYSIS")
print("="*70)

tp = ep["true_p"]  # MeV/c
rp = ep["reco_p"]  # MeV/c
tp_gev = tp / 1000
rp_gev = rp / 1000
ratio = rp / tp

print(f"\nN={len(tp)}")
print(f"True p: median={np.median(tp):.0f} MeV/c, mean={np.mean(tp):.0f}")
print(f"Reco p: median={np.median(rp):.0f} MeV/c, mean={np.mean(rp):.0f}")
print(f"p_ratio: median={np.median(ratio):.2f}, mean={np.mean(ratio):.2f}, std={np.std(ratio):.2f}")

# Is the ratio constant or does it depend on true p?
print(f"\n--- Ratio vs true p ---")
for lo, hi in [(20, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10000)]:
    m = (tp >= lo) & (tp < hi)
    if m.sum() < 3: continue
    print(f"  p_true [{lo:5d}, {hi:5d}) MeV: N={m.sum():4d}, p_ratio={np.median(ratio[m]):.2f}+-{np.std(ratio[m]):.2f}, true_p={np.median(tp[m]):.0f}, reco_p={np.median(rp[m]):.0f}")

# Scatter plot: reco_p vs true_p
print(f"\n--- What is reco_p proportional TO? ---")
# Check if reco_p is proportional to true_p:
# Linear fit: reco_p = slope * true_p + intercept
A = np.vstack([tp, np.ones_like(tp)]).T
slope, intercept = np.linalg.lstsq(A, rp, rcond=None)[0]
print(f"Linear fit: reco_p = {slope:.3f} * true_p + {intercept:.0f} MeV")
print(f"  (if slope=~4.76, it's multiplicative)")

# Check if reco_p is actually something else entirely
# What if reco_p is in units of "total charge"?
# Or what if reco_p is the sum of all shower segments' momenta?

# Check: is reco_p proportional to true_E instead of true_p?
true_E = ep["true_E"]  # MeV
slope_E, intercept_E = np.linalg.lstsq(np.vstack([true_E, np.ones_like(true_E)]).T, rp, rcond=None)[0]
print(f"reco_p vs true_E: slope={slope_E:.3f}, intercept={intercept_E:.0f} MeV")
ratio_E = rp / true_E
print(f"  reco_p / true_E: median={np.median(ratio_E):.3f}")

# Is reco_p proportional to true KE (not total E)?
true_KE = true_E - _MASS[11]*1000  # total - mass = KE for electron
slope_KE, intercept_KE = np.linalg.lstsq(np.vstack([true_KE, np.ones_like(true_KE)]).T, rp, rcond=None)[0]
print(f"reco_p vs true_KE: slope={slope_KE:.3f}, intercept={intercept_KE:.0f} MeV")
ratio_KE = rp / true_KE
print(f"  reco_p / true_KE: median={np.median(ratio_KE):.3f}")

# Check if reco_p = some constant (independent of truth)
print(f"\n--- Is reco_p independent of true momentum? ---")
print(f"  Corr coeff (reco_p, true_p): {np.corrcoef(rp, tp)[0,1]:.3f}")
print(f"  Spearman rank: {np.corrcoef(np.argsort(rp), np.argsort(tp))[0,1]:.3f}")

# =============================================================
# 2. Compare SPINE reco kinematics for tracks vs showers
# =============================================================
print("\n" + "="*70)
print("COMPARISON: electrons vs muons vs protons")
print("="*70)

for label, data in [("e", ep), ("mu+p", op)]:
    dp = data
    tp_arr = dp["true_p"]
    rp_arr = dp["reco_p"]
    rat_arr = rp_arr / tp_arr
    m = np.isfinite(rat_arr) & (tp_arr > 20)
    print(f"\n{label}: N={m.sum()}")
    print(f"  <p_true>={np.mean(tp_arr[m]):.0f}, <p_reco>={np.mean(rp_arr[m]):.0f}")
    print(f"  p_ratio median: {np.median(rat_arr[m]):.2f}")

# =============================================================
# 3. Does SPINE reco_E (total) agree with true_E for electrons?
# =============================================================
print("\n" + "="*70)
print("ENERGY CHECK: reco_E vs true_E for electrons")
print("="*70)
reco_E = ep["reco_E"]  # already total energy
true_E_arr = ep["true_E"]
E_ratio = reco_E / true_E_arr
print(f"  reco_E / true_E: median={np.median(E_ratio):.2f}, mean={np.mean(E_ratio):.2f}")
print(f"  Note: reco_E = rp_kinetic + mass, where rp_kinetic = sqrt(p²+m²)-m ≈ p for relativistic e")

# =============================================================
# 4. Check gamma similarly
# =============================================================
print("\n" + "="*70)
print("PHOTON CHECK (same as electrons?)")
print("="*70)
# Photons don't have a separate PDG in the matching easily...
# Let's collect them separately

print("\nDone.")

# =============================================================
# 5. Plot: reco_p vs true_p for electrons
# =============================================================
os.makedirs("figures", exist_ok=True)

import plot_style
plot_style.apply()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Scatter
ax = axes[0, 0]
ax.scatter(tp_gev, rp_gev, s=2, alpha=0.3, c="C0")
ax.plot([0.01, 10], [0.01*4.76, 10*4.76], "r--", lw=1, label="4.76×")
ax.plot([0.01, 10], [0.01, 10], "k--", lw=1, label="1:1")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("True p [GeV/c]"); ax.set_ylabel("Reco p [GeV/c]")
ax.set_title(f"Electrons (N={len(tp)})"); ax.legend(fontsize=8)
ax.set_xlim(0.02, 10); ax.set_ylim(0.02, 100)

# Ratio vs true p
ax = axes[0, 1]
ax.scatter(tp_gev, ratio, s=2, alpha=0.3, c="C0")
ax.axhline(4.76, color="r", ls="--", lw=1)
ax.axhline(np.median(ratio), color="orange", ls="--", lw=1, label=f"med={np.median(ratio):.2f}")
ax.set_xscale("log")
ax.set_xlabel("True p [GeV/c]"); ax.set_ylabel("p_reco / p_true")
ax.set_title("Electron momentum ratio"); ax.legend(fontsize=8)

# Ratio distribution
ax = axes[0, 2]
ax.hist(ratio[ratio < 20], bins=100, histtype="step", lw=1.5, color="C0")
ax.axvline(np.median(ratio), color="r", ls="--", lw=1, label=f"med={np.median(ratio):.1f}")
ax.set_xlabel("p_reco / p_true"); ax.set_ylabel("Counts")
ax.set_title("Electron p_ratio distribution"); ax.legend(fontsize=8)

# Reco_E vs true_E
ax = axes[1, 0]
ax.scatter(true_E_arr/1000, reco_E/1000, s=2, alpha=0.3, c="C0")
ax.plot([0.01, 10], [0.01, 10], "k--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("True E [GeV]"); ax.set_ylabel("Reco E [GeV]")
ax.set_title(f"Electron energy (N={len(tp)})")
ax.set_xlim(0.02, 10); ax.set_ylim(0.02, 100)

# E ratio vs true E
ax = axes[1, 1]
er = reco_E / true_E_arr
ax.scatter(true_E_arr/1000, er, s=2, alpha=0.3, c="C0")
ax.axhline(np.median(er), color="orange", ls="--", lw=1, label=f"med={np.median(er):.2f}")
ax.set_xscale("log")
ax.set_xlabel("True E [GeV]"); ax.set_ylabel("E_reco / E_true")
ax.set_title("Electron energy ratio"); ax.legend(fontsize=8)

# binned profile of ratio
ax = axes[1, 2]
bins = np.logspace(np.log10(0.03), np.log10(10), 20)
c, v, e, m = None, None, None, None  # skip for now

# Histogram of reco E kinetic part only
reco_Ekin = ep["reco_Ekin"] / 1000  # GeV
ax.hist(reco_Ekin, bins=np.logspace(-1, 2, 50), histtype="step", lw=1.5, color="C0")
ax.set_xscale("log")
ax.set_xlabel("Reco kinetic energy [GeV]"); ax.set_ylabel("Counts")
ax.set_title("Electron reco kinetic energy distribution")

fig.tight_layout()
fig.savefig("figures/electron_momentum_investigation.png", dpi=150)
print("\nSaved figures/electron_momentum_investigation.png")

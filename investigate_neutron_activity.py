#!/usr/bin/env python3
"""
Study J — Displaced neutron-induced visible activity.

Primary neutrons are counted as invisible in the multiplicity study (Study I):
they leave no ionization at the vertex.  But a propagating neutron scatters and
knocks out protons far downstream, which DO ionize.  This activity is real
visible energy from the neutrino, yet it is spatially disconnected from the
interaction (median ~1 m away), so neither SPINE's dlp nor Pandora's interaction
object clusters it in — it is neither counted as a reconstructible particle of
the interaction nor recoverable by the interaction-level reco.

This study quantifies that population:
  - trace each reconstructible secondary hadron (p>250, π±>100 MeV/c) to its
    primary ancestor via the parent-G4ID chain, classify neutron- vs
    charged-hadron- vs other-induced;
  - displacement distribution from the vertex;
  - fraction of CC events with displaced (>5 cm) neutron-induced protons vs Eν;
  - visible KE deposited this way, and as a fraction of the primary-neutron KE
    ("neutron visibility fraction").

Reads via PyROOT (nu.sec is a large unsplit blob — see investigate_multiplicity.py).

Output: figures/neutron_activity.png, figures/neutron_activity.json
"""

import os, sys, math, json, subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic as _bstat
from tqdm import tqdm

# ── PyROOT setup ──────────────────────────────────────────────────────────────
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

# ── constants ─────────────────────────────────────────────────────────────────
P_MIN = {2212: 0.250, 211: 0.100}    # GeV/c reconstructibility thresholds
MASS  = {2212: 0.93827, 211: 0.13957, 2112: 0.93957}
DISP_NEAR = 5.0                       # cm — boundary between near-vertex and displaced

def vmag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def ke(p, m):
    return math.sqrt(p*p + m*m) - m

def ancestor_pdg(g4, parent_map, prim_g4):
    """Walk the parent-G4ID chain until a primary is reached; return its |pdg|."""
    seen = 0
    cur = g4
    while cur in parent_map and seen < 100:
        pdg, par = parent_map[cur]
        if cur in prim_g4:
            return abs(pdg)
        cur = par
        seen += 1
    return abs(parent_map[cur][0]) if cur in parent_map else 0

# ── scan ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Study J: Displaced neutron-induced visible activity")
print("=" * 60)

f = ROOT.TFile.Open("spine_merged_reprocess.root")
t = f.Get("recTree")
N = t.GetEntries()

# accumulators
disp_by_anc = {"neutron": [], "charged_had": [], "other": []}   # displacements [cm]
rows = []          # per-event: (E, n_disp_n_protons, vis_ke_neutron_disp, prim_n_ke)
n_recon_sec_by_anc = {"neutron": 0, "charged_had": 0, "other": 0}   # all reconstructible secondaries

for ie in tqdm(range(N), desc="Scanning"):
    t.GetEntry(ie)
    rec = t.rec
    if rec.mc.nnu < 1:
        continue
    nu = rec.mc.nu[0]
    if not nu.iscc:
        continue
    E = float(nu.E)
    vx, vy, vz = nu.vtx.x, nu.vtx.y, nu.vtx.z

    # build G4ID -> (pdg, parent_g4) over prim + sec; primary G4ID set
    pmap = {}
    prim_g4 = set()
    prim_n_ke = 0.0
    for i in range(nu.nprim):
        p = nu.prim[i]
        pmap[p.G4ID] = (int(p.pdg), -1)
        prim_g4.add(p.G4ID)
        if abs(int(p.pdg)) == 2112:
            prim_n_ke += ke(vmag(p.p.px, p.p.py, p.p.pz), MASS[2112])
    for i in range(int(nu.nsec)):
        s = nu.sec[i]
        pmap[s.G4ID] = (int(s.pdg), s.parent)

    n_disp_n_p = 0
    vis_ke_n_disp = 0.0
    for i in range(int(nu.nsec)):
        s = nu.sec[i]
        pdga = abs(int(s.pdg))
        if pdga not in P_MIN:
            continue
        p = vmag(s.p.px, s.p.py, s.p.pz)
        if p < P_MIN[pdga]:
            continue
        anc = ancestor_pdg(s.G4ID, pmap, prim_g4)
        if anc == 2112:
            cls = "neutron"
        elif anc in (2212, 211, 321, 111):
            cls = "charged_had"
        else:
            cls = "other"
        n_recon_sec_by_anc[cls] += 1
        sp = s.start_pos
        d = vmag(sp.x - vx, sp.y - vy, sp.z - vz)
        disp_by_anc[cls].append(d)
        if cls == "neutron" and d > DISP_NEAR:
            n_disp_n_p += 1
            vis_ke_n_disp += ke(p, MASS[pdga])

    rows.append((E, n_disp_n_p, vis_ke_n_disp, prim_n_ke))

print(f"  CC events: {len(rows):,}")

# ── summary ───────────────────────────────────────────────────────────────────
tot_sec = sum(n_recon_sec_by_anc.values())
print("\nReconstructible secondary hadrons by primary ancestor:")
for k in ("neutron", "charged_had", "other"):
    print(f"  {k:<12}: {n_recon_sec_by_anc[k]:>8,} ({100*n_recon_sec_by_anc[k]/max(tot_sec,1):.1f}%)")

def med(v):
    return float(np.median(v)) if v else _nan
print("\nMedian displacement from vertex [cm]:")
for k in ("neutron", "charged_had"):
    print(f"  {k:<12}: {med(disp_by_anc[k]):.1f}  (N={len(disp_by_anc[k]):,})")

E_arr   = np.array([r[0] for r in rows])
ndisp   = np.array([r[1] for r in rows])
viske   = np.array([r[2] for r in rows])   # GeV
prim_nke = np.array([r[3] for r in rows])  # GeV

E_CUTS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
E_LABS = ["E<0.5 GeV", "0.5–1 GeV", "1–2 GeV", ">2 GeV"]
print("\nDisplaced neutron-induced activity by Eν bin:")
print(f"  {'bin':<12} {'evt frac':>9} {'mean N':>8} {'mean visKE':>11} {'n-vis frac':>11}")
summary = {}
for (lo, hi), lb in zip(E_CUTS, E_LABS):
    m = (E_arr >= lo) & (E_arr < hi)
    frac = float((ndisp[m] > 0).mean())
    mn   = float(ndisp[m].mean())
    mvke = float(viske[m].mean() * 1000)   # MeV
    tot_nke = prim_nke[m].sum()
    nvf  = float(viske[m].sum() / tot_nke) if tot_nke > 0 else _nan
    summary[lb] = {"event_frac": frac, "mean_n_disp": mn,
                   "mean_vis_ke_MeV": mvke, "neutron_vis_frac": nvf}
    print(f"  {lb:<12} {100*frac:>8.1f}% {mn:>8.2f} {mvke:>9.1f} MeV {100*nvf:>9.1f}%")

with open(os.path.join(outdir, "neutron_activity.json"), "w") as fh:
    json.dump({"by_ancestor": n_recon_sec_by_anc,
               "median_disp_cm": {k: med(disp_by_anc[k]) for k in disp_by_anc},
               "by_energy": summary}, fh, indent=1)

# ── figure (2×2) ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Study J — Displaced neutron-induced visible activity (CC events)", fontsize=13)

# (a) ancestor breakdown of reconstructible secondary hadrons
ax = axes[0, 0]
labels = ["Neutron", "Charged\nhadron", "Other"]
vals   = [n_recon_sec_by_anc[k] for k in ("neutron", "charged_had", "other")]
colors = ["#d62728", "#1f77b4", "#7f7f7f"]
ax.bar(labels, vals, color=colors, alpha=0.85)
for i, v in enumerate(vals):
    ax.text(i, v, f"{100*v/max(tot_sec,1):.0f}%", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Reconstructible secondary hadrons")
ax.set_title("Primary ancestor of reconstructible secondary hadrons")

# (b) displacement distributions
ax = axes[0, 1]
bins = np.logspace(-1, 3, 40)
for k, c, lb in [("neutron", "#d62728", "Neutron-induced"),
                 ("charged_had", "#1f77b4", "Charged-hadron-induced")]:
    if disp_by_anc[k]:
        ax.hist(np.clip(disp_by_anc[k], 0.1, 1000), bins=bins, histtype="step",
                lw=2.0, color=c, label=lb, density=True)
ax.axvline(DISP_NEAR, color="k", ls="--", lw=1.0, label=f"{DISP_NEAR:.0f} cm near-vertex cut")
ax.set_xscale("log")
ax.set_xlabel("Displacement from vertex [cm]")
ax.set_ylabel("Normalised")
ax.set_title("Where the secondary hadron is created")
ax.legend(fontsize=8)

# (c) event fraction with displaced neutron-induced proton vs Eν
nu_bins = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
nu_ctr  = 0.5 * (nu_bins[:-1] + nu_bins[1:])
def bfrac(x, y, bins=nu_bins):
    v, _, _ = _bstat(x, y, "mean", bins=bins)
    c, _, _ = _bstat(x, y, "count", bins=bins)
    return np.where(c >= 10, v, _nan)
ax = axes[1, 0]
ax.plot(nu_ctr, 100 * bfrac(E_arr, (ndisp > 0).astype(float)),
        color="#d62728", lw=2.0, marker="o", ms=4)
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel("CC events with displaced\nneutron-induced proton [%]")
ax.set_title(f"Prevalence vs energy (displacement > {DISP_NEAR:.0f} cm)")
ax.set_xlim(0.1, 4.0)

# (d) neutron visibility fraction vs Eν
ax = axes[1, 1]
vke_bin, _, _ = _bstat(E_arr, viske, "sum", bins=nu_bins)
nke_bin, _, _ = _bstat(E_arr, prim_nke, "sum", bins=nu_bins)
cnt, _, _ = _bstat(E_arr, viske, "count", bins=nu_bins)
nvf = np.where((cnt >= 10) & (nke_bin > 0), vke_bin / nke_bin, _nan)
ax.plot(nu_ctr, 100 * nvf, color="#9467bd", lw=2.0, marker="s", ms=4)
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel("Neutron visibility fraction [%]")
ax.set_title("Visible KE from displaced n-secondaries / primary-neutron KE")
ax.set_xlim(0.1, 4.0)

fig.tight_layout()
out = os.path.join(outdir, "neutron_activity.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")
print(f"Saved: {os.path.join(outdir,'neutron_activity.json')}")

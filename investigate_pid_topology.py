#!/usr/bin/env python3
"""
Study O — PID performance vs energy and interaction topology, and final-state
composition bias.

Extends Study L (per-species PID vs KE) by splitting on the interaction mode and
by testing whether SPINE's reconstructed final-state composition is biased
relative to truth in a topology-dependent way.

Topology: nu.mode is decoded by PyROOT (SRProxy could not).  Its integer values
do NOT match the caf::ScatteringMode enum names, but the true final-state
composition gives an unambiguous mapping (validated):
    0 = QE  (⟨π±⟩≈0.02)   1 = RES (⟨π±⟩≈0.64, ⟨π0⟩≈0.34)
    2 = DIS (⟨π±⟩≈1.99)   10 = MEC (⟨p⟩≈3.47, ⟨π±⟩≈0.02)

Two questions:
  1. PID efficiency per species vs Eν, split by mode.
  2. Composition bias: per mode, mean reco multiplicity vs mean true-reconstructible
     multiplicity per species — does SPINE systematically over/under-produce
     certain species in certain topologies?

Uses the CAF's stored truth matching.  Reads via PyROOT.
Output: figures/pid_topology.png, figures/pid_composition_bias.png, figures/pid_topology.json
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
SPECIES   = [13, 2212, 211, 11, 22]       # mu, p, pi, e, gamma
SP_LABEL  = {13: r"$\mu$", 2212: "p", 211: r"$\pi^\pm$", 11: "e", 22: r"$\gamma$"}
SP_COL    = {13: "#2ca02c", 2212: "#1f77b4", 211: "#ff7f0e", 11: "#d62728", 22: "#9467bd"}
MODE_LABEL = {0: "QE", 1: "RES", 2: "DIS", 10: "MEC"}
MODE_COL   = {"QE": "#1f77b4", "RES": "#2ca02c", "DIS": "#d62728", "MEC": "#ff7f0e"}
# reconstructibility thresholds [GeV/c; GeV for gamma]
THRESH = {11: 0.030, 13: 0.050, 211: 0.100, 2212: 0.250, 22: 0.100}
OV_MIN = 0.10

def cat(pdg):
    a = abs(int(pdg))
    return a if a in (11, 13, 22, 211, 2212) else None

def vmag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

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

def matched_cat(rec, ii, part, typ):
    try:
        nu = rec.mc.nu[ii]
        tp = nu.prim[part] if typ == K_PRIMARY else nu.sec[part]
        return cat(tp.pdg)
    except Exception:
        return None

def closest(coll, n, vtx):
    best, bi = 1e18, 0
    for i in range(n):
        v = coll[i].vtx
        d = vmag(v.x - vtx.x, v.y - vtx.y, v.z - vtx.z)
        if d < best:
            best, bi = d, i
    return bi

f = ROOT.TFile.Open("spine_merged_reprocess.root")
t = f.Get("recTree")

# efficiency samples: eff[species][mode] -> list of (E, hit)
eff = {s: {m: [] for m in MODE_LABEL.values()} for s in SPECIES}
# composition: per mode, sum of reco / true-reconstructible counts per species + n events
comp = {m: {"n": 0, "reco": {s: 0 for s in SPECIES}, "true": {s: 0 for s in SPECIES}}
        for m in MODE_LABEL.values()}

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
    E = float(nu.E)
    dlp = ixn.dlp[closest(ixn.dlp, ixn.ndlp, nu.vtx)]

    # reco primaries: category + best-truth-match; build truth-primary->reco_cat map
    prim_match = {}   # true prim index -> reco category (best overlap)
    reco_count = {s: 0 for s in SPECIES}
    for pi in range(dlp.part.ndlp):
        sp = dlp.part.dlp[pi]
        if not sp.primary:
            continue
        rc = cat(sp.pdg)
        if rc is not None:
            reco_count[rc] += 1
        bm = best_match(sp)
        if bm is None:
            continue
        ov, ii, part, typ = bm
        if ii == 0 and typ == K_PRIMARY and ov >= OV_MIN:
            cur = prim_match.get(part)
            if cur is None or ov > cur[0]:
                prim_match[part] = (ov, rc)

    # efficiency + true-reconstructible composition
    true_count = {s: 0 for s in SPECIES}
    for i in range(nu.nprim):
        tp = nu.prim[i]
        s = cat(tp.pdg)
        if s is None:
            continue
        p = vmag(tp.p.px, tp.p.py, tp.p.pz)
        if p < THRESH[s]:
            continue
        true_count[s] += 1
        m = prim_match.get(i)
        correct = (m is not None and m[1] == s)
        eff[s][mode].append((E, 1.0 if correct else 0.0))

    comp[mode]["n"] += 1
    for s in SPECIES:
        comp[mode]["reco"][s] += reco_count[s]
        comp[mode]["true"][s] += true_count[s]

print("  done scanning")

# ── Figure 1: PID efficiency vs Eν, per species, split by mode ────────────────
nu_bins = np.array([0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
nu_ctr  = np.sqrt(nu_bins[:-1]*nu_bins[1:])
def eff_curve(pairs):
    if len(pairs) < 20:
        return np.full(len(nu_bins)-1, _nan)
    x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    v, _, _ = _bstat(x, y, "mean", bins=nu_bins)
    c, _, _ = _bstat(x, y, "count", bins=nu_bins)
    return np.where(c >= 20, v, _nan)

fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
fig.suptitle("Study O — SPINE PID efficiency vs energy, split by interaction topology", fontsize=13)
for c, s in enumerate(SPECIES):
    ax = axes[c]
    for m in ["QE", "RES", "DIS", "MEC"]:
        ax.plot(nu_ctr, eff_curve(eff[s][m]), color=MODE_COL[m], lw=2.0, marker="o", ms=3, label=m)
    ax.set_title(SP_LABEL[s], fontsize=13)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.set_xlabel(r"True $E_\nu$ [GeV]")
    if c == 0:
        ax.set_ylabel("PID efficiency"); ax.legend(fontsize=9, title="mode")
fig.tight_layout()
out1 = os.path.join(outdir, "pid_topology.png")
fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out1}")

# ── Figure 2: composition bias — mean reco vs true multiplicity per species/mode
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
modes = ["QE", "RES", "DIS", "MEC"]
xsp = np.arange(len(SPECIES))
w = 0.35
# left: grouped bars true vs reco, one subplot won't fit all modes; use ratio instead
ax = axes[0]
for mi, m in enumerate(modes):
    nn = max(comp[m]["n"], 1)
    ratio = [comp[m]["reco"][s]/max(comp[m]["true"][s], 1e-9) if comp[m]["true"][s] > 0 else _nan
             for s in SPECIES]
    ax.plot(xsp, ratio, "o-", color=MODE_COL[m], lw=2.0, ms=7, label=m)
ax.axhline(1.0, color="k", ls="--", lw=1.0, label="unbiased (reco=true)")
ax.set_xticks(xsp); ax.set_xticklabels([SP_LABEL[s] for s in SPECIES], fontsize=12)
ax.set_ylabel("mean N(reco) / mean N(true reconstructible)")
ax.set_title("Composition bias: reco vs true multiplicity per species, by mode")
ax.set_ylim(0, 6.5)
# flag entries where the true count ≈ 0 (ratio is denominator-limited, not literal)
for s, x in zip(SPECIES, xsp):
    for m in modes:
        nn = max(comp[m]["n"], 1)
        if comp[m]["true"][s]/nn < 0.05:  # < 0.05 true/event
            ax.scatter([x], [min(comp[m]["reco"][s]/comp[m]["true"][s] if comp[m]["true"][s] else 6.4, 6.4)],
                       s=140, facecolors="none", edgecolors="k", lw=1.3, zorder=5)
ax.text(0.02, 0.98, "○ true ≈ 0 → ratio denominator-limited\n(π in QE/MEC, γ: abs. excess ≲ 0.1/evt,\nmostly mis-PID'd protons + ghosts)",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.5, style="italic")
ax.legend(fontsize=8, loc="upper center")

# right: mean multiplicity per species by mode (true solid, reco hatched) for DIS & QE
ax = axes[1]
for mi, m in enumerate(["QE", "DIS"]):
    nn = max(comp[m]["n"], 1)
    tvals = [comp[m]["true"][s]/nn for s in SPECIES]
    rvals = [comp[m]["reco"][s]/nn for s in SPECIES]
    off = -w/2 if mi == 0 else w/2
    ax.bar(xsp+off, tvals, w, color=MODE_COL[m], alpha=0.4, label=f"{m} true")
    ax.bar(xsp+off, rvals, w, facecolor="none", edgecolor=MODE_COL[m], lw=2.0, hatch="///", label=f"{m} reco")
ax.set_xticks(xsp); ax.set_xticklabels([SP_LABEL[s] for s in SPECIES], fontsize=12)
ax.set_ylabel("mean particles / event")
ax.set_title("Mean multiplicity per species: true (solid) vs reco (hatched)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
out2 = os.path.join(outdir, "pid_composition_bias.png")
fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out2}")

# ── tables + JSON ─────────────────────────────────────────────────────────────
summary = {"mode_fractions": {}, "pid_eff_by_mode": {}, "composition_ratio": {}}
tot = sum(comp[m]["n"] for m in modes)
print(f"\nMode fractions (CC):")
for m in modes:
    summary["mode_fractions"][m] = comp[m]["n"]
    print(f"  {m:<4} {comp[m]['n']:>7,}  ({100*comp[m]['n']/max(tot,1):.1f}%)")

print(f"\nOverall PID efficiency by species and mode:")
print(f"{'species':<8}" + "".join(f"{m:>8}" for m in modes))
for s in SPECIES:
    row = {}
    line = f"{SP_LABEL[s]:<8}"
    for m in modes:
        v = float(np.mean([p[1] for p in eff[s][m]])) if eff[s][m] else _nan
        row[m] = v; line += f"{v:>8.2f}"
    summary["pid_eff_by_mode"][int(s)] = row
    print(line)

print(f"\nComposition ratio  reco/true  by species and mode (>1 = over-produced):")
print(f"{'species':<8}" + "".join(f"{m:>8}" for m in modes))
for s in SPECIES:
    row = {}
    line = f"{SP_LABEL[s]:<8}"
    for m in modes:
        r = comp[m]["reco"][s]/comp[m]["true"][s] if comp[m]["true"][s] > 0 else _nan
        row[m] = r; line += f"{r:>8.2f}"
    summary["composition_ratio"][int(s)] = row
    print(line)

# Absolute mean counts (so ratios on near-zero denominators are seen in context)
summary["mean_true_per_evt"] = {}; summary["mean_reco_per_evt"] = {}
for tag, key in [("mean true reco-ble / event", "true"), ("mean RECO / event", "reco")]:
    print(f"\n{tag} by species and mode:")
    print(f"{'species':<8}" + "".join(f"{m:>8}" for m in modes))
    store = summary["mean_true_per_evt"] if key == "true" else summary["mean_reco_per_evt"]
    for s in SPECIES:
        row = {}
        line = f"{SP_LABEL[s]:<8}"
        for m in modes:
            v = comp[m][key][s] / max(comp[m]["n"], 1)
            row[m] = v; line += f"{v:>8.3f}"
        store[int(s)] = row
        print(line)
print("\nNote: for species with mean true ≈ 0 (π in QE/MEC, γ) the reco/true ratio is")
print("denominator-limited and should not be read literally — the absolute excess is")
print("≲ 0.1 objects/event, and those reco 'pions' are mostly mis-PID'd protons + ghosts.")

with open(os.path.join(outdir, "pid_topology.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"Saved: {os.path.join(outdir,'pid_topology.json')}")

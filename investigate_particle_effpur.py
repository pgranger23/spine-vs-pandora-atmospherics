#!/usr/bin/env python3
"""
Study L — Per-particle efficiency / PID-purity / match-overlap vs true energy,
using the CAF's STORED truth matching (sp.truth / sp.truthOverlap), the way the
atmospheric reco paper (arXiv:2601.05697) presents per-species performance.

Each reco particle stores truth matches as caf::TrueParticleID {ixn, part, type}
with a parallel truthOverlap vector.  We take the highest-overlap match and
resolve its PDG from mc.nu[ixn].prim/sec[part] (type: kPrimary=1,
kPrimaryBeforeFSI=2, kSecondary=3).  This replaces the geometric (cosθ) matching
used in Study G with the reconstruction's own truth association.

Metrics, per true species S ∈ {e, μ, γ, π, p}, vs true kinetic energy:
  Efficiency   = fraction of true-S primaries matched to a reco particle
                 (reconstruction eff), and matched to a reco of the CORRECT
                 category (PID eff).  [truth-denominated]
  PID purity   = of reco-S particles that are matched, the fraction whose matched
                 truth really is S.  [reco-denominated, binned by matched-truth KE]
  Match overlap= mean stored truthOverlap of the correct matches (hit-level
                 overlap of the best match; a completeness/purity proxy).

Both SPINE and Pandora are shown for efficiency (Pandora only outputs track=13 /
shower=11, so its per-species PID is coarse — noted on the purity panels).

Reads via PyROOT.  Output: figures/particle_effpur.png, figures/particle_effpur.json
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

# enum values (verified): kPrimary=1, kPrimaryBeforeFSI=2, kSecondary=3
K_PRIMARY = 1

MASS = {11: 0.000511, 13: 0.105658, 22: 0.0, 211: 0.139571, 2212: 0.938272}
SPECIES = [11, 13, 22, 211, 2212]
SP_LABEL = {11: "e", 13: r"$\mu$", 22: r"$\gamma$", 211: r"$\pi^\pm$", 2212: "p"}
OV_MIN = 0.10   # minimum stored overlap to call a truth particle "reconstructed"

def cat(pdg):
    a = abs(int(pdg))
    return a if a in (11, 13, 22, 211, 2212) else None

def vmag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def true_ke(pdg, p):
    m = MASS.get(abs(int(pdg)), 0.0)
    return math.sqrt(p*p + m*m) - m

def best_match(sp):
    """Return (overlap, ixn, part, type) of the highest-overlap truth match, or None."""
    n = sp.truth.size()
    if n == 0:
        return None
    bo, bk = -1.0, 0
    for k in range(sp.truthOverlap.size()):
        if sp.truthOverlap[k] > bo:
            bo, bk = sp.truthOverlap[k], k
    tid = sp.truth[bk]
    return (bo, int(tid.ixn), int(tid.part), int(tid.type))

def matched_pdg(rec, ixn_i, part_i, typ):
    try:
        nu = rec.mc.nu[ixn_i]
        if typ == K_PRIMARY:
            return int(nu.prim[part_i].pdg)
        else:
            return int(nu.sec[part_i].pdg)
    except Exception:
        return None

# ── scan ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Per-particle efficiency / purity / overlap vs true KE")
print("=" * 60)

f = ROOT.TFile.Open("spine_merged_reprocess.root")
t = f.Get("recTree")

# accumulators: lists of (true_KE, hit) per species
eff_reco   = {"spine": {s: [] for s in SPECIES}, "pandora": {s: [] for s in SPECIES}}  # reconstructed at all
eff_pid    = {"spine": {s: [] for s in SPECIES}, "pandora": {s: [] for s in SPECIES}}  # correct category
overlap    = {"spine": {s: [] for s in SPECIES}, "pandora": {s: [] for s in SPECIES}}  # (KE, overlap) correct
pur        = {"spine": {s: [] for s in SPECIES}, "pandora": {s: [] for s in SPECIES}}  # (matched-truth KE, correct?)

def process(rec, coll, ncoll, kind, key):
    # build truth-primary -> best (overlap, reco_cat); and reco purity samples
    prim_best = {}   # part_index -> (overlap, reco_cat)
    for ci in range(ncoll):
        obj = coll[ci]
        npart = obj.part.npandora - 1 if kind == "pandora" else obj.part.ndlp
        parts = obj.part.pandora if kind == "pandora" else obj.part.dlp
        for pi in range(int(npart)):
            sp = parts[pi]
            if not sp.primary:
                continue
            rc = cat(sp.pdg)
            bm = best_match(sp)
            if bm is None:
                continue
            ov, ii, part_i, typ = bm
            mp = matched_pdg(rec, ii, part_i, typ)
            if mp is None:
                continue
            mcatv = cat(mp)
            # purity sample (reco-denominated): only reco whose category is one of ours
            if rc is not None and mcatv is not None and ov >= OV_MIN:
                mke = true_ke(mp, _true_p(rec, ii, part_i, typ))
                pur[key][rc].append((mke, 1.0 if rc == mcatv else 0.0))
            # efficiency inversion: keep best overlap per true PRIMARY particle
            if typ == K_PRIMARY:
                cur = prim_best.get(part_i)
                if cur is None or ov > cur[0]:
                    prim_best[part_i] = (ov, rc)
    return prim_best

def _true_p(rec, ii, part_i, typ):
    nu = rec.mc.nu[ii]
    tp = nu.prim[part_i] if typ == K_PRIMARY else nu.sec[part_i]
    return vmag(tp.p.px, tp.p.py, tp.p.pz)

for ie in tqdm(range(t.GetEntries()), desc="Scanning"):
    t.GetEntry(ie)
    rec = t.rec
    if rec.mc.nnu < 1:
        continue
    nu = rec.mc.nu[0]
    if not nu.iscc:
        continue
    ixn = rec.common.ixn

    sp_best = process(rec, ixn.dlp, ixn.ndlp, "dlp", "spine") if ixn.ndlp > 0 else {}
    pa_best = process(rec, ixn.pandora, ixn.npandora, "pandora", "pandora") if ixn.npandora > 0 else {}

    # efficiency: loop true primaries of our species
    for i in range(nu.nprim):
        tp = nu.prim[i]
        s = cat(tp.pdg)
        if s is None:
            continue
        ke = true_ke(tp.pdg, vmag(tp.p.px, tp.p.py, tp.p.pz))
        for key, best in [("spine", sp_best), ("pandora", pa_best)]:
            m = best.get(i)
            reconstructed = (m is not None and m[0] >= OV_MIN)
            correct = (reconstructed and m[1] == s)
            eff_reco[key][s].append((ke, 1.0 if reconstructed else 0.0))
            eff_pid[key][s].append((ke, 1.0 if correct else 0.0))
            if correct:
                overlap[key][s].append((ke, m[0]))

# ── binning ───────────────────────────────────────────────────────────────────
ke_bins = np.logspace(np.log10(0.02), np.log10(3.0), 11)
ke_ctr  = np.sqrt(ke_bins[:-1] * ke_bins[1:])

def binned_frac(pairs, bins=ke_bins):
    if not pairs:
        return np.full(len(bins)-1, _nan)
    x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    v, _, _ = _bstat(x, y, "mean", bins=bins)
    c, _, _ = _bstat(x, y, "count", bins=bins)
    return np.where(c >= 15, v, _nan)

# ── figure: 3 rows (eff, purity, overlap) × 5 species ─────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharex=True)
fig.suptitle("Per-particle performance vs true kinetic energy (CC, stored truth matching)",
             fontsize=14)

for c, s in enumerate(SPECIES):
    # row 0: efficiency
    ax = axes[0, c]
    ax.plot(ke_ctr, binned_frac(eff_reco["spine"][s]),   color="#1f77b4", lw=2, marker="o", ms=3, label="SPINE recon")
    ax.plot(ke_ctr, binned_frac(eff_pid["spine"][s]),    color="#1f77b4", lw=2, ls=":",  marker="^", ms=3, label="SPINE +PID")
    ax.plot(ke_ctr, binned_frac(eff_reco["pandora"][s]), color="#2ca02c", lw=2, ls="--", marker="s", ms=3, label="Pandora recon")
    ax.set_title(f"{SP_LABEL[s]}", fontsize=12)
    ax.set_ylim(0, 1.05); ax.set_xscale("log")
    if c == 0:
        ax.set_ylabel("Efficiency")
        ax.legend(fontsize=7, loc="lower right")
    # row 1: purity
    ax = axes[1, c]
    ax.plot(ke_ctr, binned_frac(pur["spine"][s]),   color="#1f77b4", lw=2, marker="o", ms=3, label="SPINE")
    ax.plot(ke_ctr, binned_frac(pur["pandora"][s]), color="#2ca02c", lw=2, ls="--", marker="s", ms=3, label="Pandora")
    ax.set_ylim(0, 1.05); ax.set_xscale("log")
    if c == 0:
        ax.set_ylabel("PID purity")
        ax.legend(fontsize=7, loc="lower right")
    # row 2: mean match overlap
    ax = axes[2, c]
    ax.plot(ke_ctr, binned_frac(overlap["spine"][s]),   color="#1f77b4", lw=2, marker="o", ms=3, label="SPINE")
    ax.plot(ke_ctr, binned_frac(overlap["pandora"][s]), color="#2ca02c", lw=2, ls="--", marker="s", ms=3, label="Pandora")
    ax.set_ylim(0, 1.05); ax.set_xscale("log")
    ax.set_xlabel("True KE [GeV]")
    if c == 0:
        ax.set_ylabel("Mean match overlap")
        ax.legend(fontsize=7, loc="lower right")

# Pandora PID caveat annotation
axes[1, 2].text(0.5, 0.06, "Pandora PID = track(13)/shower(11) only",
                transform=axes[1, 2].transAxes, ha="center", fontsize=7,
                style="italic", color="#2ca02c")

fig.tight_layout()
out = os.path.join(outdir, "particle_effpur.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

# ── summary JSON ──────────────────────────────────────────────────────────────
def plateau(pairs, lo=0.1, hi=1.0):
    """Mean over a representative KE window (avoids the misleading raw mean that
    is dominated by sub-threshold soft particles below the plot range)."""
    v = [p[1] for p in pairs if lo <= p[0] < hi]
    return float(np.mean(v)) if len(v) >= 15 else _nan
summary = {}
print(f"\nPlateau values (true KE 0.1-1.0 GeV):")
print(f"{'species':<8}{'N_true':>8}{'SPINE eff':>11}{'SPINE PID':>11}{'SPINE pur':>11}{'Pand eff':>10}")
for s in SPECIES:
    ntrue = len(eff_reco["spine"][s])
    row = {"n_true": ntrue,
           "spine_eff_plateau": plateau(eff_reco["spine"][s]),
           "spine_pid_eff_plateau": plateau(eff_pid["spine"][s]),
           "spine_purity_plateau": plateau(pur["spine"][s]),
           "pandora_eff_plateau": plateau(eff_reco["pandora"][s])}
    summary[abs(s)] = row
    print(f"{SP_LABEL[s]:<8}{ntrue:>8,}{row['spine_eff_plateau']:>11.3f}"
          f"{row['spine_pid_eff_plateau']:>11.3f}{row['spine_purity_plateau']:>11.3f}{row['pandora_eff_plateau']:>10.3f}")
with open(os.path.join(outdir, "particle_effpur.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"Saved: {os.path.join(outdir,'particle_effpur.json')}")

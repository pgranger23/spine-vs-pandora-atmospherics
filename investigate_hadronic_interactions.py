#!/usr/bin/env python3
"""
Study Q — Fate of hadronically-interacting protons and PID-metric bias.

High-KE protons increasingly get labelled as pions (Study P).  A proton that
undergoes an inelastic hadronic interaction physically looks pion-like (a track
with a hadronic 'star'), and its hits get shared with its daughters — which can
(a) genuinely change the PID and (b) bias our truth-matched PID metric (lower
overlap, ambiguous assignment).  This study characterises exactly what happens.

end_process and daughters are NOT usable in this CAF (garbage enum / empty).  A
primary proton is tagged as HADRONICALLY INTERACTING if it has ≥1 secondary in
nu.sec whose parent-G4ID equals the proton's G4ID and whose PDG is a nucleon,
pion, or nuclear fragment.  ("Non-interacting" = ionises/stops or exits cleanly.)

Outputs:
  - fraction of protons interacting vs true KE
  - proton reco fate (proton/pion/muon/e/unmatched) vs KE, split interacted/not
  - best-match truthOverlap vs KE, interacted/not (metric-bias check)
  - how many interaction daughters are themselves reconstructed (double-count check)

Reads via PyROOT.  Output: figures/hadronic_proton_fate.png,
figures/hadronic_overlap.png, figures/hadronic_interactions.json
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

K_PRIMARY, K_SECONDARY = 1, 3
M_P = 0.938272
OV_MIN = 0.10
HADRON = lambda a: a in (2212, 2112, 211, 111, 321, 130, 311) or a > 1000000000

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

# per-proton records: (KE, interacted, reco_cat or None, overlap, n_daughters, n_daughters_reco)
rec_rows = []

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
    dlp = ixn.dlp[closest(ixn.dlp, ixn.ndlp, nu.vtx)]

    # primary protons: G4ID -> (prim_index, KE)
    prot = {}
    for i in range(nu.nprim):
        tp = nu.prim[i]
        if abs(int(tp.pdg)) != 2212:
            continue
        p = vmag(tp.p.px, tp.p.py, tp.p.pz)
        ke = math.sqrt(p*p + M_P*M_P) - M_P
        prot[int(tp.G4ID)] = (i, ke)

    if not prot:
        continue

    # interaction daughters: proton G4ID -> list of secondary indices (hadronic)
    daughters = {g: [] for g in prot}
    for si in range(int(nu.nsec)):
        s = nu.sec[si]
        par = int(s.parent)
        if par in prot and HADRON(abs(int(s.pdg))):
            daughters[par].append(si)

    # reco->truth linking: true-primary index -> best (overlap, reco_cat);
    #                      true-secondary index -> matched by a reco object?
    prim_match = {}
    sec_matched = set()
    for pi in range(dlp.part.ndlp):
        sp = dlp.part.dlp[pi]
        bm = best_match(sp)
        if bm is None:
            continue
        ov, ii, part, typ = bm
        if ii != 0 or ov < OV_MIN:
            continue
        rc = cat(sp.pdg)
        if typ == K_PRIMARY and sp.primary:
            cur = prim_match.get(part)
            if cur is None or ov > cur[0]:
                prim_match[part] = (ov, rc)
        elif typ == K_SECONDARY:
            sec_matched.add(part)

    for g, (i, ke) in prot.items():
        ndau = len(daughters[g])
        interacted = ndau > 0
        m = prim_match.get(i)
        rc = m[1] if m is not None else None
        ov = m[0] if m is not None else _nan
        ndau_reco = sum(1 for si in daughters[g] if si in sec_matched)
        rec_rows.append((ke, interacted, rc, ov, ndau, ndau_reco))

print(f"  proton records: {len(rec_rows):,}")
KE   = np.array([r[0] for r in rec_rows])
INT  = np.array([r[1] for r in rec_rows], dtype=bool)
RC   = np.array([r[2] if r[2] is not None else -1 for r in rec_rows])
OV   = np.array([r[3] for r in rec_rows])
NDR  = np.array([r[5] for r in rec_rows])

ke_bins = np.logspace(np.log10(0.02), np.log10(2.0), 13)
ke_ctr  = np.sqrt(ke_bins[:-1]*ke_bins[1:])
def prof(mask, val, stat="mean"):
    m = mask & np.isfinite(val)
    if m.sum() < 20:
        return np.full(len(ke_bins)-1, _nan)
    v, _, _ = _bstat(KE[m], val[m], stat, bins=ke_bins)
    c, _, _ = _bstat(KE[m], val[m], "count", bins=ke_bins)
    return np.where(c >= 15, v, _nan)

# ── Figure 1: fraction interacting + proton fate split by interacted ──────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# (a) fraction interacting vs KE
ax = axes[0]
ax.plot(ke_ctr, prof(np.ones_like(INT, dtype=bool), INT.astype(float)), "k-", lw=2.2, marker="o", ms=4)
ax.set_xscale("log"); ax.set_xlabel("True proton KE [GeV]"); ax.set_ylabel("Fraction hadronically interacting")
ax.set_title("Protons undergoing hadronic interaction"); ax.set_ylim(0, 1.02)

# (b) reco-as-pion fraction vs KE, interacted vs not
ax = axes[1]
for lab, mask, col in [("interacted", INT, "#d62728"), ("non-interacting", ~INT, "#1f77b4")]:
    aspi = prof(mask, (RC == 211).astype(float))
    aspro = prof(mask, (RC == 2212).astype(float))
    ax.plot(ke_ctr, aspi, color=col, lw=2.2, marker="o", ms=3, label=f"{lab}: →π")
    ax.plot(ke_ctr, aspro, color=col, lw=2.0, ls="--", marker="s", ms=3, label=f"{lab}: →p (correct)")
ax.set_xscale("log"); ax.set_xlabel("True proton KE [GeV]"); ax.set_ylabel("Fraction of protons")
ax.set_title("Proton PID fate: interacted vs not"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)

# (c) median best-match overlap vs KE, interacted vs not
ax = axes[2]
for lab, mask, col in [("interacted", INT, "#d62728"), ("non-interacting", ~INT, "#1f77b4")]:
    ax.plot(ke_ctr, prof(mask, OV, "median"), color=col, lw=2.2, marker="o", ms=3, label=lab)
ax.axhline(OV_MIN, color="k", ls=":", lw=1.0, label=f"match threshold {OV_MIN}")
ax.set_xscale("log"); ax.set_xlabel("True proton KE [GeV]"); ax.set_ylabel("Median best-match truthOverlap")
ax.set_title("Match quality: interacted vs not"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
fig.suptitle("Study Q — Hadronically-interacting protons: PID fate and matching", fontsize=13)
fig.tight_layout()
out1 = os.path.join(outdir, "hadronic_proton_fate.png")
fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out1}")

# ── summary tables ────────────────────────────────────────────────────────────
def frac(mask, cond):
    m = mask
    return float(cond[m].mean()) if m.sum() > 0 else _nan
summary = {}
print("\nBy KE band and interaction status: N, frac→p, frac→π, frac unmatched, median overlap:")
BANDS = [(0.05,0.15),(0.15,0.3),(0.3,0.6),(0.6,1.5)]
for lo,hi in BANDS:
    band = (KE>=lo)&(KE<hi)
    for lab, mm in [("interact", band&INT), ("no-int", band&~INT)]:
        n = int(mm.sum())
        if n < 20: continue
        fp = frac(mm, RC==2212); fpi = frac(mm, RC==211); fun = frac(mm, RC==-1)
        mov = float(np.nanmedian(OV[mm & np.isfinite(OV)])) if (mm&np.isfinite(OV)).sum()>0 else _nan
        print(f"  KE {lo:.2f}-{hi:.2f} {lab:<9} N={n:>6}  →p={fp:.2f}  →π={fpi:.2f}  unmatched={fun:.2f}  med_ov={mov:.2f}")
    summary[f"{lo:.2f}-{hi:.2f}"] = {
        "interact_frac": frac(band, INT),
        "int_to_pi": frac(band&INT, RC==211), "noint_to_pi": frac(band&~INT, RC==211),
        "int_to_p": frac(band&INT, RC==2212), "noint_to_p": frac(band&~INT, RC==2212),
        "int_unmatched": frac(band&INT, RC==-1), "noint_unmatched": frac(band&~INT, RC==-1),
    }

# metric-bias headline: among protons reco'd as pion, what frac interacted?
as_pi = RC == 211
print(f"\nAmong protons labelled reco-π: {100*INT[as_pi].mean():.1f}% had a hadronic interaction "
      f"(vs {100*INT.mean():.1f}% of all protons).")
# daughters double-count
print(f"Interacting protons: mean {NDR[INT].mean():.2f} of their hadronic daughters are also "
      f"reconstructed as (matched) reco objects.")
summary["reco_pi_interacted_frac"] = float(INT[as_pi].mean())
summary["all_interacted_frac"] = float(INT.mean())
summary["mean_daughters_reco_when_interact"] = float(NDR[INT].mean())

with open(os.path.join(outdir, "hadronic_interactions.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"Saved: {os.path.join(outdir,'hadronic_interactions.json')}")

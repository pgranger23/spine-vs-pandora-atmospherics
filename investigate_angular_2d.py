#!/usr/bin/env python3
"""
2D (per-axis) angular resolution and zenith hemisphere confusion.

The full 3D opening angle is not the most physically meaningful metric for an
isotropic atmospheric sample in a horizontal-drift detector: oscillation
sensitivity depends on the ZENITH angle (measured from the vertical, y).  This
study decomposes the direction error into the three per-axis polar angles:

    θ_a = arccos(d_a)         for a ∈ {x, y, z}   (d = unit direction)
    Δθ_a = θ_a,reco − θ_a,true

and reports the spread (P84-P16)/2 (the paper's ±1σ-equivalent half-width) of
each residual vs Eν, split by flavour, for SPINE (EM-calibrated all-particle
momentum sum) and Pandora (lngtrk for νμ, heshw for νe).  θ_y is the zenith
projection ("mostly Y").

It also reports the ZENITH HEMISPHERE FLIP rate — the fraction of events where
the reconstructed vertical direction points to the wrong hemisphere
(d_y,reco · d_y,true < 0), i.e. up-going / down-going confusion, the single most
important directional failure mode for atmospheric oscillation physics.  This is
the 2D analogue of the Study H2 reversal rate.

Coordinates: y = vertical (zenith axis), x = drift, z ≈ beam; the sample is
isotropic (verified: ⟨d_a⟩≈0, σ≈0.578 on every axis).

Reads via PyROOT (see investigate_multiplicity.py for why).
Output: figures/angular_2d_resolution.png, figures/angular_2d_zenith_flip.png,
        figures/angular_2d.json
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

EM_PDG  = (11, 22, 111)
EM_CORR = 0.198
P_MIN   = 0.5   # MeV/c

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

def polar(dc):
    """arccos(component) in degrees; dc is a unit-vector component in [-1,1]."""
    return math.degrees(math.acos(max(-1.0, min(1.0, dc))))

# ── scan ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("2D per-axis angular resolution + zenith hemisphere flip")
print("=" * 60)

f = ROOT.TFile.Open(caf_env.CAF_FILE)
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
    if not (math.isfinite(tmag) and tmag > 1e-9):
        continue
    tdx, tdy, tdz = tm.x/tmag, tm.y/tmag, tm.z/tmag
    is_numu = abs(int(nu.pdg)) == 14
    E = float(nu.E)

    # SPINE EM-calibrated momentum sum
    si = closest(ixn.dlp, ixn.ndlp, nu.vtx)
    dlp = ixn.dlp[si]
    sx = sy = sz = 0.0
    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        pp = p.p
        pm = vmag(pp.x, pp.y, pp.z)
        if not (math.isfinite(pm) and pm > P_MIN):
            continue
        sc = EM_CORR if abs(int(p.pdg)) in EM_PDG else 1.0
        sx += pp.x * sc; sy += pp.y * sc; sz += pp.z * sc
    smag = vmag(sx, sy, sz)
    if smag < 0.1:
        continue
    sdx, sdy, sdz = sx/smag, sy/smag, sz/smag

    # Pandora (lngtrk for νμ, heshw for νe)
    pdx = pdy = pdz = _nan
    if ixn.npandora > 0:
        pi2 = closest(ixn.pandora, ixn.npandora, nu.vtx)
        pd = ixn.pandora[pi2].dir
        v = pd.lngtrk if is_numu else pd.heshw
        pmag = vmag(v.x, v.y, v.z)
        if math.isfinite(pmag) and pmag > 1e-9:
            pdx, pdy, pdz = v.x/pmag, v.y/pmag, v.z/pmag

    row = {"E": E, "is_numu": is_numu}
    for a, (rd, td) in zip("xyz", [(sdx, tdx), (sdy, tdy), (sdz, tdz)]):
        row[f"sp_d{a}res"] = polar(rd) - polar(td)
    row["sp_yflip"] = 1.0 if (sdy * tdy < 0) else 0.0
    if math.isfinite(pdx):
        for a, (rd, td) in zip("xyz", [(pdx, tdx), (pdy, tdy), (pdz, tdz)]):
            row[f"pa_d{a}res"] = polar(rd) - polar(td)
        row["pa_yflip"] = 1.0 if (pdy * tdy < 0) else 0.0
    else:
        for a in "xyz":
            row[f"pa_d{a}res"] = _nan
        row["pa_yflip"] = _nan
    rows.append(row)

print(f"  CC events: {len(rows):,}")
A = {k: np.array([r.get(k, _nan) for r in rows], dtype=float) for k in rows[0]}
A["is_numu"] = A["is_numu"].astype(bool)

nu_bins = np.logspace(np.log10(0.1), np.log10(10.0), 29)  # finer, log-spaced, to 10 GeV
nu_ctr  = np.sqrt(nu_bins[:-1] * nu_bins[1:])              # geometric-mean centres

def res_half(x, y, bins=nu_bins):
    """Paper resolution metric: (P84 - P16)/2 — the ±1σ-equivalent half-width
    from the 16th/84th percentiles (matches arXiv:2601.05697's manual_std)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.full(len(bins)-1, _nan)
    q84, e, _ = _bstat(x[m], y[m], lambda v: np.percentile(v, 84), bins=bins)
    q16, _, _ = _bstat(x[m], y[m], lambda v: np.percentile(v, 16), bins=bins)
    c, _, _   = _bstat(x[m], y[m], "count", bins=bins)
    val = 0.5 * (q84 - q16)
    return np.where(c >= 10, val, _nan)

def frac(x, y, bins=nu_bins):
    m = np.isfinite(x) & np.isfinite(y)
    v, _, _ = _bstat(x[m], y[m], "mean", bins=bins)
    c, _, _ = _bstat(x[m], y[m], "count", bins=bins)
    return np.where(c >= 10, v, _nan)

# ── Figure 1: per-axis IQR/2, rows=flavour, cols=axis ─────────────────────────
AX_LABELS = {"x": r"$\theta_x$ (drift)", "y": r"$\theta_y$ (zenith)", "z": r"$\theta_z$ (beam)"}
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
fig.suptitle(r"Per-axis polar-angle resolution $(P_{84}-P_{16})/2$ of $\Delta\theta_a=\arccos(d_a^{reco})-\arccos(d_a^{true})$",
             fontsize=13)
for r, (numu, flav) in enumerate([(True, r"$\nu_\mu$ CC"), (False, r"$\nu_e$ CC")]):
    fm = A["is_numu"] if numu else ~A["is_numu"]
    for c, a in enumerate("xyz"):
        ax = axes[r, c]
        x = A["E"][fm]
        ax.plot(nu_ctr, res_half(x, A[f"sp_d{a}res"][fm]), color="#1f77b4", lw=2.0,
                marker="o", ms=3, label="SPINE (all-sum)")
        ax.plot(nu_ctr, res_half(x, A[f"pa_d{a}res"][fm]), color="#2ca02c", lw=2.0,
                ls="--", marker="s", ms=3, label="Pandora")
        if a == "y":
            for s in ax.spines.values():
                s.set_edgecolor("#d62728"); s.set_linewidth(2.0)
        ax.set_title(f"{flav} — {AX_LABELS[a]}", fontsize=10)
        if r == 1: ax.set_xlabel(r"True $E_\nu$ [GeV]")
        if c == 0: ax.set_ylabel(r"$(P_{84}-P_{16})/2$ of $\Delta\theta$ [°]")
        ax.set_xscale("log"); ax.set_xlim(0.1, 10.0); ax.set_ylim(0, 90)
        if r == 0 and c == 0: ax.legend(fontsize=8)
fig.tight_layout()
out1 = os.path.join(outdir, "angular_2d_resolution.png")
fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out1}")

# ── Figure 2: zenith hemisphere flip (up/down confusion) ──────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for numu, flav, cs, cp in [(True, r"$\nu_\mu$", "#1f77b4", "#2ca02c"),
                           (False, r"$\nu_e$", "#9467bd", "#ff7f0e")]:
    fm = A["is_numu"] if numu else ~A["is_numu"]
    x = A["E"][fm]
    ax.plot(nu_ctr, 100*frac(x, A["sp_yflip"][fm]), color=cs, lw=2.0, marker="o",
            ms=4, label=f"SPINE {flav}")
    ax.plot(nu_ctr, 100*frac(x, A["pa_yflip"][fm]), color=cp, lw=2.0, ls="--",
            marker="s", ms=4, label=f"Pandora {flav}")
ax.axhline(50, color="k", lw=0.8, ls=":", label="random (50%)")
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel("Zenith hemisphere flip rate [%]")
ax.set_title(r"Up/down (zenith) confusion: $d_y^{reco}\cdot d_y^{true}<0$")
ax.set_xscale("log"); ax.set_xlim(0.1, 10.0); ax.set_ylim(0, 55); ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
out2 = os.path.join(outdir, "angular_2d_zenith_flip.png")
fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out2}")

# ── summary table + JSON ──────────────────────────────────────────────────────
E_CUTS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
E_LABS = ["E<0.5", "0.5-1", "1-2", ">2"]
summ = {}
print("\nZenith (θ_y) (P84-P16)/2 [°] and hemisphere-flip [%] by flavour and Eν:")
for numu, flav in [(True, "numu"), (False, "nue")]:
    fm = A["is_numu"] if numu else ~A["is_numu"]
    print(f"  {flav}:")
    summ[flav] = {}
    for (lo, hi), lb in zip(E_CUTS, E_LABS):
        m = fm & (A["E"] >= lo) & (A["E"] < hi)
        def half(arr):
            v = arr[m & np.isfinite(arr)]
            return float(0.5*(np.percentile(v,84)-np.percentile(v,16))) if len(v) >= 10 else _nan
        sp_y = half(A["sp_dyres"]); pa_y = half(A["pa_dyres"])
        sp_fl = float(np.nanmean(A["sp_yflip"][m]))*100
        pa_fl = float(np.nanmean(A["pa_yflip"][m]))*100
        summ[flav][lb] = {"spine_thy_iqrhalf": sp_y, "pand_thy_iqrhalf": pa_y,
                          "spine_yflip_pct": sp_fl, "pand_yflip_pct": pa_fl}
        print(f"    {lb:<6} SPINE θ_y={sp_y:5.1f}° flip={sp_fl:4.1f}%   "
              f"Pandora θ_y={pa_y:5.1f}° flip={pa_fl:4.1f}%")

with open(os.path.join(outdir, "angular_2d.json"), "w") as fh:
    json.dump(summ, fh, indent=1)
print(f"Saved: {os.path.join(outdir,'angular_2d.json')}")

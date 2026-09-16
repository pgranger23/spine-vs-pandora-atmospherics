#!/usr/bin/env python3
"""
Study R — Best-of two direction estimators per energy bin, and where the switch
happens.

Two families of neutrino-direction estimator (cf. arXiv:2601.05697 Fig 14):
  • particle/momentum-weighted  — vector sum of reconstructed particle momenta;
  • calorimetric / hits-only     — charge/energy-weighted direction.

Available in this CAF:
  SPINE  : momentum sum  Σ p⃗_i           (EM-calibrated)   [dir.calo is empty]
           calorimetric  Σ (KE_i·emcal)·p̂_i   (energy-weighted proxy)
  Pandora: track/shower  lngtrk(νμ)/heshw(νe)  [particles carry no momentum]
           calorimetric  dir.calo (all-hits)

For each reconstructor we take, per energy bin, the BEST (smallest) median 3D
direction error between its two estimators, and mark the crossover energy where
the calorimetric estimator overtakes the particle/track one.

Reads via PyROOT.  Output: figures/direction_best_of.png, figures/direction_best_of.json
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

def ang3d(vx, vy, vz, tx, ty, tz):
    m = vmag(vx, vy, vz)
    if m < 1e-12:
        return _nan
    return math.degrees(math.acos(max(-1.0, min(1.0, (vx*tx+vy*ty+vz*tz)/m))))

def zen(vx, vy, vz, tdy):
    """Signed zenith residual Δθ_y = arccos(d_y^reco) − arccos(d_y^true) [deg]."""
    m = vmag(vx, vy, vz)
    if m < 1e-12:
        return _nan
    return math.degrees(math.acos(max(-1.0, min(1.0, vy/m))) -
                        math.acos(max(-1.0, min(1.0, tdy))))

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
    if tmag < 1e-9:
        continue
    tdx, tdy, tdz = tm.x/tmag, tm.y/tmag, tm.z/tmag
    is_numu = abs(int(nu.pdg)) == 14
    E = float(nu.E)

    # SPINE: momentum-weighted and energy(calorimetric)-weighted sums
    dlp = ixn.dlp[closest(ixn.dlp, ixn.ndlp, nu.vtx)]
    mx = my = mz = 0.0   # momentum sum
    cx = cy = cz = 0.0   # energy-weighted
    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        pp = p.p
        pm = vmag(pp.x, pp.y, pp.z)
        if not (math.isfinite(pm) and pm > P_MIN):
            continue
        em = EM_CORR if abs(int(p.pdg)) in EM_PDG else 1.0
        mx += pp.x*em; my += pp.y*em; mz += pp.z*em
        ke = float(p.E)               # kinetic energy [MeV] (calorimetric)
        if math.isfinite(ke) and ke > 0:
            w = ke * em / pm          # energy weight applied to the unit direction
            cx += pp.x*w; cy += pp.y*w; cz += pp.z*w

    row = {"E": E, "is_numu": is_numu}
    ok_m = vmag(mx, my, mz) > 0.1
    ok_c = vmag(cx, cy, cz) > 1e-6
    row["sp_mom"]   = ang3d(mx, my, mz, tdx, tdy, tdz) if ok_m else _nan
    row["sp_calo"]  = ang3d(cx, cy, cz, tdx, tdy, tdz) if ok_c else _nan
    row["sp_mom_z"]  = zen(mx, my, mz, tdy) if ok_m else _nan
    row["sp_calo_z"] = zen(cx, cy, cz, tdy) if ok_c else _nan

    # Pandora: track/shower direction and calorimetric (hits) direction
    row["pa_trk"] = row["pa_calo"] = _nan
    row["pa_trk_z"] = row["pa_calo_z"] = _nan
    if ixn.npandora > 0:
        pd = ixn.pandora[closest(ixn.pandora, ixn.npandora, nu.vtx)].dir
        v = pd.lngtrk if is_numu else pd.heshw
        if vmag(v.x, v.y, v.z) > 1e-6:
            row["pa_trk"]   = ang3d(v.x, v.y, v.z, tdx, tdy, tdz)
            row["pa_trk_z"] = zen(v.x, v.y, v.z, tdy)
        c = pd.calo
        if vmag(c.x, c.y, c.z) > 1e-6:
            row["pa_calo"]   = ang3d(c.x, c.y, c.z, tdx, tdy, tdz)
            row["pa_calo_z"] = zen(c.x, c.y, c.z, tdy)
    rows.append(row)

print(f"  CC events: {len(rows):,}")
A = {k: np.array([r.get(k, _nan) for r in rows], dtype=float) for k in rows[0]}

nu_bins = np.logspace(np.log10(0.1), np.log10(10.0), 21)
nu_ctr  = np.sqrt(nu_bins[:-1] * nu_bins[1:])

def med(y, bins=nu_bins):
    x = A["E"]
    m = np.isfinite(x) & np.isfinite(y)
    v, _, _ = _bstat(x[m], y[m], "median", bins=bins)
    c, _, _ = _bstat(x[m], y[m], "count",  bins=bins)
    return np.where(c >= 15, v, _nan)

def crossover(e_ctr, part, calo):
    """First energy where calo becomes <= part (calorimetric overtakes)."""
    d = calo - part
    for i in range(len(d)-1):
        if np.isfinite(d[i]) and np.isfinite(d[i+1]) and d[i] > 0 and d[i+1] <= 0:
            # linear interpolation in log-E
            f = d[i] / (d[i] - d[i+1])
            return e_ctr[i] * (e_ctr[i+1]/e_ctr[i])**f
    return _nan

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
fig.suptitle("Study R — Best of momentum-sum vs calorimetric direction, per energy bin", fontsize=13)

panels = [
    ("SPINE",   "sp_mom",  "momentum sum",       "sp_calo", "calorimetric (energy-wtd)"),
    ("Pandora", "pa_trk",  "lngtrk/heshw",        "pa_calo", "calorimetric (hits)"),
]
summary = {}
best_curves = {}
for ax, (label, kpart, lpart, kcalo, lcalo) in zip(axes, panels):
    part = med(A[kpart]); calo = med(A[kcalo])
    best = np.fmin(part, calo)
    best_curves[label] = best
    ax.plot(nu_ctr, part, color="#1f77b4", lw=1.8, ls="--", marker="o", ms=3, label=lpart)
    ax.plot(nu_ctr, calo, color="#ff7f0e", lw=1.8, ls="--", marker="s", ms=3, label=lcalo)
    ax.plot(nu_ctr, best, color="#000000", lw=2.6, label="best of the two")
    xo = crossover(nu_ctr, part, calo)
    if math.isfinite(xo):
        ax.axvline(xo, color="#d62728", ls=":", lw=1.8)
        ax.text(xo, 5, f" switch\n @ {xo:.1f} GeV", color="#d62728", fontsize=9, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(0.1, 10.0); ax.set_ylim(0, None)
    ax.set_xlabel(r"True $E_\nu$ [GeV]")
    ax.set_title(f"{label}   (switch @ {xo:.1f} GeV)" if math.isfinite(xo) else f"{label}   (no switch)")
    ax.legend(fontsize=8.5)
    summary[label] = {"crossover_GeV": xo,
                      "part_median": [float(v) for v in part],
                      "calo_median": [float(v) for v in calo]}
axes[0].set_ylabel("Median 3D direction error [°]")
fig.tight_layout()
out = os.path.join(outdir, "direction_best_of.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

# ── overlay: SPINE best vs Pandora best ───────────────────────────────────────
fig2, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(nu_ctr, best_curves["SPINE"],   color="#1f77b4", lw=2.6, marker="o", ms=4, label="SPINE (best of the two)")
ax.plot(nu_ctr, best_curves["Pandora"], color="#ff7f0e", lw=2.6, marker="s", ms=4, label="Pandora (best of the two)")
ax.set_xscale("log"); ax.set_xlim(0.1, 10.0); ax.set_ylim(0, None)
ax.set_xlabel(r"True $E_\nu$ [GeV]"); ax.set_ylabel("Median 3D direction error [°]")
ax.set_title("Best achievable direction: SPINE vs Pandora (best of momentum-sum / calorimetric)")
ax.legend(fontsize=10)
fig2.tight_layout()
out_ov = os.path.join(outdir, "direction_best_overlay.png")
fig2.savefig(out_ov, dpi=150, bbox_inches="tight"); plt.close(fig2)
print(f"Saved: {out_ov}")

# ═══ ZENITH-ONLY versions (metric = (P84−P16)/2 of Δθ_y) ══════════════════════
def res_half(y, bins=nu_bins):
    x = A["E"]
    m = np.isfinite(x) & np.isfinite(y)
    q84, _, _ = _bstat(x[m], y[m], lambda v: np.percentile(v, 84), bins=bins)
    q16, _, _ = _bstat(x[m], y[m], lambda v: np.percentile(v, 16), bins=bins)
    c,   _, _ = _bstat(x[m], y[m], "count", bins=bins)
    return np.where(c >= 15, 0.5*(q84 - q16), _nan)

panels_z = [
    ("SPINE",   "sp_mom_z",  "momentum sum",  "sp_calo_z", "calorimetric (energy-wtd)"),
    ("Pandora", "pa_trk_z",  "lngtrk/heshw",  "pa_calo_z", "calorimetric (hits)"),
]
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
fig3.suptitle(r"Study R (zenith) — Best of momentum-sum vs calorimetric for $\theta_y$, per energy bin", fontsize=13)
best_z = {}
for ax, (label, kpart, lpart, kcalo, lcalo) in zip(axes3, panels_z):
    part = res_half(A[kpart]); calo = res_half(A[kcalo])
    best = np.fmin(part, calo); best_z[label] = best
    ax.plot(nu_ctr, part, color="#1f77b4", lw=1.8, ls="--", marker="o", ms=3, label=lpart)
    ax.plot(nu_ctr, calo, color="#ff7f0e", lw=1.8, ls="--", marker="s", ms=3, label=lcalo)
    ax.plot(nu_ctr, best, color="#000000", lw=2.6, label="best of the two")
    xo = crossover(nu_ctr, part, calo)
    if math.isfinite(xo):
        ax.axvline(xo, color="#d62728", ls=":", lw=1.8)
        ax.text(xo, 3, f" switch\n @ {xo:.1f} GeV", color="#d62728", fontsize=9, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(0.1, 10.0); ax.set_ylim(0, None)
    ax.set_xlabel(r"True $E_\nu$ [GeV]")
    ax.set_title(f"{label}   (switch @ {xo:.1f} GeV)" if math.isfinite(xo) else f"{label}   (no switch)")
    ax.legend(fontsize=8.5)
    summary.setdefault(label, {})["zenith_crossover_GeV"] = xo
axes3[0].set_ylabel(r"$(P_{84}-P_{16})/2$ of $\Delta\theta_y$ [°]")
fig3.tight_layout()
out_z = os.path.join(outdir, "direction_best_of_zenith.png")
fig3.savefig(out_z, dpi=150, bbox_inches="tight"); plt.close(fig3)
print(f"Saved: {out_z}")

fig4, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(nu_ctr, best_z["SPINE"],   color="#1f77b4", lw=2.6, marker="o", ms=4, label="SPINE (best of the two)")
ax.plot(nu_ctr, best_z["Pandora"], color="#ff7f0e", lw=2.6, marker="s", ms=4, label="Pandora (best of the two)")
ax.set_xscale("log"); ax.set_xlim(0.1, 10.0); ax.set_ylim(0, None)
ax.set_xlabel(r"True $E_\nu$ [GeV]"); ax.set_ylabel(r"$(P_{84}-P_{16})/2$ of $\Delta\theta_y$ [°]")
ax.set_title("Best achievable zenith resolution: SPINE vs Pandora")
ax.legend(fontsize=10)
fig4.tight_layout()
out_zov = os.path.join(outdir, "direction_best_overlay_zenith.png")
fig4.savefig(out_zov, dpi=150, bbox_inches="tight"); plt.close(fig4)
print(f"Saved: {out_zov}")

# ── table ─────────────────────────────────────────────────────────────────────
E_CUTS = [(0,0.5),(0.5,1),(1,2),(2,5),(5,1e9)]; E_LABS=["E<0.5","0.5-1","1-2","2-5",">5"]
print(f"\nMedian 3D direction error [°] by estimator and Eν:")
for label, kpart, lpart, kcalo, lcalo in panels:
    print(f"  {label}: {'':>18}" + "".join(f"{lb:>8}" for lb in E_LABS))
    for k, ln in [(kpart, lpart), (kcalo, lcalo)]:
        line = f"    {ln:<24}"
        for (lo,hi) in E_CUTS:
            m = (A["E"]>=lo)&(A["E"]<hi)&np.isfinite(A[k])
            line += f"{np.median(A[k][m]) if m.sum()>=15 else float('nan'):>8.1f}"
        print(line)
    print(f"    -> crossover @ {summary[label]['crossover_GeV']:.2f} GeV")

with open(os.path.join(outdir, "direction_best_of.json"), "w") as fh:
    json.dump({k: {"crossover_GeV": v["crossover_GeV"]} for k, v in summary.items()}, fh, indent=1)
print(f"Saved: {os.path.join(outdir,'direction_best_of.json')}")

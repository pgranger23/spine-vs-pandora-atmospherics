#!/usr/bin/env python3
"""
Proton momentum cutoff scan.

Goal: Determine the optimal low-momentum proton threshold for angular
resolution. Perform a scan of angular resolution vs minimum true proton
momentum to find the inflection point (the "elbow") where resolution
degrades significantly.

Background: Proton angular resolution goes from 3.6° (>500 MeV) to
42° (<50 MeV). This study identifies the recommended tracking threshold.

Output: figures/proton_cutoff_scan.png, summary table
"""

import os, sys, ctypes, math, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import binned_statistic
from tqdm import tqdm

import caf_env
pySRProxy = caf_env.setup_pysrproxy()

outdir = "figures"
os.makedirs(outdir, exist_ok=True)

import plot_style
plot_style.apply()

_nan = float("nan")

def _fmt_log_ticks(ax):
    plot_style.fmt_log_ticks(ax)

# Particle masses (GeV)
MASS = {
    2212: 0.938272,  # proton
    2112: 0.939565,  # neutron
    11: 0.000511,    # electron
    13: 0.105658,    # muon
    211: 0.139570,   # pion
    111: 0.134977,   # pi0
    22: 0.0,         # photon
}

def _safe_norm(x, y, z):
    n = math.sqrt(x*x + y*y + z*z)
    return (x/n, y/n, z/n) if n > 0 else (_nan, _nan, _nan)

def _closest_dlp_idx(ixn, nu_vtx):
    best, best_i = float("inf"), 0
    nvx, nvy, nvz = nu_vtx.x, nu_vtx.y, nu_vtx.z
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = math.sqrt((v.x-nvx)**2 + (v.y-nvy)**2 + (v.z-nvz)**2)
        if d < best:
            best, best_i = d, i
    return best_i

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
print("=" * 60)
print("Loading CAF data for proton cutoff scan")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader('recTree', caf_env.CAF_FILE)

particle_records = []

for ientry in tqdm(range(prox.entries()), desc="Scanning"):
    rec = prox.entry(ientry)
    mc = rec.mc
    if not mc.nu[0].iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu = mc.nu[0]
    vtx = nu.vtx
    nu_vtx = np.array([vtx.x, vtx.y, vtx.z])

    # Find closest DLP interaction
    si = _closest_dlp_idx(ixn, vtx)
    dlp = ixn.dlp[si]

    # Loop over true primaries
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        if abs(tp.pdg) != 2212:
            continue
        # True direction from momentum vector
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        true_p = math.sqrt(px*px + py*py + pz*pz)  # GeV/c
        if true_p < 0.01:  # < 10 MeV/c — discard essentially-at-rest protons
            continue
        true_dir = np.array(_safe_norm(px, py, pz))

        # Best-matched SPINE particle via geometric match
        best_cos = -2
        best_reco_dir = None
        best_reco_pdg = None
        best_reco_ke = _nan
        for pi in range(dlp.part.ndlp):
            p = dlp.part.dlp[pi]
            rpx, rpy, rpz = p.p.x, p.p.y, p.p.z
            rp_mag = math.sqrt(rpx*rpx + rpy*rpy + rpz*rpz)
            if rp_mag < 1:
                continue
            reco_dir = np.array(_safe_norm(rpx, rpy, rpz))
            if np.any(~np.isfinite(reco_dir)):
                continue
            cos_ang = np.dot(true_dir, reco_dir)
            if cos_ang > best_cos:
                best_cos = cos_ang
                best_reco_dir = reco_dir
                best_reco_pdg = int(p.pdg)
                best_reco_ke = p.E * 0.001  # MeV → GeV

        if best_reco_dir is not None and best_cos > 0.5:
            ang_deg = math.degrees(math.acos(max(-1, min(1, best_cos))))
        else:
            ang_deg = _nan

        particle_records.append({
            "true_p": true_p,  # GeV/c (tp.p.px is stored in GeV/c in this CAF)
            "ang_deg": ang_deg,
            "reco_pdg": best_reco_pdg,
            "reco_ke_gev": best_reco_ke,
        })

print(f"\n  Matched protons: {len(particle_records)}")

arr = {k: np.array([r[k] for r in particle_records]) for k in ["true_p", "ang_deg", "reco_pdg", "reco_ke_gev"]}
valid = np.isfinite(arr["ang_deg"]) & (arr["true_p"] > 0)
p_true = arr["true_p"][valid]
ang = arr["ang_deg"][valid]

print(f"  Valid matched protons: {len(p_true)}")

# ---------------------------------------------------------------
# Figure 1: Angular resolution vs true momentum (scatter + profile)
# ---------------------------------------------------------------
print("\n--- Figure 1: Angular resolution vs proton momentum ---")
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle("Proton angular resolution vs true momentum",
              fontsize=14, fontweight="bold")

# Scatter
ax = axes1[0]
ax.scatter(p_true, ang, s=1, alpha=0.2, rasterized=True, c="C0")
ax.set_xscale("log"); _fmt_log_ticks(ax)
ax.set_xlabel(r"True proton $p$ [GeV/$c$]")
ax.set_ylabel(r"$\Delta\theta$ [deg]")
ax.set_title("Per-proton angular residual")
ax.set_xlim(0.01, 10)

# Profile (median + IQR)
ax = axes1[1]
p_bins = np.logspace(np.log10(0.02), np.log10(5), 20)
centers = 0.5 * (p_bins[:-1] + p_bins[1:])
med, _, _ = binned_statistic(p_true, ang, "median", bins=p_bins)
p84 = binned_statistic(p_true, ang, lambda x: np.percentile(x, 84), bins=p_bins)[0]
p16 = binned_statistic(p_true, ang, lambda x: np.percentile(x, 16), bins=p_bins)[0]
cnt, _, _ = binned_statistic(p_true, ang, "count", bins=p_bins)
mk = cnt > 3
ax.errorbar(centers[mk], med[mk],
            yerr=[(med[mk]-p16[mk]), (p84[mk]-med[mk])],
            marker="o", ms=6, color="C0", label="Median ± IQR/2", capsize=3)
ax.axhline(10, color="r", ls="--", lw=1, alpha=0.7, label="10° threshold")
ax.axhline(20, color="orange", ls=":", lw=1, alpha=0.7, label="20° threshold")
ax.set_xscale("log"); _fmt_log_ticks(ax)
ax.set_xlabel(r"True proton $p$ [GeV/$c$]")
ax.set_ylabel(r"$\Delta\theta$ [deg]")
ax.set_title("Angular resolution profile")
ax.set_xlim(0.01, 10)
ax.legend(fontsize=8)

# CDF for different momentum ranges
ax = axes1[2]
p_ranges = [
    (0, 0.05, "<50 MeV", "C0"),
    (0.05, 0.1, "50-100 MeV", "C1"),
    (0.1, 0.3, "100-300 MeV", "C2"),
    (0.3, 0.5, "300-500 MeV", "C3"),
    (0.5, 10, ">500 MeV", "C4"),
]
for p_min, p_max, lbl, col in p_ranges:
    mask = (p_true >= p_min) & (p_true < p_max)
    if mask.sum() < 5:
        continue
    sub = ang[mask]
    s = np.sort(sub[np.isfinite(sub)])
    cdf = np.arange(1, len(s)+1)/len(s)
    ax.plot(s, cdf, label=f"{lbl} (N={mask.sum()}, med={np.median(sub):.1f}°)", color=col, lw=1.3)
ax.set_xlabel(r"$\Delta\theta$ [deg]")
ax.set_ylabel("CDF")
ax.set_title("Angular CDF by momentum range")
ax.legend(fontsize=7)

try:
    fig1.tight_layout()
except ValueError:
    print("  Warning: tight_layout failed")
try:
    fig1.savefig(os.path.join(outdir, "proton_angular_vs_momentum.png"), dpi=150)
except ValueError:
    print("  Warning: savefig failed — setting explicit axis limits")
    # Fix: set explicit x-limits to avoid log-scale issues
    for ax_i in [0, 1]:
        ax = axes1[ax_i]
        xlim = (0.02, 5)
        ax.set_xlim(xlim)
    try:
        fig1.tight_layout()
        fig1.savefig(os.path.join(outdir, "proton_angular_vs_momentum.png"), dpi=150)
    except (ValueError, RuntimeError) as e2:
        print(f"  Still failing: {e2}")
        try:
            fig1.savefig(os.path.join(outdir, "proton_angular_vs_momentum.png"), dpi=150)
        except Exception as e3:
            print(f"  Cannot save Figure 1: {e3}")
plt.close(fig1)

# ---------------------------------------------------------------
# Figure 2: Cutoff scan — angular resolution vs min p threshold
# ---------------------------------------------------------------
print("\n--- Figure 2: Cutoff scan ---")
fig2, ax = plt.subplots(figsize=(10, 7))
fig2.suptitle("Proton momentum cutoff scan: resolution vs minimum momentum threshold",
              fontsize=13, fontweight="bold")

cut_values = np.linspace(0, 0.5, 51)  # 0 to 500 MeV
med_angs = []
p84_vals = []
p16_vals = []
n_above = []

for cut in cut_values:
    mask = p_true >= cut
    n_above.append(mask.sum())
    if mask.sum() < 5:
        med_angs.append(_nan)
        p84_vals.append(_nan)
        p16_vals.append(_nan)
    else:
        sub = ang[mask]
        med_angs.append(np.median(sub))
        p84_vals.append(np.percentile(sub, 84))
        p16_vals.append(np.percentile(sub, 16))

cut_valid = np.array([i for i, v in enumerate(med_angs) if np.isfinite(v)])
med_arr = np.array(med_angs)
p84_arr = np.array(p84_vals)
p16_arr = np.array(p16_vals)

# Dual-axis: angular resolution (primary) + N surviving (secondary)
ax.plot(cut_values[cut_valid]*1000, med_arr[cut_valid], "o-", color="C0", ms=4, label="Median")
ax.fill_between(cut_values[cut_valid]*1000,
                p16_arr[cut_valid], p84_arr[cut_valid],
                alpha=0.2, color="C0", label="16%-84%")
ax.axhline(10, color="r", ls="--", lw=1, alpha=0.7)
ax.axhline(20, color="orange", ls=":", lw=1, alpha=0.7)
ax.set_xlabel("Minimum true proton momentum [MeV/c]")
ax.set_ylabel(r"$\Delta\theta$ [deg]")
ax.set_title("Angular resolution vs momentum cutoff")
ax.legend(fontsize=9, loc="upper left")

ax2 = ax.twinx()
ax2.plot(cut_values[cut_valid]*1000, np.array(n_above)[cut_valid], "s-", color="gray", ms=3, alpha=0.7)
ax2.set_ylabel("N protons above cutoff", color="gray")
ax2.tick_params(axis='y', labelcolor="gray")

fig2.tight_layout()
fig2.savefig(os.path.join(outdir, "proton_cutoff_scan.png"), dpi=150)
plt.close(fig2)

# ---------------------------------------------------------------
# Figure 3: Proton PID mis-ID — 2D efficiency heatmap
# ---------------------------------------------------------------
print("\n--- Figure 3: Proton PID mis-ID 2D heatmap ---")
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6.8))
fig3.suptitle("Proton PID mis-identification: true momentum vs reconstructed KE",
              fontsize=14, fontweight="bold")
fig3.text(0.5, 0.005,
          "★  Full Production 46.4k CC Event Scan  |  PID confusion matrix (5k subsample) in analysis_per_particle",
          ha="center", va="bottom", fontsize=9, fontstyle="italic", color="#8b0000",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeeba", edgecolor="#e0a800", alpha=0.95))

# Determine correct ID: reco PDG == 2212 (proton)
valid_p = np.isfinite(arr["true_p"]) & np.isfinite(arr["reco_ke_gev"]) & (arr["true_p"] > 0) & (arr["reco_ke_gev"] > 0)
p_true_all = arr["true_p"][valid_p]
p_ke_reco = arr["reco_ke_gev"][valid_p]
is_correct_pid = np.abs(arr["reco_pdg"][valid_p]) == 2212
reco_pdg_vals = arr["reco_pdg"][valid_p]

# Panel a: 2D histogram of all matched protons (true p vs reco KE)
ax = axes3[0]
pbins = np.logspace(np.log10(0.02), np.log10(5), 30)
kebins = np.logspace(np.log10(0.02), np.log10(5), 30)
h = ax.hist2d(p_true_all, p_ke_reco, bins=[pbins, kebins], norm="log",
              cmap="Blues", cmin=1)
fig3.colorbar(h[3], ax=ax, label="Counts (log)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"True proton $p$ [GeV/$c$]")
ax.set_ylabel(r"Reco proton KE [GeV]")
ax.set_title(f"All matched protons (N={len(p_true_all):,})")

# Panel b: 2D efficiency heatmap (correct / total)
ax = axes3[1]
# Bin in 2D
correct_2d, _, _ = np.histogram2d(p_true_all[is_correct_pid], p_ke_reco[is_correct_pid],
                                   bins=[pbins, kebins])
total_2d, _, _ = np.histogram2d(p_true_all, p_ke_reco, bins=[pbins, kebins])
efficiency = np.divide(correct_2d, total_2d, out=np.zeros_like(correct_2d, dtype=float),
                       where=total_2d > 3)
# Mask low-stat bins
efficiency_masked = np.ma.masked_where(total_2d <= 3, efficiency)
im = ax.pcolormesh(pbins, kebins, efficiency_masked.T, cmap="RdYlGn", vmin=0, vmax=1,
                   shading="auto")
fig3.colorbar(im, ax=ax, label="Proton ID efficiency")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"True proton $p$ [GeV/$c$]")
ax.set_ylabel(r"Reco proton KE [GeV]")
ax.set_title("Proton PID efficiency (correct / total per bin)")
# Add reference regions
ax.axhline(0.1, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.axvline(0.1, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.text(0.03, 0.5, "Low p / high KE\n→ mis-ID region", fontsize=8, color="red",
        transform=ax.transAxes, ha="left", va="center")

fig3.tight_layout()
fig3.savefig(os.path.join(outdir, "proton_pid_efficiency.png"), dpi=150)
plt.close(fig3)
print("  Saved proton_pid_efficiency.png")

# Print PID summary
n_correct = is_correct_pid.sum()
n_total = len(p_true_all)
print(f"  Proton PID efficiency: {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")
misid_pdgs, counts = np.unique(reco_pdg_vals[~is_correct_pid], return_counts=True)
print(f"  Mis-ID'd as:")
for pdg_val, cnt in sorted(zip(misid_pdgs, counts), key=lambda x: -x[1]):
    print(f"    PDG {int(pdg_val):>5}: {cnt:>6} ({cnt/n_total*100:.1f}%)")

# ---------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Proton Cutoff Scan Results")
print("=" * 60)

print(f"\n  {'Cut [MeV]':>10} {'N prot':>8} {'Median [deg]':>12} {'16% [deg]':>10} {'84% [deg]':>10}")
for i in range(0, len(cut_values), 10):
    if i >= len(cut_valid):
        break
    idx = cut_valid[i] if i < len(cut_valid) else cut_valid[-1]
    print(f"  {cut_values[idx]*1000:>8.0f}   {n_above[idx]:>6}   {med_arr[idx]:>10.1f}   {p16_arr[idx]:>8.1f}   {p84_arr[idx]:>8.1f}")

# Save results
out = {
    "cut_scan": [{"cut_mev": float(cut_values[i]*1000), "n_above": int(n_above[i]),
                   "median_deg": float(med_angs[i]) if np.isfinite(med_angs[i]) else None,
                   "p16_deg": float(p16_vals[i]) if np.isfinite(p16_vals[i]) else None,
                   "p84_deg": float(p84_vals[i]) if np.isfinite(p84_vals[i]) else None}
                  for i in range(0, len(cut_values), 5)],
    "n_protons_total": int(len(p_true)),
}
with open(os.path.join(outdir, "proton_cutoff_results.json"), "w") as f:
    json.dump(out, f, indent=2)

print(f"\n  Results saved to figures/proton_cutoff_results.json")
print("Done!")

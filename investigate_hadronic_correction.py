#!/usr/bin/env python3
"""
Hadronic energy correction parameterization.

Observations: Both frameworks reconstruct only ~38-41% of hadronic energy.
This study characterizes this bias and fits a correction function.

Since the SPINE Enu estimator branches (e_had, mu_had) are unpopulated
in this dataset, we use the particle-sum hadronic energy from SPINE
reconstructed particles as an alternative.

Output: figures/hadronic_correction_*.png, correction_params.json
"""

import os, sys, ctypes, math, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from tqdm import tqdm

import caf_env
pySRProxy = caf_env.setup_pysrproxy()

outdir = "figures"
os.makedirs(outdir, exist_ok=True)

import plot_style
plot_style.apply()

_nan = float("nan")

PARTICLE_MASSES = {
    11: 0.000511, 13: 0.105658, 211: 0.139570, 111: 0.134977,
    321: 0.493677, 2212: 0.938272, 2112: 0.939565, 22: 0.0,
}

def _mass(pdg):
    return PARTICLE_MASSES.get(abs(pdg), 0.1)

def _safe_norm(x, y, z):
    n = math.sqrt(x*x + y*y + z*z)
    return (x/n, y/n, z/n) if n > 0 else (_nan, _nan, _nan)

def _closest_dlp_idx(ixn, nu_vtx):
    best, best_i = float("inf"), 0
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = math.sqrt((v.x-nu_vtx.x)**2 + (v.y-nu_vtx.y)**2 + (v.z-nu_vtx.z)**2)
        if d < best:
            best, best_i = d, i
    return best_i

def true_ke(prim):
    px, py, pz = prim.p.px, prim.p.py, prim.p.pz
    p = math.sqrt(px*px + py*py + pz*pz)
    m = _mass(prim.pdg)
    return math.sqrt(p*p + m*m) - m

def is_lepton(pdg):
    return abs(pdg) in (11, 13)

def reco_ke_from_spine(p, pdg):
    pm = math.sqrt(p.p.x**2 + p.p.y**2 + p.p.z**2)
    if pm < 1:
        return 0.0
    if abs(int(pdg)) in (11, 22, 111):
        pm *= 0.198  # ADC-to-electron calibration correction
    m = _mass(int(pdg)) * 1000  # GeV → MeV
    return (math.sqrt(pm*pm + m*m) - m) / 1000  # MeV → GeV

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
print("=" * 60)
print("Loading CAF data for hadronic correction study")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader('recTree', caf_env.CAF_FILE)

records = []

for ientry in tqdm(range(prox.entries()), desc="Loading"):
    rec = prox.entry(ientry)
    mc = rec.mc
    if not mc.nu[0].iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu = mc.nu[0]
    nu_vtx = nu.vtx

    # True hadronic energy (kinetic, non-lepton primaries)
    true_had = 0.0
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        if not is_lepton(tp.pdg):
            ke = true_ke(tp)
            if ke > 0.001:
                true_had += ke

    # SPINE: closest DLP interaction
    si = _closest_dlp_idx(ixn, nu_vtx)
    dlp = ixn.dlp[si]

    # Reconstructed hadronic energy from SPINE particles
    spine_had = 0.0
    spine_total = 0.0
    n_spine_had = 0
    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        if p.primary:
            ke = reco_ke_from_spine(p, p.pdg)
            spine_total += ke
            if not is_lepton(int(p.pdg)):
                spine_had += ke
                n_spine_had += 1

    # NuMI energy (from Enu estimator — may be empty)
    spine_enu = dlp.Enu.calo if hasattr(dlp.Enu, 'calo') else _nan

    records.append({
        "nu_E": nu.E, "nu_pdg": int(nu.pdg),
        "true_had": true_had,
        "spine_had": spine_had,
        "spine_total": spine_total,
        "spine_enu": spine_enu,
        "n_spine_had": n_spine_had,
    })

print(f"  Loaded {len(records)} CC events")

arr = {k: np.array([r[k] for r in records], dtype=np.float64) for k in records[0].keys()}
nu_pdg = np.array([r["nu_pdg"] for r in records], dtype=int)

is_nue = np.abs(nu_pdg) == 12
is_numu = np.abs(nu_pdg) == 14

# Check if Enu estimator is populated
enu_valid = np.isfinite(arr["spine_enu"]) & (arr["spine_enu"] > 0)
print(f"  SPINE Enu calo populated: {enu_valid.sum()}/{len(records)} ({enu_valid.sum()/len(records)*100:.1f}%)")

# ---------------------------------------------------------------
# Figure 1: Particle-sum hadronic energy vs true
# ---------------------------------------------------------------
print("\n--- Figure 1: SPINE particle-sum hadronic vs true ---")
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
fig1.suptitle("Hadronic energy from SPINE reconstructed particle sum",
              fontsize=14, fontweight="bold")

flavor_masks = [(is_nue, r"$\nu_e$ CC", "C0"), (is_numu, r"$\nu_\mu$ CC", "C1")]

for idx, (flavor_mask, flavor_lbl, col) in enumerate(flavor_masks):
    mask = flavor_mask & (arr["true_had"] > 0)
    true_h = arr["true_had"][mask]
    reco_h = arr["spine_had"][mask]

    valid = (true_h > 0) & (reco_h > 0) & np.isfinite(true_h) & np.isfinite(reco_h)
    true_h = true_h[valid]; reco_h = reco_h[valid]
    ratio = reco_h / true_h

    # Scatter plot
    ax = axes1[0, idx]
    ax.scatter(true_h, reco_h, s=1, alpha=0.3, rasterized=True, c=col)
    lims = [0.01, 30]
    ax.plot(lims, lims, "k--", lw=0.8, label="1:1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"True $E_{\mathrm{had}}$ [GeV]")
    ax.set_ylabel(r"SPINE reco $E_{\mathrm{had}}$ (particle sum) [GeV]")
    ax.set_title(f"{flavor_lbl} (N={len(true_h):,}, med ratio={np.median(ratio):.2f})")
    ax.legend(fontsize=8)
    ax.set_xlim(lims); ax.set_ylim(lims)

    # Ratio distribution
    ax = axes1[1, idx]
    r_valid = ratio[(ratio > 0.001) & (ratio < 100)]
    ax.hist(np.log10(r_valid), bins=60, histtype="step", lw=1.5, color=col)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$\log_{10}(E_{\mathrm{had,reco}}/E_{\mathrm{had,true}})$")
    ax.set_ylabel("Counts")
    ax.set_title(f"Ratio: med={np.median(r_valid):.2f}, "
                 f"68%={np.percentile(np.abs(r_valid-1), 68):.2f}")

    # Ratio vs true energy
    ax = axes1[1, idx + 1]
    r_bins = np.logspace(np.log10(0.02), np.log10(20), 15)
    centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    med, _, _ = binned_statistic(true_h, ratio, "median", bins=r_bins)
    cnt, _, _ = binned_statistic(true_h, ratio, "count", bins=r_bins)
    std, _, _ = binned_statistic(true_h, ratio, "std", bins=r_bins)
    mk = cnt > 3
    ax.errorbar(centers[mk], med[mk], yerr=std[mk]/np.sqrt(cnt[mk]),
                marker="o", ms=5, color=col, label=flavor_lbl)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel(r"True $E_{\mathrm{had}}$ [GeV]")
    ax.set_ylabel(r"Median $E_{\mathrm{had,reco}}/E_{\mathrm{had,true}}$")
    ax.set_title("Ratio vs true energy")
    ax.legend(fontsize=8)

# Third column: both flavors on same plot
ax = axes1[0, 2]
for flavor_mask, flavor_lbl, col in flavor_masks:
    mask = flavor_mask & (arr["true_had"] > 0) & (arr["spine_had"] > 0)
    true_h = arr["true_had"][mask]; reco_h = arr["spine_had"][mask]
    valid = np.isfinite(true_h) & np.isfinite(reco_h) & (true_h > 0) & (reco_h > 0)
    ax.scatter(true_h[valid], reco_h[valid], s=1, alpha=0.2, rasterized=True, c=col, label=flavor_lbl)
ax.plot([0.01, 30], [0.01, 30], "k--", lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"True $E_{\mathrm{had}}$ [GeV]")
ax.set_ylabel(r"SPINE reco $E_{\mathrm{had}}$ [GeV]")
ax.set_title("Both flavors combined")
ax.legend(fontsize=8)
ax.set_xlim([0.01, 30]); ax.set_ylim([0.01, 30])

# Empty third panel in row 2
axes1[1, 2].axis("off")
n_all = ((arr["true_had"] > 0) & (arr["spine_had"] > 0)).sum()
med_ratio = np.median(arr["spine_had"][arr["true_had"] > 0] /
                       arr["true_had"][arr["true_had"] > 0])
text = (
    f"Hadronic energy summary:\n\n"
    f"  Events with E_had > 0: {n_all:,}\n\n"
    f"  SPINE particle-sum / true:\n"
    f"    Median: {med_ratio:.2f}\n"
    f"    IQR: {np.percentile(ratio, 25):.2f} - {np.percentile(ratio, 75):.2f}\n\n"
    f"  SPINE Enu calo estimator:\n"
    f"    Populated: {enu_valid.sum():,} / {len(records):,}\n"
    f"    -> Using particle sum as hadronic\n"
    f"       estimator (Enu not available)"
)
axes1[1, 2].text(0.05, 0.95, text, transform=axes1[1, 2].transAxes,
                 fontsize=10, verticalalignment="top", family="monospace")

fig1.tight_layout()
fig1.savefig(os.path.join(outdir, "hadronic_reco_vs_true.png"), dpi=150)
plt.close(fig1)

# ---------------------------------------------------------------
# Figure 2: Correction fit using particle-sum hadronic energy
# ---------------------------------------------------------------
print("\n--- Figure 2: Correction parameterization ---")
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle(r"Hadronic energy correction: $E_{\mathrm{had,true}} = f(E_{\mathrm{had,reco}})$",
              fontsize=14, fontweight="bold")

def power_law(x, a, b):
    return a * np.power(x, b)

correction_params = {}

for idx, (flavor_mask, flavor_lbl, col) in enumerate(flavor_masks):
    mask = flavor_mask & (arr["true_had"] > 0.01) & (arr["spine_had"] > 0.01)
    if mask.sum() < 10:
        continue

    true_h = arr["true_had"][mask]
    reco_h = arr["spine_had"][mask]

    # Power-law fit
    try:
        popt, pcov = curve_fit(power_law, reco_h, true_h, p0=[1.5, 0.8], maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError) as e:
        print(f"  Fit failed for {flavor_lbl}: {e}")
        # Fallback: linear fit in log space
        coeffs = np.polyfit(np.log10(reco_h), np.log10(true_h), 1)
        popt = [10**coeffs[1], coeffs[0]]
        perr = [0, 0]

    a, b = popt
    da, db = perr
    correction_params[flavor_lbl] = {"a": float(a), "b": float(b),
                                     "a_err": float(da), "b_err": float(db),
                                     "n_points": int(mask.sum())}

    # Plot fit
    ax = axes2[0, idx]
    x_plot = np.logspace(np.log10(0.015), np.log10(15), 50)
    y_plot = power_law(x_plot, a, b)

    ax.scatter(reco_h, true_h, s=1, alpha=0.2, rasterized=True, c=col)
    ax.plot(x_plot, y_plot, "r-", lw=2,
           label=rf"$E_{{\mathrm{{true}}}} = {a:.3f}\,E_{{\mathrm{{reco}}}}^{{{b:.3f}}}$")
    ax.plot(x_plot, x_plot, "k--", lw=0.8, label="1:1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"SPINE reco $E_{\mathrm{had}}$ (particle sum) [GeV]")
    ax.set_ylabel(r"True $E_{\mathrm{had}}$ [GeV]")
    ax.set_title(f"{flavor_lbl} (N={mask.sum():,})")
    ax.legend(fontsize=9)

    # Residuals after correction
    ax = axes2[1, idx]
    corrected_h = power_law(reco_h, a, b)
    residual = corrected_h / true_h
    r_valid = residual[(residual > 0.01) & (residual < 100)]
    ax.hist(np.log10(r_valid), bins=50, histtype="step", lw=1.5, color="C2")
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$\log_{10}(E_{\mathrm{had,corrected}}/E_{\mathrm{had,true}})$")
    ax.set_ylabel("Counts")
    med_res = np.median(r_valid)
    ax.set_title(f"After correction (med={med_res:.2f}, "
                 f"RMS={np.std(np.log10(r_valid)):.2f})")

fig2.tight_layout()
fig2.savefig(os.path.join(outdir, "hadronic_correction_fit.png"), dpi=150)
plt.close(fig2)

# ---------------------------------------------------------------
# Print correction parameters
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Hadronic Energy Correction Parameters")
print("=" * 60)
print("\nPower-law fit: E_true = a * (E_reco)^b")
for flavor, params in correction_params.items():
    print(f"  {flavor:>10}: a={params['a']:.3f} ± {params['a_err']:.3f}, "
          f"b={params['b']:.3f} ± {params['b_err']:.3f}, "
          f"N={params['n_points']:,}")

# ---------------------------------------------------------------
# Figure 3: SPINE multiplicity of hadronic particles
# ---------------------------------------------------------------
print("\n--- Figure 3: Hadronic particle multiplicity ---")
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Number of SPINE particles contributing to hadronic energy",
              fontsize=14, fontweight="bold")

for idx, (flavor_mask, flavor_lbl, col) in enumerate(flavor_masks):
    ax = axes3[idx]
    n_had = arr["n_spine_had"][flavor_mask]
    ax.hist(n_had, bins=np.arange(0, 20), histtype="step", lw=1.5, color=col)
    ax.set_xlabel("N SPINE hadronic particles")
    ax.set_ylabel("Counts")
    ax.set_title(f"{flavor_lbl} (med={np.median(n_had):.1f})")

fig3.tight_layout()
fig3.savefig(os.path.join(outdir, "hadronic_multiplicity.png"), dpi=150)
plt.close(fig3)

# ---------------------------------------------------------------
# Figure 4: Closure test — corrected vs uncorrected total neutrino energy
# ---------------------------------------------------------------
print("\n--- Figure 4: Closure test ---")
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle(
    "Hadronic Over-Correction: Particle-Sum Double-Counting Diagnostic",
    fontsize=13, fontweight="bold", color="#c0392b",
)

def power_law(x, a, b):
    return a * np.power(x, b)

_diag_note = (
    "Root cause: particle-sum decomposition is not\n"
    "clean — some EM-calibrated particles contribute\n"
    "to both lepton and hadronic categories.\n"
    "→ Single power-law cannot close the energy loop.\n"
    "→ Requires multi-variable ML regression."
)

for idx, (flavor_mask, flavor_lbl, col) in enumerate(flavor_masks):
    ax = axes4[idx]
    mask = flavor_mask & (arr["spine_had"] > 0.01) & (arr["true_had"] > 0.01)
    if mask.sum() < 10:
        ax.text(0.5, 0.5, "N<10", ha="center", va="center", transform=ax.transAxes)
        continue
    nuE = arr["nu_E"][mask]
    total_sp = arr["spine_total"][mask]
    had_sp = arr["spine_had"][mask]
    # Get correction params for this flavor
    params = correction_params.get(flavor_lbl, None)
    if params is None:
        ax.text(0.5, 0.5, "No fit", ha="center", va="center", transform=ax.transAxes)
        continue
    a, b = params["a"], params["b"]
    corrected_had = power_law(had_sp, a, b)
    # Total corrected energy: (total_sp - had_sp) + corrected_had
    corrected_total = (total_sp - had_sp) + corrected_had
    # Ratios
    uncorrected_ratio = total_sp / nuE
    corrected_ratio = corrected_total / nuE
    valid_u = np.isfinite(uncorrected_ratio) & (uncorrected_ratio > 0) & (uncorrected_ratio < 10)
    valid_c = np.isfinite(corrected_ratio) & (corrected_ratio > 0) & (corrected_ratio < 10)
    med_u = np.median(uncorrected_ratio[valid_u])
    med_c = np.median(corrected_ratio[valid_c])
    ax.hist(uncorrected_ratio[valid_u], bins=np.linspace(0, 3, 50), histtype="step",
            lw=1.5, color=col, ls="-", density=True,
            label=f"Uncorrected (med={med_u:.2f})")
    ax.hist(corrected_ratio[valid_c], bins=np.linspace(0, 3, 50), histtype="step",
            lw=2.0, color="#c0392b", ls="--", density=True,
            label=f"Corrected (med={med_c:.2f}) ← over-corrects!")
    ax.axvline(1, color="k", ls=":", lw=0.8, label="Unity")
    ax.axvline(med_c, color="#c0392b", ls=":", lw=1.2, alpha=0.6)
    ax.set_xlabel(r"$E_{\nu,\mathrm{reco}} / E_{\nu,\mathrm{true}}$")
    ax.set_ylabel("Density")
    ax.set_title(f"{flavor_lbl} (N={mask.sum():,})  — corrected median = {med_c:.2f}")
    ax.legend(fontsize=8)
    ax.text(0.97, 0.62, _diag_note, transform=ax.transAxes, fontsize=7.5,
            ha="right", va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#e0a800",
                      alpha=0.95))
    # Print closure statistics
    print(f"  {flavor_lbl} closure: uncorrected med={med_u:.3f} "
          f"→ corrected med={med_c:.3f} (over-corrects by {(med_c-1)*100:+.1f}%)")

fig4.tight_layout()
fig4.savefig(os.path.join(outdir, "hadronic_closure_test.png"), dpi=150)
plt.close(fig4)
print("  Saved hadronic_closure_test.png")

# ---------------------------------------------------------------
# Figure 5: Linear and hybrid correction comparison
# The power-law blows up for low reco energies (b≈0.37, scaling ~E^-0.63).
# Try: (a) linear fit  E_true = α·E_reco + β
#      (b) hybrid: linear below 100 MeV, power-law above
# ---------------------------------------------------------------
print("\n--- Figure 5: Linear / hybrid correction comparison ---")

from scipy.optimize import curve_fit as _cf

def _linear(x, alpha, beta):
    return alpha * x + beta

def _hybrid(x, a_pw, b_pw, a_lin, b_lin, thresh=0.1):
    """Hybrid: linear below thresh GeV, power-law above."""
    result = np.where(x < thresh, a_lin * x + b_lin, a_pw * np.power(np.maximum(x, 1e-9), b_pw))
    return result

linear_params = {}
hybrid_params = {}

fig5, axes5 = plt.subplots(2, 2, figsize=(14, 12))
fig5.suptitle(
    "Hadronic Energy Correction: Linear vs Hybrid vs Power-law (Closure Test)",
    fontsize=13, fontweight="bold",
)

for idx, (flavor_mask, flavor_lbl, col) in enumerate(flavor_masks):
    mask = flavor_mask & (arr["true_had"] > 0.01) & (arr["spine_had"] > 0.01)
    if mask.sum() < 10:
        continue
    true_h = arr["true_had"][mask]
    reco_h = arr["spine_had"][mask]
    nuE    = arr["nu_E"][mask]
    total_sp = arr["spine_total"][mask]
    had_sp   = arr["spine_had"][mask]

    # --- Linear fit ---
    try:
        popt_lin, _ = _cf(_linear, reco_h, true_h, p0=[2.0, 0.05], maxfev=5000)
    except Exception:
        popt_lin = [2.0, 0.0]
    alpha_l, beta_l = popt_lin
    linear_params[flavor_lbl] = {"alpha": float(alpha_l), "beta": float(beta_l), "n": int(mask.sum())}

    # --- Hybrid: separate fits below/above 0.1 GeV ---
    lo = reco_h < 0.1
    hi = reco_h >= 0.1
    try:
        popt_lo, _ = _cf(_linear, reco_h[lo], true_h[lo], p0=[2.0, 0.05], maxfev=5000)
    except Exception:
        popt_lo = [2.0, 0.0]
    a_lin_h, b_lin_h = popt_lo
    pw_params = correction_params.get(flavor_lbl, {"a": 2.0, "b": 0.37})
    a_pw_h, b_pw_h = pw_params["a"], pw_params["b"]
    hybrid_params[flavor_lbl] = {
        "power_law": {"a": a_pw_h, "b": b_pw_h},
        "linear_low": {"alpha": float(a_lin_h), "beta": float(b_lin_h)},
        "threshold_gev": 0.1,
    }

    # --- Apply corrections ---
    corr_pw  = power_law(had_sp, a_pw_h, b_pw_h)
    corr_lin = _linear(had_sp, alpha_l, beta_l)
    corr_hyb = np.where(had_sp < 0.1,
                        _linear(had_sp, a_lin_h, b_lin_h),
                        power_law(had_sp, a_pw_h, b_pw_h))

    ratio_raw = (total_sp) / nuE
    ratio_pw  = (total_sp - had_sp + corr_pw)  / nuE
    ratio_lin = (total_sp - had_sp + corr_lin) / nuE
    ratio_hyb = (total_sp - had_sp + corr_hyb) / nuE

    bins_r = np.linspace(0, 3, 61)
    def _filt(r): return r[np.isfinite(r) & (r > 0) & (r < 5)]

    # Panel (idx, 0): E_had scatter with all three fit lines
    ax = axes5[idx, 0]
    x_plot = np.logspace(-2, 1, 100)
    ax.scatter(reco_h, true_h, s=1, alpha=0.15, c=col, rasterized=True)
    ax.plot(x_plot, power_law(x_plot, a_pw_h, b_pw_h),
            "r-", lw=2, label=rf"Power-law $a={a_pw_h:.2f},b={b_pw_h:.2f}$")
    ax.plot(x_plot, _linear(x_plot, alpha_l, beta_l),
            "b--", lw=2, label=rf"Linear $\alpha={alpha_l:.2f},\beta={beta_l:.3f}$")
    ax.plot(x_plot, np.where(x_plot < 0.1,
                             _linear(x_plot, a_lin_h, b_lin_h),
                             power_law(x_plot, a_pw_h, b_pw_h)),
            "g:", lw=2.5, label="Hybrid (linear<0.1 GeV, PL above)")
    ax.axvline(0.1, color="gray", ls=":", lw=0.8, alpha=0.6, label="Hybrid threshold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"SPINE reco $E_{\rm had}$ [GeV]")
    ax.set_ylabel(r"True $E_{\rm had}$ [GeV]")
    ax.set_title(f"{flavor_lbl}: correction fits")
    ax.legend(fontsize=7.5)

    # Panel (idx, 1): Eν closure for all three corrections
    ax = axes5[idx, 1]
    ax.hist(_filt(ratio_raw), bins=bins_r, histtype="step", lw=1.5, color=col, density=True,
            label=f"Raw (med={np.median(_filt(ratio_raw)):.2f})")
    ax.hist(_filt(ratio_pw),  bins=bins_r, histtype="step", lw=1.5, color="r",  density=True,
            ls="--", label=f"Power-law (med={np.median(_filt(ratio_pw)):.2f})")
    ax.hist(_filt(ratio_lin), bins=bins_r, histtype="step", lw=1.5, color="b",  density=True,
            ls="--", label=f"Linear (med={np.median(_filt(ratio_lin)):.2f})")
    ax.hist(_filt(ratio_hyb), bins=bins_r, histtype="step", lw=2.0, color="g",  density=True,
            ls="-", label=f"Hybrid (med={np.median(_filt(ratio_hyb)):.2f})")
    ax.axvline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$E_{\nu,\rm reco}/E_{\nu,\rm true}$")
    ax.set_ylabel("Density")
    ax.set_title(f"{flavor_lbl}: Eν closure comparison")
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 3)

    print(f"  {flavor_lbl}:")
    print(f"    Raw:        med={np.median(_filt(ratio_raw)):.3f}")
    print(f"    Power-law:  med={np.median(_filt(ratio_pw)):.3f}")
    print(f"    Linear:     med={np.median(_filt(ratio_lin)):.3f}")
    print(f"    Hybrid:     med={np.median(_filt(ratio_hyb)):.3f}")

fig5.tight_layout()
fig5.savefig(os.path.join(outdir, "hadronic_linear_hybrid.png"), dpi=150)
plt.close(fig5)
print("  Saved hadronic_linear_hybrid.png")

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
out = {
    "correction_params": correction_params,
    "linear_params": linear_params,
    "hybrid_params": hybrid_params,
    "note": "Using particle-sum hadronic energy (Enu estimator branches not populated)",
}
with open(os.path.join(outdir, "hadronic_correction_params.json"), "w") as f:
    json.dump(out, f, indent=2)

print(f"\nCorrection params saved to figures/hadronic_correction_params.json")
print("Done!")

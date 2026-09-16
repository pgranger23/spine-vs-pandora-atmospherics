#!/usr/bin/env python3
"""
Electron energy resolution study: median reco/true energy ratio
as a function of true energy on a log scale.
"""

import os, sys, ctypes, math, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from tqdm import tqdm

import caf_env
pySRProxy = caf_env.setup_pysrproxy()

outdir = "figures"
os.makedirs(outdir, exist_ok=True)

import plot_style
plot_style.apply()

_nan = float("nan")

def _vec_mag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def _vec_norm_dir(px, py, pz):
    n = _vec_mag(px, py, pz)
    return (px/n, py/n, pz/n) if n > 1e-6 else (_nan, _nan, _nan)

def _cos_angle(ax, ay, az, bx, by, bz):
    na = _vec_mag(ax, ay, az)
    nb = _vec_mag(bx, by, bz)
    if na < 1e-6 or nb < 1e-6:
        return _nan
    return max(-1.0, min(1.0, (ax*bx + ay*by + az*bz) / (na * nb)))

def _closest_dlp_idx(ixn, nu_vtx):
    best, best_i = float("inf"), 0
    nvx, nvy, nvz = nu_vtx.x, nu_vtx.y, nu_vtx.z
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = math.sqrt((v.x-nvx)**2 + (v.y-nvy)**2 + (v.z-nvz)**2)
        if d < best:
            best, best_i = d, i
    return best_i

def _true_energy(pdg, px, py, pz):
    m = MASS.get(abs(pdg), 0.0)
    p2 = px*px + py*py + pz*pz
    return math.sqrt(p2 + m*m)

MASS = {11: 0.000511, 13: 0.1057, 211: 0.1396, 321: 0.4937,
        2212: 0.9383, 2112: 0.9396, 22: 0.0, 111: 0.1350, 311: 0.4976,
        -11: 0.000511, -13: 0.1057, -211: 0.1396, -321: 0.4937}

_PDG_MASS = {
    11: 0.000511, 13: 0.10566, 211: 0.13957, 321: 0.49368,
    2212: 0.93827, 2112: 0.93957, 22: 0.0, 111: 0.13498,
    311: 0.49761, 310: 0.49761,
}

MAX_EVENTS = 5000
nmatch_threshold = 0.5

electrons = []

prox = pySRProxy.StandardRecordFileReader("recTree", caf_env.CAF_FILE)

for ientry in tqdm(range(prox.entries()), desc="Scanning", total=prox.entries()):
    if len(electrons) >= MAX_EVENTS and len(electrons) > 100:
        break

    rec = prox.entry(ientry)
    mc = rec.mc
    if not (mc.nnu > 0 and mc.nu[0].iscc):
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu = mc.nu[0]
    si = _closest_dlp_idx(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    truth_parts = []
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        p_mag = _vec_mag(px, py, pz)
        if p_mag < 0.02:
            continue
        dx, dy, dz = _vec_norm_dir(px, py, pz)
        if math.isnan(dx):
            continue
        truth_parts.append({
            "pdg": tp.pdg, "p_mag": p_mag,
            "dx": dx, "dy": dy, "dz": dz,
            "E": _true_energy(tp.pdg, px, py, pz),
        })

    ntruth = len(truth_parts)
    if ntruth == 0:
        continue

    spine_parts = []
    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        if not p.primary:
            continue
        px, py, pz = p.p.x, p.p.y, p.p.z
        p_mag = _vec_mag(px, py, pz)
        if not (p_mag > 1):
            continue
        dx, dy, dz = _vec_norm_dir(px, py, pz)
        if math.isnan(dx):
            continue
        mass_gev = _PDG_MASS.get(abs(p.pdg), 0)
        p_mag_gev = p_mag * 0.001
        E_total = p.E * 0.001 + mass_gev
        if abs(p.pdg) in {11, 22, 111}:
            p_mag_gev *= 0.198
            E_total = p.E * 0.001 * 0.198 + mass_gev
        spine_parts.append({
            "idx": pi, "pdg": p.pdg, "p_mag": p_mag_gev,
            "dx": dx, "dy": dy, "dz": dz,
            "E": E_total,
        })

    if len(spine_parts) == 0:
        continue

    for ti, tp in enumerate(truth_parts):
        best_cos, best_ri = -2.0, -1
        for ri, rp in enumerate(spine_parts):
            cos = _cos_angle(tp["dx"], tp["dy"], tp["dz"],
                             rp["dx"], rp["dy"], rp["dz"])
            if cos > best_cos:
                best_cos, best_ri = cos, ri
        if best_cos > nmatch_threshold and best_ri >= 0:
            rp = spine_parts[best_ri]
            if abs(tp["pdg"]) == 11:
                electrons.append({
                    "true_E": tp["E"],
                    "true_p": tp["p_mag"],
                    "reco_E": rp["E"],
                    "reco_p": rp["p_mag"],
                    "cos_match": best_cos,
                })

# Only use first 5000 electrons to match analysis convention
electrons = electrons[:5000]
ned = len(electrons)
print(f"Matched {ned} electrons")

if ned < 10:
    print("Too few electrons, aborting.")
    sys.exit(1)

# Arrays
true_E = np.array([e["true_E"] for e in electrons])
true_p = np.array([e["true_p"] for e in electrons])
reco_E = np.array([e["reco_E"] for e in electrons])
reco_p = np.array([e["reco_p"] for e in electrons])

e_ratio = reco_E / true_E
p_ratio = reco_p / true_p

# ============================================================
# Plot 1: Energy ratio vs true energy (log scale) — binned median
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_f = axes.flatten()

# Panel a: E_reco/E_true median vs true E (log x, log-linear y)
valid = np.isfinite(true_E) & np.isfinite(e_ratio) & (e_ratio > 0)
ax = axes_f[0]
# Binning — compute explicit 16th/84th percentiles (asymmetric)
emin, emax = true_E[valid].min(), true_E[valid].max()
bins = np.logspace(np.log10(emin * 0.95), np.log10(emax * 1.05), 25)
med, be, _ = binned_statistic(true_E[valid], e_ratio[valid], "median", bins=bins)
p16, _, _ = binned_statistic(true_E[valid], e_ratio[valid],
                              lambda x: np.percentile(x, 16), bins=bins)
p84, _, _ = binned_statistic(true_E[valid], e_ratio[valid],
                              lambda x: np.percentile(x, 84), bins=bins)
iqr = (p84 - p16) / 2  # kept for summary table compatibility; plots use p16/p84 directly
cnt, _, _ = binned_statistic(true_E[valid], e_ratio[valid], "count", bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
mk = cnt > 3
# Asymmetric error: clip lower bound at 0 to avoid unphysical negative ratios
yerr_lo = np.maximum(med[mk] - p16[mk], 0)
yerr_hi = p84[mk] - med[mk]
ax.fill_between(centers[mk], np.maximum(p16[mk], 0), p84[mk],
                alpha=0.18, color="C0", label="16%–84% band")
ax.errorbar(centers[mk], med[mk], yerr=[yerr_lo, yerr_hi],
            marker="o", ms=4, capsize=2, capthick=0.8, lw=1, color="C0", label="Median ± asymm. 16/84")
ax.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
ax.axhline(np.median(e_ratio[valid]), color="C0", ls="--", lw=0.8,
           label=f"Overall median = {np.median(e_ratio[valid]):.2f}")
ax.set_xscale("log")
ax.set_xlabel(r"True $E$ [GeV]")
ax.set_ylabel(r"$E_\mathrm{reco} / E_\mathrm{true}$")
ax.set_title("Median energy ratio vs true energy")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)

# Panel b: E_reco/E_true scatter + median overlay
ax = axes_f[1]
ax.scatter(true_E[valid], e_ratio[valid], s=2, alpha=0.15, rasterized=True, color="C0")
ax.fill_between(centers[mk], np.maximum(p16[mk], 0), p84[mk],
                alpha=0.25, color="red", label="16%–84% percentile band")
ax.plot(centers[mk], med[mk], "o-", ms=4, lw=1.5, color="red", label="Binned median")
ax.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
ax.axhline(np.median(e_ratio[valid]), color="red", ls="--", lw=0.8,
           label=f"Overall median = {np.median(e_ratio[valid]):.2f}")
ax.set_xscale("log")
ax.set_xlabel(r"True $E$ [GeV]")
ax.set_ylabel(r"$E_\mathrm{reco} / E_\mathrm{true}$")
ax.set_title("Energy ratio scatter + binned medians")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)

# Panel c: p_ratio vs true p (log) — compare E ratio to p ratio
valid_p = np.isfinite(true_p) & np.isfinite(p_ratio) & (p_ratio > 0)
ax = axes_f[2]
pmin, pmax = true_p[valid_p].min(), true_p[valid_p].max()
pbins = np.logspace(np.log10(pmin * 0.95), np.log10(pmax * 1.05), 25)
pmed, pbe, _ = binned_statistic(true_p[valid_p], p_ratio[valid_p], "median", bins=pbins)
pp16, _, _ = binned_statistic(true_p[valid_p], p_ratio[valid_p],
                               lambda x: np.percentile(x, 16), bins=pbins)
pp84, _, _ = binned_statistic(true_p[valid_p], p_ratio[valid_p],
                               lambda x: np.percentile(x, 84), bins=pbins)
pcnt, _, _ = binned_statistic(true_p[valid_p], p_ratio[valid_p], "count", bins=pbins)
pcenters = 0.5 * (pbins[:-1] + pbins[1:])
pmk = pcnt > 3
p_yerr_lo = np.maximum(pmed[pmk] - pp16[pmk], 0)
p_yerr_hi = pp84[pmk] - pmed[pmk]
ax.fill_between(pcenters[pmk], np.maximum(pp16[pmk], 0), pp84[pmk],
                alpha=0.18, color="C3", label="16%–84% band")
ax.errorbar(pcenters[pmk], pmed[pmk], yerr=[p_yerr_lo, p_yerr_hi],
            marker="o", ms=4, capsize=2, capthick=0.8, lw=1, color="C3", label="Median ± asymm. 16/84")
ax.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
ax.axhline(np.median(p_ratio[valid_p]), color="C3", ls="--", lw=0.8,
           label=f"Overall median = {np.median(p_ratio[valid_p]):.2f}")
ax.set_xscale("log")
ax.set_xlabel(r"True $p$ [GeV/c]")
ax.set_ylabel(r"$p_\mathrm{reco} / p_\mathrm{true}$")
ax.set_title("Momentum ratio vs true momentum")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)

# Panel d: E_reco vs true E scatter on log-log with asymmetric resolution band
ax = axes_f[3]
ax.scatter(true_E[valid], reco_E[valid], s=2, alpha=0.15, rasterized=True, color="C0")
# Binned median and explicit p16/p84 band in absolute energy space
reco_ratio = reco_E / true_E
emed, ebe, _ = binned_statistic(true_E[valid], reco_E[valid], "median", bins=bins)
e_p16_ratio, _, _ = binned_statistic(true_E[valid], reco_ratio[valid],
                                     lambda x: np.percentile(x, 16), bins=bins)
e_p84_ratio, _, _ = binned_statistic(true_E[valid], reco_ratio[valid],
                                     lambda x: np.percentile(x, 84), bins=bins)
ecnt, _, _ = binned_statistic(true_E[valid], reco_E[valid], "count", bins=bins)
emk = ecnt > 3
# Band in absolute reco-E space: lower = max(p16 * true_E, 0)
c_vals = centers[emk]
band_lower = np.maximum(e_p16_ratio[emk] * c_vals, 0)
band_upper = e_p84_ratio[emk] * c_vals
ax.fill_between(c_vals, band_lower, band_upper, alpha=0.2, color="C0",
                label="16%–84% asymmetric band")
ax.errorbar(centers[emk], emed[emk], marker="o", ms=5, capsize=2, lw=1.5, color="red", label="Binned median")
ax.plot([emin*0.9, emax*1.1], [emin*0.9, emax*1.1], "k:", lw=0.8, alpha=0.5, label="E_reco = E_true")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"True $E$ [GeV]")
ax.set_ylabel(r"Reco $E$ [GeV]")
res_med = np.median(reco_ratio[valid])
ax.set_title(f"Reco energy vs true energy (log-log, med ratio={res_med:.2f})")
ax.legend(fontsize=8)

fig.suptitle(f"Electron energy resolution (N={ned})", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "electron_energy_resolution.png"), dpi=150)
plt.close(fig)

# Save results before flavor scan (which redefines local vars)
SUMMARY_e_ratio = e_ratio.copy()
SUMMARY_valid = valid.copy()
SUMMARY_centers = centers.copy()
SUMMARY_med = med.copy()
SUMMARY_iqr = iqr.copy()
SUMMARY_cnt = cnt.copy()
SUMMARY_mk = mk.copy()
SUMMARY_p_ratio = p_ratio.copy()
SUMMARY_valid_p = valid_p.copy()

# ============================================================
# Additional diagnostics: per-flavor breakdown
# ============================================================
# Check if ratio depends on neutrino flavor
# We need to re-scan with flavor info
print("\nRe-scanning with flavor info...")
electron_flavor = []
prox2 = pySRProxy.StandardRecordFileReader("recTree", caf_env.CAF_FILE)
for ientry in tqdm(range(prox2.entries()), desc="Flavor scan", total=prox2.entries()):
    if len(electron_flavor) >= ned:
        break
    rec = prox2.entry(ientry)
    mc = rec.mc
    if not (mc.nnu > 0 and mc.nu[0].iscc):
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue
    nu = mc.nu[0]
    si = _closest_dlp_idx(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    truth_parts = []
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        p_mag = _vec_mag(px, py, pz)
        if p_mag < 0.02:
            continue
        dx, dy, dz = _vec_norm_dir(px, py, pz)
        if math.isnan(dx):
            continue
        truth_parts.append({
            "pdg": tp.pdg, "p_mag": p_mag,
            "dx": dx, "dy": dy, "dz": dz,
            "E": _true_energy(tp.pdg, px, py, pz),
        })

    if not truth_parts:
        continue

    spine_parts = []
    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        if not p.primary:
            continue
        px, py, pz = p.p.x, p.p.y, p.p.z
        p_mag = _vec_mag(px, py, pz)
        if not (p_mag > 1):
            continue
        dx, dy, dz = _vec_norm_dir(px, py, pz)
        if math.isnan(dx):
            continue
        mass_gev = _PDG_MASS.get(abs(p.pdg), 0)
        p_mag_gev = p_mag * 0.001
        E_total = p.E * 0.001 + mass_gev
        if abs(p.pdg) in {11, 22, 111}:
            p_mag_gev *= 0.198
            E_total = p.E * 0.001 * 0.198 + mass_gev
        spine_parts.append({
            "idx": pi, "pdg": p.pdg, "p_mag": p_mag_gev,
            "dx": dx, "dy": dy, "dz": dz,
            "E": E_total,
        })

    if not spine_parts:
        continue

    for ti, tp in enumerate(truth_parts):
        best_cos, best_ri = -2.0, -1
        for ri, rp in enumerate(spine_parts):
            cos = _cos_angle(tp["dx"], tp["dy"], tp["dz"],
                             rp["dx"], rp["dy"], rp["dz"])
            if cos > best_cos:
                best_cos, best_ri = cos, ri
        if best_cos > nmatch_threshold and best_ri >= 0:
            rp = spine_parts[best_ri]
            if abs(tp["pdg"]) == 11:
                electron_flavor.append({
                    "true_E": tp["E"], "reco_E": rp["E"],
                    "flavor": nu.pdg,
                })

print(f"Flavor-matched electrons: {len(electron_flavor)}")

if len(electron_flavor) > 10:
    # Separate by flavor
    ef_arr = np.array([(e["true_E"], e["reco_E"], e["flavor"]) for e in electron_flavor],
                      dtype=[("true_E", float), ("reco_E", float), ("flavor", int)])
    flavor_labels = {12: r"$\nu_e$", -12: r"$\bar{\nu}_e$", 14: r"$\nu_\mu$", -14: r"$\bar{\nu}_\mu$"}
    colors_fl = {12: "C0", -12: "C1", 14: "C2", -14: "C3"}

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    for flav in [12, -12, 14, -14]:
        m = ef_arr["flavor"] == flav
        if m.sum() < 5:
            continue
        tE = ef_arr["true_E"][m]
        rE = ef_arr["reco_E"][m]
        rat = rE / tE
        valid = np.isfinite(tE) & np.isfinite(rat) & (rat > 0)
        if valid.sum() < 5:
            continue
        emin2, emax2 = tE[valid].min(), tE[valid].max()
        bins2 = np.logspace(np.log10(emin2 * 0.95), np.log10(emax2 * 1.05), 20)
        med2, be2, _ = binned_statistic(tE[valid], rat[valid], "median", bins=bins2)
        iqr2, _, _ = binned_statistic(tE[valid], rat[valid],
                                       lambda x: np.abs(np.percentile(x, 84) - np.percentile(x, 16)) / 2, bins=bins2)
        cnt2, _, _ = binned_statistic(tE[valid], rat[valid], "count", bins=bins2)
        cen2 = 0.5 * (bins2[:-1] + bins2[1:])
        mk2 = cnt2 > 3
        ax2.errorbar(cen2[mk2], med2[mk2], yerr=iqr2[mk2], marker="o", ms=5,
                     capsize=2, color=colors_fl[flav], label=f"{flavor_labels[flav]} (N={valid.sum()})")
    ax2.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"True $E$ [GeV]")
    ax2.set_ylabel(r"$E_\mathrm{reco} / E_\mathrm{true}$")
    ax2.set_title("Electron energy ratio by neutrino flavor")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "electron_energy_by_flavor.png"), dpi=150)
    plt.close(fig2)

# ============================================================
# Print summary table
# ============================================================
print(f"\n{'='*70}")
print(f"ELECTRON ENERGY RESOLUTION SUMMARY (N={ned})")
print(f"{'='*70}")
print(f"\n  Overall median E_reco/E_true: {np.median(SUMMARY_e_ratio[SUMMARY_valid]):.4f}")
print(f"  Overall median p_reco/p_true: {np.median(SUMMARY_p_ratio[SUMMARY_valid_p]):.4f}")
print(f"  True E range: [{true_E[SUMMARY_valid].min():.4f}, {true_E[SUMMARY_valid].max():.4f}] GeV")
print(f"  E_reco/E_true IQR/2: {np.percentile(SUMMARY_e_ratio[SUMMARY_valid],84)-np.percentile(SUMMARY_e_ratio[SUMMARY_valid],16)/2:.4f}")

print(f"\n  Binned median E_reco/E_true vs true E:")
print(f"  {'E_true [GeV]':>14} {'median':>8} {'IQR/2':>8} {'N':>6}")
for i in range(len(SUMMARY_centers)):
    if SUMMARY_mk[i]:
        print(f"  [{bins[i]:.4f}, {bins[i+1]:.4f})  {SUMMARY_med[i]:>8.3f} {SUMMARY_iqr[i]:>8.3f} {int(SUMMARY_cnt[i]):>6}")

# Check constant ratio hypothesis: fit a constant
from scipy import stats
# If ratio is truly constant, slope of ratio vs log(E) should be ~0
slope, intercept, r_val, p_val, std_err = stats.linregress(np.log10(true_E[SUMMARY_valid]), SUMMARY_e_ratio[SUMMARY_valid])
print(f"\n  Constant ratio hypothesis:")
print(f"    Slope of ratio vs log10(true_E): {slope:.4f} ± {std_err:.4f}")
print(f"    (should be ~0 if ratio is energy-independent)")
print(f"    p-value: {p_val:.2e}")
if abs(slope) / std_err < 2:
    print(f"    → Ratio IS consistent with constant (within ~{abs(slope/std_err):.1f}σ)")
else:
    print(f"    → Ratio has significant energy dependence ({abs(slope/std_err):.1f}σ)")

# Also check if the scatter distribution is log-normal
log_ratio = np.log(SUMMARY_e_ratio[SUMMARY_valid])
log_mean = np.mean(log_ratio)
log_std = np.std(log_ratio)
print(f"\n  Log-normal fit to E_reco/E_true:")
print(f"    μ = {log_mean:.3f}, σ = {log_std:.3f}")
print(f"    Equivalent: median ratio = {np.exp(log_mean):.3f},")
print(f"    multiplicative spread = exp(σ) = {np.exp(log_std):.3f}")

# Check resolution (width) as a function of energy
resol_vals = SUMMARY_iqr[SUMMARY_mk] / SUMMARY_med[SUMMARY_mk] * 100  # relative IQR/2 in %
print(f"\n  Relative resolution (IQR/2 / median) vs energy:")
for i in np.where(SUMMARY_mk)[0][::3]:
    print(f"    E_true ≈ {SUMMARY_centers[i]:.3f} GeV: resol = {resol_vals[i]:.1f}% (N={int(SUMMARY_cnt[i])})")

results = {
    "n_electrons": ned,
    "median_E_ratio": float(np.median(SUMMARY_e_ratio[SUMMARY_valid])),
    "median_p_ratio": float(np.median(SUMMARY_p_ratio[SUMMARY_valid_p])),
    "E_ratio_IQR2": float(np.percentile(SUMMARY_e_ratio[SUMMARY_valid],84) - np.percentile(SUMMARY_e_ratio[SUMMARY_valid],16)/2),
    "log_mean": float(log_mean),
    "log_sigma": float(log_std),
    "slope_vs_logE": float(slope),
    "slope_err": float(std_err),
    "slope_pvalue": float(p_val),
}
with open(os.path.join(outdir, "electron_energy_resolution.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nFigure saved: {outdir}/electron_energy_resolution.png")
print("Done!")

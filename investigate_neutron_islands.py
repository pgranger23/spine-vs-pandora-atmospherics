#!/usr/bin/env python3
"""
Neutron spatial distribution and clustering analysis.

Study how neutrons are reconstructed by SPINE:
1. Spatial/angular distribution of neutron-induced activity
2. Multiplicity in events with vs without neutrons
3. Relationship between neutron energy overestimation and event complexity
4. Mis-identification patterns: neutrons reconstructed as showers

Background: Neutrons deposit energy via secondary proton scatters
detached from the primary interaction vertex. SPINE overestimates
neutron energy by 1.51x and mis-IDs 26.2% of them as EM showers.

Output: figures/neutron_islands_*.png
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

# Particle masses (GeV)
MASSES = {
    2212: 0.938272, 2112: 0.939565, 11: 0.000511,
    13: 0.105658, 211: 0.139570, 111: 0.134977, 22: 0.0,
}

def _mass(pdg):
    return MASSES.get(abs(pdg), 0.1)

def _safe_norm(x, y, z):
    n = math.sqrt(x*x + y*y + z*z)
    return (x/n, y/n, z/n) if n > 0 else (_nan, _nan, _nan)

def _vec_mag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

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
print("Loading CAF data for neutron island study")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader('recTree', caf_env.CAF_FILE)

event_records = []
neutron_particle_records = []

for ientry in tqdm(range(prox.entries()), desc="Scanning"):
    rec = prox.entry(ientry)
    mc = rec.mc
    if not mc.nu[0].iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu = mc.nu[0]
    nu_vtx = np.array([nu.vtx.x, nu.vtx.y, nu.vtx.z])
    nu_E = nu.E

    # Find closest DLP interaction
    si = _closest_dlp_idx(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    # ---- Truth-level information ----
    has_truth_neutron = False
    neutron_true_p = []
    n_truth_prim = 0  # truth primaries above momentum threshold
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        p_mag = math.sqrt(px*px + py*py + pz*pz)
        if p_mag > 0.01:
            n_truth_prim += 1
        if abs(tp.pdg) == 2112 and p_mag > 0.01:
            has_truth_neutron = True
            neutron_true_p.append(p_mag * 0.001)  # MeV → GeV

    # ---- SPINE particle information ----
    n_spine_parts = dlp.part.ndlp
    spine_protons = 0
    spine_neutrons = 0
    spine_showers = 0
    spine_p_ratios = []
    n_reco_primary = 0   # reco primaries above momentum threshold
    n_reco_matched = 0   # reco primaries matched to a truth primary (cos θ > 0.5)

    for pi in range(n_spine_parts):
        p = dlp.part.dlp[pi]
        if not p.primary:
            continue

        # Reconstructed momentum magnitude
        rpx, rpy, rpz = p.p.x, p.p.y, p.p.z
        rp_mag = math.sqrt(rpx*rpx + rpy*rpy + rpz*rpz)
        if rp_mag < 1:
            continue
        rp_gev = rp_mag * 0.001  # MeV → GeV
        n_reco_primary += 1

        # Reconstructed kinetic energy (SPINE stores kinetic)
        rke = p.E * 0.001 if p.E > 0 else _nan  # MeV → GeV

        # PDG class from SPINE
        pdg = int(p.pdg)
        abs_pdg = abs(pdg)

        if abs_pdg == 2212:
            spine_protons += 1
        elif abs_pdg == 2112:
            spine_neutrons += 1
        elif abs_pdg in (11, 22, 111):
            spine_showers += 1

        # Match to true particles via geometric matching
        best_cos = -2
        best_truth_pdg = None
        best_truth_p = _nan
        for ti2 in range(nu.nprim):
            tp2 = nu.prim[ti2]
            tpx, tpy, tpz = tp2.p.px, tp2.p.py, tp2.p.pz
            tp_mag = math.sqrt(tpx*tpx + tpy*tpy + tpz*tpz)
            if tp_mag < 0.01:
                continue
            tdir = np.array(_safe_norm(tpx, tpy, tpz))
            rdir = np.array(_safe_norm(rpx, rpy, rpz))
            if np.any(~np.isfinite(tdir)) or np.any(~np.isfinite(rdir)):
                continue
            cos_ang = np.dot(tdir, rdir)
            if cos_ang > best_cos:
                best_cos = cos_ang
                best_truth_pdg = abs(tp2.pdg)
                best_truth_p = tp_mag * 0.001

        if best_cos > 0.5 and best_truth_pdg is not None:
            n_reco_matched += 1
            p_ratio = rp_gev / best_truth_p if best_truth_p > 0 else _nan
            is_neutron_match = (best_truth_pdg == 2112)
            ang_deg = math.degrees(math.acos(max(-1, min(1, best_cos))))
            # Spatial separation between true vertex and reco particle start
            dr = _vec_mag(p.start.x - nu.vtx.x, p.start.y - nu.vtx.y, p.start.z - nu.vtx.z) * 0.1  # mm → cm
            neutron_particle_records.append({
                "true_pdg": best_truth_pdg,
                "reco_p_gev": rp_gev,
                "true_p_gev": best_truth_p,
                "p_ratio": p_ratio,
                "ang_deg": ang_deg,
                "is_neutron": is_neutron_match,
                "spine_class": abs_pdg,
                "nu_E": nu_E,
                "cos_theta": best_cos,
                "dr_cm": dr,
            })

    # Event-level record
    event_records.append({
        "has_neutron": has_truth_neutron,
        "n_neutron_true": len(neutron_true_p),
        "neutron_true_p": neutron_true_p,
        "n_spine_parts": n_spine_parts,
        "spine_protons": spine_protons,
        "spine_neutrons": spine_neutrons,
        "spine_showers": spine_showers,
        "nu_E": nu_E,
        "n_truth_prim": n_truth_prim,
        "n_reco_primary": n_reco_primary,
        "n_reco_matched": n_reco_matched,
    })

print(f"\n  Events: {len(event_records)}")
print(f"  Neutron-matched particles: {len(neutron_particle_records)}")

# Convert to numpy arrays
ev_arr = {k: np.array([r[k] for r in event_records], dtype=object if k == "neutron_true_p" else float)
          for k in ["has_neutron", "n_spine_parts", "spine_protons",
                     "spine_neutrons", "spine_showers", "nu_E",
                     "n_truth_prim", "n_reco_primary", "n_reco_matched"]}

# Particle-level arrays
parr = {}
keys = ["true_pdg", "reco_p_gev", "true_p_gev", "p_ratio", "ang_deg",
        "is_neutron", "spine_class", "nu_E", "cos_theta", "dr_cm"]
for k in keys:
    vals = [r[k] for r in neutron_particle_records]
    parr[k] = np.array(vals, dtype=float if k not in ("is_neutron",) else bool)

# Also keep integer arrays for pdg
parr["true_pdg_int"] = np.array([r["true_pdg"] for r in neutron_particle_records], dtype=int)
parr["spine_class_int"] = np.array([r["spine_class"] for r in neutron_particle_records], dtype=int)

# ---------------------------------------------------------------
# Figure 1: Neutron angular distribution and energy overestimation
# ---------------------------------------------------------------
print("\n--- Figure 1: Neutron angular and energy distributions ---")
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
fig1.suptitle("Neutron reconstruction in SPINE", fontsize=14, fontweight="bold")

n_mask = parr["is_neutron"]
not_n_mask = ~parr["is_neutron"]

# Angular distribution: neutrons vs other particles
ax = axes1[0, 0]
rng = (0, 90)
for mask, lbl, col in [(n_mask, "Neutron-matched", "C0"),
                        (not_n_mask, "Other particles", "C1")]:
    if mask.sum() > 0:
        ax.hist(parr["ang_deg"][mask], bins=np.linspace(0, 90, 45),
                histtype="step", lw=1.5, label=f"{lbl} (N={mask.sum()}, "
                f"med={np.median(parr['ang_deg'][mask]):.1f}°)", color=col)
ax.set_xlabel(r"$\Delta\theta$ [deg]")
ax.set_ylabel("Counts")
ax.set_title("Angular resolution: neutrons vs others")
ax.legend(fontsize=8)

# Angular CDF for neutrons
ax = axes1[0, 1]
for mask, lbl, col in [(n_mask, "Neutron-matched", "C0")]:
    if mask.sum() > 0:
        s = np.sort(parr["ang_deg"][mask][np.isfinite(parr["ang_deg"][mask])])
        cdf = np.arange(1, len(s)+1)/len(s)
        ax.plot(s, cdf, label=lbl, color=col, lw=1.5)
ax.set_xlabel(r"$\Delta\theta$ [deg]")
ax.set_ylabel("CDF")
ax.set_title("Neutron angular resolution CDF")
ax.legend(fontsize=9)
ax.set_xlim(0, 90)

# p_ratio distribution for neutrons
ax = axes1[1, 0]
for mask, lbl, col in [(n_mask, "Neutron-matched", "C0")]:
    if mask.sum() > 0:
        r = parr["p_ratio"][mask]
        r_valid = r[np.isfinite(r) & (r > 0.01) & (r < 100)]
        ax.hist(np.log10(r_valid), bins=50, histtype="step", lw=1.5, color=col,
                label=f"{lbl} (med={np.median(r_valid):.2f})")
ax.axvline(0, color="k", ls=":", lw=0.8)
ax.set_xlabel(r"$\log_{10}(p_{\mathrm{reco}}/p_{\mathrm{true}})$")
ax.set_ylabel("Counts")
ax.set_title("Neutron momentum ratio")
ax.legend(fontsize=9)

# p_reco vs p_true scatter for neutrons
ax = axes1[1, 1]
for mask, lbl, col in [(n_mask, "Neutron-matched", "C0")]:
    if mask.sum() > 0:
        valid = np.isfinite(parr["true_p_gev"][mask]) & np.isfinite(parr["reco_p_gev"][mask]) & (parr["true_p_gev"][mask] > 0)
        ax.scatter(parr["true_p_gev"][mask][valid], parr["reco_p_gev"][mask][valid],
                  s=5, alpha=0.4, rasterized=True, color=col)
lims = [0.01, 10]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"True neutron $p$ [GeV/$c$]")
ax.set_ylabel(r"SPINE reco $p$ [GeV/$c$]")
ax.set_title("Neutron momentum: reco vs true")
ax.set_xlim(lims); ax.set_ylim(lims)

fig1.tight_layout()
fig1.savefig(os.path.join(outdir, "neutron_angular_energy.png"), dpi=150)
plt.close(fig1)

# ---------------------------------------------------------------
# Figure 2: Event-level properties with and without neutrons
# ---------------------------------------------------------------
print("\n--- Figure 2: Event properties with/without neutrons ---")
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle("Event properties: neutron vs no-neutron events", fontsize=14, fontweight="bold")

has_n = np.array([r["has_neutron"] for r in event_records], dtype=bool)
n_parts = np.array([r["n_spine_parts"] for r in event_records], dtype=float)
n_protons = np.array([r["spine_protons"] for r in event_records], dtype=float)
n_showers = np.array([r["spine_showers"] for r in event_records], dtype=float)
nu_E_arr = np.array([r["nu_E"] for r in event_records], dtype=float)

print(f"  Events with neutrons:   {has_n.sum()} / {len(has_n)} ({has_n.sum()/len(has_n)*100:.1f}%)")
print(f"  Events without neutrons: {(~has_n).sum()} / {len(has_n)} ({(~has_n).sum()/len(has_n)*100:.1f}%)")

# SPINE particle multiplicity
ax = axes2[0, 0]
for mask, lbl, col in [(has_n, "Has neutron", "C0"), (~has_n, "No neutron", "C1")]:
    if mask.sum() > 0:
        ax.hist(n_parts[mask], bins=np.arange(0, 20), histtype="step", lw=1.5,
                label=f"{lbl} (med={np.median(n_parts[mask]):.0f})", color=col, density=True)
ax.set_xlabel("N SPINE particles"); ax.set_ylabel("Density")
ax.set_title("SPINE particle multiplicity")
ax.legend(fontsize=9)

# Number of SPINE-identified protons
ax = axes2[0, 1]
for mask, lbl, col in [(has_n, "Has neutron", "C0"), (~has_n, "No neutron", "C1")]:
    if mask.sum() > 0:
        ax.hist(n_protons[mask], bins=np.arange(0, 10), histtype="step", lw=1.5,
                label=f"{lbl} (med={np.median(n_protons[mask]):.1f})", color=col, density=True)
ax.set_xlabel("N SPINE protons"); ax.set_ylabel("Density")
ax.set_title("SPINE proton multiplicity")
ax.legend(fontsize=9)

# Number of SPINE showers
ax = axes2[1, 0]
for mask, lbl, col in [(has_n, "Has neutron", "C0"), (~has_n, "No neutron", "C1")]:
    if mask.sum() > 0:
        ax.hist(n_showers[mask], bins=np.arange(0, 10), histtype="step", lw=1.5,
                label=f"{lbl} (med={np.median(n_showers[mask]):.1f})", color=col, density=True)
ax.set_xlabel("N SPINE showers"); ax.set_ylabel("Density")
ax.set_title("SPINE shower multiplicity")
ax.legend(fontsize=9)

# Neutron presence vs neutrino energy
ax = axes2[1, 1]
# Profile: fraction of events with neutrons vs E
E_bins = np.logspace(-1, 1.5, 15)
centers = 0.5 * (E_bins[:-1] + E_bins[1:])
frac, _, _ = binned_statistic(nu_E_arr, has_n.astype(float), "mean", bins=E_bins)
cnt, _, _ = binned_statistic(nu_E_arr, has_n.astype(float), "count", bins=E_bins)
mk = cnt > 5
ax.errorbar(centers[mk], frac[mk],
            yerr=np.sqrt(frac[mk]*(1-frac[mk])/cnt[mk]),
            marker="o", ms=6, color="C0")
ax.set_xscale("log"); ax.set_xlabel(r"True neutrino $E$ [GeV]")
ax.set_ylabel("Fraction with neutrons")
ax.set_title("Neutron presence vs neutrino energy")

fig2.tight_layout()
fig2.savefig(os.path.join(outdir, "neutron_event_properties.png"), dpi=150)
plt.close(fig2)

# ---------------------------------------------------------------
# Figure 3: Neutron → shower mis-identification
# ---------------------------------------------------------------
print("\n--- Figure 3: Neutron shower mis-ID ---")
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Neutron → Shower mis-identification", fontsize=14, fontweight="bold")

# Among neutron-matched particles, how many are classified as showers vs tracks by SPINE?
n_true_neutrons = parr["is_neutron"].sum()
print(f"  Total neutron-matched particles: {n_true_neutrons}")

# For true neutrons, show SPINE's classification
neutron_spine_pdg = parr["spine_class_int"][n_mask]
unique_spine, counts_spine = np.unique(neutron_spine_pdg, return_counts=True)

ax = axes3[0]
pdg_labels = {11: "e⁻", 22: "γ", 111: "π⁰", 211: "π⁺", 13: "μ⁻",
              2212: "p", 2112: "n", 0: "other"}
colors = {11: "C0", 22: "C1", 111: "C2", 2112: "C3", 2212: "C4",
          211: "C5", 13: "C6", 0: "gray"}

for pdg_val, count in zip(unique_spine, counts_spine):
    lbl = pdg_labels.get(pdg_val, f"PDG{pdg_val}")
    col = colors.get(pdg_val, "gray")
    ax.bar(lbl, count, color=col, alpha=0.8)

# Explicitly mark neutrons as 0% reconstructed — the bar is absent because
# SPINE never classifies matched true neutrons as PDG 2112.
if 2112 not in unique_spine:
    ax.bar("n", 0, color=colors[2112], alpha=0.3)
    ax.annotate(
        "0%\nReconstructed",
        xy=("n", 0), xycoords=("data", "data"),
        xytext=(0, 12), textcoords="offset points",
        ha="center", va="bottom", fontsize=9, fontweight="bold", color="red",
        arrowprops=dict(arrowstyle="-", color="red", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", lw=1.5),
    )

count_shower = sum(counts_spine[unique_spine == 11]) + sum(counts_spine[unique_spine == 22]) + sum(counts_spine[unique_spine == 111])
count_non_shower = n_true_neutrons - count_shower
ax.set_ylabel("N neutron-matched particles")
ax.set_title(f"SPINE classification of true neutrons\n"
             f"Reconstructed as shower: {count_shower}/{n_true_neutrons} "
             f"({count_shower/n_true_neutrons*100:.1f}%)")

# p_ratio by SPINE classification for true neutrons
ax = axes3[1]
for pdg_val in unique_spine:
    mask = (parr["spine_class_int"] == pdg_val) & parr["is_neutron"]
    if mask.sum() < 5:
        continue
    r = parr["p_ratio"][mask]
    r_valid = r[np.isfinite(r) & (r > 0.01) & (r < 100)]
    lbl = pdg_labels.get(pdg_val, f"PDG{pdg_val}")
    col = colors.get(pdg_val, "gray")
    ax.hist(np.log10(r_valid), bins=40, histtype="step", lw=1.5, color=col,
            label=f"{lbl} (med={np.median(r_valid):.2f}, N={mask.sum()})")
ax.axvline(0, color="k", ls=":", lw=0.8)
ax.set_xlabel(r"$\log_{10}(p_{\mathrm{reco}}/p_{\mathrm{true}})$")
ax.set_ylabel("Counts")
ax.set_title("Neutron p_ratio by SPINE classification")
ax.legend(fontsize=8)

fig3.tight_layout()
fig3.savefig(os.path.join(outdir, "neutron_shower_misid.png"), dpi=150)
plt.close(fig3)

# ---------------------------------------------------------------
# Figure 4: Neutron spatial isolation distance (ΔR) for shower mis-ID
# ---------------------------------------------------------------
print("\n--- Figure 4: Neutron mis-ID spatial isolation distance ---")
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle("Spatial isolation of neutrons misidentified as EM showers",
              fontsize=14, fontweight="bold")

# Identify neutrons matched to shower-class SPINE particles
shower_pdgs = {11, 22, 111}
neutron_shower_mask = n_mask & np.isin(parr["spine_class_int"], list(shower_pdgs))
neutron_other_mask = n_mask & ~neutron_shower_mask

dr = parr["dr_cm"]
valid_neutrons = n_mask & np.isfinite(dr) & (dr >= 0)

# Panel a: ΔR distribution for neutron→shower vs other neutron matches
ax = axes4[0]
for mask, lbl, col in [
    (neutron_shower_mask & np.isfinite(dr) & (dr >= 0), "Neutron → EM shower", "#d62728"),
    (neutron_other_mask & np.isfinite(dr) & (dr >= 0), "Other neutron matches", "#1f77b4"),
]:
    if mask.sum() > 5:
        ax.hist(dr[mask], bins=np.linspace(0, 200, 40), histtype="step", lw=1.5,
                color=col, label=f"{lbl} (N={mask.sum()}, med={np.median(dr[mask]):.1f} cm)")
ax.set_xlabel(r"$\Delta R$ (true vtx → reco start) [cm]")
ax.set_ylabel("Counts")
ax.set_title("Spatial isolation of neutron matches")
ax.legend(fontsize=9)

# Panel b: ΔR vs true neutron energy
ax = axes4[1]
neutron_mask = n_mask & np.isfinite(dr) & (dr >= 0) & (parr["true_p_gev"] > 0)
if neutron_mask.sum() > 5:
    ax.scatter(parr["true_p_gev"][neutron_mask], dr[neutron_mask], s=3, alpha=0.3,
               rasterized=True, c=parr["spine_class_int"][neutron_mask], cmap="Set1")
    ax.set_xscale("log")
    ax.set_xlabel(r"True neutron $p$ [GeV/$c$]")
    ax.set_ylabel(r"$\Delta R$ [cm]")
    ax.set_title("ΔR vs true neutron momentum (color = SPINE class)")
# Add horizontal reference lines
ax.axhline(10, color="gray", ls="--", lw=0.8, alpha=0.5, label="10 cm")
ax.axhline(50, color="gray", ls=":", lw=0.8, alpha=0.5, label="50 cm")
ax.legend(fontsize=8)

fig4.tight_layout()
fig4.savefig(os.path.join(outdir, "neutron_spatial_isolation.png"), dpi=150)
plt.close(fig4)
print("  Saved neutron_spatial_isolation.png")

# Print ΔR summary
if n_true_neutrons > 0:
    n_shower_dr = neutron_shower_mask & np.isfinite(dr) & (dr >= 0)
    n_other_dr = neutron_other_mask & np.isfinite(dr) & (dr >= 0)
    if n_shower_dr.sum() > 0:
        print(f"  Neutron→shower ΔR: median={np.median(dr[n_shower_dr]):.1f} cm, "
              f"N={n_shower_dr.sum()}")
    if n_other_dr.sum() > 0:
        print(f"  Other neutron ΔR: median={np.median(dr[n_other_dr]):.1f} cm, "
              f"N={n_other_dr.sum()}")

# ---------------------------------------------------------------
# Figure 5 (Study D): Neutral-hadron impact on global event purity
# ---------------------------------------------------------------
print("\n--- Figure 5: Neutral hadron impact on event-level purity/completeness (Study D) ---")
no_n = ~has_n

# Per-event purity: fraction of reco primaries matched to a truth primary
# Per-event completeness: fraction of truth primaries matched to a reco primary
with np.errstate(invalid="ignore", divide="ignore"):
    ev_purity = np.where(ev_arr["n_reco_primary"] > 0,
                         ev_arr["n_reco_matched"] / ev_arr["n_reco_primary"], np.nan)
    ev_completeness = np.where(ev_arr["n_truth_prim"] > 0,
                               ev_arr["n_reco_matched"] / ev_arr["n_truth_prim"], np.nan)

valid_pur = np.isfinite(ev_purity)
valid_com = np.isfinite(ev_completeness)

fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle("Study D: Neutral Hadron Impact on Global Event Purity & Completeness",
              fontsize=13, fontweight="bold")

# Panel a: Event purity
ax = axes5[0]
bins_pur = np.linspace(0, 1.05, 43)
for mask, lbl, col in [
    (has_n & valid_pur, f"Events with ≥1 neutron (N={has_n.sum():,})", "#d62728"),
    (no_n & valid_pur,  f"Events with 0 neutrons   (N={no_n.sum():,})", "#1f77b4"),
]:
    if mask.sum() > 10:
        med_val = np.median(ev_purity[mask])
        ax.hist(ev_purity[mask], bins=bins_pur, histtype="step", lw=1.8,
                color=col, density=True,
                label=f"{lbl}\nmedian={med_val:.3f}")
        ax.axvline(med_val, color=col, ls="--", lw=1.0, alpha=0.7)
ax.set_xlabel("Event purity  (N matched / N reco primaries)")
ax.set_ylabel("Density")
ax.set_title("Event-level reconstruction purity")
ax.legend(fontsize=9)
ax.set_xlim(0, 1.05)

# Panel b: Event completeness
ax = axes5[1]
bins_com = np.linspace(0, 1.05, 43)
for mask, lbl, col in [
    (has_n & valid_com, f"Events with ≥1 neutron (N={has_n.sum():,})", "#d62728"),
    (no_n & valid_com,  f"Events with 0 neutrons   (N={no_n.sum():,})", "#1f77b4"),
]:
    if mask.sum() > 10:
        med_val = np.median(ev_completeness[mask])
        ax.hist(ev_completeness[mask], bins=bins_com, histtype="step", lw=1.8,
                color=col, density=True,
                label=f"{lbl}\nmedian={med_val:.3f}")
        ax.axvline(med_val, color=col, ls="--", lw=1.0, alpha=0.7)
ax.set_xlabel("Event completeness  (N matched / N truth primaries)")
ax.set_ylabel("Density")
ax.set_title("Event-level reconstruction completeness")
ax.legend(fontsize=9)
ax.set_xlim(0, 1.05)

fig5.tight_layout()
fig5.savefig(os.path.join(outdir, "neutron_purity_impact.png"), dpi=150)
plt.close(fig5)
print("  Saved neutron_purity_impact.png")

# Print Study D summary
for mask, lbl in [(has_n & valid_pur, "with neutrons"), (no_n & valid_pur, "without neutrons")]:
    if mask.sum() > 0:
        print(f"  Event purity  ({lbl}): median={np.median(ev_purity[mask]):.3f}")
for mask, lbl in [(has_n & valid_com, "with neutrons"), (no_n & valid_com, "without neutrons")]:
    if mask.sum() > 0:
        print(f"  Completeness  ({lbl}): median={np.median(ev_completeness[mask]):.3f}")

# ---------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Neutron Island Analysis Summary")
print("=" * 60)

print(f"\n--- Event-level statistics ---")
print(f"  Total events: {len(event_records)}")
print(f"  Events with at least one true neutron: {has_n.sum()} ({has_n.sum()/len(has_n)*100:.1f}%)")
print(f"  SPINE multiplicity (with neutrons): median {np.median(n_parts[has_n]):.1f}")
print(f"  SPINE multiplicity (no neutrons):   median {np.median(n_parts[~has_n]):.1f}")
print(f"  SPINE protons (with neutrons):     median {np.median(n_protons[has_n]):.1f}")
print(f"  SPINE protons (no neutrons):       median {np.median(n_protons[~has_n]):.1f}")
print(f"  SPINE showers (with neutrons):     median {np.median(n_showers[has_n]):.1f}")
print(f"  SPINE showers (no neutrons):       median {np.median(n_showers[~has_n]):.1f}")

print(f"\n--- Neutron particle-level statistics ---")
print(f"  Neutron-matched particles (cosθ > 0.5): {n_true_neutrons}")
if n_true_neutrons > 0:
    n_ang = parr["ang_deg"][n_mask]
    n_pr = parr["p_ratio"][n_mask]
    print(f"  Angular resolution: median {np.median(n_ang):.1f}°")
    print(f"  Momentum ratio:     median {np.median(n_pr[np.isfinite(n_pr) & (n_pr > 0.01) & (n_pr < 100)]):.2f}")
    print(f"  Reconstructed as shower (e/γ/π⁰): {count_shower}/{n_true_neutrons} ({count_shower/n_true_neutrons*100:.1f}%)")
    print(f"  Reconstructed as neutron: {counts_spine[np.where(unique_spine == 2112)][0] if 2112 in unique_spine else 0}/{n_true_neutrons}")
    if 2212 in unique_spine:
        print(f"  Reconstructed as proton: {counts_spine[np.where(unique_spine == 2212)][0]}/{n_true_neutrons}")

# Save summary
_pur_n = float(np.median(ev_purity[has_n & valid_pur])) if (has_n & valid_pur).sum() > 0 else None
_pur_0 = float(np.median(ev_purity[no_n & valid_pur])) if (no_n & valid_pur).sum() > 0 else None
_com_n = float(np.median(ev_completeness[has_n & valid_com])) if (has_n & valid_com).sum() > 0 else None
_com_0 = float(np.median(ev_completeness[no_n & valid_com])) if (no_n & valid_com).sum() > 0 else None
summary = {
    "n_events": len(event_records),
    "n_events_with_neutrons": int(has_n.sum()),
    "n_neutron_matched_particles": int(n_true_neutrons),
    "neutron_angular_resolution_median_deg": float(np.median(parr["ang_deg"][n_mask])) if n_true_neutrons > 0 else None,
    "neutron_momentum_ratio_median": float(np.median(parr["p_ratio"][n_mask][np.isfinite(parr["p_ratio"][n_mask]) & (parr["p_ratio"][n_mask] > 0.01)])) if n_true_neutrons > 0 else None,
    "neutron_shower_misid_fraction": float(count_shower / n_true_neutrons) if n_true_neutrons > 0 else None,
    "study_d_purity_with_neutrons": _pur_n,
    "study_d_purity_without_neutrons": _pur_0,
    "study_d_completeness_with_neutrons": _com_n,
    "study_d_completeness_without_neutrons": _com_0,
}
with open(os.path.join(outdir, "neutron_island_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Summary saved to figures/neutron_island_summary.json")

# ---------------------------------------------------------------
# Figure 6: ΔR geometric veto — shower purity vs efficiency trade-off
# For matched reco EM showers:
#   - True EM (true_pdg in {11,22}): we want to KEEP these
#   - Neutron fake (true_pdg == 2112): we want to REMOVE these
# A veto "reject particle if ΔR > threshold" removes displaced fakes but
# also sacrifices some photons that convert away from the vertex.
# ---------------------------------------------------------------
print("\n--- Figure 6: ΔR veto — shower purity vs efficiency ---")

shower_pdg_set = {11, 22, 111}
em_true_pdg    = {11, 22}

# Matched particles that SPINE calls an EM shower
is_spine_shower = np.isin(parr["spine_class_int"], list(shower_pdg_set))
# Among those: true EM vs neutron-induced
is_true_em      = np.isin(parr["true_pdg_int"], list(em_true_pdg)) & is_spine_shower
is_neutron_fake = (parr["true_pdg_int"] == 2112) & is_spine_shower
is_other_fake   = is_spine_shower & ~is_true_em & ~is_neutron_fake  # proton/pion → shower mis-ID

dr_all  = parr["dr_cm"]

# Finite ΔR for each category
dr_true_em = dr_all[is_true_em    & np.isfinite(dr_all)]
dr_neut_fk = dr_all[is_neutron_fake & np.isfinite(dr_all)]
dr_oth_fk  = dr_all[is_other_fake & np.isfinite(dr_all)]

# Compute purity/efficiency as function of ΔR threshold
thresholds = np.linspace(0, 50, 200)  # cm
n_true = len(dr_true_em)
n_fake_n = len(dr_neut_fk)

eff_true  = np.array([float((dr_true_em <= t).sum()) / max(n_true, 1) for t in thresholds])
rej_neut  = np.array([float((dr_neut_fk > t).sum())  / max(n_fake_n, 1) for t in thresholds])
# Purity after veto = N_true_kept / (N_true_kept + N_fake_kept)
n_true_kept = np.array([(dr_true_em <= t).sum() for t in thresholds], dtype=float)
n_nfake_kpt = np.array([(dr_neut_fk <= t).sum() for t in thresholds], dtype=float)
n_ofake_kpt = np.array([(dr_oth_fk  <= t).sum() for t in thresholds], dtype=float)
denom = n_true_kept + n_nfake_kpt + n_ofake_kpt
purity_after = np.where(denom > 0, n_true_kept / denom, float('nan'))
purity_before = float(n_true) / max(n_true + n_fake_n + len(dr_oth_fk), 1)

fig6, axes6 = plt.subplots(1, 3, figsize=(18, 6))
fig6.suptitle(r"ΔR Geometric Veto: Reject EM Showers with Distance-to-Vertex > $d_{\rm cut}$",
              fontsize=13, fontweight="bold")

# Panel a: ΔR distributions for true EM vs neutron fake showers
ax = axes6[0]
bins_dr = np.linspace(0, 50, 51)
ax.hist(dr_true_em, bins=bins_dr, histtype="step", lw=1.5, color="#2ca02c",
        density=True, label=f"True EM shower (N={n_true:,}, med={np.median(dr_true_em):.1f} cm)")
ax.hist(dr_neut_fk, bins=bins_dr, histtype="step", lw=1.5, color="#d62728",
        density=True, label=f"Neutron fake shower (N={n_fake_n:,}, med={np.median(dr_neut_fk):.1f} cm)")
ax.hist(dr_oth_fk,  bins=bins_dr, histtype="step", lw=1.5, color="#ff7f0e",
        density=True, ls="--", label=f"Other fake shower (N={len(dr_oth_fk):,})")
ax.set_xlabel(r"$\Delta R$ (reco vertex → reco shower start) [cm]")
ax.set_ylabel("Density")
ax.set_title("ΔR distributions by true particle type")
ax.legend(fontsize=8)
ax.axvline(2.0, color="k", ls=":", lw=1, label="2 cm")

# Panel b: efficiency (true EM kept) vs rejection (neutron fake removed) as function of cut
ax = axes6[1]
ax.plot(thresholds, eff_true * 100,  color="#2ca02c", lw=2, label="True EM efficiency (ΔR < cut)")
ax.plot(thresholds, rej_neut * 100,  color="#d62728", lw=2, label="Neutron fake rejection (ΔR > cut)")
# Highlight operating point at 2 cm
t2_idx = np.argmin(np.abs(thresholds - 2.0))
ax.scatter([2.0], [eff_true[t2_idx]*100],  s=80, color="#2ca02c", zorder=5, marker="*")
ax.scatter([2.0], [rej_neut[t2_idx]*100],  s=80, color="#d62728", zorder=5, marker="*")
ax.axvline(2.0, color="k", ls=":", lw=0.8)
ax.set_xlabel(r"ΔR cut threshold [cm]")
ax.set_ylabel("Fraction [%]")
ax.set_title("Efficiency vs rejection as a function of cut")
ax.legend(fontsize=8)
ax.set_xlim(0, 30); ax.set_ylim(0, 105)
ax.text(2.2, 50, f"At 2 cm:\n  EM eff: {eff_true[t2_idx]*100:.1f}%\n  n-rej: {rej_neut[t2_idx]*100:.1f}%",
        fontsize=9, va="center",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

# Panel c: shower purity vs ΔR cut
ax = axes6[2]
ax.plot(thresholds, purity_after * 100, color="#1f77b4", lw=2, label="Shower purity after veto")
ax.axhline(purity_before * 100, color="gray", ls="--", lw=1,
           label=f"Baseline purity (no cut): {purity_before*100:.1f}%")
ax.axvline(2.0, color="k", ls=":", lw=0.8)
ax.scatter([2.0], [purity_after[t2_idx]*100], s=80, color="#1f77b4", zorder=5, marker="*")
ax.set_xlabel(r"ΔR cut threshold [cm]")
ax.set_ylabel("Shower purity (matched EM / all showers) [%]")
ax.set_title("Shower purity improvement from ΔR veto")
ax.legend(fontsize=8)
ax.set_xlim(0, 30)
ax.text(2.2, purity_after[t2_idx]*100 - 5,
        f"At 2 cm: {purity_after[t2_idx]*100:.1f}%",
        fontsize=9, color="#1f77b4",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#1f77b4", alpha=0.9))

fig6.tight_layout()
fig6.savefig(os.path.join(outdir, "neutron_dr_veto.png"), dpi=150)
plt.close(fig6)
print(f"  Saved neutron_dr_veto.png")
print(f"  Baseline matched-shower purity: {purity_before*100:.1f}%")
print(f"  At ΔR > 2 cm veto: EM efficiency={eff_true[t2_idx]*100:.1f}%, "
      f"neutron rejection={rej_neut[t2_idx]*100:.1f}%, "
      f"purity={purity_after[t2_idx]*100:.1f}%")

print("Done!")

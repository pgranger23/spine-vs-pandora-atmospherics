#!/usr/bin/env python3
"""
Particle reconstruction efficiency vs true kinetic energy — SPINE vs Pandora.

For each truth primary a particle is considered "reconstructed" at two levels:
  geometric  : a reco primary exists within cosθ > 0.5 of the truth direction
  full       : geometric AND E_reco_total / E_true_total > 0.5
               (the ">50% efficiency and >50% completeness" criterion)

Efficiency = N_reconstructed / N_truth_total in each true-KE bin.

Full CC sample (no event cap) for proton statistics.

Output:
  figures/reconstruction_efficiency.png   — main 2×2 per-type efficiency
  figures/proton_reco_efficiency.png      — proton deep dive
  figures/reco_efficiency_summary.json
"""

import os, sys, ctypes, math, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from tqdm import tqdm

base_dir = os.getcwd()
lib_path = os.path.join(base_dir, "duneanaobj/build/Linux/lib")
py_path  = os.path.join(base_dir, "duneanaobj/build/Linux/python/3.13")
if py_path not in sys.path:
    sys.path.insert(0, py_path)

for _lib in ["libSRProxy_BasicTypes.dylib", "libduneanaobj_StandardRecord.dylib",
             "libduneanaobj_StandardRecordProxy.dylib"]:
    try:
        ctypes.CDLL(os.path.join(lib_path, _lib))
    except OSError:
        pass

import pySRProxy
import plot_style
plot_style.apply()
from plot_style import PARTICLE_COLORS

outdir = "figures"
os.makedirs(outdir, exist_ok=True)

_nan = float("nan")
COS_THRESH  = 0.5    # geometric matching threshold
P_MIN_GEV   = 0.02   # minimum truth primary momentum [GeV/c]
E_RATIO_MIN = 0.5    # energy completeness threshold (E_reco / E_true > this)
EM_CALIB    = 0.198  # ADC→e⁻ correction for EM particles (PDG 11, 22, 111)

EM_PDGS = {11, -11, 22, 111}

_PDG_MASS = {
    11: 0.000511, 13: 0.10566, 211: 0.13957, 321: 0.49368,
    2212: 0.93827, 2112: 0.93957, 22: 0.0, 111: 0.13498,
}

def _vmag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def _vnorm(x, y, z):
    n = _vmag(x, y, z)
    return (x/n, y/n, z/n) if n > 1e-6 else (_nan, _nan, _nan)

def _cos(ax, ay, az, bx, by, bz):
    na = _vmag(ax, ay, az)
    nb = _vmag(bx, by, bz)
    if na < 1e-6 or nb < 1e-6:
        return _nan
    return max(-1.0, min(1.0, (ax*bx + ay*by + az*bz) / (na * nb)))

def _true_E(pdg, p):
    m = _PDG_MASS.get(abs(pdg), 0.0)
    return math.sqrt(p*p + m*m)

def _true_KE(pdg, p):
    m = _PDG_MASS.get(abs(pdg), 0.0)
    return math.sqrt(p*p + m*m) - m

def _closest_dlp(ixn, nu_vtx):
    best, bi = float("inf"), 0
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = math.sqrt((v.x-nu_vtx.x)**2 + (v.y-nu_vtx.y)**2 + (v.z-nu_vtx.z)**2)
        if d < best:
            best, bi = d, i
    return bi

def pdg_cat(pdg):
    a = abs(pdg)
    if a == 11:   return "e"
    if a == 13:   return "mu"
    if a == 22:   return "gamma"
    if a == 111:  return "pi0"
    if a == 211:  return "pi"
    if a == 321:  return "K"
    if a == 2212: return "p"
    if a == 2112: return "n"
    return "other"

# ---------------------------------------------------------------
# Scan
# ---------------------------------------------------------------
print("=" * 60)
print("Particle reconstruction efficiency vs true KE")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader("recTree", "spine_merged_reprocess.root")

# Each record: truth primary properties + match status for SPINE and Pandora
truth_records = []
n_cc = 0

for ientry in tqdm(range(prox.entries()), desc="Efficiency scan"):
    rec = prox.entry(ientry)
    mc  = rec.mc
    if mc.nnu < 1 or not mc.nu[0].iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu = mc.nu[0]
    n_cc += 1

    # --- Truth primaries ---
    truth = []
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        p_mag = _vmag(px, py, pz)
        if p_mag < P_MIN_GEV:
            continue
        dx, dy, dz = _vnorm(px, py, pz)
        if math.isnan(dx):
            continue
        pdg   = int(tp.pdg)
        t_E   = _true_E(pdg, p_mag)
        t_KE  = t_E - _PDG_MASS.get(abs(pdg), 0.0)
        truth.append({"pdg": pdg, "p": p_mag, "t_E": t_E, "t_KE": t_KE,
                       "dx": dx, "dy": dy, "dz": dz})

    if not truth:
        continue

    si  = _closest_dlp(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    # --- SPINE primaries ---
    spine = []
    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        if not p.primary:
            continue
        px, py, pz = p.p.x, p.p.y, p.p.z
        pmag = _vmag(px, py, pz)
        if not (pmag > 1):     # MeV/c, NaN-safe
            continue
        dx, dy, dz = _vnorm(px, py, pz)
        if math.isnan(dx):
            continue
        rpdg = int(p.pdg)
        E_kin = float(p.E)     # kinetic energy in MeV
        if abs(rpdg) in EM_PDGS:
            E_kin *= EM_CALIB
        mass  = _PDG_MASS.get(abs(rpdg), 0.0)
        r_E   = E_kin * 0.001 + mass   # total reco energy [GeV]
        spine.append({"dx": dx, "dy": dy, "dz": dz, "r_E": r_E})

    # --- Pandora primaries ---
    has_pand = (hasattr(ixn, "pandora") and ixn.npandora > 0 and
                hasattr(rec, "fd") and hasattr(rec.fd, "hd") and
                hasattr(rec.fd.hd, "pandora"))
    pandora = []
    if has_pand:
        try:
            pand_fd    = rec.fd.hd.pandora[0]
            pand_parts = ixn.pandora[0].part
            npand      = pand_parts.npandora - 1
        except Exception:
            has_pand = False
    if has_pand:
        for pi in range(npand):
            p = pand_parts.pandora[pi]
            if not p.primary:
                continue
            rpdg = int(p.pdg)
            sd   = pand_fd.showers[pi].direction
            sdn  = math.sqrt(sd.x**2 + sd.y**2 + sd.z**2)
            if sdn > 1e-6:
                dx, dy, dz = sd.x/sdn, sd.y/sdn, sd.z/sdn
            else:
                td  = pand_fd.tracks[pi].dir
                tdn = math.sqrt(td.x**2 + td.y**2 + td.z**2)
                if tdn < 1e-6:
                    continue
                dx, dy, dz = td.x/tdn, td.y/tdn, td.z/tdn
            E_mev = float(p.E)
            if not (E_mev > 0):
                E_mev = float(pand_fd.tracks[pi].E)
            if not (E_mev > 0):
                E_mev = float(pand_fd.showers[pi].Evis)
            mass  = _PDG_MASS.get(abs(rpdg), 0.0)
            r_E   = E_mev * 0.001 + mass if E_mev > 0 else _nan
            pandora.append({"dx": dx, "dy": dy, "dz": dz, "r_E": r_E})

    # --- Match each truth primary to best SPINE / Pandora particle ---
    def _best_match(truth_p, reco_list):
        best_cos, best_E = -2.0, _nan
        for rp in reco_list:
            c = _cos(truth_p["dx"], truth_p["dy"], truth_p["dz"],
                     rp["dx"], rp["dy"], rp["dz"])
            if c > best_cos:
                best_cos, best_E = c, rp["r_E"]
        if best_cos > COS_THRESH:
            geom = True
            if math.isfinite(best_E) and best_E > 0:
                full = (best_E / truth_p["t_E"]) > E_RATIO_MIN
                e_ratio = best_E / truth_p["t_E"]
            else:
                full, e_ratio = False, _nan
        else:
            geom, full, e_ratio = False, False, _nan
        return geom, full, e_ratio

    for tp in truth:
        s_geom, s_full, s_eratio = _best_match(tp, spine)
        p_geom, p_full, p_eratio = _best_match(tp, pandora)
        truth_records.append({
            "pdg": tp["pdg"], "p": tp["p"],
            "t_E": tp["t_E"], "t_KE": tp["t_KE"],
            "s_geom": s_geom, "s_full": s_full, "s_eratio": s_eratio,
            "p_geom": p_geom, "p_full": p_full, "p_eratio": p_eratio,
        })

print(f"\n  CC events: {n_cc:,}")
print(f"  Truth primaries recorded: {len(truth_records):,}")

# Convert to arrays
arr = {k: np.array([r[k] for r in truth_records]) for k in truth_records[0]}
arr["pdg"]    = arr["pdg"].astype(int)
arr["s_geom"] = arr["s_geom"].astype(bool)
arr["s_full"] = arr["s_full"].astype(bool)
arr["p_geom"] = arr["p_geom"].astype(bool)
arr["p_full"] = arr["p_full"].astype(bool)

# ---------------------------------------------------------------
# Helper: efficiency profile
# ---------------------------------------------------------------
def _eff_profile(ke_vals, passed, bins):
    tot, _, _  = binned_statistic(ke_vals, ke_vals, statistic="count", bins=bins)
    hit, _, _  = binned_statistic(ke_vals, passed.astype(float), statistic="sum", bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    eff = np.where(tot > 0, hit / tot, _nan)
    err = np.where(tot > 0, np.sqrt(eff * (1 - np.where(np.isnan(eff), 0, eff)) / np.maximum(tot, 1)), _nan)
    return centers, eff, err, tot

# ---------------------------------------------------------------
# Figure 1: Main efficiency vs KE — 2×2 (e, mu, pi, p)
# ---------------------------------------------------------------
print("\n--- Figure 1: Reconstruction efficiency vs true KE ---")

main_cats = [("e", "Electron", np.logspace(-3, 0.5, 25)),
             ("mu", "Muon",    np.logspace(-3, 0.5, 25)),
             ("pi", "Pion",    np.logspace(-3, 0.5, 25)),
             ("p",  "Proton",  np.linspace(0, 1.5, 25))]

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(
    r"Particle Reconstruction Efficiency vs True Kinetic Energy — SPINE vs Pandora"
    "\n(geometric: cosθ > 0.5  |  full: cosθ > 0.5 AND $E_\mathrm{reco}/E_\mathrm{true}$ > 0.5)",
    fontsize=12, fontweight="bold")

for ax, (cat, cat_lbl, ke_bins) in zip(axes1.flatten(), main_cats):
    mask = np.array([pdg_cat(p) == cat for p in arr["pdg"]])
    ke   = arr["t_KE"][mask]
    valid = np.isfinite(ke) & (ke >= 0)
    ke = ke[valid]

    for label, geom_arr, full_arr, col, ls_g, ls_f in [
        ("SPINE",   arr["s_geom"][mask][valid], arr["s_full"][mask][valid], "#1f77b4", "-",  "--"),
        ("Pandora", arr["p_geom"][mask][valid], arr["p_full"][mask][valid], "#ff7f0e", "-",  "--"),
    ]:
        if ke.sum() == 0:
            continue
        c, eff_g, err_g, tot = _eff_profile(ke, geom_arr, ke_bins)
        c, eff_f, err_f, _   = _eff_profile(ke, full_arr,  ke_bins)
        mk = tot > 5
        ax.plot(c[mk], eff_g[mk], color=col, ls=ls_g, lw=2.0,
                label=f"{label} geometric")
        ax.fill_between(c[mk],
                        np.clip(eff_g[mk] - err_g[mk], 0, 1),
                        np.clip(eff_g[mk] + err_g[mk], 0, 1),
                        color=col, alpha=0.10)
        ax.plot(c[mk], eff_f[mk], color=col, ls=ls_f, lw=1.5,
                label=f"{label} full (E>50%)")

    pcol = PARTICLE_COLORS.get(cat, "black")
    ax.set_xlabel("True kinetic energy [GeV]")
    ax.set_ylabel("Reconstruction efficiency")
    ax.set_title(cat_lbl, color=pcol, fontweight="bold")
    ax.set_ylim(-0.02, 1.08)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.6, label="50%")
    if cat in ("e", "mu", "pi"):
        ax.set_xscale("log")
        plot_style.fmt_log_ticks(ax)
    ax.legend(fontsize=8, ncol=2)

fig1.tight_layout()
fig1.savefig(os.path.join(outdir, "reconstruction_efficiency.png"), dpi=150)
plt.close(fig1)
print("  Saved reconstruction_efficiency.png")

# ---------------------------------------------------------------
# Figure 2: Proton deep dive — 2×2
# ---------------------------------------------------------------
print("\n--- Figure 2: Proton deep dive ---")

pmask = np.array([pdg_cat(p) == "p" for p in arr["pdg"]])
p_ke      = arr["t_KE"][pmask]     # true kinetic energy [GeV]
p_mom     = arr["p"][pmask]         # true momentum [GeV/c]
p_s_geom  = arr["s_geom"][pmask]
p_s_full  = arr["s_full"][pmask]
p_p_geom  = arr["p_geom"][pmask]
p_p_full  = arr["p_full"][pmask]
p_s_ratio = arr["s_eratio"][pmask]
p_p_ratio = arr["p_eratio"][pmask]

valid_p   = np.isfinite(p_ke) & (p_ke >= 0)
p_ke      = p_ke[valid_p]
p_mom     = p_mom[valid_p]
p_s_geom  = p_s_geom[valid_p]; p_s_full  = p_s_full[valid_p]
p_p_geom  = p_p_geom[valid_p]; p_p_full  = p_p_full[valid_p]
p_s_ratio = p_s_ratio[valid_p]; p_p_ratio = p_p_ratio[valid_p]

print(f"  Truth protons in sample: {len(p_ke):,}")
print(f"  SPINE geom matched: {p_s_geom.sum():,} ({100*p_s_geom.mean():.1f}%)")
print(f"  SPINE full matched: {p_s_full.sum():,} ({100*p_s_full.mean():.1f}%)")
print(f"  Pandora geom matched: {p_p_geom.sum():,} ({100*p_p_geom.mean():.1f}%)")
print(f"  Pandora full matched: {p_p_full.sum():,} ({100*p_p_full.mean():.1f}%)")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))
fig2.suptitle("Proton Reconstruction Efficiency — Deep Dive\n"
              r"(geometric: cosθ > 0.5  |  full: AND $E_\mathrm{reco}/E_\mathrm{true}$ > 0.5)",
              fontsize=12, fontweight="bold")

# --- Panel [0,0]: efficiency vs KE, fine bins ---
ax = axes2[0, 0]
ke_bins_fine = np.linspace(0, 1.5, 40)
for label, geom_a, full_a, col in [
    ("SPINE",   p_s_geom, p_s_full, "#1f77b4"),
    ("Pandora", p_p_geom, p_p_full, "#ff7f0e"),
]:
    c, eff_g, err_g, tot = _eff_profile(p_ke, geom_a, ke_bins_fine)
    c, eff_f, err_f, _   = _eff_profile(p_ke, full_a,  ke_bins_fine)
    mk = tot > 10
    ax.plot(c[mk], eff_g[mk], color=col, ls="-",  lw=2.0, label=f"{label} geometric")
    ax.fill_between(c[mk],
                    np.clip(eff_g[mk] - err_g[mk], 0, 1),
                    np.clip(eff_g[mk] + err_g[mk], 0, 1),
                    color=col, alpha=0.12)
    ax.plot(c[mk], eff_f[mk], color=col, ls="--", lw=1.5, label=f"{label} full (E>50%)")

ax.axhline(0.5, color="k", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("True proton KE [GeV]")
ax.set_ylabel("Efficiency")
ax.set_title("Proton efficiency vs true KE (fine bins)")
ax.set_ylim(-0.02, 1.08); ax.set_xlim(0, 1.5)
ax.legend(fontsize=9, ncol=2)
# Mark 50% efficiency crossing points
for arr_eff, col, ls in [(p_s_full, "#1f77b4", "-"), (p_p_full, "#ff7f0e", "-")]:
    c_f, eff_f, _, tot_f = _eff_profile(p_ke, arr_eff, ke_bins_fine)
    mk = tot_f > 10
    cross = c_f[mk][np.where(eff_f[mk] >= 0.5)[0][0]] if (eff_f[mk] >= 0.5).any() else None
    if cross is not None:
        ax.axvline(cross, color=col, ls=":", lw=1.5, alpha=0.7)
        ax.text(cross + 0.02, 0.52, f"{cross*1000:.0f} MeV", color=col, fontsize=8)

# --- Panel [0,1]: efficiency vs momentum ---
ax = axes2[0, 1]
p_bins = np.linspace(0, 2.0, 35)
for label, geom_a, full_a, col in [
    ("SPINE",   p_s_geom, p_s_full, "#1f77b4"),
    ("Pandora", p_p_geom, p_p_full, "#ff7f0e"),
]:
    c, eff_g, err_g, tot = _eff_profile(p_mom, geom_a, p_bins)
    c, eff_f, err_f, _   = _eff_profile(p_mom, full_a,  p_bins)
    mk = tot > 10
    ax.plot(c[mk], eff_g[mk], color=col, ls="-",  lw=2.0, label=f"{label} geometric")
    ax.fill_between(c[mk],
                    np.clip(eff_g[mk] - err_g[mk], 0, 1),
                    np.clip(eff_g[mk] + err_g[mk], 0, 1),
                    color=col, alpha=0.12)
    ax.plot(c[mk], eff_f[mk], color=col, ls="--", lw=1.5, label=f"{label} full (E>50%)")

ax.axhline(0.5, color="k", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("True proton momentum [GeV/c]")
ax.set_ylabel("Efficiency")
ax.set_title("Proton efficiency vs true momentum")
ax.set_ylim(-0.02, 1.08); ax.set_xlim(0, 2.0)
ax.legend(fontsize=9, ncol=2)

# --- Panel [1,0]: E_reco/E_true distribution for matched protons ---
ax = axes2[1, 0]
bins_r = np.linspace(0, 2.5, 50)
for label, ratio, geom, col in [
    ("SPINE",   p_s_ratio, p_s_geom, "#1f77b4"),
    ("Pandora", p_p_ratio, p_p_geom, "#ff7f0e"),
]:
    valid_r = np.isfinite(ratio) & geom
    if valid_r.sum() > 10:
        med_r = float(np.median(ratio[valid_r]))
        ax.hist(ratio[valid_r], bins=bins_r, histtype="step", lw=1.8, color=col, density=True,
                label=f"{label} (N={valid_r.sum():,}, med={med_r:.2f})")
ax.axvline(0.5, color="k", ls="--", lw=1.2, label="50% threshold")
ax.axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
ax.set_xlabel(r"$E_\mathrm{reco\,total} / E_\mathrm{true\,total}$  (geometrically matched protons)")
ax.set_ylabel("Density")
ax.set_title("Proton energy completeness (geometrically matched)")
ax.legend(fontsize=9)
ax.set_xlim(0, 2.5)

# --- Panel [1,1]: N truth protons per KE bin (denominator statistics) ---
ax = axes2[1, 1]
ke_bins_stat = np.linspace(0, 2.0, 40)
centers_stat = 0.5 * (ke_bins_stat[:-1] + ke_bins_stat[1:])
counts_all, _ = np.histogram(p_ke, bins=ke_bins_stat)
ax.bar(centers_stat, counts_all,
       width=(ke_bins_stat[1]-ke_bins_stat[0])*0.9,
       color=PARTICLE_COLORS.get("p", "#9467bd"), alpha=0.7, edgecolor="k", lw=0.4)
ax.set_xlabel("True proton KE [GeV]")
ax.set_ylabel("N truth protons (denominator)")
ax.set_title("Proton statistics per KE bin")
ax.set_xlim(0, 2.0)
# Overlay cumulative fraction on secondary axis
ax2 = ax.twinx()
cumfrac = np.cumsum(counts_all) / max(counts_all.sum(), 1)
ax2.plot(centers_stat, cumfrac, "r-", lw=1.5, label="Cumulative fraction")
ax2.axhline(0.5, color="red", ls=":", lw=0.8, alpha=0.6)
ax2.set_ylabel("Cumulative fraction", color="red")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_ylim(0, 1.05)
ax.set_xlim(0, 2.0)

fig2.tight_layout()
fig2.savefig(os.path.join(outdir, "proton_reco_efficiency.png"), dpi=150)
plt.close(fig2)
print("  Saved proton_reco_efficiency.png")

# ---------------------------------------------------------------
# Summary statistics per particle type
# ---------------------------------------------------------------
print("\n  Overall efficiencies (all true KE):")
print(f"  {'Type':>6}  {'N truth':>9}  {'SPINE geom':>11}  {'SPINE full':>11}  {'Pand geom':>10}  {'Pand full':>10}")

cat_list = ["e", "mu", "gamma", "pi", "p", "n"]
summary = {"n_cc_events": n_cc, "n_truth_primaries": len(truth_records)}
for cat in cat_list:
    mask = np.array([pdg_cat(p) == cat for p in arr["pdg"]])
    n    = int(mask.sum())
    if n == 0:
        continue
    sg = float(arr["s_geom"][mask].mean())
    sf = float(arr["s_full"][mask].mean())
    pg = float(arr["p_geom"][mask].mean())
    pf = float(arr["p_full"][mask].mean())
    print(f"  {cat:>6}  {n:>9,}  {sg:>11.3f}  {sf:>11.3f}  {pg:>10.3f}  {pf:>10.3f}")
    summary[cat] = {"n": n, "spine_geom": round(sg, 4), "spine_full": round(sf, 4),
                    "pandora_geom": round(pg, 4), "pandora_full": round(pf, 4)}

# Proton 50%-efficiency KE threshold
for label, arr_eff, col in [("SPINE",   p_s_full, "s"),
                             ("Pandora", p_p_full, "p")]:
    c_f, eff_f, _, tot_f = _eff_profile(p_ke, arr_eff, ke_bins_fine)
    mk = tot_f > 10
    idxs = np.where(eff_f[mk] >= 0.5)[0]
    thr = float(c_f[mk][idxs[0]]) if len(idxs) > 0 else _nan
    summary[f"proton_50pct_KE_GeV_{col}"] = round(thr, 4) if math.isfinite(thr) else None
    print(f"  Proton 50%-efficiency threshold ({label}): {thr*1000:.0f} MeV KE")

with open(os.path.join(outdir, "reco_efficiency_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved reco_efficiency_summary.json")
print("\nDone!")

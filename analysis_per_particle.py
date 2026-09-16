#!/usr/bin/env python3
"""
Per-particle comparison: SPINE (dlp.part.dlp) and Pandora (pandora.part.pandora)
vs truth primaries.

Uses geometric matching (angular proximity) to associate reco particles
with truth primaries. Pandora uses fd.hd.pandora direction vectors.

Usage:
    python analysis_per_particle.py
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
from plot_style import SPINE_COLOR, PARTICLE_COLORS

import matplotlib.ticker as mticker

def _fmt_log_ticks(ax):
    plot_style.fmt_log_ticks(ax)

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

MASS = {11: 0.000511, 13: 0.1057, 211: 0.1396, 321: 0.4937,
        2212: 0.9383, 2112: 0.9396, 22: 0.0, 111: 0.1350, 311: 0.4976,
        -11: 0.000511, -13: 0.1057, -211: 0.1396, -321: 0.4937}

TRACK_PDGS = {13, 211, 321, 2212, 2112, -13, -211, -321}
SHOWER_PDGS = {11, 22, 111, 311, -11}
EM_PDGS = {11, 22, 111, -11}  # particles using calorimetry (ADC-to-electron correction)
EM_CALIB_FACTOR = 0.198       # ADC-to-electron calibration correction

def _true_energy(pdg, px, py, pz):
    m = MASS.get(abs(pdg), 0.0)
    p2 = px*px + py*py + pz*pz
    return math.sqrt(p2 + m*m)

def pdg_category(pdg):
    apdg = abs(pdg)
    if apdg == 11: return "e"
    if apdg == 13: return "mu"
    if apdg == 22: return "gamma"
    if apdg == 111: return "pi0"
    if apdg == 211: return "pi"
    if apdg == 321: return "K"
    if apdg == 2212: return "p"
    if apdg == 2112: return "n"
    return "other"

cat_order = ["e", "mu", "gamma", "pi0", "pi", "K", "p", "n", "other"]

def iqr_half(x):
    return np.abs(np.percentile(x, 84) - np.percentile(x, 16)) / 2

# ============================================================
# Load data
# ============================================================
print("=" * 60)
print("SPINE + Pandora per-particle analysis")
print("=" * 60)
print("\nNOTE: Pandora per-particle momentum is NaN in this file,")
print("so directional matching uses fd.hd.pandora track/shower dir. vectors.")
print("Reco energy is estimated from fd.track.E or fd.shower.Evis where available.\n")

prox = pySRProxy.StandardRecordFileReader("recTree", caf_env.CAF_FILE)

_PDG_MASS = {
    11: 0.000511, 13: 0.10566, 211: 0.13957, 321: 0.49368,
    2212: 0.93827, 2112: 0.93957, 22: 0.0, 111: 0.13498,
    311: 0.49761, 310: 0.49761,
}

MAX_EVENTS = 5000
mat_nmin_true = 0.02  # GeV/c
nmatch_threshold = 0.5  # cosθ threshold

particles = []        # SPINE matched pairs
pandora_particles = []  # Pandora matched pairs
event_stats = []
ntot = 0

for ientry in tqdm(range(prox.entries()), desc="Scanning events", total=prox.entries()):
    if len(event_stats) >= MAX_EVENTS:
        break

    rec = prox.entry(ientry)
    mc = rec.mc
    if not (mc.nnu > 0 and mc.nu[0].iscc):
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu = mc.nu[0]
    ntot += 1
    try:
        _nu_mode_int = int(nu.mode)
    except Exception:
        _nu_mode_int = -1
    _nu_mode_lbl = {0: "QE", 1: "RES", 2: "DIS", 3: "COH", 4: "MEC"}.get(_nu_mode_int, "other")
    _nu_E = float(nu.E)

    si = _closest_dlp_idx(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    # Truth primaries
    truth_parts = []
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        p_mag = _vec_mag(px, py, pz)
        if p_mag < mat_nmin_true:
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

    # SPINE reco particles
    spine_parts = []
    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        if not p.primary:
            continue
        px, py, pz = p.p.x, p.p.y, p.p.z
        p_mag = _vec_mag(px, py, pz)
        if not (p_mag > 1):  # MeV/c; NaN-safe
            continue
        dx, dy, dz = _vec_norm_dir(px, py, pz)
        if math.isnan(dx):
            continue
        mass_gev = _PDG_MASS.get(abs(p.pdg), 0)
        p_mag_gev = p_mag * 0.001  # MeV → GeV
        E_kin_mev = p.E           # raw kinetic energy in MeV
        if abs(p.pdg) in EM_PDGS:
            p_mag_gev *= EM_CALIB_FACTOR
            E_kin_mev *= EM_CALIB_FACTOR
        spine_parts.append({
            "idx": pi, "pdg": p.pdg, "p_mag": p_mag_gev,
            "dx": dx, "dy": dy, "dz": dz,
            "E": E_kin_mev * 0.001 + mass_gev,  # total energy (GeV)
            "E_kin_mev": E_kin_mev,
        })

    nspine = len(spine_parts)

    # Pandora reco particles
    pandora_parts = []
    has_pandora = (hasattr(ixn, "pandora") and ixn.npandora > 0 and
                   hasattr(rec, "fd") and hasattr(rec.fd, "hd") and
                   hasattr(rec.fd.hd, "pandora"))
    if has_pandora:
        try:
            pand_fd = rec.fd.hd.pandora[0]
        except:
            has_pandora = False
    if has_pandora:
        pand_fd = rec.fd.hd.pandora[0]
        pand_parts = ixn.pandora[0].part
        npand = pand_parts.npandora - 1  # skip last fake particle
        for pi in range(npand):
            p = pand_parts.pandora[pi]
            if not p.primary:
                continue
            pdg_val = int(p.pdg)
            # Direction from shower (always populated) or track
            sd = pand_fd.showers[pi].direction
            sdn = math.sqrt(sd.x**2 + sd.y**2 + sd.z**2)
            if sdn > 1e-6:
                dx, dy, dz = sd.x / sdn, sd.y / sdn, sd.z / sdn
            else:
                td = pand_fd.tracks[pi].dir
                tdn = math.sqrt(td.x**2 + td.y**2 + td.z**2)
                if tdn > 1e-6:
                    dx, dy, dz = td.x / tdn, td.y / tdn, td.z / tdn
                else:
                    continue
            # Energy estimate
            E_mev = p.E  # common E
            if E_mev <= 0:
                E_mev = pand_fd.tracks[pi].E
            if E_mev <= 0:
                E_mev = pand_fd.tracks[pi].Evis
            if E_mev <= 0:
                E_mev = pand_fd.showers[pi].Evis
            mass_gev = _PDG_MASS.get(abs(pdg_val), 0)
            E_gev = E_mev * 0.001 + mass_gev if E_mev > 0 else _nan
            pandora_parts.append({
                "idx": pi, "pdg": pdg_val,
                "dx": dx, "dy": dy, "dz": dz,
                "E": E_gev,
            })

    npandora = len(pandora_parts)

    if nspine == 0:
        continue

    # --- Geometric matching ---
    # For each truth primary, find best reco by angular distance
    matched_pairs = []  # (truth_idx, reco_idx, cosθ)
    used_truth = set()
    used_reco = set()

    for ti, tp in enumerate(truth_parts):
        best_cos = -2.0
        best_ri = -1
        for ri, rp in enumerate(spine_parts):
            cos = _cos_angle(tp["dx"], tp["dy"], tp["dz"],
                             rp["dx"], rp["dy"], rp["dz"])
            if cos > best_cos:
                best_cos = cos
                best_ri = ri
        if best_cos > nmatch_threshold and best_ri >= 0:
            matched_pairs.append((ti, best_ri, best_cos))
            used_truth.add(ti)
            used_reco.add(best_ri)

    # Record per-particle data
    for ti, ri, cos in matched_pairs:
        tp = truth_parts[ti]
        rp = spine_parts[ri]
        ang_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        particles.append({
            "true_pdg": tp["pdg"], "reco_pdg": rp["pdg"],
            "true_p": tp["p_mag"], "reco_p": rp["p_mag"],
            "true_E": tp["E"], "reco_E": rp["E"],
            "reco_Ekin_mev": rp["E_kin_mev"],
            "cos_match": cos, "ang_deg": ang_deg,
        })

    # Event-level stats
    completeness = len(used_truth) / max(ntruth, 1)
    purity = len(used_reco) / max(nspine, 1)
    ghosts = nspine - len(used_reco)
    nsplit = sum(1 for t in range(ntruth)
                 if sum(1 for pt, _, _ in matched_pairs if pt == t) > 1)

    event_stats.append({
        "ntruth": ntruth, "nspine": nspine,
        "completeness": completeness, "purity": purity,
        "ghosts": ghosts, "split": nsplit,
        "npandora_total": -1, "npandora_primary": 0,
        "pand_completeness": _nan, "pand_purity": _nan,
        "pand_ghosts": _nan, "pand_split": _nan,
        "nu_mode": _nu_mode_lbl, "nu_E": _nu_E,
    })

    # --- Pandora geometric matching ---
    if npandora > 0:
        # Total real Pandora particles = common count - 1 (exclude fake last particle)
        npand_total_real = pand_parts.npandora - 1
        pand_matched_pairs = []
        pand_used_truth = set()
        pand_used_reco = set()
        for ti, tp in enumerate(truth_parts):
            best_cos = -2.0
            best_ri = -1
            for ri, rp in enumerate(pandora_parts):
                cos = _cos_angle(tp["dx"], tp["dy"], tp["dz"],
                                 rp["dx"], rp["dy"], rp["dz"])
                if cos > best_cos:
                    best_cos = cos
                    best_ri = ri
            if best_cos > nmatch_threshold and best_ri >= 0:
                pand_matched_pairs.append((ti, best_ri, best_cos))
                pand_used_truth.add(ti)
                pand_used_reco.add(best_ri)
        pand_completeness = len(pand_used_truth) / max(ntruth, 1)
        pand_purity = len(pand_used_reco) / max(npand_total_real, 1)
        pand_ghosts = npand_total_real - len(pand_used_reco)
        pand_nsplit = sum(1 for t in range(ntruth)
                          if sum(1 for pt, _, _ in pand_matched_pairs if pt == t) > 1)
        event_stats[-1].update({
            "npandora_total": npand_total_real, "npandora_primary": npandora,
            "pand_completeness": pand_completeness, "pand_purity": pand_purity,
            "pand_ghosts": pand_ghosts, "pand_split": pand_nsplit,
        })
        for ti, ri, cos in pand_matched_pairs:
            tp = truth_parts[ti]
            rp = pandora_parts[ri]
            ang_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            pandora_particles.append({
                "true_pdg": tp["pdg"], "reco_pdg": rp["pdg"],
                "true_p": tp["p_mag"], "reco_p": _nan,
                "true_E": tp["E"], "reco_E": rp["E"],
                "cos_match": cos, "ang_deg": ang_deg,
            })

print(f"\nProcessed {len(event_stats)} events.")
print(f"  SPINE:   {len(particles)} matched pairs")
print(f"  Pandora: {len(pandora_particles)} matched pairs")

# ============================================================
# Convert to arrays
# ============================================================
sp = {k: np.array([p[k] for p in particles]) for k in
      ["true_pdg", "reco_pdg", "true_p", "reco_p", "true_E", "reco_E", "reco_Ekin_mev", "ang_deg"]}
ev = {k: np.array([e[k] for e in event_stats]) for k in event_stats[0].keys()}

# Pandora arrays
pd = None
if len(pandora_particles) > 0:
    pd = {k: np.array([p[k] for p in pandora_particles]) for k in
          ["true_pdg", "reco_pdg", "true_p", "reco_p", "true_E", "reco_E", "ang_deg"]}

# ============================================================
# ELECTRON BIAS INVESTIGATION
# ============================================================
emask = sp["true_pdg"] == 11
print(f"\n{'='*70}")
print(f"ELECTRON p_ratio INVESTIGATION (N={emask.sum()})")
print(f"{'='*70}")
if emask.sum() > 10:
    e_tp = sp["true_p"][emask]; e_rp = sp["reco_p"][emask]
    e_rat = e_rp / e_tp; e_rat = e_rat[np.isfinite(e_rat) & (e_rat > 0)]
    print(f"  p_ratio: median={np.median(e_rat):.2f}, mean={np.mean(e_rat):.2f}, std={np.std(e_rat):.2f}")
    print(f"  true_p range: [{np.min(e_tp):.4f}, {np.max(e_tp):.4f}] GeV")
    print(f"  reco_p range: [{np.min(e_rp):.4f}, {np.max(e_rp):.4f}] GeV")

    # What is reco_p proportional to?
    # Check if reco_p scales with true_E instead of true_p
    e_tE = sp["true_E"][emask]
    e_rE = sp["reco_E"][emask]
    print(f"\n  Is reco_p actually proportional to true_E?")
    slope, inter = np.polyfit(e_tE, e_rp, 1)
    print(f"    reco_p = {slope:.3f} × true_E + {inter:.4f} (slope should be ~1 if reco_p ≈ true_E)")
    e_KE = e_tE - _PDG_MASS[11]  # kinetic energy = total - mass
    slope_k, inter_k = np.polyfit(e_KE, e_rp, 1)
    print(f"    reco_p = {slope_k:.3f} × true_KE + {inter_k:.4f} (if reco_p ≈ true_KE)")

    # Check if reco_p ≈ reco_Ekin (should be, for relativistic)
    e_rEkin = sp["reco_Ekin_mev"][emask] * 0.001  # MeV → GeV
    print(f"\n  SPINE internal consistency (electrons):")
    print(f"    reco_Ekin / reco_p: median={np.median(e_rEkin/e_rp):.3f} (should be ~1)")
    print(f"    reco_p / reco_Ekin: median={np.median(e_rp/e_rEkin):.3f}")

    # Check reco_Ekin vs true_KE
    print(f"\n  reco_Ekin vs true KE:")
    print(f"    reco_Ekin / true_KE: median={np.median(e_rEkin/e_KE):.2f}, mean={np.mean(e_rEkin/e_KE):.2f}")

    # p_ratio vs true_p
    print(f"\n  p_ratio vs true_p (binned):")
    for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 5.0)]:
        m = (e_tp >= lo) & (e_tp < hi) & np.isfinite(e_rat)
        if m.sum() < 5: continue
        print(f"    p_true [{lo:.1f}, {hi:.1f}) GeV: N={m.sum():4d}, p_ratio={np.median(e_rat[m]):.3f}")

    # Is the ratio truly constant? Check scatter around median
    resid = e_rat / np.median(e_rat)
    print(f"\n  Residual scatter (ratio / median_ratio):")
    print(f"    std={np.std(resid):.3f}, IQR={np.percentile(resid,75)-np.percentile(resid,25):.3f}")

    # What if p_reco for electrons is actually the TOTAL p sum of all SPINE primaries
    # (i.e., the event's total momentum mis-assigned to the electron)?
    # We can't test this here since we only have matched data, not event totals.

    # CRITICAL CHECK: correlation between predicted and true
    print(f"\n  Correlation: ρ(p_reco, p_true) = {np.corrcoef(e_rp, e_tp)[0,1]:.3f}")
    print(f"  Correlation: ρ(p_reco, E_true) = {np.corrcoef(e_rp, e_tE)[0,1]:.3f}")

    # What if reco_p for electrons is the kin energy in MeV wrongly stored as p in MeV/c?
    # For relativistic e: E_kin ≈ p_true. So if the network outputs E_kin in MeV but
    # stores it as p in MeV/c, the ratio would be p_reco/p_true = E_kin/p_true ≈ 1.
    # That gives ~1, not ~4.76. So that's not it.

    # What if the reco_p is the TOTAL energy deposit in MeV?
    # For electron showers, total deposited energy ≈ true E (in MeV) due to EM
    # calorimetry. Then p_reco/p_true ≈ E_true/p_true ≈ E_kin+p_true*(m²/2p²)/p_true
    # For relativistic: E_true/p_true = sqrt(p²+m²)/p ≈ 1 + m²/(2p²) ≈ 1. So not 4.76.

    print(f"\n  KEY QUESTION: reco_p ~ {np.median(e_rat):.2f} × true_p")
    print(f"  This is a multiplicative factor applied to the momentum.")
    print(f"  Possible cause: SPINE uses track-range to infer momentum;")
    print(f"  for EM showers the range is ~{np.median(e_rat):.0f}× longer than MIP range")
    print(f"  at the same true momentum (wrong dE/dx assumption).")

# Also check gamma
gmask = sp["true_pdg"] == 22
if gmask.sum() > 5:
    g_rat = sp["reco_p"][gmask] / sp["true_p"][gmask]
    print(f"\n  GAMMA p_ratio: median={np.median(g_rat):.2f}, N={gmask.sum()}")
print()

# ============================================================
# FIGURE 10: PID confusion matrix
# ============================================================
print("--- Figure 10: PID confusion matrix ---")
ncat = len(cat_order)
cm = np.zeros((ncat, ncat), dtype=int)
for t, r in zip(sp["true_pdg"], sp["reco_pdg"]):
    cm[cat_order.index(pdg_category(t)), cat_order.index(pdg_category(r))] += 1

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig10, ax10 = plt.subplots(figsize=(10, 8.5))
im = ax10.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax10.set_xticks(range(ncat))
ax10.set_xticklabels(cat_order, rotation=45, ha="right")
ax10.set_yticks(range(ncat)); ax10.set_yticklabels(cat_order)
ax10.set_xlabel("Reco PDG"); ax10.set_ylabel("Truth PDG")
ax10.set_title(f"SPINE PID confusion (N={len(sp['true_pdg']):,}, cosθ>{nmatch_threshold})")
fig10.text(0.5, 0.004,
           "★  5k CC Subsample Baseline  |  Full production (46.4k CC) in proton PID efficiency study",
           ha="center", va="bottom", fontsize=9, fontstyle="italic", color="#8b0000",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeeba", edgecolor="#e0a800", alpha=0.95))
# Color code truth PDG labels to match cross-figure particle color scheme
for i, lbl in enumerate(cat_order):
    clr = PARTICLE_COLORS.get(lbl, "black")
    ax10.get_yticklabels()[i].set_color(clr)
    ax10.get_yticklabels()[i].set_fontweight("bold")
for i in range(ncat):
    for j in range(ncat):
        if cm[i, j] > 0:
            c = "white" if cm_norm[i, j] > 0.35 else "black"
            ax10.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=9,
                     fontweight="bold", color=c)
fig10.colorbar(im, ax=ax10, fraction=0.046, label="Row-normalized fraction")
fig10.tight_layout()
fig10.savefig(os.path.join(outdir, "10_pid_confusion_spine.png"), dpi=150)
plt.close(fig10)

# ============================================================
# FIGURE 11a-b: Angular resolution per particle type (split)
# ============================================================
print("--- Figure 11a-b: Angular resolution per particle type (split) ---")
lep_cats = ["e", "mu", "gamma", "pi0"]
had_cats = ["pi", "K", "p", "n"]

def _plot_ang_res(cats, title, fname, figsize=(10, 8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_f = axes.flatten()
    for idx, ptype in enumerate(cats):
        ax = axes_f[idx]; cat_idx = cat_order.index(ptype)
        mask = np.array([pdg_category(t) == ptype for t in sp["true_pdg"]])
        if mask.sum() < 5: ax.text(0.5, 0.5, "N<5", ha="center", va="center", transform=ax.transAxes); continue
        vals = sp["ang_deg"][mask]; vals = vals[np.isfinite(vals)]
        if len(vals) < 5: ax.text(0.5, 0.5, "N<5", ha="center", va="center", transform=ax.transAxes); continue
        pcol = PARTICLE_COLORS.get(ptype, "C0")
        ax.hist(vals, bins=np.linspace(0, 90, 36), histtype="step", lw=1.5, color=pcol)
        ax.axvline(np.median(vals), color=pcol, ls="--", lw=1, label=f"med={np.median(vals):.1f}°")
        ax.set_xlabel(r"$\Delta\theta$ [deg]"); ax.set_ylabel("Counts")
        ax.set_title(f"{ptype} (N={len(vals)})"); ax.legend(fontsize=8)
    for j in range(len(cats), 4): axes_f[j].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)

_plot_ang_res(lep_cats, "SPINE angular resolution: Leptons & Photons", "11a_angular_leptons_spine.png")
_plot_ang_res(had_cats, "SPINE angular resolution: Hadrons & Mesons", "11b_angular_hadrons_spine.png")

# ============================================================
# FIGURE 12a-b: Momentum resolution per particle type (split)
# ============================================================
print("--- Figure 12a-b: Momentum resolution per particle type (split) ---")
def _plot_mom_res(cats, title, fname, figsize=(10, 8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_f = axes.flatten()
    for idx, ptype in enumerate(cats):
        ax = axes_f[idx]
        mask = np.array([pdg_category(t) == ptype for t in sp["true_pdg"]])
        if mask.sum() < 5: ax.text(0.5, 0.5, "N<5", ha="center", va="center", transform=ax.transAxes); continue
        ratio = sp["reco_p"][mask] / sp["true_p"][mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0) & (ratio < 10)]
        if len(ratio) < 5: ax.text(0.5, 0.5, "N<5", ha="center", va="center", transform=ax.transAxes); continue
        pcol = PARTICLE_COLORS.get(ptype, "C0")
        ax.hist(ratio, bins=np.linspace(0, 5, 50), histtype="step", lw=1.5, color=pcol)
        ax.axvline(np.median(ratio), color=pcol, ls="--", lw=1, label=f"med={np.median(ratio):.2f}")
        ax.axvline(1, color="k", lw=0.8, ls=":")
        ax.set_xlabel(r"$p_\mathrm{reco} / p_\mathrm{true}$"); ax.set_ylabel("Counts")
        ax.set_title(f"{ptype} (N={len(ratio)})"); ax.legend(fontsize=8)
    for j in range(len(cats), 4): axes_f[j].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)

_plot_mom_res(lep_cats, "SPINE momentum resolution: Leptons & Photons", "12a_momentum_leptons_spine.png")
_plot_mom_res(had_cats, "SPINE momentum resolution: Hadrons & Mesons", "12b_momentum_hadrons_spine.png")

# ============================================================
# FIGURE 13a-b: Energy resolution per particle type (split)
# ============================================================
print("--- Figure 13a-b: Energy resolution per particle type (split) ---")
def _plot_E_res(cats, title, fname, figsize=(10, 8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_f = axes.flatten()
    for idx, ptype in enumerate(cats):
        ax = axes_f[idx]
        mask = np.array([pdg_category(t) == ptype for t in sp["true_pdg"]])
        if mask.sum() < 5: ax.text(0.5, 0.5, "N<5", ha="center", va="center", transform=ax.transAxes); continue
        ratio = sp["reco_E"][mask] / sp["true_E"][mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0) & (ratio < 10)]
        if len(ratio) < 5: ax.text(0.5, 0.5, "N<5", ha="center", va="center", transform=ax.transAxes); continue
        pcol = PARTICLE_COLORS.get(ptype, "C0")
        ax.hist(ratio, bins=np.linspace(0, 5, 50), histtype="step", lw=1.5, color=pcol)
        ax.axvline(np.median(ratio), color=pcol, ls="--", lw=1, label=f"med={np.median(ratio):.2f}")
        ax.axvline(1, color="k", lw=0.8, ls=":")
        ax.set_xlabel(r"$E_\mathrm{reco} / E_\mathrm{true}$"); ax.set_ylabel("Counts")
        ax.set_title(f"{ptype} (N={len(ratio)})"); ax.legend(fontsize=8)
    for j in range(len(cats), 4): axes_f[j].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)

_plot_E_res(lep_cats, "SPINE energy resolution: Leptons & Photons", "13a_energy_leptons_spine.png")
_plot_E_res(had_cats, "SPINE energy resolution: Hadrons & Mesons", "13b_energy_hadrons_spine.png")

# ============================================================
# FIGURE 12c: Electron calibration before/after comparison
# ============================================================
print("--- Figure 12c: Electron calibration before/after ---")
fig12c, ax12c = plt.subplots(figsize=(10, 7))
emask = sp["true_pdg"] == 11
if emask.sum() > 10:
    e_tp = sp["true_p"][emask]
    e_rp = sp["reco_p"][emask]
    # "Before" ratio: undo the EM calibration factor
    e_rp_before = e_rp / EM_CALIB_FACTOR
    e_rat_after = e_rp / e_tp
    e_rat_before = e_rp_before / e_tp
    valid_a = np.isfinite(e_rat_after) & (e_rat_after > 0) & (e_rat_after < 50)
    valid_b = np.isfinite(e_rat_before) & (e_rat_before > 0) & (e_rat_before < 50)
    bins = np.logspace(np.log10(0.1), np.log10(20), 60)
    ax12c.hist(e_rat_before[valid_b], bins=bins, histtype="step", lw=2.0,
               color="#d62728", label=f"Before calibration (med={np.median(e_rat_before[valid_b]):.2f})")
    ax12c.hist(e_rat_after[valid_a], bins=bins, histtype="step", lw=2.0,
               color="#2ca02c", label=f"After ×{EM_CALIB_FACTOR} (med={np.median(e_rat_after[valid_a]):.2f})")
    ax12c.axvline(1, color="k", ls=":", lw=1.0)
    ax12c.set_xscale("log")
    ax12c.set_xlabel(r"$p_\mathrm{reco} / p_\mathrm{true}$")
    ax12c.set_ylabel("Counts")
    ax12c.set_title(f"Electron calibration: before vs after (N={emask.sum()})")
    ax12c.legend(fontsize=11)
    ax12c.set_xlim(0.1, 20)
fig12c.tight_layout()
fig12c.savefig(os.path.join(outdir, "12c_electron_calibration_before_after.png"), dpi=150)
plt.close(fig12c)
print("Saved 12c_electron_calibration_before_after.png")

# ============================================================
# FIGURE 14: Purity vs Completeness
# ============================================================
print("--- Figure 14: Purity vs Completeness ---")
fig14, ax14 = plt.subplots(figsize=(7, 6))
ax14.scatter(ev["completeness"], ev["purity"], s=5, alpha=0.4, rasterized=True)
ax14.set_xlabel("Completeness (truth-matched / total truth)")
ax14.set_ylabel("Purity (reco-matched / total reco)")
ax14.set_title("SPINE per-event purity vs completeness")
ax14.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax14.set_xlim(0, 1.05); ax14.set_ylim(0, 1.05)
fig14.tight_layout()
fig14.savefig(os.path.join(outdir, "14_purity_completeness_spine.png"), dpi=150)
plt.close(fig14)

# ============================================================
# FIGURE 15: Ghost & split fractions
# ============================================================
print("--- Figure 15: Ghosts & splits ---")
fig15, axes15 = plt.subplots(1, 2, figsize=(14, 5))
axes15[0].hist(ev["ghosts"] / ev["nspine"].clip(1), bins=np.linspace(0, 1, 25),
               histtype="step", lw=1.5, color="C0")
axes15[0].set_xlabel("Ghost fraction (unmatched / total reco)")
axes15[0].set_ylabel("Events"); axes15[0].set_title("Ghost particles")
axes15[1].hist(ev["split"], bins=np.arange(-0.5, 10, 1),
               histtype="step", lw=1.5, color="C0")
axes15[1].set_xlabel("N split truth primaries (matched to >1 reco)")
axes15[1].set_ylabel("Events"); axes15[1].set_title("Split reconstructions")
fig15.tight_layout()
fig15.savefig(os.path.join(outdir, "15_ghosts_splits_spine.png"), dpi=150)
plt.close(fig15)

# ============================================================
# FIGURE 16a-b: Resolution vs true momentum (split)
# ============================================================
print("--- Figure 16a-b: Resolution vs true momentum (split) ---")
def _plot_res_vs_p(cats, title, fname, figsize=(10, 8), density_cats=None):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_f = axes.flatten()
    if density_cats is None:
        density_cats = set()
    for idx, ptype in enumerate(cats):
        ax = axes_f[idx]
        mask = np.array([pdg_category(t) == ptype for t in sp["true_pdg"]])
        if mask.sum() < 10:
            ax.text(0.5, 0.5, "N<10", ha="center", va="center", transform=ax.transAxes)
            continue
        true_p = sp["true_p"][mask]
        ang = sp["ang_deg"][mask]
        valid = np.isfinite(true_p) & np.isfinite(ang) & (true_p > 0.05)
        if valid.sum() < 10:
            ax.text(0.5, 0.5, "N<10", ha="center", va="center", transform=ax.transAxes)
            continue
        if ptype in density_cats:
            # 2D log-density scatter
            pbins = np.logspace(np.log10(0.08), np.log10(30), 40)
            abins = np.linspace(0, 90, 40)
            h = ax.hist2d(true_p[valid], ang[valid], bins=[pbins, abins],
                          norm="log", cmap="viridis", cmin=1)
            fig.colorbar(h[3], ax=ax, label="Counts (log)")
            # Overlay median profile
            pmin = max(true_p[valid].min(), 0.08)
            pmax = min(true_p[valid].max(), 30)
            nbin = max(min(10, int(pmax / pmin * 3)), 4)
            pbins_m = np.logspace(np.log10(pmin), np.log10(pmax), nbin + 1)
            med, _, _ = binned_statistic(true_p[valid], ang[valid], "median", bins=pbins_m)
            iqr, _, _ = binned_statistic(true_p[valid], ang[valid], iqr_half, bins=pbins_m)
            centers = 0.5 * (pbins_m[:-1] + pbins_m[1:])
            ax.plot(centers, med, "r-", lw=2.0, label="Median")
            ax.fill_between(centers, med - iqr, med + iqr, alpha=0.2, color="red", label="IQR/2")
            ax.set_ylim(bottom=0)
        else:
            pmin = max(true_p[valid].min(), 0.08)
            pmax = min(true_p[valid].max(), 30)
            nbin = max(min(10, int(pmax / pmin * 3)), 4)
            pbins = np.logspace(np.log10(pmin), np.log10(pmax), nbin + 1)
            med, _, _ = binned_statistic(true_p[valid], ang[valid], "median", bins=pbins)
            iqr, _, _ = binned_statistic(true_p[valid], ang[valid], iqr_half, bins=pbins)
            cnt, _, _ = binned_statistic(true_p[valid], ang[valid], "count", bins=pbins)
            centers = 0.5 * (pbins[:-1] + pbins[1:])
            mk = cnt > 3
            pcol = PARTICLE_COLORS.get(ptype, SPINE_COLOR)
            ax.errorbar(centers[mk], med[mk], yerr=iqr[mk], marker="o", ms=5, color=pcol)
            ax.set_ylim(bottom=0)
        ax.set_xscale("log"); _fmt_log_ticks(ax)
        ax.set_xlabel(r"True $p$ [GeV/$c$]")
        ax.set_ylabel(r"$\Delta\theta$ [deg]")
        ax.set_title(f"{ptype} ang. res. (N={valid.sum()})")
        if ptype in density_cats:
            ax.legend(fontsize=9)
    for j in range(len(cats), 4): axes_f[j].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)

_plot_res_vs_p(lep_cats, "Angular resolution vs momentum: Leptons & Photons", "16a_res_vs_p_leptons_spine.png")
_plot_res_vs_p(had_cats, "Angular resolution vs momentum: Hadrons & Mesons", "16b_res_vs_p_hadrons_spine.png",
               density_cats=set(had_cats))

# ============================================================
# FIGURE 22: Angular resolution vs momentum — SPINE vs Pandora overlay
# ============================================================
print("--- Figure 22: Angular resolution vs momentum — SPINE vs Pandora ---")

if pd is not None:
    compare_cats = ["e", "mu", "gamma", "pi", "p"]
    fig22, axes22 = plt.subplots(2, 3, figsize=(18, 11))
    fig22.suptitle(
        r"Angular Resolution vs Momentum — SPINE vs Pandora (5k CC subsample, cos$\theta$ > 0.5)",
        fontsize=13, fontweight="bold")

    axes22_flat = axes22.flatten()

    for ax, ptype in zip(axes22_flat, compare_cats):
        pcol_s = "#1f77b4"   # SPINE blue
        pcol_p = "#ff7f0e"   # Pandora orange

        for arr, col, lbl, ls, ms in [
            (sp, pcol_s, "SPINE",   "-",  "o"),
            (pd, pcol_p, "Pandora", "--", "s"),
        ]:
            mask = np.array([pdg_category(t) == ptype for t in arr["true_pdg"]])
            true_p = arr["true_p"][mask]
            ang    = arr["ang_deg"][mask]
            valid  = np.isfinite(true_p) & np.isfinite(ang) & (true_p > 0.05)
            n_pairs = int(valid.sum())
            if n_pairs < 10:
                continue
            pmin = max(true_p[valid].min(), 0.05)
            pmax = min(true_p[valid].max(), 10.0)
            nbin = max(min(10, int(pmax / pmin * 3)), 4)
            pbins   = np.logspace(np.log10(pmin), np.log10(pmax), nbin + 1)
            med, _, _  = binned_statistic(true_p[valid], ang[valid], "median", bins=pbins)
            iqr_v, _, _ = binned_statistic(true_p[valid], ang[valid], iqr_half, bins=pbins)
            cnt, _, _  = binned_statistic(true_p[valid], ang[valid], "count",  bins=pbins)
            centers = 0.5 * (pbins[:-1] + pbins[1:])
            mk = cnt > 3
            ax.errorbar(centers[mk], med[mk], yerr=iqr_v[mk],
                        marker=ms, ms=5, color=col, lw=1.5, ls=ls,
                        label=f"{lbl} (N={n_pairs:,})", capsize=3)

        pcol_lbl = PARTICLE_COLORS.get(ptype, "black")
        ax.set_xscale("log")
        _fmt_log_ticks(ax)
        ax.set_xlabel(r"True $p$ [GeV/$c$]")
        ax.set_ylabel(r"$\Delta\theta$ [deg]")
        ax.set_title(f"{ptype}", color=pcol_lbl, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)

    axes22_flat[-1].set_visible(False)  # hide unused 6th panel

    fig22.tight_layout()
    fig22.savefig(os.path.join(outdir, "22_angular_res_spine_vs_pandora.png"), dpi=150)
    plt.close(fig22)
    print("  Saved 22_angular_res_spine_vs_pandora.png")

    # Print median summary
    print(f"\n  {'Type':>6}  {'SPINE N':>8}  {'SPINE med':>10}  {'Pandora N':>10}  {'Pandora med':>12}")
    for ptype in compare_cats:
        for arr, lbl in [(sp, "SPINE"), (pd, "Pandora")]:
            mask  = np.array([pdg_category(t) == ptype for t in arr["true_pdg"]])
            ang_v = arr["ang_deg"][mask]
            ang_v = ang_v[np.isfinite(ang_v)]
            n_v   = int(mask.sum())
            med_v = float(np.median(ang_v)) if len(ang_v) > 0 else float("nan")
            if lbl == "SPINE":
                _row = f"  {ptype:>6}  {n_v:>8,}  {med_v:>10.1f}°"
            else:
                print(_row + f"  {n_v:>10,}  {med_v:>12.1f}°")
else:
    print("  Pandora data not available — skipping Figure 22.")

# ============================================================
# FIGURE 17: Track vs Shower classification
# ============================================================
print("--- Figure 17: Track vs Shower ---")
fig17, ax17 = plt.subplots(figsize=(7, 6))
true_track = np.array([abs(t) in TRACK_PDGS for t in sp["true_pdg"]])
true_shower = np.array([abs(t) in SHOWER_PDGS for t in sp["true_pdg"]])
reco_track = np.array([abs(t) in TRACK_PDGS for t in sp["reco_pdg"]])
reco_shower = np.array([abs(t) in SHOWER_PDGS for t in sp["reco_pdg"]])

track_eff = (true_track & reco_track).sum() / max(true_track.sum(), 1)
track_pur = (true_track & reco_track).sum() / max(reco_track.sum(), 1)
shower_eff = (true_shower & reco_shower).sum() / max(true_shower.sum(), 1)
shower_pur = (true_shower & reco_shower).sum() / max(reco_shower.sum(), 1)

cats = ["Track eff.", "Track pur.", "Shower eff.", "Shower pur."]
vals = [track_eff * 100, track_pur * 100, shower_eff * 100, shower_pur * 100]
colors = ["#2196F3", "#1976D2", "#FF9800", "#F57C00"]
bars = ax17.bar(cats, vals, color=colors)
for bar, v in zip(bars, vals):
    ax17.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
              f"{v:.1f}%", ha="center", fontsize=9)
ax17.set_ylabel("Percentage [%]"); ax17.set_ylim(0, 110)
ax17.set_title("SPINE track/shower classification")
fig17.tight_layout()
fig17.savefig(os.path.join(outdir, "17_track_shower_spine.png"), dpi=150)
plt.close(fig17)

# ============================================================
# FIGURE 17b: Shower purity breakdown
# ============================================================
print("--- Figure 17b: Shower purity breakdown ---")
reco_shower_pdg = sp["reco_pdg"]
true_pdg = sp["true_pdg"]
shower_mask_r = np.array([abs(r) in SHOWER_PDGS for r in reco_shower_pdg])
shower_mask_t = np.array([abs(t) in SHOWER_PDGS for t in true_pdg])

# Which truth particles are reconstructed as showers?
reco_as_shower = true_pdg[shower_mask_r]
reco_as_electron = true_pdg[reco_shower_pdg == 11]  # specifically identified as e
reco_as_gamma = true_pdg[reco_shower_pdg == 22]

# Also: which reco particles classified as showers are truth tracks?
track_misid_as_shower = (shower_mask_r & ~shower_mask_t)

# Categories for the bar chart
shower_pdg_cats = ["e", "mu", "gamma", "pi0", "pi", "K", "p", "n", "other"]
fig17b, axes17b = plt.subplots(1, 3, figsize=(18, 5.5))

for ax_i, label, arr in [
    (0, "Reco as EM shower (e/γ/π⁰)", reco_as_shower),
    (1, "Reco as e⁻", reco_as_electron),
    (2, "Reco as γ", reco_as_gamma),
]:
    counts = []
    for ptype in shower_pdg_cats:
        cat_pdgs = {"e": {11}, "mu": {13}, "gamma": {22}, "pi0": {111},
                    "pi": {211}, "K": {321, 311, 310}, "p": {2212},
                    "n": {2112}, "other": set()}
        if ptype == "other":
            c = sum(1 for a in arr if abs(a) not in {11, 13, 22, 111, 211, 321, 311, 310, 2212, 2112})
        else:
            c = sum(1 for a in arr if abs(a) in cat_pdgs[ptype])
        counts.append(c)
    total = sum(counts)
    fracs = [c / max(total, 1) * 100 for c in counts]
    colors_sh = [PARTICLE_COLORS.get(pt, "#7f7f7f") for pt in shower_pdg_cats]
    bars = axes17b[ax_i].bar(shower_pdg_cats, fracs, color=colors_sh, alpha=0.8)
    axes17b[ax_i].set_xlabel("True particle type")
    axes17b[ax_i].set_ylabel("Fraction [%]")
    axes17b[ax_i].set_title(f"{label}\n(N={total})")
    for bar, f in zip(bars, fracs):
        if f > 3:
            axes17b[ax_i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f"{f:.1f}%", ha="center", fontsize=7)

fig17b.suptitle("Shower Purity: what truth particles are reconstructed as EM showers?", fontsize=12)
fig17b.tight_layout()
fig17b.savefig(os.path.join(outdir, "17b_shower_purity_breakdown.png"), dpi=150)
plt.close(fig17b)

# Also print a table
print("\n--- Shower purity breakdown ---")
print(f"{'':>20} {'reco as shower':>16} {'reco as e⁻':>12} {'reco as γ':>12}")
total_rs = len(reco_as_shower)
total_re = len(reco_as_electron)
total_rg = len(reco_as_gamma)
print(f"{'':>20} {total_rs:>16} {total_re:>12} {total_rg:>12}")
for ptype in shower_pdg_cats:
    cat_pdgs = {"e": {11}, "mu": {13}, "gamma": {22}, "pi0": {111},
                "pi": {211}, "K": {321, 311, 310}, "p": {2212},
                "n": {2112}, "other": set()}
    def cnt(arr, ptype):
        if ptype == "other":
            return sum(1 for a in arr if abs(a) not in {11, 13, 22, 111, 211, 321, 311, 310, 2212, 2112})
        return sum(1 for a in arr if abs(a) in cat_pdgs[ptype])
    c1, c2, c3 = cnt(reco_as_shower, ptype), cnt(reco_as_electron, ptype), cnt(reco_as_gamma, ptype)
    print(f"  {ptype:>17}: {c1:>8} ({c1/max(total_rs,1)*100:4.1f}%)  {c2:>6} ({c2/max(total_re,1)*100:4.1f}%)  {c3:>6} ({c3/max(total_rg,1)*100:4.1f}%)")

# Summary text
track_misid = track_misid_as_shower.sum()
print(f"\n  Track → Shower mis-ID: {track_misid} / {shower_mask_r.sum()} shower reconstructions "
      f"({track_misid/max(shower_mask_r.sum(),1)*100:.1f}%)")
# Which tracks are most commonly mis-ID'd?
track_misid_pdgs = true_pdg[track_misid_as_shower]
from collections import Counter
top_misid = Counter(abs(p) for p in track_misid_pdgs).most_common(5)
print(f"  Top mis-ID'd as shower: {top_misid}")

# ============================================================
# FIGURE 17c: Sankey diagram — Shower purity & neutron flow
# ============================================================
print("--- Figure 17c: Sankey diagram (shower purity flow) ---")
from matplotlib.patches import FancyBboxPatch, Polygon

fig17c, ax17c = plt.subplots(figsize=(14, 9))
ax17c.set_xlim(0, 1)
ax17c.set_ylim(0, 1)
ax17c.axis("off")
fig17c.suptitle("The Neutron Invisibility Challenge: True → Reco Particle Flow",
                fontsize=16, fontweight="bold")

# Simplified categories (5 groups)
sankey_cats = [
    ("e⁻", {11}, PARTICLE_COLORS["e"], "EM lepton"),
    ("γ/π⁰", {22, 111}, PARTICLE_COLORS["gamma"], "EM shower"),
    ("n", {2112}, PARTICLE_COLORS["n"], "neutron"),
    ("p", {2212}, PARTICLE_COLORS["p"], "proton"),
    ("π/K/μ", {211, 321, 13}, PARTICLE_COLORS["pi"], "hadron"),
]

def _match_sankey(pdg, cat_set):
    return abs(pdg) in cat_set

# Compute flows
s_flows = []
for tlabel, tset, tcol, tdesc in sankey_cats:
    tmask = np.array([_match_sankey(p, tset) for p in sp["true_pdg"]])
    is_track = np.array([abs(r) in TRACK_PDGS for r in sp["reco_pdg"][tmask]])
    is_shower = np.array([abs(r) in SHOWER_PDGS for r in sp["reco_pdg"][tmask]])
    nt = int(is_track.sum())
    ns = int(is_shower.sum())
    if nt + ns > 0:
        s_flows.append((tlabel, tcol, nt, ns, nt + ns))

# Layout
lx = 0.15
rtx = 0.68
rsx = 0.88
rw = 0.045
total_s = sum(f[4] for f in s_flows)
sc = 0.78 / total_s
bott = 0.10

# Right nodes
rt_h = sum(f[2] for f in s_flows) * sc
rs_h = sum(f[3] for f in s_flows) * sc
rt_b = bott + (total_s * sc - rt_h) / 2
rs_b = bott + (total_s * sc - rs_h) / 2

rect_t = FancyBboxPatch((rtx - rw/2, rt_b), rw, rt_h,
                        boxstyle="round,pad=0.02", facecolor=SPINE_COLOR, alpha=0.85,
                        edgecolor="k", lw=2.0)
ax17c.add_patch(rect_t)
ax17c.text(rtx + rw/2 + 0.01, rt_b + rt_h/2,
          f"Reco Track\n({int(rt_h/sc):,})", ha="left", va="center", fontsize=11, fontweight="bold")

rect_s = FancyBboxPatch((rsx - rw/2, rs_b), rw, rs_h,
                        boxstyle="round,pad=0.02", facecolor="#d62728", alpha=0.85,
                        edgecolor="k", lw=2.0)
ax17c.add_patch(rect_s)
ax17c.text(rsx + rw/2 + 0.01, rs_b + rs_h/2,
          f"Reco Shower\n({int(rs_h/sc):,})", ha="left", va="center", fontsize=11, fontweight="bold")

# Draw left nodes and flows
rt_cum = rt_b
rs_cum = rs_b
cb = bott
for label, tcol, nt, ns, total_n in s_flows:
    lh = total_n * sc
    if lh < 0.005:
        cb += lh
        continue
    lc = cb + lh / 2

    # Left node
    rect = FancyBboxPatch((lx - rw/2, cb), rw, lh,
                          boxstyle="round,pad=0.02", facecolor=tcol, alpha=0.85,
                          edgecolor="k", lw=2.0)
    ax17c.add_patch(rect)
    pct = total_n / total_s * 100
    neigh = "→ " + ("T" if nt > 0 else "") + ("S" if ns > 0 else "")
    ax17c.text(lx - 0.02, lc, f"{label}\n{total_n:,} ({pct:.1f}%)",
              ha="right", va="center", fontsize=10, fontweight="bold", color=tcol)

    # Flow to track (upper portion of left node)
    if nt > 0:
        fh = nt * sc
        sb = lc + lh/2 - fh
        st = lc + lh/2
        tb = rt_cum
        tt = rt_cum + fh
        alpha = max(0.15, min(0.7, nt / max(f[2] for f in s_flows)))
        verts = [(lx + rw/2, sb), (lx + rw/2, st),
                 (rtx - rw/2, tt), (rtx - rw/2, tb)]
        poly = Polygon(verts, closed=True, facecolor=tcol, alpha=alpha,
                       edgecolor="none")
        ax17c.add_patch(poly)
        rt_cum += fh

    # Flow to shower (lower portion of left node)
    if ns > 0:
        fh = ns * sc
        sb = lc - lh/2
        st = lc - lh/2 + fh
        tb = rs_cum
        tt = rs_cum + fh
        alpha = max(0.15, min(0.7, ns / max(f[3] for f in s_flows)))
        verts = [(lx + rw/2, sb), (lx + rw/2, st),
                 (rsx - rw/2, tt), (rsx - rw/2, tb)]
        poly = Polygon(verts, closed=True, facecolor=tcol, alpha=alpha,
                       edgecolor="none")
        ax17c.add_patch(poly)
        rs_cum += fh

    cb += lh

# Add annotation
ax17c.text(0.5, 0.02,
          "Flow width proportional to particle count. "
          "Neutrons (red) → EM Showers = 41% of neutron reconstructions",
          transform=ax17c.transAxes, fontsize=11, ha="center", va="bottom",
          style="italic", color="gray")

fig17c.tight_layout()
fig17c.savefig(os.path.join(outdir, "17c_sankey_shower_purity.png"), dpi=150)
plt.close(fig17c)
print("Saved 17c_sankey_shower_purity.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"\nEvents: {len(event_stats)}, SPINE matches: {len(particles)}, Pandora matches: {len(pandora_particles)}")

print(f"\n=== SPINE event-level averages ===")
print(f"{'':>20} {'mean':>10} {'median':>10}")
for lbl, key in [("Completeness", "completeness"), ("Purity", "purity"),
                  ("N ghosts", "ghosts"), ("N split", "split")]:
    v = ev[key].astype(float)
    print(f"  {lbl:>17} {np.mean(v):>10.3f} {np.median(v):>10.3f}")

if "pand_completeness" in ev:
    pand_mask = np.isfinite(ev["pand_completeness"])
    print(f"\n=== Pandora event-level averages (N={pand_mask.sum()} events) ===")
    print(f"{'':>20} {'mean':>10} {'median':>10}")
    for lbl, key in [("Completeness", "pand_completeness"), ("Purity", "pand_purity"),
                      ("N ghosts", "pand_ghosts"), ("N split", "pand_split")]:
        v = ev[key][pand_mask].astype(float)
        print(f"  {lbl:>17} {np.mean(v):>10.3f} {np.median(v):>10.3f}")

print(f"\n=== Median angular resolution per type ===")
print(f"{'':>10} {'med [deg]':>10} {'IQR/2 [deg]':>13} {'N':>8} {'p_ratio':>8}")
for ptype in cat_order:
    mask = np.array([pdg_category(t) == ptype for t in sp["true_pdg"]])
    if mask.sum() < 5: continue
    a = sp["ang_deg"][mask]; a = a[np.isfinite(a)]
    r = sp["reco_p"][mask] / sp["true_p"][mask]
    r = r[np.isfinite(r) & (r > 0) & (r < 10)]
    print(f"  {ptype:>8} {np.median(a):>10.1f} {iqr_half(a):>13.2f} {mask.sum():>8} {np.median(r) if len(r)>0 else 0:>8.2f}")

if pd is not None:
    print(f"\n=== Pandora median angular resolution per type ===")
    print(f"{'':>10} {'med [deg]':>10} {'IQR/2 [deg]':>13} {'N':>8}")
    for ptype in cat_order:
        mask = np.array([pdg_category(t) == ptype for t in pd["true_pdg"]])
        if mask.sum() < 5: continue
        a = pd["ang_deg"][mask]; a = a[np.isfinite(a)]
        print(f"  {ptype:>8} {np.median(a):>10.1f} {iqr_half(a):>13.2f} {mask.sum():>8}")

# ---------------------------------------------------------------
# FIGURE 20: Ghost particle origin diagnostic (Study C)
# ---------------------------------------------------------------
print("\n--- Figure 20: Ghost origin diagnostic ---")
_ghost_frac = ev["ghosts"].astype(float) / np.maximum(ev["nspine"].astype(float), 1)
_nu_modes_arr = np.array([e["nu_mode"] for e in event_stats])
_nu_E_arr     = np.array([e["nu_E"]    for e in event_stats])
_ntruth_arr   = ev["ntruth"].astype(float)
_split_arr    = ev["split"].astype(float)

fig20, axes20 = plt.subplots(2, 2, figsize=(14, 10))
fig20.suptitle("Ghost Particle Origin Diagnostic (5k CC Subsample)",
               fontsize=13, fontweight="bold")

# Panel a: ghost fraction vs true multiplicity (binned median)
ax = axes20[0, 0]
_ntruth_int = _ntruth_arr.astype(int)
_bins_n = np.arange(0, min(int(_ntruth_int.max()) + 2, 10))
_med_ghost, _n_ghost = [], []
for nb in _bins_n:
    m = _ntruth_int == nb
    if m.sum() > 5:
        _med_ghost.append(float(np.median(_ghost_frac[m])))
        _n_ghost.append(m.sum())
    else:
        _med_ghost.append(float("nan"))
        _n_ghost.append(0)
_valid_n = [i for i, v in enumerate(_med_ghost) if not math.isnan(v)]
ax.bar([_bins_n[i] for i in _valid_n], [_med_ghost[i] for i in _valid_n],
       color="#1f77b4", alpha=0.7, edgecolor="navy", linewidth=0.8)
for i in _valid_n:
    ax.text(_bins_n[i], _med_ghost[i] + 0.01, f"N={_n_ghost[i]}", ha="center",
            va="bottom", fontsize=7)
ax.set_xlabel("True primary multiplicity")
ax.set_ylabel("Median ghost fraction")
ax.set_title("Ghost fraction vs true multiplicity")
ax.set_ylim(0, max([v for v in _med_ghost if not math.isnan(v)] + [0.5]) * 1.3)

# Panel b: ghost fraction by interaction mode (bar chart)
ax = axes20[0, 1]
_mode_order20 = ["QE", "RES", "MEC", "DIS", "COH", "other"]
_mode_colors20 = {"QE": "#2ca02c", "RES": "#1f77b4", "MEC": "#ff7f0e",
                  "DIS": "#d62728", "COH": "#9467bd", "other": "#8c564b"}
_mode_vals, _mode_lbls, _mode_cols = [], [], []
for m in _mode_order20:
    mk = _nu_modes_arr == m
    if mk.sum() > 5:
        _mode_vals.append(float(np.median(_ghost_frac[mk])))
        _mode_lbls.append(f"{m}\n(N={mk.sum():,})")
        _mode_cols.append(_mode_colors20[m])
xpos = np.arange(len(_mode_vals))
ax.bar(xpos, _mode_vals, color=_mode_cols, alpha=0.8, edgecolor="k", linewidth=0.6)
ax.set_xticks(xpos); ax.set_xticklabels(_mode_lbls, fontsize=9)
ax.set_ylabel("Median ghost fraction")
ax.set_title("Ghost fraction by interaction mode")
ax.set_ylim(0, max(_mode_vals + [0.5]) * 1.25)
for xi, v in zip(xpos, _mode_vals):
    ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Panel c: ghost fraction vs neutrino energy (binned median)
ax = axes20[1, 0]
_E_bins = np.linspace(0, 4, 17)
_E_centers = 0.5 * (_E_bins[:-1] + _E_bins[1:])
from scipy.stats import binned_statistic as _bstat
_med_E, _, _ = _bstat(_nu_E_arr, _ghost_frac, statistic="median", bins=_E_bins)
_cnt_E, _, _ = _bstat(_nu_E_arr, _ghost_frac, statistic="count",  bins=_E_bins)
_mk_E = _cnt_E >= 5
ax.plot(_E_centers[_mk_E], _med_E[_mk_E], "o-", color="#1f77b4", lw=1.5, ms=5)
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel("Median ghost fraction")
ax.set_title(r"Ghost fraction vs $E_\nu$")
ax.set_xlim(0, 4)

# Panel d: ghost count vs split count (2D histogram)
ax = axes20[1, 1]
_ghost_int = ev["ghosts"].astype(int)
_max_g = min(int(_ghost_int.max()), 7)
_max_s = min(int(_split_arr.max()), 7)
_h2d, _xedge, _yedge = np.histogram2d(
    _ghost_int.clip(0, _max_g), _split_arr.clip(0, _max_s),
    bins=[np.arange(-0.5, _max_g + 1.5), np.arange(-0.5, _max_s + 1.5)])
im20 = ax.imshow(_h2d.T, origin="lower",
                 extent=[_xedge[0], _xedge[-1], _yedge[0], _yedge[-1]],
                 aspect="auto", cmap="YlOrRd")
fig20.colorbar(im20, ax=ax, label="Count")
ax.set_xlabel("Ghost count")
ax.set_ylabel("Split count")
ax.set_title("Ghost count vs split count")
for xi in range(_h2d.shape[0]):
    for yi in range(_h2d.shape[1]):
        v = int(_h2d[xi, yi])
        if v > 0:
            ax.text(xi, yi, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > _h2d.max() * 0.4 else "black")

fig20.tight_layout()
fig20.savefig(os.path.join(outdir, "20_ghost_origin.png"), dpi=150)
plt.close(fig20)
print("  Saved 20_ghost_origin.png")

# ---------------------------------------------------------------
# FIGURE 21: True vs reco multiplicity migration matrix (Study D2)
# ---------------------------------------------------------------
print("\n--- Figure 21: Multiplicity migration matrix ---")
_pand_ev_mask = ev["npandora_total"] >= 0

fig21, axes21 = plt.subplots(1, 3, figsize=(18, 6))
fig21.suptitle("Particle Multiplicity Migration Matrix — SPINE vs Pandora (5k CC)",
               fontsize=13, fontweight="bold")

_nmax = 9
_nt_arr  = ev["ntruth"].astype(int).clip(0, _nmax - 1)
_ns_arr  = ev["nspine"].astype(int).clip(0, _nmax - 1)
_np_arr  = ev["npandora_primary"].astype(int).clip(0, _nmax - 1)
_delta_spine = (ev["nspine"] - ev["ntruth"]).astype(int)
_delta_pand  = (ev["npandora_primary"][_pand_ev_mask].astype(float) -
                ev["ntruth"][_pand_ev_mask].astype(float)).astype(int)

# Panel a: SPINE migration (row-normalised)
ax = axes21[0]
_mig_s = np.zeros((_nmax, _nmax))
for nt, ns in zip(_nt_arr, _ns_arr):
    _mig_s[nt, ns] += 1
_row_sum_s = _mig_s.sum(axis=1, keepdims=True).clip(min=1)
_mig_s_norm = _mig_s / _row_sum_s
im_s = ax.imshow(_mig_s_norm, cmap="Blues", vmin=0, vmax=1, origin="lower")
fig21.colorbar(im_s, ax=ax, label="Row-normalised fraction")
ax.set_xlabel("Reco multiplicity (SPINE)")
ax.set_ylabel("True primary multiplicity")
ax.set_title("SPINE migration (row-normalised)")
ax.set_xticks(range(_nmax)); ax.set_yticks(range(_nmax))
for i in range(_nmax):
    for j in range(_nmax):
        if _mig_s[i, j] > 0:
            ax.text(j, i, f"{_mig_s_norm[i,j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if _mig_s_norm[i, j] > 0.5 else "black")

# Panel b: Pandora migration (row-normalised)
ax = axes21[1]
_nt_p = _nt_arr[_pand_ev_mask]
_np_p = _np_arr[_pand_ev_mask]
_mig_p = np.zeros((_nmax, _nmax))
for nt, np_ in zip(_nt_p, _np_p):
    _mig_p[nt, np_] += 1
_row_sum_p = _mig_p.sum(axis=1, keepdims=True).clip(min=1)
_mig_p_norm = _mig_p / _row_sum_p
im_p = ax.imshow(_mig_p_norm, cmap="Oranges", vmin=0, vmax=1, origin="lower")
fig21.colorbar(im_p, ax=ax, label="Row-normalised fraction")
ax.set_xlabel("Reco multiplicity (Pandora)")
ax.set_ylabel("True primary multiplicity")
ax.set_title("Pandora migration (row-normalised)")
ax.set_xticks(range(_nmax)); ax.set_yticks(range(_nmax))
for i in range(_nmax):
    for j in range(_nmax):
        if _mig_p[i, j] > 0:
            ax.text(j, i, f"{_mig_p_norm[i,j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if _mig_p_norm[i, j] > 0.5 else "black")

# Panel c: 1D delta distribution (N_reco - N_truth)
ax = axes21[2]
_delta_range = np.arange(-4, 8)
_h_s, _ = np.histogram(_delta_spine.clip(-4, 7), bins=np.arange(-4.5, 7.6))
_h_p, _ = np.histogram(_delta_pand.clip(-4, 7),  bins=np.arange(-4.5, 7.6))
_w = 0.4
ax.bar(_delta_range - _w/2, _h_s, width=_w, color="#1f77b4", alpha=0.75, label="SPINE")
ax.bar(_delta_range + _w/2, _h_p, width=_w, color="#ff7f0e", alpha=0.75, label="Pandora")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel(r"$N_{\mathrm{reco}} - N_{\mathrm{truth}}$")
ax.set_ylabel("Events")
ax.set_title(r"Counting bias ($N_\mathrm{reco} - N_\mathrm{truth}$)")
ax.legend()
_med_s = float(np.median(_delta_spine))
_med_p = float(np.median(_delta_pand)) if len(_delta_pand) > 0 else float("nan")
ax.text(0.97, 0.95, f"SPINE median: {_med_s:+.1f}\nPandora median: {_med_p:+.1f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

fig21.tight_layout()
fig21.savefig(os.path.join(outdir, "21_multiplicity_migration.png"), dpi=150)
plt.close(fig21)
print("  Saved 21_multiplicity_migration.png")

# JSON summary
summary = {
    "n_events": len(event_stats),
    "spine_n_matched_pairs": len(particles),
    "pandora_n_matched_pairs": len(pandora_particles),
    "completeness_mean": float(np.mean(ev["completeness"])),
    "purity_mean": float(np.mean(ev["purity"])),
    "ghosts_mean": float(np.mean(ev["ghosts"])),
    "split_mean": float(np.mean(ev["split"])),
}
if "pand_completeness" in ev:
    pand_mask = np.isfinite(ev["pand_completeness"])
    summary.update({
        "pand_events": int(pand_mask.sum()),
        "pand_completeness_mean": float(np.mean(ev["pand_completeness"][pand_mask])),
        "pand_purity_mean": float(np.mean(ev["pand_purity"][pand_mask])),
        "pand_ghosts_mean": float(np.mean(ev["pand_ghosts"][pand_mask])),
        "pand_split_mean": float(np.mean(ev["pand_split"][pand_mask])),
    })
for ptype in cat_order:
    mask = np.array([pdg_category(t) == ptype for t in sp["true_pdg"]])
    if mask.sum() < 5: continue
    a = sp["ang_deg"][mask]; a = a[np.isfinite(a)]
    r = sp["reco_p"][mask] / sp["true_p"][mask]
    r = r[np.isfinite(r) & (r > 0) & (r < 10)]
    summary[f"spine_{ptype}_N"] = int(mask.sum())
    summary[f"spine_{ptype}_ang_median_deg"] = float(np.median(a)) if len(a) > 0 else None
    summary[f"spine_{ptype}_ang_iqr_deg"] = float(iqr_half(a)) if len(a) > 0 else None
    summary[f"spine_{ptype}_p_ratio_median"] = float(np.median(r)) if len(r) > 0 else None
    if pd is not None:
        pmask = np.array([pdg_category(t) == ptype for t in pd["true_pdg"]])
        if pmask.sum() >= 5:
            pa = pd["ang_deg"][pmask]; pa = pa[np.isfinite(pa)]
            summary[f"pandora_{ptype}_N"] = int(pmask.sum())
            summary[f"pandora_{ptype}_ang_median_deg"] = float(np.median(pa)) if len(pa) > 0 else None
            summary[f"pandora_{ptype}_ang_iqr_deg"] = float(iqr_half(pa)) if len(pa) > 0 else None

with open(os.path.join(outdir, "per_particle_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFigures saved to {outdir}/ (prefix: 1x-17x)")
print("Done!")

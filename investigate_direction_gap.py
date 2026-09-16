#!/usr/bin/env python3
"""
Deep investigation: why does SPINE direction resolution lag Pandora at low energy?

This script goes beyond Study H (direction_decomposition) and analyses:
  1. N_reco dependence — how does adding more reconstructed particles hurt the sum?
  2. Leading-particle correctness — does the highest-p SPINE particle match the truth lepton?
     If not, what is it (true PDG breakdown)?
  3. Truth-lepton fate — three mutually exclusive cases:
       A. Truth lepton reconstructed + correctly PID'd as lepton (PDG 11/13)
       B. Truth lepton reconstructed + wrong PDG (mis-PID'd as hadron)
       C. Truth lepton not reconstructed at all
  4. Flavor breakdown — νμ vs νe, since Pandora uses different estimators for each
  5. Summary quantification — what fraction of the SPINE–Pandora gap is explained by each factor?

Output figures (saved to figures/):
  direction_gap_nreco.png      — N_reco dependence
  direction_gap_leading.png    — leading-particle identity analysis
  direction_gap_lepton_fate.png — truth-lepton fate analysis
  direction_gap_summary.png   — quantitative summary
"""

import os, sys, ctypes, math, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binned_statistic as _bstat
from tqdm import tqdm

import caf_env
pySRProxy = caf_env.setup_pysrproxy()
import plot_style
plot_style.apply()

outdir = "figures"
os.makedirs(outdir, exist_ok=True)
_nan = float("nan")

# ── constants ──────────────────────────────────────────────────────────────────
COS_MATCH    = 0.5     # geometric matching threshold
P_MIN_GEV    = 0.02    # truth primary threshold [GeV/c]
P_MIN_MEV    = 0.5     # SPINE reco threshold [MeV/c]
EM_CORR      = 0.198   # ADC-to-electron calibration for EM particles (±11, 22, 111)
EM_PDG       = {11, -11, 22, 111}
LEPTON_PDGS  = {11, -11, 13, -13}
HADRON_PDGS  = {211, -211, 111, 2212, 2112, 321, -321, 310, 130}

# ── helpers ───────────────────────────────────────────────────────────────────
def vmag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def cos_angle(ax, ay, az, bx, by, bz):
    na = vmag(ax, ay, az)
    nb = vmag(bx, by, bz)
    if not (math.isfinite(na) and na > 1e-9) or not (math.isfinite(nb) and nb > 1e-9):
        return _nan
    return max(-1.0, min(1.0, (ax*bx + ay*by + az*bz) / (na*nb)))

def ang_deg(cos_val):
    if math.isfinite(cos_val):
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))
    return _nan

def closest_dlp(ixn, vtx):
    best, bi = float("inf"), 0
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = vmag(v.x - vtx.x, v.y - vtx.y, v.z - vtx.z)
        if d < best:
            best, bi = d, i
    return bi

def closest_pan(ixn, vtx):
    best, bi = float("inf"), 0
    for i in range(ixn.npandora):
        v = ixn.pandora[i].vtx
        d = vmag(v.x - vtx.x, v.y - vtx.y, v.z - vtx.z)
        if d < best:
            best, bi = d, i
    return bi

def pdg_category(pdg):
    """Return coarse category string for a PDG code."""
    a = abs(int(pdg))
    if a in (11,):   return "e"
    if a in (13,):   return "μ"
    if a in (211,):  return "π±"
    if a in (111,):  return "π⁰"
    if a in (2212,): return "p"
    if a in (2112,): return "n"
    if a in (22,):   return "γ"
    if a in (321, 310, 130): return "K"
    return "other"

# ── scan ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Direction gap investigation: N_reco, leading particle, lepton fate")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader("recTree", caf_env.CAF_FILE)

rows = []

for ientry in tqdm(range(prox.entries()), desc="Scanning"):
    rec = prox.entry(ientry)
    mc  = rec.mc
    if mc.nnu < 1 or not mc.nu[0].iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu     = mc.nu[0]
    nu_pdg = int(nu.pdg)
    nu_E   = float(nu.E)
    is_numu = abs(nu_pdg) in (14,)  # True = νμ, False = νe

    # True neutrino direction
    tm     = nu.momentum
    tm_mag = vmag(tm.x, tm.y, tm.z)
    if not (math.isfinite(tm_mag) and tm_mag > 1e-9):
        continue
    nu_dx, nu_dy, nu_dz = tm.x/tm_mag, tm.y/tm_mag, tm.z/tm_mag

    # ── Truth primaries ───────────────────────────────────────────────────────
    truth_parts = []        # list of dicts
    truth_lep   = None      # highest-p lepton (dict)
    truth_lep_p = 0.0

    for ti in range(nu.nprim):
        tp   = nu.prim[ti]
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        p_mag = vmag(px, py, pz)
        if p_mag < P_MIN_GEV:
            continue
        d = {"pdg": int(tp.pdg), "dx": px/p_mag, "dy": py/p_mag, "dz": pz/p_mag,
             "pmag": p_mag, "matched": False}
        truth_parts.append(d)
        if abs(int(tp.pdg)) in LEPTON_PDGS and p_mag > truth_lep_p:
            truth_lep_p = p_mag
            truth_lep   = d

    n_truth_visible = len(truth_parts)
    if n_truth_visible == 0:
        continue

    # ── SPINE primaries ───────────────────────────────────────────────────────
    si  = closest_dlp(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    spine_parts = []
    for pi in range(dlp.part.ndlp):
        sp    = dlp.part.dlp[pi]
        ppx, ppy, ppz = sp.p.x, sp.p.y, sp.p.z
        p_sp  = vmag(ppx, ppy, ppz)
        if not (math.isfinite(p_sp) and p_sp > P_MIN_MEV):
            continue
        # EM particles (e, γ, π⁰) carry raw ADC-scale momentum ~5× too large;
        # apply the ADC-to-electron calibration before summing / ranking.
        scale = EM_CORR if abs(int(sp.pdg)) in EM_PDG else 1.0
        spine_parts.append({
            "pdg": int(sp.pdg),
            "dx": ppx/p_sp, "dy": ppy/p_sp, "dz": ppz/p_sp,
            "px": ppx * scale, "py": ppy * scale, "pz": ppz * scale,
            "pmag": p_sp * scale,     # MeV/c, EM-calibrated
        })

    n_reco = len(spine_parts)

    if n_reco == 0:
        continue

    # Momentum sum
    sum_px = sum(s["px"] for s in spine_parts)
    sum_py = sum(s["py"] for s in spine_parts)
    sum_pz = sum(s["pz"] for s in spine_parts)
    sum_mag = vmag(sum_px, sum_py, sum_pz)

    spine_ang_sum = ang_deg(cos_angle(sum_px, sum_py, sum_pz, nu_dx, nu_dy, nu_dz)) \
                   if sum_mag > 0.1 else _nan

    # Leading SPINE primary (highest momentum)
    lead = max(spine_parts, key=lambda s: s["pmag"])
    spine_ang_lead = ang_deg(cos_angle(lead["dx"], lead["dy"], lead["dz"],
                                       nu_dx, nu_dy, nu_dz))

    # ── Match SPINE → truth (greedy, for each SPINE find closest truth) ───────
    # We only need: which truth primary does each SPINE primary correspond to?
    # Specifically: does the leading SPINE primary correspond to the truth lepton?
    def find_best_truth(sp):
        """Return (best_cos, best_truth_dict) for a SPINE particle."""
        best_c, best_t = COS_MATCH, None
        for tp in truth_parts:
            c = cos_angle(sp["dx"], sp["dy"], sp["dz"], tp["dx"], tp["dy"], tp["dz"])
            if math.isfinite(c) and c > best_c:
                best_c, best_t = c, tp
        return best_c, best_t

    # Identify truth lepton fate
    truth_lep_reconstructed = False
    truth_lep_reco_pdg      = None    # PDG SPINE assigned to the matched particle
    truth_lep_spine_dir     = None    # direction of SPINE particle matched to truth lep

    if truth_lep is not None:
        # Check if any SPINE primary matches the truth lepton
        best_c_lep, best_c = COS_MATCH, -1.0
        best_sp_for_lep = None
        for sp in spine_parts:
            c = cos_angle(sp["dx"], sp["dy"], sp["dz"],
                          truth_lep["dx"], truth_lep["dy"], truth_lep["dz"])
            if math.isfinite(c) and c > best_c:
                best_c = c
                best_sp_for_lep = sp
        if best_c > COS_MATCH and best_sp_for_lep is not None:
            truth_lep_reconstructed = True
            truth_lep_reco_pdg      = best_sp_for_lep["pdg"]
            truth_lep_spine_dir     = (best_sp_for_lep["dx"],
                                       best_sp_for_lep["dy"],
                                       best_sp_for_lep["dz"])

    # Is the truth lepton correctly PID'd by SPINE?
    truth_lep_correctly_pid = (truth_lep_reconstructed and
                               abs(truth_lep_reco_pdg or 0) in LEPTON_PDGS)

    # Classify lepton fate into 3 mutually exclusive cases
    if truth_lep is None:
        lep_fate = "no_truth_lep"       # shouldn't happen for CC, but be safe
    elif truth_lep_reconstructed and truth_lep_correctly_pid:
        lep_fate = "reco_correct_pid"   # A: found and correctly labeled
    elif truth_lep_reconstructed and not truth_lep_correctly_pid:
        lep_fate = "reco_wrong_pid"     # B: found but mis-PID'd as hadron
    else:
        lep_fate = "not_reco"           # C: not reconstructed at all

    # Does the leading SPINE particle (by momentum) correspond to the truth lepton?
    lead_c, lead_truth = find_best_truth(lead)
    leading_is_truth_lep = (
        truth_lep is not None and lead_truth is not None and
        abs(lead_truth["pdg"]) in LEPTON_PDGS
    )
    lead_true_pdg = lead_truth["pdg"] if lead_truth is not None else None
    lead_true_cat = pdg_category(lead_true_pdg) if lead_true_pdg is not None else "ghost"

    # Hadronic momentum fraction (SPINE primaries that are NOT matched to the truth lepton)
    had_px = had_py = had_pz = 0.0
    for sp in spine_parts:
        c_lep = cos_angle(sp["dx"], sp["dy"], sp["dz"],
                          truth_lep["dx"] if truth_lep else 0,
                          truth_lep["dy"] if truth_lep else 0,
                          truth_lep["dz"] if truth_lep else 0)
        is_lep_particle = (truth_lep is not None and
                           math.isfinite(c_lep) and c_lep > COS_MATCH)
        if not is_lep_particle:
            had_px += sp["px"]; had_py += sp["py"]; had_pz += sp["pz"]
    had_mag = vmag(had_px, had_py, had_pz)
    had_pfrac = had_mag / sum_mag if sum_mag > 0.1 else _nan

    # Truth-lepton momentum fraction in reco sum (only if reconstructed)
    if truth_lep_reconstructed and truth_lep_spine_dir is not None:
        lep_psp = None
        for sp in spine_parts:
            c = cos_angle(sp["dx"], sp["dy"], sp["dz"], *truth_lep_spine_dir)
            if math.isfinite(c) and c > 0.99:
                lep_psp = sp["pmag"]; break
        if lep_psp is None:
            # fallback: find closest
            best_c2, best_lsp = COS_MATCH, None
            for sp in spine_parts:
                c = cos_angle(sp["dx"], sp["dy"], sp["dz"], *truth_lep_spine_dir)
                if math.isfinite(c) and c > best_c2:
                    best_c2, best_lsp = c, sp
            lep_psp = best_lsp["pmag"] if best_lsp else None
        lep_pfrac = (lep_psp / sum_mag) if (lep_psp and sum_mag > 0.1) else _nan
    else:
        lep_pfrac = _nan

    # Reconstruction efficiency: how many truth primaries have a SPINE match?
    n_truth_matched = 0
    for tp in truth_parts:
        for sp in spine_parts:
            c = cos_angle(sp["dx"], sp["dy"], sp["dz"], tp["dx"], tp["dy"], tp["dz"])
            if math.isfinite(c) and c > COS_MATCH:
                n_truth_matched += 1
                break
    reco_eff = n_truth_matched / n_truth_visible if n_truth_visible > 0 else _nan

    # ── Pandora ───────────────────────────────────────────────────────────────
    pand_ang = _nan
    if ixn.npandora > 0:
        pi2 = closest_pan(ixn, nu.vtx)
        pd  = ixn.pandora[pi2].dir
        if is_numu:
            pdx, pdy, pdz = pd.lngtrk.x, pd.lngtrk.y, pd.lngtrk.z
        else:
            pdx, pdy, pdz = pd.heshw.x, pd.heshw.y, pd.heshw.z
        pand_ang = ang_deg(cos_angle(pdx, pdy, pdz, nu_dx, nu_dy, nu_dz))

        # Does Pandora's selected direction (lngtrk/heshw) match the truth lepton?
        pand_lngtrk_matches_lep = (
            truth_lep is not None and
            cos_angle(pdx, pdy, pdz,
                      truth_lep["dx"], truth_lep["dy"], truth_lep["dz"]) > COS_MATCH
        ) if vmag(pdx, pdy, pdz) > 1e-6 else False
    else:
        pand_lngtrk_matches_lep = False

    rows.append({
        "nu_E":     nu_E,
        "is_numu":  is_numu,
        "n_reco":   n_reco,
        "n_truth":  n_truth_visible,
        "reco_eff": reco_eff,
        # direction residuals
        "spine_sum":   spine_ang_sum,
        "spine_lead":  spine_ang_lead,
        "pand_ang":    pand_ang,
        # hadronic / lepton momentum fractions
        "had_pfrac":   had_pfrac,
        "lep_pfrac":   lep_pfrac,
        # leading particle analysis
        "leading_is_truth_lep":   leading_is_truth_lep,
        "lead_true_cat":          lead_true_cat,
        "lead_reco_pdg":          abs(lead["pdg"]),
        # truth lepton fate
        "lep_fate":                  lep_fate,
        "truth_lep_reco":            truth_lep_reconstructed,
        "truth_lep_correctly_pid":   truth_lep_correctly_pid,
        # Pandora kinematic selector hits the truth lepton?
        "pand_hits_lep":  pand_lngtrk_matches_lep,
    })

print(f"  CC events processed: {len(rows):,}")

# ── Convert to arrays ─────────────────────────────────────────────────────────
A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
A["nu_E"]   = A["nu_E"].astype(float)
A["n_reco"] = A["n_reco"].astype(int)

# Energy bins
E_EDGES  = [0.0, 0.5, 1.0, 2.0, 1e9]
E_LABELS = ["E<0.5 GeV", "0.5–1 GeV", "1–2 GeV", ">2 GeV"]
E_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

def ebin(E):
    return np.searchsorted(np.array(E_EDGES), E, side="right") - 1

ebin_arr = np.clip(ebin(A["nu_E"]), 0, len(E_LABELS)-1)

nu_bins   = np.linspace(0, 4.0, 25)
nu_ctr    = 0.5*(nu_bins[:-1] + nu_bins[1:])

def bstat(x, y, stat="median", bins=None):
    m = np.isfinite(x) & np.isfinite(y.astype(float))
    if m.sum() < 2:
        return np.full(len(bins)-1, _nan), np.zeros(len(bins)-1)
    v, _, _ = _bstat(x[m], y[m].astype(float), stat, bins=bins)
    c, _, _ = _bstat(x[m], y[m].astype(float), "count", bins=bins)
    return np.where(c >= 10, v, _nan), c

# ── Print summary table ───────────────────────────────────────────────────────
print("\nLepton fate breakdown by E_nu bin:")
print(f"  {'Bin':<14}  {'N':>6}  {'A: reco+PID':>12}  {'B: reco+wrong':>14}  {'C: not reco':>12}")
for bi, lb in enumerate(E_LABELS):
    m = ebin_arr == bi
    n = m.sum()
    fA = (A["lep_fate"][m] == "reco_correct_pid").mean()
    fB = (A["lep_fate"][m] == "reco_wrong_pid").mean()
    fC = (A["lep_fate"][m] == "not_reco").mean()
    print(f"  {lb:<14}  {n:>6,}  {100*fA:>11.1f}%  {100*fB:>13.1f}%  {100*fC:>11.1f}%")

print("\nLeading SPINE particle = truth lepton? by E_nu bin:")
for bi, lb in enumerate(E_LABELS):
    m = ebin_arr == bi
    frac = A["leading_is_truth_lep"][m].mean()
    print(f"  {lb:<14}: {100*frac:.1f}%")

print("\nPandora lngtrk/heshw = truth lepton? by E_nu bin:")
for bi, lb in enumerate(E_LABELS):
    m = ebin_arr == bi
    frac = A["pand_hits_lep"][m].mean()
    print(f"  {lb:<14}: {100*frac:.1f}%")

print("\nMedian direction residuals by fate and E_nu bin:")
for lep_fate, lbl in [("reco_correct_pid","A: reco+PID"), ("reco_wrong_pid","B: reco+wrong"),
                       ("not_reco","C: not reco")]:
    m_fate = A["lep_fate"] == lep_fate
    row = []
    for bi, lb in enumerate(E_LABELS):
        m2 = m_fate & (ebin_arr == bi) & np.isfinite(A["spine_sum"])
        row.append(f"{np.median(A['spine_sum'][m2]):.1f}°" if m2.sum() > 5 else "N/A")
    print(f"  SPINE sum | {lbl:<22}: {' | '.join(row)}")
for lep_fate, lbl in [("reco_correct_pid","A: reco+PID"), ("reco_wrong_pid","B: reco+wrong"),
                       ("not_reco","C: not reco")]:
    m_fate = A["lep_fate"] == lep_fate
    row = []
    for bi, lb in enumerate(E_LABELS):
        m2 = m_fate & (ebin_arr == bi) & np.isfinite(A["pand_ang"])
        row.append(f"{np.median(A['pand_ang'][m2]):.1f}°" if m2.sum() > 5 else "N/A")
    print(f"  Pandora   | {lbl:<22}: {' | '.join(row)}")

# ==============================================================================
# Figure 1: N_reco dependence
# ==============================================================================
print("\n--- Figure 1: N_reco dependence ---")

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("Direction Resolution vs Number of Reconstructed Particles ($N_\\mathrm{reco}$)\n"
              "CC events — SPINE vs Pandora", fontsize=12, fontweight="bold")

nreco_bins = np.arange(0.5, 12.5, 1)
nreco_ctr  = 0.5*(nreco_bins[:-1]+nreco_bins[1:])

# [0,0]: SPINE sum residual vs N_reco, one line per E_nu bin
ax = axes1[0, 0]
for bi, (lb, col) in enumerate(zip(E_LABELS, E_COLORS)):
    m = (ebin_arr == bi) & np.isfinite(A["spine_sum"])
    v, c = bstat(A["n_reco"][m].astype(float), A["spine_sum"][m], bins=nreco_bins)
    ax.plot(nreco_ctr[c >= 10], v[c >= 10], color=col, lw=2, marker="o", ms=5, label=lb)
ax.set_xlabel("$N_\\mathrm{reco}$ (SPINE primaries)")
ax.set_ylabel(r"Median $\Delta\theta$ [deg]")
ax.set_title("SPINE all-sum: residual vs $N_\\mathrm{reco}$")
ax.set_ylim(0, 90)
ax.legend(fontsize=9)

# [0,1]: Pandora residual vs N_reco (less sensitive by design)
ax = axes1[0, 1]
for bi, (lb, col) in enumerate(zip(E_LABELS, E_COLORS)):
    m = (ebin_arr == bi) & np.isfinite(A["pand_ang"])
    v, c = bstat(A["n_reco"][m].astype(float), A["pand_ang"][m], bins=nreco_bins)
    ax.plot(nreco_ctr[c >= 10], v[c >= 10], color=col, lw=2, marker="D", ms=5, label=lb)
ax.set_xlabel("$N_\\mathrm{reco}$ (SPINE primaries)")
ax.set_ylabel(r"Median $\Delta\theta$ [deg]")
ax.set_title("Pandora lngtrk/heshw: residual vs $N_\\mathrm{reco}$\n"
             "(kinematic selector — should be less sensitive)")
ax.set_ylim(0, 90)
ax.legend(fontsize=9)

# [1,0]: Hadronic momentum fraction vs N_reco
ax = axes1[1, 0]
m_hf = np.isfinite(A["had_pfrac"])
v_hf, c_hf = bstat(A["n_reco"][m_hf].astype(float), A["had_pfrac"][m_hf], bins=nreco_bins)
ax.plot(nreco_ctr[c_hf >= 10], v_hf[c_hf >= 10],
        color="#9467bd", lw=2, marker="s", ms=5, label="Median had. fraction")
# Add IQR band
v_lo, _ = bstat(A["n_reco"][m_hf].astype(float), A["had_pfrac"][m_hf],
                stat=lambda x: np.percentile(x,25), bins=nreco_bins)
v_hi, _ = bstat(A["n_reco"][m_hf].astype(float), A["had_pfrac"][m_hf],
                stat=lambda x: np.percentile(x,75), bins=nreco_bins)
ok_hf = c_hf >= 10
ax.fill_between(nreco_ctr[ok_hf],
                np.where(np.isfinite(v_lo[ok_hf]), v_lo[ok_hf], v_hf[ok_hf]),
                np.where(np.isfinite(v_hi[ok_hf]), v_hi[ok_hf], v_hf[ok_hf]),
                color="#9467bd", alpha=0.2, label="IQR")
ax.set_xlabel("$N_\\mathrm{reco}$ (SPINE primaries)")
ax.set_ylabel("Hadronic momentum fraction\n$|\\vec{p}_\\mathrm{had}|/|\\vec{p}_\\mathrm{all}|$")
ax.set_title("Why: more particles → more hadronic dilution")
ax.set_ylim(0, 1)
ax.legend(fontsize=9)

# [1,1]: SPINE vs Pandora gap vs N_reco (shows where the gap is largest)
ax = axes1[1, 1]
for bi, (lb, col) in enumerate(zip(E_LABELS, E_COLORS)):
    m = (ebin_arr == bi) & np.isfinite(A["spine_sum"]) & np.isfinite(A["pand_ang"])
    v_s, c_s = bstat(A["n_reco"][m].astype(float), A["spine_sum"][m], bins=nreco_bins)
    v_p, c_p = bstat(A["n_reco"][m].astype(float), A["pand_ang"][m], bins=nreco_bins)
    gap = v_s - v_p
    ok_g = (c_s >= 10) & (c_p >= 10) & np.isfinite(gap)
    ax.plot(nreco_ctr[ok_g], gap[ok_g], color=col, lw=2, marker="o", ms=5, label=lb)
ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xlabel("$N_\\mathrm{reco}$ (SPINE primaries)")
ax.set_ylabel(r"SPINE – Pandora median $\Delta\theta$ [deg]")
ax.set_title("SPINE–Pandora gap vs $N_\\mathrm{reco}$\n(positive = SPINE worse)")
ax.legend(fontsize=9)

fig1.tight_layout()
fig1.savefig(os.path.join(outdir, "direction_gap_nreco.png"), dpi=150)
plt.close(fig1)
print("  Saved direction_gap_nreco.png")

# ==============================================================================
# Figure 2: Leading-particle identity
# ==============================================================================
print("\n--- Figure 2: Leading-particle identity ---")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("Leading SPINE Primary (highest $p$): Is It the Truth Lepton?\n"
              "CC events", fontsize=12, fontweight="bold")

# [0,0]: Fraction "leading SPINE = truth lepton" vs E_nu — SPINE vs Pandora
ax = axes2[0, 0]
m_nu = A["nu_E"] < 4.0
spine_lead_lep, _ = bstat(A["nu_E"][m_nu],
                           A["leading_is_truth_lep"][m_nu].astype(float),
                           stat="mean", bins=nu_bins)
pand_lep, _       = bstat(A["nu_E"][m_nu],
                           A["pand_hits_lep"][m_nu].astype(float),
                           stat="mean", bins=nu_bins)

ax.plot(nu_ctr, spine_lead_lep, color=plot_style.SPINE_COLOR, lw=2, marker="o", ms=5,
        label="SPINE leading particle")
ax.plot(nu_ctr, pand_lep, color=plot_style.PANDORA_COLOR, lw=2, marker="D", ms=5,
        label="Pandora lngtrk/heshw")
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel("Fraction = truth lepton")
ax.set_title("Fraction of events where selected direction\nmatches truth lepton direction")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="k", lw=0.8, ls=":")
ax.legend(fontsize=9)
ax.text(0.02, 0.08,
        "SPINE: highest-p primary (PDG-agnostic)\n"
        "Pandora: longest track (νμ) / highest-E shower (νe)",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(fc="lightyellow", ec="gray", pad=3))

# [0,1]: SPINE all-sum residual split by "leading=truth lep" vs not
ax = axes2[0, 1]
for match_val, lbl, col, ls in [
    (True,  "Leading IS truth lepton", "#2ca02c",  "-"),
    (False, "Leading is NOT truth lepton", "#d62728", "--"),
]:
    m = (A["leading_is_truth_lep"] == match_val) & np.isfinite(A["spine_sum"]) & (A["nu_E"] < 4.0)
    v, c = bstat(A["nu_E"][m], A["spine_sum"][m], bins=nu_bins)
    ax.plot(nu_ctr[c >= 5], v[c >= 5], color=col, ls=ls, lw=2, marker="o", ms=5, label=lbl)
# Pandora reference
m_p = np.isfinite(A["pand_ang"]) & (A["nu_E"] < 4.0)
v_p, c_p = bstat(A["nu_E"][m_p], A["pand_ang"][m_p], bins=nu_bins)
ax.plot(nu_ctr[c_p >= 5], v_p[c_p >= 5], color=plot_style.PANDORA_COLOR,
        lw=2, marker="D", ms=5, ls="-", label="Pandora (reference)")
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel(r"Median $\Delta\theta$ [deg] (SPINE all-sum)")
ax.set_title("SPINE all-sum residual:\nsplit by leading-particle identity")
ax.set_ylim(0, 90)
ax.legend(fontsize=9)

# [1,0]: Stacked bar: true category of leading SPINE primary, by E_nu bin
ax = axes2[1, 0]
cats    = ["μ", "e", "π±", "p", "π⁰", "K", "n", "γ", "ghost", "other"]
cat_col = {"μ":"#1f77b4","e":"#2ca02c","π±":"#ff7f0e","p":"#17becf",
           "π⁰":"#8c564b","K":"#e377c2","n":"#d62728","γ":"#9467bd",
           "ghost":"#7f7f7f","other":"#bcbd22"}
bot = np.zeros(len(E_LABELS))
cat_fracs = {}
for cat in cats:
    frac = np.array([
        ((A["lead_true_cat"][ebin_arr == bi]) == cat).mean()
        for bi in range(len(E_LABELS))
    ])
    cat_fracs[cat] = frac

# Sort by total fraction (descending)
sorted_cats = sorted(cats, key=lambda c: cat_fracs[c].sum(), reverse=True)
for cat in sorted_cats:
    frac = cat_fracs[cat]
    if frac.max() < 0.01:
        continue
    ax.bar(E_LABELS, frac, bottom=bot, label=cat,
           color=cat_col.get(cat, "#7f7f7f"), alpha=0.85)
    # Label inside bar if large enough
    for xi, (b, f) in enumerate(zip(bot, frac)):
        if f > 0.05:
            ax.text(xi, b + f/2, f"{100*f:.0f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    bot += frac
ax.set_ylabel("Fraction of events")
ax.set_title("True identity of the leading SPINE primary\n(highest reconstructed momentum)")
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="upper right", ncol=2)

# [1,1]: SPINE lead-particle direction residual: when lead=truth_lep vs not
# + compare to Pandora when Pandora hits the truth lepton vs not
ax = axes2[1, 1]
for (arr_key, match_key, lbl, col, ls) in [
    ("spine_lead", "leading_is_truth_lep", "SPINE lead = truth lep", "#2ca02c",  "-"),
    ("spine_lead", None,                   "SPINE lead ≠ truth lep", "#d62728",  "--"),
    ("pand_ang",   "pand_hits_lep",        "Pandora hits truth lep", "#ff7f0e",  "-"),
    ("pand_ang",   None,                   "Pandora misses truth lep","#8c564b","--"),
]:
    if match_key is not None:
        m = (A[match_key] == True) & np.isfinite(A[arr_key]) & (A["nu_E"] < 4.0)
    else:
        match_k2 = "leading_is_truth_lep" if arr_key == "spine_lead" else "pand_hits_lep"
        m = (A[match_k2] == False) & np.isfinite(A[arr_key]) & (A["nu_E"] < 4.0)
    v, c = bstat(A["nu_E"][m], A[arr_key][m], bins=nu_bins)
    ax.plot(nu_ctr[c >= 5], v[c >= 5], color=col, ls=ls, lw=2, marker="o", ms=4, label=lbl)
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel(r"Median $\Delta\theta$ [deg]")
ax.set_title("Single-particle direction residual\nwhen selector hits vs misses truth lepton")
ax.set_ylim(0, 120)
ax.legend(fontsize=8)

fig2.tight_layout()
fig2.savefig(os.path.join(outdir, "direction_gap_leading.png"), dpi=150)
plt.close(fig2)
print("  Saved direction_gap_leading.png")

# ==============================================================================
# Figure 3: Truth-lepton fate
# ==============================================================================
print("\n--- Figure 3: Truth-lepton fate ---")

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle("Truth-Lepton Fate: Three Cases and Their Impact on Direction Resolution\n"
              "CC events", fontsize=12, fontweight="bold")

FATE_STYLES = [
    ("reco_correct_pid", "A: Reco + correct PID",  "#2ca02c", "-"),
    ("reco_wrong_pid",   "B: Reco + wrong PID",    "#ff7f0e", "--"),
    ("not_reco",         "C: Not reconstructed",   "#d62728", ":"),
]

# [0,0]: Fraction of each fate vs E_nu
ax = axes3[0, 0]
for fate, lbl, col, ls in FATE_STYLES:
    m = A["nu_E"] < 4.0
    v, c = bstat(A["nu_E"][m], (A["lep_fate"][m] == fate).astype(float),
                 stat="mean", bins=nu_bins)
    ax.plot(nu_ctr[c >= 5], v[c >= 5], color=col, ls=ls, lw=2, marker="o", ms=5, label=lbl)
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel("Fraction of CC events")
ax.set_title("Truth-lepton fate: fraction in each case vs $E_\\nu$")
ax.set_ylim(0, 1)
ax.legend(fontsize=9)

# Also show overall breakdown as text
overall = {f: (A["lep_fate"] == f).mean() for f, *_ in FATE_STYLES}
txt = "\n".join(f"{l}: {100*overall[f]:.1f}%" for f, l, *_ in FATE_STYLES)
ax.text(0.97, 0.96, txt, transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(fc="white", ec="gray", pad=3))

# [0,1]: SPINE all-sum residual split by fate, vs E_nu
ax = axes3[0, 1]
for fate, lbl, col, ls in FATE_STYLES:
    m = (A["lep_fate"] == fate) & np.isfinite(A["spine_sum"]) & (A["nu_E"] < 4.0)
    if m.sum() < 20:
        continue
    v, c = bstat(A["nu_E"][m], A["spine_sum"][m], bins=nu_bins)
    ax.plot(nu_ctr[c >= 5], v[c >= 5], color=col, ls=ls, lw=2, marker="o", ms=5, label=lbl)
# Pandora reference
m_p = np.isfinite(A["pand_ang"]) & (A["nu_E"] < 4.0)
v_p, c_p = bstat(A["nu_E"][m_p], A["pand_ang"][m_p], bins=nu_bins)
ax.plot(nu_ctr[c_p >= 5], v_p[c_p >= 5], color=plot_style.PANDORA_COLOR,
        lw=2, marker="D", ms=5, ls="-", label="Pandora (all events)", alpha=0.8)
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel(r"Median $\Delta\theta$ [deg] (SPINE all-sum)")
ax.set_title("SPINE all-sum residual by truth-lepton fate\nvs Pandora (reference)")
ax.set_ylim(0, 120)
ax.legend(fontsize=9)

# [1,0]: Pandora residual split by fate — does fate affect Pandora too?
ax = axes3[1, 0]
for fate, lbl, col, ls in FATE_STYLES:
    m = (A["lep_fate"] == fate) & np.isfinite(A["pand_ang"]) & (A["nu_E"] < 4.0)
    if m.sum() < 20:
        continue
    v, c = bstat(A["nu_E"][m], A["pand_ang"][m], bins=nu_bins)
    ax.plot(nu_ctr[c >= 5], v[c >= 5], color=col, ls=ls, lw=2, marker="D", ms=5, label=lbl)
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel(r"Median $\Delta\theta$ [deg] (Pandora)")
ax.set_title("Pandora residual split by truth-lepton fate\n"
             "(Pandora's kinematic selector vs SPINE's PID-based one)")
ax.set_ylim(0, 90)
ax.legend(fontsize=9)
ax.text(0.02, 0.96,
        "Pandora uses kinematic selector,\nnot the SPINE PDG label —\n"
        "fate B (wrong PID) should affect\nPandora less than SPINE",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(fc="lightyellow", ec="gray", pad=3))

# [1,1]: Lepton momentum fraction vs E_nu, split by fate A events only
# (when lepton IS reconstructed, how much of the momentum is it?)
ax = axes3[1, 1]
m_A = (A["lep_fate"] == "reco_correct_pid") & np.isfinite(A["lep_pfrac"]) & (A["nu_E"] < 4.0)
m_B = (A["lep_fate"] == "reco_wrong_pid")   & np.isfinite(A["lep_pfrac"]) & (A["nu_E"] < 4.0)

for m_f, lbl, col, ls in [(m_A, "A: Reco + correct PID", "#2ca02c", "-"),
                            (m_B, "B: Reco + wrong PID",   "#ff7f0e", "--")]:
    v, c = bstat(A["nu_E"][m_f], A["lep_pfrac"][m_f], bins=nu_bins)
    ax.plot(nu_ctr[c >= 5], v[c >= 5], color=col, ls=ls, lw=2, marker="o", ms=5,
            label=lbl)

ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel("Lepton momentum fraction\n$p_\\mathrm{lep}^\\mathrm{reco} / p_\\mathrm{all}^\\mathrm{reco}$")
ax.set_title("Lepton momentum fraction vs $E_\\nu$\n"
             "(when lepton is reconstructed)")
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.text(0.02, 0.12,
        "Lower fraction → hadronic system\ndominates the momentum sum",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(fc="lightyellow", ec="gray", pad=3))

fig3.tight_layout()
fig3.savefig(os.path.join(outdir, "direction_gap_lepton_fate.png"), dpi=150)
plt.close(fig3)
print("  Saved direction_gap_lepton_fate.png")

# ==============================================================================
# Figure 4: Summary — quantitative decomposition of the SPINE–Pandora gap
# ==============================================================================
print("\n--- Figure 4: Summary decomposition ---")

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
fig4.suptitle(
    "Quantitative Decomposition of the SPINE–Pandora Direction Gap\n"
    "How much does each failure mode contribute at low energy ($E_\\nu < 0.5$ GeV)?",
    fontsize=12, fontweight="bold")

# Panel [0]: Median residual breakdown at E < 0.5 GeV by fate × method
ax = axes4[0]
le_mask = (A["nu_E"] < 0.5)
combos = [
    # (spine mask, pandora mask, x_pos, label, color)
    (le_mask & np.isfinite(A["spine_sum"]),
     le_mask & np.isfinite(A["pand_ang"]),
     0, "All events", "#7f7f7f"),
    (le_mask & (A["lep_fate"]=="reco_correct_pid") & np.isfinite(A["spine_sum"]),
     le_mask & (A["lep_fate"]=="reco_correct_pid") & np.isfinite(A["pand_ang"]),
     1, "A: Reco+PID", "#2ca02c"),
    (le_mask & (A["lep_fate"]=="reco_wrong_pid") & np.isfinite(A["spine_sum"]),
     le_mask & (A["lep_fate"]=="reco_wrong_pid") & np.isfinite(A["pand_ang"]),
     2, "B: Reco+wrong PID", "#ff7f0e"),
    (le_mask & (A["lep_fate"]=="not_reco") & np.isfinite(A["spine_sum"]),
     le_mask & (A["lep_fate"]=="not_reco") & np.isfinite(A["pand_ang"]),
     3, "C: Not reco", "#d62728"),
]
x_pos = np.array([0, 1, 2, 3])
width = 0.35
spine_medians = []
pand_medians  = []
xlabels       = []
for sm, pm, xp, lbl, col in combos:
    sm_med = float(np.median(A["spine_sum"][sm])) if sm.sum() > 0 else _nan
    pm_med = float(np.median(A["pand_ang"][pm]))  if pm.sum() > 0 else _nan
    spine_medians.append(sm_med)
    pand_medians.append(pm_med)
    xlabels.append(f"{lbl}\n(N={sm.sum():,})")

spine_medians = np.array(spine_medians)
pand_medians  = np.array(pand_medians)
xp = np.arange(len(spine_medians))
ax.bar(xp - width/2, spine_medians, width, color=plot_style.SPINE_COLOR,   alpha=0.8, label="SPINE all-sum")
ax.bar(xp + width/2, pand_medians,  width, color=plot_style.PANDORA_COLOR, alpha=0.8, label="Pandora lngtrk/heshw")
for xi, (s, p) in enumerate(zip(spine_medians, pand_medians)):
    if math.isfinite(s): ax.text(xi-width/2, s+1, f"{s:.0f}°", ha="center", va="bottom", fontsize=9)
    if math.isfinite(p): ax.text(xi+width/2, p+1, f"{p:.0f}°", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xp)
ax.set_xticklabels(xlabels, fontsize=9)
ax.set_ylabel(r"Median $\Delta\theta$ [deg]")
ax.set_title("$E_\\nu < 0.5$ GeV: SPINE vs Pandora by fate")
ax.set_ylim(0, 120)
ax.legend(fontsize=9)

# Panel [1]: Fraction of events in each fate, by E_nu bin (stacked bars)
ax = axes4[1]
fate_cols = {"reco_correct_pid": "#2ca02c", "reco_wrong_pid": "#ff7f0e",
             "not_reco": "#d62728"}
fate_lbls = {"reco_correct_pid": "A: Reco+PID",
             "reco_wrong_pid":   "B: Reco+wrong PID",
             "not_reco":         "C: Not reco"}
bot2 = np.zeros(len(E_LABELS))
for fate in ["reco_correct_pid", "reco_wrong_pid", "not_reco"]:
    frac = np.array([(A["lep_fate"][ebin_arr == bi] == fate).mean()
                     for bi in range(len(E_LABELS))])
    ax.bar(E_LABELS, frac, bottom=bot2, color=fate_cols[fate],
           alpha=0.85, label=fate_lbls[fate])
    for xi, (b, f) in enumerate(zip(bot2, frac)):
        if f > 0.04:
            ax.text(xi, b + f/2, f"{100*f:.0f}%", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
    bot2 += frac
ax.set_ylabel("Fraction of CC events")
ax.set_title("Event fate breakdown by $E_\\nu$ bin")
ax.set_ylim(0, 1)
ax.legend(fontsize=9, loc="upper right")

# Panel [2]: "Pandora advantage" decomposed: for events in fate B and C,
# Pandora still gets direction right because its selector is kinematic.
# Show: fraction of events where Pandora hits truth lepton, by fate × E_nu bin
ax = axes4[2]
for fate, lbl, col, ls in FATE_STYLES:
    frac_pand_hits = []
    ns = []
    for bi in range(len(E_LABELS)):
        m = (A["lep_fate"] == fate) & (ebin_arr == bi) & np.isfinite(A["pand_ang"])
        n = m.sum()
        ns.append(n)
        if n > 10:
            frac_pand_hits.append(A["pand_hits_lep"][m].mean())
        else:
            frac_pand_hits.append(_nan)
    frac_pand_hits = np.array(frac_pand_hits, dtype=float)
    ax.plot(E_LABELS, frac_pand_hits, color=col, ls=ls, lw=2,
            marker="D", ms=7, label=lbl)
    for xi, (f, n) in enumerate(zip(frac_pand_hits, ns)):
        if math.isfinite(f) and n > 50:
            ax.text(xi, f + 0.02, f"{100*f:.0f}%", ha="center", va="bottom", fontsize=8, color=col)

ax.set_ylabel("Fraction where Pandora selector\nhits truth lepton")
ax.set_title("Pandora's kinematic robustness:\n"
             "fraction hitting truth lepton, by fate and $E_\\nu$")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)
ax.text(0.02, 0.08,
        "Pandora's selector (lngtrk/heshw)\nis kinematic and can hit the right\n"
        "particle even when SPINE mis-PID's it",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(fc="lightyellow", ec="gray", pad=3))

fig4.tight_layout()
fig4.savefig(os.path.join(outdir, "direction_gap_summary.png"), dpi=150)
plt.close(fig4)
print("  Saved direction_gap_summary.png")

# ==============================================================================
# JSON summary
# ==============================================================================
summary_out = {}
for bi, lb in enumerate(E_LABELS):
    m = ebin_arr == bi
    summary_out[lb] = {
        "n": int(m.sum()),
        "spine_sum_median":  round(float(np.median(A["spine_sum"][m & np.isfinite(A["spine_sum"])])), 2),
        "pand_median":       round(float(np.median(A["pand_ang"][m & np.isfinite(A["pand_ang"])])), 2),
        "fate_A_frac":       round(float((A["lep_fate"][m] == "reco_correct_pid").mean()), 3),
        "fate_B_frac":       round(float((A["lep_fate"][m] == "reco_wrong_pid").mean()), 3),
        "fate_C_frac":       round(float((A["lep_fate"][m] == "not_reco").mean()), 3),
        "leading_is_lep_frac": round(float(A["leading_is_truth_lep"][m].mean()), 3),
        "pand_hits_lep_frac":  round(float(A["pand_hits_lep"][m].mean()), 3),
        "had_pfrac_median":    round(float(np.median(A["had_pfrac"][m & np.isfinite(A["had_pfrac"])])), 3),
    }

with open(os.path.join(outdir, "direction_gap_summary.json"), "w") as fj:
    json.dump(summary_out, fj, indent=2)
print("  Saved direction_gap_summary.json")

print("\nDone!")

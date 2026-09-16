#!/usr/bin/env python3
"""
Implement the direction estimators from arXiv:2601.05697 (Sec 3.4.1)
applied to SPINE/DLP reconstruction and compare against our existing
SPINE all-sum and Pandora lngtrk/heshw.

Three estimators from the paper, translated to SPINE:

  spine_lep_pid   — "lepton-only" method: direction of the highest-p
                    particle with the expected lepton PDG (muon for νμ,
                    electron for νe).  Analog of Pandora lngtrk/heshw
                    but exploiting SPINE's per-species PID.

  spine_paper_sum — "all reconstructed particles" (Fig. 14 flowchart):
                    vector sum of all primaries following the paper's logic:
                      • longest non-EM track → assumed Muon → CSDA (muon mass)
                      • other tracks, proton (2212) → CSDA (proton mass)
                      • other tracks, pion (±211)  → Calo: |p.p| directly
                      • electron (±11), photon (22), π⁰ (111) → Calo × EM_CORR
                    Key: the longest-track-as-muon assumption is what distinguishes
                    this from spine_sum for events where SPINE's PDG ≠ muon on
                    the longest track.

  spine_sum       — our existing all-sum (EM correction, direct |p.p|
                    for all hadrons) — kept as the comparison baseline.

Output: figures/paper_method_comparison.png
"""

import os, sys, ctypes, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binned_statistic as _bstat
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

outdir = "figures"
os.makedirs(outdir, exist_ok=True)
_nan = float("nan")

# ── constants ──────────────────────────────────────────────────────────────────
COS_MATCH  = 0.5
P_MIN_MEV  = 0.5       # SPINE momentum threshold [MeV/c]
EM_CORR    = 0.198     # ADC-to-electron calibration for EM particles

# Particle masses [MeV/c²]
MASS = {13: 105.658, 11: 0.511, 211: 139.571, 2212: 938.272,
        22: 0.0, 111: 134.977}
# PDG codes treated as EM (shower-like) — apply EM_CORR
EM_PDG = {11, -11, 22, 111}
# PDG codes where we use KE + mass → CSDA-like momentum
CSDA_PDG = {13, -13, 2212}

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

def p_from_ke(ke_mev, mass_mev):
    """Relativistic momentum [MeV/c] from kinetic energy and rest mass."""
    E_tot = ke_mev + mass_mev
    p2 = E_tot * E_tot - mass_mev * mass_mev
    return math.sqrt(max(0.0, p2))

def closest_ixn(ixn, vtx, kind):
    """Return index of closest DLP ('dlp') or Pandora ('pandora') interaction."""
    best, bi = float("inf"), 0
    n = ixn.ndlp if kind == "dlp" else ixn.npandora
    src = ixn.dlp if kind == "dlp" else ixn.pandora
    for i in range(n):
        v = src[i].vtx
        d = vmag(v.x - vtx.x, v.y - vtx.y, v.z - vtx.z)
        if d < best:
            best, bi = d, i
    return bi

# ── scan ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Paper method (arXiv:2601.05697 Sec 3.4.1) applied to SPINE")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader("recTree", "spine_merged_reprocess.root")
rows = []

for ientry in tqdm(range(prox.entries()), desc="Scanning"):
    rec = prox.entry(ientry)
    mc  = rec.mc
    if mc.nnu < 1 or not mc.nu[0].iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu      = mc.nu[0]
    nu_pdg  = int(nu.pdg)
    nu_E    = float(nu.E)
    is_numu = abs(nu_pdg) == 14

    # True neutrino direction
    tm     = nu.momentum
    tm_mag = vmag(tm.x, tm.y, tm.z)
    if not (math.isfinite(tm_mag) and tm_mag > 1e-9):
        continue
    nu_dx, nu_dy, nu_dz = tm.x/tm_mag, tm.y/tm_mag, tm.z/tm_mag

    # ── SPINE particles ───────────────────────────────────────────────────────
    si  = closest_ixn(ixn, nu.vtx, "dlp")
    dlp = ixn.dlp[si]

    # Running sums for the three SPINE estimators
    sum_px = sum_py = sum_pz = 0.0          # existing all-sum (EM-corrected)
    pap_px = pap_py = pap_pz = 0.0          # paper all-particles (Fig. 14 flowchart)
    muon_cand  = None   # (pmag, dx, dy, dz)
    elec_cand  = None

    # ── Pass 1: find the longest non-EM track (Fig. 14: "longest track → Muon") ─
    longest_idx = -1
    longest_len = -1.0
    for pi in range(dlp.part.ndlp):
        sp = dlp.part.dlp[pi]
        ppx, ppy, ppz = sp.p.x, sp.p.y, sp.p.z
        p_mag = vmag(ppx, ppy, ppz)
        if not (math.isfinite(p_mag) and p_mag > P_MIN_MEV):
            continue
        if abs(int(sp.pdg)) in EM_PDG:
            continue  # showers excluded from longest-track selection
        try:
            tlen = vmag(float(sp.end.x) - float(sp.start.x),
                        float(sp.end.y) - float(sp.start.y),
                        float(sp.end.z) - float(sp.start.z))
        except Exception:
            tlen = 0.0
        if math.isfinite(tlen) and tlen > longest_len:
            longest_len, longest_idx = tlen, pi

    # ── Pass 2: accumulate direction estimators ───────────────────────────────
    for pi in range(dlp.part.ndlp):
        sp  = dlp.part.dlp[pi]
        ppx, ppy, ppz = sp.p.x, sp.p.y, sp.p.z
        p_mag = vmag(ppx, ppy, ppz)
        if not (math.isfinite(p_mag) and p_mag > P_MIN_MEV):
            continue

        pdg     = int(sp.pdg)
        pdg_abs = abs(pdg)
        ke      = float(sp.E)   # kinetic energy [MeV]
        dx, dy, dz = ppx/p_mag, ppy/p_mag, ppz/p_mag

        # --- Existing SPINE sum (EM correction only) ---
        scale = EM_CORR if pdg_abs in EM_PDG else 1.0
        sum_px += dx * p_mag * scale
        sum_py += dy * p_mag * scale
        sum_pz += dz * p_mag * scale

        # --- Paper method (Fig. 14 flowchart) ---
        #   Showers          → Calo × EM_CORR
        #   Longest track    → always Muon → CSDA with muon mass
        #   Other tracks     → SPINE PDG: proton→CSDA, pion→Calo
        if pdg_abs in EM_PDG:
            paper_p = p_mag * EM_CORR
        elif pi == longest_idx:
            # Longest track assumed to be the primary muon
            paper_p = p_from_ke(ke, MASS[13]) if (ke > 0 and math.isfinite(ke)) else p_mag
        elif pdg_abs in CSDA_PDG:
            m = MASS.get(pdg_abs, MASS[13])
            paper_p = p_from_ke(ke, m) if (ke > 0 and math.isfinite(ke)) else p_mag
        else:
            paper_p = p_mag   # pion / unknown → calorimetric
        pap_px += dx * paper_p
        pap_py += dy * paper_p
        pap_pz += dz * paper_p

        # --- Lepton candidate tracking (for spine_lep estimator) ---
        if pdg_abs == 13:  # muon
            if muon_cand is None or paper_p > muon_cand[0]:
                muon_cand = (paper_p, dx, dy, dz)
        elif pdg_abs == 11:  # electron
            elec_p = p_mag * EM_CORR
            if elec_cand is None or elec_p > elec_cand[0]:
                elec_cand = (elec_p, dx, dy, dz)

    if vmag(sum_px, sum_py, sum_pz) < 0.1:
        continue

    # ── Three SPINE direction estimators ─────────────────────────────────────
    spine_ang_sum = ang_deg(cos_angle(sum_px, sum_py, sum_pz, nu_dx, nu_dy, nu_dz))
    spine_ang_pap = ang_deg(cos_angle(pap_px, pap_py, pap_pz, nu_dx, nu_dy, nu_dz))

    # Lepton-only: prefer species-matched lepton; fall back to highest-p other
    if is_numu:
        lep_cand = muon_cand or elec_cand
    else:
        lep_cand = elec_cand or muon_cand

    if lep_cand:
        _, ldx, ldy, ldz = lep_cand
        spine_ang_lep = ang_deg(cos_angle(ldx, ldy, ldz, nu_dx, nu_dy, nu_dz))
    else:
        spine_ang_lep = _nan

    # ── Pandora (flavor-aware: lngtrk for νμ, heshw for νe) ──────────────────
    pand_ang = _nan
    if ixn.npandora > 0:
        pi2 = closest_ixn(ixn, nu.vtx, "pandora")
        pd  = ixn.pandora[pi2].dir
        if is_numu:
            pdx, pdy, pdz = pd.lngtrk.x, pd.lngtrk.y, pd.lngtrk.z
        else:
            pdx, pdy, pdz = pd.heshw.x, pd.heshw.y, pd.heshw.z
        pand_ang = ang_deg(cos_angle(pdx, pdy, pdz, nu_dx, nu_dy, nu_dz))

    rows.append({
        "nu_E":         nu_E,
        "is_numu":      is_numu,
        "spine_sum":    spine_ang_sum,
        "spine_paper":  spine_ang_pap,
        "spine_lep":    spine_ang_lep,
        "pand":         pand_ang,
    })

print(f"  CC events: {len(rows):,}")

# ── Convert to numpy arrays ───────────────────────────────────────────────────
A = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
A["is_numu"] = A["is_numu"].astype(bool)

nu_bins = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
nu_ctr  = 0.5 * (nu_bins[:-1] + nu_bins[1:])

def bstat_med(x, y, bins=nu_bins):
    """Median with count guard (≥10 events per bin)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.full(len(bins)-1, _nan)
    v, _, _ = _bstat(x[m], y[m], "median", bins=bins)
    c, _, _ = _bstat(x[m], y[m], "count",  bins=bins)
    return np.where(c >= 10, v, _nan)

# Print overall medians
E_CUTS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
E_LABS = ["E<0.5 GeV", "0.5–1 GeV", "1–2 GeV", ">2 GeV"]
keys   = ["spine_sum", "spine_paper", "spine_lep", "pand"]
labels = ["SPINE sum (existing)", "SPINE paper sum (CSDA μ/p)", "SPINE lep-PID", "Pandora"]

print(f"\n{'Bin':<14} " + "".join(f"{l:>24}" for l in labels))
for (lo, hi), lb in zip(E_CUTS, E_LABS):
    m = (A["nu_E"] >= lo) & (A["nu_E"] < hi)
    vals = [np.nanmedian(A[k][m]) for k in keys]
    print(f"{lb:<14} " + "".join(f"{v:>23.1f}°" for v in vals))

# ── Figure: 2×2 panels (all CC | νμ CC | νe CC | νμ-νe combined overlay) ────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(r"Direction reconstruction: paper method (arXiv:2601.05697) vs existing",
             fontsize=13)

STYLES = {
    "spine_sum":   dict(color="#1f77b4", ls="-",  lw=2.0, label="SPINE all-sum (existing)"),
    "spine_paper": dict(color="#d62728", ls="--", lw=2.0, label="SPINE paper sum (CSDA μ/p)"),
    "spine_lep":   dict(color="#ff7f0e", ls="-.", lw=2.0, label="SPINE lepton-PID only"),
    "pand":        dict(color="#2ca02c", ls=":",  lw=2.5, label="Pandora (lngtrk/heshw)"),
}

def plot_panel(ax, mask, title):
    x = A["nu_E"][mask]
    for k, st in STYLES.items():
        y = bstat_med(x, A[k][mask])
        ax.plot(nu_ctr, y, **st)
    ax.set_xlabel(r"True $E_\nu$ [GeV]")
    ax.set_ylabel(r"Median $\theta$ residual [°]")
    ax.set_title(title)
    ax.set_ylim(0, 80)
    ax.set_xlim(nu_bins[0], nu_bins[-1])
    ax.legend(fontsize=8)

plot_panel(axes[0, 0], np.ones(len(A["nu_E"]), dtype=bool), "All CC")
plot_panel(axes[0, 1], A["is_numu"],  r"$\nu_\mu$ CC only")
plot_panel(axes[1, 0], ~A["is_numu"], r"$\nu_e$ CC only")

# Bottom-right: difference (spine_paper − spine_sum) and (spine_lep − pand)
ax = axes[1, 1]
for k, ref, color, label in [
        ("spine_paper", "spine_sum",  "#d62728",
         r"SPINE paper − SPINE sum"),
        ("spine_lep",   "pand",       "#ff7f0e",
         r"SPINE lep-PID − Pandora"),
]:
    x = A["nu_E"]
    diff = A[k] - A[ref]
    y = bstat_med(x, diff)
    ax.plot(nu_ctr, y, color=color, lw=2.0, label=label)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel(r"Median $\Delta\theta$ [°]  (method − baseline)")
ax.set_title("Method differences")
ax.set_ylim(-15, 30)
ax.set_xlim(nu_bins[0], nu_bins[-1])
ax.legend(fontsize=8)

fig.tight_layout()
out = os.path.join(outdir, "paper_method_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

# ── Figure 2: split by νμ / νe and compare all four estimators ───────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(r"Flavor-split comparison: SPINE PID-informed vs Pandora", fontsize=13)

for ax, mask, fl in [(axes2[0], A["is_numu"],  r"$\nu_\mu$ CC"),
                     (axes2[1], ~A["is_numu"], r"$\nu_e$ CC")]:
    x = A["nu_E"][mask]
    for k, st in STYLES.items():
        y = bstat_med(x, A[k][mask])
        ax.plot(nu_ctr, y, **st)
    ax.set_xlabel(r"True $E_\nu$ [GeV]")
    ax.set_ylabel(r"Median $\theta$ residual [°]")
    ax.set_title(fl)
    ax.set_ylim(0, 80)
    ax.set_xlim(nu_bins[0], nu_bins[-1])
    ax.legend(fontsize=8.5)

fig2.tight_layout()
out2 = os.path.join(outdir, "paper_method_flavor_split.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")

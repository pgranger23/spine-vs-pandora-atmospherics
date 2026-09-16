#!/usr/bin/env python3
"""
Diagnostic: why does SPINE have worse neutrino direction reconstruction than
Pandora despite better per-particle angular resolution?

Strategy: compute multiple direction estimators from the same SPINE output and
compare them to the Pandora lngtrk/heshw approach used in Study B.

Direction estimators:
  spine_sum    : vector sum of ALL reco primary momenta (Study B method)
  spine_lep    : momentum of the highest-p SPINE lepton (PDG 11/13) only
  spine_lead   : direction of the single highest-momentum SPINE primary
  pand_flavor  : Pandora lngtrk (numuCC) or heshw (nueCC) — Study B method

Additional diagnostics:
  ghost_pfrac  : fraction of total SPINE reco momentum from ghost particles
  had_pfrac    : fraction of total SPINE reco momentum from hadronic primaries
  muon_found   : whether the highest-p truth muon has a SPINE geometric match

Output:
  figures/direction_decomposition.png
  figures/direction_decomposition.json
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

outdir = "figures"
os.makedirs(outdir, exist_ok=True)
_nan = float("nan")

COS_MATCH  = 0.5    # cosθ threshold for ghost identification
P_MIN_GEV  = 0.02   # truth primary momentum threshold

_LEPTON_PDGS = {11, -11, 13, -13}

def _vmag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def _cos(ax, ay, az, bx, by, bz):
    na = _vmag(ax, ay, az)
    nb = _vmag(bx, by, bz)
    if not (math.isfinite(na) and na > 1e-9) or not (math.isfinite(nb) and nb > 1e-9):
        return _nan
    return max(-1.0, min(1.0, (ax*bx + ay*by + az*bz) / (na * nb)))

def _ang_deg(cos_val):
    return math.degrees(math.acos(cos_val)) if math.isfinite(cos_val) else _nan

def _closest_dlp(ixn, nu_vtx):
    best, bi = float("inf"), 0
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = _vmag(v.x-nu_vtx.x, v.y-nu_vtx.y, v.z-nu_vtx.z)
        if d < best:
            best, bi = d, i
    return bi

def _closest_pan(ixn, nu_vtx):
    best, bi = float("inf"), 0
    for i in range(ixn.npandora):
        v = ixn.pandora[i].vtx
        d = _vmag(v.x-nu_vtx.x, v.y-nu_vtx.y, v.z-nu_vtx.z)
        if d < best:
            best, bi = d, i
    return bi

# ---------------------------------------------------------------
print("=" * 60)
print("Direction decomposition: SPINE all-sum vs lep-only vs Pandora")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader("recTree", "spine_merged_reprocess.root")

records = []
n_cc = 0

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
    n_cc  += 1

    # True neutrino direction
    tm = nu.momentum
    tm_mag = _vmag(tm.x, tm.y, tm.z)
    if not (math.isfinite(tm_mag) and tm_mag > 1e-9):
        continue
    tdx, tdy, tdz = tm.x/tm_mag, tm.y/tm_mag, tm.z/tm_mag

    # Truth primaries used for ghost identification
    truth_dirs = []
    truth_lep_dir = None        # highest-p truth lepton direction
    truth_lep_p   = 0.0
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        p_mag = _vmag(px, py, pz)
        if p_mag < P_MIN_GEV:
            continue
        dx, dy, dz = px/p_mag, py/p_mag, pz/p_mag
        truth_dirs.append((dx, dy, dz, p_mag))
        if abs(int(tp.pdg)) in _LEPTON_PDGS and p_mag > truth_lep_p:
            truth_lep_p   = p_mag
            truth_lep_dir = (dx, dy, dz)

    # ---- SPINE ----
    si  = _closest_dlp(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    # Classify each SPINE primary
    sum_px = sum_py = sum_pz = 0.0    # all-particle sum
    lep_px = lep_py = lep_pz = 0.0   # lepton sum (usually just one)
    had_px = had_py = had_pz = 0.0   # hadronic sum
    ghost_px = ghost_py = ghost_pz = 0.0  # ghost sum

    lep_p_best   = 0.0   # highest-p SPINE lepton
    lead_p_best  = 0.0   # highest-p SPINE primary overall
    lep_best_dir = (0., 0., 0.)
    lead_best_dir= (0., 0., 0.)

    n_lep_found  = 0     # N SPINE leptons (expect 0-1 for CC)
    n_ghost      = 0

    for pi in range(dlp.part.ndlp):
        sp = dlp.part.dlp[pi]
        ppx, ppy, ppz = sp.p.x, sp.p.y, sp.p.z
        p_sp = _vmag(ppx, ppy, ppz)
        if not (math.isfinite(p_sp) and p_sp > 0.5):  # MeV/c, > 0.5 MeV
            continue
        # EM particles (e, γ, π⁰): apply ADC-to-electron calibration (×0.198)
        _sc = 0.198 if abs(int(sp.pdg)) in (11, 22, 111) else 1.0
        ppx *= _sc; ppy *= _sc; ppz *= _sc
        p_sp *= _sc
        sum_px += ppx; sum_py += ppy; sum_pz += ppz

        if p_sp > lead_p_best:
            lead_p_best = p_sp
            lead_best_dir = (ppx/p_sp, ppy/p_sp, ppz/p_sp)

        rpdg = int(sp.pdg)
        is_lep = abs(rpdg) in _LEPTON_PDGS

        if is_lep:
            lep_px += ppx; lep_py += ppy; lep_pz += ppz
            n_lep_found += 1
            if p_sp > lep_p_best:
                lep_p_best = p_sp
                lep_best_dir = (ppx/p_sp, ppy/p_sp, ppz/p_sp)
        else:
            had_px += ppx; had_py += ppy; had_pz += ppz

        # Ghost check: no truth primary within COS_MATCH
        is_ghost = True
        for (tdx2, tdy2, tdz2, _) in truth_dirs:
            c = _cos(ppx, ppy, ppz, tdx2, tdy2, tdz2)
            if math.isfinite(c) and c > COS_MATCH:
                is_ghost = False
                break
        if is_ghost:
            ghost_px += ppx; ghost_py += ppy; ghost_pz += ppz
            n_ghost += 1

    # Direction angles
    sum_mag   = _vmag(sum_px, sum_py, sum_pz)
    ghost_mag = _vmag(ghost_px, ghost_py, ghost_pz)
    had_mag   = _vmag(had_px, had_py, had_pz)
    lep_mag   = _vmag(lep_px, lep_py, lep_pz)

    spine_ang_sum  = _ang_deg(_cos(sum_px, sum_py, sum_pz, tdx, tdy, tdz)) if sum_mag > 0.1 else _nan
    spine_ang_lep  = _ang_deg(_cos(*lep_best_dir, tdx, tdy, tdz)) if lep_p_best > 0.1 else _nan
    spine_ang_lead = _ang_deg(_cos(*lead_best_dir, tdx, tdy, tdz)) if lead_p_best > 0.1 else _nan

    ghost_pfrac = ghost_mag / sum_mag if sum_mag > 0.1 else _nan
    had_pfrac   = had_mag   / sum_mag if sum_mag > 0.1 else _nan

    # Angle between lep-only and all-sum (how much do hadrons rotate the sum?)
    lep_to_sum_ang = _ang_deg(_cos(lep_px, lep_py, lep_pz, sum_px, sum_py, sum_pz)) \
                     if (lep_mag > 0.1 and sum_mag > 0.1) else _nan

    # Was the truth lepton direction well-reconstructed by SPINE?
    lep_matched = False
    if truth_lep_dir is not None and n_lep_found > 0:
        c2 = _cos(*lep_best_dir, *truth_lep_dir)
        lep_matched = math.isfinite(c2) and c2 > COS_MATCH

    # ---- Pandora ----
    pand_ang = _nan
    if ixn.npandora > 0:
        pi2 = _closest_pan(ixn, nu.vtx)
        pd  = ixn.pandora[pi2].dir
        if abs(nu_pdg) == 14:
            px2, py2, pz2 = pd.lngtrk.x, pd.lngtrk.y, pd.lngtrk.z
        else:
            px2, py2, pz2 = pd.heshw.x, pd.heshw.y, pd.heshw.z
        c3 = _cos(px2, py2, pz2, tdx, tdy, tdz)
        pand_ang = _ang_deg(c3)

    records.append({
        "nu_E": nu_E, "nu_pdg": nu_pdg,
        "spine_ang_sum":  spine_ang_sum,
        "spine_ang_lep":  spine_ang_lep,
        "spine_ang_lead": spine_ang_lead,
        "pand_ang":       pand_ang,
        "ghost_pfrac":    ghost_pfrac,
        "had_pfrac":      had_pfrac,
        "lep_to_sum_ang": lep_to_sum_ang,
        "lep_matched":    lep_matched,
        "n_lep_found":    n_lep_found,
        "n_ghost":        n_ghost,
    })

print(f"  CC events: {n_cc:,}, records: {len(records):,}")

# Convert to arrays
arr = {k: np.array([r[k] for r in records]) for k in records[0]}
arr["lep_matched"] = arr["lep_matched"].astype(bool)
arr["nu_E"]  = arr["nu_E"].astype(float)
finite = lambda x: np.isfinite(arr[x])

# ---------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------
E_EDGES  = [0.0, 0.5, 1.0, 2.0, 1e9]
E_LABELS = ["E<0.5 GeV\n(QE-like)", "0.5-1 GeV\n(RES-like)",
             "1-2 GeV\n(MEC/DIS)", ">2 GeV\n(DIS-like)"]

def _e_bin(e_arr):
    bins = np.array(E_EDGES)
    idx  = np.searchsorted(bins, e_arr, side="right") - 1
    return np.clip(idx, 0, len(E_LABELS)-1)

e_bin = _e_bin(arr["nu_E"])

METHODS = [
    ("spine_ang_sum",  "SPINE all-sum (Study B)", "#1f77b4", "-",  "o"),
    ("spine_ang_lep",  "SPINE lep only",           "#1f77b4", "--", "s"),
    ("spine_ang_lead", "SPINE leading particle",   "#17becf", ":",  "^"),
    ("pand_ang",       "Pandora lngtrk/heshw",     "#ff7f0e", "-",  "D"),
]

print("\nMedian angular residual per E_nu bin:")
print(f"{'Method':<30}  " + "  ".join(f"{lb.split(chr(10))[0]:>12}" for lb in E_LABELS))
for key, lbl, *_ in METHODS:
    row = []
    for bi in range(len(E_LABELS)):
        mask = (e_bin == bi) & np.isfinite(arr[key])
        row.append(f"{np.median(arr[key][mask]):>12.2f}" if mask.sum() > 10 else f"{'N/A':>12}")
    print(f"  {lbl:<30}  {'  '.join(row)}")

# Ghost and hadronic fraction stats
gf = arr["ghost_pfrac"][np.isfinite(arr["ghost_pfrac"])]
hf = arr["had_pfrac"][np.isfinite(arr["had_pfrac"])]
lm = arr["lep_matched"]
print(f"\n  Median ghost momentum fraction: {np.median(gf):.3f}")
print(f"  Median hadronic momentum fraction: {np.median(hf):.3f}")
print(f"  Lepton matched by SPINE: {100*lm.mean():.1f}%")
print(f"  Events with no SPINE lepton found: {(arr['n_lep_found']==0).sum():,} "
      f"({100*(arr['n_lep_found']==0).mean():.1f}%)")

# ---------------------------------------------------------------
# Figure 1: Angular residual per method vs E_nu — 2×2
# ---------------------------------------------------------------
print("\n--- Figure 1: Angular residuals per method ---")

fig1, axes1 = plt.subplots(2, 2, figsize=(15, 11))
fig1.suptitle(
    r"Neutrino Direction Reconstruction: Decomposing the SPINE—Pandora Gap"
    "\nCC events — four direction estimators compared",
    fontsize=12, fontweight="bold")

# Panel [0,0]: violin per E_nu bin for all methods
ax = axes1[0, 0]
n_methods = len(METHODS)
positions_base = np.arange(len(E_LABELS)) * (n_methods + 1.5)
method_offsets = np.linspace(0, n_methods - 1, n_methods)

from matplotlib.patches import Patch
handles = []
for mi, (key, lbl, col, ls, mk) in enumerate(METHODS):
    positions = positions_base + method_offsets[mi]
    vdata = []
    for bi in range(len(E_LABELS)):
        mask = (e_bin == bi) & np.isfinite(arr[key])
        vals = np.clip(arr[key][mask], 0, 90)
        vdata.append(vals if len(vals) >= 5 else None)
    valid = [(d, p) for d, p in zip(vdata, positions) if d is not None]
    if valid:
        vd, vp = zip(*valid)
        vp2 = ax.violinplot(list(vd), positions=list(vp), widths=0.85,
                            showmedians=True, showextrema=False)
        for body in vp2["bodies"]:
            body.set_facecolor(col); body.set_alpha(0.35)
            body.set_edgecolor(col); body.set_linewidth(0.5)
            if ls == "--":
                body.set_linestyle("dashed")
        vp2["cmedians"].set_color(col); vp2["cmedians"].set_linewidth(2.0)
        if ls == "--":
            vp2["cmedians"].set_linestyle("--")
    handles.append(Patch(facecolor=col, alpha=0.6, label=lbl,
                         linestyle=ls if ls != "-" else "solid"))

xtick_pos = positions_base + method_offsets.mean()
ax.set_xticks(xtick_pos)
ax.set_xticklabels([f"{lb}\n(N={((e_bin==bi)).sum():,})"
                    for bi, lb in enumerate(E_LABELS)], fontsize=8)
ax.set_ylabel(r"$\Delta\theta$ [deg]")
ax.set_ylim(0, 90)
ax.set_title("Angular residual distribution by $E_\\nu$ bin")
ax.legend(handles=handles, fontsize=8, loc="upper right")

# Panel [0,1]: Median residual vs E_nu as line plot (cleaner comparison)
ax = axes1[0, 1]
nu_E_bins = np.linspace(0, 4.0, 20)
centers = 0.5 * (nu_E_bins[:-1] + nu_E_bins[1:])
from scipy.stats import binned_statistic as _bstat

for key, lbl, col, ls, mk in METHODS:
    mask = np.isfinite(arr[key]) & np.isfinite(arr["nu_E"]) & (arr["nu_E"] < 4.0)
    if mask.sum() < 10:
        continue
    med, _, _ = _bstat(arr["nu_E"][mask], arr[key][mask], "median", bins=nu_E_bins)
    q25, _, _ = _bstat(arr["nu_E"][mask], arr[key][mask],
                       lambda x: np.percentile(x, 25) if len(x) else _nan, bins=nu_E_bins)
    q75, _, _ = _bstat(arr["nu_E"][mask], arr[key][mask],
                       lambda x: np.percentile(x, 75) if len(x) else _nan, bins=nu_E_bins)
    cnt, _, _ = _bstat(arr["nu_E"][mask], arr[key][mask], "count", bins=nu_E_bins)
    mk2 = cnt > 20
    ax.plot(centers[mk2], med[mk2], color=col, ls=ls, marker=mk, ms=5, lw=1.8, label=lbl)
    ax.fill_between(centers[mk2],
                    np.where(np.isfinite(q25[mk2]), q25[mk2], med[mk2]),
                    np.where(np.isfinite(q75[mk2]), q75[mk2], med[mk2]),
                    color=col, alpha=0.08)

ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel(r"Median $\Delta\theta$ [deg]")
ax.set_title(r"Median angular residual vs $E_\nu$")
ax.set_ylim(0, 70)
ax.legend(fontsize=8, ncol=2)

# Panel [1,0]: Lepton-to-sum rotation vs had_pfrac
# "How many degrees does the hadronic system rotate the sum away from the lepton?"
ax = axes1[1, 0]
mk_lts = (np.isfinite(arr["lep_to_sum_ang"]) & np.isfinite(arr["had_pfrac"])
          & (arr["n_lep_found"] > 0))
hf_bins = np.linspace(0, 1, 20)
c_hf = 0.5 * (hf_bins[:-1] + hf_bins[1:])
med_lts, _, _ = _bstat(arr["had_pfrac"][mk_lts], arr["lep_to_sum_ang"][mk_lts],
                        "median", bins=hf_bins)
cnt_hf, _, _  = _bstat(arr["had_pfrac"][mk_lts], arr["lep_to_sum_ang"][mk_lts],
                        "count", bins=hf_bins)
mk_hf = cnt_hf > 10
ax.plot(c_hf[mk_hf], med_lts[mk_hf], color="#9467bd", lw=2, marker="o", ms=5,
        label="Median rotation")

# Reference: how much does this cost in final direction?
# Color events by E_nu bin
for bi, (col_b, lb) in enumerate(zip(["#2ca02c","#1f77b4","#ff7f0e","#d62728"], E_LABELS)):
    mask_b = mk_lts & (e_bin == bi)
    ax.scatter(arr["had_pfrac"][mask_b], arr["lep_to_sum_ang"][mask_b],
               s=1.0, alpha=0.06, color=col_b)

ax.set_xlabel("Hadronic momentum fraction  $|\\vec{p}_\\mathrm{had}| / |\\vec{p}_\\mathrm{all}|$")
ax.set_ylabel(r"Angle(lep-only, all-sum) [deg]")
ax.set_title("Hadronic system rotation of the momentum sum")
ax.set_xlim(0, 1); ax.set_ylim(0, 90)
ax.legend(fontsize=9)

from matplotlib.patches import Patch as _P
handles2 = [_P(facecolor=c, alpha=0.6, label=lb.split("\n")[0])
            for c, lb in zip(["#2ca02c","#1f77b4","#ff7f0e","#d62728"], E_LABELS)]
ax.legend(handles=[ax.lines[0]] + handles2,
          labels=["Median rotation"] + [lb.split("\n")[0] for lb in E_LABELS],
          fontsize=8, markerscale=2)

# Panel [1,1]: Angular residual vs ghost momentum fraction
ax = axes1[1, 1]
gf_bins = np.linspace(0, 1, 20)
c_gf = 0.5 * (gf_bins[:-1] + gf_bins[1:])

for key, lbl, col, ls, mk2 in [
    ("spine_ang_sum", "SPINE all-sum", "#1f77b4", "-",  "o"),
    ("spine_ang_lep", "SPINE lep-only","#1f77b4", "--", "s"),
]:
    mk_gf = np.isfinite(arr[key]) & np.isfinite(arr["ghost_pfrac"])
    med_gf, _, _ = _bstat(arr["ghost_pfrac"][mk_gf], arr[key][mk_gf], "median", bins=gf_bins)
    cnt_gf, _, _ = _bstat(arr["ghost_pfrac"][mk_gf], arr[key][mk_gf], "count", bins=gf_bins)
    ok = cnt_gf > 10
    ax.plot(c_gf[ok], med_gf[ok], color=col, ls=ls, marker=mk2, ms=5, lw=1.8, label=lbl)

ax.set_xlabel("Ghost momentum fraction  $|\\vec{p}_\\mathrm{ghost}| / |\\vec{p}_\\mathrm{all}|$")
ax.set_ylabel(r"Median $\Delta\theta$ [deg]")
ax.set_title("Angular residual vs ghost momentum contamination")
ax.set_xlim(0, 1); ax.set_ylim(0, 90)
ax.legend(fontsize=9)
ax.axhline(float(np.median(arr["pand_ang"][np.isfinite(arr["pand_ang"])])),
           color="#ff7f0e", ls=":", lw=1.5, label="Pandora median")

fig1.tight_layout()
fig1.savefig(os.path.join(outdir, "direction_decomposition.png"), dpi=150)
plt.close(fig1)
print("  Saved direction_decomposition.png")

# ---------------------------------------------------------------
# Figure 2: Lepton-found vs not-found breakdown
# ---------------------------------------------------------------
print("\n--- Figure 2: Muon-found vs muon-not-found breakdown ---")

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle("Effect of SPINE Lepton-Finding on Neutrino Direction\n"
              "CC events (νμ and νe combined)", fontsize=12, fontweight="bold")

# Panel 0: fraction of events with no SPINE lepton, vs E_nu
ax = axes2[0]
no_lep_mask = (arr["n_lep_found"] == 0) & np.isfinite(arr["nu_E"]) & (arr["nu_E"] < 4.0)
frac_no_lep, _, _ = _bstat(arr["nu_E"][arr["nu_E"] < 4.0],
                            (arr["n_lep_found"][arr["nu_E"] < 4.0] == 0).astype(float),
                            "mean", bins=nu_E_bins)
cnt_el, _, _ = _bstat(arr["nu_E"][arr["nu_E"] < 4.0],
                      arr["nu_E"][arr["nu_E"] < 4.0], "count", bins=nu_E_bins)
ok = cnt_el > 20
ax.plot(centers[ok], frac_no_lep[ok], color="#d62728", lw=2, marker="o", ms=5)
ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel("Fraction of CC events")
ax.set_title("Events with no SPINE lepton found\n(all primary momentum from hadrons+ghosts)")
ax.set_ylim(0, 0.40)
ax.text(0.5, 0.92,
        f"Overall: {100*(arr['n_lep_found']==0).mean():.1f}% of CC events",
        transform=ax.transAxes, ha="center", fontsize=10,
        bbox=dict(fc="white", ec="gray", pad=3))

# Panel 1: angular residual for SPINE-sum, split by whether SPINE found a lepton
ax = axes2[1]
for found, lbl, col, ls in [(True, "SPINE found lepton", "#1f77b4", "-"),
                             (False, "SPINE missed lepton", "#d62728", "--")]:
    mask = (arr["n_lep_found"] > 0) == found
    for key2, lbl2, col2, ls2 in [
        ("spine_ang_sum", "all-sum", col, ls),
    ]:
        mk2 = mask & np.isfinite(arr[key2]) & (arr["nu_E"] < 4.0)
        med2, _, _ = _bstat(arr["nu_E"][mk2], arr[key2][mk2], "median", bins=nu_E_bins)
        cnt2, _, _ = _bstat(arr["nu_E"][mk2], arr[key2][mk2], "count", bins=nu_E_bins)
        ok2 = cnt2 > 10
        ax.plot(centers[ok2], med2[ok2], color=col2, ls=ls2, lw=2, marker="o", ms=5,
                label=lbl)

# Also show Pandora as reference
mk_p = np.isfinite(arr["pand_ang"]) & (arr["nu_E"] < 4.0)
med_p, _, _ = _bstat(arr["nu_E"][mk_p], arr["pand_ang"][mk_p], "median", bins=nu_E_bins)
cnt_p, _, _ = _bstat(arr["nu_E"][mk_p], arr["pand_ang"][mk_p], "count", bins=nu_E_bins)
ok_p = cnt_p > 10
ax.plot(centers[ok_p], med_p[ok_p], color="#ff7f0e", ls="-", lw=2, marker="D", ms=5,
        label="Pandora lngtrk/heshw")

ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel(r"Median $\Delta\theta$ [deg]")
ax.set_title("SPINE all-sum direction residual:\nlepton found vs not found")
ax.set_ylim(0, 90)
ax.legend(fontsize=9)

# Panel 2: if lepton found, compare spine_sum vs spine_lep
ax = axes2[2]
found_mask = (arr["n_lep_found"] > 0)
for key, lbl, col, ls in [
    ("spine_ang_sum", "SPINE all-sum (found lep events)", "#1f77b4", "-"),
    ("spine_ang_lep", "SPINE lep-only (same events)",     "#1f77b4", "--"),
    ("pand_ang",      "Pandora lngtrk/heshw",             "#ff7f0e", "-"),
]:
    mk3 = found_mask & np.isfinite(arr[key]) & (arr["nu_E"] < 4.0)
    if mk3.sum() < 10:
        continue
    med3, _, _ = _bstat(arr["nu_E"][mk3], arr[key][mk3], "median", bins=nu_E_bins)
    cnt3, _, _ = _bstat(arr["nu_E"][mk3], arr[key][mk3], "count", bins=nu_E_bins)
    ok3 = cnt3 > 10
    ax.plot(centers[ok3], med3[ok3], color=col, ls=ls, lw=2, marker="o", ms=5, label=lbl)

ax.set_xlabel(r"True $E_\nu$ [GeV]")
ax.set_ylabel(r"Median $\Delta\theta$ [deg]")
ax.set_title("When SPINE finds the lepton:\nall-sum vs lep-only vs Pandora")
ax.set_ylim(0, 60)
ax.legend(fontsize=9)
ax.text(0.02, 0.96,
        "On events where SPINE reconstructed\na lepton primary",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(fc="lightyellow", ec="gray", pad=3))

fig2.tight_layout()
fig2.savefig(os.path.join(outdir, "direction_decomposition_muon.png"), dpi=150)
plt.close(fig2)
print("  Saved direction_decomposition_muon.png")

# ---------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------
summary = {"n_cc": n_cc}
for bi, lb in enumerate(E_LABELS):
    key_lb = lb.replace("\n", " ").replace(" ", "_")
    bm = e_bin == bi
    entry = {"n": int(bm.sum())}
    for key, lbl, *_ in METHODS:
        mk4 = bm & np.isfinite(arr[key])
        entry[lbl.replace(" ", "_")] = round(float(np.median(arr[key][mk4])), 2) \
                                       if mk4.sum() > 0 else None
    summary[key_lb] = entry

summary["overall"] = {}
for key, lbl, *_ in METHODS:
    mk5 = np.isfinite(arr[key])
    summary["overall"][lbl.replace(" ", "_")] = \
        round(float(np.median(arr[key][mk5])), 2) if mk5.sum() > 0 else None

summary["ghost_pfrac_median"]  = round(float(np.median(gf)), 4)
summary["had_pfrac_median"]    = round(float(np.median(hf)), 4)
summary["lep_found_fraction"]  = round(float(lm.mean()), 4)
summary["no_lep_fraction"]     = round(float((arr["n_lep_found"] == 0).mean()), 4)

with open(os.path.join(outdir, "direction_decomposition.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved direction_decomposition.json")
print("\nDone!")

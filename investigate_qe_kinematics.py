#!/usr/bin/env python3
"""
Study E: QE Kinematic Closure Test.

For QE CC events (nu + n -> l + p), the SPINE-reconstructed lepton and proton
should vector-sum to reproduce the neutrino 4-momentum. Deviations measure the
cumulative effect of Fermi motion, missing energy, and reco smearing.

Output: figures/qe_kinematic_closure.png, figures/qe_closure_summary.json
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

_nan = float("nan")
outdir = "figures"
os.makedirs(outdir, exist_ok=True)

PDG_MASS = {
    11: 0.000511, -11: 0.000511,
    13: 0.10566,  -13: 0.10566,
    2212: 0.93827,
    211: 0.13957, -211: 0.13957,
    111: 0.13498,
    22: 0.0,
    2112: 0.93957,
}

# EM calibration correction for SPINE (see analysis_per_particle.py)
EM_PDGS = {11, -11, 22, 111}
EM_CALIB = 0.198  # ADC→e⁻ correction factor for EM particles

LEPTON_PDGS = {11, -11, 13, -13}
PROTON_PDG  = 2212

def _vec_mag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def _vec_norm(x, y, z):
    n = _vec_mag(x, y, z)
    return (x/n, y/n, z/n) if n > 1e-9 else (_nan, _nan, _nan)

def _cos_angle(ax, ay, az, bx, by, bz):
    na = _vec_mag(ax, ay, az)
    nb = _vec_mag(bx, by, bz)
    if na < 1e-9 or nb < 1e-9:
        return _nan
    return max(-1.0, min(1.0, (ax*bx + ay*by + az*bz) / (na * nb)))

def _reco_E_gev(pdg, E_mev):
    m = PDG_MASS.get(abs(pdg), 0)
    if abs(pdg) in EM_PDGS:
        E_mev *= EM_CALIB
    return E_mev * 0.001 + m if E_mev > 0 else _nan

def _closest_dlp_idx(ixn, nu_vtx):
    best, best_i = float("inf"), 0
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = _vec_mag(v.x - nu_vtx.x, v.y - nu_vtx.y, v.z - nu_vtx.z)
        if d < best:
            best, best_i = d, i
    return best_i

print("=" * 60)
print("Study E: QE Kinematic Closure")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader("recTree", "spine_merged_reprocess.root")

records_nue = []
records_numu = []
n_qe_cc = 0
n_skipped = 0

for ientry in tqdm(range(prox.entries()), desc="QE scan"):
    rec = prox.entry(ientry)
    mc  = rec.mc
    if mc.nnu < 1:
        continue
    nu = mc.nu[0]
    if not nu.iscc:
        continue
    # nu.mode is an unregistered C++ enum — not accessible from pySRProxy.
    # Use a particle-content QE-like selection instead:
    # require exactly 1 reconstructed lepton + 1 reconstructed proton (no other significant particles).
    # (mode filtering applied post-hoc via particle content)
    pass  # selection applied below after SPINE particles are read

    n_qe_cc += 1  # total CC events seen
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        n_skipped += 1
        continue

    si = _closest_dlp_idx(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    nu_E = float(nu.E)
    nu_pdg = int(nu.pdg)

    # True neutrino direction
    tm = nu.momentum
    tnx, tny, tnz = _vec_norm(tm.x, tm.y, tm.z)

    # SPINE particles: find leading lepton and leading proton by reco momentum
    best_lep = {"p_mag": -1, "E": _nan, "px": _nan, "py": _nan, "pz": _nan}
    best_pro = {"p_mag": -1, "E": _nan, "px": _nan, "py": _nan, "pz": _nan}
    n_lep = 0
    n_pro = 0
    n_other = 0

    n_spine_parts = dlp.part.ndlp
    for pi in range(n_spine_parts):
        part = dlp.part.dlp[pi]
        pdg_v = int(part.pdg)
        p_v   = part.p
        px_v, py_v, pz_v = p_v.x, p_v.y, p_v.z  # MeV/c
        p_mag = _vec_mag(px_v, py_v, pz_v)
        E_v   = float(part.E)  # MeV (kinetic)
        if p_mag < 20:  # < 20 MeV/c — skip
            continue
        if abs(pdg_v) in LEPTON_PDGS:
            n_lep += 1
            if p_mag > best_lep["p_mag"]:
                best_lep = {"p_mag": p_mag, "E": _reco_E_gev(pdg_v, E_v),
                            "px": px_v * 0.001, "py": py_v * 0.001, "pz": pz_v * 0.001,
                            "pdg": pdg_v}
        elif pdg_v == PROTON_PDG:
            n_pro += 1
            if p_mag > best_pro["p_mag"]:
                best_pro = {"p_mag": p_mag, "E": _reco_E_gev(pdg_v, E_v),
                            "px": px_v * 0.001, "py": py_v * 0.001, "pz": pz_v * 0.001}
        else:
            n_other += 1

    # QE-like selection: exactly 1 lepton + 1 proton, no other significant particles
    if n_lep != 1 or n_pro != 1 or n_other > 0:
        n_skipped += 1
        continue

    if best_lep["p_mag"] < 0 or best_pro["p_mag"] < 0:
        n_skipped += 1
        continue

    # Energy balance
    E_lep = best_lep["E"]  # GeV
    E_pro = best_pro["E"]  # GeV (kinematic energy + mass)
    KE_pro = E_pro - PDG_MASS[2212]  # kinetic energy of proton

    if not (math.isfinite(E_lep) and math.isfinite(E_pro) and nu_E > 0.05):
        n_skipped += 1
        continue

    delta_E = (E_lep + KE_pro - nu_E) / nu_E  # fractional energy imbalance

    # Angular closure: angle between (p_lep + p_pro) and true nu direction
    sum_px = best_lep["px"] + best_pro["px"]
    sum_py = best_lep["py"] + best_pro["py"]
    sum_pz = best_lep["pz"] + best_pro["pz"]
    cos_c  = _cos_angle(sum_px, sum_py, sum_pz, tnx, tny, tnz)
    delta_theta = math.degrees(math.acos(cos_c)) if math.isfinite(cos_c) else _nan

    rec_d = {"nu_E": nu_E, "delta_E": delta_E, "delta_theta": delta_theta,
             "E_lep": E_lep, "KE_pro": KE_pro}

    if abs(nu_pdg) == 12:
        records_nue.append(rec_d)
    elif abs(nu_pdg) == 14:
        records_numu.append(rec_d)

print(f"  CC events scanned: {n_qe_cc:,}")
print(f"  QE-like (1lep+1p) selected: nue={len(records_nue):,}, numu={len(records_numu):,}")
print(f"  Skipped (no lep+p or multi-particle): {n_skipped:,}")

# ---------------------------------------------------------------
# Figure: QE kinematic closure
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(r"QE-like Kinematic Closure: SPINE $l + p$ vs True $\nu$ 4-momentum"
             "\n(1 lepton + 1 proton, no other particles)",
             fontsize=12, fontweight="bold")

for (flavor, records, col, lbl) in [
    (r"$\nu_e$ CC QE",  records_nue,  "#2ca02c", "nue"),
    (r"$\nu_\mu$ CC QE", records_numu, "#1f77b4", "numu"),
]:
    row = 0 if "nue" in lbl else 1
    if not records:
        continue

    delta_E     = np.array([r["delta_E"]     for r in records])
    delta_theta = np.array([r["delta_theta"]  for r in records])
    nu_E        = np.array([r["nu_E"]         for r in records])

    # Filter finite
    mk_E = np.isfinite(delta_E)
    mk_T = np.isfinite(delta_theta)

    # Panel (row, 0): Energy balance distribution
    ax = axes[row, 0]
    bins_e = np.linspace(-1, 1, 41)
    ax.hist(delta_E[mk_E], bins=bins_e, color=col, alpha=0.7, edgecolor="k", lw=0.4)
    med_e = float(np.median(delta_E[mk_E])) if mk_E.sum() > 0 else _nan
    std_e = float(np.std(delta_E[mk_E]))    if mk_E.sum() > 0 else _nan
    ax.axvline(med_e, color="k", ls="--", lw=1.5)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$(E_l + KE_p - E_\nu^{\rm true}) / E_\nu^{\rm true}$")
    ax.set_ylabel("Events")
    ax.set_title(f"{flavor}: Energy balance (N={mk_E.sum():,})")
    ax.text(0.97, 0.92, f"Median: {med_e:+.2f}\nStd: {std_e:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    # Panel (row, 1): Angular closure distribution
    ax = axes[row, 1]
    bins_t = np.linspace(0, 90, 46)
    ax.hist(delta_theta[mk_T], bins=bins_t, color=col, alpha=0.7, edgecolor="k", lw=0.4)
    med_t = float(np.median(delta_theta[mk_T])) if mk_T.sum() > 0 else _nan
    ax.axvline(med_t, color="k", ls="--", lw=1.5)
    ax.set_xlabel(r"Angular closure $\Delta\theta\ [\rm deg]$")
    ax.set_ylabel("Events")
    ax.set_title(f"{flavor}: Angular closure (N={mk_T.sum():,})")
    ax.text(0.97, 0.92, f"Median: {med_t:.1f} deg",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

fig.tight_layout()
fig.savefig(os.path.join(outdir, "qe_kinematic_closure.png"), dpi=150)
plt.close(fig)
print("  Saved qe_kinematic_closure.png")

def _safe_stat(arr, fn):
    arr = np.array(arr)
    arr = arr[np.isfinite(arr)]
    return float(fn(arr)) if len(arr) > 0 else None

summary = {
    "note": "QE-like selection: exactly 1 lepton + 1 proton, no other significant particles",
    "n_cc_scanned": n_qe_cc,
    "n_skipped": n_skipped,
    "nue": {
        "n": len(records_nue),
        "delta_E_median": _safe_stat([r["delta_E"] for r in records_nue], np.median),
        "delta_E_std": _safe_stat([r["delta_E"] for r in records_nue], np.std),
        "delta_theta_median_deg": _safe_stat([r["delta_theta"] for r in records_nue], np.median),
    },
    "numu": {
        "n": len(records_numu),
        "delta_E_median": _safe_stat([r["delta_E"] for r in records_numu], np.median),
        "delta_E_std": _safe_stat([r["delta_E"] for r in records_numu], np.std),
        "delta_theta_median_deg": _safe_stat([r["delta_theta"] for r in records_numu], np.median),
    },
}
with open(os.path.join(outdir, "qe_closure_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved qe_closure_summary.json")
print(f"  nue: delta_E median={summary['nue']['delta_E_median']}, "
      f"theta median={summary['nue']['delta_theta_median_deg']}")
print(f"  numu: delta_E median={summary['numu']['delta_E_median']}, "
      f"theta median={summary['numu']['delta_theta_median_deg']}")
print("Done!")

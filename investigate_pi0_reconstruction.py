#!/usr/bin/env python3
"""
Study F: Pi0 Photon Pair Reconstruction.

For events with a true pi0 primary, counts how many of the two decay photons
SPINE reconstructs (0 = fully missed, 1 = partial, 2 = both). Sub-100%
efficiency feeds directly into NC background mis-estimation and EM shower purity.

Because the two daughter photons are not stored as separate primaries, we use a
proxy: count SPINE gamma particles (PDG 22) with direction cos θ > 0.5 relative
to ANY truth primary. Also cross-checks the reco-gamma momentum sum vs true pi0 p.

Output: figures/pi0_photon_pairs.png, figures/pi0_summary.json
"""

import os, sys, ctypes, math, json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import caf_env
pySRProxy = caf_env.setup_pysrproxy()
import plot_style
plot_style.apply()

_nan = float("nan")
outdir = "figures"
os.makedirs(outdir, exist_ok=True)

EM_CALIB = 0.198  # ADC → e⁻ correction for EM particles

def _vec_mag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def _cos_angle(ax, ay, az, bx, by, bz):
    na = _vec_mag(ax, ay, az)
    nb = _vec_mag(bx, by, bz)
    if na < 1e-9 or nb < 1e-9:
        return _nan
    return max(-1.0, min(1.0, (ax*bx + ay*by + az*bz) / (na * nb)))

def _closest_dlp_idx(ixn, nu_vtx):
    best, best_i = float("inf"), 0
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = _vec_mag(v.x - nu_vtx.x, v.y - nu_vtx.y, v.z - nu_vtx.z)
        if d < best:
            best, best_i = d, i
    return best_i

COS_MATCH = 0.5   # cos θ threshold for photon-to-truth matching
P_MIN_MEV = 20.0  # minimum reco momentum to consider

print("=" * 60)
print("Study F: Pi0 Photon Pair Reconstruction")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader("recTree", caf_env.CAF_FILE)

records = []
n_pi0_events = 0
n_multi_pi0  = 0

for ientry in tqdm(range(prox.entries()), desc="Pi0 scan"):
    rec = prox.entry(ientry)
    mc  = rec.mc
    if mc.nnu < 1:
        continue
    nu = mc.nu[0]
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    # Find true pi0 primaries
    pi0_primaries = []
    truth_dirs    = []  # all truth primary directions (for matching)
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz  # GeV/c
        p_mag = _vec_mag(px, py, pz)
        if p_mag < 0.01:
            continue
        dx, dy, dz = px / p_mag, py / p_mag, pz / p_mag
        truth_dirs.append((dx, dy, dz, p_mag))
        if int(tp.pdg) == 111:
            pi0_primaries.append({"p_true": p_mag, "dx": dx, "dy": dy, "dz": dz})

    if not pi0_primaries:
        continue

    n_pi0_events += 1
    if len(pi0_primaries) > 1:
        n_multi_pi0 += 1

    # SPINE gamma particles (PDG 22) with p > threshold
    si  = _closest_dlp_idx(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    spine_gammas = []
    for pi in range(dlp.part.ndlp):
        part = dlp.part.dlp[pi]
        if int(part.pdg) != 22:
            continue
        p = part.p
        pmag = _vec_mag(p.x, p.y, p.z)
        if pmag < P_MIN_MEV:
            continue
        # Apply EM calibration (×0.198) to momentum magnitude, same as per-particle analysis.
        # SPINE calorimetric output for EM particles is ~5× the true value before this correction.
        p_mag_cal = pmag * EM_CALIB * 0.001  # MeV/c → GeV/c, EM-calibrated
        E_reco    = float(part.E) * EM_CALIB * 0.001  # kinetic energy in GeV, EM-calibrated
        spine_gammas.append({"p_mag": p_mag_cal,
                              "E_reco": E_reco,
                              "dx": p.x / pmag, "dy": p.y / pmag, "dz": p.z / pmag})

    # Count matched gammas: each SPINE gamma matched to at least one truth primary (cos>0.5)
    matched_gamma_idx = set()
    for gi, sg in enumerate(spine_gammas):
        for dx, dy, dz, _ in truth_dirs:
            c = _cos_angle(sg["dx"], sg["dy"], sg["dz"], dx, dy, dz)
            if math.isfinite(c) and c > COS_MATCH:
                matched_gamma_idx.add(gi)
                break

    n_matched = len(matched_gamma_idx)

    # Reco-gamma sum momentum vs true pi0 momentum (use first pi0)
    p_true_pi0 = pi0_primaries[0]["p_true"]  # GeV/c
    sum_px = sum(sg["dx"] * sg["p_mag"] for sg in spine_gammas)
    sum_py = sum(sg["dy"] * sg["p_mag"] for sg in spine_gammas)
    sum_pz = sum(sg["dz"] * sg["p_mag"] for sg in spine_gammas)
    p_reco_sum = _vec_mag(sum_px, sum_py, sum_pz)
    p_ratio    = p_reco_sum / p_true_pi0 if p_true_pi0 > 0.01 else _nan

    nu_mode_int = -1
    try:
        nu_mode_int = int(nu.mode)
    except Exception:
        pass
    mode_lbl = {0: "QE", 1: "RES", 2: "DIS", 3: "COH", 4: "MEC"}.get(nu_mode_int, "other")

    records.append({
        "n_spine_gamma": len(spine_gammas),
        "n_matched": n_matched,
        "n_pi0": len(pi0_primaries),
        "p_true_pi0": p_true_pi0,
        "p_reco_sum": p_reco_sum,
        "p_ratio": p_ratio,
        "nu_E": float(nu.E),
        "iscc": bool(nu.iscc),
        "mode": mode_lbl,
    })

print(f"  Events with true pi0: {n_pi0_events:,}  (multi-pi0: {n_multi_pi0:,})")

if not records:
    print("  No pi0 events found — aborting.")
    sys.exit(0)

n_mat   = np.array([r["n_matched"]     for r in records])
p_true  = np.array([r["p_true_pi0"]    for r in records])
p_ratio = np.array([r["p_ratio"]       for r in records])
n_spi_g = np.array([r["n_spine_gamma"] for r in records])

# ---------------------------------------------------------------
# Figure: Pi0 recovery bar + momentum cross-check
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(r"$\pi^0$ Photon Reconstruction: SPINE Gamma Recovery",
             fontsize=13, fontweight="bold")

# Panel a: bar chart of recovery (0/1/2+ matched gammas)
ax = axes[0, 0]
recovery_bins = [0, 1, 2, 3]
recovery_counts = [int((n_mat == k).sum()) for k in recovery_bins]
bars = ax.bar(recovery_bins, recovery_counts,
              color=["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"],
              alpha=0.8, edgecolor="k", linewidth=0.8)
total = sum(recovery_counts)
for k, (cnt, bar) in enumerate(zip(recovery_counts, bars)):
    if cnt > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f"{cnt}\n({100*cnt/total:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(recovery_bins)
ax.set_xticklabels(["0 gammas\n(fully missed)", "1 gamma\n(partial)", "2 gammas\n(both found)", "3+ gammas"])
ax.set_ylabel("Events")
ax.set_title(r"$\pi^0$ gamma recovery (N_matched reco gammas)")
ax.set_ylim(0, max(recovery_counts) * 1.25)

# Panel b: p_reco_sum / p_true (momentum fraction for events with ≥1 gamma)
ax = axes[0, 1]
mk_ratio = np.isfinite(p_ratio) & (n_mat >= 1)
if mk_ratio.sum() > 5:
    bins_r = np.linspace(0, 3, 41)
    ax.hist(p_ratio[mk_ratio], bins=bins_r, color="#1f77b4", alpha=0.7, edgecolor="k", lw=0.4)
    med_r = float(np.median(p_ratio[mk_ratio]))
    ax.axvline(med_r, color="k", ls="--", lw=1.5, label=f"Median={med_r:.2f}")
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.legend(fontsize=9)
ax.set_xlabel(r"$p_{\gamma,\rm sum}^{\rm reco} / p_{\pi^0}^{\rm true}$")
ax.set_ylabel("Events")
ax.set_title(r"Reco $\Sigma p_\gamma$ vs true $\pi^0$ momentum")

# Panel c: recovery rate vs true pi0 momentum (binned)
ax = axes[1, 0]
bins_p = np.linspace(0, 2, 11)
centers_p = 0.5 * (bins_p[:-1] + bins_p[1:])
from scipy.stats import binned_statistic as _bstat
rate_2, _, _ = _bstat(p_true, (n_mat >= 2).astype(float), statistic="mean", bins=bins_p)
rate_1, _, _ = _bstat(p_true, (n_mat >= 1).astype(float), statistic="mean", bins=bins_p)
cnt,    _, _ = _bstat(p_true, p_true, statistic="count", bins=bins_p)
mk = cnt >= 5
ax.plot(centers_p[mk], rate_2[mk], "o-", color="#2ca02c", lw=1.5, ms=5, label="Both gammas found")
ax.plot(centers_p[mk], rate_1[mk], "s--", color="#ff7f0e", lw=1.5, ms=5, label="≥1 gamma found")
ax.set_xlabel(r"True $\pi^0$ momentum [GeV/c]")
ax.set_ylabel("Recovery fraction")
ax.set_title(r"$\pi^0$ recovery rate vs true momentum")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.4)

# Panel d: recovery fraction by interaction mode
ax = axes[1, 1]
modes_arr = np.array([r["mode"] for r in records])
mode_order = ["QE", "RES", "MEC", "DIS", "COH", "other"]
mode_colors = {"QE": "#2ca02c", "RES": "#1f77b4", "MEC": "#ff7f0e",
               "DIS": "#d62728", "COH": "#9467bd", "other": "#8c564b"}
_x, _lbl, _rate0, _rate1, _rate2, _cols = [], [], [], [], [], []
for i, m in enumerate(mode_order):
    mk = modes_arr == m
    if mk.sum() < 5:
        continue
    _x.append(len(_x))
    _lbl.append(f"{m}\n(N={mk.sum():,})")
    _rate0.append(float((n_mat[mk] == 0).mean()))
    _rate1.append(float((n_mat[mk] == 1).mean()))
    _rate2.append(float((n_mat[mk] >= 2).mean()))
    _cols.append(mode_colors[m])
xpos = np.arange(len(_x))
bar_w = 0.28
ax.bar(xpos - bar_w, _rate0, width=bar_w, color="#d62728", alpha=0.8, label="0 gammas")
ax.bar(xpos,          _rate1, width=bar_w, color="#ff7f0e", alpha=0.8, label="1 gamma")
ax.bar(xpos + bar_w, _rate2, width=bar_w, color="#2ca02c", alpha=0.8, label="≥2 gammas")
ax.set_xticks(xpos); ax.set_xticklabels(_lbl, fontsize=9)
ax.set_ylabel("Fraction of pi0 events")
ax.set_title(r"$\pi^0$ recovery fraction by interaction mode")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)

fig.tight_layout()
fig.savefig(os.path.join(outdir, "pi0_photon_pairs.png"), dpi=150)
plt.close(fig)
print("  Saved pi0_photon_pairs.png")

# JSON summary
rate_0_gamma = float((n_mat == 0).mean())
rate_1_gamma = float((n_mat == 1).mean())
rate_2_gamma = float((n_mat >= 2).mean())
summary = {
    "n_pi0_events": len(records),
    "n_multi_pi0": n_multi_pi0,
    "rate_0_gamma": rate_0_gamma,
    "rate_1_gamma": rate_1_gamma,
    "rate_2plus_gamma": rate_2_gamma,
    "p_ratio_median_for_ge1": float(np.median(p_ratio[mk_ratio])) if mk_ratio.sum() > 0 else None,
}
with open(os.path.join(outdir, "pi0_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Pi0 recovery: 0 gamma={rate_0_gamma:.1%}, 1 gamma={rate_1_gamma:.1%}, "
      f">=2 gammas={rate_2_gamma:.1%}")
print("  Saved pi0_summary.json")
print("Done!")

#!/usr/bin/env python3
"""
Particle-level PID confusion — CC interactions only, full sample.

For each geometrically matched (truth→SPINE / truth→Pandora) primary particle,
compares the reconstructed PDG category to the truth PDG category.

Extends Figure 10 in analysis_per_particle.py (5k CC subsample, SPINE only)
to the full CC sample with side-by-side SPINE vs Pandora comparison.

Output: figures/particle_pid_cc.png, figures/particle_pid_cc_summary.json
"""

import os, sys, ctypes, math, json
import numpy as np
import matplotlib.pyplot as plt
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
COS_THRESH   = 0.5   # geometric matching threshold (same as analysis_per_particle.py)
P_MIN_TRUTH  = 0.02  # GeV/c — minimum truth primary momentum

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

def _closest_dlp(ixn, nu_vtx):
    best, bi = float("inf"), 0
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = math.sqrt((v.x - nu_vtx.x)**2 + (v.y - nu_vtx.y)**2 + (v.z - nu_vtx.z)**2)
        if d < best:
            best, bi = d, i
    return bi

# PDG → display category (same mapping as analysis_per_particle.py)
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

cat_order = ["e", "mu", "gamma", "pi", "p"]
ncat = len(cat_order)
cat_set = set(cat_order)   # fast membership test

# ---------------------------------------------------------------
# Scan
# ---------------------------------------------------------------
print("=" * 60)
print("Particle-level PID — CC events only, full sample")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader("recTree", "spine_merged_reprocess.root")

spine_pairs   = []  # list of (true_pdg_int, reco_pdg_int)
pandora_pairs = []
n_cc = 0

for ientry in tqdm(range(prox.entries()), desc="PID scan"):
    rec = prox.entry(ientry)
    mc  = rec.mc
    if mc.nnu < 1 or not mc.nu[0].iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu = mc.nu[0]
    n_cc += 1

    # Truth primaries with p > P_MIN_TRUTH, restricted to the 5 reco categories
    truth = []
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        if pdg_cat(int(tp.pdg)) not in cat_set:
            continue
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        if _vmag(px, py, pz) < P_MIN_TRUTH:
            continue
        dx, dy, dz = _vnorm(px, py, pz)
        if math.isnan(dx):
            continue
        truth.append({"pdg": int(tp.pdg), "dx": dx, "dy": dy, "dz": dz})
    if not truth:
        continue

    si = _closest_dlp(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    # SPINE primary particles — restricted to the 5 reco categories
    spine = []
    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        if not p.primary:
            continue
        if pdg_cat(int(p.pdg)) not in cat_set:
            continue
        px, py, pz = p.p.x, p.p.y, p.p.z
        pmag = _vmag(px, py, pz)
        if not (pmag > 1):  # MeV/c — NaN-safe
            continue
        dx, dy, dz = _vnorm(px, py, pz)
        if math.isnan(dx):
            continue
        spine.append({"pdg": int(p.pdg), "dx": dx, "dy": dy, "dz": dz})

    # Greedy truth→SPINE matching
    used_s = set()
    for tp in truth:
        best_cos, best_i = -2.0, -1
        for si2, sp in enumerate(spine):
            if si2 in used_s:
                continue
            c = _cos(tp["dx"], tp["dy"], tp["dz"], sp["dx"], sp["dy"], sp["dz"])
            if c > best_cos:
                best_cos, best_i = c, si2
        if best_cos > COS_THRESH and best_i >= 0:
            spine_pairs.append((tp["pdg"], spine[best_i]["pdg"]))
            used_s.add(best_i)

    # Pandora primary particles
    has_pand = (hasattr(ixn, "pandora") and ixn.npandora > 0 and
                hasattr(rec, "fd") and hasattr(rec.fd, "hd") and
                hasattr(rec.fd.hd, "pandora"))
    if not has_pand:
        continue
    try:
        pand_fd    = rec.fd.hd.pandora[0]
        pand_parts = ixn.pandora[0].part
        npand      = pand_parts.npandora - 1  # skip fake last particle
    except Exception:
        continue

    pandora = []
    for pi in range(npand):
        p = pand_parts.pandora[pi]
        if not p.primary:
            continue
        if pdg_cat(int(p.pdg)) not in cat_set:
            continue
        sd  = pand_fd.showers[pi].direction
        sdn = math.sqrt(sd.x**2 + sd.y**2 + sd.z**2)
        if sdn > 1e-6:
            dx, dy, dz = sd.x/sdn, sd.y/sdn, sd.z/sdn
        else:
            td  = pand_fd.tracks[pi].dir
            tdn = math.sqrt(td.x**2 + td.y**2 + td.z**2)
            if tdn < 1e-6:
                continue
            dx, dy, dz = td.x/tdn, td.y/tdn, td.z/tdn
        pandora.append({"pdg": int(p.pdg), "dx": dx, "dy": dy, "dz": dz})

    # Greedy truth→Pandora matching
    used_p = set()
    for tp in truth:
        best_cos, best_i = -2.0, -1
        for pi2, pp in enumerate(pandora):
            if pi2 in used_p:
                continue
            c = _cos(tp["dx"], tp["dy"], tp["dz"], pp["dx"], pp["dy"], pp["dz"])
            if c > best_cos:
                best_cos, best_i = c, pi2
        if best_cos > COS_THRESH and best_i >= 0:
            pandora_pairs.append((tp["pdg"], pandora[best_i]["pdg"]))
            used_p.add(best_i)

print(f"\n  CC events scanned: {n_cc:,}")
print(f"  SPINE matched pairs:   {len(spine_pairs):,}")
print(f"  Pandora matched pairs: {len(pandora_pairs):,}")

# ---------------------------------------------------------------
# Build confusion matrices
# ---------------------------------------------------------------
def build_cm(pairs):
    cm = np.zeros((ncat, ncat), dtype=int)
    for tp, rp in pairs:
        ti = cat_order.index(pdg_cat(tp))
        ri = cat_order.index(pdg_cat(rp))
        cm[ti, ri] += 1
    return cm

cm_s      = build_cm(spine_pairs)
cm_p      = build_cm(pandora_pairs)
cm_s_norm = cm_s.astype(float) / cm_s.sum(axis=1, keepdims=True).clip(min=1)
cm_p_norm = cm_p.astype(float) / cm_p.sum(axis=1, keepdims=True).clip(min=1)

# ---------------------------------------------------------------
# Figure: 2×2 layout
#   [0,0] SPINE confusion   [0,1] Pandora confusion
#   [1,0-1]  diagonal efficiency bar chart (merged)
# ---------------------------------------------------------------
fig = plt.figure(figsize=(16, 13))
fig.suptitle("Particle-level PID Confusion — CC Interactions Only, Full Sample\n"
             "(restricted to reco-used categories: e, μ, γ, π, p)",
             fontsize=13, fontweight="bold")

ax_s  = fig.add_subplot(2, 2, 1)   # SPINE confusion
ax_p  = fig.add_subplot(2, 2, 2)   # Pandora confusion
ax_b  = fig.add_subplot(2, 1, 2)   # efficiency bar chart (full bottom row)

# Confusion panels
for ax, cm, cm_norm, title, cmap in [
    (ax_s, cm_s, cm_s_norm, f"SPINE  (N = {len(spine_pairs):,} matched pairs)", "Blues"),
    (ax_p, cm_p, cm_p_norm, f"Pandora  (N = {len(pandora_pairs):,} matched pairs)", "Oranges"),
]:
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(ncat)); ax.set_xticklabels(cat_order, rotation=40, ha="right")
    ax.set_yticks(range(ncat)); ax.set_yticklabels(cat_order)
    ax.set_xlabel("Reconstructed PDG"); ax.set_ylabel("True PDG")
    ax.set_title(title)
    for i in range(ncat):
        for j in range(ncat):
            if cm[i, j] == 0:
                continue
            c = "white" if cm_norm[i, j] > 0.40 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]:,})",
                    ha="center", va="center", fontsize=9, color=c, fontweight="bold")
        # Color-code truth labels to match particle color scheme
        col = PARTICLE_COLORS.get(cat_order[i], "black")
        ax.get_yticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_fontweight("bold")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised fraction")

# Diagonal efficiency bar chart
diag_s = [float(cm_s_norm[i, i]) for i in range(ncat)]
diag_p = [float(cm_p_norm[i, i]) for i in range(ncat)]
n_truth_s = [int(cm_s[i, :].sum()) for i in range(ncat)]
n_truth_p = [int(cm_p[i, :].sum()) for i in range(ncat)]

x = np.arange(ncat)
w = 0.35
bars_s = ax_b.bar(x - w/2, diag_s, width=w, label="SPINE",   color="#1f77b4", alpha=0.85, edgecolor="k", lw=0.5)
bars_p = ax_b.bar(x + w/2, diag_p, width=w, label="Pandora", color="#ff7f0e", alpha=0.85, edgecolor="k", lw=0.5)

for xi, (vs, vp, ns, np_) in enumerate(zip(diag_s, diag_p, n_truth_s, n_truth_p)):
    if ns > 0:
        ax_b.text(xi - w/2, vs + 0.012, f"{vs:.2f}", ha="center", va="bottom", fontsize=7.5, color="#1f77b4")
    if np_ > 0:
        ax_b.text(xi + w/2, vp + 0.012, f"{vp:.2f}", ha="center", va="bottom", fontsize=7.5, color="#d65f00")
    # N truth below x-axis
    if ns > 0:
        ax_b.text(xi - w/2, -0.04, f"N={ns:,}", ha="center", va="top", fontsize=6, color="#555555", rotation=0)

ax_b.set_xticks(x)
ax_b.set_xticklabels(cat_order, fontsize=11)
ax_b.set_ylabel("PID efficiency (diagonal fraction)")
ax_b.set_title("Per-type PID efficiency: SPINE vs Pandora (CC events, matched primaries, cosθ > 0.5)")
ax_b.set_ylim(-0.08, 1.12)
ax_b.axhline(1.0, color="k", ls=":", lw=0.7, alpha=0.4)
ax_b.legend(fontsize=11)

# Color-code x-axis tick labels
for i, tick in enumerate(ax_b.get_xticklabels()):
    tick.set_color(PARTICLE_COLORS.get(cat_order[i], "black"))
    tick.set_fontweight("bold")

fig.tight_layout()
fig.savefig(os.path.join(outdir, "particle_pid_cc.png"), dpi=150)
plt.close(fig)
print("  Saved particle_pid_cc.png")

# ---------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------
print(f"\n  {'Type':>8}  {'N(truth/S)':>11}  {'SPINE eff':>10}  {'N(truth/P)':>11}  {'Pandora eff':>12}")
for i, cat in enumerate(cat_order):
    ns = cm_s[i, :].sum()
    np_ = cm_p[i, :].sum()
    se = cm_s_norm[i, i]
    pe = cm_p_norm[i, i]
    print(f"  {cat:>8}  {ns:>11,}  {se:>10.3f}  {np_:>11,}  {pe:>12.3f}")

# ---------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------
summary = {
    "n_cc_events": n_cc,
    "n_spine_matched_pairs": len(spine_pairs),
    "n_pandora_matched_pairs": len(pandora_pairs),
    "cos_threshold": COS_THRESH,
    "p_min_truth_gev": P_MIN_TRUTH,
    "spine_pid_efficiency": {cat: round(float(cm_s_norm[i, i]), 4) for i, cat in enumerate(cat_order)},
    "pandora_pid_efficiency": {cat: round(float(cm_p_norm[i, i]), 4) for i, cat in enumerate(cat_order)},
    "spine_n_truth_per_type": {cat: int(cm_s[i, :].sum()) for i, cat in enumerate(cat_order)},
    "pandora_n_truth_per_type": {cat: int(cm_p[i, :].sum()) for i, cat in enumerate(cat_order)},
}

with open(os.path.join(outdir, "particle_pid_cc_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone!")

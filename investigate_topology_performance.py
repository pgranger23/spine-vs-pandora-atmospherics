#!/usr/bin/env python3
"""
Study B: Reconstruction performance stratified by interaction mode (QE/RES/DIS/MEC).

Computes vertex resolution, angular resolution, and energy bias per event and
groups by neutrino interaction topology to reveal how event complexity affects
reconstruction quality.

Output: figures/topology_performance.png, figures/topology_performance.json
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

outdir = "figures"
_nan = float("nan")

# nu.mode is a caf::ScatteringMode enum not registered in pySRProxy → inaccessible.
# Stratify instead by true neutrino energy, which correlates strongly with topology:
#   E < 0.5 GeV  → dominantly QE-like
#   0.5–1.0 GeV  → RES-dominated
#   1.0–2.0 GeV  → MEC/RES/DIS mix
#   > 2.0 GeV    → DIS-dominated
E_BINS   = [0.0, 0.5, 1.0, 2.0, 1e9]
E_LABELS = ["E<0.5 GeV\n(QE-like)", "0.5-1 GeV\n(RES-like)",
            "1-2 GeV\n(MEC/DIS)", ">2 GeV\n(DIS-like)"]
E_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

def _e_bin(e):
    for i in range(len(E_BINS) - 1):
        if E_BINS[i] <= e < E_BINS[i + 1]:
            return i
    return len(E_BINS) - 2

def _safe_norm(x, y, z):
    n = math.sqrt(x*x + y*y + z*z)
    return (x/n, y/n, z/n) if n > 1e-9 else (_nan, _nan, _nan)

def _cos_angle(ax, ay, az, bx, by, bz):
    na = math.sqrt(ax*ax + ay*ay + az*az)
    nb = math.sqrt(bx*bx + by*by + bz*bz)
    # Guard against zero, NaN, and inf — NaN < 1e-9 is False in Python, so must check isfinite
    if not (math.isfinite(na) and na > 1e-9) or not (math.isfinite(nb) and nb > 1e-9):
        return _nan
    return max(-1.0, min(1.0, (ax*bx + ay*by + az*bz) / (na * nb)))

def _closest_idx(ixn_items, n_items, nu_vtx):
    best, best_i = float("inf"), 0
    for i in range(n_items):
        v = ixn_items[i].vtx
        d = math.sqrt((v.x - nu_vtx.x)**2 + (v.y - nu_vtx.y)**2 + (v.z - nu_vtx.z)**2)
        if d < best:
            best, best_i = d, i
    return best_i

# ---------------------------------------------------------------
print("=" * 60)
print("Study B: Topology-stratified reconstruction performance")
print("=" * 60)

prox = pySRProxy.StandardRecordFileReader("recTree", "spine_merged_reprocess.root")

records = []

for ientry in tqdm(range(prox.entries()), desc="Scanning"):
    rec = prox.entry(ientry)
    mc  = rec.mc
    if mc.nnu < 1 or not mc.nu[0].iscc:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    nu = mc.nu[0]
    nu_pdg = int(nu.pdg)
    nu_E = float(nu.E)
    e_bin = _e_bin(nu_E)

    # True direction
    tm = nu.momentum
    tdx, tdy, tdz = _safe_norm(tm.x, tm.y, tm.z)
    tvx, tvy, tvz = nu.vtx.x, nu.vtx.y, nu.vtx.z

    # --- SPINE ---
    si = _closest_idx(ixn.dlp, ixn.ndlp, nu.vtx)
    dlp = ixn.dlp[si]
    sv = dlp.vtx
    spine_dvtx = math.sqrt((sv.x-tvx)**2 + (sv.y-tvy)**2 + (sv.z-tvz)**2) * 0.1  # mm → cm

    # SPINE direction: manual sum of reconstructed particle momenta
    # dlp.dir.part_mom_sum is NOT populated in this CAF (returns zeros/NaN)
    # EM particles (e, γ, π⁰) carry raw ADC-scale momentum ~5× too large;
    # apply the ADC-to-electron calibration (×0.198) before summing.
    _spx, _spy, _spz = 0., 0., 0.
    for _pi in range(dlp.part.ndlp):
        _p  = dlp.part.dlp[_pi]
        _pp = _p.p
        _sc = 0.198 if abs(int(_p.pdg)) in (11, 22, 111) else 1.0
        _spx += _pp.x * _sc; _spy += _pp.y * _sc; _spz += _pp.z * _sc
    spine_cos = _cos_angle(_spx, _spy, _spz, tdx, tdy, tdz)
    spine_ang = math.degrees(math.acos(spine_cos)) if math.isfinite(spine_cos) else _nan

    # --- Pandora ---
    pand_dvtx = _nan
    pand_ang   = _nan
    pand_Eratio = _nan

    if ixn.npandora > 0:
        pi = _closest_idx(ixn.pandora, ixn.npandora, nu.vtx)
        pan = ixn.pandora[pi]
        pv = pan.vtx
        pand_dvtx = math.sqrt((pv.x-tvx)**2 + (pv.y-tvy)**2 + (pv.z-tvz)**2) * 0.1
        pd = pan.dir
        # Flavor-appropriate direction
        if abs(nu_pdg) == 14:
            px, py, pz = pd.lngtrk.x, pd.lngtrk.y, pd.lngtrk.z
        else:
            px, py, pz = pd.heshw.x, pd.heshw.y, pd.heshw.z
        cos_p = _cos_angle(px, py, pz, tdx, tdy, tdz)
        pand_ang = math.degrees(math.acos(cos_p)) if math.isfinite(cos_p) else _nan
        pe = pan.Enu.calo
        pand_Eratio = float(pe / nu_E) if (math.isfinite(pe) and nu_E > 0) else _nan

    records.append({
        "e_bin": e_bin, "nu_pdg": nu_pdg, "nu_E": nu_E,
        "spine_dvtx": spine_dvtx, "spine_ang": spine_ang,
        "pand_dvtx": pand_dvtx, "pand_ang": pand_ang, "pand_Eratio": pand_Eratio,
    })

print(f"  Events loaded: {len(records):,}")

# Assemble per-energy-bin arrays
n_bins = len(E_LABELS)
bin_data = [{"spine_dvtx": [], "spine_ang": [], "pand_dvtx": [],
             "pand_ang": [], "pand_Eratio": [], "n": 0}
            for _ in range(n_bins)]

for r in records:
    bd = bin_data[r["e_bin"]]
    bd["n"] += 1
    for k in ["spine_dvtx", "spine_ang", "pand_dvtx", "pand_ang", "pand_Eratio"]:
        v = r[k]
        if math.isfinite(v):
            bd[k].append(v)

# ---------------------------------------------------------------
# Figure: 3-metric violin comparison by Enu bin (topology proxy)
# ---------------------------------------------------------------
print("\n--- Figure: Energy-stratified performance ---")
for i, bd in enumerate(bin_data):
    print(f"  {E_LABELS[i]!r}: N={bd['n']:,}")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(r"Reconstruction Performance by $E_\nu$ Bin (Topology Proxy) — SPINE vs Pandora",
             fontsize=13, fontweight="bold")

metrics = [
    ("spine_dvtx", "pand_dvtx",   r"Vertex 3D $\Delta r$ [cm]",     (0, 5),   axes[0]),
    ("spine_ang",  "pand_ang",    r"Angular residual $\Delta\theta$ [deg]", (0, 90), axes[1]),
    (None,         "pand_Eratio", r"Pandora $E_{\mathrm{reco}}/E_{\mathrm{true}}$", (0, 2.5), axes[2]),
]

summary = {}
for s_key, p_key, ylabel, ylim, ax in metrics:
    positions_s = [i * 2.5      for i in range(n_bins)]
    positions_p = [i * 2.5 + 1  for i in range(n_bins)]
    xtick_pos   = [i * 2.5 + 0.5 for i in range(n_bins)]

    data_s, data_p = [], []
    for bd in bin_data:
        vals_s = np.array(bd[s_key]) if s_key else np.array([])
        vals_p = np.array(bd[p_key])
        _s = np.clip(vals_s, *ylim) if len(vals_s) >= 2 else None
        _p = np.clip(vals_p, *ylim) if len(vals_p) >= 2 else None
        data_s.append(_s)
        data_p.append(_p)

    def _do_violin(ax, datasets, positions, fc):
        valid = [(d, p) for d, p in zip(datasets, positions) if d is not None and len(d) >= 2]
        if not valid:
            return
        vd, vp = zip(*valid)
        vplot = ax.violinplot(list(vd), positions=list(vp), widths=0.8,
                              showmedians=True, showextrema=False)
        for body in vplot["bodies"]:
            body.set_facecolor(fc); body.set_alpha(0.6)
        vplot["cmedians"].set_color(fc); vplot["cmedians"].set_linewidth(2)

    if s_key:
        _do_violin(ax, data_s, positions_s, "#1f77b4")
    _do_violin(ax, data_p, positions_p, "#ff7f0e")

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f"{lbl}\n(N={bd['n']:,})" for lbl, bd in zip(E_LABELS, bin_data)],
                        fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(ylabel)

    from matplotlib.patches import Patch
    handles = []
    if s_key:
        handles.append(Patch(facecolor="#1f77b4", alpha=0.6, label="SPINE"))
    handles.append(Patch(facecolor="#ff7f0e", alpha=0.6, label="Pandora"))
    ax.legend(handles=handles, fontsize=9)

    for i, (lbl, bd, ds, dp) in enumerate(zip(E_LABELS, bin_data, data_s, data_p)):
        m_s = float(np.median(ds)) if (ds is not None and s_key) else None
        m_p = float(np.median(dp)) if dp is not None else float("nan")
        key = f"{s_key or p_key}_bin{i}"
        summary[key] = {"e_label": lbl.replace('\n', ' '), "metric": s_key or p_key,
                         "spine_median": m_s, "pandora_median": m_p, "n": bd["n"]}
        _s_str = f"{m_s:.2f}" if m_s is not None else "N/A"
        _p_str = f"{m_p:.2f}" if math.isfinite(m_p) else "N/A"
        print(f"  {lbl.split(chr(10))[0]:12s} | {s_key or p_key:20s}: SPINE={_s_str}, Pandora={_p_str}")

fig.tight_layout()
fig.savefig(os.path.join(outdir, "topology_performance.png"), dpi=150)
plt.close(fig)
print("  Saved topology_performance.png")

# Save JSON
out = {"n_events": len(records),
       "note": "nu.mode inaccessible (unregistered enum); stratified by E_nu bins instead",
       "e_bin_counts": {lbl.replace('\n', ' '): bd["n"]
                        for lbl, bd in zip(E_LABELS, bin_data)},
       "medians": summary}
with open(os.path.join(outdir, "topology_performance.json"), "w") as f:
    json.dump(out, f, indent=2)
print("  Saved topology_performance.json")
print("Done!")

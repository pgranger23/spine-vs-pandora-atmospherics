#!/usr/bin/env python3
"""
Compare reconstructed particle multiplicity (SPINE and Pandora) against
the number of 'reconstructible' truth particles per event.

Reconstructibility criteria (LArTPC-motivated):
  - Charged primaries: species-specific minimum momentum threshold
      μ  (±13)   : p > 50 MeV/c   (even slow muons leave tracks)
      e  (±11)   : p > 30 MeV/c   (produce showers)
      π± (±211)  : p > 100 MeV/c  (need ≳3 cm track)
      p  (2212)  : p > 250 MeV/c  (KE ~ 30 MeV ≈ 1 cm track)
      K± (±321, 310) : p > 100 MeV/c
  - Photons (22) : E > 100 MeV    (minimum EM shower)
  - π⁰  (111)   : NOT directly; replaced by 2γ daughters:
        count 2 if E_pi0 > 200 MeV (both gammas > 100 MeV each)
        count 1 if 100 < E_pi0 ≤ 200 MeV (one gamma above threshold)
        count 0 if E_pi0 ≤ 100 MeV
  - Neutrons (2112), K⁰_L (130), nuclear fragments: NOT reconstructible

Inelastic children (secondaries): the products of in-detector re-interactions
of pions/protons/neutrons live in nu.sec.  Most are either invisible (nuclear
fragments, thermal neutrons, sub-MeV γ) or far-displaced from the vertex
(median ~1 m — protons knocked out by propagating primary neutrons), and would
form SEPARATE reco objects rather than being clustered into this interaction.
We therefore count only NEAR-VERTEX reconstructible secondary hadrons
(p/π± above threshold, created within SEC_DISP_MAX of the vertex), i.e. prompt
re-interactions that SPINE/Pandora cluster into the interaction.  Far-displaced
neutron-knockout protons are left to the neutron-invisibility study.

Reads the CAF with PyROOT (TTree::GetEntry deserializes each entry once, O(n)
in particle count) — pySRProxy re-reads the unsplit nu.sec blob on every index
access, which is O(n²) and ~1600× slower for the secondary loop.

Output: figures/multiplicity_*.png
"""

import os, sys, math, subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import binned_statistic as _bstat
from tqdm import tqdm

# ── PyROOT setup ──────────────────────────────────────────────────────────────
base_dir = os.getcwd()
try:
    _root_libdir = subprocess.check_output(["root-config", "--libdir"]).decode().strip()
    if _root_libdir not in sys.path:
        sys.path.insert(0, _root_libdir)
except Exception:
    pass
import ROOT

_lib_path = os.path.join(base_dir, "duneanaobj/build/Linux/lib")
for _lib in ["libSRProxy_BasicTypes", "libduneanaobj_StandardRecord",
             "libduneanaobj_StandardRecordProxy"]:
    ROOT.gSystem.Load(os.path.join(_lib_path, _lib))

import plot_style
plot_style.apply()

outdir = "figures"
os.makedirs(outdir, exist_ok=True)
_nan = float("nan")

# ── reconstructibility thresholds [GeV/c for charged, GeV for γ] ──────────────
THRESH = {
    11:   0.030,   # electron
    13:   0.050,   # muon
    211:  0.100,   # charged pion
    321:  0.100,   # charged kaon
    310:  0.100,   # K0S (charged decay products visible)
    2212: 0.250,   # proton
    22:   0.100,   # photon (energy ≈ momentum)
}
M_PI0  = 0.1350   # GeV/c²

COS_MATCH    = 0.5
P_SPINE_MIN  = 0.5    # MeV/c — minimum SPINE particle momentum
SEC_DISP_MAX = 5.0    # cm — max vertex displacement for a "near-vertex" secondary

# ── helpers ───────────────────────────────────────────────────────────────────
def vmag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def cos_angle(ax, ay, az, bx, by, bz):
    na = vmag(ax, ay, az)
    nb = vmag(bx, by, bz)
    if not (math.isfinite(na) and na > 1e-9) or not (math.isfinite(nb) and nb > 1e-9):
        return _nan
    return max(-1.0, min(1.0, (ax*bx + ay*by + az*bz) / (na*nb)))

def n_reconstructible(nu):
    """Return (n_near, n_full, dirs) for a truth neutrino.

    n_near = reconstructible primaries + near-vertex reconstructible secondary
             hadrons (within SEC_DISP_MAX; prompt re-interactions that a single
             interaction object clusters).
    n_full = primaries + ALL reconstructible secondary hadrons at any displacement
             (the full visible-track content, incl. displaced hadronic-interaction
             products that would form separate reco objects — see Study J).
    dirs   = directions of the near-vertex set (for direction matching)."""
    n_prim = 0
    dirs = []

    # ── Primaries ──────────────────────────────────────────────────────────────
    for ti in range(nu.nprim):
        tp   = nu.prim[ti]
        pdga = abs(int(tp.pdg))
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        p = vmag(px, py, pz)   # GeV/c

        if pdga == 111:     # π⁰ → 2γ replacement (validated: ~73% show both γ in sec)
            E_pi0 = math.sqrt(p*p + M_PI0*M_PI0)
            if E_pi0 > 0.200:
                n_prim += 2
                if p > 1e-9:
                    dirs += [(px/p, py/p, pz/p), (px/p, py/p, pz/p)]
            elif E_pi0 > 0.100:
                n_prim += 1
                if p > 1e-9:
                    dirs.append((px/p, py/p, pz/p))
        elif pdga == 2112 or pdga == 130 or pdga > 1000000000:
            pass  # neutron / K0L / nuclear fragment — not reconstructible
        elif pdga in THRESH:
            if p >= THRESH[pdga]:
                n_prim += 1
                if p > 1e-9:
                    dirs.append((px/p, py/p, pz/p))
        # anything else (ν, heavy hadrons not listed) — skip

    # ── Reconstructible secondary hadrons, split by displacement ────────────────
    n_sec_near = 0
    n_sec_full = 0
    vx, vy, vz = nu.vtx.x, nu.vtx.y, nu.vtx.z
    for si in range(int(nu.nsec)):
        sp   = nu.sec[si]
        pdga = abs(int(sp.pdg))
        if pdga != 2212 and pdga != 211:
            continue
        px, py, pz = sp.p.px, sp.p.py, sp.p.pz
        p = vmag(px, py, pz)
        if pdga == 2212 and p < THRESH[2212]:
            continue
        if pdga == 211 and p < THRESH[211]:
            continue
        n_sec_full += 1
        s = sp.start_pos
        if vmag(s.x - vx, s.y - vy, s.z - vz) <= SEC_DISP_MAX:
            n_sec_near += 1
            if p > 1e-9:
                dirs.append((px/p, py/p, pz/p))

    return n_prim + n_sec_near, n_prim + n_sec_full, dirs

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

# ── scan ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Particle multiplicity: reconstructible vs reco (SPINE & Pandora)")
print(f"Near-vertex secondary cut: SEC_DISP_MAX = {SEC_DISP_MAX} cm")
print("=" * 60)

f = ROOT.TFile.Open("spine_merged_reprocess.root")
t = f.Get("recTree")
N = t.GetEntries()
rows = []

for ientry in tqdm(range(N), desc="Scanning"):
    t.GetEntry(ientry)
    rec = t.rec
    mc  = rec.mc
    if mc.nnu < 1:
        continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1 and ixn.npandora < 1:
        continue

    nu    = mc.nu[0]
    nu_E  = float(nu.E)
    is_cc = bool(nu.iscc)
    nu_pdg = int(nu.pdg)

    # ── Truth reconstructible count (near-vertex and full) ─────────────────────
    n_recon, n_recon_full, truth_dirs = n_reconstructible(nu)

    # ── SPINE — count primaries AND all objects (consistent with Pandora) ──────
    n_spine_prim = 0
    n_spine_all  = 0
    spine_dirs = []
    if ixn.ndlp > 0:
        si  = closest_dlp(ixn, nu.vtx)
        dlp = ixn.dlp[si]
        for pi in range(dlp.part.ndlp):
            sp = dlp.part.dlp[pi]
            ppx, ppy, ppz = sp.p.x, sp.p.y, sp.p.z
            p_sp = vmag(ppx, ppy, ppz)
            if not (math.isfinite(p_sp) and p_sp > P_SPINE_MIN):
                continue
            n_spine_all += 1
            if sp.primary:
                n_spine_prim += 1
            spine_dirs.append((ppx/p_sp, ppy/p_sp, ppz/p_sp))

    # ── Pandora — count primaries AND all PFPs (consistent with SPINE) ─────────
    n_pandora_prim = 0
    n_pandora_all  = 0
    pan_dirs  = []
    if ixn.npandora > 0:
        pi2     = closest_pan(ixn, nu.vtx)
        pan     = ixn.pandora[pi2]
        pand_fd = rec.fd.hd.pandora[pi2]
        # npandora includes a sentinel (pdg=0, primary=False) as the last entry;
        # fd.showers / fd.tracks arrays have size = npandora - 1
        n_pfp = int(pan.part.npandora) - 1
        for pi in range(n_pfp):
            p = pan.part.pandora[pi]
            n_pandora_all += 1
            if not p.primary:
                continue
            n_pandora_prim += 1
            d = None
            if int(p.pdg) == 13:
                td  = pand_fd.tracks[pi].dir
                tdn = vmag(td.x, td.y, td.z)
                if tdn > 1e-6:
                    d = (td.x/tdn, td.y/tdn, td.z/tdn)
            if d is None:
                sd  = pand_fd.showers[pi].direction
                sdn = vmag(sd.x, sd.y, sd.z)
                if sdn > 1e-6:
                    d = (sd.x/sdn, sd.y/sdn, sd.z/sdn)
            if d is not None:
                pan_dirs.append(d)

    # ── Matching efficiency (greedy: each truth matched at most once) ─────────
    def count_matched(truth_ds, reco_ds):
        if not truth_ds or not reco_ds:
            return 0
        reco_used = [False] * len(reco_ds)
        n = 0
        for td in truth_ds:
            best_c, best_ri = COS_MATCH, -1
            for ri, rd in enumerate(reco_ds):
                if reco_used[ri]:
                    continue
                c = cos_angle(*td, *rd)
                if math.isfinite(c) and c > best_c:
                    best_c, best_ri = c, ri
            if best_ri >= 0:
                n += 1
                reco_used[best_ri] = True
        return n

    n_spine_matched = count_matched(truth_dirs, spine_dirs)
    n_pan_matched   = count_matched(truth_dirs, pan_dirs)

    rows.append({
        "nu_E":         nu_E,
        "is_cc":        is_cc,
        "nu_pdg":       nu_pdg,
        "n_recon":      n_recon,          # truth: primaries + near-vertex (5 cm) sec
        "n_recon_full": n_recon_full,     # truth: primaries + ALL reconstructible sec
        "n_spine":      n_spine_all,      # SPINE all reconstructed objects
        "n_spine_prim": n_spine_prim,     # SPINE primaries only
        "n_pandora":    n_pandora_all,    # Pandora all PFPs
        "n_pandora_prim": n_pandora_prim, # Pandora primaries only
        "n_sp_match":   n_spine_matched,
        "n_pa_match":   n_pan_matched,
        "has_dlp":      ixn.ndlp > 0,
        "has_pan":      ixn.npandora > 0,
    })

print(f"  Events: {len(rows):,}")

# ── Arrays ────────────────────────────────────────────────────────────────────
A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
A["nu_E"]    = A["nu_E"].astype(float)
for _k in ("n_recon", "n_recon_full", "n_spine", "n_spine_prim",
           "n_pandora", "n_pandora_prim", "nu_pdg"):
    A[_k] = A[_k].astype(int)
A["is_cc"]   = A["is_cc"].astype(bool)
A["has_pan"] = A["has_pan"].astype(bool)
A["has_dlp"] = A["has_dlp"].astype(bool)

cc   = A["is_cc"]
hasp = A["has_pan"]
hasd = A["has_dlp"]

# guarded divisions below use np.where; silence the harmless 0/0 evaluations
np.seterr(divide="ignore", invalid="ignore")

# ── Summary statistics ────────────────────────────────────────────────────────
E_CUTS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
E_LABS = ["E<0.5 GeV", "0.5–1 GeV", "1–2 GeV", ">2 GeV"]

# Consistent counting: SPINE and Pandora are compared on the same basis (both
# "all objects" and both "primaries only"); truth is shown near-vertex and full.
print(f"\nMean particle counts (CC events) — consistent reco counting:")
print(f"{'Bin':<12}{'true(5cm)':>10}{'true(full)':>11}{'SPINE_all':>10}{'SPINE_pri':>10}"
      f"{'Pand_all':>9}{'Pand_pri':>9}")
for (lo, hi), lb in zip(E_CUTS, E_LABS):
    m  = cc & (A["nu_E"] >= lo) & (A["nu_E"] < hi)
    md = m & hasd          # SPINE valid events
    mp = m & hasp          # Pandora valid events
    def mn(key, mask):
        return float(np.mean(A[key][mask])) if mask.sum() else _nan
    print(f"{lb:<12}{mn('n_recon',m):>10.2f}{mn('n_recon_full',m):>11.2f}"
          f"{mn('n_spine',md):>10.2f}{mn('n_spine_prim',md):>10.2f}"
          f"{mn('n_pandora',mp):>9.2f}{mn('n_pandora_prim',mp):>9.2f}")

# ── Figure 1: Migration matrices (CC only) — PRIMARIES, apples-to-apples ──────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Primary-particle multiplicity migration (CC) — reconstructible truth vs reco primaries",
             fontsize=12)

MAX_N = 9
bins  = np.arange(-0.5, MAX_N + 1.5)

for ax, key, evmask, label in [
        (axes[0], "n_spine_prim",   cc & hasd, "SPINE"),
        (axes[1], "n_pandora_prim", cc & hasp, "Pandora"),
]:
    xv = A["n_recon"][evmask]
    yv = A[key][evmask]
    med_bias = float(np.median(yv - xv))

    H, xedges, yedges = np.histogram2d(
        np.clip(xv, 0, MAX_N), np.clip(yv, 0, MAX_N),
        bins=[bins, bins]
    )
    row_sum = H.sum(axis=1, keepdims=True)
    H_norm  = np.where(row_sum > 0, H / row_sum, 0.0)

    im = ax.imshow(H_norm.T, origin="lower", aspect="auto",
                   extent=[bins[0], bins[-1], bins[0], bins[-1]],
                   cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Row fraction")
    ax.plot([0, MAX_N], [0, MAX_N], "r--", lw=1.0, label="N_reco = N_truth")
    ax.set_xlabel("N truth reconstructible (primaries)")
    ax.set_ylabel(f"N {label} reco primaries")
    ax.set_title(f"{label}   (median Δ = {med_bias:+.0f})")
    ax.legend(fontsize=8)

fig.tight_layout()
out1 = os.path.join(outdir, "multiplicity_migration.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out1}")

# ── Figure 2: Signed difference distributions ─────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(r"$N_\mathrm{reco} - N_\mathrm{reconstructible}$ (CC events)", fontsize=13)

diff_bins = np.arange(-5.5, 8.5)
diff_ctr  = 0.5 * (diff_bins[:-1] + diff_bins[1:])

for row_idx, (lo, hi), lb in [(0, (0.0, 1.0), "E < 1 GeV"),
                               (1, (1.0, 1e9), "E > 1 GeV")]:
    for col_idx, (key, color, label) in enumerate([
            ("n_spine",   "#1f77b4", "SPINE"),
            ("n_pandora", "#ff7f0e", "Pandora"),
    ]):
        ax = axes[row_idx, col_idx]
        mask = cc & (A["nu_E"] >= lo) & (A["nu_E"] < hi)
        mask &= hasp if key == "n_pandora" else hasd
        diff = A[key][mask] - A["n_recon"][mask]
        counts, _ = np.histogram(np.clip(diff, diff_bins[0], diff_bins[-1]),
                                 bins=diff_bins)
        ax.bar(diff_ctr, counts / counts.sum(), width=0.8,
               color=color, alpha=0.75, label=label)
        ax.axvline(0, color="k", lw=1.0, ls="--")
        mean_d = diff.mean()
        ax.axvline(mean_d, color="red", lw=1.5, ls="-",
                   label=f"mean={mean_d:+.2f}")
        ax.set_xlabel(r"$N_\mathrm{reco} - N_\mathrm{recon}$")
        ax.set_ylabel("Fraction of events")
        ax.set_title(f"{label} — {lb}")
        ax.legend(fontsize=8)

fig.tight_layout()
out2 = os.path.join(outdir, "multiplicity_diff_distribution.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")

# ── Figure 3: Efficiency and N_reco vs N_reconstructible vs Enu ───────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Reconstructed vs true particle count (CC) — consistent counting", fontsize=13)

nu_bins = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
nu_ctr  = 0.5 * (nu_bins[:-1] + nu_bins[1:])

def bmedian(x, y, bins=nu_bins):
    m = np.isfinite(x) & np.isfinite(y.astype(float))
    if m.sum() < 2:
        return np.full(len(bins)-1, _nan)
    v, _, _ = _bstat(x[m], y[m].astype(float), "median", bins=bins)
    c, _, _ = _bstat(x[m], y[m].astype(float), "count",  bins=bins)
    return np.where(c >= 10, v, _nan)

def bmean(x, y, bins=nu_bins):
    m = np.isfinite(x) & np.isfinite(y.astype(float))
    if m.sum() < 2:
        return np.full(len(bins)-1, _nan)
    v, _, _ = _bstat(x[m], y[m].astype(float), "mean", bins=bins)
    c, _, _ = _bstat(x[m], y[m].astype(float), "count", bins=bins)
    return np.where(c >= 10, v, _nan)

m_sp_cc = cc & hasd
m_pan_cc = cc & hasp
xs = A["nu_E"][m_sp_cc]
xp = A["nu_E"][m_pan_cc]

# Panel 1: MEAN counts vs Enu — consistent reco counting, both truth targets
ax = axes[0, 0]
ax.plot(nu_ctr, bmean(A["nu_E"][cc], A["n_recon"][cc]),      "k-",  lw=2.4, label="truth: prim + near-vtx sec (5 cm)")
ax.plot(nu_ctr, bmean(A["nu_E"][cc], A["n_recon_full"][cc]), "k:",  lw=2.4, label="truth: prim + ALL reconstructible sec")
ax.plot(nu_ctr, bmean(xs, A["n_spine"][m_sp_cc]),   color="#1f77b4", ls="--", lw=2.0, label="SPINE all objects")
ax.plot(nu_ctr, bmean(xp, A["n_pandora"][m_pan_cc]),color="#ff7f0e", ls="--", lw=2.0, label="Pandora all objects")
ax.set_xlabel(r"True $E_\nu$ [GeV]"); ax.set_ylabel("Mean particle count")
ax.set_title("All reconstructed objects vs truth"); ax.legend(fontsize=7.5)

# Panel 2: primaries-only, consistent — SPINE vs Pandora vs near-vtx truth
ax = axes[0, 1]
ax.plot(nu_ctr, bmean(A["nu_E"][cc], A["n_recon"][cc]),    "k-", lw=2.2, label="truth (prim + near-vtx sec)")
ax.plot(nu_ctr, bmean(xs, A["n_spine_prim"][m_sp_cc]),     color="#1f77b4", ls="--", lw=2.0, label="SPINE primaries")
ax.plot(nu_ctr, bmean(xp, A["n_pandora_prim"][m_pan_cc]),  color="#ff7f0e", ls="-.", lw=2.0, label="Pandora primaries")
ax.set_xlabel(r"True $E_\nu$ [GeV]"); ax.set_ylabel("Mean particle count")
ax.set_title("Primaries only (apples-to-apples)"); ax.legend(fontsize=8)

# Panel 3: non-primary (secondary/fragment) reconstructed objects per event —
# this is where SPINE and Pandora diverge: SPINE reconstructs hadronic secondaries.
ax = axes[1, 0]
sp_np = (A["n_spine"] - A["n_spine_prim"]).astype(float)
pa_np = (A["n_pandora"] - A["n_pandora_prim"]).astype(float)
ax.plot(nu_ctr, bmean(xs, sp_np[m_sp_cc]), color="#1f77b4", lw=2.0, marker="o", ms=3, label="SPINE")
ax.plot(nu_ctr, bmean(xp, pa_np[m_pan_cc]), color="#ff7f0e", lw=2.0, marker="s", ms=3, label="Pandora")
ax.set_xlabel(r"True $E_\nu$ [GeV]"); ax.set_ylabel("Mean non-primary reco objects / event")
ax.set_title("Non-primary reconstructed objects (hadronic secondaries)")
ax.set_ylim(0, None); ax.legend(fontsize=8)

# Panel 4: Per-particle matching efficiency vs Enu (near-vtx truth)
ax = axes[1, 1]
for (key_match, key_cc, xv_, color, label) in [
        ("n_sp_match", m_sp_cc,  xs, "#1f77b4", "SPINE"),
        ("n_pa_match", m_pan_cc, xp, "#ff7f0e", "Pandora"),
]:
    matched = A[key_match][key_cc].astype(float)
    recon   = A["n_recon"][key_cc].astype(float)
    ratio   = np.where(recon > 0, matched / recon, _nan)
    ax.plot(nu_ctr, bmean(xv_, ratio), color=color, lw=2.0, label=label)
ax.set_xlabel(r"True $E_\nu$ [GeV]"); ax.set_ylabel("Per-particle match efficiency")
ax.set_ylim(0, 1.05); ax.set_title("Direction matching (cos θ > 0.5, near-vtx truth)"); ax.legend(fontsize=8)

fig.tight_layout()
out3 = os.path.join(outdir, "multiplicity_efficiency.png")
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out3}")

# ── Figure 4: MEAN multiplicity vs expected-true, split by flavour ────────────
# reco vs the expected reconstructible-truth count, using the MEAN (not median),
# separately for νμ, ν̄μ, νe, ν̄e.
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
fig.suptitle("Mean primaries-only multiplicity vs truth (CC, by flavour) — consistent counting",
             fontsize=13)
FLAV = [(14, r"$\nu_\mu$"), (-14, r"$\bar{\nu}_\mu$"),
        (12, r"$\nu_e$"),   (-12, r"$\bar{\nu}_e$")]
for ax, (pdg, lab) in zip(axes.flat, FLAV):
    fm = cc & (A["nu_pdg"] == pdg)
    fs = fm & hasd
    fp = fm & hasp
    ax.plot(nu_ctr, bmean(A["nu_E"][fm], A["n_recon"][fm].astype(float)),
            "k-",  lw=2.2, marker="o", ms=3, label="truth (near-vtx)")
    ax.plot(nu_ctr, bmean(A["nu_E"][fm], A["n_recon_full"][fm].astype(float)),
            "k:",  lw=2.2, ms=3, label="truth (full)")
    ax.plot(nu_ctr, bmean(A["nu_E"][fs], A["n_spine_prim"][fs].astype(float)),
            color="#1f77b4", ls="--", lw=2.0, marker="^", ms=3, label="SPINE primaries")
    ax.plot(nu_ctr, bmean(A["nu_E"][fp], A["n_pandora_prim"][fp].astype(float)),
            color="#ff7f0e", ls="-.", lw=2.0, marker="s", ms=3, label="Pandora primaries")
    ax.set_title(f"{lab} CC", fontsize=11)
    ax.set_xlim(0.1, 4.0); ax.set_ylim(0, None)
    ax.set_xlabel(r"True $E_\nu$ [GeV]"); ax.set_ylabel("Mean particle count")
    ax.legend(fontsize=7.5)
fig.tight_layout()
out4 = os.path.join(outdir, "multiplicity_mean_flavor.png")
fig.savefig(out4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out4}")

# print mean multiplicity table by flavour and Eν bin
print("\nMean primaries-only multiplicity (truth near-vtx / SPINE / Pandora) by flavour & Eν:")
for pdg, lab in FLAV:
    fm = cc & (A["nu_pdg"] == pdg)
    print(f"  {lab}:")
    for (lo, hi), lb in zip(E_CUTS, E_LABS):
        m = fm & (A["nu_E"] >= lo) & (A["nu_E"] < hi)
        md = m & hasd
        mp = m & hasp
        if m.sum() < 10:
            continue
        print(f"    {lb:<12} true={A['n_recon'][m].mean():4.2f}  "
              f"SPINE_prim={A['n_spine_prim'][md].mean() if md.sum() else float('nan'):4.2f}  "
              f"Pandora_prim={A['n_pandora_prim'][mp].mean() if mp.sum() else float('nan'):4.2f}")

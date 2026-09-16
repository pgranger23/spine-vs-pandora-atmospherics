#!/usr/bin/env python3
"""Root-cause investigation of SPINE electron momentum over-estimation.

Hypotheses:
A. p.p (momentum) is in MeV/c for all particles, but p.E (energy) has different
   meaning for EM showers (e.g., total deposited charge in ADC)
B. p.p is NOT in MeV/c for EM showers — uses wrong mass hypothesis
C. SPINE outputs track-equivalent dE/dx length and converts with wrong dE/dx
D. The network outputs a different physical quantity for EM vs track primaries
"""

import os, sys, ctypes, math
import numpy as np
from tqdm import tqdm

import caf_env
pySRProxy = caf_env.setup_pysrproxy()

prox = pySRProxy.StandardRecordFileReader("recTree", caf_env.CAF_FILE)

_MASS = {11: 0.000511, 13: 0.10566, 22: 0.0, 211: 0.13957, 321: 0.49368,
         2212: 0.93827, 2112: 0.93957, 111: 0.13498}

def mag(x, y, z): return math.sqrt(x*x + y*y + z*z)

def best_dlp_idx(ixn, vtx):
    best_d, best_i = float("inf"), 0
    for i in range(min(ixn.ndlp, 100)):
        v = ixn.dlp[i].vtx
        d = mag(v.x - vtx.x, v.y - vtx.y, v.z - vtx.z)
        if d < best_d: best_d, best_i = d, i
    return best_i

# Collect raw per-particle data (no unit conversion)
particles = []  # raw values from proxy

for entry in tqdm(range(min(5000, prox.entries())), desc="Scanning"):
    rec = prox.entry(entry); mc = rec.mc
    if not (mc.nnu > 0 and mc.nu[0].iscc): continue
    ixn = rec.common.ixn
    if ixn.ndlp < 1: continue

    nu = mc.nu[0]
    si = best_dlp_idx(ixn, nu.vtx)
    dlp = ixn.dlp[si]

    for pi in range(dlp.part.ndlp):
        p = dlp.part.dlp[pi]
        if not p.primary: continue
        px, py, pz = p.p.x, p.p.y, p.p.z
        pm = mag(px, py, pz)
        if pm < 1: continue

        particles.append({
            "pdg": p.pdg,
            "p_px": px, "p_py": py, "p_pz": pz,
            "p_mag": pm,  # raw MeV/c (presumed)
            "p_E": p.E,  # raw MeV (presumed)
            "reco_pdg": p.pdg,
        })

    # Also check truth electrons from MC
    for ti in range(nu.nprim):
        tp = nu.prim[ti]
        px, py, pz = tp.p.px, tp.p.py, tp.p.pz
        pm = mag(px, py, pz)
        if pm < 20 or abs(tp.pdg) != 11: continue  # only electrons, MeV
        particles.append({
            "pdg": tp.pdg,
            "p_px": px, "p_py": py, "p_pz": pz,
            "p_mag": pm,  # MeV/c
            "p_E": 0,  # truth doesn't have E in same format
            "reco_pdg": 0,  # truth marker
        })

print(f"\nCollected {len(particles)} primary particles")

# Separate by type
reco = [p for p in particles if p["reco_pdg"] != 0]
truth = [p for p in particles if p["reco_pdg"] == 0]

# Group reco by PDG
from collections import defaultdict
by_pdg = defaultdict(list)
for p in reco:
    by_pdg[abs(p["pdg"])].append(p)

print("\n=== SPINE raw values by PDG ===")
for pdg in sorted(by_pdg.keys()):
    vals = by_pdg[pdg]
    pm = np.array([v["p_mag"] for v in vals])
    pe = np.array([v["p_E"] for v in vals])
    name = {11: "e", 13: "mu", 22: "gamma", 211: "pi", 2212: "p", 2112: "n",
            321: "K", 111: "pi0"}.get(pdg, f"PDG{pdg}")
    mass = _MASS.get(pdg, 0)
    print(f"\n  {name} (PDG={pdg}), N={len(vals)}, mass={mass} GeV")
    print(f"    |p| mean={np.mean(pm):.0f} MeV/c, median={np.median(pm):.0f}")
    print(f"    p.E mean={np.mean(pe):.0f} MeV, median={np.median(pe):.0f}")
    # If p.E is kinetic energy: p.E + mass = sqrt(p^2 + mass^2)
    expected_Ekin_if_p = np.sqrt(pm**2 + (mass*1000)**2) - mass*1000
    expected_Ekin_from_mom = np.mean(expected_Ekin_if_p)
    print(f"    If p is momentum: expected E_kin = sqrt(p^2 + m^2) - m = {expected_Ekin_from_mom:.0f} MeV")
    print(f"    Ratio p.E / |p| = {np.mean(pe/pm):.3f}")
    print(f"    Ratio p.E / expected_Ekin = {np.mean(pe / expected_Ekin_if_p):.3f}")
    # If p.E is total energy: p.E - mass = expected_KE
    expected_T_if_Etotal = pe - mass*1000
    print(f"    If p.E is total E: T = p.E - m = {np.median(expected_T_if_Etotal):.0f} MeV")
    print(f"    If p.E is total E: |p| vs sqrt(E^2 - m^2) = {np.mean(pm / np.sqrt(np.maximum(pe**2 - (mass*1000)**2, 0))):.3f}")

# For electrons specifically: compare p_mag vs sqrt(E² - m²)
print("\n\n=== ELECTRON DEEP DIVE ===")
e_reco = by_pdg.get(11, [])
if e_reco:
    pm = np.array([v["p_mag"] for v in e_reco])
    pe = np.array([v["p_E"] for v in e_reco])
    mass_mev = _MASS[11] * 1000  # 0.511 MeV

    # Check: is p.E actually total energy deposited (calorimetric)?
    # If so, |p| vs sqrt(E² - m²) should be close to 1
    p_from_E = np.sqrt(np.maximum(pe**2 - mass_mev**2, 0))
    valid = pe > mass_mev
    print(f"  N electrons: {len(e_reco)}")
    print(f"  |p| mean: {np.mean(pm):.0f} MeV/c")
    print(f"  p.E mean: {np.mean(pe):.0f} MeV")
    print(f"  |p| / sqrt(E² - m²) (if E=total): {np.mean(pm[valid] / p_from_E[valid]):.3f}")
    print(f"  sqrt(E² - m²) / |p| (if p=momentum): {np.mean(p_from_E[valid] / pm[valid]):.3f}")

    # Check: does p.E * some_norm_factor ≈ |p| from truth?
    # What if p.E is in different units?
    print(f"\n  --- Scatter check: p_mag vs p_E ---")
    # Linear fit ratio
    ratio = pm / pe
    print(f"  p_mag / p_E: median={np.median(ratio):.3f}, mean={np.mean(ratio):.3f}, std={np.std(ratio):.3f}")
    # What factor makes p_mag agree with truth? (we know answer is ~4.76)
    # truth_p_mag ≈ 4.76 * p_E for electrons... but what about protons?
    # For protons p_mag/p_E ≈ ?

    # Check if p_E correlates with something else
    mom_sq = pm**2
    e_sq = pe**2
    # If both are valid, we should see E² = p² + m²
    # For electron: m² negligible, so E ≈ p
    # If p.E is kinetic: T ≈ p²/2m for non-rel, or ≈ p-m for ultra-rel...
    # But for relativistic electrons: T = sqrt(p²+m²) - m ≈ p - m²/(2p) ≈ p
    # So E_kin ≈ p (in natural units)
    print(f"  p.E vs |p|: E_mag_ratio = {np.median(pe/pm):.3f}")

# Compare e vs p and mu
print("\n\n=== COMPARISON: e vs p vs mu ===")
for pdg, name in [(11, "e"), (2212, "p"), (13, "mu")]:
    vals = by_pdg.get(pdg, [])
    if not vals: continue
    pm = np.array([v["p_mag"] for v in vals])
    pe = np.array([v["p_E"] for v in vals])
    mass_mev = _MASS.get(pdg, 0) * 1000
    print(f"\n  {name}:")
    print(f"    |p| / p.E = {np.median(pm/pe):.3f} (if ~1, p.E is in momentum units)")
    print(f"    sqrt(p² + m²) - m = {np.median(np.sqrt(pm**2 + mass_mev**2) - mass_mev):.0f} MeV")
    print(f"    p.E = {np.median(pe):.0f} MeV")
    # What if p.E is in "track-equivalent MeV" — i.e., what a mip-like particle
    # with dE/dx ~2 MeV/cm would deposit over the same path length?
    # For MIPs in LAr: dE/dx ~ 2.1 MeV/cm
    # For EM showers: total track length = E_crit * X0 / (dE/dx) ~ 30 MeV * 14 cm / 2.1 ≈ ...
    # This is getting speculative. Let's just check the data first.

# Check the truth electron distribution for comparison
print("\n\n=== TRUTH ELECTRON CHECK ===")
t_e = [p for p in truth if abs(p["pdg"]) == 11]
if t_e:
    t_pm = np.array([v["p_mag"] for v in t_e])
    print(f"  N truth electrons: {len(t_e)}")
    print(f"  True p mean: {np.mean(t_pm):.0f} MeV/c, median: {np.median(t_pm):.0f}")
    # Compare to SPINE electron p_mag
    if e_reco:
        r_pm = np.array([v["p_mag"] for v in e_reco])
        print(f"  SPINE reco p mean: {np.mean(r_pm):.0f} MeV/c, median: {np.median(r_pm):.0f}")
        print(f"  SPINE/Truth p ratio: {np.median(r_pm) / np.median(t_pm):.2f}")
        # What if we look at the SPINE p_mag distribution and compare to truth
        print(f"  SPINE p_mag percentiles: 10%={np.percentile(r_pm, 10):.0f}, 50%={np.percentile(r_pm, 50):.0f}, 90%={np.percentile(r_pm, 90):.0f}")
        print(f"  Truth p_mag percentiles: 10%={np.percentile(t_pm, 10):.0f}, 50%={np.percentile(t_pm, 50):.0f}, 90%={np.percentile(t_pm, 90):.0f}")

# Now let's also check: what if SPINE p.E uses the WRONG mass?
# Maybe it stores kinetic energy computed as (p_reco_network_output) with
# mass=0 assumption? Then for electron (m≈0): E_kin = sqrt(p²+0) - 0 = |p|, fine.
# But for proton: E_kin = sqrt(p²+m²) - m ≠ |p|.
# Actually we know for protons, p.E (kinetic) + m_true = total E. So the mass
# doesn't enter the stored p.E at all — it's just kinetic energy.

# What if the reconstructed momentum p_mag is actually p.E (i.e., they store
# the same quantity in both fields for some particles)?
print("\n\n=== p_mag vs p_E correlation ===")
for pdg, name in [(11, "e"), (2212, "p"), (13, "mu"), (22, "gamma")]:
    vals = by_pdg.get(pdg, [])
    if not vals or len(vals) < 5: continue
    pm = np.array([v["p_mag"] for v in vals])
    pe = np.array([v["p_E"] for v in vals])
    # Check if they're basically the same (within 1%)
    ratio = pm / pe
    print(f"  {name}: |p|/p.E shape: mean={np.mean(ratio):.2f}, std={np.std(ratio):.2f}")
    # Could p_mag be in MeV/c and p.E in the same units but with a different
    # calibration? E.g., p_mag = p.E * something?
    # For muons: ratio ~1 → they're the same
    # For protons: ratio ~? 
    # For electrons: ratio ~? 

print("\n\n=== PROTON CHECK (control sample) ===")
p_vals = by_pdg.get(2212, [])
if p_vals:
    pm = np.array([v["p_mag"] for v in p_vals])
    pe = np.array([v["p_E"] for v in p_vals])
    mass_mev = _MASS[2212] * 1000
    print(f"  N protons: {len(p_vals)}")
    print(f"  |p|: mean={np.mean(pm):.0f}, median={np.median(pm):.0f}")
    print(f"  p.E: mean={np.mean(pe):.0f}, median={np.median(pe):.0f}")
    print(f"  |p|/p.E: {np.median(pm/pe):.3f}")
    # For proton, |p| should be > p.E since p.E is kinetic energy
    # sqrt(p²+m²) - m < |p| for non-relativistic
    expected_kin = np.sqrt(pm**2 + mass_mev**2) - mass_mev
    print(f"  expected E_kin from p: mean={np.mean(expected_kin):.0f}")
    print(f"  p.E / expected_E_kin: {np.median(pe/expected_kin):.3f}")

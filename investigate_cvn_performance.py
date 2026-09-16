#!/usr/bin/env python3
"""
Study A: CVN flavor classifier ROC curves and working points.

Evaluates SPINE and Pandora CVN neural network scores for neutrino flavor
identification (νe-CC, νμ-CC, NC) using ROC analysis.

Output: figures/cvn_roc.png, figures/cvn_confusion_at_threshold.png,
        figures/cvn_summary.json
"""

import os, sys, ctypes, json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, auc, confusion_matrix
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

def _closest_dlp_idx(ixn, nu_vtx):
    best, best_i = float("inf"), 0
    for i in range(ixn.ndlp):
        v = ixn.dlp[i].vtx
        d = ((v.x - nu_vtx.x)**2 + (v.y - nu_vtx.y)**2 + (v.z - nu_vtx.z)**2)**0.5
        if d < best:
            best, best_i = d, i
    return best_i

def _closest_pan_idx(ixn, nu_vtx):
    best, best_i = float("inf"), 0
    for i in range(ixn.npandora):
        v = ixn.pandora[i].vtx
        d = ((v.x - nu_vtx.x)**2 + (v.y - nu_vtx.y)**2 + (v.z - nu_vtx.z)**2)**0.5
        if d < best:
            best, best_i = d, i
    return best_i

# ---------------------------------------------------------------
# Scan CAF for CVN scores + truth
# ---------------------------------------------------------------
print("=" * 60)
print("Study A: CVN Flavor Classifier Performance")
print("=" * 60)
print("Scanning CAF...")

prox = pySRProxy.StandardRecordFileReader("recTree", "spine_merged_reprocess.root")

records = []
_nan = float("nan")

for ientry in tqdm(range(prox.entries()), desc="CVN scan"):
    rec = prox.entry(ientry)
    mc = rec.mc
    if mc.nnu < 1:
        continue
    nu = mc.nu[0]
    nu_pdg = int(nu.pdg)
    nu_iscc = bool(nu.iscc)
    nu_E = float(nu.E)

    ixn = rec.common.ixn
    if ixn.ndlp < 1:
        continue

    si = _closest_dlp_idx(ixn, nu.vtx)
    dlp = ixn.dlp[si]
    scvn = dlp.nuhyp.cvn

    try:
        s_nue  = float(scvn.nue)
        s_numu = float(scvn.numu)
        s_nc   = float(scvn.nc)
    except Exception:
        s_nue = s_numu = s_nc = _nan

    # SPINE PID-based flavor tagger: muon → νμ, electron (if no muon) → νe, else NC
    _has_muon = False
    _has_electron = False
    for _pi in range(dlp.part.ndlp):
        _part = dlp.part.dlp[_pi]
        _pdg  = int(_part.pdg)
        _p    = _part.p
        if (_p.x**2 + _p.y**2 + _p.z**2)**0.5 < 20.0:
            continue
        if abs(_pdg) == 13:
            _has_muon = True
        elif abs(_pdg) == 11:
            _has_electron = True
    _spine_pid = 1 if _has_muon else (0 if _has_electron else 2)

    p_nue = p_numu = p_nc = _nan
    if ixn.npandora > 0:
        pi = _closest_pan_idx(ixn, nu.vtx)
        pan = ixn.pandora[pi]
        pcvn = pan.nuhyp.cvn
        try:
            p_nue  = float(pcvn.nue)
            p_numu = float(pcvn.numu)
            p_nc   = float(pcvn.nc)
        except Exception:
            pass

    records.append({
        "nu_pdg": nu_pdg, "nu_iscc": nu_iscc, "nu_E": nu_E,
        "s_nue": s_nue, "s_numu": s_numu, "s_nc": s_nc,
        "p_nue": p_nue, "p_numu": p_numu, "p_nc": p_nc,
        "spine_pid": _spine_pid,  # 0=nue, 1=numu, 2=NC
    })

print(f"  Total entries scanned: {len(records)}")

nu_pdg  = np.array([r["nu_pdg"]  for r in records], dtype=int)
nu_iscc = np.array([r["nu_iscc"] for r in records], dtype=bool)
nu_E    = np.array([r["nu_E"]    for r in records])

s_nue  = np.array([r["s_nue"]  for r in records])
s_numu = np.array([r["s_numu"] for r in records])
s_nc   = np.array([r["s_nc"]   for r in records])
p_nue  = np.array([r["p_nue"]  for r in records])
p_numu = np.array([r["p_numu"] for r in records])
p_nc   = np.array([r["p_nc"]   for r in records])
spine_pid = np.array([r["spine_pid"] for r in records], dtype=int)  # 0=nue, 1=numu, 2=NC

# True labels
true_nue_cc  = (np.abs(nu_pdg) == 12) & nu_iscc
true_numu_cc = (np.abs(nu_pdg) == 14) & nu_iscc
true_nc      = ~nu_iscc  # any NC (includes both flavors)

# Valid mask: CVN scores must be finite
s_valid = np.isfinite(s_nue) & np.isfinite(s_numu)
p_valid = np.isfinite(p_nue) & np.isfinite(p_numu)
print(f"  Events with valid SPINE CVN: {s_valid.sum():,}")
print(f"  Events with valid Pandora CVN: {p_valid.sum():,}")
print(f"  True νe-CC: {true_nue_cc.sum():,}, νμ-CC: {true_numu_cc.sum():,}, NC: {true_nc.sum():,}")

# ---------------------------------------------------------------
# Figure 1: ROC curves — νe-ID and νμ-ID
# ---------------------------------------------------------------
print("\n--- Figure 1: ROC curves ---")
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("CVN Flavor Classifier ROC Curves — SPINE vs Pandora",
              fontsize=13, fontweight="bold")

roc_results = {}

for ax, task_lbl, true_label, s_score, p_score, task_key, pid_class in [
    (axes1[0], r"$\nu_e$-CC identification",  true_nue_cc,  s_nue,  p_nue,  "nue",  0),
    (axes1[1], r"$\nu_\mu$-CC identification", true_numu_cc, s_numu, p_numu, "numu", 1),
]:
    for lbl, score, valid, col, ls in [
        ("SPINE CVN", s_score, s_valid, "#1f77b4", "-"),
        ("Pandora CVN", p_score, p_valid, "#ff7f0e", "--"),
    ]:
        # Use all valid events; background = everything that is not the signal class
        mask_v = valid
        y_true = true_label[mask_v].astype(int)
        y_score = score[mask_v]
        if y_true.sum() < 10 or (1 - y_true).sum() < 10:
            continue
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, ls=ls, lw=2,
                label=f"{lbl} (AUC={roc_auc:.3f})")
        roc_results[f"{lbl.lower().replace(' ', '_')}_{task_key}"] = {
            "auc": float(roc_auc),
            "n_signal": int(y_true.sum()),
            "n_background": int((1 - y_true).sum()),
        }
        # Mark 90% TPR working point
        idx90 = np.searchsorted(tpr, 0.90)
        if idx90 < len(fpr):
            ax.scatter(fpr[idx90], tpr[idx90], s=80, zorder=5, color=col,
                       marker="^", edgecolors="k", linewidths=0.5)

    # SPINE PID tagger: single operating point (star marker)
    pid_pred = (spine_pid == pid_class)
    n_true_total = int(true_label.sum())
    n_bkg_total  = int((~true_label).sum())
    if n_true_total > 0 and n_bkg_total > 0:
        pid_tpr = float((pid_pred & true_label).sum()) / n_true_total
        pid_fpr = float((pid_pred & ~true_label).sum()) / n_bkg_total
        ax.scatter(pid_fpr, pid_tpr, s=220, zorder=7, color="#2ca02c",
                   marker="*", edgecolors="k", linewidths=0.8,
                   label=f"SPINE PID tagger (TPR={pid_tpr:.2f}, FPR={pid_fpr:.2f})")
        roc_results[f"spine_pid_{task_key}"] = {"tpr": round(pid_tpr, 4), "fpr": round(pid_fpr, 4)}

    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(task_lbl)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.text(0.55, 0.06, "▲ = 90% TPR working point   ★ = SPINE PID tagger",
            transform=ax.transAxes, fontsize=8, alpha=0.7)

fig1.tight_layout()
fig1.savefig(os.path.join(outdir, "cvn_roc.png"), dpi=150)
plt.close(fig1)
print("  Saved cvn_roc.png")
for k, v in roc_results.items():
    if "auc" in v:
        print(f"  {k}: AUC={v['auc']:.3f}, N_sig={v['n_signal']:,}, N_bkg={v['n_background']:,}")
    else:
        print(f"  {k}: TPR={v['tpr']:.3f}, FPR={v['fpr']:.3f}")

# ---------------------------------------------------------------
# Figure 2: Score distributions + confusion matrix at threshold
# ---------------------------------------------------------------
print("\n--- Figure 2: Score distributions and confusion at threshold ---")
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("CVN Score Distributions and Classification Summary",
              fontsize=13, fontweight="bold")

_spine_cvn_available = s_valid.sum() > 0

# Panels a/b: Pandora score distributions per true class
for ax, score_arr, valid_m, score_lbl in [
    (axes2[0, 0], p_nue,  p_valid, "Pandora CVN nue score"),
    (axes2[0, 1], p_numu, p_valid, "Pandora CVN numu score"),
]:
    bins = np.linspace(0, 1, 41)
    for lbl, mask_c, col in [
        ("νe CC",  true_nue_cc,  "#2ca02c"),
        ("νμ CC",  true_numu_cc, "#1f77b4"),
        ("NC",     true_nc,      "#d62728"),
    ]:
        m = valid_m & mask_c
        if m.sum() > 5:
            ax.hist(score_arr[m], bins=bins, histtype="step", lw=1.5,
                    density=True, color=col, label=f"{lbl} (N={m.sum():,})")
    ax.set_xlabel(score_lbl)
    ax.set_ylabel("Density")
    ax.set_title(score_lbl)
    ax.legend(fontsize=8)

def _eer_threshold(true_bin, score):
    if len(true_bin) == 0 or len(np.unique(true_bin)) < 2:
        return 0.5, float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(true_bin.astype(int), score, pos_label=1)
    diff = np.abs(fpr - (1 - tpr))
    idx = np.argmin(diff)
    return float(thr[idx]), float(tpr[idx]), float(fpr[idx])

# Panel c: SPINE PID-tagger confusion matrix
ax = axes2[1, 0]
thr_nue = thr_numu = 0.5   # kept for JSON compat; not used for PID tagger
cm_norm = np.zeros((3, 3))
cat_names = ["νe CC", "νμ CC", "NC/other"]
pid_true_classes = np.where(true_nue_cc, 0, np.where(true_numu_cc, 1, np.where(true_nc, 2, 3)))
used_pid = pid_true_classes < 3
cm_pid = confusion_matrix(pid_true_classes[used_pid], spine_pid[used_pid], labels=[0, 1, 2])
cm_pid_norm = cm_pid.astype(float) / cm_pid.sum(axis=1, keepdims=True).clip(min=1)
cm_norm = cm_pid_norm  # alias for summary JSON below
im_pid = ax.imshow(cm_pid_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(cat_names, rotation=30, ha="right")
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(cat_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("SPINE PID tagger confusion\n(muon→νμ, electron→νe, else NC)")
for i in range(3):
    for j in range(3):
        c = "white" if cm_pid_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm_pid_norm[i,j]:.2f}\n({cm_pid[i,j]:,})",
                ha="center", va="center", fontsize=8, color=c, fontweight="bold")
fig2.colorbar(im_pid, ax=ax, fraction=0.046)
print(f"  SPINE PID tagger confusion diagonal: "
      f"νe={cm_pid_norm[0,0]:.2f}, νμ={cm_pid_norm[1,1]:.2f}, NC={cm_pid_norm[2,2]:.2f}")

# Panel d: same for Pandora
ax = axes2[1, 1]
pan_mask = p_valid
true_classes_p = np.where(true_nue_cc[pan_mask],  0,
                 np.where(true_numu_cc[pan_mask], 1,
                 np.where(true_nc[pan_mask],      2, 3)))

thr_nue_p,  _, _ = _eer_threshold(true_nue_cc[pan_mask],  p_nue[pan_mask])
thr_numu_p, _, _ = _eer_threshold(true_numu_cc[pan_mask], p_numu[pan_mask])
p_nue_v  = p_nue[pan_mask]
p_numu_v = p_numu[pan_mask]
pred_p = np.where(p_nue_v >= thr_nue_p,  0,
         np.where(p_numu_v >= thr_numu_p, 1, 2))

used_p = true_classes_p < 3
cm_p = confusion_matrix(true_classes_p[used_p], pred_p[used_p], labels=[0, 1, 2])
cm_norm_p = cm_p.astype(float) / cm_p.sum(axis=1, keepdims=True).clip(min=1)
cat_names = ["νe CC", "νμ CC", "NC/other"]
im2 = ax.imshow(cm_norm_p, cmap="Oranges", vmin=0, vmax=1)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(cat_names, rotation=30, ha="right")
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(cat_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Pandora CVN confusion (EER threshold)\nnue thr={thr_nue_p:.2f}, numu thr={thr_numu_p:.2f}")
for i in range(3):
    for j in range(3):
        c = "white" if cm_norm_p[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm_norm_p[i,j]:.2f}\n({cm_p[i,j]:,})",
                ha="center", va="center", fontsize=8, color=c, fontweight="bold")
fig2.colorbar(im2, ax=ax, fraction=0.046)

fig2.tight_layout()
fig2.savefig(os.path.join(outdir, "cvn_confusion_at_threshold.png"), dpi=150)
plt.close(fig2)
print("  Saved cvn_confusion_at_threshold.png")

# ---------------------------------------------------------------
# Figure 3: CC-only PID performance
# (SPINE was trained on CC only — NC excluded from background here)
# ---------------------------------------------------------------
print("\n--- Figure 3: CC-only PID performance ---")

cc_roc_results = {}

# SPINE PID: 2×3 confusion for CC events only (true rows: νe CC, νμ CC)
cm_pid_cc = np.zeros((2, 3), dtype=int)
for ti, true_mask in enumerate([true_nue_cc, true_numu_cc]):
    for pred_class in range(3):
        cm_pid_cc[ti, pred_class] = int((true_mask & (spine_pid == pred_class)).sum())
cm_pid_cc_norm = cm_pid_cc.astype(float) / cm_pid_cc.sum(axis=1, keepdims=True).clip(min=1)

# Pandora CVN: 2×3 confusion for CC events only (argmax of [nue, numu, nc] scores)
p_nc_safe = np.where(np.isfinite(p_nc), p_nc, 1.0 - p_nue - p_numu)
pan_cc_mask = p_valid & (true_nue_cc | true_numu_cc)
cm_pan_cc = np.zeros((2, 3), dtype=int)
pred_pan_cc_arr = np.full(pan_cc_mask.sum(), -1, dtype=int)
if pan_cc_mask.sum() > 10:
    scores_stack = np.column_stack([p_nue[pan_cc_mask], p_numu[pan_cc_mask], p_nc_safe[pan_cc_mask]])
    pred_pan_cc_arr = np.argmax(scores_stack, axis=1)
    for ti, true_cc_sub in enumerate([true_nue_cc[pan_cc_mask], true_numu_cc[pan_cc_mask]]):
        for pred_class in range(3):
            cm_pan_cc[ti, pred_class] = int((true_cc_sub & (pred_pan_cc_arr == pred_class)).sum())
cm_pan_cc_norm = cm_pan_cc.astype(float) / cm_pan_cc.sum(axis=1, keepdims=True).clip(min=1)

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 11))
fig3.suptitle("Flavor PID Performance — CC Interactions Only\n"
              "(SPINE trained on CC; NC excluded from background)",
              fontsize=13, fontweight="bold")

_row_lbl = ["True νe CC", "True νμ CC"]
_col_lbl = ["Pred. νe", "Pred. νμ", "Pred. NC"]

# Panel [0,0]: SPINE PID 2×3 confusion
ax = axes3[0, 0]
im3a = ax.imshow(cm_pid_cc_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(_col_lbl, rotation=20, ha="right")
ax.set_yticks([0, 1]); ax.set_yticklabels(_row_lbl)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.set_title("SPINE PID tagger — CC events only")
for i in range(2):
    for j in range(3):
        c = "white" if cm_pid_cc_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm_pid_cc_norm[i,j]:.2f}\n({cm_pid_cc[i,j]:,})",
                ha="center", va="center", fontsize=9, color=c, fontweight="bold")
fig3.colorbar(im3a, ax=ax, fraction=0.046)

# Panel [0,1]: Pandora CVN 2×3 confusion
ax = axes3[0, 1]
im3b = ax.imshow(cm_pan_cc_norm, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(_col_lbl, rotation=20, ha="right")
ax.set_yticks([0, 1]); ax.set_yticklabels(_row_lbl)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.set_title("Pandora CVN (argmax) — CC events only")
for i in range(2):
    for j in range(3):
        c = "white" if cm_pan_cc_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm_pan_cc_norm[i,j]:.2f}\n({cm_pan_cc[i,j]:,})",
                ha="center", va="center", fontsize=9, color=c, fontweight="bold")
fig3.colorbar(im3b, ax=ax, fraction=0.046)

# Panels [1,0] and [1,1]: ROC curves CC-only (background = other CC flavor, NC excluded)
for ax, task_lbl, sig_mask, bkg_mask, p_score, task_key, pid_class in [
    (axes3[1, 0], r"$\nu_e$-CC vs $\nu_\mu$-CC (CC-only)",
     true_nue_cc, true_numu_cc, p_nue, "nue_cc", 0),
    (axes3[1, 1], r"$\nu_\mu$-CC vs $\nu_e$-CC (CC-only)",
     true_numu_cc, true_nue_cc, p_numu, "numu_cc", 1),
]:
    cc_roc_mask = sig_mask | bkg_mask  # only CC events, NC excluded
    pan_roc_mask = cc_roc_mask & p_valid
    y_t = sig_mask[pan_roc_mask].astype(int)
    y_s = p_score[pan_roc_mask]
    if y_t.sum() > 5 and (1 - y_t).sum() > 5:
        fpr_cc, tpr_cc, _ = roc_curve(y_t, y_s)
        roc_auc_cc = auc(fpr_cc, tpr_cc)
        ax.plot(fpr_cc, tpr_cc, color="#ff7f0e", ls="--", lw=2,
                label=f"Pandora CVN (AUC={roc_auc_cc:.3f})")
        cc_roc_results[f"pandora_{task_key}"] = {"auc": round(roc_auc_cc, 4)}
        idx90 = np.searchsorted(tpr_cc, 0.90)
        if idx90 < len(fpr_cc):
            ax.scatter(fpr_cc[idx90], tpr_cc[idx90], s=80, zorder=5, color="#ff7f0e",
                       marker="^", edgecolors="k", linewidths=0.5)
    n_sig_cc = int(sig_mask[cc_roc_mask].sum())
    n_bkg_cc = int(bkg_mask[cc_roc_mask].sum())
    if n_sig_cc > 0 and n_bkg_cc > 0:
        pid_pred_cc = (spine_pid[cc_roc_mask] == pid_class)
        true_sig_cc = sig_mask[cc_roc_mask]
        pid_tpr_cc = float((pid_pred_cc & true_sig_cc).sum()) / n_sig_cc
        pid_fpr_cc = float((pid_pred_cc & ~true_sig_cc).sum()) / n_bkg_cc
        ax.scatter(pid_fpr_cc, pid_tpr_cc, s=220, zorder=7, color="#2ca02c",
                   marker="*", edgecolors="k", linewidths=0.8,
                   label=f"SPINE PID (TPR={pid_tpr_cc:.2f}, FPR={pid_fpr_cc:.2f})")
        cc_roc_results[f"spine_pid_{task_key}"] = {
            "tpr": round(pid_tpr_cc, 4), "fpr": round(pid_fpr_cc, 4)}
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate (other CC flavor)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(task_lbl)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.text(0.5, 0.06, "▲ = 90% TPR working point   ★ = SPINE PID tagger",
            transform=ax.transAxes, fontsize=8, alpha=0.7, ha="center")

fig3.tight_layout()
fig3.savefig(os.path.join(outdir, "cvn_cc_only.png"), dpi=150)
plt.close(fig3)
print("  Saved cvn_cc_only.png")
print(f"  SPINE PID CC-only: "
      f"νe eff={cm_pid_cc_norm[0,0]:.2f}, inter-flav FP={cm_pid_cc_norm[0,1]:.2f}, NC-tag={cm_pid_cc_norm[0,2]:.2f}")
print(f"  SPINE PID CC-only: "
      f"νμ eff={cm_pid_cc_norm[1,1]:.2f}, inter-flav FP={cm_pid_cc_norm[1,0]:.2f}, NC-tag={cm_pid_cc_norm[1,2]:.2f}")
print(f"  Pandora CC-only:   "
      f"νe eff={cm_pan_cc_norm[0,0]:.2f}, inter-flav FP={cm_pan_cc_norm[0,1]:.2f}, NC-tag={cm_pan_cc_norm[0,2]:.2f}")
print(f"  Pandora CC-only:   "
      f"νμ eff={cm_pan_cc_norm[1,1]:.2f}, inter-flav FP={cm_pan_cc_norm[1,0]:.2f}, NC-tag={cm_pan_cc_norm[1,2]:.2f}")

# ---------------------------------------------------------------
# Print and save summary
# ---------------------------------------------------------------
def _diag_frac(cm_n, i):
    return float(cm_n[i, i])

n_pid_nue  = int((spine_pid == 0).sum())
n_pid_numu = int((spine_pid == 1).sum())
n_pid_nc   = int((spine_pid == 2).sum())

summary = {
    "n_events_scanned": len(records),
    "n_with_valid_spine_cvn": int(s_valid.sum()),
    "n_with_valid_pandora_cvn": int(p_valid.sum()),
    "true_nue_cc": int(true_nue_cc.sum()),
    "true_numu_cc": int(true_numu_cc.sum()),
    "true_nc": int(true_nc.sum()),
    "roc_auc": roc_results,
    "spine_cvn_available": bool(_spine_cvn_available),
    "pandora_eer_thresholds": {"nue": thr_nue_p, "numu": thr_numu_p},
    "spine_pid_tagger": {
        "note": "muon→νμ, electron (no muon)→νe, else NC; p_min=20 MeV/c",
        "n_tagged_nue":  n_pid_nue,
        "n_tagged_numu": n_pid_numu,
        "n_tagged_nc":   n_pid_nc,
        "confusion_diagonal": {
            "nue":  _diag_frac(cm_pid_norm, 0),
            "numu": _diag_frac(cm_pid_norm, 1),
            "nc":   _diag_frac(cm_pid_norm, 2),
        },
    },
    "pandora_confusion_diagonal": {
        "nue":  _diag_frac(cm_norm_p, 0),
        "numu": _diag_frac(cm_norm_p, 1),
        "nc":   _diag_frac(cm_norm_p, 2),
    },
    "cc_only_pid": {
        "note": "CC events only (νe CC + νμ CC); NC excluded from background",
        "spine_pid": {
            "nue_eff":      float(cm_pid_cc_norm[0, 0]),
            "nue_to_numu":  float(cm_pid_cc_norm[0, 1]),
            "nue_to_nc":    float(cm_pid_cc_norm[0, 2]),
            "numu_eff":     float(cm_pid_cc_norm[1, 1]),
            "numu_to_nue":  float(cm_pid_cc_norm[1, 0]),
            "numu_to_nc":   float(cm_pid_cc_norm[1, 2]),
        },
        "pandora": {
            "nue_eff":      float(cm_pan_cc_norm[0, 0]),
            "nue_to_numu":  float(cm_pan_cc_norm[0, 1]),
            "nue_to_nc":    float(cm_pan_cc_norm[0, 2]),
            "numu_eff":     float(cm_pan_cc_norm[1, 1]),
            "numu_to_nue":  float(cm_pan_cc_norm[1, 0]),
            "numu_to_nc":   float(cm_pan_cc_norm[1, 2]),
        },
        "cc_only_roc": cc_roc_results,
    },
}
with open(os.path.join(outdir, "cvn_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\n  Summary:")
print("  SPINE CVN: not available in this CAF production")
print(f"  SPINE PID tagger: tagged nue={n_pid_nue:,}, numu={n_pid_numu:,}, NC={n_pid_nc:,}")
print(f"  SPINE PID confusion diagonal: "
      f"νe={cm_pid_norm[0,0]:.2f}, νμ={cm_pid_norm[1,1]:.2f}, NC={cm_pid_norm[2,2]:.2f}")
print(f"  Pandora CVN AUC νe={roc_results.get('pandora_cvn_nue',{}).get('auc', float('nan')):.3f}, "
      f"νμ={roc_results.get('pandora_cvn_numu',{}).get('auc', float('nan')):.3f}")
print(f"  Pandora confusion diagonal: "
      f"νe={cm_norm_p[0,0]:.2f}, νμ={cm_norm_p[1,1]:.2f}, NC={cm_norm_p[2,2]:.2f}")
print("\n  CC-only (NC excluded from background):")
print(f"  SPINE PID: νe eff={cm_pid_cc_norm[0,0]:.2f} (inter={cm_pid_cc_norm[0,1]:.2f}, NC-tag={cm_pid_cc_norm[0,2]:.2f}), "
      f"νμ eff={cm_pid_cc_norm[1,1]:.2f} (inter={cm_pid_cc_norm[1,0]:.2f}, NC-tag={cm_pid_cc_norm[1,2]:.2f})")
print(f"  Pandora:   νe eff={cm_pan_cc_norm[0,0]:.2f} (inter={cm_pan_cc_norm[0,1]:.2f}, NC-tag={cm_pan_cc_norm[0,2]:.2f}), "
      f"νμ eff={cm_pan_cc_norm[1,1]:.2f} (inter={cm_pan_cc_norm[1,0]:.2f}, NC-tag={cm_pan_cc_norm[1,2]:.2f})")
print("\nDone!")

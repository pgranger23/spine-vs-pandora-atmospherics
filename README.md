# SPINE vs Pandora Reconstruction Analysis

A comparison of **SPINE (DLP)** and **Pandora** neutrino-event reconstruction on DUNE
FD-HD atmospheric-neutrino CAFs. The analysis covers vertex, angular and energy
resolution, per-particle PID / efficiency / purity, CVN flavour classification, neutrino
direction reconstruction, and a set of focused physics studies (Studies A–R).

- **`REPORT.md`** — the full write-up: every figure, all findings, and the physics
  discussion. Start here to understand *what was found*.
- **This file** — how to set up the environment and *re-run* everything.
- **`srproxy_guide.md`** — reference for the CAF data model and the SRProxy C++/Python
  API used to read it.

> **Handoff note:** this repository contains only code, docs and small reference
> figures. The input CAF (~1.3 GB) and the `duneanaobj` reader (built separately) are
> external dependencies — see [Setup](#setup) below.

---

## Repository layout

```
.
├── README.md                       # this file — setup & run instructions
├── REPORT.md                       # full analysis report (all figures + findings)
├── srproxy_guide.md                # CAF data model + SRProxy API reference
├── requirements.txt                # Python (pip) dependencies
├── plot_style.py                   # shared matplotlib styling (imported by every script)
├── analysis_spine_vs_pandora.py    # MAIN: interaction-level comparison → most figures
├── analysis_per_particle.py        # MAIN: per-particle comparison → per-particle figures
├── investigate_*.py                # individual studies (see the index at the bottom)
└── figures/                        # committed REFERENCE outputs (PNG + JSON) to diff against
```

Scripts write their figures/JSON into `figures/`, overwriting the committed reference
copies in place. Re-running a script and getting the same picture is the quickest sanity
check that your environment is set up correctly.

---

## Setup

Three things must be in place: a ROOT-enabled Python, the pip packages, and a local build
of the `duneanaobj` reader.

### 1. ROOT / PyROOT

The scripts read CAFs through ROOT (PyROOT) and the `duneanaobj` StandardRecord
dictionary, so you need a ROOT install whose Python matches the one you run the scripts
with. At FNAL the easiest route is the DUNE software stack from cvmfs, e.g.

```bash
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup dunesw <version> -q <qualifiers>   # pulls in a matching ROOT + Python
```

Everything here was developed against **Python 3.13** and the ROOT that ships with it.
Whatever ROOT you pick, use the *same* one in step 3.

### 2. Python packages

```bash
pip install -r requirements.txt
```

(`numpy`, `scipy`, `matplotlib`, `polars`, `scikit-learn`, `tqdm`. ROOT is **not** pip —
it comes from step 1.)

### 3. Build `duneanaobj` (the CAF reader) with Python bindings

The scripts import `pySRProxy` and preload the StandardRecord shared libraries from a
local `duneanaobj` build. Use the fork with the Clingification/Python-bindings branch:

```bash
git clone https://github.com/pgranger23/duneanaobj.git
cd duneanaobj
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -Dduneanaobj_PYTHON_ENABLED=ON
make -j$(nproc)
cd ../..
```

This must sit at `./duneanaobj/` inside this repository (it is git-ignored). SRProxy is
fetched automatically at configure time via CPM. See `srproxy_guide.md` for details on
the C++ side.

> **⚠️ Platform path — read this if you are on Linux (e.g. FNAL).**
> The scripts locate the build under a hard-coded platform folder and preload macOS
> library names:
> ```python
> lib_path = os.path.join(base_dir, "duneanaobj/build/Linux/lib")
> py_path  = os.path.join(base_dir, "duneanaobj/build/Linux/python/3.13")
> _dylibs_to_preload = ["libSRProxy_BasicTypes.dylib",
>                       "libduneanaobj_StandardRecord.dylib",
>                       "libduneanaobj_StandardRecordProxy.dylib"]
> ```
> Adjust `lib_path` / `py_path` to match your actual build output directory (the folder
> name is set by CMake for your system), and if your Python is not 3.13 update the
> `python/3.X` segment. On Linux the libraries are `.so`, not `.dylib`; the preload is
> wrapped in try/except and only prints a warning if it fails, so if `import pySRProxy`
> still succeeds (RPATH resolves the libs) you can ignore the warnings — otherwise change
> the `.dylib` names to `.so`. This block is identical at the top of every script.

### 4. Get the input CAF

The single input file for every script is a merged SPINE-reprocessed CAF. At FNAL:

```
/exp/dune/data/users/pgranger/spine_merged_output/spine_merged_output_all.root
```

Every script opens it under the hard-coded name **`spine_merged_reprocess.root`** in the
repository root. The simplest thing is to symlink it so you never edit the scripts:

```bash
ln -s /exp/dune/data/users/pgranger/spine_merged_output/spine_merged_output_all.root \
      spine_merged_reprocess.root
```

(Or copy it locally under that name. If you keep a different name, update the
`StandardRecordFileReader(...)` / `ROOT.TFile.Open(...)` calls — the filename is
hard-coded in each script.)

---

## Running

All scripts are run from the repository root (they use `os.getcwd()` to find the
`duneanaobj` build and the CAF) and take no command-line arguments:

```bash
# the two main analyses first — they produce the bulk of the figures
python3 analysis_spine_vs_pandora.py
python3 analysis_per_particle.py

# then any individual study, e.g.
python3 investigate_cvn_performance.py      # Study A: CVN ROC
python3 investigate_topology_performance.py # Study B: performance by interaction mode
```

Each script prints progress (a `tqdm` bar over CAF entries) and writes its outputs into
`figures/`. Compare against the committed reference PNGs/JSON to confirm your setup.

---

## Known gotchas

- **Hard-coded input filename** (`spine_merged_reprocess.root`) and **platform build
  path** (`duneanaobj/build/Linux/...`) are repeated verbatim at the top of every script.
  Fix them once per environment (symlink for the data; edit `lib_path`/`py_path` for the
  build) — see Setup steps 3 and 4.
- **Two reader backends.** Some scripts read via `pySRProxy`, others via `ROOT.TFile`
  directly (both need the `duneanaobj` dictionary from step 3). The table below marks
  which is which.
- **`nu.mode` is not readable from Python** in this CAF (unregistered enum); studies that
  need interaction mode approximate it — see the relevant sections in `REPORT.md`.
- Several SPINE quantities need corrections applied in-script (e.g. `p.E` is kinetic
  energy; EM momenta carry an ADC→electron ×0.198 factor). These are documented in
  `REPORT.md` under *Key Findings* and applied by the scripts themselves.

---

## Script index

`R` = reads via PyROOT (`ROOT.TFile`), `S` = reads via `pySRProxy`.

| Script | Backend | What it does |
|--------|:------:|--------------|
| `analysis_spine_vs_pandora.py` | S | **Main.** Interaction-level SPINE vs Pandora: vertex/angular/energy resolution, multiplicity, CVN, mode breakdown, correlations (Figs 1–9). |
| `analysis_per_particle.py` | S | **Main.** Per-particle comparison: PID confusion, angular/momentum/energy resolution, purity/completeness, ghosts & splits (Figs 10–21). |
| `investigate_angular_2d.py` | R | Study K — 2D per-axis angular resolution and zenith up/down confusion. |
| `investigate_cvn_performance.py` | S | Study A — CVN flavour-classifier ROC curves and working points. |
| `investigate_direction_best.py` | R | Study R — best-of momentum-sum vs calorimetric direction, and the crossover. |
| `investigate_direction_budget.py` | R | Study M — direction-resolution error budget: where to invest. |
| `investigate_direction_decomposition.py` | S | Study H — neutrino direction decomposition, low-energy crossover. |
| `investigate_direction_gap.py` | S | Deep dive: why SPINE direction lags Pandora at low energy. |
| `investigate_electron_energy_resolution.py` | S | Electron energy resolution vs true energy. |
| `investigate_em_bias.py` | S | Root cause of SPINE electron momentum over-estimation. |
| `investigate_em_ratio.py` | S | Quantifies the ~4.76× SPINE electron momentum ratio (ADC→e calibration). |
| `investigate_hadronic_correction.py` | S | Hadronic energy correction parameterisation. |
| `investigate_hadronic_interactions.py` | R | Study Q — fate of hadronically-interacting protons; PID-metric bias. |
| `investigate_missing_breakdown.py` | R | Study N — which missed particles matter for direction (species/momentum). |
| `investigate_multiplicity.py` | R+S | Study I — reconstructible truth vs reco multiplicity (with near-vertex secondaries). |
| `investigate_neutron_activity.py` | R | Study J — displaced neutron-induced visible activity. |
| `investigate_neutron_islands.py` | S | Neutron spatial distribution and clustering. |
| `investigate_paper_method.py` | S | Direction estimators from arXiv:2601.05697 (Sec 3.4.1) applied to SPINE. |
| `investigate_particle_effpur.py` | R | Study L — per-particle efficiency / purity / overlap vs energy. |
| `investigate_particle_pid_cc.py` | S | Particle-level PID confusion, CC-only, full sample. |
| `investigate_pi0_reconstruction.py` | S | Study F — π0 photon-pair reconstruction. |
| `investigate_pid_particle_energy.py` | R | Study P — is PID topology dependence real or a particle-energy-spectrum effect? |
| `investigate_pid_topology.py` | R | Study O — PID vs energy & topology; final-state composition bias. |
| `investigate_proton_cutoff.py` | S | Proton momentum cutoff scan. |
| `investigate_qe_kinematics.py` | S | Study E — QE-like kinematic closure test. |
| `investigate_reconstruction_efficiency.py` | S | Study G — particle reco efficiency vs true kinetic energy. |
| `investigate_reversal_and_vtxdir.py` | S | Study H2 — direction reversal rate and vertex–direction correlation. |
| `investigate_topology_performance.py` | S | Study B — performance stratified by interaction mode (QE/RES/DIS/MEC). |
| `investigate_vertex_boundary.py` | S | Vertex resolution degradation at detector boundaries. |

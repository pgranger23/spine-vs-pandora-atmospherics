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
├── caf_env.py                      # shared env setup: resolves the CAF path + duneanaobj build
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

SRProxy is fetched automatically at configure time via CPM. See `srproxy_guide.md` for
details on the C++ side.

All scripts locate this build through **`caf_env.py`**, which auto-detects the platform
build folder (`build/Linux` or `build/Darwin`) and the correct library extension
(`.so` on Linux, `.dylib` on macOS). If your build lives somewhere else — a different
path, or your Python is not 3.13 — point `caf_env` at it with one environment variable:

```bash
export DUNEANAOBJ_BUILD=/path/to/duneanaobj/build/Linux   # contains lib/ and python/<X.Y>/
```

The default (with no variable set) is `./duneanaobj/build/<System>/`, so if you cloned and
built at `./duneanaobj/` as above, it just works.

### 4. Get the input CAF

The single input file for every script is a merged SPINE-reprocessed CAF. At FNAL:

```
/exp/dune/data/users/pgranger/spine_merged_output/spine_merged_output_all.root
```

Tell `caf_env` where it is with one environment variable:

```bash
export SPINE_CAF=/exp/dune/data/users/pgranger/spine_merged_output/spine_merged_output_all.root
```

(If `SPINE_CAF` is unset, the scripts look for `spine_merged_reprocess.root` in the
repository root — so a symlink of that name is a fine alternative to the variable.)

---

## Running

All scripts are run from the repository root and take no command-line arguments. Set the
two environment variables once (see Setup steps 3–4) and every script picks them up via
`caf_env`:

```bash
export SPINE_CAF=/exp/dune/data/users/pgranger/spine_merged_output/spine_merged_output_all.root
export DUNEANAOBJ_BUILD=$PWD/duneanaobj/build/Linux   # omit if you built at the default location

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

- **Configuration is centralised in `caf_env.py`.** The input CAF path and the
  `duneanaobj` build location come from the `SPINE_CAF` and `DUNEANAOBJ_BUILD` environment
  variables (with sensible auto-detected defaults). You should never need to edit a path
  inside an analysis script — set those two variables once. Run scripts from the
  repository root so the defaults resolve.
- **Two reader backends.** Some scripts read via `pySRProxy` (`caf_env.setup_pysrproxy()`),
  others via `ROOT.TFile` (`caf_env.setup_root()`); both need the `duneanaobj` dictionary
  from step 3. The table below marks which is which.
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

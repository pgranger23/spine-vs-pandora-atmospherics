# DUNE CAF Reading with SRProxy — C++ Guide

A reference for reading DUNE Common Analysis Files (CAFs) in compiled C++ using
the `StandardRecordProxy` interface from `duneanaobj`.

---

## 1. Build and Installation

### Prerequisites

| Requirement | Minimum version |
|-------------|----------------|
| CMake | 3.20 |
| ROOT | 6.20 |
| C++ standard | C++17 |

### Clone duneanaobj

```bash
git clone https://github.com/pgranger23/duneanaobj.git
cd duneanaobj
```

SRProxy is pulled automatically at configure time via
[CPM](https://github.com/cpm-cmake/CPM.cmake).  No manual clone is needed.
CMake fetches the fork at `pgranger23/SRProxy`, branch
`feature/Clingification_rebase3`.

### Configure and build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)
```

The Python bindings option (`duneanaobj_PYTHON_ENABLED`) is `OFF` by default.
Do not turn it on for a C++-only setup.

### Output libraries

After a successful build, the following shared libraries are in
`build/<platform>/lib/`:

```
libduneanaobj_StandardRecord.dylib      # the StandardRecord class dictionary
libduneanaobj_StandardRecordProxy.dylib # the SRProxy layer
libduneanaobj_StandardRecordFlat.dylib  # flat CAF variant
```

### Runtime environment

```bash
export SRPROXY_LIB=/path/to/duneanaobj/build/<platform>/lib
export LD_LIBRARY_PATH="$SRPROXY_LIB:$LD_LIBRARY_PATH"
# macOS: DYLD_LIBRARY_PATH instead of LD_LIBRARY_PATH
```

### Compiling your program

```bash
g++ -std=c++17 my_analysis.cxx \
    -I/path/to/duneanaobj/build/<platform>/include \
    $(root-config --cflags --libs) \
    -L$SRPROXY_LIB \
    -lduneanaobj_StandardRecord \
    -lduneanaobj_StandardRecordProxy \
    -o my_analysis
```

---

## 2. Opening a CAF and Creating the Proxy

```cpp
#include "duneanaobj/StandardRecord/Proxy/SRProxy.h"
#include "duneanaobj/StandardRecord/StandardRecord.h"
#include "TFile.h"
#include "TTree.h"

using namespace caf;

// Open file and locate tree (tree name varies by production)
TFile fin("my_caf.root", "READ");
TTree* tin = static_cast<TTree*>(fin.Get("recTree"));
if (!tin) tin = static_cast<TTree*>(fin.Get("cafmaker/cafTree"));
if (!tin) { /* handle error */ }

// Construct the proxy — second argument is the top-level branch name
StandardRecordProxy srp(tin, "rec");

for (Long64_t iev = 0; iev < tin->GetEntries(); ++iev) {
    tin->GetEntry(iev);
    // srp now reflects entry iev
}
```

`StandardRecordProxy` reads each branch lazily: only the sub-branches you
actually access are deserialised for a given entry.

---

## 3. StandardRecord Layout

```
rec
├── common                  shared reconstruction block
│   └── ixn                 interaction candidates for this spill entry
│       ├── npandora         number of Pandora interactions
│       ├── pandora[i]       i-th Pandora interaction
│       ├── ndlp             number of DLP (SPINE) interactions
│       └── dlp[i]           i-th DLP interaction
├── fd                      FD-specific block
│   └── hd                  high-definition FD reconstruction
│       └── pandora[i]       FD per-particle info for Pandora interaction i
└── mc                      Monte Carlo truth
    └── nu[i]               i-th true neutrino
        ├── prim[j]          j-th truth primary particle
        ├── nprim
        ├── iscc             true if charged-current
        └── vtx              true neutrino vertex (mm)
```

---

## 4. Pandora Reconstruction

### 4a. Interaction-level access

```cpp
auto& ixn = srp.common.ixn;
size_t npandora = ixn.npandora;   // implicit conversion to integer
if (npandora == 0) continue;

auto& pan = ixn.pandora.at(0);    // take the first (closest) interaction

// Reconstructed vertex (mm)
double vx = pan.vtx.x;
double vy = pan.vtx.y;
double vz = pan.vtx.z;

// Event-level direction estimators
// lngtrk: direction of the longest track (best for nu_mu CC)
// heshw:  direction of the highest-energy shower (best for nu_e CC)
double dx_lngtrk = pan.dir.lngtrk.x;   // pre-normalised unit vector
double dx_heshw  = pan.dir.heshw.x;

// Calorimetric neutrino energy estimate
double enu_calo = pan.Enu.calo;         // GeV (check isfinite before use)

// CVN flavor scores (sum ≈ 1)
double p_nue  = pan.nuhyp.cvn.nue;
double p_numu = pan.nuhyp.cvn.numu;
double p_nc   = pan.nuhyp.cvn.nc;      // can be entirely NaN when CVN has insufficient hits; treat whole CVN output as unavailable in that case
```

### 4b. Per-particle access

Pandora reconstructs Particle Flow Objects (PFOs).  The particle list lives
inside the interaction block; the corresponding FD reconstruction info (track
and shower directions, energies) is stored in a **parallel array** under
`rec.fd.hd.pandora`.  Both arrays must be indexed with the same index `pi`.

```cpp
auto& pand_parts = ixn.pandora.at(0).part;   // interaction-level particle list
auto& pand_fd    = srp.fd.hd.pandora.at(0);  // FD per-particle block

// The last entry in pand_parts is a sentinel for unassociated hits; stop at n-1
size_t npand = pand_parts.pandora.size() - 1;

for (size_t pi = 0; pi < npand; ++pi) {
    auto& p = pand_parts.pandora.at(pi);

    if (!p.primary) continue;   // skip non-primary PFOs

    // PDG in this CAF production is binary:
    //   PDG 11 → shower-like PFO
    //   PDG 13 → track-like PFO
    int pdg = p.pdg;

    // Direction: prefer shower direction, fall back to track direction
    auto& sd  = pand_fd.showers.at(pi).direction;
    double sdn = std::sqrt(sd.x*sd.x + sd.y*sd.y + sd.z*sd.z);
    double dx, dy, dz;
    if (sdn > 1e-6) {
        dx = sd.x/sdn;  dy = sd.y/sdn;  dz = sd.z/sdn;
    } else {
        auto& td  = pand_fd.tracks.at(pi).dir;
        double tdn = std::sqrt(td.x*td.x + td.y*td.y + td.z*td.z);
        if (tdn < 1e-6) continue;   // no valid direction
        dx = td.x/tdn;  dy = td.y/tdn;  dz = td.z/tdn;
    }

    // Energy — cascade of fallbacks (fields are sparsely populated)
    double E_mev = p.E;
    if (!(E_mev > 0)) E_mev = pand_fd.tracks.at(pi).E;
    if (!(E_mev > 0)) E_mev = pand_fd.tracks.at(pi).Evis;
    if (!(E_mev > 0)) E_mev = pand_fd.showers.at(pi).Evis;
    // E_mev is reliable only if positive; otherwise no energy measurement
}
```

**Pandora pitfalls:**

- PDG is binary (11 or 13 only).  No per-species hadron identification.
- `pand_parts.pandora.size()` includes a sentinel; iterate to `size() - 1`.
- Direction vectors (`sd.direction`, `td.dir`) may be zero-length; always
  check the magnitude before normalising.
- Energy fields for track-type PFOs use the muon mass internally, making
  `E_reco/E_true` meaningless for true pions and protons.
- `p_nc` in the CVN scores can be entirely NaN when the CVN has insufficient hits to run. There is no safe fallback — treat the entire CVN output as unavailable for that event.

---

## 5. DLP / SPINE Reconstruction

```cpp
auto& ixn = srp.common.ixn;
if (ixn.ndlp == 0) continue;

// Pick the DLP interaction closest to the true neutrino vertex
auto& dlp = ixn.dlp.at(0);   // or find closest by distance to nu.vtx

// Reconstructed vertex
double vx = dlp.vtx.x;

// Number of reconstructed primaries (no sentinel — iterate to ndlp)
size_t npart = dlp.part.ndlp;

for (size_t pi = 0; pi < npart; ++pi) {
    auto& p = dlp.part.dlp.at(pi);

    // Full per-species PDG (11, 13, 211, 2212, 22, etc.)
    int pdg = p.pdg;

    // Momentum 3-vector (MeV/c)
    double px = p.p.x;
    double py = p.p.y;
    double pz = p.p.z;
    double pmag = std::sqrt(px*px + py*py + pz*pz);

    // Kinetic energy [MeV] (note: p.E stores KE, not total energy)
    double KE_mev = p.E;

    // Total energy [GeV]: KE + rest mass
    // You must supply the mass for the PDG code yourself
}
```

**DLP/SPINE notes:**

- `p.E` stores **kinetic energy**, not total energy.  Convert with
  `E_total_GeV = p.E * 0.001 + mass_GeV`.
- The momentum vector `p.p` is in MeV/c.  For electromagnetic particles
  (e, γ, π⁰) apply the ADC-to-electron calibration factor **× 0.198** before
  using reconstructed momenta; without it the EM scale is overestimated by
  a factor of ~5.
- `dlp.dir.part_mom_sum` is not populated in all CAF productions (may return
  NaN/zero).  Compute the event direction manually from the sum of `p.p`
  vectors instead.
- `dlp.preselected` is not set in all productions; implement a fiducial volume
  cut yourself if needed.
- The DLP particle list does **not** have a sentinel entry; iterate to `ndlp`
  (not `ndlp - 1`).

---

## 6. Monte Carlo Truth

```cpp
auto& mc = srp.mc;
if (mc.nnu == 0) continue;

auto& nu = mc.nu.at(0);

bool is_cc = nu.iscc;
int  nu_pdg = nu.pdg;            // 12, 14, -12, -14, ...

// True neutrino vertex (mm, same coordinate system as reco)
double vtx_x = nu.vtx.x;

// True primary particles
for (size_t ti = 0; ti < nu.nprim; ++ti) {
    auto& tp   = nu.prim.at(ti);
    int   pdg  = tp.pdg;
    double px  = tp.p.px;   // GeV/c
    double py  = tp.p.py;
    double pz  = tp.p.pz;
}
```

---

## 7. Worked Skeleton

```cpp
#include "duneanaobj/StandardRecord/Proxy/SRProxy.h"
#include "TFile.h"
#include "TTree.h"
#include <cmath>
#include <iostream>

using namespace caf;

int main(int argc, char** argv) {
    TFile fin(argv[1], "READ");
    TTree* tin = static_cast<TTree*>(fin.Get("recTree"));
    if (!tin) tin = static_cast<TTree*>(fin.Get("cafmaker/cafTree"));

    StandardRecordProxy srp(tin, "rec");

    for (Long64_t iev = 0; iev < tin->GetEntries(); ++iev) {
        tin->GetEntry(iev);

        // Truth
        if (srp.mc.nnu == 0) continue;
        auto& nu = srp.mc.nu.at(0);

        // Pandora interaction
        auto& ixn = srp.common.ixn;
        if (ixn.npandora == 0) continue;
        auto& pan = ixn.pandora.at(0);

        // CVN score — p_nc can be entirely NaN when CVN has insufficient hits
        double p_numu = pan.nuhyp.cvn.numu;
        double p_nue  = pan.nuhyp.cvn.nue;
        double p_nc   = pan.nuhyp.cvn.nc;

        // Per-particle loop
        auto& pand_parts = pan.part;
        auto& pand_fd    = srp.fd.hd.pandora.at(0);
        size_t npand = pand_parts.pandora.size() - 1;  // exclude sentinel

        for (size_t pi = 0; pi < npand; ++pi) {
            auto& p = pand_parts.pandora.at(pi);
            if (!p.primary) continue;

            auto& sd = pand_fd.showers.at(pi).direction;
            double sdn = std::sqrt(sd.x*sd.x + sd.y*sd.y + sd.z*sd.z);
            if (sdn < 1e-6) continue;

            std::cout << "  PDG " << p.pdg
                      << " dir (" << sd.x/sdn << ", " << sd.y/sdn << ", " << sd.z/sdn << ")\n";
        }
    }
    return 0;
}
```

---

## 8. Known Limitations of This CAF Production

| Issue | Detail |
|-------|--------|
| Pandora PDG is binary | Only PDG 11 (shower) or 13 (track); no hadron species ID |
| Pandora particle energies sparse | Use the four-step fallback cascade; many fields are 0 or NaN |
| Pandora uses muon mass for all tracks | `E_reco/E_true` is meaningless for pions/protons |
| Pandora sentinel particle | `pandora.size()` includes an extra entry; iterate to `size() - 1` |
| DLP `p.E` is kinetic energy | Add rest mass manually to get total energy |
| DLP EM scale | Apply × 0.198 ADC-to-electron factor for e, γ, π⁰ |
| DLP `dir.part_mom_sum` may be zero | Build the momentum sum from individual `p.p` vectors |
| DLP `preselected` may be unset | Implement fiducial volume cut explicitly |

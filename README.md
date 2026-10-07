# miniPET

Load and work with MiniPET DAQ products from `mpdaq` / `mpeventproc`.

## Load individual files

### Singles list mode (`.slm.lr5`)

```python
from minipet import SLMFile

slm = SLMFile('run4/run4_260922-153612.slm.lr5')
singles = slm.singles()          # time-sorted; fields t, module, x, y, energy
slm.duration_s
flood = slm.flood()              # (12, 256, 256) Anger histograms
flood0 = slm.flood(module=0)     # one module
```

### Coincidence list mode (`.clm.lr5`)

```python
from minipet import CLMFile

clm = CLMFile('run4/run4_260922-153612.clm.lr5')
coinc = clm.coincidences()                 # prompts + delayed
prompts = clm.coincidences(delayed=False)
delayed = clm.coincidences(delayed=True)
times = clm.prompt_times()                 # clock ticks of T_CLK_NS
pairs = clm.pair_index(prompts)            # P0..P17 indices
clm.duration_s
```

### LOR histograms

```python
from minipet import LOR2DFile, LOR3DFile

lor2d = LOR2DFile('run4/run4_260922-153612.2dlor.lr5')
lor2d.data                    # (35, 22050)
lor2d.slice_thickness_mm

lor3d = LOR3DFile('run4/run4_260922-153612.3dlor.lr5')
lor3d.data                    # flat length N_LOR_3D
lor3d.total_counts, lor3d.randoms
g1, g2, counts = lor3d.crystal_pairs()   # nonzero bins
```

### Rates (`.rate.csv`)

```python
from minipet import read_rate, module_singles_cps, pair_rates_cps

rate = read_rate('run4/run4_260922-153612.rate.csv')  # DataFrame, rates in kcps
singles_cps = module_singles_cps(rate)                # length 12, in cps
prompts_cps, randoms_cps = pair_rates_cps(rate)       # length 18 each
```

### Crystal singles counts (`.scnt.csv`)

```python
from minipet import read_scnt

scnt = read_scnt('run4/run4_260922-153612.scnt.csv')  # (12, 1225)
```

### Coincidence spectra (`.csp.csv`)

```python
from minipet import CSPFile, read_csp, csp_dt_ns

csp = CSPFile('calibcointimetest_260922-160413.csp.csv')
csp.spectra          # (18, 401)
csp.x_ns             # time-difference axis in ns

spectra, headers = read_csp('calibcointimetest_260922-160413.csp.csv')
dt_ns = csp_dt_ns()
```

### Recon / sinogram (`.mnc`)

```python
import nibabel as nib

sino = nib.load('run4/run4_260922-153612.sino.mnc').get_fdata()
recon = nib.load('run4/run4_260922-153612.mnc').get_fdata()

# Or via the folder loader:
from minipet import MiniPET
mp = MiniPET('run4', kinds=('sino', 'recon'))
mp.sino, mp.recon
```

## Crystal assignment (Anger → crystal)

Raw `(x, y)` in list-mode are Anger positions, not crystal indices. Build a LUT from a flood acquisition, then map:

```python
from minipet import SLMFile, assign_crystals
from petlab import build_crystal_lut

slm = SLMFile('run4/run4_260922-153612.slm.lr5')
lut = build_crystal_lut(slm)                 # (12, 256, 256)
with_crystals = slm.singles_with_crystals(lut)

# Or manually:
s = slm.singles()
crystal = assign_crystals(s['module'], s['x'], s['y'], lut)
```

## Constants

Useful scanner constants from `minipet`:

```python
from minipet import (
    T_CLK_NS,      # 0.667 ns clock period
    N_MODULES,     # 12
    N_CRYS,        # 1225
    COINC_PAIRS,   # 18 (module_i, module_j) pairs
    PAIR_INDEX,    # module pair → P0..P17
)
```

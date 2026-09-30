"""Helpers for the CM2020 MiniPET lab notebooks.

Everything the students are *given* lives here, so the notebooks can stay
focused on the tasks:

* scanner geometry (crystal coordinates, module pairs / fan),
* a common in-memory format for singles (``SINGLES_DTYPE``) and loaders for
  both the real acquisitions (``load_lab_acquisitions`` /
  ``load_real_acquisition``) and the simulator,
* the raw-data calibrations the real scanner needs before its singles can be
  used: the flood crystal look-up table and the ADC energy scale,
* a simple Monte-Carlo simulator that produces acquisitions in the same
  format, for testing and for tasks that have no real acquisition yet,
* the image grid for the reconstruction (the projectors are in
  :mod:`minipet_projector`).

The DAQ file formats live in :mod:`minipet`; this module only adds the lab
layer on top.

Coordinates are in mm, with the origin at the centre of the scanner and z
along the axis. Times are in ns unless a name says ``ticks``.

Crystal numbering: a crystal is identified by ``(module, crystal)`` with
``crystal = axial_row * 35 + tangential_index``, or by the global id
``g = module * 1225 + crystal``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import minipet
from minipet import (
    COINC_PAIRS,
    CSP_ZERO_BIN,
    N_CRYS,
    N_MODULES,
    N_POS,
    N_SIDE,
    PAIR_INDEX,
    T_CLK_NS,
    MiniPET,
)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Scanner constants
# ---------------------------------------------------------------------------

N_RINGS = N_SIDE                  # axial crystal rows
CRYSTAL_WIDTH_MM = 1.27
CRYSTAL_PITCH_MM = 1.347          # Delta z = d
CRYSTAL_LENGTH_MM = 12.0          # TODO: check against the scanner spec
FACE_RADIUS_MM = 105.5            # centre -> module face (from minipet_geom)
AXIAL_FOV_MM = N_SIDE * CRYSTAL_PITCH_MM
C_MM_PER_NS = 299.792458

# Module pairs in coincidence (coincRelation = 3), in mpdaq order P0..P17.
# Always k < l, so dt = t_k - t_l is the fixed module ordering.
PAIRS = np.array(COINC_PAIRS)
N_PAIRS = len(PAIRS)
ALLOWED = PAIR_INDEX >= 0

# Common format for singles in the notebooks
SINGLES_DTYPE = np.dtype([
    ('t', '<i8'),         # time stamp in clock ticks (T_clk)
    ('module', 'u1'),     # 0..11
    ('crystal', '<u2'),   # 0..1224 = axial_row * 35 + tangential_index
    ('energy', '<f4'),    # keV
])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def crystal_xyz(flat: bool = True) -> np.ndarray:
    """Crystal coordinates in mm, centred on the scanner.

    Returns shape ``(12 * 1225, 3)`` (index = global id g) or ``(12, 1225, 3)``.
    Read from ``minipet_crystals.npz`` (extracted from the lor2pp geometry
    template ``minipet_geom.x{start,end}.bin``).
    """
    xyz = np.load(HERE / 'minipet_crystals.npz')['xyz'].astype(np.float64)
    xyz[..., 2] -= xyz[..., 2].mean()
    return xyz.reshape(-1, 3) if flat else xyz


def crystal_ring(crystal):
    """Axial row (ring) index 0..34 of a crystal (within its module)."""
    return np.asarray(crystal) // N_SIDE


def module_frames():
    """Per-module face centre, normal, tangential and axial unit vectors."""
    xyz = crystal_xyz(flat=False)
    centre = xyz.mean(axis=1)
    normal = centre.copy()
    normal[:, 2] = 0
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    tang = xyz[:, 1] - xyz[:, 0]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    axial = xyz[:, N_SIDE] - xyz[:, 0]
    axial /= np.linalg.norm(axial, axis=1, keepdims=True)
    return centre, normal, tang, axial


# ---------------------------------------------------------------------------
# Coincidences
# ---------------------------------------------------------------------------

def clm_lors(clm, lut, delayed=False):
    """Global crystal ids ``(g1, g2)`` of the coincidences in a ``.clm.lr5``.

    ``delayed=False`` gives the prompts, ``True`` the delayed-window events.
    The raw Anger positions are mapped to crystals with the flood ``lut``
    of :func:`build_crystal_lut`.
    """
    clm = clm if isinstance(clm, minipet.CLMFile) else minipet.CLMFile(clm)
    c = clm.coincidences(delayed=delayed)
    g = []
    for end in ('1', '2'):
        m = c['module' + end]
        crystal = minipet.assign_crystals(m, c['x' + end], c['y' + end], lut)
        g.append(m.astype(np.int64) * N_CRYS + crystal)
    return g[0], g[1]


# ---------------------------------------------------------------------------
# Singles I/O
# ---------------------------------------------------------------------------

def singles_rate_per_crystal(slm, lut) -> np.ndarray:
    """Singles rate of every crystal in counts per second, shape ``(12, 1225)``.

    All singles are counted, with no energy selection, as in the coincidence
    sorting of the DAQ. ``slm`` is a :class:`minipet.SLMFile`.
    """
    raw = slm.singles(sort=False)
    g = raw['module'].astype(np.int64) * N_CRYS + minipet.assign_crystals(
        raw['module'], raw['x'], raw['y'], lut)
    counts = np.bincount(g, minlength=N_MODULES * N_CRYS)
    return (counts / slm.duration_s).reshape(N_MODULES, N_CRYS)


def save_singles(path, singles, meta: dict):
    np.savez_compressed(path, singles=singles, meta=json.dumps(meta))


def load_singles(path):
    """Load a singles file in the lab format.

    Returns ``(singles, meta)``: a structured array with ``SINGLES_DTYPE``
    and a dict with at least ``T_clk_ns`` and ``T_acq_s``.
    """
    path = Path(path)
    if path.name.endswith('.slm.lr5'):
        return singles_from_slm(path)
    with np.load(path) as f:
        return f['singles'], json.loads(str(f['meta']))


# ---------------------------------------------------------------------------
# Raw-data calibrations for the real scanner
#
# The DAQ stores an Anger position pair (x, y) and an 8-bit ADC channel per
# single, not a crystal index and not keV. Both have to be calibrated before
# the singles can be used, and both calibrations come out of the data itself.
# ---------------------------------------------------------------------------

def flood_grid_1d(profile, n=N_SIDE, smooth=1.2):
    """Fit ``n`` regularly spaced crystal centres to a 1-D flood projection.

    The crystals form a regular lattice in the raw position, so a straight
    line ``centre(i) = a + b * i`` is fitted to the detected peaks rather than
    relying on finding all ``n`` of them. Returns the centres and the peaks
    that were used, for plotting.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    p = gaussian_filter1d(np.asarray(profile, float), smooth)
    pk, _ = find_peaks(p, distance=4, prominence=p.max() * 0.02)
    # the first and last raw channel collect edge pile-up, not a crystal
    pk = pk[(pk > 1) & (pk < len(p) - 2)]
    if len(pk) < 8:
        return np.linspace(0, len(p) - 1, n), pk

    idx = np.round((pk - pk[0]) / np.median(np.diff(pk))).astype(float)
    for _ in range(8):
        A = np.column_stack([np.ones(len(idx)), idx])
        (a, b), *_ = np.linalg.lstsq(A, pk.astype(float), rcond=None)
        idx = np.round((pk - a) / b)

    # slide the n-long lattice to the offset that best explains the peaks
    best, best_cost = None, np.inf
    for shift in range(-8, 9):
        c = a + b * (np.arange(n) - idx.min() + shift)
        if c[0] < -b / 2 or c[-1] > len(p) - 1 + b / 2:
            continue
        cost = np.abs(pk[:, None] - c[None, :]).min(1).mean()
        cost += 0.15 * (abs(c[0]) + abs(c[-1] - (len(p) - 1))) / b
        if cost < best_cost:
            best, best_cost = c, cost
    if best is None:
        best = np.linspace(0, len(p) - 1, n)
    return np.clip(best, 0, len(p) - 1), pk


def flood_peaks(flood, n=N_SIDE):
    """Locate the ``n x n`` crystal spots in one module's flood map.

    A separable lattice fit gives the initial guess, then each spot is
    refined to the local centroid of its cell, which absorbs the pincushion
    distortion of the Anger logic. Returns ``(n, n, 2)`` positions indexed by
    ``[axial, tangential]``.
    """
    from scipy.ndimage import gaussian_filter

    H = np.asarray(flood, float)
    cx, _ = flood_grid_1d(H.sum(1), n)
    cy, _ = flood_grid_1d(H.sum(0), n)
    gx, gy = np.meshgrid(cx, cy, indexing='ij')

    Hs = gaussian_filter(H, 1.0)
    bx = np.r_[0, (cx[1:] + cx[:-1]) / 2, H.shape[0]]
    by = np.r_[0, (cy[1:] + cy[:-1]) / 2, H.shape[1]]
    for i in range(n):
        xs, xe = int(np.ceil(bx[i])), int(np.ceil(bx[i + 1]))
        for j in range(n):
            ys, ye = int(np.ceil(by[j])), int(np.ceil(by[j + 1]))
            blk = Hs[xs:xe, ys:ye]
            if blk.size == 0 or blk.sum() <= 0:
                continue
            wx, wy = blk.sum(1), blk.sum(0)
            gx[i, j] = xs + (wx * np.arange(len(wx))).sum() / wx.sum()
            gy[i, j] = ys + (wy * np.arange(len(wy))).sum() / wy.sum()
    return np.stack([gx, gy], axis=-1)


def crystal_lut_from_flood(flood, n=N_SIDE):
    """``(256, 256)`` look-up table from raw ``(x, y)`` to crystal 0..1224.

    Every raw position is assigned to the nearest crystal spot. Raw ``x`` is
    the axial coordinate and raw ``y`` the tangential one, so the result
    follows the lab convention ``crystal = axial_row * 35 + tangential``.
    """
    from scipy.spatial import cKDTree

    peaks = flood_peaks(flood, n)
    tree = cKDTree(peaks.reshape(-1, 2))
    gx, gy = np.meshgrid(np.arange(flood.shape[0]), np.arange(flood.shape[1]),
                         indexing='ij')
    _, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
    return idx.reshape(flood.shape).astype(np.uint16)


def build_crystal_lut(source, cache=True, force=False, verbose=True):
    """Crystal look-up table ``(12, 256, 256)`` for the whole scanner.

    ``source`` is a :class:`minipet.SLMFile`, a path to one, or a ready
    ``(12, 256, 256)`` array of flood maps. The result is cached next to the
    singles file, because it is the same for every run taken with the same
    detector setup (``sd5``).

    Why this is needed: the DAQ's own map lives in ``mp4.sd5`` on the scanner
    PC and is not shipped with the data, so the lab rebuilds it from the
    intrinsic LYSO background, which floods every crystal uniformly.
    """
    flood = None
    fn = None
    if isinstance(source, np.ndarray):
        flood = source
    else:
        slm = source if isinstance(source, minipet.SLMFile) else minipet.SLMFile(source)
        fn = Path(str(slm.path).split('.slm.lr5')[0] + '.crystal_lut.npy')
        if cache and not force and fn.exists():
            return np.load(fn)
        if verbose:
            print(f'building the crystal LUT from the flood map of {slm.path.name} ...')
        flood = slm.flood()

    lut = np.empty((N_MODULES, N_POS, N_POS), dtype=np.uint16)
    for m in range(N_MODULES):
        lut[m] = crystal_lut_from_flood(flood[m])
        if verbose:
            print(f'\r  module {m + 1}/{N_MODULES}', end='')
    if verbose:
        print()
    if fn is not None and cache:
        np.save(fn, lut)
    return lut


def photopeak_adc(adc, lo=90, hi=240, smooth=3.0):
    """ADC channel of the 511 keV photopeak in a set of singles.

    Returns ``nan`` if there are too few counts to find a peak.
    """
    from scipy.ndimage import gaussian_filter1d

    adc = np.asarray(adc)
    if adc.size < 200:
        return np.nan
    h = np.bincount(adc, minlength=N_POS)[:N_POS].astype(float)
    hs = gaussian_filter1d(h, smooth)
    band = hs[lo:hi]
    if not band.any():
        return np.nan
    k = lo + int(band.argmax())
    # parabolic interpolation on the three channels around the maximum
    if 0 < k < N_POS - 1:
        y0, y1, y2 = hs[k - 1], hs[k], hs[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            k = k + 0.5 * (y0 - y2) / den
    return float(k)


def energy_scale(singles, lut=None, per='module', verbose=True):
    """keV per ADC channel, from the 511 keV photopeak.

    ``per='module'`` returns 12 values, ``per='crystal'`` returns
    ``(12, 1225)`` and falls back to the module value wherever a crystal has
    too few counts. ``lut`` is only needed for ``per='crystal'``.
    """
    mod_peak = np.array([
        photopeak_adc(singles['energy'][singles['module'] == m])
        for m in range(N_MODULES)
    ])
    if verbose:
        print('511 keV photopeak per module (ADC):',
              np.array2string(mod_peak, precision=1))
    mod_scale = 511.0 / mod_peak
    if per == 'module':
        return mod_scale

    if lut is None:
        raise ValueError("per='crystal' needs the crystal LUT")
    scale = np.repeat(mod_scale[:, None], N_CRYS, axis=1)
    for m in range(N_MODULES):
        s = singles[singles['module'] == m]
        c = lut[m, s['x'], s['y']]
        order = np.argsort(c, kind='stable')
        c, e = c[order], s['energy'][order]
        bounds = np.searchsorted(c, np.arange(N_CRYS + 1))
        for j in range(N_CRYS):
            pk = photopeak_adc(e[bounds[j]:bounds[j + 1]])
            if np.isfinite(pk) and pk > 0:
                scale[m, j] = 511.0 / pk
        if verbose:
            print(f'\r  per-crystal energy scale: module {m + 1}/{N_MODULES}', end='')
    if verbose:
        print()
    return scale


def singles_from_slm(path, lut=None, ecal=None, offsets_ns=None, verbose=True):
    """Convert a MiniPET singles list-mode file (``.slm.lr5``) to the lab format.

    The DAQ conventions used here are verified in the :mod:`minipet` docstring:
    the module comes from the packet index, the time stamp is
    ``tscount << 32 | ts`` in ticks of ``T_CLK_NS``, raw ``x`` is axial and
    raw ``y`` tangential.

    ``lut`` is the crystal look-up table (built from this file's own flood map
    if not given) and ``ecal`` the keV-per-ADC scale, either 12 per-module
    values or a ``(12, 1225)`` per-crystal array. ``offsets_ns`` optionally
    *adds* known module offsets to the time stamps, which is how the lab can
    create a miscalibrated acquisition from a scanner that is already
    calibrated.
    """
    slm = path if isinstance(path, minipet.SLMFile) else minipet.SLMFile(path)
    raw = slm.singles()
    if lut is None:
        lut = build_crystal_lut(slm, verbose=verbose)
    if ecal is None:
        ecal = energy_scale(raw, verbose=verbose)
    ecal = np.asarray(ecal, float)

    s = np.empty(len(raw), dtype=SINGLES_DTYPE)
    s['t'] = raw['t']
    s['module'] = raw['module']
    s['crystal'] = minipet.assign_crystals(raw['module'], raw['x'], raw['y'], lut)
    if ecal.ndim == 1:
        s['energy'] = raw['energy'] * ecal[raw['module']]
    else:
        s['energy'] = raw['energy'] * ecal[raw['module'], s['crystal']]

    if offsets_ns is not None:
        offsets_ns = np.asarray(offsets_ns, float)
        s['t'] = s['t'] + np.round(offsets_ns[raw['module']] / T_CLK_NS).astype(np.int64)
        s = s[np.argsort(s['t'], kind='stable')]

    T_acq = slm.duration_s
    meta = dict(T_clk_ns=T_CLK_NS, T_acq_s=T_acq, source=str(slm.path),
                simulated=False, sd5=slm.sd5,
                energy_scale=np.atleast_1d(ecal).mean(),
                injected_offsets_ns=None if offsets_ns is None else offsets_ns.tolist())
    if verbose:
        print(f'{len(s):,} singles in {T_acq:.1f} s from {slm.path.name}')
    return s, meta


def load_real_acquisition(folder, offsets_ns=None, per_crystal_energy=False,
                          verbose=True):
    """Load a real acquisition folder into the lab singles format.

    Returns ``(singles, meta, mp)`` where ``mp`` is the :class:`minipet.MiniPET`
    folder object, so the notebook can also reach ``mp.rate``, ``mp.scnt`` and
    the DAQ's own reconstruction.
    """
    mp = MiniPET(folder, kinds=('slm', 'scnt', 'rate', 'csp', 'clm'))
    if mp.slm is None:
        raise FileNotFoundError(f'no usable .slm.lr5 in {folder}')
    lut = build_crystal_lut(mp.slm, verbose=verbose)
    ecal = None
    if per_crystal_energy:
        ecal = energy_scale(mp.slm.singles(sort=False), lut=lut, per='crystal',
                            verbose=verbose)
    singles, meta = singles_from_slm(mp.slm, lut=lut, ecal=ecal,
                                     offsets_ns=offsets_ns, verbose=verbose)
    return singles, meta, mp


# Default folders for lab acquisitions A and B (relative to this package).
# Both currently point at run8 (the only run with a healthy ``.slm.lr5``).
# Point acqB at a two-source run once one is recorded, then delete
# ``lab_data/acqB.singles.npz`` (or call with ``force=True``) to rebuild.
LAB_FOLDERS = {
    'acqA': 'run8',
    'acqB': 'run8',
}


def _default_injected_offsets(seed: int = 2026) -> np.ndarray:
    """Deterministic module offsets used to miscalibrate an already-calibrated run."""
    rng = np.random.default_rng(seed)
    spread = 3.0  # ns; same scale as DEFAULT_SIM['offset_spread_ns']
    offsets = rng.uniform(-1, 1, N_MODULES) * spread
    return offsets - offsets.mean()


def load_lab_acquisitions(folders=None, cache=True, force=False,
                          inject_offsets_ns='auto', seed=2026,
                          fallback_simulate=True, verbose=True):
    """Load acquisitions A and B for the PET lab notebooks.

    Prefers real MiniPET folders listed in ``folders`` (default
    :data:`LAB_FOLDERS`). The scanner's own time calibration is typically
    already installed, so by default known module offsets are *injected* into
    both acquisitions — that is what Task 1 is meant to recover. Calibrated
    singles are cached under ``lab_data/acq{A,B}.singles.npz``.

    Parameters
    ----------
    folders : dict, optional
        ``{'acqA': path, 'acqB': path}``. Each path is a run folder containing
        a ``.slm.lr5``.
    inject_offsets_ns : 'auto' | None | array of 12
        ``'auto'`` draws a reproducible offset vector; ``None`` leaves the
        time stamps as recorded.
    fallback_simulate : bool
        If a real folder is missing, fall back to :func:`make_lab_data`.

    Returns
    -------
    data : dict
        ``{'acqA': (singles, meta), 'acqB': (singles, meta), 'paths': {...},
        'offsets_ns': array or None, 'mp': {...}}``.
    """
    folders = {k: HERE / Path(v) for k, v in (folders or LAB_FOLDERS).items()}
    cache_dir = HERE / 'lab_data'
    cache_dir.mkdir(exist_ok=True)

    if inject_offsets_ns == 'auto':
        offsets = _default_injected_offsets(seed)
    elif inject_offsets_ns is None:
        offsets = None
    else:
        offsets = np.asarray(inject_offsets_ns, float)

    have_real = all(
        folder.is_dir() and any(folder.rglob('*.slm.lr5'))
        for folder in folders.values()
    )
    if not have_real:
        if not fallback_simulate:
            missing = [str(f) for f, p in folders.items()
                       if not (p.is_dir() and any(p.rglob('*.slm.lr5')))]
            raise FileNotFoundError(
                'lab acquisition folders missing or without .slm.lr5: '
                + ', '.join(missing)
            )
        if verbose:
            print('real acquisitions not found; falling back to simulation')
        paths = make_lab_data(folder=cache_dir, seed=seed, force=force)
        out = {'paths': paths, 'offsets_ns': None, 'mp': {}}
        for name, path in paths.items():
            out[name] = load_singles(path)
        return out

    # Rebuild the cache when it still holds the old simulation, or on force.
    out = {'paths': {}, 'offsets_ns': None if offsets is None else offsets.tolist(),
           'mp': {}}
    for name, folder in folders.items():
        cache_path = cache_dir / f'{name}.singles.npz'
        truth_path = cache_dir / f'{name}.truth.json'
        reuse = False
        if cache and not force and cache_path.exists():
            singles, meta = load_singles(cache_path)
            same_src = Path(meta.get('source', '')).name in {
                p.name for p in folder.rglob('*.slm.lr5')
            }
            same_off = (
                (offsets is None and meta.get('injected_offsets_ns') is None)
                or (offsets is not None
                    and meta.get('injected_offsets_ns') is not None
                    and np.allclose(meta['injected_offsets_ns'], offsets))
            )
            reuse = (not meta.get('simulated', True)) and same_src and same_off
            if reuse:
                if verbose:
                    print(f'{name}: reusing cached {cache_path.name} '
                          f'({len(singles):,} singles)')
                out[name] = (singles, meta)
                out['paths'][name] = cache_path
                continue

        if verbose:
            print(f'{name}: loading real acquisition from {folder} ...')
        singles, meta, mp = load_real_acquisition(
            folder, offsets_ns=offsets, verbose=verbose)
        meta = dict(meta)
        meta['lab_name'] = name
        meta['folder'] = str(folder)
        if cache:
            save_singles(cache_path, singles, meta)
            truth = dict(
                offsets_ns=None if offsets is None else offsets.tolist(),
                source=meta.get('source'),
                folder=str(folder),
                simulated=False,
                T_clk_ns=meta['T_clk_ns'],
                T_acq_s=meta['T_acq_s'],
            )
            truth_path.write_text(json.dumps(truth, indent=1))
            out['paths'][name] = cache_path
        else:
            out['paths'][name] = Path(meta['source'])
        out[name] = (singles, meta)
        out['mp'][name] = mp
    return out


# ---------------------------------------------------------------------------
# Task 1 support: the DAQ's own time-offset calibration
# ---------------------------------------------------------------------------

def csp_design_matrix():
    """The 18 x 13 matrix D of Eq. (8), with a column for the common offset.

    Row ``p`` for pair ``(k, l)`` encodes ``delta_k - delta_l - mu_common``.
    Its null space is one dimensional (the global time shift), which is what
    the extra column absorbs.
    """
    Z = np.zeros((N_PAIRS, N_MODULES + 1))
    for r, (k, l) in enumerate(PAIRS):
        Z[r, k] = 1.0
        Z[r, l] = -1.0
        Z[r, -1] = -1.0
    return Z


def fit_csp_centres(spectra, fit_lo=50, fit_hi=350):
    """Gaussian centre and FWHM of each ``.csp.csv`` spectrum, in ns.

    Returns ``(centres_ns, fwhm_ns, sigma_centre_ns)``; the centres are
    relative to zero time difference.
    """
    from scipy.optimize import curve_fit

    def g(x, a, mu, sigma, b):
        return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b

    spectra = np.asarray(spectra, float)
    x_all = (np.arange(spectra.shape[1]) - CSP_ZERO_BIN) * T_CLK_NS
    centres = np.zeros(N_PAIRS)
    fwhms = np.zeros(N_PAIRS)
    errs = np.zeros(N_PAIRS)
    for p in range(N_PAIRS):
        y = spectra[p]
        sl = slice(fit_lo, fit_hi + 1)
        x, yy = x_all[sl], y[sl]
        p0 = [yy.max(), x[yy.argmax()], 2.0, np.median(yy)]
        popt, pcov = curve_fit(g, x, yy, p0=p0, maxfev=20000)
        centres[p] = popt[1]
        fwhms[p] = 2 * np.sqrt(2 * np.log(2)) * abs(popt[2])
        errs[p] = np.sqrt(abs(pcov[1, 1]))
    return centres, fwhms, errs


def solve_offsets(centres_ns, sigma_ns=None):
    """Least-squares solve of Eq. (8) for the module offsets.

    Returns ``(offsets_ns, common_ns, residual_ns)``; ``offsets_ns`` has 12
    entries with the gauge fixed by the common-offset column.
    """
    Z = csp_design_matrix()
    y = -np.asarray(centres_ns, float).ravel()
    if sigma_ns is not None:
        w = 1.0 / np.asarray(sigma_ns, float).ravel()
        sol, *_ = np.linalg.lstsq(Z * w[:, None], y * w, rcond=None)
    else:
        sol, *_ = np.linalg.lstsq(Z, y, rcond=None)
    residual = np.asarray(centres_ns, float).ravel() + Z @ sol
    return sol[:N_MODULES], float(sol[-1]), residual


def load_csp(path):
    """Read a ``.csp.csv`` file: ``(spectra, dt_ns, headers)``.

    ``spectra`` is ``(18, 401)`` in ``PAIRS`` order and ``dt_ns`` the
    time-difference axis of one bin per ``T_clk``.
    """
    spectra, headers = minipet.read_csp(path)
    return spectra, minipet.csp_dt_ns(), headers


# ---------------------------------------------------------------------------
# Monte-Carlo simulator (stand-in for the real acquisitions)
# ---------------------------------------------------------------------------

DEFAULT_SIM = dict(
    T_clk_ns=T_CLK_NS,         # time-stamp clock period, as on the real DAQ
    sigma_single_ns=0.99,       # per-detector jitter; gives a 3.3 ns CTR, as measured
    offset_spread_ns=3.0,      # module offsets ~ U(-a, a), zero mean
    mu_511_per_mm=1 / 12.0,    # LYSO
    mu_1275_per_mm=1 / 22.0,
    photofraction_511=0.45,
    photofraction_1275=0.20,
    energy_res_511=0.15,       # FWHM / E at 511 keV
    threshold_keV=100.0,       # hardware threshold on singles
    bg_rate_per_crystal=5.0,   # intrinsic 176Lu singles, measured on run8
    bg_energy_keV=(100.0, 1000.0),
    positron_sigma_mm=0.1,
    acollinearity_fwhm_deg=0.5,
    branching_beta_plus=0.90,
)


def _random_directions(rng, n):
    cos_t = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    sin_t = np.sqrt(1 - cos_t ** 2)
    return np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])


def _perturb(rng, v, sigma_rad):
    """Rotate unit vectors v by a small random angle (2D Gaussian, per axis)."""
    a = np.where(np.abs(v[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    e1 = np.cross(v, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(v, e1)
    g = rng.normal(0, sigma_rad, (len(v), 2))
    w = v + g[:, :1] * e1 + g[:, 1:] * e2
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def _detect(rng, p, v, frames, mu, crystal_length):
    """Trace photons (origin p, direction v) to the crystals.

    Returns (mask, module, crystal, path_length_mm) for the detected ones.
    """
    centre, normal, tang, axial = frames
    half = N_SIDE * CRYSTAL_PITCH_MM / 2
    n = len(p)
    module = np.full(n, -1)
    t_face = np.full(n, np.inf)
    for m in range(N_MODULES):
        den = v @ normal[m]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((centre[m] - p) @ normal[m]) / den
        h = p + t[:, None] * v
        a_u = (h - centre[m]) @ tang[m]
        a_w = (h - centre[m]) @ axial[m]
        ok = (den > 0) & (np.abs(a_u) <= half) & (np.abs(a_w) <= half) & (t < t_face)
        module[ok] = m
        t_face[ok] = t[ok]
    hit = module >= 0
    idx = np.nonzero(hit)[0]
    m = module[idx]
    den = np.einsum('ij,ij->i', v[idx], normal[m])
    depth_ray = rng.exponential(1 / mu, len(idx))
    inside = depth_ray * den <= crystal_length
    x = p[idx] + (t_face[idx] + depth_ray)[:, None] * v[idx]
    a_u = np.einsum('ij,ij->i', x - centre[m], tang[m])
    a_w = np.einsum('ij,ij->i', x - centre[m], axial[m])
    iu = np.floor((a_u + half) / CRYSTAL_PITCH_MM).astype(int)
    iw = np.floor((a_w + half) / CRYSTAL_PITCH_MM).astype(int)
    inside &= (iu >= 0) & (iu < N_SIDE) & (iw >= 0) & (iw < N_SIDE)
    det = np.zeros(n, bool)
    det[idx[inside]] = True
    out_m = m[inside]
    out_c = (iw * N_SIDE + iu)[inside]
    out_d = (t_face[idx] + depth_ray)[inside]
    return det, out_m, out_c, out_d


def _deposit(rng, E0, n, photofraction, res511):
    """Deposited energy (keV): photopeak or flat Compton continuum, blurred."""
    edge = E0 * (2 * E0 / 511.0) / (1 + 2 * E0 / 511.0)
    E = np.where(rng.random(n) < photofraction, E0, rng.uniform(0, edge, n))
    sigma = res511 / 2.3548 * np.sqrt(np.maximum(E, 1) * 511.0)
    return E + rng.normal(0, 1, n) * sigma


def simulate_acquisition(sources, T_acq_s, seed=0, offsets_ns=None,
                         chunk=1_000_000, verbose=True, **params):
    """Simulate a MiniPET singles acquisition with 22Na point sources.

    Parameters
    ----------
    sources : list of dict(pos=(x, y, z) mm, activity_Bq=..., diameter_mm=...)
    T_acq_s : acquisition time in s
    offsets_ns : module time offsets (12,); drawn at random if None

    Returns ``(singles, meta, truth)``.
    """
    P = {**DEFAULT_SIM, **params}
    rng = np.random.default_rng(seed)
    frames = module_frames()
    T_clk = P['T_clk_ns']
    if offsets_ns is None:
        offsets_ns = rng.uniform(-1, 1, N_MODULES) * P['offset_spread_ns']
        offsets_ns -= offsets_ns.mean()
    offsets_ns = np.asarray(offsets_ns, float)
    sig_acol = np.radians(P['acollinearity_fwhm_deg']) / 2.3548

    out = []

    def emit(t_ns, m, c, d_mm, E):
        keep = E >= P['threshold_keV']
        t = (t_ns + d_mm / C_MM_PER_NS + offsets_ns[m]
             + rng.normal(0, P['sigma_single_ns'], len(t_ns)))[keep]
        s = np.empty(keep.sum(), dtype=SINGLES_DTYPE)
        s['t'] = np.floor(t / T_clk).astype(np.int64)
        s['module'] = m[keep]
        s['crystal'] = c[keep]
        s['energy'] = E[keep]
        out.append(s)

    n_decays_total = 0
    for src in sources:
        n_dec = rng.poisson(src['activity_Bq'] * T_acq_s)
        n_decays_total += n_dec
        r_src = src.get('diameter_mm', 0.5) / 2
        for start in range(0, n_dec, chunk):
            n = min(chunk, n_dec - start)
            t0 = rng.uniform(0, T_acq_s * 1e9, n)
            # uniform in a sphere of the source diameter
            u = _random_directions(rng, n) * (r_src * rng.random(n) ** (1 / 3))[:, None]
            pos = np.asarray(src['pos'], float) + u
            # 1.275 MeV prompt gamma from every decay
            v = _random_directions(rng, n)
            det, m, c, d = _detect(rng, pos, v, frames, P['mu_1275_per_mm'],
                                   CRYSTAL_LENGTH_MM)
            emit(t0[det], m, c, d,
                 _deposit(rng, 1275.0, det.sum(), P['photofraction_1275'],
                          P['energy_res_511']))
            # beta+ decays: positron range, two back-to-back 511 keV photons
            bp = rng.random(n) < P['branching_beta_plus']
            pa = pos[bp] + rng.normal(0, P['positron_sigma_mm'], (bp.sum(), 3))
            ta = t0[bp]
            v1 = _random_directions(rng, len(pa))
            v2 = _perturb(rng, -v1, sig_acol)
            for vv in (v1, v2):
                det, m, c, d = _detect(rng, pa, vv, frames, P['mu_511_per_mm'],
                                       CRYSTAL_LENGTH_MM)
                emit(ta[det], m, c, d,
                     _deposit(rng, 511.0, det.sum(), P['photofraction_511'],
                              P['energy_res_511']))
        if verbose:
            print(f"  source at {src['pos']}: {n_dec:,} decays")

    # intrinsic LYSO background: independent Poisson singles in every crystal
    n_bg = rng.poisson(P['bg_rate_per_crystal'] * T_acq_s, N_MODULES * N_CRYS)
    g = np.repeat(np.arange(N_MODULES * N_CRYS), n_bg)
    emit(rng.uniform(0, T_acq_s * 1e9, len(g)), g // N_CRYS, g % N_CRYS,
         np.zeros(len(g)), rng.uniform(*P['bg_energy_keV'], len(g)))

    singles = np.concatenate(out)
    singles = singles[np.argsort(singles['t'], kind='stable')]
    meta = dict(T_clk_ns=T_clk, T_acq_s=T_acq_s, simulated=True)
    truth = dict(sources=sources, offsets_ns=offsets_ns.tolist(),
                 n_decays=int(n_decays_total), params=P,
                 crystal_length_mm=CRYSTAL_LENGTH_MM)
    if verbose:
        print(f'  {len(singles):,} singles')
    return singles, meta, truth


def make_lab_data(folder='lab_data', seed=2026, force=False):
    """Create the simulated acquisitions A and B (if not already there).

    The ground truth goes to ``*.truth.json`` next to the singles, for the
    teaching staff.
    """
    folder = HERE / folder
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-1, 1, N_MODULES) * DEFAULT_SIM['offset_spread_ns']
    offsets -= offsets.mean()
    acqs = {
        'acqA': dict(T_acq_s=60.0, sources=[
            dict(pos=(0.6, -0.9, 0.4), activity_Bq=250e3, diameter_mm=0.5)]),
        'acqB': dict(T_acq_s=90.0, sources=[
            dict(pos=(-1.4, 2.1, 1.2), activity_Bq=200e3, diameter_mm=0.5),
            dict(pos=(17.5, 9.0, -3.8), activity_Bq=150e3, diameter_mm=0.5)]),
    }
    for i, (name, cfg) in enumerate(acqs.items()):
        path = folder / f'{name}.singles.npz'
        if path.exists() and not force:
            continue
        print(f'simulating {name} ...')
        singles, meta, truth = simulate_acquisition(
            cfg['sources'], cfg['T_acq_s'], seed=seed + 1 + i, offsets_ns=offsets)
        save_singles(path, singles, meta)
        (folder / f'{name}.truth.json').write_text(json.dumps(truth, indent=1))
    return {name: folder / f'{name}.singles.npz' for name in acqs}


# ---------------------------------------------------------------------------
# Reconstruction grid (the projectors are in minipet_projector)
# ---------------------------------------------------------------------------

class ImageGrid:
    """Voxel grid centred on the scanner. ``shape`` is (nx, ny, nz)."""

    def __init__(self, voxel_mm=(0.4, 0.4, CRYSTAL_PITCH_MM / 2),
                 fov_mm=(100.0, 100.0, AXIAL_FOV_MM)):
        self.voxel = tuple(float(v) for v in voxel_mm)
        self.shape = tuple(int(np.ceil(f / v)) for f, v in zip(fov_mm, self.voxel))
        self.origin = tuple(-(n - 1) / 2 * v for n, v in zip(self.shape, self.voxel))

    def axes(self):
        """Voxel centre coordinates along x, y, z (mm)."""
        return [o + v * np.arange(n) for o, v, n in zip(self.origin, self.voxel, self.shape)]

    def __repr__(self):
        return f'ImageGrid(shape={self.shape}, voxel={self.voxel})'

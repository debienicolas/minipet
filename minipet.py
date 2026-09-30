"""MiniPET DAQ file formats: discover and load the products of an ``mpdaq`` run.

This module knows about the *scanner*, not about the lab exercises; the lab
layer lives in :mod:`petlab`.

Products written by ``mpdaq`` / ``mpeventproc``
----------------------------------------------
``.slm.lr5``    singles list mode, one 8-byte event per detected photon
``.clm.lr5``    coincidence list mode, one 16-byte event per coincidence
``.2dlor.lr5``  direct-plane LOR histogram, (35, 22050)
``.3dlor.lr5``  full LOR histogram, 18 * 1225 * 1225 = 27011250 bins
``.scnt.csv``   singles counts per crystal, 1225 rows x 12 module columns
``.rate.csv``   S / C / R rates, per pair (P0..P17) and per module (D0S..D11S)
``.csp.csv``    18 coincidence time-difference spectra, 401 bins
``.mnc``        reconstructed image; ``.sino.mnc`` the sinogram

Helpers for the lab notebooks
-----------------------------
* :func:`read_rate`, :func:`module_singles_cps`, :func:`pair_rates_cps` —
  ``.rate.csv`` in cps.
* :func:`assign_crystals`, :meth:`SLMFile.singles_with_crystals` — map Anger
  ``(x, y)`` to crystal index given a flood LUT from :mod:`petlab`.
* :class:`MiniPET` prefers repaired ``*.fixed.*`` products when the original
  file is empty.

Conventions verified against the DAQ
------------------------------------
* ``T_CLK_NS = 0.667``: the full singles time stamp is
  ``tscount << 32 | ts`` in units of this clock period. Checked by comparing
  the time-stamp span of a run with its ``lengthSec`` (118.0 s for a 120 s
  acquisition) and against the bin width of the ``.csp.csv`` spectra.
* The singles packet table holds one packet per module per time-stamp epoch,
  in module order, so ``module = packet_index % 12``. Checked two ways: the
  packets group into blocks of 12 sharing one ``tscount`` value, and the
  per-module event counts correlate at 0.991 with ``D0S..D11S`` in
  ``.rate.csv``.
* ``crystal = axial_row * 35 + tangential_index``, and a LOR histogram index
  is ``pair * 1225**2 + crystal_low_module * 1225 + crystal_high_module``.
  Read directly off the ``minipet_geom.x{start,end}.bin`` templates.
* Raw ``(x, y)`` are Anger positions, **not** crystal indices. ``x`` is the
  axial coordinate and ``y`` the tangential one; see
  :func:`petlab.build_crystal_lut`. In ``.slm.lr5`` they are 8 bit; in
  ``.clm.lr5`` they sit in the *low byte* of a 16-bit field whose high byte is
  uninitialised, so they must be masked with ``& 0xFF``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Scanner / DAQ constants
# ---------------------------------------------------------------------------

T_CLK_NS = 0.667                  # singles time-stamp clock period
N_MODULES = 12
N_SIDE = 35                       # crystals per module side
N_CRYS = N_SIDE * N_SIDE          # 1225 crystals per module
N_POS = 256                       # raw Anger position range (8 bit)
RADIUS_MM = 105.5                 # radius of the scanner

# Module pairs in coincidence for coincRelation = 3, in mpdaq order P0..P17.
# Always k < l, which fixes the sign of dt = t_k - t_l.
COINC_PAIRS = (
    (0, 5), (0, 6), (0, 7),
    (1, 6), (1, 7), (1, 8),
    (2, 7), (2, 8), (2, 9),
    (3, 8), (3, 9), (3, 10),
    (4, 9), (4, 10), (4, 11),
    (5, 10), (5, 11),
    (6, 11),
)
N_PAIRS = len(COINC_PAIRS)

# PAIR_INDEX[k, l] = index into COINC_PAIRS, or -1 if (k, l) is never in coincidence
PAIR_INDEX = np.full((N_MODULES, N_MODULES), -1, dtype=int)
for _p, (_k, _l) in enumerate(COINC_PAIRS):
    PAIR_INDEX[_k, _l] = PAIR_INDEX[_l, _k] = _p

# .csp.csv layout: 400 bins requested, odd size forced in detectorPair.cc
CSP_N_BINS = 401
CSP_ZERO_BIN = CSP_N_BINS // 2    # bin of zero time difference

N_LOR_3D = N_PAIRS * N_CRYS * N_CRYS      # 27011250
N_LOR_2D = N_PAIRS * N_SIDE * N_SIDE      # 22050 per direct plane

# Compound suffixes, longest first so '.sino.mnc' wins over '.mnc', etc.
_SUFFIXES = (
    '.sino.mnc',
    '.rate.csv',
    '.scnt.csv',
    '.csp.csv',
    '.2dlor.lr5',
    '.3dlor.lr5',
    '.clm.lr5',
    '.slm.lr5',
    '.mnc',
)

_KIND = {
    '.rate.csv': 'rate',
    '.scnt.csv': 'scnt',
    '.csp.csv': 'csp',
    '.sino.mnc': 'sino',
    '.mnc': 'recon',
    '.clm.lr5': 'clm',
    '.slm.lr5': 'slm',
    '.2dlor.lr5': 'lor2d',
    '.3dlor.lr5': 'lor3d',
}

ALL_KINDS = ('rate', 'scnt', 'csp', 'sino', 'recon', 'clm', 'slm', 'lor2d', 'lor3d')

# Raw event layouts, from the eventDescriptor attributes of the packet tables.
COINC_DTYPE = np.dtype([
    ('d1', '<u2'), ('e1', '<u2'), ('x1', '<u2'), ('y1', '<u2'),
    ('d2', '<u2'), ('e2', '<u2'), ('x2', '<u2'), ('y2', '<u2'),
])

SLM_EVENT_DTYPE = np.dtype([
    ('tscount', 'u1'),
    ('energy', 'u1'),
    ('x', 'u1'),
    ('y', 'u1'),
    ('ts', '<u4'),
])

SINGLES_DTYPE = SLM_EVENT_DTYPE          # backwards-compatible alias

# Decoded singles: what :meth:`SLMFile.singles` returns.
RAW_SINGLES_DTYPE = np.dtype([
    ('t', '<i8'),          # full time stamp in clock ticks of T_CLK_NS
    ('module', 'u1'),      # 0..11, from the packet index
    ('x', 'u1'),           # raw axial Anger position
    ('y', 'u1'),           # raw tangential Anger position
    ('energy', 'u1'),      # raw ADC channel
])

# Decoded coincidences: what :meth:`CLMFile.coincidences` returns.
RAW_COINC_DTYPE = np.dtype([
    ('module1', 'u1'), ('x1', 'u1'), ('y1', 'u1'), ('energy1', 'u1'),
    ('module2', 'u1'), ('x2', 'u1'), ('y2', 'u1'), ('energy2', 'u1'),
    ('delayed', '?'),
])


def _kind_of(name: str) -> str | None:
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            return _KIND[suffix]
    return None


def _decode(v) -> str:
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode('utf-8', 'replace').rstrip('\x00').strip()
    try:
        return b''.join(v).decode('utf-8', 'replace').rstrip('\x00').strip()
    except Exception:
        return str(v)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# HDF5 / .lr5 file hierarchy
# ---------------------------------------------------------------------------

class LR5File:
    """Parent for MiniPET HDF5 (.lr5) products.

    Opens the file once, copies useful contents into attributes, then closes.
    """

    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(self.path, 'r') as f:
            self.keys = list(f.keys())
            self.history = (
                [_decode(h) for h in f['history'][:]] if 'history' in f else []
            )
            self.scan = self._load_scan(f)
            self.sd5 = (
                _decode(f['process/sd5Info'][0]['nameOfSd5'])
                if 'process/sd5Info' in f else ''
            )
            self._load(f)

    def _load_scan(self, f) -> dict:
        acq = f['scan/acquisition'][0]
        study = f['scan/study'][0]
        patient = f['scan/patient'][0]
        return {
            'patient_id': _decode(patient['id']),
            'lengthSec': int(acq['lengthSec']),
            'startTime': int(acq['startTime']),
            'coincTimeWindow': float(acq['coincTimeWindow']),
            'coincRelation': int(acq['coincRelation']),
            'doseMBq': float(study['doseMBq']),
            'isotope': _decode(study['isotope']),
        }

    def _load(self, f):
        """Subclass hook: populate type-specific attributes from an open file."""

    def __repr__(self):
        return f'{type(self).__name__}({self.path.name!r})'


class _ListModeFile(LR5File):
    """Shared load logic for CLM / SLM event streams."""

    event_dtype: np.dtype | None = None

    def _load(self, f):
        pt = f['eventStream/packetTable']
        event_size = int(np.asarray(pt.attrs['eventSize']).reshape(-1)[0])
        self.event_type = _decode(pt.attrs.get('eventType', b''))
        self.event_descriptor = _decode(pt.attrs.get('eventDescriptor', b''))
        self.event_size = event_size
        self.packet_sequence = int(
            np.asarray(pt.attrs.get('packetSequence', [-1])).reshape(-1)[0]
        )

        self.packets = [np.asarray(pt[i], dtype=np.uint8).copy() for i in range(len(pt))]
        raw = (
            np.concatenate([p for p in self.packets if p.size])
            if any(p.size for p in self.packets)
            else np.array([], dtype=np.uint8)
        )
        n = len(raw) // event_size
        raw = raw[: n * event_size]
        dtype = self.event_dtype
        if dtype is not None and event_size == dtype.itemsize:
            self.events = raw.view(dtype)
        else:
            self.events = raw.reshape(-1, event_size) if n else raw.reshape(0, event_size)

        self.crystal_singles = (
            f['crystalSingle/data'][:] if 'crystalSingle/data' in f else None
        )

    @property
    def n_packets(self) -> int:
        return len(self.packets)

    def _packet_events(self, i):
        p = self.packets[i]
        return p.view(self.event_dtype) if p.size else np.empty(0, self.event_dtype)


class CLMFile(_ListModeFile):
    """Coincidence list-mode (.clm.lr5).

    Prompt and delayed packets alternate, prompts first. The events carry no
    time stamp, so this file cannot be used for the timing tasks; use the
    ``.slm.lr5`` singles instead.
    """

    event_dtype = COINC_DTYPE

    def _load(self, f):
        super()._load(f)
        self.shot_info = (
            f['eventStream/shotInfo'][:] if 'eventStream/shotInfo' in f else None
        )
        n = len(self.packets)
        self.n_prompt_packets = (n + 1) // 2
        self.n_delayed_packets = n // 2

    def coincidences(self, delayed: bool | None = None) -> np.ndarray:
        """Decoded coincidences with the positions masked to 8 bits.

        ``delayed=None`` returns both windows (see the ``delayed`` field),
        ``False`` only the prompts and ``True`` only the delayed window.
        """
        parts = []
        for i in range(self.n_packets):
            ev = self._packet_events(i)
            if not len(ev):
                continue
            is_delayed = bool(i % 2)
            if delayed is not None and is_delayed != delayed:
                continue
            c = np.empty(len(ev), dtype=RAW_COINC_DTYPE)
            for end, src in ((1, '1'), (2, '2')):
                c[f'module{end}'] = ev[f'd{src}']
                # the high byte of the position fields is uninitialised
                c[f'x{end}'] = ev[f'x{src}'] & 0xFF
                c[f'y{end}'] = ev[f'y{src}'] & 0xFF
                c[f'energy{end}'] = ev[f'e{src}'] & 0xFF
            c['delayed'] = is_delayed
            parts.append(c)
        if not parts:
            return np.empty(0, dtype=RAW_COINC_DTYPE)
        return np.concatenate(parts)

    def pair_index(self, coinc: np.ndarray) -> np.ndarray:
        """Index into :data:`COINC_PAIRS` for each coincidence (-1 if not allowed)."""
        return PAIR_INDEX[coinc['module1'], coinc['module2']]


class SLMFile(_ListModeFile):
    """Singles list-mode (.slm.lr5).

    The packet table holds one packet per module per time-stamp epoch, in
    module order, so ``module = packet_index % 12``.
    """

    event_dtype = SLM_EVENT_DTYPE

    def _load(self, f):
        super()._load(f)
        self.n_events = len(self.events)
        if self.n_packets % N_MODULES:
            raise ValueError(
                f'{self.path.name}: {self.n_packets} packets is not a multiple of '
                f'{N_MODULES}, so the module assignment is unknown'
            )

    def _tscount_wraps(self) -> np.ndarray:
        """Number of ``tscount`` wrap-arounds before each packet.

        ``tscount`` is only 8 bit, so it wraps every 256 epochs of
        ``2**32 * T_CLK_NS`` (733 s). The packets are in time order, which is
        used to unwrap it.
        """
        first = np.array([self._packet_events(i)['tscount'][0] if self.packets[i].size else -1
                          for i in range(self.n_packets)], dtype=np.int64)
        valid = np.flatnonzero(first >= 0)
        if not len(valid):
            return np.zeros(self.n_packets, dtype=np.int64)
        idx = np.zeros(self.n_packets, dtype=np.int64)
        idx[valid] = valid
        idx = np.maximum.accumulate(idx)
        idx[: valid[0]] = valid[0]
        filled = first[idx]
        return np.r_[0, np.cumsum(np.diff(filled) < -128)].astype(np.int64)

    def _packet_times(self, i, ev, wraps) -> np.ndarray:
        tscount = ev['tscount'].astype(np.int64) + 256 * wraps[i]
        return (tscount << 32) + ev['ts'].astype(np.int64)

    def singles(self, sort: bool = True) -> np.ndarray:
        """Decode to :data:`RAW_SINGLES_DTYPE`, time-sorted by default.

        ``t`` is the full time stamp ``tscount << 32 | ts`` in ticks of
        :data:`T_CLK_NS`, with ``tscount`` unwrapped; ``module`` comes from
        the packet index.
        """
        wraps = self._tscount_wraps()
        parts = []
        for i in range(self.n_packets):
            ev = self._packet_events(i)
            if not len(ev):
                continue
            s = np.empty(len(ev), dtype=RAW_SINGLES_DTYPE)
            s['t'] = self._packet_times(i, ev, wraps)
            s['module'] = i % N_MODULES
            s['x'] = ev['x']
            s['y'] = ev['y']
            s['energy'] = ev['energy']
            parts.append(s)
        if not parts:
            return np.empty(0, dtype=RAW_SINGLES_DTYPE)
        singles = np.concatenate(parts)
        if sort:
            singles = singles[np.argsort(singles['t'], kind='stable')]
        return singles

    @property
    def duration_s(self) -> float:
        """Acquisition length from the time-stamp span.

        Preferred over ``scan['lengthSec']``, which some repaired files report
        incorrectly (``run4_260922-153612.slm.fixed.lr5`` claims 179 s for a
        120 s run).
        """
        wraps = self._tscount_wraps()
        t_min, t_max = None, None
        for i in range(self.n_packets):
            ev = self._packet_events(i)
            if not len(ev):
                continue
            t = self._packet_times(i, ev, wraps)
            t_min = t.min() if t_min is None else min(t_min, t.min())
            t_max = t.max() if t_max is None else max(t_max, t.max())
        if t_min is None:
            return 0.0
        return float(t_max - t_min) * T_CLK_NS * 1e-9

    def flood(self, module: int | None = None, singles: np.ndarray | None = None):
        """Raw position histogram, ``(256, 256)`` for one module or ``(12, 256, 256)``.

        The 35 x 35 crystal grid shows up as a lattice of spots with a pitch of
        about 7.3 raw units; :func:`petlab.build_crystal_lut` turns it into a
        crystal look-up table.
        """
        if singles is None:
            singles = self.singles(sort=False)
        edges = np.arange(N_POS + 1)
        if module is not None:
            s = singles[singles['module'] == module]
            return np.histogram2d(s['x'], s['y'], bins=[edges, edges])[0]
        out = np.empty((N_MODULES, N_POS, N_POS))
        for m in range(N_MODULES):
            s = singles[singles['module'] == m]
            out[m] = np.histogram2d(s['x'], s['y'], bins=[edges, edges])[0]
        return out

    def singles_with_crystals(self, lut, sort: bool = True) -> np.ndarray:
        """Decode singles and attach crystal indices from a flood look-up table.

        Returns :data:`CRYSTAL_SINGLES_DTYPE`. Energy is still the raw ADC
        channel; :func:`petlab.singles_from_slm` converts to keV.
        """
        raw = self.singles(sort=sort)
        out = np.empty(len(raw), dtype=CRYSTAL_SINGLES_DTYPE)
        out['t'] = raw['t']
        out['module'] = raw['module']
        out['x'] = raw['x']
        out['y'] = raw['y']
        out['energy'] = raw['energy']
        out['crystal'] = assign_crystals(raw['module'], raw['x'], raw['y'], lut)
        return out


class LOR2DFile(LR5File):
    """2D line-of-response histogram (.2dlor.lr5), ``(35, 22050)``."""

    def _load(self, f):
        ds = f['lor2D/data']
        self.data = ds[:]
        self.frame_attr = f['lor2D/frameAttr'][0]
        self.slice_thickness_mm = float(ds.attrs['sliceThickness'].reshape(-1)[0])


class LOR3DFile(LR5File):
    """3D line-of-response histogram (.3dlor.lr5).

    ``data`` is flat with ``N_LOR_3D`` bins, indexed by
    ``pair * 1225**2 + crystal_of_low_module * 1225 + crystal_of_high_module``.
    Note that the histogram often covers only part of the acquisition; see
    ``frame_attr['timeStart']`` and ``['timeStop']``.
    """

    def _load(self, f):
        self.data = f['lor3D/data'][:]
        self.frame_attr = f['lor3D/frameAttr'][0]
        self.total_counts = float(self.data.sum())
        self.randoms = int(self.frame_attr['random'])

    def crystal_pairs(self):
        """Nonzero bins as ``(global_id_1, global_id_2, counts)``."""
        flat = np.asarray(self.data).ravel()
        nz = np.nonzero(flat)[0]
        pair, rem = np.divmod(nz, N_CRYS * N_CRYS)
        c1, c2 = np.divmod(rem, N_CRYS)
        m1 = np.array([p[0] for p in COINC_PAIRS])[pair]
        m2 = np.array([p[1] for p in COINC_PAIRS])[pair]
        return m1 * N_CRYS + c1, m2 * N_CRYS + c2, flat[nz]


# ---------------------------------------------------------------------------
# CSV products
# ---------------------------------------------------------------------------


class CSPFile:

    def __init__(self, path):

        self.path = Path(path)
        with open(path) as f:
            header = f.readline().strip()
        self.headers = [h.strip().strip('"') for h in header.split(',') if h.strip()]
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        spectra = data.T
        if spectra.shape[0] != N_PAIRS:
            raise ValueError(f'expected {N_PAIRS} coincidence columns, got {spectra.shape[0]}')
        
        self.spectra = spectra

        self.N_BINS = spectra.shape[1]
        self.ZERO_BIN = self.N_BINS // 2
        self.BIN_WIDTH = T_CLK_NS # (ns)

        self.x_ns = (np.arange(self.N_BINS) - self.ZERO_BIN) * self.BIN_WIDTH # (ns)


    



def read_csp(path):
    """Read a ``.csp.csv`` coincidence time-difference spectra file.

    Returns ``(spectra, headers)`` with ``spectra`` of shape
    ``(18, 401)`` in :data:`COINC_PAIRS` order. Bin ``CSP_ZERO_BIN`` is zero
    time difference and one bin is :data:`T_CLK_NS`.
    """
    path = Path(path)
    with open(path) as f:
        header = f.readline().strip()
    headers = [h.strip().strip('"') for h in header.split(',') if h.strip()]
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    spectra = data.T
    if spectra.shape[0] != N_PAIRS:
        raise ValueError(f'expected {N_PAIRS} coincidence columns, got {spectra.shape[0]}')
    return spectra, headers


def csp_dt_ns():
    """Time-difference axis of a ``.csp.csv`` spectrum, in ns."""
    return (np.arange(CSP_N_BINS) - CSP_ZERO_BIN) * T_CLK_NS


def pair_label(k, l):
    """Header style used by ``DetectorPair::assign`` (1-based module names)."""
    return f'D{k + 1:02d}-D{l + 1:02d}'


def read_scnt(path):
    """Read a ``.scnt.csv`` into ``(12, 1225)`` singles counts per crystal.

    Indexed by ``[module, crystal]`` with ``crystal = axial_row * 35 +
    tangential_index``. Covers the last reporting interval of the run (60 s in
    the runs seen so far), not necessarily the whole acquisition.
    """
    df = _read_csv(Path(path))
    return df.values.T.astype(np.int64)


def read_rate(path):
    """Read a ``.rate.csv`` with cleaned column names.

    Rates are stored in kcps by ``mpdaq``. Use :func:`module_singles_cps` and
    :func:`pair_rates_cps` for SI units.
    """
    return _read_csv(Path(path))


def module_singles_cps(rate, row: int = -1) -> np.ndarray:
    """Per-module singles rates in cps from a ``.rate.csv`` row (default: last)."""
    if not isinstance(rate, pd.DataFrame):
        rate = read_rate(rate)
    return np.array([rate[f'D{m}S'].iloc[row] for m in range(N_MODULES)],
                    dtype=float) * 1e3


def pair_rates_cps(rate, row: int = -1):
    """Per-pair prompt and random rates in cps from a ``.rate.csv`` row.

    Returns ``(prompts_cps, randoms_cps)``, each of length :data:`N_PAIRS`,
    in :data:`COINC_PAIRS` order.
    """
    if not isinstance(rate, pd.DataFrame):
        rate = read_rate(rate)
    prompts = np.array([rate[f'P{p}C'].iloc[row] for p in range(N_PAIRS)],
                       dtype=float) * 1e3
    randoms = np.array([rate[f'P{p}R'].iloc[row] for p in range(N_PAIRS)],
                       dtype=float) * 1e3
    return prompts, randoms


def assign_crystals(module, x, y, lut) -> np.ndarray:
    """Map raw Anger ``(x, y)`` to crystal indices 0..1224 via a flood LUT.

    ``lut`` has shape ``(12, 256, 256)`` as produced by
    :func:`petlab.build_crystal_lut`. ``module``, ``x`` and ``y`` are parallel
    arrays (the fields of :data:`RAW_SINGLES_DTYPE`).
    """
    module = np.asarray(module)
    x = np.asarray(x)
    y = np.asarray(y)
    lut = np.asarray(lut)
    if lut.shape != (N_MODULES, N_POS, N_POS):
        raise ValueError(f'lut shape {lut.shape} != ({N_MODULES}, {N_POS}, {N_POS})')
    return lut[module, x, y]


# Decoded singles with a crystal index (after applying a flood LUT).
CRYSTAL_SINGLES_DTYPE = np.dtype([
    ('t', '<i8'),
    ('module', 'u1'),
    ('crystal', '<u2'),
    ('x', 'u1'),
    ('y', 'u1'),
    ('energy', 'u1'),        # raw ADC channel; convert to keV in petlab
])


# ---------------------------------------------------------------------------
# Folder loader
# ---------------------------------------------------------------------------

def _has_content(path: Path, kind: str) -> bool:
    """Cheap emptiness check, so repaired files win over their broken originals."""
    if path.stat().st_size == 0:
        return False
    if kind in ('slm', 'clm'):
        try:
            with h5py.File(path, 'r') as f:
                pt = f['eventStream/packetTable']
                return len(pt) > 0 and any(np.asarray(pt[i]).size for i in range(len(pt)))
        except Exception:
            return False
    return True


class MiniPET:
    """Load a MiniPET acquisition / reconstruction folder into memory.

    Discovered products become attributes with data already filled in. Pass
    ``kinds`` to skip the large ones, for example
    ``MiniPET('run8', kinds=('slm', 'scnt', 'rate'))``.

    Example
    -------
    >>> mp = MiniPET('run8')
    >>> s = mp.slm.singles()          # decoded singles, time sorted
    >>> mp.slm.duration_s
    >>> mp.rate, mp.scnt, mp.sino, mp.recon
    """

    def __init__(self, folder, kinds=ALL_KINDS):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(self.folder)
        kinds = tuple(kinds)

        # every candidate per kind, best first
        self.all_paths: dict[str, list[Path]] = {}
        for path in sorted(self.folder.rglob('*')):
            if not path.is_file():
                continue
            kind = _kind_of(path.name)
            if kind is None or kind not in kinds:
                continue
            self.all_paths.setdefault(kind, []).append(path)

        # Prefer a file with content; among those, prefer a repaired
        # ``*.fixed.*`` product, then the plainest name.
        self.paths: dict[str, Path] = {}
        for kind, cands in self.all_paths.items():
            cands.sort(key=lambda p: (
                not _has_content(p, kind),
                '.fixed.' not in p.name,
                p.name.count('.'),
            ))
            self.paths[kind] = cands[0]

        self.rate = read_rate(self.paths['rate']) if 'rate' in self.paths else None
        self.scnt = _read_csv(self.paths['scnt']) if 'scnt' in self.paths else None
        self.csp = read_csp(self.paths['csp'])[0] if 'csp' in self.paths else None
        self.sino = (
            nib.load(self.paths['sino']).get_fdata() if 'sino' in self.paths else None
        )
        self.recon = (
            nib.load(self.paths['recon']).get_fdata() if 'recon' in self.paths else None
        )

        self.clm = CLMFile(self.paths['clm']) if 'clm' in self.paths else None
        self.slm = SLMFile(self.paths['slm']) if 'slm' in self.paths else None
        self.lor2d = LOR2DFile(self.paths['lor2d']) if 'lor2d' in self.paths else None
        self.lor3d = LOR3DFile(self.paths['lor3d']) if 'lor3d' in self.paths else None

    def crystal_singles(self):
        """``(12, 1225)`` singles counts per crystal from ``.scnt.csv``."""
        if 'scnt' not in self.paths:
            return None
        return read_scnt(self.paths['scnt'])

    def module_singles_cps(self, row: int = -1) -> np.ndarray | None:
        """Per-module singles rates in cps from ``.rate.csv``, or ``None``."""
        if self.rate is None:
            return None
        return module_singles_cps(self.rate, row=row)

    def pair_rates_cps(self, row: int = -1):
        """Per-pair ``(prompts_cps, randoms_cps)`` from ``.rate.csv``, or ``None``."""
        if self.rate is None:
            return None
        return pair_rates_cps(self.rate, row=row)

    def duration_s(self) -> float | None:
        """Acquisition length in seconds from the singles time-stamp span."""
        if self.slm is not None:
            return self.slm.duration_s
        if self.clm is not None:
            return float(self.clm.scan.get('lengthSec') or 0) or None
        return None

    def __repr__(self):
        kinds = ', '.join(sorted(self.paths)) or '(empty)'
        return f'MiniPET({self.folder!s}: {kinds})'

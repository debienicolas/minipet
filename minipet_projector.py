"""MiniPET sinogram model: parallelproj projector + crystal pair → bin mapping.

The scanner is modelled as a parallelproj ``RegularPolygonPETScannerGeometry``
with 12 sides of 35 x 35 crystals, which gives a sinogram of shape
``(radial, view, plane) = (139, 210, 1225)``. There is one plane per ordered
ring pair, so **one bin is exactly one LOR** and nothing is rebinned or
interpolated. ``radial_trim=140`` is chosen so that every LOR the MiniPET can
measure has a bin.

Coincidences → sinogram
-----------------------
    y = coinc_to_sinogram(clm, lut)              # prompts of a .clm.lr5
    y = coinc_to_sinogram(clm, lut, delayed=True)

A crystal is ``(module, crystal)`` with ``crystal = ring * 35 + tangential``,
and the polygon numbers its crystals by ``(ring, detector)`` with
``detector = module * 35 + tangential``. The bin of a crystal pair is therefore
two small table look-ups (:func:`sinogram_bins`): the transaxial pair
``(detector1, detector2)`` gives ``(radial, view)`` and the ring pair gives the
plane. No geometry files and no big index tables are involved.

Physical coordinates and display
--------------------------------
    coords = sinogram_coords()                   # s, phi, z_mean, tan_theta per bin
    show_sino(ax, y.sum(2), coords, 'prompts')   # on a regular (phi, s) grid
    y_phi_s = rebin_phi_s(y.sum(2), coords)

Reconstruction on a centred ``petlab.ImageGrid``
-----------------------------------------------
    P = projector(grid)                          # parallelproj projector
    sens = sensitivity_image(grid)               # P.adjoint over all measured LORs
    ybar = P(x) + r                              # forward model
    x *= P.adjoint(y / ybar) / sens               # MLEM update

``BinProjector(bins, grid)`` gives the same LORs restricted to a list of bins,
which is much faster when most bins are empty (the MLEM ratio is zero there).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import array_api_compat.numpy as xp
import numpy as np

from parallelproj.pet_lors import (
    Michelogram,
    RegularPolygonPETLORDescriptor,
    SinogramSpatialAxisOrder,
)
from parallelproj.pet_scanners import RegularPolygonPETScannerGeometry
from parallelproj.projectors import RegularPolygonPETProjector

# ---------------------------------------------------------------------------
# Geometry (mm)
# ---------------------------------------------------------------------------

CRYSTAL_SIZE_MM = 1.27
CRYSTAL_GAP_MM = 0.077
CRYSTAL_PITCH_MM = CRYSTAL_SIZE_MM + CRYSTAL_GAP_MM  # 1.347
CRYSTALS_PER_SIDE = 35
N_MODULES = 12
N_PAIRS = 18
N_CRYSTALS_PER_MODULE = CRYSTALS_PER_SIDE**2  # 1225
N_LOR_GEOMETRY = N_PAIRS * N_CRYSTALS_PER_MODULE**2  # 27011250

# Circumradius to crystal centres (from minipet_geom).
POLYGON_RADIUS_MM = 106.37
# First side / det0 near geom xstart[0] (module on +y).
POLYGON_PHI0 = math.pi / 2

# Axial mid-plane of crystal centres: pitch/2 + i*pitch, i = 0..34.
CRYSTAL_Z_MID_MM = CRYSTALS_PER_SIDE / 2 * CRYSTAL_PITCH_MM  # 23.5725

# RVP: views = 420/2 = 210; radials = 419 - 2*trim → 139 at trim=140.
SINO_NUM_RAD = 139
SINO_NUM_VIEW = 210
SINO_RADIAL_TRIM = 140
SINO_NUM_PLANES = CRYSTALS_PER_SIDE * CRYSTALS_PER_SIDE  # 1225

# mpdaq pair order P0..P17
PAIRS = (
    (0, 5), (0, 6), (0, 7),
    (1, 6), (1, 7), (1, 8),
    (2, 7), (2, 8), (2, 9),
    (3, 8), (3, 9), (3, 10),
    (4, 9), (4, 10), (4, 11),
    (5, 10), (5, 11),
    (6, 11),
)

_PAIR_INDEX = np.full((N_MODULES, N_MODULES), -1, dtype=np.int64)
for _p, (_k, _l) in enumerate(PAIRS):
    _PAIR_INDEX[_k, _l] = _PAIR_INDEX[_l, _k] = _p

_HERE = Path(__file__).resolve().parent

DEFAULT_IMG_SHAPE = (371, 371, 35)
DEFAULT_VOXEL_SIZE = (0.2694, 0.2694, CRYSTAL_PITCH_MM)


@dataclass(frozen=True)
class MiniPETSinogramProjectorBundle:
    scanner: RegularPolygonPETScannerGeometry
    lor_descriptor: RegularPolygonPETLORDescriptor
    projector: RegularPolygonPETProjector
    img_origin: tuple[float, float, float]
    xp: object
    dev: str


@dataclass(frozen=True)
class MiniPETRVPLorMap:
    """RVP bin ↔ flat lor3d index (``-1`` = no MiniPET LOR in that bin)."""

    poly_to_geom: np.ndarray  # int32 (n_rvp,)
    geom_to_poly: np.ndarray  # int32 (n_geom,)
    n_mapped: int
    sino_shape: tuple[int, int, int]

    @property
    def n_rvp(self) -> int:
        return int(self.poly_to_geom.shape[0])

    @property
    def n_geom(self) -> int:
        return int(self.geom_to_poly.shape[0])


# ---------------------------------------------------------------------------
# Image origin (geom crystal frame: xy centred, z on first crystal centre)
# ---------------------------------------------------------------------------

def default_img_origin(
    img_shape: tuple[int, int, int] = DEFAULT_IMG_SHAPE,
    voxel_size: tuple[float, float, float] = DEFAULT_VOXEL_SIZE,
) -> tuple[float, float, float]:
    ox = -(img_shape[0] - 1) / 2 * voxel_size[0]
    oy = -(img_shape[1] - 1) / 2 * voxel_size[1]
    oz = CRYSTAL_Z_MID_MM - (img_shape[2] - 1) / 2 * voxel_size[2]
    return (float(ox), float(oy), float(oz))


# ---------------------------------------------------------------------------
# RegularPolygon RVP projector
# ---------------------------------------------------------------------------

def _ring_positions(xp=xp, n_rings: int = CRYSTALS_PER_SIDE):
    return (xp.arange(n_rings, dtype=xp.float32) + 0.5) * CRYSTAL_PITCH_MM


def build_polygon_scanner(
    radius_mm: float = POLYGON_RADIUS_MM,
    phi0: float = POLYGON_PHI0,
    xp=xp,
    dev: str = "cpu",
) -> RegularPolygonPETScannerGeometry:
    return RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=radius_mm,
        num_sides=N_MODULES,
        num_lor_endpoints_per_side=CRYSTALS_PER_SIDE,
        lor_spacing=CRYSTAL_PITCH_MM,
        ring_positions=_ring_positions(xp),
        symmetry_axis=2,
        phi0=phi0,
    )


def build_sinogram_lor_descriptor(
    scanner: RegularPolygonPETScannerGeometry | None = None,
    *,
    radial_trim: int = SINO_RADIAL_TRIM,
    max_ring_difference: int | None = None,
    sinogram_order: SinogramSpatialAxisOrder = SinogramSpatialAxisOrder.RVP,
    xp=xp,
    dev: str = "cpu",
) -> RegularPolygonPETLORDescriptor:
    if scanner is None:
        scanner = build_polygon_scanner(xp=xp, dev=dev)

    michelogram = None
    if max_ring_difference is not None:
        michelogram = Michelogram(
            num_rings=scanner.num_rings,
            max_ring_difference=max_ring_difference,
            span=1,
        )

    return RegularPolygonPETLORDescriptor(
        scanner,
        michelogram=michelogram,
        radial_trim=radial_trim,
        sinogram_order=sinogram_order,
    )


def build_sinogram_projector(
    lor_descriptor: RegularPolygonPETLORDescriptor | None = None,
    img_shape: tuple[int, int, int] = DEFAULT_IMG_SHAPE,
    voxel_size: tuple[float, float, float] = DEFAULT_VOXEL_SIZE,
    img_origin=None,
    *,
    max_ring_difference: int | None = None,
    radius_mm: float = POLYGON_RADIUS_MM,
    xp=xp,
    dev: str = "cpu",
) -> MiniPETSinogramProjectorBundle:
    """Build the MiniPET RVP projector. Default out shape ``(139, 210, 1225)``."""
    if lor_descriptor is None:
        scanner = build_polygon_scanner(radius_mm=radius_mm, xp=xp, dev=dev)
        lor_descriptor = build_sinogram_lor_descriptor(
            scanner,
            max_ring_difference=max_ring_difference,
            xp=xp,
            dev=dev,
        )
    else:
        scanner = lor_descriptor.scanner

    if img_origin is None:
        img_origin = default_img_origin(img_shape, voxel_size)

    proj = RegularPolygonPETProjector(
        lor_descriptor,
        img_shape=img_shape,
        voxel_size=voxel_size,
        img_origin=img_origin,
    )
    return MiniPETSinogramProjectorBundle(
        scanner, lor_descriptor, proj, tuple(img_origin), xp, dev
    )


def show_lordesc_view(ax, view, plane, lor_desc):
    scanner = bundle.scanner
    lor_desc = bundle.lor_descriptor
    xp, dev = scanner.xp, scanner.dev
    ax.view_init(elev=-30, azim=160, roll=180, vertical_axis="y")
    scanner.show_lor_endpoints(ax)
    lor_desc.show_views(
        ax,
        views=xp.asarray([view], device=dev),
        planes=xp.asarray([plane], device=dev),
        lw=0.5,
        color="k",
    )
    ax.set_title(f"view {view}, plane {plane}")
    return ax


# ---------------------------------------------------------------------------
# Crystal pair → sinogram bin
#
# Two look-up tables, both tiny:
#   _RV[d1, d2]    flat (radial, view) index of the transaxial LOR, -1 if none
#   _SWAP[d1, d2]  True when d1 is the "end" of that bin, so the ring pair
#                  has to be swapped before the plane look-up
#   _PLANE[r1, r2] plane of an ordered ring pair (all 35 x 35 are present)
# with the detector index d = module * 35 + tangential and the ring r of a
# crystal, i.e. crystal = r * 35 + tangential.
# ---------------------------------------------------------------------------

N_DET_PER_RING = N_MODULES * CRYSTALS_PER_SIDE  # 420

_TABLES = None


def _tables():
    """``(_RV, _SWAP, _PLANE, sino_shape)`` for the default descriptor, built once."""
    global _TABLES
    if _TABLES is None:
        ld = build_sinogram_lor_descriptor()
        n_rad, n_view, n_plane = (int(n) for n in ld.spatial_sinogram_shape)
        s_ir = np.asarray(ld.start_in_ring_index)  # (view, radial)
        e_ir = np.asarray(ld.end_in_ring_index)
        flat = (np.arange(n_rad)[None, :] * n_view + np.arange(n_view)[:, None]).astype(np.int32)

        rv = np.full((N_DET_PER_RING, N_DET_PER_RING), -1, np.int32)
        swap = np.zeros((N_DET_PER_RING, N_DET_PER_RING), bool)
        rv[s_ir, e_ir] = flat
        rv[e_ir, s_ir] = flat
        swap[e_ir, s_ir] = True

        plane = np.full((CRYSTALS_PER_SIDE, CRYSTALS_PER_SIDE), -1, np.int32)
        plane[np.asarray(ld.start_plane_index), np.asarray(ld.end_plane_index)] = np.arange(
            n_plane, dtype=np.int32)

        _TABLES = (rv, swap, plane, (n_rad, n_view, n_plane))
    return _TABLES


def sinogram_shape() -> tuple[int, int, int]:
    """``(radial, view, plane)`` of the sinogram."""
    return _tables()[3]


def sinogram_bins(module1, crystal1, module2, crystal2) -> np.ndarray:
    """Flat sinogram bin of each crystal pair (``-1`` if the pair has no bin).

    ``crystal`` is ``ring * 35 + tangential`` within its module, as returned by
    the flood look-up table. The order of the two ends does not matter.
    """
    rv, swap, plane, (n_rad, n_view, n_plane) = _tables()
    c1 = np.asarray(crystal1, dtype=np.int64)
    c2 = np.asarray(crystal2, dtype=np.int64)
    d1 = np.asarray(module1, dtype=np.int64) * CRYSTALS_PER_SIDE + c1 % CRYSTALS_PER_SIDE
    d2 = np.asarray(module2, dtype=np.int64) * CRYSTALS_PER_SIDE + c2 % CRYSTALS_PER_SIDE
    k = rv[d1, d2]
    sw = swap[d1, d2]
    r1, r2 = c1 // CRYSTALS_PER_SIDE, c2 // CRYSTALS_PER_SIDE
    p = plane[np.where(sw, r2, r1), np.where(sw, r1, r2)]
    return np.where((k >= 0) & (p >= 0), k.astype(np.int64) * n_plane + p, -1)


def histogram_bins(bins, weights=None) -> np.ndarray:
    """Histogram flat bin indices (``-1`` dropped) into a ``(139, 210, 1225)`` sinogram."""
    shape = sinogram_shape()
    bins = np.asarray(bins)
    ok = bins >= 0
    w = None if weights is None else np.asarray(weights, float)[ok]
    counts = np.bincount(bins[ok], weights=w, minlength=int(np.prod(shape)))
    return counts.astype(np.float32).reshape(shape)


def coinc_to_sinogram(clm, lut, delayed: bool = False) -> np.ndarray:
    """Prompts (or delayed coincidences) of a ``.clm.lr5`` as a sinogram.

    ``lut`` is the flood crystal look-up table of :func:`petlab.build_crystal_lut`,
    so that ``lut[module, x, y]`` is the crystal of a raw Anger position.
    """
    c = clm.coincidences(delayed=delayed)
    return histogram_bins(sinogram_bins(
        c['module1'], lut[c['module1'], c['x1'], c['y1']],
        c['module2'], lut[c['module2'], c['x2'], c['y2']],
    ))


def measured_bins() -> np.ndarray:
    """Boolean sinogram, ``True`` for the bins that are a LOR the scanner measures.

    Whether a LOR is measured depends only on its two modules, and every ring
    pair is acquired, so the mask is one transaxial pattern repeated in all
    1225 planes: 22 050 of the 29 190 (radial, view) bins, 27 011 250 LORs.
    """
    rv, swap, plane, shape = _tables()
    ld = build_sinogram_lor_descriptor()
    mod1 = np.asarray(ld.start_in_ring_index).T // CRYSTALS_PER_SIDE  # (radial, view)
    mod2 = np.asarray(ld.end_in_ring_index).T // CRYSTALS_PER_SIDE
    return np.broadcast_to((_PAIR_INDEX[mod1, mod2] >= 0)[:, :, None], shape)


def randoms_from_singles(rate_per_crystal, tau_ns: float, duration_s: float) -> np.ndarray:
    """Expected randoms sinogram, ``r_ij = 2 tau S_i S_j T_acq`` (Eq. 3).

    ``rate_per_crystal`` is the singles rate in counts per second of every
    crystal, shape ``(12, 1225)`` or flat with the global id
    ``module * 1225 + crystal``. Bins that are not a measured LOR stay zero.
    """
    rv, swap, plane, shape = _tables()
    ld = build_sinogram_lor_descriptor()
    S = np.asarray(rate_per_crystal, dtype=np.float32).reshape(
        N_MODULES, CRYSTALS_PER_SIDE, CRYSTALS_PER_SIDE)          # [module, ring, tangential]
    S_det = S.transpose(0, 2, 1).reshape(N_DET_PER_RING, CRYSTALS_PER_SIDE)  # [detector, ring]

    d1 = np.asarray(ld.start_in_ring_index).T                      # (radial, view)
    d2 = np.asarray(ld.end_in_ring_index).T
    r1 = np.asarray(ld.start_plane_index)                          # (plane,)
    r2 = np.asarray(ld.end_plane_index)
    r = S_det[d1[:, :, None], r1[None, None, :]]
    r *= S_det[d2[:, :, None], r2[None, None, :]]
    r *= np.float32(2e-9 * tau_ns * duration_s)
    return np.where(measured_bins(), r, 0.0)


# ---------------------------------------------------------------------------
# lor3d → RVP binning (raw DAQ histograms; not needed for the lab notebooks)
# ---------------------------------------------------------------------------

def _lor3d_flat(lor3d: np.ndarray) -> np.ndarray:
    arr = np.asarray(lor3d)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape == (N_PAIRS, N_CRYSTALS_PER_MODULE, N_CRYSTALS_PER_MODULE):
        arr = arr.reshape(-1)
    if arr.ndim != 1 or arr.shape[0] != N_LOR_GEOMETRY:
        raise ValueError(f"unexpected lor3d shape {np.asarray(lor3d).shape}")
    return arr


def build_rvp_lor_map(
    lor_descriptor: RegularPolygonPETLORDescriptor | None = None,
) -> MiniPETRVPLorMap:
    """Map each RVP bin to a flat lor3d index via crystal-pair identity.

    Polygon crystal ``ring * 420 + side * 35 + t`` ↔ MiniPET
    ``module * 1225 + axial * 35 + tangential`` (same numbering with our
    ``phi0``). No geom ``.bin`` files required.
    """
    if lor_descriptor is None:
        lor_descriptor = build_sinogram_lor_descriptor()

    n_c = N_CRYSTALS_PER_MODULE
    n_t = CRYSTALS_PER_SIDE
    n_xtal = N_MODULES * n_c
    n_det = int(lor_descriptor.scanner.num_lor_endpoints_per_ring)
    sino_shape = tuple(int(x) for x in lor_descriptor.spatial_sinogram_shape)
    n_rad, n_view, n_plane = sino_shape
    n_rvp = n_rad * n_view * n_plane

    s_ring = np.asarray(lor_descriptor.start_in_ring_index)  # (view, rad)
    e_ring = np.asarray(lor_descriptor.end_in_ring_index)
    s_pl = np.asarray(lor_descriptor.start_plane_index)
    e_pl = np.asarray(lor_descriptor.end_plane_index)

    vv = np.arange(n_view)[:, None, None]
    rr = np.arange(n_rad)[None, :, None]
    s_ir = np.broadcast_to(s_ring[vv, rr], (n_view, n_rad, n_plane)).transpose(1, 0, 2)
    e_ir = np.broadcast_to(e_ring[vv, rr], (n_view, n_rad, n_plane)).transpose(1, 0, 2)
    s_pl_b = np.broadcast_to(s_pl, (n_rad, n_view, n_plane))
    e_pl_b = np.broadcast_to(e_pl, (n_rad, n_view, n_plane))

    # polygon linear crystal indices → MiniPET module-major ids
    def poly_to_geom_xtal(ring: np.ndarray, in_ring: np.ndarray) -> np.ndarray:
        side, t = np.divmod(in_ring, n_t)
        return side * n_c + ring * n_t + t

    ga = poly_to_geom_xtal(s_pl_b, s_ir).reshape(-1)
    gb = poly_to_geom_xtal(e_pl_b, e_ir).reshape(-1)
    poly_key = (
        np.minimum(ga, gb).astype(np.int64) * n_xtal
        + np.maximum(ga, gb).astype(np.int64)
    )

    idx = np.arange(N_LOR_GEOMETRY, dtype=np.int64)
    pair = idx // (n_c * n_c)
    rem = idx % (n_c * n_c)
    c1 = rem // n_c
    c2 = rem % n_c
    m1 = np.array([p[0] for p in PAIRS], dtype=np.int64)[pair]
    m2 = np.array([p[1] for p in PAIRS], dtype=np.int64)[pair]
    g0 = m1 * n_c + c1
    g1 = m2 * n_c + c2
    geom_key = (
        np.minimum(g0, g1).astype(np.int64) * n_xtal
        + np.maximum(g0, g1).astype(np.int64)
    )

    order = np.argsort(geom_key, kind="mergesort")
    sorted_keys = geom_key[order]
    pos = np.searchsorted(sorted_keys, poly_key)
    in_range = pos < N_LOR_GEOMETRY
    pos_c = np.clip(pos, 0, N_LOR_GEOMETRY - 1)
    matched = in_range & (sorted_keys[pos_c] == poly_key)

    poly_to_geom = np.full(n_rvp, -1, dtype=np.int32)
    poly_to_geom[matched] = order[pos_c[matched]].astype(np.int32)

    geom_to_poly = np.full(N_LOR_GEOMETRY, -1, dtype=np.int32)
    geom_to_poly[poly_to_geom[matched]] = np.nonzero(matched)[0].astype(np.int32)

    return MiniPETRVPLorMap(
        poly_to_geom=poly_to_geom,
        geom_to_poly=geom_to_poly,
        n_mapped=int(matched.sum()),
        sino_shape=sino_shape,
    )


def load_lor_map(cache: bool = True) -> MiniPETRVPLorMap:
    """:func:`build_rvp_lor_map` for the default descriptor, cached in ``lab_data/``."""
    fn = _HERE / "lab_data" / f"rvp_lor_map_trim{SINO_RADIAL_TRIM}.npz"
    if cache and fn.exists():
        with np.load(fn) as f:
            return MiniPETRVPLorMap(
                poly_to_geom=f["poly_to_geom"],
                geom_to_poly=f["geom_to_poly"],
                n_mapped=int(f["n_mapped"]),
                sino_shape=tuple(int(x) for x in f["sino_shape"]),
            )
    lor_map = build_rvp_lor_map()
    if cache:
        fn.parent.mkdir(exist_ok=True)
        np.savez(fn, poly_to_geom=lor_map.poly_to_geom, geom_to_poly=lor_map.geom_to_poly,
                 n_mapped=lor_map.n_mapped, sino_shape=np.array(lor_map.sino_shape))
    return lor_map


def lor3d_index(g1, g2) -> np.ndarray:
    """Flat lor3d index of crystal pairs (global ids ``module * 1225 + crystal``).

    The order of the two ends does not matter; pairs of modules that are not
    in coincidence give ``-1``.
    """
    g1 = np.asarray(g1, dtype=np.int64)
    g2 = np.asarray(g2, dtype=np.int64)
    n_c = N_CRYSTALS_PER_MODULE
    swap = g1 // n_c > g2 // n_c
    lo = np.where(swap, g2, g1)
    hi = np.where(swap, g1, g2)
    pair = _PAIR_INDEX[lo // n_c, hi // n_c]
    return np.where(pair >= 0, pair * n_c * n_c + (lo % n_c) * n_c + hi % n_c, -1)


def lor3d_crystals() -> tuple[np.ndarray, np.ndarray]:
    """Global crystal ids ``(g_low_module, g_high_module)`` of every lor3d bin, in lor3d order."""
    n_c = N_CRYSTALS_PER_MODULE
    idx = np.arange(N_LOR_GEOMETRY, dtype=np.int64)
    pair, rem = np.divmod(idx, n_c * n_c)
    c1, c2 = np.divmod(rem, n_c)
    pairs = np.array(PAIRS, dtype=np.int64)
    return (pairs[pair, 0] * n_c + c1).astype(np.int32), (pairs[pair, 1] * n_c + c2).astype(np.int32)


def lors_to_sinogram(g1, g2, weights=None) -> np.ndarray:
    """Histogram crystal pairs given as global ids ``module * 1225 + crystal``."""
    g1 = np.asarray(g1, dtype=np.int64)
    g2 = np.asarray(g2, dtype=np.int64)
    bins = sinogram_bins(g1 // N_CRYSTALS_PER_MODULE, g1 % N_CRYSTALS_PER_MODULE,
                         g2 // N_CRYSTALS_PER_MODULE, g2 % N_CRYSTALS_PER_MODULE)
    return histogram_bins(bins, weights)


def clm_to_sinogram(clm, lut, lor_map=None, delayed: bool = False) -> np.ndarray:
    """Deprecated alias of :func:`coinc_to_sinogram`; ``lor_map`` is ignored."""
    return coinc_to_sinogram(clm, lut, delayed=delayed)


# ---------------------------------------------------------------------------
# Physical coordinates of the sinogram bins
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SinogramCoords:
    """Physical coordinates of every bin of an RVP sinogram ``(rad, view, plane)``.

    The polygon sinogram is not a regular (phi, s) grid: within one view phi
    changes by a few degrees along the radial axis, and s jumps where the LOR
    crosses a module boundary. These arrays give the true values per bin.

    ``s``, ``phi`` (in [0, pi)) and ``chord`` (transaxial LOR length) have
    shape ``(rad, view)`` and are the same in every plane. The LOR is
    ``x cos(phi) + y sin(phi) = s``; ``u = (-sin(phi), cos(phi))`` points
    along it. ``ring_1``, ``ring_2`` (shape ``(plane,)``) are the rings of
    the two ends *before* phi is folded into [0, pi); ``flip`` marks the bins
    where folding swapped them. z = 0 is the axial centre of the scanner.
    """

    s: np.ndarray
    phi: np.ndarray
    chord: np.ndarray
    flip: np.ndarray
    ring_1: np.ndarray
    ring_2: np.ndarray
    z_ring: np.ndarray

    @property
    def z_mean(self) -> np.ndarray:
        """Mean axial position of each plane (mm), shape ``(plane,)``."""
        return 0.5 * (self.z_ring[self.ring_1] + self.z_ring[self.ring_2])

    @property
    def direct(self) -> np.ndarray:
        """Planes with both ends in the same ring, shape ``(plane,)``."""
        return self.ring_1 == self.ring_2

    @property
    def ring_difference(self) -> np.ndarray:
        """Signed ring difference d along ``u``, shape ``(rad, view, plane)``."""
        d = (self.ring_1 - self.ring_2)[None, None, :]
        return np.where(self.flip[..., None], -d, d)

    @property
    def tan_theta(self) -> np.ndarray:
        """tan of the polar angle: z grows by ``chord * tan_theta`` along ``u``; ``(rad, view, plane)``."""
        dz = (self.z_ring[self.ring_1] - self.z_ring[self.ring_2])[None, None, :]
        dz = np.where(self.flip[..., None], -dz, dz)
        return dz / self.chord[..., None]


def sinogram_coords(lor_descriptor: RegularPolygonPETLORDescriptor | None = None) -> SinogramCoords:
    """Per-bin (s, phi, z, tan theta) of the RVP sinogram, see :class:`SinogramCoords`."""
    if lor_descriptor is None:
        lor_descriptor = build_sinogram_lor_descriptor()
    sc = lor_descriptor.scanner
    n_det = int(sc.num_lor_endpoints_per_ring)
    xy = np.asarray(sc.get_lor_endpoints(xp.zeros(n_det, dtype=xp.int32),
                                         xp.arange(n_det, dtype=xp.int32)))[:, :2].astype(np.float64)
    # (view, rad) in the descriptor -> (rad, view)
    p1 = xy[np.asarray(lor_descriptor.start_in_ring_index)].transpose(1, 0, 2)
    p2 = xy[np.asarray(lor_descriptor.end_in_ring_index)].transpose(1, 0, 2)
    dx, dy = p2[..., 0] - p1[..., 0], p2[..., 1] - p1[..., 1]
    chord = np.hypot(dx, dy)
    phi = np.arctan2(dx, p1[..., 1] - p2[..., 1])
    s = (p2[..., 0] * p1[..., 1] - p1[..., 0] * p2[..., 1]) / chord
    flip = phi < 0
    phi = np.where(flip, phi + np.pi, phi)
    phi = np.where(phi >= np.pi, phi - np.pi, phi)
    s = np.where(flip, -s, s)
    return SinogramCoords(
        s=s, phi=phi, chord=chord, flip=flip,
        ring_1=np.asarray(lor_descriptor.start_plane_index).astype(np.int64),
        ring_2=np.asarray(lor_descriptor.end_plane_index).astype(np.int64),
        z_ring=np.asarray(sc.ring_positions, dtype=np.float64) - CRYSTAL_Z_MID_MM,
    )


# Regular (phi, s) grid for display and for methods that need equal spacing
S_BIN_MM = CRYSTAL_PITCH_MM / 2
S_EDGES = np.arange(-50, 50 + S_BIN_MM, S_BIN_MM) - S_BIN_MM / 2
S_CENTRES = 0.5 * (S_EDGES[1:] + S_EDGES[:-1])
N_PHI = 180
PHI_EDGES = np.linspace(0, 180, N_PHI + 1)
PHI_CENTRES = 0.5 * (PHI_EDGES[1:] + PHI_EDGES[:-1])      # degrees


def rebin_phi_s(sino: np.ndarray, coords: SinogramCoords) -> np.ndarray:
    """Put a ``(rad, view)`` or ``(rad, view, plane)`` sinogram on the regular grid.

    Returns ``(N_PHI, len(S_CENTRES))`` or ``(plane, N_PHI, len(S_CENTRES))``,
    with axes :data:`PHI_CENTRES` (deg) and :data:`S_CENTRES` (mm). Every bin
    goes to the grid cell that contains its (phi, s), so counts are conserved;
    because the two samplings differ slightly, some cells get two bins and
    some none, which shows up as a faint moire pattern.
    """
    sino = np.asarray(sino)
    pi = np.minimum((coords.phi / np.pi * N_PHI).astype(np.int64), N_PHI - 1)
    si = np.digitize(coords.s, S_EDGES) - 1
    ok = (si >= 0) & (si < len(S_CENTRES))
    cell = (pi * len(S_CENTRES) + si)[ok]
    n_cell = N_PHI * len(S_CENTRES)
    if sino.ndim == 2:
        return np.bincount(cell, sino[ok], minlength=n_cell).reshape(N_PHI, len(S_CENTRES))
    out = np.empty((sino.shape[2], N_PHI, len(S_CENTRES)))
    for k in range(sino.shape[2]):
        out[k] = np.bincount(cell, sino[..., k][ok], minlength=n_cell).reshape(N_PHI, len(S_CENTRES))
    return out


def show_sino(ax, sino: np.ndarray, coords: SinogramCoords | None = None, title: str = "", **kw):
    """Display a ``(rad, view)`` sinogram on the regular (phi, s) grid.

    With ``coords=None``, ``sino`` is taken to be on that grid already, as
    returned by :func:`rebin_phi_s`.
    """
    y = np.asarray(sino) if coords is None else rebin_phi_s(sino, coords)
    im = ax.imshow(y.T, aspect="auto", origin="lower", interpolation="nearest",
                   extent=[0, 180, S_EDGES[0], S_EDGES[-1]], **kw)
    ax.set(xlabel="φ (deg)", ylabel="s (mm)", title=title)
    return im


def lor3d_to_rvp(lor3d: np.ndarray, lor_map: MiniPETRVPLorMap) -> np.ndarray:
    """Scatter lor3d into RVP ``(139, 210, 1225)`` (extras stay 0)."""
    flat = _lor3d_flat(lor3d)
    out = np.zeros(lor_map.n_rvp, dtype=np.result_type(flat, np.float32))
    m = lor_map.poly_to_geom >= 0
    out[m] = flat[lor_map.poly_to_geom[m]]
    return out.reshape(lor_map.sino_shape)


def rvp_to_lor3d(sino: np.ndarray, lor_map: MiniPETRVPLorMap) -> np.ndarray:
    """Gather RVP back to flat lor3d order."""
    arr = np.asarray(sino)
    if arr.shape != lor_map.sino_shape:
        raise ValueError(f"expected {lor_map.sino_shape}, got {arr.shape}")
    flat_sino = arr.reshape(-1)
    out = np.zeros(lor_map.n_geom, dtype=np.result_type(flat_sino, np.float32))
    m = lor_map.geom_to_poly >= 0
    out[m] = flat_sino[lor_map.geom_to_poly[m]]
    return out


# ---------------------------------------------------------------------------
# Reconstruction (lab notebooks)
#
# The image grid is centred on the scanner (petlab.ImageGrid: ``shape``,
# ``voxel``, ``origin`` with z = 0 at the axial centre); the projectors work
# in the scanner frame, whose z starts at the first ring.
# ---------------------------------------------------------------------------

def _scanner_origin(grid) -> tuple[float, float, float]:
    ox, oy, oz = grid.origin
    return (float(ox), float(oy), float(oz) + CRYSTAL_Z_MID_MM)


def grid_projector(grid) -> MiniPETSinogramProjectorBundle:
    """Sinogram projector ``(139, 210, 1225)`` for a centred image grid."""
    return build_sinogram_projector(img_shape=tuple(grid.shape), voxel_size=tuple(grid.voxel),
                                    img_origin=_scanner_origin(grid))


def projector(grid) -> RegularPolygonPETProjector:
    """The parallelproj projector of the scanner for a centred image grid.

    ``P(x)`` is the forward projection of an image into the ``(139, 210, 1225)``
    sinogram and ``P.adjoint(y)`` the back-projection.
    """
    return grid_projector(grid).projector


def sensitivity_image(grid, lor_map=None, cache: bool = True, verbose: bool = True) -> np.ndarray:
    """sigma_j = sum_i a_ij over all measured LORs (the back-projection of a sinogram of ones).

    Takes a few seconds to a few minutes depending on the CPU; cached in
    ``lab_data/``. ``lor_map`` is ignored (kept for older notebooks).
    """
    fn = _HERE / "lab_data" / (
        "sens_rvp_{}x{}x{}_{:.4f}_{:.4f}_{:.4f}.npy".format(*grid.shape, *grid.voxel))
    if cache and fn.exists():
        return np.load(fn)
    if verbose:
        print("computing the sensitivity image (once, then cached) ...")
    sens = np.asarray(projector(grid).adjoint(
        xp.asarray(np.ascontiguousarray(measured_bins(), dtype=np.float32))))
    if cache:
        fn.parent.mkdir(exist_ok=True)
        np.save(fn, sens)
    return sens


class BinProjector:
    """System matrix A restricted to a subset of sinogram bins.

    ``bins`` are flat indices into the ``(139, 210, 1225)`` sinogram, e.g.
    ``np.flatnonzero(y)`` for the bins with counts. ``A.fwd(x)`` gives the
    expected counts in those bins, ``A.back(v)`` the back-projection of one
    value per bin. The LORs are the same as those of :func:`grid_projector`,
    but only the listed ones are computed, which is much faster when most bins
    are empty.
    """

    def __init__(self, bins, grid, lor_descriptor: RegularPolygonPETLORDescriptor | None = None):
        from parallelproj.projectors import ListmodePETProjector

        if lor_descriptor is None:
            lor_descriptor = build_sinogram_lor_descriptor()
        self.grid = grid
        self.bins = np.asarray(bins, dtype=np.int64)
        sino_shape = tuple(int(n) for n in lor_descriptor.spatial_sinogram_shape)
        rad, view, plane = np.unravel_index(self.bins, sino_shape)
        sc = lor_descriptor.scanner
        ends = []
        for ring, in_ring in ((lor_descriptor.start_plane_index, lor_descriptor.start_in_ring_index),
                              (lor_descriptor.end_plane_index, lor_descriptor.end_in_ring_index)):
            r = xp.asarray(np.asarray(ring)[plane].astype(np.int32))
            i = xp.asarray(np.asarray(in_ring)[view, rad].astype(np.int32))
            ends.append(xp.asarray(np.ascontiguousarray(sc.get_lor_endpoints(r, i), dtype=np.float32)))
        self._P = ListmodePETProjector(ends[0], ends[1], tuple(grid.shape), tuple(grid.voxel),
                                       xp.asarray(np.array(_scanner_origin(grid), dtype=np.float32)))

    @property
    def n_bins(self) -> int:
        return len(self.bins)

    def fwd(self, x) -> np.ndarray:
        return np.asarray(self._P(xp.asarray(np.asarray(x, dtype=np.float32))))

    def back(self, v) -> np.ndarray:
        return np.asarray(self._P.adjoint(xp.asarray(np.asarray(v, dtype=np.float32))))


if __name__ == "__main__":
    bundle = build_sinogram_projector(img_shape=(40, 40, 8), voxel_size=(2.5, 2.5, 6.0))
    print(f"radial_trim   {SINO_RADIAL_TRIM}")
    print(f"out_shape     {bundle.projector.out_shape}")

    measured = measured_bins()
    print(f"measured LORs {measured.sum():,} of {measured.size:,} bins "
          f"(expected {N_LOR_GEOMETRY:,})")
    assert measured.sum() == N_LOR_GEOMETRY

    # every crystal pair of every module pair in coincidence has exactly one bin
    n_c, n_t = N_CRYSTALS_PER_MODULE, CRYSTALS_PER_SIDE
    c = np.arange(n_c)
    hit = np.zeros(np.prod(sinogram_shape()), np.int32)
    for k, l in PAIRS:
        b = sinogram_bins(np.full(n_c, k), c, np.full(n_c, l), c[::-1])
        assert (b >= 0).all()
        hit[b] += 1
    print(f"round trip    {(hit > 0).sum():,} distinct bins for {len(PAIRS) * n_c:,} test pairs")

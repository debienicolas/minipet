#!/usr/bin/env python3
"""Estimate acquisition length of a MiniPET .clm.lr5 from its packet table.

Coincidence events carry no time stamp. The DAQ still writes one prompt packet
and one delayed packet per 32-bit clock epoch, so the number of packet pairs
times the epoch duration is the acquisition length:

    T = (n_packets // 2) * 2**32 * T_CLK_NS * 1e-9   [seconds]

Example:
  python clm_acquisition_length.py run4/run4_260922-153612.clm.lr5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minipet import T_CLK_NS, CLMFile

EPOCH_S = (1 << 32) * T_CLK_NS * 1e-9  # ≈ 2.865 s


def acquisition_length_from_packets(clm: CLMFile) -> float:
    """Return acquisition length in seconds from the CLM packet count."""
    n_pairs = clm.n_packets // 2  # prompt + delayed alternate, prompts first
    return n_pairs * EPOCH_S


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clm', type=Path, help='path to a .clm.lr5 file')
    args = ap.parse_args(argv)

    if not args.clm.exists():
        print(f'error: file not found: {args.clm}', file=sys.stderr)
        return 2

    clm = CLMFile(args.clm)
    T = acquisition_length_from_packets(clm)
    length_sec = clm.scan.get('lengthSec')

    print(f'file:              {clm.path}')
    print(f'packets:           {clm.n_packets} '
          f'({clm.n_prompt_packets} prompt + {clm.n_delayed_packets} delayed)')
    print(f'epoch:             {EPOCH_S:.6f} s  (= 2**32 × {T_CLK_NS} ns)')
    print(f'acquisition length:{T:8.2f} s  (from packet pairs)')
    if length_sec is not None:
        print(f'scan.lengthSec:    {length_sec:8d} s  (metadata, for comparison)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

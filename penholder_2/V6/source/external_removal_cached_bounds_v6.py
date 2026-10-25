#!/usr/bin/env python3
"""Prune impossible conservative bounds within one unchanged scene attempt.

For each supplied piece, its zero-motion convex hull is contained in every
convex translation bound used by the original verifier. If that one hull
already exceeds the original collision tolerance, its conservative route
cannot pass. The existing actual-stock boundary-prism fallback remains
unchanged. A clear precheck never certifies a route.
"""
import json
import math
import time

import numpy as np
import manifold3d as mf

import external_removal_phase_v6 as F
import plan_external_removal_v6 as P

PHASES = F.PHASES
_ORIGINAL_SAFE = P.safe_sweep
_BASE_MAKE = None
_BASE_SAVE = None
_CACHE = None
_INSTALLED = False
_STATS = dict(hulls_tested=0, cache_hits=0, conservative_calls_pruned=0,
              invalid_prechecks_fell_through=0)


def safe_sweep(pieces, direction, distance, original, obstacles):
    if _CACHE is None:
        return _ORIGINAL_SAFE(pieces, direction, distance, original, obstacles)
    for piece in pieces:
        key = (id(original), id(obstacles), id(piece))
        cached = _CACHE.get(key)
        if cached is not None:
            # Strong references make object-id reuse impossible during this
            # attempt. Manifold transforms return new objects, so their
            # positions cannot alias this key.
            old_original, old_context, old_piece, overlap = cached
            if old_original is not original or old_context is not obstacles or old_piece is not piece:
                raise RuntimeError('Unexpected identity collision in bound cache')
            _STATS['cache_hits'] += 1
        else:
            overlap = None
            _STATS['hulls_tested'] += 1
            try:
                hull = P.H.sweep(piece, np.array([1., 0., 0.]), 0.)
                if hull.status() != mf.Error.NoError:
                    raise ValueError('Invalid zero-motion hull')
                signed_volume = float(hull.volume())
                if not math.isfinite(signed_volume) or signed_volume < 0.:
                    raise ValueError('Invalid zero-motion hull volume')
                candidate = float(obstacles.obstruction(hull, original, convex=False))
                if not math.isfinite(candidate) or candidate < 0.:
                    raise ValueError('Invalid zero-motion collision result')
                overlap = candidate
            except Exception:
                # Failure to establish the bound is not a route result.
                # The original continuous verifier still runs below.
                _STATS['invalid_prechecks_fell_through'] += 1
            _CACHE[key] = (original, obstacles, piece, overlap)
        if overlap is not None and overlap > P.TOL:
            _STATS['conservative_calls_pruned'] += 1
            return False, overlap
    # Do not sum subthreshold piece overlaps and do not return success here.
    # Both would change the original per-piece continuous acceptance gate.
    return _ORIGINAL_SAFE(pieces, direction, distance, original, obstacles)


def make_plan(item, original, obstacles, broad_search=True):
    global _CACHE
    previous = _CACHE
    _CACHE = {}
    before = dict(_STATS)
    try:
        result = _BASE_MAKE(item, original, obstacles, broad_search)
        if result[0] is not None:
            result[0]['zero_motion_pruning_counts'] = {
                key: _STATS[key] - before[key] for key in _STATS}
        return result
    finally:
        # field.active changes only after this call returns. Fresh sibling
        # contexts, clipped/transformed pieces, partitions and later attempts
        # have distinct references or receive a new cache.
        _CACHE = previous


def save_plan(item, plan, sequence):
    row = _BASE_SAVE(item, plan, sequence)
    row['zero_motion_pruning_counts'] = plan.get('zero_motion_pruning_counts')
    row['zero_motion_bound_verifier'] = 'source/external_removal_cached_bounds_v6.py'
    return row


def install(phase):
    global _INSTALLED, _BASE_MAKE, _BASE_SAVE
    F.install(phase)
    P.frozen_bytes(__file__)
    if not _INSTALLED:
        _BASE_MAKE = P.make_plan
        _BASE_SAVE = P.save_plan
        _INSTALLED = True
    P.safe_sweep = safe_sweep
    P.make_plan = make_plan
    P.save_plan = save_plan


def finish_report(path):
    F.finish_report(path)
    report = json.loads(path.read_text())
    report['zero_motion_bound_verifier'] = 'source/external_removal_cached_bounds_v6.py'
    report['zero_motion_pruning_counts'] = dict(_STATS)
    report['zero_motion_pruning_scope'] = (
        'One outer make_plan attempt only; exact original/context/piece identities with strong references. '
        'A single supplied convex hull must exceed the unchanged collision tolerance at zero motion. '
        'Clear or invalid prechecks always fall through to the original continuous verifier; '
        'actual-stock boundary-prism checks and final exits are unchanged.')
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


if __name__ == '__main__':
    raise SystemExit('Use the verified continuation driver for this analysis-only overlay.')

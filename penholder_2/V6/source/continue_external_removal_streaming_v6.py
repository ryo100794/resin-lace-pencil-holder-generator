#!/usr/bin/env python3
"""Focused fine-partition continuation with lazy certified boundary prisms.

Candidates may stop only on rejection. Acceptance requires exhaustion of the
same actual-stock/all-boundary-component stream, its complete coverage proof,
the unchanged summed collision gate, and the existing final scene exit.
"""
from pathlib import Path
import argparse
import copy
import json
import math
import sys
import time
import numpy as np

import continue_external_removal_fine_v6 as F
import external_removal_extended_v6 as E
import external_removal_prism_stream_v6 as B
import plan_external_removal_v6 as P

_BASE_INSTALL = F.install
_BASE_FINISH = F.finish
_BINDINGS = None
_CURRENT_TASK = None
_LAST_PROGRESS = 0.
_BASE_FRAGMENT_ROUTE = None
_BASE_TRY_BODY = None
_BASE_OUTER_SWEEP = None
_STATS = dict(calls=0, conservative_bounds_accepted=0, exact_streams_started=0,
              exact_components_checked=0, streams_rejected_early=0,
              streams_fully_exhausted_and_accepted=0, derivation_failures=0)


def fragment_route(item, fragment, pieces, original, obstacles, preferred=(), broad_search=True):
    result = _BASE_FRAGMENT_ROUTE(item, fragment, pieces, original, obstacles, preferred, broad_search)
    if result is not None or not broad_search:
        return result
    pieces, proxy_proof = E._fragment_bounds(item, fragment, pieces)
    bounds = np.asarray(fragment.bounding_box(), float)
    directions = E.candidate_directions(item, (bounds[:3] + bounds[3:]) / 2, preferred)
    return E._short_departures(item, fragment, pieces, directions[:12], original, obstacles,
                               allow_exact=True, proxy_proof=proxy_proof)


def logged_body_cuts(item, rem, cuts, original, obstacles, saved_routes=None,
                     preferred=None, broad_search=True):
    global _CURRENT_TASK, _LAST_PROGRESS
    active = F._ACTIVE_OPTION
    tracked = (saved_routes is None and active is not None and active['id'] == item['id']
               and active['cuts'] == cuts and not active['used'])
    previous = _CURRENT_TASK
    if tracked:
        _CURRENT_TASK = dict(id=item['id'], cut_count=len(cuts))
        started = _LAST_PROGRESS = time.monotonic()
        print('Fine partition start', item['id'], 'cuts', len(cuts), flush=True)
    try:
        result = _BASE_TRY_BODY(item, rem, cuts, original, obstacles, saved_routes, preferred, broad_search)
    finally:
        _CURRENT_TASK = previous
    if tracked:
        print('Fine partition routing finished', item['id'], 'cuts', len(cuts),
              'candidate_found', result is not None, 'seconds', round(time.monotonic() - started, 2), flush=True)
    return result


def progress_sweep(*args, **kwargs):
    global _LAST_PROGRESS
    now = time.monotonic()
    if _CURRENT_TASK is not None and now - _LAST_PROGRESS >= 30.:
        _LAST_PROGRESS = now
        print('Fine search progress', json.dumps(dict(task=_CURRENT_TASK, streaming_counts=_STATS,
              early_pose_counts=F.W._STATS, later_pose_counts=F._LATE_POSE_STATS)), flush=True)
    return _BASE_OUTER_SWEEP(*args, **kwargs)


def sweep(fragment, pieces, direction, distance, original, obstacles,
          allow_exact=False, exact_only=False, proxy_proof=None):
    _STATS['calls'] += 1
    if not exact_only:
        passed, collision = P.safe_sweep(pieces, direction, distance, original, obstacles)
        if passed:
            _STATS['conservative_bounds_accepted'] += 1
            proof = dict(method='continuous_conservative_convex_piece_bounds')
            if proxy_proof is not None:
                proof['fragment_proxy_coverage'] = proxy_proof
            return True, collision, proof
        if not allow_exact:
            return False, collision, None
    _STATS['exact_streams_started'] += 1
    collision, initial_overlap, count = 0., None, 0
    try:
        stream = B.iter_translation_sweep(fragment, direction, distance)
        for index, (component, record) in enumerate(stream):
            if record.get('component_index', index) != index:
                raise B.CertificateError('Stream component order changed')
            convex_prefilter = index > 0 and initial_overlap == 0.
            overlap = float(obstacles.obstruction(component, original, convex=convex_prefilter))
            _STATS['exact_components_checked'] += 1
            if not math.isfinite(overlap) or overlap < 0.:
                _STATS['streams_rejected_early'] += 1
                return False, math.inf, None
            if index == 0:
                initial_overlap = overlap
            collision += overlap
            count += 1
            if collision > P.TOL:
                _STATS['streams_rejected_early'] += 1
                return False, collision, None
        # This method itself rejects partial consumption. There is no route
        # acceptance before every nonzero certified boundary cell is tested.
        proof = stream.coverage_proof()
        if count != proof['verification_component_count'] or count < 1:
            raise B.CertificateError('Incomplete boundary stream')
        allowance = float(proof.get('serialization_uncovered_bound_mm3', 0.))
        if not math.isfinite(allowance) or allowance < 0.:
            raise B.CertificateError('Invalid complete stream coverage allowance')
        collision += allowance
        if collision > P.TOL:
            return False, collision, None
    except B.CertificateError:
        _STATS['derivation_failures'] += 1
        return False, math.inf, None
    _STATS['streams_fully_exhausted_and_accepted'] += 1
    return True, collision, dict(
        method='continuous_union_of_initial_solid_and_all_boundary_triangle_prisms_with_summed_overlap_bound',
        component_count=count, summed_complete_scene_overlap_mm3=collision,
        initial_actual_component_overlap_mm3=initial_overlap,
        boundary_only_convex_prefilter_requires_exact_zero_initial_overlap=True,
        streaming_acceptance_requires_complete_exhaustion=True,
        construction=proof)


def install(phase):
    global _BINDINGS, _BASE_FRAGMENT_ROUTE, _BASE_TRY_BODY, _BASE_OUTER_SWEEP
    _BASE_INSTALL(phase)
    P.frozen_bytes(__file__)
    _BINDINGS = B.source_bindings()
    for entry in _BINDINGS:
        P.frozen_bytes(P.ROOT / entry['path'])
        if P.INPUT_HASHES[entry['path']] != entry['sha256']:
            raise RuntimeError('Streaming boundary helper changed')
    # Fine later-pose -> W early-pose -> this unchanged-acceptance sweep.
    # Existing wrappers and their source bytes are not modified.
    F.W._BASE_SWEEP = sweep
    _BASE_FRAGMENT_ROUTE = F._BASE_ROUTE
    F._BASE_ROUTE = fragment_route
    _BASE_TRY_BODY = P.try_body_cuts
    P.try_body_cuts = logged_body_cuts
    _BASE_OUTER_SWEEP = E._sweep
    E._sweep = progress_sweep


def finish(path):
    _BASE_FINISH(path)
    report = json.loads(path.read_text())
    report['streamed_boundary_verifier'] = dict(
        source='source/continue_external_removal_streaming_v6.py',
        source_bindings=copy.deepcopy(_BINDINGS), counts=dict(_STATS),
        actual_initial_stock_and_every_nonzero_boundary_component_required=True,
        early_stop_only_rejects=True,
        complete_coverage_proof_required_before_any_stream_acceptance=True,
        unchanged_summed_overlap_tolerance_mm3=P.TOL,
        existing_sibling_state_and_final_exit_gates_unchanged=True,
        fragment_candidate_schedule='Existing exact direct routes, then the existing short-departure family for each actual fine fragment; whole-body discovery remains disabled.',
        progress_logging_does_not_affect_success_criteria=True)
    report['global_search_phase'] = 'bound_fine_partitions_with_direct_then_short_fragment_routes'
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--partition-menu', type=Path, required=True)
    own, rest = parser.parse_known_args()
    F._MENU_PATH = own.partition_menu.resolve()
    F.require(F._MENU_PATH.is_relative_to(P.ROOT), 'Fine menu must be inside the bound project')
    sys.argv = [sys.argv[0]] + rest
    F.D.verify_prior = F.verify_prior
    F.D.S.install, F.D.S.finish_report = install, finish
    F.D.main()

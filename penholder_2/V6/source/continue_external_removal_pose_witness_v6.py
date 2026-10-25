#!/usr/bin/env python3
"""Reject blocked exact-sweep candidates using actual translated stock poses.

A colliding intermediate pose cannot belong to a collision-free continuous
translation. This optional precheck only rejects. Every clear or unavailable
sample delegates to the unchanged complete continuous sweep and exit verifier.
"""
import json
import math
import numpy as np

import continue_external_removal_exact_direct_v6 as J
import external_removal_extended_v6 as E
import plan_external_removal_v6 as P

_BASE_INSTALL = J.install
_BASE_FINISH = J.finish
_BASE_SWEEP = E._sweep
SAMPLE_DISTANCES_MM = (.05, .10, .20, .40)
_STATS = dict(calls=0, actual_poses_tested=0, blocked_sweeps_rejected=0,
              clear_calls_delegated_to_continuous_verifier=0, invalid_samples_fell_through=0)


def sweep(fragment, pieces, direction, distance, original, obstacles,
          allow_exact=False, exact_only=False, proxy_proof=None):
    _STATS['calls'] += 1
    vector = np.asarray(direction, float)
    distance = float(distance)
    if vector.shape == (3,) and np.all(np.isfinite(vector)) and math.isfinite(distance) and distance > 0:
        distances = sorted({min(distance, sample) for sample in SAMPLE_DISTANCES_MM})
        for travel in distances:
            try:
                posed = fragment.translate(vector * travel)
                collision = float(obstacles.obstruction(posed, original, convex=False))
                _STATS['actual_poses_tested'] += 1
                if not math.isfinite(collision) or collision < 0:
                    raise ValueError('Invalid optional actual-pose overlap')
            except Exception:
                # An unavailable rejection witness never supplies clearance.
                _STATS['invalid_samples_fell_through'] += 1
                continue
            if collision > P.TOL:
                _STATS['blocked_sweeps_rejected'] += 1
                return False, collision, dict(
                    method='rejection_only_intermediate_actual_stock_pose',
                    sampled_distance_mm=travel, complete_scene_overlap_mm3=collision,
                    actual_fragment_used=True, conservative_proxy_not_used=True,
                    no_clearance_claim_from_samples=True)
    _STATS['clear_calls_delegated_to_continuous_verifier'] += 1
    return _BASE_SWEEP(fragment, pieces, direction, distance, original, obstacles,
                       allow_exact=allow_exact, exact_only=exact_only, proxy_proof=proxy_proof)


def install(phase):
    _BASE_INSTALL(phase)
    P.frozen_bytes(__file__)
    E._sweep = sweep


def finish(path):
    _BASE_FINISH(path)
    report = json.loads(path.read_text())
    report['actual_pose_rejection_precheck'] = dict(
        source='source/continue_external_removal_pose_witness_v6.py',
        sample_distances_mm=list(SAMPLE_DISTANCES_MM), counts=dict(_STATS),
        actual_signed_fragment_used=True, complete_scene_convex_prefilter_disabled=True,
        acceptance_shortcut=False, clear_samples_always_delegate_to_unchanged_continuous_verifier=True,
        tolerance_mm3=P.TOL)
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


if __name__ == '__main__':
    J.D.verify_prior = J.verify_prior
    J.D.S.install, J.D.S.finish_report = install, finish
    J.D.main()

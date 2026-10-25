#!/usr/bin/env python3
"""Revalidate explicit residual instructions against a verified prior scene.

The existing continuation verifies the complete immutable prior proof and
recreates its exact residual obstacle state. This overlay permits only saved
instruction replay, then lets that same state machine close a complete report.
No new path discovery or geometry change is performed.
"""
import json

import continue_external_removal_cached_bounds_v6 as D
import external_removal_extended_v6 as E
import plan_external_removal_v6 as P

_BASE_INSTALL = D.S.install
_BASE_FINISH = D.S.finish_report
_BASE_SWEEP = E._sweep
_ACTIVE_ID = None
_CHECKED_IDS = set()


def diagnostic_sweep(*args, **kwargs):
    result = _BASE_SWEEP(*args, **kwargs)
    if _ACTIVE_ID is not None and not result[0]:
        proof = result[2] or {}
        summary = {key: value for key, value in proof.items()
                   if key not in ('components', 'component_proofs', 'triangle_prisms',
                                  'source_triangles', 'prism_certificates')}
        print('Candidate failed sweep', _ACTIVE_ID,
              json.dumps(dict(overlap_mm3=result[1], proof=summary,
                              direction=list(args[2]), distance_mm=float(args[3]),
                              fragment_bounds_mm=list(args[0].bounding_box())), default=str)[:4500],
              flush=True)
    return result


def install(phase):
    _BASE_INSTALL(phase)
    P.frozen_bytes(__file__)
    base_make = P.make_plan
    base_body = P.try_body_cuts
    base_verify = P.verify_saved_legs

    def no_discovery(*args, **kwargs):
        return None

    def replay_body(item, rem, cuts, original, obstacles, saved_routes=None,
                    preferred=None, broad_search=True):
        if saved_routes is None:
            return None
        return base_body(item, rem, cuts, original, obstacles, saved_routes,
                         preferred, broad_search)

    def verify(item, fragment, pieces, legs, original, obstacles):
        global _ACTIVE_ID
        old = _ACTIVE_ID
        _ACTIVE_ID = item['id']
        try:
            result = base_verify(item, fragment, pieces, legs, original, obstacles)
        finally:
            _ACTIVE_ID = old
        if result is None:
            print('Explicit instruction not revalidated', item['id'], flush=True)
        return result

    def make(item, original, obstacles, broad_search=True):
        if item['id'] not in P.ROUTE_SEEDS:
            return None, dict(reason='No explicit candidate supplied in this continuation phase')
        _CHECKED_IDS.add(item['id'])
        plan, error = base_make(item, original, obstacles, broad_search)
        if plan is not None and plan['mode'] != 'endpoint_cuts_only' and not plan.get('candidate_route_revalidated'):
            raise RuntimeError('New route discovery occurred in candidate-only continuation')
        return plan, error

    P.route_fragment = no_discovery
    P.try_body_cuts = replay_body
    P.verify_saved_legs = verify
    P.make_plan = make
    E._sweep = diagnostic_sweep


def finish(path):
    _BASE_FINISH(path)
    report = json.loads(path.read_text())
    accepted = [row['id'] for row in report['plans']]
    residual = [row['id'] for row in report['remaining_failures']]
    if len(set(accepted)) != len(accepted) or len(set(residual)) != len(residual) or set(accepted) & set(residual):
        raise RuntimeError('Candidate continuation has duplicated accepted/residual IDs')
    all_ids = set(accepted) | set(residual)
    if len(all_ids) != report['initial_support_count'] or not set(P.ROUTE_SEEDS) <= all_ids:
        raise RuntimeError('Candidate continuation does not cover the source operation set')
    if 'unsearched_operation_ids' in report:
        report['prior_prefix_unsearched_operation_ids'] = report.pop('unsearched_operation_ids')
    report['candidate_only_continuation'] = dict(
        source='source/continue_external_removal_candidates_v6.py',
        requested_ids=sorted(P.ROUTE_SEEDS), checked_ids=sorted(_CHECKED_IDS),
        all_new_non_cut_only_plans_are_explicit_instruction_replays=True,
        new_path_discovery_disabled=True)
    report['global_search_phase'] = 'explicit_candidate_continuation_only'
    report['candidate_family_order'] = 'Only supplied residual instructions are revalidated; no new route discovery.'
    report['phase_scope'] = ('The immutable prior accepted sequence is reused only after complete byte binding. '
        'Every newly accepted operation is fully revalidated in the actual remaining scene. '
        'All unresolved source IDs are explicit; no infeasibility claim is made.')
    report['complete_input_and_residual_ledger'] = True
    tmp = path.with_suffix('.json.part')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(path)


if __name__ == '__main__':
    D.S.install = install
    D.S.finish_report = finish
    D.main()

#!/usr/bin/env python3
"""Freshly certify selected instructions and close a complete partial proof.

No interrupted-run outcome is reused. All source/aggregate bindings, actual
stock and cuts, every continuous move, sibling state and final exit are
revalidated by the existing verifier. New route discovery is disabled.
Unselected operations receive explicit residual entries, so the unchanged
main state machine can finish its full input ledger without a --limit.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import external_removal_cached_bounds_v6 as C
import plan_external_removal_v6 as P

_SELECTED_IDS = set()
_SOURCE_IDS = set()
_BASE_MAKE = None
_BASE_LOAD = None
_BASE_TRY_BODY = None


def load_parts():
    global _SOURCE_IDS
    result = _BASE_LOAD()
    _SOURCE_IDS = {item['id'] for item in result[0]}
    if len(_SOURCE_IDS) != len(result[0]):
        raise ValueError('Duplicate source operation ID')
    if not _SELECTED_IDS <= _SOURCE_IDS:
        raise ValueError('Instruction prefix contains unknown operation IDs')
    return result


def no_new_routes(*args, **kwargs):
    return None


def replay_body_cuts(item, rem, cuts, original, obstacles, saved_routes=None,
                     preferred=None, broad_search=True):
    if saved_routes is None:
        return None
    return _BASE_TRY_BODY(item, rem, cuts, original, obstacles,
                         saved_routes=saved_routes, preferred=preferred,
                         broad_search=broad_search)


def make_plan(item, original, obstacles, broad_search=True):
    if item['id'] not in _SELECTED_IDS:
        return None, dict(reason='Not searched in this instruction-prefix certification phase')
    plan, error = _BASE_MAKE(item, original, obstacles, broad_search)
    if plan is not None:
        if plan['mode'] != 'endpoint_cuts_only' and not plan.get('candidate_route_revalidated'):
            raise RuntimeError('A new route was generated in instruction-only certification')
    return plan, error


def finish_report(path):
    C.finish_report(path)
    report = json.loads(path.read_text())
    actual = {row['id'] for row in report['plans']}
    residual = {row['id'] for row in report['remaining_failures']}
    if len(actual) != len(report['plans']) or len(residual) != len(report['remaining_failures']):
        raise RuntimeError('Duplicate accepted or residual operation ID')
    if not actual <= _SELECTED_IDS or actual | residual != _SOURCE_IDS or actual & residual:
        raise RuntimeError('Instruction-prefix report does not close the exact source operation set')
    report['global_search_phase'] = 'instruction_prefix_certification_only'
    report['prefix_certification_source'] = 'source/certify_external_removal_prefix_v6.py'
    report['candidate_family_order'] = 'Only frozen selected instructions are revalidated; new route discovery is disabled.'
    report['search_order'] = 'Selected immutable candidate order; explicit unsearched residuals; no checkpoint outcomes reused.'
    report['phase_scope'] = ('Complete fresh validation of the accepted selected instructions and the full input ledger. '
        'Every source operation is explicitly accepted or residual. Unselected operations were not searched, '
        'and no claim of infeasibility is made for them.')
    report['instruction_prefix_requested_count'] = len(_SELECTED_IDS)
    report['instruction_prefix_validated_count'] = len(actual)
    report['instruction_prefix_unvalidated_ids'] = sorted(_SELECTED_IDS - actual)
    report['unsearched_operation_ids'] = sorted(_SOURCE_IDS - _SELECTED_IDS)
    report['complete_input_and_residual_ledger'] = True
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def main():
    global _SELECTED_IDS, _BASE_MAKE, _BASE_LOAD, _BASE_TRY_BODY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--assembly-context', type=Path, required=True)
    args = parser.parse_args()
    candidate_path = args.candidates.resolve()
    context_path = args.assembly_context.resolve()
    if not candidate_path.is_relative_to(P.ROOT) or not context_path.is_relative_to(P.ROOT):
        raise ValueError('Inputs must be inside the bound project')
    C.install('direct')
    P.frozen_bytes(__file__)
    seed = P.metadata(candidate_path)
    _SELECTED_IDS = {row['id'] for row in seed['plans']}
    if not _SELECTED_IDS or len(_SELECTED_IDS) != len(seed['plans']):
        raise ValueError('Empty or duplicate instruction prefix')
    _BASE_MAKE, _BASE_LOAD, _BASE_TRY_BODY = P.make_plan, P.load_parts, P.try_body_cuts
    P.make_plan, P.load_parts = make_plan, load_parts
    P.route_fragment, P.try_body_cuts = no_new_routes, replay_body_cuts
    sys.argv = [sys.argv[0], '--assembly-context', str(context_path),
                '--reuse-plans', str(candidate_path)]
    target = P.ROOT / 'reports/v6_external_removal_validation.json'
    started_ns = time.time_ns()
    try:
        P.main()
    except SystemExit as error:
        if error.code == 2 and target.exists() and target.stat().st_mtime_ns >= started_ns:
            finish_report(target)
        raise
    finish_report(target)


if __name__ == '__main__':
    main()

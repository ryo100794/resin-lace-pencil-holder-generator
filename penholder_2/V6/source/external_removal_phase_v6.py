#!/usr/bin/env python3
"""Finish one global search phase with a complete immutable partial proof.

Every remaining operation receives the selected candidate family before a
later phase is considered. The unchanged planner writes its complete input
ledger and explicit remaining set even when this phase has not solved all
operations. A separate, verified continuation can then use that proof.
"""
import argparse
import hashlib
import json
import sys
import time

import external_removal_staged_v6 as G
import external_removal_extended_v6 as E
import plan_external_removal_v6 as P

PHASES = ('direct', 'short_departures', 'exact_components')
_PHASE = None


def make_plan(item, original, obstacles, broad_search=True):
    G._STAGE = _PHASE
    plan, error = E.make_plan(item, original, obstacles, broad_search)
    if plan is not None:
        plan['candidate_search_stage'] = _PHASE if broad_search else 'preferred_direct'
    return plan, error


def install(phase):
    global _PHASE
    if phase not in PHASES:
        raise ValueError('Unknown search phase: ' + str(phase))
    if _PHASE is not None and _PHASE != phase:
        raise ValueError('A proof process cannot change its frozen search phase')
    _PHASE = phase
    G.install()
    source = P.frozen_bytes(__file__)
    config = dict(schema='external_removal_search_phase_v1', search_phase=phase,
                  source='source/external_removal_phase_v6.py',
                  source_sha256=hashlib.sha256(source).hexdigest(),
                  all_stock_cut_collision_and_exit_gates_unchanged=True)
    data = (json.dumps(config, indent=2, sort_keys=True) + '\n').encode()
    digest = hashlib.sha256(data).hexdigest()
    path = P.OUT / ('search_phase_' + digest[:20] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Immutable search phase configuration changed')
    else:
        path.write_bytes(data)
    P.frozen_bytes(path)
    P.make_plan = make_plan


def finish_report(path):
    G.finish_report(path)
    report = json.loads(path.read_text())
    report['global_search_phase'] = _PHASE
    report['global_phase_controller'] = 'source/external_removal_phase_v6.py'
    report['candidate_family_order'] = (
        'Revalidated saved instructions and preferred direct passes, then global passes of the frozen '
        + _PHASE + ' candidate family. No later family is searched in this process; unresolved IDs are '
        'explicit. Saved instructions are fully revalidated regardless of their original discovery family.')
    report['phase_scope'] = ('All remaining operations received preferred direct passes and the selected '
        'global candidate family. Unresolved operations are explicit. This report records the full source '
        'ledger and actual accepted sequence and may serve as a verified continuation prefix.')
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--search-phase', choices=PHASES, required=True)
    parser.add_argument('--limit', type=int)
    options, remaining = parser.parse_known_args()
    if options.limit is not None:
        parser.error('--limit is disabled: a complete global pass and residual ledger are required')
    sys.argv = [sys.argv[0]] + remaining
    install(options.search_phase)
    started_ns = time.time_ns()
    target = P.ROOT / 'reports/v6_external_removal_validation.json'
    try:
        P.main()
    except SystemExit as error:
        if error.code == 2 and target.exists() and target.stat().st_mtime_ns >= started_ns:
            finish_report(target)
        raise
    finish_report(target)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Replay residual instructions, then search only bounded fine body partitions.

The immutable menu proposes cut planes, never routes. Every partition, actual
kerf, endpoint cut, sequential sibling state, continuous movement and exit is
recomputed through the existing verifier. Whole-body discovery is disabled.
"""
from pathlib import Path
import argparse
import copy
import json
import math
import sys
import numpy as np

import continue_external_removal_pose_witness_v6 as W
import external_removal_extended_v6 as E
import plan_external_removal_v6 as P

D = W.J.D
_BASE_VERIFY = W.J.verify_prior
_BASE_INSTALL = W.install
_BASE_FINISH = W.finish
_MENU_PATH = None
_MENU = None
_MENU_HASH = None
_RECORDS = {}
_PRIOR = None
_BASE_MAKE = None
_BASE_ROUTE = None
_BASE_LOAD = None
_BASE_SAVE = None
_BASE_SWEEP = None
_BASE_TRY_BODY = None
_ACTIVE_OPTION = None
_ATTEMPTS = []
_LATE_POSE_STATS = dict(tested=0, rejected=0, invalid=0)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def small(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= P.TOL


def verify_prior(prior):
    global _PRIOR
    result = _BASE_VERIFY(prior)
    _PRIOR = prior
    return result


def load_parts():
    result = _BASE_LOAD()
    by_id = {item['id']: item for item in result[0]}
    require(set(_RECORDS) <= set(by_id), 'Fine menu contains an unknown source operation')
    for identifier, row in _RECORDS.items():
        item = by_id[identifier]
        require('endpoint_partition' in item, 'Fine menu expected the adopted finite-cut stock')
        expected = item['endpoint_partition']['paths']['removable']
        require(row['actual_stock'] == expected, 'Fine menu actual stock differs for ' + identifier)
        require(P.INPUT_HASHES[expected['path']] == expected['sha256'], 'Fine menu stock hash changed')
    return result


def cuts_only_route(item, fragment, pieces, original, obstacles, preferred=(), broad_search=True):
    if not item.get('_active_body_cuts'):
        return None
    return _BASE_ROUTE(item, fragment, pieces, original, obstacles, preferred, broad_search)


def restricted_body_cuts(item, rem, cuts, original, obstacles, saved_routes=None,
                         preferred=None, broad_search=True):
    if saved_routes is None:
        if (_ACTIVE_OPTION is None or _ACTIVE_OPTION['id'] != item['id'] or
                _ACTIVE_OPTION['cuts'] != cuts or _ACTIVE_OPTION['used']):
            return None
        _ACTIVE_OPTION['used'] = True
    return _BASE_TRY_BODY(item, rem, cuts, original, obstacles, saved_routes,
                          preferred, broad_search)


def later_pose_sweep(fragment, pieces, direction, distance, original, obstacles,
                     allow_exact=False, exact_only=False, proxy_proof=None):
    vector = np.asarray(direction, float)
    distance = float(distance)
    if vector.shape == (3,) and np.all(np.isfinite(vector)) and math.isfinite(distance) and distance > .4:
        for travel in sorted({min(distance, sample) for sample in (.8, 1.6, 3.2, 6.4, 12.8)} | {distance}):
            if travel <= .4:
                continue
            try:
                collision = float(obstacles.obstruction(fragment.translate(vector * travel), original, convex=False))
                _LATE_POSE_STATS['tested'] += 1
                if not math.isfinite(collision) or collision < 0:
                    raise ValueError('Invalid optional later-pose overlap')
            except Exception:
                _LATE_POSE_STATS['invalid'] += 1
                continue
            if collision > P.TOL:
                _LATE_POSE_STATS['rejected'] += 1
                return False, collision, dict(method='rejection_only_later_actual_stock_pose',
                    sampled_distance_mm=travel, complete_scene_overlap_mm3=collision,
                    no_clearance_claim_from_samples=True)
    return _BASE_SWEEP(fragment, pieces, direction, distance, original, obstacles,
                       allow_exact=allow_exact, exact_only=exact_only, proxy_proof=proxy_proof)


def runtime_fine_checks(item, plan, option):
    parts = [entry[0] for entry in plan['fragments']]
    require(len(parts) == option['resulting_fragment_count'], 'Fine partition interval count changed')
    require(all(part.volume() > 0. for part in parts), 'Fine partition contains an empty/nonpositive interval')
    rem, kerf = item['removable'], plan['actual_body_kerf']
    checks = dict(
        sum_fragment_outside_actual_removable_mm3=sum(P.H.volume(part - rem) for part in parts),
        kerf_outside_actual_removable_mm3=P.H.volume(kerf - rem),
        sum_fragment_kerf_interior_overlap_mm3=sum(P.H.volume(part ^ kerf) for part in parts),
        sum_pairwise_fragment_interior_overlap_mm3=sum(P.H.volume(parts[i] ^ parts[j])
            for i in range(len(parts)) for j in range(i)))
    return checks if all(small(x) for x in checks.values()) else None


def make_plan(item, original, obstacles, broad_search=True):
    global _ACTIVE_OPTION
    # Supplied seeds include any interrupted-run delta as instructions only.
    # They receive the ordinary complete current-state replay before new cuts.
    old_seed = P.ROUTE_SEEDS.get(item['id'])
    if old_seed is not None:
        plan, error = _BASE_MAKE(item, original, obstacles, False)
        if plan is not None:
            require(plan.get('candidate_route_revalidated') is True or plan['mode'] == 'endpoint_cuts_only',
                    'Unexpected route discovery during explicit instruction replay')
            return plan, None
    row = _RECORDS.get(item['id'])
    if row is None:
        return None, dict(reason='No explicitly validated fine partition menu for this residual operation')
    last_error = None
    for option in row['options']:
        # P's explicit-body-cut branch already accepts saved_routes=None to
        # generate routes for supplied planes. All its normal preconditions
        # run first. The outer controller corrects the generic seed provenance.
        seed = dict(id=item['id'], mode='explicit_body_cuts',
                    additional_body_cuts=option['additional_body_cuts'], fragments=None)
        P.ROUTE_SEEDS[item['id']] = seed
        require(_ACTIVE_OPTION is None, 'Nested fine partition attempt')
        _ACTIVE_OPTION = dict(id=item['id'], cuts=option['additional_body_cuts'], used=False)
        try:
            plan, error = _BASE_MAKE(item, original, obstacles, False)
        finally:
            _ACTIVE_OPTION = None
            if old_seed is None:
                P.ROUTE_SEEDS.pop(item['id'], None)
            else:
                P.ROUTE_SEEDS[item['id']] = old_seed
        if plan is not None:
            require(plan['mode'] == 'explicit_body_cuts' and plan['cuts'] == option['additional_body_cuts'],
                    'A returned fine plan used different cut instructions')
            checks = runtime_fine_checks(item, plan, option)
            if checks is None:
                _ATTEMPTS.append(dict(id=item['id'], option_id=option['option_id'],
                    passed=False, reason='Runtime fine-stock containment/disjointness failed'))
                continue
            plan['candidate_route_revalidated'] = False
            plan['route_candidate_source'] = 'new_current_state_routes_for_bound_fine_partition_menu'
            plan['fine_partition_menu'] = dict(path=str(_MENU_PATH.relative_to(P.ROOT)), sha256=_MENU_HASH,
                option_id=option['option_id'], actual_stock=row['actual_stock'],
                requested_piece_count=option['requested_piece_count'],
                actual_projection_pitch_mm=option['actual_projection_pitch_mm'],
                all_fragment_routes_newly_verified=True, runtime_extra_stock_checks=checks)
            _ATTEMPTS.append(dict(id=item['id'], option_id=option['option_id'], passed=True))
            return plan, None
        last_error = error
        _ATTEMPTS.append(dict(id=item['id'], option_id=option['option_id'], passed=False, reason=error['reason']))
        if error['reason'] not in {'Preferred departures blocked; deferred until easier neighbours are removed',
                                  'No certified whole or segmented route with remaining model/supports present',
                                  'No clear whole-piece route'}:
            return None, error
    return None, dict(reason='No complete-scene direct extraction for any bound fine partition option',
                      last_partition_failure=last_error)


def save_plan(item, plan, sequence):
    row = _BASE_SAVE(item, plan, sequence)
    if 'fine_partition_menu' in plan:
        row['fine_partition_menu'] = copy.deepcopy(plan['fine_partition_menu'])
    return row


def install(phase):
    global _MENU, _MENU_HASH, _RECORDS, _BASE_MAKE, _BASE_ROUTE, _BASE_LOAD, _BASE_SAVE, _BASE_SWEEP, _BASE_TRY_BODY
    _BASE_INSTALL(phase)
    P.frozen_bytes(__file__)
    _MENU = P.metadata(_MENU_PATH)
    _MENU_HASH = P.INPUT_HASHES[str(_MENU_PATH.relative_to(P.ROOT))]
    require(_MENU['schema'] == 'finer_body_partition_menu_v1' and
            _MENU['unchanged_volume_tolerance_mm3'] == P.TOL and
            _MENU['printed_geometry_modified'] is False, 'Wrong fine partition scope')
    bindings = dict(_MENU['source_bindings'])
    bindings[_MENU['source']['path']] = _MENU['source']['sha256']
    for path, digest in bindings.items():
        P.frozen_bytes(P.ROOT / path)
        require(P.INPUT_HASHES[path] == digest, 'Fine menu source changed: ' + path)
    all_records = {row['id']: row for row in _MENU['records']}
    require(len(all_records) == len(_MENU['records']), 'Duplicate fine menu ID')
    active_ids = {row['id'] for row in _PRIOR['remaining_failures']}
    source_ids = active_ids | {row['id'] for row in _PRIOR['plans']}
    require(set(all_records) <= source_ids, 'Fine menu contains unknown source IDs')
    _RECORDS = {identifier: row for identifier, row in all_records.items() if identifier in active_ids}
    for row in _RECORDS.values():
        for option in row['options']:
            require(option['every_requested_interval_preserved'] is True and
                    option['all_signed_components_preserved'] is True and
                    option['resulting_fragment_count'] == option['requested_piece_count'] and
                    all(small(x) for x in option['local_coverage_disjointness_checks'].values()),
                    'Unverified local fine partition option')
    _BASE_MAKE, _BASE_ROUTE, _BASE_LOAD, _BASE_SAVE = P.make_plan, P.route_fragment, P.load_parts, P.save_plan
    _BASE_TRY_BODY = P.try_body_cuts
    _BASE_SWEEP = E._sweep
    P.make_plan, P.route_fragment, P.load_parts, P.save_plan = make_plan, cuts_only_route, load_parts, save_plan
    P.try_body_cuts = restricted_body_cuts
    E._sweep = later_pose_sweep


def finish(path):
    _BASE_FINISH(path)
    report = json.loads(path.read_text())
    if 'candidate_only_continuation' in report:
        report['prior_candidate_only_continuation'] = report.pop('candidate_only_continuation')
    report['global_search_phase'] = 'explicit_seed_replay_then_bound_fine_partitions_only'
    report['fine_partition_search'] = dict(source='source/continue_external_removal_fine_v6.py',
        menu_path=str(_MENU_PATH.relative_to(P.ROOT)), menu_sha256=_MENU_HASH,
        whole_body_discovery_disabled=True, new_routes_recomputed_for_actual_cut_stock=True,
        targeted_remaining_ids=sorted(_RECORDS), attempts=_ATTEMPTS,
        later_actual_pose_rejection_counts=_LATE_POSE_STATS,
        later_pose_distances='min(leg,.8/1.6/3.2/6.4/12.8mm) and leg end; rejection only')
    report['candidate_family_order'] = ('Explicit residual instructions are revalidated first. Only the bound fine '
        'partition menu is then searched through unchanged endpoint/stock/kerf/sequential-sibling/sweep/exit gates. '
        'No whole-body discovery or additional generic cut menu is searched.')
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--partition-menu', type=Path, required=True)
    own, rest = parser.parse_known_args()
    _MENU_PATH = own.partition_menu.resolve()
    require(_MENU_PATH.is_relative_to(P.ROOT), 'Fine menu must be inside the bound project')
    sys.argv = [sys.argv[0]] + rest
    D.verify_prior = verify_prior
    D.S.install, D.S.finish_report = install, finish
    D.main()

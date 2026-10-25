#!/usr/bin/env python3
"""Plan direct routes and all cut partitions before expensive route fallbacks.

This scheduling overlay preserves every stock, partition, complete-scene
collision and final-exit gate in the frozen extended and sequential verifiers.
It changes only the order in which candidate families are searched.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

import external_removal_extended_v6 as E
import external_removal_plate_stock_v6_b2 as B2
import plan_external_removal_v6 as P

_STAGE = 'direct'
_INSTALLED = False


def prepare_plate(item, original):
    """Bind the compositional policy without editing the earlier verifier."""
    if item['id'] not in E.B.PLATE_IDS or '_plate_stock' in item:
        return
    source_keys = ['cache/original_v5.npz', 'source/external_removal_plate_stock_v6.py',
                   'source/external_removal_plate_stock_v6_b2.py']
    source_keys += ['cache/custom_holder_supports/' + item['record'][field]
                    for field in ['full_mesh_file', 'removable_mesh_file']]
    expected = {key: P.INPUT_HASHES[key] for key in source_keys}
    prepared = B2.prepare_plate_stock(item, original,
        original_entry=dict(path='cache/original_v5.npz', sha256=expected['cache/original_v5.npz']),
        expected_input_hashes=expected,
        cache_root=P.ROOT / 'cache/external_removal_plate_stock_v6_b2')
    manifest_path = Path(prepared['manifest_path'])
    manifest_data = P.frozen_bytes(manifest_path)
    record = prepared['record']
    if record.get('passed') is not True:
        raise ValueError('Actual finishing-stock partition has not passed: ' + item['id'])
    for entry in record['bindings'].values():
        E._freeze_entry(entry)
    for entry in record['files']:
        E._freeze_entry(entry)
    item['_plate_stock'] = dict(manifest_path=str(manifest_path.relative_to(P.ROOT)),
        manifest_sha256=hashlib.sha256(manifest_data).hexdigest(), record=record)
    item['_original_nominal_removable'] = item['removable']
    item['removable'] = prepared['stock']
    item['pieces'] = [prepared['stock']]


def route_fragment(item, fragment, pieces, original, obstacles, preferred=(), broad_search=True):
    if not broad_search:
        return E.route_fragment(item, fragment, pieces, original, obstacles, preferred, False)
    if _STAGE == 'exact_components':
        return E.route_fragment(item, fragment, pieces, original, obstacles, preferred, True)
    pieces, proxy_proof = E._fragment_bounds(item, fragment, pieces)
    if item['kind'] == 'plate_perimeter_tack':
        return E._plate_routes(item, fragment, pieces, original, obstacles,
                               broad_search=_STAGE == 'short_departures')
    bb = np.asarray(fragment.bounding_box(), float)
    center = (bb[:3] + bb[3:]) / 2
    directions = E.candidate_directions(item, center, preferred)
    direct = E._direct_routes(item, fragment, pieces, directions, original, obstacles,
                               allow_exact=False, proxy_proof=proxy_proof)
    if direct is not None or _STAGE == 'direct':
        return direct
    return E._short_departures(item, fragment, pieces, directions[:16], original, obstacles,
                                allow_exact=False, proxy_proof=proxy_proof)


def make_plan(item, original, obstacles, broad_search=True):
    global _STAGE
    if not broad_search:
        _STAGE = 'direct'
        plan, error = E.make_plan(item, original, obstacles, False)
        if plan is not None:
            plan['candidate_search_stage'] = 'preferred_direct'
        return plan, error
    error = None
    for stage in ['direct', 'short_departures', 'exact_components']:
        _STAGE = stage
        # The unchanged make_plan traverses declared cuts, the whole body and
        # every generic cut count within this stage. Thus direct split-body
        # solutions are considered before any expensive whole-body fallback.
        plan, error = E.make_plan(item, original, obstacles, True)
        if plan is not None:
            plan['candidate_search_stage'] = stage
            return plan, None
        if error['reason'] not in {
            'No certified whole or segmented route with remaining model/supports present',
            'No clear whole-piece route',
        }:
            # A failed stock identity, endpoint cut, or initial-overlap gate
            # cannot be repaired by selecting another translation path.
            return None, error
    return None, error


def save_plan(item, plan, sequence):
    row = E.save_plan(item, plan, sequence)
    row['candidate_search_stage'] = plan.get('candidate_search_stage')
    row['candidate_scheduler'] = 'source/external_removal_staged_v6.py'
    return row


def install():
    global _INSTALLED
    E.install()
    P.frozen_bytes(__file__)
    P.frozen_bytes(B2.__file__)
    if not _INSTALLED:
        E.prepare_plate = prepare_plate
        P.route_fragment = route_fragment
        P.make_plan = make_plan
        P.save_plan = save_plan
        _INSTALLED = True


def finish_report(path):
    E.finish_report(path)
    report = json.loads(path.read_text())
    report['candidate_scheduler'] = 'source/external_removal_staged_v6.py'
    report['candidate_family_order'] = (
        'Preferred direct passes, then for each remaining operation: direct whole/declared/all generic '
        'body-cut routes; short whole/body-fragment departures; conservative boundary-prism component '
        'fallback last. Every accepted cut, swept move and full exit is checked in its actual state.')
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def main():
    install()
    started_ns = time.time_ns()
    path = P.ROOT / 'reports/v6_external_removal_validation.json'
    try:
        P.main()
    except SystemExit as error:
        if error.code == 2 and path.exists() and path.stat().st_mtime_ns >= started_ns:
            finish_report(path)
        raise
    finish_report(path)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Bounded local diagnosis and INSTRUCTION-ONLY candidates for L3 support 11.

Loads only this support's finite-cut stock, caps and five covering proxies.
No whole original, scene index, printed geometry or active input is changed.
Local cap clearance is not a full-scene route or physical access certificate.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

import custom_holder_supports as H

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / 'cache/external_removal_v6/pre_prefix_certification_3222abc225274e22.json'
SNAPSHOT_SHA = '3222abc225274e2279b98f68b28bf33f0c9321edb163f0bfcfa9072d59bbc0bc'
TOL = 1e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def immutable(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('An immutable diagnostic path already has different bytes')
    else:
        path.write_bytes(data)


def main():
    if sha(SNAPSHOT) != SNAPSHOT_SHA:
        raise ValueError('Frozen instruction snapshot changed')
    remaining = set(json.loads(SNAPSHOT.read_text())['remaining_ids'])
    target_ids = [f'S3_A{a:03d}_011' for a in (80, 200, 320)]
    if not set(target_ids) <= remaining:
        raise ValueError('Requested IDs are not all in the frozen remaining set')
    manifest_path = ROOT / 'reports/sphere_removal_partition_v6.json'
    manifest = json.loads(manifest_path.read_text())
    row = next(r for r in manifest['records'] if r.get('layer_count') == 3
               and r.get('support_index') == 11 and r.get('placement') is None)
    inputs = {str(Path(__file__).relative_to(ROOT)): sha(__file__),
              str(manifest_path.relative_to(ROOT)): sha(manifest_path),
              str(SNAPSHOT.relative_to(ROOT)): SNAPSHOT_SHA,
              'source/custom_holder_supports.py': sha(ROOT / 'source/custom_holder_supports.py')}

    def load(entry):
        path = ROOT / entry['path']
        digest = sha(path)
        if digest != entry['sha256']:
            raise ValueError('Finite-cut mesh binding changed')
        inputs[entry['path']] = digest
        return H.load(path)[0]

    actual = load(row['paths']['removable'])
    caps = load(row['paths']['retained'])
    old = [load(e) for e in row['paths']['proxies']]
    box = actual.bounding_box()
    tight = [p.trim_by_plane((0, 1, 0), box[1])
               .trim_by_plane((0, -1, 0), -box[4]) for p in old]
    tight_union = H.union(tight)
    coverage = dict(actual_uncovered_mm3=H.volume(actual - tight_union),
                    covering_excess_mm3=H.volume(tight_union - actual))
    if max(coverage.values()) > TOL:
        raise ValueError('Local tight convex cover failed')
    tests = []
    for down in (.025, .030, .035):
        result = dict(initial_local_downward_travel_mm=down)
        for label, pieces in [('old_proxy', old), ('actual_tight_cover', tight)]:
            values = []
            current = pieces
            for direction, distance in [([0, -1, 0], down),
                                        ([0, 0, 1], 20.), ([0, 1, 0], 120.)]:
                # This sum is conservative even if query pieces overlap.
                values.append(sum(H.volume(H.sweep(piece, direction, distance) ^ caps)
                                  for piece in current))
                current = [piece.translate(np.asarray(direction) * distance)
                           for piece in current]
            result[label + '_own_cap_summed_overlap_per_leg_mm3'] = values
        tests.append(result)
    if any(max(t['actual_tight_cover_own_cap_summed_overlap_per_leg_mm3']) > TOL
           for t in tests):
        raise ValueError('Suggested departure fails even its own local caps')

    plans = []
    for angle, target in zip((80, 200, 320), target_ids):
        radians = np.deg2rad(angle)
        radial = np.array([np.cos(radians), np.sin(radians), 0.])
        radial /= np.linalg.norm(radial)
        plans.append(dict(id=target, mode='whole_piece', additional_body_cuts=[],
            fragments=[dict(body_section_index=0, translation_legs=[
                dict(direction=[0., 0., -1.], distance_mm=.030),
                dict(direction=radial.tolist(), distance_mm=20.),
                dict(direction=[0., 0., 1.], distance_mm=120.)])],
            fragment_extraction_order=[0]))
    candidate_data = encoded(dict(plans=plans))
    digest = hashlib.sha256(candidate_data).hexdigest()
    candidate_path = ROOT / ('cache/external_removal_v6/'
                             f'sphere3_support11_departure_candidates_{digest}.json')
    immutable(candidate_path, candidate_data)
    report = dict(scope='Local support-11 cap diagnosis only; no whole-scene acceptance',
        input_sha256=inputs, candidate_path=str(candidate_path.relative_to(ROOT)),
        candidate_sha256=digest, candidate_count=len(plans),
        candidate_status='Instruction-only; full endpoint cuts, every sweep and final exit must be revalidated',
        whole_scene_loaded=False, printed_geometry_modified=False,
        local_coordinates='Build +Y; +Z radially outward; +X tangential',
        actual_body_bounds_mm=list(box), actual_body_volume_mm3=H.volume(actual),
        old_proxy_excess_over_actual_mm3=H.volume(H.union(old) - actual),
        actual_tight_cover_checks=coverage, local_tests=tests,
        interpretation=('The existing .05 mm minimum short departure exceeds the .04 mm '
            'finite endpoint clearance. A .030 mm downward departure fits within that clearance. '
            'Old proxies extend .020 mm into the removed endpoint bands and falsely obstruct '
            'this path against the lower cap. Actual-stock replay is required. The outward '
            'and upward legs have only been checked against this support\'s own caps.'),
        candidate_alternatives_initial_down_mm=[.025, .035],
        full_scene_proof_reused=False)
    report_data = encoded(report)
    report_sha = hashlib.sha256(report_data).hexdigest()
    report_path = ROOT / ('reports/'
        f'sphere3_support11_departure_diagnostic_{report_sha[:16]}.json')
    immutable(report_path, report_data)
    print(json.dumps(dict(candidate_path=str(candidate_path.relative_to(ROOT)),
        candidate_sha256=digest, report_path=str(report_path.relative_to(ROOT)),
        report_sha256=report_sha, local_tests=tests), indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Local-only feasibility check for nine residual holder removal operations.

No original/full scene is loaded. Printed stock, retained caps, active sources
and running planner inputs remain unchanged. The only candidate is an
instruction-only H0257 finite body-cut plan, requiring complete revalidation.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

import custom_holder_supports as H
import plan_external_removal_v6 as P

ROOT = Path(__file__).resolve().parent.parent
IDS = [206, 226, 257, 267, 277, 384, 492, 550, 613]
SNAPSHOT = 'cache/external_removal_v6/pre_prefix_certification_3222abc225274e22.json'
SNAPSHOT_SHA = '3222abc225274e2279b98f68b28bf33f0c9321edb163f0bfcfa9072d59bbc0bc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def immutable(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Immutable diagnostic path collision')
    else:
        path.write_bytes(data)


def main():
    if sha(ROOT / SNAPSHOT) != SNAPSHOT_SHA:
        raise ValueError('Frozen instruction source changed')
    seeds = {r['id']: r for r in json.loads((ROOT / SNAPSHOT).read_text())['plans']}
    metadata_path = ROOT / 'cache/custom_holder_supports/holder_supports.json'
    metadata = {r['index']: r for r in json.loads(metadata_path.read_text())['parts']}
    bindings = {SNAPSHOT: SNAPSHOT_SHA,
                str(metadata_path.relative_to(ROOT)): sha(metadata_path),
                str(Path(__file__).relative_to(ROOT)): sha(__file__),
                'source/plan_external_removal_v6.py': sha(P.__file__),
                'source/custom_holder_supports.py': sha(H.__file__)}
    records = []
    candidate = None
    for index in IDS:
        row = metadata[index]
        loaded = []
        for field in ['full_mesh_file', 'removable_mesh_file']:
            path = ROOT / 'cache/custom_holder_supports' / row[field]
            bindings[str(path.relative_to(ROOT))] = sha(path)
            loaded.append(H.load(path)[0])
        full, removable = loaded
        # Preserve the exact signed subtraction result used by the planner.
        # Decomposition below is diagnostic only; nothing is discarded.
        retained = full - removable
        axes = np.asarray(row['neck_axes'], float)
        seed = seeds[f'H{index:04d}']
        record = dict(id=seed['id'], kind=row['kind'], style=row['support_style'],
            neck_length_mm=row['neck_length_mm'], endpoint_normal_dot=float(axes[0] @ axes[1]),
            nominal_removable_retained_min_gap_mm=float(removable.min_gap(retained, .1)),
            measured_coincident_intersection_mm3=H.volume(removable ^ retained),
            nominal_removable_volume_mm3=H.volume(removable),
            retained_signed_component_count=len(retained.decompose()),
            old_candidate_mode=seed['mode'],
            first_departure_dot_endpoint_inward_normals=[
                (axes @ np.asarray(f['translation_legs'][0]['direction'])).tolist()
                for f in seed['fragments']],
            classification=('Parallel horizontal interfaces permit two same-normal finite bands '
                'inside actual removable stock.' if index == 257 else
                'Nonparallel pitched interfaces; two endpoint-normal bands do not fit the current '
                'single-common-normal body-cut grammar. No automatic end-wedge fragments proposed.'))
        if index == 226:
            record['coincident_intersection_caution'] = (
                'The measured R intersection (F minus R) is a numerical set-operation '
                'inconsistency on coincident serialized stock. It is not treated as a '
                'physical intrusion or a passed route; no cap geometry is repaired or removed.')
        if index == 257:
            bounds = removable.bounding_box()
            root = np.asarray(row['root_mm'], float)
            target = np.asarray(row['target_mm'], float)
            width = .05
            levels = [bounds[2] + width / 2, bounds[5] - width / 2]
            cuts = []
            for level in levels:
                center = root + (target - root) * ((level - root[2]) / (target[2] - root[2]))
                cuts.append(dict(plane_normal=[0., 0., 1.], plane_offset_mm=float(level),
                    centerline_reference_mm=center.tolist(), modeled_removed_kerf_width_mm=width))
            partition = P.body_partition(removable, cuts)
            if partition is None or len(partition[0]) != 1:
                raise ValueError('H0257 does not leave exactly one valid substantial fragment')
            fragments, kerf, checks = partition
            fragment = fragments[0]
            extra_checks = dict(actual_kerf_outside_removable_mm3=H.volume(kerf - removable),
                                actual_kerf_retained_intersection_mm3=H.volume(kerf ^ retained))
            if max(extra_checks.values()) > P.TOL:
                raise ValueError('H0257 kerf leaves removable stock or cuts retained material')
            gap = float(fragment.min_gap(retained, .1))
            if gap < .00999:
                raise ValueError('H0257 finite bands failed to clear the unchanged signed retained stock')
            legs = [dict(direction=list(l['direction']), distance_mm=float(l['distance_mm']))
                    for l in seed['fragments'][0]['translation_legs']]
            moved = fragment
            own_cap_overlaps = []
            for leg in legs:
                direction = np.asarray(leg['direction'], float)
                own_cap_overlaps.append(H.volume(H.sweep(moved, direction, leg['distance_mm']) ^ retained))
                moved = moved.translate(direction * leg['distance_mm'])
            if max(own_cap_overlaps) > P.TOL:
                raise ValueError('H0257 old directions fail even unchanged own retained stock')
            candidate = dict(id=seed['id'], mode='explicit_body_cuts', additional_body_cuts=cuts,
                fragments=[dict(body_section_index=0, translation_legs=legs)], fragment_extraction_order=[0])
            record['local_finite_band_diagnostic'] = dict(
                body_cut_partition_checks=checks, stock_checks=extra_checks,
                remaining_fragment_count=1, remaining_fragment_volume_mm3=H.volume(fragment),
                actual_removed_body_band_volume_mm3=H.volume(kerf),
                entire_unchanged_retained_min_gap_mm=gap,
                own_retained_conservative_hull_sweep_overlap_per_leg_mm3=own_cap_overlaps,
                bands_start_at_nominal_removable_interfaces=True,
                bands_extend_into_temporary_carrier_beyond_constant_waist_mm=.01,
                no_original_or_other_support_checks_performed=True)
        records.append(record)
    data = encoded(dict(plans=[candidate]))
    digest = hashlib.sha256(data).hexdigest()
    candidate_path = ROOT / ('cache/external_removal_v6/'
        f'holder0257_finite_band_candidates_{digest}.json')
    immutable(candidate_path, data)
    report = dict(scope='Small local stock/cap feasibility only, not a full removal certificate',
        source_bindings=bindings, printed_geometry_modified=False, whole_scene_loaded=False,
        signed_retained_components_discarded=False, volume_tolerance_mm3=P.TOL,
        records=records, candidate_path=str(candidate_path.relative_to(ROOT)),
        candidate_sha256=digest, candidate_status='Instructions only; all cuts and route legs require full-scene verification')
    report_data = encoded(report)
    report_digest = hashlib.sha256(report_data).hexdigest()
    report_path = ROOT / f'reports/holder_endpoint_clearance_diagnostic_{report_digest[:16]}.json'
    immutable(report_path, report_data)
    print(json.dumps(dict(candidate_path=str(candidate_path.relative_to(ROOT)), candidate_sha256=digest,
        report_path=str(report_path.relative_to(ROOT)), report_sha256=report_digest), indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Small-stock finer body-cut menu, with no scene or route acceptance claims.

Uses the existing P.body_partition grammar and .050 mm actual-stock kerfs.
The result is a separate menu, not a --candidates route seed file. It never
changes any printed input, active source, current cut definition or checkpoint.
"""
from pathlib import Path
import hashlib
import json
import math
import re

import numpy as np

import custom_holder_supports as H
import plan_external_removal_v6 as P
from build_holder import SPHERES, sphere_transform

ROOT = Path(__file__).resolve().parent.parent
PROOF = 'reports/external_removal_endpoint_partial_d2e85a5d25ea33e2.json'
PROOF_SHA = 'd2e85a5d25ea33e2419f7999e293efe8ddc0f2bfc8409f4a6e3f7efa37e19e76'
PITCHES = (.50, .35, .25)
WIDTH = .05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sha(ROOT / PROOF) != PROOF_SHA:
        raise ValueError('Immutable residual-source proof changed')
    proof = json.loads((ROOT / PROOF).read_text())
    ledger = {entry['path']: entry['sha256'] for entry in proof['input_files']}
    inputs = {PROOF: PROOF_SHA}

    def bind(path):
        digest = sha(ROOT / path)
        if path not in ledger or ledger[path] != digest:
            raise ValueError('Actual stock/source no longer matches the immutable proof: ' + path)
        inputs[path] = digest
        return ROOT / path

    for path in ['source/plan_external_removal_v6.py', 'source/custom_holder_supports.py',
                 'source/build_holder.py']:
        bind(path)
    hp = 'reports/holder_finite_endpoint_partition_v6.json'
    sp = 'reports/sphere_removal_partition_v6.json'
    holders = json.loads(bind(hp).read_text())
    spheres = json.loads(bind(sp).read_text())
    if not holders['passed'] or not spheres['passed']:
        raise ValueError('Actual finite endpoint partition is not validated')
    holder_records = {r['support_index']: r for r in holders['records']}
    sphere_records = {(r['layer_count'], r['support_index'], r['placement']): r
                      for r in spheres['records'] if r['placement'] is not None}
    holder_metadata = {r['index']: r for r in json.loads(bind(
        'cache/custom_holder_supports/holder_supports.json').read_text())['parts']}
    sphere_metadata = {n: {r['support_index']: r for r in json.loads(bind(
        f'cache/custom_sphere_supports/sphere_{n}layers_supports.json').read_text())['supports']}
                       for n in (3, 4, 5)}
    records = []
    for failure in proof['remaining_failures']:
        identity = failure['id']
        if identity.startswith('H'):
            index = int(identity[1:])
            endpoint = holder_records[index]
            row = holder_metadata[index]
            root, target = np.asarray(row['root_mm']), np.asarray(row['target_mm'])
            endpoint_manifest = hp
        else:
            layers, angle, index = map(int, re.fullmatch(r'S(\d)_A(\d+)_(\d+)', identity).groups())
            placement, sphere = next((i, s) for i, s in enumerate(SPHERES)
                                    if s['layers'] == layers and s['theta_deg'] == angle)
            endpoint = sphere_records[layers, index, placement]
            row = sphere_metadata[layers][index]
            transform = sphere_transform(sphere)
            root = transform[:, :3] @ np.asarray(row['anchor_point']) + transform[:, 3]
            target = transform[:, :3] @ np.asarray(row['contact_point']) + transform[:, 3]
            endpoint_manifest = sp
        stock_entry = endpoint['paths']['removable']
        stock_path = bind(stock_entry['path'])
        if sha(stock_path) != stock_entry['sha256']:
            raise ValueError('Finite stock record hash mismatch')
        stock = H.load(stock_path)[0]
        points = np.asarray(stock.to_mesh64().vert_properties[:, :3], float)
        chord = target - root
        chord_length = float(np.linalg.norm(chord))
        if chord_length <= 0.:
            raise ValueError('No usable support chord')
        axis = chord / chord_length
        projection = points @ axis
        low, high = float(projection.min()), float(projection.max())
        span = high - low
        requests = {}
        for pitch in PITCHES:
            count = max(2, math.ceil(span / pitch))
            requests.setdefault(count, []).append(dict(target_pitch_mm=pitch))
        for count in (6, 8):
            requests.setdefault(count, []).append(dict(requested_piece_count=count))
        options, rejected = [], []
        for count, requests_here in sorted(requests.items()):
            pitch = span / count
            if pitch <= WIDTH:
                rejected.append(dict(requested_piece_count=count, reason='Projection interval does not exceed the .050 mm kerf'))
                continue
            levels = np.linspace(low, high, count + 1)[1:-1]
            cuts = [dict(plane_normal=axis.tolist(), plane_offset_mm=float(level),
                centerline_reference_mm=(root + axis * (level - float(root @ axis))).tolist(),
                modeled_removed_kerf_width_mm=WIDTH) for level in levels]
            result = P.body_partition(stock, cuts)
            if result is None:
                rejected.append(dict(requested_piece_count=count, reason='Existing P.body_partition coverage gate failed'))
                continue
            fragments, kerf, base_checks = result
            # One result for EVERY requested nonempty interval ensures that
            # P's small-volume interval filter did not omit a body fragment.
            # Every returned signed component stays in its complete fragment.
            if len(fragments) != count:
                rejected.append(dict(requested_piece_count=count, actual_fragment_count=len(fragments),
                    reason='An interval was empty or below P cutoff; no fragment-discarding menu option retained'))
                continue
            checks = dict(base_checks)
            pairwise = 0.
            for i, a in enumerate(fragments):
                for b in fragments[i + 1:]:
                    if H.bbox_intersects(a, b):
                        pairwise += H.volume(a ^ b)
            checks['sum_pairwise_fragment_interior_overlap_mm3'] = pairwise
            checks['sum_fragment_kerf_interior_overlap_mm3'] = sum(H.volume(part ^ kerf) for part in fragments)
            checks['kerf_outside_actual_removable_mm3'] = H.volume(kerf - stock)
            checks['sum_fragment_outside_actual_removable_mm3'] = sum(H.volume(part - stock) for part in fragments)
            if max(checks.values()) > P.TOL:
                rejected.append(dict(requested_piece_count=count,
                    reason='Additional complete-stock/disjointness gate failed', checks=checks))
                continue
            components = [part.decompose() for part in fragments]
            options.append(dict(option_id=f'{identity}_chord_{count:02d}_pieces', requests=requests_here,
                actual_projection_pitch_mm=pitch, requested_piece_count=count,
                resulting_fragment_count=len(fragments), additional_body_cuts=cuts,
                fragment_body_section_indices=list(range(len(fragments))),
                fragment_volumes_mm3=[float(part.volume()) for part in fragments],
                fragment_component_counts=[len(parts) for parts in components],
                fragment_signed_component_volumes_mm3=[[float(part.volume()) for part in parts] for parts in components],
                kerf_volume_mm3=float(kerf.volume()), local_coverage_disjointness_checks=checks,
                every_requested_interval_preserved=True, all_signed_components_preserved=True,
                full_scene_cut_and_route_verification_required=True))
        records.append(dict(id=identity, actual_stock=stock_entry,
            adopted_endpoint_partition_manifest=endpoint_manifest,
            root_mm=root.tolist(), target_mm=target.tolist(), chord_axis=axis.tolist(),
            projection_interval_mm=[low, high], projection_span_mm=span,
            original_actual_stock_volume_mm3=float(stock.volume()), options=options,
            rejected_options=rejected))
        print(json.dumps(dict(id=identity, kept_counts=[o['resulting_fragment_count'] for o in options],
            rejected_counts=[o['requested_piece_count'] for o in rejected])), flush=True)
    for path, digest in inputs.items():
        if sha(ROOT / path) != digest:
            raise ValueError('Bound input changed during menu preparation')
    menu = dict(schema='finer_body_partition_menu_v1',
        scope='Local actual-stock partition MENU ONLY. No route, cutter access, strength or complete-scene acceptance claim.',
        source=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__)),
        source_bindings=inputs, residual_source_proof=PROOF,
        input_residual_count=len(records), intended_projection_pitches_mm=list(PITCHES),
        extra_piece_counts=[6, 8], actual_removed_kerf_width_mm=WIDTH,
        unchanged_volume_tolerance_mm3=P.TOL, records=records,
        not_a_route_candidate_seed_file=True, no_fragment_directions_or_extraction_order_prescribed=True,
        only_apply_to_still_remaining_ids=True, printed_geometry_modified=False,
        active_inputs_modified=False, original_or_whole_scene_loaded=False)
    data = (json.dumps(menu, indent=2, allow_nan=False) + '\n').encode()
    digest = hashlib.sha256(data).hexdigest()
    output = ROOT / f'cache/external_removal_v6/finer_body_partition_menu_{digest}.json'
    if output.exists() and output.read_bytes() != data:
        raise ValueError('Content-addressed menu collision')
    if not output.exists():
        output.write_bytes(data)
    print(json.dumps(dict(menu=str(output.relative_to(ROOT)), sha256=digest,
        operations=len(records), options=sum(len(row['options']) for row in records),
        rejected_options=sum(len(row['rejected_options']) for row in records))), flush=True)


if __name__ == '__main__':
    main()

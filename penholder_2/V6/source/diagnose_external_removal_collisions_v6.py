#!/usr/bin/env python3
"""Attribute failed residual instructions without accepting routes or changing stock.

The complete immutable prior proof is checked with the existing continuation
validator. Its exact remaining obstacle field is recreated once. Diagnostic
queries preserve every permanent solid, every active support and every unexited
sibling. They identify the first failing component of the unchanged boundary
sweep construction and report individual positive intersections. No accepted
operation, input geometry, existing source, or authoritative proof is written.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

import continue_external_removal_cached_bounds_v6 as D
import external_removal_plate_stock_v6 as B
import plan_external_removal_v6 as P


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rel(path):
    return str(Path(path).resolve().relative_to(P.ROOT))


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.part.' + str(os.getpid()))
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def intersection(query, obstacle):
    result = query ^ obstacle
    raw = float(result.volume()) if not result.is_empty() else 0.
    require(math.isfinite(raw), 'Nonfinite diagnostic Boolean volume')
    return dict(signed_volume_mm3=raw, positive_volume_mm3=max(0., raw),
                status=str(result.status()), triangle_count=int(result.num_tri()),
                bounds_mm=(list(result.bounding_box()) if not result.is_empty() else None))


def permanent_labels():
    labels = []
    holder = P.metadata(P.ROOT / 'cache/custom_holder_supports/holder_supports.json')
    for row in holder['parts']:
        labels.append(dict(category='permanent', support_id=f'H{row["index"]:04d}',
            permanent_kind=('fixed_addition' if row['kind'] == 'fixed_micro_reinforcement'
                            else 'computed_full_minus_saved_removable'),
            source_kind=row['kind']))
    for sphere in P.SPHERES:
        n = sphere['layers']
        data = P.metadata(P.ROOT / f'cache/custom_sphere_supports/sphere_{n}layers_supports.json')
        for row in data['supports']:
            labels.append(dict(category='permanent',
                support_id=f'S{n}_A{int(sphere["theta_deg"]):03d}_{row["support_index"]:03d}',
                permanent_kind=('fixed_addition' if row['kind'] == 'fixed_micro_reinforcement'
                                else 'finite_endpoint_partition_retained'),
                source_kind=row['kind']))
    return labels


def field_labels(field, permanent, items):
    descriptions = permanent_labels()
    require(len(descriptions) == len(permanent), 'Permanent source order differs from load_parts')
    labels = []
    for solid, description in zip(permanent, descriptions):
        if not solid.is_empty():
            labels.append(description)
    for item in items:
        if not item['full'].is_empty():
            labels.append(dict(category='active_support', support_id=item['id'],
                               source_kind=item['kind']))
    require(len(labels) == len(field.solids), 'Obstacle labels differ from field construction')
    for index, label in enumerate(labels):
        owner = field.owner[index]
        require((owner is None) == (label['category'] == 'permanent'), 'Obstacle owner mismatch')
        if owner is not None:
            require(owner == label['support_id'], 'Active source order mismatch')
    return labels


def attribution(query, original, context, labels, sibling_indices=()):
    """Complete per-owner Booleans, with no early return or boundary prefilter."""
    rows = []
    checked = 0
    negative = []

    def add(solid, label):
        nonlocal checked
        detail = intersection(query, solid)
        checked += 1
        if detail['signed_volume_mm3'] < 0.:
            negative.append(dict(**label, **detail))
        if detail['positive_volume_mm3'] > 0.:
            rows.append(dict(**label, **detail))

    for index in context.field.tree.intersection(tuple(query.bounding_box())):
        owner = context.field.owner[index]
        if owner is not None and (owner == context.excluded or owner not in context.field.active):
            continue
        add(context.field.solids[index], dict(field_index=index, **labels[index]))
    require(not sibling_indices or len(sibling_indices) == len(context.extra), 'Sibling labels differ')
    for index, sibling in enumerate(context.extra):
        if P.H.bbox_intersects(query, sibling):
            add(sibling, dict(category='unexited_sibling', support_id=context.excluded,
                body_section_index=(sibling_indices[index] if sibling_indices else index)))
    add(original, dict(category='original_v5', source='cache/original_v5.npz'))
    rows.sort(key=lambda row: row['positive_volume_mm3'], reverse=True)
    return dict(complete_positive_overlap_sum_mm3=sum(row['positive_volume_mm3'] for row in rows),
                positive_owner_intersections=rows, checked_obstacle_count=checked,
                negative_signed_intersections=negative,
                duplicate_scene_coverage_counted_separately=True,
                full_booleans_without_convex_prefilter=True)


def construct_components(fragment, direction, distance):
    """Capture provenance from the unchanged constructor, preserving its objects."""
    original_checked = B._checked_prism
    provenance = []

    def capture(points, ideal, delta, face_index):
        component, proof = original_checked(points, ideal, delta, face_index)
        provenance.append(dict(component_index=len(provenance)+1, face_index=int(face_index),
            construction=proof['construction'], padding_mm=proof['padding_mm'],
            padding_basis=proof['padding_basis'], conservative_box=proof['conservative_box'],
            unpadded_failures=proof['unpadded_failures'],
            raw_target_containment=proof['raw_target_containment'],
            serialized_target_containment=proof['serialized_target_containment'],
            query_is_original_convex_constructor_object=proof['query_is_original_convex_constructor_object'],
            six_rounded_target_points_mm=np.asarray(points, float).tolist()))
        return component, proof

    B._checked_prism = capture
    try:
        components, proof = B.translation_sweep_components(fragment, direction, distance)
    finally:
        B._checked_prism = original_checked
    require(len(provenance)+1 == len(components), 'Boundary component provenance mismatch')
    return components, proof, provenance


def diagnose_leg(fragment, direction, distance, original, context, labels, sibling_indices):
    initial_gate = float(context.obstruction(fragment, original, convex=False))
    initial = attribution(fragment, original, context, labels, sibling_indices)
    base = dict(direction=np.asarray(direction, float).tolist(), distance_mm=float(distance),
                fragment_bounds_mm=list(fragment.bounding_box()),
                actual_initial_obstruction_return_mm3=initial_gate,
                initial_actual_stock_attribution=initial)
    if initial_gate > P.TOL:
        return dict(**base, failed_component_index=0,
            failure_stage='actual_initial_signed_stock', exact_sweep_sum_through_failure_mm3=initial_gate,
            failed_component_provenance=dict(construction='literal_actual_starting_signed_solid'),
            failed_component_attribution=initial, continuous_component_gate_passed=False)
    try:
        components, proof, provenance = construct_components(fragment, direction, distance)
    except B.CertificateError as error:
        return dict(**base, failure_stage='component_derivation', reason=str(error),
                    derivation_checks=error.checks, continuous_component_gate_passed=False)
    total = float(proof.get('serialization_uncovered_bound_mm3', 0.))
    values = []
    actual_initial = None
    for index, component in enumerate(components):
        convex = index > 0 and actual_initial == 0.
        overlap = float(context.obstruction(component, original, convex=convex))
        require(math.isfinite(overlap) and overlap >= 0., 'Invalid component gate result')
        if index == 0:
            actual_initial = overlap
        total += overlap
        if overlap > 0.:
            values.append(dict(component_index=index, overlap_return_mm3=overlap,
                               convex_prefilter_allowed=convex))
        if total > P.TOL:
            detail = initial if index == 0 else attribution(component, original, context, labels, sibling_indices)
            return dict(**base, failure_stage='summed_boundary_component_collision',
                failed_component_index=index, exact_sweep_sum_through_failure_mm3=total,
                first_failed_component_overlap_return_mm3=overlap,
                convex_prefilter_allowed_for_failed_component=convex,
                failed_component_bounds_mm=list(component.bounding_box()),
                failed_component_provenance=(dict(construction='literal_actual_starting_signed_solid')
                    if index == 0 else provenance[index-1]),
                failed_component_attribution=detail,
                positive_component_returns_through_failure=values,
                component_count=len(components), sweep_construction=proof,
                continuous_component_gate_passed=False)
    return dict(**base, continuous_component_gate_passed=True,
                exact_sweep_overlap_sum_mm3=total, component_count=len(components),
                positive_component_returns=values, sweep_construction=proof)


def diagnose_item(item, candidate, original, field, labels):
    context = field.context(item['id'])
    rem = item['removable']
    row = dict(id=item['id'], mode=candidate['mode'], kind=item['kind'],
               actual_whole_removable_attribution=attribution(rem, original, context, labels), legs=[])
    if candidate['mode'] == 'whole_piece':
        require(len(candidate['fragments']) == 1, 'Invalid whole-piece instruction')
        fragments = [rem]
        order = [0]
    else:
        require(candidate['mode'] == 'explicit_body_cuts', 'Unsupported diagnostic candidate mode')
        partition = P.body_partition(rem, candidate['additional_body_cuts'])
        if partition is None:
            row['stopped_at'] = 'body_partition_failed'
            return row
        fragments, kerf, checks = partition
        row['body_partition_checks'] = checks
        row['body_kerf_attribution'] = attribution(kerf, original, context, labels)
        order = [record.get('body_section_index', index)
                 for index, record in enumerate(candidate['fragments'])]
        require(sorted(order) == list(range(len(fragments))), 'Invalid body section correspondence')
    remaining = list(range(len(fragments)))
    for fragment_index, record in zip(order, candidate['fragments']):
        others = [index for index in remaining if index != fragment_index]
        state = context.with_extra([fragments[index] for index in others])
        fragment = fragments[fragment_index]
        for leg_index, leg in enumerate(record['translation_legs']):
            detail = diagnose_leg(fragment, leg['direction'], leg['distance_mm'], original, state, labels, others)
            row['legs'].append(dict(body_section_index=fragment_index, leg_index=leg_index, **detail))
            if not detail['continuous_component_gate_passed']:
                row['stopped_at'] = 'first_failed_component_leg'
                return row
            fragment = fragment.translate(np.asarray(leg['direction'], float) * float(leg['distance_mm']))
        bb = fragment.bounding_box()
        escaped = bb[5] < P.SCENE_MIN_Z if item['kind'] == 'plate_perimeter_tack' else bb[2] > P.SCENE_MAX_Z
        if not escaped:
            row['stopped_at'] = 'fragment_did_not_exit_initial_scene'
            return row
        # Local sibling bookkeeping only. The global active field is never changed.
        remaining.remove(fragment_index)
    row['stopped_at'] = 'all_diagnostic_component_queries_clear_not_a_route_certificate'
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--prior-sha256', required=True)
    ap.add_argument('--candidates', type=Path, required=True)
    ap.add_argument('--candidates-sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ids', nargs='*')
    args = ap.parse_args()
    started = time.monotonic()
    output = args.output.resolve()
    require(output.is_relative_to(P.ROOT / 'reports') and not output.exists(),
            'Diagnostic output must be a new project report')
    P.frozen_bytes(__file__)
    prior = P.metadata(args.prior.resolve())
    require(P.INPUT_HASHES[rel(args.prior)] == args.prior_sha256, 'Wrong prior report bytes')
    plans = D.verify_prior(prior)
    candidates = P.metadata(args.candidates.resolve())
    require(P.INPUT_HASHES[rel(args.candidates)] == args.candidates_sha256, 'Wrong candidate bytes')
    seeds = {row['id']: row for row in candidates['plans']}
    require(len(seeds) == len(candidates['plans']), 'Duplicate candidate ID')
    items, permanent, _, all_full = P.load_parts()
    del all_full
    by_id = {item['id']: item for item in items}
    require(len(by_id) == len(items) == prior['initial_support_count'], 'Source operation set changed')
    removed = {row['id'] for row in plans}
    for row in plans:
        item = by_id[row['id']]
        require(all(row[k] == item[k] for k in ('group', 'source_index', 'kind')), 'Prior source identity changed')
    residual = {row['id'] for row in prior['remaining_failures']}
    require(residual == set(by_id)-removed, 'Prior residual set is incomplete')
    requested = args.ids or ([row['id'] for row in prior['remaining_failures']
                             if row['id'].startswith('S')] + (['H0226'] if 'H0226' in residual else []))
    require(len(set(requested)) == len(requested) and set(requested) <= residual & set(seeds),
            'Requested diagnostics are not unique residual candidates')
    original_path = P.ROOT / 'cache/original_v5.npz'
    digest = P.sha(original_path)
    require(digest == prior['original_v5_sha256'], 'Original changed')
    surface = P.H.Surface(original_path)
    P.H.ACTIVE_SURFACE = surface
    original = surface.solid
    require(P.sha(original_path) == digest, 'Original changed during load')
    P.SCENE_MIN_Z, P.SCENE_MAX_Z = prior['initial_scene_z_bounds_mm']
    field = P.ObstacleField(items, permanent)
    field.active.difference_update(removed)
    require(field.active == residual, 'Incorrect reconstructed obstacle field')
    labels = field_labels(field, permanent, items)
    report = dict(schema='v6_residual_collision_attribution_v1', diagnostic_only=True,
        route_acceptance_or_geometry_mutation=False, numerical_tolerance_mm3=P.TOL,
        immutable_prior_report=rel(args.prior), prior_sha256=args.prior_sha256,
        candidate_instructions=rel(args.candidates), candidate_sha256=args.candidates_sha256,
        source=rel(__file__), source_sha256=P.INPUT_HASHES[rel(__file__)],
        recreated_active_ids=sorted(residual), initial_support_count=len(items),
        prior_accepted_count=len(plans), requested_ids=requested, results=[])
    partial = output.with_suffix('.partial.json')
    require(not partial.exists(), 'Refusing to overwrite an earlier diagnostic checkpoint')
    for ident in requested:
        before = time.monotonic()
        result = diagnose_item(by_id[ident], seeds[ident], original, field, labels)
        result['elapsed_seconds'] = time.monotonic()-before
        report['results'].append(result)
        atomic_json(partial, report)
        last = result['legs'][-1] if result['legs'] else {}
        print(json.dumps(dict(id=ident, stop=result['stopped_at'],
            component=last.get('failed_component_index'),
            volume=last.get('exact_sweep_sum_through_failure_mm3'),
            owners=[{k:v for k,v in owner.items() if k in ('category', 'support_id', 'positive_volume_mm3', 'body_section_index')}
                    for owner in last.get('failed_component_attribution', {}).get('positive_owner_intersections', [])[:4]],
            elapsed_seconds=result['elapsed_seconds'])), flush=True)
        require(field.active == residual, 'Diagnostic changed global active state')
    changed = [path for path, digest in P.INPUT_HASHES.items() if P.sha(P.ROOT/path) != digest]
    require(not changed, 'Frozen input changed during diagnosis: ' + str(changed))
    report['input_files'] = [dict(path=path, sha256=digest) for path,digest in sorted(P.INPUT_HASHES.items())]
    report['all_input_bytes_unchanged'] = True
    report['elapsed_seconds'] = time.monotonic()-started
    report['complete'] = True
    atomic_json(output, report)
    partial.unlink()
    print('Diagnostic complete', rel(output), hashlib.sha256(output.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()

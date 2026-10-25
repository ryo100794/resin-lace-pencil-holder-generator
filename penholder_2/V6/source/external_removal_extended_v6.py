#!/usr/bin/env python3
"""Revalidate actual removal with cut-aware departures and exact small sweeps.

The printed model, the original planner, and the sequential sibling verifier
are immutable. This overlay adds candidate paths, not acceptance exceptions.
All accepted moves still use the complete current obstacle state and a
continuous swept-solid test, followed by an exit from the entire scene.

The forty unresolved underside finishing pads use their actual added stock,
R minus the unchanged original. Their independently recorded set identities
justify retaining the old full-part obstacles and old permanent caps.
"""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np

import external_removal_sequential_v6 as S
import plan_external_removal_v6 as P
import external_removal_plate_stock_v6 as B

_BASE_CANDIDATE = P.candidate_directions
_BASE_MAKE_PLAN = P.make_plan
_BASE_VERIFY_SAVED_LEGS = P.verify_saved_legs
_INSTALLED = False
_SWEEP_DERIVATION_FAILURE_COUNT = 0
_SWEEP_DERIVATION_FAILURE_EXAMPLES = []


def _unit(vector):
    value = np.asarray(vector, float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    length = float(np.linalg.norm(value))
    return value / length if length > 1e-10 else None


def _unique(vectors):
    result = []
    for vector in vectors:
        value = _unit(vector)
        if value is not None and not any(np.linalg.norm(value - old) < 1e-5 for old in result):
            result.append(value)
    return result


def _endpoint_normals(item):
    endpoint = item.get('endpoint_partition')
    if endpoint is not None:
        return _unique([cut['axis_toward_free_body'] for cut in endpoint['cuts']])
    row = item['record']
    normals = row.get('cut_plane_normals', row.get('neck_axes', []))
    if item['group'] == 'sphere':
        normals = [item['transform'][:, :3] @ np.asarray(n, float) for n in normals]
    return _unique(normals)


def departure_directions(item, center, preferred=()):
    """Geometric candidates only; no normal test replaces a collision test."""
    center = np.asarray(center, float)
    radial = _unit([center[0], center[1], 0.])
    if radial is None:
        radial = np.array([1., 0., 0.])
    tangent = np.array([-radial[1], radial[0], 0.])
    normals = _endpoint_normals(item)
    body_normals = []
    nearby_cuts = sorted(item.get('_active_body_cuts', []), key=lambda cut:
        abs(float(np.dot(center, np.asarray(cut['plane_normal'], float))) - float(cut['plane_offset_mm'])))[:2]
    for cut in nearby_cuts:
        axis = np.asarray(cut['plane_normal'], float)
        offset = float(cut['plane_offset_mm'])
        body_normals.append(axis if float(np.dot(center, axis)) >= offset else -axis)
    candidates = list(preferred)
    if len(normals) >= 2:
        candidates += [normals[0] + normals[1],
                       np.cross(normals[0], normals[1]),
                       -np.cross(normals[0], normals[1])]
    candidates += body_normals
    candidates += [body_normal + endpoint_normal
                   for body_normal in body_normals for endpoint_normal in normals]
    candidates += normals + [tangent, -tangent]
    for normal in normals + body_normals:
        for base in [radial, -radial, tangent, -tangent]:
            projection = base - float(np.dot(base, normal)) * normal
            candidates += [projection, projection + .15 * normal]
    candidates += [radial, -radial]
    candidates += _BASE_CANDIDATE(item, center, preferred)
    return _unique(candidates)


def candidate_directions(item, center, preferred=()):
    if item['kind'] == 'plate_perimeter_tack':
        return [np.array([0., 0., -1.])]
    return departure_directions(item, center, preferred)


def _fragment_bounds(item, fragment, supplied):
    """Clip the existing covering proxies by the actual body-cut interval.

    Plane clipping preserves convexity of each generated proxy. Coverage of
    the actual serialized fragment is tested before these tighter bounds can
    replace its single global convex hull. Failure returns the old bound.
    """
    cuts = item.get('_active_body_cuts')
    if not cuts or len(item.get('pieces', [])) <= 1:
        return supplied, None
    axis = np.asarray(cuts[0]['plane_normal'], float)
    mesh = fragment.to_mesh64()
    vertices = np.asarray(mesh.vert_properties[:, :3], float)
    values = vertices @ axis
    low, high = float(values.min()), float(values.max())
    padding = 1e-9
    clipped = [piece.trim_by_plane(axis, low - padding)
                    .trim_by_plane(-axis, -(high + padding))
               for piece in item['pieces']]
    clipped = [piece for piece in clipped if not piece.is_empty()]
    if not clipped:
        return supplied, None
    missing = P.H.volume(fragment - P.H.union(clipped))
    if missing > P.TOL:
        return supplied, None
    proof = dict(method='generated_convex_proxies_clipped_to_actual_body_interval',
                 axis=axis.tolist(), projection_interval_mm=[low, high],
                 outward_plane_padding_mm=padding,
                 uncovered_actual_fragment_mm3=missing,
                 proxy_count=len(clipped))
    return clipped, proof


def _sweep(fragment, pieces, direction, distance, original, obstacles,
           allow_exact=False, exact_only=False, proxy_proof=None):
    """Sum component intersections to bound the entire continuous union.

    The initial actual solid and every boundary-triangle prism cover the
    continuous translation. Summing all their complete-scene overlap volumes
    is an upper bound for the overlap of their union, including duplicates.
    No fragile Boolean union and no maximum-per-component shortcut is used.
    """
    if not exact_only:
        passed, collision = P.safe_sweep(pieces, direction, distance, original, obstacles)
        if passed:
            proof = dict(method='continuous_conservative_convex_piece_bounds')
            if proxy_proof is not None:
                proof['fragment_proxy_coverage'] = proxy_proof
            return True, collision, proof
        if not allow_exact:
            return False, collision, None
    global _SWEEP_DERIVATION_FAILURE_COUNT
    try:
        components, proof = B.translation_sweep_components(fragment, direction, distance)
    except B.CertificateError as error:
        _SWEEP_DERIVATION_FAILURE_COUNT += 1
        if len(_SWEEP_DERIVATION_FAILURE_EXAMPLES) < 32:
            _SWEEP_DERIVATION_FAILURE_EXAMPLES.append(dict(
                reason=str(error), checks=error.checks,
                fragment_bounds_mm=list(fragment.bounding_box()),
                direction=np.asarray(direction, float).tolist(), distance_mm=float(distance)))
        return False, math.inf, None
    collision = float(proof.get('serialization_uncovered_bound_mm3', 0.))
    if not math.isfinite(collision) or collision < 0. or collision > P.TOL:
        return False, math.inf, None
    initial_overlap = None
    for index, component in enumerate(components):
        # The boundary-only original-model broadphase additionally needs an
        # outside-original starting witness. A tolerated nonzero initial
        # intersection does not supply one, so use the full Boolean then.
        convex_prefilter = index > 0 and initial_overlap == 0.
        component_overlap = float(obstacles.obstruction(component, original, convex=convex_prefilter))
        if not math.isfinite(component_overlap) or component_overlap < 0.:
            return False, math.inf, None
        if index == 0:
            initial_overlap = component_overlap
        collision += component_overlap
        if collision > P.TOL:
            return False, collision, None
    return True, collision, dict(
        method='continuous_union_of_initial_solid_and_all_boundary_triangle_prisms_with_summed_overlap_bound',
        component_count=len(components), summed_complete_scene_overlap_mm3=collision,
        initial_actual_component_overlap_mm3=initial_overlap,
        boundary_only_convex_prefilter_requires_exact_zero_initial_overlap=True,
        construction=proof)


def _leg(direction, distance, overlap, proof):
    return dict(direction=np.asarray(direction, float).tolist(), distance_mm=float(distance),
                maximum_swept_intersection_mm3=float(overlap), continuous_sweep=proof)


def _finish_exit(item, fragment, pieces, original, obstacles, prefix,
                 allow_exact, proxy_proof=None):
    if item['kind'] == 'plate_perimeter_tack':
        return prefix if fragment.bounding_box()[5] < P.SCENE_MIN_Z else None
    distance = max(0., P.SCENE_MAX_Z + 1. - float(fragment.bounding_box()[2]))
    legs = list(prefix)
    if distance:
        direction = np.array([0., 0., 1.])
        passed, overlap, proof = _sweep(fragment, pieces, direction, distance, original,
                                      obstacles, allow_exact=allow_exact, proxy_proof=proxy_proof)
        if not passed:
            return None
        legs.append(_leg(direction, distance, overlap, proof))
        fragment = fragment.translate(direction * distance)
    return legs if fragment.bounding_box()[2] > P.SCENE_MAX_Z else None


def _direct_routes(item, fragment, pieces, directions, original, obstacles,
                   allow_exact=False, proxy_proof=None):
    for direction in directions:
        for distance in [20., 35., 55., 90.]:
            passed, overlap, proof = _sweep(fragment, pieces, direction, distance,
                                          original, obstacles, allow_exact=allow_exact,
                                          proxy_proof=proxy_proof)
            if not passed:
                break  # Every longer collinear sweep contains this one.
            moved = fragment.translate(direction * distance)
            moved_pieces = [piece.translate(direction * distance) for piece in pieces]
            legs = _finish_exit(item, moved, moved_pieces, original, obstacles,
                                [_leg(direction, distance, overlap, proof)], allow_exact,
                                proxy_proof)
            if legs is not None:
                return legs
    return None


def _escape_directions(item, center):
    radial = _unit([center[0], center[1], 0.])
    if radial is None:
        radial = np.array([1., 0., 0.])
    tangent = np.array([-radial[1], radial[0], 0.])
    # Both the inner pen cavity and the outside are candidates. An upward
    # exit is always checked after whichever second leg is selected.
    candidates = [radial, -radial, tangent, -tangent]
    candidates += _BASE_CANDIDATE(item, center)
    return _unique(candidates)


def _short_departures(item, fragment, pieces, directions, original, obstacles,
                       allow_exact=False, proxy_proof=None):
    # A small initial release can clear a cut cap before a radial departure.
    # Every segment includes its starting point and all intermediate solids.
    first_distances = [.05, .15, .4, 1., 2.5]
    second_distances = [2., 6., 12., 20., 35., 55., 90.]
    for direction in directions:
        for distance in first_distances:
            passed, overlap, proof = _sweep(fragment, pieces, direction, distance,
                                          original, obstacles, allow_exact=allow_exact,
                                          proxy_proof=proxy_proof)
            if not passed:
                break
            moved = fragment.translate(direction * distance)
            moved_pieces = [piece.translate(direction * distance) for piece in pieces]
            prefix = [_leg(direction, distance, overlap, proof)]
            legs = _finish_exit(item, moved, moved_pieces, original, obstacles, prefix,
                                allow_exact, proxy_proof)
            if legs is not None:
                return legs
            bb = np.asarray(moved.bounding_box(), float)
            center = (bb[:3] + bb[3:]) / 2
            for second_direction in _escape_directions(item, center)[:12]:
                if np.linalg.norm(second_direction - direction) < 1e-5:
                    continue
                for second_distance in second_distances:
                    passed2, overlap2, proof2 = _sweep(moved, moved_pieces, second_direction,
                        second_distance, original, obstacles, allow_exact=allow_exact,
                        proxy_proof=proxy_proof)
                    if not passed2:
                        break
                    offset = second_direction * second_distance
                    end = moved.translate(offset)
                    end_pieces = [piece.translate(offset) for piece in moved_pieces]
                    legs = _finish_exit(item, end, end_pieces, original, obstacles,
                        prefix + [_leg(second_direction, second_distance, overlap2, proof2)],
                        allow_exact, proxy_proof)
                    if legs is not None:
                        return legs
    return None


def _plate_routes(item, fragment, pieces, original, obstacles, broad_search):
    downward = np.array([0., 0., -1.])

    def exit_below(start, start_pieces, prefix):
        distance = max(.5, float(start.bounding_box()[5] - P.SCENE_MIN_Z + .05))
        passed, overlap, proof = _sweep(start, start_pieces, downward, distance, original,
            obstacles, allow_exact=True, exact_only=True)
        if not passed:
            return None
        end = start.translate(downward * distance)
        if end.bounding_box()[5] >= P.SCENE_MIN_Z:
            return None
        return prefix + [_leg(downward, distance, overlap, proof)]

    direct = exit_below(fragment, pieces, [])
    if direct is not None or not broad_search:
        return direct
    bb = np.asarray(fragment.bounding_box(), float)
    center = (bb[:3] + bb[3:]) / 2
    radial = _unit([center[0], center[1], 0.])
    if radial is None:
        radial = np.array([1., 0., 0.])
    tangent = np.array([-radial[1], radial[0], 0.])
    directions = _unique([radial, radial + .5 * downward, radial + downward,
                          radial + 2. * downward, tangent, -tangent,
                          tangent + .5 * downward, -tangent + .5 * downward,
                          -radial + downward])
    for direction in directions:
        for distance in [.05, .15, .4, 1.]:
            passed, overlap, proof = _sweep(fragment, pieces, direction, distance,
                original, obstacles, allow_exact=True, exact_only=True)
            if not passed:
                break
            offset = direction * distance
            result = exit_below(fragment.translate(offset),
                [piece.translate(offset) for piece in pieces],
                [_leg(direction, distance, overlap, proof)])
            if result is not None:
                return result
    return None


def route_fragment(item, fragment, pieces, original, obstacles, preferred=(), broad_search=True):
    pieces, proxy_proof = _fragment_bounds(item, fragment, pieces)
    bb = np.asarray(fragment.bounding_box(), float)
    if item['kind'] == 'plate_perimeter_tack':
        return _plate_routes(item, fragment, pieces, original, obstacles, broad_search)
    center = (bb[:3] + bb[3:]) / 2
    directions = candidate_directions(item, center, preferred)
    if not broad_search:
        directions = directions[:4]
    result = _direct_routes(item, fragment, pieces, directions, original, obstacles,
                            proxy_proof=proxy_proof)
    if result is not None or not broad_search:
        return result
    result = _short_departures(item, fragment, pieces, directions[:16], original, obstacles,
                               proxy_proof=proxy_proof)
    if result is not None:
        return result
    # Exact sweeps are more expensive and are reserved for the remaining
    # conservative-bound failures. The obstacle and volume gates are identical.
    result = _direct_routes(item, fragment, pieces, directions[:16], original, obstacles,
                            allow_exact=True, proxy_proof=proxy_proof)
    if result is not None:
        return result
    return _short_departures(item, fragment, pieces, directions[:12], original, obstacles,
                              allow_exact=True, proxy_proof=proxy_proof)


def verify_saved_legs(item, fragment, pieces, legs, original, obstacles):
    # Previously successful instructions usually pass their original bound
    # immediately in the identical recreated scene. Recheck that full bound
    # before doing any new proxy or prism construction; this is validation,
    # never reuse of an old collision result.
    original_bound_result = _BASE_VERIFY_SAVED_LEGS(
        item, fragment, pieces, legs, original, obstacles)
    if original_bound_result is not None:
        for leg in original_bound_result:
            leg['continuous_sweep'] = dict(
                method='original_continuous_conservative_bounds_revalidated_in_current_complete_scene')
        return original_bound_result
    pieces, proxy_proof = _fragment_bounds(item, fragment, pieces)
    moved = pieces
    end = fragment
    verified = []
    for leg in legs:
        direction = np.asarray(leg['direction'], float)
        distance = float(leg['distance_mm'])
        if direction.shape != (3,) or not np.all(np.isfinite(direction)) or not math.isfinite(distance):
            return None
        if distance < 0 or abs(float(np.linalg.norm(direction)) - 1.) > 1e-8:
            return None
        passed, overlap, proof = _sweep(end, moved, direction, distance, original, obstacles,
            allow_exact=True, exact_only=item['kind'] == 'plate_perimeter_tack',
            proxy_proof=proxy_proof)
        if not passed:
            return None
        verified.append(_leg(direction, distance, overlap, proof))
        offset = direction * distance
        moved = [piece.translate(offset) for piece in moved]
        end = end.translate(offset)
    bb = end.bounding_box()
    escaped = bb[5] < P.SCENE_MIN_Z if item['kind'] == 'plate_perimeter_tack' else bb[2] > P.SCENE_MAX_Z
    return verified if escaped else None


def try_body_cuts(item, rem, cuts, original, obstacles, saved_routes=None, preferred=None, broad_search=True):
    augmented = dict(item, _active_body_cuts=cuts)
    return S.try_body_cuts(augmented, rem, cuts, original, obstacles, saved_routes, preferred, broad_search)


def _freeze_entry(entry):
    path = P.ROOT / entry['path']
    P.frozen_bytes(path)
    if P.INPUT_HASHES[entry['path']] != entry['sha256']:
        raise ValueError('Derived finishing stock changed: ' + entry['path'])


def prepare_plate(item, original):
    if item['id'] not in B.PLATE_IDS or '_plate_stock' in item:
        return
    original_entry = dict(path='cache/original_v5.npz', sha256=P.INPUT_HASHES['cache/original_v5.npz'])
    source_keys = ['cache/original_v5.npz', 'source/external_removal_plate_stock_v6.py']
    source_keys += ['cache/custom_holder_supports/' + item['record'][field]
                    for field in ['full_mesh_file', 'removable_mesh_file']]
    expected = {key: P.INPUT_HASHES[key] for key in source_keys}
    prepared = B.prepare_plate_stock(item, original, original_entry=original_entry,
        expected_input_hashes=expected, cache_root=P.ROOT / 'cache/external_removal_plate_stock_v6')
    manifest_path = Path(prepared['manifest_path'])
    manifest_data = P.frozen_bytes(manifest_path)
    manifest_relative = str(manifest_path.relative_to(P.ROOT))
    record = prepared['record']
    if record.get('passed') is not True:
        raise ValueError('Actual finishing-stock partition has not passed: ' + item['id'])
    for entry in record['files']:
        _freeze_entry(entry)
    item['_plate_stock'] = dict(manifest_path=manifest_relative,
        manifest_sha256=hashlib.sha256(manifest_data).hexdigest(), record=record)
    item['_original_nominal_removable'] = item['removable']
    item['removable'] = prepared['stock']
    item['pieces'] = [prepared['stock']]


def make_plan(item, original, obstacles, broad_search=True):
    if item['kind'] == 'plate_perimeter_tack':
        try:
            prepare_plate(item, original)
        except B.CertificateError as error:
            return None, dict(reason='Actual added finishing-stock partition failed closed: ' + str(error),
                              partition_checks=error.checks)
    plan, failure = _BASE_MAKE_PLAN(item, original, obstacles, broad_search)
    if plan is not None and '_plate_stock' in item:
        plan['plate_stock_partition'] = item['_plate_stock']
    return plan, failure


def save_plan(item, plan, sequence):
    row = S.save_plan(item, plan, sequence)
    if 'plate_stock_partition' in plan:
        partition = plan['plate_stock_partition']
        row['finishing_stock_partition'] = dict(
            manifest_path=partition['manifest_path'], sha256=partition['manifest_sha256'],
            removed_stock='Actual serialized nominal removable minus unchanged original V5 material.',
            original_material_preserved=True, physical_retained_state_unchanged=True,
            physical_tool_access_certified=False)
    row['extended_route_verifier'] = 'source/external_removal_extended_v6.py'
    return row


def install():
    global _INSTALLED
    S.install()
    P.frozen_bytes(__file__)
    P.frozen_bytes(B.__file__)
    if not _INSTALLED:
        P.candidate_directions = candidate_directions
        P.route_fragment = route_fragment
        P.verify_saved_legs = verify_saved_legs
        P.try_body_cuts = try_body_cuts
        P.make_plan = make_plan
        P.save_plan = save_plan
        _INSTALLED = True


def finish_report(path):
    S.finish_report(path)
    report = json.loads(path.read_text())
    report['extended_route_verifier'] = 'source/external_removal_extended_v6.py'
    report['method'] = ('Continuous swept-solid tests against the complete original, actual nominal retained stock, '
        'fixed additions and every not-yet-removed support. Conservative convex bounds are used where sufficient; '
        'the initial solid plus a checked bound for every swept boundary-triangle prism covers nonconvex '
        'translations without a Boolean union; any conservative padding and serialization coverage allowance '
        'are included in the summed complete-scene intersection bound. '
        'Extracted siblings are omitted only after their certified complete exits. Actual added underside finishing '
        'stock excludes the unchanged original under explicit partition and physical-state identity proofs.')
    report['plate_stock_partition_count'] = sum('finishing_stock_partition' in row for row in report['plans'])
    report['rejected_sweep_derivation_count'] = _SWEEP_DERIVATION_FAILURE_COUNT
    report['rejected_sweep_derivation_examples'] = _SWEEP_DERIVATION_FAILURE_EXAMPLES
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

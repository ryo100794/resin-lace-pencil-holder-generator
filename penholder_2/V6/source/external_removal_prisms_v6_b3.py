#!/usr/bin/env python3
"""Tight, certified translation-prism bounds; no printed geometry changes.

Frozen B's actual-stock preparation and exact rational predicates are reused.
The new constructor extrudes a convex canonical triangle, then maps its axes
to the two original triangle edges and the translation. This retains the
slanted end planes rather than filling a long transverse rectangle.

Every ideal binary-float input vertex and exact input-plus-delta endpoint must
lie inside every raw AND reloaded boundary halfspace. Nonzero cells are never
discarded. All analysis padding counts against the unchanged complete-scene
SUM <= 1e-8 mm3 gate. The actual signed starting solid remains component zero.
Callers must bind BOTH this source and frozen B in their immutable ledger.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import math

import manifold3d as mf
import numpy as np

import external_removal_plate_stock_v6 as B

ROOT = Path(__file__).resolve().parent.parent
BASE_SOURCE_SHA256 = 'bdcc40203f31000d8918066de93de0a525e9cc493f7c4ceb29c13f631b11b019'
NOMINAL_PADDING_MM = 1e-9
TOL = B.TOL
CertificateError = B.CertificateError
PLATE_IDS = B.PLATE_IDS
prepare_plate_stock = B.prepare_plate_stock


def source_bindings():
    base_path = Path(B.__file__).resolve()
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if base_hash != BASE_SOURCE_SHA256:
        raise CertificateError('Frozen base sweep helper changed')
    return [dict(path=str(Path(__file__).resolve().relative_to(ROOT)),
                 sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            dict(path=str(base_path.relative_to(ROOT)), sha256=base_hash)]


def _exact_matrix_det(matrix):
    columns = [tuple(Fraction.from_float(float(value)) for value in matrix[:, j])
               for j in range(3)]
    zero = (Fraction(0),) * 3
    return B._exact_det(zero, columns[0], columns[1], columns[2])


def _vertex_distance_bound(solid, ideals):
    """An exact-rational, outward-rounded vertex enclosure-size diagnostic.

Each output vertex is matched to its nearest ideal vertex. Convexity makes
the maximum distance a conservative bound from the entire constructor hull
to the ideal hull. This is not used to excuse any scene overlap.
"""
    vertices, _ = B._arrays(solid)
    largest = Fraction(0)
    for vertex in vertices:
        point = tuple(Fraction.from_float(float(x)) for x in vertex)
        squared = min(sum((point[k] - target[k]) ** 2 for k in range(3))
                      for target in ideals)
        largest = max(largest, squared)
    bound = math.sqrt(float(largest))
    while Fraction.from_float(bound) ** 2 < largest:
        bound = float(np.nextafter(bound, math.inf))
    return bound


def _certify(prism, target_points, ideals, face_index, construction, padding,
             failures, details=None):
    B._valid(prism, f'Boundary prism {face_index}')
    if prism.is_empty() or B._volume(prism) <= 0.:
        raise CertificateError('Nonzero prism collapsed')
    raw = B._contains_target_hull(prism, target_points, ideals)
    if not raw['passed']:
        raise CertificateError('Raw constructor excludes an ideal prism vertex', raw)
    reloaded, _, serialization = B._roundtrip(prism)
    B._valid(reloaded, f'Reloaded boundary prism {face_index}')
    if reloaded.is_empty() or B._volume(reloaded) <= 0.:
        raise CertificateError('Serialized nonzero prism collapsed')
    saved = B._contains_target_hull(reloaded, target_points, ideals)
    if not saved['passed']:
        raise CertificateError('Serialized constructor excludes an ideal prism vertex', saved)
    proof = dict(construction=construction, padding_mm=padding,
        padding_basis=('None' if padding == 0. else
            'Expanded canonical triangle and axial interval; all added analysis stock is counted in collisions.'),
        conservative_box=None, unpadded_failures=list(failures),
        raw_target_containment=raw, serialized_target_containment=saved,
        serialization=serialization, query_is_original_convex_constructor_object=True,
        maximum_vertex_to_ideal_vertex_distance_mm=_vertex_distance_bound(prism, ideals))
    if details is not None:
        proof['affine_prism'] = details
    return prism, proof


def _affine_triangular_prism(target_points, delta):
    """A convex triangle extrusion followed by a nonsingular affine map."""
    origin = target_points[0]
    u = target_points[1] - origin
    v = target_points[2] - origin
    matrix = np.column_stack([u, v, np.asarray(delta, float)])
    determinant = _exact_matrix_det(matrix)
    if determinant == 0:
        raise CertificateError('Rounded affine basis is singular; another certified bound is required')
    swapped = determinant < 0
    if swapped:
        matrix[:, [0, 1]] = matrix[:, [1, 0]]
    lengths = np.linalg.norm(matrix, axis=0)
    if not np.all(np.isfinite(lengths)) or np.min(lengths) <= 0.:
        raise CertificateError('Invalid affine prism basis lengths')
    planar_pad = NOMINAL_PADDING_MM / min(lengths[0], lengths[1])
    axial_pad = NOMINAL_PADDING_MM / lengths[2]
    if not math.isfinite(planar_pad + axial_pad):
        raise CertificateError('Affine padding is not representable')
    # This triangle contains (0,0),(1,0),(0,1), with positive margin on
    # every side. The exact world-space test remains authoritative after
    # all projection, extrusion and transform rounding.
    points = np.array([[-planar_pad, -planar_pad],
                       [1. + 2. * planar_pad, -planar_pad],
                       [-planar_pad, 1. + 2. * planar_pad]])
    section = mf.CrossSection.hull_points(points)
    prism = section.extrude(1. + 2. * axial_pad).translate((0., 0., -axial_pad))
    prism = prism.transform(np.column_stack([matrix, origin]))
    return prism, dict(
        constructor='convex CrossSection.hull_points, untwisted constant-section extrusion, nonsingular affine transform',
        preserves_triangle_cross_section_and_slanted_ends=True,
        nominal_padding_parameter_mm=NOMINAL_PADDING_MM,
        canonical_triangle_edge_margin=planar_pad,
        canonical_axial_margin=axial_pad,
        positive_orientation_basis_swap=bool(swapped),
        basis_column_lengths_mm=lengths.tolist(),
        extra_analysis_geometry_is_fully_charged_to_scene_overlap=True)


def checked_prism(target_points, target_rationals, delta, face_index):
    """Use a tight certified prism, retaining the old safe fallback if needed."""
    target_points = np.asarray(target_points, float)
    failures = []
    try:
        return _certify(mf.Manifold.hull_points(target_points), target_points,
                        target_rationals, face_index, 'exact_hull', 0., failures)
    except CertificateError as error:
        failures.append(str(error))
    try:
        prism, details = _affine_triangular_prism(target_points, delta)
        return _certify(prism, target_points, target_rationals, face_index,
            'expanded_affine_triangular_prism', NOMINAL_PADDING_MM, failures, details)
    except (CertificateError, ValueError, RuntimeError) as error:
        failures.append(str(error))
    # Near-singular cases still receive an explicitly certified enclosure;
    # frozen B may return its larger bound or reject the whole candidate.
    # No face is dropped and no collision tolerance changes here.
    prism, proof = B._checked_prism(target_points, target_rationals, delta, face_index)
    proof['b3_tight_attempt_failures'] = failures
    proof['maximum_vertex_to_ideal_vertex_distance_mm'] = _vertex_distance_bound(prism, target_rationals)
    return prism, proof


def translation_sweep_components(solid, direction, distance):
    """Return actual stock plus all certified boundary-prism enclosures."""
    bindings = source_bindings()
    B._valid(solid, 'Starting actual sweep stock')
    direction = np.asarray(direction, float)
    distance = float(distance)
    if (direction.shape != (3,) or not np.all(np.isfinite(direction))
            or not math.isfinite(distance) or distance < 0.
            or abs(float(np.linalg.norm(direction)) - 1.) > 1e-8):
        raise CertificateError('Translation requires a finite unit direction and nonnegative distance')
    delta = direction * distance
    vertices, faces = B._arrays(solid)
    _, _, initial_serialization = B._roundtrip(solid)
    qvertices = [tuple(Fraction.from_float(float(x)) for x in vertex) for vertex in vertices]
    qdelta = tuple(Fraction.from_float(float(x)) for x in delta)
    components, derivations, zero_faces = [solid], [], []
    for face_index, face in enumerate(faces):
        qa, qb, qc = [qvertices[int(i)] for i in face]
        determinant = B._exact_det(qa, qb, qc, qdelta)
        if determinant == 0:
            zero_faces.append(int(face_index))
            continue
        expected = float(abs(determinant) / 2)
        if expected <= 0. or not math.isfinite(expected):
            raise CertificateError('Nonzero exact prism volume is not representable',
                                   dict(face_index=int(face_index)))
        triangle = vertices[face]
        ideals = [qa, qb, qc] + [tuple(point[k] + qdelta[k] for k in range(3))
                                for point in (qa, qb, qc)]
        prism, certificate = checked_prism(np.vstack([triangle, triangle + delta]),
                                          ideals, delta, face_index)
        components.append(prism)
        rt = certificate['serialization']
        derivations.append(dict(face_index=int(face_index), component_index=len(components) - 1,
            expected_prism_volume_mm3=expected, constructed_bound_volume_mm3=B._volume(prism),
            conservative_extra_volume_mm3=max(0., float(prism.volume()) - expected),
            expected_volume_error_mm3=abs(float(prism.volume()) - expected),
            serialized_sha256=rt['serialized_sha256'],
            identical_oriented_triangles=rt['identical_oriented_triangles'],
            serialization_uncovered_mm3=rt['uncovered_source_volume_mm3'],
            **{key: value for key, value in certificate.items() if key != 'serialization'}))
    proof = dict(
        method='unassembled_all_boundary_triangle_prism_union_with_summed_scene_overlap_affine_v3',
        representation='Mathematical union certificate; components are not Boolean-unioned. Expanded prisms are conservative bounds.',
        source=bindings[0], base_source=bindings[1], required_source_bindings=bindings,
        units='mm', direction=direction.tolist(), distance_mm=distance, translation_mm=delta.tolist(),
        starting_oriented_triangles_sha256=B._oriented_triangles_sha(solid),
        actual_starting_stock_preserved=True, starting_serialization=initial_serialization,
        boundary_triangle_count=len(faces), convex_prism_count=len(derivations),
        verification_component_count=len(components), exactly_zero_volume_face_count=len(zero_faces),
        exactly_zero_volume_face_indices=zero_faces,
        zero_volume_proof='Exact rational determinant of binary-float source points and binary-float delta equals zero.',
        near_parallel_nonzero_faces_discarded=0,
        padded_prism_count=sum(row['padding_mm'] > 0. for row in derivations),
        affine_triangular_prism_count=sum(row['construction'] == 'expanded_affine_triangular_prism' for row in derivations),
        conservative_box_face_indices=[row['face_index'] for row in derivations if row['construction'] == 'thin_oriented_box'],
        padded_face_indices=[row['face_index'] for row in derivations if row['padding_mm'] > 0.],
        maximum_nominal_padding_parameter_mm=max((row['padding_mm'] for row in derivations), default=0.),
        maximum_vertex_enclosure_distance_mm=max((row['maximum_vertex_to_ideal_vertex_distance_mm'] for row in derivations), default=0.),
        enclosure_distance_basis='Maximum exact-rational, outward-rounded output-vertex distance to a nearest ideal vertex; diagnostic upper bound, never a collision exemption.',
        padded_volume_included_in_collision_sum=True,
        analytical_coverage='The starting signed solid plus every boundary triangle swept by delta covers its full continuous translation, including cavity boundaries.',
        actual_component_containment='All six ideal points satisfy every raw and reloaded oriented boundary halfspace exactly. Query components are original convex constructor objects.',
        ideal_endpoint_coordinates='Exact Fraction(source binary-float coordinate) plus Fraction(binary-float delta), without rounding the sum.',
        serialization_uncovered_bound_mm3=0.,
        overlap_rule='SUM every complete-scene component overlap, counting duplicates, plus serialization_uncovered_bound_mm3; accept only total<=1e-8 mm3.',
        prism_derivation_records_sha256=B._sha(json.dumps(derivations, sort_keys=True).encode()),
        derivation_records_scope='Regenerate from the bound actual mesh, translation and both source files; no prior collision proof is reused.',
        raw_and_serialized_exact_halfspace_tests=sum(row['raw_target_containment']['exact_rational_tests']
            + row['serialized_target_containment']['exact_rational_tests'] for row in derivations),
        target_containment_violations=0,
        maximum_conservative_extra_volume_mm3=max((row['conservative_extra_volume_mm3'] for row in derivations), default=0.),
        total_conservative_extra_volume_mm3=sum(row['conservative_extra_volume_mm3'] for row in derivations),
        expected_prism_volume_error_sum_mm3=sum(row['expected_volume_error_mm3'] for row in derivations),
        expected_prism_volume_error_scope='Diagnostic only; literal target containment supplies coverage and all added volume is charged to collisions.',
        literal_roundtrip_triangle_identity_count=sum(row['identical_oriented_triangles'] for row in derivations),
        maximum_roundtrip_uncovered_volume_mm3=max((row['serialization_uncovered_mm3'] for row in derivations), default=0.),
        static_geometry_checks_passed=True, current_complete_scene_check_required=True)
    return components, proof


def validate_fixtures(path):
    """Small fixture geometry only; no original or scene collision claims."""
    path = Path(path)
    data = path.read_bytes()
    fixture = json.loads(data)
    rows = []
    for item in fixture['results']:
        for leg in item['legs']:
            prior = leg.get('failed_component_provenance', {})
            if prior.get('construction') != 'thin_oriented_box':
                continue
            points = np.asarray(prior['six_rounded_target_points_mm'], float)
            delta = np.asarray(leg['direction'], float) * float(leg['distance_mm'])
            qtriangle = [tuple(Fraction.from_float(float(x)) for x in point) for point in points[:3]]
            qdelta = tuple(Fraction.from_float(float(x)) for x in delta)
            ideals = qtriangle + [tuple(point[k] + qdelta[k] for k in range(3)) for point in qtriangle]
            prism, proof = checked_prism(points, ideals, delta, prior['face_index'])
            old, old_proof = B._checked_prism(points, ideals, delta, prior['face_index'])
            expected = float(abs(B._exact_det(*qtriangle, qdelta)) / 2)
            rows.append(dict(id=item['id'], face_index=prior['face_index'],
                new_construction=proof['construction'], old_construction=old_proof['construction'],
                actual_ideal_prism_volume_mm3=expected, old_bound_volume_mm3=B._volume(old),
                new_bound_volume_mm3=B._volume(prism),
                new_conservative_extra_volume_mm3=max(0., B._volume(prism) - expected),
                proof=proof))
    report = dict(scope='Small fixture containment/serialization only; no complete-scene route check',
        source_bindings=source_bindings(), fixture_path=str(path.resolve().relative_to(ROOT)),
        fixture_sha256=hashlib.sha256(data).hexdigest(),
        passed=bool(rows) and all(row['new_construction'] == 'expanded_affine_triangular_prism' for row in rows),
        records=rows, printed_geometry_modified=False)
    output = ROOT / 'reports/external_removal_prisms_b3_fixture_validation.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(report=str(output.relative_to(ROOT)), sha256=B._sha(output.read_bytes()),
        passed=report['passed'], fixture_count=len(rows), source_sha256=report['source_bindings'][0]['sha256'])), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-fixtures', type=Path, required=True)
    validate_fixtures(parser.parse_args().validate_fixtures)

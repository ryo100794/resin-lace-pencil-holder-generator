#!/usr/bin/env python3
"""Analysis-only stock partition and continuous translation certificates.

No printed mesh, existing support partition or planner input is modified.
The 40 named plate operations may remove C = serialized R minus original O;
the old permanent obstacle K = F minus R remains unchanged.  The identity

    O union (F minus C) == O union K

is checked in a local ROI containing the complete F and R.  Original O is
supplied by the caller's already hash-bound scene; this module never loads a
second original scene.  The caller must keep O and the old K in its field.

Continuous translation uses an UNASSEMBLED UNION CERTIFICATE: the actual
starting solid plus the triangular prism swept by EVERY boundary triangle.
It does not Boolean-union those coincidence-heavy prisms.  Mathematically
their union is exactly the continuous sweep.  The caller must SUM complete
scene intersections of all components, counting duplicates, and accept only
if that sum plus the reported serialization coverage allowance is <= 1e-8
mm^3.  A maximum over individual overlaps is not a valid certificate.

Public APIs:
  prepare_plate_stock(item, original, original_entry, expected_input_hashes,
                      cache_root=None) -> {stock, record, manifest_path}
  translation_sweep_components(solid, direction, distance)
                      -> (components, proof)
  verify_translation(prepared, direction, distance, original, obstacles)
                      -> result including the conservative SUM certificate

The first component is the original signed solid, with cavities intact;
subsequent components are individually checked convex triangular prisms or
explicit conservative enclosures with microscopic analysis-only padding.
Exactly parallel triangles are omitted only after an exact rational
determinant proves that their swept set has zero three-dimensional volume.
Near-parallel nonzero prisms are never discarded by a numerical threshold.

Plate stock/cache paths are ROOT-relative in records.  Prism serialization
is checked in memory; no per-leg prism files are written.  Finishing this
stock is a nominal underside trimming operation, not a finite cutter-path,
tool-access, manual-accuracy or material-strength certification.
"""
from pathlib import Path
from fractions import Fraction
import hashlib
import io
import json
import math
import os
import time

import manifold3d as mf
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve()
TOL = 1e-8
ROI_MARGIN_MM = .02
ANALYSIS_PRISM_PAD_MM = 1e-9
PLATE_IDS = frozenset(
    [f'H{i:04d}' for i in
     [24, 26, 29, 31, 80, 82, 85, 87, 104, 107, 110, 111,
      112, 115, 118, 119, 120, 122, 125, 127, 144, 146, 149, 151]]
    + [f'H{i:04d}' for i in range(579, 595)])


class CertificateError(ValueError):
    """A candidate failed without changing any frozen source geometry."""
    def __init__(self, message, checks=None):
        super().__init__(message)
        self.checks = checks or {}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _relative(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve().relative_to(ROOT))


def _source_entry():
    return dict(path=_relative(SOURCE), sha256=_sha(SOURCE.read_bytes()))


def _volume(solid):
    if solid.is_empty():
        return 0.
    value = float(solid.volume())
    if not math.isfinite(value) or value < -TOL:
        raise CertificateError('Invalid signed volume in a certificate',
                               dict(signed_volume_mm3=value))
    return max(0., value)


def _valid(solid, label):
    if solid.status() != mf.Error.NoError:
        raise CertificateError(label + ': invalid Manifold status',
                               dict(status=str(solid.status())))
    _volume(solid)


def _union(parts):
    nonempty = [part for part in parts if not part.is_empty()]
    return (mf.Manifold.batch_boolean(nonempty, mf.OpType.Add)
            if nonempty else mf.Manifold())


def _arrays(solid):
    mesh = solid.to_mesh64()
    vertices = np.array(mesh.vert_properties[:, :3], dtype=np.float64, copy=True)
    faces = np.array(mesh.tri_verts, dtype=np.uint64, copy=True)
    if not np.all(np.isfinite(vertices)):
        raise CertificateError('Nonfinite mesh coordinates')
    return vertices, faces


def _from_arrays(vertices, faces):
    result = mf.Manifold(mf.Mesh64(np.array(vertices, np.float64, copy=True),
                                 np.array(faces, np.uint64, copy=True)))
    _valid(result, 'Reloaded mesh')
    return result


def _bytes(solid):
    vertices, faces = _arrays(solid)
    data = io.BytesIO()
    np.savez_compressed(data, vertices=vertices, faces=faces)
    return data.getvalue()


def _load_bytes(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        vertices = np.array(source['vertices'], np.float64, copy=True)
        faces = np.array(source['faces'], np.uint64, copy=True)
    return _from_arrays(vertices, faces)


def _oriented_triangles_sha(solid):
    """Independent of face order, vertex numbering and cyclic face rotation."""
    vertices, faces = _arrays(solid)
    triangles = []
    for face in faces:
        p = vertices[face]
        p[p == 0.] = 0.  # +0 and -0 denote the same coordinate.
        cyclic = [tuple(np.roll(p, -k, axis=0).reshape(-1)) for k in range(3)]
        triangles.append(min(cyclic))
    triangles.sort()
    array = np.asarray(triangles, dtype='<f8').reshape((-1, 9))
    return _sha(array.tobytes())


def _roundtrip(solid):
    """Keep actual signed solids intact; never drop boundary components."""
    _valid(solid, 'Input mesh')
    data = _bytes(solid)
    loaded = _load_bytes(data)
    before = _oriented_triangles_sha(solid)
    after = _oriented_triangles_sha(loaded)
    exact_faces = before == after
    # Literal oriented-triangle equality proves identity without an unstable
    # Boolean subtraction of two coincident copies of the same surface.
    missing = 0. if exact_faces else _volume(solid - loaded)
    extra = 0. if exact_faces else _volume(loaded - solid)
    volume_error = abs(float(solid.volume()) - float(loaded.volume()))
    checks = dict(
        identical_oriented_triangles=exact_faces,
        uncovered_source_volume_mm3=missing,
        added_volume_mm3=extra,
        signed_volume_error_mm3=volume_error,
        original_oriented_triangles_sha256=before,
        loaded_oriented_triangles_sha256=after,
        serialized_sha256=_sha(data))
    if max(missing, extra, volume_error) > TOL:
        raise CertificateError('Serialization changes the analysis solid', checks)
    return loaded, data, checks


def _entry(path, expected_hashes):
    rel = _relative(path)
    if rel not in expected_hashes:
        raise CertificateError('Caller has not hash-bound source input: ' + rel)
    data = (ROOT / rel).read_bytes()
    digest = _sha(data)
    if digest != expected_hashes[rel]:
        raise CertificateError('Frozen source input changed: ' + rel)
    return dict(path=rel, sha256=digest), data


def _immutable_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise CertificateError('Refusing to overwrite immutable cache: ' + str(path))
        return
    temp = path.with_name(path.name + f'.part.{os.getpid()}.{time.time_ns()}')
    try:
        temp.write_bytes(data)
        # Link is atomic and refuses to replace an existing immutable result.
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _check_gate(checks, message):
    bad = {name: float(value) for name, value in checks.items()
           if not math.isfinite(float(value)) or float(value) > TOL}
    if bad:
        raise CertificateError(message, checks)


def _state_checks(full, old_removable, original_local, clean, excluded,
                  old_retained, post_trim, roi):
    cover = _union([clean, excluded])
    old_scene = _union([original_local, old_retained])
    new_scene = _union([original_local, post_trim])
    actual_post_removal = _union([original_local, full]) - clean
    checks = dict(
        original_removable_outside_full_mm3=_volume(old_removable - full),
        full_outside_roi_mm3=_volume(full - roi),
        original_removable_outside_roi_mm3=_volume(old_removable - roi),
        clean_outside_original_removable_mm3=_volume(clean - old_removable),
        clean_outside_full_mm3=_volume(clean - full),
        clean_intersects_original_mm3=_volume(clean ^ original_local),
        excluded_stock_outside_original_mm3=_volume(excluded - original_local),
        partition_uncovered_original_removable_mm3=_volume(old_removable - cover),
        partition_outside_original_removable_mm3=_volume(cover - old_removable),
        clean_excluded_interior_overlap_mm3=_volume(clean ^ excluded),
        partition_signed_volume_error_mm3=abs(
            clean.volume() + excluded.volume() - old_removable.volume()),
        changed_retained_stock_outside_original_mm3=_volume(
            (post_trim - old_retained) - original_local),
        old_scene_missing_from_new_scene_mm3=_volume(old_scene - new_scene),
        new_scene_outside_old_scene_mm3=_volume(new_scene - old_scene),
        physical_removal_missing_from_old_scene_mm3=_volume(old_scene - actual_post_removal),
        physical_removal_outside_old_scene_mm3=_volume(actual_post_removal - old_scene))
    _check_gate(checks, 'Plate stock/state identity failed')
    return checks


def prepare_plate_stock(item, original, original_entry,
                        expected_input_hashes, cache_root=None):
    """Derive and persist only C=actual saved R-original for the 40 plate IDs.

    original_entry is {path,sha256}; expected_input_hashes is the caller's
    frozen ROOT-relative input map (for example P.INPUT_HASHES).  Passing the
    original object asserts it came from that already bound source entry.
    Only the small F/R files are loaded again here, with matching hashes.
    Returned old_retained_reference is evidence only, never a replacement
    for the caller's existing permanent obstacle or maximum-retained gate.
    """
    if item.get('id') not in PLATE_IDS or item.get('kind') != 'plate_perimeter_tack':
        raise CertificateError('Plate correction is outside its declared 40-part scope')
    _valid(original, 'Caller-bound original scene')
    orig_entry = dict(path=_relative(original_entry['path']),
                      sha256=original_entry['sha256'])
    if expected_input_hashes.get(orig_entry['path']) != orig_entry['sha256']:
        raise CertificateError('Original scene binding does not match the caller ledger')
    row = item['record']
    folder = ROOT / 'cache/custom_holder_supports'
    full_entry, full_bytes = _entry(folder / row['full_mesh_file'], expected_input_hashes)
    rem_entry, rem_bytes = _entry(folder / row['removable_mesh_file'], expected_input_hashes)
    source_entry, _ = _entry(SOURCE, expected_input_hashes)
    bindings = dict(source=source_entry, original=orig_entry,
                    full=full_entry, original_removable=rem_entry)
    policy = dict(id=item['id'], tolerance_mm3=TOL, roi_margin_mm=ROI_MARGIN_MM,
                  method='serialized_R_minus_original_preserve_old_K_v1')
    key = _sha(json.dumps(dict(bindings=bindings, policy=policy), sort_keys=True).encode())
    base = Path(cache_root) if cache_root is not None else ROOT / 'cache/plate_stock_v6'
    if not base.is_absolute():
        base = ROOT / base
    folder_out = base / item['id'] / key
    manifest_path = folder_out / 'partition.json'
    if manifest_path.exists():
        record = json.loads(manifest_path.read_bytes())
        if record.get('bindings') != bindings or record.get('policy') != policy or record.get('passed') is not True:
            raise CertificateError('Existing immutable plate cache has invalid bindings')
        loaded = {}
        for entry in record['files']:
            data = (ROOT / entry['path']).read_bytes()
            if _sha(data) != entry['sha256']:
                raise CertificateError('Immutable derived plate cache changed: ' + entry['path'])
            loaded[entry['role']] = _load_bytes(data)
        return dict(stock=loaded['cleaned_removable'], record=record,
                    manifest_path=str(manifest_path))

    full = _load_bytes(full_bytes)
    old_removable = _load_bytes(rem_bytes)
    binding_checks = dict(
        caller_full_missing_mm3=_volume(full - item['full']),
        caller_full_extra_mm3=_volume(item['full'] - full),
        caller_removable_missing_mm3=_volume(old_removable - item['removable']),
        caller_removable_extra_mm3=_volume(item['removable'] - old_removable))
    _check_gate(binding_checks, 'Passed item is not its actual serialized F/R')
    bounds = np.asarray(full.bounding_box(), float)
    rbox = np.asarray(old_removable.bounding_box(), float)
    low = np.minimum(bounds[:3], rbox[:3]) - ROI_MARGIN_MM
    high = np.maximum(bounds[3:], rbox[3:]) + ROI_MARGIN_MM
    roi = mf.Manifold.cube(tuple(high - low)).translate(low)
    original_local = original ^ roi
    clean = old_removable - original_local
    excluded = old_removable ^ original_local
    old_retained = full - old_removable
    post_trim = full - clean
    _state_checks(full, old_removable, original_local, clean, excluded,
                  old_retained, post_trim, roi)

    solids = dict(cleaned_removable=clean, excluded_original_stock=excluded,
                  old_retained_reference=old_retained,
                  post_trim_part_reference=post_trim,
                  original_local_context=original_local)
    reloaded = {}
    blobs = {}
    roundtrips = {}
    files = []
    for role, solid in solids.items():
        loaded, data, proof = _roundtrip(solid)
        path = folder_out / (role + '.npz')
        reloaded[role] = loaded
        blobs[role] = (path, data)
        roundtrips[role] = proof
        files.append(dict(role=role, path=_relative(path), sha256=_sha(data)))
    checks = _state_checks(
        full, old_removable, reloaded['original_local_context'],
        reloaded['cleaned_removable'], reloaded['excluded_original_stock'],
        reloaded['old_retained_reference'], reloaded['post_trim_part_reference'], roi)
    # The cached original crop must still stand for the actual original crop.
    checks['cached_original_crop_uncovered_mm3'] = _volume(
        original_local - reloaded['original_local_context'])
    checks['cached_original_crop_extra_mm3'] = _volume(
        reloaded['original_local_context'] - original_local)
    _check_gate(checks, 'Serialized plate state identity failed')
    record = dict(
        schema='plate_stock_partition_v1', passed=True, units='mm',
        bindings=bindings, policy=policy, files=files,
        roi_bounds_mm=np.r_[low, high].tolist(),
        caller_binding_checks=binding_checks, checks=checks,
        serialization_checks=roundtrips,
        volumes_mm3=dict(original_removable=_volume(old_removable),
                         cleaned_removable=_volume(reloaded['cleaned_removable']),
                         preserved_original_overlap=_volume(reloaded['excluded_original_stock'])),
        physical_state_identity='O union (F minus (R minus O)) = O union (F minus R)',
        static_obstacle_rule='Keep the existing original O and old permanent K=F-R unchanged.',
        original_binding='Caller supplied the already loaded solid from the frozen original path/hash; no second scene load.',
        cut_method='Underside finishing of actual added stock; original-overlap material remains in the original model.',
        cut_boundary_basis='The actual serialized R surface clipped by original O; historical trim height is not reconstructed.',
        finite_cutter_or_manual_accuracy_certified=False,
        full_scene_route_validation='Required separately against the current complete obstacle state.')
    for role, (path, data) in blobs.items():
        _immutable_write(path, data)
    _immutable_write(manifest_path, (json.dumps(record, indent=2, sort_keys=True) + '\n').encode())
    return dict(stock=reloaded['cleaned_removable'], record=record,
                manifest_path=str(manifest_path))


def _exact_det(a, b, c, delta):
    u = [b[k] - a[k] for k in range(3)]
    v = [c[k] - a[k] for k in range(3)]
    cross = [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2],
             u[0]*v[1] - u[1]*v[0]]
    return sum(cross[k] * delta[k] for k in range(3))


def _contains_target_hull(solid, target_points, target_rationals=None):
    """Exact rational halfspace proof for all six target hull vertices.

    Every point must lie in every inward boundary-face halfspace. Their
    convex hull therefore does too. This is sufficient for containment,
    including after roundtrip; no volume tolerance stands in for this test.
    """
    vertices, faces = _arrays(solid)
    qv = [tuple(Fraction.from_float(float(x)) for x in row) for row in vertices]
    qp = (target_rationals if target_rationals is not None else
          [tuple(Fraction.from_float(float(x)) for x in row) for row in target_points])
    violations = []
    tested = 0
    for face_index, face in enumerate(faces):
        a, b, c = [qv[int(index)] for index in face]
        for point_index, point in enumerate(qp):
            value = _exact_det(a, b, c, [point[k] - a[k] for k in range(3)])
            tested += 1
            if value > 0:
                violations.append(dict(face_index=int(face_index), point_index=point_index,
                                       positive_oriented_six_volume_mm3=float(value)))
    return dict(passed=not violations, exact_rational_tests=tested,
                violating_point_face_pairs=len(violations),
                maximum_positive_oriented_six_volume_mm3=max(
                    (row['positive_oriented_six_volume_mm3'] for row in violations), default=0.))


def _checked_prism(target_points, target_rationals, delta, face_index):
    """Keep a raw convex hull query, with literal target coverage verified.

    A collapsed or inward-rounded nonzero hull is replaced by a conservative
    hull of microscopic axis-aligned boxes around all six points. Original
    points are also retained. This adds analysis-only volume; none of that
    volume is excused from the scene overlap sum.
    """
    reasons = []
    for strategy in ('exact_hull', 'padded_hull', 'thin_oriented_box'):
        padding = 0. if strategy == 'exact_hull' else ANALYSIS_PRISM_PAD_MM
        box_info = None
        if strategy == 'thin_oriented_box':
            # A hull implementation can itself collapse a very thin padded
            # hull. An affine cube preserves that small dimension explicitly.
            # This is a conservative box, not claimed to be an exact prism.
            origin = target_points.mean(axis=0)
            w = np.asarray(delta, float) / np.linalg.norm(delta)
            centered = target_points - origin
            flat = centered - (centered @ w)[:, None]*w
            differences = flat[:, None, :] - flat[None, :, :]
            lengths = np.linalg.norm(differences, axis=2)
            ia, ib = np.unravel_index(int(lengths.argmax()), lengths.shape)
            if lengths[ia, ib] > 0.:
                u = differences[ia, ib] / lengths[ia, ib]
            else:
                basis = np.eye(3)[int(np.argmin(np.abs(w)))]
                u = np.cross(w, basis)
                u /= np.linalg.norm(u)
            v = np.cross(w, u)
            v /= np.linalg.norm(v)
            matrix = np.column_stack([u, v, w])
            local = centered @ matrix
            low = local.min(axis=0) - padding
            high = local.max(axis=0) + padding
            prism = mf.Manifold.cube(tuple(high-low)).transform(
                np.column_stack([matrix, origin + matrix @ low]))
            box_info = dict(local_padding_per_side_mm=padding,
                            local_extent_mm=(high-low).tolist(),
                            tangential_rectangle_overfill_included_in_scene_check=True)
        elif padding:
            signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.)
                              for z in (-1., 1.)])
            expanded = (target_points[:, None, :] + padding*signs[None, :, :]).reshape((-1, 3))
            points = np.vstack([target_points, expanded])
            prism = mf.Manifold.hull_points(points)
        else:
            prism = mf.Manifold.hull_points(target_points)
        _valid(prism, f'Prism of boundary triangle {face_index}')
        if prism.is_empty() or _volume(prism) <= 0.:
            reasons.append('Nonzero hull collapsed')
            continue
        raw_containment = _contains_target_hull(prism, target_points, target_rationals)
        if not raw_containment['passed']:
            reasons.append('Raw hull fails literal six-point containment')
            continue
        try:
            reloaded, _, rt = _roundtrip(prism)
        except CertificateError as error:
            reasons.append(str(error))
            continue
        if reloaded.is_empty():
            reasons.append('Serialized nonzero hull collapsed')
            continue
        loaded_containment = _contains_target_hull(reloaded, target_points, target_rationals)
        if not loaded_containment['passed']:
            reasons.append('Roundtrip fails literal six-point containment')
            continue
        # Keep the actual hull_points or affine-cube object as the query:
        # convexity follows from construction, not a volume comparison on an
        # independently reconstructed mesh.
        return prism, dict(padding_mm=padding,
            construction=strategy,
            padding_basis=('None' if not padding else
                ('Conservative affine box containing all six ideal points.' if box_info else
                 'Conservative hull of original six points and axis-aligned +/-padding boxes around them.')),
            conservative_box=box_info,
            unpadded_failures=reasons, raw_target_containment=raw_containment,
            serialized_target_containment=loaded_containment,
            serialization=rt, query_is_original_convex_constructor_object=True)
    raise CertificateError('Cannot construct a stable covering nonzero prism; candidate rejected',
                           dict(face_index=int(face_index), attempts=reasons,
                                maximum_attempted_analysis_padding_mm=ANALYSIS_PRISM_PAD_MM))


def translation_sweep_components(solid, direction, distance):
    """Return an unassembled exact-union representation and coverage proof.

    Components[0] is the actual signed starting solid. Every later component
    is a convex prism or an explicitly recorded microscopic conservative
    enclosure of one. The sum of their scene-intersection volumes is a
    conservative upper bound on the mathematical union's intersection.
    Literal rational six-point containment is checked for every raw and
    serialized prism, so no volume-based coverage allowance is inferred.
    The proof's serialization_uncovered_bound_mm3 is retained as an API
    field and must be added to the sum (it is zero for this proof method).
    No whole-sweep convex hull or Boolean union is constructed.
    """
    _valid(solid, 'Starting sweep stock')
    d = np.asarray(direction, float)
    travel = float(distance)
    if d.shape != (3,) or not np.all(np.isfinite(d)) or not math.isfinite(travel):
        raise CertificateError('Invalid translation vector')
    if travel < 0. or abs(float(np.linalg.norm(d)) - 1.) > 1e-8:
        raise CertificateError('Translation must have unit direction and nonnegative distance')
    delta = d * travel
    vertices, faces = _arrays(solid)
    _, _, initial_roundtrip = _roundtrip(solid)
    # The original starting solid itself is kept, so its roundtrip missing
    # volume is not an omission from this verifier's first component.
    components = [solid]
    qvertices = [tuple(Fraction.from_float(float(x)) for x in row) for row in vertices]
    qdelta = tuple(Fraction.from_float(float(x)) for x in delta)
    zero_faces = []
    facet_rows = []
    uncovered_bound = 0.
    expected_volume_error_sum = 0.
    padded_count = 0
    for face_index, face in enumerate(faces):
        qa, qb, qc = [qvertices[int(index)] for index in face]
        det = _exact_det(qa, qb, qc, qdelta)
        if det == 0:
            zero_faces.append(int(face_index))
            continue
        expected_volume = float(abs(det) / 2)
        if expected_volume <= 0. or not math.isfinite(expected_volume):
            raise CertificateError('A nonzero prism is not representable in float64',
                                   dict(face_index=int(face_index)))
        triangle = vertices[face]
        ideal_points = [qa, qb, qc] + [tuple(point[k] + qdelta[k] for k in range(3))
                                      for point in (qa, qb, qc)]
        prism, prism_proof = _checked_prism(
            np.vstack([triangle, triangle + delta]), ideal_points, delta, face_index)
        rt = prism_proof['serialization']
        expected_error = abs(float(prism.volume()) - expected_volume)
        expected_volume_error_sum += expected_error
        padded_count += prism_proof['padding_mm'] > 0.
        components.append(prism)
        facet_rows.append(dict(
            face_index=int(face_index), component_index=len(components)-1,
            expected_prism_volume_mm3=expected_volume,
            expected_volume_error_mm3=expected_error,
            constructed_bound_volume_mm3=_volume(prism),
            conservative_extra_volume_mm3=max(0., float(prism.volume())-expected_volume),
            serialized_sha256=rt['serialized_sha256'],
            identical_oriented_triangles=rt['identical_oriented_triangles'],
            serialization_uncovered_mm3=rt['uncovered_source_volume_mm3'],
            serialization_extra_mm3=rt['added_volume_mm3'],
            **{key: value for key, value in prism_proof.items() if key != 'serialization'}))
    proof = dict(
        method='unassembled_all_boundary_triangle_prism_union_with_summed_scene_overlap_v1',
        representation='Mathematical union; verification components are intentionally not Boolean-unioned. Any explicitly padded prism is a conservative enclosure, not an exact prism.',
        units='mm', source=_source_entry(),
        starting_oriented_triangles_sha256=_oriented_triangles_sha(solid),
        direction=d.tolist(), distance_mm=travel, translation_mm=delta.tolist(),
        actual_starting_stock_preserved=True, starting_serialization=initial_roundtrip,
        boundary_triangle_count=len(faces), convex_prism_count=len(facet_rows),
        verification_component_count=len(components),
        exactly_zero_volume_face_count=len(zero_faces),
        exactly_zero_volume_face_indices=zero_faces,
        zero_volume_proof='Exact Fraction determinant of binary float input coordinates and translation equals zero.',
        near_parallel_nonzero_faces_discarded=0,
        padded_prism_count=int(padded_count),
        maximum_analysis_padding_mm=(ANALYSIS_PRISM_PAD_MM if padded_count else 0.),
        padded_volume_included_in_collision_sum=True,
        analytical_coverage='Sweep(S,d)=S union all(T+[0,d]) over every boundary triangle T, including cavity faces. A point reached outside S crosses a boundary along its translation line.',
        actual_component_containment='Every six-point target hull lies in all inward boundary-face halfspaces of both raw and serialized component, checked with exact rational determinants. Queries use original convex-hull or affine-cube constructor objects.',
        ideal_endpoint_coordinates='Exact Fraction(source binary-float coordinate) + Fraction(binary-float translation component), not rounded numpy endpoint addition.',
        serialization_uncovered_bound_mm3=uncovered_bound,
        expected_prism_volume_error_sum_mm3=expected_volume_error_sum,
        expected_prism_volume_error_scope='Diagnostic only, not a coverage allowance; explicit target-hull containment proves coverage and all padded extra stock is included in overlap tests.',
        overlap_rule='SUM all component complete-scene intersections, counting duplicates, plus serialization_uncovered_bound_mm3; accept only total<=1e-8 mm3.',
        prism_derivation_records_sha256=_sha(json.dumps(facet_rows, sort_keys=True).encode()),
        derivation_records_scope='Transient per-prism records are summarized here; every component is regenerated from the bound starting mesh, translation vector and source code, and revalidated.',
        raw_and_serialized_exact_halfspace_tests=sum(
            row['raw_target_containment']['exact_rational_tests']
            + row['serialized_target_containment']['exact_rational_tests'] for row in facet_rows),
        target_containment_violations=0,
        padded_face_indices=[row['face_index'] for row in facet_rows if row['padding_mm'] > 0.],
        conservative_box_face_indices=[row['face_index'] for row in facet_rows
                                       if row['construction'] == 'thin_oriented_box'],
        maximum_conservative_extra_volume_mm3=max(
            (row['conservative_extra_volume_mm3'] for row in facet_rows), default=0.),
        total_conservative_extra_volume_mm3=sum(
            row['conservative_extra_volume_mm3'] for row in facet_rows),
        literal_roundtrip_triangle_identity_count=sum(
            row['identical_oriented_triangles'] for row in facet_rows),
        maximum_roundtrip_uncovered_volume_mm3=max(
            (row['serialization_uncovered_mm3'] for row in facet_rows), default=0.),
        static_geometry_checks_passed=True,
        current_complete_scene_check_required=True)
    return components, proof


def verify_translation(prepared, direction, distance, original, obstacles):
    """Evaluate the conservative SUM against the caller's complete state.

    Dynamic collision results are never reused from a cache. The caller is
    responsible for verifying final escape from the scene and sequencing.
    """
    components, proof = translation_sweep_components(prepared['stock'], direction, distance)
    total = float(proof['serialization_uncovered_bound_mm3'])
    overlaps = []
    for index, component in enumerate(components):
        # The existing convex boundary prefilter also assumes an outside
        # witness. Convexity alone does not prove that precondition for tiny
        # prisms, so every new fallback component receives the full test.
        overlap = float(obstacles.obstruction(component, original, convex=False))
        if not math.isfinite(overlap) or overlap < 0.:
            raise CertificateError('Invalid complete-scene overlap result')
        overlaps.append(overlap)
        total += overlap
        if total > TOL:
            break
    passed = len(overlaps) == len(components) and total <= TOL
    return dict(
        passed=passed, maximum_swept_intersection_mm3=total,
        conservative_total_overlap_bound_mm3=total,
        component_overlap_sum_mm3=sum(overlaps),
        component_overlap_volumes_mm3=overlaps,
        checked_component_count=len(overlaps), total_component_count=len(components),
        duplicate_intersections_counted=True, tolerance_mm3=TOL,
        sweep_record=proof,
        partition_manifest_path=prepared.get('manifest_path'),
        physical_tool_access_certified=False,
        final_scene_exit_validation='Required from the final translated actual stock bounds.')

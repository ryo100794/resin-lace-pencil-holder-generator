#!/usr/bin/env python3
"""Local finite endpoint-cut partitions of eight immutable holder supports.

This analysis never changes printed F. It cuts two bounded .040 mm wafers
from F on the temporary-body side of the existing nominal neck planes.
All oriented boundary components are preserved. When first introducing this
definition, the caller must keep each complete F active until its cut and
separately check actual finite cutters, removed stock and extraction against
the complete current scene. Later continuations may reuse accepted operations
only when the same manifest, derived meshes and endpoint-cut validation are
explicitly bound by the prior complete proof.

No original model or whole-scene spatial index is loaded here.
"""
from pathlib import Path
import hashlib
import io
import json

import manifold3d as mf
import numpy as np

import custom_holder_supports as H

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / 'reports/holder_finite_endpoint_partition_v6.json'
SOURCE = ROOT / 'cache/custom_holder_supports'
OUT = ROOT / 'cache/holder_finite_endpoint_partition_v6'
INDICES = (206, 226, 267, 277, 384, 492, 550, 613)
WIDTH = .040
RADIAL_ALLOWANCE = .003
TOL = 1e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compose(parts):
    return mf.Manifold.compose([p for p in parts if not p.is_empty()])


def vertices(solid):
    return np.asarray(solid.to_mesh64().vert_properties[:, :3], float)


def oriented_triangle_sha(solid):
    mesh = solid.to_mesh64()
    v = np.asarray(mesh.vert_properties[:, :3], float)
    rows = []
    for face in np.asarray(mesh.tri_verts):
        points = v[face].copy()
        points[points == 0.] = 0.
        rows.append(min(tuple(np.roll(points, -k, axis=0).reshape(-1)) for k in range(3)))
    return hashlib.sha256(np.asarray(sorted(rows), dtype='<f8').reshape((-1, 9)).tobytes()).hexdigest()


def valid(solid):
    if solid.status() != mf.Error.NoError or not np.isfinite(solid.volume()):
        raise ValueError('Invalid derived solid')


def partition_checks(full, old_retained, maximum, retained, removable, kerf, waists):
    combined = compose([retained, removable, kerf])
    return dict(
        old_retained_outside_unchanged_full_mm3=H.volume(old_retained - full),
        new_retained_outside_unchanged_full_mm3=H.volume(retained - full),
        new_removable_outside_unchanged_full_mm3=H.volume(removable - full),
        actual_kerf_outside_unchanged_full_mm3=H.volume(kerf - full),
        new_retained_outside_existing_maximum_mm3=H.volume(retained - maximum),
        actual_kerf_outside_original_waists_mm3=H.volume(kerf - H.union(waists)),
        retained_removable_interior_overlap_mm3=H.volume(retained ^ removable),
        retained_kerf_interior_overlap_mm3=H.volume(retained ^ kerf),
        removable_kerf_interior_overlap_mm3=H.volume(removable ^ kerf),
        partition_uncovered_printed_mm3=H.volume(full - combined),
        partition_outside_printed_mm3=H.volume(combined - full),
        signed_partition_volume_error_mm3=abs(retained.volume() + removable.volume()
                                             + kerf.volume() - full.volume()))


def construct(row, full, old_removable, maximum, old_proxies):
    if row['support_style'] != 'pitched_carriers' or row['neck_count'] != 2:
        raise ValueError('Unsupported endpoint construction')
    radius = row['neck_diameter_mm'] / 2
    old_retained = full - old_removable
    cutters, cuts, waists, probes = [], [], [], []
    for j, (point, normal, offset) in enumerate(zip(
            [row['root_mm'], row['target_mm']], row['neck_axes'],
            row['nominal_cut_offsets_along_outward_axis_mm'])):
        point = np.asarray(point, float)
        axis = np.asarray(normal, float)
        axis /= np.linalg.norm(axis)
        half_waist = row['neck_length_mm'] / 2
        if WIDTH >= half_waist:
            raise ValueError('Body-side band does not fit within the existing waist')
        cutters.append(H.axial_profile(point, axis,
            [(offset, radius + RADIAL_ALLOWANCE), (offset + WIDTH, radius + RADIAL_ALLOWANCE)]))
        waists.append(H.axial_profile(point, axis,
            [(offset - half_waist, radius), (offset + half_waist, radius)]))
        # A small interior sample ties the classified cap to its intended
        # attachment end; this is not a substitute for the full containment.
        probes.append(H.axial_profile(point, axis, [(.025, radius / 4), (.035, radius / 4)]))
        center = point + axis * (offset + WIDTH / 2)
        cuts.append(dict(endpoint_index=j, center_mm=center.tolist(),
            axis_toward_free_body=axis.tolist(), kerf_width_mm=WIDTH,
            original_nominal_cut_center_mm=row['neck_centers_mm'][j],
            original_nominal_offset_along_axis_mm=offset,
            actual_band_offsets_along_axis_mm=[offset, offset + WIDTH],
            waist_length_mm=row['neck_length_mm'], waist_radius_mm=radius,
            cutter_radial_allowance_mm=RADIAL_ALLOWANCE,
            width_basis='axial distance along the existing pitched endpoint axis',
            minimum_axial_margin_to_waist_end_mm=half_waist - WIDTH,
            fixed_side_plane_offset_mm=float(point @ axis + offset),
            free_side_plane_offset_mm=float(point @ axis + offset + WIDTH)))
    all_cutters = H.union(cutters)
    kerf = full ^ all_cutters
    left = full - all_cutters
    components = left.decompose()
    positive = [i for i, part in enumerate(components) if part.volume() > 1e-12]
    if not positive:
        raise ValueError('Endpoint cuts left no positive stock')
    assignments, records, secondary_parent = {}, [], {}
    fixed_endpoint_components = {0: [], 1: []}
    for i in positive:
        part = components[i]
        points = vertices(part)
        endpoints = [cut['endpoint_index'] for cut in cuts
                     if float(np.max(points @ np.asarray(cut['axis_toward_free_body'])
                                     - cut['fixed_side_plane_offset_mm'])) <= 1e-8]
        outside = H.volume(part - maximum)
        if endpoints and outside > TOL:
            raise ValueError('A geometric endpoint cap exceeds its unchanged retained maximum')
        assignment = 'retained' if endpoints else 'removable'
        assignments[i] = assignment
        for endpoint in endpoints:
            fixed_endpoint_components[endpoint].append(i)
        records.append(dict(component_index=i, signed_volume_mm3=float(part.volume()),
            assignment=assignment, endpoint_candidates=endpoints,
            outside_existing_retained_maximum_mm3=outside))
    if any(len(indices) != 1 for indices in fixed_endpoint_components.values()):
        raise ValueError('Each intended fixed end must have exactly one positive cap')
    # Compose retains signed secondary shells instead of treating each cavity
    # as new positive material. None is discarded, whatever its volume.
    for i, part in enumerate(components):
        if i in assignments:
            continue
        gap, parent = min((float(part.min_gap(components[j], 100.)), j) for j in positive)
        assignments[i] = assignments[parent]
        secondary_parent[i] = parent
        records.append(dict(component_index=i, signed_volume_mm3=float(part.volume()),
            assignment=assignments[i], oriented_secondary_boundary_preserved=True,
            nearest_positive_component=parent, surface_gap_mm=gap))
    retained = compose([part for i, part in enumerate(components) if assignments[i] == 'retained'])
    removable = compose([part for i, part in enumerate(components) if assignments[i] == 'removable'])
    fragments = [compose([components[j]] + [components[i] for i, parent in secondary_parent.items()
                                            if parent == j])
                 for j in positive if assignments[j] == 'removable']
    if not fragments:
        raise ValueError('No substantial removable fragment remains')
    attachment_checks = {}
    for endpoint, indices in fixed_endpoint_components.items():
        missing = H.volume(probes[endpoint] - components[indices[0]])
        attachment_checks[f'endpoint_{endpoint}_interior_attachment_probe_outside_cap_mm3'] = missing
    checks = partition_checks(full, old_retained, maximum, retained, removable, kerf, waists)
    checks.update(attachment_checks)
    for j, cutter in enumerate(cutters):
        checks[f'cutter_{j}_new_retained_overlap_mm3'] = H.volume(cutter ^ retained)
        checks[f'cutter_{j}_new_removable_overlap_mm3'] = H.volume(cutter ^ removable)
    # Tighten only analytical convex sweep bounds. The printed source and
    # actual derived free body are untouched by this proxy clipping.
    proxies = []
    for proxy in old_proxies:
        for cut in cuts:
            proxy = proxy.trim_by_plane(cut['axis_toward_free_body'], cut['free_side_plane_offset_mm'])
        if not proxy.is_empty():
            proxies.append(proxy)
    checks['tight_proxy_uncovered_actual_removable_mm3'] = H.volume(removable - H.union(proxies))
    if max(checks.values()) > TOL:
        raise ValueError('Constructive local partition gate failed: ' + json.dumps(checks))
    for part in [retained, removable, kerf, *cutters, *fragments, *proxies]:
        valid(part)
    return dict(retained=retained, removable=removable, kerf=kerf, cutters=cutters,
        fragments=fragments, proxies=proxies, cuts=cuts, components=records,
        checks=checks, waists=waists, old_retained=old_retained,
        minimum_retained_body_gap_mm=float(retained.min_gap(removable, .1)))


def save_solid(path, solid):
    mesh = solid.to_mesh64()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, vertices=np.asarray(mesh.vert_properties[:, :3]), faces=mesh.tri_verts)
    data = buffer.getvalue()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('An immutable derived mesh path changed')
    else:
        path.write_bytes(data)
    loaded = H.load(path)[0]
    same = oriented_triangle_sha(solid) == oriented_triangle_sha(loaded)
    if not same:
        raise ValueError('Derived mesh serialization changes oriented triangles')
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path)), loaded


def load_record(support_index, report_path=REPORT):
    report = json.loads(Path(report_path).read_text())
    if report.get('passed') is not True:
        raise ValueError('Local holder endpoint partition has not passed')
    return next(row for row in report['records'] if row['support_index'] == support_index)


def main():
    metadata_path = SOURCE / 'holder_supports.json'
    rows = {row['index']: row for row in json.loads(metadata_path.read_text())['parts']}
    inputs = {str(metadata_path.relative_to(ROOT)): sha(metadata_path),
              str(Path(__file__).relative_to(ROOT)): sha(__file__),
              str(Path(H.__file__).relative_to(ROOT)): sha(H.__file__)}
    for index in INDICES:
        row = rows[index]
        filenames = [row[k] for k in ['full_mesh_file', 'removable_mesh_file', 'retained_mesh_file']]
        filenames += row.get('generated_convex_sweep_meshes', [])
        for name in filenames:
            path = SOURCE / name
            inputs[str(path.relative_to(ROOT))] = sha(path)
    run_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    output = OUT / run_hash[:16]
    output.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for index in INDICES:
        row = rows[index]
        try:
            full = H.load(SOURCE / row['full_mesh_file'])[0]
            old_removable = H.load(SOURCE / row['removable_mesh_file'])[0]
            maximum = H.load(SOURCE / row['retained_mesh_file'])[0]
            proxies = [H.load(SOURCE / name)[0] for name in row['generated_convex_sweep_meshes']]
            result = construct(row, full, old_removable, maximum, proxies)
            paths, loaded = {}, {}
            for name in ['retained', 'removable', 'kerf']:
                paths[name], loaded[name] = save_solid(output / f'H{index:04d}_{name}.npz', result[name])
            for name in ['cutters', 'fragments', 'proxies']:
                saved = [save_solid(output / f'H{index:04d}_{name}_{j:02d}.npz', part)
                         for j, part in enumerate(result[name])]
                paths[name] = [entry for entry, part in saved]
                loaded[name] = [part for entry, part in saved]
            checks = partition_checks(full, result['old_retained'], maximum,
                loaded['retained'], loaded['removable'], loaded['kerf'], result['waists'])
            checks['serialized_fragments_uncovered_removable_mm3'] = H.volume(
                loaded['removable'] - compose(loaded['fragments']))
            checks['serialized_fragments_outside_removable_mm3'] = H.volume(
                compose(loaded['fragments']) - loaded['removable'])
            checks['serialized_proxy_uncovered_removable_mm3'] = H.volume(
                loaded['removable'] - H.union(loaded['proxies']))
            checks['serialized_actual_kerf_outside_cutters_mm3'] = H.volume(
                loaded['kerf'] - H.union(loaded['cutters']))
            checks['serialized_cutter_stock_uncovered_by_kerf_mm3'] = H.volume(
                (full ^ H.union(loaded['cutters'])) - loaded['kerf'])
            if max(checks.values()) > TOL:
                raise ValueError('Serialized local partition gate failed: ' + json.dumps(checks))
            record = dict(id=f'H{index:04d}', support_index=index, paths=paths, cuts=result['cuts'],
                checks=checks, construction_checks=result['checks'], passed=True,
                component_records=result['components'],
                minimum_retained_body_gap_mm=float(loaded['retained'].min_gap(loaded['removable'], .1)),
                serialized_oriented_triangles_identical=True,
                all_oriented_components_preserved=True,
                source_full_mesh_file=str((SOURCE / row['full_mesh_file']).relative_to(ROOT)),
                source_full_sha256=sha(SOURCE / row['full_mesh_file']),
                source_removable_mesh_file=str((SOURCE / row['removable_mesh_file']).relative_to(ROOT)),
                source_removable_sha256=sha(SOURCE / row['removable_mesh_file']),
                source_maximum_mesh_file=str((SOURCE / row['retained_mesh_file']).relative_to(ROOT)),
                source_maximum_sha256=sha(SOURCE / row['retained_mesh_file']),
                full_scene_cut_and_route_validation_required=True)
            records.append(record)
            print(json.dumps(dict(id=record['id'], local_partition_passed=True,
                fragments=len(paths['fragments']), maximum_numeric_error_mm3=max(checks.values()),
                minimum_retained_body_gap_mm=record['minimum_retained_body_gap_mm'])), flush=True)
        except Exception as error:
            failure = dict(id=f'H{index:04d}', error=str(error))
            failures.append(failure)
            print(json.dumps(failure), flush=True)
    changed_inputs = [path for path, digest in inputs.items() if sha(ROOT / path) != digest]
    if changed_inputs:
        failures.append(dict(error='Frozen input changed during local generation', paths=changed_inputs))
    report = dict(schema='holder_finite_endpoint_partition_v1', passed=not failures,
        scope='Local finite-cut partition only. No original/current-scene cutter or route test performed.',
        volume_tolerance_mm3=TOL, source_sha256=sha(__file__), inputs=inputs,
        records=records, failures=failures, printed_geometry_modified=False,
        prior_state_requirement='When first introducing these definitions, every affected ID must '
            'still be active in the reused certified prefix; its complete source F stays occupied '
            'until the explicit endpoint cut. A later continuation may reuse an already accepted '
            'affected operation only if the prior complete proof explicitly binds this same manifest, '
            'the same derived mesh hashes and its passed endpoint-cut validation. It must not '
            'retroactively replace the cut definition of an operation proved under older inputs.',
        physical_band_definition='Two .040 mm axial bands toward the temporary body from existing '
            'nominal planes, wholly within the original .140 mm waists; bounded radius has .003 mm allowance.',
        no_tool_access_strength_or_hand_accuracy_certification=True)
    temporary = REPORT.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(REPORT)
    print(json.dumps(dict(report=str(REPORT.relative_to(ROOT)), sha256=sha(REPORT),
        passed=report['passed'], records=len(records), failures=len(failures))), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

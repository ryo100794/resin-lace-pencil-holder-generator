#!/usr/bin/env python3
"""Make a V5 five-layer sphere coupon without modifying its mechanism.

The authoritative cached sphere is kept at its original size and orientation.
A small base touches only the fixed outer mounting collar. Local +Y becomes
export +Z, preserving the holder's bottom-first build order. The exported
STL and core 3MF contain exactly the same float32 triangle coordinates.

This is a geometry and file-integrity check, not a resin-strength or slicer
certificate. The fixed outer ornaments may still require external supports.

Example, from the project directory::

    python source/build_test_coupon.py

Only this script's model outputs and validation report are written. The
sphere cache and all other source files are read-only inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback

import manifold3d as mf
import numpy as np
import trimesh

import audit_motion
import export_models
from embroidery_technique import point_contact_vertices
from sphere_geometry import _collar


ROOT = Path(__file__).resolve().parent.parent
EPSILON_MM = 0.00003
INTERSECTION_EPSILON_MM3 = 1.0e-9
MINIMUM_ADDED_FIXTURE_CLEARANCE_MM = 0.15


def log(message):
    print(message, flush=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def mesh_arrays(solid):
    mesh = solid.to_mesh64()
    return (np.asarray(mesh.vert_properties[:, :3], dtype=np.float64),
            np.asarray(mesh.tri_verts, dtype=np.uint32))


def overlap_volume(a, b):
    intersection = a ^ b
    require(intersection.status() == mf.Error.NoError, 'Intersection Boolean failed')
    return 0.0 if intersection.is_empty() else max(0.0, float(intersection.volume()))


def box(low, high):
    low, high = np.asarray(low, float), np.asarray(high, float)
    return mf.Manifold.cube(tuple(high-low)).translate(low)


class Inputs:
    """Load each input once and keep an auditable immutable-input manifest."""

    def __init__(self, cache):
        self.cache = Path(cache).resolve()
        self.meshes = {}
        self.hashes = {}

    def remember(self, path):
        path = Path(path).resolve()
        digest = export_models.file_sha256(path)
        if path in self.hashes:
            require(self.hashes[path] == digest, f'Input changed while reading: {path}')
        self.hashes[path] = digest
        return path

    def mesh(self, filename):
        path = (self.cache/filename).resolve()
        require(path.is_relative_to(self.cache), 'Cache mesh path escapes its directory')
        if path not in self.meshes:
            self.remember(path)
            self.meshes[path] = audit_motion.Mesh(path)
        return self.meshes[path]

    def verify_unchanged(self):
        for path, digest in self.hashes.items():
            require(export_models.file_sha256(path) == digest,
                    f'Input changed during coupon construction: {path}')

    def manifest(self):
        return [dict(path=str(path), sha256=digest)
                for path, digest in sorted(self.hashes.items(), key=lambda row: str(row[0]))]


def make_fixture(radius):
    """Base, stem and concave collar seat in the sphere's local coordinates."""
    center_height = radius + 5.0
    base = box((-16., -center_height, -10.),
               (16., -center_height+2., 10.))
    stem = box((-4., -center_height+1.85, -1.8),
               (4., -radius-.90, 1.8))
    seat = box((-4., -radius-1.05, -.36),
               (4., -radius+.25, .36))
    # Circumscribe the analytical exclusion circle. The flat polygon sides
    # therefore never enter the R+0.20 mm collar-only radial keep-out zone.
    segments = 360
    exclusion_radius = (radius+.20)/math.cos(math.pi/segments)
    cutter = mf.Manifold.cylinder(4., exclusion_radius, exclusion_radius,
                                 segments).translate((0., 0., -2.))
    seat = seat-cutter
    fixture = mf.Manifold.batch_boolean([base, stem, seat], mf.OpType.Add)
    require(fixture.status() == mf.Error.NoError and len(fixture.decompose()) == 1,
            'The added base is not one connected valid solid')
    # Proper rotation, determinant +1. Local +Z/front becomes world -Y;
    # local +Y/up becomes world +Z. No reflection or winding reversal.
    transform = np.array([[1., 0., 0., 0.],
                          [0., 0., -1., 0.],
                          [0., 1., 0., center_height]])
    dimensions = dict(base_width_mm=32., base_depth_mm=20., base_thickness_mm=2.,
        stem_width_mm=8., stem_depth_mm=3.6, seat_width_mm=8.,
        seat_thickness_mm=.72, collar_exclusion_radius_mm=radius+.20,
        sphere_center_height_mm=center_height,
        base_stem_and_stem_seat_overlap_mm=.15)
    return fixture, transform, dimensions


def minimum_surface_radius(solid):
    vertices, faces = mesh_arrays(solid)
    result = math.inf
    for start in range(0, len(faces), 32768):
        triangles = vertices[faces[start:start+32768]]
        nearest = trimesh.triangles.closest_point(triangles, np.zeros((len(triangles), 3)))
        result = min(result, float(np.linalg.norm(nearest, axis=1).min()))
    return result


def fixture_motion_check(fixture, meta, layers):
    """Bound every rotated moving layer, including accumulated journal play.

    Actual bore and pin meshes are measured by the existing motion auditor.
    The radius of any point of a rotated layer is unchanged; center offsets
    from all ancestor joints accumulate by the triangle inequality. This
    verifies only clearance from the newly added fixture. The separate full
    module audit remains responsible for clearance between spherical layers.
    """
    bearings = []
    for bearing in meta['bearings']:
        opposite = [item for item in meta['bearings']
                    if item['host_index'] == bearing['host_index']
                    and item['moving_index'] == bearing['moving_index']
                    and item['side'] == -bearing['side']]
        require(len(opposite) == 1, 'An opposed bearing is missing')
        measured = audit_motion.bearing_measurements(
            bearing, opposite[0], layers, meta['axis_unit_vector'])
        require(measured.get('passed'), 'Actual bearing measurement failed: '+str(measured))
        bearings.append(measured)
    cumulative = [0.]
    per_interface = []
    for index in range(len(layers)-1):
        pair = [row for row in bearings if row['host_index'] == index]
        require(len(pair) == 2, 'Each interface must have two opposed measured bearings')
        bounds = []
        for row in pair:
            radial = max(0., row['conservative_maximum_radial_play_mm'])
            mouth = row['actual_mouth_axial_mm']-EPSILON_MM
            require(mouth > 0., 'Invalid bearing mouth position')
            theta = math.atan2(radial, mouth)
            axial = (max(0., row['axial_endplay_mm'])+EPSILON_MM
                     +(max(0., row['actual_pin_tip_axial_mm'])+EPSILON_MM)
                     *(1-math.cos(theta)))
            bounds.append(max(row['conservative_center_displacement_per_interface_mm'],
                              math.hypot(axial, radial)))
        per_interface.append(dict(host_index=index, moving_index=index+1,
                                  center_play_bound_mm=max(bounds)))
        cumulative.append(cumulative[-1]+max(bounds))
    minimum_fixture_radius = minimum_surface_radius(fixture)
    rows = []
    for index, layer in enumerate(layers[1:], start=1):
        radius = float(np.linalg.norm(layer.vertices, axis=1).max())
        gap = minimum_fixture_radius-radius-cumulative[index]-EPSILON_MM
        row = dict(layer_index=index, centered_maximum_radius_mm=radius,
            cumulative_center_play_bound_mm=cumulative[index],
            all_angle_fixture_clearance_bound_mm=gap,
            passed=gap >= MINIMUM_ADDED_FIXTURE_CLEARANCE_MM)
        rows.append(row)
        require(row['passed'], f'Fixture enters moving-layer envelope: {row}')
    return dict(method='Actual triangle radial enclosure and measured accumulated journal play',
        motion_scope='All centered rotations, conservative joint translation and tilt center displacement',
        stump_scope='Actual released_with_stumps_layer_meshes supplied by this module',
        minimum_fixture_surface_radius_mm=minimum_fixture_radius,
        export_rounding_allowance_mm=EPSILON_MM, measured_bearings=bearings,
        interface_play=per_interface, moving_layers=rows,
        minimum_clearance_bound_mm=min(row['all_angle_fixture_clearance_bound_mm'] for row in rows),
        passed=True)


def separated(fixture, other, label):
    volume = overlap_volume(fixture, other)
    gap = float(fixture.min_gap(other, 2.0))
    passed = volume <= INTERSECTION_EPSILON_MM3 and gap-EPSILON_MM >= MINIMUM_ADDED_FIXTURE_CLEARANCE_MM
    row = dict(label=label, intersection_volume_mm3=volume,
        mesh_gap_mm=gap, gap_search_limit_mm=2.0,
        conservative_gap_after_export_mm=gap-EPSILON_MM, passed=passed)
    require(passed, f'Added fixture obstructs protected geometry: {row}')
    return row


def canonical_float32_mesh(solid):
    """Quantize once, weld identical coordinates, remove only collapsed faces.

    No added material, decimation, smoothing, approximate welding, or normal
    repair is allowed here. Any remaining invalid surface fails closed.
    """
    vertices64, faces = mesh_arrays(solid)
    quantized = np.asarray(vertices64, dtype=np.float32)
    error = float(np.linalg.norm(quantized.astype(float)-vertices64, axis=1).max())
    vertices, inverse = np.unique(quantized, axis=0, return_inverse=True)
    mapped_faces = inverse[faces].astype(np.uint32)
    keep = ((mapped_faces[:, 0] != mapped_faces[:, 1])
            & (mapped_faces[:, 1] != mapped_faces[:, 2])
            & (mapped_faces[:, 2] != mapped_faces[:, 0]))
    collapsed = int((~keep).sum())
    mapped_faces = mapped_faces[keep]
    used, inverse = np.unique(mapped_faces, return_inverse=True)
    vertices = np.ascontiguousarray(vertices[used], dtype=np.float32)
    faces = np.ascontiguousarray(inverse.reshape((-1, 3)), dtype=np.uint32)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    triangle_points = mesh.triangles
    double_areas = np.linalg.norm(np.cross(triangle_points[:, 1]-triangle_points[:, 0],
                                         triangle_points[:, 2]-triangle_points[:, 0]), axis=1)
    zero_area = int(np.count_nonzero(double_areas == 0.))
    near_zero = int(np.count_nonzero(double_areas*.5 <= 1.0e-14))
    incidence = np.bincount(mesh.edges_unique_inverse)
    bad_edges = int(np.count_nonzero(incidence != 2))
    require(np.isfinite(vertices).all(), 'Export vertices are not finite')
    require(zero_area == 0 and near_zero == 0, 'Float32 export has zero or near-zero area faces')
    require(bad_edges == 0 and mesh.is_watertight and mesh.is_winding_consistent,
            'Float32 export has an invalid edge or winding')
    point_contacts = point_contact_vertices(mesh)
    require(len(point_contacts) == 0, 'Float32 export contains a nonmanifold vertex')
    exact = mf.Manifold(mf.Mesh64(vertices.astype(np.float64), faces))
    require(exact.status() == mf.Error.NoError, 'Float32 surface is not accepted as a manifold')
    components = exact.decompose()
    volumes = [float(piece.volume()) for piece in components]
    require(len(components) == 1 and volumes[0] > 0., 'Export must be one positive connected solid')
    require(error < EPSILON_MM/2., 'Export position error exceeds the clearance allowance')
    report = dict(vertex_count=len(vertices), triangle_count=len(faces),
        source_triangle_count=len(solid.to_mesh64().tri_verts),
        identical_coordinate_welding=True, collapsed_faces_removed=collapsed,
        no_added_material_or_approximate_repair=True,
        maximum_vertex_rounding_error_mm=error,
        exact_zero_area_triangles=zero_area, near_zero_area_triangles=near_zero,
        near_zero_area_threshold_mm2=1.0e-14, bad_edge_incidence_count=bad_edges,
        nonmanifold_vertex_count=len(point_contacts), components=len(components),
        component_volumes_mm3=volumes, volume_mm3=float(mesh.volume),
        watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
        bounds_mm=mesh.bounds.tolist(), dimensions_mm=np.ptp(vertices.astype(float), axis=0).tolist(),
        passed=True)
    return vertices, faces, report


def build(args, report):
    started = time.monotonic()
    inputs = Inputs(args.cache)
    metadata_path = inputs.remember(args.cache/args.metadata)
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    require(meta.get('version') == 5, 'Coupon requires regenerated V5 metadata, not mixed V3 cache')
    require(meta.get('layer_count') == 5 and abs(meta['outer_nominal_radius_mm']-13.) < 1e-8,
            'Coupon is specifically the final 26 mm, five-layer test sphere')
    require(np.allclose(meta['build_direction'], [0., 1., 0.]), 'Unexpected sphere build direction')
    for source in [Path(__file__), Path(export_models.__file__), Path(audit_motion.__file__),
                   ROOT/'source'/'sphere_geometry.py', ROOT/'source'/'embroidery_technique.py']:
        inputs.remember(source)
    supported = inputs.mesh(meta['supported_mesh_file'])
    fixed = inputs.mesh(meta['free_layer_meshes'][0])
    source_topology = supported.topology(1)
    require(source_topology['passed'], 'Supported source sphere has invalid topology')
    require(len(supported.faces) == meta['triangle_count'], 'Cached sphere and metadata disagree on triangles')
    require(abs(supported.manifold.volume()-meta['volume_mm3']) < 1e-5,
            'Cached sphere and metadata disagree on volume')
    require(np.allclose(np.r_[supported.vertices.min(axis=0), supported.vertices.max(axis=0)],
                        meta['bounds_mm'], atol=1e-8, rtol=0.), 'Cached sphere bounds do not match metadata')
    fixture, transform, dimensions = make_fixture(meta['outer_nominal_radius_mm'])
    collar = _collar(meta['outer_nominal_radius_mm'])
    collar_contact = overlap_volume(fixture, collar)
    fixed_contact = overlap_volume(fixture, fixed.manifold)
    noncollar_contact = overlap_volume(fixture, fixed.manifold-collar)
    require(collar_contact > 1.0 and fixed_contact > 1.0, 'Collar seat has insufficient finite attachment')
    require(noncollar_contact <= INTERSECTION_EPSILON_MM3,
            'Fixture touches the fixed sphere outside its mounting collar')
    require(abs(collar_contact-fixed_contact) < 1e-8, 'Diagnostic collar differs from the actual contact surface')
    report['source_module'] = dict(name=meta['name'], metadata_file=str(metadata_path),
        metadata_sha256=inputs.hashes[metadata_path], version=meta['version'],
        shell_radii_mm=meta['layer_radii_mm'], support_count=meta['wire_count'],
        breakaway_neck_count=meta['neck_count'], topology=source_topology)
    report['fixture'] = dict(**dimensions, local_to_export_transform=transform.tolist(),
        transform_determinant=float(np.linalg.det(transform[:, :3])),
        local_build_direction=[0., 1., 0.], export_build_direction=[0., 0., 1.],
        export_front_direction=[0., -1., 0.],
        orientation='Bottom-first; model base is nearest the build plate even in a hanging bottom-up vat printer',
        collar_attachment_volume_mm3=collar_contact,
        actual_fixed_outer_layer_attachment_volume_mm3=fixed_contact,
        noncollar_fixed_material_intersection_mm3=noncollar_contact)
    log('Collar-only attachment verified; checking all moving layers and service routes')
    released = [inputs.mesh(name) for name in meta['released_with_stumps_layer_meshes']]
    require(len(released) == meta['layer_count'], 'Missing released layer meshes')
    report['moving_layer_static_checks'] = [
        separated(fixture, layer.manifold, f'moving_layer_{index}')
        for index, layer in enumerate(released[1:], start=1)]
    report['moving_layer_all_angle_enclosure'] = fixture_motion_check(fixture, meta, released)
    # Adding the fixture to the outer released layer must retain the five
    # independently separated solids, including the allowed fracture stumps.
    released_coupon = mf.Manifold.batch_boolean([fixture]+[layer.manifold for layer in released], mf.OpType.Add)
    released_parts = released_coupon.decompose()
    require(len(released_parts) == 5 and all(part.volume() > 0. for part in released_parts),
            'Fixture joined a moving released layer or split a fixed one')
    report['released_coupon_components'] = len(released_parts)
    wires, tool_paths, extractions = [], [], []
    for wire in sorted(meta['release_wires'], key=lambda row: row['sequence_index']):
        label = f"wire_{wire['wire_index']}"
        wires.append(separated(fixture, inputs.mesh(wire['mesh_file']).manifold, label))
        removable = inputs.mesh(wire['removable_mesh_file'])
        direction = np.asarray(wire['extraction_vector'], float)
        direction /= np.linalg.norm(direction)
        displacement = direction*float(wire['extraction_distance_mm'])
        sweep = mf.Manifold.hull_points(np.vstack([removable.vertices,
                                                   removable.vertices+displacement]))
        row = separated(fixture, sweep, label+'_continuous_extraction')
        row.update(sequence_index=wire['sequence_index'], kind=wire['kind'],
                   extraction_vector_local=direction.tolist(),
                   distance_mm=float(wire['extraction_distance_mm']),
                   method='Convex hull enclosing every continuously translated actual removable-piece triangle')
        extractions.append(row)
        for endpoint_index, endpoint in enumerate(wire['endpoints']):
            corridor = inputs.mesh(endpoint['tool_corridor_mesh_file']).manifold
            # Include the last deliberate contact segment up to the actual
            # neck center. The fixture must not obstruct this segment either.
            extension = audit_motion.cylinder_between(endpoint['tool_tip_end'],
                endpoint.get('tool_contact_point', endpoint['neck_center']),
                float(endpoint['tool_tip_diameter_mm'])/2.)
            corridor = corridor+extension
            row = separated(fixture, corridor, f'{label}_endpoint_{endpoint_index}_tool')
            row.update(endpoint_index=endpoint_index, wire_index=wire['wire_index'],
                shaft_diameter_mm=endpoint['tool_diameter_mm'],
                tip_diameter_mm=endpoint['tool_tip_diameter_mm'],
                entry_local_mm=endpoint['tool_entry'], target_local_mm=endpoint['neck_center'])
            tool_paths.append(row)
    require(len(wires) == meta['wire_count'] and len(tool_paths) == meta['neck_count'],
            'Incomplete support/tool-path verification')
    report['temporary_support_checks'] = wires
    report['neck_access_checks'] = tool_paths
    report['continuous_extraction_checks'] = extractions
    report['release_sequence'] = meta['release_sequence']
    report['release_order'] = [dict(wire_index=wire['wire_index'], sequence_index=wire['sequence_index'],
                                  kind=wire['kind'], host_index=wire['host_index'])
                               for wire in sorted(meta['release_wires'], key=lambda row: row['sequence_index'])]
    assembly = supported.manifold+fixture
    require(assembly.status() == mf.Error.NoError and len(assembly.decompose()) == 1,
            'Supported coupon is not one valid connected solid')
    lost_volume = max(0., float((supported.manifold-assembly).volume()))
    require(lost_volume <= INTERSECTION_EPSILON_MM3, 'Coupon construction removed original module material')
    report['source_module_material_removed_mm3'] = lost_volume
    vertices, faces, report['export_mesh_validation'] = canonical_float32_mesh(assembly.transform(transform))
    triangle_hash = export_models.triangle_sha256(vertices, faces)
    inputs.verify_unchanged()
    args.out.mkdir(parents=True, exist_ok=True)
    stl = args.out/(args.stem+'.stl')
    three_mf = args.out/(args.stem+'.3mf')
    temporary_stl = stl.with_suffix('.stl.part')
    temporary_3mf = three_mf.with_suffix('.3mf.part')
    log(f'Writing and rereading {len(faces):,} full-resolution coupon triangles')
    export_models.write_stl(temporary_stl, vertices, faces)
    with temporary_stl.open('r+b') as stream:
        stream.write(b'V5 five-layer sphere test coupon; mm; bottom-first; temporary ties included'.ljust(80, b' '))
    export_models.write_3mf(temporary_3mf, vertices, faces,
                           'Ivory sphere V5 five-layer test coupon; millimetres; bottom-first')
    stl_validation = export_models.verify_stl(temporary_stl, vertices, faces, triangle_hash)
    three_mf_validation = export_models.verify_3mf(temporary_3mf, vertices, faces, triangle_hash)
    inputs.verify_unchanged()
    temporary_stl.replace(stl)
    temporary_3mf.replace(three_mf)
    report['exports'] = []
    for path, validation in [(stl, stl_validation), (three_mf, three_mf_validation)]:
        report['exports'].append(dict(path=str(path.resolve()), bytes=path.stat().st_size,
            sha256=export_models.file_sha256(path), roundtrip_validation=validation))
    report['input_manifest'] = inputs.manifest()
    report['inputs_unchanged_during_run'] = True
    report['elapsed_seconds'] = time.monotonic()-started
    report['passed'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'cache'/'spheres')
    parser.add_argument('--metadata', default='sphere_5layers.json')
    parser.add_argument('--out', type=Path, default=ROOT/'models')
    parser.add_argument('--stem', default='ivory_sphere_v5_test_coupon')
    parser.add_argument('--report', type=Path, default=ROOT/'reports'/'test_coupon_validation.json')
    args = parser.parse_args()
    args.cache, args.out, args.report = args.cache.resolve(), args.out.resolve(), args.report.resolve()
    require(Path(args.stem).name == args.stem, 'Stem must be a filename, without directories')
    report = dict(schema_version=1, design_version=5, units='mm',
        created_utc=datetime.now(timezone.utc).isoformat(), passed=False,
        physical_print_tested=False,
        scope='Addition of a collar-only fixture to the unchanged cached V5 five-layer sphere; exact STL/3MF roundtrip',
        limitations=[
            'Printer, resin, exposure, lift/peel load and green strength are not specified or physically verified.',
            'The fixed outer sphere and its decorative overhangs may require additional external slicer supports. Inspect all build layers.',
            'Do not place added slicer supports in the rotating gaps, bearing cavities, front service windows, or declared removal routes.',
            'This one-sphere coupon tests local manufacture and release; it does not reproduce the complete holder\'s peel area or wall stiffness.',
            'Internal layer-to-layer rotation, print-start connectivity and material performance remain separate checks; fixture checks do not replace them.',
            'The base and collar seat are permanent. Only the sphere\'s declared temporary supports are removed in its recorded sequence.',
            'A hanging bottom-up printer does not reverse model-coordinate build order: base first, then sphere bottom, then sphere top.',
        ])
    try:
        build(args, report)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
        write_json(args.report, report)
        raise
    write_json(args.report, report)
    log(json.dumps(dict(passed=True, report=str(args.report),
        outputs=[row['path'] for row in report['exports']],
        triangles=report['export_mesh_validation']['triangle_count'],
        dimensions_mm=report['export_mesh_validation']['dimensions_mm'],
        minimum_added_fixture_motion_clearance_mm=report['moving_layer_all_angle_enclosure']['minimum_clearance_bound_mm']),
        ensure_ascii=False))


if __name__ == '__main__':
    main()

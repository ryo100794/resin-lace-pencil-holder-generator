#!/usr/bin/env python3
"""Audit geometric reinforcement, not mechanical strength or print success.

The regular floral-opening certificate uses enclosing conical caps and the
actual inradius of the sphere tessellation used by the source. It explicitly
excludes service slots, journals, wash holes, local trims, and relief details.
Actual cached shell samples, bearing sections, roots, and support waists are
measured separately. All lengths are mm and all areas are mm2.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
from pathlib import Path
import time

import manifold3d as mf
import numpy as np
import trimesh

from audit_motion import Mesh, frame, mesh_from_manifold, sha256, transformed, unit


EXPORT_ALLOWANCE_MM = .00003


class Inputs:
    def __init__(self, root):
        self.root = root.resolve()
        self.hashes = {}
        self.meshes = {}

    def remember(self, path):
        path = Path(path).resolve()
        digest = sha256(path)
        previous = self.hashes.setdefault(path, digest)
        if previous != digest:
            raise RuntimeError(f'Input changed during audit: {path}')
        return path

    def text(self, path):
        return self.remember(path).read_text()

    def json(self, path):
        return json.loads(self.text(path))

    def mesh(self, path):
        path = self.remember(path)
        if path not in self.meshes:
            self.meshes[path] = Mesh(path)
        return self.meshes[path]

    def relative(self, path):
        try:
            return str(Path(path).resolve().relative_to(self.root))
        except ValueError:
            return str(Path(path).resolve())

    def final_fingerprints(self):
        changed = [self.relative(p) for p, digest in self.hashes.items()
                   if not p.exists() or sha256(p) != digest]
        return {self.relative(p): digest for p, digest in self.hashes.items()}, changed


def source_configuration(root, inputs):
    path = root/'source/sphere_geometry.py'
    text = inputs.text(path)
    tree = ast.parse(text)
    constants = {}
    segment_expression = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        if isinstance(node, ast.FunctionDef) and node.name == '_build_layer':
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == 'seg' for t in statement.targets):
                    segment_expression = statement.value
    if segment_expression is None:
        raise RuntimeError('Cannot identify the source sphere tessellation expression')
    allowed = (ast.IfExp, ast.Compare, ast.Name, ast.Load, ast.Constant,
               ast.GtE, ast.Gt, ast.LtE, ast.Lt, ast.Eq, ast.NotEq)
    if any(not isinstance(node, allowed) for node in ast.walk(segment_expression)):
        raise RuntimeError('Unsupported sphere tessellation expression; audit it explicitly')
    if any(isinstance(node, ast.Name) and node.id != 'radius'
           for node in ast.walk(segment_expression)):
        raise RuntimeError('Unexpected dependency in sphere tessellation expression')
    code = compile(ast.Expression(segment_expression), str(path), 'eval')
    for helper_name in ('service_edge_ribs.py', 'printable_shell_starts.py'):
        helper = root/'source'/helper_name
        if helper.exists():
            inputs.remember(helper)
    return dict(constants=constants, source_path=path,
                segments=lambda radius: int(eval(code, {'__builtins__': {}}, {'radius': radius})))


_UNIT_FACTORS = {}


def tessellation_inradius(segments):
    if segments not in _UNIT_FACTORS:
        # A small reference primitive, not a regenerated design. The two
        # source skin surfaces are homothetic copies of this tessellation.
        vertices, faces = mesh_from_manifold(mf.Manifold.sphere(1., segments))
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        distances = np.abs(np.einsum('ij,ij->i', normals, triangles[:, 0]))/np.linalg.norm(normals, axis=1)
        _UNIT_FACTORS[segments] = float(distances.min())
    return _UNIT_FACTORS[segments]


def aperture_centres(rotation_degrees=0.):
    ico = trimesh.creation.icosphere(subdivisions=0, radius=1.)
    p = np.asarray(ico.vertices, float)
    a, b = unit(p[4]), np.array([0., 0., 1.])
    v = np.cross(a, b)
    c = float(a@b)
    k = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    rot = np.eye(3)+k+k@k/(1+c)
    phi = math.radians(rotation_degrees)
    rz = np.array([[math.cos(phi), -math.sin(phi), 0.],
                   [math.sin(phi), math.cos(phi), 0.], [0., 0., 1.]])
    p = p@(rz@rot).T
    small = p[np.asarray(ico.faces)].sum(axis=1)
    small /= np.linalg.norm(small, axis=1)[:, None]
    return p, small


def cone_ligaments(spec, inner_inradius):
    large, small = aperture_centres(float(spec['rotation_degrees']))
    groups = [('large', n, math.radians(float(spec['port_angle_degrees'])
               +abs(float(spec['port_amplitude_degrees'])))) for n in large]
    if spec['small_ports']:
        groups.extend(('small', n, math.radians(float(spec['small_port_angle_degrees'])
                       +abs(float(spec['small_port_amplitude_degrees'])))) for n in small)
    closest = {}
    for i, j in itertools.combinations(range(len(groups)), 2):
        ka, a, aa = groups[i]
        kb, b, ab = groups[j]
        separation = math.acos(float(np.clip(a@b, -1., 1.)))
        angle_gap = separation-aa-ab
        lower = 2*inner_inradius*math.sin(max(0., angle_gap)/2)
        key = '-'.join(sorted((ka, kb)))
        row = dict(aperture_indices=[i, j], centre_separation_degrees=math.degrees(separation),
                   enclosing_half_angles_degrees=[math.degrees(aa), math.degrees(ab)],
                   angular_separation_lower_bound_degrees=math.degrees(angle_gap),
                   inner_surface_distance_lower_bound_mm=lower,
                   distance_after_export_allowance_mm=lower-EXPORT_ALLOWANCE_MM)
        if key not in closest or lower < closest[key]['inner_surface_distance_lower_bound_mm']:
            closest[key] = row
    return dict(pair_count=len(groups)*(len(groups)-1)//2, closest_by_pair_type=closest,
                minimum_distance_after_export_allowance_mm=min(
                    row['distance_after_export_allowance_mm'] for row in closest.values()),
                proof='Each actual floral cone lies inside its declared maximum-angle cap. For cap-axis separation gamma, every pair of cut-boundary directions is separated by at least gamma-alpha-beta. At radius >= r, their Euclidean distance is >= 2*r*sin((gamma-alpha-beta)/2). This is a conservative bound, not the exact minimum of the petal boundaries.',
                excluded_features=['service slots', 'bearing bores and wash holes',
                                   'local flattening cuts', 'raised relief tips and their attachment'])


def shell_samples(mesh, spec, factor):
    radius, thickness = float(spec['radius_mm']), float(spec['thickness_mm'])
    inner = radius-thickness
    large, small = aperture_centres(float(spec['rotation_degrees']))
    count = 4096
    i = np.arange(count)
    y = 1-2*(i+.5)/count
    phi = i*math.pi*(3-math.sqrt(5))
    rr = np.sqrt(1-y*y)
    dirs = np.column_stack((rr*np.cos(phi), y, rr*np.sin(phi)))
    # Back hemisphere avoids all forward service cuts/ribs; keep clear of
    # the journal axes, equatorial collar, and bottom receiving pads.
    keep = (dirs[:, 2] < -.25) & (np.abs(dirs[:, 0]) < .70) & (np.abs(dirs[:, 1]) < .80)
    angle = float(spec['port_angle_degrees'])+abs(float(spec['port_amplitude_degrees']))+.25
    keep &= (dirs@large.T < math.cos(math.radians(angle))).all(axis=1)
    if spec['small_ports']:
        angle = float(spec['small_port_angle_degrees'])+abs(float(spec['small_port_amplitude_degrees']))+.25
        keep &= (dirs@small.T < math.cos(math.radians(angle))).all(axis=1)
    candidates = dirs[keep]
    rows = []
    if len(candidates):
        candidates = candidates[np.unique(np.linspace(0, len(candidates)-1, min(240, len(candidates)), dtype=int))]
    for direction in candidates:
        positions = sorted(float(np.asarray(hit.position)@direction)
                           for hit in mesh.manifold.ray_cast([0., 0., 0.], direction*(radius+1.)))
        distinct = []
        for value in positions:
            if not distinct or value-distinct[-1] > 1e-7:
                distinct.append(value)
        if len(distinct) != 2:
            continue
        a, b = distinct
        if not (inner*factor-1e-5 <= a <= inner+1e-5 and radius*factor-1e-5 <= b <= radius+1e-5):
            continue
        rows.append(dict(direction=direction.tolist(), inner_hit_radius_mm=a,
                         outer_hit_radius_mm=b, radial_material_thickness_mm=b-a))
        if len(rows) == 16:
            break
    minimum = min((row['radial_material_thickness_mm'] for row in rows), default=None)
    maximum = max((row['radial_material_thickness_mm'] for row in rows), default=None)
    return dict(sample_count=len(rows), samples=rows, minimum_sampled_radial_thickness_mm=minimum,
                maximum_sampled_radial_thickness_mm=maximum,
                passed=bool(len(rows) >= 8 and minimum >= thickness*factor-EXPORT_ALLOWANCE_MM
                            and maximum <= thickness+EXPORT_ALLOWANCE_MM),
                scope='Samples on retained ordinary skin, away from journals, service cuts, pads and raised relief. Sample minima are not global minima for every feature.')


def contour_metrics(p):
    p = np.asarray(p, float)
    q = np.roll(p, -1, axis=0)
    d = q-p
    u = np.clip(-np.einsum('ij,ij->i', p, d)/np.maximum(np.einsum('ij,ij->i', d, d), 1e-30), 0., 1.)
    near = p+u[:, None]*d
    return dict(minimum_radius_mm=float(np.linalg.norm(near, axis=1).min()),
                maximum_radius_mm=float(np.linalg.norm(p, axis=1).max()),
                area_mm2=.5*abs(float(np.sum(p[:, 0]*q[:, 1]-p[:, 1]*q[:, 0]))))


def section(mesh, axis, coordinate):
    return [np.asarray(p) for p in transformed(mesh.manifold, frame(axis).T).slice(float(coordinate)).to_polygons()
            if len(p) >= 3]


def bearing_feature(meta, bearing, role):
    field = {'socket': 'housing_mesh_file', 'pivot': 'pin_mesh_file'}[role]
    if bearing.get(field):
        return bearing[field]
    return next(item['mesh_file'] for item in meta['bearing_feature_meshes']
                if item['bearing_index'] == bearing['bearing_index'] and item['role'] == role)


def measure_bearing(meta, bearing, cache, inputs):
    axis = unit(bearing['outward_direction'])
    host, child = int(bearing['host_index']), int(bearing['moving_index'])
    housing = inputs.mesh(cache/bearing_feature(meta, bearing, 'socket'))
    pin = inputs.mesh(cache/bearing_feature(meta, bearing, 'pivot'))
    bare = inputs.mesh(cache/meta['shell_only_meshes'][child])
    mouth, stop, tip = (float(bearing[k]) for k in ('mouth_axial_mm', 'stop_axial_mm', 'pin_tip_axial_mm'))
    vent_s = float(np.asarray(bearing['wash_port_center'])@axis)
    vent_r = float(bearing['wash_port_diameter_mm'])/2
    end = min(stop, vent_s-vent_r)
    if end-mouth <= .02:
        raise RuntimeError('No unvented journal section available for wall measurement')
    station = (mouth+end)/2
    contours = sorted((contour_metrics(p) for p in section(housing, axis, station)),
                      key=lambda row: row['maximum_radius_mm'], reverse=True)
    if len(contours) != 2:
        raise RuntimeError('Expected outer and inner contours of the isolated housing')
    outside, bore = contours
    shaft = [contour_metrics(p) for p in section(pin, axis, (mouth+tip)/2)]
    root_s = float(bearing['pin_root_axial_mm'])+.03
    root = [contour_metrics(p) for p in section(pin, axis, root_s)]
    if len(shaft) != 1 or not root:
        raise RuntimeError('Cannot isolate shaft or root section')
    root = max(root, key=lambda row: row['area_mm2'])
    overlap = max(0., float((pin.manifold ^ bare.manifold).volume()))
    ro = float(meta['layer_radii_mm'][host])
    ri = float(meta['layer_radii_mm'][child])
    wall = outside['minimum_radius_mm']-bore['maximum_radius_mm']
    return dict(host_index=host, moving_index=child, side=int(bearing['side']),
        unvented_section_coordinate_mm=station,
        actual_housing_outer_minimum_radius_mm=outside['minimum_radius_mm'],
        actual_housing_outer_maximum_radius_mm=outside['maximum_radius_mm'],
        actual_bore_maximum_radius_mm=bore['maximum_radius_mm'],
        unvented_ring_wall_lower_bound_mm=wall,
        declared_radial_wall_mm=float(bearing['housing_radius_mm'])-float(bearing['bore_radius_mm']),
        shaft_radius_mm=shaft[0]['maximum_radius_mm'], shaft_section_area_mm2=shaft[0]['area_mm2'],
        measured_radial_running_clearance_mm=bore['minimum_radius_mm']-shaft[0]['maximum_radius_mm'],
        nominal_blind_cap_thickness_mm=ro-stop, axial_endplay_mm=stop-tip,
        pin_root_embed_below_nominal_sphere_mm=ri-float(bearing['pin_root_axial_mm']),
        root_section_coordinate_mm=root_s, root_section=root,
        pin_and_land_overlap_with_bare_shell_mm3=overlap,
        wash_port_diameter_mm=2*vent_r,
        wash_port_edge_to_blind_stop_axial_distance_mm=stop-(vent_s+vent_r),
        passed=bool(wall > 0 and overlap > 1e-8 and shaft[0]['area_mm2'] > 0),
        scope='Housing wall is measured on an unvented ring cross-section. The wash-hole edge distance and root attachment volume are separate geometric quantities, not minimum local strength or bond strength.')


def support_sections(meta, cache, inputs):
    rows = []
    for wire in meta['release_wires']:
        mesh = inputs.mesh(cache/wire['mesh_file'])
        ends = []
        for ep in wire['endpoints']:
            axis = unit(ep['inward_tangent'])
            transform = frame(axis).T
            centre = np.asarray(ep['neck_center'], float)
            cs = transformed(mesh.manifold, transform, -transform@centre).slice(0)
            width = float(ep.get('neck_width_mm', ep['neck_diameter_mm']))
            thick = float(ep.get('neck_thickness_mm', ep['neck_diameter_mm']))
            limit = .55*max(width, thick)+.005
            polygons = [np.asarray(p) for p in cs.to_polygons()
                        if len(p) >= 3 and np.linalg.norm(p, axis=1).max() <= limit]
            if len(polygons) != 1:
                raise RuntimeError('Cannot isolate the actual support waist')
            metrics = contour_metrics(polygons[0])
            ends.append(dict(layer_index=int(ep['layer_index']),
                declared_width_mm=width, declared_thickness_mm=thick,
                actual_section_area_mm2=metrics['area_mm2'],
                actual_maximum_diameter_mm=2*metrics['maximum_radius_mm']))
        rows.append(dict(wire_index=int(wire['wire_index']), kind=wire['kind'],
            host_index=int(wire['host_index']), moving_index=int(wire['moving_index']),
            body_width_mm=float(wire.get('body_width_mm', wire['body_diameter_mm'])),
            body_thickness_mm=float(wire.get('body_thickness_mm', wire['body_diameter_mm'])),
            endpoint_sections=ends,
            minimum_measured_neck_area_mm2=min(e['actual_section_area_mm2'] for e in ends)))
    return rows


def audit_module(path, configuration, inputs, expected_version, shell_target, ligament_target):
    meta = inputs.json(path)
    if int(meta.get('version', -1)) != expected_version:
        raise RuntimeError(f'{path.name}: expected version {expected_version}, got {meta.get("version")}')
    cache = path.parent
    layers = []
    for index, spec in enumerate(meta['layer_specs']):
        r, t = float(spec['radius_mm']), float(spec['thickness_mm'])
        segments = configuration['segments'](r)
        factor = tessellation_inradius(segments)
        inner = (r-t)*factor
        mesh = inputs.mesh(cache/meta['free_layer_meshes'][index])
        # Bare shells are also used by the pin-root attachment measurements.
        inputs.mesh(cache/meta['shell_only_meshes'][index])
        cone = cone_ligaments(spec, inner)
        samples = shell_samples(mesh, spec, factor)
        declared_floor = spec.get('ligament_inner_radius_lower_bound_mm')
        floor_valid = declared_floor is None or inner >= float(declared_floor)-1e-6
        nominal_target = t >= shell_target-1e-8 if shell_target is not None else True
        ligament_ok = (cone['minimum_distance_after_export_allowance_mm'] >= ligament_target-1e-6
                       if ligament_target is not None else True)
        layers.append(dict(layer_index=index, radius_mm=r, nominal_shell_thickness_mm=t,
            sphere_segments=segments, unit_sphere_facet_inradius=factor,
            verified_ordinary_inner_facet_radius_floor_mm=inner,
            declared_ligament_inner_radius_floor_mm=declared_floor,
            declared_floor_verified=floor_valid,
            ordinary_face_normal_thickness_lower_bound_mm=t*factor-EXPORT_ALLOWANCE_MM,
            nominal_removed_shell_fraction=spec['nominal_removed_fraction'],
            ordinary_perforation_ligaments=cone, actual_skin_samples=samples,
            nominal_thickness_target_passed=nominal_target, ligament_target_passed=ligament_ok,
            passed=bool(floor_valid and nominal_target and ligament_ok and samples['passed'])))
    bearing_rows = [measure_bearing(meta, b, cache, inputs) for b in meta['bearings']]
    supports = support_sections(meta, cache, inputs)
    mesh_hashes = {p.name: digest for p, digest in inputs.hashes.items()
                   if p.parent == cache.resolve() and p.suffix == '.npz'}
    return dict(layer_count=int(meta['layer_count']), version=int(meta['version']),
        metadata_file=path.name, metadata_sha256=inputs.hashes[path.resolve()],
        source_mesh_hashes=mesh_hashes, layers=layers, bearings=bearing_rows,
        support_sections=supports, layer_radii_mm=meta['layer_radii_mm'],
        nominal_radial_gaps_mm=meta['clear_radial_gaps_mm'],
        passed=bool(all(q['passed'] for q in layers) and all(q['passed'] for q in bearing_rows)),
        scope='Geometry of the source modules. No global minimum wall claim is made for service openings, bearing or wash-hole intersections, local trims, or relief tips.')


def comparisons(current, baseline):
    old = {m['layer_count']: m for m in baseline}
    rows = []
    for m in current:
        b = old[m['layer_count']]
        for layer, previous in zip(m['layers'], b['layers']):
            rows.append(dict(layer_count=m['layer_count'], layer_index=layer['layer_index'],
                v3_nominal_thickness_mm=previous['nominal_shell_thickness_mm'],
                v5_nominal_thickness_mm=layer['nominal_shell_thickness_mm'],
                nominal_thickness_ratio=layer['nominal_shell_thickness_mm']/previous['nominal_shell_thickness_mm'],
                v3_cone_ligament_lower_bound_mm=previous['ordinary_perforation_ligaments']['minimum_distance_after_export_allowance_mm'],
                v5_cone_ligament_lower_bound_mm=layer['ordinary_perforation_ligaments']['minimum_distance_after_export_allowance_mm'],
                v3_removed_shell_fraction=previous['nominal_removed_shell_fraction'],
                v5_removed_shell_fraction=layer['nominal_removed_shell_fraction']))
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=root/'cache/spheres')
    parser.add_argument('--reference-root', type=Path, default=root/'reference/v3')
    parser.add_argument('--report', type=Path, default=root/'reports/strength_validation.json')
    parser.add_argument('--minimum-shell-thickness', type=float, default=1.20)
    parser.add_argument('--minimum-ligament', type=float, default=1.10)
    args = parser.parse_args()
    started = time.monotonic()
    inputs = Inputs(root)
    inputs.remember(Path(__file__))
    inputs.remember(root/'source/audit_motion.py')
    source = source_configuration(root, inputs)
    reference = source_configuration(args.reference_root, inputs)
    paths = [args.cache/f'sphere_{n}layers.json' for n in (3, 4, 5)]
    # Reject a mixed or incomplete cache before any geometric calculation.
    for n, path in zip((3, 4, 5), paths):
        meta = inputs.json(path)
        if meta.get('version') != 5 or meta.get('layer_count') != n:
            raise RuntimeError(f'All three complete version-5 modules are required: {path}')
    source_targets_match = bool(
        abs(float(source['constants']['SHELL_THICKNESS'])-args.minimum_shell_thickness) < 1e-8
        and abs(float(source['constants']['MIN_PERFORATION_LIGAMENT'])-args.minimum_ligament) < 1e-8)
    modules, baseline = [], []
    for n, path in zip((3, 4, 5), paths):
        modules.append(audit_module(path, source, inputs, 5, args.minimum_shell_thickness, args.minimum_ligament))
        baseline.append(audit_module(args.reference_root/'cache/spheres'/path.name,
                                     reference, inputs, 3, None, None))
        print(f'L{n}: V5 geometric reinforcement passed={modules[-1]["passed"]}', flush=True)
    fingerprints, changed = inputs.final_fingerprints()
    report = dict(audit='V5 geometric reinforcement compared with actual V3 source meshes', units='mm',
        version=5, nominal_shell_thickness_target_mm=args.minimum_shell_thickness,
        ordinary_perforation_ligament_target_mm=args.minimum_ligament,
        export_coordinate_allowance_mm=EXPORT_ALLOWANCE_MM,
        source_targets_match=source_targets_match, modules=modules, reference_modules=baseline,
        reference_root=inputs.relative(args.reference_root), comparisons=comparisons(modules, baseline),
        input_sha256=fingerprints, fingerprints_stable=not changed, changed_inputs=changed,
        passed=bool(source_targets_match and not changed and all(m['passed'] for m in modules)
                    and all(m['passed'] for m in baseline)),
        elapsed_seconds=time.monotonic()-started,
        limits=[
            'The 1.20 mm requirement is the nominal CAD shell thickness; the inscribed facets give the separately reported ordinary face-normal lower bound.',
            'The regular opening ligament proof uses enclosing cones and a verified tessellation inradius. It excludes service slots, journals, wash holes, local cuts, relief tips, and their intersections.',
            'Shell ray measurements are samples in retained ordinary skin, not a complete minimum-thickness field.',
            'An unvented housing section and positive root overlap do not establish minimum strength at every journal or wash-hole intersection.',
            'This is not finite-element analysis or a strength, peel-load, green-resin, cure, fatigue, dimensional-calibration, support, or print-success certification.',
            'Free rotation, retention, startup connectivity, tool access, and extraction require the separate mechanism and early-support audits.'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'Report: {args.report}; passed={report["passed"]}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independent mesh-based validation of the V5 rotating nested spheres.

The generator is not imported. This audit reads its cached meshes and declared
mechanism coordinates, then measures engagement, verifies tool access, and
certifies the continuous interval between sampled angular positions using a
rigid-motion displacement bound. Units are millimetres.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import manifold3d as mf
import trimesh
from audit_geometry import projected_minimum, clip_at_z


# At world-coordinate magnitudes below128 mm, float32 coordinate rounding is
# at most sqrt(3)*2^-18 mm per vertex. This allowance exceeds the sum for two
# independently rounded surfaces and also leaves room for Boolean roundoff.
EXPORT_PRECISION_ALLOWANCE_MM = .00003


def log(message):
    print(message, flush=True)


def unit(value):
    value = np.asarray(value, dtype=float)
    return value / np.linalg.norm(value)


def frame(axis):
    a = unit(axis)
    seed = np.array([1., 0., 0.]) if abs(a[0]) < .8 else np.array([0., 0., 1.])
    u = unit(seed-a*(a @ seed))
    return np.column_stack((u, np.cross(a, u), a))


def rotation(axis, degrees):
    a = unit(axis)
    t = math.radians(degrees)
    x, y, z = a
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)*math.cos(t)+(1-math.cos(t))*np.outer(a, a)+math.sin(t)*k


def transformed(m, r, offset=(0., 0., 0.)):
    return m.transform(np.column_stack((r, offset)))


def sha256(path):
    hh = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            hh.update(block)
    return hh.hexdigest()


class Mesh:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(path) as data:
            self.vertices = np.asarray(data['vertices'], dtype=np.float64)
            self.faces = np.asarray(data['faces'], dtype=np.uint32)
        self.manifold = mf.Manifold(mf.Mesh64(self.vertices, self.faces))
        if self.manifold.status() != mf.Error.NoError:
            raise ValueError(f'Invalid input mesh: {path}: {self.manifold.status()}')

    def topology(self, expected_components=1):
        v, f = self.vertices, self.faces
        tv = v[f]
        areas = .5*np.linalg.norm(np.cross(tv[:, 1]-tv[:, 0], tv[:, 2]-tv[:, 0]), axis=1)
        tt = trimesh.Trimesh(v, f, process=False)
        comps = self.manifold.decompose()
        volumes = [float(c.volume()) for c in comps]
        result = dict(file=self.path.name, sha256=sha256(self.path), vertices=len(v),
            triangles=len(f), components=len(comps), expected_components=expected_components,
            component_volumes_mm3=volumes, zero_area_triangles=int(np.count_nonzero(areas == 0)),
            watertight=bool(tt.is_watertight), winding_consistent=bool(tt.is_winding_consistent))
        result['passed'] = bool(len(comps) == expected_components and all(q > 0 for q in volumes)
            and result['zero_area_triangles'] == 0 and result['watertight'] and result['winding_consistent'])
        return result

    def radius_bounds(self, axis):
        vv = self.vertices
        maximum = float(np.linalg.norm(vv, axis=1).max())
        minimum = np.inf
        for start in range(0, len(self.faces), 100000):
            t = vv[self.faces[start:start+100000]]
            closest = trimesh.triangles.closest_point(t, np.zeros((len(t), 3)))
            minimum = min(minimum, float(np.linalg.norm(closest, axis=1).min()))
        s = vv @ unit(axis)
        rho = np.sqrt(np.maximum(0., np.einsum('ij,ij->i', vv, vv)-s*s))
        return dict(minimum_surface_radius_mm=minimum, maximum_radius_mm=maximum,
                    maximum_distance_to_axis_mm=float(rho.max()),
                    axial_bounds_mm=[float(s.min()), float(s.max())])


def mesh_from_manifold(m):
    mm = m.to_mesh64()
    return np.asarray(mm.vert_properties[:, :3]), np.asarray(mm.tri_verts)


def positive_intersection_volume(a, b):
    intersection = a ^ b
    return 0. if intersection.is_empty() else max(0., float(intersection.volume()))


def cylinder_between(start, end, radius, end_radius=None, segments=32):
    p, q = np.asarray(start, float), np.asarray(end, float)
    delta = q-p
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return mf.Manifold()
    end_radius = radius if end_radius is None else end_radius
    # Circumscribe the specified circular tool, rather than using an inscribed
    # polygon that would silently test a marginally narrower shaft.
    radius /= math.cos(math.pi/segments)
    end_radius /= math.cos(math.pi/segments)
    return transformed(mf.Manifold.cylinder(length, radius, end_radius, segments),
                       frame(delta), p)


def axis_band_triangles(mesh, axis, start, end):
    """Actual triangle portions inside the open axial bearing interval."""
    vv = mesh.vertices @ frame(axis)
    tris = vv[mesh.faces]
    selected = (tris[:, :, 2].max(axis=1) >= start) & (tris[:, :, 2].min(axis=1) <= end)
    out = []
    for tri in tris[selected]:
        lower = [tri] if np.all(tri[:, 2] >= start) else clip_at_z(tri, start)
        for lo in lower:
            lo = np.asarray(lo)
            if np.all(lo[:, 2] <= end):
                out.append(lo)
            else:
                reverse = lo.copy(); reverse[:, 2] *= -1
                for clipped in clip_at_z(reverse, -end):
                    clipped = np.asarray(clipped); clipped[:, 2] *= -1
                    out.append(clipped)
    return np.asarray(out)


def first_ray_position(m, start, end):
    hits = m.ray_cast(start, end)
    return None if not hits else np.asarray(hits[0].position, float)


def bearing_measurements(bearing, opposite, layers, axis):
    host = layers[int(bearing['host_index'])]
    child = layers[int(bearing['moving_index'])]
    a = unit(axis)*float(bearing['side'])
    basis = frame(a)
    mouth = float(bearing['mouth_axial_mm'])
    stop = float(bearing['stop_axial_mm'])
    tip = float(bearing['pin_tip_axial_mm'])
    pr = float(bearing['pin_radius_mm'])
    br = float(bearing['bore_radius_mm'])
    hr = float(bearing.get('housing_radius_mm', 1.18))
    far = stop+4.
    actual_tip_position = first_ray_position(child.manifold, far*a, (mouth-.5)*a)
    actual_stop_position = first_ray_position(host.manifold, (mouth+.03)*a, far*a)
    if actual_tip_position is None or actual_stop_position is None:
        return dict(host_index=bearing['host_index'], moving_index=bearing['moving_index'],
                    side=bearing['side'], passed=False, error='Missing pin tip or blind stop on actual-axis ray')
    actual_tip = float(actual_tip_position @ a)
    actual_stop = float(actual_stop_position @ a)
    rim_radius = (br+hr)/2
    mouths = []
    for phi in np.arange(8)*math.pi/4:
        radial = rim_radius*(math.cos(phi)*basis[:, 0]+math.sin(phi)*basis[:, 1])
        position = first_ray_position(host.manifold, (mouth-.6)*a+radial, (stop+.3)*a+radial)
        if position is not None:
            mouths.append(float(position @ a))
    if len(mouths) != 8:
        return dict(host_index=bearing['host_index'], moving_index=bearing['moving_index'],
                    side=bearing['side'], passed=False, error='Housing mouth missing at one or more measured azimuths')
    actual_mouth = max(mouths)
    interval_start, interval_end = actual_mouth+.025, min(actual_tip, actual_stop)-.025
    ht = axis_band_triangles(host, a, interval_start, interval_end)
    ct = axis_band_triangles(child, a, interval_start, interval_end)
    radial2, _ = projected_minimum(ht)
    bore_min = math.sqrt(radial2)
    pin_max = float(np.linalg.norm(ct[:, :, :2], axis=2).max())
    pin_min2, _ = projected_minimum(ct)
    pin_min = math.sqrt(pin_min2)
    # Use the actual bore contour near the closed mouth to obtain an upper
    # bound on permissible play, distinct from the lower running-clearance
    # bound measured along the engaged shaft.
    cross_section = transformed(host.manifold, basis.T).slice(actual_mouth+.01)
    bore_contours = [np.asarray(poly) for poly in cross_section.to_polygons()
                     if len(poly) >= 3 and np.linalg.norm(poly, axis=1).max() < rim_radius]
    if len(bore_contours) != 1:
        return dict(host_index=bearing['host_index'], moving_index=bearing['moving_index'],
                    side=bearing['side'], passed=False, error='Cannot isolate the actual closed bore-mouth contour')
    bore_max = float(np.linalg.norm(bore_contours[0], axis=1).max())
    gap = bore_min-pin_max
    maximum_radial_play = bore_max-pin_min+EXPORT_PRECISION_ALLOWANCE_MM
    endplay = actual_stop-actual_tip
    opposite_endplay = float(opposite['stop_axial_mm'])-float(opposite['pin_tip_axial_mm'])
    engagement = actual_tip-actual_mouth
    retained = engagement-opposite_endplay
    theta_bound = math.atan2(max(0., maximum_radial_play), max(actual_mouth, 1e-9))
    # Conservative rigid-body tilt correction: permits additional endplay,
    # tip axial retraction, and the axial extent of a tilted pin-end circle.
    tilt_loss = 2*actual_tip*(1-math.cos(theta_bound))+pin_max*math.sin(theta_bound)
    displacement = math.hypot(endplay+actual_tip*(1-math.cos(theta_bound)), max(0., maximum_radial_play))
    metadata_error = max(abs(actual_tip-tip), abs(actual_stop-stop), abs(actual_mouth-mouth))
    result = dict(host_index=int(bearing['host_index']), moving_index=int(bearing['moving_index']),
        side=int(bearing['side']), actual_mouth_axial_mm=actual_mouth,
        actual_blind_stop_axial_mm=actual_stop, actual_pin_tip_axial_mm=actual_tip,
        actual_minimum_bore_radius_mm=bore_min, actual_maximum_pin_radius_mm=pin_max,
        actual_maximum_bore_mouth_radius_mm=bore_max, actual_minimum_pin_radius_mm=pin_min,
        conservative_maximum_radial_play_mm=maximum_radial_play,
        all_angle_radial_clearance_bound_mm=gap, axial_endplay_mm=endplay,
        engagement_mm=engagement, retained_engagement_after_opposite_endplay_mm=retained,
        conservative_tilt_bound_degrees=math.degrees(theta_bound),
        retained_engagement_with_conservative_tilt_mm=retained-tilt_loss,
        conservative_center_displacement_per_interface_mm=displacement,
        declared_geometry_maximum_error_mm=metadata_error)
    result['passed'] = bool(gap > .05 and endplay > .03 and retained-tilt_loss > .3 and metadata_error < .015)
    return result


def certify_rotation(host, child, axis, label, search_length=.6):
    """Prove the continuous rotation using sample gap minus motion bound."""
    h, m = host.manifold, child.manifold
    rho = child.radius_bounds(axis)['maximum_distance_to_axis_mm']
    overlap = positive_intersection_volume(h, m)
    initial = float(h.min_gap(m, search_length))
    if overlap > 1e-9 or initial <= 1e-5:
        return dict(pair=label, passed=False, initial_overlap_mm3=overlap, initial_gap_mm=initial)
    maximum_step = min(math.radians(15), 4*math.asin(min(.99, .78*initial/(2*rho))))
    n = max(24, int(math.ceil(2*math.pi/maximum_step)))
    if n > 1440:
        return dict(pair=label, passed=False, error='Clearance too small for bounded practical angular verification',
                    initial_gap_mm=initial, required_samples=n)
    gaps = [initial]
    started = time.monotonic()
    for k in range(1, n):
        moving = transformed(m, rotation(axis, k*360/n))
        gaps.append(float(h.min_gap(moving, search_length)))
        if k % 80 == 0:
            log(f'{label}: {k}/{n} angular samples, min gap={min(gaps):.6f}mm')
    displacement_bound = 2*rho*math.sin(math.pi/(2*n))
    numerical_allowance = EXPORT_PRECISION_ALLOWANCE_MM
    certified = min(gaps)-displacement_bound-numerical_allowance
    result = dict(pair=label, initial_overlap_mm3=overlap, sample_count=n,
        angular_step_degrees=360/n, smallest_sample_gap_mm=min(gaps),
        greatest_sample_gap_mm=max(gaps), worst_sample_angle_degrees=int(np.argmin(gaps))*360/n,
        moving_maximum_distance_to_axis_mm=rho,
        maximum_motion_to_nearest_sample_mm=displacement_bound,
        numerical_allowance_mm=numerical_allowance,
        certified_continuous_rotation_gap_mm=certified,
        elapsed_seconds=time.monotonic()-started,
        proof='Each unsampled angle is within half one angular step of a sample; every moving point travels at most2*rho_max*sin(step/4).')
    result['passed'] = bool(certified > 0)
    log(f'{label}: continuous360deg bound={certified:.6f}mm from{n} samples')
    return result


def union(items):
    return mf.Manifold.batch_boolean(items, mf.OpType.Add)


def neck_measurements(wire, endpoint, cache):
    center = np.asarray(endpoint['neck_center'], float)
    point = np.asarray(endpoint['attachment_point'], float)
    direction = unit(endpoint['inward_tangent'])
    r = frame(direction).T
    section = transformed(wire.manifold, r, -r@center).slice(0)
    profile = endpoint.get('neck_profile', 'circle')
    declared_width = float(endpoint.get('neck_width_mm', endpoint['neck_diameter_mm']))
    declared_thickness = float(endpoint.get('neck_thickness_mm', endpoint['neck_diameter_mm']))

    def contains_origin(poly):
        following = np.roll(poly, -1, axis=0)
        straddles = (poly[:, 1] > 0) != (following[:, 1] > 0)
        p, q = poly[straddles], following[straddles]
        intercepts = p[:, 0]-p[:, 1]*(q[:, 0]-p[:, 0])/(q[:, 1]-p[:, 1])
        return bool(np.count_nonzero(intercepts > 0) % 2)

    # The previous circular-only radius cutoff excluded real flattened
    # waists. Select the contour surrounding the declared centre, with a
    # bound derived from the declared profile; other wire branches crossing
    # this plane must not silently contribute to the neck area.
    contour_bound = max(.20, .55*max(declared_width, declared_thickness)+.005)
    local_contours = [np.asarray(poly) for poly in section.to_polygons()
                      if len(poly) >= 3 and np.linalg.norm(poly, axis=1).max() < contour_bound
                      and contains_origin(np.asarray(poly))]
    if len(local_contours) != 1:
        return dict(passed=False, error='The actual wire does not contain exactly one isolated waist at the declared neck centre')
    contour = local_contours[0]
    maximum_radius = float(np.linalg.norm(contour, axis=1).max())
    following = np.roll(contour, -1, axis=0)
    area = .5*abs(float(np.sum(contour[:, 0]*following[:, 1]-contour[:, 1]*following[:, 0])))
    stump = Mesh(cache/endpoint['stump_mesh_file'])
    actual_stump = float(((stump.vertices-point) @ direction).max())
    diameter = 2*maximum_radius
    width_axis = r@unit(endpoint.get('neck_width_axis', r.T[:, 0]))
    thickness_axis = r@unit(endpoint.get('neck_thickness_axis', r.T[:, 1]))
    axes_in_plane = bool(abs(width_axis[2]) < 1e-6 and abs(thickness_axis[2]) < 1e-6
                         and abs(width_axis@thickness_axis) < 1e-6)
    width = float(np.ptp(contour@unit(width_axis[:2])))
    thickness = float(np.ptp(contour@unit(thickness_axis[:2])))
    segments = endpoint.get('neck_profile_segments')
    expected_area = (float(segments)*declared_width*declared_thickness/8
                     *math.sin(2*math.pi/float(segments))) if segments is not None else None
    declared_area = endpoint.get('neck_area_mm2')
    area_matches = all(abs(area-float(value)) <= max(5e-7, .0025*float(value))
                       for value in (expected_area, declared_area) if value is not None)
    dimensions_match = (abs(width-declared_width) < .002
                        and abs(thickness-declared_thickness) < .002) if profile == 'ellipse' else (
                        abs(diameter-float(endpoint['neck_diameter_mm'])) < .002)
    return dict(neck_profile=profile,
        actual_neck_maximum_diameter_mm=diameter,
        actual_neck_width_mm=width, actual_neck_thickness_mm=thickness,
        actual_neck_cross_section_area_mm2=area,
        area_equivalent_circular_diameter_mm=2*math.sqrt(area/math.pi),
        actual_maximum_axial_residue_mm=actual_stump,
        declared_neck_width_mm=declared_width,
        declared_neck_thickness_mm=declared_thickness,
        expected_polygon_section_area_mm2=expected_area,
        declared_neck_area_mm2=declared_area,
        section_axes_in_plane_and_orthogonal=axes_in_plane,
        declared_neck_diameter_mm=float(endpoint['neck_diameter_mm']),
        declared_maximum_residue_mm=float(endpoint['maximum_residue_mm']),
        passed=bool(dimensions_match and area_matches and axes_in_plane
                    and actual_stump <= float(endpoint['maximum_residue_mm'])+.00003))


def projected_wire_separation(meta_path):
    """Check the cross-section that each complete support sweeps in +Z."""
    meta_path = Path(meta_path)
    cache = meta_path.parent
    meta = json.loads(meta_path.read_text())
    extrusions = []
    for w in meta['release_wires']:
        wire = Mesh(cache/w['mesh_file'])
        projection = mf.CrossSection(wire.manifold.project().to_polygons(), mf.FillRule.Positive)
        extrusions.append(projection.extrude(1.))
    pairs = []
    for i, first in enumerate(extrusions):
        for j in range(i+1, len(extrusions)):
            gap = float(first.min_gap(extrusions[j], .3))
            pairs.append(dict(wires=[i, j], projected_gap_lower_bound_mm=gap))
    closest = min(pairs, key=lambda q: q['projected_gap_lower_bound_mm'])
    bound = closest['projected_gap_lower_bound_mm']-EXPORT_PRECISION_ALLOWANCE_MM
    return dict(metadata_file=meta_path.name, metadata_sha256=sha256(meta_path),
        wire_count=len(extrusions), pair_count=len(pairs), closest_pair=closest,
        smallest_projected_gap_after_export_allowance_mm=bound,
        minimum_required_projected_gap_mm=.20,
        gap_search_limit_mm=.3,
        passed=bool(bound >= .20),
        proof='Positive-fill XY projections of complete actual wire meshes are extruded to a common slab. Their separation is unchanged by any relative translation along Z and therefore bounds wire-to-wire extraction clearance continuously.')


def certify_extraction(removable, obstacles, direction, distance):
    """Certify a translation with adaptively partitioned triangle sweeps.

    The sweep equals the original solid plus the extrusion of all boundary
    triangles facing the direction of travel. A convex hull of a group of
    those extrusions is a conservative bound. Only intersecting group bounds
    are subdivided; at a single triangle the bound is its exact prism. This
    avoids falsely filling the open bend between a wire and its own stump.
    """
    direction = unit(direction)
    delta = direction*distance
    triangles = removable.vertices[removable.faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    facing = (normals @ direction) > 1e-13
    faces = triangles[facing]
    centers = faces.mean(axis=1)
    tolerance = 1e-8
    original_overlap = positive_intersection_volume(removable.manifold, obstacles)
    hull = mf.Manifold.hull_points(np.vstack((removable.vertices, removable.vertices+delta)))
    coarse_overlap = positive_intersection_volume(hull, obstacles)
    visited = 0
    exact_prisms = 0
    unresolved_volume = 0.
    collision_bounds = []
    if coarse_overlap > tolerance:
        pending = [np.arange(len(faces))]
        while pending:
            indices = pending.pop()
            points = faces[indices].reshape(-1, 3)
            bound = mf.Manifold.hull_points(np.vstack((points, points+delta)))
            intersection = bound ^ obstacles
            overlap = max(0., float(intersection.volume()))
            visited += 1
            if overlap <= tolerance:
                continue
            if len(indices) == 1:
                exact_prisms += 1
                unresolved_volume += overlap
                if len(collision_bounds) < 8:
                    collision_bounds.append(list(intersection.bounding_box()))
                continue
            spread_axis = int(np.ptp(centers[indices], axis=0).argmax())
            ordered = indices[np.argsort(centers[indices, spread_axis])]
            middle = len(ordered)//2
            pending.extend((ordered[:middle], ordered[middle:]))
    return dict(passed=bool(original_overlap < tolerance and unresolved_volume == 0.),
        certificate='Original solid plus all forward-facing triangle prisms covers its full straight sweep. Convex bounds are recursively divided until clear or an exact single-triangle prism is reached.',
        numerical_volume_tolerance_mm3=tolerance,
        initial_obstruction_volume_mm3=original_overlap,
        coarse_convex_hull_obstruction_volume_mm3=coarse_overlap,
        adaptive_sweep_groups_checked=visited,
        colliding_exact_triangle_prisms=exact_prisms,
        sum_of_colliding_prism_volumes_mm3=unresolved_volume,
        collision_bounds_mm=collision_bounds)


def access_and_extraction(meta, cache, clean, stump_layers):
    wires = meta.get('release_wires', [])
    wire_models = [Mesh(cache/w['mesh_file']) for w in wires]
    permanent_clean = union([m.manifold for m in clean])
    nominal_stump_files = [ep.get('nominal_stump_mesh_file') for w in wires for ep in w['endpoints']]
    if nominal_stump_files and all(nominal_stump_files):
        permanent_stumps = union([permanent_clean]+[Mesh(cache/p).manifold for p in nominal_stump_files])
        extraction_state = 'Actual central-piece mesh and matching nominal break-plane stumps; maximum stumps are separately covered by the rotation certificate.'
    else:
        permanent_stumps = union([m.manifold for m in stump_layers])
        extraction_state = 'Maximum retained stumps; nominal extraction-state meshes were not all supplied.'
    declared_order = meta.get('release_sequence')
    if isinstance(declared_order, list):
        order = [int(x['wire_index']) if isinstance(x, dict) else int(x) for x in declared_order]
    else:
        order = sorted(range(len(wires)), key=lambda i: int(wires[i].get('sequence_index', i)))
    if len(order) != len(wires) or set(order) != set(range(len(wires))):
        raise ValueError('The declared release order does not cover every wire exactly once')
    remaining = set(range(len(wires)))
    results = []
    for number in order:
        w = wires[number]
        obstacles = union([permanent_clean]+[wire_models[i].manifold for i in sorted(remaining) if i != number])
        endpoints = []
        for ep in w['endpoints']:
            entry = np.asarray(ep['tool_entry'], float)
            shaft_end = np.asarray(ep['tool_shaft_end'], float)
            tip_end = np.asarray(ep['tool_tip_end'], float)
            shaft_radius = float(ep['tool_diameter_mm'])/2
            tip_radius = float(ep.get('tool_tip_diameter_mm', .20))/2
            needle_end = np.asarray(ep.get('tool_needle_end', tip_end), float)
            tool = union([cylinder_between(entry, shaft_end, shaft_radius),
                          cylinder_between(shaft_end, needle_end, shaft_radius, tip_radius),
                          cylinder_between(needle_end, tip_end, tip_radius)])
            overlap = positive_intersection_volume(tool, obstacles)
            contact = np.asarray(ep.get('tool_contact_point', ep['neck_center']), float)
            contact_tool = union([tool, cylinder_between(tip_end, contact, tip_radius)])
            contact_overlap = positive_intersection_volume(contact_tool, obstacles)
            neck = neck_measurements(wire_models[number], ep, cache)
            endpoints.append(dict(neck_center=ep['neck_center'],
                tool_shaft_diameter_mm=2*shaft_radius, tool_tip_diameter_mm=2*tip_radius,
                shaft_length_mm=float(np.linalg.norm(entry-shaft_end)),
                taper_length_mm=float(np.linalg.norm(shaft_end-needle_end)),
                needle_length_mm=float(np.linalg.norm(needle_end-tip_end)),
                obstruction_volume_mm3=overlap,
                approach_to_actual_neck_center_obstruction_volume_mm3=contact_overlap,
                actual_neck_geometry=neck,
                passed=bool(overlap < 1e-8 and contact_overlap < 1e-8 and neck['passed'])))
        extraction = dict(passed=False, error='No actual removable central-piece mesh declared')
        removable_file = w.get('removable_mesh_file')
        if removable_file:
            removable = Mesh(cache/removable_file)
            direction = unit(w.get('extraction_vector', [0, 0, 1]))
            distance = float(w.get('extraction_distance_mm', 2*meta.get('outer_nominal_radius_mm', 13)+6))
            extraction_obstacles = union([permanent_stumps]+[wire_models[i].manifold
                for i in sorted(remaining) if i != number])
            extraction = dict(removable_mesh_file=removable_file,
                direction=direction.tolist(), translation_mm=distance,
                retained_stump_state=extraction_state,
                **certify_extraction(removable, extraction_obstacles, direction, distance))
        results.append(dict(wire_index=number, host_index=w['host_index'], moving_index=w['moving_index'],
                            neck_access=endpoints, extraction=extraction,
                            passed=all(e['passed'] for e in endpoints) and extraction['passed']))
        remaining.remove(number)
        log(f'release wire{number}: access={all(e["passed"] for e in endpoints)} extraction={extraction["passed"]}')
    return dict(declared_sequence=order, wires=results, passed=bool(wires) and all(r['passed'] for r in results))


def initial_print_connectivity(supported, clean, build_direction=(0, 1, 0)):
    """Check that the first moving pin material is already tied to its host.

    This is deliberately narrower than a full slicer/support validation. It
    catches an internal captive part whose decorative support appears only
    many layers after the part itself starts printing.
    """
    build = unit(build_direction)
    results = []
    for child_index in range(1, len(clean)):
        child = clean[child_index]
        host = clean[child_index-1]
        start = float((child.vertices @ build).min())
        prefix_results = []
        for rise in [.01, .04, .10]:
            h = start+rise
            prefix = supported.manifold.trim_by_plane(-build, -h)
            host_prefix = host.manifold.trim_by_plane(-build, -h)
            child_prefix = child.manifold.trim_by_plane(-build, -h)
            child_volume = float(child_prefix.volume())
            host_connected = False
            touching_components = 0
            for component in prefix.decompose():
                touched_child = positive_intersection_volume(component, child_prefix) > 1e-12
                if touched_child:
                    touching_components += 1
                    if positive_intersection_volume(component, host_prefix) > 1e-10:
                        host_connected = True
            prefix_results.append(dict(height_above_first_moving_material_mm=rise,
                absolute_build_coordinate_mm=h, moving_prefix_volume_mm3=child_volume,
                components_containing_moving_prefix=touching_components,
                connected_to_parent_material=host_connected,
                passed=bool(child_volume > 1e-12 and touching_components == 1 and host_connected)))
        results.append(dict(moving_layer=child_index, first_moving_build_coordinate_mm=start,
                            early_prefix_checks=prefix_results,
                            passed=all(r['passed'] for r in prefix_results)))
    return dict(build_direction=build.tolist(), moving_layers=results,
        passed=all(r['passed'] for r in results),
        scope='Checks captive moving-part startup at0.01,0.04,0.10mm above its first material; this is not a full support or slicer certification.')


def audit_module(meta_path, skip_motion=False, skip_access=False):
    started = time.monotonic()
    meta_path = Path(meta_path)
    cache = meta_path.parent
    meta = json.loads(meta_path.read_text())
    axis = unit(meta['axis_unit_vector'])
    clean = [Mesh(cache/p) for p in meta['free_layer_meshes']]
    stump_layers = [Mesh(cache/p) for p in meta['released_with_stumps_layer_meshes']]
    n = len(clean)
    topologies = [m.topology(1) for m in stump_layers]
    supported = Mesh(cache/meta['supported_mesh_file'])
    supported_topology = supported.topology(1)
    released = union([m.manifold for m in stump_layers])
    released_components = released.decompose()
    bounds = [m.radius_bounds(axis) for m in stump_layers]
    bearings = []
    for b in meta['bearings']:
        opposites = [other for other in meta['bearings']
            if int(other['host_index']) == int(b['host_index']) and int(other['moving_index']) == int(b['moving_index'])
            and int(other['side']) == -int(b['side'])]
        if len(opposites) != 1:
            raise ValueError('Each bearing must have exactly one opposed bearing')
        bearings.append(bearing_measurements(b, opposites[0], stump_layers, axis))
    # Each joint can translate its child's centre relative to its parent.
    # A fixed V3 allowance is invalid when journal clearances change. Use
    # the larger measured bound from the two opposed bearings at each
    # interface; triangle inequality then bounds all descendant centres.
    interface_play = []
    epsilon = EXPORT_PRECISION_ALLOWANCE_MM
    for host_index in range(n-1):
        pair = [b for b in bearings if b['host_index'] == host_index
                and b['moving_index'] == host_index+1]
        required = ('conservative_center_displacement_per_interface_mm',
                    'conservative_maximum_radial_play_mm', 'axial_endplay_mm',
                    'actual_pin_tip_axial_mm', 'actual_mouth_axial_mm')
        valid = bool(len(pair) == 2 and all(
            all(key in b and math.isfinite(float(b[key])) for key in required)
            for b in pair))
        padded = []
        if valid:
            for b in pair:
                radial = max(0., float(b['conservative_maximum_radial_play_mm']))
                # The measured radial-play bound already includes epsilon.
                # Pad the axial end gap and the coordinates used for tilt
                # as well, so export rounding cannot enlarge a joint beyond
                # its accumulated source-geometry envelope.
                mouth = float(b['actual_mouth_axial_mm'])-epsilon
                if mouth <= 0:
                    valid = False
                    break
                theta = math.atan2(radial, mouth)
                axial = (max(0., float(b['axial_endplay_mm']))+epsilon
                         +(max(0., float(b['actual_pin_tip_axial_mm']))+epsilon)
                         *(1-math.cos(theta)))
                padded.append(max(float(b['conservative_center_displacement_per_interface_mm']),
                                  math.hypot(axial, radial)))
        interface_play.append(dict(host_index=host_index, moving_index=host_index+1,
            source_measured_center_displacement_bound_mm=max(
                float(b['conservative_center_displacement_per_interface_mm']) for b in pair) if valid else None,
            float32_padded_center_displacement_bound_mm=max(padded) if valid else None,
            opposed_bearing_sides=[b['side'] for b in pair], passed=valid))
    play_bounds_valid = all(row['passed'] for row in interface_play)
    cumulative_play = [0.]
    for row in interface_play:
        allowance = row['float32_padded_center_displacement_bound_mm']
        cumulative_play.append(cumulative_play[-1]+allowance
                               if cumulative_play[-1] is not None and allowance is not None else None)
    rotations = []
    if not skip_motion:
        for i in range(n-1):
            rotations.append(certify_rotation(stump_layers[i], stump_layers[i+1], axis,
                                             f'{meta_path.stem} layers{i}/{i+1}'))
    nonadjacent = []
    for i in range(n):
        for j in range(i+2, n):
            centered_lower = bounds[i]['minimum_surface_radius_mm']-bounds[j]['maximum_radius_mm']
            relative_play = sum(row['float32_padded_center_displacement_bound_mm']
                                for row in interface_play[i:j]) if all(
                                row['passed'] for row in interface_play[i:j]) else None
            lower = centered_lower-relative_play-epsilon if relative_play is not None else None
            overlap = positive_intersection_volume(stump_layers[i].manifold, stump_layers[j].manifold)
            nonadjacent.append(dict(outer_layer=i, inner_layer=j, initial_overlap_mm3=overlap,
                centered_all_angle_spherical_separation_bound_mm=centered_lower,
                relative_center_displacement_bound_mm=relative_play,
                surface_export_rounding_allowance_mm=epsilon,
                all_angle_spherical_separation_bound_mm=lower,
                passed=bool(lower is not None and lower > 0 and overlap < 1e-9)))
    # Actual material after the declared breaks must remain inside the printed
    # supported model; disappearing material is the deliberately removed wire.
    extra_after_breaks = max(0., float((released-supported.manifold).volume()))
    removed = supported.manifold-released
    removed_volume = float(removed.volume())
    print_start = initial_print_connectivity(supported, clean, meta.get('build_direction', [0, 1, 0]))
    access = None if skip_access else access_and_extraction(meta, cache, clean, stump_layers)
    projected_supports = None if skip_access else projected_wire_separation(meta_path)
    wall_envelope = max(bounds[i]['maximum_radius_mm']+cumulative_play[i]
                        for i in range(1, n)) if play_bounds_valid else None
    result = dict(metadata_file=meta_path.name, metadata_sha256=sha256(meta_path), layer_count=n,
        source_cache=str(cache.resolve()),
        certificate_mesh_scope='Actual source-module meshes named and hashed below. Whole-holder exported topology is audited separately; these geometry certificates transfer through rigid assembly when mechanism surfaces are preserved apart from bounded coordinate rounding.',
        axis_unit_vector=axis.tolist(), actual_with_stumps_layer_topology=topologies,
        supported_topology=supported_topology, actual_released_components=len(released_components),
        expected_released_components=n, layer_envelopes=bounds,
        supported_minus_released_volume_mm3=removed_volume,
        released_material_absent_from_supported_mm3=extra_after_breaks,
        measured_bearings=bearings, continuous_rotation=rotations,
        initial_print_connectivity=print_start,
        nonadjacent_shell_clearance=nonadjacent, release_access_and_extraction=access,
        support_projection_separation=projected_supports,
        moving_centered_radius_bound_mm=max(bounds[i]['maximum_radius_mm'] for i in range(1, n)),
        moving_radius_including_conservative_play_mm=wall_envelope,
        measured_play_bounds_by_interface=interface_play,
        cumulative_play_allowance_by_layer_mm=cumulative_play,
        play_bounds_valid=play_bounds_valid,
        # Kept for older report readers as a conservative uniform bound;
        # the actual wall and nonadjacent tests above use each joint's bound.
        play_allowance_per_nested_interface_mm=max(
            row['float32_padded_center_displacement_bound_mm'] for row in interface_play) if play_bounds_valid else None,
        play_allowance_method='Maximum of opposed measured journal bounds per interface, including export rounding; accumulated along each nested parent-child chain.',
        float32_export_allowance_mm=EXPORT_PRECISION_ALLOWANCE_MM,
        elapsed_seconds=time.monotonic()-started)
    result['passed'] = bool(all(t['passed'] for t in topologies) and supported_topology['passed']
        and len(released_components) == n and all(c.volume() > 0 for c in released_components)
        and all(b['passed'] for b in bearings) and play_bounds_valid and not skip_motion
        and all(r['passed'] for r in rotations) and all(q['passed'] for q in nonadjacent)
        and extra_after_breaks < 1e-8 and not skip_access and access['passed']
        and projected_supports['passed'] and print_start['passed'])
    result['scope'] = [
        'The motion certificate includes the actual maximum retained breakaway stumps supplied in each layer mesh.',
        'Centered continuous360degree relative rotation is certified for adjacent layers; independent rotations compose because they share one axis.',
        'Bearing retention uses measured tip/stop/mouth coordinates with conservative opposite-endplay and tilt allowances.',
        'Wall envelopes and nonadjacent spherical separation include the cumulative measured relative centre displacement along the intervening bearing chain.',
        'Tool access and extraction apply to the declared aligned print orientation and declared release order.',
        'Manufacturing tolerances, cured-resin friction and actual fracture behaviour still depend on the selected resin and printer.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path(__file__).resolve().parent.parent/'cache'/'spheres')
    parser.add_argument('--report', type=Path, default=Path(__file__).resolve().parent.parent/'reports'/'motion_validation.json')
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument('--skip-motion', action='store_true')
    parser.add_argument('--skip-access', action='store_true')
    parser.add_argument('--projection-only', action='store_true', help='Audit the full support XY projections and their continuous relative +Z separation')
    args = parser.parse_args()
    result = dict(audit='Independent V5 mechanism audit', units='mm', modules=[])
    for layer_count in args.layers:
        log(f'Auditing{layer_count}layer mechanism')
        module = (projected_wire_separation(args.cache/f'sphere_{layer_count}layers.json') if args.projection_only
                  else audit_module(args.cache/f'sphere_{layer_count}layers.json', args.skip_motion, args.skip_access))
        module.setdefault('layer_count', layer_count)
        result['modules'].append(module)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2)+'\n')
    result['passed'] = all(m['passed'] for m in result['modules'])
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    log(json.dumps(dict(passed=result['passed'], modules=[dict(layers=m['layer_count'],passed=m['passed']) for m in result['modules']]), indent=2))

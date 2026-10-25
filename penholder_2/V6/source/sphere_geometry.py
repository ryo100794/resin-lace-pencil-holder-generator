"""Print-in-place concentric floral spheres with captured opposed journals.

Units are millimetres.  Local +Z faces out of the vessel, +Y points upward,
and +X is tangential to the vessel.  The equatorial mounting collar lies
in z=0.  Opposed blind journal pockets retain each independently rotating
layer.  One accessible front-window wire and two straight bottom starters temporarily join each pair;
both wire ends have short notched breakaway necks.  The bottom starters have
elliptical necks and separate grasping fins. No permanent interlayer
bridge from V1 is included.

Dependencies: numpy, manifold3d, trimesh.  Example::

    PYTHONPATH=/tmp/sphere_model_deps python sphere_geometry.py

The public function ``build_sphere(radius, layers)`` returns (Manifold,
metadata).  Running the module caches all three requested sizes and their
separate, outer-first spherical layers for review and section rendering.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import manifold3d as mf
import trimesh


DEG = math.pi / 180.0
PIVOT_AXIS = np.array([1.0, 0.0, 0.0])
PIN_RADIUS = 0.60
BORE_RADIUS = 1.00
HOUSING_RADIUS = 1.65
RUNNING_GAP = 0.40
WIRE_RADIUS = 0.275
NECK_RADIUS = 0.20
SHELL_THICKNESS = 1.20
MIN_PERFORATION_LIGAMENT = 1.10
SPHERE_FACET_ALLOWANCE = 0.04


def _unit(a):
    a = np.asarray(a, dtype=float)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _rotation_from_to(a, b):
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        q = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            q = np.array([0.0, 1.0, 0.0])
        u = _unit(np.cross(a, q))
        return 2.0 * np.outer(u, u) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def _about_z(degrees):
    a = degrees * DEG
    return np.array([[math.cos(a), -math.sin(a), 0],
                     [math.sin(a), math.cos(a), 0], [0, 0, 1]], float)


def _xf(rotation, translation=(0, 0, 0)):
    return np.column_stack([rotation, np.asarray(translation, dtype=float)])


def _union(items):
    return mf.Manifold.batch_boolean(items, mf.OpType.Add)


def _mesh_manifold(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    # This also normalizes the winding of the explicit decorative sweeps.
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    if np.einsum('ij,ij->i', a, np.cross(b, c)).sum() < 0:
        faces = np.ascontiguousarray(faces[:, ::-1])
    result = mf.Manifold(mf.Mesh(vertices, faces))
    if result.status() != mf.Error.NoError:
        raise RuntimeError(f"Explicit surface construction failed: {result.status()}")
    return result


def _icosahedral_frames(rotation_degrees=0.0):
    ico = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
    points = np.asarray(ico.vertices, dtype=float)
    rotation = _rotation_from_to(points[4], [0, 0, 1])
    rotation = _about_z(rotation_degrees) @ rotation
    points = points @ rotation.T
    frames = []
    for index, n in enumerate(points):
        dots = points @ n
        neighbors = np.argsort(dots)[-6:-1]
        # Every vertex of an icosahedron has five equidistant neighbours.
        q = points[int(neighbors[0])]
        u = _unit(q - n * np.dot(q, n))
        v = np.cross(n, u)
        frames.append((n, u, v))
    small_frames = []
    for triangle in ico.faces:
        n = _unit(points[triangle].sum(axis=0))
        q = points[int(triangle[0])]
        u = _unit(q - n * np.dot(q, n))
        v = np.cross(n, u)
        small_frames.append((n, u, v))
    return frames, small_frames


def _radial_port(frame, reach, base_angle, amplitude, lobes, phase=0, samples=80):
    """Solid radial cone whose angular boundary is a petalled rosette."""
    n, u, v = frame
    theta = np.linspace(0, 2 * math.pi, samples, endpoint=False)
    alpha = (base_angle + amplitude * np.cos(lobes * theta + phase)) * DEG
    rho = reach * np.tan(alpha)
    polygon = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
    profile = mf.CrossSection([polygon])
    # Extrusion tapers to an apex at the origin after this rigid transform.
    cutter = mf.Manifold.extrude(profile, reach, scale_top=(0, 0))
    cutter = cutter.rotate((180, 0, 0)).translate((0, 0, reach))
    # The 180 degree rotation reverses local Y.  Compensate so its angular
    # phase agrees precisely with the raised rim and analytical clearance.
    return cutter.transform(_xf(np.column_stack([u, -v, n])))


def _closed_sweep(points, outward, half_width=0.27, height=0.34,
                  embed=0.14, stitch_periods=0):
    """A narrow, five-sided stitched relief with a crisp raised ridge."""
    points = np.asarray(points, float)
    outward = _unit(outward)
    tangent = _unit(np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0))
    across = _unit(np.cross(tangent, outward))
    count = len(points)
    t = np.arange(count) * 2 * math.pi / count
    modulation = 1.0 + 0.07 * np.cos(stitch_periods * t) if stitch_periods else np.ones(count)
    relief = height * (0.92 + 0.08 * np.cos(stitch_periods * t)) if stitch_periods else np.full(count, height)
    # Base penetrates the receiving skin; the ridge forms actual geometry.
    side_profile = [(-1, -embed), (1, -embed), (1, 0.06), (0, None), (-1, 0.06)]
    rings = []
    for x, y in side_profile:
        offset = relief if y is None else np.full(count, y)
        rings.append(points + across * (x * half_width * modulation)[:, None]
                     + outward * offset[:, None])
    vertices = np.stack(rings, axis=1).reshape((-1, 3))
    faces = []
    sides = len(side_profile)
    for i in range(count):
        for j in range(sides):
            a, b = i * sides + j, ((i + 1) % count) * sides + j
            c = ((i + 1) % count) * sides + (j + 1) % sides
            d = i * sides + (j + 1) % sides
            faces.extend(((a, b, c), (a, c, d)))
    return _mesh_manifold(vertices, faces)


def _port_border(frame, radius, base_angle, amplitude, lobes, phase=0,
                 small=False):
    n, u, v = frame
    samples = 150 if not small else 60
    theta = np.linspace(0, 2 * math.pi, samples, endpoint=False)
    # Broad, shallow relief sits entirely on the receiving shell instead of
    # forming a thin overhanging lip at the edge of the perforation.
    half_width = 0.30 if small else 0.38
    embed = 0.24
    supported_offset = math.asin((half_width * 1.07 + 0.10) / (radius - embed)) / DEG
    alpha = (base_angle + amplitude * np.cos(lobes * theta + phase) + supported_offset) * DEG
    radial = (np.cos(alpha)[:, None] * n + np.sin(alpha)[:, None] *
              (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v))
    return _closed_sweep(radial * (radius + 0.015), radial,
                         half_width=half_width,
                         height=0.16 if small else 0.22,
                         embed=embed, stitch_periods=12 if small else 30)


def _layer_parameters(radius, layer_index):
    if radius <= 6.5:
        angle, amp = 22.0, 2.5
    else:
        angle, amp = 24.0, 3.3
    # Preserve the floral layout while widening every bridge between its
    # conical openings.  The smallest radius is on the INSIDE of the wall;
    # using the outside radius would overstate the remaining bridge width.
    large, small = _icosahedral_frames(layer_index * 3.0)
    inner_floor = radius - SHELL_THICKNESS - SPHERE_FACET_ALLOWANCE
    required_angle = 2 * math.asin(MIN_PERFORATION_LIGAMENT / (2 * inner_floor)) / DEG
    separation = min(math.acos(float(np.clip(a[0] @ b[0], -1, 1))) / DEG
                     for i, a in enumerate(large) for b in large[i+1:])
    max_angle = (separation - required_angle) / 2
    small_angle = 5.0 if radius >= 12 else 4.2
    small_amplitude = 1.05 if radius >= 12 else 0.85
    if layer_index == 0:
        cross_separation = min(math.acos(float(np.clip(a[0] @ b[0], -1, 1))) / DEG
                               for a in large for b in small)
        max_angle = min(max_angle, cross_separation - small_angle - small_amplitude - required_angle)
    angle = min(angle, max_angle - amp)
    if angle <= amp:
        raise ValueError('Requested perforations cannot retain the structural web')
    return {
        'radius_mm': float(radius),
        'thickness_mm': SHELL_THICKNESS,
        'minimum_perforation_ligament_target_mm': MIN_PERFORATION_LIGAMENT,
        'ligament_inner_radius_lower_bound_mm': inner_floor,
        'ligament_scope': 'Between floral perforation envelopes; bearings and service access cuts are audited separately.',
        'rotation_degrees': layer_index * 3.0,
        'petal_phase_radians': layer_index * 0.12,
        'port_angle_degrees': angle,
        'port_amplitude_degrees': amp,
        'small_ports': bool(layer_index == 0),
        'small_port_angle_degrees': small_angle,
        'small_port_amplitude_degrees': small_amplitude,
    }


def _build_layer(radius, layer_index, outer):
    spec = _layer_parameters(radius, layer_index)
    seg = 80 if radius >= 9 else (64 if radius >= 6 else 48)
    outer_sphere = mf.Manifold.sphere(radius, seg)
    inner_sphere = mf.Manifold.sphere(radius - spec['thickness_mm'], seg)
    blank = outer_sphere - inner_sphere
    large_frames, small_frames = _icosahedral_frames(spec['rotation_degrees'])
    cuts = [_radial_port(frame, radius + 2, spec['port_angle_degrees'],
                         spec['port_amplitude_degrees'], 5,
                         spec['petal_phase_radians'], 100) for frame in large_frames]
    if spec['small_ports']:
        cuts.extend(_radial_port(frame, radius + 2,
                                 spec['small_port_angle_degrees'],
                                 spec['small_port_amplitude_degrees'], 3, 0, 48)
                    for frame in small_frames)
    shell = mf.Manifold.batch_boolean([blank] + cuts, mf.OpType.Subtract)
    unadorned_volume = shell.volume()
    spec['nominal_removed_fraction'] = float(1 - unadorned_volume / blank.volume())
    if outer:
        decoration = [_port_border(frame, radius, spec['port_angle_degrees'],
                                    spec['port_amplitude_degrees'], 5,
                                    spec['petal_phase_radians']) for frame in large_frames]
        # Fine secondary trefoil rims leave the wider windows visually dominant.
        decoration.extend(_port_border(frame, radius,
                                        spec['small_port_angle_degrees'],
                                        spec['small_port_amplitude_degrees'], 3, small=True)
                          for frame in small_frames)
        shell = _union([shell] + decoration)
    spec['triangle_count'] = int(shell.num_tri())
    spec['volume_mm3'] = float(shell.volume())
    spec['components'] = len(shell.decompose())
    spec['status'] = str(shell.status())
    if spec['components'] != 1 or shell.status() != mf.Error.NoError:
        raise RuntimeError(f"Perforated layer is not a single valid solid: {spec}")
    return shell, spec


def _candidate_directions(count=18000):
    i = np.arange(count)
    y = 1.0 - 2.0 * (i + 0.5) / count
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    r = np.sqrt(1 - y * y)
    p = np.column_stack([r * np.cos(phi), y, r * np.sin(phi)])
    return p[(p[:, 1] < -0.30) & (p[:, 2] < -0.20)]


def _port_clearance(points, spec):
    """Conservative distance on each skin to the nearest pierced boundary."""
    large, small = _icosahedral_frames(spec['rotation_degrees'])
    margin = np.full(len(points), np.inf)
    groups = [(large, 5, spec['port_angle_degrees'], spec['port_amplitude_degrees'],
               spec['petal_phase_radians'])]
    if spec['small_ports']:
        groups.append((small, 3, spec['small_port_angle_degrees'],
                       spec['small_port_amplitude_degrees'], 0))
    for frames, lobes, angle, amp, phase in groups:
        for n, u, v in frames:
            gamma = np.arccos(np.clip(points @ n, -1, 1))
            phi = np.arctan2(points @ v, points @ u)
            threshold = (angle + amp * np.cos(lobes * phi + phase)) * DEG
            margin = np.minimum(margin, (gamma - threshold) * spec['radius_mm'])
    return margin


def _rod(a, b, radius=0.52):
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = b - a
    length = float(np.linalg.norm(delta))
    return mf.Manifold.cylinder(length, radius, radius, 12).transform(
        _xf(_rotation_from_to([0, 0, 1], delta / length), a))


def _axial_cylinder(direction, start, end, radius, segments=64):
    direction = np.asarray(direction, dtype=float)
    return mf.Manifold.cylinder(end - start, radius, radius, segments).transform(
        _xf(_rotation_from_to([0, 0, 1], direction), direction * start))


def _bearing_parts(specs):
    additions = [[] for _ in specs]
    cuts = [[] for _ in specs]
    features, bearings = [], []
    for host in range(len(specs) - 1):
        moving = host + 1
        ro, ri = specs[host]['radius_mm'], specs[moving]['radius_mm']
        mouth, stop = ri + RUNNING_GAP, ro - 0.65
        tip = stop - 0.35
        for side in (-1, 1):
            direction = PIVOT_AXIS * side
            housing_blank = _axial_cylinder(direction, mouth, ro, HOUSING_RADIUS)
            bore = _axial_cylinder(direction, mouth - 0.22, stop, BORE_RADIUS)
            vent_center = direction * (stop - 0.55)
            vent_reach = HOUSING_RADIUS + 0.35
            vent = _rod(vent_center - [0, 0, vent_reach], vent_center + [0, 0, vent_reach], 0.35)
            housing = housing_blank - bore - vent
            pin = _axial_cylinder(direction, ri - 0.65, tip, PIN_RADIUS)
            # These local lands make bearing attachment independent of the
            # surrounding cutwork's phase.  Their radial profile is spherical.
            land_cylinder = _axial_cylinder(direction, ri - 1.10, ri + 0.08, 1.25)
            land_skin = (mf.Manifold.sphere(ri + 0.025, 64)
                         - mf.Manifold.sphere(ri - specs[moving]['thickness_mm'], 64))
            pin_land = land_cylinder ^ land_skin
            pin = pin + pin_land
            additions[host].append(housing_blank)
            additions[moving].append(pin)
            cuts[host].extend([bore, vent])
            serial = len(bearings)
            bearings.append({
                'bearing_index': serial, 'host_index': host, 'moving_index': moving,
                'side': side, 'axis_unit_vector': PIVOT_AXIS.tolist(),
                'outward_direction': direction.tolist(),
                'mouth_axial_mm': mouth, 'stop_axial_mm': stop,
                'pin_tip_axial_mm': tip, 'pin_root_axial_mm': ri - 0.65,
                'pin_radius_mm': PIN_RADIUS, 'bore_radius_mm': BORE_RADIUS,
                'housing_radius_mm': HOUSING_RADIUS,
                'radial_clearance_nominal_mm': BORE_RADIUS - PIN_RADIUS,
                'axial_end_clearance_mm': stop - tip,
                'initial_engagement_mm': tip - mouth,
                'minimum_engagement_at_full_axial_play_mm': tip - mouth - (stop - tip),
                'blind_cap_thickness_mm': ro - stop,
                'housing_mouth_to_inner_nominal_sphere_mm': mouth - ri,
                'wash_port_diameter_mm': 0.70,
                'wash_port_center': vent_center.tolist(),
                'mouth_point': (direction * mouth).tolist(),
                'stop_point': (direction * stop).tolist(),
                'pin_tip_point': (direction * tip).tolist(),
                'retention': 'opposed blind end stops; remaining engagement exceeds full axial play',
                'circular_segments': 64,
            })
            features.extend([
                {'model': housing, 'role': 'socket', 'layer_index': host,
                 'host_index': host, 'moving_index': moving, 'side': side,
                 'bearing_index': serial},
                {'model': pin, 'role': 'pivot', 'layer_index': moving,
                 'host_index': host, 'moving_index': moving, 'side': side,
                 'bearing_index': serial},
                {'model': bore, 'role': 'bore_cut', 'layer_index': host,
                 'host_index': host, 'moving_index': moving, 'side': side,
                 'bearing_index': serial},
            ])
    return additions, cuts, features, bearings


def _front_window_frame(spec):
    frames, _ = _icosahedral_frames(spec['rotation_degrees'])
    return max(frames, key=lambda f: float(f[0][2]))


def _window_attachment(spec, azimuth):
    n, u, v = _front_window_frame(spec)
    planar = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    phi = math.atan2(np.dot(planar, v), np.dot(planar, u))
    theta = (spec['port_angle_degrees'] + spec['port_amplitude_degrees'] *
             math.cos(5 * phi + spec['petal_phase_radians'])) * DEG
    radial = math.cos(theta) * n + math.sin(theta) * planar
    inward = math.sin(theta) * n - math.cos(theta) * planar
    # The foot contacts the actual conical window sidewall close to its
    # inner edge.  Its embedded length is distinct from the visible nub.
    distance = spec['radius_mm'] - spec['thickness_mm'] + 0.03
    # Recess the nominal foot 0.02 mm into the opening so the thicker
    # breakaway waist clears the faceted inner edge after cutting.
    point = radial * distance + inward * 0.02
    # A widened carrier following the narrow waist must grow upward in
    # the local +Y build direction.  The natural aperture tangent is too
    # shallow near the left/right petals: its widened head can appear as
    # an island before the waist reaches it.  Keep the true wall foot,
    # but pitch the carrier toward +Y inside the lower-half aperture.
    if inward[1] < 0.75:
        transverse = math.hypot(inward[0], inward[2])
        inward = np.array([inward[0] / transverse * math.sqrt(1 - 0.75**2),
                           0.75,
                           inward[2] / transverse * math.sqrt(1 - 0.75**2)])
    return point, inward, theta


def _profiled_wire_endpoint(point, direction, carrier_length=0.95, receiving_layer=None):
    direction = _unit(direction)
    q = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, q)) > 0.9:
        q = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(direction, q))
    v = np.cross(direction, u)
    stations = [(-0.40, WIRE_RADIUS), (0.015, WIRE_RADIUS),
                (0.08, NECK_RADIUS), (0.22, NECK_RADIUS),
                (0.34, WIRE_RADIUS), (carrier_length, WIRE_RADIUS)]
    sides = 16
    angles = np.arange(sides) * 2 * math.pi / sides
    circles = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    vertices = np.concatenate([point + direction * s + radius * circles
                               for s, radius in stations])
    faces = []
    for i in range(len(stations) - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = (i + 1) * sides + (j + 1) % sides, (i + 1) * sides + j
            faces.extend(((a, b, c), (a, c, d)))
    vertices = np.vstack([vertices, point + direction * stations[0][0],
                          point + direction * stations[-1][0]])
    for j in range(sides):
        faces.append((len(vertices) - 2, (j + 1) % sides, j))
        a = (len(stations) - 1) * sides
        faces.append((len(vertices) - 1, a + j, a + (j + 1) % sides))
    endpoint = _mesh_manifold(vertices, faces)
    neck = _rod(point + direction * 0.08, point + direction * 0.22, NECK_RADIUS)
    seating = {'foot_flatten_applied': False,
               'original_first_material_y_mm': float(endpoint.bounding_box()[1])}
    if receiving_layer is not None:
        seat = endpoint ^ receiving_layer
        if seat.volume() > .005:
            original_contact_y = float(seat.bounding_box()[1])
            floor_y = original_contact_y + .02
            neck_first_y = float(neck.bounding_box()[1])
            seating.update(original_first_shell_contact_y_mm=original_contact_y,
                           neck_first_material_y_mm=neck_first_y)
            # Only flatten the buried foot.  Keep a positive separation
            # below the complete narrow waist so its specified diameter,
            # length and deliberate fracture plane are not changed.
            if floor_y < neck_first_y - .02:
                previous_volume = float(endpoint.volume())
                endpoint = endpoint.trim_by_plane((0.,1.,0.), floor_y)
                seating.update(foot_flatten_applied=True,
                               foot_flatten_height_y_mm=floor_y,
                               removed_foot_volume_mm3=previous_volume-float(endpoint.volume()),
                               separation_below_full_neck_mm=neck_first_y-floor_y)
    seating['first_material_y_mm'] = float(endpoint.bounding_box()[1])
    if receiving_layer is not None:
        seat = endpoint ^ receiving_layer
        if seat.volume() > 0:
            seating['first_shell_contact_y_mm'] = float(seat.bounding_box()[1])
    stump = endpoint.trim_by_plane(-direction, float(np.dot(-direction, point + direction * 0.20)))
    return endpoint, neck, stump, seating


def _tube_path(points, radius=WIRE_RADIUS):
    pieces = [_rod(a, b, radius) for a, b in zip(points[:-1], points[1:])]
    pieces.extend(mf.Manifold.sphere(radius, 12).translate(p) for p in points)
    return _union(pieces)


def _tool_corridor(neck_center, radius):
    entry = np.array([neck_center[0], neck_center[1], radius + 4.0])
    shaft_end = neck_center + [0, 0, 0.55]
    tip_end = neck_center + [0, 0, 0.13]
    shaft = _rod(shaft_end, entry, 0.25)
    tip = mf.Manifold.cylinder(0.42, 0.10, 0.25, 24).translate(tip_end)
    return shaft + tip, entry, shaft_end, tip_end


def _make_release_wire(specs, host, azimuth, number, outer_radius, bow_sign=None, carrier_length=0.62, bow_size=0.35, receiving_layers=None):
    endpoints, necks, stumps, details, corridors = [], [], [], [], []
    q_points = []
    for layer in (host, host + 1):
        point, inward, theta = _window_attachment(specs[layer], azimuth)
        receiving = receiving_layers[layer] if receiving_layers is not None else None
        endpoint, neck, stump, seating = _profiled_wire_endpoint(point, inward, carrier_length, receiving)
        center = point + inward * 0.15
        corridor, entry, shaft_end, tip_end = _tool_corridor(center, outer_radius)
        endpoints.append(endpoint)
        necks.append(neck)
        stumps.append((layer, stump))
        corridors.append(corridor)
        q_points.append(point + inward * carrier_length)
        details.append({
            'layer_index': layer, 'attachment_point': point.tolist(),
            'inward_tangent': inward.tolist(), 'neck_center': center.tolist(),
            'nominal_break_plane_point': center.tolist(),
            'nominal_exposed_residue_mm': 0.15, 'maximum_residue_mm': 0.20,
            'embedded_anchor_length_mm': 0.40, 'carrier_length_mm': carrier_length,
            'build_seating': seating,
            'neck_diameter_mm': 2 * NECK_RADIUS, 'neck_length_mm': 0.14,
            'tool_entry': entry.tolist(), 'tool_shaft_end': shaft_end.tolist(),
            'tool_tip_end': tip_end.tolist(), 'tool_contact_point': center.tolist(),
            'tool_diameter_mm': 0.50, 'tool_tip_diameter_mm': 0.20,
            'tool_taper_length_mm': 0.42,
            'tool_contact': 'shaft and tapered tip stop before intended neck cutting contact',
        })
    q0, q1 = q_points
    if bow_sign is None:
        bow_sign = 1 if host % 2 else -1
    tangent = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0]) * bow_sign
    c0 = q0 + [0, 0, 0.60] + tangent * bow_size
    c1 = q1 + [0, 0, 0.75] + tangent * bow_size
    # Preserve the outward grip bow without a downward lobe that starts
    # in air.  The centreline progresses monotonically between its two
    # shell-rooted carriers in the local +Y build direction.
    c0[1] = q0[1] + (q1[1] - q0[1]) / 3
    c1[1] = q0[1] + 2 * (q1[1] - q0[1]) / 3
    t = np.linspace(0, 1, 17)[:, None]
    path = (1-t)**3 * q0 + 3*(1-t)**2*t*c0 + 3*(1-t)*t*t*c1 + t**3*q1
    # Convex capsules share identical complete end polyhedra.  A union
    # of cylinders with low-resolution end spheres leaves tiny cylinder
    # corners outside those spheres; those corners can begin as separate
    # islands even when the centreline is monotone in Y.
    nodes = [mf.Manifold.sphere(WIRE_RADIUS, 16).translate(p) for p in path]
    body_pieces = [mf.Manifold.batch_hull((a,b)) for a,b in zip(nodes[:-1],nodes[1:])]
    body = _union(body_pieces)
    model = _union(endpoints + [body])
    metadata = {
        '_endpoint_models': endpoints,
        '_body_convex_parts': body_pieces,
        'wire_index': number, 'host_index': host, 'moving_index': host + 1,
        'kind': 'front_window',
        'azimuth_degrees': azimuth / DEG,
        'grip_bow_sign': bow_sign, 'grip_tangent_bow_mm': bow_size,
        'carrier_length_mm': carrier_length,
        'minimum_carrier_build_direction_component': 0.75,
        'grip_curve_build_y_monotonic': bool(np.all(np.diff(path[:,1]) >= -1e-9)),
        'body_diameter_mm': 2 * WIRE_RADIUS, 'neck_diameter_mm': 2 * NECK_RADIUS, 'neck_length_mm': 0.14,
        'endpoints': details, 'grip_point': path[8].tolist(),
        'grip_curve_points': path.tolist(),
        'release_order': 'deepest interface first, then outward; one front wire then left and right bottom starters per interface',
    }
    return model, necks, stumps, corridors, body, metadata


def _front_wire_mechanism_clearance(candidate, permanent, outer_radius):
    """Keep the whole waist free and certify the actual straight pull.

    The generated body is a union of convex capsules.  Split each detached
    endpoint at its profile stations; every resulting frustum is convex.
    Translating each convex piece is exactly its hull with the translated
    copy.  This avoids filling the open bend between the wire and its foot.
    The separate mesh audit later certifies the final exported geometry by
    triangle sweeps, without using these construction pieces.
    """
    wire, necks, stumps, corridors, body, detail = candidate
    pieces = list(detail['_body_convex_parts'])
    nominal_stumps, detached = [], []
    waist_rows = []
    for index, (endpoint, endpoint_model) in enumerate(zip(detail['endpoints'], detail['_endpoint_models'])):
        point = np.asarray(endpoint['attachment_point'])
        direction = np.asarray(endpoint['inward_tangent'])
        guard = _rod(point + direction * .0801, point + direction * .2199, .30)
        foreign = [body] + [model for j,model in enumerate(detail['_endpoint_models']) if j != index]
        intrusion = max(0., float((_union(foreign) ^ guard).volume()))
        waist_rows.append(dict(layer_index=endpoint['layer_index'],
                               guarded_waist_station_range_mm=[.0801,.2199],
                               guard_radius_mm=.30,
                               foreign_material_volume_mm3=intrusion))
        if intrusion > 1e-10:
            return dict(passed=False, reason='curve re-enters a constant waist',
                        waist_checks=waist_rows)
        cut = float(np.dot(direction, point + direction * .15))
        nominal_stumps.append(endpoint_model.trim_by_plane(-direction, -cut))
        end = endpoint_model.trim_by_plane(direction, cut)
        detached.append(end)
        # Three convex solids: constant waist, expanding cone, carrier.
        for lo, hi in ((.15,.22), (.22,.34), (.34,detail['carrier_length_mm']+.001)):
            piece = end.trim_by_plane(direction, float(np.dot(direction,point+direction*lo)))
            piece = piece.trim_by_plane(-direction, float(np.dot(-direction,point+direction*hi)))
            if piece.volume() > 1e-12:
                pieces.append(piece)
    removable = _union([body] + detached)
    obstacles = _union([permanent] + nominal_stumps)
    original_overlap = max(0.,float((removable ^ obstacles).volume()))
    if original_overlap > 1e-10:
        return dict(passed=False, reason='detached wire already intersects retained material',
                    initial_obstruction_volume_mm3=original_overlap,
                    waist_checks=waist_rows)
    delta = np.array([0.,0.,2*outer_radius+4.])
    total_overlap = 0.
    for index, piece in enumerate(pieces):
        swept = mf.Manifold.batch_hull((piece,piece.translate(delta)))
        overlap = max(0.,float((swept ^ obstacles).volume()))
        total_overlap += overlap
        if total_overlap > 1e-10:
            return dict(passed=False, reason='a detached convex piece strikes retained material during pull',
                        failing_convex_piece=index,
                        initial_obstruction_volume_mm3=original_overlap,
                        sum_convex_sweep_obstruction_volume_mm3=total_overlap,
                        waist_checks=waist_rows)
    return dict(passed=True, waist_checks=waist_rows,
                initial_obstruction_volume_mm3=original_overlap,
                convex_sweep_pieces_checked=len(pieces),
                sum_convex_sweep_obstruction_volume_mm3=total_overlap,
                translation_mm=float(delta[2]), numerical_volume_tolerance_mm3=1e-10,
                scope='Actual generated convex pieces versus retained layers and own nominal stumps; inter-wire clearance separately certified by complete XY projections')


def _front_wire_prefix_seats(wire, receiving_layers, step=0.05):
    """Reject a front tie if any sampled build prefix has no shell seat.

    This is a candidate-selection check. The independent full-height audit
    subsequently checks the complete assembly and critical birth heights.
    """
    mesh = wire.to_mesh64()
    vertices = np.asarray(mesh.vert_properties[:, :3])
    lower, upper = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    faces = np.asarray(mesh.tri_verts)
    neighbor_min = np.full(len(vertices), np.inf)
    for a, b in ((0,1), (1,2), (2,0)):
        np.minimum.at(neighbor_min, faces[:,a], vertices[faces[:,b],1])
        np.minimum.at(neighbor_min, faces[:,b], vertices[faces[:,a],1])
    birth_levels = sorted(set(float(y) for y in vertices[
        vertices[:,1] <= neighbor_min + 1e-9, 1]))
    heights = sorted(set([lower + 0.001, lower + 0.01, upper + 0.001]
                         + list(np.arange(lower + step, upper, step))
                         + [y + offset for y in birth_levels for offset in (.0001,.001,.005,.025)]))
    receiving = _union(receiving_layers)
    full_contact = wire ^ receiving
    first_contact = (float(full_contact.bounding_box()[1])
                     if full_contact.volume() > 0 else math.inf)
    minimum_contact = math.inf
    for height in heights:
        prefix = wire.trim_by_plane((0., -1., 0.), -height)
        for piece in prefix.decompose():
            if piece.volume() <= 1e-12:
                continue
            contact = max(0., float((piece ^ receiving).volume()))
            minimum_contact = min(minimum_contact, contact)
            if contact <= 1e-12:
                return dict(passed=False, failed_height_y_mm=height,
                            unsupported_volume_mm3=float(piece.volume()),
                            unsupported_bounds_mm=list(piece.bounding_box()))
    return dict(passed=True, heights_checked=len(heights), step_mm=step,
                critical_surface_minima_checked=len(birth_levels),
                first_material_y_mm=lower,
                first_shell_contact_y_mm=first_contact,
                unsupported_lead_before_shell_contact_mm=max(0.,first_contact-lower),
                critical_minimum_offsets_mm=[.0001,.001,.005,.025],
                first_sample_above_material_mm=0.0001,
                minimum_prefix_shell_contact_mm3=minimum_contact,
                scope='Actual temporary-wire prefixes seated in receiving layers; full assembly checked independently')



def _elliptical_y_prism(x, y0, y1, width, thickness, z=0., segments=24):
    """Closed elliptical prism with a straight local Y centreline."""
    shape = mf.Manifold.cylinder(y1-y0, 1., 1., segments).scale((width/2, thickness/2, 1))
    return shape.transform(_xf(np.array([[1.,0.,0.], [0.,0.,1.], [0.,-1.,0.]]), (x,y0,z)))


def _profiled_starter_endpoint(point, direction, width, thickness, body_width, body_thickness):
    direction = _unit(direction)
    stations = [(-.18,body_width,body_thickness), (.015,body_width,body_thickness),
                (.08,width,thickness), (.22,width,thickness),
                (.34,body_width,body_thickness), (.45,body_width,body_thickness)]
    angles = np.arange(24)*2*math.pi/24
    vertices = np.concatenate([point + direction*s +
        np.column_stack((w/2*np.cos(angles),np.zeros(24),t/2*np.sin(angles)))
        for s,w,t in stations])
    faces=[]
    for i in range(len(stations)-1):
        for j in range(24):
            a,b=i*24+j,i*24+(j+1)%24
            c,d=(i+1)*24+(j+1)%24,(i+1)*24+j
            faces.extend(((a,b,c),(a,c,d)))
    vertices=np.vstack([vertices,point+direction*stations[0][0],point+direction*stations[-1][0]])
    for j in range(24):
        faces.append((len(vertices)-2,(j+1)%24,j))
        a=(len(stations)-1)*24
        faces.append((len(vertices)-1,a+j,a+(j+1)%24))
    endpoint=_mesh_manifold(vertices,faces)
    ya,yb=sorted((float(point[1]+direction[1]*.08),float(point[1]+direction[1]*.22)))
    neck=_elliptical_y_prism(float(point[0]),ya,yb,width,thickness)
    stump=endpoint.trim_by_plane(-direction,float(np.dot(-direction,point+direction*.20)))
    return endpoint,neck,stump


def _make_bottom_starters(specs, host, outer_radius):
    """Two independent first-layer paths, flat receiving lands, accessible grips.

    The moving sphere's original bottom cap is trimmed separately.  Its two
    new first-material lands are each directly backed by one already-built
    starter. A separate fin provides a grip without bending the load stem.
    """
    moving=host+1
    ro,ri=specs[host]['radius_mm'],specs[moving]['radius_mm']
    parent_inner=ro-specs[host]['thickness_mm']
    width,thickness,body_width,body_thickness=.74,.54,.82,.76
    parent_y,child_y=-parent_inner+.08,-ri-.04
    candidates,landings=[],[]
    for side in (-1,1):
        x=side*.56
        parent_land=_elliptical_y_prism(x,-parent_inner-.20,parent_y,body_width+.04,body_thickness+.04)
        child_land=_elliptical_y_prism(x,child_y,-ri+.20,body_width+.04,body_thickness+.04)
        landings.extend([{'model':parent_land,'layer_index':host,'host_index':host,'side':side,'role':'parent_inner_land'},
                         {'model':child_land,'layer_index':moving,'host_index':host,'side':side,'role':'moving_first_layer_land'}])
        points=[np.array([x,parent_y,0.]),np.array([x,child_y,0.])]
        directions=[np.array([0.,1.,0.]),np.array([0.,-1.,0.])]
        endpoints,necks,stumps,corridors,details,qs=[],[],[],[],[],[]
        for layer,point,inward in zip((host,moving),points,directions):
            endpoint,neck,stump=_profiled_starter_endpoint(point,inward,width,thickness,body_width,body_thickness)
            center=point+.15*inward
            tip_end=center+[0.,0.,.13]
            needle_end=center+[0.,0.,1.50]
            shaft_end=center+[0.,0.,2.10]
            entry=np.array([center[0],center[1],outer_radius+4.])
            needle=_rod(tip_end,needle_end,.10)
            taper=mf.Manifold.cylinder(.60,.10,.25,24).translate(needle_end)
            shaft=_rod(shaft_end,entry,.25)
            corridor=_union([needle,taper,shaft])
            endpoints.append(endpoint);necks.append(neck);stumps.append((layer,stump))
            corridors.append(corridor);qs.append(point+.45*inward)
            details.append({
                'layer_index':layer,'attachment_point':point.tolist(),
                'inward_tangent':inward.tolist(),'neck_center':center.tolist(),
                'nominal_break_plane_point':center.tolist(),
                'nominal_exposed_residue_mm':.15,'maximum_residue_mm':.20,
                'embedded_anchor_length_mm':.18,'carrier_length_mm':.45,
                'neck_profile':'ellipse','neck_width_mm':width,'neck_thickness_mm':thickness,
                'neck_width_axis':[1.,0.,0.],'neck_thickness_axis':[0.,0.,1.],
                'neck_diameter_mm':width,'neck_diameter_definition':'major axis of elliptical section',
                'neck_length_mm':.14,'neck_section_segments':24,'neck_profile_segments':24,
                'nominal_neck_polygon_area_mm2':12*(width/2)*(thickness/2)*math.sin(2*math.pi/24),
                'neck_area_mm2':12*(width/2)*(thickness/2)*math.sin(2*math.pi/24),
                'tool_entry':entry.tolist(),'tool_shaft_end':shaft_end.tolist(),
                'tool_needle_end':needle_end.tolist(),'tool_tip_end':tip_end.tolist(),
                'tool_contact_point':center.tolist(),'tool_diameter_mm':.50,
                'tool_tip_diameter_mm':.20,'tool_taper_length_mm':.60,'tool_needle_length_mm':1.37,
                'tool_contact':'0.20 mm needle reaches the waist from +Z; wider shaft starts at z=2.10 mm',
            })
        q0,q1=qs
        # Finite overlaps avoid precision-dependent face-only joints at
        # either end of the full-section stem, far beyond both waists.
        stem=_elliptical_y_prism(x,float(q0[1]-.025),float(q1[1]+.025),body_width,body_thickness)
        mid=(q0[1]+q1[1])/2
        grip_tip=np.array([x,mid,.68])
        # A growing triangular fin has no isolated first print material.
        profile=mf.CrossSection([np.array([[q0[1],body_thickness/2-.07],
                                          [q1[1],body_thickness/2-.07],
                                          [grip_tip[1],grip_tip[2]]])])
        grip=mf.Manifold.extrude(profile,.40).transform(
            _xf(np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]]),(x-.20,0.,0.)))
        body=_union([stem,grip])
        model=_union(endpoints+[body])
        metadata={
            '_endpoint_models':endpoints,'_structural_stem_model':stem,'_grip_model':grip,
            'wire_index':-1,'host_index':host,'moving_index':moving,'side':side,
            'kind':'bottom_starter','azimuth_degrees':-90.,
            'body_profile':'ellipse','body_width_mm':body_width,'body_thickness_mm':body_thickness,
            'body_diameter_mm':body_width,'neck_profile':'ellipse',
            'neck_width_mm':width,'neck_thickness_mm':thickness,
            'neck_diameter_mm':width,'neck_length_mm':.14,
            'endpoints':details,'grip_point':grip_tip.tolist(),
            'grip_curve_points':[q0.tolist(),q1.tolist()],
            'structural_stem_centerline':[q0.tolist(),q1.tolist()],
            'structural_stem_eccentricity_mm':0.,'grip_is_in_primary_load_path':False,
            'release_order':'deepest interface first; one front wire, then left and right bottom starters',
            'build_direction':[0.,1.,0.],
            'initial_moving_layer_minimum_build_y_mm':child_y,
            'bond_reaches_below_initial_moving_build_y_mm':.18,
            'starter_lower_attachment_y_mm':parent_y,
            'moving_original_pole_y_mm':-ri,'moving_shell_start_after_cap_trim_y_mm':-ri+.04,
            'paired_starter_x_offsets_mm':[-.56,.56],
            'service_opening_width_mm':2.40,'service_opening_starts_z_mm':.38,
            'service_opening_applies_to_layer_indices':list(range(host+1)),
        }
        candidates.append((model,necks,stumps,corridors,body,metadata))
    # A continuous slot meets the front rosette without fragile strips
    # between separate service windows. Its side ribs carry the opening.
    window_low, window_high = parent_y - .17, -1.50
    corner=.12
    top_extension=window_high-(child_y+.17)
    opening_profile=mf.CrossSection.square((2.40-2*corner,window_high-window_low-2*corner)).offset(corner,circular_segments=24)
    opening=mf.Manifold.extrude(opening_profile,outer_radius+4.-.38).translate((-1.20+corner,window_low+corner,.38))
    for candidate in candidates:
        candidate[5]['service_opening_upper_extension_mm']=top_extension
        candidate[5]['service_opening_y_bounds_mm']=[window_low,window_high]
        candidate[5]['service_opening_continuous_to_front_rosette']=True
    return candidates,opening,landings


def _collar(radius):
    count = 360
    angle = np.linspace(0, 2 * math.pi, count, endpoint=False)
    outer_r = radius + 0.59 + 0.20 * np.cos(30 * angle)
    inner_r = radius - 0.40
    outer = np.column_stack([outer_r * np.cos(angle), outer_r * np.sin(angle)])
    inner = np.column_stack([inner_r * np.cos(angle[::-1]), inner_r * np.sin(angle[::-1])])
    profile = mf.CrossSection([outer, inner])
    ring = mf.Manifold.extrude(profile, 0.92).translate((0, 0, -0.46))
    ornaments = []
    for sign in (-1, 1):
        rr = radius + 0.31 + 0.10 * np.cos(30 * angle)
        points = np.column_stack([rr * np.cos(angle), rr * np.sin(angle),
                                  np.full(count, sign * 0.445)])
        radial = np.tile([0, 0, sign], (count, 1))
        ornaments.append(_closed_sweep(points, radial, half_width=0.16,
                                        height=0.23, embed=0.12,
                                        stitch_periods=60))
    return _union([ring] + ornaments)


def _build_details(radius, layers, verbose=False):
    radius, layers = float(radius), int(layers)
    if radius < 8.5 or layers < 2:
        raise ValueError('Use an outer radius >=8.5 mm and at least two layers')
    radii = np.linspace(radius, 3.8, layers)
    if np.min(-np.diff(radii)) < 1.74:
        raise ValueError('The requested layer count leaves insufficient clear spacing')
    shells, specs = [], []
    started = time.monotonic()
    for index, r in enumerate(radii):
        if verbose:
            print(f'R{radius:g} L{layers}: shell {index + 1}/{layers}, r={r:.2f}', flush=True)
        shell, spec = _build_layer(float(r), index, index == 0)
        if index:
            # Avoid an isolated original sphere pole between the two pads.
            # Both pads already have positive parent-supported area when
            # this retained spherical cap first begins at +0.08 mm.
            shell = shell.trim_by_plane((0.,1.,0.),-float(r)+.04)
            spec['original_bottom_cap_trim_mm'] = .04
            spec['first_material_pad_offset_below_nominal_pole_mm'] = .04
        shells.append(shell)
        specs.append(spec)
    additions, cuts, features, bearings = _bearing_parts(specs)
    local_start_trims = []
    if layers == 4 and abs(radius - 11.) < 1e-9:
        # Prefix inspection locates this inherited downward petal tip's
        # attachment at y=-3.5698093 mm. Retain it only above that join.
        tip_cut = mf.Manifold.cube((1.05, .0712, .82)).translate((4.27, -3.64, 5.71))
        cuts[1].append(tip_cut)
        local_start_trims.append(dict(layer_index=1,
            box_min_mm=[4.27,-3.64,5.71], box_max_mm=[5.32,-3.5688,6.53],
            original_tip_first_y_mm=-3.6279074,
            measured_connection_y_mm=-3.5698093,
            flatten_height_mm=.0591074,
            actual_removed_shell_volume_mm3=float((shells[1] ^ tip_cut).volume()),
            role='Flatten a local downward petal tip until its first material joins the earlier skin'))
    starter_candidates, service_openings, starter_landings = {}, [], []
    for host in range(layers - 1):
        pair, opening, landings = _make_bottom_starters(specs, host, radius)
        starter_candidates[host] = pair
        service_openings.append(opening)
        starter_landings.extend(landings)
        for landing in landings:
            additions[landing['layer_index']].append(landing['model'])
        for ancestor in range(host + 1):
            cuts[ancestor].append(opening)
    from service_edge_ribs import build_service_edge_ribs
    service_edge_ribs = []
    for index in range(layers - 1):
        ribs, rib_meta = build_service_edge_ribs(specs[index]['radius_mm'], x_offset=1.77, tube_radius=.62, radial_inset=.72, upper_radial_inset=.64, service_half_width=1.20, top_y=-1.25)
        r = specs[index]['radius_mm']
        segments = 80 if r >= 9 else (64 if r >= 6 else 48)
        skin = mf.Manifold.sphere(r, segments) - mf.Manifold.sphere(r-SHELL_THICKNESS, segments)
        bands = []
        for side in (-1, 1):
            x0 = 1.20 if side > 0 else -2.25
            region = mf.Manifold.cube((1.05, r-1.25, r+1)).translate((x0, -r, 0))
            bands.append(skin ^ region)
        backing = _union(bands)
        ribs = ribs + backing
        rib_meta.update(full_thickness_backing_band=True,
                        band_x_abs_interval_mm=[1.20, 2.25],
                        band_radial_thickness_mm=SHELL_THICKNESS,
                        band_top_y_mm=-1.25,
                        backing_band_volume_mm3=backing.volume(),
                        role='Full-thickness backing of service edges, continuous from the earlier lower spherical band')
        shells[index] = shells[index] + ribs
        service_edge_ribs.append(dict(model=ribs, layer_index=index, **rib_meta))
    shells[0] = shells[0] + _collar(radius)
    from printable_shell_starts import flatten_disconnected_starts
    free_layers, shell_only = [], []
    for index, shell in enumerate(shells):
        base = mf.Manifold.batch_boolean([shell] + cuts[index], mf.OpType.Subtract) if cuts[index] else shell
        free = _union([shell] + additions[index])
        if cuts[index]:
            free = mf.Manifold.batch_boolean([free] + cuts[index], mf.OpType.Subtract)
        if len(free.decompose()) != 1 or free.status() != mf.Error.NoError:
            raise RuntimeError(f'Bearing-equipped layer {index} is not one solid')
        if index:
            protected = [feature['model'] for feature in features
                         if feature['layer_index'] == index]
            free, trims, tip_cutters = flatten_disconnected_starts(
                free, layer_index=index, protected_parts=protected)
            if tip_cutters:
                base = mf.Manifold.batch_boolean([base] + tip_cutters, mf.OpType.Subtract)
                local_start_trims.extend(trims)
            if verbose:
                removed = sum(row['actual_removed_shell_volume_mm3'] for row in trims)
                print(f'R{radius:g} L{layers}: layer {index} start trims={len(trims)}, '
                      f'removed={removed:.9f} mm3', flush=True)
        shell_only.append(base)
        free_layers.append(free)
    permanent = _union(free_layers)
    if len(permanent.decompose()) != layers:
        overlaps = []
        for i in range(layers):
            for j in range(i + 1, layers):
                overlaps.append((i, j, (free_layers[i] ^ free_layers[j]).volume()))
        raise RuntimeError(f'Released bearings collide: {overlaps}')
    wires, wire_details, wire_parts = [], [], []
    all_stumps = [[] for _ in specs]
    all_nominal_stumps = [[] for _ in specs]
    previous_corridors = []
    accepted_projections = []
    all_starters = [value for pair in starter_candidates.values() for value in pair]
    all_starter_corridors = [corridor for value in all_starters for corridor in value[3]]
    all_starter_models = [value[0] for value in all_starters]
    starter_projections = [mf.CrossSection(part.project().to_polygons()) for part in all_starter_models]
    for candidate in all_starters:
        host=candidate[5]['host_index']
        if (candidate[4] ^ permanent).volume() > 1e-8:
            raise RuntimeError(f'Bottom starter body touches permanent shell at interface {host}')
        for endpoint_index, corridor in enumerate(candidate[3]):
            if (corridor ^ permanent).volume() > 1e-8:
                raise RuntimeError(f'Bottom starter tool {host}/{endpoint_index} blocked by permanent shell: {(corridor ^ permanent).volume()}')
        for endpoint, endpoint_model in zip(candidate[5]['endpoints'],candidate[5]['_endpoint_models']):
            contact=(endpoint_model ^ free_layers[endpoint['layer_index']]).volume()
            endpoint['measured_embedded_contact_volume_mm3']=float(contact)
            if contact < .005:
                raise RuntimeError(f'Bottom starter lacks finite receiving contact: host {host}, side {candidate[5]["side"]}, {contact}')

    def accept_wire(candidate):
        wire, necks, stumps, corridors, body, detail = candidate
        nominal_stumps = []
        for (layer, stump), endpoint in zip(stumps, detail['endpoints']):
            all_stumps[layer].append(stump)
            point = np.asarray(endpoint['attachment_point'])
            inward = np.asarray(endpoint['inward_tangent'])
            nominal = stump.trim_by_plane(-inward, float(np.dot(-inward, point + inward * 0.15)))
            nominal_stumps.append(nominal)
            all_nominal_stumps[layer].append(nominal)
            vm = np.asarray(stump.to_mesh().vert_properties[:, :3], dtype=float)
            endpoint['maximum_residue_radial_bound_mm'] = float(np.linalg.norm(vm, axis=1).max())
        detached_ends = []
        for endpoint_model, endpoint in zip(detail.pop('_endpoint_models'), detail['endpoints']):
            inward = np.asarray(endpoint['inward_tangent'])
            plane_point = np.asarray(endpoint['nominal_break_plane_point'])
            detached_ends.append(endpoint_model.trim_by_plane(inward, float(np.dot(inward, plane_point))))
        removable = _union([body] + detached_ends)
        for nominal_stump in nominal_stumps:
            if (removable ^ nominal_stump).volume() > 1e-9:
                raise RuntimeError('Direct positive/negative fracture partition overlaps')
        removable_pieces = removable.decompose()
        if len(removable_pieces) != 1:
            meaningful = [part for part in removable_pieces if part.volume() > 1e-9]
            if len(meaningful) == 1:
                removable = meaningful[0]
            else:
                raise RuntimeError(f'Detached wire not one piece: {[p.volume() for p in removable_pieces]}')
        detail['extraction_vector'] = [0.0, 0.0, 1.0]
        detail['extraction_distance_mm'] = 2 * radius + 4.0
        detail['extraction_state'] = 'nominal 0.15 mm stumps; earlier/deeper wires already removed'
        detail['sequence_index'] = len(wires)
        detail['wire_index'] = len(wires)
        wires.append(wire); wire_details.append(detail)
        accepted_projections.append(mf.CrossSection(wire.project().to_polygons()))
        extra_parts={}
        for extra in ('structural_stem','grip'):
            if '_'+extra+'_model' in detail:
                extra_parts[extra]=detail.pop('_'+extra+'_model')
        wire_parts.append({'model': wire, 'necks': necks, 'stumps': stumps,
                           'nominal_stumps': nominal_stumps, 'removable': removable,
                           'corridors': corridors, 'body': body, **extra_parts})
        previous_corridors.extend(corridors)
        if verbose:
            print(f'R{radius:g} L{layers}: accessible {detail["kind"]} wire {len(wires)} host {detail["host_index"]} phi {detail["azimuth_degrees"]:.1f}', flush=True)

    front = _front_window_frame(specs[0])
    reference_angle = math.atan2(front[1][1], front[1][0])
    for host in reversed(range(layers - 1)):
        for branch, petal in enumerate((1,) if host % 2 else (0,)):
            desired = reference_angle + (petal * 72 + host * (3.0 - 0.12 / 5 / DEG)) * DEG
            accepted = None
            failures = []
            default_bow = 1 if host % 2 else -1
            offsets = []
            for local in (0, 6, -6, 12, -12, 18, -18, 24, -24, 30, -30, 36):
                for sector in (0, 72, -72, 144, -144):
                    offset = (sector + local) % 360
                    if offset not in offsets:
                        offsets.append(offset)
            # The final outer tie must share the same aperture with
            # several already-placed inner ties.  Refine the azimuth
            # between the coarse petal candidates before rejecting the
            # layout; all physical clearance gates remain unchanged.
            offsets.extend(offset for offset in range(360) if offset not in offsets)
            options = [(offset, bow) for offset in offsets for bow in (default_bow, -default_bow)]
            for offset, bow in options:
                if math.sin(desired + offset * DEG) > -0.15:
                    continue
                candidate = _make_release_wire(specs, host, desired + offset * DEG, len(wires), radius, bow,
                                               receiving_layers=free_layers)
                wire, necks, stumps, corridors, body, detail = candidate
                # Enlarging the service slot can remove an old attachment
                # location.  Require a finite seat in its intended layer,
                # and prohibit the endpoint from joining another layer.
                endpoints_ok = True
                for endpoint, endpoint_model in zip(detail['endpoints'], detail['_endpoint_models']):
                    receiving = int(endpoint['layer_index'])
                    contact = float((endpoint_model ^ free_layers[receiving]).volume())
                    endpoint['measured_embedded_contact_volume_mm3'] = contact
                    if contact < .005 or any((endpoint_model ^ other).volume() > 1e-8
                                            for j, other in enumerate(free_layers) if j != receiving):
                        endpoints_ok = False
                        break
                if not endpoints_ok:
                    failures.append((offset, 'no isolated structural endpoint seat'))
                    continue
                if (body ^ permanent).volume() > 1e-8:
                    failures.append((offset, 'body touches permanent shell'))
                    continue
                if any((corridor ^ permanent).volume() > 1e-8 for corridor in corridors):
                    failures.append((offset, 'tool blocked by shell'))
                    continue
                if any((wire ^ other).volume() > 1e-8 for other in wires + all_starter_models):
                    failures.append((offset, 'wires join'))
                    continue
                projection = mf.CrossSection(wire.project().to_polygons())
                guard = projection.offset(0.205, circular_segments=32)
                if any((guard ^ other).area() > 1e-10 for other in accepted_projections + starter_projections):
                    failures.append((offset, 'less than 0.20 mm projected extraction spacing'))
                    continue
                if any((corridor ^ wire).volume() > 1e-8 for corridor in previous_corridors + all_starter_corridors):
                    failures.append((offset, 'blocks earlier tool corridor'))
                    continue
                mechanism = _front_wire_mechanism_clearance(candidate, permanent, radius)
                if not mechanism['passed']:
                    failures.append((offset, mechanism['reason']))
                    continue
                startup = _front_wire_prefix_seats(wire, [free_layers[host], free_layers[host+1]])
                if not startup['passed']:
                    failures.append((offset, 'front support has an unseated build prefix'))
                    continue
                detail['sampled_build_prefix_connectivity'] = startup
                detail['generated_piece_release_clearance'] = mechanism
                detail.pop('_body_convex_parts')
                accepted = candidate
                break
            if accepted is None:
                raise RuntimeError(f'No accessible wire for interface {host}, branch {branch}: {failures}')
            accept_wire(accepted)
        for starter in starter_candidates[host]:
            accept_wire(starter)
    stump_layers = [_union([layer] + stumps) for layer, stumps in zip(free_layers, all_stumps)]
    nominal_stump_layers = [_union([layer] + stumps) for layer, stumps in zip(free_layers, all_nominal_stumps)]
    released = _union(stump_layers)
    if len(released.decompose()) != layers:
        raise RuntimeError('Conservative post-break stumps join released layers')
    assembled = _union(free_layers + wires)
    components = len(assembled.decompose())
    if assembled.status() != mf.Error.NoError or components != 1:
        raise RuntimeError(f'Supported assembly invalid: {assembled.status()}, {components} components')
    metadata = {
        'name': f'sphere_R{radius:g}_L{layers}',
        'units': 'mm',
        'cache_coordinate_precision': 'float64; final print export is separately canonicalized at float32 precision',
        'outer_nominal_radius_mm': radius,
        'layer_count': layers,
        'layer_radii_mm': radii.tolist(),
        'clear_radial_gaps_mm': [float(radii[i] - specs[i]['thickness_mm'] - radii[i+1])
                                 for i in range(layers - 1)],
        'layer_specs': specs,
        'local_coordinates': {'outward': '+Z', 'up': '+Y', 'tangent': '+X'},
        'collar_plane': 'z=0',
        'collar_nominal_max_radius_mm': radius + 0.79,
        'version': 5,
        'axis_unit_vector': PIVOT_AXIS.tolist(),
        'fixation': 'horizontal opposed blind journals; one front wire and two straight bottom starters per interface, all with two breakaway necks',
        'permanent_interlayer_bridges': 0,
        'rotational_degrees_of_freedom_per_interface': 1,
        'permitted_rotation_degrees': 360,
        'bearings': bearings,
        'release_wires': wire_details,
        'wire_count': len(wires),
        'neck_count': 2 * len(wires),
        'wire_body_diameter_mm': 2 * WIRE_RADIUS,
        'neck_diameter_mm': 2 * NECK_RADIUS,
        'neck_length_mm': 0.14,
        'front_wire_neck_diameter_mm':2 * NECK_RADIUS,
        'bottom_starter_neck_profile':'ellipse, wide along local X and thin along local Z',
        'outermost_starter_neck_width_thickness_mm':[.74,.54],
        'other_starter_neck_width_thickness_mm':[.74,.54],
        'bottom_starters_per_interface':2,
        'receiving_lands_begin_before_trimmed_sphere_mm':.08,
        'strength_scope':'Geometric improvement and load sensitivity only; no printer-specific peel force, green strength or physical print certification.',
        'localized_print_start_trims':local_start_trims,
        'nominal_residue_mm': 0.15,
        'maximum_residue_mm': 0.20,
        'release_sequence': 'deepest interface first; one front wire then left and right bottom starters; sequence_index orders individual wire removal',
        'build_direction': [0., 1., 0.],
        'front_wire_count': layers - 1,
        'inter_wire_projected_extraction_clearance_mm': 0.20,
        'fracture_partition_construction': 'Each original endpoint is independently clipped to opposite closed half-spaces at the nominal break plane; positive endpoint pieces are united with the central grip.',
        'measured_minimum_static_inter_wire_gap_mm': min(wires[i].min_gap(wires[j], 0.5) for i in range(len(wires)) for j in range(i)),
        'partition_check_maximum_removable_nominal_stump_overlap_mm3': max(max(0., (part['removable'] ^ stump).volume()) for part in wire_parts for stump in part['nominal_stumps']),
        'bottom_starter_count': 2 * (layers - 1),
        'released_components': len(released.decompose()),
        'released_volume_mm3': float(released.volume()),
        'bare_released_components': len(permanent.decompose()),
        'triangle_count': int(assembled.num_tri()),
        'vertex_count': int(assembled.num_vert()),
        'volume_mm3': float(assembled.volume()),
        'surface_area_mm2': float(assembled.surface_area()),
        'components': components,
        'status': str(assembled.status()),
        'bounds_mm': list(assembled.bounding_box()),
        'generation_seconds': time.monotonic() - started,
    }
    parts = {'shell_only': shell_only, 'features': features, 'free_layers': free_layers,
             'stump_layers': stump_layers, 'nominal_stump_layers': nominal_stump_layers,
             'wires': wire_parts, 'released': released, 'permanent': permanent,
             'service_openings': service_openings, 'starter_landings': starter_landings,
             'service_edge_ribs': service_edge_ribs}
    return assembled, metadata, parts


def build_sphere(radius, layers):
    """Return the supported one-solid assembly and all mechanical dimensions."""
    assembled, metadata, _ = _build_details(radius, layers)
    return assembled, metadata


def _save_npz(path, model):
    mesh = model.to_mesh64()
    vertices = np.ascontiguousarray(mesh.vert_properties[:, :3], dtype=np.float64)
    faces = np.ascontiguousarray(mesh.tri_verts, dtype=np.uint32)
    triangles = vertices[faces]
    doubled_areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(vertices).all() or np.any(doubled_areas == 0):
        raise RuntimeError(f'Invalid exact cache coordinates in {path.name}')
    np.savez_compressed(path, vertices=vertices, faces=faces)
    return vertices, faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'cache' / 'spheres')
    parser.add_argument('--only', choices=('small', 'medium', 'large'))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    sizes = [('small', 9, 3), ('medium', 11, 4), ('large', 13, 5)]
    for label, radius, layers in sizes:
        if args.only and label != args.only:
            continue
        model, metadata, parts = _build_details(radius, layers, verbose=True)
        stem = metadata['name']
        vertices, faces = _save_npz(args.cache / f'{stem}.npz', model)
        shutil.copyfile(args.cache / f'{stem}.npz', args.cache / f'sphere_{layers}layers.npz')
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        metadata['trimesh_watertight'] = bool(tm.is_watertight)
        metadata['trimesh_winding_consistent'] = bool(tm.is_winding_consistent)
        metadata['trimesh_positive_volume'] = bool(tm.volume > 0)
        metadata['supported_mesh_file'] = f'sphere_{layers}layers.npz'
        metadata['released_mesh_file'] = f'sphere_{layers}layers_released.npz'
        _save_npz(args.cache / metadata['released_mesh_file'], parts['released'])
        for key, field in [('shell_only', 'shell_only_meshes'), ('free_layers', 'free_layer_meshes'),
                           ('stump_layers', 'released_with_stumps_layer_meshes'),
                           ('nominal_stump_layers', 'nominal_stump_layer_meshes')]:
            filenames = []
            for index, part in enumerate(parts[key]):
                filename = f'{stem}_{key}_{index + 1:02d}.npz'
                _save_npz(args.cache / filename, part)
                filenames.append(filename)
            metadata[field] = filenames
        metadata['service_opening_meshes'] = []
        for host, opening in enumerate(parts['service_openings']):
            filename = f'{stem}_service_opening_{host:02d}.npz'
            _save_npz(args.cache / filename, opening)
            metadata['service_opening_meshes'].append({'host_index': host, 'mesh_file': filename,
                                                     'role': 'diagnostic_void_cut'})
        metadata['bearing_feature_meshes'] = []
        for index, feature in enumerate(parts['features']):
            filename = f'{stem}_bearing_{index:02d}_{feature["role"]}.npz'
            _save_npz(args.cache / filename, feature['model'])
            description = {k: v for k, v in feature.items() if k != 'model'}
            description['mesh_file'] = filename
            metadata['bearing_feature_meshes'].append(description)
            bearing = metadata['bearings'][feature['bearing_index']]
            role_key = {'socket': 'housing_mesh_file', 'pivot': 'pin_mesh_file', 'bore_cut': 'bore_cut_mesh_file'}[feature['role']]
            bearing[role_key] = filename
        metadata['service_edge_rib_meshes']=[]
        for index, rib in enumerate(parts['service_edge_ribs']):
            filename=f'{stem}_service_edge_ribs_{index:02d}.npz'
            _save_npz(args.cache/filename,rib['model'])
            metadata['service_edge_rib_meshes'].append({**{k:v for k,v in rib.items() if k!='model'},'mesh_file':filename})
        metadata['starter_landing_meshes']=[]
        for index,landing in enumerate(parts['starter_landings']):
            filename=f'{stem}_starter_landing_{index:02d}.npz'
            _save_npz(args.cache/filename,landing['model'])
            metadata['starter_landing_meshes'].append({**{k:v for k,v in landing.items() if k!='model'},'mesh_file':filename})
        for index, (wire, detail) in enumerate(zip(parts['wires'], metadata['release_wires'])):
            for key, field in [('model', 'mesh_file'), ('removable', 'removable_mesh_file'), ('body', 'body_mesh_file')]:
                filename = f'{stem}_wire_{index:02d}_{key}.npz'
                _save_npz(args.cache / filename, wire[key])
                detail[field] = filename
            for extra in ('structural_stem','grip'):
                if extra in wire:
                    filename=f'{stem}_wire_{index:02d}_{extra}.npz'
                    _save_npz(args.cache/filename,wire[extra])
                    detail[extra+'_mesh_file']=filename
            detail['neck_mesh_files'] = []
            for endpoint_index, neck in enumerate(wire['necks']):
                filename = f'{stem}_wire_{index:02d}_neck_{endpoint_index}.npz'
                _save_npz(args.cache / filename, neck)
                detail['neck_mesh_files'].append(filename)
                endpoint = detail['endpoints'][endpoint_index]
                for key, model_part in [('stump', wire['stumps'][endpoint_index][1]),
                                        ('nominal_stump', wire['nominal_stumps'][endpoint_index]),
                                        ('tool_corridor', wire['corridors'][endpoint_index])]:
                    filename = f'{stem}_wire_{index:02d}_{key}_{endpoint_index}.npz'
                    _save_npz(args.cache / filename, model_part)
                    endpoint[key + '_mesh_file'] = filename
        (args.cache / f'{stem}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        (args.cache / f'sphere_{layers}layers.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps({key: metadata[key] for key in (
            'name', 'triangle_count', 'volume_mm3', 'components', 'status',
            'trimesh_watertight', 'trimesh_winding_consistent', 'generation_seconds')}), flush=True)


if __name__ == '__main__':
    main()

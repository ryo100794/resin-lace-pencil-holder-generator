#!/usr/bin/env python3
"""Independent early support paths and nominal load sensitivity, in mm.

The sphere generator is never imported.  This reads actual cached solids and
the declared endpoint frames.  It measures when each separate temporary tie
first joins finite parent and child material, and slices the assembled module
at the startup and receiving-land transition.  It does not estimate resin
strength, peel force, fatigue, stress concentration, or load sharing.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from audit_motion import (Mesh, frame, mesh_from_manifold, neck_measurements,
                          positive_intersection_volume, sha256, transformed,
                          union, unit)


VOLUME_EPS_MM3 = 1e-12
HEIGHT_RESOLUTION_MM = 1e-5
STARTUP_RISES_MM = (.01, .04, .079, .081, .10, .20, .24)


def prefix(solid, direction, height):
    return solid.trim_by_plane(-direction, -float(height))


def volume(solid):
    return 0. if solid.is_empty() else max(0., float(solid.volume()))


def build_bounds(solid, direction):
    vertices, _ = mesh_from_manifold(solid)
    if not len(vertices):
        return None
    coordinates = vertices@direction
    return [float(coordinates.min()), float(coordinates.max())]


def first_bond(wire, parent, child, direction):
    """A path must join both finite anchor volumes within one wire component."""
    parent_anchor = wire.manifold ^ parent.manifold
    child_anchor = wire.manifold ^ child.manifold
    parent_volume, child_volume = volume(parent_anchor), volume(child_anchor)
    result = dict(parent_anchor_volume_mm3=parent_volume,
                  moving_anchor_volume_mm3=child_volume,
                  parent_anchor_build_bounds_mm=build_bounds(parent_anchor, direction),
                  moving_anchor_build_bounds_mm=build_bounds(child_anchor, direction))
    if min(parent_volume, child_volume) <= VOLUME_EPS_MM3:
        return dict(result, passed=False, error='No finite anchor in both permanent layers')

    def joined(height):
        for component in prefix(wire.manifold, direction, height).decompose():
            if (positive_intersection_volume(component, parent_anchor) > VOLUME_EPS_MM3
                    and positive_intersection_volume(component, child_anchor) > VOLUME_EPS_MM3):
                return True
        return False

    start = float((child.vertices@direction).min())
    low = start-HEIGHT_RESOLUTION_MM
    high = float((wire.vertices@direction).max())+HEIGHT_RESOLUTION_MM
    if not joined(high):
        return dict(result, passed=False, error='Full wire does not contain a parent-to-child path')
    while high-low > HEIGHT_RESOLUTION_MM:
        mid = (low+high)/2
        if joined(mid):
            high = mid
        else:
            low = mid
    return dict(result, passed=True,
                first_finite_connection_build_coordinate_bracket_mm=[low, high],
                first_finite_connection_above_first_moving_material_bracket_mm=[low-start, high-start])


def contains_origin(polygon):
    following = np.roll(polygon, -1, axis=0)
    crossings = (polygon[:, 1] > 0) != (following[:, 1] > 0)
    p, q = polygon[crossings], following[crossings]
    return bool(np.count_nonzero(p[:, 0]-p[:, 1]*(q[:, 0]-p[:, 0])/(q[:, 1]-p[:, 1]) > 0) % 2)


def polygon_moments(polygon):
    """Signed polygon integration; x is declared width and y is thickness."""
    p = np.asarray(polygon, float)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0]*q[:, 1]-q[:, 0]*p[:, 1]
    area = .5*float(cross.sum())
    if area < 0:
        return polygon_moments(p[::-1])
    if area <= 0:
        raise ValueError('Nonpositive neck section area')
    center = np.sum((p+q)*cross[:, None], axis=0)/(6*area)
    ix = float(np.sum((p[:, 1]**2+p[:, 1]*q[:, 1]+q[:, 1]**2)*cross)/12-area*center[1]**2)
    iy = float(np.sum((p[:, 0]**2+p[:, 0]*q[:, 0]+q[:, 0]**2)*cross)/12-area*center[0]**2)
    return dict(area_mm2=area, centroid_mm=center.tolist(),
                width_mm=float(np.ptp(p[:, 0])), thickness_mm=float(np.ptp(p[:, 1])),
                second_moment_about_width_axis_mm4=ix,
                second_moment_about_thickness_axis_mm4=iy,
                section_modulus_about_width_axis_mm3=ix/float(np.abs(p[:, 1]-center[1]).max()),
                section_modulus_about_thickness_axis_mm3=iy/float(np.abs(p[:, 0]-center[0]).max()))


def neck_sections(wire, endpoint):
    direction = unit(endpoint['inward_tangent'])
    r = frame(direction).T
    center = np.asarray(endpoint['neck_center'], float)
    declared_width = float(endpoint.get('neck_width_mm', endpoint['neck_diameter_mm']))
    declared_thickness = float(endpoint.get('neck_thickness_mm', endpoint['neck_diameter_mm']))
    width_axis = unit((r@unit(endpoint.get('neck_width_axis', r.T[:, 0])))[:2])
    thickness_axis = unit((r@unit(endpoint.get('neck_thickness_axis', r.T[:, 1])))[:2])
    plane_axes = np.column_stack((width_axis, thickness_axis))
    contour_bound = max(.20, .55*max(declared_width, declared_thickness)+.005)
    length = float(endpoint.get('neck_length_mm', .14))
    rows = []
    for offset in (-.49*length, 0., .49*length):
        section = transformed(wire.manifold, r, -r@(center+direction*offset)).slice(0)
        candidates = [np.asarray(p) for p in section.to_polygons()
                      if len(p) >= 3 and np.linalg.norm(p, axis=1).max() < contour_bound
                      and contains_origin(np.asarray(p))]
        if len(candidates) != 1:
            return dict(passed=False, error='Expected one waist section', axial_offset_mm=offset)
        rows.append(dict(axial_offset_from_neck_center_mm=offset,
                         **polygon_moments(candidates[0]@plane_axes)))
    return dict(passed=True, measured_sections=rows,
                minimum_measured_neck_area_mm2=min(row['area_mm2'] for row in rows),
                minimum_section_modulus_about_width_axis_mm3=min(
                    row['section_modulus_about_width_axis_mm3'] for row in rows),
                minimum_section_modulus_about_thickness_axis_mm3=min(
                    row['section_modulus_about_thickness_axis_mm3'] for row in rows),
                scope='Three actual slices across each declared constant waist; the measured minimum is not a global stress or strength estimate.')


def alignment_measurements(wire, detail, direction, cache, parent, child):
    p0, p1 = [np.asarray(ep['attachment_point'], float) for ep in detail['endpoints']]
    axis = unit(p1-p0)
    r = frame(axis).T
    core = wire.manifold
    core_details = None
    if detail.get('structural_stem_mesh_file'):
        stem = Mesh(cache/detail['structural_stem_mesh_file'])
        limits = stem.vertices@axis
        overlap = min(.01, .1*float(np.ptp(limits)))
        # Retain the actual endpoint solids and the independent exported
        # central stem. Finite overlapping slices prove a complete material
        # path even with the optional off-axis grip branch omitted.
        candidate_core = union([prefix(wire.manifold, axis, float(limits.min())+overlap),
                                stem.manifold,
                                wire.manifold.trim_by_plane(axis, float(limits.max())-overlap)])
        candidate_outside = volume(candidate_core-wire.manifold)
        # Independently exported coplanar faces can differ in the last few
        # coordinate bits. Certify a core made from the actual full wire,
        # rather than exempting excess candidate material by raising a
        # tolerance. It still has to be one finite parent-to-child path.
        core = candidate_core ^ wire.manifold
        outside = volume(core-wire.manifold)
        core_vertices, _ = mesh_from_manifold(core)
        bond = first_bond(SimpleNamespace(manifold=core, vertices=core_vertices), parent, child, direction)
        core_details = dict(structural_stem_mesh_file=stem.path.name,
                            structural_stem_mesh_sha256=sha256(stem.path),
                            candidate_core_excess_removed_by_actual_wire_intersection_mm3=candidate_outside,
                            core_volume_outside_actual_wire_mm3=outside,
                            core_positive_components=sum(volume(p) > VOLUME_EPS_MM3 for p in core.decompose()),
                            first_bond=bond)
        core_details['passed'] = bool(outside <= 1e-10 and core_details['core_positive_components'] == 1
                                      and bond['passed'])
    local = transformed(core, r, -r@p0)
    length = float(np.linalg.norm(p1-p0))
    neck_coordinates = sorted(float((np.asarray(ep['neck_center'])-p0)@axis)
                              for ep in detail['endpoints'])
    centers = []
    for height in np.linspace(*neck_coordinates, 17):
        section = local.slice(float(height))
        # CrossSection integrates the complete union, including all branches
        # in this slice. Its centroid is measured below by signed contours.
        total_area, weighted_center = 0., np.zeros(2)
        contours = section.to_polygons()
        for polygon in contours:
            p = np.asarray(polygon)
            q = np.roll(p, -1, axis=0)
            cross = p[:, 0]*q[:, 1]-q[:, 0]*p[:, 1]
            signed_area = .5*float(cross.sum())
            total_area += signed_area
            weighted_center += np.sum((p+q)*cross[:, None], axis=0)/6
        if abs(total_area) <= 1e-14:
            return dict(passed=False, error='Empty section in temporary load path')
        centroid = weighted_center/total_area
        centers.append(dict(axial_coordinate_from_parent_attachment_mm=float(height),
                            transverse_centroid_mm=centroid.tolist(),
                            centroid_offset_from_attachment_line_mm=float(np.linalg.norm(centroid)),
                            section_area_mm2=abs(total_area), contour_count=len(contours)))
    endpoint_angles = [math.degrees(math.acos(float(np.clip(abs(unit(ep['inward_tangent'])@direction), 0, 1))))
                       for ep in detail['endpoints']]
    return dict(passed=core_details is None or core_details['passed'], attachment_span_mm=length,
                grip_independent_straight_core=core_details,
                attachment_line_angle_to_build_direction_degrees=math.degrees(
                    math.acos(float(np.clip(abs(axis@direction), 0, 1)))),
                endpoint_axis_angles_to_build_direction_degrees=endpoint_angles,
                maximum_measured_section_centroid_offset_mm=max(
                    row['centroid_offset_from_attachment_line_mm'] for row in centers),
                measured_sections=centers,
                scope='If a separate structural stem is provided, actual endpoint regions plus that stem must form one continuous subset of the full wire. The grip branch is then omitted from the section-centroid measurement. Section offsets do not predict the actual bending moment or resin stress.')


def staged_connectivity(supported, parent, child, direction, wire_rows, clean, wires):
    start = float((child.vertices@direction).min())
    checks = []
    for rise in STARTUP_RISES_MM:
        height = start+rise
        assembled_prefix = prefix(supported.manifold, direction, height)
        parent_prefix = prefix(parent.manifold, direction, height)
        moving_prefix = prefix(child.manifold, direction, height)
        moving_volume = volume(moving_prefix)
        components = [p for p in assembled_prefix.decompose() if volume(p) > VOLUME_EPS_MM3]
        permanent_prefixes = [prefix(part.manifold, direction, height) for part in clean]
        fixed_parts = [part for part in permanent_prefixes[0].decompose() if volume(part) > VOLUME_EPS_MM3]
        fixed_anchor = max(fixed_parts, key=volume) if fixed_parts else None
        connected_volume, disconnected_volume = 0., 0.
        touched = 0
        anchored_components = 0
        unsupported = []
        for component in components:
            root_connected = fixed_anchor is not None and (
                positive_intersection_volume(component, fixed_anchor) > VOLUME_EPS_MM3)
            if root_connected:
                anchored_components += 1
            else:
                vertices, faces = mesh_from_manifold(component)
                layers_present, wires_present = {}, {}
                for layer_index, part in enumerate(permanent_prefixes):
                    part_overlap = positive_intersection_volume(component, part)
                    if part_overlap > VOLUME_EPS_MM3:
                        layers_present[str(layer_index)] = part_overlap
                for wire_index, part in wires.items():
                    part_overlap = positive_intersection_volume(component, part.manifold)
                    if part_overlap > VOLUME_EPS_MM3:
                        wires_present[str(wire_index)] = part_overlap
                unsupported.append(dict(volume_mm3=volume(component), triangles=len(faces),
                    bounds_mm=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                    permanent_layer_overlap_volumes_mm3=layers_present,
                    temporary_wire_overlap_volumes_mm3=wires_present,
                    fixed_layer_only=bool(not wires_present and set(layers_present) == {'0'})))
            overlap = positive_intersection_volume(component, moving_prefix)
            if overlap <= VOLUME_EPS_MM3:
                continue
            touched += 1
            if positive_intersection_volume(component, parent_prefix) > VOLUME_EPS_MM3:
                connected_volume += overlap
            else:
                disconnected_volume += overlap
        active = [row['wire_index'] for row in wire_rows if row['first_bond']['passed']
                  and row['first_bond']['first_finite_connection_build_coordinate_bracket_mm'][1] <= height]
        accounted = abs(connected_volume-moving_volume) <= max(1e-10, moving_volume*1e-7)
        current_child_connected = bool(moving_volume > VOLUME_EPS_MM3 and touched == 1
                                       and accounted and disconnected_volume <= 1e-10)
        all_moving_and_wires_connected = bool(anchored_components == 1
                                              and all(row['fixed_layer_only'] for row in unsupported))
        checks.append(dict(height_above_first_moving_material_mm=rise,
                           build_coordinate_mm=height,
                           moving_prefix_volume_mm3=moving_volume,
                           moving_prefix_positive_components=sum(
                               volume(p) > VOLUME_EPS_MM3 for p in moving_prefix.decompose()),
                           supported_prefix_positive_components=len(components),
                           supported_components_containing_moving_material=touched,
                           moving_volume_connected_to_parent_mm3=connected_volume,
                           disconnected_moving_volume_mm3=disconnected_volume,
                           current_child_material_parent_connected=current_child_connected,
                           all_moving_layers_and_temporary_wires_connected_to_fixed_parent=all_moving_and_wires_connected,
                           entire_supported_prefix_connected=bool(len(components) == 1),
                           components_outside_main_fixed_parent=unsupported,
                           individually_active_temporary_paths=active,
                           passed=current_child_connected and all_moving_and_wires_connected))
    return dict(first_moving_material_build_coordinate_mm=start, stages=checks,
                passed=all(row['passed'] for row in checks))


def independent_paths(wires, details, parent, child):
    permanent = parent.manifold+child.manifold
    spans = {i: wires[i].manifold-permanent for i in details}
    span_checks = [dict(wire_index=i, volume_outside_permanent_layers_mm3=volume(part),
                        positive_components=sum(volume(q) > VOLUME_EPS_MM3 for q in part.decompose()))
                   for i, part in spans.items()]
    for row in span_checks:
        row['passed'] = bool(row['volume_outside_permanent_layers_mm3'] > VOLUME_EPS_MM3
                             and row['positive_components'] == 1)
    rows = []
    for i, j in itertools.combinations(details, 2):
        overlap = positive_intersection_volume(spans[i], spans[j])
        gap = float(spans[i].min_gap(spans[j], .6))
        rows.append(dict(wire_indices=[i, j], overlap_outside_permanent_layers_mm3=overlap,
                         minimum_gap_lower_bound_outside_permanent_layers_mm=gap, gap_search_limit_mm=.6,
                         passed=bool(overlap <= 1e-10 and gap > 1e-5)))
    return dict(individual_span_checks=span_checks, pair_checks=rows,
                passed=all(row['passed'] for row in rows+span_checks),
                interpretation='Separate full spans outside the permanent layers, with two necks in series on each path; endpoint neck counts are not parallel path counts.')


def nominal_loads(material_volume, density, path_areas):
    mass_g = material_volume*density/1000
    force_n = mass_g*9.80665/1000
    minimum_area = min(path_areas) if path_areas else None
    total_area = sum(path_areas)
    return dict(material_volume_mm3=material_volume,
                assumed_resin_density_g_cm3=density, mass_g=mass_g, static_weight_n=force_n,
                minimum_individual_path_area_mm2=minimum_area, parallel_path_area_sum_mm2=total_area,
                all_force_on_weakest_one_path_nominal_stress_mpa=force_n/minimum_area if minimum_area else None,
                ideal_shared_nominal_stress_mpa=force_n/total_area if total_area else None,
                one_path_stress_coefficient_mpa_per_n=1/minimum_area if minimum_area else None,
                ideal_shared_stress_coefficient_mpa_per_n=1/total_area if total_area else None,
                hypothetical_applied_force_cases=[dict(applied_force_n=f,
                    one_path_nominal_stress_mpa=f/minimum_area if minimum_area else None,
                    ideal_shared_nominal_stress_mpa=f/total_area if total_area else None)
                    for f in (.1, .5, 1.)],
                limitations='F/A only. Equal stress or force sharing is not established. No resin green strength, stress concentration, peel force, fatigue, or strength safety factor is assumed.')


def audit_module(meta_path, density=1.2, required_early_paths=2, activation_limit=.03):
    started = time.monotonic()
    meta_path = Path(meta_path)
    cache = meta_path.parent
    meta = json.loads(meta_path.read_text())
    direction = unit(meta.get('build_direction', (0, 1, 0)))
    clean = [Mesh(cache/p) for p in meta['free_layer_meshes']]
    supported = Mesh(cache/meta['supported_mesh_file'])
    details = {int(w['wire_index']): w for w in meta['release_wires']}
    wires = {i: Mesh(cache/w['mesh_file']) for i, w in details.items()}
    wire_rows = {}
    for index, detail in details.items():
        parent, child = clean[detail['host_index']], clean[detail['moving_index']]
        wire = wires[index]
        necks = []
        for ep in detail['endpoints']:
            check = neck_measurements(wire, ep, cache)
            sections = neck_sections(wire, ep)
            necks.append(dict(layer_index=ep['layer_index'], center_check=check,
                              waist_sections=sections, passed=check['passed'] and sections['passed']))
        starter = 'starter' in detail.get('kind', '').lower() or detail.get('role') == 'early_support'
        row = dict(wire_index=index, kind=detail.get('kind'), is_starter=starter,
                   host_index=detail['host_index'], moving_index=detail['moving_index'],
                   mesh_file=wire.path.name, mesh_sha256=sha256(wire.path),
                   first_bond=first_bond(wire, parent, child, direction), necks=necks,
                   minimum_measured_path_neck_area_mm2=min(
                       n['waist_sections'].get('minimum_measured_neck_area_mm2', 0.) for n in necks))
        if starter:
            row['alignment'] = alignment_measurements(wire, detail, direction, cache, parent, child)
            core = row['alignment'].get('grip_independent_straight_core')
            if core is not None:
                row['whole_wire_first_bond'] = row['first_bond']
                row['first_bond'] = core['first_bond']
        row['passed'] = bool(row['first_bond']['passed'] and all(n['passed'] for n in necks)
                             and row.get('alignment', {'passed': True})['passed'])
        wire_rows[index] = row

    interfaces = []
    for host_index in range(len(clean)-1):
        child_index = host_index+1
        selected = {i: w for i, w in details.items() if w['host_index'] == host_index}
        rows = [wire_rows[i] for i in selected]
        starters = [row for row in rows if row['is_starter']]
        child_start = float((clean[child_index].vertices@direction).min())
        independent = independent_paths(wires, selected, clean[host_index], clean[child_index])
        startup = staged_connectivity(supported, clean[host_index], clean[child_index], direction, rows, clean, wires)
        ordered = sorted((row for row in rows if row['first_bond']['passed']),
                         key=lambda row: row['first_bond']['first_finite_connection_build_coordinate_bracket_mm'][1])
        active_early = [row for row in starters if row['first_bond']['passed'] and
                        row['first_bond']['first_finite_connection_build_coordinate_bracket_mm'][1]
                        <= child_start+activation_limit]
        all_descendants = union([q.manifold for q in clean[child_index:]])
        suspended = union([all_descendants]+[wires[i].manifold for i, w in details.items()
                                            if w['host_index'] >= host_index])
        first_two = None
        if len(ordered) >= 2:
            first, second = ordered[:2]
            one = first['first_bond']['first_finite_connection_build_coordinate_bracket_mm']
            two = second['first_bond']['first_finite_connection_build_coordinate_bracket_mm']
            h = two[1]
            first_two = dict(first_wire_index=first['wire_index'], second_wire_index=second['wire_index'],
                second_path_build_coordinate_bracket_mm=two,
                second_path_delay_above_first_moving_material_bracket_mm=[two[0]-child_start, two[1]-child_start],
                resolved_single_path_interval_mm=max(0., two[0]-one[1]),
                connection_brackets_overlap=bool(two[0] <= one[1]),
                moving_permanent_material_at_second_path_upper_bracket_mm3=volume(prefix(all_descendants, direction, h)),
                conservative_material_including_wires_at_second_path_upper_bracket=nominal_loads(
                    volume(prefix(suspended, direction, h)), density,
                    [first['minimum_measured_path_neck_area_mm2']]),
                interpretation='Material is measured at the upper threshold bracket, conservatively including all incoming and descendant wire material. Overlapping brackets do not resolve a one-path interval.')
        early_areas = [row['minimum_measured_path_neck_area_mm2'] for row in active_early]
        joined_stages = [stage for stage in startup['stages']
                         if stage['moving_prefix_positive_components'] == 1
                         and stage['current_child_material_parent_connected']]
        first_joined_stage = joined_stages[0] if joined_stages else None
        interfaces.append(dict(host_index=host_index, moving_index=child_index,
            starter_wire_indices=[row['wire_index'] for row in starters],
            required_early_paths=required_early_paths, activation_limit_above_first_material_mm=activation_limit,
            active_starter_paths_within_limit=[row['wire_index'] for row in active_early],
            individual_wires=rows, independence=independent, startup=startup,
            first_two_temporary_paths=first_two,
            first_verified_common_child_component=dict(
                height_above_first_moving_material_mm=first_joined_stage['height_above_first_moving_material_mm'],
                moving_prefix_volume_mm3=first_joined_stage['moving_prefix_volume_mm3']) if first_joined_stage else None,
            initial_land_interpretation='Before separate receiving lands merge, each starter supports its own seed material. The starter area sum describes possible parallel paths for the shared shell only after those lands join; actual load sharing is not assumed.',
            complete_descendant_material_static_sensitivity=nominal_loads(volume(suspended), density, early_areas),
            passed=bool(len(active_early) >= required_early_paths and all(row['passed'] for row in rows)
                        and independent['passed'] and startup['passed'])))
        print(f"L{len(clean)} interface {host_index}: early paths={len(active_early)}, "
              f"stages={'pass' if startup['passed'] else 'FAIL'}, "
              f"minimum starter area={min(early_areas) if early_areas else 0:.6f} mm2", flush=True)

    inputs = [meta_path, supported.path]+[q.path for q in clean]+[q.path for q in wires.values()]
    inputs += [cache/ep['stump_mesh_file'] for detail in details.values() for ep in detail['endpoints']]
    inputs += [cache/detail['structural_stem_mesh_file'] for detail in details.values()
               if detail.get('structural_stem_mesh_file')]
    return dict(metadata_file=str(meta_path), layer_count=len(clean), metadata_sha256=sha256(meta_path),
                source_mesh_hashes={p.name: sha256(p) for p in inputs},
                build_direction=direction.tolist(), interfaces=interfaces,
                passed=all(row['passed'] for row in interfaces), elapsed_seconds=time.monotonic()-started)


def compare_modules(current, reference):
    by_layers = {row['layer_count']: row for row in reference}
    comparisons = []
    for module in current:
        baseline = by_layers.get(module['layer_count'])
        if baseline is None:
            continue
        for new, old in zip(module['interfaces'], baseline['interfaces']):
            n = new['complete_descendant_material_static_sensitivity']
            o = old['complete_descendant_material_static_sensitivity']
            comparisons.append(dict(layer_count=module['layer_count'], host_index=new['host_index'],
                previous_early_path_count=len(old['active_starter_paths_within_limit']),
                current_early_path_count=len(new['active_starter_paths_within_limit']),
                previous_path_neck_area_mm2=o['minimum_individual_path_area_mm2'],
                current_path_neck_area_mm2=n['minimum_individual_path_area_mm2'],
                individual_neck_area_ratio=n['minimum_individual_path_area_mm2']/o['minimum_individual_path_area_mm2']
                    if o['minimum_individual_path_area_mm2'] else None,
                parallel_early_area_ratio=n['parallel_path_area_sum_mm2']/o['parallel_path_area_sum_mm2']
                    if o['parallel_path_area_sum_mm2'] else None,
                previous_second_path_delay_mm=old['first_two_temporary_paths']['second_path_delay_above_first_moving_material_bracket_mm']
                    if old['first_two_temporary_paths'] else None,
                current_second_path_delay_mm=new['first_two_temporary_paths']['second_path_delay_above_first_moving_material_bracket_mm']
                    if new['first_two_temporary_paths'] else None,
                previous_maximum_starter_section_centroid_offset_mm=max(
                    row['alignment']['maximum_measured_section_centroid_offset_mm']
                    for row in old['individual_wires'] if row['is_starter']),
                current_maximum_starter_section_centroid_offset_mm=max(
                    row['alignment']['maximum_measured_section_centroid_offset_mm']
                    for row in new['individual_wires'] if row['is_starter'])))
    return comparisons


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=root/'cache/spheres')
    parser.add_argument('--reference-cache', type=Path, help='Optional unchanged V2 source cache')
    parser.add_argument('--reference-report', type=Path, help='Optional previous run of this audit, avoiding a repeat baseline audit')
    parser.add_argument('--report', type=Path, default=root/'reports/early_support_validation.json')
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument('--density', type=float, default=1.2, help='Illustrative density in g/cm3, not a material specification')
    parser.add_argument('--required-early-paths', type=int, default=2)
    parser.add_argument('--activation-limit', type=float, default=.03)
    args = parser.parse_args()
    if args.density <= 0 or args.required_early_paths < 1 or args.activation_limit <= 0:
        parser.error('Density, path count, and activation limit must be positive')
    modules = [audit_module(args.cache/f'sphere_{n}layers.json', args.density,
                            args.required_early_paths, args.activation_limit) for n in args.layers]
    reference = []
    if args.reference_cache:
        reference = [audit_module(args.reference_cache/f'sphere_{n}layers.json', args.density, 1,
                                  args.activation_limit) for n in args.layers]
    elif args.reference_report:
        reference = json.loads(args.reference_report.read_text())['modules']
    report = dict(passed=all(module['passed'] for module in modules), modules=modules,
                  reference_modules=reference, comparison_to_reference=compare_modules(modules, reference),
                  finite_anchor_volume_threshold_mm3=VOLUME_EPS_MM3,
                  first_connection_height_resolution_mm=HEIGHT_RESOLUTION_MM,
                  startup_height_offsets_mm=STARTUP_RISES_MM,
                  audit_source_sha256=sha256(Path(__file__)),
                  reference_report_sha256=sha256(args.reference_report) if args.reference_report else None,
                  scope='Source-module geometry certificate for base-first +Y construction. Separate receiving-land islands are allowed only when the actual supported prefix joins all their material to the parent. Every moving-layer and temporary-wire region present at a tested height must join the component containing the main fixed-parent prefix. Fixed-outer-layer-only islands are retained as explicit diagnostics and are not part of this captive-mechanism gate. The 0.079/0.081 mm stages bracket the declared 0.08 mm skin transition. This is not a full slicer, overhang, resin strength, peel-load, cure, or printer-process certification. Bearings are not counted as bonded temporary paths; journal contact may carry load after clearance is taken up. Final export topology and rounding are checked separately.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(f"Report: {args.report}; passed={report['passed']}", flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

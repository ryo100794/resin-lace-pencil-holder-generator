#!/usr/bin/env python3
"""Locate every final precision repair relative to the rotating mechanisms.

Input repair locations are read from preparation telemetry. Edge cylinders
are enclosed in capsules, point joints in balls, and filled microscopic
cavities in the balls of their measured bounding boxes. All comparisons are
conservative distance bounds; no model or repair geometry is modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import manifold3d as mf


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def point_segment_distance(point, a, b):
    edge = b-a
    length2 = float(edge @ edge)
    fraction = float(np.clip((point-a) @ edge/length2, 0., 1.)) if length2 else 0.
    return float(np.linalg.norm(point-(a+fraction*edge)))


def segment_distance(a, b, c, d):
    """Minimum distance of two closed line segments, including point cases."""
    result = min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
                 point_segment_distance(c, a, b), point_segment_distance(d, a, b))
    u, v, w = b-a, d-c, a-c
    uu, vv, uv, uw, vw = float(u@u), float(v@v), float(u@v), float(u@w), float(v@w)
    denominator = uu*vv-uv*uv
    if denominator > 1e-14*max(uu*vv, 1e-30):
        s, t = (uv*vw-vv*uw)/denominator, (uu*vw-uv*uw)/denominator
        if 0 <= s <= 1 and 0 <= t <= 1:
            result = min(result, float(np.linalg.norm(w+s*u-t*v)))
    return result


def make_protected(root, source_cache):
    design = json.loads((root/'reports'/'design.json').read_text())
    envelopes = json.loads((root/'reports'/'moving_envelopes.json').read_text())
    modules, bearings, necks = [], [], []
    for index, placement in enumerate(design['spheres']):
        theta = math.radians(placement['theta_deg'])
        centre = np.array([placement['center_radius']*math.cos(theta),
                           placement['center_radius']*math.sin(theta), placement['z']])
        # The documented local +X is tangent, +Y is vertical, +Z is outward.
        rotation = np.column_stack(([-math.sin(theta), math.cos(theta), 0.],
                                    [0., 0., 1.], [math.cos(theta), math.sin(theta), 0.]))
        transform = lambda point: rotation@np.asarray(point, float)+centre
        meta_path = source_cache/f"sphere_{placement['layers']}layers.json"
        meta = json.loads(meta_path.read_text())
        modules.append(dict(module_index=index, layer_count=placement['layers'],
            centre=centre, radius=float(envelopes[str(placement['layers'])]),
            metadata_file=meta_path.name, metadata_sha256=sha256(meta_path)))
        for b in meta['bearings']:
            bearings.append(dict(module_index=index, host_index=b['host_index'],
                moving_index=b['moving_index'], side=b['side'],
                a=transform(b['mouth_point']), b=transform(b['stop_point']),
                radius=float(b['bore_radius_mm'])+.00003))
        for w in meta['release_wires']:
            for endpoint_index, ep in enumerate(w['endpoints']):
                point = np.asarray(ep['attachment_point'], float)
                direction = np.asarray(ep['inward_tangent'], float)
                # Protect the full taper as well as the waist. The V3
                # elliptical starters use their wider major-axis radius.
                protective_radius=max(float(w.get('body_width_mm',w.get('body_diameter_mm',.46))),
                                      float(w.get('body_thickness_mm',0.)))/2+.00003
                necks.append(dict(module_index=index, wire_index=w['wire_index'],
                    endpoint_index=endpoint_index,
                    a=transform(point), b=transform(point+.34*direction), radius=protective_radius))
    return modules, bearings, necks


def repair_shapes(preparation):
    result = []
    def add_edges(entries, prefix):
        for entry in entries:
            for i, edge in enumerate(entry['edge_endpoints_mm']):
                a, b = np.asarray(edge, float)
                delta = b-a
                length = float(np.linalg.norm(delta))
                if length < 1e-10:
                    continue
                extension = float(entry['endpoint_extension_mm'])
                delta /= length
                result.append(dict(kind='edge_bond', identifier=f'{prefix}/{entry["pass"]}/{i}',
                    a=a-extension*delta, b=b+extension*delta,
                    radius=float(entry['bond_radius_mm'])+.00003))
    add_edges(preparation['edge_bonds'], 'edge')
    for entry in preparation['vertex_bonds']:
        for i, point in enumerate(entry['coordinates_mm']):
            p = np.asarray(point, float)
            result.append(dict(kind='vertex_joint', identifier=f'vertex/{entry["pass"]}/{i}',
                a=p, b=p, radius=float(entry['faceted_joint_radius_mm'])+.00003))
        add_edges(entry['subsequent_edge_repairs'], f'vertex/{entry["pass"]}/edge')
    return result


def cavity_shapes(input_path):
    with np.load(input_path) as data:
        model = mf.Manifold(mf.Mesh64(np.asarray(data['vertices'], np.float64),
                                     np.asarray(data['faces'], np.uint32)))
    if model.status() != mf.Error.NoError:
        raise ValueError('The original assembly is not a valid manifold input')
    result = []
    for i, component in enumerate(model.decompose()):
        volume = float(component.volume())
        if volume >= 0:
            continue
        box = np.asarray(component.bounding_box(), float)
        midpoint = (box[:3]+box[3:])/2
        radius = float(np.linalg.norm(box[3:]-box[:3])/2)
        result.append(dict(kind='microscopic_cavity_fill', identifier=f'cavity/{i}',
            a=midpoint, b=midpoint, radius=radius+.00003,
            signed_volume_mm3=volume, measured_bounds_mm=box.tolist(),
            triangles=component.num_tri()))
    return result


def audit(root, source_cache):
    prep_path = root/'reports'/'print_preparation.json'
    preparation = json.loads(prep_path.read_text())
    modules, bearings, necks = make_protected(root, source_cache)
    repairs = repair_shapes(preparation)
    cavities = cavity_shapes(root/'cache'/'final.npz')
    cavity_volume = -sum(c['signed_volume_mm3'] for c in cavities)
    cavity_match = len(cavities) == preparation['filled_microscopic_cavity_surfaces'] and \
        abs(cavity_volume-preparation['filled_volume_mm3']) < 1e-8
    results = []
    for repair in repairs+cavities:
        a, b, radius = repair['a'], repair['b'], repair['radius']
        distances = [(point_segment_distance(m['centre'], a, b)-radius-m['radius'], m)
                     for m in modules]
        moving_gap, moving = min(distances, key=lambda x: x[0])
        gap_distances = [(segment_distance(a, b, p['a'], p['b'])-radius-p['radius'], p)
                        for p in bearings]
        bearing_gap, bearing = min(gap_distances, key=lambda x: x[0])
        neck_distances = [(segment_distance(a, b, p['a'], p['b'])-radius-p['radius'], p)
                         for p in necks]
        neck_gap, neck = min(neck_distances, key=lambda x: x[0])
        row = dict(kind=repair['kind'], identifier=repair['identifier'],
            bounding_capsule_endpoints_mm=[a.tolist(), b.tolist()], bounding_capsule_radius_mm=radius,
            minimum_distance_to_any_moving_envelope_mm=moving_gap,
            closest_moving_module=moving['module_index'],
            minimum_distance_to_any_bearing_gap_capsule_mm=bearing_gap,
            closest_bearing={k:v for k,v in bearing.items() if k not in ('a', 'b', 'radius')},
            minimum_distance_to_any_neck_taper_capsule_mm=neck_gap,
            closest_neck={k:v for k,v in neck.items() if k not in ('a', 'b', 'radius')},
            passed=bool(min(moving_gap, bearing_gap, neck_gap) > 0))
        if 'signed_volume_mm3' in repair:
            row.update(signed_volume_mm3=repair['signed_volume_mm3'],
                       measured_bounds_mm=repair['measured_bounds_mm'], triangles=repair['triangles'])
        results.append(row)
    if preparation['removed_tetrahedral_residues']:
        raise ValueError('A removed residue requires an explicit geometric scope audit')
    return dict(audit='Independent scope check of all final precision repairs', units='mm',
        preparation_report_sha256=sha256(prep_path), original_assembly_sha256=sha256(root/'cache'/'final.npz'),
        source_cache=str(source_cache.resolve()),
        edge_bond_count=sum(r['kind']=='edge_bond' for r in repairs),
        vertex_joint_count=sum(r['kind']=='vertex_joint' for r in repairs),
        cavity_fill_count=len(cavities), cavity_fill_volume_mm3=cavity_volume,
        cavity_fill_matches_preparation=cavity_match,
        protected_moving_envelopes=len(modules), protected_bearing_gap_capsules=len(bearings),
        protected_neck_taper_capsules=len(necks),
        minimum_distance_to_moving_envelopes_mm=min(r['minimum_distance_to_any_moving_envelope_mm'] for r in results),
        minimum_distance_to_bearing_gap_capsules_mm=min(r['minimum_distance_to_any_bearing_gap_capsule_mm'] for r in results),
        minimum_distance_to_neck_taper_capsules_mm=min(r['minimum_distance_to_any_neck_taper_capsule_mm'] for r in results),
        modules=[{k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in m.items()} for m in modules],
        repairs=results, passed=bool(cavity_match and all(r['passed'] for r in results)),
        method='Every added primitive or filled cavity is enclosed in a conservative capsule or bounding-box ball. Exact segment distances certify separation from all moving-material envelopes and all bearing/neck protective capsules. Bounds include coordinate-rounding allowances.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--source-cache', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    source_cache = args.source_cache or args.root/'cache'/'spheres'
    report_path = args.report or args.root/'reports'/'precision_repairs_scope.json'
    result = audit(args.root, source_cache)
    report_path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('repairs','modules')}, indent=2))

#!/usr/bin/env python3
"""Transfer only route INSTRUCTIONS between repeated sphere azimuths.

Reads JSON/text only. A successfully parsed checkpoint is copied verbatim
to an immutable content-addressed snapshot. For remaining sphere IDs, the
best accepted same-layer/same-support-index counterpart supplies directions,
distances, extra cut planes and fragment order. No mesh, collision result,
endpoint-cut proof, acceptance flag or success claim is copied.

Every emitted candidate requires the normal whole-scene partition, cutter,
continuous-sweep, sibling-state and final-exit revalidation. The containing
holder is not assumed to have rotational symmetry.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import ast
import hashlib
import json
import math
import os
import re
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'cache/external_removal_v6'
ID = re.compile(r'^S([345])_A(\d{3})_(\d{3})$')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def immutable(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refusing to replace an immutable instruction artifact')
        return
    temporary = path.with_name(path.name+f'.part.{os.getpid()}.{time.time_ns()}')
    try:
        temporary.write_bytes(data)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def key(sid):
    match = ID.fullmatch(sid)
    if match is None:
        raise ValueError('Invalid sphere operation ID: '+sid)
    return tuple(int(x) for x in match.groups())  # layer, azimuth, index


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite instruction value')
    return value


def vector(value, unit=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('Instruction vector is not three-dimensional')
    result = [finite(x) for x in value]
    if unit and abs(math.sqrt(sum(x*x for x in result))-1.) > 1e-8:
        raise ValueError('Instruction direction/normal is not unit length')
    return result


def rz(point, angle_deg):
    x, y, z = vector(point)
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [c*x-s*y, s*x+c*y, z]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def layout_from_source(path):
    """Read the literal layout table without importing any geometry code."""
    raw = path.read_bytes()
    tree = ast.parse(raw.decode())
    layout = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Tuple):
            continue
        if [getattr(x, 'id', None) for x in node.target.elts] != ['kind', 'r', 'l', 'z', 'phase']:
            continue
        table = ast.literal_eval(node.iter)
        for kind_name, radius, layers, height, phase in table:
            layout[int(layers)] = dict(kind=kind_name, radius_mm=float(radius),
                height_mm=float(height), phase_deg=int(phase),
                azimuths_deg=[int(phase)+120*k for k in range(3)], center_radius_mm=44.8)
    if set(layout) != {3, 4, 5}:
        raise ValueError('Cannot ground the three repeated sphere placements')
    # Both the layout entry and sphere_center use this fixed radius in the
    # bound source. Reject an unexpected edit instead of guessing a transform.
    text = raw.decode()
    if 'center_radius=44.8' not in text or '44.8*math.cos(a)' not in text or '44.8*math.sin(a)' not in text:
        raise ValueError('Unexpected sphere-center definition')
    return layout, raw


def placement_rotation_error(layers, source_angle, target_angle, delta, layout):
    row = layout[layers]
    if source_angle not in row['azimuths_deg'] or target_angle not in row['azimuths_deg']:
        raise ValueError('Operation azimuth is absent from the bound layout')
    def basis_and_center(angle):
        a = math.radians(angle)
        return [[-math.sin(a), math.cos(a), 0.], [0., 0., 1.],
                [math.cos(a), math.sin(a), 0.],
                [row['center_radius_mm']*math.cos(a),
                 row['center_radius_mm']*math.sin(a), row['height_mm']]]
    source, target = basis_and_center(source_angle), basis_and_center(target_angle)
    return max(abs(a-b) for p, q in zip(source, target) for a, b in zip(rz(p, delta), q))


def transfer(source, target_id, layout):
    layers, target_angle, index = key(target_id)
    slayers, source_angle, sindex = key(source['id'])
    if (layers, index) != (slayers, sindex) or source_angle == target_angle:
        raise ValueError('Route counterpart identity does not match')
    delta = (target_angle-source_angle) % 360
    if delta not in (120, 240):
        raise ValueError('Counterpart is not one threefold rotation away')
    signed_delta = delta if delta <= 180 else delta-360
    placement_error = placement_rotation_error(layers, source_angle, target_angle, signed_delta, layout)
    if placement_error > 1e-10:
        raise ValueError('Placement transform does not equal the requested global-Z rotation')
    mode = source['mode']
    if mode not in ('whole_piece', 'explicit_body_cuts'):
        raise ValueError('No transferable extraction route in this operation mode')
    cuts = []
    maximum_plane_error = 0.
    for cut in source.get('additional_body_cuts', []):
        normal = vector(cut['plane_normal'], unit=True)
        rotated_normal = rz(normal, signed_delta)
        vector(rotated_normal, unit=True)
        new = dict(plane_normal=rotated_normal,
                   plane_offset_mm=finite(cut['plane_offset_mm']),
                   modeled_removed_kerf_width_mm=finite(cut['modeled_removed_kerf_width_mm']))
        if not 0. < new['modeled_removed_kerf_width_mm'] <= .05+1e-12:
            raise ValueError('Body-cut width is outside the unchanged verifier range')
        if 'centerline_reference_mm' in cut:
            point = vector(cut['centerline_reference_mm'])
            new['centerline_reference_mm'] = rz(point, signed_delta)
            maximum_plane_error = max(maximum_plane_error,
                abs(dot(normal, point)-dot(rotated_normal, new['centerline_reference_mm'])))
        cuts.append(new)
    fragments = []
    for i, fragment in enumerate(source['fragments']):
        section = fragment.get('body_section_index', i)
        if not isinstance(section, int) or isinstance(section, bool) or section < 0:
            raise ValueError('Invalid body-section instruction index')
        legs = []
        for leg in fragment['translation_legs']:
            direction = rz(vector(leg['direction'], unit=True), signed_delta)
            vector(direction, unit=True)
            distance = finite(leg['distance_mm'])
            if distance < 0.:
                raise ValueError('Negative translation distance')
            legs.append(dict(direction=direction, distance_mm=distance))
        if not legs:
            raise ValueError('Extraction fragment lacks translation instructions')
        fragments.append(dict(body_section_index=section, translation_legs=legs))
    order = [fragment['body_section_index'] for fragment in fragments]
    if sorted(order) != list(range(len(fragments))):
        raise ValueError('Fragment order is not a complete permutation')
    if 'fragment_extraction_order' in source and source['fragment_extraction_order'] != order:
        raise ValueError('Source fragment-order fields disagree')
    if mode == 'whole_piece' and (cuts or len(fragments) != 1):
        raise ValueError('Whole-piece instructions have an invalid partition')
    if mode == 'explicit_body_cuts' and not cuts:
        raise ValueError('Explicit-cut route has no cut instructions')
    if maximum_plane_error > 1e-10:
        raise ValueError('Rotation failed the cut-plane offset invariant')
    candidate = dict(id=target_id, mode=mode, additional_body_cuts=cuts,
                     fragments=fragments, fragment_extraction_order=order)
    rank = (len(cuts), sum(len(x['translation_legs']) for x in fragments),
            len(fragments), sum(leg['distance_mm'] for x in fragments for leg in x['translation_legs']),
            source['id'])
    provenance = dict(target_id=target_id, source_id=source['id'], layers=layers,
        support_index=index, source_azimuth_deg=source_angle, target_azimuth_deg=target_angle,
        rotation_about_world_Z_deg=delta, equivalent_signed_rotation_deg=signed_delta,
        source_sequence_index=source.get('sequence_index'),
        ranking=dict(additional_cut_count=rank[0], translation_leg_count=rank[1],
                     fragment_count=rank[2], total_translation_mm=rank[3]),
        maximum_placement_coordinate_error_mm=placement_error,
        maximum_rotated_plane_invariant_error_mm=maximum_plane_error,
        source_plan_sha256=sha(json.dumps(source, sort_keys=True).encode()))
    return candidate, provenance, rank


def freeze_checkpoint(path):
    for attempt in range(8):
        raw = path.read_bytes()
        try:
            data = json.loads(raw)
            plans, remaining = data['plans'], data['remaining_ids']
            accepted_ids = [row['id'] for row in plans]
            if len(set(accepted_ids)) != len(accepted_ids) or len(set(remaining)) != len(remaining):
                raise ValueError('Duplicate operation IDs in checkpoint')
            if set(accepted_ids) & set(remaining):
                raise ValueError('Accepted/remaining checkpoint sets overlap')
            if len(plans)+len(remaining) != data['initial_support_count']:
                raise ValueError('Checkpoint operation count is inconsistent')
            snapshot = OUT/('azimuth_route_snapshot_'+sha(raw)+'.json')
            immutable(snapshot, raw)
            return data, snapshot, raw
        except (json.JSONDecodeError, KeyError):
            if attempt == 7:
                raise
            time.sleep(.05)
    raise RuntimeError('Could not read a complete checkpoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=OUT/'checkpoint.json')
    args = parser.parse_args()
    checkpoint, snapshot, raw = freeze_checkpoint(args.checkpoint.resolve())
    layout_path = ROOT/'source/build_holder.py'
    layout, layout_bytes = layout_from_source(layout_path)
    sources = defaultdict(list)
    for row in checkpoint['plans']:
        if ID.fullmatch(row['id']) is None:
            continue
        layers, angle, index = key(row['id'])
        if row.get('passed') is not True or row.get('group') != 'sphere':
            continue
        if (row.get('layers'), row.get('source_index'), row.get('azimuth_deg')) != (layers, index, angle):
            raise ValueError('Accepted source metadata/ID mismatch')
        sources[layers, index].append(row)
    remaining = [sid for sid in checkpoint['remaining_ids'] if ID.fullmatch(sid)]
    chosen = []
    unavailable = []
    groups = defaultdict(list)
    invalid_sources = []
    for target in remaining:
        layers, _, index = key(target)
        groups[layers, index].append(target)
        alternatives = []
        for source in sources[layers, index]:
            try:
                alternatives.append(transfer(source, target, layout))
            except ValueError as error:
                invalid_sources.append(dict(target_id=target, source_id=source['id'], reason=str(error)))
        if not alternatives:
            unavailable.append(dict(target_id=target, layers=layers, support_index=index,
                reason=('No accepted same-layer/index counterpart in this checkpoint.'
                        if not sources[layers, index] else 'No usable route instruction mode/shape.')))
            continue
        alternatives.sort(key=lambda entry: entry[2])
        candidate, provenance, rank = alternatives[0]
        provenance['available_accepted_counterparts'] = [row['id'] for row in sources[layers, index]]
        chosen.append((candidate, provenance, rank))
    chosen.sort(key=lambda entry: (entry[2][:-1], entry[0]['id']))
    candidates = dict(plans=[entry[0] for entry in chosen])
    candidate_bytes = (json.dumps(candidates, indent=2, sort_keys=True)+'\n').encode()
    candidate_digest = sha(candidate_bytes)
    candidate_path = OUT/('azimuth_route_candidates_'+candidate_digest+'.json')
    immutable(candidate_path, candidate_bytes)
    source_path = Path(__file__).resolve()
    report = dict(
        instruction_generation_completed=True, routes_validated=False,
        scope='Instruction-only azimuth transfer. No meshes, collision/partition proofs, endpoint-cut results or success flags are copied into candidates.',
        builder=dict(path=str(source_path.relative_to(ROOT)), sha256=sha(source_path.read_bytes())),
        checkpoint_snapshot=dict(path=str(snapshot.relative_to(ROOT)), sha256=sha(raw)),
        layout_source=dict(path=str(layout_path.relative_to(ROOT)), sha256=sha(layout_bytes)),
        candidates=dict(path=str(candidate_path.relative_to(ROOT)), sha256=candidate_digest,
                        count=len(chosen)),
        snapshot_counts=dict(accepted=len(checkpoint['plans']), remaining_total=len(checkpoint['remaining_ids']),
            remaining_spheres=len(remaining), remaining_layer_index_groups=len(groups),
            remaining_group_size_histogram=dict(Counter(len(rows) for rows in groups.values()))),
        selection_rule='Minimum additional-cut count, then leg count, fragment count, total travel and source ID.',
        transform_rule='Rotate world directions, body-cut normals and reference points around world Z; leave distances, plane offsets and kerf widths unchanged. Preserve source fragment order/body_section_index.',
        required_revalidation='Recompute target actual partition/cutter state, every complete-scene continuous sweep, sequential sibling state and final scene exit under unchanged gates. Holder rotational symmetry is not assumed.',
        provenance=[entry[1] for entry in chosen], unavailable=unavailable,
        unusable_source_routes=invalid_sources,
        active_input_or_printed_geometry_modified=False)
    report_bytes = (json.dumps(report, indent=2, sort_keys=True)+'\n').encode()
    report_path = ROOT/'reports'/('azimuth_route_candidates_'+candidate_digest[:16]
                                  +'_'+sha(report_bytes)[:16]+'.json')
    immutable(report_path, report_bytes)
    print(json.dumps(dict(candidate_path=str(candidate_path.relative_to(ROOT)),
        candidate_sha256=candidate_digest, candidate_count=len(chosen),
        checkpoint_snapshot=str(snapshot.relative_to(ROOT)), snapshot_sha256=sha(raw),
        report_path=str(report_path.relative_to(ROOT)), report_sha256=sha(report_bytes),
        remaining_without_counterpart=len(unavailable)), indent=2))


if __name__ == '__main__':
    main()

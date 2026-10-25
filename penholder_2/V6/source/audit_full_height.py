#!/usr/bin/env python3
"""Independent whole-height sampled connectivity check for nested spheres.

The generator is not imported.  This reads the actual supported mesh,
permanent layers and temporary wires, clips the accumulated printed solid
at regular heights and near mesh local minima, and finds disconnected
internal material.  Fixed-outer-shell-only islands are recorded separately
because that shell may receive external slicer supports.  This is not a
strength, peel-load, overhang-angle or printer/slicer certification.
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


ROOT = Path(__file__).resolve().parent.parent
VOLUME_EPSILON_MM3 = 1e-10
MAX_CERTIFIED_GRID_STEP_MM = .05
LOCAL_MINIMUM_OFFSETS_MM = (.0001, .001, .005, .025)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def volume(part):
    return 0. if part.is_empty() else max(0., float(part.volume()))


def overlap(a, b):
    if a.is_empty() or b.is_empty():
        return 0.
    aa, bb = np.asarray(a.bounding_box()), np.asarray(b.bounding_box())
    if np.any(aa[3:] < bb[:3]) or np.any(bb[3:] < aa[:3]):
        return 0.
    return volume(a ^ b)


def prefix(part, direction, height):
    return part.trim_by_plane(-direction, -float(height))


class InputMesh:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = digest(path)
        with np.load(path, allow_pickle=False) as data:
            self.vertices = np.asarray(data['vertices'], dtype=np.float64)
            self.faces = np.asarray(data['faces'], dtype=np.uint32)
        self.solid = mf.Manifold(mf.Mesh64(self.vertices, self.faces))
        if self.solid.status() != mf.Error.NoError:
            raise ValueError('Invalid source mesh: '+str(path))


def local_minimum_levels(mesh, direction):
    """Find boundary-vertex local minima, including horizontal plateaus.

    Plateaus can create redundant candidates.  Keeping those candidates
    is conservative; the clipping stage decides actual solid connectivity.
    The returned levels are sampling targets, not a proof that every
    possible critical topology event has been enumerated.
    """
    levels = mesh.vertices @ direction
    neighbor_minimum = np.full(len(levels), np.inf)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        a, b = mesh.faces[:, first], mesh.faces[:, second]
        np.minimum.at(neighbor_minimum, a, levels[b])
        np.minimum.at(neighbor_minimum, b, levels[a])
    selected = levels <= neighbor_minimum + 1e-10
    return np.unique(np.round(levels[selected], 10))


def height_schedule(mesh, direction, grid_step, internal_parts):
    levels = mesh.vertices @ direction
    minimum, maximum = float(levels.min()), float(levels.max())
    targets = {}

    def add(height, kind):
        height = float(height)
        if not minimum < height <= maximum:
            return
        key = round(height, 10)
        targets.setdefault(key, set()).add(kind)

    first = math.floor(minimum/grid_step)*grid_step
    for height in np.arange(first, maximum+grid_step*.5, grid_step):
        add(height, 'regular_grid')
    add(maximum, 'complete_model')
    # Refine the internal material. Fixed-outer-only ornaments explicitly
    # need external slicer supports and are not this audit's release gate.
    # Any internal component appearing in a union contains material from
    # at least one of these actual operand meshes, so their minima are
    # meaningful candidates even if a later union hides a boundary vertex.
    minima = np.unique(np.concatenate([local_minimum_levels(part, direction)
                                       for part in internal_parts]))
    for height in minima:
        for offset in LOCAL_MINIMUM_OFFSETS_MM:
            add(height+offset, 'local_minimum_plus_'+format(offset, '.4f')+'_mm')
    return [(height, sorted(kinds)) for height, kinds in sorted(targets.items())], minima


def wire_onsets(wires, free, direction, details):
    """Diagnostic: onset of wire material versus any permanent contact.

    A wire may legitimately be supported by another temporary part.  The
    complete supported-prefix test, not this diagnostic alone, is the gate.
    """
    rows = []
    for index, wire in enumerate(wires):
        start = float((wire.vertices@direction).min())
        contact_minima = []
        for layer in free:
            contact = wire.solid ^ layer.solid
            if volume(contact) <= VOLUME_EPSILON_MM3:
                continue
            vertices = np.asarray(contact.to_mesh64().vert_properties[:, :3])
            contact_minima.append(float((vertices@direction).min()))
        first = min(contact_minima) if contact_minima else None
        rows.append(dict(wire_index=int(details[index]['wire_index']),
                         kind=details[index].get('kind'), mesh_file=wire.path.name,
                         first_material_build_coordinate_mm=start,
                         earliest_permanent_contact_coordinate_mm=first,
                         material_precedes_first_permanent_contact_mm=
                         None if first is None else max(0., first-start)))
    return rows


def audit_module(meta_path, grid_step=.05):
    start_time = time.monotonic()
    meta_path = Path(meta_path)
    cache = meta_path.parent
    metadata_hash = digest(meta_path)
    meta = json.loads(meta_path.read_text())
    direction = np.asarray(meta.get('build_direction', [0., 1., 0.]), dtype=float)
    direction /= np.linalg.norm(direction)
    supported = InputMesh(cache/meta['supported_mesh_file'])
    free = [InputMesh(cache/name) for name in meta['free_layer_meshes']]
    details = meta['release_wires']
    wires = [InputMesh(cache/row['mesh_file']) for row in details]
    inputs = {mesh.path.name: mesh.sha256 for mesh in [supported]+free+wires}
    schedule, minima = height_schedule(supported, direction, grid_step, free[1:]+wires)
    fixed_outer_contained = volume(free[0].solid-supported.solid) <= VOLUME_EPSILON_MM3
    fixed_start = float((free[0].vertices@direction).min())
    stages, internal_records, fixed_records = [], [], []
    worst_volume = 0.
    for serial, (height, kinds) in enumerate(schedule):
        accumulated = prefix(supported.solid, direction, height)
        components = [part for part in accumulated.decompose()
                      if volume(part) > VOLUME_EPSILON_MM3]
        # The fixed outer layer's largest current component is the same
        # anchor convention used by audit_early_supports.py.  Isolated
        # exterior ornament fragments remain explicit diagnostics.
        anchor = None
        anchored_count, internal_here, fixed_here = 0, [], []
        if len(components) == 1 and fixed_outer_contained and height > fixed_start+1e-7:
            # Set inclusion survives intersection with a half-space. The
            # one positive accumulated component therefore contains the
            # nonempty fixed prefix. The full-set inclusion was measured
            # independently above, avoiding a repeated large Boolean.
            anchored_count = 1
        if anchored_count != 1 or len(components) != 1:
            outer_prefix = prefix(free[0].solid, direction, height)
            anchor_parts = [part for part in outer_prefix.decompose()
                            if volume(part) > VOLUME_EPSILON_MM3]
            anchor = max(anchor_parts, key=volume) if anchor_parts else None
            anchored_count = 0
            for part in components:
                if anchor is not None and overlap(part, anchor) > VOLUME_EPSILON_MM3:
                    anchored_count += 1
                    continue
                layers_present = {}
                for index, layer in enumerate(free):
                    amount = overlap(part, layer.solid)
                    if amount > VOLUME_EPSILON_MM3:
                        layers_present[str(index)] = amount
                wires_present = {}
                for index, wire in enumerate(wires):
                    amount = overlap(part, wire.solid)
                    if amount > VOLUME_EPSILON_MM3:
                        wires_present[str(details[index]['wire_index'])] = amount
                fixed_only = bool(not wires_present and set(layers_present) == {'0'})
                row = dict(build_coordinate_mm=height, sampling_reasons=kinds,
                           volume_mm3=volume(part), triangles=int(part.num_tri()),
                           bounds_mm=list(part.bounding_box()),
                           permanent_layer_overlap_volumes_mm3=layers_present,
                           temporary_wire_overlap_volumes_mm3=wires_present,
                           fixed_layer_only=fixed_only)
                if fixed_only:
                    fixed_here.append(row)
                else:
                    internal_here.append(row)
                    worst_volume = max(worst_volume, row['volume_mm3'])
        passed = bool(not internal_here and
                      (anchored_count == 1 or not components))
        stages.append(dict(build_coordinate_mm=height, sampling_reasons=kinds,
                           supported_prefix_positive_components=len(components),
                           anchored_components=anchored_count,
                           internal_disconnected_component_count=len(internal_here),
                           fixed_outer_only_component_count=len(fixed_here),
                           passed=passed))
        internal_records.extend(internal_here)
        fixed_records.extend(fixed_here)
        if serial and serial % 200 == 0:
            print(f'L{meta["layer_count"]}: {serial}/{len(schedule)} heights; '
                  f'internal disconnected records={len(internal_records)}', flush=True)
    source_stable = digest(meta_path) == metadata_hash and all(
        digest(cache/name) == expected for name, expected in inputs.items())
    topology_parts = supported.solid.decompose()
    topology_ok = len(topology_parts) == 1 and volume(topology_parts[0]) > 0 and fixed_outer_contained
    coverage_ok = grid_step <= MAX_CERTIFIED_GRID_STEP_MM and bool(minima.size)
    passed = bool(source_stable and topology_ok and coverage_ok
                  and stages and all(stage['passed'] for stage in stages))
    report = dict(metadata_file=meta_path.name, metadata_sha256=metadata_hash,
                  layer_count=meta['layer_count'], design_version=meta.get('version'),
                  source_mesh_hashes=inputs, inputs_unchanged_during_run=source_stable,
                  build_direction=direction.tolist(), grid_step_mm=grid_step,
                  fixed_outer_contained_in_supported_mesh=fixed_outer_contained,
                  regular_grid_and_local_minimum_coverage_passed=coverage_ok,
                  local_minimum_offsets_mm=list(LOCAL_MINIMUM_OFFSETS_MM),
                  local_minimum_levels_mm=minima.tolist(),
                  heights_checked=len(stages), stages=stages,
                  disconnected_internal_records=internal_records,
                  disconnected_fixed_outer_only_records=fixed_records,
                  maximum_disconnected_internal_component_volume_mm3=worst_volume,
                  wire_onset_diagnostics=wire_onsets(wires, free, direction, details),
                  final_supported_topology_passed=topology_ok, passed=passed,
                  elapsed_seconds=time.monotonic()-start_time,
                  record_count_interpretation='One physical feature can appear at several sampled heights; records are not unique defect counts.')
    print(f'L{meta["layer_count"]}: passed={passed}, heights={len(stages)}, '
          f'internal records={len(internal_records)}, '
          f'largest={worst_volume:.9f} mm3', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'cache'/'spheres')
    parser.add_argument('--report', type=Path, default=ROOT/'reports'/'full_height_validation.json')
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument('--step-mm', type=float, default=.05,
                        help='Default .05 mm. Coarser diagnostic scans cannot pass the release coverage gate.')
    args = parser.parse_args()
    if not 0 < args.step_mm <= 1.:
        parser.error('step-mm must be positive and at most 1 mm')
    result = dict(audit='Independent whole-height sampled accumulated-solid connectivity',
                  units='mm', volume_epsilon_mm3=VOLUME_EPSILON_MM3,
                  maximum_release_grid_step_mm=MAX_CERTIFIED_GRID_STEP_MM,
                  audit_source_sha256=digest(Path(__file__)), modules=[], passed=False,
                  scope=[
                      'Actual supported meshes are clipped at a regular grid across their full build height and just above boundary-vertex local minima of all internal permanent layers and temporary wires.',
                      'Every sampled disconnected component containing an internal layer or a temporary support fails the internal-connectivity gate.',
                      'Components belonging solely to the fixed outer shell remain explicit diagnostics and require appropriate external slicer support.',
                      'The largest current fixed-outer-shell component is the reference anchor. The holder wall and printer-generated external supports are not part of this module audit.',
                      'This finite sampling does not certify all slicer layers, layer phases, pixels, sub-threshold volumes, overhang angles or continuous solid connectivity between tested heights.',
                      'No resin strength, peel force, fatigue, shrinkage, exposure, post-processing damage or physical-print performance is certified.'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for layer_count in args.layers:
        result['modules'].append(audit_module(args.cache/f'sphere_{layer_count}layers.json', args.step_mm))
        args.report.write_text(json.dumps(result, indent=2)+'\n')
    result['passed'] = all(module['passed'] for module in result['modules'])
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

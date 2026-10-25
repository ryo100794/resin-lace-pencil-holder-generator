#!/usr/bin/env python3
"""Derive release summaries from the actual audited mechanism and source hashes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import manifold3d as mf

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT/'cache'/'spheres'
REPORTS = ROOT/'reports'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    with np.load(path, allow_pickle=False) as d:
        return mf.Manifold(mf.Mesh64(d['vertices'], d['faces'].astype(np.uint32)))

def write(name, value):
    (REPORTS/name).write_text(json.dumps(value, indent=2)+'\n')

def main():
    motion = json.loads((REPORTS/'motion_validation.json').read_text())
    if not motion.get('passed'):
        raise RuntimeError('Complete motion/access validation must pass first')
    details, envelopes, generation, projections = {}, {}, [], []
    for module in motion['modules']:
        n = module['layer_count']
        path = CACHE/module['metadata_file']
        if digest(path) != module['metadata_sha256']:
            raise RuntimeError('Metadata changed after validation: '+str(path))
        meta = json.loads(path.read_text())
        epsilon = module['float32_export_allowance_mm']
        moving = []
        for index in range(1, n):
            topology = module['actual_with_stumps_layer_topology'][index]
            source = CACHE/topology['file']
            if digest(source) != topology['sha256']:
                raise RuntimeError('Moving geometry changed after validation')
            radius = module['layer_envelopes'][index]['maximum_radius_mm']
            play = module['cumulative_play_allowance_by_layer_mm'][index]
            moving.append(dict(layer_index=index, mesh_file=source.name,
                mesh_sha256=topology['sha256'], centered_radius_mm=radius,
                play_allowance_mm=play, radius_including_play_mm=radius+play))
        bound = module['moving_radius_including_conservative_play_mm']
        envelopes[str(n)] = bound+epsilon
        details[str(n)] = dict(source_cache='cache/spheres',
            metadata_file=path.name, metadata_sha256=digest(path),
            moving_layers=moving,
            maximum_centered_radius_mm=module['moving_centered_radius_bound_mm'],
            radius_including_play_mm=bound, float32_export_allowance_mm=epsilon,
            certified_wall_envelope_radius_mm=bound+epsilon)
        projections.append(module['support_projection_separation'])
        free = [load(CACHE/name) for name in meta['free_layer_meshes']]
        bonds = []
        for wire in meta['release_wires']:
            for endpoint in wire['endpoints']:
                stump = load(CACHE/endpoint['stump_mesh_file'])
                bonds.append(float((stump ^ free[endpoint['layer_index']]).volume()))
        topology = module['supported_topology']
        generation.append(dict(layers=n, faces=topology['triangles'],
            supports=meta['wire_count'], endpoints=meta['neck_count'],
            min_endpoint_shell_bond_volume_mm3=min(bonds),
            zero_area_triangles=topology['zero_area_triangles'],
            supported_sha256=digest(CACHE/meta['supported_mesh_file']),
            minimum_wire_gap_mm=meta['measured_minimum_static_inter_wire_gap_mm'],
            partition_overlap_mm3=meta['partition_check_maximum_removable_nominal_stump_overlap_mm3'],
            passed=bool(min(bonds)>.005 and topology['passed'])))
    if not all(row['passed'] for row in generation):
        raise RuntimeError('Source bond/topology summary failed')
    write('moving_envelopes.json', envelopes)
    write('moving_envelopes_details.json', details)
    write('support_projection_clearance.json', dict(
        audit='V5 complete-support projection certificate', units='mm',
        modules=projections, passed=all(row['passed'] for row in projections)))
    write('sphere_generation_checks.json', generation)
    print(json.dumps(dict(passed=True, modules=len(generation),
        minimum_bond_volume_mm3=min(row['min_endpoint_shell_bond_volume_mm3'] for row in generation))))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Bounded read-only direct-original overlap diagnostic for 24 plate stocks."""
from pathlib import Path
import hashlib
import json
import resource
import time

import manifold3d as mf
import numpy as np
import external_removal_plate_stock_v6 as B

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        vertices = np.array(data['vertices'], np.float64, copy=True)
        faces = np.array(data['faces'], np.uint64, copy=True)
    solid = mf.Manifold(mf.Mesh64(vertices, faces))
    if solid.status() != mf.Error.NoError:
        raise ValueError('Invalid source solid')
    return solid


def main():
    start = time.time()
    original_path = ROOT / 'cache/original_v5.npz'
    inputs = {str(original_path.relative_to(ROOT)): sha(original_path),
              'source/external_removal_plate_stock_v6.py': sha(B.__file__)}
    if inputs['cache/original_v5.npz'] != 'de722a37883d54ad0d9cb8383aff327d08f2885eceab08ae9d96981c37ece348':
        raise ValueError('Unexpected immutable original hash')
    original = load(original_path)
    original.num_tri()
    print('One original Manifold loaded; no Surface or acceleration trees.', flush=True)
    metadata_path = ROOT / 'cache/custom_holder_supports/holder_supports.json'
    inputs[str(metadata_path.relative_to(ROOT))] = sha(metadata_path)
    rows = json.loads(metadata_path.read_text())['parts']
    results = []
    for sid in sorted(x for x in B.PLATE_IDS if int(x[1:]) < 579):
        row = rows[int(sid[1:])]
        paths = {key: ROOT/'cache/custom_holder_supports'/row[key]
                 for key in ('full_mesh_file', 'removable_mesh_file')}
        for path in paths.values():
            inputs[str(path.relative_to(ROOT))] = sha(path)
        full = load(paths['full_mesh_file'])
        removable = load(paths['removable_mesh_file'])
        overlap = removable ^ original
        outside = removable - full
        entry = dict(id=sid, intersection_is_empty=overlap.is_empty(),
                     intersection_signed_volume_mm3=float(overlap.volume()),
                     intersection_triangles=overlap.num_tri(),
                     R_outside_F_is_empty=outside.is_empty(),
                     R_outside_F_signed_volume_mm3=float(outside.volume()),
                     literal_removable_sha256=inputs[str(paths['removable_mesh_file'].relative_to(ROOT))])
        results.append(entry)
        print(json.dumps(entry), flush=True)
    report = dict(
        scope='One original Manifold only; direct intersections of24 small actual saved R meshes with unchanged original. No Surface, no full-support assembly and no geometry changes.',
        passed=all(row['intersection_is_empty'] and row['R_outside_F_is_empty'] for row in results),
        inputs=inputs, results=results, elapsed_s=time.time()-start,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        diagnostic_source=dict(path='source/diagnose_plate_noop_v6.py', sha256=sha(__file__)))
    dest = ROOT / 'reports/plate_stock_literal_noop_diagnostic_v6.json'
    if dest.exists():
        raise ValueError('Refusing to overwrite prior diagnostic')
    dest.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(dest.relative_to(ROOT)), sha256=sha(dest),
                          passed=report['passed'], elapsed_s=report['elapsed_s'],
                          peak_process_rss_kib=report['peak_process_rss_kib'])), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Read-only exact cross-contact audit of sphere and holder support stock.

Original V5 geometry is deliberately not a collision obstacle here: only a
new support bonding to another support's removable stock or neck is tested.
Every input byte stream used is copied into an immutable, hashed snapshot.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import manifold3d as mf
import numpy as np

import custom_holder_supports as H
from build_holder import SPHERES, sphere_transform

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-9


def sha(data):
    return hashlib.sha256(data).hexdigest()


def volume(solid):
    return 0.0 if solid.is_empty() else max(0.0, float(solid.volume()))


def union(parts):
    parts = [p for p in parts if not p.is_empty()]
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add) if parts else mf.Manifold()


def intersects_boxes(a, b):
    return bool(np.all(a[3:] >= b[:3]) and np.all(b[3:] >= a[:3]))


def main():
    started = time.monotonic()
    holder_dir = ROOT / 'cache/custom_holder_supports'
    sphere_dir = ROOT / 'cache/custom_sphere_supports'
    checkpoint = holder_dir / 'generation_checkpoint.json'
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_hash = sha(checkpoint_bytes)
    holder_metadata = json.loads(checkpoint_bytes)
    snapshot = sphere_dir / 'cross_contacts_snapshot' / checkpoint_hash[:16]
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / 'holder_generation_checkpoint.json').write_bytes(checkpoint_bytes)
    inputs = []
    loaded = {}

    def frozen_bytes(path):
        path = Path(path)
        relative = path.relative_to(ROOT)
        key = str(relative)
        if key in loaded:
            return loaded[key]
        data = path.read_bytes()
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError(f'Immutable snapshot filename changed: {dest}')
        dest.write_bytes(data)
        inputs.append(dict(path=key, sha256=sha(data), bytes=len(data),
                           snapshot_path=str(dest.relative_to(ROOT))))
        loaded[key] = data
        return data

    def solid(path):
        with np.load(io.BytesIO(frozen_bytes(path)), allow_pickle=False) as data:
            result = mf.Manifold(mf.Mesh64(np.asarray(data['vertices'], float),
                                          np.asarray(data['faces'], np.uint32)))
        if result.status() != mf.Error.NoError:
            raise ValueError(f'Invalid support mesh: {path}')
        return result

    sphere_modules = {}
    sphere_records = []
    for placement, sphere in enumerate(SPHERES):
        n = sphere['layers']
        if n not in sphere_modules:
            metadata = json.loads(frozen_bytes(sphere_dir / f'sphere_{n}layers_supports.json'))
            if not metadata.get('passed') or not metadata.get('finalization_complete'):
                raise RuntimeError(f'Sphere {n} finalization is incomplete')
            module = []
            for row in metadata['supports']:
                full = solid(sphere_dir / row['mesh_file'])
                removable = solid(sphere_dir / row['removable_mesh_file'])
                necks = []
                if row['kind'] == 'removable_fixed_support':
                    for j in range(2):
                        path = sphere_dir / f'sphere_{n}layers_neck_{row["support_index"]:03d}_{j}.npz'
                        necks.append(solid(path) ^ full)
                module.append((row, full, removable, union(necks)))
            sphere_modules[n] = module
        transform = sphere_transform(sphere)
        for row, full, removable, neck in sphere_modules[n]:
            full = full.transform(transform)
            removable = removable.transform(transform)
            neck = neck.transform(transform)
            sphere_records.append(dict(id=f'S{n}_A{int(sphere["theta_deg"]):03d}_{row["support_index"]:03d}',
                                       layer_count=n, placement=placement, azimuth_deg=sphere['theta_deg'],
                                       support_index=row['support_index'], kind=row['kind'], full=full,
                                       removable=removable, neck=neck, bounds=np.asarray(full.bounding_box())))

    sphere_bounds = np.asarray([r['bounds'] for r in sphere_records])
    rows = []
    permitted_fixed_contacts = []
    bbox_pairs = 0
    actual_pairs = 0
    tested = 0
    for index, holder in enumerate(holder_metadata['parts']):
        hfull = solid(holder_dir / holder['full_mesh_file'])
        hb = np.asarray(hfull.bounding_box())
        candidates = np.flatnonzero(np.all(sphere_bounds[:, 3:] >= hb[:3], axis=1)
                                    & np.all(sphere_bounds[:, :3] <= hb[3:], axis=1))
        bbox_pairs += len(candidates)
        if not len(candidates):
            continue
        hremovable = solid(holder_dir / holder['removable_mesh_file'])
        necks = []
        count = holder.get('neck_count', 0)
        if count:
            radius = holder['neck_diameter_mm'] / 2
            if holder.get('support_style') == 'compact_horizontal_sections':
                q = np.asarray(holder['root_mm'])
                p = np.asarray(holder['target_mm'])
                dz = p[2] - q[2]
                shear = np.array([[1, 0, (p[0]-q[0])/dz, q[0]],
                                  [0, 1, (p[1]-q[1])/dz, q[1]], [0, 0, 1, q[2]]])
                necks = [H.profile([0, 0, 0], [(a, radius), (b, radius)]).transform(shear)
                         for a, b in [(.06, .14), (dz-.14, dz-.06)]]
            else:
                half = holder['neck_length_mm'] / 2
                necks = [H.axial_profile(np.asarray(c), np.asarray(a), [(-half, radius), (half, radius)])
                         for c, a in zip(holder['neck_centers_mm'], holder['neck_axes'])]
            necks = [part ^ hfull for part in necks]
        hneck = union(necks)
        for si in candidates:
            sphere = sphere_records[int(si)]
            full_contact = sphere['full'] ^ hfull
            full_volume = volume(full_contact)
            tested += 1
            if full_volume <= TOL:
                continue
            actual_pairs += 1
            conflicts = []
            for kind, first, second in [
                ('sphere_removable_vs_holder_full', sphere['removable'], hfull),
                ('sphere_neck_vs_holder_full', sphere['neck'], hfull),
                ('holder_removable_vs_sphere_full', hremovable, sphere['full']),
                ('holder_neck_vs_sphere_full', hneck, sphere['full']),
            ]:
                if first.is_empty() or second.is_empty():
                    continue
                if not intersects_boxes(np.asarray(first.bounding_box()), np.asarray(second.bounding_box())):
                    continue
                intersection = first ^ second
                amount = volume(intersection)
                if amount <= TOL:
                    continue
                regions = []
                for part in intersection.decompose():
                    part_volume = volume(part)
                    if part_volume <= TOL:
                        continue
                    bounds = np.asarray(part.bounding_box())
                    regions.append(dict(volume_mm3=part_volume, bounds_mm=bounds.tolist(),
                                        bounds_center_mm=((bounds[:3]+bounds[3:])/2).tolist()))
                conflicts.append(dict(test=kind, volume_mm3=amount, regions=regions))
            record = dict(holder_id=f'H{holder["index"]:04d}', holder_index=holder['index'],
                          holder_kind=holder['kind'], holder_root_mm=holder['root_mm'],
                          holder_target_mm=holder['target_mm'], sphere_id=sphere['id'],
                          sphere_layers=sphere['layer_count'], sphere_placement=sphere['placement'],
                          sphere_azimuth_deg=sphere['azimuth_deg'], sphere_support_index=sphere['support_index'],
                          sphere_kind=sphere['kind'], full_contact_volume_mm3=full_volume)
            if conflicts:
                rows.append(dict(**record, conflicts=conflicts))
            else:
                permitted_fixed_contacts.append(record)
        if index % 200 == 0:
            print(f'holder {index}/{len(holder_metadata["parts"])}; forbidden pairs {len(rows)}', flush=True)

    changed_inputs = [item['path'] for item in inputs
                      if not (ROOT/item['path']).exists() or sha((ROOT/item['path']).read_bytes()) != item['sha256']]
    current_checkpoint_hash = sha(checkpoint.read_bytes())
    report = dict(passed=not rows and not changed_inputs,
                  scope='Support-to-support accidental fusion only. Every sphere module is repeated unfiltered at all three placements.',
                  units='mm', volume_threshold_mm3=TOL, holder_checkpoint_sha256=checkpoint_hash,
                  current_holder_checkpoint_sha256=current_checkpoint_hash,
                  checkpoint_changed_after_snapshot=current_checkpoint_hash != checkpoint_hash,
                  holder_checkpoint_complete=holder_metadata.get('complete', False),
                  holder_part_count=len(holder_metadata['parts']), sphere_placed_part_count=len(sphere_records),
                  bounding_box_candidate_pairs=bbox_pairs, boolean_full_pairs_tested=tested,
                  positive_full_contact_pairs=actual_pairs, forbidden_pair_count=len(rows),
                  forbidden_pairs=rows, permitted_fixed_only_contact_pairs=permitted_fixed_contacts,
                  inputs=inputs, loaded_input_files_changed_since_snapshot=changed_inputs,
                  snapshot_folder=str(snapshot.relative_to(ROOT)),
                  neck_reconstruction='Sphere actual saved neck solids clipped to full support. Holder pitched/axial circular necks reconstructed from recorded center/axis/length/radius; compact sheared necks from recorded endpoints/radius. All holder necks clipped to the actual full mesh.',
                  holder_geometry_helper_sha256=sha(Path(H.__file__).read_bytes()),
                  audit_source_sha256=sha(Path(__file__).read_bytes()), seconds=time.monotonic()-started)
    target = ROOT/'reports/custom_sphere_supports_cross_contacts.json'
    temporary = target.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(target)
    print(json.dumps({k: report[k] for k in ['passed', 'holder_part_count', 'sphere_placed_part_count',
                     'bounding_box_candidate_pairs', 'positive_full_contact_pairs', 'forbidden_pair_count',
                     'checkpoint_changed_after_snapshot', 'loaded_input_files_changed_since_snapshot', 'seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()

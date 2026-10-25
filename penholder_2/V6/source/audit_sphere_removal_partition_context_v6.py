#!/usr/bin/env python3
"""Local-context precheck for finite neck wafers; no model mutation."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import custom_sphere_supports as S
from build_holder import SPHERES,sphere_transform

ROOT=Path(__file__).resolve().parent.parent
TOL=1e-9

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    started=time.monotonic();manifest=ROOT/'reports/sphere_removal_partition_v6.json';plan=json.loads(manifest.read_text())
    if not plan['passed']:raise RuntimeError('Partition manifest not passed')
    inputs={str(manifest.relative_to(ROOT)):sha(manifest)};records=[];failures=[];dependencies=[]
    holder_path=ROOT/'cache/custom_holder_supports/holder_supports.npz';holder=S.load(holder_path);inputs[str(holder_path.relative_to(ROOT))]=sha(holder_path)
    for n in [3,4,5]:
        meta_path=ROOT/f'cache/spheres/sphere_{n}layers.json';meta=json.loads(meta_path.read_text());inputs[str(meta_path.relative_to(ROOT))]=sha(meta_path)
        own_paths=[ROOT/'cache/spheres'/name for name in meta['free_layer_meshes']]+[ROOT/'cache/spheres'/w['mesh_file'] for w in meta['release_wires']]
        own_original=S.join([S.load(p) for p in own_paths])
        for p in own_paths:inputs[str(p.relative_to(ROOT))]=sha(p)
        supports_path=ROOT/f'cache/custom_sphere_supports/sphere_{n}layers_supports.json';supports=json.loads(supports_path.read_text())['supports'];inputs[str(supports_path.relative_to(ROOT))]=sha(supports_path)
        other_full={r['support_index']:S.load(ROOT/'cache/custom_sphere_supports'/r['mesh_file']) for r in supports}
        local=[r for r in plan['records'] if r['layer_count']==n and r['placement'] is None]
        local_meshes={r['support_index']:([S.load(ROOT/p['path']) for p in r['paths']['cutters']],S.load(ROOT/r['paths']['kerf']['path']),S.load(ROOT/r['paths']['removable']['path'])) for r in local}
        after_removal={r['support_index']:S.load(ROOT/r['paths']['retained']['path']) for r in local}
        for r in local:
            i=r['support_index'];cutters,kerf,body=local_meshes[i]
            checks=dict(cutter_original_sphere_overlap_mm3=max(S.overlap(c,own_original) for c in cutters),kerf_original_sphere_overlap_mm3=S.overlap(kerf,own_original),free_body_original_sphere_overlap_mm3=S.overlap(body,own_original))
            neighbors=[]
            for j,other in other_full.items():
                if j==i:continue
                amount=max(S.overlap(c,other) for c in cutters)
                if amount>TOL:
                    permanent=after_removal.get(j,other);remaining=max(S.overlap(c,permanent) for c in cutters)
                    allowed=remaining<=TOL and j in after_removal
                    entry=dict(other_support_index=j,cutter_overlap_mm3=amount,overlap_after_other_removed_mm3=remaining,
                        resolved_by_removing_other_first=allowed)
                    neighbors.append(entry)
                    if allowed:
                        for placement,sphere in enumerate(SPHERES):
                            if sphere['layers']==n:dependencies.append(dict(layer_count=n,placement=placement,remove_support_index=j,before_cutting_support_index=i,initial_tool_overlap_mm3=amount,retained_overlap_mm3=remaining))
            row=dict(layer_count=n,support_index=i,placement=None,checks=checks,neighbor_collisions=neighbors,passed=max(checks.values())<=TOL and all(x['resolved_by_removing_other_first'] for x in neighbors))
            records.append(row)
            if not row['passed']:failures.append(row)
        for placement,sphere in enumerate(SPHERES):
            if sphere['layers']!=n:continue
            environment_path=ROOT/f'cache/support_environment/sphere_{n}layers_holder_{int(sphere["theta_deg"])}_source_float32.npz'
            environment=S.load(environment_path);inputs[str(environment_path.relative_to(ROOT))]=sha(environment_path);transform=sphere_transform(sphere)
            for r in local:
                i=r['support_index'];cutters,kerf,body=local_meshes[i]
                checks=dict(cutter_original_holder_environment_overlap_mm3=max(S.overlap(c,environment) for c in cutters),
                    kerf_original_holder_environment_overlap_mm3=S.overlap(kerf,environment),
                    free_body_original_holder_environment_overlap_mm3=S.overlap(body,environment),
                    cutter_added_holder_overlap_mm3=max(S.overlap(c.transform(transform),holder) for c in cutters))
                row=dict(layer_count=n,support_index=i,placement=placement,checks=checks,passed=max(checks.values())<=TOL)
                records.append(row)
                if not row['passed']:failures.append(row)
        print('L',n,'complete; failures',len(failures),flush=True)
    changed=[path for path,h in inputs.items() if sha(ROOT/path)!=h]
    report=dict(passed=not failures and not changed,scope='Local original sphere layers/wires plus each exact matched original-holder environment; complete current added-holder aggregate and all other added sphere parts. Passed requires the explicit remove-before-cut dependencies below. Independent direct original_v5 world check remains an additional audit.',
        overlap_tolerance_mm3=TOL,records=records,failures=failures,removal_dependencies=dependencies,inputs=inputs,changed_inputs=changed,source_sha256=sha(__file__),seconds=time.monotonic()-started)
    path=ROOT/'reports/sphere_removal_partition_local_context_v6.json';temp=path.with_suffix('.json.part');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)
    print(json.dumps(dict(passed=report['passed'],checks=len(records),failures=len(failures),changed_inputs=changed,seconds=report['seconds'])),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Fresh in-memory complete-union provenance-reset and bounded export trials."""
from pathlib import Path
import argparse
import gc
import json
import time
import numpy as np
import manifold3d as mf
from scipy.spatial import cKDTree
from build_holder import SPHERES,sphere_transform
from diagnose_v6_bounded_preparation import sha,save,write,strict_attempt,exact_geometry_equal
from audit_precision_transfer_v6 import derive_motion_transfer
from prepare_v6_export import v5_rounding_chain

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'cache/point_preparation_trials'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tolerances',type=float,nargs='+',default=[1e-5,1e-4,1e-3]);args=ap.parse_args()
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);inputs={}
    def remember(path):
        key=str(path.relative_to(ROOT));digest=sha(path)
        if key in inputs and inputs[key]!=digest:raise ValueError('Input changed')
        inputs[key]=digest
    def load(path):
        remember(path)
        with np.load(path,allow_pickle=False)as d:
            m=mf.Manifold(mf.Mesh64(np.asarray(d['vertices'],float),np.asarray(d['faces'],np.uint32)))
        if m.status()!=mf.Error.NoError:raise ValueError('Invalid immutable component')
        return m
    for path in [Path(__file__),ROOT/'source/diagnose_v6_bounded_preparation.py',ROOT/'source/build_test_coupon.py',ROOT/'source/build_holder.py',ROOT/'source/audit_precision_transfer_v6.py']:
        remember(path)
    raw_path=ROOT/'cache/assembled_v6_float64.npz';remember(raw_path)
    base=load(ROOT/'cache/original_v5.npz')
    spheres={n:load(ROOT/f'cache/custom_sphere_supports/sphere_{n}layers_supports.npz')for n in [3,4,5]}
    parts=[load(ROOT/'cache/custom_holder_supports/holder_supports.npz')]
    parts.extend(spheres[s['layers']].transform(sphere_transform(s))for s in SPHERES)
    print('Reconstruct fresh additions from frozen components',flush=True)
    additions=mf.Manifold.batch_boolean(parts,mf.OpType.Add);additions.to_mesh64()
    print('Reconstruct fresh complete raw union',flush=True)
    raw=base+additions;before=raw.to_mesh64()
    v0=np.asarray(before.vert_properties[:,:3]);f0=np.asarray(before.tri_verts)
    with np.load(raw_path,allow_pickle=False)as data:
        same=exact_geometry_equal(np.asarray(data['vertices'],float),np.asarray(data['faces'],np.uint32),v0,f0)
    print('Same complete frozen raw triangle surface:',same,flush=True)
    if not same['passed']:raise ValueError('Fresh reconstruction differs from the saved raw; do not reuse its proof')
    reset=raw.as_original();after=reset.to_mesh64()
    same_reset=exact_geometry_equal(v0,f0,np.asarray(after.vert_properties[:,:3]),np.asarray(after.tri_verts))
    print('as_original complete geometry match:',same_reset,flush=True)
    if not same_reset['passed']:raise ValueError('as_original changed actual geometry')
    source_vertices=v0.copy()
    del base,spheres,parts,additions,raw,before,after,v0,f0;gc.collect()
    tree=cKDTree(source_vertices)
    gap_path=ROOT/'reports/v6_final_additions_motion_context.json';remember(gap_path)
    gap_context=json.loads(gap_path.read_text());gap_checks=gap_context['all_angle_moving_envelope_checks']
    chain=v5_rounding_chain(ROOT);trials=[]
    for tolerance in args.tolerances:
        if tolerance>1e-3:raise ValueError('Trial exceeds authorized tolerance')
        start=time.monotonic();name=f'fresh_as_original_{tolerance:.0e}mm';stem=OUT/name
        print('Trial',name,flush=True)
        model=reset.simplify(tolerance);mesh=model.to_mesh64();v=np.asarray(mesh.vert_properties[:,:3]);f=np.asarray(mesh.tri_verts)
        subset_distance=0.
        for off in range(0,len(v),65536):
            distance,_=tree.query(v[off:off+65536],workers=1);subset_distance=max(subset_distance,float(distance.max(initial=0.)))
        if subset_distance!=0.:raise ValueError('Simplify moved a retained vertex')
        path=stem.with_name(stem.name+'_float64.npz');save(path,v,f)
        final_round=float(np.linalg.norm(v.astype(np.float32).astype(float)-v,axis=1).max())
        effective=max(tolerance,float(reset.get_tolerance()))
        sigma=chain['prior_per_surface_rounding_upper_bound_mm']+effective+final_round
        transfer=derive_motion_transfer(sigma,root=ROOT,added_gap_checks=gap_checks)
        strict=strict_attempt(model,stem)
        strict.update(name=name,requested_tolerance_mm=tolerance,effective_tolerance_mm=effective,
            float64_file=str(path.relative_to(ROOT)),float64_sha256=sha(path),vertices=len(v),triangles=len(f),
            vertices_exact_subset_of_complete_raw=True,maximum_retained_vertex_displacement_mm=0.,
            final_rounding_error_mm=final_round,complete_source_to_final_surface_error_upper_bound_mm=sigma,
            scalar_motion_transfer=transfer,elapsed_seconds=time.monotonic()-start)
        trials.append(strict)
        report=dict(passed=bool(strict['passed']and transfer['mathematical_transfer_passed']),diagnostic_only=True,
            original_and_adopted_meshes_unmodified=True,raw_reimport_not_used=True,
            reconstructed_raw_same_complete_geometry=same,as_original_same_complete_geometry=same_reset,
            source_raw_sha256=inputs[str(raw_path.relative_to(ROOT))],source_chain=chain,trials=trials,
            no_component_filtering=True,unchanged_strict_validator=True,input_sha256=inputs,
            actual_all_placement_gap_binding_present=True,global_material_birth_gate_required=True,
            elapsed_seconds=time.monotonic()-started)
        write(ROOT/'reports/v6_point_bounded_preparation_trials.json',report)
        print(name,'strict',strict['passed'],'motion',transfer['mathematical_transfer_passed'],
            'triangles',len(f),'seconds',strict['elapsed_seconds'],
            'failure',strict.get('failure_reason'),flush=True)
        del model,mesh,v,f;gc.collect()
        if strict['passed']and transfer['mathematical_transfer_passed']:break
    for path,digest in inputs.items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen trial input changed: '+path)

if __name__=='__main__':main()

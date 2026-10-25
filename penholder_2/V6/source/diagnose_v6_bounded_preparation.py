#!/usr/bin/env python3
"""Bounded, unfiltered representation trials of the frozen complete V6 union.

No source/adopted mesh is changed. Every trial starts from the same complete
raw Mesh64, resets face provenance only, applies one documented simplify
operation, and invokes the unchanged strict float32 canonicalizer.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import time
import numpy as np
import manifold3d as mf
from scipy.spatial import cKDTree
import build_test_coupon as B
from audit_precision_transfer_v6 import derive_motion_transfer
from prepare_v6_export import v5_rounding_chain
from prove_mesh_import_surface_v6 import prove as prove_import_surface

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'cache/bounded_preparation_trials'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,obj):
    temp=Path(str(path)+'.part');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)

def save(path,vertices,faces):
    temp=Path(str(path)+'.part.npz');np.savez_compressed(temp,vertices=vertices,faces=faces);temp.replace(path)

def inspect_degeneracies(path):
    with np.load(path,allow_pickle=False) as d:
        v=np.asarray(d['vertices'],float);f=np.asarray(d['faces'],np.uint32)
    q=v.astype(np.float32).astype(float);rows=[];collapsed=0;zero=0;near=0;minimum_nonzero=float('inf')
    max_round=0.
    for start in range(0,len(v),100000):
        max_round=max(max_round,float(np.linalg.norm(q[start:start+100000]-v[start:start+100000],axis=1).max(initial=0.)))
    for start in range(0,len(f),65536):
        ff=f[start:start+65536];tri=q[ff]
        collapsed_mask=np.any(np.all(tri==tri[:,[1,2,0]],axis=2),axis=1)
        collapsed+=int(collapsed_mask.sum())
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        zero+=int(((area==0)&~collapsed_mask).sum())
        near_mask=(area<=1e-14)&~collapsed_mask;near+=int(near_mask.sum())
        nz=area[(area>0)&~collapsed_mask]
        if len(nz):minimum_nonzero=min(minimum_nonzero,float(nz.min()))
        for k in np.flatnonzero(near_mask):
            original=v[ff[k]]
            original_area=float(np.linalg.norm(np.cross(original[1]-original[0],original[2]-original[0]))/2)
            rows.append(dict(face_index=int(start+k),float32_vertices_mm=tri[k].tolist(),
                original_vertices_mm=original.tolist(),original_area_mm2=original_area,
                float32_area_mm2=float(area[k]),center_mm=tri[k].mean(0).tolist(),
                edge_lengths_mm=np.linalg.norm(tri[k]-tri[k][[1,2,0]],axis=1).tolist()))
    return dict(input_file=str(path.relative_to(ROOT)),input_sha256=sha(path),vertices=len(v),triangles=len(f),
        collapsed_coordinate_faces=collapsed,zero_area_noncollapsed=zero,near_zero_noncollapsed=near,
        threshold_mm2=1e-14,minimum_nonzero_area_mm2=minimum_nonzero,
        maximum_vertex_rounding_error_mm=max_round,degenerate_faces=rows)

def cyclic_order(f):
    which=np.argmin(f,axis=1)
    return np.take_along_axis(f,(which[:,None]+np.arange(3)[None,:])%3,axis=1)

def exact_geometry_equal(v0,f0,v1,f1):
    """Exact complete oriented triangle multiset check, ignoring index order."""
    if len(v0)==len(v1)and len(f0)==len(f1)and np.array_equal(v0,v1)and np.array_equal(f0,f1):
        return dict(passed=True,method='Exact identical vertex and triangle arrays')
    if len(f0)!=len(f1):return dict(passed=False,reason='Triangle count changed',before=len(f0),after=len(f1))
    unique,inverse=np.unique(v0,axis=0,return_inverse=True)
    tree=cKDTree(unique);dist,index=tree.query(v1,workers=1)
    if np.any(dist):return dict(passed=False,reason='Vertex coordinates changed',maximum=float(dist.max()))
    t0=cyclic_order(inverse[f0]);t1=cyclic_order(index[f1])
    o0=np.lexsort((t0[:,2],t0[:,1],t0[:,0]));o1=np.lexsort((t1[:,2],t1[:,1],t1[:,0]))
    equal=np.array_equal(t0[o0],t1[o1])
    return dict(passed=bool(equal),method='Exact complete oriented triangle coordinate multiset')

def import_collapse_bound(v0,f0,v1,f1):
    """Prove any importer collapse as a bounded complete-surface vertex map.

    This does not filter the trial mesh. It compares both complete meshes.
    Every noncollapsed mapped source face must equal an output face, with
    multiplicity and orientation. A mapped degenerate source face must lie
    on an actual output edge/vertex. Barycentric interpolation then bounds
    both directed surface distances by the largest mapped-vertex movement.
    """
    unique,new_inverse=np.unique(v1,axis=0,return_inverse=True)
    tree=cKDTree(unique);distance,old_inverse=tree.query(v0,workers=1)
    bound=float(distance.max(initial=0.))
    mapped=old_inverse[f0];collapsed=((mapped[:,0]==mapped[:,1])|(mapped[:,1]==mapped[:,2])|(mapped[:,2]==mapped[:,0]))
    old=cyclic_order(mapped[~collapsed]);new=cyclic_order(new_inverse[f1])
    if len(old)!=len(new):
        return dict(passed=False,reason='Imported changes exceed a collapsed-index vertex map',
            largest_nearest_vertex_movement_mm=bound,mapped_noncollapsed_triangles=len(old),output_triangles=len(new),
            collapsed_source_triangles=int(collapsed.sum()))
    so=np.lexsort((old[:,2],old[:,1],old[:,0]));sn=np.lexsort((new[:,2],new[:,1],new[:,0]))
    if not np.array_equal(old[so],new[sn]):
        return dict(passed=False,reason='Complete oriented mapped face multisets differ',largest_nearest_vertex_movement_mm=bound,
            collapsed_source_triangles=int(collapsed.sum()))
    collapsed_faces=mapped[collapsed];needed=set()
    for triangle in collapsed_faces:
        ends=np.unique(triangle)
        if len(ends)==2:needed.add((int(ends[0]),int(ends[1])))
    missing=set(needed)
    # Only the few collapsed edge endpoints need membership checks.
    if missing:
        by_vertex={}
        for a,b in missing:by_vertex.setdefault(a,set()).add(b)
        for a,targets in by_vertex.items():
            incident=new[np.any(new==a,axis=1)]
            adjacent=set(map(int,incident.ravel()))
            for b in targets:
                if b in adjacent:missing.discard((a,b))
    return dict(passed=not missing,maximum_vertex_displacement_mm=bound,
        complete_bidirectional_surface_distance_upper_bound_mm=bound,
        complete_oriented_nondegenerate_face_multiset_equal=True,
        collapsed_source_triangles=len(collapsed_faces),collapsed_edges=len(needed),
        all_collapsed_faces_map_to_actual_output_edges_or_vertices=not missing,
        missing_output_edges=[list(e)for e in sorted(missing)],
        proof='The complete source triangulation maps by bounded vertex movement to exactly the complete output triangulation. Every collapsed face maps to an actual output edge or vertex. Barycentric interpolation bounds all source/output surface points, including every removed degenerate triangle, by the maximum vertex displacement.')

def strict_attempt(m,out):
    try:
        v,f,report=B.canonical_float32_mesh(m)
        path=out.with_name(out.name+'_float32.npz');save(path,v,f)
        return dict(passed=True,topology=report,float32_file=str(path.relative_to(ROOT)),float32_sha256=sha(path))
    except Exception as error:
        result=dict(passed=False,failure_reason=str(error));tb=error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_name=='canonical_float32_mesh':
                state=tb.tb_frame.f_locals
                for key in ['collapsed','zero_area','near_zero','bad_edges','error']:
                    if key in state:result[key]=float(state[key])if key=='error'else int(state[key])
                if 'components'in state:result['component_signed_volumes_mm3']=[float(x.volume())for x in state['components']]
                if 'point_contacts'in state:result['nonmanifold_vertex_count']=len(state['point_contacts'])
                if 'vertices'in state and'faces'in state:
                    path=out.with_name(out.name+'_failed_float32.npz');save(path,state['vertices'],state['faces'])
                    result['failed_float32_file']=str(path.relative_to(ROOT));result['failed_float32_sha256']=sha(path)
            tb=tb.tb_next
        return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inspect-only',action='store_true');ap.add_argument('--tolerances',type=float,nargs='+',default=[1e-5,1e-4,1e-3]);ap.add_argument('--gap-report',type=Path)
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    raw_path=ROOT/'cache/assembled_v6_float64.npz';prepared_path=ROOT/'cache/assembled_v6_prepared_float64.npz'
    inputs={str(p.relative_to(ROOT)):sha(p)for p in [raw_path,prepared_path,Path(__file__),ROOT/'source/build_test_coupon.py',ROOT/'source/audit_precision_transfer_v6.py']}
    diagnostic=inspect_degeneracies(prepared_path)
    write(ROOT/'reports/v6_prepared_degeneracy_diagnostic.json',diagnostic)
    print('Existing prepared degeneracies:',diagnostic['zero_area_noncollapsed'],diagnostic['near_zero_noncollapsed'],'collapsed:',diagnostic['collapsed_coordinate_faces'],flush=True)
    if args.inspect_only:return
    with np.load(raw_path,allow_pickle=False)as d:
        source_v=np.asarray(d['vertices'],float);source_f=np.asarray(d['faces'],np.uint32)
    # Explicit per-triangle face IDs prevent import-time coplanar face
    # consolidation. Their removal is deferred to the separately documented
    # as_original/simplify stage, rather than an unbudgeted constructor step.
    raw=mf.Manifold(mf.Mesh64(source_v,source_f,face_id=np.arange(len(source_f),dtype=np.uint64)))
    if raw.status()!=mf.Error.NoError:raise ValueError('Raw reload failed')
    reloaded=raw.to_mesh64();reset=raw.as_original();rr=reset.to_mesh64()
    same_load=exact_geometry_equal(source_v,source_f,np.asarray(reloaded.vert_properties[:,:3]),np.asarray(reloaded.tri_verts))
    same_reset=exact_geometry_equal(np.asarray(reloaded.vert_properties[:,:3]),np.asarray(reloaded.tri_verts),np.asarray(rr.vert_properties[:,:3]),np.asarray(rr.tri_verts))
    reload_bound=(dict(passed=True,complete_bidirectional_surface_distance_upper_bound_mm=0.,method=same_load['method'])if same_load['passed']else import_collapse_bound(source_v,source_f,np.asarray(reloaded.vert_properties[:,:3]),np.asarray(reloaded.tri_verts)))
    if not reload_bound['passed']:
        reload_bound=prove_import_surface(source_v,source_f,np.asarray(reloaded.vert_properties[:,:3]),np.asarray(reloaded.tri_verts))
        write(ROOT/'reports/v6_raw_import_surface_proof.json',reload_bound)
    if not reload_bound['passed']or not same_reset['passed']:
        write(ROOT/'reports/v6_bounded_preparation_trials.json',dict(passed=False,raw_reload_same_geometry=same_load,raw_reload_surface_mapping=reload_bound,as_original_same_geometry=same_reset,input_sha256=inputs))
        raise ValueError('Importer surface mapping is not proved; no trial proceeded')
    print('Raw import full-surface bound:',reload_bound['complete_bidirectional_surface_distance_upper_bound_mm'],'as_original retains arrays exactly',flush=True)
    del reloaded,rr,source_f,raw;gc.collect();tree=cKDTree(source_v)
    chain=v5_rounding_chain(ROOT);rows=[];gap_checks=None
    if args.gap_report:
        gap_source=json.loads(args.gap_report.read_text());gap_checks=gap_source['all_angle_moving_envelope_checks']
        inputs[str(args.gap_report.relative_to(ROOT))]=sha(args.gap_report)
    for tolerance in args.tolerances:
        if tolerance>1e-3:raise ValueError('Diagnostic tolerance exceeds authorized bound')
        t0=time.monotonic();name=f'reloaded_as_original_{tolerance:.0e}mm';stem=OUT/name
        print('Trial',name,flush=True);m=reset.simplify(tolerance);mm=m.to_mesh64()
        v=np.asarray(mm.vert_properties[:,:3]);f=np.asarray(mm.tri_verts);maxdist=0.
        for off in range(0,len(v),65536):
            distance,_=tree.query(v[off:off+65536],workers=1);maxdist=max(maxdist,float(distance.max(initial=0.)))
        if maxdist!=0.:raise ValueError('Simplified vertex was not in the frozen complete raw input')
        path=stem.with_name(stem.name+'_float64.npz');save(path,v,f)
        final_round=float(np.linalg.norm(v.astype(np.float32).astype(float)-v,axis=1).max())
        effective=max(float(reset.get_tolerance()),tolerance)
        sigma=chain['prior_per_surface_rounding_upper_bound_mm']+reload_bound['complete_bidirectional_surface_distance_upper_bound_mm']+effective+final_round
        transfer=derive_motion_transfer(sigma,root=ROOT,added_gap_checks=gap_checks)
        result=strict_attempt(m,stem)
        result.update(name=name,requested_tolerance_mm=tolerance,effective_tolerance_mm=effective,
            raw_mesh_input_sha256=inputs[str(raw_path.relative_to(ROOT))],
            float64_file=str(path.relative_to(ROOT)),float64_sha256=sha(path),vertices=len(v),triangles=len(f),
            vertices_exact_subset_of_complete_raw=True,final_rounding_error_mm=final_round,
            complete_source_to_final_surface_error_upper_bound_mm=sigma,scalar_motion_transfer=transfer,
            elapsed_seconds=time.monotonic()-t0)
        rows.append(result)
        report=dict(passed=bool(result['passed']and transfer['mathematical_transfer_passed']),diagnostic_only=True,
            no_source_or_adopted_input_modified=True,raw_reload_same_geometry=same_load,as_original_same_geometry=same_reset,
            raw_reload_surface_mapping=reload_bound,
            raw_import_has_unique_per_triangle_face_ids=True,
            no_component_filtering=True,unchanged_strict_validator=True,source_chain=chain,trials=rows,
            input_sha256=inputs,elapsed_seconds=time.monotonic()-start,
            global_material_birth_gate_required=True,actual_all_placement_gap_binding_present=gap_checks is not None)
        write(ROOT/'reports/v6_bounded_preparation_trials.json',report)
        print(name,'strict',result['passed'],'motion',transfer['mathematical_transfer_passed'], 'triangles',len(f),'elapsed',result['elapsed_seconds'],flush=True)
        del m,mm,v,f;gc.collect()
        if result['passed']and transfer['mathematical_transfer_passed']:break
    for name,digest in inputs.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen input changed during bounded trial: '+name)

if __name__=='__main__':main()

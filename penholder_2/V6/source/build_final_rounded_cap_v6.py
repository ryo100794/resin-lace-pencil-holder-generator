#!/usr/bin/env python3
"""One explicit convex rounded fixed-cap expansion enclosing every remaining defect."""
from pathlib import Path
import json,time
import numpy as np
import manifold3d as mf
import diagnose_fixed_precision_v6 as D
from build_fixed_precision_repairs_v6 import canonical_probe
ROOT=D.ROOT;OUT=ROOT/'cache/rim_cap_bedding_v6/final_rounded_cap'

def dump(p,d):
 t=p.with_suffix(p.suffix+'.part');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def main():
 OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 dp=ROOT/'reports/final_all_gates_diagnostic_v6.json';hp=ROOT/'cache/custom_holder_supports/holder_supports.json';cp=ROOT/'reports/rim_cap_bedding_candidates_v6.json';pp=ROOT/'cache/assembled_v6_prepared_float64.npz';rp=ROOT/'cache/assembled_v6_float64.npz'
 inputs={str(p.relative_to(ROOT)):D.sha(p)for p in[dp,hp,cp,pp,rp,Path(__file__)]};diag=json.loads(dp.read_text());holder=json.loads(hp.read_text());parts={r['index']:r for r in holder['parts']}
 if diag['bad_edge_count']!=1 or diag['zero_area_count']or diag['near_zero_area_count']or diag['independent_nonmanifold_point_count']or diag['inconsistent_two_face_edge_count']or diag['regular_edge_face_component_count']!=2:raise ValueError('Unexpected complete defect inventory')
 if D.sha(pp)!=diag['source_sha256']:raise ValueError('Prepared diagnostic input changed')
 oldpath=hp.parent/parts[1303]['full_mesh_file'];old=D.load(oldpath);inputs[str(oldpath.relative_to(ROOT))]=D.sha(oldpath)
 historical=next(r for r in json.loads(cp.read_text())['repairs']if r['affected_support_index']==450)
 if D.sha(oldpath)!=historical['mesh_sha256']:raise ValueError('Expected unchanged fourth symmetric pad')
 source=parts[450];fullpath=hp.parent/source['full_mesh_file'];rempath=hp.parent/source['removable_mesh_file'];full=D.load(fullpath);rem=D.load(rempath)
 for p in[fullpath,rempath]:inputs[str(p.relative_to(ROOT))]=D.sha(p)
 # Recover the entire four-triangle positive shell from the exact canonical
 # coordinates. The reported component box contains exactly those four faces.
 component=diag['regular_edge_face_components'][1];bounds=np.asarray(component['bounds_mm'])
 with np.load(pp,allow_pickle=False)as data:v=np.asarray(data['vertices'],np.float32).astype(float);f=np.asarray(data['faces'],np.uint32)
 mask=np.all(v>=bounds[:3]-1e-9,axis=1)&np.all(v<=bounds[3:]+1e-9,axis=1);tri=v[f[np.all(mask[f],axis=1)]];tri=tri[~np.any(np.all(tri==tri[:,[1,2,0]],axis=2),axis=1)]
 if len(tri)!=4:raise ValueError('Did not recover the complete tiny shell')
 vv,iv=np.unique(tri.reshape(-1,3),axis=0,return_inverse=True);lobe=mf.Manifold(mf.Mesh64(vv,iv.reshape(-1,3).astype(np.uint32)))
 if lobe.status()!=mf.Error.NoError or abs(lobe.volume()-component['signed_centered_volume_mm3'])>1e-15:raise ValueError('Recovered tiny shell differs from all-gates diagnostic')
 lpath=OUT/'complete_positive_sliver.npz';D.save(lpath,lobe);del v,f,mask,tri
 oldbb=np.asarray(old.bounding_box());floor=float(oldbb[2]);radius=.07
 center=np.array([oldbb[3]-.006,oldbb[1]+.006,floor+radius+.005])
 sphere=mf.Manifold.sphere(radius,24).translate(center)
 patch=mf.Manifold.hull_points(np.vstack((np.asarray(old.to_mesh64().vert_properties)[:,:3],np.asarray(sphere.to_mesh64().vert_properties)[:,:3])))
 if not(old-patch).is_empty():raise ValueError('Convex rounded replacement removed old fixed material')
 if not(lobe-patch).is_empty():raise ValueError('Rounded replacement does not cover complete positive shell')
 if abs(patch.bounding_box()[2]-floor)>1e-10:raise ValueError('Original rooted floor changed')
 first=patch.trim_by_plane((0,0,-1),-(floor+.00005));contact=first^(full-rem)
 if contact.is_empty()or contact.volume()<=1e-12 or abs(contact.bounding_box()[2]-floor)>1e-8:raise ValueError('Rounded replacement lacks persistent fixed first-face seating')
 if not(patch^rem).is_empty():raise ValueError('Rounded replacement intersects own removable body')
 offset=float(((np.asarray(patch.to_mesh64().vert_properties)[:,:3]-np.asarray(source['target_mm']))@np.asarray(source['neck_axes'][1])).max())
 if offset>=.075:raise ValueError('Rounded replacement leaves permanent fixed head')
 print('Read current full raw only for the scoped local strict diagnostic',flush=True)
 raw=D.load(rp);bb=np.asarray(patch.bounding_box());context=raw^D.box(bb[:3]-[.22,.22,.12],bb[3:]+[.22,.22,.12]);del raw
 before=canonical_probe(context.as_original());after=canonical_probe((context+patch).as_original())
 if not after['passed']:raise ValueError('Rounded replacement failed unchanged local strict gate: '+str(after))
 components=sorted(context.decompose(),key=lambda m:-float(m.volume()));reference=context;reference_note='Complete current raw crop; no component filtering'
 if len(components)==2 and 0<float(components[1].volume())<1e-5 and(components[1]-components[0]).is_empty():
  reference=components[0];reference_note='For net occupied-volume accounting only, the complete tiny positive component is exactly contained in the large component; full unfiltered raw context is used for the actual Boolean and strict check.'
 path=OUT/'H1303_rounded_fixed_cap.npz';D.save(path,patch);contextpath=OUT/'H1303_current_raw_context.npz';D.save(contextpath,context)
 row=dict(id='final_rounded_cap_H1303',replacement_holder_index=1303,affected_support_index=450,startup_dependency_support_index=450,
  kind='fixed_micro_reinforcement',temporary=False,repair_purpose='complete_remaining_positive_sliver_enclosure_with_rounded_convex_fixed_cap',
  mesh_file=str(path.relative_to(ROOT)),mesh_sha256=D.sha(path),old_piece_file=str(oldpath.relative_to(ROOT)),old_piece_sha256=D.sha(oldpath),
  complete_old_piece_contained=True,old_piece_outside_replacement_empty=True,original_floor_unchanged=True,
  sphere_radius_mm=radius,sphere_segments=24,sphere_center_mm=center.tolist(),sphere_lower_pole_above_old_floor_mm=.005,
  construction='Convex hull of the complete old fixed box and one raised faceted sphere; no old material or temporary support geometry removed.',
  complete_diagnosed_sliver_file=str(lpath.relative_to(ROOT)),complete_diagnosed_sliver_sha256=D.sha(lpath),complete_diagnosed_sliver_volume_mm3=float(lobe.volume()),complete_sliver_outside_replacement_empty=True,
  all_remaining_topology_defects_enclosed=True,source_all_gates_diagnostic_file=str(dp.relative_to(ROOT)),source_all_gates_diagnostic_sha256=D.sha(dp),
  target_mm=((bb[:3]+bb[3:])/2).tolist(),root_mm=[float((bb[0]+bb[3])/2),float((bb[1]+bb[4])/2),floor],bounds_mm=bb.tolist(),neck_count=0,volume_mm3=float(patch.volume()),
  actual_added_volume_over_occupied_context_mm3=float((patch-reference).volume()),occupied_context_definition=reference_note,
  maximum_box_dimension_mm=float((bb[3:]-bb[:3]).max()),source_support_removable_gap_mm=float(patch.min_gap(rem,1.)),maximum_outward_axis_offset_from_contact_mm=offset,
  first_prefix_height_mm=floor+.00005,first_prefix_contact_volume_mm3=float(contact.volume()),first_prefix_contact_fraction=float(contact.volume()/first.volume()),
  nearby_existing_fixed_context=[dict(owner='holder',index=450)],
  final_context_file=str(contextpath.relative_to(ROOT)),final_context_sha256=D.sha(contextpath),
  local_context_scope='Diagnostic crop imported from complete saved raw Mesh64. Exact fresh whole-model assembly and unchanged strict validation remain mandatory; raw import is not asserted to preserve every source triangle.',
  local_before_strict_probe=before,local_after_strict_probe=after,local_geometry_passed=True,
  persistent_fixed_contact=dict(holder_support_indices=[450],nominal_retained_stock_used=True,contact_volume_mm3=float(contact.volume()),first_prefix_height_mm=floor+.00005,first_contact_bottom_mm=float(contact.bounding_box()[2]),patch_bottom_mm=floor,patch_is_convex=True,all_positive_height_prefixes_connected=True,passed=True),
  existing_temporary_and_cut_geometry_unchanged=True,historical_bedding_record=historical)
 for rel,digest in inputs.items():
  if D.sha(ROOT/rel)!=digest:raise ValueError('Input changed during construction')
 report=dict(local_geometry_passed=True,complete_exclusion_checks_required=True,global_validation_required=True,source_models_modified=False,repair_count=1,replacements_only_no_extra_part_count=True,units='mm',original_model_and_all_temporary_geometry_unchanged=True,
  aggregate_mesh_file=str(path.relative_to(ROOT)),aggregate_sha256=D.sha(path),aggregate_volume_mm3=float(patch.volume()),repairs=[row],input_sha256=inputs,source_sha256=D.sha(__file__),elapsed_seconds=time.monotonic()-start)
 dump(ROOT/'reports/final_rounded_cap_candidates_v6.json',report);print('DONE',row['local_after_strict_probe'],'net added',row['actual_added_volume_over_occupied_context_mm3'],'gap',row['source_support_removable_gap_mm'],report['elapsed_seconds'],flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Convex, positively padded caps cover the complete last floating families."""
from pathlib import Path
import json,time,shutil
import numpy as np
import manifold3d as mf
import diagnose_fixed_precision_v6 as D
from build_fixed_precision_repairs_v6 import canonical_probe
ROOT=D.ROOT;OUT=ROOT/'cache/residual_rounded_caps_v6';EXPECTED='9027aa1127d8628e2fed58e384257c6e3f4b4f9c1cdce594013193a3202dd018'

def dump(p,d):
 t=p.with_suffix(p.suffix+'.part');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def component_at(band,z,point):
 prefix=band.trim_by_plane((0,0,-1),-z);matches=[]
 for c in prefix.decompose():
  if float(c.volume())<=1e-12:continue
  v=np.asarray(c.to_mesh64().vert_properties)[:,:3];distance=float(np.linalg.norm(v-point,axis=1).min())
  if distance<1e-6:matches.append(c)
 floating=[c for c in matches if c.bounding_box()[2]>99.2+1e-7]
 if floating:
  if len(floating)!=1:raise ValueError('Ambiguous complete floating component')
  return False,floating[0]
 rooted=[c for c in matches if c.bounding_box()[2]<=99.2+1e-7]
 if not rooted:raise ValueError('Birth vertex missing from actual prefix')
 return True,rooted[0]

def inward_padding(container,points):
 mm=container.to_mesh64();v=np.asarray(mm.vert_properties)[:,:3];tri=v[np.asarray(mm.tri_verts)];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(n,axis=1);n=n/length[:,None];h=np.einsum('ij,ij->i',n,tri[:,0]);return float((h-(points@n.T).max(0)).min())

def main():
 OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();vp=ROOT/'reports/residual_cap_prefix_validation_v6.json';hp=ROOT/'cache/custom_holder_supports/holder_supports.json';mp=ROOT/'cache/print_ready_v6.npz';rp=ROOT/'cache/assembled_v6_float64.npz';ap=ROOT/'reports/custom_supports_validation.json'
 inputs={str(p.relative_to(ROOT)):D.sha(p)for p in[vp,hp,mp,rp,ap,Path(__file__)]};validation=json.loads(vp.read_text());holder=json.loads(hp.read_text());parts={r['index']:r for r in holder['parts']}
 if D.sha(mp)!=EXPECTED or not validation['complete']or validation['source_final_mesh_sha256']!=EXPECTED:raise ValueError('Unexpected strict prefix source')
 print('Load strict source and current raw only once; use full-XY bands',flush=True)
 m=D.load(mp);band=m.trim_by_plane((0,0,1),99.2).trim_by_plane((0,0,-1),-100.2);del m
 raw=D.load(rp);rawband=raw.trim_by_plane((0,0,1),99.2).trim_by_plane((0,0,-1),-100.2);del raw
 rows=[];patches=[]
 for event in validation['events']:
  index=event['pad_index'];owner=event['source_support_index'];point=np.asarray(event['position_mm']);oldpath=hp.parent/parts[index]['full_mesh_file'];old=D.load(oldpath);inputs[str(oldpath.relative_to(ROOT))]=D.sha(oldpath)
  source=parts[owner];fullpath=hp.parent/source['full_mesh_file'];rempath=hp.parent/source['removable_mesh_file'];full=D.load(fullpath);rem=D.load(rempath)
  for p in[fullpath,rempath]:inputs[str(p.relative_to(ROOT))]=D.sha(p)
  low=event['surface_join_z_mm']-1e-7;high=event['surface_join_z_mm']+1e-7
  isroot,last=component_at(band,low,point)
  if isroot:raise ValueError('Expected complete floating component before candidate join')
  rooted,root_component=component_at(band,high,point);checks=[dict(height_mm=low,rooted=False),dict(height_mm=high,rooted=rooted)]
  if not rooted:
   low=high;high=99.63;rooted,root_component=component_at(band,high,point)
   if not rooted:raise ValueError('Floating family extends past verified Z99.63')
   while high-low>1e-7:
    middle=(low+high)/2;rooted,current=component_at(band,middle,point);checks.append(dict(height_mm=middle,rooted=rooted))
    if rooted:high=middle;root_component=current
    else:low=middle;last=current
  lv=np.asarray(last.to_mesh64().vert_properties)[:,:3];lpath=OUT/f'H{index:04d}_complete_last_floating_family.npz';D.save(lpath,last)
  oldbb=np.asarray(old.bounding_box());floor=float(oldbb[2]);side=int(np.argmin(np.r_[point-oldbb[:3],oldbb[3:]-point]));axis=side%3
  if axis==2:raise ValueError('Expected a side-plane residual')
  outward=np.zeros(3);outward[axis]=-1 if side<3 else 1
  trials=[]
  for radius in[.12,.16]:
   center=point-outward*.008;center[2]=floor+radius+.005;sphere=mf.Manifold.sphere(radius,28).translate(center)
   patch=mf.Manifold.hull_points(np.vstack((np.asarray(old.to_mesh64().vert_properties)[:,:3],np.asarray(sphere.to_mesh64().vert_properties)[:,:3])))
   margin=inward_padding(patch,lv)
   if not(old-patch).is_empty()or not(last-patch).is_empty():raise ValueError('Candidate loses old material or complete floating family')
   if margin<.01:trials.append(dict(radius_mm=radius,minimum_family_inward_padding_mm=margin,passed=False,reason='Padding below0.01 mm'));continue
   if abs(patch.bounding_box()[2]-floor)>1e-9:raise ValueError('Existing rooted floor changed')
   bb=np.asarray(patch.bounding_box());context=rawband^D.box(bb[:3]-[.2,.2,.08],bb[3:]+[.2,.2,.08]);before=canonical_probe(context.as_original());after=canonical_probe((context+patch).as_original())
   trials.append(dict(radius_mm=radius,minimum_family_inward_padding_mm=margin,before=before,after=after,passed=after['passed']))
   if after['passed']:break
  else:raise ValueError(f'H{index} rounded trials failed: {trials}')
  prefix=patch.trim_by_plane((0,0,-1),-(floor+.00005));contact=prefix^(full-rem)
  if contact.is_empty()or contact.volume()<=1e-12 or abs(contact.bounding_box()[2]-floor)>1e-8:raise ValueError('No persistent fixed first-face contact')
  if not(patch^rem).is_empty():raise ValueError('Rounded pad intersects own removable stock')
  offset=float(((np.asarray(patch.to_mesh64().vert_properties)[:,:3]-np.asarray(source['target_mm']))@np.asarray(source['neck_axes'][1])).max())
  if offset>=.075:raise ValueError('Rounded pad leaves fixed contact-head side')
  ident=f'residual_rounded_cap_H{index:04d}';path=OUT/(ident+'.npz');D.save(path,patch);contextpath=OUT/(ident+'_raw_context.npz');D.save(contextpath,context);patches.append(patch)
  row=dict(id=ident,replacement_holder_index=index,affected_support_index=owner,startup_dependency_support_index=owner,kind='fixed_micro_reinforcement',temporary=False,repair_purpose='positive_rounded_enclosure_of_complete_last_floating_family',mesh_file=str(path.relative_to(ROOT)),mesh_sha256=D.sha(path),old_piece_file=str(oldpath.relative_to(ROOT)),old_piece_sha256=D.sha(oldpath),complete_old_piece_contained=True,old_piece_outside_replacement_empty=True,original_floor_unchanged=True,
   source_final_event=event,complete_last_floating_family_file=str(lpath.relative_to(ROOT)),complete_last_floating_family_sha256=D.sha(lpath),complete_last_floating_family_volume_mm3=float(last.volume()),complete_last_floating_family_bounds_mm=list(last.bounding_box()),actual_last_floating_probe_z_mm=low,actual_rooted_connection_probe_z_mm=high,actual_rooted_prefix_touches_z99_2=True,root_connection_checks=checks,complete_family_outside_replacement_empty=True,minimum_complete_family_inward_padding_mm=margin,requirement_mm=.01,
   sphere_radius_mm=radius,sphere_segments=28,sphere_center_mm=center.tolist(),sphere_lower_pole_above_old_floor_mm=.005,construction='Convex hull of complete old fixed pad and a raised faceted sphere; the complete last actual floating component has at least0.01 mm of inward supporting-plane clearance.',rounded_trials=trials,
   target_mm=((bb[:3]+bb[3:])/2).tolist(),root_mm=[float((bb[0]+bb[3])/2),float((bb[1]+bb[4])/2),floor],bounds_mm=bb.tolist(),neck_count=0,volume_mm3=float(patch.volume()),actual_added_volume_over_occupied_context_mm3=float((patch-context).volume()),maximum_box_dimension_mm=float((bb[3:]-bb[:3]).max()),source_support_removable_gap_mm=float(patch.min_gap(rem,1.)),maximum_outward_axis_offset_from_contact_mm=offset,
   first_prefix_height_mm=floor+.00005,first_prefix_contact_volume_mm3=float(contact.volume()),first_prefix_contact_fraction=float(contact.volume()/prefix.volume()),nearby_existing_fixed_context=[dict(owner='holder',index=owner)],final_context_file=str(contextpath.relative_to(ROOT)),final_context_sha256=D.sha(contextpath),local_before_strict_probe=before,local_after_strict_probe=after,local_geometry_passed=True,
   local_context_scope='Current raw crop is a diagnostic import; authoritative complete-family rooting is independently clipped from the actual strict final9027aa mesh. Fresh complete source assembly, topology and whole-height audit remain mandatory.',persistent_fixed_contact=dict(holder_support_indices=[owner],nominal_retained_stock_used=True,contact_volume_mm3=float(contact.volume()),first_prefix_height_mm=floor+.00005,first_contact_bottom_mm=float(contact.bounding_box()[2]),patch_bottom_mm=floor,patch_is_convex=True,all_positive_height_prefixes_connected=True,passed=True),existing_temporary_and_cut_geometry_unchanged=True)
  rows.append(row);dump(OUT/'candidate_checkpoint.json',dict(repairs=rows));print('Pad',index,'owner',owner,'last/root',low,high,'radius',radius,'complete padding',margin,'strict pass','net',row['actual_added_volume_over_occupied_context_mm3'],flush=True)
 aggregate=mf.Manifold.batch_boolean(patches,mf.OpType.Add);path=OUT/'residual_rounded_caps.npz';D.save(path,aggregate)
 for rel,digest in inputs.items():
  if D.sha(ROOT/rel)!=digest:raise ValueError('Input changed during candidate construction')
 # Historical strict source must survive the next actual assembly overwrite.
 shutil.copyfile(mp,OUT/'source_strict_final9027aa.npz');shutil.copyfile(ap,OUT/'source_full_height_nine_births.json')
 report=dict(local_geometry_passed=True,complete_exclusion_checks_required=True,global_validation_required=True,source_models_modified=False,repair_count=len(rows),replacements_only_no_extra_part_count=True,units='mm',original_model_and_all_temporary_geometry_unchanged=True,aggregate_mesh_file=str(path.relative_to(ROOT)),aggregate_sha256=D.sha(path),aggregate_volume_mm3=float(aggregate.volume()),repairs=rows,input_sha256=inputs,source_sha256=D.sha(__file__),elapsed_seconds=time.monotonic()-start)
 dump(ROOT/'reports/residual_rounded_caps_candidates_v6.json',report);print('DONE',len(rows),report['elapsed_seconds'],flush=True)
if __name__=='__main__':main()

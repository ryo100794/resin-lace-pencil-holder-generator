#!/usr/bin/env python3
"""Three explicit positive side expansions of diagnosed fixed cap pads."""
from pathlib import Path
import json,time
import numpy as np
import manifold3d as mf
import diagnose_fixed_precision_v6 as D
from build_fixed_precision_repairs_v6 import canonical_probe
ROOT=D.ROOT;OUT=ROOT/'cache/rim_cap_bedding_v6/corner_expansions'

def dump(p,d):
 t=p.with_suffix(p.suffix+'.part');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def main():
 OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 candidate_path=ROOT/'reports/rim_cap_bedding_candidates_v6.json';holder_path=ROOT/'cache/custom_holder_supports/holder_supports.json';diag_path=ROOT/'reports/rim_cap_bedding_prepared_degeneracies_two_pass_v6.json';raw_path=ROOT/'cache/assembled_v6_float64.npz'
 inputs={str(p.relative_to(ROOT)):D.sha(p)for p in[candidate_path,holder_path,diag_path,raw_path,Path(__file__)]}
 candidate=json.loads(candidate_path.read_text());holder=json.loads(holder_path.read_text());parts={r['index']:r for r in holder['parts']};diag=json.loads(diag_path.read_text())
 if len(diag['degenerate_faces'])!=3:raise ValueError('Expected exactly three diagnosed corner facets')
 print('Loading current complete raw context once for bounded local diagnostics',flush=True)
 raw=D.load(raw_path);band=raw.trim_by_plane((0,0,1),99.35).trim_by_plane((0,0,-1),-100.15);print('Band',band.num_tri(),flush=True);del raw
 repairs=[];solids=[]
 for index,owner in[(1288,431),(1294,440),(1307,458)]:
  oldrow=next(r for r in candidate['repairs']if r['affected_support_index']==owner);oldpath=ROOT/'cache/custom_holder_supports'/parts[index]['full_mesh_file'];old=D.load(oldpath);inputs[str(oldpath.relative_to(ROOT))]=D.sha(oldpath)
  if D.sha(oldpath)!=oldrow['mesh_sha256']:raise ValueError('Adopted fixed pad differs from the original candidate')
  source=parts[owner];fullpath=holder_path.parent/source['full_mesh_file'];rempath=holder_path.parent/source['removable_mesh_file'];full=D.load(fullpath);rem=D.load(rempath)
  for p in[fullpath,rempath]:inputs[str(p.relative_to(ROOT))]=D.sha(p)
  oldbb=np.asarray(old.bounding_box());matched=[f for f in diag['degenerate_faces']if np.all(np.asarray(f['center_mm'])>=oldbb[:3]-1e-8)and np.all(np.asarray(f['center_mm'])<=oldbb[3:]+1e-8)]
  if len(matched)!=1:raise ValueError('Expected one exact collinear facet per replacement')
  trials=[]
  for padding in[.02,.04]:
   lo=oldbb[:3]-[padding,padding,0];hi=oldbb[3:]+[padding,padding,0];patch=D.box(lo,hi)
   if not(old-patch).is_empty():raise ValueError('Positive expansion lost old fixed material')
   vertices=np.asarray(matched[0]['original_vertices_mm'])
   if np.any(vertices<lo+1e-8)or np.any(vertices>hi-1e-8):raise ValueError('Old full degenerate facet not strictly inside expansion')
   context=band^D.box(lo-[.22,.22,.12],hi+[.22,.22,.12]);before=canonical_probe(context.as_original());after=canonical_probe((context+patch).as_original())
   trials.append(dict(extra_xy_padding_mm=padding,before=before,after=after))
   if after['passed']:break
  else:raise ValueError('Both bounded local expansion attempts failed: '+str(trials))
  floor=float(lo[2]);probe=floor+.00005;prefix=patch.trim_by_plane((0,0,-1),-probe);contact=prefix^(full-rem)
  if contact.is_empty()or contact.volume()<=1e-12 or abs(contact.bounding_box()[2]-floor)>1e-8:raise ValueError('Expanded pad lacks unchanged fixed first-face seat')
  if not(patch^rem).is_empty():raise ValueError('Expanded pad enters removable stock')
  offset=float(((np.asarray(patch.to_mesh64().vert_properties)[:,:3]-np.asarray(source['target_mm']))@np.asarray(source['neck_axes'][1])).max())
  if offset>=.075:raise ValueError('Expanded pad leaves permanent contact head')
  ident=f'cap_corner_H{index:04d}';path=OUT/(ident+'.npz');D.save(path,patch);cpath=OUT/(ident+'_raw_context.npz');D.save(cpath,context)
  record=dict(oldrow);record.update(id=ident,replacement_holder_index=index,repair_purpose='positive_fixed_pad_corner_zone_expansion',mesh_file=str(path.relative_to(ROOT)),mesh_sha256=D.sha(path),old_piece_file=str(oldpath.relative_to(ROOT)),old_piece_sha256=D.sha(oldpath),complete_old_piece_contained=True,old_piece_outside_replacement_empty=True,extra_xy_padding_mm=padding,unchanged_z_extent=True,bounds_mm=list(patch.bounding_box()),target_mm=((lo+hi)/2).tolist(),root_mm=[float((lo[0]+hi[0])/2),float((lo[1]+hi[1])/2),floor],volume_mm3=float(patch.volume()),actual_added_volume_over_occupied_context_mm3=float((patch-context).volume()),maximum_box_dimension_mm=float((hi-lo).max()),local_before_strict_probe=before,local_after_strict_probe=after,corner_expansion_trials=trials,failed_full_prepared_facet=matched[0],failed_facet_strictly_inside_added_box=True,source_support_removable_gap_mm=float(patch.min_gap(rem,1.)),maximum_outward_axis_offset_from_contact_mm=offset,first_prefix_contact_volume_mm3=float(contact.volume()),first_prefix_contact_fraction=float(contact.volume()/prefix.volume()),final_context_file=str(cpath.relative_to(ROOT)),final_context_sha256=D.sha(cpath),local_context_scope='Diagnostic crop imported from complete saved raw Mesh64. Exact fresh whole-model assembly and unchanged strict validation remain mandatory; no claim that raw import preserves every source triangle.',persistent_fixed_contact=dict(holder_support_indices=[owner],nominal_retained_stock_used=True,contact_volume_mm3=float(contact.volume()),first_prefix_height_mm=probe,first_contact_bottom_mm=float(contact.bounding_box()[2]),patch_bottom_mm=floor,patch_is_convex=True,all_positive_height_prefixes_connected=True,passed=True),local_geometry_passed=True)
  repairs.append(record);solids.append(patch);print(ident,'padding',padding,'strict',after,'gap',record['source_support_removable_gap_mm'],flush=True)
 aggregate=mf.Manifold.batch_boolean(solids,mf.OpType.Add);apath=OUT/'corner_expansions.npz';D.save(apath,aggregate)
 for rel,digest in inputs.items():
  if D.sha(ROOT/rel)!=digest:raise ValueError('Input changed during build')
 report=dict(local_geometry_passed=True,complete_exclusion_checks_required=True,global_validation_required=True,source_models_modified=False,repair_count=3,replacements_only_no_extra_part_count=True,units='mm',original_model_and_all_temporary_geometry_unchanged=True,aggregate_mesh_file=str(apath.relative_to(ROOT)),aggregate_sha256=D.sha(apath),aggregate_volume_mm3=float(aggregate.volume()),repairs=repairs,input_sha256=inputs,source_sha256=D.sha(__file__),elapsed_seconds=time.monotonic()-start)
 dump(ROOT/'reports/rim_cap_corner_expansions_candidates_v6.json',report);print('DONE',report['elapsed_seconds'],flush=True)
if __name__=='__main__':main()

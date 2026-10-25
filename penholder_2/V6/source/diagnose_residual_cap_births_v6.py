#!/usr/bin/env python3
"""Actual closed-solid prefix validation for every residual final cap event."""
from pathlib import Path
import json,time
import numpy as np
import manifold3d as mf
import diagnose_fixed_precision_v6 as D
ROOT=D.ROOT;OUT=ROOT/'cache/residual_cap_births_v6';REPORT=ROOT/'reports/residual_cap_prefix_validation_v6.json'

def dump(p,d):
 t=p.with_suffix(p.suffix+'.part');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def main():
 OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic();mp=ROOT/'cache/print_ready_v6.npz';ap=ROOT/'reports/custom_supports_validation.json';holderpath=ROOT/'cache/custom_holder_supports/holder_supports.json'
 audit=json.loads(ap.read_text());events=[r for r in audit['rows']if r['material_birth']and not r['plate_contact']]
 if len(events)!=9 or D.sha(mp)!=audit['input_sha256']:raise ValueError('Wrong final diagnostic input')
 inputs={str(p.relative_to(ROOT)):D.sha(p)for p in[mp,ap,holderpath,Path(__file__)]}
 holder=json.loads(holderpath.read_text());pads=[r for r in holder['parts']if r['index']>=1285]
 for k,e in enumerate(events):
  p=np.asarray(e['position_mm']);near=[r for r in pads if np.all(p>=np.asarray(r['bounds_mm'])[:3]-1e-8)and np.all(p<=np.asarray(r['bounds_mm'])[3:]+1e-8)]
  if len(near)!=1:raise ValueError('Ambiguous existing pad')
  e['event_index']=k;e['pad_index']=near[0]['index'];e['source_support_index']=near[0]['precision_fixed_fill']['source_evidence']['affected_support_index']
 lower=99.2;upper=100.2;model=D.load(mp);band=model.trim_by_plane((0,0,1),lower).trim_by_plane((0,0,-1),-upper);print('Full XY band triangles',band.num_tri(),flush=True);del model
 # No source material birth exists below 99.6208 in the complete final audit.
 # Every positive prefix component touching the Z99.2 crop therefore reaches
 # material that is already plate-rooted; a component entirely above it floats.
 if any(r['birth_z_mm']<lower for r in events):raise ValueError('Unrooted source material below band base')
 heights=sorted(set([e['probe_z_mm']for e in events]+[e['surface_join_z_mm']-1e-7 for e in events]+[99.63,99.64,99.66,99.69,99.72,99.75,99.8,99.9,100.]))
 records=[]
 for n,height in enumerate(heights):
  prefix=band.trim_by_plane((0,0,-1),-height);components=prefix.decompose();rows=[];event_locations={}
  for j,c in enumerate(components):
   volume=float(c.volume());bb=np.asarray(c.bounding_box());mm=c.to_mesh64();v=np.asarray(mm.vert_properties)[:,:3];touches=bb[2]<=lower+1e-7
   row=dict(component_index=j,signed_volume_mm3=volume,triangles=c.num_tri(),bounds_mm=bb.tolist(),touches_rooted_lower_plane=bool(touches),positive_material_component=volume>1e-12,floating_positive_component=volume>1e-12 and not touches)
   matched=[]
   for e in events:
    if e['birth_z_mm']>=height:continue
    p=np.asarray(e['position_mm']);dist=float(np.linalg.norm(v-p,axis=1).min(initial=float('inf')))
    if dist<1e-6:matched.append(e['event_index']);event_locations[e['event_index']]=dict(component_index=j,nearest_original_birth_vertex_distance_mm=dist,floating_positive_component=row['floating_positive_component'],touches_rooted_lower_plane=bool(touches),signed_volume_mm3=volume,bounds_mm=bb.tolist())
   row['contains_birth_vertices']=matched
   if row['floating_positive_component']:
    path=OUT/f'prefix_{n:02d}_floating_{j:02d}.npz';D.save(path,c);row['mesh_file']=str(path.relative_to(ROOT));row['mesh_sha256']=D.sha(path)
   rows.append(row)
  record=dict(height_mm=height,components=rows,event_locations=event_locations)
  records.append(record);print('Z',height,'positive floating',[(r['signed_volume_mm3'],r['contains_birth_vertices'],r['bounds_mm'])for r in rows if r['floating_positive_component']],'event locations',event_locations,flush=True)
  dump(REPORT,dict(diagnostic_only=True,complete=False,source_final_mesh_sha256=audit['input_sha256'],events=events,lower_rooted_plane_mm=lower,upper_crop_mm=upper,records=records,input_sha256=inputs))
 for p,d in inputs.items():
  if D.sha(ROOT/p)!=d:raise ValueError('Input changed')
 result=dict(diagnostic_only=True,complete=True,source_final_mesh_sha256=audit['input_sha256'],events=events,lower_rooted_plane_mm=lower,upper_crop_mm=upper,records=records,input_sha256=inputs,elapsed_seconds=time.monotonic()-started,
  classification='Actual Boolean intersection of the complete XY model band with z<=h, decomposed as closed solids; all positive components above the already-rooted Z99.2 plane are reported, including every merged floating family.')
 dump(REPORT,result);print('DONE',result['elapsed_seconds'],flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Read-only, all-gates diagnostic of the exact prepared float32 surface.
No topology gate is skipped or converted to a release pass.
"""
from pathlib import Path
import json,time,hashlib,gc
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parent.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
 q=p.with_suffix(p.suffix+'.part');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)

def main():
 start=time.monotonic();path=ROOT/'cache/assembled_v6_prepared_float64.npz';digest=sha(path)
 with np.load(path,allow_pickle=False)as data:v64=np.asarray(data['vertices'],float);f0=np.asarray(data['faces'],np.uint32)
 q=v64.astype(np.float32);rounding=float(np.linalg.norm(q.astype(float)-v64,axis=1).max());del v64
 v,inv=np.unique(q,axis=0,return_inverse=True);f=inv[f0].astype(np.uint32);del q,inv,f0
 keep=(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,2]!=f[:,0]);collapsed=int((~keep).sum());f=f[keep];del keep
 used,inv=np.unique(f,return_inverse=True);v=v[used];f=inv.reshape(-1,3).astype(np.int32);del used,inv
 nv=len(v);nf=len(f);nh=3*nf;zero=[];near=[]
 for off in range(0,nf,100000):
  tri=v[f[off:off+100000]].astype(float);areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
  for i in np.flatnonzero(areas<=1e-14):
   row=dict(face_index=int(off+i),area_mm2=float(areas[i]),vertices_mm=tri[i].tolist(),center_mm=tri[i].mean(0).tolist());near.append(row)
   if areas[i]==0:zero.append(row)
 origin=f.ravel();dest=f[:,[1,2,0]].ravel();keys=np.minimum(origin,dest).astype(np.int64)*nv+np.maximum(origin,dest);order=np.argsort(keys);sortedkeys=keys[order]
 starts=np.r_[0,np.flatnonzero(sortedkeys[1:]!=sortedkeys[:-1])+1];counts=np.diff(np.r_[starts,nh]);bad=np.flatnonzero(counts!=2)
 edges=[];bad_vertices=set()
 for i in bad:
  halves=order[starts[i]:starts[i]+counts[i]];a=int(origin[halves[0]]);b=int(dest[halves[0]]);bad_vertices.update([a,b]);edges.append(dict(edge_index=int(i),vertex_indices=[a,b],incidence=int(counts[i]),face_indices=(halves//3).astype(int).tolist(),endpoints_mm=v[[a,b]].astype(float).tolist(),center_mm=v[[a,b]].mean(0).astype(float).tolist(),length_mm=float(np.linalg.norm(v[a].astype(float)-v[b]))))
 pair_starts=starts[counts==2];ha=order[pair_starts];hb=order[pair_starts+1];consistent=(origin[ha]==dest[hb])&(dest[ha]==origin[hb]);winding=[]
 for a,b in zip(ha[~consistent],hb[~consistent]):
  vi=[int(origin[a]),int(dest[a])];bad_vertices.update(vi);winding.append(dict(vertex_indices=vi,endpoints_mm=v[vi].astype(float).tolist(),face_indices=[int(a//3),int(b//3)]))
 report=dict(diagnostic_only=True,passed=False,source_file=str(path.relative_to(ROOT)),source_sha256=digest,canonical_vertex_count=nv,canonical_triangle_count=nf,collapsed_repeated_index_faces_removed=collapsed,maximum_rounding_error_mm=rounding,zero_area_count=len(zero),near_zero_area_count=len(near),degenerate_faces=near,bad_edge_count=len(edges),bad_edges=edges,inconsistent_two_face_edge_count=len(winding),inconsistent_two_face_edges=winding,complete_point_and_component_checks_pending=True)
 target=ROOT/'reports/final_all_gates_diagnostic_v6.json';dump(target,report)
 print('Early counts:',{k:report[k]for k in['zero_area_count','near_zero_area_count','bad_edge_count','inconsistent_two_face_edge_count']},flush=True)
 for edge in edges:print('EDGE',edge,flush=True)
 del keys,sortedkeys,starts,counts,pair_starts,order,dest;gc.collect()
 # Complete fan graph on all vertices with valid two-face edges. Vertices
 # incident to invalid edges are identified separately rather than blessed.
 gooda=ha[consistent];goodb=hb[consistent];succ=np.arange(nh,dtype=np.int32)
 succ[gooda]=(goodb//3)*3+(goodb%3+1)%3;succ[goodb]=(gooda//3)*3+(gooda%3+1)%3
 graph=csr_matrix((np.ones(nh,dtype=np.int8),succ,np.arange(nh+1,dtype=np.int32)),shape=(nh,nh))
 fan_count,labels=connected_components(graph,directed=False);del graph,succ
 fan_vertex=np.empty(fan_count,dtype=np.int32);fan_vertex[labels]=origin;mult=np.bincount(fan_vertex,minlength=nv);fan_indices=np.flatnonzero(mult>1)
 point_rows=[dict(vertex_index=int(i),position_mm=v[i].astype(float).tolist(),surface_fan_count=int(mult[i]),incident_to_invalid_edge=int(i)in bad_vertices)for i in fan_indices]
 independent=[r for r in point_rows if not r['incident_to_invalid_edge']]
 report.update(independent_nonmanifold_point_count=len(independent),nonmanifold_points=independent,invalid_edge_vertex_fan_diagnostics=[r for r in point_rows if r['incident_to_invalid_edge']])
 print('Independent nonmanifold points:',len(independent),flush=True)
 del labels,fan_vertex,mult;gc.collect()
 # Regular-edge face components reveal point-connected shells and tiny
 # sliver surfaces even though a global invalid edge blocks Manifold import.
 aa=(ha//3).astype(np.int32);bb=(hb//3).astype(np.int32)
 graph=csr_matrix((np.ones(len(aa),dtype=np.int8),(aa,bb)),shape=(nf,nf))
 nc,face_labels=connected_components(graph,directed=False);del graph,aa,bb,ha,hb,gooda,goodb,consistent;gc.collect()
 component_rows=[]
 for i in range(nc):
  selected=np.flatnonzero(face_labels==i);ff=f[selected];verts=np.unique(ff);points=v[verts].astype(float);lo=points.min(0);hi=points.max(0);center=(lo+hi)/2;volume=0.
  for off in range(0,len(ff),100000):
   tri=v[ff[off:off+100000]].astype(float)-center;volume+=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6.)
  boundary_edges=[]
  for e in edges:
   own=[j for j in e['face_indices']if face_labels[j]==i]
   if own and len(own)!=2:boundary_edges.append(dict(edge_index=e['edge_index'],own_incidence=len(own)))
  component_rows.append(dict(component_index=i,triangles=len(selected),vertices=len(verts),signed_centered_volume_mm3=volume,bounds_mm=np.r_[lo,hi].tolist(),center_mm=center.tolist(),regular_edge_component_closed=len(boundary_edges)==0,remaining_invalid_boundary_edges=boundary_edges))
 component_rows.sort(key=lambda r:-abs(r['signed_centered_volume_mm3']))
 report.update(regular_edge_face_component_count=nc,regular_edge_face_components=component_rows,complete_point_and_component_checks_pending=False,
  manifold_component_gate_not_claimed_due_to_invalid_edges=bool(edges or winding),scope='All canonical coordinate welding and collapsed-index-face removal exactly match the unchanged strict validator. Area, every edge, winding and every unaffected vertex link are examined without first-failure early exit. Face components are diagnostic surfaces split across invalid edges; their closed flags and signed volumes are reported, never filtered or used as a release pass.',elapsed_seconds=time.monotonic()-start)
 if sha(path)!=digest:raise ValueError('Prepared input changed during diagnostic')
 dump(target,report);print('Components',[(r['triangles'],r['signed_centered_volume_mm3'],r['regular_edge_component_closed'],r['center_mm'])for r in component_rows],flush=True);print('DONE',report['elapsed_seconds'],flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Extract scoped fixed-rim/negative-boundary diagnostics from a frozen trial."""
from pathlib import Path
import gc
import hashlib
import json
import time
import numpy as np
import manifold3d as mf
from scipy import sparse
from scipy.sparse.csgraph import connected_components

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'cache/fixed_precision_repairs'
RAW=ROOT/'cache/fresh_v6_float64_trial.npz'
CANON=ROOT/'cache/fresh_v6_simplify_1e-5mm_float32_trial.npz'
EDGE=ROOT/'reports/custom_sphere_supports_fresh_v6_canonical_1e-5mm_edge_attribution.json'
COMP=ROOT/'reports/custom_sphere_supports_fresh_v6_canonical_1e-5mm_components.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(path):
    with np.load(path,allow_pickle=False) as d:
        m=mf.Manifold(mf.Mesh64(np.asarray(d['vertices'],float),np.asarray(d['faces'],np.uint32)))
    if m.status()!=mf.Error.NoError:raise ValueError('Invalid solid '+str(path))
    return m

def save(path,m):
    mesh=m.to_mesh64();part=Path(str(path)+'.part.npz')
    np.savez_compressed(part,vertices=np.asarray(mesh.vert_properties[:,:3]),faces=np.asarray(mesh.tri_verts))
    part.replace(path)

def volume(m):return 0. if m.is_empty() else max(0.,float(m.volume()))

def box(lo,hi):return mf.Manifold.cube(tuple(np.asarray(hi)-lo)).translate(lo)

def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [RAW,CANON,EDGE,COMP,ROOT/'cache/original_v5.npz']}
    edge=json.loads(EDGE.read_text());component=json.loads(COMP.read_text())
    if inputs[str(CANON.relative_to(ROOT))]!=edge['source_mesh_sha256']:raise ValueError('Trial hash changed')
    cases=[]
    for e in edge['rows']:
        if not e['original_surface_matches']:continue
        pts=np.asarray(e['endpoints_mm']);lo=pts.min(0)-.08;hi=pts.max(0)+.08
        cases.append(dict(id=f'rim_{e["index"]:02d}',kind='original_rim_edge',edge=e,
                          roi_bounds_mm=np.r_[lo,hi].tolist()))
    with np.load(CANON,allow_pickle=False) as d:
        v=np.asarray(d['vertices'],float);f=np.asarray(d['faces'],np.int32)
    rr=np.r_[f[:,0],f[:,0]];cc=np.r_[f[:,1],f[:,2]]
    graph=sparse.coo_matrix((np.ones(len(rr),np.uint8),(rr,cc)),shape=(len(v),len(v))).tocsr()
    count,labels=connected_components(graph,directed=False)
    del rr,cc,graph
    face_labels=labels[f[:,0]]
    for c in component['all_components']:
        if c['signed_volume_mm3']>=0:continue
        inds=np.flatnonzero(labels==c['index']);faces=f[face_labels==c['index']]
        if len(inds)!=c['vertices'] or len(faces)!=c['triangles']:raise ValueError('Component labeling mismatch')
        mapping=np.full(len(v),-1,np.int32);mapping[inds]=np.arange(len(inds));faces=mapping[faces[:,::-1]]
        solid=mf.Manifold(mf.Mesh64(v[inds],faces.astype(np.uint32)))
        if solid.status()!=mf.Error.NoError or solid.volume()<=0:raise ValueError('Reversed cavity is not a positive solid')
        path=OUT/f'cavity_{c["index"]}_positive_void.npz';save(path,solid)
        bb=np.asarray(solid.bounding_box());lo=bb[:3]-.02;hi=bb[3:]+.02
        cases.append(dict(id=f'cavity_{c["index"]}',kind='negative_boundary',component=c,
            positive_void_file=str(path.relative_to(ROOT)),positive_void_sha256=sha(path),
            positive_void_volume_mm3=volume(solid),roi_bounds_mm=np.r_[lo,hi].tolist()))
    del v,f,labels,face_labels,mapping,faces,inds;gc.collect()
    for name,path in [('original',ROOT/'cache/original_v5.npz'),('fresh_raw',RAW)]:
        model=load(path);print('Loaded',name,model.num_tri(),flush=True)
        for case in cases:
            bb=np.asarray(case['roi_bounds_mm']);local=model^box(bb[:3],bb[3:])
            target=OUT/f'{case["id"]}_{name}_context.npz';save(target,local)
            row=dict(context_file=str(target.relative_to(ROOT)),context_sha256=sha(target),
                     local_volume_mm3=volume(local),triangles=local.num_tri())
            if case['kind']=='negative_boundary':
                void=load(ROOT/case['positive_void_file']);inside=void^local;outside=void-local
                row.update(void_inside_volume_mm3=volume(inside),void_outside_volume_mm3=volume(outside),
                           void_inside_fraction=volume(inside)/volume(void),void_outside_empty=outside.is_empty(),
                           void_inside_empty=inside.is_empty(),void_boundary_min_gap_mm=float(void.min_gap(local,.01)))
            case[name]=row
            print(case['id'],name,row,flush=True)
        del model,local;gc.collect()
    if any(sha(ROOT/p)!=digest for p,digest in inputs.items()):raise ValueError('Frozen input changed')
    report=dict(diagnostic_only=True,original_source_modified=False,source_sha256=sha(__file__),
                inputs=[dict(path=p,sha256=d)for p,d in inputs.items()],cases=cases,elapsed_seconds=time.monotonic()-started)
    (ROOT/'reports/fixed_precision_scope_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Complete scoped contexts',len(cases),flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent +Z surface-sublevel support candidate analysis.

This is a geometric support audit, not a resin-strength certificate. A
closed mesh boundary merge tree can include cavity surfaces. Signed cap
volume distinguishes positive material from negative cavity births before
their first boundary merge. Plate-contact roots at Z=0 are independent valid
roots; they need not be a single component before the base joins. The final
gate additionally checks that the complete input is one closed valid solid.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CPP = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <array>
#include <cmath>
struct Event {uint64_t child,elder;double birth,death,height,volume;uint64_t triangles;};
using P=std::array<double,3>;
P mix(const P&a,const P&b,double t){return {a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]),a[2]+t*(b[2]-a[2])};}
double term(const P&a,const P&b,const P&c,double h){
  double nz=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  return nz*.5*((a[2]+b[2]+c[2])/3.-h);
}
double clipped(const P&a,const P&b,const P&c,double h){
  P p[3]={a,b,c};int below=0;for(int i=0;i<3;++i)below+=p[i][2]<=h;
  if(below==0)return 0.;if(below==3)return term(a,b,c,h);
  int unique=0;
  for(int i=0;i<3;++i)if((p[i][2]<=h)==(below==1))unique=i;
  const P &x=p[unique],&y=p[(unique+1)%3],&z=p[(unique+2)%3];
  P q=mix(x,y,(h-x[2])/(y[2]-x[2])),r=mix(x,z,(h-x[2])/(z[2]-x[2]));
  double small=term(x,q,r,h);
  return below==1?small:term(a,b,c,h)-small;
}
int main(int argc, char** argv) {
  if(argc != 3) return 1;
  FILE* in=fopen(argv[1],"rb"); FILE* out=fopen(argv[2],"wb");
  if(!in || !out) return 2;
  uint64_t n,m; if(fread(&n,8,1,in)!=1 || fread(&m,8,1,in)!=1)return 3;
  std::vector<float> v(n*3); if(fread(v.data(),4,n*3,in)!=n*3)return 4;
  std::vector<uint32_t> faces(m*3); if(fread(faces.data(),4,m*3,in)!=m*3)return 5; fclose(in);
  std::vector<uint32_t> degree(n,0),order(n),rank(n),parent(n);
  for(uint64_t i=0;i<m*3;++i) degree[faces[i]]+=2;
  std::vector<uint64_t> offsets(n+1,0),cursor(n);
  for(uint64_t i=0;i<n;++i) offsets[i+1]=offsets[i]+degree[i];
  std::copy(offsets.begin(), offsets.begin()+n,cursor.begin());
  std::vector<uint32_t> adj(m*6);
  std::vector<uint64_t> fo(n+1,0),fc(n);
  for(uint64_t i=0;i<n;++i)fo[i+1]=fo[i]+degree[i]/2;
  std::copy(fo.begin(),fo.begin()+n,fc.begin());
  std::vector<uint32_t> fa(m*3);
  for(uint64_t i=0;i<m;++i) {
    uint32_t a=faces[3*i],b=faces[3*i+1],c=faces[3*i+2];
    adj[cursor[a]++]=b;adj[cursor[a]++]=c;
    adj[cursor[b]++]=a;adj[cursor[b]++]=c;
    adj[cursor[c]++]=a;adj[cursor[c]++]=b;
    fa[fc[a]++]=i;fa[fc[b]++]=i;fa[fc[c]++]=i;
  }
  std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){return v[3*a+2]<v[3*b+2] || (v[3*a+2]==v[3*b+2] && a<b);});
  for(uint32_t i=0;i<n;++i){rank[order[i]]=i;parent[i]=i;}
  auto find=[&](uint32_t a){uint32_t r=a;while(parent[r]!=r)r=parent[r];while(parent[a]!=a){uint32_t b=parent[a];parent[a]=r;a=b;}return r;};
  uint64_t count=0;
  std::vector<Event> events;
  for(uint32_t it=0;it<n;++it) {
    uint32_t a=order[it];
    for(uint64_t j=offsets[a];j<offsets[a+1];++j) {
      uint32_t b=adj[j]; if(rank[b]>=it)continue;
      uint32_t x=find(a), y=find(b);if(x==y)continue;
      if(rank[x]<rank[y])std::swap(x,y);
      double birth=v[3*x+2],death=v[3*a+2];
      if(death>birth+1e-8) {
        events.push_back({x,y,birth,death,0.,0.,0});++count;
      }
      parent[x]=y;
    }
  }
  std::vector<uint32_t> vertex_seen(n,0),face_seen(m,0),queue;
  uint32_t serial=0;
  for(auto &e:events) {
    e.height=e.birth+std::min(.025,(e.death-e.birth)*.25);
    if(e.birth>1e-6) {
      ++serial;queue.clear();queue.push_back(e.child);vertex_seen[e.child]=serial;
      double sum=0.,compensation=0.;
      for(uint64_t qi=0;qi<queue.size();++qi) {
        uint32_t x=queue[qi];
        for(uint64_t j=fo[x];j<fo[x+1];++j) {
          uint32_t f=fa[j];if(face_seen[f]==serial)continue;face_seen[f]=serial;++e.triangles;
          P points[3];
          for(int k=0;k<3;++k) {
            uint32_t w=faces[3*f+k];points[k]={v[3*w],v[3*w+1],v[3*w+2]};
            if(v[3*w+2]<=e.height && vertex_seen[w]!=serial){vertex_seen[w]=serial;queue.push_back(w);}
          }
          double value=clipped(points[0],points[1],points[2],e.height);
          double corrected=value-compensation,next=sum+corrected;
          compensation=(next-sum)-corrected;sum=next;
        }
      }
      e.volume=sum;
    }
    fwrite(&e.child,8,1,out);fwrite(&e.elder,8,1,out);
    fwrite(&e.birth,8,1,out);fwrite(&e.death,8,1,out);
    fwrite(&e.height,8,1,out);fwrite(&e.volume,8,1,out);fwrite(&e.triangles,8,1,out);
  }
  fclose(out);fprintf(stderr,"merge records: %llu\n",(unsigned long long)count);
}
'''


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def attribute_original_surfaces(rows, design):
    """Attribute baseline minima to original fixed spheres or holder.

    This is only for planning the original V5 additions. A new support's
    boundary naturally has no V5 source match, so this is not a release gate.
    """
    import trimesh
    positive=[i for i,row in enumerate(rows) if row['material_birth'] and not row['plate_contact']]
    meshes={}
    for layers in (3,4,5):
        meta=json.loads((ROOT/f'cache/spheres/sphere_{layers}layers.json').read_text())
        meshes[layers]={}
        for kind,filename in [('fixed_outer',meta['free_layer_meshes'][0]),('supported_module',meta['supported_mesh_file'])]:
            with np.load(ROOT/'cache/spheres'/filename,allow_pickle=False) as data:
                meshes[layers][kind]=trimesh.Trimesh(data['vertices'],data['faces'],process=False)
    for i in positive:rows[i]['source_surface_matches']=[]
    for index,sphere in enumerate(design['spheres']):
        angle=np.deg2rad(sphere['theta_deg'])
        rotation=np.column_stack(([-np.sin(angle),np.cos(angle),0],[0,0,1],[np.cos(angle),np.sin(angle),0]))
        center=np.array([sphere['center_radius']*np.cos(angle),sphere['center_radius']*np.sin(angle),sphere['z']])
        candidates=[i for i in positive if np.linalg.norm(np.asarray(rows[i]['position_mm'])-center)<sphere['radius']+1.1]
        if not candidates:continue
        points=(np.array([rows[i]['position_mm'] for i in candidates])-center)@rotation
        for kind,mesh in meshes[sphere['layers']].items():
            _,distances,faces=trimesh.proximity.closest_point(mesh,points)
            for row_index,distance,face in zip(candidates,distances,faces):
                if distance<.0001:
                    rows[row_index]['source_surface_matches'].append(dict(sphere_index=index,
                        sphere_kind=sphere['kind'],surface=kind,distance_mm=float(distance),triangle_index=int(face)))
    counts={}
    for i in positive:
        kinds={match['surface'] for match in rows[i]['source_surface_matches']}
        kind='fixed_sphere_surface' if 'fixed_outer' in kinds else ('sphere_non_outer_surface' if kinds else 'fixed_holder_surface')
        rows[i]['source_attribution']=kind
        counts[kind]=counts.get(kind,0)+1
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'cache/original_v5.npz')
    parser.add_argument('--out',type=Path,default=ROOT/'reports/support_analysis_candidates.json')
    parser.add_argument('--min-persistence',type=float,default=0.)
    parser.add_argument('--attribute-original',action='store_true',
        help='Attribute original V5 births to actual fixed sphere/holder surfaces.')
    parser.add_argument('--final-gate','--require-rooted',dest='final_gate',action='store_true',
        help='Require one complete solid, Z=0 plate roots, and no positive non-plate material births.')
    args=parser.parse_args();started=time.monotonic();input_hash=sha(args.input)
    with np.load(args.input,allow_pickle=False) as d:
        vertices=np.asarray(d['vertices'],dtype='<f4');faces=np.asarray(d['faces'],dtype='<u4')
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Expected finite, nonempty vertices with shape (N,3)')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or int(faces.max())>=len(vertices):
        raise ValueError('Expected valid, nonempty face indices with shape (M,3)')
    if sha(args.input)!=input_hash:raise RuntimeError('Input changed while being read')
    with tempfile.TemporaryDirectory(prefix='support_merge_') as tmp:
        tmp=Path(tmp);(tmp/'merge.cpp').write_text(CPP)
        subprocess.run(['g++','-O3','-std=c++17',str(tmp/'merge.cpp'),'-o',str(tmp/'merge')],check=True)
        with (tmp/'input.bin').open('wb') as f:
            np.array([len(vertices),len(faces)],dtype='<u8').tofile(f)
            vertices.tofile(f);faces.tofile(f)
        subprocess.run([str(tmp/'merge'),str(tmp/'input.bin'),str(tmp/'events.bin')],check=True)
        events=np.fromfile(tmp/'events.bin',dtype=[('child','<u8'),('elder','<u8'),('birth','<f8'),('death','<f8'),('height','<f8'),('volume','<f8'),('triangles','<u8')])
    normals=np.zeros((len(vertices),3),dtype=np.float64)
    for start in range(0,len(faces),65536):
        ff=faces[start:start+65536];tri=vertices[ff].astype(float)
        nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        for k in range(3):np.add.at(normals,ff[:,k],nn)
    lengths=np.linalg.norm(normals,axis=1);normals/=np.maximum(lengths[:,None],1e-30)
    design=json.loads((ROOT/'reports/design.json').read_text())
    sphere_rows=[]
    for s in design['spheres']:
        a=np.deg2rad(s['theta_deg']);center=np.array([s['center_radius']*np.cos(a),s['center_radius']*np.sin(a),s['z']])
        sphere_rows.append((s,center))
    rows=[]
    for e in events:
        vertex=int(e['child']);p=vertices[vertex].astype(float)
        near=[]
        for i,(s,c) in enumerate(sphere_rows):
            distance=float(np.linalg.norm(p-c))
            if distance<=s['radius']+1.0:near.append(dict(index=i,kind=s['kind'],distance_from_center_mm=distance))
        duration=float(e['death']-e['birth'])
        if duration<args.min_persistence:continue
        rows.append(dict(vertex_index=vertex,position_mm=p.tolist(),
            outward_area_weighted_normal=normals[vertex].tolist(),
            downward_birth=bool(normals[vertex,2]<-.01),
            probe_z_mm=float(e['height']),signed_capped_surface_volume_mm3=float(e['volume']),
            capped_surface_triangle_count=int(e['triangles']),
            material_birth=bool(e['volume']>1e-12),cavity_birth=bool(e['volume']< -1e-12),
            birth_z_mm=float(e['birth']),surface_join_z_mm=float(e['death']),
            surface_lifetime_mm=duration,plate_contact=bool(e['birth']<=1e-6),
            elder_vertex=int(e['elder']),near_sphere=near))
    fixed=[r for r in rows if r['material_birth'] and not r['plate_contact'] and not r['near_sphere']]
    positive=[r for r in rows if r['material_birth'] and not r['plate_contact']]
    cavities=[r for r in rows if r['cavity_birth'] and not r['plate_contact']]
    unresolved=[r for r in rows if not r['material_birth'] and not r['cavity_birth'] and not r['plate_contact']]
    report=dict(input_file=str(args.input),input_sha256=input_hash,auditor_sha256=sha(__file__),
        analysis='Boundary surface merge tree, exact signed cap volume distinguishes material from cavity births; representative full-solid confirmation recommended',
        units='mm',build_direction=[0,0,1],source_vertices=len(vertices),source_triangles=len(faces),
        candidate_count=len(rows),outside_heuristic_sphere_envelope_positive_birth_count=len(fixed),
        positive_non_plate_material_birth_count=len(positive),
        negative_cavity_birth_count=len(cavities),
        numerical_undetermined_non_plate_birth_count=len(unresolved),
        numerical_undetermined_maximum_lifetime_mm=max((r['surface_lifetime_mm'] for r in unresolved),default=0.),
        numerical_undetermined_maximum_absolute_cap_volume_mm3=max((abs(r['signed_capped_surface_volume_mm3']) for r in unresolved),default=0.),
        plate_contact_birth_count=sum(r['plate_contact'] for r in rows),
        positive_capped_volume_threshold_mm3=1e-12,
        plate_root_height_tolerance_mm=1e-6,
        birth_probe_height_rule='birth + min(0.025 mm, 0.25 * (first boundary merge height - birth))',
        boundary_merge_event_height_resolution_mm=1e-8,
        finite_precision='Input vertices converted to float32; final print export uses this precision.',
        outside_heuristic_sphere_envelope_lifetime_histogram={str(cut):sum(r['surface_lifetime_mm']>=cut for r in fixed) for cut in [.0001,.001,.005,.01,.025,.05,.1,.25,.5,1,2,5]},
        near_sphere_exclusion='Heuristic radius +1 mm, not authoritative source attribution',
        rows=rows,
        limitations=['No resin, peel force, exposure, pixel raster, finite contact strength or overhang deflection simulation.',
            'The small 1e-12 mm3 positive cap-volume threshold excludes numerical material below that sampled volume.',
            'Existing surface graph roots lying on Z=0 count as independently plate attached; platform adhesion is not measured.',
            'The classification is geometric; verify the slicer layer preview for the actual printer and layer phase.'],
        elapsed_seconds=time.monotonic()-started)
    if args.attribute_original:
        report['positive_birth_source_attribution_counts']=attribute_original_surfaces(rows,design)
        report['fixed_holder_positive_material_birth_count']=report['positive_birth_source_attribution_counts'].get('fixed_holder_surface',0)
        report['fixed_sphere_positive_material_birth_count']=report['positive_birth_source_attribution_counts'].get('fixed_sphere_surface',0)
        report['source_attribution_method']='Nearest actual original cached fixed-outer and supported-module surface, proper inverse sphere transform, distance <0.0001 mm. Coincident holder/sphere boundary is assigned sphere.'
    if args.final_gate:
        import manifold3d as mf
        solid=mf.Manifold(mf.Mesh64(vertices.astype(float),faces))
        status=solid.status()==mf.Error.NoError
        parts=solid.decompose() if status else []
        positive_parts=[part for part in parts if part.volume()>1e-12]
        negative_parts=[part for part in parts if part.volume()< -1e-12]
        report['complete_mesh_geometry']=dict(manifold_status=str(solid.status()),
            positive_component_count=len(positive_parts),negative_component_count=len(negative_parts),
            bounds_mm=list(solid.bounding_box()) if status else None,volume_mm3=float(solid.volume()) if status else None)
        report['passed']=bool(status and len(positive_parts)==1 and not negative_parts and
            abs(float(vertices[:,2].min()))<=1e-6 and not positive and args.min_persistence==0.)
        report['passed_scope']='Complete geometry valid and all positive non-plate material boundary births eliminated at declared numerical thresholds; no physical-print certification.'
    report['elapsed_seconds']=time.monotonic()-started
    if sha(args.input)!=input_hash:raise RuntimeError('Input changed during audit')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    if args.final_gate and not report['passed']:
        sys.exit(2)

if __name__=='__main__':main()

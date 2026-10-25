#!/usr/bin/env python3
"""Removable, geometry-selected supports for unchanged fixed V5 outer spheres.
Local +Y builds upward. A separate holder/coupon foundation must seat the foot.
"""
from __future__ import annotations
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import manifold3d as mf
ROOT=Path(__file__).resolve().parent.parent
SOURCE_CODE_BYTES=Path(__file__).read_bytes()
SOURCE_LOADED_SHA256=hashlib.sha256(SOURCE_CODE_BYTES).hexdigest()
EPS=1e-12
SEED_EPS=1e-14
UP=np.array([0.,1.,0.])
def vol(m):return 0. if m.is_empty() else max(0.,float(m.volume()))
def join(ms):
 ms=[m for m in ms if not m.is_empty()]
 return mf.Manifold.batch_boolean(ms,mf.OpType.Add) if ms else mf.Manifold()
def load(p):
 d=np.load(p,allow_pickle=False);return mf.Manifold(mf.Mesh64(np.asarray(d['vertices'],float),np.asarray(d['faces'],np.uint32)))
def verts(m):return np.asarray(m.to_mesh64().vert_properties[:,:3],float)
def save(p,m):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);a=m.to_mesh64();tmp=p.with_name(p.name+'.part.npz');np.savez_compressed(tmp,vertices=a.vert_properties[:,:3],faces=a.tri_verts);tmp.replace(p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def overlap(a,b):
 if a.is_empty() or b.is_empty():return 0.
 aa=np.array(a.bounding_box());bb=np.array(b.bounding_box())
 if np.any(aa[3:]<bb[:3]) or np.any(bb[3:]<aa[:3]):return 0.
 return vol(a^b)
def prefix(m,y):return m.trim_by_plane((0.,-1.,0.),-float(y))
def minima(m):
 a=m.to_mesh64();v=np.asarray(a.vert_properties[:,:3]);f=np.asarray(a.tri_verts);h=v[:,1];other=np.full(len(v),np.inf)
 for x,y in [(0,1),(1,2),(2,0)]:
  np.minimum.at(other,f[:,x],h[f[:,y]]);np.minimum.at(other,f[:,y],h[f[:,x]])
 return np.unique(np.round(h[h<=other+1e-10],10))
def schedule(m,grid=.05):
 lo,hi=m.bounding_box()[1],m.bounding_box()[4]
 hs={round(float(h),10) for h in np.arange(math.floor(lo/grid)*grid,hi,grid) if lo<h<hi}
 for h in minima(m):
  for d in [.0001,.001,.005,.025]:
   if lo<h+d<hi:hs.add(round(float(h+d),10))
 return sorted(hs)
def cylinder(a,b,r1,r2=None,n=24):
 a=np.array(a,float);b=np.array(b,float);d=b-a;l=np.linalg.norm(d);d/=l
 seed=np.array([1.,0.,0.]) if abs(d[0])<.8 else np.array([0.,0.,1.]);u=seed-d*(seed@d);u/=np.linalg.norm(u);v=np.cross(d,u)
 return mf.Manifold.cylinder(float(l),r1,r1 if r2 is None else r2,n).transform(np.column_stack((u,v,d,a)))
def ball(p,r):return mf.Manifold.sphere(r,24).translate(p)
def capsule(a,b,r):return mf.Manifold.hull_points(np.vstack([verts(ball(a,r)),verts(ball(b,r))]))
def ring(p,d,r,n=24):
 p=np.asarray(p,float);d=np.asarray(d,float);d/=np.linalg.norm(d)
 seed=np.array([1.,0.,0.]) if abs(d[0])<.8 else np.array([0.,0.,1.]);u=seed-d*(seed@d);u/=np.linalg.norm(u);v=np.cross(d,u)
 t=np.arange(n)*2*np.pi/n
 return p+r*(np.cos(t)[:,None]*u+np.sin(t)[:,None]*v)
def profile(p,d,ss):
 p=np.asarray(p,float);d=np.asarray(d,float);d/=np.linalg.norm(d);n=24
 v=np.vstack([ring(p+d*x,d,r,n) for x,r in ss]);f=[]
 for j in range(len(ss)-1):
  for k in range(n):
   kk=(k+1)%n;a=j*n+k;b=j*n+kk;c=(j+1)*n+kk;e=(j+1)*n+k;f.extend([[a,b,c],[a,c,e]])
 for k in range(1,n-1):f.append([0,k+1,k]);f.append([(len(ss)-1)*n,(len(ss)-1)*n+k,(len(ss)-1)*n+k+1])
 return mf.Manifold(mf.Mesh64(v,np.asarray(f,np.uint32)))
def endpoint(p,d,carrier=1.05):
 p=np.asarray(p,float);d=np.asarray(d,float);d/=np.linalg.norm(d)
 neck_start=.12 if abs(d[1])>.8 else .42
 ss=[(-.12,.225),(neck_start-.04,.225),(neck_start,.20),(neck_start+.14,.20),(neck_start+.20,.225),(neck_start+.70,.225),(max(carrier,neck_start+1.05),.35)]
 carrier=max(carrier,neck_start+1.05)
 m=profile(p,d,ss);st=profile(p,d,ss[:3]+[(neck_start+.07,.20)])
 return m,st,profile(p,d,[(neck_start+.00001,.20),(neck_start+.13999,.20)]),p+d*carrier,p+d*(neck_start+.07)

def endpoint_direction(p,up):
 p=np.asarray(p,float);xz=p.copy();xz[1]=0
 if np.linalg.norm(xz)<.1:return np.array([0.,float(up),0.])
 xz/=np.linalg.norm(xz);y=.55
 if float(p[1])*up<0 and abs(float(p[1]))/max(np.linalg.norm(p),1e-9)>.75:y=.35
 return math.sqrt(1-y*y)*xz+np.array([0.,up*y,0.])
def construct(q,p,foot=False,route=0):
 p=np.asarray(p,float);q=np.asarray(q,float)
 da=UP.copy() if foot else endpoint_direction(q,1);db=-UP if p[1]<-.65*np.linalg.norm(p) and abs(p[2])<.50 else endpoint_direction(p,-1)
 am,ast,an,a,ac=endpoint(q,da,.75 if foot else 1.05)
 if foot:
  am=profile(q,UP,[(-.2,.35),(.40,.35)]);ast=mf.Manifold();an=mf.Manifold();a=q+UP*.40;ac=q
 bm,bst,bn,b,bc=endpoint(p,db,1.05 if abs(db[1])>.5 else 1.30)
 if b[1]<=a[1]+.18:return None
 if route and foot and abs(p[2])>.5:
  yy=min(b[1]-.20,q[1]+1.60);t=(yy-a[1])/(b[1]-a[1]);mid=a*(1-t)+b*t;mid[1]=yy;mid[2]=np.sign(p[2])*max(1.20,abs(mid[2]))
  body=join([mf.Manifold.hull_points(np.vstack([ring(a,da,.35),ring(mid,UP,.35)])),mf.Manifold.hull_points(np.vstack([ring(mid,UP,.35),ring(b,db,.35)]))])
 else:body=mf.Manifold.hull_points(np.vstack([ring(a,da,.35),ring(b,db,.35)]))
 m=join([am,bm,body]);st=join([ast,bst])
 return m,st,[an,bn],dict(anchor_point=q.tolist(),contact_point=p.tolist(),root_kind='foundation_foot' if foot else 'previous_fixed_shell',routing_variant=route,body_diameter_mm=.70,neck_diameter_mm=.40,neck_length_mm=.14,cut_points=[ac.tolist(),bc.tolist()],centerline=[q.tolist(),a.tolist(),b.tolist(),p.tolist()],endpoint_directions=[da.tolist(),db.tolist()],retained_stump_axial_length_mm=.19,lower_foundation_required=foot)
def first_unrooted(m,root_y,start=None,grid=.05):
 for h in schedule(m,grid):
  if start is not None and h<start-1e-7:continue
  for c in prefix(m,h).decompose():
   if vol(c)>EPS and c.bounding_box()[1]>root_y+1e-7:return h,c
 return None
def validate(m,root_y):
 hs=schedule(m,.05);bad=[]
 for h in hs:
  for c in prefix(m,h).decompose():
   if vol(c)>EPS and c.bounding_box()[1]>root_y+1e-7:bad.append(dict(height_y_mm=h,volume_mm3=vol(c),bounds_mm=list(c.bounding_box())))
 return dict(passed=not bad,heights_checked=len(hs),grid_step_mm=.05,local_minimum_offsets_mm=[.0001,.001,.005,.025],unrooted_component_records=bad,volume_epsilon_mm3=EPS,scope='Actual fixed shell and all added supports; anchor foot must be seated by separate foundation')
def build(n,resume=False):
 cache=ROOT/'cache/spheres';out=ROOT/'cache/custom_sphere_supports';out.mkdir(exist_ok=True)
 mp=cache/f'sphere_{n}layers.json';meta=json.loads(mp.read_text());r=meta['outer_nominal_radius_mm'];outer=load(cache/meta['free_layer_meshes'][0]);base_v=verts(outer)
 forbidden=join([load(cache/p) for p in meta['free_layer_meshes'][1:]]+[load(cache/w['mesh_file']) for w in meta['release_wires']])
 foot_c=np.array([0.,-r-2.8,0.]);foot=cylinder(foot_c-[0,.3,0],foot_c+[0,.3,0],.6,n=32);root_y=foot.bounding_box()[1]
 save(out/f'sphere_{n}layers_anchor_foot.npz',foot)
 current=outer+foot;supports=[foot];stumps=[];records=[];start=None
 if resume:
  import trimesh
  for path in sorted(out.glob(f'sphere_{n}layers_support_*.npz')):
   k=int(path.stem.rsplit('_',1)[1]);sp=out/f'sphere_{n}layers_stumps_{k:03d}.npz'
   if not sp.exists():break
   m=load(path);st=load(sp);cuts=[];dirs=[]
   for piece in st.decompose():
    a=piece.to_mesh64();tm=trimesh.Trimesh(np.asarray(a.vert_properties[:,:3]),np.asarray(a.tri_verts),process=False)
    opts=[]
    for facet,area in zip(tm.facets,tm.facets_area):
     if abs(float(area)-.12423314165)<.003:
      fi=np.asarray(facet);center=np.average(tm.triangles_center[fi],axis=0,weights=tm.area_faces[fi]);normal=tm.face_normals[fi[0]];opts.append((abs(float(area)-.12423314165),center,normal))
    if not opts:raise RuntimeError('Cannot recover exact cached cut disk')
    _,center,normal=min(opts,key=lambda t:t[0]);cuts.append(center.tolist());dirs.append(normal.tolist())
   pairs=sorted(zip(cuts,dirs),key=lambda p:p[0][1]);cuts=[p[0] for p in pairs];dirs=[p[1] for p in pairs]
   kind='foundation_foot' if len(cuts)==1 else 'previous_fixed_shell'
   if len(cuts)==1:cuts=[foot_c.tolist()]+cuts;dirs=[UP.tolist()]+dirs
   contacts=[(np.array(c)-np.array(d)*(.19 if abs(d[1])>.8 else .49)).tolist() for c,d in zip(cuts,dirs)]
   if kind=='foundation_foot':contacts[0]=foot_c.tolist()
   rec=dict(support_index=k,anchor_point=contacts[0],contact_point=contacts[1],root_kind=kind,cut_points=cuts,endpoint_directions=dirs,body_diameter_mm=.70,neck_diameter_mm=.40,neck_length_mm=.14,mesh_file=path.name,stump_mesh_file=sp.name,geometry_reused_and_finally_revalidated=True)
   records.append(rec);supports.append(m);stumps.append(st)
  current=join([outer]+supports)
  print(f'L{n} resume {len(records)} cached supports',flush=True)
 for k in range(len(records),150):
  found=first_unrooted(current,root_y,start)
  if found is None:break
  h,seed=found;v=verts(seed);p=np.mean(v[v[:,1]<=v[:,1].min()+1e-8],axis=0)
  mask=(base_v[:,1]<p[1]-1.6)&(base_v[:,1]>p[1]-7.)&(np.linalg.norm(base_v,axis=1)>r-.12)&(base_v[:,1]>-r*.91)
  cs=base_v[mask]
  if len(cs):
   cs=cs[np.argsort(np.linalg.norm(cs-p,axis=1))];_,ids=np.unique(np.round(cs/.35).astype(int),axis=0,return_index=True);cs=cs[np.sort(ids)][:100]
  choices=[(q,False,0) for q in cs]
  if p[1]<-r*.45:choices.extend([(foot_c,True,0),(foot_c,True,1)])
  accepted=None;reasons={}
  for q,isfoot,route in choices:
   result=construct(q,p,isfoot,route)
   if result is None:reasons['too_short']=reasons.get('too_short',0)+1;continue
   m,st,necks,rec=result
   if overlap(m,forbidden)>1e-10:reasons['inner_collision']=reasons.get('inner_collision',0)+1;continue
   # Avoid unplanned fixed material bonds, preserving actual narrow waists.
   foreign=(m^outer)-join([ball(p,.65),ball(q,.65)])
   if vol(foreign)>1e-8:reasons['unintended_fixed_contacts']=reasons.get('unintended_fixed_contacts',0)+1;continue
   if max(overlap(nn,outer) for nn in (necks[1:] if isfoot else necks))>1e-9:reasons['buried_neck']=reasons.get('buried_neck',0)+1;continue
   anchor=foot if isfoot else prefix(current,p[1]-1.)
   contact=m^anchor
   if vol(contact)<1e-7:reasons['missing_base_contact']=reasons.get('missing_base_contact',0)+1;continue
   first=float(contact.bounding_box()[1]);ytrim=first+.002;m=m.trim_by_plane(UP,ytrim);st=st.trim_by_plane(UP,ytrim)
   if overlap(m,seed)<min(EPS,vol(seed)*.1):reasons['missed_seed']=reasons.get('missed_seed',0)+1;continue
   candidate=current+m
   fail=first_unrooted(candidate,root_y,start=float(m.bounding_box()[1]),grid=.15)
   if fail and fail[0]<h-1e-6:reasons['unseated_support']=reasons.get('unseated_support',0)+1;continue
   containing=[c for c in prefix(candidate,h).decompose() if overlap(c,seed)>min(EPS,vol(seed)*.1)]
   if not containing or any(c.bounding_box()[1]>root_y+1e-7 for c in containing):reasons['target_unrooted']=reasons.get('target_unrooted',0)+1;continue
   rec.update(support_index=k,detection_height_y_mm=float(h),tip_minimum_y_mm=float(p[1]),seed_volume_mm3=vol(seed),base_first_contact_y_mm=first,anchor_flatten_y_mm=ytrim,model_volume_mm3=vol(m),forbidden_static_overlap_mm3=overlap(m,forbidden),previous_candidate_rejections=reasons)
   accepted=m,st,rec,candidate;break
  if accepted is None:
   diag=dict(module=n,failed_index=k,tip=p.tolist(),height=h,reasons=reasons);(ROOT/'reports'/f'custom_sphere_supports_{n}_failure.json').write_text(json.dumps(diag,indent=2));save(out/f'sphere_{n}layers_partial.npz',current);raise RuntimeError(str(diag))
  m,st,rec,current=accepted;supports.append(m);stumps.append(st);records.append(rec)
  save(out/f'sphere_{n}layers_support_{k:03d}.npz',m);save(out/f'sphere_{n}layers_stumps_{k:03d}.npz',st)
  rec['mesh_file']=f'sphere_{n}layers_support_{k:03d}.npz';rec['stump_mesh_file']=f'sphere_{n}layers_stumps_{k:03d}.npz'
  print(f'L{n} support {k+1}: y={p[1]:.4f}, root={rec["root_kind"]}, rejected={reasons}',flush=True);start=float(m.bounding_box()[1])
 else:raise RuntimeError('Support limit exceeded')
 check=validate(current,root_y)
 if not check['passed']:raise RuntimeError('Final connectivity failed')
 only=join(supports);save(out/f'sphere_{n}layers_supports.npz',only);save(out/f'sphere_{n}layers_supported_outer.npz',current);save(out/f'sphere_{n}layers_retained_stumps.npz',join(stumps))
 report=dict(passed=True,module_layers=n,source_metadata_sha256=digest(mp),source_outer_mesh=meta['free_layer_meshes'][0],source_outer_sha256=digest(cache/meta['free_layer_meshes'][0]),build_direction=[0,1,0],supports=records,support_count=len(records),foundation_foot=dict(center=foot_c.tolist(),diameter_mm=1.2,height_mm=.6,bounds=list(foot.bounding_box()),mesh_file=f'sphere_{n}layers_anchor_foot.npz',requirement='Entire first cross-section must attach to earlier holder/coupon foundation'),connectivity=check,permanent_shells_modified=False,internal_mechanism_modified=False,support_only_mesh=f'sphere_{n}layers_supports.npz',removal_sequence='Cut both narrow necks before gripping each twig. Remove all outer supports before internal mechanism supports.',limitations=['No physical peel or resin strength validation','Whole-holder removal and foundation connectivity require separate checks'])
 (ROOT/'reports'/f'custom_sphere_supports_{n}.json').write_text(json.dumps(report,indent=2)+'\n');(out/f'sphere_{n}layers_supports.json').write_text(json.dumps(report,indent=2)+'\n')
 print(f'L{n} done: {len(records)} supports; {check["heights_checked"]} heights',flush=True)

# V6 contextual generator. Diagnostic V1 routes above are not release inputs.
def compact_y(q,p,rbody,rneck):
 q=np.asarray(q,float);p=np.asarray(p,float);dy=p[1]-q[1]
 if not .42<=dy<=4.5:raise ValueError('Compact vertical range')
 ss=[(-.08,rbody),(.005,rbody),(.06,rneck),(.14,rneck),(.18,rbody),(dy-.18,rbody),(dy-.14,rneck),(dy-.06,rneck),(dy+.005,rbody),(dy+.08,rbody)]
 # Section centers are sheared; every section is perpendicular to build Y.
 shear=np.array([[1.,(p[0]-q[0])/dy,0.,q[0]],[0.,1.,0.,q[1]],[0.,(p[2]-q[2])/dy,1.,q[2]]])
 full=profile(np.zeros(3),UP,ss).transform(shear)
 ns=[profile(np.zeros(3),UP,[(.06,rneck),(.14,rneck)]).transform(shear),profile(np.zeros(3),UP,[(dy-.14,rneck),(dy-.06,rneck)]).transform(shear)]
 st=join([full.trim_by_plane(-UP,-float(q[1]+.10)),full.trim_by_plane(UP,float(p[1]-.10))])
 pieces=[profile(np.zeros(3),UP,[a,b]).transform(shear) for a,b in zip(ss[:-1],ss[1:])]
 angle_cos=dy/np.linalg.norm(p-q)
 return full,st,ns,pieces,dict(style='rising_horizontal_sections',cut_points=[(q+(p-q)*(.10/dy)).tolist(),(p-(p-q)*(.10/dy)).tolist()],cut_plane_normals=[[0,1,0],[0,1,0]],neck_length_build_mm=.08,horizontal_neck_diameter_mm=2*rneck,horizontal_body_diameter_mm=2*rbody,minimum_neck_minor_diameter_normal_to_axis_mm=2*rneck*angle_cos*math.cos(math.pi/24),polygon_segments=24,polygon_inradius_factor=math.cos(math.pi/24),retained_height_from_each_contact_mm=.10,anchor_point=q.tolist(),contact_point=p.tolist(),centerline=[q.tolist(),p.tolist()])

def pitched_y(q,p,nq,rbody=.24,rneck=.16,offset=.16,carrier=.52):
 q=np.asarray(q,float);p=np.asarray(p,float);nq=np.asarray(nq,float)
 def pitched(n,sgn):
  n=np.asarray(n,float);n/=np.linalg.norm(n);xz=n.copy();xz[1]=0;xz/=max(np.linalg.norm(xz),1e-12)
  if n[1]*sgn<.75:n=math.sqrt(1-.75**2)*xz+np.array([0.,sgn*.75,0.])
  return n
 da=pitched(nq,1);db=pitched(p/np.linalg.norm(p),-1)
 ss=[(-.10,rbody),(.015,rbody),(offset,rneck),(offset+.14,rneck),(carrier-.04,rbody),(carrier,rbody)]
 if not all(a[0]<b[0] for a,b in zip(ss,ss[1:])):raise ValueError('endpoint station order')
 a=q+carrier*da;b=p+carrier*db
 if b[1]<=a[1]+.03:raise ValueError('non-rising body')
 em=[profile(q,da,ss),profile(p,db,ss)];ns=[profile(q,da,[(offset+.00001,rneck),(offset+.13999,rneck)]),profile(p,db,[(offset+.00001,rneck),(offset+.13999,rneck)])]
 body=mf.Manifold.hull_points(np.vstack([ring(a,da,rbody),ring(b,db,rbody)]))
 if any(overlap(body,x)>1e-10 for x in ns):raise ValueError('body obscures neck')
 full=join(em+[body]);cut=offset+.07
 st=join([profile(q,da,ss[:3]+[(cut,rneck)]),profile(p,db,ss[:3]+[(cut,rneck)])])
 pieces=[body]+[profile(pt,dr,[aa,bb]) for pt,dr in [(q,da),(p,db)] for aa,bb in zip(ss,ss[1:])]
 return full,st,ns,pieces,dict(style='short_normal_axis_necks',anchor_point=q.tolist(),contact_point=p.tolist(),cut_points=[(q+da*cut).tolist(),(p+db*cut).tolist()],cut_plane_normals=[da.tolist(),db.tolist()],neck_length_mm=.14,neck_diameter_mm=2*rneck,body_diameter_mm=2*rbody,minimum_neck_minor_diameter_normal_to_axis_mm=2*rneck*math.cos(math.pi/24),polygon_segments=24,polygon_inradius_factor=math.cos(math.pi/24),centerline=[q.tolist(),a.tolist(),b.tolist(),p.tolist()],end_profiles=[dict(point=pt.tolist(),direction=dr.tolist(),stations=ss,nominal_cut_axial_mm=cut,maximum_retained_axial_mm=offset+.14) for pt,dr in [(q,da),(p,db)]])

def axial_y(q,p):
 q=np.asarray(q,float);p=np.asarray(p,float);u=p-q;length=float(np.linalg.norm(u));u/=length
 if not .40<=length<=3.0 or u[1]<.20:raise ValueError('Short axial range')
 rn=.135;transverse=math.sqrt(max(0.,1-u[1]**2));rb=rn+min(.015,.8*.045*u[1]/max(transverse,1e-12))
 ss=[(-.08,rb),(.005,rb),(.05,rn),(.13,rn),(.18,rb),(length-.18,rb),(length-.13,rn),(length-.05,rn),(length+.005,rb),(length+.08,rb)]
 if np.any(np.diff([d*u[1]-rr*transverse for d,rr in ss])<-1e-10):raise ValueError('Backward shoulder')
 full=profile(q,u,ss);ends=[(-.08,rb),(.005,rb),(.05,rn),(.13,rn)];cut=.09
 st=join([profile(q,u,ends[:3]+[(cut,rn)]),profile(p,-u,ends[:3]+[(cut,rn)])]);ns=[profile(q,u,[(.05001,rn),(.12999,rn)]),profile(p,-u,[(.05001,rn),(.12999,rn)])]
 pieces=[profile(q,u,[a,b]) for a,b in zip(ss,ss[1:])]
 return full,st,ns,pieces,dict(style='short_round_axis_necks',anchor_point=q.tolist(),contact_point=p.tolist(),cut_points=[(q+cut*u).tolist(),(p-cut*u).tolist()],cut_plane_normals=[u.tolist(),(-u).tolist()],neck_length_mm=.08,neck_diameter_mm=2*rn,body_diameter_mm=2*rb,minimum_neck_minor_diameter_normal_to_axis_mm=2*rn*math.cos(math.pi/24),polygon_segments=24,polygon_inradius_factor=math.cos(math.pi/24),centerline=[q.tolist(),p.tolist()],end_profiles=[dict(point=pt.tolist(),direction=dr.tolist(),stations=ends,nominal_cut_axial_mm=cut,maximum_retained_axial_mm=.13) for pt,dr in [(q,u),(p,-u)]])

def radial_escape_y(q,p,rbody=.21,rneck=.135):
 q=np.asarray(q,float);p=np.asarray(p,float)
 def outward(pt,up):
  yz=pt[1]/np.linalg.norm(pt);horizontal=pt.copy();horizontal[1]=0;horizontal/=np.linalg.norm(horizontal)
  yy=(.35 if yz<-.15 else .75) if up>0 else (-.75 if yz<.4 else -.20)
  return horizontal*math.sqrt(1-yy*yy)+np.array([0.,yy,0.])
 da=outward(q,1);db=outward(p,-1)
 specs=[];ends=[];ns=[];st=[];pieces=[];cuts=[];carriers=[]
 for pt,d in [(q,da),(p,db)]:
  off=.40 if d[1]>0 and d[1]<.5 else (.55 if d[1]<0 and abs(d[1])<.5 else .16)
  carrier=max(off+.44,.76 if abs(d[1])>.5 else 1.08)
  ss=[(-.08,.15),(.015,.15),(off,rneck),(off+.14,rneck),(carrier-.04,rbody),(carrier,rbody)]
  low=np.array([x*abs(d[1])-rr*math.sqrt(1-d[1]*d[1]) for x,rr in ss])
  if np.any(np.diff(low)<-1e-10):raise ValueError('radial endpoint backward shoulder')
  en=profile(pt,d,ss);ends.append(en);ns.append(profile(pt,d,[(off+.00001,rneck),(off+.13999,rneck)]));st.append(profile(pt,d,ss[:3]+[(off+.07,rneck)]));cuts.append((pt+d*(off+.07)).tolist());carriers.append(pt+d*carrier);pieces.extend(profile(pt,d,[a,b]) for a,b in zip(ss,ss[1:]));specs.append(dict(point=pt.tolist(),direction=d.tolist(),stations=ss,nominal_cut_axial_mm=off+.07,maximum_retained_axial_mm=off+.14))
 a,b=carriers
 if b[1]<=a[1]+.04:raise ValueError('radial body not rising')
 body=mf.Manifold.hull_points(np.vstack([ring(a,da,rbody),ring(b,db,rbody)]));pieces.append(body)
 if any(overlap(body,x)>1e-10 for x in ns):raise ValueError('radial body covers waist')
 full=join(ends+[body])
 return full,join(st),ns,pieces,dict(style='outward_bent_round_necks',anchor_point=q.tolist(),contact_point=p.tolist(),cut_points=cuts,cut_plane_normals=[da.tolist(),db.tolist()],neck_length_mm=.14,neck_diameter_mm=2*rneck,body_diameter_mm=2*rbody,contact_cap_diameter_mm=.30,minimum_neck_minor_diameter_normal_to_axis_mm=2*rneck*math.cos(math.pi/24),polygon_segments=24,polygon_inradius_factor=math.cos(math.pi/24),centerline=[q.tolist(),a.tolist(),b.tolist(),p.tolist()],end_profiles=specs)

def inner_surface_arc_y(q,p,nq):
 q=np.asarray(q,float);p=np.asarray(p,float);da=np.asarray(nq,float);da/=np.linalg.norm(da);db=-p/np.linalg.norm(p)
 if da[1]<.15 or db[1]>-.60 or p[1]-q[1]<.4:raise ValueError('Inner arc endpoint orientation')
 rn=.103;rb=.106;specs=[];ends=[];ns=[];st=[];pieces=[];cuts=[];carriers=[]
 for pt,d,carrier in [(q,da,.20),(p,db,.135)]:
  ss=[(-.065,.11),(.002,.11),(.025,rn),(.105,rn),(carrier-.01,rb),(carrier,rb)]
  if np.any(np.diff([x*abs(d[1])-rr*math.sqrt(max(0.,1-d[1]*d[1])) for x,rr in ss])<-1e-10):raise ValueError('Arc endpoint backward shoulder')
  end=profile(pt,d,ss);ends.append(end);ns.append(profile(pt,d,[(.02501,rn),(.10499,rn)]));st.append(profile(pt,d,ss[:3]+[(.065,rn)]));cuts.append((pt+d*.065).tolist());carriers.append(pt+d*carrier);pieces.extend(profile(pt,d,[a,b]) for a,b in zip(ss,ss[1:]));specs.append(dict(point=pt.tolist(),direction=d.tolist(),stations=ss,nominal_cut_axial_mm=.065,maximum_retained_axial_mm=.105))
 a,b=carriers
 if b[1]<a[1]+.10:raise ValueError('Arc body must rise')
 la=np.linalg.norm(a);lb=np.linalg.norm(b);ua=a/la;ub=b/lb;ang=math.acos(float(np.clip(ua@ub,-1,1)));count=max(2,int(math.ceil(np.linalg.norm(b-a)/.45))+1)
 ts=np.linspace(0,1,count);points=[]
 for t in ts:
  u=ua*(1-t)+ub*t;u/=np.linalg.norm(u);points.append(u*(la*(1-t)+lb*t))
 points=np.asarray(points);directions=[da]+[(points[i+1]-points[i-1])/np.linalg.norm(points[i+1]-points[i-1]) for i in range(1,count-1)]+[db]
 sections=[ring(pt,d,rb) for pt,d in zip(points,directions)]
 bodyparts=[mf.Manifold.hull_points(np.vstack([aa,bb])) for aa,bb in zip(sections,sections[1:])];body=join(bodyparts)
 if any(overlap(body,x)>1e-10 for x in ns):raise ValueError('Arc body covers neck')
 pieces+=bodyparts;full=join(ends+[body])
 return full,join(st),ns,pieces,dict(style='inner_surface_curved_round_necks',anchor_point=q.tolist(),contact_point=p.tolist(),cut_points=cuts,cut_plane_normals=[da.tolist(),db.tolist()],neck_length_mm=.08,neck_diameter_mm=2*rn,body_diameter_mm=2*rb,minimum_neck_minor_diameter_normal_to_axis_mm=2*rn*math.cos(math.pi/24),polygon_segments=24,polygon_inradius_factor=math.cos(math.pi/24),centerline=[q.tolist()]+points.tolist()+[p.tolist()],body_section_rings=[dict(center=pt.tolist(),normal=d.tolist(),radius_mm=rb) for pt,d in zip(points,directions)],end_profiles=specs,design_note='Curved narrow temporary bridge follows fixed inner roof while remaining outside the certified moving envelope; it must be cut into short fragments if the global extraction plan requires this.')

def reconstruct_support_pieces(rec):
 q=np.asarray(rec['anchor_point']);p=np.asarray(rec['contact_point']);style=rec['style']
 if style=='permanent_micro_bridge':return []
 if style=='rising_horizontal_sections':return compact_y(q,p,rec['horizontal_body_diameter_mm']/2,rec['horizontal_neck_diameter_mm']/2)[3]
 if style=='short_round_axis_necks':return axial_y(q,p)[3]
 if style in ['short_normal_axis_necks','outward_bent_round_necks','inner_surface_curved_round_necks']:
  pieces=[];rings=[]
  for end in rec['end_profiles']:
   pt=np.asarray(end['point']);d=np.asarray(end['direction']);ss=end['stations']
   pieces.extend(profile(pt,d,[a,b]) for a,b in zip(ss,ss[1:]));rings.append(ring(pt+d*ss[-1][0],d,ss[-1][1]))
  if style=='inner_surface_curved_round_necks':
   rings=[ring(e['center'],e['normal'],e['radius_mm']) for e in rec['body_section_rings']]
   pieces.extend(mf.Manifold.hull_points(np.vstack([a,b])) for a,b in zip(rings,rings[1:]))
  else:pieces.append(mf.Manifold.hull_points(np.vstack(rings)))
  return pieces
 raise RuntimeError(f'Cannot reconstruct convex pieces for {style}')

def nominal_stump_primitives(rec,pieces):
 if rec['style']=='rising_horizontal_sections':
  q=np.asarray(rec['anchor_point']);p=np.asarray(rec['contact_point'])
  return [x for part in pieces for x in [part.trim_by_plane(-UP,-float(q[1]+.10)),part.trim_by_plane(UP,float(p[1]-.10))] if not x.is_empty()]
 result=[]
 for end in rec['end_profiles']:
  pt=np.asarray(end['point']);d=np.asarray(end['direction']);cut=end['nominal_cut_axial_mm']
  for a,b in zip(end['stations'],end['stations'][1:]):
   if a[0]>=cut:break
   upper=min(b[0],cut);radius=a[1]+(b[1]-a[1])*(upper-a[0])/(b[0]-a[0])
   result.append(profile(pt,d,[a,(upper,radius)]))
 return result

def convex_difference(part,cutter):
 if overlap(part,cutter)<=1e-12:return [part]
 if vol(part-cutter)<=1e-12:return []
 mesh=cutter.to_mesh64();vv=np.asarray(mesh.vert_properties[:,:3]);tri=vv[np.asarray(mesh.tri_verts)];nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);lens=np.linalg.norm(nn,axis=1);use=lens>1e-14;nn=nn[use]/lens[use,None];dd=np.einsum('ij,ij->i',nn,tri[use,0]);seen=set();remaining=part;outside=[]
 for normal,d in zip(nn,dd):
  key=tuple(np.round(np.r_[normal,d],9))
  if key in seen:continue
  seen.add(key);piece=remaining.trim_by_plane(normal,float(d));remaining=remaining.trim_by_plane(-normal,float(-d))
  if vol(piece)>1e-12:outside.append(piece)
  if remaining.is_empty():break
 return outside

def persist_removable_convex(n,rec,full,st,out,pieces=None):
 if rec['kind']=='fixed_micro_reinforcement':rec['generated_convex_sweep_meshes']=[];return
 removable=full-st
 if pieces is None and rec.get('generated_convex_sweep_meshes'):
  cached=[load(out/name) for name in rec['generated_convex_sweep_meshes']];joined=join(cached);missing=vol(removable-joined);extra=vol(joined-removable);convex_error=max((vol(mf.Manifold.hull_points(verts(x))-x) for x in cached),default=0.)
  if missing<=1e-8 and convex_error<=1e-9:
   rec['removable_convex_decomposition'].update(final_saved_meshes_revalidated=True,union_uncovered_actual_volume_mm3=missing,union_extra_proxy_volume_mm3=extra,union_symmetric_difference_mm3=missing+extra,maximum_convex_hull_fill_mm3=convex_error,conservative_sweep_outer_cover=True);return
 pieces=reconstruct_support_pieces(rec) if pieces is None else pieces
 kept=[];paths=[];local_fill=0.;cutters=None
 for piece in pieces:
  x=piece^removable
  if x.is_empty():continue
  parts=x.decompose();maxextra=max((vol(mf.Manifold.hull_points(verts(a))-a) for a in parts),default=0.)
  if maxextra>1e-9:
   if cutters is None:cutters=nominal_stump_primitives(rec,pieces)
   parts=[piece.trim_by_plane(UP,float(rec['root_flatten_y_mm']))]
   for cutter in cutters:parts=[fragment for a in parts for fragment in convex_difference(a,cutter)]
  for part in parts:
   # Saved union meshes can contain tiny seam pockets absent from the ideal
   # construction frustum. Hull only this local actual fragment, never the
   # entire bent support. Its swept volume conservatively contains reality.
   actual=part^removable
   for connected in actual.decompose():
    if vol(connected)<1e-12:continue
    hull=mf.Manifold.hull_points(verts(connected));local_fill=max(local_fill,vol(hull-connected));kept.append(hull)
 joined=join(kept);missing=vol(removable-joined)
 if missing>1e-8:
  for residual in (removable-joined).decompose():
   if vol(residual)>1e-12:kept.append(mf.Manifold.hull_points(verts(residual)))
  joined=join(kept);missing=vol(removable-joined)
 extra=vol(joined-removable)
 if missing>1e-8:raise RuntimeError(f'Convex sweep proxies miss actual body L{n}/{rec["support_index"]}: {missing}')
 convex_error=max((vol(mf.Manifold.hull_points(verts(x))-x) for x in kept),default=0.)
 if convex_error>1e-9:raise RuntimeError('A saved sweep primitive is not convex')
 for i,part in enumerate(kept):
  name=f'sphere_{n}layers_removable_convex_{rec["support_index"]:03d}_{i:02d}.npz';save(out/name,part);paths.append(name)
 rec['generated_convex_sweep_meshes']=paths;rec['removable_convex_decomposition']=dict(passed=True,piece_count=len(kept),conservative_sweep_outer_cover=True,maximum_convex_hull_fill_mm3=convex_error,maximum_local_actual_fragment_hull_fill_mm3=local_fill,union_uncovered_actual_volume_mm3=missing,union_extra_proxy_volume_mm3=extra,union_symmetric_difference_mm3=missing+extra,coverage_volume_tolerance_mm3=1e-8,source='Construction primitives intersected with the exact nominal removable solid and partitioned at endpoint stump planes. Each local fragment is conservatively convex-hulled; all actual removable material is covered. Small reported extra seam fill makes extraction collision checks stricter. Printed geometry is unchanged.')

def maximum_stump(full,rec):
 if rec['style']=='permanent_micro_bridge':return full
 if rec['style']=='rising_horizontal_sections':
  q=np.array(rec['anchor_point']);p=np.array(rec['contact_point']);return join([full.trim_by_plane(-UP,-float(q[1]+.14)),full.trim_by_plane(UP,float(p[1]-.14))])
 return join([profile(np.array(e['point']),np.array(e['direction']),e['stations'][:4]) for e in rec['end_profiles']]) ^ full

def fixed_environment_paths(n):
 base=ROOT/'cache/support_environment'
 paths=sorted(base.glob(f'sphere_{n}layers_holder_*_source_float32.npz'))
 paths=[p for p in paths if 'all_placements' not in p.name and p.stat().st_size>0]
 if len(paths)!=3:raise RuntimeError(f'Need three finished actual-placement environments for L{n}, have {paths}')
 return paths

def root_contact(c,envs):
 # Routing acceleration only. Final exact union-prefix and whole-holder
 # validation remain mandatory; a near gap alone is not a release proof.
 return all(c.min_gap(e,.00002)<=.000003 for e in envs)

def context_failure(solid,envs,start=None,grid=.05):
 for h in schedule(solid,grid):
  if start is not None and h<start-1e-7:continue
  for c in prefix(solid,h).decompose():
   if vol(c)>EPS and not root_contact(c,envs):return h,c
 return None

def min_radius(m):
 import trimesh
 a=m.to_mesh64();v=np.asarray(a.vert_properties[:,:3]);f=np.asarray(a.tri_verts);tri=v[f]
 close=trimesh.triangles.closest_point(tri,np.zeros((len(tri),3)))
 return float(np.linalg.norm(close,axis=1).min())

def context_exact_validation(solid,envs):
 hs=schedule(solid,.05);rows=[];allpass=True
 for ei,env in enumerate(envs):
  merged=solid+env;bad=[]
  for h in hs:
   for c in prefix(merged,h).decompose():
    if vol(c)<=EPS:continue
    if overlap(c,env)<=EPS:
     bad.append(dict(height_y_mm=h,volume_mm3=vol(c),bounds_mm=list(c.bounding_box())))
  rows.append(dict(environment_index=ei,passed=not bad,heights_checked=len(hs),unrooted_records=bad));allpass&=not bad
 return dict(passed=allpass,placement_checks=rows,grid_step_mm=.05,local_minimum_offsets_mm=[.0001,.001,.005,.025],volume_epsilon_mm3=EPS,scope='Every added/outer prefix component must reach actual fixed holder material; whole-holder build validation is a separate required gate')

def certified_motion_cylinder(n,meta,cache,out,margin=.025):
 report_path=ROOT/'reports/motion_validation.json';motion=json.loads(report_path.read_text());mod=next(m for m in motion['modules'] if m['layer_count']==n)
 if not motion['passed'] or not mod['passed']:raise RuntimeError('Original motion certificate failed')
 if mod['metadata_sha256']!=digest(cache/f'sphere_{n}layers.json'):raise RuntimeError('Motion metadata mismatch')
 if np.linalg.norm(np.array(mod['axis_unit_vector'])-np.array([1.,0.,0.]))>1e-12:raise RuntimeError('Expected common X axis')
 expected={r['file']:r['sha256'] for r in mod['actual_with_stumps_layer_topology']};rows=[];theta=0.;epsilon=.00003
 for index,name in enumerate(meta['released_with_stumps_layer_meshes'][1:],1):
  path=cache/name
  if digest(path)!=expected[name]:raise RuntimeError('Moving-layer certificate hash mismatch')
  pair=[b for b in mod['measured_bearings'] if b['moving_index']==index]
  step=max(math.atan2(float(b['conservative_maximum_radial_play_mm']),float(b['actual_mouth_axial_mm'])-epsilon) for b in pair);theta+=step
  vv=verts(load(path));xx=np.abs(vv[:,0]);rr=np.linalg.norm(vv[:,1:],axis=1)
  tr=np.minimum(np.arctan2(xx,rr),theta);tx=np.minimum(np.arctan2(rr,xx),theta)
  rawrho=float(np.max(rr*np.cos(tr)+xx*np.sin(tr)));rawx=float(np.max(xx*np.cos(tx)+rr*np.sin(tx)))
  displacement=float(mod['cumulative_play_allowance_by_layer_mm'][index]);rho=rawrho+displacement+epsilon;ax=rawx+displacement+epsilon
  rows.append(dict(layer_index=index,mesh_file=name,mesh_sha256=digest(path),joint_tilt_bound_degrees=math.degrees(step),cumulative_tilt_bound_degrees=math.degrees(theta),cumulative_center_displacement_mm=displacement,raw_rotation_tilt_radial_bound_mm=rawrho,raw_rotation_tilt_axial_bound_mm=rawx,export_allowance_mm=epsilon,cylinder_radius_mm=rho,cylinder_axial_half_length_mm=ax))
 rho=max(a['cylinder_radius_mm'] for a in rows);ax=max(a['cylinder_axial_half_length_mm'] for a in rows);segments=128
 def make(pad):return cylinder([-ax-pad-1e-6,0,0],[ax+pad+1e-6,0,0],(rho+pad+1e-6)/math.cos(math.pi/segments),n=segments)
 cert=make(0.);guard=make(margin)
 def side_inradius(m):
  vv=verts(m);lo=vv[:,0].min();yz=vv[abs(vv[:,0]-lo)<1e-7,1:];yz=yz[np.argsort(np.arctan2(yz[:,1],yz[:,0]))];bb=np.roll(yz,-1,axis=0);dd=bb-yz;t=np.clip(-np.einsum('ij,ij->i',yz,dd)/np.einsum('ij,ij->i',dd,dd),0,1);return float(np.linalg.norm(yz+t[:,None]*dd,axis=1).min())
 gi=side_inradius(guard);ci=side_inradius(cert)
 if ci<rho or gi<rho+margin:raise RuntimeError('Cylinder polygon underbounds analytic envelope')
 save(out/f'sphere_{n}layers_motion_envelope.npz',cert);save(out/f'sphere_{n}layers_motion_guard.npz',guard)
 proof=dict(passed=True,layer_count=n,axis=[1,0,0],source_motion_report_sha256=digest(report_path),source_metadata_sha256=mod['metadata_sha256'],layers=rows,cylinder_radius_mm=rho,cylinder_axial_half_length_mm=ax,additional_support_clearance_margin_mm=margin,circumscribed_polygon_segments=segments,measured_certificate_polygon_inradius_mm=ci,measured_guard_polygon_inradius_mm=gi,mesh_file=f'sphere_{n}layers_motion_envelope.npz',guard_mesh_file=f'sphere_{n}layers_motion_guard.npz',formula='For each source vertex x=abs(X), r=hypot(Y,Z): max radial coordinate r*cos(t)+x*sin(t), t=min(atan2(x,r),sum joint tilt); analogous axial maximum uses t=min(atan2(r,x),sum tilt). Add the certified cumulative centre displacement and export allowance.',proof='Arbitrary rotations about the nominal X axis preserve x and r. Each ancestor joint changes axis direction by at most its measured/padded tilt; angular triangle inequality bounds their sum. Maximizing over tilt azimuth gives the stated expression. For each fixed pose the norm of a linear projection is convex, so triangle interiors cannot exceed the maximum over their vertices. The same argument bounds axial projection. Adding the Euclidean displacement bound to each projection is conservative. Every rotating source layer is thus inside the union of these cylinders; the largest radius/half-length cylinder contains that union.',scope='Rigid original layers with original measured journal clearances, including retained internal support stumps; no resin deformation or printer shrinkage model')
 (out/f'sphere_{n}layers_motion_envelope.json').write_text(json.dumps(proof,indent=2));(ROOT/'reports'/f'custom_sphere_supports_{n}_motion_envelope.json').write_text(json.dumps(proof,indent=2));return cert,guard,proof

def certified_motion_support_envelope(n,meta,cache,out,margin=.025):
 # The cylinder is retained as a simple independently checkable outer bound.
 # Oblique fixed inner faces need the tighter convex support-function envelope.
 cyl,cylguard,proof=certified_motion_cylinder(n,meta,cache,out,margin)
 save(out/f'sphere_{n}layers_motion_cylinder.npz',cyl)
 save(out/f'sphere_{n}layers_motion_cylinder_guard.npz',cylguard)
 angles=np.linspace(0.,math.pi/2,65);aa=np.cos(angles);bb=np.sin(angles)
 bounds=np.zeros(len(angles));per_layer=[]
 for row in proof['layers']:
  vv=verts(load(cache/row['mesh_file']));xx=np.abs(vv[:,0]);rr=np.linalg.norm(vv[:,1:],axis=1)
  radius=np.hypot(xx,rr);beta=np.arctan2(rr,xx);theta=math.radians(row['cumulative_tilt_bound_degrees'])
  extra=row['cumulative_center_displacement_mm']+row['export_allowance_mm']
  hh=np.array([float(np.max(radius*np.cos(np.maximum(np.abs(beta-phi)-theta,0.))))+extra for phi in angles])
  bounds=np.maximum(bounds,hh);per_layer.append(dict(layer_index=row['layer_index'],support_bounds_mm=hh.tolist()))
 def polygon(pad):
  hs=bounds+pad+1e-6
  poly=[np.array([0.,0.]),np.array([hs[0],0.]),np.array([hs[0],hs[-1]]),np.array([0.,hs[-1]])]
  for a,b,h in zip(aa,bb,hs):
   normal=np.array([a,b]);outp=[]
   for u,v in zip(poly,poly[1:]+poly[:1]):
    du=float(normal@u-h);dv=float(normal@v-h);iu=du<=1e-12;iv=dv<=1e-12
    if iu:outp.append(u)
    if iu!=iv:outp.append(u+(v-u)*(du/(du-dv)))
   poly=outp
  return np.asarray(poly)
 segments=128
 def revolve(poly):
  pts=[];ts=np.arange(segments)*2*math.pi/segments
  for x,rad in poly:
   if x<1e-10 and rad<1e-10:continue
   # A circumscribed polygon contains the full analytic disk at every station.
   rad=rad/math.cos(math.pi/segments)
   for sign in [-1.,1.]:pts.extend(np.column_stack([np.full(segments,sign*x),rad*np.cos(ts),rad*np.sin(ts)]))
  return mf.Manifold.hull_points(np.asarray(pts))
 poly=polygon(0.);gpoly=polygon(margin);cert=revolve(poly);guard=revolve(gpoly)
 save(out/f'sphere_{n}layers_motion_envelope.npz',cert);save(out/f'sphere_{n}layers_motion_guard.npz',guard)
 proof.update(envelope_type='circumscribed_axisymmetric_convex_support_envelope',support_direction_count=len(angles),support_direction_angles_radians=angles.tolist(),support_direction_axial_weights=aa.tolist(),support_direction_radial_weights=bb.tolist(),support_direction_bounds_mm=bounds.tolist(),per_layer_support_bounds=per_layer,certificate_first_quadrant_profile_mm=poly.tolist(),guard_first_quadrant_profile_mm=gpoly.tolist(),profile_numerical_padding_mm=1e-6,angular_circumscription_factor=1/math.cos(math.pi/segments),cylinder_mesh_file=f'sphere_{n}layers_motion_cylinder.npz',cylinder_guard_mesh_file=f'sphere_{n}layers_motion_cylinder_guard.npz',mesh_sha256=digest(out/f'sphere_{n}layers_motion_envelope.npz'),guard_mesh_sha256=digest(out/f'sphere_{n}layers_motion_guard.npz'),formula='For each direction phi in [0,pi/2], h(phi)=max_over_layers_and_vertices sqrt(x*x+r*r)*cos(max(abs(atan2(r,x)-phi)-theta,0)) + cumulative_center_displacement + export_allowance, with x=abs(X), r=hypot(Y,Z), and cumulative measured joint tilt theta. Each halfplane cos(phi)*abs(X)+sin(phi)*rho <= h(phi) is an independently valid outer bound. The intersection is revolved with a circumscribed polygon, never an inscribed one.',proof='Any permitted joint composition has deviation from a nominal X rotation at most the sum of joint tilt bounds. The polar angle beta of a vertex can therefore change by at most theta. The positive-weight axial/radial support functional is maximized by moving beta toward phi, yielding the formula. For a fixed pose, a*abs(X)+b*hypot(Y,Z) is convex; triangle/solid interiors cannot exceed vertex maxima. A Euclidean displacement contributes at most its norm because (a,b) is unit length. Intersecting these valid halfplanes remains an outer bound, including all arbitrary axial rolls and their permitted accumulated tilts/play. The profile is convex, and each angular polygon is circumscribed; linear interpolation of aligned polygon rings contains the analytic rotated profile at all axial positions. Convex hull of those rings therefore cannot underbound the analytic enclosure. Additional support margin is independently added to every supporting direction before polygonization.')
 (out/f'sphere_{n}layers_motion_envelope.json').write_text(json.dumps(proof,indent=2));(ROOT/'reports'/f'custom_sphere_supports_{n}_motion_envelope.json').write_text(json.dumps(proof,indent=2))
 return cert,guard,proof

def outside_support_envelope(points,proof,pad=0.):
 pp=np.asarray(points,float);single=pp.ndim==1;pp=np.atleast_2d(pp);xx=np.abs(pp[:,0]);rr=np.linalg.norm(pp[:,1:],axis=1);outside=np.zeros(len(pp),bool)
 for a,b,h in zip(proof['support_direction_axial_weights'],proof['support_direction_radial_weights'],proof['support_direction_bounds_mm']):outside|=a*xx+b*rr>h+pad
 return bool(outside[0]) if single else outside

def build_contextual(n,resume=False):
 import custom_holder_supports as H
 import trimesh
 cache=ROOT/'cache/spheres';out=ROOT/'cache/custom_sphere_supports';out.mkdir(exist_ok=True)
 mp=cache/f'sphere_{n}layers.json';meta=json.loads(mp.read_text());r=meta['outer_nominal_radius_mm'];outer=load(cache/meta['free_layer_meshes'][0])
 ep=fixed_environment_paths(n);environment_hashes_at_load={str(p):digest(p) for p in ep};envs=[load(p) for p in ep];env=join(envs).as_original()
 run_id=f'L{n}_{time.time_ns()}';(out/f'generator_source_{SOURCE_LOADED_SHA256}.py').write_bytes(SOURCE_CODE_BYTES)
 # True mechanism and already modelled supports, not just nominal spheres.
 forbidden=join([load(cache/p) for p in meta['free_layer_meshes'][1:]]+[load(cache/w['mesh_file']) for w in meta['release_wires']])
 corridors=join([load(cache/e['tool_corridor_mesh_file']) for w in meta['release_wires'] for e in w['endpoints']])
 extraction=[]
 for w in meta['release_wires']:
  wm=load(cache/w['removable_mesh_file']);bb=wm.bounding_box();proj=mf.CrossSection(wm.project().to_polygons(),mf.FillRule.Positive);extraction.append(mf.Manifold.extrude(proj,r+4.-bb[2]).translate((0,0,bb[2])))
 corridors=join([corridors]+extraction)
 bound={3:8.56380847,4:10.55667117,5:12.55228651}[n]
 motion_cylinder,motion_guard,motion_proof=certified_motion_support_envelope(n,meta,cache,out);rho_bound=motion_proof['cylinder_radius_mm'];axial_bound=motion_proof['cylinder_axial_half_length_mm']
 rot=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0]],float);rotation=rot[:,:3]
 source_path=out/f'sphere_{n}layers_context_surface_z.npz';save(source_path,(outer+env).transform(rot));surf=H.Surface(source_path)
 original_local=(outer+env).as_original();originals=[(outer+e).as_original() for e in envs];current=outer;supports=[];stumps=[];necks=[];micro=[];records=[];start=None
 if resume and (out/f'sphere_{n}layers_checkpoint.json').exists():
  checkpoint=json.loads((out/f'sphere_{n}layers_checkpoint.json').read_text());records=checkpoint['records'];skipped=checkpoint.get('skipped_targets',[])
  for rec in records:
   full=load(out/rec['mesh_file']);st=load(out/rec['stump_mesh_file']);supports.append(full);stumps.append(st)
   if rec['kind']=='fixed_micro_reinforcement':micro.append(full)
   else:necks.extend(load(out/f'sphere_{n}layers_neck_{rec["support_index"]:03d}_{j}.npz') for j in [0,1])
  current=join([outer]+supports);print(f'L{n} restored {len(records)} contextual supports',flush=True)
 targets=json.loads((ROOT/f'cache/support_environment/sphere_{n}layers_birth_targets.json').read_text())['union_targets']
 targets=sorted(targets,key=lambda row:row['representative_local_position_mm'][1]);skipped=[]
 for ti,target in enumerate(targets):
  k=len(records);p=np.array(target['representative_local_position_mm'],float);h=float(p[1]+min(.001,target['minimum_surface_lifetime_mm']*.25))
  candidates=[c for c in prefix(current,h).decompose() if vol(c)>SEED_EPS and c.min_gap(ball(p,.012),.01)<.001]
  seed=min(candidates,key=lambda c:c.min_gap(ball(p,.0000001),.02)) if candidates else None
  earlier=join(supports)
  if seed is not None and not earlier.is_empty() and overlap(seed,earlier)>SEED_EPS:
   skipped.append(dict(target_index=ti,reason='The target print-prefix component is already attached to an earlier added support',point=p.tolist()));continue
  print(f'L{n} context target {ti+1}/{len(targets)} addition {k+1} at {p.tolist()}',flush=True)
  pz=rotation@p
  # Roots are actual upward faces hit by a descending ray, not arbitrary vertices.
  root_candidates=surf.roots(pz,max_length=10.,minimum_drop=.025)
  delta=surf.centers-pz;dd=np.linalg.norm(delta,axis=1);mask=(delta[:,2]<-.50)&(delta[:,2]>-4.5)&(dd<8.)&(outside_support_envelope(surf.centers,motion_proof,.04)|(np.linalg.norm(surf.centers,axis=1)>bound+.04))
  ids=np.flatnonzero(mask);ids=ids[np.argsort(dd[ids])];used=set();extra=[]
  for fi in ids:
   key=tuple(np.round(surf.centers[fi]/.30).astype(int))
   if key in used:continue
   used.add(key);extra.append((float(dd[fi]),surf.centers[fi],surf.norm[fi],int(fi)))
   if len(extra)>=80:break
  root_candidates+=extra
  if p[1]>r*.65:
   mask=(delta[:,2]<-.4)&(delta[:,2]>-7.)&(dd<11.)&(surf.norm[:,2]>.15)&outside_support_envelope(surf.centers,motion_proof,.025)
   ids=np.flatnonzero(mask);ids=ids[np.argsort(dd[ids])];used=set();extra=[]
   for fi in ids:
    key=tuple(np.round(surf.centers[fi]/.30).astype(int))
    if key in used:continue
    used.add(key);extra.append((float(dd[fi]),surf.centers[fi],surf.norm[fi],int(fi)))
    if len(extra)>=120:break
   root_candidates+=extra
  micro_limit=.55 if p[1]>r*.65 else .40
  near=np.asarray(surf.kd.query_ball_point(pz,micro_limit+surf.max_radius),dtype=int)
  near=near[np.linalg.norm(surf.centers[near]-pz,axis=1)-surf.radii[near]<=micro_limit]
  clipped=[]
  for level_drop in ([.002,.0005,.0001] if p[1]>r*.65 else [.030]):
   level=pz[2]-level_drop
   for fi in near:
    poly=list(surf.tri[fi]);pp=[]
    for aa,bb in zip(poly,poly[1:]+poly[:1]):
     ai=aa[2]<=level;bi=bb[2]<=level
     if ai:pp.append(aa)
     if ai!=bi:pp.append(aa+(bb-aa)*((level-aa[2])/(bb[2]-aa[2])))
    for j in range(1,len(pp)-1):clipped.append([pp[0],pp[j],pp[j+1]])
  if clipped:
   tris=np.asarray(clipped);points=trimesh.triangles.closest_point(tris,np.broadcast_to(pz,(len(tris),3)).copy());ds=np.linalg.norm(points-pz,axis=1);used=set()
   for kk in np.argsort(ds):
    if ds[kk]>micro_limit:break
    key=tuple(np.round(points[kk]/.025).astype(int))
    if key in used:continue
    used.add(key);root_candidates.append((float(ds[kk]),points[kk],np.array([0.,0.,1.]),-1))
    if len(used)>=40:break
  choices=[]
  for dist,qz,nz,face in root_candidates:
   q=rotation.T@qz
   if np.linalg.norm(q)<bound+.025 and not outside_support_envelope(q,motion_proof,.025):continue
   if p[1]>r*.65 and nz[2]>.15 and p[1]-q[1]>.4:choices.append((q,'inner_arc',.106,.103,rotation.T@nz))
   if dist<=micro_limit:
    choices.extend([(q,'micro',.12,.12,rotation.T@nz),(q,'micro',.10,.10,rotation.T@nz)])
   if .4<=np.linalg.norm(p-q)<=3.0 and (p[1]-q[1])/np.linalg.norm(p-q)>=.20:choices.append((q,'axial',.15,.135,rotation.T@nz))
   if p[1]-q[1]>=.70:choices.extend((q,'radial',a,.135,rotation.T@nz) for a in [.21,.18])
   if p[1]-q[1]>=.42:
    structural=bool(abs(p[2])<.7 and p[1]<-r+.6)
    sizes=[(.40,.30),(.32,.24)] if structural else [(.28,.21),(.24,.18),(.21,.155)]
    for aa,bb in sizes:
     cosang=(p[1]-q[1])/np.linalg.norm(p-q);bb=max(bb,.102/(cosang*math.cos(math.pi/24)));aa=max(aa,bb+.035)
     if aa<=.45:choices.append((q,'compact',aa,bb,rotation.T@nz))
    choices.extend((q,'pitched',aa,bb,rotation.T@nz) for aa,bb in [(.28,.18),(.24,.15),(.21,.12)])
  if p[1]>r*.65:choices.sort(key=lambda choice:0 if choice[1]=='micro' else 1)
  reasons={};accepted=None;local_cache={}
  for q,kind,rb,rn,nq in choices:
   try:
    if kind=='micro':
     full=capsule(q,p,rb);st=full;ns=[];pieces=[full]
     rec=dict(style='permanent_micro_bridge',anchor_point=q.tolist(),contact_point=p.tolist(),nominal_diameter_mm=2*rb,centerline=[q.tolist(),p.tolist()],maximum_centerline_span_mm=micro_limit,actual_centerline_span_mm=float(np.linalg.norm(p-q)),micro_span_design_exception=('Fixed inner-roof local gap may span up to 0.55 mm with the same 0.05 mm3 added-volume cap; all motion and service checks remain required' if np.linalg.norm(p-q)>.40 else None))
    elif kind=='compact':full,st,ns,pieces,rec=compact_y(q,p,rb,rn)
    elif kind=='pitched':full,st,ns,pieces,rec=pitched_y(q,p,nq,rb,rn)
    elif kind=='radial':full,st,ns,pieces,rec=radial_escape_y(q,p,rb,rn)
    elif kind=='inner_arc':full,st,ns,pieces,rec=inner_surface_arc_y(q,p,nq)
    else:full,st,ns,pieces,rec=axial_y(q,p)
   except ValueError:continue
   key=tuple(np.round(q,8))
   if key not in local_cache:
    fb=np.array(full.bounding_box());lo=np.minimum(np.minimum(q,p)-2.,fb[:3]-.1);hi=np.maximum(np.maximum(q,p)+2.,fb[3:]+.1);box=mf.Manifold.cube(tuple(hi-lo)).translate(lo);local_cache[key]=((original_local^box).as_original(),[(oo^box).as_original() for oo in originals],lo,hi)
   query_original,query_originals,boxlow,boxhigh=local_cache[key];fb=np.array(full.bounding_box())
   if np.any(fb[:3]<boxlow) or np.any(fb[3:]>boxhigh):raise RuntimeError('Candidate escaped exact local obstacle box')
   # Positive, earlier anchor contact in every actual placement.
   tight_roof_micro=kind=='micro' and p[1]>r*.65
   anchor_probe_drop=min(.015,max(.00001,(p[1]-q[1])*.25)) if tight_roof_micro else .015
   floor_padding=min(.002,max(.000001,(p[1]-q[1])*.01)) if tight_roof_micro else .002
   low=prefix(full,p[1]-anchor_probe_drop)
   if any(overlap(low,oo)<1e-8 for oo in query_originals):reasons['missing_anchor']=reasons.get('missing_anchor',0)+1;continue
   contacts=[full^oo for oo in query_originals]
   if any(vol(x)<1e-8 for x in contacts):continue
   floor=max(float(x.bounding_box()[1]) for x in contacts)+floor_padding
   if floor>=p[1]-floor_padding:reasons['late_start']=reasons.get('late_start',0)+1;continue
   full=full.trim_by_plane(UP,floor);st=st.trim_by_plane(UP,floor);pieces=[x.trim_by_plane(UP,floor) for x in pieces]
   if kind=='micro' and vol(full-query_original)>.05:reasons['large_micro']=reasons.get('large_micro',0)+1;continue
   radial_min=min_radius(full);axis_checked=radial_min<bound+.025
   if axis_checked and overlap(full,motion_guard)>1e-10:reasons['moving_envelope']=reasons.get('moving_envelope',0)+1;continue
   if overlap(full,forbidden)>1e-10:reasons['mechanism_collision']=reasons.get('mechanism_collision',0)+1;continue
   maximum_st=maximum_stump(full,rec)
   if overlap(maximum_st,corridors)>1e-10:reasons['retained_tool_collision']=reasons.get('retained_tool_collision',0)+1;continue
   if kind!='micro':
    if any(overlap(x,query_original)>1e-9 for x in ns):reasons['buried_neck']=reasons.get('buried_neck',0)+1;continue
    if overlap(full-st,query_original)>1e-8:reasons['extra_contact']=reasons.get('extra_contact',0)+1;continue
    if any(overlap(x,join(supports))>1e-9 for x in ns):reasons['new_neck_bypass']=reasons.get('new_neck_bypass',0)+1;continue
    if any(overlap(full,x)>1e-9 for x in necks):reasons['old_neck_bypass']=reasons.get('old_neck_bypass',0)+1;continue
   if kind!='micro' and rec['minimum_neck_minor_diameter_normal_to_axis_mm']<.20:reasons['neck_too_small']=reasons.get('neck_too_small',0)+1;continue
   if any(overlap(full,x)>1e-9 for x in necks):reasons['old_neck_bypass']=reasons.get('old_neck_bypass',0)+1;continue
   old_removable=join([a-b for a,b in zip(supports,stumps)])
   if overlap(full,old_removable)>1e-9:reasons['old_body_bond']=reasons.get('old_body_bond',0)+1;continue
   if kind!='micro' and overlap(full-st,join(supports))>1e-9:reasons['new_body_bond']=reasons.get('new_body_bond',0)+1;continue
   candidate=(current+full).as_original()
   bad=False
   for yy in schedule(full,.05):
    if yy>p[1]-anchor_probe_drop:break
    for part in prefix(full,yy).decompose():
     if vol(part)>EPS and any(overlap(part,oo)<=EPS for oo in query_originals):bad=True;break
    if bad:break
   if bad:reasons['unseated_new_prefix']=reasons.get('unseated_new_prefix',0)+1;continue
   if vol(ball(p,.0005)-full)>1e-16:reasons['missed_target_ball']=reasons.get('missed_target_ball',0)+1;continue
   rec.update(generation_source_sha256=SOURCE_LOADED_SHA256,generation_run_id=run_id,source_target_index=ti,required_in_placements=target['required_in_placements'],support_index=k,kind='fixed_micro_reinforcement' if kind=='micro' else 'removable_fixed_support',detected_y_mm=h,tip_min_y_mm=float(p[1]),root_flatten_y_mm=floor,root_floor_padding_mm=floor_padding,earlier_anchor_probe_drop_mm=anchor_probe_drop,forbidden_mechanism_overlap_mm3=overlap(full,forbidden),minimum_radial_distance_mm=min_radius(full),spherical_moving_overbound_mm=bound,moving_clearance_method='certified_rotation_tilt_play_support_envelope' if axis_checked else 'conservative_spherical_bound',motion_envelope_certificate=f'sphere_{n}layers_motion_envelope.json',motion_guard_overlap_mm3=overlap(full,motion_guard) if axis_checked else 0.,added_volume_outside_original_mm3=vol(full-query_original),retained_tool_corridor_overlap_mm3=overlap(st,corridors),candidate_rejections=reasons)
   accepted=full,st,ns,pieces,rec,candidate;break
  if accepted is None:
   diag=dict(layer_count=n,failed_target=p.tolist(),height_y_mm=h,candidates=len(choices),reasons=reasons,completed=len(records));(ROOT/'reports'/f'custom_sphere_supports_{n}_context_failure.json').write_text(json.dumps(diag,indent=2));save(out/f'sphere_{n}layers_context_partial.npz',current);raise RuntimeError(str(diag))
  full,st,ns,pieces,rec,current=accepted;supports.append(full);stumps.append(st);necks+=ns;records.append(rec)
  if rec['kind']=='fixed_micro_reinforcement':micro.append(full)
  rec['mesh_file']=f'sphere_{n}layers_support_{k:03d}.npz';rec['stump_mesh_file']=f'sphere_{n}layers_stumps_{k:03d}.npz';rec['removable_mesh_file']=f'sphere_{n}layers_removable_{k:03d}.npz'
  save(out/rec['mesh_file'],full);save(out/rec['stump_mesh_file'],st);save(out/rec['removable_mesh_file'],full-st)
  persist_removable_convex(n,rec,full,st,out,pieces)
  for j,x in enumerate(ns):save(out/f'sphere_{n}layers_neck_{k:03d}_{j}.npz',x)
  ck=out/f'sphere_{n}layers_checkpoint.json';tmp=ck.with_name(ck.name+'.part');tmp.write_text(json.dumps(dict(run_id=run_id,source_sha256=SOURCE_LOADED_SHA256,records=records,skipped_targets=skipped),indent=2));tmp.replace(ck)
  print(f'L{n} accepted {k+1} {rec["kind"]}, volume={rec["added_volume_outside_original_mm3"]:.5f}, rejected={reasons}',flush=True)
  start=h-.001
 print(f'L{n}: routing complete ({len(records)} additions); whole-holder audit required separately',flush=True)
 check=dict(passed=True,scope='Generation checks only: each new support has actual earlier anchors in all three placements and every positive added prefix reaches original fixed material; all authoritative targets were processed. Full assembled float32 all-height birth validation is required separately.',per_added_prefix_grid_step_mm=.05,local_minimum_offsets_mm=[.0001,.001,.005,.025],volume_epsilon_mm3=EPS,full_model_build_validation='required separately',full_context_grid_redundant_with_global_gate=True)
 assert all(digest(p)==environment_hashes_at_load[str(p)] for p in ep),'Environment changed during generation'
 save(out/f'sphere_{n}layers_supports.npz',join(supports));save(out/f'sphere_{n}layers_supported_outer.npz',current);save(out/f'sphere_{n}layers_retained_stumps.npz',join(stumps));save(out/f'sphere_{n}layers_micro_reinforcements.npz',join(micro))
 report=dict(passed=False,finalization_complete=False,generation_checks_passed=check['passed'],full_model_build_validation='required separately',module_layers=n,generation_run_id=run_id,generation_source_sha256=SOURCE_LOADED_SHA256,source_metadata_sha256=digest(mp),source_outer_sha256=digest(cache/meta['free_layer_meshes'][0]),permanent_original_vertices_unchanged=True,internal_mechanism_modified=False,motion_envelope=motion_proof,build_direction=[0,1,0],supports=records,support_count=len(records),globally_detected_target_count=len(targets),skipped_targets=skipped,removable_support_count=sum(x['kind']=='removable_fixed_support' for x in records),fixed_micro_reinforcement_count=len(micro),foundation_foot=dict(used=False),environment_inputs=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in ep],connectivity=check,limits=['Geometric build support only; no physical resin or peel-force test','Permanent microbridges touch only fixed features','Final full-holder validation and tool/extraction review are separate release gates'])
 (out/f'sphere_{n}layers_supports.json').write_text(json.dumps(report,indent=2));(ROOT/'reports'/f'custom_sphere_supports_{n}.json').write_text(json.dumps(report,indent=2))
 print(f'L{n} complete: {len(records)} additions, exact placement checks {check["passed"]}',flush=True)
 if not check['passed']:raise RuntimeError('Exact context validation failed')

def finalize_retained(n):
 out=ROOT/'cache/custom_sphere_supports';cache=ROOT/'cache/spheres';meta=json.loads((cache/f'sphere_{n}layers.json').read_text());rp=out/f'sphere_{n}layers_supports.json';report=json.loads(rp.read_text())
 tool_parts=[load(cache/e['tool_corridor_mesh_file']) for w in meta['release_wires'] for e in w['endpoints']]
 extrusion_parts=[]
 for w in meta['release_wires']:
  m=load(cache/w['removable_mesh_file']);bb=m.bounding_box();polys=m.project().to_polygons();proj=mf.CrossSection(polys,mf.FillRule.Positive)
  extrusion_parts.append(mf.Manifold.extrude(proj,meta['outer_nominal_radius_mm']+4.-bb[2]).translate((0,0,bb[2])))
 tool_keepout=join(tool_parts);extraction_keepout=join(extrusion_parts);stumps=[];checks=[]
 for rec in report['supports']:
  full=load(out/rec['mesh_file'])
  nominal_st=load(out/rec['stump_mesh_file']);persist_removable_convex(n,rec,full,nominal_st,out)
  if rec['kind']=='fixed_micro_reinforcement':st=full
  else:
   st=maximum_stump(full,rec)
   if rec['style']=='rising_horizontal_sections':rec['conservative_retained_height_from_each_contact_mm']=.14;rec['nominal_cut_height_from_each_contact_mm']=.10
  fname=f'sphere_{n}layers_retained_max_{rec["support_index"]:03d}.npz';save(out/fname,st);rec['conservative_retained_mesh_file']=fname;stumps.append(st)
  a=overlap(st,tool_keepout);b=overlap(st,extraction_keepout)
  checks.append(dict(support_index=rec['support_index'],kind=rec['kind'],tool_corridor_overlap_mm3=a,internal_wire_translation_envelope_overlap_mm3=b,passed=a<=1e-9 and b<=1e-9))
 save(out/f'sphere_{n}layers_retained_stumps.npz',join(stumps))
 report['retained_state_service_check']=dict(passed=all(x['passed'] for x in checks),support_checks=checks,wire_extraction_envelope='Positive-fill actual XY removable-wire projection extruded from each original minimum Z to beyond outer sphere; conservative superset of all +Z translations',compact_max_retained_build_height_mm=.14,compact_nominal_cut_build_height_mm=.10,normal_axis_profiles='See each end_profiles entry for actual nominal and maximum axial dimensions')
 report['finalization_complete']=True
 report['passed']=report['generation_checks_passed'] and report['retained_state_service_check']['passed'] and all(x['kind']=='fixed_micro_reinforcement' or x.get('removable_convex_decomposition',{}).get('passed',False) for x in report['supports'])
 rp.write_text(json.dumps(report,indent=2));(ROOT/'reports'/f'custom_sphere_supports_{n}.json').write_text(json.dumps(report,indent=2))
 print(f'L{n} retained service gate: {report["retained_state_service_check"]["passed"]}',flush=True)
 if not report['passed']:raise RuntimeError('Retained-state service validation failed')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--layers',type=int,choices=[3,4,5],required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();build_contextual(a.layers,a.resume);finalize_retained(a.layers)

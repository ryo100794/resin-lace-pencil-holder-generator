#!/usr/bin/env python3
"""Nested openwork spheres and embroidery pen holder, rotating variation V5.

Dimensions in millimetres. Each ornament is actual closed 3D geometry.
The inherited embroidery module supplies the faceted thread, stitched
leaf, textured floor and 24 petal picot rim techniques, without a dragon.
Run with numpy, scipy, shapely, trimesh, manifold3d and mapbox-earcut.
"""
from pathlib import Path
import argparse, hashlib, json, math, sys, time
import numpy as np
import manifold3d as mf
from scipy.interpolate import CubicSpline
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from shapely import affinity
import embroidery_technique as E

ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'cache'
TAU=2*math.pi
E.R_REF=46.0
E.CIRC=TAU*E.R_REF
E.Y_SCALE=1.0
E.RAD=CubicSpline([0,3.2,12,28,55,82,99,112],
                  [46.6,46.8,46.4,45.8,45.0,45.9,46.8,47.4],bc_type='natural')
OUTER=E.RAD
INNER=lambda z:32.8+0.3*np.clip((np.asarray(z)-84)/17,0,1)
SPHERES=[]
for kind,r,l,z,phase in [('large',13.,5,45.,0),('medium',11.,4,78.,40),('small',9.,3,26.,80)]:
    for k in range(3):
        SPHERES.append(dict(kind=kind,radius=r,layers=l,z=z,theta_deg=phase+120*k,center_radius=44.8))

def log(s):E.log(s)

def save(m,name):
    mm=m.to_mesh64()
    np.savez_compressed(CACHE/(name+'.npz'),vertices=mm.vert_properties[:,:3],faces=mm.tri_verts)
    log(f'Saved {name}: {m.num_tri():,} triangles, {m.volume():.2f} mm3')
    return m

def load(name):
    d=np.load(CACHE/(name+'.npz'))
    return E.status(mf.Manifold(mf.Mesh64(d['vertices'],d['faces'].astype(np.uint32))),name)

def stage(name,fn,rebuild=False):
    if (CACHE/(name+'.npz')).exists() and not rebuild:
        log('Reuse '+name)
        return load(name)
    return save(fn(),name)

def join(items,name):
    items=[m for m in items if m is not None and not m.is_empty()]
    if not items:raise ValueError('Empty group: '+name)
    return E.merge(items,name)

def tube3(points,r=.5,sides=12,closed=False):
    """Closed polygonal sweep with transported normal, finite capped ends."""
    p=np.asarray(points,float)
    if np.linalg.norm(p[0]-p[-1])<1e-7:
        p=p[:-1];closed=True
    p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-7]]
    if len(p)<2:raise ValueError('Short path')
    tang=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0) if closed else np.gradient(p,axis=0)
    tang/=np.linalg.norm(tang,axis=1)[:,None]
    seed=np.array([0.,0.,1.])
    if abs(tang[0]@seed)>.85:seed=np.array([1.,0.,0.])
    u=seed-tang[0]*(seed@tang[0]);u/=np.linalg.norm(u)
    vv=[]
    tt=np.arange(sides)*TAU/sides
    for q,t in zip(p,tang):
        u-=t*(u@t);u/=np.linalg.norm(u)
        v=np.cross(t,u)
        vv.append(q+r*(np.cos(tt)[:,None]*u+np.sin(tt)[:,None]*v))
    vv=np.vstack(vv)
    f=[]
    for i in range(len(p) if closed else len(p)-1):
        j=(i+1)%len(p)
        for k in range(sides):
            n=(k+1)%sides
            f.extend([[i*sides+k,i*sides+n,j*sides+n],[i*sides+k,j*sides+n,j*sides+k]])
    if not closed:
        for k in range(1,sides-1):
            f.append([0,k+1,k])
            f.append([(len(p)-1)*sides,(len(p)-1)*sides+k,(len(p)-1)*sides+k+1])
    f=np.asarray(f,np.uint32)
    tri=vv[f]
    if np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()<0:f=f[:,::-1].copy()
    return E.status(mf.Manifold(mf.Mesh64(vv,f)),'spatial bridge')

def sphere_center(s):
    a=math.radians(s['theta_deg'])
    return np.array([44.8*math.cos(a),44.8*math.sin(a),s['z']])

def sphere_transform(s):
    a=math.radians(s['theta_deg'])
    return np.column_stack(([-math.sin(a),math.cos(a),0],[0,0,1],
                            [math.cos(a),math.sin(a),0],sphere_center(s)))

def cutters():
    # Keep wall lace outside the complete sphere volumes, so no crossing
    # lattice hides the innermost layers or occupies an intended gap.
    return join([mf.Manifold.sphere(s['radius']+.48,80).translate(sphere_center(s))
                 for s in SPHERES],'sphere aperture cutters')

def pattern_mask(padding=2.6):
    p=[]
    for s in SPHERES:
        a=math.radians(s['theta_deg'])*E.R_REF
        for shift in [-E.CIRC,0,E.CIRC]:
            p.append(Point(a+shift,s['z']).buffer(s['radius']+padding,quad_segs=32))
    return unary_union(p)

def ground(inner=False):
    old=E.RAD
    if inner:E.RAD=INNER
    count=44 if inner else 60
    zz=np.linspace(2.70,99.45,265)
    family=[]
    try:
        for hand in [-1,1]:
            parts=[]
            for k in range(count):
                uv=-E.CIRC/2+k*E.CIRC/count+hand*.79*(zz-2.7)
                uv+=.10*np.sin(TAU*(zz-2.7)/(E.CIRC/count/.79))
                E.add_thread(parts,np.column_stack((uv,zz)),
                    width=.48 if inner else .60,depth=1.28 if inner else 1.72,
                    centre=-.04 if inner else -.12,step=.44,grain=.042)
            family.append(join(parts,('inner' if inner else 'outer')+f' weave {hand}'))
        return join(family,'two families of cutwork')
    finally:E.RAD=old

def circle_uv(s,r,n=181):
    t=np.linspace(0,TAU,n)
    return np.array([math.radians(s['theta_deg'])*E.R_REF,s['z']])+r*np.column_stack((np.cos(t),np.sin(t)))

def bezels():
    groups=[]
    for i,s in enumerate(SPHERES):
        parts=[];c=np.array([math.radians(s['theta_deg'])*E.R_REF,s['z']])
        r=s['radius']+1.28
        for delta,w,depth,mid in [(0,1.03,3.5,-.50),(1.0,.40,2.0,.75),(1.70,.32,1.6,.86)]:
            E.add_thread(parts,circle_uv(s,r+delta),width=w,depth=depth,
                         centre=mid,step=.30,grain=.04)
        # Paired concentric seams and dense transverse satin stitches.
        for th in np.arange(0,TAU,.71/(r+1.35)):
            v=np.array([math.cos(th),math.sin(th)])
            E.add_thread(parts,[c+v*(r+.92),c+v*(r+1.78)],width=.28,depth=1.5,
                         centre=.97,step=.25,grain=.025)
        # Twelve connected pointed leaves form a delicate calyx, behind the
        # spherical collar rather than filling any sphere viewing window.
        for k in range(12):
            th=TAU*k/12
            v=np.array([math.cos(th),math.sin(th)])
            E.lace_leaf(parts,c+v*(r+.95),c+v*(r+5.1),width=1.85,raised=.28)
        groups.append(join(parts,f'embroidered sphere bezel {i+1}/9'))
    return join(groups,'nine decorated load spreading bezels')

def botanical():
    groups=[];mask=pattern_mask(4.8)
    # Large paisleys occupy the diagonal spaces; all inherit exact .27-.42
    # mm thread sections and independent raised veins from the prior work.
    for k in range(3):
        c=[math.radians(78+120*k)*E.R_REF,58.0]
        parts=[]
        poly=E.lace_paisley(parts,c,hand=1,angle=-.22)
        groups.append(join(parts,f'paisley panel {k+1}'))
        p=poly.buffer(1.2)
        mask=unary_union([mask,p,affinity.translate(p,xoff=-E.CIRC),affinity.translate(p,xoff=E.CIRC)])
    # Fans at the foot and blossoms below the lace crown vary motif size.
    for k in range(9):
        parts=[];u=(k+.25)*E.CIRC/9
        if k%3==0:E.lace_fan(parts,[u,12.0],width=14.0,height=9.0)
        else:E.lace_rosette(parts,[u,12.0],4.6,6,phase=k*.3)
        E.lace_rosette(parts,[u+1.8,94.0],3.6,6,phase=k*.25,small=True)
        groups.append(join(parts,f'foot and crown embroidery {k+1}'))
    # Flowing tendrils follow uncluttered space, with fine open vein leaves.
    for k in range(15):
        base=(k+.36)*E.CIRC/15
        zz=np.linspace(5.0,98,240)
        xx=base+3.0*np.sin((zz-7)*TAU/65+k*.58)
        parts=[]
        for hand in [-1,1]:
            pp=np.column_stack((xx+hand*.19,zz))
            for ln in E.line_parts(LineString(pp).difference(mask)):
                if ln.length>1.0:E.add_thread(parts,np.asarray(ln.coords),width=.41,depth=1.9,
                    centre=.12+hand*.06,step=.35,grain=.035)
        for j,z in enumerate(np.arange(17,89,8.8)):
            u=base+3*np.sin((z-7)*TAU/65+k*.58)
            hand=-1 if (k+j)%2 else 1
            a=np.array([u,z]);b=a+[hand*(4.0+1.0*(j%2)),3.1]
            if not mask.intersects(LineString([a,b]).buffer(1.8)):
                E.lace_leaf(parts,a,b,width=2.6,raised=.23)
        if parts:groups.append(join(parts,f'twining embroidery {k+1}'))
    return join(groups,'all botanical filling')

def ring3(radius,z,r=.42):
    t=np.linspace(0,TAU,max(180,int(TAU*radius/.4))+1)
    return tube3(np.column_stack((radius*np.cos(t),radius*np.sin(t),np.full(len(t),z))),r=r,sides=10,closed=True)

def inner_anchor(theta,z):
    # Land each radial link exactly on an inner helical thread, not in a hole.
    pitch=E.CIRC/44
    shift=.79*(z-2.7)+.10*np.sin(TAU*(z-2.7)/(pitch/.79))
    k=round((theta*E.R_REF+E.CIRC/2-shift)/pitch)
    a=(-E.CIRC/2+k*pitch+shift)/E.R_REF
    return np.array([float(INNER(z))*math.cos(a),float(INNER(z))*math.sin(a),z])

def bridges():
    parts=[]
    # Fine pierced wall depth comes from 60 bowed links between two lace
    # skins. There is no solid thick cylindrical slab behind the ornament.
    for z in [9.0,30.0,52.0,74.0,96.0]:
        for k in range(12):
            a=TAU*(k+.22*(int(z)//10%2))/12
            pitch=E.CIRC/60
            shift=.79*(z-2.7)+.10*np.sin(TAU*(z-2.7)/(pitch/.79))
            index=round((a*E.R_REF+E.CIRC/2-shift)/pitch)
            a=(-E.CIRC/2+index*pitch+shift)/E.R_REF
            o=np.array([float(OUTER(z))*math.cos(a),float(OUTER(z))*math.sin(a),z])
            end=inner_anchor(a,z)
            t=np.linspace(0,1,25)
            path=o[None,:]*(1-t[:,None])+end[None,:]*t[:,None]
            path[:,2]+=1.25*np.sin(np.pi*t)
            parts.append(tube3(path,r=.51,sides=12))
    # Each spherical seat gets four independent curved roots. Top/bottom
    # links remain outside its sphere volume and connect to the inner weave.
    for s in SPHERES:
        a=math.radians(s['theta_deg']);n=np.array([math.cos(a),math.sin(a),0])
        tangent=np.array([-math.sin(a),math.cos(a),0])
        for sign in [-1,1]:
            z=s['z']+sign*(s['radius']+1.38)
            start=44.8*n+np.array([0,0,z])
            end=inner_anchor(a,z)
            t=np.linspace(0,1,25)
            p=start*(1-t[:,None])+end*t[:,None]
            p[:,2]+=sign*.75*np.sin(np.pi*t)
            parts.append(tube3(p,r=.62,sides=12))
        # Curved paired calyx arms transfer from local flat equator hoop to
        # developed cylindrical bezel along its tangential sides.
        for sign in [-1,1]:
            p0=sphere_center(s)+sign*(s['radius']+.40)*tangent
            au=a+sign*(s['radius']+1.60)/E.R_REF
            p1=np.array([float(OUTER(s['z']))*math.cos(au),float(OUTER(s['z']))*math.sin(au),s['z']])
            t=np.linspace(0,1,15)
            parts.append(tube3(p0*(1-t[:,None])+p1*t[:,None],r=.65,sides=12))
    # Open radial embroidery across the lip ties both skins into one collar.
    for k in range(36):
        a=TAU*k/36
        t=np.linspace(0,1,41)
        r=33.1+(float(OUTER(99.0))-33.1)*t
        phi=a+.020*np.sin(np.pi*t)
        z=98.7+1.15*np.sin(np.pi*t)
        p=np.column_stack((r*np.cos(phi),r*np.sin(phi),z))
        parts.append(tube3(p,r=.38,sides=10))
        # Three fine embroidery tacks along each lip spoke leave open depth.
        for q in [.28,.50,.72]:
            rr=33.1+(float(OUTER(99))-33.1)*q
            ts=np.linspace(-1,1,15)
            ph=a+ts*.012
            zz=98.7+1.15*math.sin(math.pi*q)+.18*np.sin(np.pi*(ts+1))
            parts.append(tube3(np.column_stack((rr*np.cos(ph),rr*np.sin(ph),zz)),r=.16,sides=8))
    for r,z,w in [(33.1,98.7,.52),(float(OUTER(99)),98.7,.55),
                  (float(OUTER(8)),8,.50),(float(OUTER(97)),97,.42)]:
        parts.append(ring3(r,z,w))
    return join(parts,'bowed radial links and pierced lip')

def assemble_spheres(released=False,sphere_cache=None):
    sphere_cache=sphere_cache or CACHE/'spheres'
    parts=[]
    for s in SPHERES:
        suffix='_released' if released else ''
        path=sphere_cache/f"sphere_{s['layers']}layers{suffix}.npz"
        if not path.exists():raise FileNotFoundError(path)
        d=np.load(path)
        m=mf.Manifold(mf.Mesh64(d['vertices'],d['faces'].astype(np.uint32)))
        parts.append(E.status(m.transform(sphere_transform(s)),s['kind']))
    return join(parts,'nine real concentric sphere ornaments')

def assembly_fingerprint(released=False,sphere_cache=None):
    """Do not silently reuse old sphere assemblies after mechanism changes."""
    suffix='_released' if released else ''
    paths=[CACHE/'wall_snapped.npz',Path(__file__)]
    sphere_cache=sphere_cache or CACHE/'spheres'
    paths += [sphere_cache/f'sphere_{n}layers{suffix}.npz' for n in (3,4,5)]
    result={}
    for path in paths:
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
        result[str(path.relative_to(ROOT))]=digest.hexdigest()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--wall-only',action='store_true');p.add_argument('--rebuild',action='store_true')
    p.add_argument('--released',action='store_true',help='Assemble the reference state after temporary support removal.')
    p.add_argument('--sphere-cache',type=Path,default=CACHE/'spheres')
    args=p.parse_args();CACHE.mkdir(exist_ok=True,parents=True)
    (ROOT/'reports'/'design.json').write_text(json.dumps(dict(name='Spherical ivory lace pen holder V5 with reinforced spherical shells',
      unit='mm',spheres=SPHERES,skin_radii_mm=[32.8,46.0],thread_width_mm=[.27,.62],
      sphere_mobility='opposed blind journals with removable necked support wires',
      surface='actual perforated geometry'),indent=2))
    if (CACHE/'wall_snapped.npz').exists() and not args.rebuild:
        wall=load('wall_snapped')
    else:
        cut=stage('apertures',cutters,args.rebuild)
        floor=stage('floor',lambda:E.make_textured_base(OUTER,1.0,3.2),args.rebuild)
        outer=stage('outer',lambda:ground(False)-cut,args.rebuild)
        inner=stage('inner',lambda:ground(True)-cut,args.rebuild)
        bezel=stage('bezels',bezels,args.rebuild)
        flowers=stage('botanical',lambda:botanical()-cut,args.rebuild)
        web=stage('bridges_snapped',lambda:bridges()-cut,args.rebuild)
        rim=stage('rim',E.build_open_frill,args.rebuild)
        wall=stage('wall_snapped',lambda:join([floor,outer,inner,bezel,flowers,web,rim],'thick openwork vessel'),args.rebuild)
    if args.wall_only:return
    state='released' if args.released else 'supported'
    args.sphere_cache=args.sphere_cache.resolve()
    fingerprint=assembly_fingerprint(args.released,args.sphere_cache)
    fingerprint_path=CACHE/('assembly_inputs_'+state+'.json')
    previous=json.loads(fingerprint_path.read_text()) if fingerprint_path.exists() else None
    changed=args.rebuild or previous!=fingerprint
    # Only the inherited decorative wall is simplified. Moving shells,
    # journals and fragile necks retain the geometry checked by audit_motion.
    wall=E.status(wall.as_original().set_tolerance(.001).simplify(.005),'fine wall simplification')
    save(wall,'wall_for_assembly')
    sphere=stage('sphere_instances_'+state,lambda:assemble_spheres(args.released,args.sphere_cache),changed)
    final=stage('assembled_'+state,lambda:join([wall,sphere],'complete nine sphere pen holder'),changed)
    final=E.status(final,'assembled mechanism surfaces preserved')
    save(final,'final_released' if args.released else 'final')
    fingerprint_path.write_text(json.dumps(fingerprint,indent=2)+'\n')
    log(f'Build finished; components={len(final.decompose())}')

if __name__=='__main__':main()

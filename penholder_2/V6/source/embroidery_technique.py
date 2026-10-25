#!/usr/bin/env python3
"""Parametric, one-piece floral embroidery pen holder, revision 5.
All modelling dimensions are millimetres. The walls, floral petals and frill are open embroidery structures.

Dependencies: numpy, scipy, shapely>=2, trimesh, manifold3d, mapbox-earcut.
Run: python build_flower_holder.py
The local .deps directory is optional and is not needed with installed packages.
This file is the editable design source, not a slicer profile.
"""
from pathlib import Path
import sys, math, json, time, zipfile
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon, Point, box, MultiPolygon, LineString
from shapely import affinity
from shapely.ops import unary_union
import manifold3d as mf
import trimesh

R_REF = 42.0
WALL = 2.2
Y_SCALE = 0.96
BOTTOM = 3.2
BODY_HEIGHT = 102.4
PROFILE = np.array([[0,39.2],[3.2,40.0],[12,42.0],[26,42.8],
                    [52,39.5],[70,39.3],[86,41.1],[99,43.4],
                    [103,44.0],[111,45.1]],dtype=float)
RAD = CubicSpline(PROFILE[:,0], PROFILE[:,1], bc_type='natural')
TAU = 2*math.pi
CIRC = TAU*R_REF
T0 = time.time()

def log(msg):
    print(f'{time.time()-T0:7.1f}s {msg}', flush=True)

def polys(g):
    if g.is_empty: return []
    if g.geom_type=='Polygon': return [g]
    return [p for q in g.geoms for p in polys(q)]

def cross(g):
    contours=[]
    for p in polys(g):
        contours.append(np.array(p.exterior.coords[:-1]))
        contours.extend(np.array(h.coords[:-1]) for h in p.interiors)
    return mf.CrossSection(contours, mf.FillRule.EvenOdd)

def status(m,name):
    if m.status()!=mf.Error.NoError or m.is_empty():
        raise ValueError(f'{name}: {m.status()}, empty={m.is_empty()}')
    return m

def mapped(m, reference='outer', resolution=1.6):
    if resolution: m=m.refine_to_length(resolution)
    def warp(v):
        s,z,d = v.T
        th=s/R_REF
        r=RAD(z)+d
        if reference=='inner': r=r-WALL
        return np.column_stack((r*np.cos(th),Y_SCALE*r*np.sin(th),z))
    return status(m.warp_batch(warp), 'warped solid')

def sheet(g, inner=-WALL, thick=WALL, resolution=1.65):
    m=cross(g).extrude(thick).translate((0,0,inner))
    return mapped(m,resolution=resolution)

def merge(parts, name):
    log(f'{name}: union {len(parts)} pieces')
    m=status(mf.Manifold.batch_boolean(parts,mf.OpType.Add),name)
    log(f'{name}: {m.num_tri():,} triangles')
    return m

def revolved_profile(profile,n=384):
    # Closed CCW meridional polygon: (radius,height).
    pts=np.array(profile,float)
    if Polygon(pts).exterior.is_ccw is False: pts=pts[::-1]
    return cross(Polygon(pts)).revolve(n).scale((1,Y_SCALE,1))

def leaf(w,h):
    # Pointed ends; lightly bulging sides give the lace a botanical character.
    t=np.linspace(0,1,11)
    x=(w/2)*np.sin(np.pi*t)**0.83
    x[0]=x[-1]=0.0
    left=np.column_stack((-x, -h/2+h*t))
    right=np.column_stack((x[-2:0:-1],(-h/2+h*t)[-2:0:-1]))
    return Polygon(np.vstack((left,right))).buffer(0)

FLOWERS=[(-math.pi/2,58.5,28.3,24,18,12),
         (math.pi/6,57.0,23.5,22,16,10),
         (5*math.pi/6,54.0,23.5,22,16,10)]
PETAL_PLANS=[]
FLOWER_MASKS=[]

def petal_plan(theta,cz,r0,r1,w,phi,edge,ridge):
    length=r1-r0
    # A 0.64 mm truncated end retains the angular silhouette without a needle.
    outline=np.array([[r0,-0.60],[r0+0.44*length,-w],
                      [r1-1.8,-0.77],[r1,-0.32],[r1,0.32],
                      [r1-1.8,0.77],[r0+0.44*length,w],[r0,0.60]])
    rot=np.array([[math.cos(phi),-math.sin(phi)],[math.sin(phi),math.cos(phi)]])
    center=np.array([theta*R_REF,cz])
    xy=outline@rot.T+center
    apex=(np.array([r0+0.53*length,0.0])@rot.T)+center
    poly=Polygon(xy)
    PETAL_PLANS.append((xy,apex,edge,ridge))
    return poly

for theta,cz,r,na,nb,nc in FLOWERS:
    pp=[]
    for n,r0,r1,w,e,h,phase in [
        (na,r*.28,r,2.45 if r>25 else 2.25,0.48,2.6,0.0),
        (nb,r*.15,r*.73,2.55 if r>25 else 2.25,1.0,3.7,0.50),
        (nc,r*.04,r*.43,2.05 if r>25 else 1.8,2.0,4.45,0.15)]:
        for k in range(n):
            phi=TAU*(k+phase)/n + math.pi/2
            # Small deterministic changes avoid a mechanical gear silhouette.
            tip=r1*(1+0.018*math.sin(3*phi+0.6))
            pp.append(petal_plan(theta,cz,r0,tip,w,phi,e,h))
    mask=unary_union(pp+[Point(theta*R_REF,cz).buffer(r*.20,quad_segs=24)])
    # Backing follows the petal silhouette; it is not a circular plaque.
    FLOWER_MASKS.append(mask.buffer(0.62,join_style=2))

def petal_solid(plan):
    xy,apex,edge,ridge=plan
    n=len(xy)
    # Local XY are developed circumferential distance and height.
    vv=np.vstack((np.column_stack((xy,np.full(n,-0.32))),
                  np.column_stack((xy,np.full(n,edge))),
                  [apex[0],apex[1],ridge]))
    ff=[]
    cap=np.array(mf.triangulate([xy]),int)
    for f in cap: ff.append(f[::-1])
    for i in range(n):
        j=(i+1)%n
        ff.extend([[i,j,j+n],[i,j+n,i+n],[i+n,j+n,2*n]])
    m=mf.Manifold(mf.Mesh(np.array(vv,np.float32),np.array(ff,np.uint32)))
    status(m,'petal')
    return mapped(m,resolution=1.35)


# All strands have a six-sided, nearly diamond section.  Both the outside
# and the inside therefore show edges, rather than a smooth backing panel.
COUNTS={'load_bearing_strands':0,'botanical_stems':0,'open_leaves':0,
        'small_flowers':0,'open_main_petals':0,'petal_stitches':0,
        'picot_loops':0,'filaments':0,'lace_rosettes':0,'lace_fans':0,
        'paisley_motifs':0,'trefoil_sprigs':0,'stitched_garlands':0,
        'lace_petal_stitches':0,'border_stitches':0,'rim_petals':0,
        'rim_cutwork_windows':0,'rim_stitches':0,'rim_bridges':0}

def triangle_wave(x):
    return 1.0-4.0*np.abs(np.mod(x,1.0)-0.5)

def thread_path(points,width=0.65,depth=2.2,centre=-1.1,
                step=0.60,grain=0.035,depth_fn=None):
    """Sweep a faceted thread in developed circumferential coordinates.

    width is in the developed surface; depth is radial.  centre and depth_fn
    may be functions of an array of (circumferential distance, height).
    A narrow 0.06 mm crest is retained on both faces.  Periodic circumferential
    paths join with shared vertices; there is no slit at the wrap seam.
    """
    p=np.asarray(points,dtype=float)
    p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-7]]
    if len(p)<2:return None
    delta=p[-1]-p[0]
    cyclic=(np.linalg.norm(delta)<1e-6)
    periodic=(abs(abs(delta[0])-CIRC)<1e-5 and abs(delta[1])<1e-5)
    closed=cyclic or periodic
    shift=delta if periodic else np.zeros(2)
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if arc[-1]<0.12:return None
    samples=np.linspace(0,arc[-1],max(4,int(math.ceil(arc[-1]/step))+1))
    pp=np.column_stack([np.interp(samples,arc,p[:,i]) for i in range(2)])
    if closed:pp=pp[:-1];samples=samples[:-1]
    if closed:
        before=np.roll(pp,1,axis=0);after=np.roll(pp,-1,axis=0)
        before[0]-=shift;after[-1]+=shift
        tangent=after-before
    else:tangent=np.gradient(pp,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))
    # Fine axial facets are real geometry, not a normal map or texture.
    modulation=1.0+grain*triangle_wave(samples/0.82)
    w=width*modulation
    crest=np.minimum(0.060,w*0.15)
    lateral=np.column_stack((-w/2,-crest/2,crest/2,w/2,crest/2,-crest/2))
    xy=pp[:,None,:]+lateral[:,:,None]*normal[:,None,:]
    uv=xy.reshape(-1,2)
    mid=np.asarray(centre(uv),float) if callable(centre) else np.full(len(uv),float(centre))
    dep=np.asarray(depth_fn(uv),float) if depth_fn else np.full(len(uv),depth)
    radial=np.tile(np.array([0,0.5,0.5,0,-0.5,-0.5]),len(pp))*dep+mid
    vv=np.column_stack((uv,radial))
    ff=[]
    for i in range(len(pp) if closed else len(pp)-1):
        ni=(i+1)%len(pp)
        for j in range(6):
            k=(j+1)%6;a=i*6+j;b=ni*6+j;c=ni*6+k;d=i*6+k
            ff.extend([[a,b,c],[a,c,d]])
    if not closed:
        for j in range(1,5):
            ff.append([0,j,j+1])
            ff.append([(len(pp)-1)*6,(len(pp)-1)*6+j+1,(len(pp)-1)*6+j])
    ff=np.asarray(ff,dtype=np.uint32)
    tri=vv[ff]
    if np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()<0:
        ff=ff[:,::-1].copy()
    solid=mf.Manifold(mf.Mesh(np.asarray(vv,np.float32),ff))
    status(solid,'faceted thread')
    COUNTS['filaments']+=1
    return mapped(solid,resolution=None)

def add_thread(parts,points,**kwargs):
    m=thread_path(points,**kwargs)
    if m is not None:parts.append(m)

def build_open_ground():
    """Two interlaced families of structural embroidery strands.

    No cylindrical wall is generated.  The spaces between these strands
    communicate directly from the inside of the holder to the outside.
    """
    groups=[]
    pitch=CIRC/64
    zz=np.linspace(2.65,103.25,241)
    for handed in [-1,1]:
        pieces=[]
        for k in range(64):
            s=-CIRC/2+k*pitch+handed*0.79*(zz-2.65)
            s+=0.12*np.sin(TAU*(zz-2.65)/(pitch/0.79))
            centre=lambda uv,h=handed: -1.1+h*0.10*np.sin(TAU*uv[:,1]/5.23)
            add_thread(pieces,np.column_stack((s,zz)),width=0.72,depth=2.2,
                       centre=centre,step=0.52,grain=0.055)
            COUNTS['load_bearing_strands']+=1
        groups.append(merge(pieces,f'ground weave family {handed:+d}'))
    return merge(groups,'permeable embroidery ground')

def open_leaf(parts,start,tip,width=2.6):
    start=np.asarray(start,float);tip=np.asarray(tip,float)
    length=np.linalg.norm(tip-start);axis=(tip-start)/length
    normal=np.array([-axis[1],axis[0]])
    local=np.asarray(leaf(width,length).exterior.coords)
    xy=start+normal[None,:]*local[:,0,None]+axis[None,:]*(local[:,1,None]+length/2)
    add_thread(parts,xy,width=0.42,depth=2.0,centre=-0.30,step=0.36,grain=0.04)
    add_thread(parts,[start,tip],width=0.40,depth=2.0,centre=-0.20,step=0.40)
    for side in [-1,1]:
        for i,t in enumerate([0.23,0.39,0.55,0.71]):
            t+=0.022 if side>0 else 0
            edge_t=min(t+0.10,0.90)
            a=start+axis*(length*t-0.26)
            half=width/2*np.sin(np.pi*edge_t)**0.83
            b=start+axis*(length*edge_t)+side*normal*half
            add_thread(parts,[a,b],width=0.29,depth=1.85,centre=-0.16,step=0.36)
    COUNTS['open_leaves']+=1

def open_small_flower(parts,c,radius=2.30):
    c=np.asarray(c,float)
    for k in range(6):
        phi=TAU*k/6
        ax=np.array([math.cos(phi),math.sin(phi)])
        n=np.array([-ax[1],ax[0]])
        p=np.array([c+ax*.46,c+ax*1.28+n*.50,c+ax*radius,
                    c+ax*1.28-n*.50,c+ax*.46])
        add_thread(parts,p,width=0.35,depth=1.9,centre=-0.04,step=0.35)
        add_thread(parts,[c,c+ax*radius],width=0.27,depth=1.9,centre=0.0,step=0.38)
    tt=np.linspace(0,TAU,17)
    add_thread(parts,c+np.column_stack((np.cos(tt),np.sin(tt)))*.60,
               width=.37,depth=2.0,centre=.0,step=.3)
    COUNTS['small_flowers']+=1

LACE_LAYOUT = {
    'ground_and_microsection_preserved_from_revision': 3,
    'paisley_nominal_width_height_mm': [18.0,27.0],
    'rosette_nominal_diameters_mm': [16.0,19.0],
    'fan_nominal_width_height_mm': [16.0,11.5],
    'upper_flower_nominal_diameter_mm': 8.0,
    'small_flower_nominal_diameters_mm': [4.3,5.2],
    'trefoil_nominal_span_mm': 5.6,
    'vine_leaf_nominal_lengths_mm': [5.5,7.8,10.5],
    'stitch_pitch_mm': .78,
    'motif_clearance_from_filler_mm': 1.25,
    'scales_apply_to_paths_not_thread_sections': True,
    'motif_interior': 'open frames, ribs and stitches; no filled plates',
}

def bezier(a,b,c,d,n=33):
    t=np.linspace(0,1,n)[:,None]
    return (1-t)**3*np.asarray(a)+3*(1-t)**2*t*np.asarray(b)+3*(1-t)*t*t*np.asarray(c)+t**3*np.asarray(d)

def lace_leaf(parts,start,tip,width=3.0,raised=.10):
    """A pointed embroidered leaf/petal; scale the outline, keep stitch pitch."""
    start=np.asarray(start,float);tip=np.asarray(tip,float)
    length=np.linalg.norm(tip-start);axis=(tip-start)/length
    normal=np.array([-axis[1],axis[0]])
    local=np.asarray(leaf(width,length).exterior.coords)
    xy=start+normal[None,:]*local[:,0,None]+axis[None,:]*(local[:,1,None]+length/2)
    def crest(uv):
        t=np.clip((uv-start)@axis/length,0,1)
        return raised+.36*(1-np.abs(2*t-1))
    add_thread(parts,xy,width=.42,depth=2.0,centre=crest,step=.36,grain=.04)
    add_thread(parts,[start,tip],width=.40,depth=2.0,
               centre=lambda uv:crest(uv)+.10,step=.38)
    for side in [-1,1]:
        for dist in np.arange(.85,length-.48,.78):
            d=dist+(.15 if side>0 else 0)
            edge_t=min((d+.27)/length,.94)
            a=start+axis*max(.15,d-.25)
            half=width/2*np.sin(np.pi*edge_t)**.83
            b=start+axis*(length*edge_t)+side*normal*half
            if np.linalg.norm(b-a)<.32:continue
            add_thread(parts,[a,b],width=.29,depth=1.85,
                       centre=lambda uv:crest(uv)+.14,step=.36,grain=.025)
            COUNTS['lace_petal_stitches']+=1
    COUNTS['open_leaves']+=1

def lace_rosette(parts,c,radius,petals=8,phase=0.0,small=False):
    c=np.asarray(c,float)
    for k in range(petals):
        phi=TAU*(k/ petals)+phase
        axis=np.array([math.cos(phi),math.sin(phi)])
        r=radius*(1+.025*math.cos(phi*3))
        lace_leaf(parts,c+axis*(radius*.16),c+axis*r,
                  width=radius*TAU/petals*.48,raised=.14)
    tt=np.linspace(0,TAU,41)
    for rr in [radius*.13,radius*.21]:
        add_thread(parts,c+np.column_stack((np.cos(tt),np.sin(tt)))*rr,
                   width=.35,depth=2.0,centre=.25,step=.32)
    for k in range(petals):
        ph=TAU*k/petals+phase
        ax=np.array([math.cos(ph),math.sin(ph)])
        add_thread(parts,[c,c+ax*(radius*.27)],width=.27,depth=1.9,
                   centre=.25,step=.32)
    COUNTS['small_flowers' if small else 'lace_rosettes']+=1

def paisley_outline(c,width=18.0,height=27.0,hand=1,angle=0.0):
    paths=[bezier((0,-.5),(-.64,-.48),(-.66,0),(-.27,.26)),
           bezier((-.27,.26),(-.03,.44),(.49,.64),(.45,.21)),
           bezier((.45,.21),(.24,.48),(.18,.17),(.32,-.08)),
           bezier((.32,-.08),(.53,-.36),(.28,-.54),(0,-.5))]
    xy=np.vstack([p[:-1] for p in paths])
    xy=(xy-(xy.min(axis=0)+xy.max(axis=0))/2)/np.ptp(xy,axis=0)
    xy*=np.array([width*hand,height])
    co=math.cos(angle);si=math.sin(angle)
    xy=xy@np.array([[co,si],[-si,co]])+np.asarray(c)
    p=Polygon(xy)
    if not p.is_valid:raise ValueError('Invalid paisley outline')
    return p

def stitched_border(parts,poly,offset=.88,raised=.08):
    inner=poly.buffer(-offset,join_style=2)
    if inner.geom_type!='Polygon':raise ValueError('Split stitched border')
    outer_line=poly.exterior;inner_line=inner.exterior
    for ln,w in [(outer_line,.42),(inner_line,.35)]:
        add_thread(parts,np.asarray(ln.coords),width=w,depth=2.0,
                   centre=raised,step=.34,grain=.04)
    for distance in np.arange(.2,outer_line.length,.78):
        a=outer_line.interpolate(distance)
        b=inner_line.interpolate(inner_line.project(a))
        if a.distance(b)>1.8:continue
        add_thread(parts,[[a.x,a.y],[b.x,b.y]],width=.27,depth=1.85,
                   centre=raised+.08,step=.32,grain=.025)
        COUNTS['border_stitches']+=1

def lace_paisley(parts,c,hand=1,angle=0.0):
    c=np.asarray(c,float)
    poly=paisley_outline(c,hand=hand,angle=angle)
    stitched_border(parts,poly,raised=.14)
    co=math.cos(angle);si=math.sin(angle)
    rot=np.array([[co,si],[-si,co]])
    def world(xy):return np.asarray(xy)*np.array([hand,1])@rot+c
    spine=bezier((-1,-11),(-6,-1),(1,9),(5.3,9.2),61)
    for delta in [-.19,.19]:
        add_thread(parts,world(spine+np.array([delta,0])),width=.40,
                   depth=2.0,centre=.27+delta*.30,step=.36)
    # A miniature spray follows the curl instead of filling the teardrop.
    for t,side in [(0.18,-1),(0.29,1),(0.40,-1),(0.51,1),(.63,-1),(.73,1)]:
        p=spine[int(t*(len(spine)-1))]
        a=world(p);b=world(p+np.array([side*3.6,3.4]))
        if poly.buffer(-.60).covers(LineString([a,b])):
            lace_leaf(parts,a,b,width=2.5,raised=.19)
    lace_rosette(parts,world((2.2,-5.0)),2.45,petals=5,phase=.25,small=True)
    COUNTS['paisley_motifs']+=1
    return poly

def lace_fan(parts,c,width=16.0,height=11.5,hand=1):
    c=np.asarray(c,float);base=c+np.array([0,-height/2])
    tt=np.linspace(0,1,101)
    def arc(t,scale=1.0):
        t=np.asarray(t)
        x=width*(t-.5)
        z=height*(.52+.48*np.sin(np.pi*t))+.42*np.sin(7*np.pi*t)**2
        return base+np.column_stack((x*hand,z))*scale
    # Twin scalloped arcs with close stitches and seven radiating ribs.
    for scale in [1.0,.915]:
        add_thread(parts,arc(tt,scale),width=.42 if scale==1 else .35,
                   depth=2.0,centre=.15,step=.35,grain=.04)
    for t in np.linspace(0,1,8):
        tip=arc([t])[0]
        add_thread(parts,[base,tip],width=.40,depth=2.0,centre=.20,step=.38)
    add_thread(parts,[arc([0])[0],base,arc([1])[0]],width=.42,
               depth=2.0,centre=.15,step=.35)
    line=LineString(arc(tt))
    for d in np.arange(.1,line.length,.78):
        p=np.array(line.interpolate(d).coords[0]);q=base+(p-base)*.915
        add_thread(parts,[p,q],width=.27,depth=1.85,centre=.23,step=.3)
        COUNTS['border_stitches']+=1
    for scale in [.35,.51,.67,.81]:
        add_thread(parts,arc(tt,scale),width=.29,depth=1.85,centre=.22,step=.35)
    COUNTS['lace_fans']+=1

def trefoil(parts,c,angle=0.0):
    c=np.asarray(c,float)
    for ph in [math.pi/2,math.pi/2+2.18,math.pi/2-2.18]:
        ph+=angle;ax=np.array([math.cos(ph),math.sin(ph)])
        n=np.array([-ax[1],ax[0]])
        p=[c,c+ax*1.6+n*.70,c+ax*2.8,c+ax*1.6-n*.70,c]
        add_thread(parts,p,width=.35,depth=1.9,centre=.02,step=.34)
        add_thread(parts,[c,c+ax*2.8],width=.27,depth=1.85,centre=.10,step=.34)
    COUNTS['trefoil_sprigs']+=1

def line_parts(g):
    if g.is_empty:return []
    if g.geom_type=='LineString':return [g]
    return [p for q in g.geoms for p in line_parts(q)]

def build_botanical_fill():
    """Large, medium and tiny open motifs on the unchanged fine ground."""
    groups=[];reserved=[];pitch=CIRC/6
    # Three larger motifs occupy the gaps between the original main flowers.
    gaps=[-5*CIRC/12,-CIRC/12,CIRC/4]
    for k,(s,z) in enumerate(zip(gaps,[47.5,54.0,51.0])):
        parts=[];hand=1 if k!=1 else -1;angle=[-.12,.15,-.08][k]
        poly=lace_paisley(parts,[s,z],hand,angle)
        reserved.append(poly.buffer(1.25))
        groups.append(merge(parts,f'curled paisley {k+1}/3'))
    for k,(s,z) in enumerate(zip(gaps,[79.0,82.0,80.5])):
        parts=[];lace_rosette(parts,[s,z],9.5,10,phase=.16+k*.17)
        reserved.append(Point(s,z).buffer(10.75))
        groups.append(merge(parts,f'medium rosette {k+1}/3'))
    for k in range(6):
        s=-CIRC/2+(k+.5)*pitch;parts=[]
        if k%2==0:
            z=16.8+1.0*math.sin(k)
            lace_rosette(parts,[s,z],8.0,8,phase=.25)
            reserved.append(Point(s,z).buffer(9.25))
        else:
            z=16.5+.9*math.sin(k)
            lace_fan(parts,[s,z],hand=1 if k!=3 else -1)
            reserved.append(box(s-9.25,z-7.0,s+9.25,z+7.6))
        upper_z=95.0+.30*math.sin(k)
        lace_rosette(parts,[s,upper_z],4.0,6,phase=k*.21,small=True)
        reserved.append(Point(s,upper_z).buffer(5.1))
        groups.append(merge(parts,f'alternating fan and blossom panel {k+1}/6'))
    for th,cz,r,*_ in FLOWERS:
        reserved.append(Point(th*R_REF,cz).buffer(r*.90+1.25))
    mask=unary_union(reserved)
    mask=unary_union([affinity.translate(mask,xoff=x) for x in [-CIRC,0,CIRC]])
    # Stems flow around the motifs; sizes change without thickening the thread.
    for k in range(12):
        anchor=-CIRC/2+(k+.5)*CIRC/12;phase=(k%3)*TAU/3
        def stem(z):return anchor+4.3*np.sin((np.asarray(z)-5)*TAU/61+phase)
        zz=np.linspace(3.4,101.2,301);parts=[]
        for hand in [-1,1]:
            s=stem(zz)+hand*.18*np.sin(TAU*zz/3.0)
            path=LineString(np.column_stack((s,zz))).difference(mask)
            for line in line_parts(path):
                if line.length>.75:
                    add_thread(parts,np.asarray(line.coords),width=.43,depth=2.1,
                               centre=-.30+hand*.12,step=.40)
        COUNTS['botanical_stems']+=1
        for row,z in enumerate(np.arange(5.9,98,6.7)):
            for side in [-1,1]:
                node_z=z+(.80 if side>0 else 0)
                a=np.array([float(stem(node_z)),node_z])
                scale=(row+k+(side>0))%3
                length=[5.5,7.8,10.5][scale];w=[2.35,3.25,4.3][scale]
                direction=np.array([side*(.78+.08*math.sin(row)),.65])
                b=a+length*direction/np.linalg.norm(direction)
                if b[1]>101.0 or mask.intersects(LineString([a,b]).buffer(w*.48)):
                    continue
                lace_leaf(parts,a,b,width=w,raised=-.25)
        if parts:groups.append(merge(parts,f'varied fern vine {k+1}/12'))
    # Tiny blossoms and trefoils weave through the remaining interstices.
    for row,z in enumerate(np.arange(7.5,98.0,11.4)):
        parts=[]
        for k in range(12):
            s=-CIRC/2+(k+.15+.40*(row%2))*CIRC/12
            c=np.array([s,z+.65*math.sin(k*1.7)])
            if mask.intersects(Point(c).buffer(3.25)):continue
            if (k+row)%3==0:trefoil(parts,c,angle=.20*math.sin(k))
            else:open_small_flower(parts,c,radius=2.15 if k%2 else 2.60)
        if parts:groups.append(merge(parts,f'scattered small lace motifs {row+1}'))
    # Stitched scallops connect the lower and upper motif fields.
    for level,amp,phase in [(28.5,2.8,.15),(90.0,1.8,.6)]:
        parts=[];ss=np.linspace(-CIRC/2,CIRC/2,1025)
        def zz(s):return level+amp*np.cos(TAU*np.asarray(s)*6/CIRC+phase)
        for delta in [-.42,.42]:
            add_thread(parts,np.column_stack((ss,zz(ss)+delta)),width=.43,
                       depth=2.0,centre=-.06,step=.36,grain=.04)
        path=LineString(np.column_stack((ss,zz(ss))))
        for d in np.arange(.25,path.length,.78):
            p=np.array(path.interpolate(d).coords[0])
            add_thread(parts,[p+[0,-.42],p+[.15,.42]],width=.27,
                       depth=1.85,centre=.04,step=.30,grain=.025)
            COUNTS['border_stitches']+=1
        COUNTS['stitched_garlands']+=1
        groups.append(merge(parts,f'scalloped embroidered garland at {level} mm'))
    return merge(groups,'multiscale open embroidery composition')
def fan_height(plan):
    xy,apex,edge,peak=plan
    def value(query):
        q=np.asarray(query,float)
        result=np.full(len(q),edge,float)
        best=np.full(len(q),-1e9)
        for i in range(len(xy)):
            a=xy[i];b=xy[(i+1)%len(xy)]
            v0=b-a;v1=apex-a
            det=v0[0]*v1[1]-v0[1]*v1[0]
            if abs(det)<1e-12:continue
            d=q-a
            u=(d[:,0]*v1[1]-d[:,1]*v1[0])/det
            v=(v0[0]*d[:,1]-v0[1]*d[:,0])/det
            margin=np.minimum(np.minimum(u,v),1-u-v)
            take=margin>best
            result[take]=edge+(peak-edge)*np.clip(v[take],0,1)
            best[take]=margin[take]
        return result
    return value

def open_main_petal(plan):
    xy,apex,edge,peak=plan
    poly=Polygon(xy)
    start=(xy[0]+xy[-1])/2;tip=(xy[3]+xy[4])/2
    direction=(tip-start)/np.linalg.norm(tip-start)
    normal=np.array([-direction[1],direction[0]])
    length=np.linalg.norm(tip-start)
    surf=fan_height(plan)
    centre=lambda uv:surf(uv)-.26
    pieces=[]
    add_thread(pieces,np.vstack((xy,xy[0])),width=.62,depth=1.40,
               centre=centre,step=.45,grain=.045)
    add_thread(pieces,[start,apex,tip],width=.59,depth=1.38,
               centre=centre,step=.42,grain=.04)
    # Herringbone satin stitches span the actual opening from rim to ridge.
    for k,dist in enumerate(np.arange(1.15,length-.35,.78)):
        for side in [-1,1]:
            d=dist+(.18 if side>0 else 0)
            c=start+direction*d
            cut=LineString([c-normal*7,c+normal*7]).intersection(poly)
            if cut.geom_type!='LineString' or cut.length<.70:continue
            endpoints=np.asarray(cut.coords)
            projections=(endpoints-c)@normal
            b=endpoints[np.argmax(projections) if side>0 else np.argmin(projections)]
            a=start+direction*max(.2,d-.38)
            if np.linalg.norm(b-a)<.38:continue
            add_thread(pieces,[a,b],width=.285,depth=1.13,
                       centre=lambda uv:surf(uv)-.12,step=.40,grain=.025)
            COUNTS['petal_stitches']+=1
    COUNTS['open_main_petals']+=1
    return merge(pieces,f'open petal {COUNTS["open_main_petals"]}/150')

def build_main_flowers():
    petals=[open_main_petal(p) for p in PETAL_PLANS]
    hubs=[]
    for th,cz,r,*_ in FLOWERS:
        c=np.array([th*R_REF,cz])
        radius=r*.138
        def top(uv,c=c):
            rr=np.linalg.norm(uv-c,axis=1)
            return 5.12-.10*rr+.14*triangle_wave(rr/.77)
        centre=lambda uv,top=top:(top(uv)-1.7)/2
        dep=lambda uv,top=top:top(uv)+1.7
        for rr in np.arange(.70,radius+.2,.72):
            tt=np.linspace(0,TAU,49)
            add_thread(hubs,c+np.column_stack((np.cos(tt),np.sin(tt)))*rr,
                       width=.40,centre=centre,depth_fn=dep,step=.34)
        for k in range(18):
            phi=TAU*k/18
            ax=np.array([math.cos(phi),math.sin(phi)])
            add_thread(hubs,[c,c+ax*radius],width=.38,centre=centre,
                       depth_fn=dep,step=.34)
    return merge(petals+[merge(hubs,'embroidered flower centres')],
                 'three folded openwork flowers')

RIM_DETAILS={
    'style':'cupped petal scallops, raised cutwork windows and contour-following picots',
    'petal_count':24,'nominal_petal_width_mm':CIRC/24,
    'nominal_petal_height_mm':11.0,'peak_height_variation_mm':.45,
    'radial_flare_mm':2.05,'tip_return_mm':.35,
    'main_thread_width_mm':[.40,.48],'fine_thread_width_mm':[.27,.34],
    'nominal_picot_outer_width_mm':1.30,'nominal_picot_projection_mm':1.2,
    'picot_thread_width_mm':.30,'picot_arc_spacing_mm':2.05,
    'nominal_border_stitch_pitch_mm':.58,'window_outline_pitch_mm':.62,
    'framed_cutwork_windows_per_petal':3,'continuous_backing':False,
    'veins_and_bridges_raised_mm':.38,
    'picots_follow_local_contour_normals':True,
}

def rim_stitched_outline(parts,poly,centre,offset=.60,outer_width=.40,
                         inner_width=.30,depth=1.20,pitch=.58):
    """Sewn edges around an actual open window, without a backing sheet."""
    inset=poly.buffer(-offset,join_style=1)
    if inset.geom_type!='Polygon' or inset.is_empty:
        raise ValueError('Cutwork outline is too narrow for paired stitching')
    outer=poly.exterior;inner=inset.exterior
    for line,width in [(outer,outer_width),(inner,inner_width)]:
        add_thread(parts,np.asarray(line.coords),width=width,depth=depth,
                   centre=centre,step=.27,grain=.04)
    for dist in np.arange(.15,outer.length,pitch):
        a=outer.interpolate(dist);b=inner.interpolate(inner.project(a))
        if a.distance(b)>1.30:continue
        add_thread(parts,[[a.x,a.y],[b.x,b.y]],width=.27,depth=depth-.10,
                   centre=lambda uv,c=centre:c(uv)+.035,step=.25,grain=.025)
        COUNTS['rim_stitches']+=1

def rim_window(parts,start,tip,width,centre):
    start=np.asarray(start,float);tip=np.asarray(tip,float)
    length=np.linalg.norm(tip-start);axis=(tip-start)/length
    normal=np.array([-axis[1],axis[0]])
    shape=np.asarray(leaf(width,length).exterior.coords)
    points=start+normal[None,:]*shape[:,0,None]+axis[None,:]*(shape[:,1,None]+length/2)
    rim_stitched_outline(parts,Polygon(points),centre,offset=.44,
                         outer_width=.34,inner_width=.28,depth=1.12,pitch=.62)
    COUNTS['rim_cutwork_windows']+=1

def rim_petal(k):
    pitch=CIRC/24;half=pitch/2;s=-CIRC/2+(k+.5)*pitch
    z0=98.90+.14*math.cos(TAU*k/3)
    high=7.80+.45*math.cos(math.pi*k)
    u=np.linspace(-1,1,161)
    z=102.0+high*np.maximum(0,1-u*u)**.58+.32*(1-u*u)*np.cos(4*math.pi*u)
    top=np.column_stack((s+half*u,z))
    left=bezier((s-.72,z0),(s-2.9,z0+.65),(s-half,101.30),top[0],33)
    right=bezier(top[-1],(s+half,101.30),(s+2.9,z0+.65),(s+.72,z0),33)
    outline=np.vstack((left[:-1],top[:-1],right))
    poly=Polygon(outline)
    if not poly.is_valid:raise ValueError('Invalid petal scallop')
    top_z=float(z.max())
    def surface(uv):
        t=np.clip((uv[:,1]-z0)/(top_z-z0),0,1)
        x=np.clip((uv[:,0]-s)/half,-1,1)
        flare=2.05*(3*t*t-2*t*t*t)-.35*np.exp(-((t-1)/.13)**2)
        cup=.46*(1-x*x)*np.sin(math.pi*t)
        twist=.19*((-1)**k)*x*np.sin(math.pi*t)
        return -.55+flare+cup+twist
    def raised(uv):
        t=np.clip((uv[:,1]-z0)/(top_z-z0),0,1)
        return surface(uv)+.38*np.sin(math.pi*t)
    parts=[]
    rim_stitched_outline(parts,poly,surface)
    # Three stitched teardrops and a lifted central vein create open cutwork.
    for a,b,w in [((s-.10,z0+2.0),(s+.05,top_z-1.65),3.0),
                  ((s-.42,z0+1.45),(s-3.65,z0+6.35),2.15),
                  ((s+.42,z0+1.45),(s+3.65,z0+6.35),2.15)]:
        rim_window(parts,a,b,w,raised)
    spine=bezier((s,z0-.40),(s-.10,z0+3.2),(s+.15,top_z-2.3),(s,top_z),65)
    add_thread(parts,spine,width=.44,depth=1.42,centre=raised,step=.30,grain=.04)
    for side in [-1,1]:
        branch=bezier((s,z0+.45),(s+side*1.35,z0+2.9),
                      (s+side*2.75,z0+4.7),(s+side*3.65,z0+6.35),49)
        add_thread(parts,branch,width=.40,depth=1.35,centre=raised,step=.30)
        # Each window joins the outside embroidery with several separate bridges.
        targets=[((s+side*2.6,z0+4.8),(s+side*4.80,z0+5.35)),
                 ((s+side*3.65,z0+6.35),(s+side*4.1,z0+7.25)),
                 ((s+side*1.22,z0+7.25),(s+side*2.30,z0+9.10))]
        for a,b in targets:
            # Continue the bridge until it intersects the actual curved border.
            line=LineString([a,b])
            d=np.asarray(b)-np.asarray(a)
            ray=LineString([a,np.asarray(a)+d*2.5])
            section=ray.intersection(poly)
            if section.geom_type!='LineString' or section.length<.2:continue
            path=np.asarray(section.coords)
            add_thread(parts,path,width=.30,depth=1.20,centre=raised,step=.28)
            COUNTS['rim_bridges']+=1
    # Picots are small U-shaped stitches with two feet, oriented to the edge.
    edge=LineString(top)
    for j,dist in enumerate(np.arange(1.55,edge.length-1.55,2.05)):
        c=np.array(edge.interpolate(dist).coords[0])
        pa=np.array(edge.interpolate(max(0,dist-.06)).coords[0])
        pb=np.array(edge.interpolate(min(edge.length,dist+.06)).coords[0])
        tangent=(pb-pa)/np.linalg.norm(pb-pa)
        outward=np.array([-tangent[1],tangent[0]])
        tt=np.linspace(0,math.pi,31)
        height=1.03 if j%3 else 1.12
        points=c+(.50*np.cos(tt))[:,None]*tangent+(height*np.sin(tt)-.12)[:,None]*outward
        add_thread(parts,points,width=.30,depth=1.08,centre=surface,step=.20,grain=.035)
        COUNTS['picot_loops']+=1
    COUNTS['rim_petals']+=1
    return merge(parts,f'cupped cutwork petal {k+1}/24')

def build_open_frill():
    """Petal-shaped cutwork lace with tiny picots following its scalloped edge."""
    parts=[];ss=np.linspace(-CIRC/2,CIRC/2,1025)
    def wave(s):return .12*np.cos(TAU*np.asarray(s)*24/CIRC)
    for level in [98.55,99.25]:
        add_thread(parts,np.column_stack((ss,level+wave(ss))),width=.43,
                   depth=1.60,centre=-.46,step=.32,grain=.045)
    for s in np.arange(-CIRC/2,CIRC/2,.60):
        add_thread(parts,[[s,98.55+wave(s)],[s+.14,99.25+wave(s+.14)]],
                   width=.27,depth=1.40,centre=-.39,step=.25,grain=.025)
        COUNTS['rim_stitches']+=1
    # Short brides across the valleys give adjacent petals finite shared roots.
    for k in range(24):
        s=-CIRC/2+k*CIRC/24
        add_thread(parts,[[s-.50,101.78],[s+.50,101.78]],width=.40,
                   depth=1.50,centre=-.12,step=.25,grain=.035)
        COUNTS['rim_bridges']+=1
    seam=merge(parts,'fine stitched root and valley bridges')
    return merge([seam]+[rim_petal(k) for k in range(24)],'petal cutwork and fine picot frill')


BASE_DETAILS = {
    "pattern": "continuous diagonal diamond facets on both faces and outside edge",
    "planar_pitch_mm": 0.90,
    "top_relief_mm": 0.35,
    "underside_relief_mm": 0.32,
    "edge_relief_mm": 0.32,
    "edge_nominal_pitch_mm": 0.90,
    "z_min_mm": 0.0,
    "top_z_relative_to_bottom_mm": [0.0, 0.35],
    "minimum_vertical_solid_thickness_mm_at_default_bottom": 2.88,
    "solid_floor": True,
    "large_untextured_flat_regions": False,
}


def _clipped_diamond_grid(radius, pitch=0.90, circle_segments=720):
    """Triangulate an angular heightfield inside a regular circular polygon.

    A vertex's third coordinate is the pyramid height in [0, 1]. Clipping
    preserves this coordinate by linear interpolation, so neighbouring
    triangles share both position and height even on the circular edge.
    """
    angles = 2.0 * math.pi * (np.arange(circle_segments) + 0.5) / circle_segments
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    apothem = radius * math.cos(math.pi / circle_segments)
    vertex_index = {}
    points = []
    faces = []

    def vertex_id(point):
        key = tuple(np.round(point[:2], 9))
        idx = vertex_index.get(key)
        if idx is None:
            idx = len(points)
            vertex_index[key] = idx
            points.append(point)
        elif abs(points[idx][2] - point[2]) > 2.0e-7:
            raise RuntimeError("Diamond height discontinuity at a shared edge")
        return idx

    def emit(poly):
        ids = [vertex_id(v) for v in poly]
        for j in range(1, len(ids) - 1):
            if len({ids[0], ids[j], ids[j + 1]}) == 3:
                faces.append([ids[0], ids[j], ids[j + 1]])

    def clip(tri):
        # Only planes that cut the original triangle need to be processed.
        candidates = np.flatnonzero(np.max(tri[:, :2] @ normals.T, axis=0) > apothem)
        poly = tri
        for ni in candidates:
            normal = normals[ni]
            distances = poly[:, :2] @ normal - apothem
            inside = distances <= 1.0e-12
            if inside.all():
                continue
            if not inside.any():
                return
            clipped = []
            for j in range(len(poly)):
                previous = (j - 1) % len(poly)
                if inside[j] != inside[previous]:
                    alpha = distances[previous] / (distances[previous] - distances[j])
                    clipped.append(poly[previous] + alpha * (poly[j] - poly[previous]))
                if inside[j]:
                    clipped.append(poly[j])
            poly = np.asarray(clipped)
        emit(poly)

    first = math.floor(-radius / pitch)
    last = math.ceil(radius / pitch)
    for iy in range(first, last):
        y0, y1 = iy * pitch, (iy + 1) * pitch
        for ix in range(first, last):
            x0, x1 = ix * pitch, (ix + 1) * pitch
            nearest_x = max(x0, 0.0, -x1)
            nearest_y = max(y0, 0.0, -y1)
            if nearest_x ** 2 + nearest_y ** 2 > radius ** 2:
                continue
            corners = np.array([[x0, y0, 0.0], [x1, y0, 0.0],
                                [x1, y1, 0.0], [x0, y1, 0.0]])
            center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, 1.0])
            entirely_inside = np.max(np.linalg.norm(corners[:, :2], axis=1)) < apothem
            for j in range(4):
                tri = np.vstack((corners[j], corners[(j + 1) % 4], center))
                if entirely_inside:
                    emit(tri)
                else:
                    clip(tri)

    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.uint32)
    # Every single-use edge belongs to the one convex circular boundary.
    directed_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    undirected_edges = np.sort(directed_edges, axis=1)
    _, inverse, counts = np.unique(undirected_edges, axis=0, return_inverse=True, return_counts=True)
    single_edges = directed_edges[counts[inverse] == 1]
    boundary = np.unique(single_edges)
    theta = np.arctan2(points[boundary, 1], points[boundary, 0])
    boundary = boundary[np.argsort(theta)]
    # Assert that angle sorting has retained the actual tessellation boundary.
    actual = {tuple(sorted(e)) for e in single_edges.tolist()}
    ordered = {tuple(sorted((int(a), int(b))))
               for a, b in zip(boundary, np.roll(boundary, -1))}
    if actual != ordered:
        raise RuntimeError("The textured base has an inconsistent boundary")
    return points, faces, boundary


def make_textured_base(rad_func, y_scale=0.96, bottom=3.2):
    """Return one closed textured floor, suitable for union with side strands.

    ``rad_func(z)`` supplies the original vessel's outer radius at height z.
    The floor occupies the entire interior and reaches this radius even at
    its edge valleys. Side strands starting below z=bottom will overlap it.
    """
    if bottom <= 0.80:
        raise ValueError("The textured floor needs bottom > 0.80 mm")
    if y_scale <= 0.0:
        raise ValueError("y_scale must be positive")
    top_relief = BASE_DETAILS["top_relief_mm"]
    underside_relief = BASE_DETAILS["underside_relief_mm"]
    radius = float(rad_func(bottom))
    points, disk_faces, boundary = _clipped_diamond_grid(radius)
    n = len(points)
    heights = np.clip(points[:, 2], 0.0, 1.0)
    bottom_z = underside_relief * (1.0 - heights)
    top_z = bottom + top_relief * heights

    # A 45 degree grid gives diagonal stitches relative to the vessel axes.
    angle = math.pi / 4.0
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    xy_unit = points[:, :2] @ rotation.T / radius

    def radii(z):
        try:
            result = np.asarray(rad_func(z), dtype=np.float64)
            if result.shape == np.shape(z):
                return result
        except (TypeError, ValueError):
            pass
        return np.asarray([float(rad_func(float(v))) for v in np.ravel(z)]).reshape(np.shape(z))

    def face_vertices(z):
        xy = xy_unit * radii(z)[:, None]
        xy[:, 1] *= y_scale
        return np.column_stack((xy, z))

    vertex_groups = [face_vertices(bottom_z), face_vertices(top_z)]
    face_groups = [disk_faces[:, ::-1], disk_faces + n]
    nb = len(boundary)
    t_bottom = bottom_z[boundary]
    t_top = top_z[boundary]
    edge_unit = xy_unit[boundary]
    edge_radius = np.linalg.norm(edge_unit, axis=1)
    edge_direction = edge_unit / edge_radius[:, None]
    theta = np.arctan2(edge_direction[:, 1], edge_direction[:, 0])
    nominal_edge_radius = (float(rad_func(0.0)) + float(rad_func(bottom))) * 0.5
    periods = max(12, round(2.0 * math.pi * nominal_edge_radius / 0.90))
    ring_ids = [boundary.astype(np.uint32)]
    offset = 2 * n
    ring_steps = 16

    def triangle_wave(phase):
        return 1.0 - np.abs(2.0 * np.mod(phase, 1.0) - 1.0)

    for j in range(1, ring_steps):
        t = j / ring_steps
        z = t_bottom * (1.0 - t) + t_top * t
        # Diamond roof facets continue around the edge. Their amplitude
        # closes continuously onto the already textured face boundaries.
        around = triangle_wave(theta * periods / (2.0 * math.pi))
        vertical = triangle_wave(z / 0.90 + 0.5)
        corrugation = (BASE_DETAILS["edge_relief_mm"] * math.sin(math.pi * t)
                       * np.minimum(around, vertical))
        r = edge_radius * radii(z) + corrugation
        xy = edge_direction * r[:, None]
        xy[:, 1] *= y_scale
        vertex_groups.append(np.column_stack((xy, z)))
        ring_ids.append(np.arange(offset, offset + nb, dtype=np.uint32))
        offset += nb
    ring_ids.append((boundary + n).astype(np.uint32))
    for lower, upper in zip(ring_ids[:-1], ring_ids[1:]):
        ln, un = np.roll(lower, -1), np.roll(upper, -1)
        face_groups.extend((np.column_stack((lower, ln, un)),
                            np.column_stack((lower, un, upper))))

    vertices = np.vstack(vertex_groups).astype(np.float32)
    faces = np.vstack(face_groups).astype(np.uint32)
    # Deduplicate at the exact coordinate precision used by binary STL.
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces].astype(np.uint32)
    valid = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
             & (faces[:, 2] != faces[:, 0]))
    faces = faces[valid]
    result = mf.Manifold(mf.Mesh(vertices, faces))
    if result.status() != mf.Error.NoError or result.is_empty():
        raise RuntimeError(f"Textured floor mesh failed: {result.status()}")
    if result.volume() <= 0:
        raise RuntimeError("Textured floor has non-positive volume")
    return result

def make_3mf(tm,path):
    # Standard core 3MF, millimetres, one object, no printer-specific settings.
    import xml.etree.ElementTree as ET
    NS='http://schemas.microsoft.com/3dmanufacturing/core/2015/02'
    ET.register_namespace('',NS)
    model=ET.Element(f'{{{NS}}}model',{'unit':'millimeter','xml:lang':'en-US'})
    ET.SubElement(model,f'{{{NS}}}metadata',{'name':'Title'}).text='Petal Cutwork Floral Embroidery Pen Holder v5'
    ET.SubElement(model,f'{{{NS}}}metadata',{'name':'Description'}).text='One-piece original parametric model. Units: mm.'
    res=ET.SubElement(model,f'{{{NS}}}resources')
    obj=ET.SubElement(res,f'{{{NS}}}object',{'id':'1','type':'model','name':'Openwork embroidery pen holder'})
    mesh=ET.SubElement(obj,f'{{{NS}}}mesh')
    vertices=ET.SubElement(mesh,f'{{{NS}}}vertices')
    for x,y,z in tm.vertices:
        ET.SubElement(vertices,f'{{{NS}}}vertex',{'x':f'{x:.17g}','y':f'{y:.17g}','z':f'{z:.17g}'})
    triangles=ET.SubElement(mesh,f'{{{NS}}}triangles')
    for a,b,c in tm.faces:
        ET.SubElement(triangles,f'{{{NS}}}triangle',{'v1':str(a),'v2':str(b),'v3':str(c)})
    build=ET.SubElement(model,f'{{{NS}}}build')
    ET.SubElement(build,f'{{{NS}}}item',{'objectid':'1'})
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('[Content_Types].xml','<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr('_rels/.rels','<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr('3D/3dmodel.model',ET.tostring(model,encoding='utf-8',xml_declaration=True))


def quantized_mesh(solid):
    mm=solid.to_mesh64()
    tm=trimesh.Trimesh(np.asarray(mm.vert_properties,dtype=np.float32)[:,:3],
                       np.asarray(mm.tri_verts),process=True)
    f=tm.faces
    tm.update_faces((f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2]))
    tm.remove_unreferenced_vertices()
    # Keep inward-facing boundaries of closed microcavities distinguishable
    # from detached positive solids. They are filled explicitly below.
    tm.fix_normals(multibody=False)
    return tm

def close_near_contact_edges(solid):
    """Give numerically coincident embroidery contacts a finite local bond.

    This adds short octagonal threads only at non-manifold welded edges.
    It does not delete ornaments or globally remesh the small openings.
    """
    details=[]
    for iteration,radius in enumerate([.045,.060,.080,.100]):
        tm=quantized_mesh(solid)
        incidence=np.bincount(tm.edges_unique_inverse)
        bad=tm.edges_unique[incidence!=2]
        if len(bad)==0:return solid,tm,details
        boundary=int((incidence==1).sum())
        if boundary:raise ValueError(f'Unexpected open boundary: {boundary} edges')
        log(f'Finite contact bonds, pass {iteration+1}: {len(bad)} coincident edges')
        pieces=[]
        for e in bad:
            a,b=tm.vertices[e]
            axis=b-a;length=np.linalg.norm(axis)
            if length<1e-10:continue
            axis=axis/length
            seed=np.array([1.,0.,0.]) if abs(axis[0])<.8 else np.array([0.,1.,0.])
            u=np.cross(seed,axis);u/=np.linalg.norm(u);v=np.cross(axis,u)
            transform=np.column_stack((u,v,axis,(a+b)/2))
            bond=mf.Manifold.cylinder(length+.20,radius,radius,8,center=True)
            pieces.append(bond.transform(transform))
        solid=merge([solid]+pieces,f'finite embroidery contacts {iteration+1}')
        # Preserve the new joint faces; a second broad simplification could
        # collapse the same tiny contact back to an edge.
        details.append({'pass':iteration+1,'edge_count':len(bad),'bond_radius_mm':radius,
                        'endpoint_extension_mm':.10,
                        'edge_endpoints_mm':np.asarray(tm.vertices)[bad].tolist()})
    tm=quantized_mesh(solid)
    return solid,tm,details

def numerical_residue_info(component):
    """Identify only tiny zero-volume or sub-coordinate-precision remnants.

    A four-vertex tetrahedron can survive float32 welding as a very thin
    positive shell after a Boolean contact collapses.  Its lateral area alone
    is not a useful thickness test.  Require a tiny volume and footprint,
    exactly four triangular faces, and thickness no larger than the normal
    projection of the two endpoints' float32 rounding envelopes.
    """
    volume=abs(float(component.volume));area=float(component.area)
    if volume>=1e-7:return None
    if area<1e-3:
        return {'kind':'zero_volume_and_area','volume_mm3':volume,'area_mm2':area}
    if len(component.faces)!=4 or len(component.vertices)!=4:return None
    vv=np.asarray(component.vertices,float)
    diameter=float(np.max(np.linalg.norm(vv[:,None]-vv[None,:],axis=2)))
    if diameter>=.15 or area>=.02:return None
    centred=vv-vv.mean(axis=0)
    _,_,axes=np.linalg.svd(centred,full_matrices=False)
    normal=axes[-1]
    thickness=float(np.ptp(centred@normal))
    spacing=np.abs(np.spacing(vv.astype(np.float32)).astype(float))
    half_rounding=float(np.max(spacing@np.abs(normal))*.5)
    if thickness>2*half_rounding:return None
    return {'kind':'sub_float32_precision_tetrahedron','face_count':4,
            'volume_mm3':volume,'area_mm2':area,'diameter_mm':diameter,
            'best_fit_plane_full_thickness_mm':thickness,
            'normal_projected_half_rounding_envelope_mm':half_rounding,
            'bounds_mm':component.bounds.tolist()}

def point_contact_vertices(tm):
    """Find vertices with multiple local surface fans after STL-style welding."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    faces=np.asarray(tm.faces,dtype=np.int32)
    origin=faces.ravel();dest=faces[:,[1,2,0]].ravel()
    nv=len(tm.vertices);nh=len(origin)
    key=np.minimum(origin,dest).astype(np.int64)*nv+np.maximum(origin,dest)
    order=np.argsort(key);ordered=key[order]
    if nh%2 or not np.array_equal(ordered[0::2],ordered[1::2]):
        raise ValueError('Vertex-link check requires closed edges')
    if np.any(ordered[1:-1:2]==ordered[2::2]):
        raise ValueError('Vertex-link check requires two faces per edge')
    opposite=np.empty(nh,dtype=np.int32)
    opposite[order[0::2]]=order[1::2];opposite[order[1::2]]=order[0::2]
    if not np.array_equal(origin[opposite],dest):
        raise ValueError('Vertex-link check requires consistent winding')
    successor=(opposite//3)*3+(opposite%3+1)%3
    del dest,key,order,ordered,opposite
    graph=csr_matrix((np.ones(nh,dtype=np.int8),successor,
                      np.arange(nh+1,dtype=np.int32)),shape=(nh,nh))
    fan_count,labels=connected_components(graph,directed=False)
    del graph,successor
    fan_vertex=np.empty(fan_count,dtype=np.int32)
    fan_vertex[labels]=origin
    multiplicity=np.bincount(fan_vertex,minlength=nv)
    return np.asarray(tm.vertices)[multiplicity>1]

def close_point_contacts(solid,tm):
    """Replace point-only filament contacts with tiny faceted solid joints."""
    details=[]
    for iteration,radius in enumerate([.055,.075,.100]):
        points=point_contact_vertices(tm)
        if len(points)==0:return solid,tm,details
        log(f'Finite vertex joints, pass {iteration+1}: {len(points)} point contacts')
        joints=[mf.Manifold.sphere(radius,12).translate(p) for p in points]
        solid=merge([solid]+joints,f'finite embroidery vertex joints {iteration+1}')
        # New local facets stay unsimplified. Recheck welded edges as well.
        solid,tm,edge_repairs=close_near_contact_edges(solid)
        details.append({'pass':iteration+1,'point_contact_count':len(points),
                        'faceted_joint_radius_mm':radius,'coordinates_mm':points.tolist(),
                        'subsequent_edge_repairs':edge_repairs})
    if len(point_contact_vertices(tm)):
        raise ValueError('Unresolved point-only contact in embroidery')
    return solid,tm,details

def attach_positive_slivers(solid, mesh, skip_component=None,
                            maximum_volume=1.0e-5, maximum_diameter=0.35,
                            maximum_gap=0.20, joint_radius=0.055,
                            endpoint_extension=0.08):
    """Return (unioned Manifold, attachment_records), retaining every solid.

    ``mesh`` must be the current, consistently wound, closed export mesh of
    ``solid``. A negative signed component represents an enclosed cavity and
    is left to the caller. ``skip_component`` may be the caller's unchanged
    numerical-residue predicate; skipping here never deletes that component.

    Every unskipped positive component except the main body must satisfy all
    three attachment gates. Otherwise this function raises for inspection.
    A cKDTree of the main triangle centres narrows the search; exact closest
    points on the 64 selected triangles are then evaluated for each original
    vertex, quarter-edge sample, and face centre of the fragment.
    """
    import numpy as np
    import manifold3d as mf
    import trimesh
    from scipy.spatial import cKDTree

    if not (mesh.is_watertight and mesh.is_winding_consistent):
        raise ValueError("Positive-fragment attachment requires a closed, consistently wound mesh")
    components = mesh.split(only_watertight=False, repair=False)
    positive = [c for c in components if c.volume > 0]
    if not positive:
        raise ValueError("No positive main body found")
    main = max(positive, key=lambda c: c.volume)
    candidates = []
    for component in positive:
        if component is main:
            continue
        if skip_component is not None and skip_component(component):
            continue
        vv = np.asarray(component.vertices, dtype=float)
        centred_tri = np.asarray(component.triangles) - vv.mean(axis=0)
        volume = abs(float(np.einsum(
            "ij,ij->i", centred_tri[:, 0],
            np.cross(centred_tri[:, 1], centred_tri[:, 2])).sum() / 6.0))
        if volume > maximum_volume:
            raise ValueError(f"Detached positive component needs inspection: volume {volume:.9g} mm3")
        # The box diagonal is a conservative upper bound for complex tiny
        # fragments. Compute the exact vertex diameter for the small meshes
        # handled here, without creating an unbounded pairwise matrix.
        if len(vv) <= 128:
            diameter = float(np.linalg.norm(vv[:, None] - vv[None, :], axis=2).max())
            diameter_method = "exact_vertex_pair_distance"
        else:
            diameter = float(np.linalg.norm(component.extents))
            diameter_method = "bounding_box_diagonal_upper_bound"
        if diameter > maximum_diameter:
            raise ValueError(f"Detached positive component needs inspection: diameter {diameter:.9g} mm")
        candidates.append((component, volume, diameter, diameter_method))
    if not candidates:
        return solid, []

    main_triangles = np.asarray(main.triangles)
    tree = cKDTree(main_triangles.mean(axis=1))
    neighbours = min(64, len(main_triangles))
    joints = []
    records = []
    for component, volume, diameter, diameter_method in candidates:
        vv = np.asarray(component.vertices)
        ee = np.asarray(component.edges_unique)
        endpoints = vv[ee]
        samples = [vv, np.asarray(component.triangles).mean(axis=1)]
        for t in (0.25, 0.50, 0.75):
            samples.append((1.0 - t) * endpoints[:, 0] + t * endpoints[:, 1])
        queries = np.unique(np.vstack(samples), axis=0)
        _, indices = tree.query(queries, k=neighbours)
        indices = np.asarray(indices).reshape(len(queries), neighbours)
        triangles = main_triangles[indices.ravel()]
        repeated_queries = np.repeat(queries, neighbours, axis=0)
        closest = trimesh.triangles.closest_point(triangles, repeated_queries)
        distances = np.linalg.norm(closest - repeated_queries, axis=1)
        best = int(np.argmin(distances))
        gap = float(distances[best])
        if gap > maximum_gap:
            raise ValueError(f"Detached positive component needs inspection: attachment gap {gap:.9g} mm")
        fragment_point = repeated_queries[best]
        main_point = closest[best]
        if gap < 1.0e-8:
            joint = mf.Manifold.sphere(joint_radius, 12).translate(fragment_point)
            shape = "faceted_contact_sphere"
        else:
            axis = (main_point - fragment_point) / gap
            seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            u = np.cross(seed, axis)
            u /= np.linalg.norm(u)
            w = np.cross(axis, u)
            transform = np.column_stack((u, w, axis, (fragment_point + main_point) * 0.5))
            joint = mf.Manifold.cylinder(
                gap + 2.0 * endpoint_extension, joint_radius, joint_radius,
                8, center=True).transform(transform)
            shape = "octagonal_finite_attachment"
        if joint.status() != mf.Error.NoError or joint.is_empty():
            raise ValueError("Failed to construct a finite local attachment")
        # Give the whole extremely thin fragment a finite axial backing.
        # A convex octagonal prism containing every source vertex also
        # contains all of its triangular faces. This avoids leaving a thin
        # tail detached when a Boolean clips the nearest corner of a sliver.
        spine_center = vv.mean(axis=0)
        _, _, spine_axes = np.linalg.svd(vv - spine_center, full_matrices=False)
        spine_axis = spine_axes[0]
        axial = (vv - spine_center) @ spine_axis
        perpendicular = (vv - spine_center) - axial[:, None] * spine_axis
        perpendicular_radius = float(np.linalg.norm(perpendicular, axis=1).max())
        apothem = joint_radius * np.cos(np.pi / 8.0)
        if perpendicular_radius >= apothem:
            raise ValueError("Tiny fragment exceeds the fixed axial backing radius; inspect locally")
        spine_seed = (np.array([1.0, 0.0, 0.0]) if abs(spine_axis[0]) < 0.8
                      else np.array([0.0, 1.0, 0.0]))
        spine_u = np.cross(spine_seed, spine_axis)
        spine_u /= np.linalg.norm(spine_u)
        spine_w = np.cross(spine_axis, spine_u)
        axial_midpoint = (float(axial.min()) + float(axial.max())) * 0.5
        spine_transform = np.column_stack((spine_u, spine_w, spine_axis,
                                           spine_center + axial_midpoint * spine_axis))
        spine_length = float(np.ptp(axial)) + 2.0 * endpoint_extension
        spine = mf.Manifold.cylinder(spine_length, joint_radius, joint_radius,
                                     8, center=True).transform(spine_transform)
        if spine.status() != mf.Error.NoError or spine.is_empty():
            raise ValueError("Failed to construct finite fragment backing")
        joints.extend((joint, spine))
        records.append({
            "kind": shape,
            "fragment_face_count": len(component.faces),
            "fragment_vertex_count": len(component.vertices),
            "fragment_volume_mm3": volume,
            "fragment_area_mm2": float(component.area),
            "fragment_diameter_mm": diameter,
            "diameter_measurement": diameter_method,
            "fragment_bounds_mm": component.bounds.tolist(),
            "gap_mm": gap,
            "fragment_contact_mm": fragment_point.tolist(),
            "main_contact_mm": main_point.tolist(),
            "radius_mm": joint_radius,
            "endpoint_extension_mm": endpoint_extension,
            "finite_backing": {
                "kind": "octagonal_axial_spine",
                "radius_mm": joint_radius,
                "length_mm": spine_length,
                "axis": spine_axis.tolist(),
                "center_mm": (spine_center + axial_midpoint * spine_axis).tolist(),
                "fragment_maximum_perpendicular_radius_mm": perpendicular_radius,
                "guaranteed_prism_apothem_mm": float(apothem),
                "all_fragment_vertices_contained": True,
            },
            "sample_count": len(queries),
            "nearby_triangle_count_per_sample": neighbours,
            "selection_limits": {
                "positive_volume_mm3": maximum_volume,
                "diameter_mm": maximum_diameter,
                "attachment_gap_mm": maximum_gap,
            },
        })
    joined = mf.Manifold.batch_boolean([solid] + joints, mf.OpType.Add)
    if joined.status() != mf.Error.NoError or joined.is_empty():
        raise ValueError(f"Finite positive-fragment union failed: {joined.status()}")
    return joined, records


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-stages',action='store_true',
                        help='Reuse unchanged intermediate geometry from the local cache.')
    args=parser.parse_args()
    cache=ROOT/'cache_v5'
    cache.mkdir(exist_ok=True)

    def stage(name,builder):
        path=cache/(name+'.npz')
        if args.reuse_stages and path.exists():
            log(f'Reusing stage {name}')
            data=np.load(path)
            m=mf.Manifold(mf.Mesh64(data['vertices'],data['faces']))
            if 'counts' in data:
                saved=json.loads(str(data['counts']))
                for key,value in saved.items():COUNTS[key]+=value
            return status(m,name)
        previous=COUNTS.copy()
        m=builder()
        mesh=m.to_mesh64()
        changes={k:COUNTS[k]-previous[k] for k in COUNTS}
        np.savez_compressed(path,vertices=mesh.vert_properties[:,:3],
                            faces=mesh.tri_verts,counts=json.dumps(changes))
        log(f'Stage {name} saved: {m.num_tri():,} triangles')
        return m

    log('Building dense openwork embroidery for resin printing')
    floor=stage('floor',lambda:make_textured_base(RAD,Y_SCALE,BOTTOM))
    ground=stage('ground',build_open_ground)
    botanical=stage('botanical',build_botanical_fill)
    flowers=stage('flowers',build_main_flowers)
    rim=stage('rim',build_open_frill)
    final=stage('assembly',lambda:merge([floor,ground,botanical,flowers,rim],
                                     'complete openwork embroidery holder'))
    final=status(final.as_original().set_tolerance(0.002).simplify(0.010),
                 'simplified openwork holder')
    final,tm,contact_bonds=close_near_contact_edges(final)
    final,tm,vertex_bonds=close_point_contacts(final,tm)
    microfragment_bonds=[]
    for attachment_pass in range(3):
        final,attachments=attach_positive_slivers(
            final,tm,skip_component=numerical_residue_info)
        if not attachments:break
        log(f'Finite attachments, pass {attachment_pass+1}: {len(attachments)} positive microfragments')
        final,tm,additional_edges=close_near_contact_edges(final)
        final,tm,additional_vertices=close_point_contacts(final,tm)
        microfragment_bonds.append({'pass':attachment_pass+1,'attachments':attachments,
                                    'subsequent_edge_repairs':additional_edges,
                                    'subsequent_vertex_repairs':additional_vertices})
    if not (tm.is_watertight and tm.is_winding_consistent and tm.volume>0):
        log(f'QA failure: watertight={tm.is_watertight}, winding={tm.is_winding_consistent}')
        np.savez(ROOT/'mesh_v5_debug.npz',vertices=tm.vertices,faces=tm.faces)
        raise ValueError('Openwork mesh failed integrity verification')
    comps=tm.split(only_watertight=False,repair=False)
    # The combined threads can enclose tiny blind pockets.  A closed inward
    # boundary has negative signed volume. Removing that boundary fills the
    # enclosed pocket with material; all exterior openwork remains intact.
    # Never mistake a negative cavity boundary for a detached decoration.
    cavities=[c for c in comps if c.volume<0]
    residue=[];residue_details=[];useful=[]
    for c in comps:
        if c.volume<0:continue
        info=numerical_residue_info(c)
        if info is None:useful.append(c)
        else:residue.append(c);residue_details.append(info)
    cavity_details={'filled_cavity_count':len(cavities),
                    'filled_volume_mm3':float(-sum(c.volume for c in cavities)),
                    'largest_filled_cavity_mm3':float(max([-c.volume for c in cavities],default=0)),
                    'zero_thickness_numerical_residues':len(residue),
                    'numerical_residue_details':residue_details}
    log(f'Enclosed microcavities filled: {json.dumps(cavity_details)}')
    if cavities or residue:tm=trimesh.util.concatenate(useful)
    if len(useful)!=1:
        np.savez(ROOT/'mesh_v5_debug.npz',vertices=tm.vertices,faces=tm.faces)
        details=[{'faces':len(c.faces),'volume_mm3':float(c.volume),
                  'bounds':c.bounds.tolist()} for c in sorted(useful,key=lambda c:-len(c.faces))[:12]]
        log(json.dumps(details))
        raise ValueError(f'Expected one connected solid; got {len(useful)}')
    tm.export(ROOT/'floral_lace_pen_holder.stl')
    saved=trimesh.load_mesh(ROOT/'floral_lace_pen_holder.stl',process=True)
    assert saved.is_watertight and saved.is_winding_consistent and saved.is_volume
    log(f'STL exported: {len(tm.faces):,} triangles')
    make_3mf(tm,ROOT/'floral_lace_pen_holder.3mf')
    display=tm.copy()
    display.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2,[1,0,0]))
    display.apply_scale(.001)
    display.visual=trimesh.visual.ColorVisuals(display,face_colors=[222,214,200,255])
    display.export(ROOT/'floral_lace_pen_holder.glb')
    import hashlib
    report={'design_revision':5,'units':'mm','printing_process_intent':'resin photopolymer',
            'extents_mm':tm.extents.tolist(),'bounds_mm':tm.bounds.tolist(),
            'triangles':len(tm.faces),'vertices':len(tm.vertices),'watertight':bool(tm.is_watertight),
            'winding_consistent':bool(tm.is_winding_consistent),'connected_components':len(useful),
            'volume_cm3':float(tm.volume/1000),'surface_area_cm2':float(tm.area/100),
            'continuous_side_wall':False,'sidewall_perforations':True,
            'all_regions_structured':['outer and inner wall strands','open main flower petals',
                                      'small flowers and leaf veins','frill front and back',
                                      'picot loops','floor top and underside','floor perimeter'],
            'load_bearing_strand_width_mm':.72,'load_bearing_strand_radial_depth_mm':2.2,
            'decorative_strand_width_mm':[.27,.62],'nominal_thread_crest_width_mm':.060,'thread_crest_width_range_mm':[.04,.06],
            'main_petal_stitch_pitch_mm':.78,'base_details':BASE_DETAILS,
            'multiscale_lace_layout':LACE_LAYOUT,'rim_details':RIM_DETAILS,
            'flower_petal_counts':[sum(f[3:]) for f in FLOWERS],
            'embroidery_counts':COUNTS,'model_only_3mf':True,
            'finite_contact_bonds':contact_bonds,
            'finite_vertex_contact_bonds':vertex_bonds,
            'finite_microfragment_attachments':microfragment_bonds,
            'vertex_links_manifold':True,
            'closed_microcavities':cavity_details,
            'stl_sha256':hashlib.sha256((ROOT/'floral_lace_pen_holder.stl').read_bytes()).hexdigest(),
            'note':'Geometrical validation is not a physical print test. Dimensions are mm.'}
    (ROOT/'geometry_report.json').write_text(json.dumps(report,indent=2)+'\n')
    log(json.dumps(report,indent=2))

if __name__=='__main__':main()

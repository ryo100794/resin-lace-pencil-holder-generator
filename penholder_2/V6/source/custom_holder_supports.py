#!/usr/bin/env python3
"""Model-specific, removable supports for the immutable V5 holder.

The input boundary merge-tree events are independently classified by
audit_custom_supports.py.  This generator routes short rising supports
between actual surface contacts; no printer-specific load guarantee is made.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import manifold3d as mf
from scipy.spatial import cKDTree
import trimesh
from build_holder import SPHERES, sphere_transform, sphere_center
import build_holder as B

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'cache'/'custom_holder_supports'
EPS=1e-9
ACTIVE_SURFACE=None
CERTIFIED_MOTION=[]
NEW_SPHERE_REMOVABLE=[]
NEW_SPHERE_FIXED=[]
MOTION_CERTIFICATE=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(path):
    with np.load(path,allow_pickle=False) as d:
        v=np.asarray(d['vertices'],float);f=np.asarray(d['faces'],np.uint32)
    m=mf.Manifold(mf.Mesh64(v,f))
    if m.status()!=mf.Error.NoError:raise ValueError('Invalid manifold '+str(path))
    return m,v,f

def save(path,solid):
    mesh=solid.to_mesh64()
    np.savez_compressed(path,vertices=np.asarray(mesh.vert_properties[:,:3]),faces=mesh.tri_verts)

def union(parts):
    p=[x for x in parts if not x.is_empty()]
    return mf.Manifold.batch_boolean(p,mf.OpType.Add) if p else mf.Manifold()

def volume(m):return max(0.,float(m.volume())) if not m.is_empty() else 0.

def box(lo,hi):return mf.Manifold.cube(tuple(np.asarray(hi)-lo)).translate(lo)

def profile(point,stations,sides=16):
    """Closed vertical-axis surface of revolution; no cylinder corner islands."""
    p=np.asarray(point,float);a=np.arange(sides)*2*np.pi/sides
    ring=np.column_stack((np.cos(a),np.sin(a)))
    v=np.vstack([p+np.column_stack((r*ring,np.full(sides,z))) for z,r in stations])
    f=[]
    for i in range(len(stations)-1):
        for k in range(sides):
            a=i*sides+k;b=i*sides+(k+1)%sides;c=(i+1)*sides+(k+1)%sides;d=(i+1)*sides+k
            f.extend(((a,b,c),(a,c,d)))
    for k in range(1,sides-1):
        f.append((0,k+1,k));a=(len(stations)-1)*sides;f.append((a,a+k,a+k+1))
    m=mf.Manifold(mf.Mesh64(v,np.asarray(f,np.uint32)))
    if m.status()!=mf.Error.NoError:raise ValueError('profile topology')
    return m

def sphere_envelope_ok(m,envelopes):
    if CERTIFIED_MOTION:
        from audit_rotation_envelope import check_added_material
        worst=1.
        for placed in CERTIFIED_MOTION:
            # Complete-solid checks include every triangle of the generated
            # stock; a root hit is therefore classified as fixed material only
            # after the whole candidate excludes motion and every old wire.
            bb=np.asarray(m.bounding_box());eb=np.asarray(placed['envelope'].bounding_box())
            aabb_gap=np.linalg.norm(np.maximum(np.maximum(bb[:3]-eb[3:],eb[:3]-bb[3:]),0.))
            if aabb_gap<=.08003:
                result=check_added_material(m,placed['envelope'],placed['guard'],margin=.08)
                worst=min(worst,result['continuous_motion_clearance_lower_bound_mm'])
                if not result['passed']:return False,result['continuous_motion_clearance_lower_bound_mm']
            else:worst=min(worst,float(aabb_gap)-.00003)
            forbidden=placed['original_moving_and_internal_wires']
            fb=np.asarray(forbidden.bounding_box())
            fg=np.linalg.norm(np.maximum(np.maximum(bb[:3]-fb[3:],fb[:3]-bb[3:]),0.))
            if fg<.0001:
                if not (m^forbidden).is_empty() or float(m.min_gap(forbidden,.0001))<.00006:return False,-.00006
        for forbidden in NEW_SPHERE_REMOVABLE:
            if bbox_intersects(m,forbidden) and not (m^forbidden).is_empty():return False,-.00001
        return True,worst
    mm=m.to_mesh64();v=np.asarray(mm.vert_properties[:,:3]);f=np.asarray(mm.tri_verts)
    tri=v[f];worst=1e6
    for s in SPHERES:
        c=sphere_center(s);r=envelopes[str(s['layers'])]
        bb=np.asarray(m.bounding_box());dd=np.maximum(np.maximum(bb[:3]-c,c-bb[3:]),0)
        if np.linalg.norm(dd)>r+1:continue
        q=trimesh.triangles.closest_point(tri,np.broadcast_to(c,(len(tri),3)).copy())
        gap=float(np.linalg.norm(q-c,axis=1).min()-r);worst=min(worst,gap)
        if gap<.08:return False,gap
    return True,None if worst==1e6 else worst

def initialize_certified_motion_context():
    """Independently certified moving envelope plus immutable old mechanisms."""
    global CERTIFIED_MOTION,MOTION_CERTIFICATE,NEW_SPHERE_REMOVABLE,NEW_SPHERE_FIXED
    from audit_rotation_envelope import load_certified_envelopes
    report=ROOT/'reports/rotation_envelope_certificate.json'
    if not report.exists():return
    envelopes=load_certified_envelopes(ROOT);guards=load_certified_envelopes(ROOT,guards=True)
    originals={};sphere_removables={};sphere_fixed={};inputs={str(report.relative_to(ROOT)):sha(report)}
    for n in [3,4,5]:
        meta=json.loads((ROOT/f'cache/spheres/sphere_{n}layers.json').read_text())
        names=meta['free_layer_meshes'][1:]+[r['mesh_file'] for r in meta['release_wires']]
        solids=[]
        for name in names:
            path=ROOT/'cache/spheres'/name;obj,_,_=load(path);solids.append(obj)
            inputs[str(path.relative_to(ROOT))]=sha(path)
        originals[n]=union(solids)
        path=ROOT/f'cache/custom_sphere_supports/sphere_{n}layers_supports.json'
        if path.exists():
            data=json.loads(path.read_text());inputs[str(path.relative_to(ROOT))]=sha(path)
            solids=[];fixed=[]
            for row in data['supports']:
                fullfile=ROOT/'cache/custom_sphere_supports'/row['mesh_file']
                full,_,_=load(fullfile);inputs[str(fullfile.relative_to(ROOT))]=sha(fullfile)
                if row['kind']=='fixed_micro_reinforcement':fixed.append(full);continue
                file=ROOT/'cache/custom_sphere_supports'/row['removable_mesh_file']
                obj,_,_=load(file);solids.append(obj);inputs[str(file.relative_to(ROOT))]=sha(file)
                fixed.append(full-obj)
                length=row.get('neck_length_mm',row.get('neck_length_build_mm',.14))
                for point,normal in zip(row.get('cut_points',[]),row.get('cut_plane_normals',[])):
                    normal=np.asarray(normal,float);normal/=np.linalg.norm(normal)
                    offset=float(np.dot(point,normal));half=float(length)/2
                    neck=full.trim_by_plane(normal,offset-half).trim_by_plane(-normal,-offset-half)
                    solids.append(neck)
            sphere_removables[n]=union(solids)
            sphere_fixed[n]=union(fixed)
    CERTIFIED_MOTION=[];NEW_SPHERE_REMOVABLE=[];NEW_SPHERE_FIXED=[]
    for index,s in enumerate(SPHERES):
        transform=sphere_transform(s);n=s['layers']
        CERTIFIED_MOTION.append(dict(sphere_index=index,envelope=envelopes[n].transform(transform),
            guard=guards[n].transform(transform),original_moving_and_internal_wires=originals[n].transform(transform)))
        if n in sphere_removables:NEW_SPHERE_REMOVABLE.append(sphere_removables[n].transform(transform))
        if n in sphere_fixed:NEW_SPHERE_FIXED.append(sphere_fixed[n].transform(transform))
    MOTION_CERTIFICATE=dict(independent_report=str(report.relative_to(ROOT)),input_sha256=inputs,
        minimum_requested_motion_gap_mm=.08,additional_numeric_distance_allowance_mm=.00003,
        actual_original_moving_and_internal_wire_minimum_gap_mm=.00006,
        new_sphere_removable_stock_and_full_necks_excluded=bool(NEW_SPHERE_REMOVABLE),
        new_sphere_permanent_stock_in_fixed_contact_and_extraction_context=bool(NEW_SPHERE_FIXED),
        root_classification='A first hit on the immutable original surface is fixed only if the whole support excludes the independently certified moving envelope and every actual original moving-layer/internal-wire solid.')
    print('Certified whole-solid motion and original-wire context ready',flush=True)

def fixed_root_candidate(p):
    # The former radius heuristic remains a fallback only.  With the certified
    # context every proposed complete support receives the stricter solid gate.
    return bool(CERTIFIED_MOTION) or is_holder_point(p)

class Surface:
    def __init__(self,path,root_ray_density=1):
        t=time.monotonic();self.solid,self.vertices,self.faces=load(path)
        self.tri=self.vertices[self.faces]
        self.centers=self.tri.mean(axis=1)
        self.radii=np.linalg.norm(self.tri-self.centers[:,None],axis=2).max(axis=1)
        self.kd=cKDTree(self.centers)
        self.max_radius=float(self.radii.max())
        self.e1=self.tri[:,1]-self.tri[:,0];self.e2=self.tri[:,2]-self.tri[:,0]
        self.norm=np.cross(self.e1,self.e2);self.norm/=np.maximum(np.linalg.norm(self.norm,axis=1)[:,None],1e-20)
        self.bounds_tree=trimesh.triangles.bounds_tree(self.tri)
        self.root_ray_density=int(root_ray_density)
        print('surface loaded',len(self.faces),'triangles',time.monotonic()-t,flush=True)

    def roots(self,p,max_length=10.,minimum_drop=.04):
        """Exact first triangle hits on a downward cone, shortest hits first."""
        p=np.asarray(p,float);ids=np.asarray(self.kd.query_ball_point(p,max_length+self.max_radius),dtype=int)
        # Discard distant large-box candidates using exact center radius bound.
        ids=ids[np.linalg.norm(self.centers[ids]-p,axis=1)<=max_length+self.radii[ids]]
        orig=p-np.array([0.,0.,.0001]);tvec=orig-self.tri[ids,0]
        e1=self.e1[ids];e2=self.e2[ids];qvec=np.cross(tvec,e1)
        out=[];directions=[np.array([0.,0.,-1.])]
        rings=[(15,8),(30,12),(45,16),(60,20)]
        if self.root_ray_density>1:rings.extend([(22.5,10),(37.5,14),(52.5,18)])
        for angle,count in rings:
            aa=np.deg2rad(angle)
            count*=self.root_ray_density
            for phi in np.arange(count)*2*np.pi/count:
                directions.append(np.array([np.sin(aa)*np.cos(phi),np.sin(aa)*np.sin(phi),-np.cos(aa)]))
        for dr in directions:
            pv=np.cross(np.broadcast_to(dr,e2.shape),e2);det=np.einsum('ij,ij->i',e1,pv)
            inv=np.zeros_like(det);nz=np.abs(det)>1e-14;inv[nz]=1./det[nz]
            u=np.einsum('ij,ij->i',tvec,pv)*inv;vv=(qvec@dr)*inv
            tt=np.einsum('ij,ij->i',e2,qvec)*inv
            good=nz&(u>=-1e-8)&(vv>=-1e-8)&(u+vv<=1+1e-8)&(tt>1e-4)&(tt<=max_length)
            loc=np.flatnonzero(good)
            if not len(loc):continue
            k=loc[np.argmin(tt[loc])];q=orig+dr*tt[k]
            if p[2]-q[2]<minimum_drop:continue
            if self.norm[ids[k],2]<.15:continue
            out.append((float(tt[k]),q,self.norm[ids[k]],int(ids[k])))
        out.sort(key=lambda x:x[0]);return out

    def closest(self,p,max_gap=.15):
        p=np.asarray(p,float);ids=np.asarray(self.kd.query_ball_point(p,max_gap+self.max_radius),dtype=int)
        ids=ids[np.linalg.norm(self.centers[ids]-p,axis=1)<=max_gap+self.radii[ids]]
        if not len(ids):return None
        tri=self.tri[ids];points=trimesh.triangles.closest_point(tri,np.broadcast_to(p,(len(tri),3)).copy())
        dd=np.linalg.norm(points-p,axis=1);k=int(dd.argmin())
        if dd[k]>max_gap:return None
        return points[k],self.norm[ids[k]],int(ids[k]),float(dd[k])

    def nearby_surface_points(self,p,radius=.40):
        p=np.asarray(p,float)
        ids=np.fromiter(self.bounds_tree.intersection(tuple(np.r_[p-radius,p+radius])),dtype=int)
        if not len(ids):return []
        tri=self.tri[ids]
        points=trimesh.triangles.closest_point(tri,np.broadcast_to(p,(len(tri),3)).copy())
        dd=np.linalg.norm(points-p,axis=1)
        valid=(dd>.001)&(dd<=radius)
        points=points[valid];dd=dd[valid];ids=ids[valid]
        order=np.lexsort((dd,points[:,2]));seen=set();out=[]
        for k in order:
            key=tuple(np.round(points[k]/.025).astype(int))
            if key in seen:continue
            seen.add(key);out.append((float(dd[k]),points[k],self.norm[ids[k]],int(ids[k])))
            if len(out)>=36:break
        return out

    def may_hit_convex(self,query):
        """Conservative boundary-triangle rejection before expensive Booleans.

        Every query is a convex continuous sweep containing a fragment whose
        original-solid intersection was already excluded. If no original
        boundary triangle can intersect the convex volume, the sweep cannot
        enter the original solid. A triangle is rejected only when all its
        vertices lie outside one exact hull face plane (with outward margin).
        Remaining possibilities still use the original exact Boolean gate.
        """
        bb=np.asarray(query.bounding_box())
        ids=np.fromiter(self.bounds_tree.intersection(tuple(np.r_[bb[:3]-1e-7,bb[3:]+1e-7])),dtype=int)
        if not len(ids):return False
        triangles=self.tri[ids]
        mask=np.all(triangles.max(axis=1)>=bb[:3]-1e-7,axis=1)&np.all(triangles.min(axis=1)<=bb[3:]+1e-7,axis=1)
        triangles=triangles[mask]
        if not len(triangles):return False
        mesh=query.to_mesh64();v=np.asarray(mesh.vert_properties[:,:3]);f=np.asarray(mesh.tri_verts)
        tri=v[f];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(norm,axis=1)
        tri=tri[length>1e-14];norm=norm[length>1e-14]/length[length>1e-14,None]
        offsets=-np.einsum('ij,ij->i',norm,tri[:,0])
        _,index=np.unique(np.round(np.column_stack((norm,offsets)),10),axis=0,return_index=True)
        norm=norm[index];offsets=offsets[index]
        for first in range(0,len(triangles),512):
            tt=triangles[first:first+512]
            distances=np.einsum('ijk,lk->ijl',tt,norm)+offsets
            rejected=np.any(np.min(distances,axis=1)>1e-7,axis=1)
            if np.any(~rejected):return True
        return False

def stability_contacts(surface):
    """Two contacts on each surviving long radial link or lip spoke."""
    proposals=[];skipped=[];serial=0;TAU=2*np.pi
    for z in [9.,30.,52.,74.,96.]:
        for k in range(12):
            a=TAU*(k+.22*(int(z)//10%2))/12
            pitch=B.E.CIRC/60
            shift=.79*(z-2.7)+.10*np.sin(TAU*(z-2.7)/(pitch/.79))
            index=round((a*B.E.R_REF+B.E.CIRC/2-shift)/pitch)
            a=(-B.E.CIRC/2+index*pitch+shift)/B.E.R_REF
            o=np.array([float(B.OUTER(z))*np.cos(a),float(B.OUTER(z))*np.sin(a),z])
            end=B.inner_anchor(a,z)
            for t in [1/3,2/3]:
                p=o*(1-t)+end*t;p[2]+=1.25*np.sin(np.pi*t)-.51
                proposals.append((p,'radial_link',z,k,t,float(np.linalg.norm((o-end)[:2]))))
    for k in range(36):
        a=TAU*k/36
        for t in [1/3,2/3]:
            rr=33.1+(float(B.OUTER(99.0))-33.1)*t;phi=a+.020*np.sin(np.pi*t)
            p=np.array([rr*np.cos(phi),rr*np.sin(phi),98.7+1.15*np.sin(np.pi*t)-.38])
            proposals.append((p,'lip_spoke',98.7,k,t,float(B.OUTER(99.0))-33.1))
    events=[]
    for guess,kind,z,k,t,span in proposals:
        hit=surface.closest(guess,.15)
        if hit is None:
            skipped.append(dict(kind=kind,level_z_mm=z,member_index=k,fraction=t,
                                predicted_contact_mm=guess.tolist(),reason='No surviving original surface within 0.15 mm, e.g. sphere aperture cut.'))
            continue
        p,n,face,gap=hit
        if n[2]>=-.15 or not is_holder_point(p):
            skipped.append(dict(kind=kind,level_z_mm=z,member_index=k,fraction=t,
                                predicted_contact_mm=guess.tolist(),reason='Contact is not on a clear downward fixed-holder surface.'))
            continue
        events.append(dict(vertex_index=-1-len(events),position_mm=p.tolist(),outward_area_weighted_normal=n.tolist(),
                           birth_z_mm=float(p[2]),surface_lifetime_mm=0.,material_birth=False,
                           support_purpose='connected_span_stabilization',member_kind=kind,member_level_z_mm=z,
                           member_index=k,member_fraction=t,original_projected_span_mm=span,
                           proposed_three_section_projected_span_mm=span/3,actual_surface_gap_from_proposal_mm=gap,
                           contact_face=face))
    return events,skipped

def alternative_span_station(surface,event,fraction):
    """Move an arbitrary design station along its same actual fixed member."""
    if event['member_kind']!='radial_link':raise ValueError('Only radial-link alternatives are implemented')
    z=event['member_level_z_mm'];k=event['member_index'];t=float(fraction);TAU=2*np.pi
    a=TAU*(k+.22*(int(z)//10%2))/12;pitch=B.E.CIRC/60
    shift=.79*(z-2.7)+.10*np.sin(TAU*(z-2.7)/(pitch/.79))
    index=round((a*B.E.R_REF+B.E.CIRC/2-shift)/pitch)
    a=(-B.E.CIRC/2+index*pitch+shift)/B.E.R_REF
    o=np.array([float(B.OUTER(z))*np.cos(a),float(B.OUTER(z))*np.sin(a),z]);end=B.inner_anchor(a,z)
    guess=o*(1-t)+end*t;guess[2]+=1.25*np.sin(np.pi*t)-.51
    hit=surface.closest(guess,.15)
    if hit is None:return None
    p,n,face,gap=hit
    if n[2]>=-.15 or not is_holder_point(p):return None
    result=dict(event);result.update(position_mm=p.tolist(),outward_area_weighted_normal=n.tolist(),
        birth_z_mm=float(p[2]),contact_face=face,actual_surface_gap_from_proposal_mm=gap,
        member_original_proposed_fraction=event['member_fraction'],member_fraction=t,
        member_station_adjustment_reason='Modest relocation along the same surviving fixed member to give the temporary neck and withdrawal path enough space.')
    return result

def span_coverage_report(stability,skipped,rows,lip_coverage):
    groups={}
    for event in stability:
        if event['member_kind']!='radial_link':continue
        key=(event['member_level_z_mm'],event['member_index'])
        groups.setdefault(key,dict(member_kind='radial_link',level_z_mm=key[0],member_index=key[1],
            original_projected_span_mm=event['original_projected_span_mm'],requested_actual_source_stations=[],validated_stations=[],missing_source_stations=[]))['requested_actual_source_stations'].append(event['member_fraction'])
    for event in skipped:
        if event['kind']!='radial_link':continue
        key=(event['level_z_mm'],event['member_index'])
        if key in groups:groups[key]['missing_source_stations'].append(event)
    for row in rows:
        info=row.get('span_stabilization',{})
        if info.get('member_kind')!='radial_link':continue
        key=(info['member_level_z_mm'],info['member_index'])
        if key in groups:groups[key]['validated_stations'].append(dict(fraction=info['member_fraction'],support_index=row['index'],
            source_proposal_fraction=info.get('member_original_proposed_fraction',info['member_fraction'])))
    result=[]
    for item in groups.values():
        positions=sorted(set(station['fraction'] for station in item['validated_stations']))
        complete=len(item['validated_stations'])>=len(item['requested_actual_source_stations'])
        item['all_surviving_requested_stations_supported']=complete
        item['maximum_projected_section_mm']=(float(np.diff([0.]+positions+[1.]).max()*item['original_projected_span_mm']) if not item['missing_source_stations'] else None)
        item['scope']='Station spacing along the original projected member; clipped-away proposals are listed separately. This is not a stiffness, resin-strength or peel-force certification.'
        result.append(item)
    maxima=[r['maximum_projected_section_mm'] for r in result if r['maximum_projected_section_mm'] is not None and r['all_surviving_requested_stations_supported']]
    return dict(radial_members=result,all_surviving_requested_stations_supported=all(r['all_surviving_requested_stations_supported'] for r in result),
        maximum_unclipped_radial_projected_section_mm=max(maxima,default=None),
        lip_spoke_count=len(lip_coverage or []),lip_spokes_with_all_three_existing_stations_verified=sum(r['passed'] for r in lip_coverage or []),
        maximum_lip_projected_section_mm=max([r['maximum_projected_gap_between_supported_stations_mm'] for r in lip_coverage or []],default=None))

def verified_lip_support_stations(parts,rows,original):
    """Confirm existing printed embroidery tacks brace their actual lip spoke."""
    report=[];TAU=2*np.pi;outer=float(B.OUTER(99.0));span=outer-33.1
    for k in range(36):
        a=TAU*k/36;t=np.linspace(0,1,41)
        rr=33.1+span*t;phi=a+.020*np.sin(np.pi*t);zz=98.7+1.15*np.sin(np.pi*t)
        spoke=B.tube3(np.column_stack((rr*np.cos(phi),rr*np.sin(phi),zz)),r=.38,sides=10)
        stations=[]
        for q in [.28,.50,.72]:
            r=33.1+span*q;ts=np.linspace(-1,1,15);ph=a+ts*.012
            z=98.7+1.15*np.sin(np.pi*q)+.18*np.sin(np.pi*(ts+1))
            tack=B.tube3(np.column_stack((r*np.cos(ph),r*np.sin(ph),z)),r=.16,sides=8)
            joint=tack^spoke;actual_joint=joint^original
            jt=volume(actual_joint);found=[]
            if jt>1e-8:
                actual_tack=tack^original
                tack_parts=[piece for piece in actual_tack.decompose() if volume(piece)>1e-10]
                for support,row in zip(parts,rows):
                    if row.get('candidate_vertex',-1)<0 or not row.get('passed') or not bbox_intersects(tack,support):continue
                    if support.bounding_box()[2]>=tack.bounding_box()[2]-1e-4:continue
                    for piece in tack_parts:
                        contact=volume(piece^support);joint_here=volume(piece^actual_joint)
                        if contact>1e-8 and joint_here>1e-8:
                            found.append(dict(support_index=row['index'],actual_tack_contact_mm3=contact,
                                              actual_spoke_joint_mm3=joint_here,connected_actual_tack_component_mm3=volume(piece)))
                            break
            stations.append(dict(fraction=q,actual_tack_to_spoke_joint_volume_mm3=jt,validated_lower_supports=found,passed=bool(found)))
        good=[r['fraction'] for r in stations if r['passed']]
        gap=float(np.diff([0.]+good+[1.]).max()*span)
        report.append(dict(spoke_index=k,stations=stations,original_projected_span_mm=span,
                           maximum_projected_gap_between_supported_stations_mm=gap,
                           passed=all(r['passed'] for r in stations) and gap<=4.5,
                           scope='Finite actual-geometry joints and lower support paths; does not certify equal stiffness or peel resistance.'))
    return report

def pitched_normal(n,sign):
    n=np.asarray(n,float);n/=np.linalg.norm(n)
    if n[2]*sign<.75:
        xy=n[:2];length=np.linalg.norm(xy)
        n=np.r_[xy/max(length,1e-12)*math.sqrt(1-.75**2),sign*.75]
    return n

def axial_profile(point,direction,stations):
    n=np.asarray(direction,float);n/=np.linalg.norm(n)
    seed=np.array([1.,0.,0.]) if abs(n[0])<.9 else np.array([0.,1.,0.])
    u=np.cross(seed,n);u/=np.linalg.norm(u);v=np.cross(n,u)
    return profile([0,0,0],stations).transform(np.column_stack((u,v,n,np.asarray(point,float))))

def exact_ring(point,direction,radius,sides=16):
    n=np.asarray(direction,float);n/=np.linalg.norm(n)
    seed=np.array([1.,0.,0.]) if abs(n[0])<.9 else np.array([0.,1.,0.])
    u=np.cross(seed,n);u/=np.linalg.norm(u);v=np.cross(n,u)
    a=np.arange(sides)*2*np.pi/sides
    return np.asarray(point)+radius*(np.cos(a)[:,None]*u+np.sin(a)[:,None]*v)

def low_axis(part,point,direction,offset):
    n=np.asarray(direction,float)
    return part.trim_by_plane(-n,-float(np.dot(n,point)+offset))

def high_axis(part,point,direction,offset):
    n=np.asarray(direction,float)
    return part.trim_by_plane(n,float(np.dot(n,point)+offset))

def make_twig(q,p,rbody=.30,rneck=.175,root_normal=(0,0,1),tip_normal=(0,0,-1)):
    """Surface-normal end necks pitched upward/downward and a rising capsule."""
    q=np.asarray(q,float);p=np.asarray(p,float)
    nr=pitched_normal(root_normal,1);nt=pitched_normal(tip_normal,-1)
    def carrier(n):
        nz=abs(n[2]);return max(.60,(rbody+rneck*math.sqrt(1-nz*nz)+.22*nz+.045)/nz)
    cr,ct=carrier(nr),carrier(nt)
    if p[2]-q[2]<1.30:raise ValueError('Not enough vertical separation for two accessible necks')
    # A contact head that widens too quickly can start a floating sector on
    # the lower face of a tilted tip.  Limit its shoulder slope, while keeping
    # the load-bearing waist and body diameters unchanged.  The 20% margin is
    # applied to the exact circular-section minimum-Z growth condition.
    def head_radius(n):
        nz=abs(n[2]);transverse=math.sqrt(max(0.,1-nz*nz))
        return min(rbody,rneck+.8*.065*nz/max(transverse,1e-12))
    hr,ht=head_radius(nr),head_radius(nt)
    root=axial_profile(q,nr,[(-.16,hr),(.015,hr),(.08,rneck),(.22,rneck),(.34,rbody),(cr,rbody)])
    tip=axial_profile(p,nt,[(-.16,ht),(.015,ht),(.08,rneck),(.22,rneck),(.34,rbody),(ct,rbody)])
    a=q+nr*cr;b=p+nt*ct
    if b[2]<=a[2]:raise ValueError('Nonrising body')
    lower=np.vstack((exact_ring(a-nr*.03,nr,rbody),exact_ring(a,nr,rbody)))
    upper=np.vstack((exact_ring(b,nt,rbody),exact_ring(b-nt*.03,nt,rbody)))
    if upper[:,2].min()<=lower[:,2].min()+.01:raise ValueError('Upper body cap begins before the seated lower cap')
    body=mf.Manifold.hull_points(np.vstack((lower,upper)))
    full=root+body+tip
    necks=[axial_profile(q,nr,[(.08,rneck),(.22,rneck)]),axial_profile(p,nt,[(.08,rneck),(.22,rneck)])]
    rootstations=[(-.16,hr),(.015,hr),(.08,rneck),(.22,rneck),(.34,rbody),(cr,rbody)]
    tipstations=[(-.16,ht),(.015,ht),(.08,rneck),(.22,rneck),(.34,rbody),(ct,rbody)]
    convex=[body]
    for pt,n,stations in [(q,nr,rootstations),(p,nt,tipstations)]:
        convex.extend(axial_profile(pt,n,[a,b]) for a,b in zip(stations,stations[1:]))
    return full,necks,nr,nt,convex,dict(cut=.15,retained=.20,contact_limit=.075,neck_length=.14,style='pitched_carriers',
        endpoint_contact_head_diameters_mm=[2*hr,2*ht],contact_head_growth_margin=0.8)

def make_compact_tack(q,p,rbody=.18,rneck=.10):
    """Short twigs use monotonically rising horizontal polygon sections.

    No spherical end caps and no hidden cylinder corners are introduced.
    Each endpoint still has an explicit narrow, 0.08 mm long neck.
    """
    q=np.asarray(q,float);p=np.asarray(p,float);dz=p[2]-q[2]
    if not .42<=dz<=2.2:raise ValueError('Compact tack vertical range')
    cosine=dz/float(np.linalg.norm(p-q))
    # Horizontal circles become ellipses in the load-normal section.
    # Include polygon apothem and a 0.01 mm dimensional margin.
    rneck=max(rneck,.21/(2*math.cos(math.pi/16)*cosine))
    rbody=max(rbody,rneck+.04)
    nr=np.array([0.,0.,1.]);nt=-nr
    shear=np.array([[1.,0.,(p[0]-q[0])/dz,q[0]],
                    [0.,1.,(p[1]-q[1])/dz,q[1]],
                    [0.,0.,1.,q[2]]])
    stations=[(-.08,rbody),(.005,rbody),(.06,rneck),(.14,rneck),(.18,rbody),
              (dz-.18,rbody),(dz-.14,rneck),(dz-.06,rneck),(dz+.005,rbody),(dz+.08,rbody)]
    full=profile([0,0,0],stations).transform(shear)
    necks=[profile([0,0,0],[(.06,rneck),(.14,rneck)]).transform(shear),
           profile([0,0,0],[(dz-.14,rneck),(dz-.06,rneck)]).transform(shear)]
    convex=[profile([0,0,0],[a,b]).transform(shear) for a,b in zip(stations,stations[1:])]
    return full,necks,nr,nt,convex,dict(cut=.10,retained=.13,contact_limit=.05,neck_length=.08,
        style='compact_horizontal_sections',neck_radius=rneck,body_radius=rbody,
        minimum_neck_minor_width_lower_bound_mm=2*rneck*math.cos(math.pi/16)*cosine)

def make_axial_short_tack(q,p):
    """Short circular-axis necks avoid shear narrowing in cramped positions."""
    q=np.asarray(q,float);p=np.asarray(p,float);u=p-q;length=float(np.linalg.norm(u));u/=length
    if not .40<=length<=3.0 or u[2]<.20:raise ValueError('Short axial tack range')
    rneck=.135
    transverse=math.sqrt(max(0.,1-u[2]**2))
    increase=min(.015,.8*.045*u[2]/max(transverse,1e-12))
    rbody=rneck+increase
    stations=[(-.08,rbody),(.005,rbody),(.05,rneck),(.13,rneck),(.18,rbody),
              (length-.18,rbody),(length-.13,rneck),(length-.05,rneck),(length+.005,rbody),(length+.08,rbody)]
    # Minimum Z of every circular station must progress in build order.
    first_levels=np.array([s*u[2]-r*transverse for s,r in stations])
    if np.any(np.diff(first_levels)<-1e-10):raise ValueError('Axial tack has a backward-growing shoulder')
    full=axial_profile(q,u,stations)
    necks=[axial_profile(q,u,[(.05,rneck),(.13,rneck)]),axial_profile(p,-u,[(.05,rneck),(.13,rneck)])]
    convex=[axial_profile(q,u,[aa,bb]) for aa,bb in zip(stations,stations[1:])]
    return full,necks,u,-u,convex,dict(cut=.09,retained=.12,contact_limit=.045,neck_length=.08,
        style='short_axial_profile',neck_radius=rneck,body_radius=rbody,
        minimum_neck_minor_width_lower_bound_mm=2*rneck*math.cos(math.pi/16),
        maximum_temporary_length_mm=3.0)

def seated(support,solid,point):
    """Trim only free lower edge below the first actual root contact."""
    local=box(np.asarray(point)-[1.,1.,.4],np.asarray(point)+[1.,1.,.35])
    contact=support^solid^local
    if volume(contact)<=1e-7:return None,None
    floor=float(contact.bounding_box()[2])+.001
    if floor>point[2]+.04:return None,None
    result=support.trim_by_plane((0,0,1),floor)
    return result,floor

def sweep(part,direction,travel):
    mesh=part.to_mesh64();v=np.asarray(mesh.vert_properties[:,:3]);d=np.asarray(direction,float)
    return mf.Manifold.hull_points(np.vstack((v,v+d*travel)))

def paths_at(point,vertical_preference=0):
    radial=np.array([point[0],point[1],0.]);radial/=np.linalg.norm(radial)
    # Inner skin is removed into the pen cavity, outer skin toward the exterior.
    sign=1 if np.linalg.norm(point[:2])>40 else -1
    d=sign*radial;t=np.array([-radial[1],radial[0],0.])
    flat=[d,(d+.45*t)/np.linalg.norm(d+.45*t),(d-.45*t)/np.linalg.norm(d-.45*t),
          (d+.85*t)/np.linalg.norm(d+.85*t),(d-.85*t)/np.linalg.norm(d-.85*t),-d]
    tilted=[]
    for zz in ([.50,-.50] if vertical_preference>=0 else [-.50,.50]):
        for v in flat:
            w=v+[0,0,zz];tilted.append(w/np.linalg.norm(w))
    # Some bottom contact ends can only clear their own cut face with a
    # steeper diagonal withdrawal.  These are explicit additional candidates;
    # every continuous sweep still undergoes the same exact collision gate.
    alternate=[(-d+.45*t)/np.linalg.norm(-d+.45*t),(-d-.45*t)/np.linalg.norm(-d-.45*t)]
    steep=[]
    for height in [1.,2.]:
        for zz in ([height,-height] if vertical_preference>=0 else [-height,height]):
            for v in flat+alternate:
                w=v+[0,0,zz];steep.append(w/np.linalg.norm(w))
    return flat[:3]+tilted+flat[3:]+steep

def bbox_intersects(a,b):
    aa=np.asarray(a.bounding_box());bb=np.asarray(b.bounding_box())
    return bool(np.all(aa[:3]<=bb[3:]) and np.all(bb[:3]<=aa[3:]))

def obstruction(query,original,previous):
    # Small, nearby retained ends reject impossible departures cheaply before
    # the same exact whole-model Boolean gate is evaluated.
    v=0.
    for other in previous:
        if bbox_intersects(query,other):
            v+=volume(query^other)
            if v>1e-8:return v
    if ACTIVE_SURFACE is not None and original is ACTIVE_SURFACE.solid and not ACTIVE_SURFACE.may_hit_convex(query):return v
    v+=volume(query^original)
    return v

def removal_plan(removable,convex,original,previous,q,p,release_planes=None):
    """Whole-piece extraction, or explicit further body cuts into small pieces."""
    if volume(removable^original)>1e-8:return None
    travel=20.;mid=(np.asarray(q)+p)/2
    def active_release_normals(part):
        if not release_planes:return []
        mesh=part.to_mesh64();vv=np.asarray(mesh.vert_properties[:,:3]);active=[]
        for point,normal in release_planes:
            normal=np.asarray(normal,float)
            if abs(float(np.min((vv-np.asarray(point))@normal)))<1e-6:active.append(normal)
        return active
    def can_depart(direction,normals):
        return all(float(np.dot(direction,normal))>=-1e-12 for normal in normals)
    whole_normals=active_release_normals(removable)
    for direction in paths_at(mid):
        if not can_depart(direction,whole_normals):continue
        sv=obstruction(sweep(removable,direction,travel),original,previous)
        if sv>1e-8:
            sv=0.
            for piece in convex:
                sv+=obstruction(sweep(piece,direction,travel),original,previous)
                if sv>1e-8:break
        if sv<=1e-8:return dict(mode='whole_piece',cuts=[],fragments=[(removable,direction,travel,sv)],convex_sweep_parts=convex)
    axis=np.asarray(p)-q;axis/=np.linalg.norm(axis)
    mesh=removable.to_mesh64();vv=np.asarray(mesh.vert_properties[:,:3]);lo,hi=(vv@axis).min(),(vv@axis).max()
    for count in sorted(set([2,3,max(3,int(math.ceil((hi-lo)/1.8)))])):
        levels=np.linspace(lo,hi,count+1)[1:-1];fragments=[];cuts=[]
        for i in range(count):
            part=removable
            if i>0:part=part.trim_by_plane(axis,float(levels[i-1]+.025))
            if i<count-1:part=part.trim_by_plane(-axis,-float(levels[i]-.025))
            if volume(part)>1e-12:fragments.append(part)
        succeeded=[]
        # Discarded kerfs are explicit geometry; each remaining fragment must
        # clear the other fragments as well as all earlier supports.
        for i,part in enumerate(fragments):
            chosen=None;bb=np.asarray(part.bounding_box());center=(bb[:3]+bb[3:])/2
            others=previous+[f for j,f in enumerate(fragments) if j!=i]
            pref=1 if np.linalg.norm(center-q)<np.linalg.norm(center-p) else -1
            normals=active_release_normals(part)
            for direction in paths_at(center,pref):
                if not can_depart(direction,normals):continue
                sv=obstruction(sweep(part,direction,travel),original,others)
                if sv<=1e-8:chosen=(part,direction,travel,sv);break
            if chosen is None:break
            succeeded.append(chosen)
        if len(succeeded)==len(fragments):
            for level in levels:
                t=float((level-np.dot(axis,q))/np.dot(axis,np.asarray(p)-q))
                cuts.append(dict(plane_normal=axis.tolist(),plane_offset_mm=float(level),
                                 centerline_reference_mm=(np.asarray(q)+t*(np.asarray(p)-q)).tolist(),
                                 modeled_removed_kerf_width_mm=.05,
                                 instruction='Additional snip through temporary body after the two endpoint necks; preserve the fixed model.'))
            return dict(mode='explicit_body_cuts',cuts=cuts,fragments=succeeded)
    return None

def store_removal_plan(index,row,plan):
    stem=f'holder_support_{index:04d}';fragment_rows=[]
    for n,(piece,direction,travel,sv) in enumerate(plan['fragments']):
        name=stem+f'_fragment_{n:02d}.npz';save(OUT/name,piece)
        fragment_rows.append(dict(mesh_file=name,direction=direction.tolist(),travel_mm=travel,
                                  swept_overlap_volume_mm3=sv,bounds_mm=list(piece.bounding_box())))
    row['removal_mode']=plan['mode'];row['additional_body_cuts']=plan['cuts'];row['removal_fragments']=fragment_rows
    row.pop('generated_convex_sweep_meshes',None)
    if plan.get('convex_sweep_parts'):
        names=[]
        for n,piece in enumerate(plan['convex_sweep_parts']):
            name=stem+f'_convex_sweep_piece_{n:02d}.npz';save(OUT/name,piece);names.append(name)
        row['generated_convex_sweep_meshes']=names

def restore_declared_fragments(row):
    """Recreate cache fragments exactly from authoritative cuts and stock.

    This also makes an interrupted non-authoritative route optimization safe:
    no changed fragment cache is allowed to silently disagree with its saved
    checkpoint's explicit cut planes.
    """
    declared=row.get('removal_fragments',[])
    if not declared:return
    removable,_,_=load(OUT/row['removable_mesh_file'])
    cuts=row.get('additional_body_cuts',[])
    if not cuts:
        if len(declared)!=1:raise ValueError('Unexplained multiple fragments')
        save(OUT/declared[0]['mesh_file'],removable);return
    normal=np.asarray(cuts[0]['plane_normal'],float)
    if any(np.linalg.norm(np.asarray(c['plane_normal'])-normal)>1e-10 for c in cuts):raise ValueError('Nonparallel saved cut plan')
    cuts=sorted(cuts,key=lambda c:c['plane_offset_mm']);pieces=[]
    for i in range(len(cuts)+1):
        part=removable
        if i>0:
            c=cuts[i-1];part=part.trim_by_plane(normal,c['plane_offset_mm']+c['modeled_removed_kerf_width_mm']/2)
        if i<len(cuts):
            c=cuts[i];part=part.trim_by_plane(-normal,-c['plane_offset_mm']+c['modeled_removed_kerf_width_mm']/2)
        if volume(part)>1e-12:pieces.append(part)
    if len(pieces)!=len(declared):raise ValueError('Declared fragment count differs from exact saved cuts')
    for piece,rowpart in zip(pieces,declared):save(OUT/rowpart['mesh_file'],piece)

def revalidate_routes(parts,rows,original):
    """Check every route against ALL nominal ends and every permanent join."""
    nominal=[]
    for full,row in zip(parts,rows):
        if row.get('temporary',True):
            removable,_,_=load(OUT/row['removable_mesh_file']);nominal.append(full-removable)
        else:nominal.append(full)
    obstacle=union(nominal)
    failures=[];checked=0;replanned=0
    for index,(full,row) in enumerate(zip(parts,rows)):
        if not row.get('temporary',True) or row['kind']=='plate_perimeter_tack':continue
        removable,_,_=load(OUT/row['removable_mesh_file'])
        initial=volume(removable^original)
        bad=initial>1e-8
        if not bad:
            for frag in row.get('removal_fragments',[]):
                names=(row.get('generated_convex_sweep_meshes') if len(row['removal_fragments'])==1 else None) or [frag['mesh_file']]
                for name in names:
                    piece,_,_=load(OUT/name);query=sweep(piece,frag['direction'],frag['travel_mm'])
                    if bbox_intersects(query,obstacle) and volume(query^obstacle)>1e-8:bad=True;break
                if bad:break
        if bad:
            q=np.asarray(row['root_mm']);p=np.asarray(row['target_mm'])
            convex=[]
            for name in row.get('generated_convex_sweep_meshes',[]):
                piece,_,_=load(OUT/name);convex.append(piece)
            if not convex:convex=[removable]
            plan=removal_plan(removable,convex,original,parts[:index]+[obstacle],q,p)
            if plan is None:
                failures.append(dict(index=index,kind=row['kind'],reason='No route clear of original, all nominal ends and final permanent joins',initial_original_overlap_mm3=initial))
                row['own_nominal_stumps_and_final_permanent_joins_checked']=False
                continue
            store_removal_plan(index,row,plan);replanned+=1
        row['own_nominal_stumps_and_final_permanent_joins_checked']=True
        row['all_holder_nominal_stumps_checked']=True
        row['initial_removable_original_overlap_mm3']=initial
        checked+=1
    return dict(checked_temporary_parts=checked,replanned_parts=replanned,all_holder_nominal_stumps_included=True,
        failures=failures,passed=not failures)

def is_holder_point(p):
    return all(np.linalg.norm(np.asarray(p)-sphere_center(s))>s['radius']+.55 for s in SPHERES)

def micro_reinforcement(surface,original,local,keepout,previous,p,envelopes,maximum_gap=.55):
    """Tiny fixed-only joins; <=0.4 mm preferred, <=0.55 mm documented exception."""
    candidates=surface.roots(p,max_length=maximum_gap,minimum_drop=.00001)
    candidates.extend(surface.nearby_surface_points(p,maximum_gap))
    for distance,q,norm,face in candidates:
        if not fixed_root_candidate(q):continue
        for radius in [.12,.10]:
            a=mf.Manifold.sphere(radius,16).translate(q)
            b=mf.Manifold.sphere(radius,16).translate(p)
            part=mf.Manifold.batch_hull([a,b])
            part,floor=seated(part,local,q)
            if part is None:continue
            if floor>=p[2]-.0001:continue
            if volume(part)>.05:continue
            ok,gap=sphere_envelope_ok(part,envelopes)
            if not ok or volume(part^keepout)>1e-9:continue
            collision=False
            for old in previous:
                if bbox_intersects(part,old) and volume((part^old)-local)>1e-8:collision=True;break
            if collision:continue
            contacts=[]
            for at in [q,p]:contacts.append(volume(part^local^mf.Manifold.sphere(.055,16).translate(at)))
            if min(contacts)<1e-10:continue
            prefix_ok,checks=add_prefix_checks(part,local,p[2]+radius+.01)
            if not prefix_ok:continue
            return part,dict(kind='fixed_micro_reinforcement',temporary=False,target_mm=np.asarray(p).tolist(),root_mm=q.tolist(),
                actual_fixed_surface_gap_mm=distance,diameter_mm=2*radius,maximum_added_part_volume_mm3=.05,
                local_gap_exception_over_preferred_0_40_mm=distance>.40,
                maximum_permitted_fixed_only_gap_mm=maximum_gap,
                fixed_gap_design_exception=(dict(reason='No lower fixed surface exists within the normal 0.55 mm search at this repeated flower start; a short fixed-only bridge replaces inaccessible removable necks.',maximum_mm=maximum_gap) if maximum_gap>.55 else None),
                volume_mm3=volume(part),added_volume_outside_original_mm3=volume(part-local),
                endpoint_probe_contact_volumes_mm3=contacts,root_face=face,neck_count=0,
                moving_envelope_clearance_mm=gap,original_service_path_intersection_mm3=0.,prefix_checks=checks,
                initial_material_precedes_target_start_mm=float(p[2]-floor),
                root_seating_floor_z_mm=floor,removal='Permanent fixed-holder local reinforcement; do not cut.',passed=True)
    return None

def terminal_micro_reinforcement(surface,local,keepout,previous,rows,p,envelopes):
    """Join nearby fixed tips using an already supported retained terminal.

    The permanent bridge must avoid every removable body and preserve the
    previously validated withdrawal path. Its startup may depend on the
    earlier temporary support's terminal, which remains attached to the fixed
    model after that support is cut. This dependency is recorded explicitly.
    """
    p=np.asarray(p,float)
    for prior,old in zip(previous,rows):
        if old.get('kind') not in ('local_holder_twig','span_stabilization_twig'):continue
        ot=np.asarray(old['target_mm'],float)
        if np.linalg.norm(ot-p)>.80:continue
        ids=np.asarray(surface.kd.query_ball_point(ot,.20+surface.max_radius),dtype=int)
        tri=surface.tri[ids]
        if not len(tri):continue
        near=trimesh.triangles.closest_point(tri,np.broadcast_to(ot,(len(tri),3)).copy())
        tri=tri[np.linalg.norm(near-ot,axis=1)<.20]
        if not len(tri):continue
        points=trimesh.triangles.closest_point(tri,np.broadcast_to(p,(len(tri),3)).copy())
        dist=np.linalg.norm(points-p,axis=1)
        order=np.argsort(dist)
        for j in order[:12]:
            q=points[j];gap=float(dist[j])
            if not .025<gap<=.55 or np.linalg.norm(q-ot)>.20:continue
            for radius in [.10,.12]:
                base=mf.Manifold.batch_hull([mf.Manifold.sphere(radius,16).translate(p),mf.Manifold.sphere(radius,16).translate(q)])
                for below in [.06,.04,.02,.01]:
                    floor=min(p[2],q[2]) - below
                    part=base.trim_by_plane((0,0,1),float(floor))
                    if volume(part)>.05:continue
                    ok,mgap=sphere_envelope_ok(part,envelopes)
                    if not ok or volume(part^keepout)>1e-9:continue
                    bad=False
                    for candidate,row in zip(previous,rows):
                        if not bbox_intersects(part,candidate):continue
                        removable,_,_=load(OUT/row['removable_mesh_file'])
                        if volume(part^removable)>1e-9:bad=True;break
                        # Permanent additions cannot invalidate an older
                        # support's prescribed extraction trajectory.
                        for frag in row.get('removal_fragments',[]):
                            direction=frag['direction'];travel=frag['travel_mm']
                            convex_names=row.get('generated_convex_sweep_meshes') if len(row['removal_fragments'])==1 else None
                            names=convex_names or [frag['mesh_file']]
                            for name in names:
                                piece,_,_=load(OUT/name);query=sweep(piece,direction,travel)
                                if bbox_intersects(query,part) and volume(query^part)>1e-9:bad=True;break
                            if bad:break
                        if bad:break
                    if bad:continue
                    contacts=[volume(part^local^mf.Manifold.sphere(.055,16).translate(at)) for at in (q,p)]
                    if min(contacts)<1e-10:continue
                    context=local+prior
                    prefix_ok,checks=add_prefix_checks(part,context,p[2]+radius+.01)
                    if not prefix_ok:continue
                    return part,dict(kind='fixed_micro_reinforcement',temporary=False,target_mm=p.tolist(),root_mm=q.tolist(),
                        actual_fixed_surface_gap_mm=gap,diameter_mm=2*radius,volume_mm3=volume(part),
                        local_gap_exception_over_preferred_0_40_mm=gap>.40,
                        maximum_permitted_fixed_only_gap_mm=.55,
                        maximum_added_part_volume_mm3=.05,added_volume_outside_original_mm3=volume(part-local),
                        endpoint_probe_contact_volumes_mm3=contacts,moving_envelope_clearance_mm=mgap,
                        original_service_path_intersection_mm3=0.,neck_count=0,prefix_checks=checks,
                        startup_dependency_support_index=old['index'],startup_dependency_region='Retained terminal above the older support cut; no removable body intersection.',
                        previous_removable_body_and_withdrawal_intersections_mm3=0.,root_seating_floor_z_mm=floor,
                        removal='Permanent fixed-holder local reinforcement; do not cut.',passed=True)
    return None

def add_prefix_checks(part,original,neck_floor):
    """Entire height: every positive support prefix touches original material."""
    bb=np.asarray(part.bounding_box());lo,hi=bb[2],bb[5]
    mesh=part.to_mesh64();vv=np.asarray(mesh.vert_properties[:,:3]);ff=np.asarray(mesh.tri_verts);zz=vv[:,2]
    neighbor=np.full(len(vv),np.inf)
    for i,j in [(0,1),(1,2),(2,0)]:
        np.minimum.at(neighbor,ff[:,i],zz[ff[:,j]]);np.minimum.at(neighbor,ff[:,j],zz[ff[:,i]])
    minima=np.unique(zz[zz<=neighbor+1e-10]);offsets=np.array([.0001,.001,.005,.025])
    heights=np.unique(np.r_[lo+offsets,np.arange(lo+.05,hi,.05),(minima[:,None]+offsets).ravel(),hi])
    contact=part^original
    rows=[]
    for z in heights:
        if z>hi+1e-9:continue
        pre=part.trim_by_plane((0,0,-1),-float(z))
        pieces=[x for x in pre.decompose() if volume(x)>1e-12]
        if len(pieces)==1:values=[volume(contact.trim_by_plane((0,0,-1),-float(z)))]
        else:values=[volume(x^original) for x in pieces]
        ok=all(v>1e-12 for v in values)
        rows.append(dict(z_mm=float(z),parts=len(pieces),actual_original_contact_volumes_mm3=values,passed=ok))
        if not ok:return False,rows
    return True,rows

def tool_keepout():
    parts=[];cache=ROOT/'cache/spheres'
    for s in SPHERES:
        meta=json.loads((cache/f"sphere_{s['layers']}layers.json").read_text())
        tr=sphere_transform(s)
        for w in meta['release_wires']:
            for endpoint in w['endpoints']:
                m,_,_=load(cache/endpoint['tool_corridor_mesh_file']);parts.append(m.transform(tr))
            m,v,_=load(cache/w['removable_mesh_file'])
            direct=np.asarray(w.get('extraction_vector',[0,0,1]),float)
            dist=float(w.get('extraction_distance_mm',2*meta['outer_nominal_radius_mm']+4))
            hull=mf.Manifold.hull_points(np.vstack((v,v+dist*direct)))
            parts.append(hull.transform(tr))
    return union(parts)

def main():
    global ACTIVE_SURFACE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min-lifetime',type=float,default=.025)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--skip-extraction',action='store_true',help='Diagnostic only; cannot pass release gate')
    ap.add_argument('--skip-foundations',action='store_true')
    ap.add_argument('--skip-stability',action='store_true')
    ap.add_argument('--resume',action='store_true',help='Keep already validated parts from generation_checkpoint.json and retry remaining contacts.')
    ap.add_argument('--candidate-ids-file',type=Path,help='JSON integer list restricting this pass to specific unresolved source candidates.')
    ap.add_argument('--finalize-only',action='store_true',help='Recheck all removal routes and regenerate aggregates from an existing checkpoint.')
    ap.add_argument('--local-route-revalidation',action='store_true',help='Optional diagnostic; the final combined external-removal planner is authoritative.')
    ap.add_argument('--fixed-gap-exception-ids',type=Path,help='JSON IDs of explicitly diagnosed fixed flower starts permitting a <=0.65 mm micro bridge.')
    ap.add_argument('--root-ray-density',type=int,choices=[1,2,4],default=1)
    ap.add_argument('--span-alternatives',type=Path,help='JSON mapping source station IDs to alternate fractions on the same member, tried after the original station.')
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    fixed_gap_exception_ids=set(json.loads(args.fixed_gap_exception_ids.read_text())) if args.fixed_gap_exception_ids else set()
    started=time.monotonic();source=ROOT/'cache/original_v5.npz';surface=Surface(source,args.root_ray_density)
    ACTIVE_SURFACE=surface
    original=surface.solid;env=json.loads((ROOT/'reports/moving_envelopes.json').read_text())
    initialize_certified_motion_context()
    classified=ROOT/'reports/support_analysis_fixed_holder_contacts.json'
    if classified.exists():events=json.loads(classified.read_text())
    else:
        events=json.loads((ROOT/'reports/support_analysis_candidates.json').read_text())
        events=[r for r in events['rows'] if r.get('material_birth',r['downward_birth']) and not r['plate_contact'] and is_holder_point(r['position_mm'])]
    selected=[r for r in events if r['surface_lifetime_mm']>=args.min_lifetime]
    stability,stability_skipped=([],[]) if args.skip_stability else stability_contacts(surface)
    selected.extend(stability)
    if args.span_alternatives:
        alternatives=json.loads(args.span_alternatives.read_text())
        for event in stability:
            for fraction in alternatives.get(str(event['vertex_index']),[]):
                changed=alternative_span_station(surface,event,fraction)
                if changed is not None:selected.append(changed)
    selected.sort(key=lambda r:(bool(r.get('support_purpose')),r['birth_z_mm']))
    if args.limit:selected=selected[:args.limit]
    keepout=tool_keepout();print('Internal service keepout ready; selected',len(selected),flush=True)
    parts=[];stumps=[];rows=[];failed=[];resume_count=0;lip_station_coverage=None;redundant_stability=[];previous_failed=[]
    if args.resume and (OUT/'generation_checkpoint.json').exists():
        saved=json.loads((OUT/'generation_checkpoint.json').read_text())
        if saved['source_sha256']!=sha(source):raise ValueError('Resume source changed')
        for row in saved['parts']:
            if row['kind'] not in ('plate_perimeter_tack','fixed_micro_reinforcement') and row.get('body_construction')!='exact_endrings_v3':continue
            full,_,_=load(OUT/row['full_mesh_file']);retained,_,_=load(OUT/row['retained_mesh_file'])
            parts.append(full);stumps.append(retained);rows.append(row)
        # All geometry has been loaded before reused filenames are compacted.
        for idx,(full,retained,row) in enumerate(zip(parts,stumps,rows)):
            if row.get('body_construction')=='exact_endrings_v3':
                if row['index']!=idx:raise ValueError('A mixed-generation checkpoint needs a clean run')
                restore_declared_fragments(row)
                continue
            row['index']=idx;stem=f'holder_support_{idx:04d}'
            removable=full-retained
            for label,obj in [('full',full),('retained',retained),('removable',removable)]:
                name=stem+'_'+label+'.npz';save(OUT/name,obj);row[label+'_mesh_file']=name
        resume_count=len(rows);done={r.get('candidate_vertex') for r in rows}
        selected=[e for e in selected if e['vertex_index'] not in done]
        previous_failed=[r for r in saved.get('unresolved_selected_candidates',[]) if r['candidate_vertex'] not in done]
        print('Resumed',resume_count,'parts;',len(selected),'candidate contacts remain',flush=True)
    if args.candidate_ids_file:
        wanted=set(json.loads(args.candidate_ids_file.read_text()))
        selected=[e for e in selected if e['vertex_index'] in wanted]
    if args.finalize_only:selected=[]
    selected_ids={e['vertex_index'] for e in selected}
    failed=[r for r in previous_failed if r['candidate_vertex'] not in selected_ids]

    def commit(full,retained,removable,row):
        idx=len(parts);row['index']=idx;stem=f'holder_support_{idx:04d}'
        row.setdefault('temporary',True)
        row['body_construction']='exact_endrings_v3'
        for label,obj in [('full',full),('retained',retained),('removable',removable)]:
            name=stem+'_'+label+'.npz';save(OUT/name,obj);row[label+'_mesh_file']=name
        if '_removal_plan' in row:
            store_removal_plan(idx,row,row.pop('_removal_plan'))
        parts.append(full);stumps.append(retained);rows.append(row)
        failed[:]=[entry for entry in failed if entry['candidate_vertex']!=row.get('candidate_vertex')]
        (OUT/'generation_checkpoint.json').write_text(json.dumps(dict(complete=False,parts=rows,
            unresolved_selected_candidates=failed,source_sha256=sha(source)),indent=2)+'\n')
        print('accepted',idx,row['kind'],np.round(row['target_mm'],3),flush=True)

    for serial,e in enumerate(selected):
        if any(row.get('candidate_vertex')==e['vertex_index'] for row in rows):continue
        if e.get('support_purpose') and e.get('member_kind')=='lip_spoke':
            if lip_station_coverage is None:lip_station_coverage=verified_lip_support_stations(parts,rows,original)
            coverage=lip_station_coverage[e['member_index']]
            if coverage['passed']:
                redundant_stability.append(dict(member_kind='lip_spoke',member_index=e['member_index'],member_fraction=e['member_fraction'],
                    reason='All three existing embroidery tacks have validated lower supports and finite actual tack-to-spoke joints.',
                    maximum_projected_gap_mm=coverage['maximum_projected_gap_between_supported_stations_mm']))
                continue
        p=np.asarray(e['position_mm'],float);reason_counts={};accepted=False
        roi=box(p-[12.,12.,12.],p+[12.,12.,1.]);source_local=original^roi
        nearby_fixed=[part for part,row in zip(parts,rows) if not row.get('temporary',True) and bbox_intersects(part,roi)]
        nearby_fixed.extend(part for part in NEW_SPHERE_FIXED if bbox_intersects(part,roi))
        local=source_local+union(nearby_fixed)
        if p[2]<.40:
            # A perimeter tip missing z=0 is seated directly on the plate.
            # Only one model contact exists; the bottom end is the plate.
            radius=.12
            pad=profile([p[0],p[1],0.],[(0.,radius),(max(.001,p[2]-.025),radius),(p[2]+.08,radius)])
            contact=pad^local
            if volume(contact)>1e-8:
                retained=pad.trim_by_plane((0,0,1),max(0.,p[2]-.025))
                rem=pad-retained
                commit(pad,retained,rem,dict(kind='plate_perimeter_tack',target_mm=p.tolist(),root_mm=[p[0],p[1],0.],
                    candidate_vertex=e['vertex_index'],birth_lifetime_mm=e['surface_lifetime_mm'],body_diameter_mm=2*radius,
                    model_contact_volume_mm3=volume(contact),cut_method='Trim/finish the short plate-root tack from the underside; one model contact, no separate lower neck.',
                    extraction_direction=[0,0,-1],extraction_distance_mm=.5,neck_count=0,passed=True))
                continue
        if not e.get('support_purpose'):
            micro=micro_reinforcement(surface,original,local,keepout,parts,p,env,
                maximum_gap=.65 if e['vertex_index'] in fixed_gap_exception_ids else .55)
            if micro is None:micro=terminal_micro_reinforcement(surface,local,keepout,parts,rows,p,env)
            if micro is not None:
                part,row=micro;row.update(candidate_vertex=e['vertex_index'],birth_lifetime_mm=e['surface_lifetime_mm'])
                commit(part,part,mf.Manifold(),row);continue
        roots=surface.roots(p,max_length=10.)
        for distance,q,norm,face in roots:
            if not fixed_root_candidate(q):continue
            for style,rbody,rneck in [('pitched',.30,.175),('pitched',.25,.15),('short_axial',.15,.135),('compact',.18,.10)]:
                try:
                    if style=='compact':full,necks,nr,nt,convex,cfg=make_compact_tack(q,p,rbody,rneck)
                    elif style=='short_axial':full,necks,nr,nt,convex,cfg=make_axial_short_tack(q,p)
                    else:full,necks,nr,nt,convex,cfg=make_twig(q,p,rbody,rneck,norm,e['outward_area_weighted_normal'])
                except ValueError:continue
                rneck=cfg.get('neck_radius',rneck);rbody=cfg.get('body_radius',rbody)
                minor_width=cfg.get('minimum_neck_minor_width_lower_bound_mm',2*rneck*math.cos(math.pi/16))
                if minor_width<.20-1e-10:raise ValueError('Temporary neck thinner than release floor')
                cut,ret,limit=cfg['cut'],cfg['retained'],cfg['contact_limit']
                ok,gap=sphere_envelope_ok(full,env)
                if not ok:reason_counts['moving_envelope']=reason_counts.get('moving_envelope',0)+1;continue
                full,floor=seated(full,local,q)
                if full is None:reason_counts['root_seat']=reason_counts.get('root_seat',0)+1;continue
                if any(volume(x^local)>1e-9 for x in necks):reason_counts['neck_bypass']=reason_counts.get('neck_bypass',0)+1;continue
                allowed=union([low_axis(full,q,nr,limit),low_axis(full,p,nt,limit)])
                contact=full^local
                stray=volume(contact-allowed)
                if stray>1e-8:reason_counts['unintended_contact']=reason_counts.get('unintended_contact',0)+1;continue
                joined=False
                for other in parts:
                    if bbox_intersects(full,other) and volume((full^other)-local)>1e-8:joined=True;break
                if joined:reason_counts['joins_previous_support']=reason_counts.get('joins_previous_support',0)+1;continue
                rootcontact=volume(low_axis(contact,q,nr,limit))
                tipcontact=volume(low_axis(contact,p,nt,limit))
                if min(rootcontact,tipcontact)<1e-7:reason_counts['contact_too_small']=reason_counts.get('contact_too_small',0)+1;continue
                retained=union([low_axis(full,q,nr,ret),low_axis(full,p,nt,ret)])
                if volume(retained^keepout)>1e-9:reason_counts['service_stump']=reason_counts.get('service_stump',0)+1;continue
                removable=high_axis(high_axis(full,q,nr,cut),p,nt,cut)
                nominal_stumps=full-removable
                removable_convex=[high_axis(high_axis(piece,q,nr,cut),p,nt,cut).trim_by_plane((0,0,1),floor) for piece in convex]
                removable_convex=[piece for piece in removable_convex if volume(piece)>1e-12]
                if args.skip_extraction:
                    plan=dict(mode='unchecked_diagnostic',cuts=[],fragments=[(removable,paths_at((q+p)/2)[0],20.,None)])
                else:plan=removal_plan(removable,removable_convex,original,[nominal_stumps]+NEW_SPHERE_FIXED+parts,q,p,
                    release_planes=[(q+nr*cut,nr),(p+nt*cut,nt)])
                if plan is None:reason_counts['extraction']=reason_counts.get('extraction',0)+1;continue
                prefix_ok,prefix=add_prefix_checks(full,local,q[2]+.30)
                if not prefix_ok:reason_counts['prefix']=reason_counts.get('prefix',0)+1;continue
                if len(full.decompose())!=1:reason_counts['topology']=reason_counts.get('topology',0)+1;continue
                commit(full,retained,removable,dict(kind='span_stabilization_twig' if e.get('support_purpose') else 'local_holder_twig',target_mm=p.tolist(),root_mm=q.tolist(),
                    candidate_vertex=e['vertex_index'],birth_lifetime_mm=e['surface_lifetime_mm'],root_face=face,
                    root_search_ray_density=args.root_ray_density,
                    span_stabilization={k:v for k,v in e.items() if k.startswith('member_') or 'span_mm' in k},
                    body_diameter_mm=2*rbody,neck_diameter_mm=2*rneck,neck_length_mm=cfg['neck_length'],neck_count=2,
                    support_style=cfg['style'],
                    endpoint_contact_head_diameters_mm=cfg.get('endpoint_contact_head_diameters_mm'),
                    contact_head_growth_margin=cfg.get('contact_head_growth_margin'),
                    minimum_neck_minor_width_lower_bound_mm=minor_width,
                    nominal_cut_offsets_along_outward_axis_mm=[cut,cut],retained_offsets_along_outward_axis_mm=[ret,ret],
                    neck_axes=[nr.tolist(),nt.tolist()],
                    neck_centers_mm=[(q+nr*cut).tolist(),(p+nt*cut).tolist()],
                    root_contact_volume_mm3=rootcontact,tip_contact_volume_mm3=tipcontact,
                    unintended_contact_volume_mm3=stray,moving_envelope_clearance_mm=gap,
                    moving_clearance_method='independent_complete_solid_rotation_envelope_and_actual_internal_wire_exclusion' if CERTIFIED_MOTION else 'conservative_spherical_bound',
                    fixed_attachment_dependencies=[dict(support_index=row['index'],contact_volume_mm3=volume(full^part)) for part,row in zip(parts,rows) if not row.get('temporary',True) and bbox_intersects(full,part) and volume(full^part)>1e-9],
                    sphere_fixed_attachment_dependencies=[dict(sphere_index=idx,contact_volume_mm3=volume(full^part)) for idx,part in enumerate(NEW_SPHERE_FIXED) if bbox_intersects(full,part) and volume(full^part)>1e-9],
                    _removal_plan=plan,
                    extraction_order='Reverse support index: each part was cleared against all earlier-added holder supports.',
                    root_seating_floor_z_mm=floor,prefix_checks=prefix,
                    original_model_service_paths_retained_overlap_mm3=0.,passed=not args.skip_extraction))
                accepted=True;break
            if accepted:break
        if not accepted:
            failed.append(dict(candidate_vertex=e['vertex_index'],position_mm=p.tolist(),lifetime_mm=e['surface_lifetime_mm'],
                               ray_root_candidates=len(roots),rejection_counts=reason_counts))
            print('unresolved',serial,np.round(p,3),reason_counts,flush=True)

    if stability and lip_station_coverage is None:lip_station_coverage=verified_lip_support_stations(parts,rows,original)
    if lip_station_coverage is not None:
        accepted_ids={row.get('candidate_vertex') for row in rows}
        recorded={(row['member_index'],row['member_fraction']) for row in redundant_stability}
        for event in stability:
            key=(event['member_index'],event['member_fraction'])
            if event['member_kind']=='lip_spoke' and event['vertex_index'] not in accepted_ids and key not in recorded:
                coverage=lip_station_coverage[event['member_index']]
                if coverage['passed']:
                    redundant_stability.append(dict(member_kind='lip_spoke',member_index=event['member_index'],member_fraction=event['member_fraction'],
                        reason='All three existing embroidery tacks have validated lower supports and finite actual tack-to-spoke joints.',
                        maximum_projected_gap_mm=coverage['maximum_projected_gap_between_supported_stations_mm']))
                    recorded.add(key)

    # Foundation implementation is separate because the supported foot belongs
    # to the detachable sphere tree, and its whole flat first section is seated.
    foundation_failures=[]
    if not args.skip_foundations:
        for index,s in enumerate(SPHERES):
            manifests=list((ROOT/'cache/custom_sphere_supports').glob(f"sphere_{s['layers']}layers*.json"))
            explicitly_used=any(json.loads(path.read_text()).get('foundation_foot',{}).get('used') is True for path in manifests)
            if not explicitly_used:continue
            footpath=ROOT/'cache/custom_sphere_supports'/f"sphere_{s['layers']}layers_anchor_foot.npz"
            if not footpath.exists():foundation_failures.append(dict(index=index,reason='foot_mesh_missing'));continue
            foot,_,_=load(footpath);foot=foot.transform(sphere_transform(s));bb=np.asarray(foot.bounding_box())
            p=(bb[:3]+bb[3:])/2;p[2]=bb[2]+.45
            target=p.copy();raypoint=p.copy();raypoint[2]=bb[2]-.65
            accepted=False
            for _,q,norm,face in surface.roots(raypoint,max_length=12.,minimum_drop=.75):
                if not fixed_root_candidate(q):continue
                nr=pitched_normal(norm,1);nz=nr[2]
                cr=(.60+.35*math.sqrt(1-nz*nz)+.22*nz+.045)/nz
                rootstations=[(-.22,.55),(.015,.55),(.08,.35),(.22,.35),(.40,.60),(cr,.60)]
                root=axial_profile(q,nr,rootstations)
                top=profile([p[0],p[1],0.],[(bb[2]-.80,.60),(bb[2]-.60,.65),(p[2],.65)])
                a=mf.Manifold.sphere(.60,20).translate(q+nr*cr);b=mf.Manifold.sphere(.65,20).translate([p[0],p[1],bb[2]-.2])
                body=mf.Manifold.batch_hull([a,b]);full=root+body+top
                full,floor=seated(full,original,q)
                if full is None:continue
                neck=axial_profile(q,nr,[(.08,.35),(.22,.35)])
                if volume(neck^original)>1e-9:continue
                if volume(neck^body)>1e-9:continue
                contact=full^original;allowed=low_axis(full,q,nr,.075)
                if volume(contact-allowed)>1e-8:continue
                if volume(full^foot)<.1:continue
                firstfoot=foot.trim_by_plane((0,0,-1),-float(bb[2]+.01))
                if volume(firstfoot-full)>1e-9:continue
                ok,gap=sphere_envelope_ok(full,env)
                if not ok:continue
                retained=low_axis(full,q,nr,.20)
                if volume(retained^keepout)>1e-9:continue
                removable=high_axis(full,q,nr,.15)+foot
                convex=[body,top,foot]+[axial_profile(q,nr,[aa,bb]) for aa,bb in zip(rootstations,rootstations[1:])]
                convex=[high_axis(piece,q,nr,.15).trim_by_plane((0,0,1),floor) for piece in convex]
                convex=[piece for piece in convex if volume(piece)>1e-12]
                nominal_stumps=full-high_axis(full,q,nr,.15)
                plan=removal_plan(removable,convex,original,parts+[nominal_stumps],q,p)
                if plan is None:continue
                pre_ok,prefix=add_prefix_checks(full,original,q[2]+.30)
                if not pre_ok:continue
                commit(full,retained,removable,dict(kind='sphere_foundation',sphere_index=index,layer_count=s['layers'],
                    target_mm=p.tolist(),root_mm=q.tolist(),body_diameter_mm=1.2,neck_diameter_mm=.7,neck_length_mm=.14,
                    neck_count=1,neck_centers_mm=[(q+nr*.15).tolist()],neck_axes=[nr.tolist()],root_contact_volume_mm3=volume(contact),
                    foot_overlap_volume_mm3=volume(full^foot),foot_bounds_mm=bb.tolist(),
                    foot_first_0_01_mm_uncovered_volume_mm3=volume(firstfoot-full),
                    _removal_plan=plan,prefix_checks=prefix,
                    root_seating_floor_z_mm=floor,moving_envelope_clearance_mm=gap,
                    cut_method='Remove sphere twigs first, then cut the foundation root neck and remove the remaining foot assembly.',passed=True))
                accepted=True;break
            if not accepted:foundation_failures.append(dict(index=index,reason='no_clear_root_route',target_mm=p.tolist()))

    route_validation=(revalidate_routes(parts,rows,original) if args.local_route_revalidation else
        dict(performed=False,authoritative_required_report='external_removal_v6.json',
             scope='Per-candidate routes exclude the immutable original, prior holder additions and own nominal ends. The independent final planner must check all fixed stock, all nominal stumps, all remaining supports and the complete exit path.'))
    allparts=union(parts);allstumps=union(stumps)
    micros=union([p for p,r in zip(parts,rows) if r['kind']=='fixed_micro_reinforcement'])
    save(OUT/'holder_supports.npz',allparts);save(OUT/'holder_retained_stumps.npz',allstumps)
    save(OUT/'holder_micro_reinforcements.npz',micros)
    report=dict(version='V6',build_direction=[0,0,1],units='mm',input_sha256=sha(source),
        generator_sha256=sha(Path(__file__)),selected_lifetime_threshold_mm=args.min_lifetime,
        actual_material_birth_candidates=len(events),selected_candidates=len(selected),supports=len(rows),
        resumed_previous_part_count=resume_count,
        temporary_support_count=sum(r.get('temporary',True) for r in rows),
        permanent_micro_reinforcement_count=sum(r['kind']=='fixed_micro_reinforcement' for r in rows),
        permanent_micro_reinforcement_volume_mm3=volume(micros),
        certified_motion_context=MOTION_CERTIFICATE,
        own_stump_and_final_fixed_join_removal_validation=route_validation,
        connected_span_candidates=len(stability),connected_span_proposals_without_clear_actual_surface=stability_skipped,
        existing_lip_tack_support_coverage=lip_station_coverage,
        connected_span_coverage=span_coverage_report(stability,stability_skipped,rows,lip_station_coverage),
        current_pass_root_ray_density=args.root_ray_density,
        redundant_stability_contacts_omitted=redundant_stability,
        unresolved_selected_candidates=failed,foundation_failures=foundation_failures,parts=rows,
        support_mesh_file='holder_supports.npz',retained_stumps_mesh_file='holder_retained_stumps.npz',
        support_volume_mm3=volume(allparts),elapsed_seconds=time.monotonic()-started,
        generation_checks_passed=not failed and not foundation_failures and all(r['passed'] for r in rows),
        passed=False,
        full_model_build_validation='Independent whole-model birth audit and combined external-removal plan required separately before release.',
        scope=['Generated supports only; original mesh unchanged.',
               'Whole-model island audit is independent and required; local contact tests do not prove rooted material beneath each contact.',
               'Larger external twigs are removed before original internal wire supports.',
               'Physical peel forces, resin strength, exposure and removal force are untested.'])
    (OUT/'holder_supports.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports/custom_holder_supports.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'generation_checkpoint.json').write_text(json.dumps(dict(complete=True,parts=rows,
        unresolved_selected_candidates=failed,source_sha256=sha(source),
        generator_sha256=sha(Path(__file__)),aggregate_report='holder_supports.json'),indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['parts','unresolved_selected_candidates']},indent=2),flush=True)

if __name__=='__main__':main()

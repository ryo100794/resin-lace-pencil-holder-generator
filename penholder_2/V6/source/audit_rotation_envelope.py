#!/usr/bin/env python3
"""Independently certify V6 continuous-motion outer envelopes.

The fixed sphere generator is not imported. Actual retained-layer meshes and
opposed journal measurements determine a support function for every allowed
axial rotation, accumulated joint tilt and centre translation. Every facet
of the exported convex envelope is checked against that support function.
The added-material checker uses complete solid intersections and minimum
triangle-surface distances, never just support vertices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import manifold3d as mf
from threadpoolctl import threadpool_limits
import audit_motion

ROOT=Path(__file__).resolve().parent.parent
EPSILON_MM=0.00003
EXTRA_GUARD_NUMERICAL_PADDING_MM=0.000001
ANGLE_ROUND_DECIMALS=9


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def load_mesh(path):
    with np.load(path,allow_pickle=False) as data:
        v=np.asarray(data['vertices'],float);f=np.asarray(data['faces'],np.uint32)
    require(np.isfinite(v).all(),'Nonfinite mesh coordinates')
    solid=mf.Manifold(mf.Mesh64(v,f))
    require(solid.status()==mf.Error.NoError,'Invalid manifold: '+str(path))
    return solid,v,f


def save_mesh(path,solid):
    path=Path(path);mesh=solid.to_mesh64();temp=path.with_suffix('.part')
    with temp.open('wb') as stream:
        np.savez_compressed(stream,vertices=np.asarray(mesh.vert_properties[:,:3],float),faces=np.asarray(mesh.tri_verts,np.uint32))
    temp.replace(path)
    return dict(file=str(path.relative_to(ROOT)),sha256=sha(path),stored_vertices=len(mesh.vert_properties),
        stored_triangles=len(mesh.tri_verts),volume_mm3=float(solid.volume()),bounds_mm=list(solid.bounding_box()))


def source_bounds(n,root=ROOT):
    """Re-measure journals and derive conservative joint/vertex bounds."""
    root=Path(root);cache=root/'cache/spheres';meta_path=cache/f'sphere_{n}layers.json'
    report_path=root/'reports/motion_validation.json';report=json.loads(report_path.read_text())
    require(report.get('passed') is True,'Original motion validation did not pass')
    prior=next(x for x in report['modules'] if x['layer_count']==n)
    meta=json.loads(meta_path.read_text());axis=np.asarray(meta['axis_unit_vector'],float)
    require(np.allclose(axis,[1,0,0],atol=1e-12,rtol=0),'Expected common local X journal axis')
    require(meta['permitted_rotation_degrees']==360 and meta['rotational_degrees_of_freedom_per_interface']==1,'Unexpected rotation freedom')
    require(prior['metadata_sha256']==sha(meta_path),'Original motion report metadata hash mismatch')
    names=meta['released_with_stumps_layer_meshes']
    meshes=[audit_motion.Mesh(cache/name) for name in names]
    manifest={str(meta_path.relative_to(root)):sha(meta_path),str(report_path.relative_to(root)):sha(report_path),
        'source/audit_motion.py':sha(root/'source/audit_motion.py')}
    for name,mesh in zip(names,meshes):
        path=cache/name;digest=sha(path);manifest[str(path.relative_to(root))]=digest
        declared=next(x for x in prior['actual_with_stumps_layer_topology'] if x['file']==name)
        require(declared['sha256']==digest,'Retained moving-layer geometry changed')
    measured=[]
    for bearing in meta['bearings']:
        require(np.allclose(bearing['axis_unit_vector'],axis,atol=1e-12,rtol=0),'Noncoaxial nominal interface')
        require(bearing['moving_index']==bearing['host_index']+1,'Nonsequential joint chain')
        for key in ('mouth_point','stop_point','pin_tip_point'):
            point=np.asarray(bearing[key],float)
            require(np.linalg.norm(point[1:])<1e-12,'Off-axis journal coordinate')
        opposite=[b for b in meta['bearings'] if b['host_index']==bearing['host_index'] and b['moving_index']==bearing['moving_index'] and b['side']==-bearing['side']]
        require(len(opposite)==1,'Missing opposed bearing')
        row=audit_motion.bearing_measurements(bearing,opposite[0],meshes,axis)
        require(row.get('passed') is True,'Actual opposed journal measurement/retention failed')
        measured.append(row)
    joints=[];cum_theta=0.;cum_play=0.;layer_data=[];layer_rows=[]
    for index in range(1,n):
        pair=[b for b in measured if b['moving_index']==index]
        require(len(pair)==2 and {b['side'] for b in pair}=={-1,1},'Expected two opposed journal mouths')
        radial=max(max(0.,b['conservative_maximum_radial_play_mm']) for b in pair)
        mouths=[b['actual_mouth_axial_mm']-EPSILON_MM for b in pair]
        require(min(mouths)>0,'Invalid positive journal lever arm')
        theta=max(math.atan2(max(0.,b['conservative_maximum_radial_play_mm']),m) for b,m in zip(pair,mouths))
        axial=max(max(0.,b['axial_endplay_mm'])+EPSILON_MM+(max(0.,b['actual_pin_tip_axial_mm'])+EPSILON_MM)*(1-math.cos(theta)) for b in pair)
        # The centre remains between the opposed mouth planes. Its radial
        # location is a convex combination of the two constrained pin-axis
        # intersections, so radial <= max opposed radial play.
        require(axial<min(mouths),'Centre may leave opposed-mouth interval')
        measured_bound=max(math.hypot(axial,radial),*(b['conservative_center_displacement_per_interface_mm'] for b in pair))
        prior_bound=prior['measured_play_bounds_by_interface'][index-1]['float32_padded_center_displacement_bound_mm']
        play=max(measured_bound,float(prior_bound))
        cum_theta+=theta;cum_play+=play
        joints.append(dict(host_index=index-1,moving_index=index,opposed_mouth_lower_bounds_mm=mouths,
            radial_centre_bound_mm=radial,axial_centre_bound_mm=axial,joint_tilt_bound_radians=theta,
            joint_tilt_bound_degrees=math.degrees(theta),centre_displacement_bound_mm=play,
            prior_certified_centre_displacement_mm=prior_bound,
            retained_engagement_minimum_mm=min(b['retained_engagement_with_conservative_tilt_mm'] for b in pair)))
        v=meshes[index].vertices;s=np.abs(v[:,0]);r=np.hypot(v[:,1],v[:,2])
        length=np.hypot(s,r);beta=np.arctan2(r,s)
        layer_data.append((length,beta,cum_theta,cum_play))
        layer_rows.append(dict(layer_index=index,mesh_file=names[index],mesh_sha256=manifest[str((cache/names[index]).relative_to(root))],
            cumulative_axis_tilt_radians=cum_theta,cumulative_axis_tilt_degrees=math.degrees(cum_theta),
            cumulative_centre_displacement_mm=cum_play,maximum_vertex_norm_mm=float(length.max()),
            maximum_raw_axis_radius_mm=float(r.max()),maximum_raw_axial_absolute_mm=float(s.max())))
    return layer_data,dict(layer_count=n,axis=[1,0,0],independent_axial_rotation_degrees=360,
        measured_bearings=measured,interfaces=joints,moving_layers=layer_rows,input_sha256=manifest,
        export_coordinate_allowance_mm=EPSILON_MM)


def support_function(phi,layer_data):
    angles=np.atleast_1d(np.asarray(phi,float));result=np.full(len(angles),-np.inf)
    for start in range(0,len(angles),32):
        angle=angles[start:start+32];best=np.full(len(angle),-np.inf)
        for length,beta,theta,play in layer_data:
            angular_error=np.maximum(np.abs(beta[:,None]-angle[None,:])-theta,0.)
            values=(length[:,None]*np.cos(angular_error)).max(axis=0)+play+EPSILON_MM
            best=np.maximum(best,values)
        result[start:start+len(angle)]=best
    return result


def face_planes(solid):
    mesh=solid.to_mesh64();v=np.asarray(mesh.vert_properties[:,:3],float);f=np.asarray(mesh.tri_verts,np.uint32)
    tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal,axis=1)
    valid=length>0
    # Exactly zero-area triangles impose no halfspace. The authoritative
    # solid is produced by Manifold.hull(), and all nonzero facets are checked.
    normal=normal[valid]/length[valid,None];offset=np.einsum('ij,ij->i',normal,tri[valid,0])
    require(np.isfinite(normal).all() and np.all(offset>0),'Hull must contain the origin strictly')
    return v,f,normal,offset,int((~valid).sum())


def certify_faces(solid,layer_data,additional_margin):
    vertices,faces,normals,offsets,zero=face_planes(solid)
    phi=np.arctan2(np.hypot(normals[:,1],normals[:,2]),np.abs(normals[:,0]))
    rounded=np.round(phi,ANGLE_ROUND_DECIMALS);angles,inverse=np.unique(rounded,return_inverse=True)
    # Angular grouping is padded by a support-function Lipschitz bound;
    # therefore it cannot lower a required facet offset.
    norm_bound=max(float(length.max())+play+EPSILON_MM for length,beta,theta,play in layer_data)
    rounding_pad=norm_bound*np.abs(phi-rounded)
    required=support_function(angles,layer_data)[inverse]+rounding_pad+additional_margin
    slack=offsets-required;worst=int(np.argmin(slack))
    return dict(passed=bool(np.all(slack>=0.)),nonzero_facets_checked=len(normals),
        exact_zero_area_nonbounding_triangles=zero,distinct_axial_radial_directions_checked=len(angles),
        additional_motion_margin_mm=additional_margin,minimum_face_plane_slack_mm=float(slack.min()),
        maximum_angular_grouping_upper_padding_mm=float(rounding_pad.max(initial=0.)),
        worst_face_normal=normals[worst].tolist(),worst_plane_offset_mm=float(offsets[worst]),
        worst_required_offset_mm=float(required[worst]),
        method='Every nonzero outward convex-hull facet offset is at least the independently recomputed exact continuous-motion support function for its normal, including angle-grouping upper padding.')


def envelope_to_guard_margin(envelope,guard):
    vertices=np.asarray(envelope.to_mesh64().vert_properties[:,:3],float)
    _,_,normals,offsets,_=face_planes(guard);minimum=math.inf
    for start in range(0,len(normals),256):
        normal=normals[start:start+256];support=(vertices@normal.T).max(axis=0)
        minimum=min(minimum,float((offsets[start:start+len(normal)]-support).min()))
    return minimum


def load_certified_envelopes(root=ROOT,guards=False):
    root=Path(root);report_path=root/'reports/rotation_envelope_certificate.json'
    report=json.loads(report_path.read_text());require(report.get('passed') is True,'Motion envelope certificate did not pass')
    for path,digest in report['input_sha256'].items():
        require(sha(root/path)==digest,'Motion certificate input changed: '+path)
    result={}
    for row in report['modules']:
        item=row['guard_mesh'] if guards else row['envelope_mesh'];path=root/item['file']
        require(sha(path)==item['sha256'],'Certified envelope bytes changed')
        # Match the convex representation whose actual facets were certified.
        result[row['layer_count']]=load_mesh(path)[0].hull()
    return result


def check_added_material(added,envelope,guard,margin=.025,search_length=1.):
    """For complete added solids in the same frame as placed envelopes."""
    require(margin>=.025,'Do not reduce the declared V6 minimum margin')
    require(added.status()==mf.Error.NoError,'Invalid added-material mesh')
    if added.is_empty():return dict(passed=True,empty_added_material=True)
    intersection=added^envelope;guard_intersection=added^guard
    require(intersection.status()==mf.Error.NoError and guard_intersection.status()==mf.Error.NoError,'Intersection failed')
    gap=float(added.min_gap(envelope,search_length));guard_gap=float(added.min_gap(guard,search_length))
    # The guard contains the full certified envelope plus a radius-margin
    # ball. Positive clearance to it thus proves at least margin clearance.
    # Loaded authoritative guards are proved for at least 0.025 mm. A
    # caller asking for a stricter value (e.g. 0.08 mm holder supports)
    # must satisfy the measured certificate distance independently.
    certified_guard_margin=.025
    lower=max(gap,certified_guard_margin+guard_gap)-EPSILON_MM if guard_intersection.is_empty() else gap-EPSILON_MM
    required_gap=margin+(EPSILON_MM if margin>certified_guard_margin else 0.)
    passed=bool(intersection.is_empty() and guard_intersection.is_empty() and gap>EPSILON_MM and gap>=required_gap)
    return dict(passed=passed,intersection_empty=intersection.is_empty(),intersection_volume_mm3=float(intersection.volume()),
        guard_intersection_empty=guard_intersection.is_empty(),guard_intersection_volume_mm3=float(guard_intersection.volume()),
        computed_envelope_minimum_gap_mm=gap,computed_guard_minimum_gap_mm=guard_gap,
        search_distance_cap_mm=search_length,envelope_distance_capped=bool(gap>=search_length-1e-12),
        guard_distance_capped=bool(guard_gap>=search_length-1e-12),certified_guard_margin_mm=certified_guard_margin,
        requested_minimum_mesh_clearance_mm=margin,required_computed_envelope_gap_mm=required_gap,
        numerical_distance_allowance_mm=EPSILON_MM,continuous_motion_clearance_lower_bound_mm=max(0.,lower),
        method='Complete-solid Boolean exclusions and complete triangular-surface min_gap against independently certified convex envelopes; capped distances are lower bounds.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,default=ROOT/'reports/rotation_envelope_certificate.json')
    parser.add_argument('--source',type=Path,default=ROOT/'cache/custom_sphere_supports');parser.add_argument('--margin',type=float,default=.025)
    args=parser.parse_args();require(args.margin>=.025,'Do not reduce the declared V6 guard margin')
    started=time.monotonic();out=ROOT/'cache/rotation_envelopes';out.mkdir(parents=True,exist_ok=True)
    rows=[];inputs={'source/audit_rotation_envelope.py':sha(__file__)}
    with threadpool_limits(limits=2):
        for n in (3,4,5):
            layer_data,basis=source_bounds(n);inputs.update(basis.pop('input_sha256'))
            provided_json=args.source/f'sphere_{n}layers_motion_envelope.json';provided=json.loads(provided_json.read_text())
            raw_path=args.source/provided['mesh_file'];guard_path=args.source/provided['guard_mesh_file']
            raw_hash=sha(raw_path);guard_hash=sha(guard_path)
            require(raw_hash==provided['mesh_sha256'] and guard_hash==provided['guard_mesh_sha256'],'Generator envelope input changed during read')
            raw=load_mesh(raw_path)[0].hull();raw_guard=load_mesh(guard_path)[0].hull()
            require(sha(raw_path)==raw_hash and sha(guard_path)==guard_hash,'Generator rewrote envelope during snapshot')
            # A tiny outward homothety makes the independently checked
            # envelope-plus-ball containment strict despite roundoff.
            _,_,_,offsets,_=face_planes(raw_guard)
            scale=1.+EXTRA_GUARD_NUMERICAL_PADDING_MM/float(offsets.min())
            guard=raw_guard.scale((scale,scale,scale)).hull()
            envelope_record=save_mesh(out/f'sphere_{n}layers_certified_envelope.npz',raw)
            guard_record=save_mesh(out/f'sphere_{n}layers_certified_guard.npz',guard)
            # Certify exactly the authoritative stored representations.
            envelope=load_mesh(ROOT/envelope_record['file'])[0].hull();guard=load_mesh(ROOT/guard_record['file'])[0].hull()
            face_proof=certify_faces(envelope,layer_data,0.)
            guard_proof=certify_faces(guard,layer_data,args.margin)
            separation=envelope_to_guard_margin(envelope,guard)
            passed=bool(face_proof['passed'] and guard_proof['passed'] and separation>=args.margin)
            row=dict(**basis,source_generated_envelope_sha256=raw_hash,source_generated_guard_sha256=guard_hash,
                envelope_mesh=envelope_record,guard_mesh=guard_record,envelope_facet_certificate=face_proof,
                guard_facet_certificate=guard_proof,guard_contains_envelope_plus_ball_radius_mm=separation,
                required_additional_guard_margin_mm=args.margin,guard_outward_scale_for_numerics=scale,
                guard_minimum_outward_numerical_padding_mm=EXTRA_GUARD_NUMERICAL_PADDING_MM,passed=passed)
            rows.append(row);print('certified',n,'passed',passed,'facet slack',face_proof['minimum_face_plane_slack_mm'],'guard radius',separation,flush=True)
    result=dict(passed=all(x['passed'] for x in rows),modules=rows,input_sha256=inputs,
        independent_formula='For vertex length L and beta=atan2(hypot(Y,Z),abs(X)), unit axial-radial direction phi has support L*cos(max(abs(beta-phi)-sum_joint_tilt,0))+sum_centre_displacement+export_allowance. Maximise independently over all retained-layer vertices and layers.',
        proof=['All nominal joint axes are measured/declaratively checked as common local X; arbitrary independent full axial rolls do not add axis deviation.',
            'Opposed captured journal mouths bound each relative axis tilt. Angular triangle inequality bounds the nested axis cone; triangle inequality bounds the sum of centre translations.',
            'For a fixed pose a linear support functional reaches its maximum on a mesh vertex. The independent maximum over vertices and all allowed poses therefore covers triangle interiors and the complete solid.',
            'Each exported convex-hull face is individually checked against that continuous-motion support maximum. Finite generator direction sampling is not relied upon as an inner approximation.',
            'The guard is independently proved to contain the entire certified envelope plus a ball of the declared margin; added solids are checked by full Boolean intersection and complete surface minimum distance.'],
        limitations=['Rigid-body geometry and the existing measured opposed journal capture assumptions only; no resin deformation, cure shrinkage, wear or measured printer tolerances.',
            'The envelope over-approximates reachable poses; its measured distance is a conservative lower bound on actual moving-part clearance, not the exact minimum over physical poses.',
            'Exactly zero-area hull triangles carry no bounding halfspace and are counted separately. All nonzero facets are certified.'],
        elapsed_seconds=time.monotonic()-started)
    temp=args.report.with_suffix('.part');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(args.report)
    require(result['passed'],'Independent motion envelope certification failed')


if __name__=='__main__':main()

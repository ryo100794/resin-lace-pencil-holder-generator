#!/usr/bin/env python3
"""Build immutable-V5 holder collision environments in sphere coordinates.

Each placed sphere is surrounded by an exact transformed clipping cube.
The original supported sphere module is subtracted, leaving nearby holder
geometry. No small residues are silently removed. A second representation
subtracts a world-float32-rounded source sphere, allowing the effect of
export quantization to be measured rather than guessed.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import manifold3d as mf
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'cache/support_environment'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as d:
        vertices=np.asarray(d['vertices'],float);faces=np.asarray(d['faces'],np.uint32)
    solid=mf.Manifold(mf.Mesh64(vertices,faces))
    if solid.status()!=mf.Error.NoError:raise ValueError('Invalid source: '+str(path))
    return solid,vertices,faces


def save(path,solid):
    if solid.status()!=mf.Error.NoError:raise ValueError('Invalid output: '+str(path))
    mesh=solid.to_mesh64()
    temporary=path.with_suffix('.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,vertices=np.asarray(mesh.vert_properties[:,:3],float),faces=np.asarray(mesh.tri_verts,np.uint32))
    with np.load(temporary,allow_pickle=False) as data:
        if len(data['faces'])!=solid.num_tri():raise ValueError('Incomplete NPZ write')
    temporary.replace(path)
    return dict(file=path.name,sha256=digest(path),triangles=solid.num_tri(),
        volume_mm3=float(solid.volume()),bounds_mm=list(solid.bounding_box()))


def transform(sphere):
    a=np.deg2rad(sphere['theta_deg'])
    rotation=np.column_stack(([-np.sin(a),np.cos(a),0],[0,0,1],[np.cos(a),np.sin(a),0]))
    center=np.array([sphere['center_radius']*np.cos(a),sphere['center_radius']*np.sin(a),sphere['z']])
    forward=np.column_stack((rotation,center))
    inverse=np.column_stack((rotation.T,-rotation.T@center))
    assert abs(np.linalg.det(rotation)-1.)<1e-12
    return forward,inverse,rotation,center


def component_diagnostics(solid):
    parts=solid.decompose()
    small=[];negative=[]
    for index,part in enumerate(parts):
        volume=float(part.volume())
        row=dict(index=index,volume_mm3=volume,triangles=part.num_tri(),bounds_mm=list(part.bounding_box()))
        if abs(volume)<1e-6:small.append(row)
        if volume<0:negative.append(row)
    return dict(component_count=len(parts),components_below_1e_minus_6_mm3=small,
        small_component_absolute_volume_mm3=sum(abs(x['volume_mm3']) for x in small),negative_components=negative,
        removed_components=0)


def birth_targets(design):
    baseline=ROOT/'reports/support_analysis_v5_baseline.json'
    audit=json.loads(baseline.read_text())
    if audit['input_sha256']!=digest(ROOT/'cache/original_v5.npz'):
        raise ValueError('Birth audit does not match original model')
    placed=[]
    for index,sphere in enumerate(design['spheres']):
        _,inverse,rotation,center=transform(sphere)
        rows=[]
        for row in audit['rows']:
            if not row.get('material_birth') or row['plate_contact']:continue
            matches=[x for x in row.get('source_surface_matches',[]) if x['surface']=='fixed_outer' and x['sphere_index']==index]
            if not matches:continue
            point=(np.asarray(row['position_mm'])-center)@rotation
            rows.append(dict(original_vertex_index=row['vertex_index'],local_position_mm=point.tolist(),
                world_position_mm=row['position_mm'],local_birth_y_mm=row['birth_z_mm']-sphere['z'],
                local_surface_join_y_mm=row['surface_join_z_mm']-sphere['z'],
                surface_lifetime_mm=row['surface_lifetime_mm'],probe_cap_volume_mm3=row['signed_capped_surface_volume_mm3'],
                local_area_weighted_surface_normal=(np.asarray(row['outward_area_weighted_normal'])@rotation).tolist(),
                source_match=matches[0]))
        placed.append(dict(sphere_index=index,layer_count=sphere['layers'],azimuth_deg=sphere['theta_deg'],
            local_to_world=transform(sphere)[0].tolist(),world_to_local=inverse.tolist(),targets=rows))
    reports=[]
    for n in (3,4,5):
        members=[x for x in placed if x['layer_count']==n]
        all_rows=[]
        for member in members:
            for row in member['targets']:
                all_rows.append(dict(**row,sphere_index=member['sphere_index'],azimuth_deg=member['azimuth_deg']))
        points=np.array([x['local_position_mm'] for x in all_rows])
        groups=[]
        if len(points):
            tree=cKDTree(points);seen=set()
            for i in range(len(points)):
                if i in seen:continue
                # Use actual nearby source coordinates, not a newly shifted average.
                indices=sorted(set(tree.query_ball_point(points[i],.0001))-seen)
                seen.update(indices)
                groups.append(dict(representative_local_position_mm=all_rows[i]['local_position_mm'],
                    required_in_placements=sorted(set(all_rows[j]['sphere_index'] for j in indices)),
                    source_vertex_indices=[all_rows[j]['original_vertex_index'] for j in indices],
                    maximum_surface_lifetime_mm=max(all_rows[j]['surface_lifetime_mm'] for j in indices),
                    minimum_surface_lifetime_mm=min(all_rows[j]['surface_lifetime_mm'] for j in indices),
                    maximum_local_coordinate_distance_from_representative_mm=float(max(np.linalg.norm(points[j]-points[i]) for j in indices)),
                    member_indices=indices))
        result=dict(layer_count=n,units='mm',local_build_direction=[0,1,0],
            original_model_sha256=audit['input_sha256'],source_audit_sha256=digest(baseline),
            distinction='Only positive fixed-sphere material births observed in the complete placed holder are listed; outer-shell-only artificial starts are not added.',
            placements=members,all_placed_targets=all_rows,union_targets=groups,
            clustering_tolerance_mm=.0001,raw_target_count=len(all_rows),union_target_count=len(groups))
        filename=f'sphere_{n}layers_birth_targets.json';(OUT/filename).write_text(json.dumps(result,indent=2)+'\n')
        reports.append(dict(layer_count=n,file=filename,sha256=digest(OUT/filename),placed_target_count=len(all_rows),union_target_count=len(groups)))
        print('birth targets',n,len(all_rows),'deduplicated',len(groups),flush=True)
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers',nargs='+',type=int,choices=(3,4,5),default=[5,4,3])
    parser.add_argument('--raw',action='store_true',help='Also build expensive unrounded-source diagnostics and unions.')
    parser.add_argument('--force',action='store_true',help='Recreate rather than reuse complete existing matched environments.')
    args=parser.parse_args();started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    design=json.loads((ROOT/'reports/design.json').read_text())
    source=ROOT/'cache/original_v5.npz';original_hash=digest(source)
    targets=birth_targets(design)
    report_path=ROOT/'reports/support_environment.json'
    previous=json.loads(report_path.read_text()) if report_path.exists() else {}
    if previous and previous.get('original_model_sha256')!=original_hash:
        raise ValueError('Existing environment report belongs to another original model')
    records={x['sphere_index']:x for x in previous.get('placements',[])}
    original=None
    unions=previous.get('all_placement_unions',{})

    def checkpoint():
        result=dict(original_model_sha256=original_hash,generator_sha256=digest(__file__),birth_targets=targets,
            placements=[records[k] for k in sorted(records)],all_placement_unions=unions,
            precision_policy='Working environments subtract the original supported source sphere after matching world float32 print-export rounding. Raw unrounded variants are optional diagnostics. No small components are removed.',
            uses='The all-placement union is a conservative collision guard only; it must not establish connectivity or roots for an individual placement.',
            limitations=['Collision context is clipped to the recorded R+6 mm cube; complete extraction paths outside that cube need whole-model checking.',
                'Tiny quantization residues are retained and disclosed. The actual global birth target list and final whole-holder root audit remain authoritative.'],
            elapsed_seconds=time.monotonic()-started)
        complete=True
        for n in (3,4,5):
            wanted=OUT/f'sphere_{n}layers_holder_all_placements_source_float32.npz'
            if not wanted.exists() or wanted.stat().st_size==0:complete=False
        result['complete']=complete;result['passed']=complete and len(records)==9
        result['passed_scope']='Valid local collision context meshes and source-attributed target lists extracted from unchanged V5; not support or print validation.'
        temp=report_path.with_suffix('.part');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(report_path)

    for n in args.layers:
        meta=json.loads((ROOT/f'cache/spheres/sphere_{n}layers.json').read_text())
        path=ROOT/'cache/spheres'/meta['supported_mesh_file']
        module,vertices,faces=load(path);contexts=[];raw_contexts=[]
        for index,sphere in enumerate(design['spheres']):
            if sphere['layers']!=n:continue
            tick=time.monotonic();r=float(sphere['radius']);half=r+6.
            forward,inverse,rotation,center=transform(sphere);angle=int(sphere['theta_deg'])
            stem=f'sphere_{n}layers_holder_{angle}';matched_path=OUT/(stem+'_source_float32.npz')
            record=records.get(index,dict(sphere_index=index,layer_count=n,azimuth_deg=angle))
            record.update(local_build_direction=[0,1,0],local_clip_bounds_mm=[-half,-half,-half,half,half,half],
                local_to_world=forward.tolist(),world_to_local=inverse.tolist(),rotation_determinant=float(np.linalg.det(rotation)),
                original_supported_module_file=path.name,original_supported_module_sha256=digest(path))
            context=None;clipped=None
            if not args.force and matched_path.exists() and matched_path.stat().st_size>0:
                try:
                    context,_,_=load(matched_path)
                    known=record.get('world_float32_source_environment',{}).get('sha256')
                    if known and digest(matched_path)!=known:raise ValueError('Saved matched environment hash mismatch')
                    print('reuse matched',n,angle,context.num_tri(),flush=True)
                except Exception:
                    context=None
            if context is None or args.raw:
                if original is None:original,_,_=load(source)
                cube=mf.Manifold.cube((2*half,2*half,2*half),center=True)
                clipped=(original^cube.transform(forward)).transform(inverse)
            if context is None:
                world_vertices=(vertices@rotation.T+center).astype(np.float32).astype(float)
                rounded_world=mf.Manifold(mf.Mesh64(world_vertices,faces))
                record['world_float32_source_status']=str(rounded_world.status())
                if rounded_world.status()!=mf.Error.NoError:raise ValueError('Rounded own module is invalid')
                context=clipped-rounded_world.transform(inverse)
                record['world_float32_source_environment']=save(matched_path,context)
            else:
                record['world_float32_source_environment']=dict(file=matched_path.name,sha256=digest(matched_path),
                    triangles=context.num_tri(),volume_mm3=float(context.volume()),bounds_mm=list(context.bounding_box()))
            if args.raw:
                raw=clipped-module;record['raw_environment']=save(OUT/(stem+'.npz'),raw)
                record['raw_component_diagnostics']=component_diagnostics(raw);raw_contexts.append(raw)
            contexts.append(context);record['latest_elapsed_seconds']=time.monotonic()-tick;records[index]=record
            checkpoint();print('matched environment',n,angle,'ready',round(time.monotonic()-tick,2),flush=True)
        destination=OUT/f'sphere_{n}layers_holder_all_placements_source_float32.npz'
        if not args.force and destination.exists() and destination.stat().st_size>0:
            try:combined,_,_=load(destination)
            except Exception:combined=mf.Manifold.batch_boolean(contexts,mf.OpType.Add)
        else:combined=mf.Manifold.batch_boolean(contexts,mf.OpType.Add)
        unions[str(n)]=[save(destination,combined)]
        if args.raw:
            raw_union=mf.Manifold.batch_boolean(raw_contexts,mf.OpType.Add)
            unions[str(n)].append(save(OUT/f'sphere_{n}layers_holder_all_placements.npz',raw_union))
        checkpoint();print('MATCHED UNION READY',n,combined.num_tri(),flush=True)
        del contexts,raw_contexts,combined,module,vertices,faces
        gc.collect()
    if digest(source)!=original_hash:raise RuntimeError('Original model changed during extraction')
    checkpoint();print('completed selected types',args.layers,round(time.monotonic()-started,2),'seconds',flush=True)


if __name__=='__main__':main()

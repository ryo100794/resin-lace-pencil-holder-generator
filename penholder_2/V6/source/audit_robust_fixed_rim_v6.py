#!/usr/bin/env python3
"""Scope the one additional fixed-rim fill against actual mechanism stock.

This certifies only the declared additions. Complete export topology, every
material birth, and the assembled external removal plan remain release gates.
No source or owning support geometry is edited here.
"""
from pathlib import Path
import hashlib
import io
import json
import time
import numpy as np
import manifold3d as mf
import audit_motion as A
import audit_rotation_envelope as R
from build_holder import SPHERES, sphere_transform

ROOT=Path(__file__).resolve().parent.parent
INPUTS={}
DISTANCE_ALLOWANCE=.00003
STATIC_MINIMUM_GAP=.00006

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frozen(path):
    path=Path(path);data=path.read_bytes();key=str(path.relative_to(ROOT));digest=hashlib.sha256(data).hexdigest()
    if key in INPUTS and INPUTS[key]!=digest:raise ValueError('Input changed: '+key)
    INPUTS[key]=digest
    return data

def metadata(path):return json.loads(frozen(path))

def load(path):
    with np.load(io.BytesIO(frozen(path)),allow_pickle=False) as data:
        v=np.asarray(data['vertices'],float);f=np.asarray(data['faces'],np.uint32)
    m=mf.Manifold(mf.Mesh64(v,f))
    if m.status()!=mf.Error.NoError:raise ValueError('Invalid stock: '+str(path))
    return m,v

def bbox_gap(a,b):
    return float(np.linalg.norm(np.maximum(np.maximum(a[:3]-b[3:],b[:3]-a[3:]),0.)))

def positive_volume(m):
    if m.is_empty():return 0.
    value=float(m.volume())
    if value < -1e-14:raise ValueError('Unexpected negative boundary volume in positive stock')
    return max(0.,value)

def main():
    started=time.monotonic()
    source=ROOT/'reports/fixed_precision_robust_rim_candidates.json'
    candidate=metadata(source)
    if candidate.get('local_geometry_passed') is not True:raise ValueError('Local strict checks did not pass')
    rows=[];patches=[]
    for record in candidate['repairs']:
        m,_=load(ROOT/record['mesh_file'])
        if sha(ROOT/record['mesh_file'])!=record['mesh_sha256']:raise ValueError('Candidate geometry changed')
        patches.append(m)
        rows.append(dict(id=record['id'],mesh_file=record['mesh_file'],mesh_sha256=record['mesh_sha256'],
            bounds_mm=list(m.bounding_box()),categories={},persistent_fixed_contact=None))
    category_counts={};near_tests=[]

    def check(stock,category,label,minimum=STATIC_MINIMUM_GAP):
        if stock.is_empty():return
        category_counts[category]=category_counts.get(category,0)+1
        bb=np.asarray(stock.bounding_box())
        for patch,row in zip(patches,rows):
            gap=bbox_gap(np.asarray(row['bounds_mm']),bb)
            method='complete-solid AABB separation lower bound'
            exact=False
            if gap<1.:
                overlap=patch^stock
                if overlap.status()!=mf.Error.NoError:raise ValueError('Forbidden-stock Boolean failed')
                if not overlap.is_empty():raise ValueError(f'{row["id"]}: forbidden intersection {category} {label}, signed volume {overlap.volume()}')
                gap=float(patch.min_gap(stock,1.))
                method='empty complete-solid Boolean and capped complete-surface distance'
                exact=True
            lower=gap-DISTANCE_ALLOWANCE
            if lower<minimum:raise ValueError(f'{row["id"]}: insufficient {category} clearance {lower}')
            previous=row['categories'].get(category)
            item=dict(closest_stock=label,clearance_lower_bound_mm=lower,required_mm=minimum,
                      distance_allowance_mm=DISTANCE_ALLOWANCE,method=method,passed=True)
            if previous is None or lower<previous['clearance_lower_bound_mm']:row['categories'][category]=item
            if exact:near_tests.append(dict(repair_id=row['id'],category=category,stock=label,**item))

    frozen(ROOT/'source/audit_robust_fixed_rim_v6.py')
    frozen(ROOT/'source/build_robust_fixed_rim_v6.py')
    frozen(ROOT/'source/audit_rotation_envelope.py')
    frozen(ROOT/'source/audit_motion.py')
    frozen(ROOT/'source/build_holder.py')
    cert=metadata(ROOT/'reports/rotation_envelope_certificate.json')
    env=R.load_certified_envelopes(ROOT);guards=R.load_certified_envelopes(ROOT,guards=True)
    for path,digest in cert['input_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Certified envelope input changed')
        INPUTS[path]=digest
    for record in cert['modules']:
        for key in ['envelope_mesh','guard_mesh']:
            path=ROOT/record[key]['file'];frozen(path)
    for i,s in enumerate(SPHERES):
        n=s['layers'];t=sphere_transform(s)
        check(env[n].transform(t),'continuous_motion_envelope',f'placement_{i}_L{n}',.08)
        check(guards[n].transform(t),'continuous_motion_guard',f'placement_{i}_L{n}',.00006)

    cache=ROOT/'cache/spheres'
    for n in [3,4,5]:
        meta=metadata(cache/f'sphere_{n}layers.json')
        placements=[(i,sphere_transform(s))for i,s in enumerate(SPHERES)if s['layers']==n]
        for name in meta['free_layer_meshes'][1:]:
            m,_=load(cache/name)
            for i,t in placements:check(m.transform(t),'original_moving_layers',f'placement_{i}:{name}')
        for w in meta['release_wires']:
            full,_=load(cache/w['mesh_file'])
            removable,v=load(cache/w['removable_mesh_file'])
            direction=np.asarray(w.get('extraction_vector',[0,0,1]),float)
            direction/=np.linalg.norm(direction)
            distance=float(w.get('extraction_distance_mm',2*meta['outer_nominal_radius_mm']+6))
            sweep=mf.Manifold.hull_points(np.vstack((v,v+direction*distance)))
            for i,t in placements:
                check(full.transform(t),'original_internal_wires',f'placement_{i}:wire_{w["wire_index"]}')
                check(sweep.transform(t),'original_continuous_wire_extraction',f'placement_{i}:wire_{w["wire_index"]}')
            for j,endpoint in enumerate(w['endpoints']):
                corridor,_=load(cache/endpoint['tool_corridor_mesh_file'])
                tip=np.asarray(endpoint['tool_tip_end'],float)
                contact=np.asarray(endpoint.get('tool_contact_point',endpoint['neck_center']),float)
                extension=A.cylinder_between(tip,contact,float(endpoint.get('tool_tip_diameter_mm',.2))/2)
                for i,t in placements:
                    check(corridor.transform(t),'original_internal_tool_corridors',f'placement_{i}:wire_{w["wire_index"]}:endpoint_{j}')
                    check(extension.transform(t),'original_internal_tool_contact_extensions',f'placement_{i}:wire_{w["wire_index"]}:endpoint_{j}')
    print('Original mechanism, motion and internal continuous service paths checked',flush=True)

    hp=ROOT/'cache/custom_holder_supports/holder_supports.json'
    holder=metadata(hp);folder=hp.parent
    dependency_indices={x['index']for row in candidate['repairs']for x in row['nearby_existing_fixed_context']if x['owner']=='holder'}
    fixed_dependencies={}
    for r in holder['parts']:
        full,_=load(folder/r['full_mesh_file'])
        if r['kind']=='fixed_micro_reinforcement':
            if r['index']in dependency_indices:fixed_dependencies[r['index']]=full
            continue
        removable,_=load(folder/r['removable_mesh_file'])
        check(removable,'new_holder_nominal_removable',f'H{r["index"]:04d}')
        centers=r.get('neck_centers_mm',[]);axes=r.get('neck_axes',[])
        if len(centers)!=len(axes)or len(centers)!=r.get('neck_count',0):raise ValueError('Holder neck description incomplete')
        for j,(center,normal) in enumerate(zip(centers,axes)):
            normal=np.asarray(normal,float);normal/=np.linalg.norm(normal)
            offset=float(np.dot(center,normal));half=float(r['neck_length_mm'])/2
            neck=full.trim_by_plane(normal,offset-half).trim_by_plane(-normal,-offset-half)
            if neck.is_empty():raise ValueError('Holder declared neck band has no actual stock')
            check(neck,'new_holder_full_necks',f'H{r["index"]:04d}:neck_{j}')
        if r['index']in dependency_indices:fixed_dependencies[r['index']]=full-removable

    for n in [3,4,5]:
        folder=ROOT/'cache/custom_sphere_supports'
        meta=metadata(folder/f'sphere_{n}layers_supports.json')
        if not meta.get('passed')or not meta.get('finalization_complete'):raise ValueError('Sphere addition source is not finalized')
        placements=[(i,sphere_transform(s))for i,s in enumerate(SPHERES)if s['layers']==n]
        for r in meta['supports']:
            if r['kind']=='fixed_micro_reinforcement':continue
            full,_=load(folder/r['mesh_file']);removable,_=load(folder/r['removable_mesh_file'])
            centers=r.get('cut_points',[]);axes=r.get('cut_plane_normals',[])
            if len(centers)!=2 or len(axes)!=2:raise ValueError('Sphere neck description incomplete')
            necks=[]
            for center,normal in zip(centers,axes):
                normal=np.asarray(normal,float);normal/=np.linalg.norm(normal)
                offset=float(np.dot(center,normal));half=float(r.get('neck_length_mm',r.get('neck_length_build_mm',.14)))/2
                necks.append(full.trim_by_plane(normal,offset-half).trim_by_plane(-normal,-offset-half))
            for i,t in placements:
                check(full.transform(t),'new_sphere_entire_temporary_support',f'placement_{i}:support_{r["support_index"]}')
                check(removable.transform(t),'new_sphere_nominal_removable',f'placement_{i}:support_{r["support_index"]}')
                for j,neck in enumerate(necks):check(neck.transform(t),'new_sphere_full_necks',f'placement_{i}:support_{r["support_index"]}:neck_{j}')
    print('All new removable stock and complete neck bands excluded',flush=True)

    for patch,row,record in zip(patches,rows,candidate['repairs']):
        original,_=load(ROOT/f'cache/fixed_precision_repairs/additional_rim/{record["id"]}_original_context.npz')
        deps=[x['index']for x in record['nearby_existing_fixed_context']if x['owner']=='holder']
        persistent=mf.Manifold.batch_boolean([original]+[fixed_dependencies[i]for i in deps],mf.OpType.Add)
        # Every patch is convex. Any positive Z prefix is connected. Its
        # flat first face seats in original/nominally retained fixed stock,
        # not in a support body that is scheduled for removal.
        mm=patch.to_mesh64();hull=patch.hull()
        if positive_volume(hull-patch)>1e-12:raise ValueError('Expected a convex fixed fill')
        lower=float(patch.bounding_box()[2]);height=min(lower+.00005,float(patch.bounding_box()[5]))
        prefix=patch.trim_by_plane((0,0,-1),-height);contact=prefix^persistent
        vol=positive_volume(contact)
        if vol<=1e-12 or abs(contact.bounding_box()[2]-lower)>1e-8:raise ValueError('No persistent fixed first-prefix contact')
        row['persistent_fixed_contact']=dict(holder_support_indices=deps,nominal_retained_stock_used=True,
            original_context_file=f'cache/fixed_precision_repairs/additional_rim/{record["id"]}_original_context.npz',
            first_prefix_height_mm=height,contact_volume_mm3=vol,
            first_contact_bottom_mm=float(contact.bounding_box()[2]),patch_bottom_mm=lower,
            patch_is_convex=True,all_positive_height_prefixes_connected=True,passed=True)
        row['passed']=True
    for path,digest in INPUTS.items():
        if sha(ROOT/path)!=digest:raise ValueError('Input changed before scope freeze: '+path)
    report=dict(passed=True,scope='Only the one additional declared fixed-rim repair solid',
        global_topology_and_build_order_required=True,external_removal_plan_required=True,
        continuous_motion_requested_margin_mm=.08,static_forbidden_stock_required_gap_mm=STATIC_MINIMUM_GAP,
        numerical_distance_allowance_mm=DISTANCE_ALLOWANCE,
        all_stock_classes_checked=category_counts,repairs=rows,near_complete_surface_tests=near_tests,
        input_sha256=INPUTS,elapsed_seconds=time.monotonic()-started,
        exclusions='No intersections with all-angle moving envelopes, actual original moving/internal-wire stock, continuous declared internal extraction and tool-contact paths, or any current added nominal removable stock or complete neck band. This does not certify a cutter handle, fingers, or cutting force.')
    path=ROOT/'reports/fixed_precision_robust_rim_scope_checks.json'
    part=Path(str(path)+'.part');part.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');part.replace(path)
    print('Fixed repair scope passed:',len(rows),'pieces;',len(INPUTS),'frozen files;',report['elapsed_seconds'],'seconds',flush=True)

if __name__=='__main__':main()

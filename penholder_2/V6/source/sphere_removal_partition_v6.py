#!/usr/bin/env python3
"""Analysis-only physical neck-cut partitions of frozen sphere additions.

No printed source mesh is edited.  Two bounded wafers remove finite 0.04 mm
bands within the existing narrow waists.  Disconnected oriented boundaries
are classified by their endpoint geometry and preserved with Compose.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import manifold3d as mf
import numpy as np

import custom_sphere_supports as S
from build_holder import SPHERES, sphere_transform

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'cache/custom_sphere_supports'
OUT=ROOT/'cache/sphere_removal_partition_v6'
REPORT=ROOT/'reports/sphere_removal_partition_v6.json'
KERF_WIDTH=.04
RADIAL_ALLOWANCE=.003
VOLUME_TOL=1e-8
KERF_WIDTH_BY_PART={(3,3):.035,(5,40):.035}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compose(parts):
    """Preserve signed disconnected surfaces, including tiny cavity shells."""
    parts=[p for p in parts if not p.is_empty()]
    return mf.Manifold.compose(parts) if parts else mf.Manifold()


def write_json(path,data):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_name(path.name+'.part');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def endpoint_cutters(row,width=KERF_WIDTH,radial_allowance=RADIAL_ALLOWANCE):
    cutters=[];records=[]
    for j,point in enumerate(row['cut_points']):
        point=np.asarray(point,float);axis=np.asarray(row['cut_plane_normals'][j],float);axis/=np.linalg.norm(axis)
        if row['style']=='rising_horizontal_sections':
            q=np.asarray(row['anchor_point']);p=np.asarray(row['contact_point']);dy=p[1]-q[1]
            axial=.10 if j==0 else dy-.10;radius=row['horizontal_neck_diameter_mm']/2
            shear=np.array([[1.,(p[0]-q[0])/dy,0.,q[0]],[0.,1.,0.,q[1]],[0.,(p[2]-q[2])/dy,1.,q[2]]])
            cutter=S.profile(np.zeros(3),S.UP,[(axial-width/2,radius+radial_allowance),(axial+width/2,radius+radial_allowance)]).transform(shear)
            axis=np.array([0.,1. if j==0 else -1.,0.])
            waist_length=row['neck_length_build_mm'];width_basis='local build-Y plane separation; cross sections follow the original shear'
        else:
            radius=row['neck_diameter_mm']/2
            cutter=S.profile(point,axis,[(-width/2,radius+radial_allowance),(width/2,radius+radial_allowance)])
            waist_length=row['neck_length_mm'];width_basis='axial plane separation along the original endpoint axis'
        if width>=waist_length:raise RuntimeError('Kerf does not fit wholly inside the existing waist')
        cutters.append(cutter)
        records.append(dict(endpoint_index=j,center_mm=point.tolist(),axis_toward_free_body=axis.tolist(),
            kerf_width_mm=width,waist_length_mm=waist_length,waist_radius_mm=radius,
            cutter_radial_allowance_mm=radial_allowance,width_basis=width_basis,
            minimum_axial_margin_to_waist_end_mm=(waist_length-width)/2,
            fixed_side_plane_offset_mm=float(point@axis-width/2),
            free_side_plane_offset_mm=float(point@axis+width/2)))
    return cutters,records


def partition(row,n,width=KERF_WIDTH):
    full=S.load(SOURCE/row['mesh_file']);maximum=S.load(SOURCE/row['conservative_retained_mesh_file'])
    cutters,cuts=endpoint_cutters(row,width)
    cutter=S.join(cutters);kerf=full^cutter;left=full-cutter
    components=left.decompose();positives=[i for i,p in enumerate(components) if p.volume()>1e-12]
    if not positives:raise RuntimeError('No positive material remains after cuts')
    assignments={};component_records=[]
    for i in positives:
        part=components[i];v=S.verts(part);outside=S.vol(part-maximum);endpoints=[]
        for cut in cuts:
            d=np.asarray(cut['axis_toward_free_body']);limit=cut['fixed_side_plane_offset_mm']
            if float(np.max(v@d-limit))<=1e-8:endpoints.append(cut['endpoint_index'])
        fixed=outside<=VOLUME_TOL and bool(endpoints)
        assignments[i]='retained' if fixed else 'removable'
        component_records.append(dict(component_index=i,signed_volume_mm3=float(part.volume()),
            assignment=assignments[i],endpoint_candidates=endpoints,volume_outside_maximum_retained_mm3=outside))
    # Never discard numerical secondary shells. Attach each to its nearest
    # positive surface and preserve the original orientation via Compose.
    for i,part in enumerate(components):
        if i in assignments:continue
        gaps=[(float(part.min_gap(components[j],100.)),j) for j in positives]
        gap,j=min(gaps);assignments[i]=assignments[j]
        component_records.append(dict(component_index=i,signed_volume_mm3=float(part.volume()),assignment=assignments[i],
            oriented_secondary_boundary_preserved=True,nearest_positive_component=j,surface_gap_mm=gap))
    fixed_parts=[p for i,p in enumerate(components) if assignments[i]=='retained']
    free_parts=[p for i,p in enumerate(components) if assignments[i]=='removable']
    retained=compose(fixed_parts);removable=compose(free_parts)
    # A free body may have several independent positive components. Keep those
    # separate for extraction, preserving associated oriented cavity shells.
    groups=[]
    for j in positives:
        if assignments[j]!='removable':continue
        secondary=[p for i,p in enumerate(components) if i not in positives and assignments[i]=='removable'
            and next(x for x in component_records if x['component_index']==i)['nearest_positive_component']==j]
        groups.append(compose([components[j]]+secondary))
    necks=[S.load(SOURCE/f'sphere_{n}layers_neck_{row["support_index"]:03d}_{j}.npz') for j in range(2)]
    checks=dict(retained_outside_maximum_mm3=S.vol(retained-maximum),kerf_outside_original_waist_mm3=S.vol(kerf-S.join(necks)),
        retained_removable_interior_overlap_mm3=S.overlap(retained,removable),
        retained_kerf_interior_overlap_mm3=S.overlap(retained,kerf),
        removable_kerf_interior_overlap_mm3=S.overlap(removable,kerf),
        sum_signed_volume_error_mm3=abs(retained.volume()+removable.volume()+kerf.volume()-full.volume()),
        partition_uncovered_printed_volume_mm3=S.vol(full-compose([retained,removable,kerf])),
        partition_outside_printed_volume_mm3=S.vol(compose([retained,removable,kerf])-full))
    if max(checks.values())>VOLUME_TOL:
        raise RuntimeError(f'Local finite-cut partition failed L{n}/{row["support_index"]}: '+json.dumps(checks))
    proxies=[]
    for filename in row.get('generated_convex_sweep_meshes',[]):
        proxy=S.load(SOURCE/filename)
        # Old proxies cover the original nominal removable body. The new free
        # body is shorter at its waists; retain the conservative old cover.
        if not proxy.is_empty():proxies.append(proxy)
    uncovered=S.vol(removable-S.join(proxies))
    if uncovered>VOLUME_TOL:raise RuntimeError(f'Old convex cover misses cut body L{n}/{row["support_index"]}: {uncovered}')
    return dict(full=full,maximum=maximum,retained=retained,removable=removable,kerf=kerf,
        cutters=cutters,fragments=groups,proxies=proxies,cuts=cuts,component_records=component_records,checks=checks,
        convex_proxy_uncovered_actual_mm3=uncovered)


def save_record(n,row,result,placement=None):
    index=row['support_index'];stem=f'S{n}_{index:03d}'
    transform=None
    if placement is not None:
        sphere=SPHERES[placement];transform=sphere_transform(sphere);stem+=f'_P{placement}'
    paths={}
    for name in ['retained','removable','kerf']:
        mesh=result[name].transform(transform) if transform is not None else result[name]
        path=OUT/(stem+'_'+name+'.npz');S.save(path,mesh);paths[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
    for key in ['cutters','fragments','proxies']:
        paths[key]=[]
        for j,mesh in enumerate(result[key]):
            if transform is not None:mesh=mesh.transform(transform)
            path=OUT/f'{stem}_{key}_{j:02d}.npz';S.save(path,mesh);paths[key].append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
    cuts=json.loads(json.dumps(result['cuts']))
    if transform is not None:
        for cut in cuts:
            point=transform[:,:3]@np.asarray(cut['center_mm'])+transform[:,3];axis=transform[:,:3]@np.asarray(cut['axis_toward_free_body'])
            cut['center_mm']=point.tolist();cut['axis_toward_free_body']=axis.tolist()
            width=cut['kerf_width_mm'];cut['fixed_side_plane_offset_mm']=float(point@axis-width/2);cut['free_side_plane_offset_mm']=float(point@axis+width/2)
    saved={key:S.load(ROOT/paths[key]['path']) for key in ['retained','removable','kerf']}
    full=result['full'].transform(transform) if transform is not None else result['full']
    maximum=result['maximum'].transform(transform) if transform is not None else result['maximum']
    checks=dict(serialized_retained_outside_maximum_mm3=S.vol(saved['retained']-maximum),
        serialized_partition_uncovered_printed_mm3=S.vol(full-compose(list(saved.values()))),
        serialized_partition_outside_printed_mm3=S.vol(compose(list(saved.values()))-full),
        serialized_volume_error_mm3=abs(sum(m.volume() for m in saved.values())-full.volume()),
        serialized_retained_removable_interior_overlap_mm3=S.overlap(saved['retained'],saved['removable']),
        serialized_retained_kerf_interior_overlap_mm3=S.overlap(saved['retained'],saved['kerf']),
        serialized_removable_kerf_interior_overlap_mm3=S.overlap(saved['removable'],saved['kerf']))
    checks['serialized_proxy_uncovered_actual_mm3']=S.vol(saved['removable']-S.join([S.load(ROOT/item['path']) for item in paths['proxies']]))
    passed=max(checks.values())<=VOLUME_TOL
    record=dict(layer_count=n,support_index=index,placement=placement,paths=paths,cuts=cuts,checks=checks,passed=passed,
        source_full_mesh_file=str((SOURCE/row['mesh_file']).relative_to(ROOT)),source_full_sha256=sha(SOURCE/row['mesh_file']),
        source_maximum_mesh_file=str((SOURCE/row['conservative_retained_mesh_file']).relative_to(ROOT)),source_maximum_sha256=sha(SOURCE/row['conservative_retained_mesh_file']))
    if placement is None:record.update(component_records=result['component_records'],construction_checks=result['checks'],convex_proxy_uncovered_actual_mm3=result['convex_proxy_uncovered_actual_mm3'])
    return record


def load_record(layer_count,support_index,placement,report_path=REPORT):
    report=json.loads(Path(report_path).read_text())
    if not report.get('passed'):raise RuntimeError('Sphere endpoint-cut partition has not passed')
    return next(r for r in report['records'] if r['layer_count']==layer_count and r['support_index']==support_index and r['placement']==placement)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layers',type=int);ap.add_argument('--index',type=int);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic();records=[];failures=[];inputs={}
    for n in ([args.layers] if args.layers else [3,4,5]):
        metadata=SOURCE/f'sphere_{n}layers_supports.json';inputs[str(metadata.relative_to(ROOT))]=sha(metadata)
        for row in json.loads(metadata.read_text())['supports']:
            if row['kind']!='removable_fixed_support' or (args.index is not None and row['support_index']!=args.index):continue
            try:
                result=partition(row,n,KERF_WIDTH_BY_PART.get((n,row['support_index']),KERF_WIDTH));local=save_record(n,row,result);records.append(local)
                for placement,sphere in enumerate(SPHERES):
                    if sphere['layers']==n:records.append(save_record(n,row,result,placement))
                bad=[r for r in records[-4:] if not r['passed']]
                if bad:failures.extend(bad)
                print(json.dumps(dict(n=n,index=row['support_index'],passed=not bad,fragments=len(result['fragments']),checks=local['checks'])),flush=True)
            except Exception as error:
                failures.append(dict(layer_count=n,support_index=row['support_index'],error=str(error)));print('FAILED',n,row['support_index'],str(error),flush=True)
    report=dict(passed=not failures,scope='Analysis-only finite endpoint-cut partition; original printed geometry is immutable. Cutter approach/extraction against complete surroundings is a separate planner gate.',
        kerf_width_mm=KERF_WIDTH,radial_cutter_allowance_mm=RADIAL_ALLOWANCE,volume_tolerance_mm3=VOLUME_TOL,
        declared_kerf_width_exceptions=[dict(layer_count=n,support_index=i,kerf_width_mm=w) for (n,i),w in KERF_WIDTH_BY_PART.items()],
        records=records,failures=failures,inputs=inputs,source_sha256=sha(__file__),seconds=time.monotonic()-started)
    target=REPORT if args.layers is None else ROOT/f'reports/sphere_removal_partition_v6_{args.layers}_{args.index}.json'
    write_json(target,report);print(json.dumps(dict(passed=report['passed'],records=len(records),failures=len(failures),seconds=report['seconds'])),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()

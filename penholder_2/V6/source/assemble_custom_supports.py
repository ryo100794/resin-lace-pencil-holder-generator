#!/usr/bin/env python3
"""Union model-specific removable additions with the unchanged V5 print mesh.

The inputs are actual solid meshes in millimetres.  Spherical-module additions
use local +Y build coordinates; the complete print uses +Z.  This script adds
material only, preserves the original mechanism, checks enclosure clearance,
and fails rather than silently repairing a damaged mechanism. The export
representation uses a separately bounded precision-preparation step.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import manifold3d as mf
from build_holder import SPHERES, sphere_transform
from audit_rotation_envelope import load_certified_envelopes,check_added_material
from prepare_v6_export import prepare

ROOT=Path(__file__).resolve().parent.parent
BASE_SHA256='de722a37883d54ad0d9cb8383aff327d08f2885eceab08ae9d96981c37ece348'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):
    with np.load(path,allow_pickle=False) as d:
        v=np.asarray(d['vertices'],dtype=np.float64)
        f=np.asarray(d['faces'],dtype=np.uint32)
    m=mf.Manifold(mf.Mesh64(v,f))
    if m.status()!=mf.Error.NoError or m.is_empty():raise ValueError('Invalid solid: '+str(path))
    return m

def save(m,path):
    mm=m.to_mesh64()
    np.savez_compressed(path,vertices=mm.vert_properties[:,:3],faces=mm.tri_verts)

def join(parts):return mf.Manifold.batch_boolean(parts,mf.OpType.Add)

def raw_component_diagnostic(solid):
    """Read-only face connectivity and centred volumes; no decomposition/filtering.

    A sparse graph avoids constructing over a thousand separate Manifold
    objects for the raw numerical boundary remnants.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    mm=solid.to_mesh64();v=np.asarray(mm.vert_properties[:,:3],float)
    f=np.asarray(mm.tri_verts,dtype=np.int32)
    rows=np.concatenate((f[:,0],f[:,0]));cols=np.concatenate((f[:,1],f[:,2]))
    graph=coo_matrix((np.ones(len(rows),dtype=np.int8),(rows,cols)),shape=(len(v),len(v))).tocsr()
    count,labels=connected_components(graph,directed=False)
    del graph,rows,cols
    first=np.full(count,len(v),dtype=np.int64)
    np.minimum.at(first,labels,np.arange(len(v)))
    origins=v[first];volumes=np.zeros(count)
    for start in range(0,len(f),32768):
        faces=f[start:start+32768];groups=labels[faces[:,0]]
        triangles=v[faces]-origins[groups,None,:]
        values=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2]))/6.
        volumes+=np.bincount(groups,weights=values,minlength=count)
    return dict(count=int(count),signed_volumes_mm3=volumes.tolist(),
        component_filtering=False,
        method='Undirected vertex/triangle connectivity graph; signed triangle volumes centred separately on each component.',
        scope='Read-only diagnostic before bounded representation preparation. Every raw face passes to preparation; strict one-positive-component gate applies to the final float32 mesh.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sphere-dir',type=Path,default=ROOT/'cache/custom_sphere_supports')
    p.add_argument('--holder',type=Path,default=ROOT/'cache/custom_holder_supports/holder_supports.npz')
    a=p.parse_args(); started=time.monotonic()
    base_path=ROOT/'cache/original_v5.npz'
    if sha(base_path)!=BASE_SHA256:raise ValueError('The released V5 input has changed')
    base=read(base_path)
    sources=[base_path,a.holder,Path(__file__),ROOT/'source/prepare_v6_export.py',
        ROOT/'source/build_test_coupon.py',ROOT/'source/build_holder.py']
    frozen={str(x):sha(x) for x in sources}
    def remember(path):
        digest=sha(path)
        if str(path) in frozen and frozen[str(path)]!=digest:
            raise ValueError('Assembly input changed during loading: '+str(path))
        frozen[str(path)]=digest
        if path not in sources:sources.append(path)
    holder_report=ROOT/'reports/custom_holder_supports.json'
    holder_generation=json.loads(holder_report.read_text())
    if holder_generation.get('generation_checks_passed',holder_generation.get('passed')) is not True:
        raise ValueError('Holder support generation has not passed')
    if holder_generation.get('unresolved_selected_candidates') or holder_generation.get('foundation_failures'):
        raise ValueError('Holder support generation still has unresolved design candidates')
    remember(holder_report)
    sphere_parts={}
    for n in (3,4,5):
        path=a.sphere_dir/f'sphere_{n}layers_supports.npz'
        metadata=a.sphere_dir/f'sphere_{n}layers_supports.json'
        sphere_report=json.loads(metadata.read_text())
        if sphere_report.get('passed') is not True or sphere_report.get('finalization_complete') is not True:
            raise ValueError('Sphere support generation has not passed: '+str(n))
        remember(metadata);remember(path)
        sphere_parts[n]=read(path)
    parts=[read(a.holder)]
    for s in SPHERES:parts.append(sphere_parts[s['layers']].transform(sphere_transform(s)))
    print('Union of actual support additions',flush=True)
    additions=join(parts)
    if additions.status()!=mf.Error.NoError:raise ValueError('Support union failed')
    save(additions,ROOT/'cache/added_supports_v6.npz')
    micro_parts=[];retained_parts=[]
    holder_retained=a.holder.parent/'holder_retained_stumps.npz'
    if not holder_retained.exists():raise ValueError('Missing holder retained-state geometry')
    remember(holder_retained);retained_parts.append(read(holder_retained))
    holder_micro=a.holder.parent/'holder_micro_reinforcements.npz'
    if holder_micro.exists():
        with np.load(holder_micro) as data:nonempty=len(data['faces'])>0
        if nonempty:remember(holder_micro);micro_parts.append(read(holder_micro))
    for s in SPHERES:
        n=s['layers'];tr=sphere_transform(s)
        retained_path=a.sphere_dir/f'sphere_{n}layers_retained_stumps.npz'
        if not retained_path.exists():raise ValueError('Missing sphere retained-state geometry')
        remember(retained_path);retained_parts.append(read(retained_path).transform(tr))
        micro_path=a.sphere_dir/f'sphere_{n}layers_micro_reinforcements.npz'
        if micro_path.exists():
            with np.load(micro_path) as data:nonempty=len(data['faces'])>0
            if nonempty:
                remember(micro_path);micro_parts.append(read(micro_path).transform(tr))
    retained=join(retained_parts)
    save(retained,ROOT/'cache/added_retained_v6.npz')
    micro=join(micro_parts) if micro_parts else mf.Manifold()
    if not micro.is_empty():save(micro,ROOT/'cache/added_fixed_micro_v6.npz')
    temporary=additions-micro
    save(temporary,ROOT/'cache/added_temporary_v6.npz')
    envelopes=load_certified_envelopes(root=ROOT,guards=False)
    guards=load_certified_envelopes(root=ROOT,guards=True)
    remember(ROOT/'reports/rotation_envelope_certificate.json')
    clearance=[]
    for s in SPHERES:
        transform=sphere_transform(s)
        checked=check_added_material(additions,envelopes[s['layers']].transform(transform),
            guards[s['layers']].transform(transform),margin=.025,search_length=1.)
        clearance.append(dict(kind=s['kind'],layers=s['layers'],azimuth_deg=s['theta_deg'],
            **checked))
    if not all(x['passed'] for x in clearance):
        (ROOT/'reports/custom_support_envelope_failure.json').write_text(json.dumps(clearance,indent=2)+'\n')
        raise ValueError('An added support enters a conservative moving envelope')
    print('Union with unchanged released print mesh',flush=True)
    assembled=base+additions
    if assembled.status()!=mf.Error.NoError:raise ValueError('Assembly Boolean failed')
    save(assembled,ROOT/'cache/assembled_v6_float64.npz')
    print('Complete raw union saved; inspect components without filtering',flush=True)
    raw_components=raw_component_diagnostic(assembled)
    raw_components['input_sha256']=sha(ROOT/'cache/assembled_v6_float64.npz')
    raw_volumes=raw_components['signed_volumes_mm3']
    (ROOT/'reports/v6_raw_component_diagnostic.json').write_text(json.dumps(raw_components,indent=2)+'\n')
    print('Raw components',raw_components['count'],'largest nonmain absolute volume',
        max(sorted(map(abs,raw_volumes))[:-1],default=0.),flush=True)
    print('Confirming original material inclusion before representation preparation',flush=True)
    lost=max(0.,float((base-assembled).volume()))
    if lost>1e-7:raise ValueError('Assembly removed original model material: '+str(lost))
    print('Bounded representation preparation and strict float32 export topology',flush=True)
    vertices,faces,topology,precision=prepare(assembled,ROOT/'cache/assembled_v6_float64.npz')
    target=ROOT/'cache/print_ready_v6.npz'
    np.savez_compressed(target,vertices=vertices,faces=faces)
    precision['exported_mesh_sha256']=sha(target)
    (ROOT/'reports/v6_precision_preparation.json').write_text(json.dumps(precision,indent=2)+'\n')
    changed=[path for path,digest in frozen.items() if sha(path)!=digest]
    if changed:raise ValueError('Assembly input changed during execution: '+str(changed))
    report=dict(passed=True,design_version=6,physical_print_tested=False,
        inputs=[dict(path=str(x.relative_to(ROOT)),sha256=frozen[str(x)]) for x in sources],
        added_support_mesh_sha256=sha(ROOT/'cache/added_supports_v6.npz'),
        exported_mesh_sha256=sha(target),original_material_removed_before_quantization_mm3=lost,
        added_material_mm3=float(assembled.volume()-base.volume()),
        added_fixed_micro_material_mm3=max(0.,float((micro-base).volume())) if not micro.is_empty() else 0.,
        retained_geometry_sha256=sha(ROOT/'cache/added_retained_v6.npz'),
        export_mesh_validation=topology,all_angle_moving_envelope_checks=clearance,
        precision_preparation_report_sha256=sha(ROOT/'reports/v6_precision_preparation.json'),
        raw_component_diagnostic_sha256=sha(ROOT/'reports/v6_raw_component_diagnostic.json'),
        elapsed_seconds=time.monotonic()-started,
        scope=['V5 print geometry is an immutable input; this assembly only unions extra material.',
               'Temporary supports and explicitly declared permanent micro-reinforcements are added only to fixed geometry.',
               'Every addition is connected in the completed model. Build-order connectivity is independently audited.',
               'The independently certified support-function enclosure includes nested rotations, cumulative measured bearing play/tilt and existing cut residues.',
               'The immutable source is unchanged; the final export has a separately reported bounded simplification/rounding error. A final hash-bound motion-transfer audit is required.',
               'Topology and geometric clearance do not establish resin strength or peeling-load capacity.'])
    (ROOT/'reports/custom_support_assembly.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,triangles=len(faces),dimensions_mm=topology['dimensions_mm'],
        added_material_mm3=report['added_material_mm3'])),flush=True)

if __name__=='__main__':main()

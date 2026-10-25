#!/usr/bin/env python3
"""Check original internal tools and all continuous extractions after V6 cuts.

All V6 external temporary pieces have been removed in this state.  Their
declared maximum stumps and the deliberately permanent fixed micro-braces
remain.  Original V5 mechanism clearances have a separate immutable-input
certificate; this checks only the new potential obstructions.
"""
from pathlib import Path
import argparse,json,time,numpy as np
import manifold3d as mf
import audit_motion as A
from assemble_custom_supports import read,sha
from build_holder import SPHERES,sphere_transform

ROOT=Path(__file__).resolve().parent.parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly-context',type=Path,default=ROOT/'reports/custom_support_assembly.json')
    args=parser.parse_args();context_path=args.assembly_context.resolve()
    start=time.monotonic();path=ROOT/'cache/added_retained_v6.npz'
    frozen={}
    def remember(p):
        p=Path(p);key=str(p.relative_to(ROOT));digest=sha(p)
        if key in frozen and frozen[key]!=digest:raise ValueError('Service input changed: '+key)
        frozen[key]=digest
    remember(path)
    for p in [Path(__file__),ROOT/'source/audit_motion.py',ROOT/'source/build_holder.py',
            context_path]:remember(p)
    assembly=json.loads(context_path.read_text())
    if not assembly.get('passed') or assembly['retained_geometry_sha256']!=frozen[str(path.relative_to(ROOT))]:
        raise ValueError('Retained state differs from the validated assembly')
    retained=read(path);cache=ROOT/'cache/spheres';placements=[]
    for s in SPHERES:
        matrix=np.eye(4);matrix[:3]=sphere_transform(s)
        obstacles=retained.transform(np.linalg.inv(matrix)[:3])
        metadata_path=cache/f'sphere_{s["layers"]}layers.json'
        remember(metadata_path)
        meta=json.loads(metadata_path.read_text());wires=[]
        for w in sorted(meta['release_wires'],key=lambda r:r['sequence_index']):
            endpoints=[]
            for ep in w['endpoints']:
                remember(cache/ep['tool_corridor_mesh_file'])
                tool=read(cache/ep['tool_corridor_mesh_file'])
                tool+=A.cylinder_between(ep['tool_tip_end'],
                    ep.get('tool_contact_point',ep['neck_center']),ep.get('tool_tip_diameter_mm',.2)/2.)
                collision=max(0.,float((tool^obstacles).volume()))
                endpoints.append(dict(neck_center=ep['neck_center'],
                    added_obstruction_volume_mm3=collision,passed=bool(collision<1e-8)))
            remember(cache/w['removable_mesh_file']);removable=A.Mesh(cache/w['removable_mesh_file'])
            extraction=A.certify_extraction(removable,obstacles,w.get('extraction_vector',[0,0,1]),
                float(w['extraction_distance_mm']))
            wires.append(dict(wire_index=w['wire_index'],endpoints=endpoints,extraction=extraction,
                passed=all(e['passed'] for e in endpoints) and extraction['passed']))
        row=dict(layers=s['layers'],azimuth_deg=s['theta_deg'],
            module_metadata_sha256=sha(metadata_path),wires=wires,passed=all(w['passed'] for w in wires))
        placements.append(row)
        print('Internal access',s['layers'],s['theta_deg'],row['passed'],flush=True)
    for name,digest in frozen.items():
        if sha(ROOT/name)!=digest:raise ValueError('Service input changed during audit: '+name)
    report=dict(passed=all(x['passed'] for x in placements),units='mm',
        source_context_file=str(context_path.relative_to(ROOT)),
        source_context_type=assembly.get('context_type','validated_print_assembly'),
        input_files=[dict(path=p,sha256=digest) for p,digest in sorted(frozen.items())],
        retained_state_sha256=sha(path),placements=placements,elapsed_seconds=time.monotonic()-start,
        order='Remove V6 external temporary supports first; permanent micro-braces remain; then follow the unchanged V5 internal wire order.',
        scope='Actual tool volumes to neck centres and adaptive continuous triangle-prism extraction sweeps. No physical cutting-force certification.')
    (ROOT/'reports/v6_internal_service_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(2)

if __name__=='__main__':main()

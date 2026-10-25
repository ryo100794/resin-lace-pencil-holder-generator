#!/usr/bin/env python3
"""Check that holder lace does not obstruct sphere-support service paths.

Module-level access is independently checked in audit_motion.py. Here exact
tools and conservative whole-piece extraction hulls are compared with the
complete assembly wall, after each of the nine placement transforms.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import manifold3d as mf
from build_holder import SPHERES, sphere_transform

ROOT=Path(__file__).resolve().parent.parent

def load(path):
    d=np.load(path)
    m=mf.Manifold(mf.Mesh64(np.asarray(d['vertices'],dtype=np.float64),
                          np.asarray(d['faces'],dtype=np.uint32)))
    if m.status()!=mf.Error.NoError:raise ValueError(f'Invalid mesh: {path}')
    return m,np.asarray(d['vertices'],dtype=np.float64)

def merge(parts):
    return mf.Manifold.batch_boolean(parts,mf.OpType.Add)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'cache'/'spheres')
    parser.add_argument('--report',type=Path,default=ROOT/'reports'/'holder_service_access.json')
    parser.add_argument('--wall',type=Path,default=ROOT/'cache'/'wall_for_assembly.npz')
    args=parser.parse_args()
    started=time.monotonic()
    wall_path=args.wall
    wall,_=load(wall_path)
    modules={}
    for n in (3,4,5):
        meta_path=args.cache/f'sphere_{n}layers.json'
        meta=json.loads(meta_path.read_text())
        tools=[];sweeps=[]
        for wire in meta['release_wires']:
            for endpoint in wire['endpoints']:
                tool,_=load(args.cache/endpoint['tool_corridor_mesh_file'])
                tools.append(tool)
            _,vertices=load(args.cache/wire['removable_mesh_file'])
            direction=np.asarray(wire.get('extraction_vector',[0,0,1]),dtype=float)
            direction/=np.linalg.norm(direction)
            travel=float(wire.get('extraction_distance_mm',2*meta['outer_nominal_radius_mm']+4))
            sweeps.append(mf.Manifold.hull_points(np.vstack((vertices,vertices+travel*direction))))
        modules[n]=dict(tools=merge(tools),extraction_hulls=merge(sweeps),
                        tool_count=len(tools),support_count=len(sweeps),
                        metadata_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest())
    report=dict(audit='Whole holder service access',units='mm',
                wall_sha256=hashlib.sha256(wall_path.read_bytes()).hexdigest(),placements=[])
    for sphere in SPHERES:
        module=modules[sphere['layers']]
        transform=sphere_transform(sphere)
        checks={}
        for field in ('tools','extraction_hulls'):
            query=module[field].transform(transform)
            overlap=wall^query
            volume=max(0.,float(overlap.volume()))
            gap=float(wall.min_gap(query,.15))
            checks[field]=dict(intersection_volume_mm3=volume,
                               clearance_mm_capped_at_0_15=gap,
                               passed=bool(volume<1e-8 and gap>.02))
        row=dict(kind=sphere['kind'],layers=sphere['layers'],azimuth_degrees=sphere['theta_deg'],
                 tool_corridors=module['tool_count'],removable_supports=module['support_count'],
                 module_metadata_sha256=module['metadata_sha256'],checks=checks,
                 passed=all(x['passed'] for x in checks.values()))
        report['placements'].append(row)
        print(f"{row['kind']} {row['azimuth_degrees']}: {row['passed']}",flush=True)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    report['passed']=all(x['passed'] for x in report['placements'])
    report['scope']=[
        'Tool corridors use the precise exported module tool volumes.',
        'Each extraction query is the convex hull of all wire vertices before and after a straight outward translation; it conservatively contains every intermediate position.',
        'This test addresses obstruction by the holder wall; aligned shell, bearing and remaining-wire checks are in motion_validation.json.',
        'No physical cutting-force or human hand-access validation is implied.']
    report['elapsed_seconds']=time.monotonic()-started
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()

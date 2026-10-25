#!/usr/bin/env python3
"""Certify rotating module envelopes against the actual assembly wall.

This checks the actual triangles of the cached wall, including its uncut
bezels and mounting arms. A sphere centered on each module bounds all moving
shells and their retained support stumps for every rotation about that center.
The nearest point on each candidate wall triangle gives an exact clearance
from that enclosing sphere, independent of angular sampling.
"""
from pathlib import Path
import argparse,json
import numpy as np
import trimesh
import build_holder as B

ROOT=Path(__file__).resolve().parent.parent

def nearest_wall(vertices,faces,center):
    best=np.inf
    point=None
    for first in range(0,len(faces),120000):
        tri=vertices[faces[first:first+120000]].astype(np.float64)
        delta=np.maximum(np.maximum(tri.min(axis=1)-center,center-tri.max(axis=1)),0)
        lower=np.einsum('ij,ij->i',delta,delta)
        tri=tri[lower<best*best]
        if not len(tri):continue
        pp=trimesh.triangles.closest_point(tri,np.broadcast_to(center,(len(tri),3)).copy())
        distance=np.linalg.norm(pp-center,axis=1)
        at=int(np.argmin(distance))
        if distance[at]<best:
            best=float(distance[at]);point=pp[at]
    return best,point

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--envelopes',required=True,type=Path,
        help='JSON map from layer count to radius bounding moving material in mm.')
    parser.add_argument('--report',type=Path,default=ROOT/'reports'/'holder_motion_clearance.json')
    parser.add_argument('--wall',type=Path,default=ROOT/'cache'/'wall_for_assembly.npz')
    args=parser.parse_args()
    env=json.loads(args.envelopes.read_text())
    with np.load(args.wall) as d:
        v=d['vertices'];f=d['faces']
    rows=[]
    for s in B.SPHERES:
        radius=float(env[str(s['layers'])])
        center=B.sphere_center(s)
        nearest,point=nearest_wall(v,f,center)
        row=dict(kind=s['kind'],layers=s['layers'],azimuth_degrees=s['theta_deg'],
            moving_envelope_radius_mm=radius,wall_nearest_radius_mm=nearest,
            minimum_clearance_lower_bound_mm=nearest-radius,
            nearest_wall_point_mm=point.tolist())
        rows.append(row)
        print(json.dumps(row),flush=True)
    report=dict(passed=all(r['minimum_clearance_lower_bound_mm']>0 for r in rows),
        method='Exact point-to-triangle minimum against a rotation-invariant spherical envelope',
        angular_coverage='all orientations about the module center',
        basis='actual wall used for assembly and conservative bounds supplied for moving solids',
        wall_mesh=args.wall.name,
        modules=rows)
    args.report.write_text(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()

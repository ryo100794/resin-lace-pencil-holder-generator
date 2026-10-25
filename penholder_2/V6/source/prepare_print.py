#!/usr/bin/env python3
"""Prepare a watertight float32 mesh for binary STL and identical 3MF."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np
import manifold3d as mf
import embroidery_technique as E

ROOT=Path(__file__).resolve().parent.parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--released',action='store_true')
    args=p.parse_args()
    expected=28 if args.released else 1
    d=np.load(ROOT/'cache'/('final_released.npz' if args.released else 'final.npz'))
    m=mf.Manifold(mf.Mesh64(d['vertices'],d['faces'].astype(np.uint32)))
    E.log('Decomposing microscopic enclosed boundaries')
    components=m.decompose()
    volumes=np.array([c.volume() for c in components])
    negative=volumes<0
    positives=[c for c,v in zip(components,volumes) if v>0]
    positives.sort(key=lambda c:c.volume(),reverse=True)
    residues=[]
    if len(positives)<expected:raise ValueError(f'Expected {expected} major solid components')
    for c in positives[expected:]:
        bb=np.asarray(c.bounding_box());ext=bb[3:]-bb[:3]
        if not (c.num_tri()==4 and c.volume()<1e-5 and np.linalg.norm(ext)<.15 and ext.min()<.02):
            raise ValueError('Unexpected detached positive feature; manual review required')
        residues.append(dict(triangles=4,volume_mm3=c.volume(),bounds_mm=bb.tolist(),
                             reason='isolated sub-feature-size tetrahedral Boolean residue'))
    if not positives:raise ValueError('Missing main solid')
    if abs(volumes[negative].sum())>.005:raise ValueError('Unexpected cavity volume; manual review required')
    kept=positives[:expected]
    if any(c.volume()<1 for c in kept):raise ValueError('Unexpectedly small intended component')
    m=kept[0] if expected==1 else mf.Manifold.batch_boolean(kept,mf.OpType.Add)
    report={'input_components':len(components),'expected_solid_components':expected,
        'state':'released' if args.released else 'supported_for_printing',
        'filled_microscopic_cavity_surfaces':int(negative.sum()),
        'filled_volume_mm3':float(-volumes[negative].sum()),'original_volume_mm3':float(volumes.sum()),
        'removed_tetrahedral_residues':residues}
    E.log('Checking finite contacts at binary STL coordinate precision')
    m,tm,edge_bonds=E.close_near_contact_edges(m)
    m,tm,vertex_bonds=E.close_point_contacts(m,tm)
    report.update(edge_bonds=edge_bonds,vertex_bonds=vertex_bonds)
    tm.remove_unreferenced_vertices()
    if not tm.is_watertight or not tm.is_winding_consistent or tm.volume<=0:
        raise ValueError('Print precision topology check failed')
    vertices=np.asarray(tm.vertices,dtype=np.float32)
    faces=np.asarray(tm.faces,dtype=np.uint32)
    np.savez_compressed(ROOT/'cache'/('released_v5.npz' if args.released else 'print_ready_v5.npz'),vertices=vertices,faces=faces)
    report.update(vertices=len(vertices),triangles=len(faces),volume_mm3=float(tm.volume),
                  dimensions_mm=tm.extents.tolist(),bounds_mm=tm.bounds.tolist(),
                  watertight=bool(tm.is_watertight),consistent_winding=bool(tm.is_winding_consistent),
                  coordinate_precision='float32; identical vertex coordinates for STL and 3MF')
    (ROOT/'reports'/('released_preparation.json' if args.released else 'print_preparation.json')).write_text(json.dumps(report,indent=2))
    E.log(f'Print mesh ready: {len(faces):,} triangles')

if __name__=='__main__':main()

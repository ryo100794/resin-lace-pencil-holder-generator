#!/usr/bin/env python3
"""Bind the finalized local V6 sphere additions to continuous-motion bounds."""
from pathlib import Path
import hashlib
import io
import json
import time
import numpy as np
import manifold3d as mf
from audit_rotation_envelope import load_certified_envelopes,check_added_material

ROOT=Path(__file__).resolve().parent.parent

def main():
    started=time.monotonic();inputs={}
    def frozen(path):
        data=Path(path).read_bytes();key=str(Path(path).relative_to(ROOT))
        digest=hashlib.sha256(data).hexdigest()
        if key in inputs and inputs[key]!=digest:raise ValueError('Changing source: '+key)
        inputs[key]=digest
        return data
    cert_path=ROOT/'reports/rotation_envelope_certificate.json'
    cert=json.loads(frozen(cert_path))
    envelope=load_certified_envelopes(ROOT);guard=load_certified_envelopes(ROOT,guards=True)
    for path,digest in cert['input_sha256'].items():
        frozen(ROOT/path)
        if inputs[path]!=digest:raise ValueError('Motion input changed')
    for row in cert['modules']:
        frozen(ROOT/row['envelope_mesh']['file']);frozen(ROOT/row['guard_mesh']['file'])
    frozen(Path(__file__));rows=[]
    for n in [3,4,5]:
        meta=json.loads(frozen(ROOT/f'cache/custom_sphere_supports/sphere_{n}layers_supports.json'))
        if meta.get('passed') is not True or meta.get('finalization_complete') is not True:
            raise ValueError('Sphere geometry has not finalized')
        meshes=[]
        for kind in ['supports','retained_stumps']:
            path=ROOT/f'cache/custom_sphere_supports/sphere_{n}layers_{kind}.npz'
            with np.load(io.BytesIO(frozen(path)),allow_pickle=False) as data:
                m=mf.Manifold(mf.Mesh64(np.asarray(data['vertices'],float),np.asarray(data['faces'],np.uint32)))
            result=check_added_material(m,envelope[n],guard[n],margin=.025,search_length=1.)
            if not result['passed']:raise ValueError('Sphere motion exclusion failed')
            meshes.append(dict(**result,kind=kind,mesh_file=str(path.relative_to(ROOT)),mesh_sha256=inputs[str(path.relative_to(ROOT))]))
        rows.append(dict(layer_count=n,passed=True,meshes=meshes))
        print('Sphere',n,'full/retained continuous gaps:',[r['continuous_motion_clearance_lower_bound_mm']for r in meshes],flush=True)
    for path,digest in inputs.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Input changed after validation')
    report=dict(passed=True,certificate_sha256=inputs[str(cert_path.relative_to(ROOT))],
        modules=rows,input_sha256=inputs,elapsed_seconds=time.monotonic()-started,
        scope='Complete finalized local additions and retained stock checked against the certified all-angle nested mechanism envelope. Final assembly checks all nine placements and transfers the export precision separately.')
    path=ROOT/'reports/rotation_envelope_sphere_additions.json'
    temp=Path(str(path)+'.part');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(path)

if __name__=='__main__':main()

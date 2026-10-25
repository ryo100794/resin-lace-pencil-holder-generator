#!/usr/bin/env python3
"""Recover the unchanged, hash-pinned V5 print mesh from its released 3MF."""
from pathlib import Path
import argparse, hashlib, json, zipfile
import numpy as np

V5_SHA256 = '093dba5106df663af08ab9f95c80bf7775486c299e7cf0fce832b71e9e59cae1'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent.parent/'cache/original_v5.npz')
    a = p.parse_args()
    h = hashlib.sha256(a.input.read_bytes()).hexdigest()
    if h != V5_SHA256:
        raise ValueError('The input is not the verified V5 release')
    vertices = np.empty((1588304, 3), np.float32)
    faces = np.empty((3227960, 3), np.uint32)
    nv = nf = 0
    with zipfile.ZipFile(a.input) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt 3MF')
        with archive.open('3D/3dmodel.model') as stream:
            for line in stream:
                if line.startswith(b'<vertex '):
                    fields = line.split(b'"')
                    vertices[nv] = [float(fields[i]) for i in (1,3,5)]
                    nv += 1
                elif line.startswith(b'<triangle '):
                    fields = line.split(b'"')
                    faces[nf] = [int(fields[i]) for i in (1,3,5)]
                    nf += 1
    if nv != len(vertices) or nf != len(faces):
        raise ValueError((nv,nf))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, vertices=vertices, faces=faces)
    print(json.dumps({'source_sha256':h, 'vertices':nv, 'triangles':nf,
        'bounds_mm':[vertices.min(0).tolist(),vertices.max(0).tolist()]}), flush=True)

if __name__ == '__main__': main()

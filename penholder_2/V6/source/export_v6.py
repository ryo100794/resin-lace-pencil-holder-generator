#!/usr/bin/env python3
"""Write and reread the actual V6 STL/3MF geometry without simplification."""
from pathlib import Path
import argparse,json,time
import export_models as E

ROOT=Path(__file__).resolve().parent.parent

def export(input_path, output_dir, stem, title, formats=('3mf','stl')):
    v,f=E.load_source(input_path);digest=E.triangle_sha256(v,f)
    results=[];output_dir.mkdir(exist_ok=True,parents=True)
    for extension in formats:
        target=output_dir/(stem+'.'+extension);temp=Path(str(target)+'.part')
        print('Export '+target.name,flush=True)
        if extension=='3mf':
            E.write_3mf(temp,v,f,title)
            validation=E.verify_3mf(temp,v,f,digest)
        elif extension=='stl':
            E.write_stl(temp,v,f)
            with temp.open('r+b') as stream:
                stream.write(b'Ivory V6 model-specific removable supports; mm; bottom-first +Z'.ljust(80,b' '))
            validation=E.verify_stl(temp,v,f,digest)
        else:raise ValueError(extension)
        temp.replace(target)
        results.append(dict(name=target.name,bytes=target.stat().st_size,
            sha256=E.file_sha256(target),roundtrip_validation=validation))
    return dict(passed=True,input_sha256=E.file_sha256(input_path),
        triangle_count=len(f),vertex_count=len(v),canonical_triangle_sha256=digest,
        dimensions_mm=(v.max(0).astype(float)-v.min(0).astype(float)).tolist(),files=results)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=ROOT/'cache/print_ready_v6.npz')
    p.add_argument('--out',type=Path,default=ROOT/'deliverables')
    p.add_argument('--stem',default='ivory_penholder_v6_custom_supported')
    p.add_argument('--report',type=Path,default=ROOT/'reports/v6_export_validation.json')
    a=p.parse_args()
    report=export(a.input,a.out,a.stem,'Ivory pen holder V6; model-specific removable supports; millimetres; bottom-first +Z')
    a.report.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

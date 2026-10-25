#!/usr/bin/env python3
"""Reassemble and verify V6 from its retained, model-specific support parts.

The default reproduces the released geometry; it does not rerun a potentially
long support search. --regenerate-supports is an explicit design operation.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys

ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/'source'

def run(script,*args):
    subprocess.run([sys.executable,str(SRC/script),*map(str,args)],cwd=ROOT,check=True)

def require(name):
    d=json.loads((ROOT/'reports'/name).read_text())
    if d.get('passed') is not True:raise ValueError('Required report failed: '+name)
    return d

def require_holder_generation():
    d=json.loads((ROOT/'reports/custom_holder_supports.json').read_text())
    if d.get('generation_checks_passed',d.get('passed')) is not True:
        raise ValueError('Holder local generation checks did not pass')
    if d.get('unresolved_selected_candidates') or d.get('foundation_failures'):
        raise ValueError('Holder generation still has unresolved candidates')
    return d

def verify_original_mechanism():
    expected=require('v5_mechanism_unchanged.json')
    for item in expected['checks']:
        p=ROOT/'cache/spheres'/item['file']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Immutable V5 mechanism input changed: '+str(p))
    print('Original V5 sphere mechanism: all '+str(len(expected['checks']))+' files match.',flush=True)

def require_packaged_companion():
    """Fail clearly before expensive work if only one project ZIP was unpacked."""
    main_manifest=ROOT/'FILE_MANIFEST.json'
    if not main_manifest.exists():return
    main=json.loads(main_manifest.read_text())
    if not main.get('required_support_assets_archive'):return
    asset_manifest=ROOT/'ASSET_MANIFEST.json'
    instruction='Extract ivory_v6_project.zip and ivory_v6_support_assets.zip into the same directory, combining their project/ folders.'
    if not asset_manifest.is_file():raise ValueError('Missing mandatory support assets. '+instruction)
    data=asset_manifest.read_bytes()
    if hashlib.sha256(data).hexdigest()!=main['required_asset_manifest_sha256']:
        raise ValueError('The support-assets archive belongs to a different release. '+instruction)
    missing=[]
    for row in json.loads(data)['files']:
        relative=Path(row['path'])
        if not relative.parts or relative.parts[0]!='project' or '..' in relative.parts:
            raise ValueError('Invalid companion manifest path')
        if not (ROOT/Path(*relative.parts[1:])).is_file():missing.append(row['path'])
    if missing:raise ValueError('Incomplete support-assets extraction: '+', '.join(missing[:5])+'. '+instruction)
    print('Mandatory support-assets companion is present; actual input hashes are checked by the following validation steps.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--regenerate-supports',action='store_true',
        help='Repeat support discovery/routing. May be slow; final verification remains mandatory.')
    p.add_argument('--render',action='store_true')
    p.add_argument('--package',action='store_true')
    a=p.parse_args()
    require_packaged_companion()
    for d in ['cache','reports','deliverables']:(ROOT/d).mkdir(exist_ok=True)
    verify_original_mechanism()
    run('audit_rotation_envelope.py');require('rotation_envelope_certificate.json')
    if a.regenerate_supports:
        run('build_support_environment.py')
        for n in (3,4,5):run('custom_sphere_supports.py','--layers',n)
        run('custom_holder_supports.py','--min-lifetime',0)
    require_holder_generation()
    for n in (3,4,5):require(f'custom_sphere_supports_{n}.json')
    run('assemble_custom_supports.py');require('custom_support_assembly.json')
    run('audit_custom_supports.py','--input',ROOT/'cache/print_ready_v6.npz',
        '--out',ROOT/'reports/custom_supports_validation.json','--require-rooted')
    require('custom_supports_validation.json')
    run('audit_v6_release.py','--geometry-only');require('v6_precision_motion_validation.json')
    candidate_index=ROOT/'reports/v6_rebuild_route_candidates.json'
    if not candidate_index.is_file():
        raise ValueError('Missing released removal-instruction index; unpack both complete project archives.')
    saved=json.loads(candidate_index.read_text());candidates=ROOT/saved['path']
    if hashlib.sha256(candidates.read_bytes()).hexdigest()!=saved['sha256']:
        raise ValueError('Released removal instruction snapshot changed')
    require('holder_finite_endpoint_partition_v6.json')
    run('external_removal_holder_endpoints_v6.py',
        '--assembly-context',ROOT/'reports/custom_support_assembly.json',
        '--candidates',candidates)
    require('v6_external_removal_validation.json')
    run('audit_v6_service.py');require('v6_internal_service_validation.json')
    verify_original_mechanism()
    run('export_v6.py');require('v6_export_validation.json')
    run('audit_v6_release.py');require('v6_release_validation.json')
    if a.render or a.package:run('render_custom_supports.py');require('custom_support_render.json')
    run('write_v6_delivery_notes.py');require('v6_delivery_summary.json')
    if a.package:run('package_v6_delivery.py')
    print('V6 assembly, geometry checks and export completed. Physical printing is not part of this pipeline.',flush=True)

if __name__=='__main__':main()

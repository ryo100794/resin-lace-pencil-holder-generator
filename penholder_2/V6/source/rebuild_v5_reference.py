#!/usr/bin/env python3
"""One entry point to recreate the nested-sphere embroidery pen holder."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys

SRC=Path(__file__).resolve().parent
ROOT=SRC.parent

def run(name,*arguments):
    subprocess.run([sys.executable,str(SRC/name),*map(str,arguments)],check=True,cwd=ROOT)

def require_pass(path):
    report=json.loads(path.read_text())
    if report.get('passed') is not True:raise RuntimeError(f'Validation failed: {path}')
    return report

def copy_rendered_images():
    """Use the versioned JPEG names collected by package_delivery.py."""
    images = (
        ('final', 'hero.jpg', 'hero'),
        ('final', 'top.jpg', 'top'),
        ('final', 'views.jpg', 'views'),
        ('final', 'detail.jpg', 'detail'),
        ('mechanism', 'mechanism_overview.jpg', 'mechanism'),
        ('mechanism', 'neck_detail.jpg', 'neck_detail'),
        ('mechanism', 'starter_access.jpg', 'starter_access'),
        ('mechanism', 'bearing_section.jpg', 'bearing_section'),
    )
    for directory, filename, label in images:
        source = ROOT/'previews'/directory/filename
        destination = ROOT/'deliverables'/f'ivory_multisphere_pen_holder_v5_{label}.jpg'
        shutil.copyfile(source, destination)
    print(f'Copied {len(images)} rendered JPEGs to deliverables.', flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--render',action='store_true',
                   help='Render model and mechanism JPEGs and copy them to deliverables.')
    p.add_argument('--fresh',action='store_true',help='Recompute holder geometry after design changes.')
    a=p.parse_args()
    for d in ['cache','deliverables','reports','previews']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    run('sphere_geometry.py','--cache',ROOT/'cache'/'spheres')
    motion_path=ROOT/'reports'/'motion_validation.json'
    run('audit_motion.py','--cache',ROOT/'cache'/'spheres','--report',motion_path)
    require_pass(motion_path)
    early_path=ROOT/'reports'/'early_support_validation.json'
    baseline=ROOT/'reports'/'early_support_v4_baseline.json'
    run('audit_early_supports.py','--cache',ROOT/'cache'/'spheres',
        '--report',early_path,*(['--reference-report',baseline] if baseline.exists() else []))
    require_pass(early_path)
    full_height=ROOT/'reports'/'full_height_validation.json'
    run('audit_full_height.py','--cache',ROOT/'cache'/'spheres','--report',full_height)
    require_pass(full_height)
    strength=ROOT/'reports'/'strength_validation.json'
    run('audit_strength.py','--cache',ROOT/'cache'/'spheres',
        '--reference-root',ROOT/'reference'/'v3','--report',strength)
    require_pass(strength)
    run('finalize_reports.py')
    run('build_holder.py',*(['--rebuild'] if a.fresh else []))
    run('prepare_print.py')
    topology_path=ROOT/'reports'/'geometry_validation.json'
    run('audit_geometry.py',ROOT/'cache'/'print_ready_v5.npz','--report',topology_path)
    require_pass(topology_path)
    envelopes=ROOT/'reports'/'moving_envelopes.json'
    repair_scope=ROOT/'reports'/'precision_repairs_scope.json'
    run('audit_precision_repairs.py','--source-cache',ROOT/'cache'/'spheres')
    require_pass(repair_scope)
    clearance=ROOT/'reports'/'holder_motion_clearance.json'
    run('audit_holder_clearance.py','--envelopes',envelopes,'--report',clearance)
    require_pass(clearance)
    access=ROOT/'reports'/'holder_service_access.json'
    run('audit_holder_access.py','--report',access)
    require_pass(access)
    run('export_models.py','--input',ROOT/'cache'/'print_ready_v5.npz',
        '--out',ROOT/'deliverables','--stem','ivory_multisphere_pen_holder_v5_printpath_20261025')
    coupon=ROOT/'reports'/'test_coupon_validation.json'
    run('build_test_coupon.py','--cache',ROOT/'cache'/'spheres',
        '--out',ROOT/'models','--report',coupon)
    require_pass(coupon)
    if a.render:
        run('render_model.py',ROOT/'cache'/'print_ready_v5.npz','--out',ROOT/'previews'/'final',
            '--only','hero','top','views','detail')
        run('render_mechanism.py',ROOT/'cache'/'spheres'/'sphere_5layers.json',
            '--out',ROOT/'previews'/'mechanism','--only','overview','neck','starter','bearing')
        run('render_coupon.py')
        require_pass(ROOT/'reports'/'test_coupon_render.json')
        copy_rendered_images()
    run('write_release_notes.py')

if __name__=='__main__':main()

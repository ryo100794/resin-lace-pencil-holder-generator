#!/usr/bin/env python3
"""Render the exact exported V5 coupon STL; do not modify any model surface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

import render_model as render


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, default=ROOT/'models'/'ivory_sphere_v5_test_coupon.stl')
    parser.add_argument('--validation', type=Path, default=ROOT/'reports'/'test_coupon_validation.json')
    parser.add_argument('--out', type=Path, default=ROOT/'deliverables'/'ivory_sphere_v5_test_coupon.jpg')
    parser.add_argument('--report', type=Path, default=ROOT/'reports'/'test_coupon_render.json')
    args = parser.parse_args()
    start = time.monotonic()
    validation = json.loads(args.validation.read_text())
    if not validation.get('passed'):
        raise ValueError('A successful coupon mesh validation is required before rendering')
    digest = render.sha256(args.mesh)
    exports = [row for row in validation['exports'] if Path(row['path']).suffix == '.stl']
    if len(exports) != 1 or exports[0]['sha256'] != digest:
        raise ValueError('STL content differs from the validated coupon')
    mesh = render.load_mesh(args.mesh)
    expected_triangles = validation['export_mesh_validation']['triangle_count']
    if mesh.GetNumberOfPolys() != expected_triangles:
        raise ValueError('STL reader did not preserve every validated triangle')
    bounds = np.asarray(mesh.GetBounds(), float)
    dimensions = bounds[1::2]-bounds[::2]
    if not np.allclose(dimensions, validation['export_mesh_validation']['dimensions_mm'],
                       atol=1e-8, rtol=0.):
        raise ValueError('STL reader bounds disagree with the measured export bounds')

    width, view_height = 1600, 1250
    header, footer = 112, 158
    studio = render.Studio(mesh, ao=True, size=(width, view_height))
    try:
        # The aperture points toward export -Y. A modest X offset shows the
        # stand depth while preserving the view through the central window.
        picture = studio.render((width, view_height), azimuth=-79., elevation=13.,
                                margin=1.10, show_ground=True)
        backend = studio.window.GetClassName()
        ao_active = studio.ao_pass is not None
    finally:
        studio.close()
    canvas = Image.new('RGB', (width, header+view_height+footer), render.bg_rgb())
    canvas.paste(picture, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((58, 29), 'V5  /  FIVE-LAYER SPHERE TEST COUPON',
              font=render.font(31, bold=True), fill=render.INK)
    draw.text((60, 75), 'ACTUAL EXPORTED STL  /  FIXED COLLAR ON BASE',
              font=render.font(18), fill=render.QUIET_INK)
    # This is a direction legend, deliberately not a dimension leader.
    arrow_x, arrow_bottom, arrow_top = 96, header+1015, header+820
    draw.line([(arrow_x, arrow_bottom), (arrow_x, arrow_top)], fill=render.QUIET_INK, width=3)
    draw.polygon([(arrow_x, arrow_top-12), (arrow_x-8, arrow_top+4),
                  (arrow_x+8, arrow_top+4)], fill=render.QUIET_INK)
    draw.text((60, arrow_bottom+17), 'BUILD +Z', font=render.font(16, bold=True), fill=render.INK)
    draw.text((60, arrow_bottom+44), 'BASE FIRST', font=render.font(14), fill=render.QUIET_INK)
    y = header+view_height+10
    draw.line([(60, y), (width-60, y)], fill=(113, 104, 91), width=1)
    dimension_text = (f'{dimensions[0]:.3f} x {dimensions[1]:.3f} x {dimensions[2]:.3f} mm'
                      '  (X x Y x Z)')
    draw.text((60, y+22), dimension_text, font=render.font(25, bold=True), fill=render.INK)
    source = validation['source_module']
    draw.text((60, y+62),
              f"{source['support_count']} TEMPORARY SUPPORTS  /  {source['breakaway_neck_count']} BREAKAWAY NECKS  /  PERMANENT BASE",
              font=render.font(18), fill=render.QUIET_INK)
    draw.text((60, y+100),
              'External ornament supports may be required in the slicer. Physical print performance is not certified.',
              font=render.font(17), fill=render.QUIET_INK)
    if render.sha256(args.mesh) != digest:
        raise ValueError('Input STL changed during rendering')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render.atomic_image(canvas, args.out)
    report = dict(passed=True, source_mesh=str(args.mesh.resolve()), source_sha256=digest,
        input_triangles=mesh.GetNumberOfPolys(), rendered_without_geometry_modification=True,
        display_normals_only='35 degree crease-aware Phong normals',
        clipped=False, decimated=False, dimensions_mm=dimensions.tolist(),
        export_build_axis='+Z', aperture_direction='-Y',
        camera=dict(azimuth_degrees=-79., elevation_degrees=13., parallel_projection=True),
        display_backend=backend, ambient_occlusion_active=ao_active,
        output=str(args.out.resolve()), image_size=list(canvas.size),
        output_sha256=render.sha256(args.out), elapsed_seconds=time.monotonic()-start)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

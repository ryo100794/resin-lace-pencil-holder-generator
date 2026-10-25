#!/usr/bin/env python3
"""Render actual V6 additions in orange against the unchanged V5 geometry."""
from pathlib import Path
import argparse, json
from PIL import Image, ImageDraw
import vtk
import render_model as R

ROOT = Path(__file__).resolve().parent.parent
ORANGE = (0.99, 0.35, 0.075)

def add_actor(studio, path, color=ORANGE):
    mesh = R.load_mesh(path)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.SplittingOn()
    normals.SetFeatureAngle(35)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetAmbient(.34)
    actor.GetProperty().SetDiffuse(.68)
    actor.GetProperty().SetSpecular(.06)
    studio.renderer.AddActor(actor)
    actor._pipeline = (mesh, normals, mapper)
    return actor

def label(image, title, subtitle):
    canvas = Image.new('RGB', (image.width, image.height+112), R.bg_rgb())
    canvas.paste(image, (0, 76))
    draw = ImageDraw.Draw(canvas)
    draw.text((32,19), title, font=R.font(26,True), fill=R.INK)
    draw.text((33,53), subtitle, font=R.font(16), fill=R.QUIET_INK)
    draw.text((32,canvas.height-28), 'V6  /  ACTUAL MESH  /  ORANGE: REMOVE  /  BLUE: RETAIN FIXED MICRO-BRACES',
              font=R.font(15), fill=(233,228,217))
    return canvas

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original',type=Path,default=ROOT/'cache/original_v5.npz')
    p.add_argument('--supports',type=Path,default=ROOT/'cache/added_temporary_v6.npz')
    p.add_argument('--micro',type=Path,default=ROOT/'cache/added_fixed_micro_v6.npz')
    p.add_argument('--out',type=Path,default=ROOT/'deliverables')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    studio=R.Studio(R.load_mesh(a.original),ao=True)
    added=add_actor(studio,a.supports)
    fixed=add_actor(studio,a.micro,(.13,.63,.86)) if a.micro.exists() else None
    specs=[
      ('support_overview','MODEL-SPECIFIC SUPPORTS',
       'Bottom-first build  |  Existing nested-shell mechanism retained',
       dict(size=(1300,1450),azimuth=14,elevation=16,margin=1.07)),
      ('sphere_support_detail','FIXED OUTER SPHERE SUPPORTS',
       '5-layer module  |  Short local branches and defined cutting points',
       dict(size=(1400,1180),azimuth=3,elevation=3,focus=[44.8,0,45],parallel_scale=18.0,show_ground=False)),
      ('wall_support_detail','FIXED LACE SUPPORTS',
       'Additional supports at printed starts of ornament and lace',
       dict(size=(1400,1180),azimuth=84,elevation=6,focus=[5.,45.,73.],parallel_scale=12.,show_ground=False)),
      ('base_support_detail','DIRECT BUILD-PLATE CONTACT',
       'Original textured base preserved  |  Local start pads highlighted',
       dict(size=(1300,1300),bottom=True,margin=1.07)),
    ]
    outputs=[]
    for name,title,subtitle,pose in specs:
        print('Render '+name,flush=True)
        size=pose.pop('size')
        result=label(studio.render(size=size,**pose),title,subtitle)
        target=a.out/f'ivory_v6_{name}.jpg'
        result.save(target,quality=94,subsampling=0)
        outputs.append(dict(file=target.name,sha256=R.sha256(target),pixels=list(result.size)))
    studio.close()
    report=dict(passed=True,original_sha256=R.sha256(a.original),
        support_mesh_sha256=R.sha256(a.supports),
        fixed_micro_sha256=R.sha256(a.micro) if a.micro.exists() else None,outputs=outputs,
        scope='Actual supplied triangle geometry. Orange denotes removable additions; blue denotes permanent fixed micro-braces. Color is explanatory. No geometry modification or smoothing.')
    (ROOT/'reports/custom_support_render.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

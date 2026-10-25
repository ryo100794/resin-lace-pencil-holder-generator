#!/usr/bin/env python3
"""Render the actual nested-sphere pen-holder mesh using an EGL studio.

The renderer changes camera and lighting only.  It neither smooths nor decimates
the source geometry.  The optional sphere study is explicitly marked SECTION;
its display clipping planes are never written back to the supplied meshes.

Typical use::

    PYTHONPATH=/tmp/sphere_model_deps python source/render_model.py cache/print_ready_v5.npz
    PYTHONPATH=/tmp/sphere_model_deps python source/render_model.py cache/print_ready_v5.npz \
        --only hero detail views top
    PYTHONPATH=/tmp/sphere_model_deps python source/render_model.py --self-test
    PYTHONPATH=/tmp/sphere_model_deps python source/render_model.py cache/print_ready_v5.npz \
        --only sections --sphere3 cache/spheres/sphere3.npz \
        --sphere4 cache/spheres/sphere4.npz --sphere5 cache/spheres/sphere5.npz

Input NPZ arrays may be named vertices/faces or v/f.  Lengths are millimetres and
the complete pen holder is Z-up.  Sphere studies take local module meshes with
+Z outward, +Y upward and +X tangent, matching the modelling generator.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import io
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/ivory_sphere_mesa_cache")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


_HEADLESS_XVFB = None


def ensure_render_display():
    """Use the installed Xvfb when an older Linux VTK has no EGL backend."""
    global _HEADLESS_XVFB
    if (not sys.platform.startswith("linux") or os.environ.get("DISPLAY")
            or hasattr(vtk, "vtkEGLRenderWindow")
            or hasattr(vtk, "vtkOSOpenGLRenderWindow")):
        return
    executable = shutil.which("Xvfb")
    if executable is None:
        raise RuntimeError("This VTK build needs an X display. Install Xvfb or use an EGL-enabled VTK build.")
    process = subprocess.Popen(
        [executable, "-displayfd", "1", "-screen", "0", "2048x2048x24",
         "-nolisten", "tcp", "-noreset"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ready, _, _ = select.select([process.stdout], [], [], 10.0)
    number = process.stdout.readline().strip() if ready else ""
    if not number.isdecimal() or process.poll() is not None:
        if process.poll() is None:
            process.terminate()
        try:
            _, diagnostics = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            _, diagnostics = process.communicate()
        raise RuntimeError("Xvfb could not create a local render display: "
                           + (diagnostics.strip()[-1500:] or "startup timed out"))
    os.environ["DISPLAY"] = ":" + number
    if os.environ.get("VTK_DEFAULT_OPENGL_WINDOW") == "vtkEGLRenderWindow":
        os.environ.pop("VTK_DEFAULT_OPENGL_WINDOW")
    _HEADLESS_XVFB = process

    def stop_own_display():
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    atexit.register(stop_own_display)
    print(f"Started local Xvfb render display {os.environ['DISPLAY']}", flush=True)


BACKGROUND = (0.278, 0.257, 0.230)
IVORY = (0.928, 0.900, 0.827)
INK = (233, 228, 217)
QUIET_INK = (194, 186, 170)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VERSION = 5


def revision_details(metadata=None):
    """Read the design revision from its supplied modelling metadata."""
    metadata = metadata or {}
    data = metadata.get("mechanism", metadata)
    version = metadata.get("version", data.get("version", DEFAULT_VERSION))
    if isinstance(version, str):
        version = version.strip().lower().removeprefix("ivory_spheres_").removeprefix("v")
    version = int(version)
    if version < 1:
        raise ValueError("The design metadata must contain a positive version number")
    return {"number": version, "label": f"V{version}", "id": f"ivory_spheres_v{version}"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_image(image: Image.Image, path: Path) -> None:
    """Save a complete baseline JPEG or standard RGB PNG and verify decoding."""
    image = image.convert("RGB")
    buffer = io.BytesIO()
    if path.suffix.lower() in (".jpg", ".jpeg"):
        image.save(buffer, format="JPEG", quality=94, subsampling=0,
                   optimize=True, progressive=False)
    else:
        image.save(buffer, format="PNG", optimize=False)
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as check:
        check.load()
        if check.size != image.size or check.mode != "RGB":
            raise OSError("Image verification failed")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    with Image.open(path) as check:
        check.load()


def save_image(image: Image.Image, base: Path, png: bool = False) -> list[Path]:
    paths = [base.with_suffix(".jpg")]
    if png:
        paths.append(base.with_suffix(".png"))
    for path in paths:
        atomic_image(image, path)
    return paths


def font(size: int, bold: bool = False):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
        "/usr/share/fonts/truetype/liberation2/LiberationSans%s.ttf" % ("-Bold" if bold else "-Regular"),
    ]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def bg_rgb() -> tuple[int, int, int]:
    return tuple(round(channel * 255) for channel in BACKGROUND)


def arrays_to_mesh(vertices, faces) -> vtk.vtkPolyData:
    # Keep modelling caches in their supplied precision through normal
    # computation.  Float32 print coordinates stay float32; Boolean diagnostic
    # pieces can contain tiny but valid float64 facets near a curved union.
    vertices = np.asarray(vertices)
    coordinate_dtype = np.float64 if vertices.dtype.itemsize > 4 else np.float32
    vertices = np.ascontiguousarray(vertices, dtype=coordinate_dtype)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be a (n,3) array")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be a (n,3) array of triangle indices")
    if len(faces) == 0 or len(vertices) == 0:
        raise ValueError("Source mesh is empty")
    if not np.isfinite(vertices).all():
        raise ValueError("Source mesh contains nonfinite coordinates")
    index_dtype = np.int32 if max(len(vertices), faces.size) < 2**31 else np.int64
    connectivity = np.ascontiguousarray(faces.reshape(-1), dtype=index_dtype)
    if int(connectivity.min()) < 0 or int(connectivity.max()) >= len(vertices):
        raise ValueError("Triangle indices fall outside the vertex array")
    offsets = np.arange(0, connectivity.size + 1, 3, dtype=index_dtype)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=False))
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtk(offsets, deep=False),
                  numpy_to_vtk(connectivity, deep=False))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    # Explicit references keep all zero-copy arrays alive for the VTK pipeline.
    mesh._numpy_arrays = (vertices, connectivity, offsets)
    return mesh


def load_mesh(path: Path, local_sphere: bool = False) -> vtk.vtkPolyData:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            v_key = "vertices" if "vertices" in data.files else "v"
            f_key = "faces" if "faces" in data.files else "f"
            vertices = data[v_key]
            faces = data[f_key]
            if local_sphere:
                # [local Z, local X, local Y] puts the aperture on +world X.
                vertices = vertices[:, [2, 0, 1]]
            return arrays_to_mesh(vertices, faces)
    if path.suffix.lower() == ".stl":
        reader = vtk.vtkSTLReader()
    elif path.suffix.lower() == ".ply":
        reader = vtk.vtkPLYReader()
    else:
        raise ValueError("Supported mesh inputs are NPZ, STL and PLY")
    reader.SetFileName(str(path))
    reader.Update()
    result = vtk.vtkPolyData()
    result.ShallowCopy(reader.GetOutput())
    if not result.GetNumberOfPolys():
        raise ValueError("Source mesh has no polygons")
    if local_sphere:
        transform = vtk.vtkTransform()
        transform.SetMatrix((0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1))
        filt = vtk.vtkTransformPolyDataFilter()
        filt.SetTransform(transform)
        filt.SetInputData(result)
        filt.Update()
        result = filt.GetOutput()
    return result


def test_mesh() -> vtk.vtkPolyData:
    append = vtk.vtkAppendPolyData()
    for radius in [32, 25, 18]:
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetCenter(0, 0, 32)
        sphere.SetThetaResolution(144)
        sphere.SetPhiResolution(96)
        sphere.Update()
        # A test view of three hemispheres reveals if AO and back lighting work.
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, 0)
        plane.SetNormal(0, 1, 0)
        clip = vtk.vtkClipPolyData()
        clip.SetInputConnection(sphere.GetOutputPort())
        clip.SetClipFunction(plane)
        clip.Update()
        append.AddInputData(clip.GetOutput())
    append.Update()
    result = vtk.vtkPolyData()
    result.ShallowCopy(append.GetOutput())
    return result


class Studio:
    def __init__(self, mesh: vtk.vtkPolyData, ao: bool = True,
                 size: tuple[int, int] = (1360, 1536), scene_scale=None):
        ensure_render_display()
        self.mesh = mesh
        self.bounds = np.asarray(mesh.GetBounds(), dtype=float)
        self.center = (self.bounds[::2] + self.bounds[1::2]) * 0.5
        self.extent = self.bounds[1::2] - self.bounds[::2]
        self.scale = float(max(self.extent) if scene_scale is None else scene_scale)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*BACKGROUND)
        self.renderer.SetAmbient(0.24, 0.24, 0.24)
        self.renderer.AutomaticLightCreationOff()
        self.renderer.UseFXAAOn()
        self.ao_pass = None
        self.ao_fallback_reason = None
        self.ao_unsupported_optional_settings = []
        self._ao_errors = []
        if ao:
            try:
                steps = vtk.vtkRenderStepsPass()
                ao_pass = vtk.vtkSSAOPass()
                ao_pass.SetDelegatePass(steps)
                # VTK 9.3.1 supports SSAO, but not the newer depth-format and
                # intensity controls.  Keep its working defaults for optional
                # settings that are absent instead of disabling the pass.
                options = [("SetDepthFormat", getattr(vtk.vtkTextureObject, "Float32", None)),
                           ("SetRadius", self.scale * 0.027),
                           ("SetBias", self.scale * 0.0016),
                           ("SetKernelSize", 96), ("SetBlur", True),
                           ("SetIntensityScale", 0.88)]
                for method, value in options:
                    if hasattr(ao_pass, method) and value is not None:
                        getattr(ao_pass, method)(value)
                    else:
                        self.ao_unsupported_optional_settings.append(method)
                ao_pass.AddObserver("ErrorEvent", lambda *_: self._ao_errors.append("SSAO render error"))
                self.renderer.SetPass(ao_pass)
                self.ao_pass = ao_pass
                self._render_steps = steps
            except (AttributeError, RuntimeError, TypeError) as error:
                self.renderer.SetPass(None)
                self.ao_fallback_reason = str(error)
                print(f"SSAO unavailable; using standard lighting: {error}", flush=True)

        self.normals = vtk.vtkPolyDataNormals()
        self.normals.SetInputData(mesh)
        # A crease-aware normal field keeps Boolean cut edges crisp.  Averaging
        # every adjacent face produces spurious triangular highlights where a
        # very thin shell meets a raised border.  This changes lighting normals
        # only: all source vertex positions and triangles are retained.
        self.normals.SplittingOn()
        self.normals.SetFeatureAngle(35)
        self.normals.ConsistencyOn()
        self.normals.AutoOrientNormalsOn()
        self.normals.ComputePointNormalsOn()
        self.normals.ComputeCellNormalsOff()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.normals.GetOutputPort())
        mapper.ScalarVisibilityOff()
        self.mapper = mapper
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        prop = self.actor.GetProperty()
        prop.SetColor(*IVORY)
        prop.SetAmbient(0.22)
        prop.SetDiffuse(0.73)
        prop.SetSpecular(0.095)
        prop.SetSpecularPower(28)
        prop.SetInterpolationToPhong()
        prop.BackfaceCullingOff()
        self.renderer.AddActor(self.actor)

        ground_z = float(self.bounds[4] - self.scale * 0.0015)
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-6 * self.scale, -6 * self.scale, ground_z)
        plane.SetPoint1(6 * self.scale, -6 * self.scale, ground_z)
        plane.SetPoint2(-6 * self.scale, 6 * self.scale, ground_z)
        ground_mapper = vtk.vtkPolyDataMapper()
        ground_mapper.SetInputConnection(plane.GetOutputPort())
        self.ground = vtk.vtkActor()
        self.ground.SetMapper(ground_mapper)
        self.ground.GetProperty().SetColor(*BACKGROUND)
        self.ground.GetProperty().SetAmbient(0.97)
        self.ground.GetProperty().SetDiffuse(0.03)
        self.ground.GetProperty().SetSpecular(0)
        self.renderer.AddActor(self.ground)

        self.lights = []
        for color, intensity in [
            ((1.00, 0.99, 0.96), 0.91),
            ((0.95, 0.98, 1.00), 0.28),
            ((1.00, 0.98, 0.93), 0.42),
            ((0.97, 0.99, 1.00), 0.09),
        ]:
            light = vtk.vtkLight()
            light.SetLightTypeToSceneLight()
            light.PositionalOff()
            light.SetColor(*color)
            light.SetIntensity(intensity)
            self.renderer.AddLight(light)
            self.lights.append(light)

        if hasattr(vtk, "vtkEGLRenderWindow") and not os.environ.get("DISPLAY"):
            self.window = vtk.vtkEGLRenderWindow()
        elif hasattr(vtk, "vtkOSOpenGLRenderWindow") and not os.environ.get("DISPLAY"):
            self.window = vtk.vtkOSOpenGLRenderWindow()
        else:
            self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetMultiSamples(0)
        self.window.SetAlphaBitPlanes(0)
        self.window.SetSize(*size)
        self.window.AddRenderer(self.renderer)
        self.camera = self.renderer.GetActiveCamera()
        self.camera.ParallelProjectionOn()
        self.camera.SetViewUp(0, 0, 1)

    def set_section(self, enabled: bool):
        """Display only world Y >= 0 for the explicitly labelled sphere study."""
        self.mapper.RemoveAllClippingPlanes()
        if enabled:
            plane = vtk.vtkPlane()
            plane.SetOrigin(0, 0, 0)
            plane.SetNormal(0, 1, 0)
            self.mapper.AddClippingPlane(plane)

    def pose(self, azimuth=14.0, elevation=18.0, margin=1.10, focus=None,
             parallel_scale=None, top=False, bottom=False):
        a, e = math.radians(azimuth), math.radians(elevation)
        if top or bottom:
            direction = np.array([0.0, 0.0, -1.0 if bottom else 1.0])
            up = np.array([0.0, 1.0, 0.0])
        else:
            direction = np.array([math.cos(e) * math.cos(a),
                                  math.cos(e) * math.sin(a), math.sin(e)])
            up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        actual_up = np.cross(direction, right)
        corners = np.array([[x, y, z] for x in self.bounds[0:2]
                            for y in self.bounds[2:4] for z in self.bounds[4:6]])
        local = corners - self.center
        width = np.ptp(local @ right)
        height = np.ptp(local @ actual_up)
        w, h = self.window.GetSize()
        frame_scale = max(height / 2, width / 2 * h / w) * margin
        target = self.center if focus is None else np.asarray(focus, dtype=float)
        if parallel_scale is not None:
            frame_scale = parallel_scale
        self.camera.SetPosition(*(target + direction * self.scale * 2.4))
        self.camera.SetFocalPoint(*target)
        self.camera.SetViewUp(*up)
        self.camera.SetParallelScale(float(frame_scale))
        self.camera.SetClippingRange(self.scale * 0.50, self.scale * 4.4)
        # The light rig follows the view, so the back and close views are as
        # readable as the hero.  All lights remain physically directional.
        for light, (forward, lateral, vertical) in zip(self.lights, [
            (1.7, -1.5, 2.7), (2.0, 2.7, 0.6), (-1.2, 1.1, 2.3),
            (1.4, -0.2, -1.4),
        ]):
            position = target + self.scale * (
                direction * forward + right * lateral + actual_up * vertical)
            light.SetPosition(*position)
            light.SetFocalPoint(*target)

    def render(self, size=(1360, 1536), show_ground=True, **pose) -> Image.Image:
        self.window.SetSize(*size)
        if pose.get("top") or pose.get("bottom"):
            show_ground = False
        self.ground.SetVisibility(show_ground)
        self.pose(**pose)
        self._ao_errors.clear()
        try:
            self.window.Render()
        except RuntimeError as error:
            if self.ao_pass is None:
                raise
            self._ao_errors.append(str(error))
        if self._ao_errors and self.ao_pass is not None:
            self.ao_fallback_reason = "; ".join(self._ao_errors)
            self.renderer.SetPass(None)
            self.ao_pass = None
            print(f"SSAO failed; retrying with standard lighting: {self.ao_fallback_reason}", flush=True)
            self.window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        result = capture.GetOutput()
        width, height, _ = result.GetDimensions()
        pixels = vtk_to_numpy(result.GetPointData().GetScalars())
        pixels = pixels.reshape(height, width, 3)[::-1].copy()
        return Image.fromarray(pixels)

    def close(self):
        self.window.Finalize()


def sphere_focus(angle: float, height: float, radial_center=44.8):
    theta = math.radians(angle)
    return [radial_center * math.cos(theta), radial_center * math.sin(theta), height]


def scaled_size(width, height, resolution):
    return round(width * resolution), round(height * resolution)


def render_views(studio, resolution=1.0, revision_label=None):
    w, h = scaled_size(690, 900, resolution)
    gap = round(20 * resolution)
    header = round(62 * resolution)
    footer = round(50 * resolution)
    canvas = Image.new("RGB", (w * 2 + gap * 3, h + header + footer), bg_rgb())
    draw = ImageDraw.Draw(canvas)
    poses = [dict(azimuth=0, elevation=0), dict(azimuth=180, elevation=13)]
    labels = ["FRONT", "REVERSE"]
    for index, (pose, label) in enumerate(zip(poses, labels)):
        image = studio.render((w, h), margin=1.10, **pose)
        x = gap + index * (w + gap)
        canvas.paste(image, (x, header))
        draw.text((x + round(22 * resolution), round(20 * resolution)), label,
                  font=font(round(19 * resolution)), fill=QUIET_INK)
    draw.text((gap + round(22 * resolution), h + header + round(12 * resolution)),
              "ACTUAL MESH  /  IVORY SPHERES " + (revision_label or revision_details()["label"]),
              font=font(round(15 * resolution)), fill=QUIET_INK)
    return canvas


def render_detail(studio, resolution=1.0):
    # The largest five-layer sphere is angle 0, centre (44.8, 0, 45) mm.
    return studio.render(scaled_size(1440, 1200, resolution), azimuth=4, elevation=4,
                         focus=sphere_focus(0, 45), parallel_scale=19.1,
                         show_ground=False)


def render_details(studio, resolution=1.0):
    width, height = scaled_size(680, 680, resolution)
    gap = round(22 * resolution)
    header, footer = round(63 * resolution), round(60 * resolution)
    canvas = Image.new("RGB", (width * 2 + gap * 3, height + header + footer), bg_rgb())
    draw = ImageDraw.Draw(canvas)
    views = [("5 LAYERS", 0, 45, 17.4), ("4 LAYERS", 40, 78, 15.0)]
    for index, (label, angle, height_mm, frame_scale) in enumerate(views):
        picture = studio.render((width, height), azimuth=angle + 4, elevation=5,
                                focus=sphere_focus(angle, height_mm),
                                parallel_scale=frame_scale, show_ground=False)
        x = gap + index * (width + gap)
        canvas.paste(picture, (x, header))
        draw.text((x + round(16 * resolution), round(22 * resolution)), label,
                  font=font(round(20 * resolution)), fill=INK)
    draw.text((gap + round(16 * resolution), header + height + round(16 * resolution)),
              "ACTUAL MESH  /  THROUGH-CUT LACE AND NESTED SHELLS",
              font=font(round(15 * resolution)), fill=QUIET_INK)
    return canvas


def render_rim(studio, resolution=1.0):
    return studio.render(scaled_size(1440, 1050, resolution), azimuth=14, elevation=12,
                         focus=[45, 11, float(studio.bounds[5] - 6.5)],
                         parallel_scale=18.0, show_ground=False)


def render_sections(paths: dict[int, Path], ao=True, resolution=1.0):
    tile_w, tile_h = scaled_size(460, 450, resolution)
    gap, header, subheader, footer = [round(value * resolution)
                                     for value in (20, 72, 32, 76)]
    canvas = Image.new("RGB", (tile_w * 3 + gap * 4,
                               tile_h * 2 + header + subheader + footer), bg_rgb())
    draw = ImageDraw.Draw(canvas)
    draw.text((gap + round(10 * resolution), round(22 * resolution)),
              "NESTED SPHERES  /  ACTUAL GEOMETRY",
              font=font(round(23 * resolution)), fill=INK)
    for index, layer_count in enumerate((3, 4, 5)):
        mesh = load_mesh(paths[layer_count], local_sphere=True)
        studio = Studio(mesh, ao=ao)
        try:
            x = gap + index * (tile_w + gap)
            # Identical physical camera scale preserves the 18/22/26 mm size
            # comparison between modules; every individual module is unscaled.
            common_scale = 15.2
            normal = studio.render((tile_w, tile_h), azimuth=4, elevation=4,
                                   focus=[0, 0, 0], parallel_scale=common_scale,
                                   show_ground=False)
            canvas.paste(normal, (x, header))
            draw.text((x + round(15 * resolution), header + tile_h + round(4 * resolution)),
                      f"{layer_count} SHELLS", font=font(round(19 * resolution)), fill=INK)
            studio.set_section(True)
            section = studio.render((tile_w, tile_h), azimuth=-46, elevation=12,
                                    focus=[0, 0, 0], parallel_scale=common_scale,
                                    show_ground=False)
            canvas.paste(section, (x, header + tile_h + subheader))
        finally:
            studio.close()
    y = header + tile_h * 2 + subheader + round(12 * resolution)
    draw.text((gap + round(10 * resolution), y),
              "LOWER ROW: DISPLAY CUTAWAY  /  SOURCE MESHES REMAIN WHOLE",
              font=font(round(15 * resolution)), fill=QUIET_INK)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", nargs="?", type=Path, default=ROOT / "cache" / "print_ready_v5.npz")
    parser.add_argument("--metadata", type=Path,
                        help="Modelling metadata supplying the revision; defaults to the canonical five-layer metadata when present")
    parser.add_argument("--out", type=Path, default=ROOT / "previews")
    parser.add_argument("--only", nargs="+", choices=["hero", "front", "back", "top",
                        "detail", "details", "rim", "views", "sections"],
                        default=["hero", "views", "detail", "top"])
    parser.add_argument("--sphere3", type=Path)
    parser.add_argument("--sphere4", type=Path)
    parser.add_argument("--sphere5", type=Path)
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Pixel-size multiplier; default image long edge is 1536 px")
    parser.add_argument("--png", action="store_true", help="Also save lossless PNG files")
    parser.add_argument("--state", choices=["supported", "released"], default="supported",
                        help="Label the physical assembly state shown by the supplied source mesh")
    parser.add_argument("--no-ao", action="store_true")
    parser.add_argument("--no-report", action="store_true",
                        help="Write only the requested JPEG/PNG images")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 0.15 <= args.resolution <= 2.0:
        parser.error("--resolution must be between 0.15 and 2.0")
    if "sections" in args.only and not all([args.sphere3, args.sphere4, args.sphere5]):
        parser.error("sections requires --sphere3, --sphere4 and --sphere5")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata
    if metadata_path is None:
        candidate = ROOT / "cache" / "spheres" / "sphere_5layers.json"
        metadata_path = candidate if candidate.is_file() else None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path else {}
    revision = revision_details(metadata)
    started = time.time()
    report = {"revision": revision["id"], "assembly_state": args.state,
              "source_metadata": str(metadata_path) if metadata_path else None,
              "metadata_sha256": sha256(metadata_path) if metadata_path else None,
              "vtk_version": vtk.vtkVersion.GetVTKVersion(),
              "source_mesh": None if args.self_test else str(args.mesh.resolve()),
              "source_sha256": None if args.self_test or args.only == ["sections"] else sha256(args.mesh),
              "geometry_processing": "No smoothing, remeshing, rescaling or decimation; point normals for lighting only.",
              "images": []}
    studio = None
    try:
        if args.self_test or any(view != "sections" for view in args.only):
            mesh = test_mesh() if args.self_test else load_mesh(args.mesh)
            studio = Studio(mesh, ao=not args.no_ao)
            report.update({"source_triangles": mesh.GetNumberOfPolys(),
                           "bounds_mm": studio.bounds.tolist(),
                           "extents_mm": studio.extent.tolist(),
                           "render_window": studio.window.GetClassName()})
            print(f"VTK {vtk.vtkVersion.GetVTKVersion()}, {studio.window.GetClassName()}, "
                  f"{mesh.GetNumberOfPolys():,} triangles", flush=True)
        if args.self_test:
            picture = studio.render((640, 700), azimuth=-35, elevation=16, margin=1.10)
            output_files = save_image(picture, args.out / "renderer_test", args.png)
        else:
            output_files = []
            for view in args.only:
                r = args.resolution
                if view == "hero":
                    picture = studio.render(scaled_size(1360, 1536, r),
                                            azimuth=14, elevation=12, margin=1.07)
                    state_label = ("SUPPORTED PRINT STATE" if args.state == "supported"
                                   else "RELEASED ASSEMBLY")
                    ImageDraw.Draw(picture).text((round(38 * r), picture.height - round(44 * r)),
                                                revision["label"] + "  /  " + state_label,
                                                font=font(round(18 * r)), fill=QUIET_INK)
                    report["hero_view_pose"] = {"azimuth_deg": 14,
                                                "elevation_deg": 12, "margin": 1.07}
                elif view == "front":
                    picture = studio.render(scaled_size(1280, 1440, r),
                                            azimuth=0, elevation=0, margin=1.10)
                elif view == "back":
                    picture = studio.render(scaled_size(1280, 1440, r),
                                            azimuth=180, elevation=13, margin=1.09)
                elif view == "top":
                    picture = studio.render(scaled_size(1360, 1360, r), top=True, margin=1.09)
                elif view == "detail":
                    picture = render_detail(studio, r)
                elif view == "details":
                    picture = render_details(studio, r)
                elif view == "rim":
                    picture = render_rim(studio, r)
                elif view == "views":
                    picture = render_views(studio, r, revision["label"])
                elif view == "sections":
                    picture = render_sections({3: args.sphere3, 4: args.sphere4,
                                               5: args.sphere5}, ao=not args.no_ao,
                                              resolution=r)
                    report["sphere_study"] = {
                        "upper_row": "complete supplied sphere module meshes",
                        "lower_row": "display clipping retains local X >= 0; source geometry unchanged",
                        "same_physical_scale": True,
                        "sources": {str(n): {"path": str(path), "sha256": sha256(path)}
                                    for n, path in [(3, args.sphere3), (4, args.sphere4),
                                                    (5, args.sphere5)]}}
                paths = save_image(picture, args.out / view, args.png)
                output_files.extend(paths)
                print(f"Rendered {view}: {picture.width} x {picture.height} px "
                      f"({time.time() - started:.1f} s)", flush=True)
        for path in output_files:
            with Image.open(path) as image:
                report["images"].append({"name": path.name, "size_px": list(image.size),
                                         "bytes": path.stat().st_size,
                                         "sha256": sha256(path), "mode": image.mode,
                                         "format": image.format,
                                         "baseline_jpeg": image.format == "JPEG" and not bool(image.info.get("progressive"))})
        report["elapsed_seconds"] = round(time.time() - started, 2)
        if studio is not None:
            report.update({"ao_enabled": studio.ao_pass is not None,
                           "ao_fallback_reason": studio.ao_fallback_reason,
                           "ao_unsupported_optional_settings": studio.ao_unsupported_optional_settings,
                           "automatic_xvfb": _HEADLESS_XVFB is not None})
        suffix = "test" if args.self_test else "_".join(args.only)
        if not args.no_report:
            (args.out / f"render_report_{suffix}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Complete in {time.time() - started:.1f} s", flush=True)
    finally:
        if studio is not None:
            studio.close()


if __name__ == "__main__":
    main()

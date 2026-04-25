# Resin Lace Pencil Holder Generator

Procedural Python generator for a high-resolution SLA/MSLA resin-printable pencil holder with continuous floral relief, a reinforced lower Ortega-style band, dense upper lace relief, motif-aligned perforations, and ornamental resin drain holes.

The generator writes a binary STL and a parameter JSON file to the current working directory.

## Features

- Cylindrical pencil holder, 105 mm tall, with a flat inner bottom floor.
- Eight-panel continuous relief layout.
- Fractal floral body relief with vines and ornamental recesses.
- Lower Ortega-style geometric band with stronger raised threads and sharper engraved cuts.
- Double rounded raised bead around the upper edge of the lower band.
- Upper lace band without through-holes, using dense 0.2 mm-class raised thread relief.
- Motif-aligned through-holes below the lace band.
- Decorative lower side drain holes for resin escape.
- Signed soft limiting for relief height to avoid hard clipping and flattened detail.
- High output resolution intended for 0.05 mm layer-height resin printing.

## Requirements

```bash
python3 -m pip install numpy
```

Only NumPy is required.

## Usage

```bash
python3 generate_v9_dense_lace_mesh_bottom_pencil_holder.py
```

Outputs:

```text
continuous_relief_fractal_floral_pencil_holder_v9_dense_lace_mesh_bottom.stl
continuous_relief_fractal_floral_pencil_holder_v9_dense_lace_mesh_bottom_parameters.json
```

Generated STL files are intentionally ignored by Git because the high-resolution model can exceed 300 MB.

## Current Resolution

```text
N_THETA = 1568
N_Z = 960
N_RADIAL = 96
```

This gives approximately:

```text
arc pitch: about 0.15 mm
z pitch: about 0.109 mm
```

That is designed to preserve 0.2 mm-class lace relief while staying more practical than a true 0.05 mm mesh pitch.

## Notes For Printing

- Intended process: SLA/MSLA resin printing.
- Suggested layer height: 0.05 mm.
- Inspect the generated STL in your slicer before printing.
- The lower side drain holes are included to help trapped resin escape.
- The STL is dense; slicers may take time to load and repair/check the mesh.

## Documentation

See [docs/design.md](docs/design.md) for the design and formula notes.

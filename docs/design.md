# Design Notes

## Goal

This script generates a decorative, resin-printable pencil holder using procedural mathematical fields rather than a hand-modeled mesh. The model is designed as a cylindrical object with continuous relief, floral motifs, dense lace surfaces, functional drain holes, and a flat usable interior floor.

## Coordinate System

The cylinder is sampled in two main directions:

- `theta`: angular direction around the holder.
- `z`: vertical direction.

The circumference is divided into eight repeated panels using:

```python
PANEL_COUNT = 8
```

Panel-local coordinates are used for symmetry-sensitive design features, especially holes and lower-band motifs.

## Relief Pipeline

The generator builds several scalar fields:

- `ridge`: raised detail.
- `valley_deep`: strong engraved recesses.
- `valley_mid`: medium recesses.
- `valley_shallow`: shallow cuts.
- `rib`: panel ribs and structural raised lines.
- `micro`: fine surface variation.
- `occupancy`: motif density guidance.

These fields are combined into a signed radial displacement:

```python
outer_delta = raise_term - engrave_term
```

The displacement is then passed through a signed `tanh` soft limiter. This avoids hard clipping, which would otherwise flatten dense areas of floral and lace detail.

## Lower Ortega-Style Band

The lower band uses panel-local coordinates:

- diamond fields,
- zig/zag diagonals,
- narrow raised threads,
- companion engraved cuts.

The band has been strengthened so the lower design remains visible after slicing and resin printing.

At the top of this band, the script adds a continuous double rounded bead:

```python
ortega_top_double_bead
```

This creates a two-line rounded border, roughly 1 mm-class in visual width, wrapping around the full cylinder.

## Floral Relief

The floral body is created from recursive rosette fields:

- large central flower,
- upper and lower secondary flowers,
- satellite rosettes around petals,
- shallow valleys and ridges to separate petals.

The flower construction intentionally mixes radial waves and small phase distortions so the surface is decorative rather than mechanically uniform.

## Upper Lace Relief

The upper lace band does not use through-holes. Instead, it relies on surface relief:

- scallops,
- loop fields,
- bead chains,
- taut needlework strands,
- ladder knots,
- hairline incisions,
- braided cords,
- fine net crossings,
- 0.2 mm-class dense warp and weft thread fields,
- vertical thread ribs,
- knot lattice highlights.

This is intended for 0.05 mm resin printing, where fine raised relief is usually more reliable than very small unsupported perforations.

## Perforations

Decorative holes are excluded from the upper lace band. For the remaining body, holes are generated from floral and vine candidates, then filtered by final relief alignment.

The function:

```python
relief_aligned_hole_gate()
```

evaluates the motif fields at cell centers and keeps decorative perforations near valleys, grooves, and vine recesses. This reduces visual mismatch between holes and surface relief.

Hole rims are softened by shifting radii near hole-edge cells:

- outer radius moves inward,
- inner radius moves outward.

This creates a beveled effect instead of a harsh perpendicular cut.

## Resin Drain Holes

The lower side wall includes ornamental drain holes above the bottom floor. These are designed as part of the lower-band vocabulary:

- central arch-like drain,
- side elongated openings,
- small upper vents.

The bottom floor remains solid and flat.

## Mesh Construction

The script constructs:

- outer surface triangles,
- inner surface triangles,
- side walls around perforations,
- top rim cap,
- flat inner floor,
- flat bottom surface,
- vertical bottom wall between inner floor and outer bottom.

The STL writer outputs binary STL directly using Python `struct`.

## Output Scale

All dimensions are in millimeters.

Important defaults:

```python
HEIGHT = 105.0
BOTTOM_THICKNESS = 4.0
INNER_RADIUS_BASE = 36.0
OUTER_RADIUS_BASE = 38.35
MIN_WALL_THICKNESS = 1.55
```

Current high-resolution defaults:

```python
N_THETA = 1568
N_Z = 960
N_RADIAL = 96
```

Generated STL files can be large, around hundreds of megabytes.

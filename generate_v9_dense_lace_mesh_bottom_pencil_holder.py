import math
import os
import json
import struct
import numpy as np

MODEL_NAME = "continuous_relief_fractal_floral_pencil_holder_v9_dense_lace_mesh_bottom"
MODEL_TITLE = "continuous relief fractal floral pencil holder v9 dense lace mesh bottom"
OUT_DIR = os.getcwd()
STL_PATH = os.path.join(OUT_DIR, f"{MODEL_NAME}.stl")
PARAM_PATH = os.path.join(OUT_DIR, f"{MODEL_NAME}_parameters.json")
os.makedirs(OUT_DIR, exist_ok=True)

HEIGHT = 105.0
BOTTOM_THICKNESS = 4.0
INNER_RADIUS_BASE = 36.0
OUTER_RADIUS_BASE = 38.35
MIN_WALL_THICKNESS = 1.55
BASE_WALL_THICKNESS = OUTER_RADIUS_BASE - INNER_RADIUS_BASE
PANEL_COUNT = 8
LOWER_ORTEGA_HEIGHT = 18.0
UPPER_EMBROIDERY_HEIGHT = 16.0
OUTER_RAISE_LIMIT = 2.20
OUTER_ENGRAVE_LIMIT = 1.95
N_THETA = 1568
N_Z = 960
N_RADIAL = 96
MID_RADIUS = 0.5 * (INNER_RADIUS_BASE + OUTER_RADIUS_BASE)
PANEL_WIDTH = (2.0 * math.pi * MID_RADIUS) / PANEL_COUNT


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gauss(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2)


def rotated_ellipse_inside(dx, dz, rx, rz, angle):
    ca = math.cos(angle)
    sa = math.sin(angle)
    x = ca * dx + sa * dz
    y = -sa * dx + ca * dz
    return (x * x) / (rx * rx) + (y * y) / (rz * rz) < 1.0


def active_main_zone(z):
    lower = smoothstep(LOWER_ORTEGA_HEIGHT - 3.0, LOWER_ORTEGA_HEIGHT + 4.0, z)
    upper = 1.0 - smoothstep(HEIGHT - UPPER_EMBROIDERY_HEIGHT - 5.0, HEIGHT - UPPER_EMBROIDERY_HEIGHT, z)
    return lower * upper


def soft_limit_signed(delta, negative_limit, positive_limit):
    positive = positive_limit * np.tanh(np.maximum(delta, 0.0) / positive_limit)
    negative = negative_limit * np.tanh(np.maximum(-delta, 0.0) / negative_limit)
    return positive - negative


def dilate_cells(mask, iterations=1):
    result = mask.copy()
    for _ in range(iterations):
        result |= np.roll(result, 1, axis=0)
        result |= np.roll(result, -1, axis=0)
        up = np.zeros_like(result)
        up[:, 1:] = result[:, :-1]
        down = np.zeros_like(result)
        down[:, :-1] = result[:, 1:]
        result |= up | down
    return result


def cell_mask_to_vertex_weight(mask):
    weight = np.zeros((N_THETA, N_Z + 1), dtype=np.float64)
    weight[:, :-1] += mask
    weight[:, 1:] += mask
    weight[:, :-1] += np.roll(mask, 1, axis=0)
    weight[:, 1:] += np.roll(mask, 1, axis=0)
    return np.clip(weight / 4.0, 0.0, 1.0)


theta = np.linspace(0.0, 2.0 * math.pi, N_THETA, endpoint=False, dtype=np.float64)
z = np.linspace(0.0, HEIGHT, N_Z + 1, dtype=np.float64)
THETA, Z = np.meshgrid(theta, z, indexing="ij")
X = (THETA % (2.0 * math.pi)) * MID_RADIUS
PANEL_INDEX = np.floor(X / PANEL_WIDTH).astype(np.int32)
LOCAL_X = X - PANEL_INDEX * PANEL_WIDTH
CENTER_X = PANEL_WIDTH / 2.0

theta_c = theta + math.pi / N_THETA
z_c = (np.arange(N_Z, dtype=np.float64) + 0.5) * HEIGHT / N_Z
THETA_C, Z_C = np.meshgrid(theta_c, z_c, indexing="ij")
X_C = (THETA_C % (2.0 * math.pi)) * MID_RADIUS
PANEL_INDEX_C = np.floor(X_C / PANEL_WIDTH).astype(np.int32)
LOCAL_X_C = X_C - PANEL_INDEX_C * PANEL_WIDTH
CENTER_X_C = PANEL_WIDTH / 2.0


def fbm_periodic(theta_grid, z_grid, octaves=6):
    acc = np.zeros_like(theta_grid, dtype=np.float64)
    amp = 1.0
    norm = 0.0
    for k in range(octaves):
        fth = 2 ** k
        fz = 2 ** max(0, k - 1)
        term = (
            np.sin(PANEL_COUNT * fth * theta_grid + 2.0 * math.pi * z_grid / (35.0 / fz) + 0.63 * k)
            + 0.65 * np.cos(PANEL_COUNT * 0.73 * fth * theta_grid - 2.0 * math.pi * z_grid / (23.0 / fz) + 0.81 * k)
            + 0.40 * np.sin(PANEL_COUNT * 1.61 * fth * theta_grid + 2.0 * math.pi * z_grid / (15.0 / fz) + 0.37 * k)
        )
        acc += amp * term
        norm += amp * (1.0 + 0.65 + 0.40)
        amp *= 0.55
    return acc / norm


def ortega_field(theta_grid, z_grid, local_x, center_x):
    band = 1.0 - smoothstep(LOWER_ORTEGA_HEIGHT - 2.0, LOWER_ORTEGA_HEIGHT + 2.0, z_grid)
    u = (local_x - center_x) / (0.5 * PANEL_WIDTH)
    v = z_grid / LOWER_ORTEGA_HEIGHT
    stripe = np.abs(np.sin(PANEL_COUNT * theta_grid * 0.5))
    zig = np.abs(((u * 3.0 + v * 4.3) % 2.0) - 1.0)
    zag = np.abs(((u * 3.0 - v * 4.3) % 2.0) - 1.0)
    diamond = np.abs(u) * 1.15 + np.abs(v - 0.48) * 2.15
    lower_thread = (
        0.42 * gauss(diamond - 0.72, 0.030)
        + 0.30 * (gauss(zig - 0.52, 0.026) + gauss(zag - 0.52, 0.026))
        + 0.20 * gauss(np.abs(u) - 0.42, 0.026) * gauss(v - 0.50, 0.34)
        + 0.16 * gauss(np.abs(u) - 0.68, 0.022) * gauss(v - 0.54, 0.28)
    )
    lower_cut = (
        0.30 * gauss(diamond - 0.88, 0.040)
        + 0.24 * (gauss(zig - 0.70, 0.036) + gauss(zag - 0.70, 0.036))
    )
    ridge = band * (
        0.64 * gauss(diamond - 0.72, 0.045)
        + 0.36 * (gauss(zig - 0.52, 0.035) + gauss(zag - 0.52, 0.035))
        + 0.25 * smoothstep(0.42, 0.95, stripe) * gauss(np.abs(v - 0.52), 0.26)
        + lower_thread
    )
    valley = band * (
        0.34 * gauss(diamond - 0.90, 0.055)
        + 0.22 * (gauss(zig - 0.72, 0.050) + gauss(zag - 0.72, 0.050))
        + lower_cut
    )
    return ridge, valley


def embroidery_lace_field(theta_grid, z_grid, local_x, center_x):
    top_start = HEIGHT - UPPER_EMBROIDERY_HEIGHT
    band = smoothstep(top_start - 2.0, top_start + 0.8, z_grid)
    u = (local_x - center_x) / (0.5 * PANEL_WIDTH)
    w = np.clip((z_grid - top_start) / UPPER_EMBROIDERY_HEIGHT, 0.0, 1.0)

    # base scallops
    c1 = 0.28 + 0.16 * np.cos(PANEL_COUNT * theta_grid)
    c2 = 0.45 + 0.100 * np.cos(PANEL_COUNT * 2.0 * theta_grid + 0.60)
    c3 = 0.61 + 0.072 * np.cos(PANEL_COUNT * 3.0 * theta_grid + 1.05)
    c4 = 0.76 + 0.040 * np.cos(PANEL_COUNT * 4.0 * theta_grid + 0.75)
    c5 = 0.88 + 0.022 * np.cos(PANEL_COUNT * 5.0 * theta_grid + 1.35)

    scallop1_r = gauss(w - c1, 0.026)
    scallop1_v = gauss(w - c1 - 0.052, 0.040)
    scallop2_r = gauss(w - c2, 0.020)
    scallop2_v = gauss(w - c2 - 0.038, 0.030)
    scallop3_r = gauss(w - c3, 0.017)
    scallop3_v = gauss(w - c3 - 0.028, 0.026)
    scallop4_r = gauss(w - c4, 0.014)
    scallop4_v = gauss(w - c4 - 0.021, 0.022)
    scallop5_r = gauss(w - c5, 0.012)
    scallop5_v = gauss(w - c5 - 0.018, 0.018)

    loop1 = np.abs(np.sin(PANEL_COUNT * 0.5 * theta_grid))
    loop2 = np.abs(np.sin(PANEL_COUNT * 1.5 * theta_grid + 0.7))
    loop3 = np.abs(np.sin(PANEL_COUNT * 2.5 * theta_grid + 1.2))
    loop4 = np.abs(np.sin(PANEL_COUNT * 3.5 * theta_grid + 1.6))

    loop1_r = gauss(loop1 - 0.58, 0.042) * gauss(w - 0.36, 0.16)
    loop1_v = gauss(loop1 - 0.80, 0.056) * gauss(w - 0.36, 0.16)
    loop2_r = gauss(loop2 - 0.62, 0.035) * gauss(w - 0.51, 0.12)
    loop2_v = gauss(loop2 - 0.82, 0.047) * gauss(w - 0.51, 0.12)
    loop3_r = gauss(loop3 - 0.66, 0.030) * gauss(w - 0.66, 0.09)
    loop3_v = gauss(loop3 - 0.84, 0.043) * gauss(w - 0.66, 0.10)
    loop4_r = gauss(loop4 - 0.69, 0.024) * gauss(w - 0.79, 0.07)
    loop4_v = gauss(loop4 - 0.86, 0.038) * gauss(w - 0.79, 0.07)

    bead1 = gauss(np.abs(w - 0.82), 0.020) * (0.40 + 0.60 * np.sin(PANEL_COUNT * 3.0 * theta_grid) ** 2)
    bead2 = gauss(np.abs(w - 0.90), 0.014) * (0.35 + 0.65 * np.sin(PANEL_COUNT * 5.0 * theta_grid + 0.4) ** 2)
    bead3 = gauss(np.abs(w - 0.96), 0.010) * (0.30 + 0.70 * np.sin(PANEL_COUNT * 7.0 * theta_grid + 1.0) ** 2)

    stitch1 = gauss(np.abs(u) - 0.46, 0.030) * gauss(w - 0.52, 0.25)
    stitch2 = gauss(np.abs(u) - 0.24, 0.022) * gauss(w - 0.66, 0.18)
    stitch3 = gauss(np.abs(u) - 0.66, 0.020) * gauss(w - 0.74, 0.15)
    stitch4 = gauss(np.abs(u) - 0.12, 0.015) * gauss(w - 0.86, 0.09)

    # sharp micro lines
    line_a = gauss(np.sin(PANEL_COUNT * 6.0 * theta_grid + 18.0 * w), 0.20) * gauss(w - 0.56, 0.28)
    line_b = gauss(np.sin(PANEL_COUNT * 8.0 * theta_grid - 23.0 * w + 0.7), 0.18) * gauss(w - 0.63, 0.24)
    line_c = gauss(np.sin(PANEL_COUNT * 11.0 * theta_grid + 31.0 * w + 1.1), 0.16) * gauss(w - 0.77, 0.18)
    line_d = gauss(np.sin(PANEL_COUNT * 14.0 * theta_grid - 37.0 * w + 1.5), 0.15) * gauss(w - 0.89, 0.10)

    # taut needlework strands with sharper but printable line widths
    thread_a_center = 0.47 * np.sin(2.0 * math.pi * (w - 0.34) * 1.30 + 0.55 * np.sin(PANEL_COUNT * theta_grid))
    thread_b_center = -0.47 * np.sin(2.0 * math.pi * (w - 0.34) * 1.30 + 0.55 * np.sin(PANEL_COUNT * theta_grid))
    thread_c_center = 0.23 * np.sin(2.0 * math.pi * (w - 0.42) * 2.15 + 0.35 * np.cos(PANEL_COUNT * 2.0 * theta_grid + 0.4))
    thread_d_center = -0.23 * np.sin(2.0 * math.pi * (w - 0.42) * 2.15 + 0.35 * np.cos(PANEL_COUNT * 2.0 * theta_grid + 0.4))
    thread_env = smoothstep(0.28, 0.39, w) * (1.0 - smoothstep(0.88, 0.97, w))
    thread_r = thread_env * (
        0.92 * gauss(u - thread_a_center, 0.020)
        + 0.92 * gauss(u - thread_b_center, 0.020)
        + 0.58 * gauss(u - thread_c_center, 0.015)
        + 0.58 * gauss(u - thread_d_center, 0.015)
    )
    thread_v = thread_env * (
        0.52 * gauss(u - thread_a_center - 0.038, 0.026)
        + 0.52 * gauss(u - thread_b_center + 0.038, 0.026)
        + 0.36 * gauss(u - thread_c_center - 0.030, 0.020)
        + 0.36 * gauss(u - thread_d_center + 0.030, 0.020)
    )

    ladder_phase = np.mod(PANEL_COUNT * 4.0 * theta_grid + 13.5 * w, 2.0 * math.pi)
    ladder_r = gauss(np.sin(ladder_phase), 0.105) * gauss(np.abs(u) - 0.54, 0.018) * gauss(w - 0.72, 0.16)
    ladder_v = gauss(np.cos(ladder_phase), 0.130) * gauss(np.abs(u) - 0.54, 0.026) * gauss(w - 0.72, 0.16)

    hairline_a = gauss(np.sin(PANEL_COUNT * 18.0 * theta_grid + 48.0 * w + 0.2), 0.090) * gauss(w - 0.74, 0.17)
    hairline_b = gauss(np.sin(PANEL_COUNT * 21.0 * theta_grid - 54.0 * w + 1.1), 0.085) * gauss(w - 0.83, 0.10)
    hairline_mask = smoothstep(0.30, 0.62, np.abs(u)) * (1.0 - smoothstep(0.82, 0.96, np.abs(u)))
    hairline = hairline_mask * (0.60 * hairline_a + 0.45 * hairline_b)

    micro_picot = gauss(np.abs(u) - (0.86 - 0.045 * np.cos(PANEL_COUNT * 4.0 * theta_grid + 2.0 * w)), 0.013) * gauss(w - 0.955, 0.028)

    cord_a = gauss(np.sin(PANEL_COUNT * 10.0 * theta_grid + 26.0 * w), 0.070) * gauss(w - 0.68, 0.23)
    cord_b = gauss(np.sin(PANEL_COUNT * 10.0 * theta_grid - 26.0 * w + 0.65), 0.070) * gauss(w - 0.68, 0.23)
    cord_gate = smoothstep(0.16, 0.34, np.abs(u)) * (1.0 - smoothstep(0.78, 0.94, np.abs(u)))
    braided_cord = cord_gate * (0.54 * cord_a + 0.54 * cord_b + 0.28 * cord_a * cord_b)

    fine_net_a = gauss(np.sin(PANEL_COUNT * 16.0 * theta_grid + 42.0 * w + 0.3), 0.060)
    fine_net_b = gauss(np.sin(PANEL_COUNT * 16.0 * theta_grid - 42.0 * w + 1.1), 0.060)
    fine_net = fine_net_a * fine_net_b * gauss(w - 0.70, 0.19) * (1.0 - smoothstep(0.86, 0.98, np.abs(u)))

    pinstripe = (
        gauss(np.abs(u) - 0.36, 0.010)
        + 0.85 * gauss(np.abs(u) - 0.58, 0.010)
        + 0.70 * gauss(np.abs(u) - 0.72, 0.009)
    ) * gauss(w - 0.80, 0.14)
    pin_cut = (
        gauss(np.abs(u) - 0.42, 0.014)
        + 0.70 * gauss(np.abs(u) - 0.64, 0.014)
    ) * gauss(w - 0.70, 0.20)

    # 0.2 mm-class dense raised lace threads for resin printing.
    lace_core = smoothstep(0.24, 0.34, w) * (1.0 - smoothstep(0.93, 0.985, w))
    lace_side_gate = 1.0 - smoothstep(0.90, 0.99, np.abs(u))
    dense_warp = (
        gauss(np.sin(PANEL_COUNT * 24.0 * theta_grid + 72.0 * w + 0.15), 0.052)
        + gauss(np.sin(PANEL_COUNT * 24.0 * theta_grid - 72.0 * w + 0.95), 0.052)
    ) * lace_core * lace_side_gate
    dense_weft = (
        gauss(np.sin(PANEL_COUNT * 30.0 * theta_grid + 0.35), 0.048) * gauss(w - 0.58, 0.19)
        + 0.82 * gauss(np.sin(PANEL_COUNT * 34.0 * theta_grid + 1.10), 0.046) * gauss(w - 0.78, 0.13)
    ) * lace_side_gate
    vertical_threads = (
        gauss(np.abs(u) - 0.18, 0.007)
        + gauss(np.abs(u) - 0.30, 0.007)
        + gauss(np.abs(u) - 0.42, 0.007)
        + gauss(np.abs(u) - 0.54, 0.007)
        + gauss(np.abs(u) - 0.66, 0.007)
        + gauss(np.abs(u) - 0.78, 0.007)
    ) * gauss(w - 0.70, 0.22)
    knot_lattice = dense_warp * dense_weft * (0.55 + 0.45 * np.cos(PANEL_COUNT * 6.0 * theta_grid) ** 2)
    sharp_lace_r = 0.52 * dense_warp + 0.34 * dense_weft + 0.42 * vertical_threads + 0.58 * knot_lattice
    sharp_lace_v = (
        0.32 * gauss(np.sin(PANEL_COUNT * 24.0 * theta_grid + 72.0 * w + 1.72), 0.070) * lace_core
        + 0.22 * gauss(np.sin(PANEL_COUNT * 30.0 * theta_grid + 1.92), 0.060) * gauss(w - 0.69, 0.20)
        + 0.26 * (
            gauss(np.abs(u) - 0.24, 0.011)
            + gauss(np.abs(u) - 0.48, 0.011)
            + gauss(np.abs(u) - 0.72, 0.011)
        ) * gauss(w - 0.68, 0.20)
    ) * lace_side_gate

    # needle-lace mesh
    m1 = 0.5 + 0.5 * np.sin(PANEL_COUNT * 4.0 * theta_grid + 10.0 * w)
    m2 = 0.5 + 0.5 * np.sin(PANEL_COUNT * 3.0 * theta_grid - 13.0 * w + 0.8)
    m3 = 0.5 + 0.5 * np.sin(PANEL_COUNT * 5.0 * theta_grid + 7.0 * w + 1.4)
    m4 = 0.5 + 0.5 * np.sin(PANEL_COUNT * 9.0 * theta_grid - 17.0 * w + 0.2)
    mesh = (m1 * m2 * 0.55 + m3 * 0.25 + m4 * 0.20)
    mesh_r = mesh * gauss(w - 0.60, 0.26)
    mesh_v = (1.0 - mesh) * gauss(w - 0.60, 0.24)

    picot_chain = gauss(np.abs(u) - (0.78 - 0.08 * np.cos(PANEL_COUNT * 2.0 * theta_grid)), 0.020) * gauss(w - 0.92, 0.06)

    ridge = band * (
        0.30 * scallop1_r + 0.22 * scallop2_r + 0.17 * scallop3_r + 0.14 * scallop4_r + 0.10 * scallop5_r
        + 0.22 * loop1_r + 0.20 * loop2_r + 0.16 * loop3_r + 0.13 * loop4_r
        + 0.18 * bead1 + 0.16 * bead2 + 0.13 * bead3
        + 0.16 * mesh_r + 0.12 * line_a + 0.11 * line_b + 0.10 * line_c + 0.08 * line_d
        + 0.15 * thread_r + 0.12 * ladder_r + 0.10 * hairline + 0.08 * picot_chain + 0.08 * micro_picot
        + 0.13 * braided_cord + 0.11 * fine_net + 0.09 * pinstripe + 0.16 * sharp_lace_r
    )
    valley = band * (
        0.18 * scallop1_v + 0.15 * scallop2_v + 0.12 * scallop3_v + 0.09 * scallop4_v + 0.07 * scallop5_v
        + 0.12 * loop1_v + 0.11 * loop2_v + 0.10 * loop3_v + 0.08 * loop4_v
        + 0.12 * stitch1 + 0.10 * stitch2 + 0.09 * stitch3 + 0.07 * stitch4
        + 0.15 * mesh_v + 0.07 * line_a + 0.07 * line_b + 0.06 * line_c + 0.05 * line_d
        + 0.095 * thread_v + 0.08 * ladder_v + 0.065 * hairline
        + 0.075 * fine_net + 0.060 * pin_cut + 0.095 * sharp_lace_v
    )
    return ridge, valley


def add_rosette(ridge, valley_deep, valley_mid, valley_shallow, occupancy,
                dx, dz, scale_x, scale_z, base_petals, phase, weight, levels, active):
    fbm_local = (
        0.46 * np.sin(dx / (0.51 * scale_x) + dz / (0.43 * scale_z) + phase)
        + 0.32 * np.cos(dx / (0.27 * scale_x) - dz / (0.23 * scale_z) + 1.2)
        + 0.20 * np.sin(dx / (0.15 * scale_x) + dz / (0.13 * scale_z) + 2.1)
    )
    q = 0.56
    for level in range(levels):
        s = q ** level
        petals = int(base_petals * (2 ** level))
        amp = weight * (0.82 ** level)
        rx = max(scale_x * s, 1.35)
        rz = max(scale_z * s, 1.55)
        ddx = dx + 0.10 * rx * np.sin((level + 1) * dz / (0.75 * scale_z) + phase)
        ddz = dz + 0.10 * rz * np.cos((level + 1) * dx / (0.72 * scale_x) - phase)
        uu = ddx / rx
        vv = ddz / rz
        rr = np.sqrt(uu * uu + vv * vv) + 1e-9
        aa = np.arctan2(vv, uu)
        petal_wave = (
            np.cos(petals * aa + phase + 0.55 * fbm_local)
            + 0.48 * np.cos(2 * petals * aa - 0.6 * phase + 0.35 * fbm_local)
            + 0.22 * np.cos(4 * petals * aa + 0.9)
        )
        target = 0.78 + 0.18 * petal_wave + 0.055 * np.sin(7.0 * rr + 0.4 * level + phase)
        ring = gauss(rr - target, 0.038 + 0.009 * level)
        shoulder = gauss(rr - (target + 0.10), 0.082 + 0.016 * level)
        basin = gauss(rr - (target + 0.02), 0.058 + 0.012 * level)
        ridge += amp * 0.32 * ring * active
        valley_deep += amp * 0.42 * basin * active
        valley_mid += amp * 0.34 * shoulder * active
        valley_shallow += amp * 0.18 * gauss(rr - (target + 0.18), 0.15 + 0.018 * level) * active
        occupancy += amp * 0.30 * gauss(rr - target, 0.19 + 0.026 * level) * active
        if level < levels - 1:
            tip_focus = gauss(rr - (target + 0.01), 0.030 + 0.007 * level)
            tip_mod = 0.5 + 0.5 * np.cos(petals * aa + phase)
            micro_rosette = 0.5 + 0.5 * np.cos((petals * 2) * aa + 0.7 * phase + 0.9 * fbm_local)
            ridge += amp * 0.13 * tip_focus * micro_rosette * tip_mod * active
            valley_mid += amp * 0.11 * tip_focus * (1.0 - micro_rosette) * active
    hub = np.sqrt((dx / (0.26 * scale_x)) ** 2 + (dz / (0.20 * scale_z)) ** 2)
    ridge += weight * 0.34 * gauss(hub, 0.63) * active
    valley_shallow += weight * 0.10 * gauss(hub - 1.1, 0.38) * active
    return ridge, valley_deep, valley_mid, valley_shallow, occupancy


def add_fractal_flower_complex(ridge, valley_deep, valley_mid, valley_shallow, occupancy,
                               z_grid, local_x, center_x, flower_z, scale_x, scale_z,
                               base_petals, phase, weight, levels=5):
    active = active_main_zone(z_grid)
    dx0 = local_x - center_x
    dz0 = z_grid - flower_z
    dx = dx0 + 0.11 * scale_x * np.sin(0.25 * dz0 + 2.0 * dx0 / scale_x + phase)
    dz = dz0 + 0.09 * scale_z * np.cos(0.22 * dx0 / scale_x - 0.30 * dz0 + phase)
    ridge, valley_deep, valley_mid, valley_shallow, occupancy = add_rosette(
        ridge, valley_deep, valley_mid, valley_shallow, occupancy,
        dx, dz, scale_x, scale_z, base_petals, phase, weight, levels, active
    )
    q_sat = 0.40
    for k in range(base_petals):
        a = phase + 2.0 * math.pi * k / base_petals
        cx1 = math.cos(a) * scale_x * 0.80
        cz1 = math.sin(a) * scale_z * 0.84
        sweight = weight * 0.36
        sx = scale_x * q_sat
        sz = scale_z * q_sat
        sd_x = dx0 - cx1
        sd_z = dz0 - cz1
        active_sat = active * gauss(np.sqrt((sd_x / (sx * 2.0))**2 + (sd_z / (sz * 2.0))**2), 0.8)
        ridge, valley_deep, valley_mid, valley_shallow, occupancy = add_rosette(
            ridge, valley_deep, valley_mid, valley_shallow, occupancy,
            sd_x, sd_z, sx, sz, max(4, base_petals - 1), a + 0.35, sweight, 3, active_sat
        )
        if k % 2 == 0:
            b = a + 0.55
            cx2 = cx1 + math.cos(b) * sx * 0.85
            cz2 = cz1 + math.sin(b) * sz * 0.92
            sx2 = sx * 0.45
            sz2 = sz * 0.45
            bd_x = dx0 - cx2
            bd_z = dz0 - cz2
            active_b = active * gauss(np.sqrt((bd_x / (sx2 * 2.2))**2 + (bd_z / (sz2 * 2.2))**2), 0.9)
            ridge, valley_deep, valley_mid, valley_shallow, occupancy = add_rosette(
                ridge, valley_deep, valley_mid, valley_shallow, occupancy,
                bd_x, bd_z, sx2, sz2, 4, b - 0.25, weight * 0.18, 3, active_b
            )
    return ridge, valley_deep, valley_mid, valley_shallow, occupancy


def build_motif_fields(theta_grid, z_grid, local_x, center_x):
    ridge = np.zeros_like(z_grid, dtype=np.float64)
    valley_deep = np.zeros_like(z_grid, dtype=np.float64)
    valley_mid = np.zeros_like(z_grid, dtype=np.float64)
    valley_shallow = np.zeros_like(z_grid, dtype=np.float64)
    rib = np.zeros_like(z_grid, dtype=np.float64)
    occupancy = np.zeros_like(z_grid, dtype=np.float64)

    fbm_global = fbm_periodic(theta_grid, z_grid, octaves=6)
    fbm_global2 = fbm_periodic(theta_grid + 0.17, z_grid * 1.09 + 2.3, octaves=5)
    micro = (
        0.19 * fbm_global + 0.12 * fbm_global2 + 0.06 * np.sin(19.0 * theta_grid + 0.31 * z_grid)
        + 0.05 * np.sin(33.0 * theta_grid - 0.26 * z_grid + 0.8) + 0.04 * np.cos(57.0 * theta_grid + 0.15 * z_grid)
    )

    side_d = np.minimum(local_x, PANEL_WIDTH - local_x)
    rib += 0.72 * (1.0 - smoothstep(2.0, 5.8, side_d))
    rib += 0.12 * (1.0 - smoothstep(1.0, 3.1, np.abs(local_x - center_x)))
    rib *= 0.52 + 0.48 * smoothstep(0.08, 0.92, np.mod(z_grid / 5.0, 1.0))

    ridge, valley_deep, valley_mid, valley_shallow, occupancy = add_fractal_flower_complex(
        ridge, valley_deep, valley_mid, valley_shallow, occupancy,
        z_grid, local_x, center_x, 53.0, 14.0, 21.0, 6, 0.0, 1.00, levels=5
    )
    ridge, valley_deep, valley_mid, valley_shallow, occupancy = add_fractal_flower_complex(
        ridge, valley_deep, valley_mid, valley_shallow, occupancy,
        z_grid, local_x, center_x, 28.0, 10.2, 12.0, 5, 0.45, 0.78, levels=4
    )
    ridge, valley_deep, valley_mid, valley_shallow, occupancy = add_fractal_flower_complex(
        ridge, valley_deep, valley_mid, valley_shallow, occupancy,
        z_grid, local_x, center_x, 79.5, 10.2, 12.0, 5, -0.35, 0.78, levels=4
    )

    for sign in (-1.0, 1.0):
        vx = center_x + sign * (10.4 + 1.9 * np.sin((z_grid - 16.0) / 7.4) + 0.9 * np.sin((z_grid - 16.0) / 3.2 + 0.7) + 0.35 * np.sin((z_grid - 16.0) / 1.7 + 1.4))
        dv = np.abs(local_x - vx)
        vine_env = 0.5 + 0.5 * np.sin((z_grid - 15.0) * math.pi / 4.7)
        valley_deep += 0.18 * gauss(dv, 0.46) * vine_env**2
        valley_mid += 0.23 * gauss(dv, 0.82) * vine_env**2
        valley_shallow += 0.15 * gauss(dv, 1.38) * vine_env**2
        ridge += 0.13 * gauss(dv - 1.40, 0.30)
        branch_phase = 0.5 + 0.5 * np.cos(3.1 * z_grid + sign)
        ridge += 0.08 * branch_phase * gauss(dv - 2.1, 0.52)
        valley_mid += 0.09 * branch_phase * gauss(dv - 2.4, 0.70)

    ortega_ridge, ortega_valley = ortega_field(theta_grid, z_grid, local_x, center_x)
    embroidery_ridge, embroidery_valley = embroidery_lace_field(theta_grid, z_grid, local_x, center_x)
    ridge += ortega_ridge + embroidery_ridge
    valley_mid += 0.72 * ortega_valley + 0.86 * embroidery_valley
    valley_shallow += 0.42 * ortega_valley + 0.66 * embroidery_valley
    return ridge, valley_deep, valley_mid, valley_shallow, rib, micro, occupancy


ridge, valley_deep, valley_mid, valley_shallow, rib, micro, occupancy = build_motif_fields(THETA, Z, LOCAL_X, CENTER_X)

depth_envelope = 0.46 + 0.30 * fbm_periodic(THETA + 0.2, Z * 1.02 + 1.0, octaves=5) + 0.17 * np.sin(THETA * 3.1 - Z / 17.0)
depth_envelope = smoothstep(-0.30, 0.76, depth_envelope)
flow_blend = smoothstep(0.08, 0.80, occupancy)
engrave_term = (
    (1.05 + 0.31 * depth_envelope) * valley_deep
    + (0.72 + 0.22 * flow_blend) * valley_mid
    + (0.38 + 0.20 * (1.0 - depth_envelope)) * valley_shallow
)
raise_term = 0.96 * ridge + 0.54 * rib + 0.52 * micro
ortega_top_double_bead = (
    0.44 * gauss(Z - (LOWER_ORTEGA_HEIGHT - 0.45), 0.34)
    + 0.34 * gauss(Z - (LOWER_ORTEGA_HEIGHT + 0.55), 0.32)
)
outer_delta = raise_term - engrave_term
outer_delta += ortega_top_double_bead
outer_delta += 0.34 * gauss(Z - 6.6, 1.10)
outer_delta += 0.50 * gauss(HEIGHT - Z - 3.0, 1.00)
outer_delta += 0.22 * (1.0 - smoothstep(0.0, 4.8, Z))
outer_delta = soft_limit_signed(outer_delta, OUTER_ENGRAVE_LIMIT, OUTER_RAISE_LIMIT)
R_OUT = OUTER_RADIUS_BASE + outer_delta

# inner follows the same signed ratio as outer
inner_delta = outer_delta.copy()
R_IN = INNER_RADIUS_BASE + inner_delta
R_IN = np.minimum(R_IN, R_OUT - MIN_WALL_THICKNESS)
R_OUT = np.maximum(R_OUT, R_IN + MIN_WALL_THICKNESS)


def relief_aligned_hole_gate():
    ridge_c, valley_deep_c, valley_mid_c, valley_shallow_c, rib_c, micro_c, occupancy_c = build_motif_fields(
        THETA_C, Z_C, LOCAL_X_C, CENTER_X_C
    )
    depth_envelope_c = 0.46 + 0.30 * fbm_periodic(THETA_C + 0.2, Z_C * 1.02 + 1.0, octaves=5) + 0.17 * np.sin(THETA_C * 3.1 - Z_C / 17.0)
    depth_envelope_c = smoothstep(-0.30, 0.76, depth_envelope_c)
    flow_blend_c = smoothstep(0.08, 0.80, occupancy_c)
    engrave_c = (
        (1.05 + 0.31 * depth_envelope_c) * valley_deep_c
        + (0.72 + 0.22 * flow_blend_c) * valley_mid_c
        + (0.38 + 0.20 * (1.0 - depth_envelope_c)) * valley_shallow_c
    )
    raise_c = 0.96 * ridge_c + 0.54 * rib_c + 0.52 * micro_c
    relief_delta_c = raise_c - engrave_c
    valley_gate = (engrave_c > 0.18) & (relief_delta_c < 0.04)
    groove_gate = ((valley_deep_c + 0.75 * valley_mid_c) > 0.16) & (ridge_c < 0.40)
    vine_gate = (valley_mid_c > 0.14) & (raise_c < 0.36)
    gate = (valley_gate | groove_gate | vine_gate) & (Z_C < HEIGHT - UPPER_EMBROIDERY_HEIGHT)
    return dilate_cells(gate, iterations=1)


def add_fractal_flower_holes(mask, z_grid, local_x, center_x, flower_z, scale_x, scale_z, petal_count, phase, weight):
    dx = local_x - center_x
    dz = z_grid - flower_z
    q = 0.58
    for level in range(3):
        s = q ** level
        petals = petal_count * (2 ** level)
        rx = max(scale_x * s, 1.5)
        rz = max(scale_z * s, 1.8)
        rr = np.sqrt((dx / rx) ** 2 + (dz / rz) ** 2)
        aa = np.arctan2(dz / rz, dx / rx)
        target = 0.78 + 0.16 * np.cos(petals * aa + phase) + 0.06 * np.cos(2 * petals * aa - 0.4)
        local_ring = np.abs(rr - target) < (0.028 + 0.010 * level)
        sparse = np.cos(petals * aa + phase) > 0.80
        mask |= local_ring & sparse
    for k in range(petal_count):
        a0 = phase + 2.0 * math.pi * k / petal_count
        h0x = math.cos(a0) * scale_x * 0.62
        h0z = math.sin(a0) * scale_z * 0.66
        mask |= rotated_ellipse_inside(dx - h0x, dz - h0z, 0.52 * weight, 1.14 * weight, a0)


def lower_drain_hole_mask():
    u = (LOCAL_X_C - CENTER_X_C) / (0.5 * PANEL_WIDTH)
    drain = np.zeros_like(Z_C, dtype=bool)
    z_ok = (Z_C > BOTTOM_THICKNESS + 1.7) & (Z_C < LOWER_ORTEGA_HEIGHT - 2.4)
    panel_ok = (LOCAL_X_C > 4.2) & (LOCAL_X_C < PANEL_WIDTH - 4.2)

    main_z = 9.2
    dx = LOCAL_X_C - CENTER_X_C
    dz = Z_C - main_z
    arch = rotated_ellipse_inside(dx, dz, 1.55, 2.55, 0.0)
    arch &= dz < 1.75

    side = np.zeros_like(Z_C, dtype=bool)
    for offset, angle in [(-5.9, -0.30), (5.9, 0.30)]:
        side_dx = LOCAL_X_C - (CENTER_X_C + offset)
        side_dz = Z_C - 8.3
        side |= rotated_ellipse_inside(side_dx, side_dz, 0.86, 1.75, angle)

    small_vents = np.zeros_like(Z_C, dtype=bool)
    for offset in [-8.8, 8.8]:
        vent_dx = LOCAL_X_C - (CENTER_X_C + offset)
        vent_dz = Z_C - 11.7
        small_vents |= rotated_ellipse_inside(vent_dx, vent_dz, 0.58, 0.92, 0.18 * np.sign(offset))

    drain |= arch | side | small_vents
    drain &= z_ok & panel_ok
    return drain


def build_perforation_mask():
    mask = np.zeros_like(Z_C, dtype=bool)
    z_ok = (Z_C > 13.5) & (Z_C < 91.5)
    structure_ok = (LOCAL_X_C > 3.2) & (LOCAL_X_C < PANEL_WIDTH - 3.2) & (np.abs(LOCAL_X_C - CENTER_X_C) > 1.10)
    lower_drain = lower_drain_hole_mask()
    relief_gate = relief_aligned_hole_gate()
    _, ortega_valley_c = ortega_field(THETA_C, Z_C, LOCAL_X_C, CENTER_X_C)
    motif_valley_gate = (ortega_valley_c > 0.13) & (Z_C < LOWER_ORTEGA_HEIGHT + 2.0)
    add_fractal_flower_holes(mask, Z_C, LOCAL_X_C, CENTER_X_C, 53.0, 14.0, 21.0, 6, 0.0, 0.95)
    add_fractal_flower_holes(mask, Z_C, LOCAL_X_C, CENTER_X_C, 28.0, 10.2, 12.0, 5, 0.45, 0.76)
    add_fractal_flower_holes(mask, Z_C, LOCAL_X_C, CENTER_X_C, 79.5, 10.2, 12.0, 5, -0.35, 0.76)
    for ez in [17.0, 24.0, 31.5, 39.0, 46.0, 62.0, 69.2, 76.0, 84.0]:
        for ex_offset, angle in [(-8.0, -0.25), (-4.8, 0.18), (4.8, -0.18), (8.0, 0.25)]:
            dx = LOCAL_X_C - (CENTER_X_C + ex_offset + 0.52 * np.sin((Z_C - ez) / 4.0))
            dz = Z_C - ez
            vine_alignment = gauss(dx, 2.8) * gauss(dz, 2.0)
            mask |= rotated_ellipse_inside(dx, dz, 0.66, 1.42, angle) & (vine_alignment > 0.18)
    mask &= relief_gate
    mask &= z_ok
    mask &= structure_ok | motif_valley_gate
    mask &= ~(Z_C < LOWER_ORTEGA_HEIGHT)
    mask |= lower_drain
    mask &= ~(Z_C > HEIGHT - 4.8)
    mask &= ~(Z_C > HEIGHT - UPPER_EMBROIDERY_HEIGHT)
    return mask

solid = ~build_perforation_mask()
solid[:, z_c < BOTTOM_THICKNESS + 1.0] = True
solid[:, z_c > HEIGHT - 4.8] = True
hole = ~solid
neighbor_hole_count = (
    np.roll(hole, 1, axis=0)
    + np.roll(hole, -1, axis=0)
    + np.pad(hole[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    + np.pad(hole[:, 1:], ((0, 0), (0, 1)), constant_values=False)
)
solid[hole & (neighbor_hole_count == 0)] = True

hole = ~solid
hole_edge_cells = solid & (
    np.roll(hole, 1, axis=0)
    | np.roll(hole, -1, axis=0)
    | np.pad(hole[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    | np.pad(hole[:, 1:], ((0, 0), (0, 1)), constant_values=False)
)
outer_bevel = cell_mask_to_vertex_weight(hole_edge_cells)
outer_bevel += 0.46 * cell_mask_to_vertex_weight(dilate_cells(hole_edge_cells, iterations=1) & solid)
outer_bevel = smoothstep(0.08, 0.72, outer_bevel)
outer_bevel *= smoothstep(BOTTOM_THICKNESS + 0.8, BOTTOM_THICKNESS + 2.2, Z) * (1.0 - smoothstep(HEIGHT - 7.0, HEIGHT - 4.8, Z))
R_OUT -= 0.34 * outer_bevel
R_IN += 0.26 * outer_bevel
R_IN = np.minimum(R_IN, R_OUT - MIN_WALL_THICKNESS)
R_OUT = np.maximum(R_OUT, R_IN + MIN_WALL_THICKNESS)


def cylindrical_vertices(radius_grid, theta_grid, z_grid):
    return np.stack([radius_grid * np.cos(theta_grid), radius_grid * np.sin(theta_grid), z_grid], axis=2).astype(np.float32)

outer_vertices_grid = cylindrical_vertices(R_OUT, THETA, Z)
inner_vertices_grid = cylindrical_vertices(R_IN, THETA, Z)
V_outer = outer_vertices_grid.reshape(-1, 3)
V_inner = inner_vertices_grid.reshape(-1, 3)
outer_count = V_outer.shape[0]
inner_count = V_inner.shape[0]
outer_index = np.arange(outer_count, dtype=np.int32).reshape(N_THETA, N_Z + 1)
inner_index = (outer_count + np.arange(inner_count, dtype=np.int32)).reshape(N_THETA, N_Z + 1)
base_vertex_count = outer_count + inner_count
face_parts = []

cell_i, cell_j = np.nonzero(solid)
ip = (cell_i + 1) % N_THETA
j = cell_j
face_parts.append(np.stack([outer_index[cell_i, j], outer_index[ip, j], outer_index[ip, j + 1]], axis=1))
face_parts.append(np.stack([outer_index[cell_i, j], outer_index[ip, j + 1], outer_index[cell_i, j + 1]], axis=1))
face_parts.append(np.stack([inner_index[ip, j], inner_index[cell_i, j], inner_index[cell_i, j + 1]], axis=1))
face_parts.append(np.stack([inner_index[ip, j], inner_index[cell_i, j + 1], inner_index[ip, j + 1]], axis=1))

left_boundary = solid & ~np.roll(solid, 1, axis=0)
ci, cj = np.nonzero(left_boundary)
face_parts.append(np.stack([outer_index[ci, cj], inner_index[ci, cj], inner_index[ci, cj + 1]], axis=1))
face_parts.append(np.stack([outer_index[ci, cj], inner_index[ci, cj + 1], outer_index[ci, cj + 1]], axis=1))
right_boundary = solid & ~np.roll(solid, -1, axis=0)
ci, cj = np.nonzero(right_boundary)
cip = (ci + 1) % N_THETA
face_parts.append(np.stack([inner_index[cip, cj], outer_index[cip, cj], outer_index[cip, cj + 1]], axis=1))
face_parts.append(np.stack([inner_index[cip, cj], outer_index[cip, cj + 1], inner_index[cip, cj + 1]], axis=1))
down_boundary = np.zeros_like(solid)
down_boundary[:, 1:] = solid[:, 1:] & ~solid[:, :-1]
ci, cj = np.nonzero(down_boundary)
cip = (ci + 1) % N_THETA
face_parts.append(np.stack([inner_index[ci, cj], outer_index[ci, cj], outer_index[cip, cj]], axis=1))
face_parts.append(np.stack([inner_index[ci, cj], outer_index[cip, cj], inner_index[cip, cj]], axis=1))
up_boundary = np.zeros_like(solid)
up_boundary[:, :-1] = solid[:, :-1] & ~solid[:, 1:]
ci, cj = np.nonzero(up_boundary)
cip = (ci + 1) % N_THETA
jp = cj + 1
face_parts.append(np.stack([outer_index[ci, jp], inner_index[ci, jp], inner_index[cip, jp]], axis=1))
face_parts.append(np.stack([outer_index[ci, jp], inner_index[cip, jp], outer_index[cip, jp]], axis=1))
ci = np.arange(N_THETA, dtype=np.int32)
cip = (ci + 1) % N_THETA
top_j = N_Z
face_parts.append(np.stack([outer_index[ci, top_j], outer_index[cip, top_j], inner_index[cip, top_j]], axis=1))
face_parts.append(np.stack([outer_index[ci, top_j], inner_index[cip, top_j], inner_index[ci, top_j]], axis=1))

extra_vertices = []

def add_extra_vertices(points):
    start = base_vertex_count + sum(v.shape[0] for v in extra_vertices)
    extra_vertices.append(points.astype(np.float32))
    return start + np.arange(points.shape[0], dtype=np.int32)


def add_radial_cap(z_value, radius_values, upward):
    faces = []
    for i in range(N_THETA):
        th0 = float(theta[i])
        th1 = float(theta[i] + 2.0 * math.pi / N_THETA)
        r0 = float(radius_values[i])
        r1 = float(radius_values[(i + 1) % N_THETA])
        points = np.empty(((N_RADIAL + 1) * 2, 3), dtype=np.float32)
        for k in range(N_RADIAL + 1):
            t = k / N_RADIAL
            z0 = z_value
            z1 = z_value
            points[2 * k] = [r0 * t * math.cos(th0), r0 * t * math.sin(th0), z0]
            points[2 * k + 1] = [r1 * t * math.cos(th1), r1 * t * math.sin(th1), z1]
        idx = add_extra_vertices(points)
        for k in range(N_RADIAL):
            a0 = idx[2 * k]
            b0 = idx[2 * k + 1]
            a1 = idx[2 * (k + 1)]
            b1 = idx[2 * (k + 1) + 1]
            if upward:
                faces.append([a0, a1, b1])
                faces.append([a0, b1, b0])
            else:
                faces.append([b0, b1, a1])
                faces.append([b0, a1, a0])
    face_parts.append(np.array(faces, dtype=np.int32))


bottom_j = int(round(BOTTOM_THICKNESS / HEIGHT * N_Z))
inner_floor_radius = R_IN[:, bottom_j]
outer_bottom_radius = R_OUT[:, 0]

# bottom surface and inner floor stay flat
add_radial_cap(BOTTOM_THICKNESS, inner_floor_radius, upward=True)
add_radial_cap(0.0, outer_bottom_radius, upward=False)

for i in range(N_THETA):
    ip2 = (i + 1) % N_THETA
    th0 = float(theta[i])
    th1 = float(theta[i] + 2.0 * math.pi / N_THETA)
    r0 = float(inner_floor_radius[i])
    r1 = float(inner_floor_radius[ip2])
    points = np.array([
        [r1 * math.cos(th1), r1 * math.sin(th1), 0.0],
        [r0 * math.cos(th0), r0 * math.sin(th0), 0.0],
        [r0 * math.cos(th0), r0 * math.sin(th0), BOTTOM_THICKNESS],
        [r1 * math.cos(th1), r1 * math.sin(th1), BOTTOM_THICKNESS],
    ], dtype=np.float32)
    idx = add_extra_vertices(points)
    face_parts.append(np.array([[idx[0], idx[1], idx[2]], [idx[0], idx[2], idx[3]]], dtype=np.int32))

V = np.vstack([V_outer, V_inner] + extra_vertices).astype(np.float32)
F = np.vstack(face_parts).astype(np.int32)


def write_binary_stl(path, vertices, faces, name):
    with open(path, "wb") as f:
        header = name.encode("ascii", errors="ignore")[:80]
        f.write(header + b" " * (80 - len(header)))
        f.write(struct.pack("<I", faces.shape[0]))
        chunk = 150000
        for start in range(0, faces.shape[0], chunk):
            end = min(start + chunk, faces.shape[0])
            tri = vertices[faces[start:end]]
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            lens = np.linalg.norm(normals, axis=1)
            lens[lens == 0.0] = 1.0
            normals = (normals.T / lens).T.astype(np.float32)
            for n, t in zip(normals, tri):
                f.write(struct.pack("<3f", *n))
                f.write(struct.pack("<3f", *t[0]))
                f.write(struct.pack("<3f", *t[1]))
                f.write(struct.pack("<3f", *t[2]))
                f.write(struct.pack("<H", 0))

write_binary_stl(STL_PATH, V, F, MODEL_NAME)

wall = R_OUT - R_IN
open_cells = int(np.size(solid) - np.count_nonzero(solid))
open_ratio = float(open_cells / np.size(solid))
arc_pitch = float(2.0 * math.pi * MID_RADIUS / N_THETA)
z_pitch = float(HEIGHT / N_Z)

metadata = {
    "units": "mm",
    "model": MODEL_TITLE,
    "description": "v9 adds denser, sharper upper embroidery-lace detail while preserving the fractal floral cylindrical pencil-holder geometry with a flat inner bottom floor.",
    "geometry": {
        "height_mm": HEIGHT,
        "bottom_thickness_mm": BOTTOM_THICKNESS,
        "inner_base_radius_mm": INNER_RADIUS_BASE,
        "outer_base_radius_mm": OUTER_RADIUS_BASE,
        "base_wall_thickness_mm": BASE_WALL_THICKNESS,
        "minimum_wall_thickness_mm": MIN_WALL_THICKNESS,
        "sampled_wall_thickness_range_mm": [float(np.min(wall)), float(np.max(wall))]
    },
    "formula_revision": {
        "lpf_enabled": False,
        "inner_relief_mode": "same_signed_ratio_as_outer",
        "upper_lace_revision": [
            "added higher-order scallops",
            "added micro filigree line fields",
            "added finer beads and picot chain",
            "sharpened line widths",
            "added taut needlework strands, ladder knots, hairline incisions, and micro picots",
            "added braided cords, fine net crossings, and pinstripe relief without upper through-holes",
            "added 0.2 mm-class dense warp, weft, vertical thread, and knot lattice relief"
        ],
        "bottom_revision": "flat inner bottom floor",
        "relief_limiter": "signed tanh soft limit to preserve detail without hard clipping",
        "perforation_revision": "motif-gated openings with softened beveled hole rims, excluding the upper lace band",
        "resin_drain_revision": "ornamental lower side drain holes above the bottom floor",
        "symmetry_revision": "theta segments are divisible by panel count and lace/drain holes use panel-local mirrored placement",
        "hole_alignment_revision": "decorative perforations are gated by cell-centered final relief valleys and grooves",
        "lower_band_revision": "stronger Ortega-style raised threads and sharper engraved cuts",
        "ortega_top_border_revision": "double rounded 1 mm-class raised bead around the upper edge of the lower Ortega band",
        "resolution_revision": "increased theta/z/radial output resolution for 0.05 mm resin printing"
    },
    "relief": {
        "outer_relief_range_from_base_mm": [float(np.min(R_OUT - OUTER_RADIUS_BASE)), float(np.max(R_OUT - OUTER_RADIUS_BASE))],
        "inner_relief_range_from_base_mm": [float(np.min(R_IN - INNER_RADIUS_BASE)), float(np.max(R_IN - INNER_RADIUS_BASE))]
    },
    "mesh": {
        "theta_segments": N_THETA,
        "z_segments": N_Z,
        "radial_cap_segments": N_RADIAL,
        "arc_pitch_mm": arc_pitch,
        "z_pitch_mm": z_pitch,
        "vertices": int(V.shape[0]),
        "triangles": int(F.shape[0])
    },
    "through_holes": {
        "enabled": True,
        "open_grid_ratio": open_ratio,
        "open_grid_cells": open_cells
    }
}
with open(PARAM_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Generated:")
print(STL_PATH)
print(PARAM_PATH)
print(f"Vertices: {V.shape[0]:,}")
print(f"Triangles: {F.shape[0]:,}")
print(f"Wall thickness range: {np.min(wall):.2f} to {np.max(wall):.2f} mm")
print(f"Outer relief range: {np.min(R_OUT - OUTER_RADIUS_BASE):.2f} to {np.max(R_OUT - OUTER_RADIUS_BASE):.2f} mm")
print(f"Inner relief range: {np.min(R_IN - INNER_RADIUS_BASE):.2f} to {np.max(R_IN - INNER_RADIUS_BASE):.2f} mm")
print(f"Through-hole ratio: {open_ratio:.4f}")
print(f"STL size: {os.path.getsize(STL_PATH) / 1024 / 1024:.1f} MB")

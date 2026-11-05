"""Scene geometry, sheet layout and layer stack for the synthetic web.
Pure Python, no bpy.

CD = cross direction = breadth (across the web). MD = machine direction =
length (along the web, the direction of travel).
"""
import math
import struct
from pathlib import Path


MM_PER_INCH = 25.4


def px_to_mm(px, dpi):
    return px / dpi * MM_PER_INCH


# The frame: the project's fixed camera and frame numbers.
FRAME_WIDTH_PX = 1920
FRAME_HEIGHT_PX = 1080
FRAME_MM_PER_PX = 0.5


def frame_geometry():
    return {
        "width_px": FRAME_WIDTH_PX,
        "height_px": FRAME_HEIGHT_PX,
        "mm_per_px": FRAME_MM_PER_PX,
        "cd_mm": FRAME_WIDTH_PX * FRAME_MM_PER_PX,
        "md_mm": FRAME_HEIGHT_PX * FRAME_MM_PER_PX,
    }


def fps_target(line_speed_m_per_min, geometry):
    # The web travels along the MD, so frames per second is line speed over
    # the frame's length along the MD (not its breadth).
    line_speed_m_per_s = line_speed_m_per_min / 60.0
    return line_speed_m_per_s / (geometry["md_mm"] / 1000.0)


def md_pitch_mm(sheet_length_mm, gap_mm):
    return sheet_length_mm + gap_mm


def gaps_in_window(offset_mm, window_mm, sheet_length_mm, gap_mm):
    """Sheet gaps along the MD that fall inside the window, clipped to it and
    expressed in window coordinates. The window starts at `offset_mm` along
    the web; sheet k starts at k * pitch and its trailing gap follows it."""
    pitch = md_pitch_mm(sheet_length_mm, gap_mm)
    window_end = offset_mm + window_mm
    gaps = []
    k = math.floor(offset_mm / pitch) - 1
    while True:
        start = k * pitch + sheet_length_mm
        end = (k + 1) * pitch
        if start >= window_end:
            break
        if end > offset_mm:
            gaps.append((max(start, offset_mm) - offset_mm, min(end, window_end) - offset_mm))
        k += 1
    return gaps


def lane_visible_fractions(lane_width_mm, n_lanes, window_cd_mm, lane_gap_mm, window_cd_offset_mm=0.0):
    """Fraction of each lane visible in the frame window, which spans CD
    `window_cd_offset_mm` to that plus `window_cd_mm` (offset 0 is the left
    camera). Lanes are `lane_gap_mm` apart across the web."""
    fractions = []
    for i in range(n_lanes):
        lane_x0 = i * (lane_width_mm + lane_gap_mm)
        visible_mm = min(lane_x0 + lane_width_mm, window_cd_offset_mm + window_cd_mm) - max(
            lane_x0, window_cd_offset_mm
        )
        fractions.append(min(max(visible_mm / lane_width_mm, 0.0), 1.0))
    return fractions


# Sourced ranges for the layer stack. VLT is stored as a fraction.
PET_DENSITY_RANGE = (1.3, 1.4)
BARRIER_VLT_RANGE = (0.87, 0.89)
BARRIER_GAUGES_UM = (9, 10, 12)  # three real product grades, pick one
SEALANT_DENSITY_RANGE = (0.91, 0.98)
SEALANT_GAUGE_RANGE_UM = (50, 105)
BOND_GAP_RANGE_UM = (3.0, 5.0)  # typical inter-layer adhesive gap


def _truncated_normal(rng, low, high):
    """Mean at the range midpoint, SD = range / 4 (a range read as
    plus or minus two standard deviations), redrawn until the value falls inside the range."""
    mean = (low + high) / 2
    sd = (high - low) / 4
    while True:
        value = rng.gauss(mean, sd)
        if low <= value <= high:
            return value


def layer_stack(rng, pet_gauge_um, barrier_gauge_um):
    if pet_gauge_um is None or pet_gauge_um <= 0:
        raise ValueError("pet_gauge_um must be supplied and positive")
    if barrier_gauge_um not in BARRIER_GAUGES_UM:
        raise ValueError(f"barrier_gauge_um must be one of {BARRIER_GAUGES_UM}")
    pet_density = _truncated_normal(rng, *PET_DENSITY_RANGE)
    barrier_vlt = _truncated_normal(rng, *BARRIER_VLT_RANGE)
    sealant_density = _truncated_normal(rng, *SEALANT_DENSITY_RANGE)
    sealant_gauge_um = _truncated_normal(rng, *SEALANT_GAUGE_RANGE_UM)
    bond_gap_um = _truncated_normal(rng, *BOND_GAP_RANGE_UM)
    return {
        "layers": [
            {
                "name": "pet",
                "material": "PET (BOPET, corona-treated)",
                "density": pet_density,
                "thickness_um": pet_gauge_um,
            },
            {
                "name": "barrier",
                "material": "AlOx-coated PET",
                "vlt": barrier_vlt,
                "thickness_um": barrier_gauge_um,
            },
            {
                "name": "sealant",
                "material": "LLDPE",
                "density": sealant_density,
                "thickness_um": sealant_gauge_um,
            },
        ],
        "bond_gap_um": bond_gap_um,
        "provenance": {
            "pet_density": PET_DENSITY_RANGE,
            "barrier_vlt": BARRIER_VLT_RANGE,
            "sealant_density": SEALANT_DENSITY_RANGE,
            "sealant_gauge_um": SEALANT_GAUGE_RANGE_UM,
            "bond_gap_um": BOND_GAP_RANGE_UM,
            "pet_gauge_um": "caller-supplied",
            "barrier_gauge_um": "datasheet grade",
        },
    }


def print_layer_stack(rng, pet_gauge_um):
    """The Placement A material state: printed film viewed
    through the clear outer PET, pre-lamination. No barrier, no sealant and no
    bond gap exist yet at this point in the line."""
    if pet_gauge_um is None or pet_gauge_um <= 0:
        raise ValueError("pet_gauge_um must be supplied and positive")
    pet_density = _truncated_normal(rng, *PET_DENSITY_RANGE)
    return {
        "layers": [{"name": "pet", "material": "PET (BOPET, corona-treated)", "density": pet_density, "thickness_um": pet_gauge_um}],
        "provenance": {"pet_density": PET_DENSITY_RANGE, "pet_gauge_um": "caller-supplied"},
    }


def layer_z_positions_um(stack):
    """(top, bottom) of each layer in microns, measured down from the top of
    the outer PET, with the bond gap between consecutive layers."""
    positions = []
    z = 0.0
    for layer in stack["layers"]:
        top = z
        bottom = top + layer["thickness_um"]
        positions.append((top, bottom))
        z = bottom + stack.get("bond_gap_um", 0.0)
    return positions


MAP_SUFFIXES = {
    "artwork": "_artwork.png",
    "height": "_Height.png",
    "metallic": "_Metallic.png",
    "roughness": "_Roughness.png",
}


def sku_map_paths(sku_dir):
    """The artwork and its three maps in a SKU directory, found by suffix so
    the same call works for every SKU. Exactly one file per suffix."""
    sku_dir = Path(sku_dir)
    paths = {}
    for name, suffix in MAP_SUFFIXES.items():
        matches = sorted(sku_dir.glob(f"*{suffix}"))
        if len(matches) != 1:
            raise ValueError(f"expected one *{suffix} in {sku_dir}, found {len(matches)}")
        paths[name] = matches[0]
    return paths


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_size(path):
    """(width_px, height_px) from the PNG IHDR chunk, without decoding."""
    with open(path, "rb") as f:
        header = f.read(24)
    if header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def check_maps_aligned(paths):
    """Pixel size shared by the artwork and its maps. Raises ValueError if any
    differ, so a partial re-export can't be rendered."""
    sizes = {name: _png_size(path) for name, path in paths.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"maps differ in pixel size: {sizes}")
    return next(iter(sizes.values()))


def artwork_size_mm(path, dpi):
    """(breadth across the web, length along the web) of one artwork sheet in
    mm. Artwork width is the lane's CD breadth; height is the sheet's MD
    length."""
    width_px, height_px = _png_size(path)
    return px_to_mm(width_px, dpi), px_to_mm(height_px, dpi)


ARTWORK_DPI = 300  # A10


def artwork_dimensions_mm(sku_dir):
    """(lane width, sheet length) in mm of a SKU's artwork at `ARTWORK_DPI`: the
    size the scene is built from, so the planner must use it rather than a
    rounded figure."""
    return artwork_size_mm(sku_map_paths(sku_dir)["artwork"], ARTWORK_DPI)


def web_breadth_mm(lane_width_mm, n_lanes, lane_gap_mm):
    """Web breadth across the CD: the lanes plus the gaps where they join."""
    return n_lanes * lane_width_mm + (n_lanes - 1) * lane_gap_mm


def ink_sheets_in_window(
    lane_width_mm,
    sheet_length_mm,
    gap_mm,
    n_lanes,
    window_cd_mm,
    window_md_mm,
    md_offset_mm,
    lane_gap_mm,
    window_cd_offset_mm=0.0,
):
    """Ink sheets overlapping the frame window, in web coordinates: x across
    the web (CD) from its left edge, y along it (MD). Sheet k of every lane
    starts at k * pitch; lanes are `lane_gap_mm` apart. Ordered by lane, then
    sheet index. `cd_visible_fraction` is how much of the sheet's breadth the
    window shows."""
    pitch = md_pitch_mm(sheet_length_mm, gap_mm)
    window_end = md_offset_mm + window_md_mm
    fractions = lane_visible_fractions(
        lane_width_mm, n_lanes, window_cd_mm, lane_gap_mm, window_cd_offset_mm
    )
    sheets = []
    for lane in range(n_lanes):
        if fractions[lane] <= 0.0:
            continue
        index = math.floor(md_offset_mm / pitch) - 1
        while index * pitch < window_end:
            y0 = index * pitch
            if y0 + sheet_length_mm > md_offset_mm:
                sheets.append(
                    {
                        "lane": lane,
                        "index": index,
                        "x0_mm": lane * (lane_width_mm + lane_gap_mm),
                        "y0_mm": y0,
                        "width_mm": lane_width_mm,
                        "length_mm": sheet_length_mm,
                        "cd_visible_fraction": fractions[lane],
                    }
                )
            index += 1
    return sheets


def layer_mesh_spec(stack):
    """What the scene builds for Placement B, top to bottom: the clear PET,
    the reverse-printed ink as a zero-thickness sheet on the PET's inner face,
    the AlOx-coated PET barrier, and the LLDPE sealant. Film layers run
    continuously (through the sheet gaps); only the ink is per sheet. Only the
    ink carries the artwork and its maps, with Height as a bump. z is in um
    from the top of the PET; the scene script decides any scale for rendering."""
    pet_z, barrier_z, sealant_z = layer_z_positions_um(stack)

    def film(name, z):
        return {
            "name": name,
            "kind": "solid",
            "z_top_um": z[0],
            "z_bottom_um": z[1],
            "maps": [],
            "extent": "continuous",
        }

    ink = {
        "name": "ink",
        "kind": "sheet",
        "z_top_um": pet_z[1],
        "z_bottom_um": pet_z[1],
        "maps": list(MAP_SUFFIXES),
        "height_mode": "bump",
        "extent": "per_sheet",
    }
    return [film("pet", pet_z), ink, film("barrier", barrier_z), film("sealant", sealant_z)]


# Sheet layout on the web.
SHEET_GAP_MM = 10.0
LANE_GAP_MM = 10.0
N_LANES = 2
FILM_MARGIN_MM = 50.0  # film runs this far past the window along the web

# Refractive indices of the laminate layers at normal incidence. The PET
# layers are biaxially oriented film (BOPET): about 1.65 in the plane (MD = TD) and 1.50
# through the thickness, so light at normal incidence sees 1.65. Unoriented PET is 1.575
# (en.wikipedia.org/wiki/Polyethylene_terephthalate) and is the lower bound. LLDPE is taken as
# polyethylene, about 1.51 (LDPE 1.51-1.52). The BOPET and LLDPE values are secondary:
# search-result summaries of patents and polymer tables, not a supplier datasheet.
PET_REFRACTIVE_INDEX = 1.65
PET_ISOTROPIC_REFRACTIVE_INDEX = 1.575
LLDPE_REFRACTIVE_INDEX = 1.51
# The intact bond is an adhesive slab filling the bond gap: index-matched to the layer
# below it, so it adds no air gap of its own. An air-gap defect (tunneling, bubbles, silvering) is the
# absence of the slab. The adhesive's own index is not sourced; matching the neighbour is an assumption.
ADHESIVE_REFRACTIVE_INDEX = {"1-2": PET_REFRACTIVE_INDEX, "2-3": LLDPE_REFRACTIVE_INDEX}


def bank_cd_offset_mm(camera, lane_width_mm):
    """CD offset of a bank camera's window: the left camera at 0, the right one
    flush with the right web edge."""
    if camera == "left":
        return 0.0
    if camera == "right":
        return web_breadth_mm(lane_width_mm, N_LANES, LANE_GAP_MM) - frame_geometry()["cd_mm"]
    raise ValueError(f"camera must be 'left' or 'right', got {camera!r}")


def scene_plan(stack, lane_width_mm, sheet_length_mm, md_offset_mm, z_scale=1.0, cd_offset_mm=0.0):
    """Everything the Blender scene script builds, in web coordinates: mm
    across the web (x, CD) and along it (y, MD), um through the stack (z,
    measured down from the top of the PET, scaled by `z_scale`; 1.0 is true
    scale). The camera is orthographic so the mm-per-pixel scale is exact. Its
    window starts at CD `cd_offset_mm`, anywhere from the left edge to flush
    with the right one."""
    g = frame_geometry()
    breadth_mm = web_breadth_mm(lane_width_mm, N_LANES, LANE_GAP_MM)
    if not 0.0 <= cd_offset_mm <= breadth_mm - g["cd_mm"]:
        raise ValueError(f"cd_offset_mm must keep the window on the web, got {cd_offset_mm}")
    ink_sheets = ink_sheets_in_window(
        lane_width_mm,
        sheet_length_mm,
        SHEET_GAP_MM,
        N_LANES,
        g["cd_mm"],
        g["md_mm"],
        md_offset_mm,
        LANE_GAP_MM,
        cd_offset_mm,
    )
    # Film is continuous, so it must also cover the whole of every ink sheet.
    y0 = min([md_offset_mm - FILM_MARGIN_MM] + [s["y0_mm"] for s in ink_sheets])
    y1 = max(
        [md_offset_mm + g["md_mm"] + FILM_MARGIN_MM]
        + [s["y0_mm"] + s["length_mm"] for s in ink_sheets]
    )
    z = layer_z_positions_um(stack)
    # Each layer keeps its own true z-span, a real (if tiny) gap from the layer above
    # (two attempts to close this gap -- a separate bond object, then extending a layer to touch its
    # neighbour directly -- both produced a coincident-face rendering artifact; reverted).
    film = [
        {
            "name": layer["name"],
            "kind": "solid",
            "x0_mm": 0.0,
            "x1_mm": breadth_mm,
            "y0_mm": y0,
            "y1_mm": y1,
            "z_top_um": top * z_scale,
            "z_bottom_um": bottom * z_scale,
        }
        for layer, (top, bottom) in zip(stack["layers"], z)
    ]
    ink_z_um = z[0][1] * z_scale  # the PET's inner face
    return {
        "camera": {
            "type": "ortho",
            "x_mm": cd_offset_mm + g["cd_mm"] / 2,
            "cd_offset_mm": cd_offset_mm,
            "y_mm": md_offset_mm + g["md_mm"] / 2,
            "ortho_scale_mm": g["cd_mm"],
            "width_px": g["width_px"],
            "height_px": g["height_px"],
        },
        "web": {"breadth_mm": breadth_mm},
        "film": film,
        "ink_sheets": [dict(sheet, z_um=ink_z_um) for sheet in ink_sheets],
    }


def camera_response(x, exposure, gamma, bits):
    """Digital number for a scene-linear value `x` (1.0 = sensor saturation).
    Industrial global-shutter cameras respond linearly and clip at
    saturation; gamma (1.0 = none) is an optional setting. Exposure (gain x
    time) is applied before clipping."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if bits < 1:
        raise ValueError("bits must be at least 1")
    v = min(max(x * exposure, 0.0), 1.0)
    return v ** (1.0 / gamma) * (2**bits - 1)


def exposure_gain(exposure, master_scale):
    """Gain on the scene-linear master frame for a camera exposure. The film
    output clips at 1.0, so the scene is rendered at `master_scale` times the
    reference light power (low enough that nothing clips) and exposure, where
    1.0 is the reference power, is applied afterwards."""
    if exposure <= 0 or master_scale <= 0:
        raise ValueError("exposure and master_scale must be positive")
    return exposure / master_scale


def camera_response_lut(exposure, gamma, bits, size):
    """(xs, ys) lookup table of `camera_response` over scene-linear 0..1, for
    interpolating a whole image in the Blender script."""
    xs = [i / (size - 1) for i in range(size)]
    return xs, [camera_response(x, exposure, gamma, bits) for x in xs]


# Light rig geometry, stated assumptions: the back panel sits a
# clearance below the stack; the front bars sit just outside the camera's field.
BACK_CLEARANCE_MM = 50.0
LIGHT_MARGIN_MM = 50.0
FRONT_BAR_WIDTH_MM = 100.0
# The back panel is sized per frame, so its power scales with its area to keep
# the same radiance in every window. The
# requested back power is that of the panel at the scripts' default 100 mm
# window, where the light-power sweep was made.
BACK_REFERENCE_AREA_MM2 = 1210.16 * 800.0


def light_rig(plan, back_power_w, front_to_back_ratio, front_angle_deg):
    """Transflected illumination as light sources, in web coordinates (mm; x
    across, y along, z up with the top of the PET at 0). A large diffuse back
    panel under the web gives the transmitted light; two front bars either
    side of the camera axis, angled `front_angle_deg` off vertical and aimed at
    the frame centre, give the reflected light, keeping their mirror
    reflection out of the perpendicular camera. Total front power is
    `front_to_back_ratio` x the back power, split evenly."""
    if front_to_back_ratio < 0:
        raise ValueError("front_to_back_ratio must not be negative")
    if not 0.0 < front_angle_deg < 90.0:
        raise ValueError("front_angle_deg must be between 0 and 90 degrees")
    cam = plan["camera"]
    breadth_mm = plan["web"]["breadth_mm"]
    film_y0 = min(f["y0_mm"] for f in plan["film"])
    film_y1 = max(f["y1_mm"] for f in plan["film"])
    deepest_mm = max(f["z_bottom_um"] for f in plan["film"]) / 1000.0
    back_x, back_y, back_z = breadth_mm / 2, (film_y0 + film_y1) / 2, -(deepest_mm + BACK_CLEARANCE_MM)
    back_area_mm2 = (breadth_mm + 2 * LIGHT_MARGIN_MM) * ((film_y1 - film_y0) + 2 * LIGHT_MARGIN_MM)
    lights = [
        {
            "name": "back",
            "role": "back",
            "size_x_mm": breadth_mm + 2 * LIGHT_MARGIN_MM,
            "size_y_mm": (film_y1 - film_y0) + 2 * LIGHT_MARGIN_MM,
            "position_mm": (back_x, back_y, back_z),
            "aim_mm": (back_x, back_y, back_z + 1.0),
            "power_w": back_power_w * back_area_mm2 / BACK_REFERENCE_AREA_MM2,
        }
    ]
    # The bars sit just outside the camera's field along the MD (an orthographic
    # camera would otherwise see them as glowing bands); the angle then sets how
    # high they hang.
    md_half_mm = cam["ortho_scale_mm"] * cam["height_px"] / cam["width_px"] / 2
    run_mm = md_half_mm + FRONT_BAR_WIDTH_MM / 2 + LIGHT_MARGIN_MM
    height_mm = run_mm / math.tan(math.radians(front_angle_deg))
    for name, sign in (("front_a", -1), ("front_b", 1)):
        lights.append(
            {
                "name": name,
                "role": "front",
                "size_x_mm": cam["ortho_scale_mm"] + 2 * LIGHT_MARGIN_MM,
                "size_y_mm": FRONT_BAR_WIDTH_MM,
                "position_mm": (cam["x_mm"], cam["y_mm"] + sign * run_mm, height_mm),
                "aim_mm": (cam["x_mm"], cam["y_mm"], 0.0),
                "power_w": front_to_back_ratio * back_power_w / 2,
            }
        )
    return lights

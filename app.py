# app.py

import streamlit as st
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import mount_builder           # unchanged; ensure mount_builder.py is in the same directory
from stl import mesh           # requires: pip install numpy-stl

st.set_page_config(page_title="Airfoil Toolkit", layout="centered")
st.title("Airfoil Toolkit")

# ─────────── Helper Functions ───────────

def load_dat_coords(dat_bytes):
    """
    Parse a .dat file’s bytes into an (N×2) numpy array of (x, y).
    Returns an empty (0×2) array if parsing fails or fewer than 3 valid points.
    """
    raw_lines = dat_bytes.decode("utf-8", errors="ignore").splitlines()
    coord_lines = raw_lines[1:]  # skip any header line
    pts = []
    for line in coord_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
                pts.append([x_val, y_val])
            except:
                continue
    pts = np.array(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        return np.zeros((0, 2))
    return pts

def compute_polygon_centroid(points):
    """
    Compute the centroid (Cx, Cy) of a 2D polygon given by points (N×2).
    If invalid, returns (0.0, 0.0).
    """
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        return 0.0, 0.0
    x = points[:, 0]
    y = points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    a = x * y_next - x_next * y
    area = 0.5 * np.sum(a)
    if abs(area) < 1e-9:
        return x.mean(), y.mean()
    Cx = (1.0 / (6.0 * area)) * np.sum((x + x_next) * a)
    Cy = (1.0 / (6.0 * area)) * np.sum((y + y_next) * a)
    return Cx, Cy

def compute_midchord_center(points):
    """
    Compute mid-chord center: X = (minX+maxX)/2, Y = midpoint of upper & lower surfaces at that X.
    If invalid, returns (0.0, 0.0).
    """
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        return 0.0, 0.0
    xs = points[:, 0]
    ys = points[:, 1]
    minx, maxx = xs.min(), xs.max()
    midx = 0.5 * (minx + maxx)
    intersect_ys = []
    N = points.shape[0]
    for i in range(N):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % N]
        if (x1 - midx) * (x2 - midx) <= 0 and abs(x2 - x1) > 1e-9:
            t = (midx - x1) / (x2 - x1)
            y_at = y1 + t * (y2 - y1)
            intersect_ys.append(y_at)
    if len(intersect_ys) < 2:
        return midx, ys.mean()
    y_top = max(intersect_ys)
    y_bot = min(intersect_ys)
    midy = 0.5 * (y_top + y_bot)
    return midx, midy

def make_box(origin, length, width, height):
    """
    Build a rectangular prism (box).
    origin = (x0, y0, z0), length = size along X, width = size along Y, height = size along Z.
    Returns (verts, faces).
    """
    x0, y0, z0 = origin
    dx, dy, dz = length, width, height
    verts = np.array([
        [x0,     y0,     z0],
        [x0+dx,  y0,     z0],
        [x0+dx,  y0+dy,  z0],
        [x0,     y0+dy,  z0],
        [x0,     y0,     z0+dz],
        [x0+dx,  y0,     z0+dz],
        [x0+dx,  y0+dy,  z0+dz],
        [x0,     y0+dy,  z0+dz],
    ], dtype=float)
    faces = np.array([
        [0,1,2], [0,2,3],       # bottom
        [4,6,5], [4,7,6],       # top
        [0,5,1], [0,4,5],       # front (y=y0)
        [3,2,6], [3,6,7],       # back  (y=y0+dy)
        [0,3,7], [0,7,4],       # left  (x=x0)
        [1,5,6], [1,6,2],       # right (x=x0+dx)
    ], dtype=int)
    return verts, faces

def make_tapered_hex_prism(cx, cy, top_f2f, bot_f2f, depth, z_top=0.0):
    """
    Build a tapered hexagonal prism for centroid testing:
    - Top face (Z = z_top) has flat-to-flat = top_f2f
    - Bottom face (Z = z_top - depth) has flat-to-flat = bot_f2f
    Returns (verts, faces).
    """
    def hex_corners(f2f, x0, y0, z):
        R = (f2f / 2.0) / np.cos(np.pi/6)
        pts = []
        for k in range(6):
            theta = np.pi/6 + k * (np.pi/3)
            x = x0 + R * np.cos(theta)
            y = y0 + R * np.sin(theta)
            pts.append([x, y, z])
        return np.array(pts, dtype=float)

    top_hex = hex_corners(top_f2f, cx, cy, z_top)
    bot_hex = hex_corners(bot_f2f, cx, cy, z_top - depth)
    verts = np.vstack([top_hex, bot_hex])
    faces = []
    for i in range(6):
        i_next = (i + 1) % 6
        top_i     = i
        top_inext = i_next
        bot_i     = i + 6
        bot_inext = i_next + 6
        faces.append([top_i, top_inext, bot_inext])
        faces.append([top_i, bot_inext, bot_i])
    for i in range(1, 5):
        faces.append([0, i, i + 1])
    faces.append([0, 5, 1])
    base = 6
    for i in range(1, 5):
        faces.append([base, base + i + 1, base + i])
    faces.append([base, base + 7, base + 11])
    return verts, np.array(faces, dtype=int)

def subtract_tapered_hex(verts, faces, cx, cy, top_f2f, bot_f2f, depth, foil_thickness):
    """
    Remove any triangular face whose centroid falls inside the tapered hex volume:
    - Hex top at Z=foil_thickness has flat-to-flat = top_f2f
    - Hex bottom at Z=foil_thickness - depth has flat-to-flat = bot_f2f
    Returns (new_verts, new_faces) with unused vertices trimmed.
    """
    z_top = foil_thickness
    z_bot = z_top - depth
    kept = []
    cos30 = np.cos(np.pi/6)
    sin30 = np.sin(np.pi/6)

    for tri in faces:
        centroid = verts[tri].mean(axis=0)
        x_c, y_c, z_c = centroid
        if not (z_bot <= z_c <= z_top):
            kept.append(tri)
            continue
        lam = (z_top - z_c) / depth
        f2f_at_z = top_f2f - lam * (top_f2f - bot_f2f)
        dx = x_c - cx
        dy = y_c - cy
        x_r = dx * cos30 + dy * sin30
        y_r = -dx * sin30 + dy * cos30
        if max(abs(x_r), abs(y_r) / cos30) <= (f2f_at_z / 2.0):
            # Inside hex → remove
            continue
        kept.append(tri)

    if not kept:
        return np.zeros((0,3)), np.zeros((0,3), dtype=int)

    kept = np.array(kept, dtype=int)
    unique_v = np.unique(kept.flatten())
    idx_map = {old: new for new, old in enumerate(unique_v)}
    new_verts = verts[unique_v]
    new_faces = np.vectorize(lambda i: idx_map[i])(kept)
    return new_verts, new_faces

def extrude_airfoil_with_hex_hole(dat_coords, thickness_mm,
                                  top_f2f, bot_f2f, depth,
                                  scale, center_x, center_y):
    """
    1) Extrude the 2D foil (N×2) into a 3D plate (z=0 to z=thickness_mm).
    2) Carve a tapered hex hole centered at (center_x, center_y).
    3) Scale X/Y by 'scale' (Z remains in absolute mm).
    Returns (verts_carved, faces_carved).
    """
    foil_2d = dat_coords.copy()
    N = foil_2d.shape[0]
    bottom_layer = np.concatenate([foil_2d, np.zeros((N,1))], axis=1)
    top_layer    = np.concatenate([foil_2d, np.ones((N,1)) * thickness_mm], axis=1)
    verts = np.vstack([bottom_layer, top_layer])
    faces = []
    for i in range(N):
        i_next = (i + 1) % N
        faces.append([i, i_next, N + i_next])
        faces.append([i, N + i_next, N + i])
    faces = np.array(faces, dtype=int)

    # Prepare a scaled copy for carving tests
    verts_for_carving = verts.copy()
    verts_for_carving[:, 0:2] *= scale
    cx_scaled = center_x * scale
    cy_scaled = center_y * scale
    top_f2f_s = top_f2f * scale
    bot_f2f_s = bot_f2f * scale
    depth_s   = depth  # Z not scaled

    verts_carved, faces_carved = subtract_tapered_hex(
        verts_for_carving, faces,
        cx_scaled, cy_scaled,
        top_f2f_s, bot_f2f_s, depth_s,
        foil_thickness=thickness_mm
    )

    return verts_carved, faces_carved

def draw_hex_2d(ax, cx, cy, f2f):
    """
    Draw a 2D hexagon (flat-to-flat = f2f) centered at (cx, cy) on the given Matplotlib axis.
    """
    R = (f2f / 2.0) / np.cos(np.pi/6)
    angles = [np.pi/6 + k * (np.pi/3) for k in range(6)]  # 30°, 90°, ...
    verts = [(cx + R * np.cos(theta), cy + R * np.sin(theta)) for theta in angles]
    polygon = Polygon(verts, closed=True, fill=False, edgecolor="red", linewidth=1.5)
    ax.add_patch(polygon)

# ─────────── Tabs ───────────

tab1, tab2, tab3 = st.tabs([
    "View Airfoil",
    "Extrude to STL (► Tapered Hex Hole)",
    "Build Mount"
])

# ─────────── Tab 1: View Airfoil ───────────

with tab1:
    st.header("👁 View Airfoil")

    uploaded_dat = st.file_uploader("Upload a .dat file", type="dat")
    if uploaded_dat:
        dat_bytes = uploaded_dat.read()
        coords = load_dat_coords(dat_bytes)

        if coords.size == 0:
            st.error("Unable to parse valid (x, y) points from the .dat file.")
        else:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(coords[:,0], coords[:,1], "-b", linewidth=1.5)
            ax.set_aspect("equal", "box")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_title("2D Airfoil Profile")
            st.pyplot(fig)

# ─────────── Tab 2: Extrude to STL with Tapered Hex Hole ───────────

with tab2:
    st.header("🌬 Extrude Airfoil + Tapered Hexagonal Hole")

    uploaded_dat2 = st.file_uploader("Upload a .dat file", type="dat", key="extrude_dat")
    foil_thickness = st.number_input(
        "Foil thickness (depth in Z) [mm]",
        min_value=0.5, value=5.0, step=0.5
    )
    scale_factor = st.number_input(
        "X/Y scale factor",
        min_value=0.01, max_value=10.0,
        value=1.0, step=0.01,
        help="Multiply all X and Y dims by this factor."
    )

    st.markdown("---")
    st.markdown("#### Hex-Hole Positioning Method")
    hole_loc_method = st.radio(
        "Choose where to place the hex hole:",
        ("Centroid of foil", "Mid­chord (center X, mid Y at that X)", "Custom (use 2D graph)"),
    )

    st.markdown("---")
    st.markdown("#### Hex-Hole Geometry (all in mm)")

    hex_top_f2f = st.number_input(
        "Hexagon top flat-to-flat [mm]", min_value=1.0, value=6.0, step=0.1
    )
    hex_bot_f2f = st.number_input(
        "Hexagon bottom flat-to-flat [mm]", min_value=0.1, value=5.8, step=0.1
    )
    hex_depth = st.number_input(
        "Hexagon hole depth [mm]", min_value=0.1, max_value=foil_thickness,
        value=foil_thickness, step=0.5
    )

    st.markdown("---")

    if uploaded_dat2:
        dat_bytes2 = uploaded_dat2.read()
        coords2 = load_dat_coords(dat_bytes2)

        if coords2.size == 0:
            st.error("Unable to parse valid (x, y) points from the .dat file.")
        else:
            # Compute centroid and midchord by default
            Cx_centroid, Cy_centroid = compute_polygon_centroid(coords2)
            Cx_midchord, Cy_midchord = compute_midchord_center(coords2)

            # If custom, show 2D plot allowing user to move hex via sliders
            if hole_loc_method == "Custom (use 2D graph)":
                st.markdown("**Click 'Reset to Centroid' to return to centroid.**")
                # Let the user type in the hex‐center X and Y (default = centroid):
                custom_x = st.number_input("Hex center X-coordinate [mm]", value=float(Cx_centroid))
                custom_y = st.number_input("Hex center Y-coordinate [mm]", value=float(Cy_centroid))

                # Create the 2D preview
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.plot(coords2[:,0], coords2[:,1], "-b", linewidth=1.5, label="Foil Profile")
                draw_hex_2d(ax2, custom_x, custom_y, hex_top_f2f)

                # Compute a combined bounding box for foil and hex (top f2f)
                # 1) Foil bounds:
                foil_minx, foil_maxx = coords2[:,0].min(), coords2[:,0].max()
                foil_miny, foil_maxy = coords2[:,1].min(), coords2[:,1].max()

                # 2) Hex bounding box (assuming flat‐to‐flat = hex_top_f2f)
                # The circumradius of that hex is R = (f2f/2) / cos(30°):
                R = (hex_top_f2f / 2.0) / np.cos(np.pi/6)
                hex_minx = custom_x - R
                hex_maxx = custom_x + R
                hex_miny = custom_y - R
                hex_maxy = custom_y + R

                # 3) Combine both sets of bounds:
                overall_minx = min(foil_minx, hex_minx)
                overall_maxx = max(foil_maxx, hex_maxx)
                overall_miny = min(foil_miny, hex_miny)
                overall_maxy = max(foil_maxy, hex_maxy)

                # 4) Add a small 5% margin around that combined box:
                dx = overall_maxx - overall_minx
                dy = overall_maxy - overall_miny
                margin_x = dx * 0.05 if dx > 0 else 1.0
                margin_y = dy * 0.05 if dy > 0 else 1.0

                ax2.set_xlim(overall_minx - margin_x, overall_maxx + margin_x)
                ax2.set_ylim(overall_miny - margin_y, overall_maxy + margin_y)

                ax2.set_aspect("equal", "box")
                ax2.set_xlabel("x [mm]")
                ax2.set_ylabel("y [mm]")
                ax2.set_title("2D Airfoil with Hex Overlay (Custom Placement)")
                ax2.legend(loc="upper right")
                st.pyplot(fig2)

                Cx, Cy = custom_x, custom_y


            elif hole_loc_method == "Centroid of foil":
                Cx, Cy = Cx_centroid, Cy_centroid
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.plot(coords2[:,0], coords2[:,1], "-b", linewidth=1.5)
                draw_hex_2d(ax2, Cx, Cy, hex_top_f2f)
                ax2.set_aspect("equal", "box")
                ax2.set_xlabel("x [mm]")
                ax2.set_ylabel("y [mm]")
                ax2.set_title("2D Airfoil with Hex at Centroid")
                st.pyplot(fig2)

            else:  # Midchord
                Cx, Cy = Cx_midchord, Cy_midchord
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.plot(coords2[:,0], coords2[:,1], "-b", linewidth=1.5)
                draw_hex_2d(ax2, Cx, Cy, hex_top_f2f)
                ax2.set_aspect("equal", "box")
                ax2.set_xlabel("x [mm]")
                ax2.set_ylabel("y [mm]")
                ax2.set_title("2D Airfoil with Hex at Mid­chord")
                st.pyplot(fig2)

            # Estimate bounding box
            verts_dummy, faces_dummy = extrude_airfoil_with_hex_hole(
                dat_coords=coords2,
                thickness_mm=foil_thickness,
                top_f2f=hex_top_f2f,
                bot_f2f=hex_bot_f2f,
                depth=hex_depth,
                scale=scale_factor,
                center_x=Cx,
                center_y=Cy
            )
            if verts_dummy.size > 0:
                min_xyz = verts_dummy.min(axis=0)
                max_xyz = verts_dummy.max(axis=0)
                dx = max_xyz[0] - min_xyz[0]
                dy = max_xyz[1] - min_xyz[1]
                dz = max_xyz[2] - min_xyz[2]
                st.markdown("**Estimated final dimensions (mm)**")
                st.write(f"• X (length) = {dx:.2f} mm")
                st.write(f"• Y (width)  = {dy:.2f} mm")
                st.write(f"• Z (height) = {dz:.2f} mm")
            else:
                st.markdown("**Error:** Hex hole removed entire mesh. Try smaller hex or shallower depth.")
            st.markdown("---")

    # Generate STL
    if uploaded_dat2 and coords2.size > 0 and st.button("Generate foil_with_hex_hole.stl"):
        if hole_loc_method == "Custom (use 2D graph)":
            # Reuse custom_x, custom_y from above
            Cx, Cy = float(custom_x), float(custom_y)
        elif hole_loc_method == "Centroid of foil":
            Cx, Cy = Cx_centroid, Cy_centroid
        else:
            Cx, Cy = Cx_midchord, Cy_midchord

        verts_final, faces_final = extrude_airfoil_with_hex_hole(
            dat_coords=coords2,
            thickness_mm=foil_thickness,
            top_f2f=hex_top_f2f,
            bot_f2f=hex_bot_f2f,
            depth=hex_depth,
            scale=scale_factor,
            center_x=Cx,
            center_y=Cy
        )

        def make_stl_bytes(verts, faces):
            buf = io.BytesIO()
            foil_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    foil_mesh.vectors[i][j] = verts[f[j], :]
            foil_mesh.save(buf, mode=mesh.Mode.ASCII)
            buf.seek(0)
            return buf.getvalue()

        if verts_final.size == 0:
            st.error("Failed to generate STL—hex hole removed all triangles.")
        else:
            st.success("✔ Foil + Hex Hole STL is ready!")
            st.download_button(
                label="Download foil_with_hex_hole.stl",
                data=make_stl_bytes(verts_final, faces_final),
                file_name="foil_with_hex_hole.stl",
                mime="application/octet-stream"
            )
            # Optional wireframe preview
            fig3 = plt.figure(figsize=(5, 3))
            ax3 = fig3.add_subplot(111, projection="3d")
            for tri in faces_final:
                pts = verts_final[tri]
                for e in range(3):
                    xs = [pts[e][0], pts[(e+1)%3][0]]
                    ys = [pts[e][1], pts[(e+1)%3][1]]
                    zs = [pts[e][2], pts[(e+1)%3][2]]
                    ax3.plot(xs, ys, zs, color="gray", linewidth=0.4)
            ax3.set_title("Wireframe: Foil + Hex Hole")
            ax3.set_xlabel("X (mm)"); ax3.set_ylabel("Y (mm)"); ax3.set_zlabel("Z (mm)")
            ax3.set_box_aspect((1, 0.5, 0.2))
            st.pyplot(fig3)

# ─────────── Tab 3: Build Mount ───────────

with tab3:
    st.header("🛠 Build Wind-Tunnel Mount")
    st.markdown("All dimensions below are in millimeters (mm).")

    plate_W = st.number_input("Plate width (W) [mm]", min_value=10.0, value=50.0, step=1.0)
    plate_H = st.number_input("Plate height (H) [mm]", min_value=10.0, value=70.0, step=1.0)
    plate_T = st.number_input("Plate thickness (T) [mm]", min_value=1.0, value=5.0, step=0.5)

    spacing = st.number_input("Spacing between vertical plates (S) [mm]", min_value=1.0, value=5.0, step=1.0)

    hole_r_vert = st.number_input("Vertical plate hole radius [mm]", min_value=0.5, value=3.0, step=0.5)
    hole_offset = st.number_input("Vertical hole offset from bottom [mm]", min_value=0.0, value=20.0, step=1.0)

    base_th = st.number_input("Base plate thickness [mm]", min_value=1.0, value=3.0, step=0.5)

    st.markdown("**Base‐plate mounting‐hole offsets**")
    sep_x = st.number_input("Base hole offset from X edge [mm]", min_value=0.0, value=10.0, step=1.0)
    sep_y = st.number_input("Base hole offset from Y edge [mm]", min_value=0.0, value=10.0, step=1.0)
    hole_r_base = st.number_input("Base plate hole radius [mm]", min_value=0.5, value=3.0, step=0.5)

    base_len = 2 * plate_T + spacing
    base_wid = plate_W
    base_hole_centers = [
        (sep_x,             sep_y,            -base_th/2),
        (base_len - sep_x,  sep_y,            -base_th/2),
        (base_len - sep_x,  base_wid - sep_y, -base_th/2),
        (sep_x,             base_wid - sep_y, -base_th/2),
    ]

    if st.button("Generate Mount STL"):
        params = {
            'plate_W': plate_W,
            'plate_H': plate_H,
            'plate_T': plate_T,
            'spacing_between_plates': spacing,
            'hole_radius': hole_r_vert,
            'hole_offset_vert': hole_offset,
            'base_plate_length': base_len,
            'base_plate_width': base_wid,
            'base_plate_thick': base_th,
            'base_hole_radius': hole_r_base,
            'base_hole_centers': base_hole_centers
        }
        verts_mount, faces_mount = mount_builder.assemble_mount(params)

        buf = io.BytesIO()
        mount_mesh = mesh.Mesh(np.zeros(faces_mount.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces_mount):
            for j in range(3):
                mount_mesh.vectors[i][j] = verts_mount[f[j], :]
        mount_mesh.save(buf, mode=mesh.Mode.ASCII)
        buf.seek(0)

        st.success("✔ Mount geometry generated successfully!")
        st.download_button(
            label="Download wind_tunnel_mount.stl",
            data=buf.getvalue(),
            file_name="wind_tunnel_mount.stl",
            mime="application/octet-stream"
        )

        # Optional bracket preview
        fig4 = plt.figure(figsize=(4, 4))
        ax4 = fig4.add_subplot(111, projection="3d")
        for tri in faces_mount:
            pts = verts_mount[tri]
            for e in range(3):
                xs = [pts[e][0], pts[(e+1)%3][0]]
                ys = [pts[e][1], pts[(e+1)%3][1]]
                zs = [pts[e][2], pts[(e+1)%3][2]]
                ax4.plot(xs, ys, zs, color="gray", linewidth=0.5)
        ax4.set_box_aspect((1,1,1))
        ax4.set_xlabel("X (mm)"); ax4.set_ylabel("Y (mm)"); ax4.set_zlabel("Z (mm)")
        st.pyplot(fig4)

    st.markdown(
        """
        **How to use this bracket**  
        1. Print “wind_tunnel_mount.stl” on your 3D printer.  
        2. Bolt the two vertical plates to your wind‐tunnel’s side rings using M6 bolts (holes are Ø6 mm).  
        3. Bolt the base plate to your test stand or table.  
        """
    )

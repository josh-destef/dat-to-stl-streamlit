# app.py

import streamlit as st
import numpy as np
import io
import matplotlib.pyplot as plt

# ────────── External Imports ──────────
import mount_builder           # unchanged from before
from stl import mesh           # requires: pip install numpy-stl

st.set_page_config(page_title="Airfoil Toolkit", layout="centered")
st.title("Airfoil Toolkit")


# ────────── Define Tabs ──────────
tab1, tab2, tab3 = st.tabs([
    "View Airfoil",
    "Extrude to STL (► Tapered Hex Hole)",
    "Build Mount"
])


# ────────── Tab 1: View Airfoil ──────────
with tab1:
    st.header("👁 View Airfoil")

    uploaded_dat = st.file_uploader("Upload a .dat file", type="dat")
    if uploaded_dat:
        dat_bytes = uploaded_dat.read()

        # ─── Helper: load_dat_coords ───
        def load_dat_coords(dat_bytes):
            raw_lines = dat_bytes.decode("utf-8").splitlines()
            coord_lines = raw_lines[1:]  # skip header/description
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
                        pass
            return np.array(pts)

        coords = load_dat_coords(dat_bytes)

        # ─── Helper: plot_airfoil_2d ───
        def plot_airfoil_2d(coords):
            xs, ys = coords[:, 0], coords[:, 1]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(xs, ys, "-b", linewidth=1.5)
            ax.set_aspect("equal", "box")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_title("2D Airfoil Profile")
            return fig

        fig1 = plot_airfoil_2d(coords)
        st.pyplot(fig1)


# ────────── Tab 2: Extrude to STL with Tapered Hex Hole ──────────
with tab2:
    st.header("🌬 Extrude Airfoil + Tapered Hexagonal Hole")

    # 1) Upload a .dat file
    uploaded_dat2 = st.file_uploader("Upload a .dat file", type="dat", key="extrude_dat")

    # 2) Foil thickness in Z (mm)
    foil_thickness = st.number_input(
        "Foil thickness (depth in Z) [mm]",
        min_value=0.5, value=5.0, step=0.5
    )

    # 3) Optional uniform X/Y scale factor
    scale_factor = st.number_input(
        "X/Y scale factor",
        min_value=0.01, max_value=10.0,
        value=1.0, step=0.01,
        help="Multiply all X and Y dimensions by this factor."
    )

    st.markdown("---")
    st.markdown("#### Hex-Hole Positioning Method")
    hole_loc_method = st.radio(
        "Choose where to place the hex hole:",
        ("Centroid of foil", "Mid­chord (center X, mid Y at that X)")
    )

    st.markdown("---")
    st.markdown("#### Hex-Hole Geometry (all in millimeters)")

    # 4) Top flat-to-flat (at Z = foil_thickness)
    hex_top_f2f = st.number_input(
        "Hexagon top flat-to-flat [mm]",
        min_value=1.0, value=6.0, step=0.1
    )
    # 5) Bottom flat-to-flat (at Z = foil_thickness - hex_depth)
    hex_bot_f2f = st.number_input(
        "Hexagon bottom flat-to-flat [mm]",
        min_value=0.1, value=5.8, step=0.1
    )
    # 6) Hex hole depth (must be ≤ foil_thickness if you want it piercing)
    hex_depth = st.number_input(
        "Hexagon hole depth [mm]",
        min_value=0.1, max_value=foil_thickness, value=foil_thickness, step=0.5
    )

    st.markdown("---")

    # ─── Helper: load_dat_coords (duplicate for Tab 2) ───
    def load_dat_coords(dat_bytes):
        raw_lines = dat_bytes.decode("utf-8").splitlines()
        coord_lines = raw_lines[1:]
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
                    pass
        return np.array(pts)

    # ─── Helper: compute_polygon_centroid ───
    def compute_polygon_centroid(points):
        """
        Given an (N×2) sequence of (x,y) defining a polygon (must be closed or nearly so),
        compute its centroid via the shoelace formula. If area≈0, fallback to average.
        """
        x = points[:, 0]
        y = points[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        a = x * y_next - x_next * y
        area = 0.5 * np.sum(a)
        if abs(area) < 1e-9:
            return np.mean(x), np.mean(y)
        Cx = (1.0 / (6.0 * area)) * np.sum((x + x_next) * a)
        Cy = (1.0 / (6.0 * area)) * np.sum((y + y_next) * a)
        return Cx, Cy

    # ─── Helper: compute_midchord_center ───
    def compute_midchord_center(points):
        """
        Find the mid-chord X (=(minX+maxX)/2). Then for that X, find
        the two intersection Ys (upper & lower) by linearly interpolating
        the foil polygon edges. Return (midX, midY), where midY = (Y_top+Y_bot)/2.
        """
        xs = points[:, 0]
        ys = points[:, 1]
        minx, maxx = xs.min(), xs.max()
        midx = 0.5 * (minx + maxx)

        intersect_ys = []
        N = points.shape[0]
        for i in range(N):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % N]
            # Check if the segment (x1→x2) crosses x = midx
            if (x1 - midx) * (x2 - midx) <= 0 and abs(x2 - x1) > 1e-9:
                t = (midx - x1) / (x2 - x1)
                y_at = y1 + t * (y2 - y1)
                intersect_ys.append(y_at)
        if len(intersect_ys) < 2:
            # Degenerate or very narrow: fallback to average Y
            return midx, np.mean(ys)
        # Usually, we get exactly 2 intersection Ys: one upper, one lower
        y_top = max(intersect_ys)
        y_bot = min(intersect_ys)
        midy = 0.5 * (y_top + y_bot)
        return midx, midy

    # ─── Helper: make_box (rectangular prism) ───
    def make_box(origin, length, width, height):
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
            [0,1,2], [0,2,3],      # bottom
            [4,6,5], [4,7,6],      # top
            [0,5,1], [0,4,5],      # front (y=y0)
            [3,2,6], [3,6,7],      # back  (y=y0+dy)
            [0,3,7], [0,7,4],      # left  (x=x0)
            [1,5,6], [1,6,2],      # right (x=x0+dx)
        ], dtype=int)
        return verts, faces

    # ─── Helper: build a tapered hex prism for centroid testing ───
    def make_tapered_hex_prism(cx, cy, top_f2f, bot_f2f, depth, z_top=0.0):
        """
        Returns (verts, faces) of a tapered hex prism:
         - top face (Z = z_top) has flat-to-flat = top_f2f
         - bottom face (Z = z_top - depth) has flat-to-flat = bot_f2f
        """
        def hex_corners(f2f, x0, y0, z):
            R = (f2f / 2.0) / np.cos(np.pi/6)  # circumradius
            pts = []
            for k in range(6):
                theta = np.pi/6 + k * (np.pi/3)  # 30°, 90°, 150°, …
                x = x0 + R * np.cos(theta)
                y = y0 + R * np.sin(theta)
                pts.append([x, y, z])
            return np.array(pts, dtype=float)

        top_hex = hex_corners(top_f2f, cx, cy, z_top)
        bot_hex = hex_corners(bot_f2f, cx, cy, z_top - depth)
        verts = np.vstack([top_hex, bot_hex])  # shape = (12, 3)
        faces = []

        # Side faces (6 quads → 12 triangles)
        for i in range(6):
            i_next = (i + 1) % 6
            top_i     = i
            top_inext = i_next
            bot_i     = i + 6
            bot_inext = i_next + 6
            faces.append([top_i, top_inext, bot_inext])
            faces.append([top_i, bot_inext, bot_i])

        # Top cap (fan around vertex 0)
        for i in range(1, 5):
            faces.append([0, i, i + 1])
        faces.append([0, 5, 1])

        # Bottom cap (fan around vertex 6)
        base = 6
        for i in range(1, 5):
            faces.append([base, base + i + 1, base + i])
        faces.append([base, base + 7, base + 11])

        return verts, np.array(faces, dtype=int)

    # ─── Helper: carve out triangles whose centroids fall inside tapered hex prism ───
    def subtract_tapered_hex(verts, faces, cx, cy, top_f2f, bot_f2f, depth):
        """
        Given a mesh (verts, faces), remove any triangle whose centroid lies
        inside the tapered hex prism defined by (cx, cy, top_f2f, bot_f2f, depth).
        Returns (new_verts, new_faces) with unused vertices trimmed out.
        """
        # We only need the hex verts to infer shape, but we'll do centroid tests directly
        z_top = foil_thickness
        z_bot = z_top - depth

        kept = []
        for tri in faces:
            centroid = verts[tri].mean(axis=0)
            x_c, y_c, z_c = centroid

            # 1) Z check: outside if not in [z_bot..z_top]
            if not (z_bot <= z_c <= z_top):
                kept.append(tri)
                continue

            # 2) At that Z, find f2f via linear interp
            lam = (z_top - z_c) / depth  # 0→1 as z goes from z_top to z_bot
            f2f_at_z = top_f2f - lam * (top_f2f - bot_f2f)

            # 3) Test (x_c, y_c) in a non-rotated hex of flat-to-flat = f2f_at_z, centered at (cx, cy)
            dx = x_c - cx
            dy = y_c - cy

            cos30 = np.cos(np.pi/6)
            sin30 = np.sin(np.pi/6)
            # Rotate by -30° so that hex flats align with axes
            x_r = dx * cos30 + dy * sin30
            y_r = -dx * sin30 + dy * cos30

            if max(abs(x_r), abs(y_r) / cos30) <= (f2f_at_z / 2.0):
                # inside hex cross-section → remove
                continue
            else:
                kept.append(tri)

        if not kept:
            return np.zeros((0,3)), np.zeros((0,3), dtype=int)

        kept = np.array(kept, dtype=int)
        unique_v = np.unique(kept.flatten())
        idx_map = {old: new for new, old in enumerate(unique_v)}
        new_verts = verts[unique_v]
        new_faces = np.vectorize(lambda i: idx_map[i])(kept)
        return new_verts, new_faces

    # ─── Helper: Extrude foil → 3D plate, carve hex hole, scale X/Y ───
    def extrude_airfoil_with_hex_hole(dat_coords, thickness_mm,
                                      top_f2f, bot_f2f, depth, scale,
                                      center_x, center_y):
        """
        1) Build a 3D plate from the 2D foil polygon (N×2) by extruding in Z.
        2) Carve out a tapered hex hole centered at (center_x, center_y) in XY.
        3) Scale the final mesh’s X/Y by 'scale' (Z remains as absolute mm).
        Returns: (final_verts, final_faces).
        """
        foil_2d = dat_coords.copy()
        N = foil_2d.shape[0]

        bottom_layer = np.concatenate([
            foil_2d,
            np.zeros((N,1))          # z = 0
        ], axis=1)

        top_layer = np.concatenate([
            foil_2d,
            np.ones((N,1)) * thickness_mm  # z = thickness_mm
        ], axis=1)

        verts = np.vstack([bottom_layer, top_layer])  # shape = (2N, 3)
        faces = []
        for i in range(N):
            i_next = (i + 1) % N
            faces.append([i, i_next, N + i_next])
            faces.append([i, N + i_next, N + i])
        faces = np.array(faces, dtype=int)

        # 2) Carve the tapered hex hole
        #    Note: center_x and center_y are given in original coordinate space.
        #    Because we will scale X/Y later, we must test centroids at (center_x*scale, center_y*scale).
        cx_scaled = center_x * scale
        cy_scaled = center_y * scale
        top_f2f_scaled = top_f2f * scale
        bot_f2f_scaled = bot_f2f * scale
        depth_scaled   = depth    # depth is in Z (no scaling for Z)

        # But note: verts are still unscaled. We'll perform scaling only on X/Y after carving.
        # To do centroid‐test in scaled X/Y, we temporarily scale the X/Y of a copy:
        verts_for_carving = verts.copy()
        verts_for_carving[:, 0:2] *= scale  # scale X/Y for carving test

        verts_carved, faces_carved = subtract_tapered_hex(
            verts_for_carving, faces,
            cx_scaled, cy_scaled,
            top_f2f_scaled, bot_f2f_scaled, depth_scaled
        )

        # 3) Now, all 'verts_carved' are already in the scaled coordinate for X/Y.
        #    Z coordinates were never scaled and remain correct. So we can return directly:
        return verts_carved, faces_carved

    # ────────── If user uploaded a .dat, compute bounding-box preview ──────────
    if uploaded_dat2:
        dat_bytes2 = uploaded_dat2.read()
        coords2 = load_dat_coords(dat_bytes2)

        # Determine (center_x, center_y) based on the chosen method
        if hole_loc_method == "Centroid of foil":
            Cx, Cy = compute_polygon_centroid(coords2)
        else:  # "Mid­chord"
            Cx, Cy = compute_midchord_center(coords2)

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

    # ─── “Generate & Download” button ───
    if uploaded_dat2 and st.button("Generate foil_with_hex_hole.stl"):
        dat_bytes2 = uploaded_dat2.read()
        coords2 = load_dat_coords(dat_bytes2)

        if hole_loc_method == "Centroid of foil":
            Cx, Cy = compute_polygon_centroid(coords2)
        else:
            Cx, Cy = compute_midchord_center(coords2)

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

            # Optional: show wireframe preview
            fig2 = plt.figure(figsize=(5, 3))
            ax2 = fig2.add_subplot(111, projection="3d")
            for tri in faces_final:
                pts = verts_final[tri]
                for e in range(3):
                    xs = [pts[e][0], pts[(e+1)%3][0]]
                    ys = [pts[e][1], pts[(e+1)%3][1]]
                    zs = [pts[e][2], pts[(e+1)%3][2]]
                    ax2.plot(xs, ys, zs, color="gray", linewidth=0.4)
            ax2.set_title("Wireframe: Foil + Hex Hole")
            ax2.set_xlabel("X (mm)"); ax2.set_ylabel("Y (mm)"); ax2.set_zlabel("Z (mm)")
            ax2.set_box_aspect((1, 0.5, 0.2))
            st.pyplot(fig2)


# ────────── Tab 3: Build Mount (unchanged) ──────────
with tab3:
    st.header("🛠 Build Wind-Tunnel Mount")
    st.markdown("All dimensions below are in millimeters (mm).")

    # 1) Vertical plate dimensions (W × H × T)
    plate_W = st.number_input("Plate width (W) [mm]", min_value=10.0, value=50.0, step=1.0)
    plate_H = st.number_input("Plate height (H) [mm]", min_value=10.0, value=70.0, step=1.0)
    plate_T = st.number_input("Plate thickness (T) [mm]", min_value=1.0, value=5.0, step=0.5)

    # 2) Spacing between the two vertical plates
    spacing = st.number_input("Spacing between vertical plates (S) [mm]", min_value=1.0, value=5.0, step=1.0)

    # 3) Vertical-plate hole parameters
    hole_r_vert = st.number_input("Vertical plate hole radius [mm]", min_value=0.5, value=3.0, step=0.5)
    hole_offset = st.number_input("Vertical hole offset from bottom [mm]", min_value=0.0, value=20.0, step=1.0)

    # 4) Base plate thickness
    base_th = st.number_input("Base plate thickness [mm]", min_value=1.0, value=3.0, step=0.5)

    # 5) Base-plate hole offsets from edges
    st.markdown("**Base-plate mounting-hole offsets**")
    sep_x = st.number_input("Base hole offset from X edge [mm]", min_value=0.0, value=10.0, step=1.0)
    sep_y = st.number_input("Base hole offset from Y edge [mm]", min_value=0.0, value=10.0, step=1.0)
    hole_r_base = st.number_input("Base plate hole radius [mm]", min_value=0.5, value=3.0, step=0.5)

    # Calculate base plate length & width so it spans under the two vertical plates:
    base_len = 2 * plate_T + spacing      # total X-span
    base_wid = plate_W                     # match plate width

    # Four corner-hole centers (Z is −base_th/2)
    base_hole_centers = [
        (sep_x,            sep_y,            -base_th/2),
        (base_len - sep_x, sep_y,            -base_th/2),
        (base_len - sep_x, base_wid - sep_y, -base_th/2),
        (sep_x,            base_wid - sep_y, -base_th/2),
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

        # Optional: preview bracket wireframe
        def preview_bracket(verts, faces):
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")
            for tri in faces:
                pts = verts[tri]
                for e in range(3):
                    xs = [pts[e][0], pts[(e+1)%3][0]]
                    ys = [pts[e][1], pts[(e+1)%3][1]]
                    zs = [pts[e][2], pts[(e+1)%3][2]]
                    ax.plot(xs, ys, zs, color="gray", linewidth=0.5)
            ax.set_box_aspect((1,1,1))
            ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
            return fig

        fig_mount = preview_bracket(verts_mount, faces_mount)
        st.pyplot(fig_mount)

    st.markdown(
        """
        **How to use this bracket**  
        1. Print “wind_tunnel_mount.stl” on your 3D printer.  
        2. Bolt the two vertical plates to your wind‐tunnel’s side rings using M6 bolts (holes are Ø6 mm).  
        3. Bolt the base plate to your test stand or table.  
        """
    )

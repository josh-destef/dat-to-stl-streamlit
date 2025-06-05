# app.py

import streamlit as st
import numpy as np
import io
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
import pandas as pd
import re
import base64
=======

# ────────── External Imports ──────────
import mount_builder           # unchanged from before
from stl import mesh           # requires: pip install numpy-stl
>>>>>>> Stashed changes

st.set_page_config(page_title="Airfoil Toolkit", layout="centered")
st.title("Airfoil Toolkit")

<<<<<<< Updated upstream
tab1, tab2, tab3 = st.tabs(["View Airfoil", "Extrude to STL", "Metadata"])
=======
>>>>>>> Stashed changes

# ────────── Define Tabs ──────────
tab1, tab2, tab3 = st.tabs([
    "View Airfoil",
    "Extrude to STL (► Tapered Hex Hole)",
    "Build Mount"
])

<<<<<<< Updated upstream
def load_dat_coords(dat_bytes):
    """
    Given raw bytes of a .dat file (UTF-8), skip the first line,
    parse the remaining lines as two-column floats,
    and return an (N×2) numpy array.
    """
    raw_lines = dat_bytes.decode("utf-8", errors="ignore").splitlines()
    coord_lines = raw_lines[1:]  # skip first-line description
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
            except ValueError:
                pass
    return np.array(pts, dtype=float)

def plot_2d_profile(coords, title="Airfoil Profile"):
    """
    Given an (N×2) array of coordinates, plot x vs. y with equal axis.
    Returns the Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(coords[:, 0], coords[:, 1], "-k", linewidth=2)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig

def resample_profile_xy(xy, num_pts=200):
    """
    Fit a non‐periodic spline through the given XY points, then
    resample to num_pts points. Ensures no duplicate endpoint.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    tck, _ = splprep([x, y], s=0, per=0)
    u_new = np.linspace(0.0, 1.0, num_pts)
    out = splev(u_new, tck)
    pts_interp = np.vstack(out).T
    if np.allclose(pts_interp[0], pts_interp[-1], atol=1e-6):
        pts_interp = pts_interp[:-1]
    return pts_interp

def point_in_triangle(pt, a, b, c):
    v0 = c - a
    v1 = b - a
    v2 = pt - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) and (v >= 0) and (u + v < 1)

def is_convex(prev_pt, curr_pt, next_pt):
    v1 = prev_pt - curr_pt
    v2 = next_pt - curr_pt
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    return cross_z < 0

def triangulate_polygon(contour_xy):
    """
    Ear‐clipping triangulation on a CCW 2D contour.
    Returns list of index triples.
    """
    n = len(contour_xy)
    if n < 3:
        return []
    idxs = list(range(n))
    triangles = []
    guard = 0
    while len(idxs) > 3 and guard < n * n:
        guard += 1
        ear_found = False
        for ii in range(len(idxs)):
            i = idxs[ii]
            prev_idx = idxs[ii - 1]
            next_idx = idxs[(ii + 1) % len(idxs)]
            A = contour_xy[prev_idx]
            B = contour_xy[i]
            C = contour_xy[next_idx]
            if not is_convex(A, B, C):
                continue
            is_ear = True
            for other in idxs:
                if other in (prev_idx, i, next_idx):
                    continue
                P = contour_xy[other]
                if point_in_triangle(P, A, B, C):
                    is_ear = False
                    break
            if not is_ear:
                continue
            triangles.append((prev_idx, i, next_idx))
            idxs.pop(ii)
            ear_found = True
            break
        if not ear_found:
            break
    if len(idxs) == 3:
        triangles.append((idxs[0], idxs[1], idxs[2]))
    return triangles

def extrude_to_vertices_faces(xy_points, thickness_mm, num_interp=200):
    """
    Resample xy_points to num_interp, then build vertices/faces for extrusion.
    Returns vertices (2n×3) and faces (F×3).
    """
    contour = resample_profile_xy(xy_points, num_interp)
    n = contour.shape[0]
    vertices = np.zeros((2 * n, 3), dtype=float)
    for i in range(n):
        x_val, y_val = contour[i]
        vertices[i]     = [x_val, y_val, 0.0]
        vertices[n + i] = [x_val, y_val, thickness_mm]
    faces = []
    bottom_triangles = triangulate_polygon(contour)
    for (i, j, k) in bottom_triangles:
        faces.append([i, k, j])
    for (i, j, k) in bottom_triangles:
        faces.append([n + i, n + j, n + k])
    for i in range(n):
        i_next = (i + 1) % n
        b0 = i
        b1 = i_next
        t0 = n + i
        t1 = n + i_next
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
    return vertices, np.array(faces, dtype=int)

def write_stl_ascii(vertices, faces, solid_name="airfoil_extrusion"):
    """
    Write ASCII STL from vertices/faces. Return as string.
    """
    def compute_normal(v1, v2, v3):
        n = np.cross(v2 - v1, v3 - v1)
        length = np.linalg.norm(n)
        if length == 0:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        return n / length
    output = io.StringIO()
    output.write(f"solid {solid_name}\n")
    for tri in faces:
        v1 = vertices[tri[0]]
        v2 = vertices[tri[1]]
        v3 = vertices[tri[2]]
        n = compute_normal(v1, v2, v3)
        output.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        output.write("    outer loop\n")
        output.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
        output.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
        output.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
        output.write("    endloop\n")
        output.write("  endfacet\n")
    output.write(f"endsolid {solid_name}\n")
    return output.getvalue()

def format_num(v):
    """
    Format a float to string: no decimal if integer, else one decimal place.
    """
    return f"{int(v)}" if float(v).is_integer() else f"{v:.1f}"

def extract_naca_code(text):
    """
    Search for NACA 4-digit or 5-digit code in a string (case-insensitive).
    Returns the code string if found, else empty string.
    """
    m = re.search(r"(?i)NACA\s*([0-9]{4,5})", text)
    return m.group(1) if m else ""

def explain_naca4(code):
    """
    Given a 4-digit string, return (m, p, t) and explanation text.
    """
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    return m, p, t

def explain_naca5(code):
    """
    Return placeholder text and link for 5-digit series.
    """
    return (
        "NACA 5-digit decoding is more involved. "
        "Refer to https://en.wikipedia.org/wiki/NACA_airfoil#5-digit_series for details."
    )

# -------------------------
# Tab 1: View Airfoil
# -------------------------
with tab1:
    st.header("View Airfoil Profile")
    dat_file_v = st.file_uploader(
        "Upload `.dat` to view (first line is description)",
        type=["dat"], key="view"
    )
    if dat_file_v:
        coords_v = load_dat_coords(dat_file_v.read())
        st.subheader("2D Profile Plot")
        fig_v = plot_2d_profile(coords_v, title="Uploaded Airfoil")
        st.pyplot(fig_v)
        if st.checkbox("Show coordinate table"):
            df = pd.DataFrame(coords_v, columns=["x", "y"])
            st.dataframe(df, height=200)
    else:
        st.info("Upload a `.dat` file to view the 2D profile.")
=======

# ────────── Tab 1: View Airfoil ──────────
with tab1:
    st.header("👁 View Airfoil")
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    st.header("Extrude Airfoil to STL")
    dat_file_e = st.file_uploader(
        "Upload `.dat` for extrusion (first line is description)",
        type=["dat"], key="extrude"
=======
    st.header("🌬 Extrude Airfoil + Tapered Hexagonal Hole")

    # 1) Upload a .dat file
    uploaded_dat2 = st.file_uploader("Upload a .dat file", type="dat", key="extrude_dat")

    # 2) Foil thickness in Z (mm)
    foil_thickness = st.number_input(
        "Foil thickness (depth in Z) [mm]",
        min_value=0.5, value=5.0, step=0.5
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
        # Determine airfoil code from filename or first-line text
        filename_text = dat_file_e.name or ""
        first_line = raw_bytes.decode("utf-8", errors="ignore").splitlines()[0] if raw_bytes else ""
        code = extract_naca_code(filename_text) or extract_naca_code(first_line) or "airfoil"

        st.subheader("Generated Filename")
        auto_name = f"{code}_{format_num(scale_factor)}_{format_num(thickness)}"
        st.markdown(f"**{auto_name}.stl**")

        st.subheader("Generate & Preview STL")
        if st.button("Create STL"):
            verts, faces = extrude_to_vertices_faces(scaled_coords, thickness, num_interp=num_pts)
            stl_text = write_stl_ascii(verts, faces, solid_name=auto_name)
            stl_bytes = stl_text.encode("utf-8")

            # 1) Show 3D preview via embedded Three.js
            b64 = base64.b64encode(stl_bytes).decode()
            html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8" />
              <title>STL Preview</title>
              <style>
                body {{ margin: 0; }}
                #viewer {{ width: 100%; height: 400px; }}
              </style>
            </head>
            <body>
              <div id="viewer"></div>
              <script src="https://cdn.jsdelivr.net/npm/three@0.150/build/three.min.js"></script>
              <script src="https://cdn.jsdelivr.net/npm/three@0.150/examples/js/loaders/STLLoader.js"></script>
              <script>
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, 2, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, 400);
                document.getElementById('viewer').appendChild(renderer.domElement);

                const ambient = new THREE.AmbientLight(0x404040);
                scene.add(ambient);
                const directional = new THREE.DirectionalLight(0xffffff, 1);
                directional.position.set(1, 1, 1).normalize();
                scene.add(directional);

                const loader = new THREE.STLLoader();
                loader.load('data:application/sla;base64,{b64}', function (geometry) {{
                  const material = new THREE.MeshNormalMaterial();
                  const mesh = new THREE.Mesh(geometry, material);
                  scene.add(mesh);

                  const box = new THREE.Box3().setFromObject(mesh);
                  const size = box.getSize(new THREE.Vector3()).length();
                  const center = box.getCenter(new THREE.Vector3());

                  mesh.position.x += (mesh.position.x - center.x);
                  mesh.position.y += (mesh.position.y - center.y);
                  mesh.position.z += (mesh.position.z - center.z);

                  camera.position.set(center.x, center.y, size * 2);
                  camera.lookAt(center);

                  function animate() {{
                    requestAnimationFrame(animate);
                    mesh.rotation.x += 0.01;
                    mesh.rotation.y += 0.01;
                    renderer.render(scene, camera);
                  }}
                  animate();
                }});
              </script>
            </body>
            </html>
            """
            st.components.v1.html(html, height=420)

            # 2) Download button
            st.download_button(
                label="Download `.stl`",
                data=stl_bytes,
                file_name=f"{auto_name}.stl",
                mime="application/vnd.ms-pki.stl"
            )
            st.success(f"STL `{auto_name}.stl` is ready!")
    else:
        st.info("Upload a `.dat` file to enable extrusion.")

# -------------------------
# Tab 3: Metadata
# -------------------------
with tab3:
    st.header("Airfoil Metadata Parser")
    st.markdown(
        "Upload a `.dat` file or enter a NACA code manually to see parsed information."
    )
    dat_file_m = st.file_uploader(
        "Upload `.dat` for metadata (optional)", type=["dat"], key="meta"
    )
    code_manual = st.text_input(
        "Or enter NACA code (e.g., 2412 or 24121)", value=""
    )

    detected_code = ""
    if dat_file_m:
        first_line_m = dat_file_m.read().decode("utf-8", errors="ignore").splitlines()[0]
        name_from_file = dat_file_m.name or ""
        detected_code = extract_naca_code(name_from_file) or extract_naca_code(first_line_m)
    if code_manual.strip():
        detected_code = code_manual.strip()

    if detected_code:
        st.subheader(f"Parsed Code: {detected_code}")
        if len(detected_code) == 4:
            m, p, t = explain_naca4(detected_code)
            st.markdown(f"- **Max camber (m):** {m*100:.1f}% of chord")
            st.markdown(f"- **Camber location (p):** {p*100:.1f}% of chord")
            st.markdown(f"- **Max thickness (t):** {t*100:.1f}% of chord")
            st.markdown(
                "For more details, see "
                "[NACA 4-digit airfoil](https://en.wikipedia.org/wiki/NACA_airfoil#4-digit_series)."
            )
        elif len(detected_code) == 5:
            st.markdown(explain_naca5(detected_code))
            st.markdown(
                "For more details, see "
                "[NACA 5-digit series](https://en.wikipedia.org/wiki/NACA_airfoil#5-digit_series)."
            )
        else:
            st.error("Unsupported code length. Enter 4 or 5 digits.")
    else:
        st.info("No NACA code detected. Upload a `.dat` or enter a code above.")
=======
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
>>>>>>> Stashed changes

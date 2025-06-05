import streamlit as st
import numpy as np
from scipy.interpolate import splprep, splev
import io
import matplotlib.pyplot as plt
import pandas as pd
import base64

st.set_page_config(page_title="Airfoil Toolkit", layout="centered")
st.title("Airfoil Toolkit")

tab1, tab2 = st.tabs(["View Airfoil", "Extrude to STL"])

# -------------------------
# 1) Shared helper functions
# -------------------------

def load_dat_coords(dat_bytes):
    """
    Given raw bytes of a .dat file (UTF-8), skip the first line,
    parse the remaining lines as two-column floats, and return an (N×2) numpy array.
    """
    raw_lines = dat_bytes.decode("utf-8", errors="ignore").splitlines()
    coord_lines = raw_lines[1:]  # skip first‐line description
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
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
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
    """
    Return True if point pt lies inside triangle (a,b,c) in 2D.
    (Barycentric method.)
    """
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
    """
    Return True if (prev_pt→curr_pt→next_pt) makes a convex “ear” in CCW ordering.
    """
    v1 = prev_pt - curr_pt
    v2 = next_pt - curr_pt
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    return cross_z < 0


def triangulate_polygon(contour_xy):
    """
    Ear‐clipping triangulation for a single CCW 2D contour (N×2).
    Returns a list of index‐triples (i, j, k).
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


def write_stl_ascii(vertices, faces, solid_name="airfoil_extrusion"):
    """
    Write an ASCII STL string from (vertices, faces). Return as one string.
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


def triangulate_with_hole(outer_xy, inner_xy):
    """
    Build a simple “single‐ring” polygon by connecting the outer contour to the hole contour 
    with a single bridge, then ear‐clip that. Returns a list of index triples into the combined array,
    plus the combined XY array itself.
    - outer_xy: (n×2), CCW
    - inner_xy: (m×2), CCW (we will reverse it to CW)
    Combined polygon is:
      outer[oi], outer[oi+1], …, outer wrap…, outer[oi-1],  outer[oi],
      followed by inner[hi], inner[hi-1], …, inner wrap …, inner[hi-1].
    where 
      hi = argmin(inner_xy[:,0])  (hole’s leftmost vertex),
      oi = argmin(outer_xy[:,0])  (outer’s leftmost vertex).
    """
    n = len(outer_xy)
    m = len(inner_xy)

    # Pick the hole‐vertex hi with minimal x:
    hi = int(np.argmin(inner_xy[:, 0]))
    # Pick outer vertex oi with minimal x:
    oi = int(np.argmin(outer_xy[:, 0]))

    # Build an “outer ring” starting at oi, wrapping around
    outer_seq = []
    for i in range(n):
        idx = (oi + i) % n
        outer_seq.append(tuple(outer_xy[idx]))

    # Build a “hole ring” in **reverse** (CW) starting at hi:
    inner_seq = []
    for i in range(m):
        idx = (hi - i) % m
        inner_seq.append(tuple(inner_xy[idx]))

    # Combined = outer_seq + inner_seq
    combined = outer_seq + inner_seq
    combined_arr = np.array(combined, dtype=float)

    # Now ear‐clip that combined_arr
    tri_indices = triangulate_polygon(combined_arr)

    # tri_indices refers to indices in [0 .. (n + m - 1)]
    # Return both the triangle list and the combined array
    return tri_indices, combined_arr


def build_hex_2d(f2f, x0, y0):
    """
    Build a regular hexagon (flat‐to‐flat = f2f) in 2D, centered at (x0, y0), 
    returning an (6 × 2) array in CCW order.
    Formula: radius R = (f2f/2)/cos(π/6), vertices at 30° + k*60°.
    """
    R = (f2f / 2.0) / np.cos(np.pi / 6)
    pts = []
    for k in range(6):
        theta = np.pi / 6 + k * (np.pi / 3)  # 30°, 90°, 150°, …
        x = x0 + R * np.cos(theta)
        y = y0 + R * np.sin(theta)
        pts.append([x, y])
    return np.array(pts, dtype=float)  # shape (6, 2)


# -------------------------
# 2) TAB 1: View Airfoil
# -------------------------
with tab1:
    st.header("View Airfoil Profile")
    dat_file_v = st.file_uploader(
        "Upload `.dat` to view (first line is description)",
        type=["dat"], key="view"
    )
    if dat_file_v:
        coords_v = load_dat_coords(dat_file_v.read())
        if coords_v.shape[0] < 3:
            st.error("Need at least 3 points to display a valid airfoil.")
        else:
            st.subheader("2D Profile Plot")
            fig_v = plot_2d_profile(coords_v, title="Uploaded Airfoil")
            st.pyplot(fig_v)
            if st.checkbox("Show coordinate table"):
                df = pd.DataFrame(coords_v, columns=["x", "y"])
                st.dataframe(df, height=200)
    else:
        st.info("Upload a `.dat` file to view the 2D profile.")


# -------------------------
# 3) TAB 2: Extrude to STL (with “Hole Built In”)
# -------------------------
with tab2:
    st.header("Extrude Airfoil to STL + Drill a Finite‐Depth Hex Hole")

    # 1) Upload a DAT
    dat_file_e = st.file_uploader(
        "Upload `.dat` for extrusion (first line is description)",
        type=["dat"],
        key="extrude"
    )

    if dat_file_e:
        raw_bytes = dat_file_e.read()
        coords_e = load_dat_coords(raw_bytes)

        if coords_e.shape[0] < 3:
            st.error("Failed to parse enough points. Please upload a valid airfoil DAT.")
        else:
            # 2) Main parameters
            st.subheader("Parameters")
            col1, col2 = st.columns(2)
            with col1:
                scale_factor = st.number_input(
                    "Scaling factor (mm per unit chord)",
                    value=100.0,
                    format="%.3f"
                )
            with col2:
                thickness = st.number_input(
                    "Extrusion thickness (mm)",
                    value=1.0,
                    format="%.3f"
                )

            # 3) Show bounding‐box in X, Y, Z
            st.subheader("Airfoil Bounding-Box (X, Y, Z)")
            scaled_coords_xy = coords_e * scale_factor
            xs_scaled = scaled_coords_xy[:, 0]
            ys_scaled = scaled_coords_xy[:, 1]
            x_min, x_max = float(xs_scaled.min()), float(xs_scaled.max())
            y_min, y_max = float(ys_scaled.min()), float(ys_scaled.max())
            z_min, z_max = 0.0, thickness
            st.write(f"• X span: {x_max - x_min:.3f} mm   (from {x_min:.3f} to {x_max:.3f})")
            st.write(f"• Y span: {y_max - y_min:.3f} mm   (from {y_min:.3f} to {y_max:.3f})")
            st.write(f"• Z span: {z_max - z_min:.3f} mm")

            # 4) Optional: 2D resampled preview
            st.subheader("Optional: Resampled 2D Preview")
            num_pts = st.slider(
                "Resample points for smoothness",
                min_value=50, max_value=500,
                value=300, step=10
            )
            # Resample and scale
            contour_xy = resample_profile_xy(scaled_coords_xy, num_pts)  # shape (n,2)

            fig2d = plot_2d_profile(contour_xy, title="Resampled & Scaled (2D)")
            st.pyplot(fig2d)

            # 5) Let user choose Hex center (x_choice, y_choice)
            st.subheader("Choose `x` and `y` for the Hex Center (in mm)")
            x_choice = st.slider(
                "Hole center X [mm]",
                min_value=x_min, max_value=x_max,
                value=(x_min + x_max) / 2.0,
                step=(x_max - x_min) / 1000.0
            )
            y_choice = st.slider(
                "Hole center Y [mm]",
                min_value=y_min, max_value=y_max,
                value=(y_min + y_max) / 2.0,
                step=(y_max - y_min) / 1000.0
            )

            # Show the 2D contour plus a red dot at (x_choice, y_choice)
            fig_marker, ax_marker = plt.subplots(figsize=(5, 2.5))
            ax_marker.plot(contour_xy[:, 0], contour_xy[:, 1], "-k", linewidth=2)
            ax_marker.scatter([x_choice], [y_choice], c="red", s=60, label="Hex Center")
            ax_marker.set_aspect("equal", "box")
            ax_marker.set_xlabel("x [mm]")
            ax_marker.set_ylabel("y [mm]")
            ax_marker.set_title("2D Airfoil (Chosen Hex Center)")
            ax_marker.grid(True, linestyle="--", alpha=0.4)
            ax_marker.legend()
            st.pyplot(fig_marker)

            # 6) Tapered‐Hex parameters & hole depth
            st.subheader("Tapered‐Hex Hole Parameters")
            col_ht, col_hb, col_depth = st.columns(3)
            with col_ht:
                top_f2f = st.number_input(
                    "Hex TOP flat‐to‐flat [mm] (at z = hole‐top)",
                    min_value=0.5, value=5.0, step=0.1
                )
            with col_hb:
                bot_f2f = st.number_input(
                    "Hex BOTTOM flat‐to‐flat [mm] (at z = 0)",
                    min_value=0.1, value=4.8, step=0.1
                )
            with col_depth:
                depth = st.number_input(
                    "Hole DEPTH (mm)",
                    min_value=0.0, max_value=thickness,
                    value=min( (y_max - y_min) * 0.3, thickness ),  # default 30% of chord
                    step=0.1
                )

            # 7) Generate & preview final mesh
            st.subheader("Generate & Preview STL")
            stl_name = st.text_input("Filename (no extension)", value="airfoil_extrusion")
            if not stl_name.strip():
                st.error("Please enter a valid filename.")
            else:
                if st.button("Create STL with Finite‐Depth Hex Hole"):
                    # —————————————————————————————
                    # A) Build the 2D polygons for triangulation
                    # —————————————————————————————
                    n = contour_xy.shape[0]
                    m = 6  # hex has 6 vertices

                    # 1) Outer ring (airfoil), CCW, at z = 0 and z = depth and z = thickness
                    outer_xy = contour_xy.copy()  # shape (n,2)

                    # 2) Build two hex contours (bot and top of hole):
                    #    - Hex at z=0 uses bot_f2f
                    hex_bot_xy = build_hex_2d(bot_f2f, x_choice, y_choice)  # (6,2)
                    #    - Hex at z=depth uses top_f2f
                    hex_top_xy = build_hex_2d(top_f2f, x_choice, y_choice)  # (6,2)

                    # ——————————
                    # B) Triangulate the 2D “polygon with hole” at z = 0
                    # ——————————
                    # We need bottom cap of the hole: outer ring minus hex_bot ring.
                    tri_bot_indices, combined_bot = triangulate_with_hole(outer_xy, hex_bot_xy)
                    # combined_bot has length n + m; tri_bot_indices refers to combined indices in [0..n+m-1]

                    # ——————————
                    # C) Triangulate the 2D “polygon with hole” at z = depth
                    # ——————————
                    tri_depth_indices, combined_depth = triangulate_with_hole(outer_xy, hex_top_xy)
                    # combined_depth length also n + m

                    # ——————————
                    # D) Triangulate the 2D outer airfoil at z = thickness (no hole)
                    # ——————————
                    tri_top_only = triangulate_polygon(outer_xy)  # each triple in [0..n-1]

                    # —————————————————————————————
                    # E) Build the 3D vertex array
                    # —————————————————————————————
                    # We'll allocate in this order:
                    #   1) outer_z0:   indices [0 .. n-1],  z=0
                    #   2) hex_z0:     indices [ n .. n+m-1],  z=0
                    #   3) outer_zD:   indices [ n+m .. n+m+n-1],  z=depth
                    #   4) hex_zD:     indices [ n+m+n .. n+m+n+m-1],  z=depth
                    #   5) outer_zT:   indices [ n+m+n+m .. n+m+n+m+n-1],  z=thickness
                    #
                    #           total vertices = 3n + 2m
                    #
                    total_vertices = 3 * n + 2 * m
                    verts_3d = np.zeros((total_vertices, 3), dtype=float)

                    # 1) Outer at z=0
                    for i in range(n):
                        x_i, y_i = outer_xy[i]
                        verts_3d[i] = [x_i, y_i, 0.0]

                    # 2) Hex bottom at z=0
                    for j in range(m):
                        xh, yh = hex_bot_xy[j]
                        verts_3d[n + j] = [xh, yh, 0.0]

                    # 3) Outer at z=depth
                    offset_outer_zD = n + m
                    for i in range(n):
                        x_i, y_i = outer_xy[i]
                        verts_3d[offset_outer_zD + i] = [x_i, y_i, depth]

                    # 4) Hex top at z=depth
                    offset_hex_zD = n + m + n
                    for j in range(m):
                        xh, yh = hex_top_xy[j]
                        verts_3d[offset_hex_zD + j] = [xh, yh, depth]

                    # 5) Outer at z=thickness (no hole)
                    offset_outer_zT = n + m + n + m
                    for i in range(n):
                        x_i, y_i = outer_xy[i]
                        verts_3d[offset_outer_zT + i] = [x_i, y_i, thickness]

                    # —————————————————————————————
                    # F) Build the triangular faces
                    # —————————————————————————————
                    faces_3d = []

                    # F.1) Bottom‐cap (z=0) with hole: use tri_bot_indices on combined_bot.
                    #     For each combined index k, if k < n, it maps to “outer_z0 _ index k” = k
                    #                              else, it maps to “hex_z0 index (k - n)” = (n + (k-n)) = k
                    for (a, b, c) in tri_bot_indices:
                        ia = int(a)
                        ib = int(b)
                        ic = int(c)
                        # map each to 3D index
                        va = ia if ia < n else n + (ia - n)
                        vb = ib if ib < n else n + (ib - n)
                        vc = ic if ic < n else n + (ic - n)
                        faces_3d.append([va, vb, vc])

                    # F.2) Cap “at depth” (the top of the hole): use tri_depth_indices on combined_depth.
                    #     Combined indices in [0..n-1] map to “outer_zD” = offset_outer_zD + i
                    #     Combined indices in [n..n+m-1] map to “hex_zD” = offset_hex_zD + (i-n)
                    for (a, b, c) in tri_depth_indices:
                        ia = int(a)
                        ib = int(b)
                        ic = int(c)
                        va = (offset_outer_zD + ia) if ia < n else (offset_hex_zD + (ia - n))
                        vb = (offset_outer_zD + ib) if ib < n else (offset_hex_zD + (ib - n))
                        vc = (offset_outer_zD + ic) if ic < n else (offset_hex_zD + (ic - n))
                        faces_3d.append([va, vb, vc])

                    # F.3) Top cap (z=thickness) without hole: tri_top_only on outer_xy.
                    #     Each index i maps to offset_outer_zT + i
                    for (i, j, k) in tri_top_only:
                        faces_3d.append([
                            offset_outer_zT + int(i),
                            offset_outer_zT + int(j),
                            offset_outer_zT + int(k)
                        ])

                    # F.4) Side‐walls of outer between z=0 → z=depth
                    for i in range(n):
                        i_next = (i + 1) % n
                        # bottom‐outer: i, i_next
                        # depth‐outer: offset_outer_zD + i, offset_outer_zD + i_next
                        b0 = i
                        b1 = i_next
                        d0 = offset_outer_zD + i
                        d1 = offset_outer_zD + i_next
                        # two triangles:
                        faces_3d.append([b0, b1, d1])
                        faces_3d.append([b0, d1, d0])

                    # F.5) Side‐walls of outer between z=depth → z=thickness
                    for i in range(n):
                        i_next = (i + 1) % n
                        d0 = offset_outer_zD + i
                        d1 = offset_outer_zD + i_next
                        t0 = offset_outer_zT + i
                        t1 = offset_outer_zT + i_next
                        faces_3d.append([d0, d1, t1])
                        faces_3d.append([d0, t1, t0])

                    # F.6) Side‐walls of the hole between z=0 → z=depth
                    for j in range(m):
                        j_next = (j + 1) % m
                        hb0 = n + j         # bottom‐hex index
                        hb1 = n + j_next
                        hd0 = offset_hex_zD + j
                        hd1 = offset_hex_zD + j_next
                        faces_3d.append([hb0, hb1, hd1])
                        faces_3d.append([hb0, hd1, hd0])

                    # Convert face list to NumPy
                    faces_np = np.array(faces_3d, dtype=int)

                    # —————————————————————————————
                    # G) Write the STL and preview
                    # —————————————————————————————
                    stl_text = write_stl_ascii(verts_3d, faces_np, solid_name=stl_name)
                    stl_bytes = stl_text.encode("utf-8")

                    # Three.js preview
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
                        const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
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

                    # Download button
                    st.download_button(
                        label="Download `.stl`",
                        data=stl_bytes,
                        file_name=f"{stl_name}.stl",
                        mime="application/vnd.ms-pki.stl"
                    )
                    st.success(f"STL `{stl_name}.stl` is ready!")
    else:
        st.info("Upload a `.dat` file to enable extrusion.")

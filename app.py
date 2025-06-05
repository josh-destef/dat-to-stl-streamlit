import streamlit as st
import numpy as np
from scipy.interpolate import splprep, splev
import io
import matplotlib.pyplot as plt
import pandas as pd
import base64

# ----- New import for mesh‐boolean -----
import trimesh  

st.set_page_config(page_title="Airfoil Toolkit", layout="centered")
st.title("Airfoil Toolkit")

tab1, tab2 = st.tabs(["View Airfoil", "Extrude to STL"])


# -------------------------
# 1) Shared helper functions
# -------------------------

def load_dat_coords(dat_bytes):
    """
    Parse .dat bytes into an (N×2) numpy array of (x, y).
    """
    raw_lines = dat_bytes.decode("utf-8", errors="ignore").splitlines()
    coord_lines = raw_lines[1:]  # skip header
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
                continue
    return np.array(pts, dtype=float)


def plot_2d_profile(coords, title="Airfoil Profile"):
    """
    Given an (N×2) array, plot x vs. y with equal axis.
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


# -------------------------
# 2) Hex‐prism construction & membership test
# -------------------------

def build_tapered_hex_prism(cx, cy, top_f2f, bot_f2f, depth, top_z):
    """
    Build a standalone 3D mesh for a *tapered hex prism*:
      - The top hex face (at z=top_z) has flat‐to‐flat = top_f2f.
      - The bottom hex face (at z=top_z - depth) has flat‐to‐flat = bot_f2f.
      - Centered horizontally at (cx, cy).

    Returns (verts_hex, faces_hex) where:
    - verts_hex is (12×3) array (6 top corners + 6 bottom corners).
    - faces_hex is (12×3) integer array of triangle indices:
       • 6 quads for the sides (split each into 2 triangles → 12 triangles)
       • 6 triangles for top cap (fan)
       • 6 triangles for bottom cap (fan, flipped)
    """
    # 1) Compute the circumradius R from flat‐to‐flat f2f:  R = (f2f/2)/cos(π/6)
    def hex_corners(f2f, x0, y0, z):
        R = (f2f / 2.0) / np.cos(np.pi / 6)
        pts = []
        for k in range(6):
            theta = np.pi / 6 + k * (np.pi / 3)  # 30°, 90°, 150°, ...
            x = x0 + R * np.cos(theta)
            y = y0 + R * np.sin(theta)
            pts.append([x, y, z])
        return np.array(pts, dtype=float)

    top_pts = hex_corners(top_f2f, cx, cy, top_z)
    bot_pts = hex_corners(bot_f2f, cx, cy, top_z - depth)
    verts_hex = np.vstack([top_pts, bot_pts])  # (12×3)

    faces = []
    # 2) Side‐walls: each i in [0..5] → quad between top[i], top[i+1], bot[i+1], bot[i]
    for i in range(6):
        i_next = (i + 1) % 6
        top_i = i
        top_inext = i_next
        bot_i = 6 + i
        bot_inext = 6 + i_next
        # split quad into two triangles
        faces.append([top_i, top_inext, bot_inext])
        faces.append([top_i, bot_inext, bot_i])

    # 3) Triangulate the top hex face by “fan” around vertex 0 → 4 triangles (0,1,2), (0,2,3), (0,3,4), (0,4,5)
    for i in range(1, 5):
        faces.append([0, i, i + 1])
    # 4) Triangulate the bottom hex face (indices 6..11) similarly—but with flipped winding
    base = 6
    for i in range(1, 5):
        faces.append([base, base + i + 1, base + i])
    # last triangle to close the fan: (6, 11, 7)
    faces.append([base, base + 7, base + 11])

    return verts_hex, np.array(faces, dtype=int)


def find_max_thickness_x(contour, num_samples=500):
    """
    Given a 2D contour (N×2), sample num_samples x‐values from min(x) to max(x).
    At each xg, find where the polygon intersects the vertical line x=xg.
    Compute thickness = max(intersect_ys) - min(intersect_ys). Return the xg that maximizes thickness.
    """
    xs = contour[:, 0]
    minx, maxx = xs.min(), xs.max()
    x_grid = np.linspace(minx, maxx, num_samples)
    best_x = minx
    best_thk = 0.0

    for xg in x_grid:
        y_ints = []
        for i in range(len(contour)):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + 1) % len(contour)]
            if (x1 - xg) * (x2 - xg) <= 0 and abs(x2 - x1) > 1e-8:
                t = (xg - x1) / (x2 - x1)
                yi = y1 + t * (y2 - y1)
                y_ints.append(yi)
        if len(y_ints) >= 2:
            y_top = max(y_ints)
            y_bot = min(y_ints)
            thickness_here = y_top - y_bot
            if thickness_here > best_thk:
                best_thk = thickness_here
                best_x = xg

    return best_x


# -------------------------
# 3) TAB 1: View Airfoil
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
# 4) TAB 2: Extrude to STL + Boolean‐Hex
# -------------------------
with tab2:
    st.header("Extrude Airfoil to STL with Tapered Hex Hole (Boolean)")

    # 1) File upload
    dat_file_e = st.file_uploader(
        "Upload `.dat` for extrusion (first line is description)",
        type=["dat"], key="extrude"
    )
    if dat_file_e:
        raw_bytes = dat_file_e.read()
        coords_e = load_dat_coords(raw_bytes)

        if coords_e.shape[0] < 3:
            st.error("Failed to parse enough points. Please upload a valid airfoil `.dat`.")
        else:
            # 2) Parameters
            st.subheader("Parameters")
            col1, col2 = st.columns(2)
            with col1:
                scale_factor = st.number_input(
                    "Scaling factor (mm per unit chord)",
                    value=100.0, format="%.3f"
                )
            with col2:
                thickness = st.number_input(
                    "Extrusion thickness (mm)",
                    value=1.0, format="%.3f"
                )

            # 3) Bounding‐Box Preview
            st.subheader("Airfoil Bounding-Box (X, Y, Z)")
            xs_scaled = coords_e[:, 0] * scale_factor
            ys_scaled = coords_e[:, 1] * scale_factor
            x_min, x_max = xs_scaled.min(), xs_scaled.max()
            y_min, y_max = ys_scaled.min(), ys_scaled.max()
            z_min, z_max = 0.0, thickness
            st.write(f"• X span: {x_max - x_min:.3f} mm   (from {x_min:.3f} to {x_max:.3f})")
            st.write(f"• Y span: {y_max - y_min:.3f} mm   (from {y_min:.3f} to {y_max:.3f})")
            st.write(f"• Z span: {z_max - z_min:.3f} mm")

            # 4) Optional: Resampled Preview
            st.subheader("Optional: Resampled Preview")
            num_pts = st.slider(
                "Resample points for smoothness",
                min_value=50, max_value=500, value=300, step=10
            )
            scaled_coords = coords_e * scale_factor
            contour_preview = resample_profile_xy(scaled_coords, num_pts)
            fig_e = plot_2d_profile(contour_preview, title="Resampled & Scaled (Preview)")
            st.pyplot(fig_e)

            # 5) Hex‐Hole parameters
            st.subheader("Tapered Hex‐Hole Parameters")
            col1h, col2h, col3h = st.columns(3)
            with col1h:
                top_f2f = st.number_input(
                    "Hexagon TOP flat-to-flat (mm)",
                    min_value=0.5, value=5.0, step=0.1
                )
            with col2h:
                bot_f2f = st.number_input(
                    "Hexagon BOTTOM flat-to-flat (mm)",
                    min_value=0.1, value=4.8, step=0.1
                )
            with col3h:
                depth = st.number_input(
                    "Hexagon hole DEPTH (mm)",
                    min_value=0.1, max_value=thickness, value=thickness, step=0.1
                )

            # 6) Generate & Preview STL
            st.subheader("Generate & Preview STL")
            stl_name = st.text_input("Filename (no extension)", value="airfoil_extrusion")
            if not stl_name.strip():
                st.error("Please enter a valid filename.")
            else:
                if st.button("Create STL with Hex Hole"):
                    # a) Build the extruded foil mesh
                    verts_foil, faces_foil = extrude_to_vertices_faces(
                        scaled_coords, thickness, num_interp=num_pts
                    )

                    # b) Find the unscaled x at max thickness
                    unscaled_contour = resample_profile_xy(coords_e, num_pts)
                    x_unscaled_max = find_max_thickness_x(unscaled_contour, num_samples=500)
                    cx = x_unscaled_max * scale_factor
                    cy = 0.0

                    # c) Build the tapered hex‐prism mesh
                    verts_hex, faces_hex = build_tapered_hex_prism(
                        cx, cy, top_f2f, bot_f2f, depth, top_z=thickness
                    )

                    # d) Convert both to trimesh.Trimesh and do boolean difference
                    foil_mesh = trimesh.Trimesh(vertices=verts_foil, faces=faces_foil, process=False)
                    hex_mesh  = trimesh.Trimesh(vertices=verts_hex,  faces=faces_hex,  process=False)

                    try:
                        # If you have OpenSCAD installed, you can specify engine="scad"
                        result_mesh = foil_mesh.difference(hex_mesh)
                    except BaseException as e:
                        st.error(f"Boolean‐difference failed: {e}")
                        return

                    verts_final = result_mesh.vertices
                    faces_final = result_mesh.faces

                    # Inform user where the hole was placed
                    st.info(
                        f"Unscaled thickest station: x = {x_unscaled_max:.6f}  →  "
                        f"Scaled center at {cx:.3f} mm"
                    )

                    if verts_final.size == 0 or faces_final.size == 0:
                        st.error("Resulting mesh is empty—perhaps the hex was too large or depth too great.")
                    else:
                        # 7) Generate ASCII STL text
                        stl_text = write_stl_ascii(verts_final, faces_final, solid_name=stl_name)
                        stl_bytes = stl_text.encode("utf-8")

                        # 8) Show Three.js preview via embedded HTML
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

                        # 9) Download button
                        st.download_button(
                            label="Download `.stl`",
                            data=stl_bytes,
                            file_name=f"{stl_name}.stl",
                            mime="application/vnd.ms-pki.stl"
                        )
                        st.success(f"STL `{stl_name}.stl` is ready!")
    else:
        st.info("Upload a `.dat` file to enable extrusion.")

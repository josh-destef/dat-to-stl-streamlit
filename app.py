import streamlit as st
import numpy as np
from scipy.interpolate import splprep, splev
import io
import matplotlib.pyplot as plt
import pandas as pd
import base64
import struct

st.set_page_config(page_title="Airfoil Toolkit", layout="centered")
st.title("Airfoil Toolkit")

# Three tabs: 
#  - tab1 = “View Airfoil”
#  - tab2 = “Extrude to STL”
#  - tab3 = “Modify Existing STL”

tab1, tab2, tab3 = st.tabs(["View Airfoil", "Extrude to STL", "Modify Existing STL"])


# -------------------------
# 1) Shared helper functions
# -------------------------

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
    """
    Return True if point pt lies inside triangle (a,b,c) in 2D.
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
    Ear‐clipping triangulation for a CCW 2D contour. Returns a list of index‐triples (i, j, k).
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
    1) Resample xy_points to num_interp (smooth contour).  
    2) Build a 3D “plate” with bottom at z=0 and top at z=thickness_mm.  
    3) Triangulate the 2D contour for bottom & top caps.  
    4) Add side‐wall faces.  
    Returns (vertices (2n×3), faces (F×3)).
    """
    contour = resample_profile_xy(xy_points, num_interp)
    n = contour.shape[0]
    vertices = np.zeros((2 * n, 3), dtype=float)
    for i in range(n):
        x_val, y_val = contour[i]
        vertices[i] = [x_val, y_val, 0.0]
        vertices[n + i] = [x_val, y_val, thickness_mm]

    faces = []
    bottom_tris = triangulate_polygon(contour)
    for (i, j, k) in bottom_tris:
        faces.append([i, k, j])            # bottom cap
    for (i, j, k) in bottom_tris:
        faces.append([n + i, n + j, n + k])  # top cap

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


# -------------------------
# 2) Find the chordwise station of maximum thickness
# -------------------------

def find_max_thickness_x(contour, num_samples=500):
    """
    Given a 2D contour (N×2), sample num_samples x-values between min(x) and max(x).
    For each xg, find where the polygon intersects the vertical line x=xg.
    Compute vertical thickness = max(Y)-min(Y). Return the xg that maximizes that thickness.
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
            thk = y_top - y_bot
            if thk > best_thk:
                best_thk = thk
                best_x = xg

    return best_x


# -------------------------
# 3) Build a “unit” hex prism in Y–Z at x=0
# -------------------------

def build_unit_hex_prism(f2f_top, f2f_bot, depth, thickness):
    """
    Build a 12‐vertex unit hex prism whose axis is along +Z, all at x=0:
      - The top cap (z = thickness) has flat‐to‐flat = f2f_top.
      - The bottom cap (z = thickness - depth) has flat‐to‐flat = f2f_bot.
    Each vertex is [0, y, z].  
    Returns (verts_hex, faces_hex):
      • verts_hex is shape (12×3).  
      • faces_hex is shape (12,3) of triangle indices:
        – 6 quads for side walls → 12 triangles  
        – 4 triangles for top cap  
        – 4 triangles for bottom cap  
    """
    def hex_corners(f2f, center_y, center_z):
        R = (f2f / 2.0) / np.cos(np.pi / 6)  # circumradius
        pts = []
        for k in range(6):
            theta = np.pi / 6 + k * (np.pi / 3)  # 30°, 90°, 150°, ...
            y = center_y + R * np.cos(theta)
            z = center_z + R * np.sin(theta)
            pts.append([0.0, y, z])
        return np.array(pts, dtype=float)

    top_center_z = thickness
    bottom_center_z = thickness - depth

    top_pts = hex_corners(f2f_top, 0.0, top_center_z)      # indices [0..5]
    bot_pts = hex_corners(f2f_bot, 0.0, bottom_center_z)    # indices [6..11]

    verts_hex = np.vstack([top_pts, bot_pts])  # (12×3)

    faces = []
    # 1) Side‐walls: each i=0..5 → quad between top[i], top[i+1], bot[i+1], bot[i]
    for i in range(6):
        i_next = (i + 1) % 6
        top_i = i
        top_inext = i_next
        bot_i = 6 + i
        bot_inext = 6 + i_next
        faces.append([top_i, top_inext, bot_inext])
        faces.append([top_i, bot_inext, bot_i])

    # 2) Top hex cap: fan around vertex 0 (indices [0..5]): (0,1,2),(0,2,3),(0,3,4),(0,4,5)
    for i in range(1, 5):
        faces.append([0, i, i + 1])

    # 3) Bottom hex cap: fan around vertex 6 (indices [6..11]): (6,7,8),(6,8,9),(6,9,10),(6,10,11)
    for corner in range(7, 11):
        faces.append([6, corner, corner + 1])

    return verts_hex, np.array(faces, dtype=int)


# -------------------------
# 4) “Slice‐and‐Insert Hex” on an existing mesh
# -------------------------

def slice_and_insert_hex(verts, faces, x_center, f2f_top, f2f_bot, depth, z_center):
    """
    1) Identify all vertices whose |x - x_center| < tol AND |z - z_center| < (depth/2). 
       Collect their indices in slice_verts.  
    2) Remove any face that references any of slice_verts → keep_faces.  
    3) Build a “unit” tapered hex prism centered in Y–Z:
         - Top cap at z = z_center + (depth/2), flat‐to‐flat = f2f_top
         - Bottom cap at z = z_center - (depth/2), flat‐to‐flat = f2f_bot
       (This prism has 12 vertices at x=0.)
    4) Translate that prism’s vertices to x = x_center (so that the hex is drilled through).
    5) Append the 12 hex‐verts to the original vertices array.
    6) Reindex the 8 hex faces by adding n_old = len(verts) (since they come after).
    7) Return (verts_new, faces_new) where:
         verts_new  = np.vstack([verts, verts_hex_translated])  
         faces_new  = np.vstack([keep_faces, faces_hex_reindexed])
    """
    thickness = np.max(verts[:, 2])
    tol = 1e-6

    # a) Collect any vertex within the tolerance window around x_center and z_center
    x_coords = verts[:, 0]
    z_coords = verts[:, 2]
    mask_slice = (np.abs(x_coords - x_center) < tol) & \
                 (np.abs(z_coords - z_center) < (depth / 2.0 + tol))
    slice_verts = np.nonzero(mask_slice)[0]

    # b) Remove faces that reference ANY slice_vert
    keep_faces = []
    for tri in faces:
        if any(v in slice_verts for v in tri):
            continue
        keep_faces.append(tri)
    keep_faces = np.array(keep_faces, dtype=int)

    # c) Build “unit” hex prism at x=0 in Y–Z, but centered around z_center:
    #    We want top cap at z_top = z_center + (depth/2), bottom cap at z_bot = z_center - (depth/2).
    #    So call build_unit_hex_prism with:
    f2f_top_actual = f2f_top
    f2f_bot_actual = f2f_bot
    # But build_unit_hex_prism assumes top cap at z = thickness, bottom at z = thickness - depth.
    # So we temporarily “trick” it by passing (depth, thickness) = (depth, (z_center + depth/2)) 
    # so that: 
    #   top_z = (z_center + depth/2),
    #   bottom_z = (z_center + depth/2) - depth = z_center - (depth/2).
    #
    z_temp_top = z_center + depth / 2.0
    z_temp_thickness = z_temp_top
    verts_hex_unit, faces_hex_unit = build_unit_hex_prism(
        f2f_top_actual, f2f_bot_actual, depth, z_temp_thickness
    )

    # d) Translate hex‐unit verts from x=0 to x=x_center
    verts_hex = verts_hex_unit.copy()
    verts_hex[:, 0] += x_center

    # e) Append hex verts to the original verts
    n_old = verts.shape[0]
    verts_new = np.vstack([verts, verts_hex])  # size = n_old + 12

    # f) Reindex hex faces (+ n_old) so they refer to the new vertices
    faces_hex = faces_hex_unit + n_old

    # g) Combine the kept original faces + new hex faces
    faces_new = np.vstack([keep_faces, faces_hex])

    return verts_new, faces_new


# -------------------------
# 5) Lightweight STL loader (ASCII or Binary)
# -------------------------

def load_stl_mesh(stl_bytes):
    """
    Minimal STL loader that handles both ASCII and binary.
    Returns (vertices (V×3), faces (F×3)) as NumPy arrays.
    """
    header = stl_bytes[:80]
    try:
        # If the header does NOT start with “solid”, treat as binary
        if not header.lstrip().lower().startswith(b"solid"):
            # Attempt binary parsing
            #  - Byte 80..84 is the unsigned int # of triangles
            num_tris = struct.unpack("<I", stl_bytes[80:84])[0]
            # Each triangle: 50 bytes (12 floats + 2‐byte attribute)
            expected_len = 84 + num_tris * 50
            if len(stl_bytes) < expected_len:
                raise ValueError
            data = stl_bytes[84:84 + num_tris * 50]
            vertices = []
            faces = []
            offset = 0
            for i in range(num_tris):
                # skip normal (3 floats)
                offset += 12
                v0 = struct.unpack("<3f", data[offset:offset + 12]); offset += 12
                v1 = struct.unpack("<3f", data[offset:offset + 12]); offset += 12
                v2 = struct.unpack("<3f", data[offset:offset + 12]); offset += 12
                vertices.append(v0)
                vertices.append(v1)
                vertices.append(v2)
                faces.append([3 * i, 3 * i + 1, 3 * i + 2])
                # skip attribute short
                offset += 2
            verts_np = np.array(vertices, dtype=float)
            faces_np = np.array(faces, dtype=int)
            return verts_np, faces_np
    except Exception:
        # Fall through to ASCII parse
        pass

    # ASCII parsing: read lines, collect “vertex x y z”
    text = stl_bytes.decode("utf-8", errors="ignore").splitlines()
    verts_list = []
    faces_list = []
    tri_verts = []
    for line in text:
        line = line.strip().lower()
        if line.startswith("vertex"):
            parts = line.split()
            if len(parts) == 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                tri_verts.append((x, y, z))
                if len(tri_verts) == 3:
                    # Add these three vertices & a new face
                    start_idx = len(verts_list)
                    verts_list.extend(tri_verts)
                    faces_list.append([start_idx, start_idx + 1, start_idx + 2])
                    tri_verts = []
    verts_np = np.array(verts_list, dtype=float)
    faces_np = np.array(faces_list, dtype=int)
    return verts_np, faces_np


# -------------------------
# 6) TAB 1: View Airfoil
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
# 7) TAB 2: Extrude to STL
# -------------------------
with tab2:
    st.header("Extrude Airfoil to STL")
    dat_file_e = st.file_uploader(
        "Upload `.dat` for extrusion (first line is description)", 
        type=["dat"], key="extrude"
    )
    if dat_file_e:
        raw_bytes = dat_file_e.read()
        coords_e = load_dat_coords(raw_bytes)

        if coords_e.shape[0] < 3:
            st.error("Failed to parse enough points. Please upload a valid airfoil DAT.")
        else:
            # Parameters
            st.subheader("Parameters")
            col1, col2 = st.columns(2)
            with col1:
                scale_factor = st.number_input(
                    "Scaling factor (e.g., chord in mm per unit)",
                    value=100.0,
                    format="%.3f"
                )
            with col2:
                thickness = st.number_input(
                    "Extrusion thickness (mm)",
                    value=1.0,
                    format="%.3f"
                )

            # Bounding‐Box Preview
            st.subheader("Airfoil Bounding-Box (X, Y, Z)")
            xs_scaled = coords_e[:, 0] * scale_factor
            ys_scaled = coords_e[:, 1] * scale_factor
            x_min, x_max = xs_scaled.min(), xs_scaled.max()
            y_min, y_max = ys_scaled.min(), ys_scaled.max()
            z_min, z_max = 0.0, thickness
            st.write(f"• X span: {x_max - x_min:.3f} mm   (from {x_min:.3f} to {x_max:.3f})")
            st.write(f"• Y span: {y_max - y_min:.3f} mm   (from {y_min:.3f} to {y_max:.3f})")
            st.write(f"• Z span: {z_max - z_min:.3f} mm")

            # Optional: Resampled Preview
            st.subheader("Optional: Resampled Preview")
            num_pts = st.slider(
                "Resample points for smoothness",
                min_value=50, max_value=500, value=300, step=10
            )
            scaled_coords = coords_e * scale_factor
            contour_preview = resample_profile_xy(scaled_coords, num_pts)
            fig_e = plot_2d_profile(contour_preview, title="Resampled & Scaled (Preview)")
            st.pyplot(fig_e)

            # Generate & Preview STL
            st.subheader("Generate & Preview STL")
            stl_name = st.text_input("Filename (no extension)", value="airfoil_extrusion")
            if not stl_name.strip():
                st.error("Please enter a valid filename.")
            else:
                if st.button("Create STL"):
                    verts, faces = extrude_to_vertices_faces(scaled_coords, thickness, num_interp=num_pts)
                    stl_text = write_stl_ascii(verts, faces, solid_name=stl_name)
                    stl_bytes = stl_text.encode("utf-8")

                    # Three.js Preview via embedded HTML
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


# -------------------------
# 8) TAB 3: Modify Existing STL
# -------------------------
with tab3:
    st.header("Modify an Existing STL: Drill a Tapered Hex Hole")

    st.markdown(
        """
        1. Upload an existing `.stl` file (ASCII or binary).  
        2. Preview it.  
        3. Specify the exact `(x, y, z)` center in **mm** where you want to cut a hex hole.  
        4. Enter the hex parameters (top‐flat, bottom‐flat, depth).  
        5. Click **“Cut Hex Hole”** → you’ll get a final 3D preview and can download the modified STL.
        """
    )

    st.subheader("Step 1: Upload an STL")
    stl_file = st.file_uploader(
        "Upload a `.stl` file to modify", 
        type=["stl"], key="modify_stl"
    )

    if stl_file:
        raw_stl = stl_file.read()
        try:
            verts_orig, faces_orig = load_stl_mesh(raw_stl)
        except Exception as e:
            st.error(f"Failed to parse STL: {e}")
            st.stop()

        # Step 2: Preview original STL
        st.subheader("Original STL Preview")
        stl_bytes_orig = raw_stl  # raw bytes
        b64_orig = base64.b64encode(stl_bytes_orig).decode()
        html_preview = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <title>Original STL Preview</title>
          <style>
            body {{ margin: 0; }}
            #viewer {{ width: 100%; height: 300px; }}
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
            renderer.setSize(window.innerWidth, 300);
            document.getElementById('viewer').appendChild(renderer.domElement);

            const ambient = new THREE.AmbientLight(0x404040);
            scene.add(ambient);
            const directional = new THREE.DirectionalLight(0xffffff, 1);
            directional.position.set(1, 1, 1).normalize();
            scene.add(directional);

            const loader = new THREE.STLLoader();
            loader.load('data:application/sla;base64,{b64_orig}', function (geometry) {{
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
                mesh.rotation.x += 0.005;
                mesh.rotation.y += 0.005;
                renderer.render(scene, camera);
              }}
              animate();
            }});
          </script>
        </body>
        </html>
        """
        st.components.v1.html(html_preview, height=320)

        # Step 3: Get (x, y, z) center of the hex hole
        st.subheader("Step 3: Specify Hole Center (mm)")
        colx, coly, colz = st.columns(3)
        with colx:
            x_center = st.number_input("X-center [mm]", format="%.3f", value=float(np.mean(verts_orig[:, 0])))
        with coly:
            y_center = st.number_input("Y-center [mm]", format="%.3f", value=float(np.mean(verts_orig[:, 1])))
        with colz:
            z_center = st.number_input("Z-center [mm]", format="%.3f", value=float(np.mean(verts_orig[:, 2])))

        # Step 4: Get hex parameters
        st.subheader("Step 4: Hex Hole Parameters")
        colht, colhb, cold = st.columns(3)
        with colht:
            top_f2f = st.number_input(
                "Hex TOP flat-to-flat [mm]", min_value=0.5, value=5.0, step=0.1
            )
        with colhb:
            bot_f2f = st.number_input(
                "Hex BOTTOM flat-to-flat [mm]", min_value=0.1, value=4.8, step=0.1
            )
        with cold:
            depth = st.number_input(
                "Hex DEPTH [mm]", min_value=0.1, max_value=float(np.ptp(verts_orig[:, 2])), value=float(np.ptp(verts_orig[:, 2])), step=0.1
            )

        # Step 5: Cut Hex Hole when button clicked
        if st.button("Cut Hex Hole"):
            # 1) Remove + insert hex:
            verts_mod, faces_mod = slice_and_insert_hex(
                verts_orig,
                faces_orig,
                x_center,
                top_f2f,
                bot_f2f,
                depth,
                z_center
            )

            # 2) Generate ASCII STL of modified mesh
            stl_text_mod = write_stl_ascii(verts_mod, faces_mod, solid_name="modified_with_hex")
            stl_bytes_mod = stl_text_mod.encode("utf-8")

            # 3) Preview modified STL
            st.subheader("Modified STL Preview")
            b64_mod = base64.b64encode(stl_bytes_mod).decode()
            html_mod = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8" />
              <title>Modified STL Preview</title>
              <style>
                body {{ margin: 0; }}
                #viewer {{ width: 100%; height: 300px; }}
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
                renderer.setSize(window.innerWidth, 300);
                document.getElementById('viewer').appendChild(renderer.domElement);

                const ambient = new THREE.AmbientLight(0x404040);
                scene.add(ambient);
                const directional = new THREE.DirectionalLight(0xffffff, 1);
                directional.position.set(1, 1, 1).normalize();
                scene.add(directional);

                const loader = new THREE.STLLoader();
                loader.load('data:application/sla;base64,{b64_mod}', function (geometry) {{
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
                    mesh.rotation.x += 0.005;
                    mesh.rotation.y += 0.005;
                    renderer.render(scene, camera);
                  }}
                  animate();
                }});
              </script>
            </body>
            </html>
            """
            st.components.v1.html(html_mod, height=320)

            # 4) Download button for modified STL
            st.download_button(
                label="Download Modified `.stl`",
                data=stl_bytes_mod,
                file_name="modified_with_hex.stl",
                mime="application/vnd.ms-pki.stl"
            )
            st.success("Modified STL is ready to download!")
    else:
        st.info("Upload a `.stl` file in order to cut a hex hole.")

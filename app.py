import streamlit as st
import numpy as np
from scipy.interpolate import splprep, splev
import io
import matplotlib.pyplot as plt
import pandas as pd
import re
import base64

st.set_page_config(page_title="Airfoil Toolkit", layout="centered")
st.title("Airfoil Toolkit")

tab1, tab2, tab3 = st.tabs(["View Airfoil", "Extrude to STL", "Metadata"])

# -------------------------
# Shared helper functions
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

# -------------------------
# Tab 2: Extrude to STL
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

        st.subheader("Optional: Resampled Preview")
        num_pts = st.slider("Resample points", min_value=50, max_value=500, value=300, step=10)
        scaled_coords = coords_e * scale_factor
        contour_preview = resample_profile_xy(scaled_coords, num_pts)
        fig_e = plot_2d_profile(contour_preview, title="Resampled & Scaled (Preview)")
        st.pyplot(fig_e)

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

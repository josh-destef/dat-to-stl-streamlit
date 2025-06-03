import streamlit as st
import numpy as np
from scipy.interpolate import splprep, splev
import io
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_3d_viewer import st_viewer

<<<<<<< HEAD
st.set_page_config(page_title="Airfoil Toolkit", layout="centered")
st.title("Airfoil Toolkit")

tab1, tab2 = st.tabs(["View Airfoil", "Extrude to STL"])
=======
st.set_page_config(page_title="DAT → STL Extruder", layout="centered")
st.title("DAT → STL Extruder")
>>>>>>> 1d9681a1debc2a4bfa163e787dec11e4cf2e855b

# -------------------------
# Shared helper functions
# -------------------------

def load_dat_coords(dat_bytes):
    """
<<<<<<< HEAD
    Given raw bytes of a .dat file (UTF-8), skip the first line,
    parse the remaining lines as two-column floats,
    and return an (N×2) numpy array.
    """
    raw_lines = dat_bytes.decode("utf-8").splitlines()
    coord_lines = raw_lines[1:]  # skip first-line description
    pts = []
=======
Upload a `.dat` file (the first line is treated as a description), specify a scaling factor (to convert from nondimensional or other units into millimeters), enter an extrusion thickness, and choose a filename. Click **Generate & Download** to get the STL.
"""
)

# -------------------------
# Helper functions
# -------------------------

def resample_profile_xy(xy, num_pts=200):
    """
    Fit a non‐periodic spline through the given XY points, then
    resample to num_pts points. Ensures no duplicate endpoint.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    # Always use non-periodic spline (per=0) to preserve sharp corners
    tck, _ = splprep([x, y], s=0, per=0)
    u_new = np.linspace(0.0, 1.0, num_pts)
    out = splev(u_new, tck)
    pts_interp = np.vstack(out).T  # shape (num_pts, 2)

    # Remove duplicate last point if it matches the first
    if np.allclose(pts_interp[0], pts_interp[-1], atol=1e-6):
        pts_interp = pts_interp[:-1]

    return pts_interp  # no duplicate endpoint


def point_in_triangle(pt, a, b, c):
    """
    Check if point pt is inside triangle ABC using a barycentric technique.
    pt, a, b, c are arrays or lists of length 2.
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
        return False  # degenerate triangle

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) and (v >= 0) and (u + v < 1)


def is_convex(prev_pt, curr_pt, next_pt):
    """
    Return True if angle (prev_pt → curr_pt → next_pt) is convex (CCW turn).
    """
    v1 = prev_pt - curr_pt
    v2 = next_pt - curr_pt
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    return cross_z < 0  # negative for CCW (assuming y-axis up)


def triangulate_polygon(contour_xy):
    """
    Perform ear‐clipping triangulation on a 2D polygon given by contour_xy,
    an (n × 2) numpy array of vertices in CCW order. Returns a list of
    index triples (i, j, k) indexing contour_xy that form non‐overlapping triangles.
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
            # Could be degenerate or not strictly CCW
            break

    if len(idxs) == 3:
        triangles.append((idxs[0], idxs[1], idxs[2]))

    return triangles


def extrude_to_vertices_faces(xy_points, thickness_mm, num_interp=200):
    """
    Given an (M×2) XY array (the 2D profile, already scaled), resample
    to num_interp points and build vertices/faces for a 3D extrusion.
    Returns:
      - vertices: (2*n × 3) array of floats
      - faces:    (F × 3) array of int indices
    """
    contour = resample_profile_xy(xy_points, num_interp)
    n = contour.shape[0]

    # Build vertices: bottom layer (z=0), top layer (z=thickness_mm)
    vertices = np.zeros((2 * n, 3), dtype=float)
    for i in range(n):
        x_val, y_val = contour[i]
        vertices[i]     = [x_val, y_val, 0.0]
        vertices[n + i] = [x_val, y_val, thickness_mm]

    faces = []

    # Triangulate bottom cap
    bottom_triangles = triangulate_polygon(contour)
    for (i, j, k) in bottom_triangles:
        # Reverse winding so normal points downward
        faces.append([i, k, j])

    # Triangulate top cap
    for (i, j, k) in bottom_triangles:
        faces.append([n + i, n + j, n + k])  # normals point upward

    # Side walls
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
    Write ASCII STL from vertices (V×3) and faces (F×3). Returns the STL as a string.
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
# Main Streamlit Interface
# -------------------------

# 1. File uploader
dat_file = st.file_uploader("1. Upload your `.dat` file (first line is a description)", type=["dat"])
if dat_file:
    raw_lines = dat_file.read().decode("utf-8").splitlines()
    coord_lines = raw_lines[1:]  # skip first line
    pts_list = []
>>>>>>> 1d9681a1debc2a4bfa163e787dec11e4cf2e855b
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
<<<<<<< HEAD
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

        st.subheader("Generate & Preview STL")
        stl_name = st.text_input("Filename (no extension)", value="airfoil_extrusion")
        if not stl_name:
            st.error("Please enter a valid filename.")
        else:
            if st.button("Create STL"):
                verts, faces = extrude_to_vertices_faces(scaled_coords, thickness, num_interp=num_pts)
                stl_text = write_stl_ascii(verts, faces, solid_name=stl_name)
                stl_bytes = stl_text.encode("utf-8")

                st.markdown("**3D Preview**")
                st_viewer(stl_bytes, file_type="stl")

                st.download_button(
                    label="Download `.stl`",
                    data=stl_bytes,
                    file_name=f"{stl_name}.stl",
                    mime="application/vnd.ms-pki.stl"
                )
                st.success(f"STL `{stl_name}.stl` is ready!")
    else:
        st.info("Upload a `.dat` file to enable extrusion.")
=======

    if len(pts_list) < 3:
        st.error("The `.dat` file must contain at least three valid coordinate pairs after the first line.")
        st.stop()

    raw_xy = np.array(pts_list, dtype=float)

    # 2. User inputs: scale factor and thickness
    st.markdown("**2. Enter scaling factor and extrusion thickness**")
    col1, col2 = st.columns(2)
    with col1:
        scale_factor = st.number_input(
            "Scaling factor (e.g., chord length in mm per unit)",
            value=100.0,
            format="%.3f",
            help="Multiply all X,Y coordinates by this factor."
        )
    with col2:
        thickness = st.number_input(
            "Extrusion thickness (mm)",
            value=1.0,
            format="%.3f",
            help="Thickness of the extruded 3D profile."
        )

    # 3. Filename
    st.markdown("**3. Name your STL file**")
    st.write("Filename (no extension):")
    stl_name = st.text_input("", value="airfoil_extrusion").strip()
    if not stl_name:
        st.error("Please provide a valid name for the STL file.")
        st.stop()

    generate_button = st.button("Generate & Download STL")

    if generate_button:
        # 4. Apply scaling to coordinates
        scaled_xy = raw_xy * scale_factor

        # 5. Extrude into vertices & faces
        vertices, faces = extrude_to_vertices_faces(scaled_xy, thickness, num_interp=300)

        # 6. Create ASCII STL text
        stl_text = write_stl_ascii(vertices, faces, solid_name=stl_name)

        # 7. Offer download
        stl_bytes = stl_text.encode("utf-8")
        filename = f"{stl_name}.stl"
        st.download_button(
            label="Click to Download",
            data=stl_bytes,
            file_name=filename,
            mime="application/vnd.ms-pki.stl"
        )
        st.success(f"STL `{filename}` is ready for download!")

else:
    st.info("Please upload a `.dat` file to begin.")
>>>>>>> 1d9681a1debc2a4bfa163e787dec11e4cf2e855b

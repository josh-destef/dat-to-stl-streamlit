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
# Shared helper functions
# -------------------------

def load_dat_coords(dat_bytes):
    """
    Given raw bytes of a .dat file, skip the first line,
    parse the remaining lines as two-column floats,
    and return an (N×2) numpy array. If fewer than 3 valid points, returns an empty array.
    """
    raw_lines = dat_bytes.decode("utf-8", errors="ignore").splitlines()
    coord_lines = raw_lines[1:]  # skip first‐line header/description
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
    pts = np.array(pts, dtype=float)
    if pts.shape[0] < 3:
        return np.zeros((0, 2))
    return pts

def plot_2d_profile(coords, title="Airfoil Profile"):
    """
    Plot a 2D airfoil profile from an (N×2) array, with equal aspect.
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
    Fit a non‐periodic spline through XY points, then resample to num_pts points.
    Ensures no duplicate endpoint.
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
    Barycentric‐test if point pt lies inside triangle ABC.
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
    Return True if angle (prev_pt→curr_pt→next_pt) is convex (cross‐product test).
    """
    v1 = prev_pt - curr_pt
    v2 = next_pt - curr_pt
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    return cross_z < 0

def triangulate_polygon(contour_xy):
    """
    Ear‐clipping triangulation of a CCW 2D polygon.
    Returns a list of index triples (i,j,k) for each triangle.
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
    bottom_triangles = triangulate_polygon(contour)
    # Bottom cap: invert winding so normal points downward
    for (i, j, k) in bottom_triangles:
        faces.append([i, k, j])
    # Top cap: normal upward
    for (i, j, k) in bottom_triangles:
        faces.append([n + i, n + j, n + k])
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
# Hex-hole carving helpers
# -------------------------

def subtract_tapered_hex(verts, faces, cx, cy, top_f2f, bot_f2f, depth, foil_thickness):
    """
    Remove any triangular face whose centroid lies inside the tapered hex volume:
      - Top hex (Z = foil_thickness) has flat-to-flat = top_f2f
      - Bottom hex (Z = foil_thickness − depth) has flat-to-flat = bot_f2f
    Returns (new_verts, new_faces), trimming out unused vertices.
    """
    z_top = foil_thickness
    z_bot = z_top - depth
    kept = []
    cos30 = np.cos(np.pi / 6)
    sin30 = np.sin(np.pi / 6)

    for tri in faces:
        centroid = verts[tri].mean(axis=0)
        x_c, y_c, z_c = centroid
        # 1) Only carve if centroid z is between z_bot and z_top
        if not (z_bot <= z_c <= z_top):
            kept.append(tri)
            continue
        # 2) Interpolate flat-to-flat size at z_c
        lam = (z_top - z_c) / depth  # 0 at z_top, 1 at z_bot
        f2f_at_z = top_f2f - lam * (top_f2f - bot_f2f)
        # 3) Rotate (x_c - cx, y_c - cy) by -30° so hex flats align with axes
        dx = x_c - cx
        dy = y_c - cy
        x_r = dx * cos30 + dy * sin30
        y_r = -dx * sin30 + dy * cos30
        # 4) If inside that hex cross-section, remove
        if max(abs(x_r), abs(y_r) / cos30) <= (f2f_at_z / 2.0):
            continue
        kept.append(tri)

    if not kept:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    kept = np.array(kept, dtype=int)
    unique_v = np.unique(kept.flatten())
    idx_map = {old: new for new, old in enumerate(unique_v)}
    new_verts = verts[unique_v]
    new_faces = np.vectorize(lambda i: idx_map[i])(kept)
    return new_verts, new_faces

def find_max_thickness_x(contour, num_samples=500):
    """
    Given a 2D contour (N×2), sample `num_samples` values of x from min to max.
    At each sampled xg, find all intersections of contour edges with x = xg,
    compute y_top = max(intersections), y_bot = min(intersections),
    thickness = y_top - y_bot. Return xg that yields the maximum thickness.
    """
    xs = contour[:, 0]
    minx, maxx = xs.min(), xs.max()
    x_grid = np.linspace(minx, maxx, num_samples)
    best_x = xs[0]
    best_thk = 0.0

    for xg in x_grid:
        y_ints = []
        # check each segment for intersection with xg
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

def extrude_and_carve_hex(xy_points, thickness_mm, num_interp,
                          scale, top_f2f, bot_f2f, depth):
    """
    1) Resample and scale the 2D airfoil profile.
    2) Extrude to a 3D plate of thickness_mm.
    3) Find x-location of maximum thickness on the unscaled resampled contour,
       set hex center at (x_max*scale, 0).
    4) Carve a tapered hex hole through [z_bot..z_top].
    Returns (verts_final, faces_final, (contour_scaled, cx, cy)).
    """
    # 1) Resample and scale
    contour = resample_profile_xy(xy_points, num_interp)
    contour_scaled = contour * scale

    # 2) Extrude
    n = contour_scaled.shape[0]
    vertices = np.zeros((2 * n, 3), dtype=float)
    for i in range(n):
        x_val, y_val = contour_scaled[i]
        vertices[i] = [x_val, y_val, 0.0]
        vertices[n + i] = [x_val, y_val, thickness_mm]
    faces = []
    bottom_triangles = triangulate_polygon(contour_scaled)
    for (i, j, k) in bottom_triangles:
        faces.append([i, k, j])           # bottom cap (normal down)
    for (i, j, k) in bottom_triangles:
        faces.append([n + i, n + j, n + k])  # top cap (normal up)
    for i in range(n):
        i_next = (i + 1) % n
        b0 = i
        b1 = i_next
        t0 = n + i
        t1 = n + i_next
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
    faces = np.array(faces, dtype=int)

    # 3) Find x of maximum thickness on unscaled contour
    x_unscaled_max = find_max_thickness_x(contour, num_samples=500)
    cx = x_unscaled_max * scale
    cy = 0.0  # centerline for hole

    # 4) Carve hex hole
    verts_carved, faces_carved = subtract_tapered_hex(
        vertices, faces, cx, cy, top_f2f, bot_f2f, depth, thickness_mm
    )

    return verts_carved, faces_carved, (contour_scaled, cx, cy)


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
        if coords_v.shape[0] < 3:
            st.error("Need at least 3 (x,y) points to display a valid airfoil.")
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
# Tab 2: Extrude to STL
# -------------------------
with tab2:
    st.header("Extrude Airfoil to STL with Tapered Hex Hole")

    dat_file_e = st.file_uploader(
        "Upload `.dat` for extrusion (first line is description)",
        type=["dat"], key="extrude"
    )
    if dat_file_e:
        raw_bytes = dat_file_e.read()
        coords_e = load_dat_coords(raw_bytes)

        if coords_e.shape[0] < 3:
            st.error("Failed to parse enough points from the DAT. Please upload a valid airfoil file.")
        else:
            # -------------------------
            # 1) Extrusion parameters
            # -------------------------
            st.subheader("Extrusion Parameters")
            col1, col

import streamlit as st
import numpy as np
from scipy.interpolate import splprep, splev
import io

st.set_page_config(page_title="DAT → STL Extruder", layout="centered")

st.title("DAT → STL Extruder")

st.markdown(
    """
Upload a `.dat` file containing a 2D airfoil cross-section, specify a scaling factor (to convert from nondimensional units to millimeters or your desired units), enter an extrusion thickness, and name the resulting STL. Click **Generate & Download** to receive the STL file.
"""
)

# ---- File Uploader ----
dat_file = st.file_uploader("1. Upload your `.dat` file", type=["dat"])
if dat_file:
    # Read all lines
    raw_lines = dat_file.read().decode("utf-8").splitlines()
    # First line is a description (skip it)
    coord_lines = raw_lines[1:]
    # Parse the rest into a NumPy array
    pts_list = []
    for line in coord_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
                pts_list.append([x_val, y_val])
            except ValueError:
                continue
    if len(pts_list) < 3:
        st.error("The `.dat` file must contain at least three valid coordinate pairs after the first line.")
        st.stop()
    raw_xy = np.array(pts_list, dtype=float)

    # ---- User Inputs ----
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

    st.markdown("**3. Name your STL file**")
    st.write("Filename (no extension):")
    stl_name = st.text_input("", value="airfoil_extrusion").strip()
    if not stl_name:
        st.error("Please provide a valid name for the STL file.")
        st.stop()

    def resample_profile_xy(xy, num_pts=200):
        """
        Fit a periodic (or fallback to nonperiodic) spline through XY points,
        then sample evenly to get `num_pts` points.
        """
        x = xy[:, 0]
        y = xy[:, 1]
        # Always use non-periodic spline to preserve sharp LE corner
        tck, _ = splprep([x, y], s=0, per=0)

        u_new = np.linspace(0.0, 1.0, num_pts)
        out = splev(u_new, tck)
        pts_interp = np.vstack(out).T
        return pts_interp

    def extrude_to_vertices_faces(xy_points, thickness_mm, num_interp=200):
        """
        Given an (M×2) array of XY profile points (already scaled), resample
        to `num_interp` points and build vertices/faces for a 3D extrusion.
        Returns (vertices, faces) where:
          - vertices is a (2*num_interp × 3) float array
          - faces is a (F × 3) int array of triangle indices
        """
        # 1) Resample
        contour = resample_profile_xy(xy_points, num_interp)  # shape (num_interp,2)

        # 2) Ensure closed loop
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        n = contour.shape[0]  # will be num_interp+1 if closed now
        nn = n - 1  # unique points

        # 3) Build vertices: bottom (z=0) and top (z=thickness_mm)
        vertices = np.zeros((2 * nn, 3), dtype=float)
        for i in range(nn):
            x_val, y_val = contour[i]
            vertices[i] = [x_val, y_val, 0.0]
            vertices[nn + i] = [x_val, y_val, thickness_mm]

        faces = []

        # 4) Bottom cap triangulation (fan)
        for i in range(1, nn - 1):
            faces.append([0, i, i + 1])

        # 5) Top cap triangulation (reversed winding to point outward/up)
        top_offset = nn
        for i in range(1, nn - 1):
            faces.append([top_offset + 0, top_offset + i + 1, top_offset + i])

        # 6) Side faces (two triangles per quad)
        for i in range(nn):
            i_next = (i + 1) % nn
            b0 = i
            b1 = i_next
            t0 = nn + i
            t1 = nn + i_next
            faces.append([b0, b1, t1])
            faces.append([b0, t1, t0])

        return vertices, np.array(faces, dtype=int)

    def write_stl_ascii(vertices, faces):
        """
        Write ASCII STL to a string, given vertices (V×3) and faces (F×3).
        """
        def compute_normal(v1, v2, v3):
            n = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(n)
            if norm == 0:
                return np.array([0.0, 0.0, 0.0], dtype=float)
            return n / norm

        output = io.StringIO()
        output.write(f"solid {stl_name}\n")
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
        output.write(f"endsolid {stl_name}\n")
        return output.getvalue()

    generate_button = st.button("Generate & Download STL")

    if generate_button:
        # 1) Apply scaling
        scaled_xy = raw_xy * scale_factor

        # 2) Extrude to get vertices and faces
        vertices, faces = extrude_to_vertices_faces(scaled_xy, thickness, num_interp=300)

        # 3) Create ASCII STL string
        stl_text = write_stl_ascii(vertices, faces)

        # 4) Offer download
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

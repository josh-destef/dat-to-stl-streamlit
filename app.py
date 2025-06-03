import numpy as np
from scipy.interpolate import splprep, splev

def resample_profile_xy(xy, num_pts=200):
    """
    Fit a non‐periodic spline through the given XY points, then
    resample to num_pts points. Ensures no duplicate endpoint.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    # Always use non-periodic spline (per=0)
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
    Check if point pt is inside triangle ABC using barycentric technique.
    pt, a, b, c are arrays or lists of length 2.
    """
    # Compute vectors
    v0 = c - a
    v1 = b - a
    v2 = pt - a

    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Compute barycentric coordinates
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
    # Compute cross product z-component of vectors (curr→next) x (curr→prev)
    v1 = prev_pt - curr_pt
    v2 = next_pt - curr_pt
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    return cross_z < 0  # negative for CCW ordering (y-axis up)


def triangulate_polygon(contour_xy):
    """
    Perform ear‐clipping triangulation on a 2D polygon given by contour_xy,
    an (n × 2) numpy array of vertices in CCW order. Returns a list of
    index triples (i, j, k) indexing contour_xy that form non‐overlapping triangles.
    """
    n = len(contour_xy)
    if n < 3:
        return []

    # Initial list of vertex indices (assume contour_xy is CCW)
    idxs = list(range(n))
    triangles = []

    def get_point(i):
        return contour_xy[i]

    # Loop until only one triangle remains
    guard = 0
    while len(idxs) > 3 and guard < n * n:
        guard += 1
        ear_found = False

        # Try each vertex as a potential "ear tip"
        for ii in range(len(idxs)):
            i = idxs[ii]
            prev_idx = idxs[ii - 1]
            next_idx = idxs[(ii + 1) % len(idxs)]

            A = get_point(prev_idx)
            B = get_point(i)
            C = get_point(next_idx)

            # 1) Check if angle ABC is convex
            if not is_convex(A, B, C):
                continue

            # 2) Check no other point lies inside triangle (A, B, C)
            is_ear = True
            for other in idxs:
                if other in (prev_idx, i, next_idx):
                    continue
                P = get_point(other)
                if point_in_triangle(P, A, B, C):
                    is_ear = False
                    break

            if not is_ear:
                continue

            # If we reach here, (A, B, C) is an ear: clip it
            triangles.append((prev_idx, i, next_idx))
            idxs.pop(ii)  # remove the ear tip from polygon
            ear_found = True
            break

        if not ear_found:
            # The polygon might be degenerate or not strictly CCW.
            break

    # Finally, whatever remains should form one triangle
    if len(idxs) == 3:
        triangles.append((idxs[0], idxs[1], idxs[2]))

    return triangles


def extrude_to_vertices_faces(xy_points, thickness_mm, num_interp=200):
    """
    Given an (M×2) XY array (the 2D profile, already scaled), resample
    to num_interp points and build the 3D extrusion with top+bottom caps.
    Returns:
      - vertices: (2*n × 3) array
      - faces:    (F × 3) array of triangle indices
    """
    # 1) Resample to a smooth (n × 2) contour with no duplicate endpoint
    contour = resample_profile_xy(xy_points, num_interp)
    n = contour.shape[0]  # number of unique points

    # 2) Build 3D vertices: bottom layer (z=0), top layer (z=thickness_mm)
    vertices = np.zeros((2 * n, 3), dtype=float)
    for i in range(n):
        x_val, y_val = contour[i]
        vertices[i] = [x_val, y_val, 0.0]
        vertices[n + i] = [x_val, y_val, thickness_mm]

    faces = []

    # 3) Triangulate the bottom cap (z=0) using ear clipping on contour
    bottom_triangles = triangulate_polygon(contour)
    for (i, j, k) in bottom_triangles:
        # For the bottom cap, we want normals pointing downward,
        # so we reverse the winding order: (i, k, j)
        faces.append([i, k, j])

    # 4) Triangulate the top cap (z=thickness_mm),
    # flipping to ensure normals point upward:
    for (i, j, k) in bottom_triangles:
        # Add offset n for top vertices
        faces.append([n + i, n + j, n + k])

    # 5) Side walls: connect each edge on bottom to corresponding on top
    for i in range(n):
        i_next = (i + 1) % n
        b0 = i
        b1 = i_next
        t0 = n + i
        t1 = n + i_next
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])

    return vertices, np.array(faces, dtype=int)

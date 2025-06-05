# mount_builder.py

import numpy as np


def make_box(origin, length, width, height):
    """
    Returns (verts, faces) for an axis-aligned rectangular prism.
    origin = (x0, y0, z0), length = size in X, width = size in Y, height = size in Z.
    """
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
        [0,1,2], [0,2,3],       # bottom
        [4,6,5], [4,7,6],       # top
        [0,5,1], [0,4,5],       # front (y=y0)
        [3,2,6], [3,6,7],       # back  (y=y0+dy)
        [0,3,7], [0,7,4],       # left  (x=x0)
        [1,5,6], [1,6,2],       # right (x=x0+dx)
    ], dtype=int)

    return verts, faces


def remove_triangles_inside_cylinder(box_verts, box_faces, cyl_center, cyl_radius, axis='z'):
    """
    Remove any triangle whose centroid lies within the circular cross-section
    (cylinder axis parallel to Z). Returns new (verts, faces) trimmed of unused vertices.
    """
    kept = []
    for tri in box_faces:
        centroid = box_verts[tri].mean(axis=0)
        dx = centroid[0] - cyl_center[0]
        dy = centroid[1] - cyl_center[1]
        if np.hypot(dx, dy) > cyl_radius:
            kept.append(tri)

    if not kept:
        return np.zeros((0,3)), np.zeros((0,3), dtype=int)

    kept = np.array(kept, dtype=int)
    unique_v = np.unique(kept.flatten())
    index_map = {old: new for new, old in enumerate(unique_v)}
    new_verts = box_verts[unique_v]
    new_faces = np.vectorize(lambda i: index_map[i])(kept)
    return new_verts, new_faces


def assemble_mount(params):
    """
    Builds a “U‐bracket” (two vertical plates + base) with circular holes removed.
    params keys:
      - plate_W, plate_H, plate_T, spacing_between_plates
      - hole_radius, hole_offset_vert
      - base_plate_length, base_plate_width, base_plate_thick
      - base_hole_radius, base_hole_centers (list of 4 (x,y,z) tuples)
    Returns (all_verts, all_faces) for the combined mesh.
    """
    verts_list = []
    faces_list = []
    v_offset = 0

    W = params['plate_W']
    H = params['plate_H']
    T = params['plate_T']
    S = params['spacing_between_plates']
    hole_r = params['hole_radius']
    hole_z = params['hole_offset_vert']

    # Left vertical plate at (0,0,0)
    left_origin = (0.0, 0.0, 0.0)
    verts_L, faces_L = make_box(left_origin, T, W, H)
    # Carve two holes
    hole_centers_L = [
        (left_origin[0] + T/2, left_origin[1] + W/2, left_origin[2] + hole_z),
        (left_origin[0] + T/2, left_origin[1] + W/2, left_origin[2] + H - hole_z)
    ]
    for hc in hole_centers_L:
        verts_L, faces_L = remove_triangles_inside_cylinder(verts_L, faces_L, hc, hole_r)
    verts_list.append(verts_L)
    faces_list.append(faces_L + v_offset)
    v_offset += verts_L.shape[0]

    # Right vertical plate at (T+S, 0, 0)
    right_origin = (T + S, 0.0, 0.0)
    verts_R, faces_R = make_box(right_origin, T, W, H)
    hole_centers_R = [
        (right_origin[0] + T/2, right_origin[1] + W/2, right_origin[2] + hole_z),
        (right_origin[0] + T/2, right_origin[1] + W/2, right_origin[2] + H - hole_z)
    ]
    for hc in hole_centers_R:
        verts_R, faces_R = remove_triangles_inside_cylinder(verts_R, faces_R, hc, hole_r)
    verts_list.append(verts_R)
    faces_list.append(faces_R + v_offset)
    v_offset += verts_R.shape[0]

    # Base plate (top at z=0, thickness extends downward)
    base_len = params['base_plate_length']
    base_wid = params['base_plate_width']
    base_th  = params['base_plate_thick']
    base_origin = (0.0, 0.0, -base_th)
    verts_B, faces_B = make_box(base_origin, base_len, base_wid, base_th)
    for bhc in params['base_hole_centers']:
        verts_B, faces_B = remove_triangles_inside_cylinder(verts_B, faces_B, bhc, params['base_hole_radius'])
    verts_list.append(verts_B)
    faces_list.append(faces_B + v_offset)

    all_verts = np.vstack(verts_list)
    all_faces = np.vstack(faces_list)
    return all_verts, all_faces

"""
OFF file I/O utilities for ModelNet10 3D models.
OFF format:
    OFF
    <num_vertices> <num_faces> <num_edges>
    x1 y1 z1
    ...
    n v1 v2 ... vn   (face: n=number of vertices, then vertex indices)
"""

import numpy as np


def read_off(filepath):
    """
    Read an OFF file and return vertices and faces.

    Returns:
        vertices : np.ndarray of shape (n_verts, 3)
        faces    : list of lists (face vertex indices)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    idx = 0
    # Skip the 'OFF' header line (sometimes counts follow on same line)
    header = lines[idx].strip()
    if header == 'OFF':
        idx += 1
        count_parts = lines[idx].strip().split()
    elif header.startswith('OFF'):
        count_parts = header[3:].strip().split()
        idx += 1
    else:
        count_parts = header.split()
        idx += 1

    n_verts = int(count_parts[0])
    n_faces = int(count_parts[1])
    idx += 1  # move past count line (already consumed if inline)

    # When header and counts are on separate lines, idx is already right.
    # When they were on same line, we already incremented past counts.
    # Re-check: after header processing, idx should point to vertices.
    vertices = []
    for i in range(n_verts):
        coords = list(map(float, lines[idx + i].strip().split()))
        vertices.append(coords[:3])
    vertices = np.array(vertices, dtype=np.float64)
    idx += n_verts

    faces = []
    for i in range(n_faces):
        parts = list(map(int, lines[idx + i].strip().split()))
        # parts[0] is the face valence (number of vertices in face)
        faces.append(parts[1:parts[0] + 1])

    return vertices, faces


def write_off(filepath, vertices, faces):
    """Write vertices and faces to an OFF file."""
    with open(filepath, 'w') as f:
        f.write('OFF\n')
        f.write(f'{len(vertices)} {len(faces)} 0\n')
        for v in vertices:
            f.write(f'{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n')
        for face in faces:
            f.write(f'{len(face)}')
            for vi in face:
                f.write(f' {vi}')
            f.write('\n')

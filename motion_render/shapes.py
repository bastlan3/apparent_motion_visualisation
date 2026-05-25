from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class ShapeData:
    vertices: Tensor  # [N, 3] float32
    faces: Tensor     # [F, 3] int64
    edges: Tensor     # [E, 2] int64


def _faces_to_edges(faces: Tensor) -> Tensor:
    pairs = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], dim=0)
    pairs = torch.sort(pairs, dim=1).values
    return torch.unique(pairs, dim=0)


def cube() -> ShapeData:
    verts = torch.tensor([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0, 2, 1], [0, 3, 2],  # back
        [4, 5, 6], [4, 6, 7],  # front
        [0, 1, 5], [0, 5, 4],  # bottom
        [2, 3, 7], [2, 7, 6],  # top
        [0, 4, 7], [0, 7, 3],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ], dtype=torch.long)

    return ShapeData(vertices=verts, faces=faces, edges=_faces_to_edges(faces))


def tetrahedron() -> ShapeData:
    verts = torch.tensor([
        [ 1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
    ], dtype=torch.float32) * 0.5

    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ], dtype=torch.long)

    return ShapeData(vertices=verts, faces=faces, edges=_faces_to_edges(faces))


def sphere(subdivisions: int = 2) -> ShapeData:
    t = (1.0 + 5.0 ** 0.5) / 2.0

    raw = [
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ]
    verts = []
    for v in raw:
        n = (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
        verts.append([v[0]/n, v[1]/n, v[2]/n])

    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    cache = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        v1, v2 = verts[a], verts[b]
        m = [(v1[i] + v2[i]) / 2 for i in range(3)]
        n = (m[0]**2 + m[1]**2 + m[2]**2) ** 0.5
        m = [x / n for x in m]
        idx = len(verts)
        verts.append(m)
        cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_faces = []
        for f in faces:
            a, b, c = f
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces

    verts_t = torch.tensor(verts, dtype=torch.float32) * 0.5
    faces_t = torch.tensor(faces, dtype=torch.long)

    return ShapeData(vertices=verts_t, faces=faces_t, edges=_faces_to_edges(faces_t))

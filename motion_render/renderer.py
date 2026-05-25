import math
import torch
from torch import Tensor

from .transforms import look_at_matrix


def _clip_segment(x0, y0, x1, y1, W, H):
    """Cohen-Sutherland line clipping to [0,W] x [0,H]. Returns clipped coords or None."""
    INSIDE = 0; LEFT = 1; RIGHT = 2; ABOVE = 4; BELOW = 8

    def region(x, y):
        c = INSIDE
        if x < 0:   c |= LEFT
        elif x > W: c |= RIGHT
        if y < 0:   c |= ABOVE
        elif y > H: c |= BELOW
        return c

    c0, c1 = region(x0, y0), region(x1, y1)

    for _ in range(12):
        if not (c0 | c1):
            return x0, y0, x1, y1
        if c0 & c1:
            return None

        out = c0 if c0 else c1
        if out & BELOW:
            t = (H - y0) / (y1 - y0) if y1 != y0 else 0.0
            x, y = x0 + t * (x1 - x0), float(H)
        elif out & ABOVE:
            t = (0.0 - y0) / (y1 - y0) if y1 != y0 else 0.0
            x, y = x0 + t * (x1 - x0), 0.0
        elif out & RIGHT:
            t = (W - x0) / (x1 - x0) if x1 != x0 else 0.0
            x, y = float(W), y0 + t * (y1 - y0)
        else:
            t = (0.0 - x0) / (x1 - x0) if x1 != x0 else 0.0
            x, y = 0.0, y0 + t * (y1 - y0)

        if out == c0:
            x0, y0, c0 = x, y, region(x, y)
        else:
            x1, y1, c1 = x, y, region(x, y)

    return None


def _draw_segment(canvas: Tensor, x0, y0, x1, y1, H, W, value=1.0):
    """Xiaolin Wu anti-aliased line onto canvas in-place."""
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    slope = (y1 - y0) / dx if dx != 0.0 else 0.0

    for x in range(int(math.ceil(x0)), int(math.floor(x1)) + 1):
        y_exact = y0 + slope * (x - x0)
        y_lo = int(math.floor(y_exact))
        frac = y_exact - y_lo

        for y_idx, alpha in ((y_lo, 1.0 - frac), (y_lo + 1, frac)):
            r, c = (x, y_idx) if steep else (y_idx, x)
            if 0 <= r < H and 0 <= c < W:
                cur = canvas[r, c].item()
                canvas[r, c] = max(cur, value * alpha)


def _front_facing_mask(world_verts: Tensor, faces: Tensor, camera_eye: Tensor) -> Tensor:
    """Returns [F] bool: True if face normal points toward the camera."""
    v0 = world_verts[faces[:, 0]]
    v1 = world_verts[faces[:, 1]]
    v2 = world_verts[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)          # [F, 3]
    centroid = (v0 + v1 + v2) / 3.0                          # [F, 3]
    to_cam = camera_eye.unsqueeze(0) - centroid               # [F, 3]
    return (normals * to_cam).sum(dim=1) > 0                  # [F]


def _face_cam_depths(world_verts: Tensor, faces: Tensor, V: Tensor) -> Tensor:
    """Returns [F] z-coordinate of face centroids in camera space."""
    v0 = world_verts[faces[:, 0]]
    v1 = world_verts[faces[:, 1]]
    v2 = world_verts[faces[:, 2]]
    centroid = (v0 + v1 + v2) / 3.0                          # [F, 3]
    ones = torch.ones(centroid.shape[0], 1)
    c_cam = (V @ torch.cat([centroid, ones], dim=1).T).T      # [F, 4]
    return c_cam[:, 2]                                        # [F]


def _face_shading(world_verts: Tensor, faces: Tensor, light_dir: Tensor,
                  ambient: float, fill_max: float) -> Tensor:
    """Returns [F] fill brightness using diffuse + ambient shading."""
    v0 = world_verts[faces[:, 0]]
    v1 = world_verts[faces[:, 1]]
    v2 = world_verts[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    diffuse = torch.clamp((normals * light_dir).sum(dim=1), min=0.0)  # [F]
    return ambient + (fill_max - ambient) * diffuse


def _fill_triangle(canvas: Tensor, p0, p1, p2, H: int, W: int, value: float):
    """Fill a triangle onto canvas using a vectorized barycentric test."""
    x_min = max(0, int(math.floor(min(p0[0], p1[0], p2[0]))))
    x_max = min(W - 1, int(math.ceil(max(p0[0], p1[0], p2[0]))))
    y_min = max(0, int(math.floor(min(p0[1], p1[1], p2[1]))))
    y_max = min(H - 1, int(math.ceil(max(p0[1], p1[1], p2[1]))))

    if x_max < x_min or y_max < y_min:
        return

    ys = torch.arange(y_min, y_max + 1, dtype=torch.float32)
    xs = torch.arange(x_min, x_max + 1, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')

    def edge_fn(ax, ay, bx, by):
        return (bx - ax) * (gy - ay) - (by - ay) * (gx - ax)

    e0 = edge_fn(p0[0], p0[1], p1[0], p1[1])
    e1 = edge_fn(p1[0], p1[1], p2[0], p2[1])
    e2 = edge_fn(p2[0], p2[1], p0[0], p0[1])

    inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))

    patch = canvas[y_min:y_max + 1, x_min:x_max + 1]
    canvas[y_min:y_max + 1, x_min:x_max + 1] = torch.where(
        inside, torch.full_like(patch, value), patch
    )


def _visible_edge_mask(faces: Tensor, front_mask: Tensor, edges: Tensor) -> Tensor:
    """Returns [E] bool: True if edge borders at least one front-facing face."""
    edge_keys = {(edges[ei, 0].item(), edges[ei, 1].item()): ei for ei in range(edges.shape[0])}
    visible = torch.zeros(edges.shape[0], dtype=torch.bool)

    for fi in range(faces.shape[0]):
        if not front_mask[fi].item():
            continue
        f = faces[fi].tolist()
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(a, b), max(a, b))
            if key in edge_keys:
                visible[edge_keys[key]] = True

    return visible


class Renderer:
    def __init__(
        self,
        height: int,
        width: int,
        background: float = 0.0,
        edge_value: float = 1.0,
        ambient: float = 0.25,
        fill_max: float = 0.75,
    ):
        self.height = height
        self.width = width
        self.background = background
        self.edge_value = edge_value
        self.ambient = ambient
        self.fill_max = fill_max
        ld = torch.tensor([1.0, 2.0, 3.0])
        self.light_dir = ld / ld.norm()

    def render_scene(self, scene, camera) -> Tensor:
        """Returns [H, W] float32 tensor with values in [0, 1]."""
        canvas = torch.full((self.height, self.width), self.background, dtype=torch.float32)

        up = torch.tensor([0.0, 1.0, 0.0])
        V = look_at_matrix(camera.eye, camera.target, up)

        for body in scene.bodies:
            wv = body.world_vertices()          # [N, 3]
            px, valid = camera.project(wv)      # [N, 2], [N]
            faces = body.shape.faces            # [F, 3]
            edges = body.shape.edges            # [E, 2]

            front = _front_facing_mask(wv, faces, camera.eye)
            depths = _face_cam_depths(wv, faces, V)
            shading = _face_shading(wv, faces, self.light_dir, self.ambient, self.fill_max)

            # Fill faces back-to-front (painter's algorithm)
            fi_front = front.nonzero(as_tuple=False).flatten()
            if fi_front.numel() > 0:
                order = depths[fi_front].argsort(descending=False)  # furthest (most -z) first
                for fi in fi_front[order]:
                    fi_i = fi.item()
                    i0, i1, i2 = faces[fi_i, 0].item(), faces[fi_i, 1].item(), faces[fi_i, 2].item()
                    if valid[i0].item() and valid[i1].item() and valid[i2].item():
                        p0 = [px[i0, 0].item(), px[i0, 1].item()]
                        p1 = [px[i1, 0].item(), px[i1, 1].item()]
                        p2 = [px[i2, 0].item(), px[i2, 1].item()]
                        _fill_triangle(canvas, p0, p1, p2, self.height, self.width, shading[fi_i].item())

            # Draw edges of visible (front-facing) faces on top
            vis = _visible_edge_mask(faces, front, edges)
            for ei in vis.nonzero(as_tuple=False).flatten():
                ei_i = ei.item()
                i0, i1 = edges[ei_i, 0].item(), edges[ei_i, 1].item()
                if valid[i0].item() and valid[i1].item():
                    x0, y0 = px[i0, 0].item(), px[i0, 1].item()
                    x1, y1 = px[i1, 0].item(), px[i1, 1].item()
                    seg = _clip_segment(x0, y0, x1, y1, self.width - 1, self.height - 1)
                    if seg is not None:
                        _draw_segment(canvas, *seg, self.height, self.width, self.edge_value)

        return canvas

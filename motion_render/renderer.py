import math
import torch
from torch import Tensor


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


class Renderer:
    def __init__(self, height: int, width: int, background: float = 0.0):
        self.height = height
        self.width = width
        self.background = background

    def render_scene(self, scene, camera) -> Tensor:
        """Returns [H, W] float32 tensor with values in [0, 1]."""
        canvas = torch.full(
            (self.height, self.width), self.background, dtype=torch.float32
        )

        for body in scene.bodies:
            wv = body.world_vertices()          # [N, 3]
            px, valid = camera.project(wv)      # [N, 2], [N]
            edges = body.shape.edges            # [E, 2]

            for e in range(edges.shape[0]):
                i0 = edges[e, 0].item()
                i1 = edges[e, 1].item()

                if not (valid[i0].item() and valid[i1].item()):
                    continue

                x0, y0 = px[i0, 0].item(), px[i0, 1].item()
                x1, y1 = px[i1, 0].item(), px[i1, 1].item()

                seg = _clip_segment(x0, y0, x1, y1, self.width - 1, self.height - 1)
                if seg is not None:
                    _draw_segment(canvas, *seg, self.height, self.width)

        return canvas

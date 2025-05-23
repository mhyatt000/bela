"""Lightweight plotting utilities."""
from __future__ import annotations

import numpy as np
import torch


def project_points(points: torch.Tensor | np.ndarray, K: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Project 3D camera points to pixel coordinates."""
    if isinstance(points, np.ndarray):
        points = torch.as_tensor(points)
    if isinstance(K, np.ndarray):
        K = torch.as_tensor(K)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return torch.stack((u, v), dim=-1)


def draw_points(img: torch.Tensor, xy: torch.Tensor, color: tuple[float, float, float] = (1.0, 0.0, 0.0), radius: int = 2) -> torch.Tensor:
    """Overlay points on an image tensor."""
    img = img.clone()
    c = torch.tensor(color, dtype=img.dtype, device=img.device).view(3, 1, 1)
    h, w = img.shape[-2:]
    xy = xy.round().to(torch.int32)
    for x, y in xy.reshape(-1, 2):
        if 0 <= x < w and 0 <= y < h:
            xs = slice(max(x - radius, 0), min(x + radius + 1, w))
            ys = slice(max(y - radius, 0), min(y + radius + 1, h))
            img[:, ys, xs] = c
    return img


def plot_cam_pose(batch: dict, K: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return image with cam.pose points overlaid."""
    img = batch["observation.shared.image.low"]
    pts = batch["observation.shared.cam.pose"][..., :3]
    if pts.ndim > 2:
        pts = pts.reshape(-1, 3)
    xy = project_points(pts, K)
    return draw_points(img, xy)

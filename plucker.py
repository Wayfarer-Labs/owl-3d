# https://github.com/chenguolin/DiffSplat/blob/main/src/utils/geo_util.py
def plucker_ray(h: int, w: int, C2W: Tensor, fxfycxcy: Tensor, bug: bool = True) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """Get Plucker ray embeddings.

    Inputs:
        - `h`: image height
        - `w`: image width
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `plucker`: (B, V, 6, `h`, `w`)
        - `ray_o`: (B, V, 3, `h`, `w`)
        - `ray_d`: (B, V, 3, `h`, `w`)
    """
    device, dtype = C2W.device, C2W.dtype
    B, V = C2W.shape[:2]

    C2W = C2W.reshape(B*V, 4, 4).float()
    fxfycxcy = fxfycxcy.reshape(B*V, 4).float()

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")  # OpenCV/COLMAP camera convention
    y, x = y.to(device), x.to(device)
    if bug:  # BUG !!! same here: https://github.com/camenduru/GRM/blob/master/model/visual_encoder/vit_gs.py#L85
        y = y[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) / (h - 1)
        x = x[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) / (w - 1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    else:
        y = (y[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) + 0.5) / h
        x = (x[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) + 0.5) / w
        x = (x - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # (B*V, h*w, 3)
    ray_d = torch.bmm(ray_d, C2W[:, :3, :3].transpose(1, 2))  # (B*V, h*w, 3)
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # (B*V, h*w, 3)
    ray_o = C2W[:, :3, 3][:, None, :].expand_as(ray_d)  # (B*V, h*w, 3)

    ray_o = ray_o.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)
    ray_d = ray_d.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)
    plucker = torch.cat([torch.cross(ray_o, ray_d, dim=2).to(dtype), ray_d.to(dtype)], dim=2)

    return plucker, (ray_o, ray_d)

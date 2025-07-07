#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imageio import imread
from tqdm import tqdm

import gsplat

# ---------------------------------------
# 1. Dataset Definition
# ---------------------------------------
class VideoFrameDataset(Dataset):
    def __init__(self, image_dir, depth_dir, camera_file):
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.endswith('.png'))
        self.depth_dir    = depth_dir
        self.image_dir    = image_dir
        self.cameras      = json.load(open(camera_file, 'r'))

    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        fname = self.image_files[idx]
        # Load image
        img = imread(os.path.join(self.image_dir, fname)).astype(np.float32) / 255.0
        # Load depth (assumed .npy)
        depth = np.load(os.path.join(self.depth_dir, fname.replace('.png', '.npy')))
        # Camera intrinsics & extrinsics
        intr = np.array(self.cameras['intrinsics'], dtype=np.float32)
        extr = np.array(self.cameras['extrinsics'][idx], dtype=np.float32)
        return img, depth, intr, extr

# ---------------------------------------
# 2. Argument Parsing
# ---------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GSplat Video Trainer")
    p.add_argument("--image_dir",    required=True)
    p.add_argument("--depth_dir",    required=True)
    p.add_argument("--camera_file",  required=True)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=1)
    p.add_argument("--lr",           type=float, default=1e-2)
    p.add_argument("--init_scale",   type=float, default=0.01)
    p.add_argument("--init_opacity", type=float, default=0.5)
    return p.parse_args()

# ---------------------------------------
# 3. Main Training Loop
# ---------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Loader
    dataset = VideoFrameDataset(args.image_dir, args.depth_dir, args.camera_file)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 3D Gaussian Initialization containers
    all_pts, all_cols = [], []

    # Unpack camera intrinsics once
    intr0 = np.array(dataset.cameras['intrinsics'], dtype=np.float32)
    fx, fy = intr0[0,0], intr0[1,1]
    cx, cy = intr0[0,2], intr0[1,2]

    # Preprocess: backproject depths
    for img, depth, intr, extr in tqdm(loader, desc="Initializing Gaussians"):
        img   = img[0].cpu().numpy()
        depth = depth[0].cpu().numpy()

        ys, xs = np.where(depth > 0)                                  # [N,]
        z       = depth[ys, xs]
        x_cam   = (xs - cx) * z / fx
        y_cam   = (ys - cy) * z / fy
        pts_cam = np.stack([x_cam, y_cam, z], axis=-1)                # [N,3]

        R = extr[0,:3,:3]
        t = extr[0,:3,3]
        pts_w = (R @ pts_cam.T + t[:,None]).T                        # [N,3]

        all_pts.append(pts_w)
        all_cols.append(img[ys, xs, :])

    pts = torch.tensor(np.concatenate(all_pts, axis=0), dtype=torch.float32, device=device)
    cols = torch.tensor(np.concatenate(all_cols, axis=0), dtype=torch.float32, device=device)

    # GSplat parameter initialization
    means     = torch.nn.Parameter(pts)
    scales    = torch.nn.Parameter(torch.ones_like(pts) * args.init_scale)
    quats     = torch.nn.Parameter(torch.tensor([[1,0,0,0]], device=device).repeat(pts.shape[0],1))
    colors    = torch.nn.Parameter(cols)
    opacities = torch.nn.Parameter(torch.ones(pts.shape[0], device=device) * args.init_opacity)

    optimizer = torch.optim.Adam([means, scales, quats, colors, opacities], lr=args.lr)

    # Training
    for epoch in range(args.epochs):
        total_loss = 0.0
        for img, depth, intr, extr in loader:
            img   = img.to(device)[0]
            extr  = extr.to(device)[0]

            optimizer.zero_grad()
            # Project
            xys, depths2d, radii, conics, num_hits = gsplat.project_gaussians(
                means, scales, 1.0, quats, extr, fx, fy, cx, cy,
                img.shape[1], img.shape[2], 16
            )
            # Rasterize
            rendered, _ = gsplat.rasterize_gaussians(
                xys, depths2d, radii, conics, num_hits,
                colors, opacities, img.shape[1], img.shape[2], 16
            )
            # Loss
            loss = (rendered - img).pow(2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {total_loss/len(loader):.6f}")

if __name__ == "__main__":
    main()

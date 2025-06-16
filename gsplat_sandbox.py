from gsplat import rasterization
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Optional
from torch.autograd import Function

class GaussianRenderer(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_c2w, test_intr, 
               W, H, sh_degree, near_plane, far_plane):
        # Remove extra batch dimensions for gsplat rasterization
        # Expected shapes: xyz[N,3], rotation[N,4], scale[N,3], opacity[N], feature[N,D]
        opacity = opacity.sigmoid().squeeze(-1)  # [N, 1] -> [N]
        scale = scale.exp()  # [N, 3]
        rotation = F.normalize(rotation, p=2, dim=-1)  # [N, 4]
        
        # Convert c2w to w2c (viewmat) and ensure correct shape [1, 4, 4]
        test_w2c = test_c2w.float().inverse().unsqueeze(0) # (1, 4, 4)
        
        # Create intrinsics matrix [1, 3, 3] 
        test_intr_i = torch.zeros(1, 3, 3).to(test_intr.device)
        test_intr_i[0, 0, 0] = test_intr[0]  # fx
        test_intr_i[0, 1, 1] = test_intr[1]  # fy  
        test_intr_i[0, 0, 2] = test_intr[2]  # cx
        test_intr_i[0, 1, 2] = test_intr[3]  # cy
        test_intr_i[0, 2, 2] = 1
        
        rendering, _, _ = rasterization(xyz, rotation, scale, opacity, feature,
                                        test_w2c, test_intr_i, W, H, 
                                        near_plane=near_plane, far_plane=far_plane,
                                        render_mode="RGB",
                                        backgrounds=torch.ones(1, 3).to(test_intr.device),
                                        rasterize_mode='classic') # (1, H, W, 3) 
        return rendering # (1, H, W, 3)

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
                W, H, sh_degree, near_plane, far_plane):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        
        B, V, _ = test_intr.shape
        renderings = torch.zeros(B, V, H, W, 3).to(xyz.device)
        for ib in range(B):
            for iv in range(V):
                renderings[ib, iv:iv+1] = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
                                                                  test_c2ws[ib,iv], test_intr[ib,iv], W, H, sh_degree, near_plane, far_plane)
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        sh_degree = ctx.sh_degree
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            B, V, _ = test_intr.shape
            for ib in range(B):
                for iv in range(V):
                    rendering = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
                                                        test_c2ws[ib,iv], test_intr[ib,iv], W, H, sh_degree, near_plane, far_plane)
                    rendering.backward(grad_output[ib, iv:iv+1])

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None
            
if __name__ == "__main__":
    import json
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import os
    
    # Use data located in output/1K/ from mega-sam
    data_root = "output/1K"
    
    # Get the first available scene
    scene_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    if not scene_dirs:
        print("No scene directories found!")
        exit(1)
    
    scene_dir = os.path.join(data_root, scene_dirs[0])
    print(f"Using scene: {scene_dir}")
    
    # Load transforms.json
    transforms_path = os.path.join(scene_dir, "transforms.json")
    with open(transforms_path, 'r') as f:
        transforms_data = json.load(f)
    
    # Extract camera intrinsics
    W, H = transforms_data['w'], transforms_data['h']
    fx, fy = transforms_data['fl_x'], transforms_data['fl_y']
    cx, cy = transforms_data['cx'], transforms_data['cy']
    
    # Create test Gaussian splats (structured data for better visualization)
    num_gaussians = 1000
    device = torch.device('cuda')  # Force CUDA since gsplat requires it
    print(f"Using device: {device}")
    
    # Create a more structured scene with some geometry
    # Create a grid of Gaussians forming a simple 3D structure
    grid_size = 10
    gaussians_per_dim = int(np.cbrt(num_gaussians))  # Cube root for 3D grid
    
    # Create grid positions
    x_coords = torch.linspace(-1, 1, gaussians_per_dim, device=device)
    y_coords = torch.linspace(-1, 1, gaussians_per_dim, device=device)  
    z_coords = torch.linspace(1, 3, gaussians_per_dim, device=device)  # Forward from camera
    
    # Create meshgrid
    xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    xyz_grid = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
    
    # Take only the first num_gaussians points
    xyz_grid = xyz_grid[:num_gaussians]
    
    # Add some random noise for variation
    xyz = xyz_grid.unsqueeze(0) + torch.randn(1, num_gaussians, 3, device=device) * 0.1
    
    # Create more interesting colors - vary by position
    feature = torch.zeros(1, num_gaussians, 3, device=device)
    feature[0, :, 0] = (xyz[0, :, 0] + 1) / 2  # Red varies with X
    feature[0, :, 1] = (xyz[0, :, 1] + 1) / 2  # Green varies with Y  
    feature[0, :, 2] = (xyz[0, :, 2] - 1) / 2  # Blue varies with Z
    feature = torch.clamp(feature, 0, 1)
    
    # Initialize scales to be small and uniform (log space)
    scale = torch.full((1, num_gaussians, 3), -4.0, device=device) + torch.randn(1, num_gaussians, 3, device=device) * 0.1
    
    # Initialize rotations properly (normalized quaternions)
    rotation = torch.randn(1, num_gaussians, 4, device=device)
    rotation = F.normalize(rotation, p=2, dim=-1)  # Normalize to unit quaternions
    
    # Initialize opacity to be mostly visible
    opacity = torch.full((1, num_gaussians, 1), 2.0, device=device) + torch.randn(1, num_gaussians, 1, device=device) * 0.5
    
    # Prepare camera parameters from transforms.json
    frames = transforms_data['frames'][:4]  # Use first 4 frames for testing
    num_views = len(frames)
    
    # Convert transform matrices to c2w format
    c2w_matrices = []
    for frame in frames:
        transform_matrix = np.array(frame['transform_matrix'])
        c2w_matrices.append(torch.tensor(transform_matrix, dtype=torch.float32, device=device))
    
    test_c2ws = torch.stack(c2w_matrices).unsqueeze(0)  # (1, V, 4, 4)
    
    # Create intrinsics tensor
    test_intr = torch.tensor([fx, fy, cx, cy], device=device).unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1)  # (1, V, 4)
    
    # Rendering parameters
    render_W, render_H = 512, 288  # Reduced resolution for testing
    sh_degree = 0  # Simple RGB without spherical harmonics
    near_plane = 0.1
    far_plane = 100.0
    
    print(f"Rendering {num_views} views at {render_W}x{render_H}")
    print(f"Number of Gaussians: {num_gaussians}")
    
    # Load ground truth images for optimization
    gt_images = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(scene_dir, frames[i]['file_path'].replace('images/', 'images_4/'))
        if os.path.exists(frame_path):
            gt_img = np.array(Image.open(frame_path))
            # Resize to match render size and normalize to [0,1]
            gt_img_resized = np.array(Image.fromarray(gt_img).resize((render_W, render_H)))
            gt_img_tensor = torch.tensor(gt_img_resized, dtype=torch.float32, device=device) / 255.0
            gt_images.append(gt_img_tensor)
        else:
            print(f"Warning: Ground truth image not found for frame {i}")
            # Create a dummy image if GT not available
            gt_images.append(torch.zeros(render_H, render_W, 3, device=device))
    
    gt_images = torch.stack(gt_images).unsqueeze(0)  # (1, V, H, W, 3)
    
    # OPTIMIZATION PHASE
    print("Starting Gaussian optimization...")
    
    # Make parameters trainable
    xyz.requires_grad_(True)
    feature.requires_grad_(True)
    scale.requires_grad_(True)
    rotation.requires_grad_(True)
    opacity.requires_grad_(True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam([
        {'params': [xyz], 'lr': 0.01},
        {'params': [feature], 'lr': 0.01},
        {'params': [scale], 'lr': 0.01},
        {'params': [rotation], 'lr': 0.01},
        {'params': [opacity], 'lr': 0.01}
    ])
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Render current Gaussians
        renderings = GaussianRenderer.apply(
            xyz, feature, scale, rotation, opacity,
            test_c2ws, test_intr,
            render_W, render_H, sh_degree, near_plane, far_plane
        )
        
        # Compute loss (L2 + regularization)
        mse_loss = F.mse_loss(renderings, gt_images)
        
        # Add regularization terms
        opacity_reg = torch.mean(torch.abs(opacity))  # Encourage sparsity
        scale_reg = torch.mean(scale.exp())  # Prevent overly large Gaussians
        
        total_loss = mse_loss + 0.01 * opacity_reg + 0.001 * scale_reg
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Normalize rotations to keep them as valid quaternions
        with torch.no_grad():
            rotation.data = F.normalize(rotation.data, p=2, dim=-1)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {total_loss.item():.6f}, MSE = {mse_loss.item():.6f}")
    
    print("Optimization complete!")
    
    # EVALUATION PHASE
    try:
        with torch.no_grad():
            renderings = GaussianRenderer.apply(
                xyz, feature, scale, rotation, opacity,
                test_c2ws, test_intr,
                render_W, render_H, sh_degree, near_plane, far_plane
            )
        
        print(f"Final rendering successful! Output shape: {renderings.shape}")
        
        # Save rendered images
        output_dir = "rendered_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(num_views):
            # Convert to numpy and normalize properly
            img = renderings[0, i].cpu().numpy()
            # Clamp to [0,1] range and convert to 0-255
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            
            # Save image
            Image.fromarray(img).save(f"{output_dir}/rendered_view_{i:03d}.png")
            print(f"Saved rendered_view_{i:03d}.png")
        
        # Create a visualization comparing rendered views
        fig, axes = plt.subplots(2, num_views, figsize=(4*num_views, 8))
        if num_views == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_views):
            # Show rendered image
            rendered_img = renderings[0, i].cpu().numpy()
            rendered_img = np.clip(rendered_img, 0, 1)  # Ensure valid range for imshow
            axes[0, i].imshow(rendered_img)
            axes[0, i].set_title(f'Rendered View {i}')
            axes[0, i].axis('off')
            
            # Load and show corresponding ground truth image (if available)
            frame_path = os.path.join(scene_dir, frames[i]['file_path'].replace('images/', 'images_4/'))
            if os.path.exists(frame_path):
                gt_img = np.array(Image.open(frame_path))
                # Resize to match rendered size
                gt_img_resized = np.array(Image.fromarray(gt_img).resize((render_W, render_H)))
                axes[1, i].imshow(gt_img_resized)
                axes[1, i].set_title(f'Ground Truth {i}')
            else:
                axes[1, i].text(0.5, 0.5, 'GT not found', ha='center', va='center')
                axes[1, i].set_title(f'Ground Truth {i} (missing)')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to {output_dir}/comparison.png")
        
        # Test gradient computation
        print("\nTesting gradient computation...")
        xyz.requires_grad_(True)
        feature.requires_grad_(True)
        scale.requires_grad_(True)
        rotation.requires_grad_(True)
        opacity.requires_grad_(True)
        
        renderings = GaussianRenderer.apply(
            xyz, feature, scale, rotation, opacity,
            test_c2ws, test_intr,
            render_W, render_H, sh_degree, near_plane, far_plane
        )
        
        # Simple loss for testing gradients
        loss = renderings.mean()
        loss.backward()
        
        print(f"Gradients computed successfully!")
        print(f"xyz grad norm: {xyz.grad.norm().item():.6f}")
        print(f"feature grad norm: {feature.grad.norm().item():.6f}")
        print(f"scale grad norm: {scale.grad.norm().item():.6f}")
        print(f"rotation grad norm: {rotation.grad.norm().item():.6f}")
        print(f"opacity grad norm: {opacity.grad.norm().item():.6f}")
        
        # OPTIMIZATION PHASE - Train Gaussians to match ground truth images
        print("\n" + "="*50)
        print("STARTING GAUSSIAN OPTIMIZATION")
        print("="*50)
        
        # Reset gradients and prepare for optimization
        xyz.grad = None
        feature.grad = None
        scale.grad = None
        rotation.grad = None
        opacity.grad = None
        
        # Load ground truth images for optimization targets
        gt_images = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(scene_dir, frames[i]['file_path'].replace('images/', 'images_4/'))
            if os.path.exists(frame_path):
                gt_img = Image.open(frame_path).convert('RGB')
                gt_img_resized = gt_img.resize((render_W, render_H))
                gt_tensor = torch.tensor(np.array(gt_img_resized), dtype=torch.float32, device=device) / 255.0
                gt_images.append(gt_tensor)
            else:
                print(f"Warning: Ground truth image not found for frame {i}")
                # Create a dummy target (black image)
                gt_images.append(torch.zeros(render_H, render_W, 3, device=device))
        
        gt_targets = torch.stack(gt_images).unsqueeze(0)  # (1, V, H, W, 3)
        print(f"Loaded {len(gt_images)} ground truth images for optimization")
        
        # Optimization parameters
        num_epochs = 500
        learning_rates = {
            'xyz': 1e-3,
            'feature': 1e-2,
            'scale': 1e-3,
            'rotation': 1e-3,
            'opacity': 1e-2
        }
        
        # Create optimizers for different parameter groups
        optimizers = {
            'xyz': torch.optim.Adam([xyz], lr=learning_rates['xyz']),
            'feature': torch.optim.Adam([feature], lr=learning_rates['feature']),
            'scale': torch.optim.Adam([scale], lr=learning_rates['scale']),
            'rotation': torch.optim.Adam([rotation], lr=learning_rates['rotation']),
            'opacity': torch.optim.Adam([opacity], lr=learning_rates['opacity'])
        }
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(num_epochs):
            # Zero gradients
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            
            # Forward pass (remove torch.no_grad() to allow gradients)
            renderings = GaussianRenderer.apply(
                xyz, feature, scale, rotation, opacity,
                test_c2ws, test_intr,
                render_W, render_H, sh_degree, near_plane, far_plane
            )
            
            # Compute losses
            mse_loss = F.mse_loss(renderings, gt_targets)
            
            # Regularization losses
            opacity_reg = 0.01 * torch.mean(torch.abs(opacity.sigmoid() - 0.5))  # Encourage moderate opacity
            scale_reg = 0.001 * torch.mean(torch.abs(scale))  # Prevent scales from becoming too large
            
            total_loss = mse_loss + opacity_reg + scale_reg
            
            # Backward pass
            total_loss.backward()
            
            # Update parameters
            for optimizer in optimizers.values():
                optimizer.step()
            
            # Logging
            if epoch % 50 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}/{num_epochs}: "
                      f"MSE: {mse_loss.item():.6f}, "
                      f"Opacity Reg: {opacity_reg.item():.6f}, "
                      f"Scale Reg: {scale_reg.item():.6f}, "
                      f"Total: {total_loss.item():.6f}")
                
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    
                    # Save optimized renderings periodically
                    if epoch % 100 == 0:
                        with torch.no_grad():
                            opt_renderings = GaussianRenderer.apply(
                                xyz, feature, scale, rotation, opacity,
                                test_c2ws, test_intr,
                                render_W, render_H, sh_degree, near_plane, far_plane
                            )
                            
                            for i in range(num_views):
                                img = opt_renderings[0, i].cpu().numpy()
                                img = np.clip(img, 0, 1)
                                img = (img * 255).astype(np.uint8)
                                Image.fromarray(img).save(f"{output_dir}/optimized_view_{i:03d}_epoch_{epoch:03d}.png")
        
        print(f"\nOptimization completed! Best loss: {best_loss:.6f}")
        
        # Final optimized renderings
        print("Generating final optimized renderings...")
        with torch.no_grad():
            final_renderings = GaussianRenderer.apply(
                xyz, feature, scale, rotation, opacity,
                test_c2ws, test_intr,
                render_W, render_H, sh_degree, near_plane, far_plane
            )
            
            # Save final optimized images
            for i in range(num_views):
                img = final_renderings[0, i].cpu().numpy()
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                Image.fromarray(img).save(f"{output_dir}/final_optimized_view_{i:03d}.png")
                print(f"Saved final_optimized_view_{i:03d}.png")
            
            # Create before/after comparison
            fig, axes = plt.subplots(3, num_views, figsize=(4*num_views, 12))
            if num_views == 1:
                axes = axes.reshape(3, 1)
            
            for i in range(num_views):
                # Original random rendering
                original_img = renderings[0, i].cpu().numpy()
                original_img = np.clip(original_img, 0, 1)
                axes[0, i].imshow(original_img)
                axes[0, i].set_title(f'Original Random View {i}')
                axes[0, i].axis('off')
                
                # Optimized rendering
                optimized_img = final_renderings[0, i].cpu().numpy()
                optimized_img = np.clip(optimized_img, 0, 1)
                axes[1, i].imshow(optimized_img)
                axes[1, i].set_title(f'Optimized View {i}')
                axes[1, i].axis('off')
                
                # Ground truth
                if i < len(gt_images):
                    gt_img = gt_images[i].cpu().numpy()
                    axes[2, i].imshow(gt_img)
                    axes[2, i].set_title(f'Ground Truth {i}')
                else:
                    axes[2, i].text(0.5, 0.5, 'GT not found', ha='center', va='center')
                    axes[2, i].set_title(f'Ground Truth {i} (missing)')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/before_after_optimization.png", dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"Before/after comparison saved to {output_dir}/before_after_optimization.png")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

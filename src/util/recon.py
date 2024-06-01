"""
Mesh reconstruction tools
"""

import warnings

import mcubes
import numpy as np
import torch
import tqdm
from matplotlib import colormaps
import plotly.graph_objects as go

import featurenerf.src.util as util

def vis_grid_plotly(grid : np.ndarray, up=None, center=None, eye=None, size=2, output_name=None):
    """Visualize grid in plotly
    
    Args:
        grid (np.ndarray): (X, Y, Z) grid with values
    """
    # Create data_ls
    data_ls = []
    X, Y, Z = grid.shape
    L = max(max(X, Y), Z)
    grid_max = grid.max()
    grid_min = grid.min()
    cmap = colormaps.get_cmap("plasma")
    x_ls = []
    y_ls = []
    z_ls = []
    plotly_colors = []
    for x_i in range(X):
        for y_i in range(Y):
            for z_i in range(Z):
                x, y, z = x_i / L, y_i / L, z_i / L
                value = grid[x_i, y_i, z_i]
                colors = cmap((value - grid_min) / (grid_max - grid_min))[:3]

                # append
                x_ls.append(x)
                y_ls.append(y)
                z_ls.append(z)
                plotly_colors.append(f'rgb({colors[0]}, {colors[1]}, {colors[2]})')
    data_ls.append(go.Scatter3d(x=np.array(x_ls), y=np.array(y_ls), z=np.array(z_ls), mode='markers', marker=dict(size=size, color=plotly_colors,)))
        
    # Plot mesh
    go_pcd = go.Figure(data=data_ls,
                       layout=go.Layout(scene=dict(aspectmode='data'),))

    if up is not None:
        camera = dict(
            up=dict(x=up[0], y=up[1], z=up[2]),
            center=dict(x=center[0], y=center[1], z=center[2]),
            eye=dict(x=eye[0], y=eye[1], z=eye[2])
        )

        go_pcd.update_layout(scene_camera=camera)

    go_pcd.update_layout(margin=dict(l=5,r=5,b=5,t=5,), showlegend=False)
    go_pcd.show()
    if output_name is not None:
        go_pcd.write_html(f'{output_name}.html')

def vis_pts_with_values_plotly(pts : np.ndarray, values : np.ndarray, up=None, center=None, eye=None, size=2, output_name=None):
    """Visualize grid in plotly
    
    Args:
        pts (np.ndarray): (N, 3) points
        values (np.ndarray): (N, ) values
    """
    assert pts.shape[0] == values.shape[0]
    # Create data_ls
    data_ls = []
    N = pts.shape[0]
    grid_max = values.max()
    grid_min = values.min()
    cmap = colormaps.get_cmap("plasma")
    colors = cmap((values - grid_min) / (grid_max - grid_min))[:, :3]
    plotly_colors = []
    for i in range(N):
        # append
        plotly_colors.append(f'rgb({colors[i, 0]}, {colors[i, 1]}, {colors[i, 2]})')
    data_ls.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers', marker=dict(size=size, color=plotly_colors,)))
        
    # Plot mesh
    go_pcd = go.Figure(data=data_ls,
                       layout=go.Layout(scene=dict(aspectmode='data'),))

    if up is not None:
        camera = dict(
            up=dict(x=up[0], y=up[1], z=up[2]),
            center=dict(x=center[0], y=center[1], z=center[2]),
            eye=dict(x=eye[0], y=eye[1], z=eye[2])
        )

        go_pcd.update_layout(scene_camera=camera)

    go_pcd.update_layout(margin=dict(l=5,r=5,b=5,t=5,), showlegend=False)
    go_pcd.show()
    if output_name is not None:
        go_pcd.write_html(f'{output_name}.html')

def vis_pts_with_rgbs_plotly(pts : np.ndarray, rgbs : np.ndarray, up=None, center=None, eye=None, size=2, output_name=None):
    """Visualize points with colors in plotly
    
    Args:
        pts (np.ndarray): (N, 3) points
        rgbs (np.ndarray): (N, 3) colors
    """
    assert pts.shape[0] == rgbs.shape[0]
    # Create data_ls
    data_ls = []
    N = pts.shape[0]
    plotly_colors = []
    for i in range(N):
        # append
        plotly_colors.append(f'rgb({rgbs[i, 0]}, {rgbs[i, 1]}, {rgbs[i, 2]})')
    data_ls.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers', marker=dict(size=size, color=plotly_colors,)))

    # Plot mesh
    go_pcd = go.Figure(data=data_ls,
                          layout=go.Layout(scene=dict(aspectmode='data'),))
    
    if up is not None:
        camera = dict(
            up=dict(x=up[0], y=up[1], z=up[2]),
            center=dict(x=center[0], y=center[1], z=center[2]),
            eye=dict(x=eye[0], y=eye[1], z=eye[2])
        )

        go_pcd.update_layout(scene_camera=camera)
    
    go_pcd.update_layout(margin=dict(l=5,r=5,b=5,t=5,), showlegend=False)
    go_pcd.show()
    if output_name is not None:
        go_pcd.write_html(f'{output_name}.html')

def marching_cubes(
    occu_net,
    c1=[-1, -1, -1],
    c2=[1, 1, 1],
    reso=[128, 128, 128],
    isosurface=50.0,
    sigma_idx=3,
    eval_batch_size=100000,
    coarse=True,
    device=None,
    pose=None,
):
    """
    Run marching cubes on network. Uses PyMCubes.
    WARNING: does not make much sense with viewdirs in current form, since
    sigma depends on viewdirs.
    :param occu_net main NeRF type network
    :param c1 corner 1 of marching cube bounds x,y,z
    :param c2 corner 2 of marching cube bounds x,y,z (all > c1)
    :param reso resolutions of marching cubes x,y,z
    :param isosurface sigma-isosurface of marching cubes
    :param sigma_idx index of 'sigma' value in last dimension of occu_net's output
    :param eval_batch_size batch size for evaluation
    :param coarse whether to use coarse NeRF for evaluation
    :param device optionally, device to put points for evaluation.
    By default uses device of occu_net's first parameter.
    """
    if occu_net.use_viewdirs:
        warnings.warn("Running marching cubes with fake view dirs (pointing to origin), output may be invalid")
    with torch.no_grad():
        grid = util.gen_grid(*zip(c1, c2, reso), ij_indexing=True)
        is_train = occu_net.training

        print("Evaluating sigma @", grid.size(0), "points")
        occu_net.eval()

        all_sigmas = []
        all_rgbs = []
        if device is None:
            device = next(occu_net.parameters()).device
        grid_spl = torch.split(grid, eval_batch_size, dim=0)
        if occu_net.use_viewdirs:
            # # generate viewdirs
            # grid_rays = util.gen_rays(pose.reshape(-1, 4, 4), 128, 128, 140, 0.8, 1.8)
            # all_rays = grid_rays.reshape(1, -1, 8)
            # fake_viewdirs = all_rays[..., 3:6]

            fake_viewdirs = -grid / torch.norm(grid, dim=-1).unsqueeze(-1)
            vd_spl = torch.split(fake_viewdirs, eval_batch_size, dim=0)
            for pnts, vd in tqdm.tqdm(zip(grid_spl, vd_spl), total=len(grid_spl)):
                outputs = occu_net(pnts.to(device=device)[None], coarse=coarse, viewdirs=vd.to(device=device))
                sigmas = outputs[..., sigma_idx]
                all_sigmas.append(sigmas.cpu()[0])
                all_rgbs.append(outputs[..., :3].cpu()[0])
        else:
            for pnts in tqdm.tqdm(grid_spl):
                outputs = occu_net(pnts.to(device=device), coarse=coarse)
                sigmas = outputs[..., sigma_idx]
                all_sigmas.append(sigmas.cpu()[0])
                all_rgbs.append(outputs[..., :3].cpu()[0])
        sigmas = torch.cat(all_sigmas, dim=0)
        sigmas = sigmas.view(*reso).cpu().numpy()

        rgbs = torch.cat(all_rgbs, dim=0)
        rgbs = rgbs.view(*reso, 3).cpu().numpy()

        vis_pts_with_values_plotly(grid.cpu().numpy()[::100], sigmas.reshape(-1)[::100])
        # vis_grid_plotly(sigmas[::4][:, ::4][..., ::4])

        print("Running marching cubes")
        vertices, triangles = mcubes.marching_cubes(sigmas, isosurface)
        vertices_color = []
        for v in vertices:
            x, y, z = v
            rgb = rgbs[int(x), int(y), int(z)]
            vertices_color.append(rgb)
        vertices_color = np.array(vertices_color)
        # Scale
        c1, c2 = np.array(c1), np.array(c2)
        vertices *= (c2 - c1) / np.array(reso)

    if is_train:
        occu_net.train()
    return vertices + c1, triangles, vertices_color


def save_obj(vertices, triangles, path, vert_rgb=None):
    """
    Save OBJ file, optionally with vertex colors.
    This version is faster than PyMCubes and supports color.
    Taken from PIFu.
    :param vertices (N, 3)
    :param triangles (N, 3)
    :param vert_rgb (N, 3) rgb
    """
    file = open(path, "w")
    if vert_rgb is None:
        # No color
        for v in vertices:
            file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    else:
        # Color
        for idx, v in enumerate(vertices):
            c = vert_rgb[idx]
            file.write("v %.4f %.4f %.4f %.4f %.4f %.4f\n" % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in triangles:
        f_plus = f + 1
        file.write("f %d %d %d\n" % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

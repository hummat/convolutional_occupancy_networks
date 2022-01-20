import copy
import glob
import os
import time
import sys

import numpy as np
import torch
from torchvision import transforms
import tqdm
import trimesh
import open3d as o3d
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation
from easy_o3d.utils import eval_data, draw_geometries, process_point_cloud, OutlierTypes

sys.path.append("")
from src import config, data
from src.checkpoints import CheckpointIO
from src.data import fields, seed_all_rng
from src.eval import MeshEvaluator
from src.common import look_at


def load_mesh(file_path: str, process: bool = True, padding: float = 0.1):
    mesh = trimesh.load(file_path, process=False)

    if process:
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        scale = total_size / (1 - padding)
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / scale)

    return mesh


def process_mesh(mesh, padding: float = 0, flip_yz: bool = True, with_transforms: bool = True):
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max() / (1 - padding)

    mesh.apply_translation(-loc)
    mesh.apply_scale(1 / scale)

    if flip_yz:
        angle = 90 / 180 * np.pi
        R = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
        mesh.apply_transform(R)

    if with_transforms:
        return mesh, loc, scale
    return mesh


def from_pointcloud(use_trimesh=True, visualize=True):
    path_prefix = "/home/matthias/Data/Ubuntu/git/convolutional_occupancy_networks"
    default_path = os.path.join(path_prefix, "configs/default.yaml")
    model_path = os.path.join(path_prefix, "configs/pointcloud/shapenet_grid32_depth_like_upper.yaml")
    cfg = config.load_config(model_path, default_path)
    device = torch.device("cuda")
    # file_path = "/home/matthias/Data2/datasets/bop/ycb/models/021_bleach_cleanser/google_512k/nontextured.ply"
    file_path = "/home/matthias/Data/Ubuntu/data/agile_justin/exp8/object_points.ply"

    if use_trimesh:
        mesh = load_mesh(file_path)
        mesh, loc, scale = process_mesh(mesh, flip_yz=True)
        # T = np.eye(4)
        # T[:2] *= -1
        # mesh.apply_transform(T)

        #points = mesh.sample(100000).astype(np.float32)
        all_points = mesh.vertices
        all_points[:, 1] *= -1
        # side = np.random.randint(3)
        # xb = [points[:, side].min(), points[:, side].max()]
        # length = np.random.uniform(0.5 * (xb[1] - xb[0]), (xb[1] - xb[0]))
        # ind = (points[:, side] - xb[0]) <= length
        # points = points[ind]

        indices = np.random.randint(all_points.shape[0], size=len(all_points))
        points = all_points[indices, :].astype(np.float32)

        noise = 0.005 * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        points = points + noise

        if visualize:
            trimesh.PointCloud(points).show()
            # visualize_pointcloud(points, show=True)
    else:
        import open3d as o3d
        o3d_pcd = o3d.io.read_point_cloud(filename="/home/matthias/Data/Ubuntu/data/agile_justin/scene3/scene_points.ply")
        o3d_pcd.points = o3d.utility.Vector3dVector(np.asarray(o3d_pcd.points) * 0.002)
        scene = copy.deepcopy(o3d_pcd)

        # Crop with model bbox
        o3d_mesh = o3d.io.read_triangle_mesh(filename=file_path)
        max_size = o3d_mesh.get_axis_aligned_bounding_box().get_max_extent() * 1.2
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(0, 0, 0), max_bound=(max_size, max_size, max_size))
        bounding_box = bbox.translate((0.5, 0.5, 0.5), relative=False)
        o3d_pcd = o3d_pcd.crop(bounding_box=bounding_box)

        # Remove table
        points = np.asarray(o3d_pcd.points)
        points = points[points[:, 2] > 0.465]
        o3d_pcd.points = o3d.utility.Vector3dVector(points)

        # Record current position
        pos = o3d_pcd.get_center()

        # Normalize
        total_size = (o3d_pcd.get_max_bound() - o3d_pcd.get_min_bound()).max()
        padding = 0.1
        scale = total_size / (1 - padding)
        o3d_pcd = o3d_pcd.translate(translation=[0, 0, 0], relative=False)
        o3d_pcd.points = o3d.utility.Vector3dVector(np.asarray(o3d_pcd.points) * np.reciprocal(scale))

        # Rotate
        R1 = o3d_pcd.get_rotation_matrix_from_xyz(np.array([np.radians(0), np.radians(0), np.radians(6)]))
        o3d_pcd = o3d_pcd.rotate(R=R1, center=o3d_pcd.get_center())
        R2 = np.asarray([0, 1, 0, 0, 0, 1, 1, 0, 0]).reshape(3, 3)
        o3d_pcd.rotate(R=R2, center=o3d_pcd.get_center())

        # Downsample and add noise
        o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=0.015)
        points = np.asarray(o3d_pcd.points, dtype=np.float32)
        indices = np.random.randint(points.shape[0], size=3000)
        points = points[indices, :]

        noise = 0.005 * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        points = points + noise

        if visualize:
            o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            o3d_pcd.paint_uniform_color([0.8, 0.0, 0.0])
            to_draw = [o3d_pcd, o3d.geometry.TriangleMesh().create_coordinate_frame(size=total_size)]
            o3d.visualization.draw_geometries(to_draw)

    data = {'inputs': torch.unsqueeze(torch.from_numpy(points), dim=0)}

    model = config.get_model(cfg, device)
    checkpoint_io = CheckpointIO("..", model=model)
    checkpoint_io.load(os.path.join(path_prefix, cfg['training']['out_dir'], cfg['test']['model_file']))
    # checkpoint_io.load(cfg['test']['model_file'])
    model.eval()

    generator = config.get_generator(model, cfg, device)
    mesh_pred = generator.generate_mesh(data, return_stats=False)

    evaluator = MeshEvaluator()
    print(evaluator.eval_pointcloud(mesh_pred.sample(len(all_points)), all_points))

    if visualize:
        if use_trimesh:
            mesh_pred.show()
        else:
            obj = o3d.geometry.TriangleMesh()
            obj.vertices = o3d.utility.Vector3dVector(mesh_pred.vertices)
            obj.triangles = o3d.utility.Vector3iVector(mesh_pred.faces)

            obj.rotate(R=R2.T, center=obj.get_center())
            obj.rotate(R=R1.T, center=obj.get_center())

            obj.scale(scale, center=obj.get_center())
            obj.translate(translation=pos, relative=False)

            scene.paint_uniform_color([0.8, 0.8, 0.8])
            scene.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

            obj.paint_uniform_color([0.8, 0.0, 0.0])
            obj.compute_triangle_normals()
            obj.compute_vertex_normals()
            o3d.visualization.draw_geometries([scene, obj])
    else:
        mesh_pred.export("smile.off")


def data_test():
    seed_all_rng(63)
    # path = np.array(sorted(glob.glob("/home/matthias/Data2/datasets/shapenet/ShapeNetCore.v1/*/*")))
    path = np.array(sorted(glob.glob("/home/matthias/Data2/datasets/shapenet/occupancy_networks/ShapeNet/extra/02876657/*")))
    path = np.array(sorted(glob.glob("/home/matthias/Data2/datasets/shapenet/depth/02876657/*")))
    # path = np.array([p.replace("occupancy_networks/ShapeNet/core", "ShapeNetCore.v1") for p in path])
    path = np.array([p for p in path if not p.endswith(".lst")])
    transform = [
        data.SubsamplePointcloud(3000),
        data.PointcloudNoise(0.005)
    ]
    transform = transforms.Compose(transform)
    input_field = fields.DepthLikePointCloudField("pointcloud.npz",
                                                  upper_hemisphere=True,
                                                  sample_camera_position='',
                                                  rotate_object='yx',
                                                  transform=transform)  # 22.7s (mesh), 36.6s (pcd)
    input_field = fields.BlenderProcDepthPointCloudField(transform=transform, unscale=True, project=True)
    pointcloud_field = fields.PointCloudField("pointcloud.npz")
    points_field = fields.PointsField("points.npz", unpackbits=True)

    data_transform = transforms.Compose([data.Rotate(to_cam_frame=True),
                                         # data.Scale(scale_range=(0.1, 1)),
                                         data.Normalize(scale=False)])
    data_transform = data.Rotate(to_world_frame=True)

    path = sorted(list(np.random.choice(path, size=10, replace=False)) * 10)
    inputs = Parallel(n_jobs=16)(delayed(input_field.load)(p, i, 0) for i, p in enumerate(tqdm.tqdm(path)))
    path = [p.replace("depth", "occupancy_networks/ShapeNet/extra") for p in path]
    pointclouds = Parallel(n_jobs=16)(delayed(pointcloud_field.load)(p, i, 0) for i, p in enumerate(tqdm.tqdm(path)))
    points = Parallel(n_jobs=16)(delayed(points_field.load)(p, i, 0) for i, p in enumerate(tqdm.tqdm(path)))
    for i, (inp, pcd, point) in enumerate(zip(inputs, pointclouds, points)):
        data_fields = {"inputs": inp[None],
                       "inputs.rot": inp["rot"],
                       "inputs.x_angle": inp["x_angle"],
                       "pointcloud": pcd[None],
                       "points": point[None]}

        data_fields = data_transform(data_fields)
        trafo = np.eye(4)
        trafo[:3, :3] = inp["rot"].T
        trafo[:3, 3] = inp["cam"]
        #points, points_gt = input_data["inputs"], input_data["pointcloud"]

        forward = [0, 0, 1]
        up = [0, 1, 0]
        # forward, right, up = look_at(cams[i], return_frame=True)
        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_fields["inputs"])),
                                           o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_fields["pointcloud"])).paint_uniform_color([0.8, 0.8, 0.8]),
                                           o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
                                           o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(trafo)],
                                          window_name=path[i].split('/')[-1],
                                          zoom=1,
                                          lookat=[0, 0, 0],
                                          front=forward,
                                          up=up)


def loader_test():
    cfg = config.load_config(os.path.abspath('configs/pointcloud/shapenet_grid32_depth_like_world_bottle.yaml'), 'configs/default.yaml')
    # cfg['data']['input_type'] = 'blenderproc'
    val_dataset = config.get_dataset('val', cfg, return_idx=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=cfg['training']['n_workers_val'],
                                             shuffle=False,
                                             collate_fn=data.collate_remove_none,
                                             worker_init_fn=data.worker_init_reset_seed)

    for _ in range(3):
        for _ in tqdm.tqdm(val_loader):
            pass


if __name__ == "__main__":
    loader_test()

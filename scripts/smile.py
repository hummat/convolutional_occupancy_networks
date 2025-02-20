import copy
import glob
import os
import pathlib
import time
import sys
from multiprocessing import cpu_count

import numpy as np
import torch
from torch.utils.data import DataLoader
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
    default_path = os.path.join(path_prefix, "configs/default_new.yaml")
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
    seed_all_rng()

    #cfg = config.load_config(os.path.abspath('configs/bottle/partial.yaml'), 'configs/default_new.yaml')
    #model = config.get_model(cfg)
    #model.load_state_dict(torch.load(os.path.abspath("out/pointcloud/shapenet_grid32_partial/model_best.pt"))["model"])
    #model.eval()

    # generator = config.get_generator(model, cfg, device="cuda")

    # path = np.array(sorted(glob.glob("/home/matthias/Data2/datasets/shapenet/ShapeNetCore.v1/*/*")))
    # path = np.array(sorted(glob.glob("/home/matthias/Data2/datasets/shapenet/occupancy_networks/ShapeNet/extra/02876657/*")))
    path = np.array(sorted(glob.glob("/home/matthias/Data2/datasets/shapenet/distorted/02876657/*")))
    path = np.array([p for p in path if os.path.exists(os.path.join(p, "scales.npy"))])
    path = np.array([p.replace("distorted", "occupancy_networks/ShapeNet/extra") for p in path])
    path = np.array([p for p in path if not p.endswith(".lst")])
    transform = [
        # data.PointcloudNoise(0),
        # data.VoxelizePointcloud(voxel_size=0.002)
    ]
    transform = transforms.Compose(transform)
    input_field = fields.DepthLikePointCloudField("pointcloud.npz",
                                                  upper_hemisphere=True,
                                                  sample_camera_position='',
                                                  rotate_object='yx',
                                                  transform=transform)
    input_field = fields.BlenderProcDepthPointCloudField(transform=transform,
                                                         path_prefix="/home/matthias/Data2/datasets/shapenet/depth")
    # input_field = fields.PartialPointCloudField("pointcloud.npz",
    #                                             transform=transform,
    #                                             rotate_object='y',
    #                                             axes='z',
    #                                             part_ratio=[0.4, 0.6])
    pointcloud_field = fields.PointCloudField("pointcloud.npz")
    points_field = fields.PointsField("points.npz", unpackbits=True)

    data_transform = transforms.Compose([data.Scale(axes="xz", amount=0.5, random=True),
                                         data.Rotate(to_world_frame=True),
                                         # data.VoxelizeInputs(voxel_size=0.002),
                                         data.Normalize(scale=False)
                                         ])
    # data_transform = data.Rotate(axis='y')

    path = sorted(list(np.random.choice(path, size=min(10, len(path)), replace=False)) * 10)
    # path = ["/home/matthias/Data2/datasets/shapenet/occupancy_networks/ShapeNet/extra/02876657/9edfe64416a9646e30612f5c0ef21eb8"] * 100
    if len(path) >= 256:
        inputs = Parallel(n_jobs=16)(delayed(input_field.load)(p, i, 0) for i, p in enumerate(tqdm.tqdm(path)))
        pointclouds = Parallel(n_jobs=16)(delayed(pointcloud_field.load)(p, i, 0) for i, p in enumerate(tqdm.tqdm(path)))
        points = Parallel(n_jobs=16)(delayed(points_field.load)(p, i, 0) for i, p in enumerate(tqdm.tqdm(path)))
    else:
        inputs = [input_field.load(p, 0, 0) for p in tqdm.tqdm(path)]
        pointclouds = [pointcloud_field.load(p, 0, 0) for p in tqdm.tqdm(path)]
        points = [points_field.load(p, 0, 0) for p in tqdm.tqdm(path)]
    for i, (inp, pcd, point) in enumerate(zip(inputs, pointclouds, points)):
        data_fields = {"inputs": inp[None],
                       "inputs.rot": inp["rot"],
                       "inputs.x_angle": inp["x_angle"],
                       "inputs.scale": inp["scale"],
                       "pointcloud": pcd[None],
                       "points": point[None],
                       "occ": point["occ"]}

        data_fields = data_transform(data_fields)

        """
        input_points = torch.unsqueeze(torch.from_numpy(data_fields["inputs"].astype(np.float32)), dim=0)
        mesh_data = generator.generate_mesh({"inputs": input_points}, return_stats=False)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_data.vertices),
                                         o3d.utility.Vector3iVector(mesh_data.faces))
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        """

        geometries = [
            # mesh,
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_fields["inputs"])),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_fields["points"][data_fields["occ"] >= 0.5])).paint_uniform_color([0.8, 0.8, 0.8])
        ]
        frames = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
                  # o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).translate(inp["cam"], relative=False)
                  ]
        # for loc in inp["locs"]:
        #     print(inp["cam"] - loc)
        #     frames.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).translate(loc, relative=False))
        geometries.extend(frames)
        #points, points_gt = input_data["inputs"], input_data["pointcloud"]

        forward = [0, 0, 1]
        up = [0, 1, 0]
        # forward, right, up = look_at(cams[i], return_frame=True)
        o3d.visualization.draw_geometries(geometries,
                                          window_name=path[i].split('/')[-1],
                                          zoom=1,
                                          lookat=[0, 0, 0],
                                          front=forward,
                                          up=up)


def loader_test():
    seed_all_rng(0)
    generator = torch.Generator().manual_seed(0)

    cfg = config.load_config(os.path.abspath('configs/agile/test.yaml'), 'configs/default_new.yaml')
    dataset = config.get_dataset('train', cfg, return_idx=False)

    print(f"Using {cpu_count()} CPUs.")
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=cfg['training']['batch_size'],
                                         num_workers=cpu_count(),
                                         shuffle=True,
                                         pin_memory=True,
                                         collate_fn=data.heterogeneous_batching,
                                         worker_init_fn=data.worker_init_reset_seed,
                                         generator=generator,
                                         persistent_workers=True)

    for i in range(10):
        print("Run", i)
        for batch in tqdm.tqdm(loader, disable=False):
            pass
        print()


if __name__ == "__main__":
    data_test()

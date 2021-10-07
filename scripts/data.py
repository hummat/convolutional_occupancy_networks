import glob
import time
import os

import numpy as np
import open3d as o3d
import trimesh
import h5py
from scipy.interpolate import RegularGridInterpolator


def load_sdf(sdf_file, sdf_res):
    intsize = 4
    floatsize = 8
    sdf = {
        "param": [],
        "value": []
    }
    with open(sdf_file, "rb") as f:
        try:
            bytes = f.read()
            ress = np.frombuffer(bytes[:intsize * 3], dtype=np.int32)
            if -1 * ress[0] != sdf_res or ress[1] != sdf_res or ress[2] != sdf_res:
                raise Exception(sdf_file, "res not consistent with ", str(sdf_res))
            positions = np.frombuffer(bytes[intsize * 3:intsize * 3 + floatsize * 6], dtype=np.float64)
            # bottom left corner, x,y,z and top right corner, x, y, z
            sdf["param"] = [positions[0], positions[1], positions[2], positions[3], positions[4], positions[5]]
            sdf["param"] = np.float32(sdf["param"])
            sdf["value"] = np.frombuffer(bytes[intsize * 3 + floatsize * 6:], dtype=np.float32)
            sdf["value"] = np.reshape(sdf["value"], (sdf_res + 1, sdf_res + 1, sdf_res + 1))
        finally:
            f.close()
    return sdf


def eval_sdf():
    synthset = "02876657"
    obj_id = "1ffd7113492d375593202bf99dddc268"

    disn_prefix = "/home/matthias/Data2/datasets/shapenet/disn"
    occnet_prefix = "/home/matthias/Data/Ubuntu/git/occupancy_networks/data/ShapeNet.build"

    occnet_path = os.path.join(occnet_prefix, synthset, "4_watertight_scaled", obj_id + ".off")
    disn_path = os.path.join(disn_prefix, synthset, obj_id, "isosurf.obj")
    disn_sdf_path = os.path.join(disn_prefix, "SDF_v1", synthset, obj_id, "ori_sample.h5")
    shapenet_path = os.path.join("/home/matthias/Data2/datasets/shapenet/ShapeNetCore.v1", synthset, obj_id,
                                 "model.obj")
    shapenet2_path = os.path.join("/home/matthias/Data2/datasets/shapenet/ShapeNetCore.v2", synthset, obj_id,
                                  "models/model_normalized.obj")
    sdfgen_path = os.path.join("/home/matthias/Data2/datasets/shapenet/sdfgen", synthset, obj_id, "output_0.hdf5")

    sdf = True
    pcd = False
    mesh = False
    sdfgen = False
    if pcd:
        with h5py.File(disn_sdf_path, "r") as f:
            print("Keys: %s" % f.keys())
            points = f["pc_sdf_sample"].value[:, :3]
            sdf = f["pc_sdf_sample"].value[:, 3]
            inside = points[sdf <= 0]
            outside = points[sdf > 0]
            pcd = trimesh.PointCloud(inside)
            print(pcd.bounds)
            pcd.show()
    elif mesh:
        file = "/home/matthias/Data/Ubuntu/git/DISN/02876657/1ffd7113492d375593202bf99dddc268/1ffd7113492d375593202bf99dddc268.obj"
        mesh = o3d.io.read_triangle_mesh(file)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        size = 0.5 * (mesh.get_max_bound() - mesh.get_min_bound()).max()
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=size)
        o3d.visualization.draw_geometries([mesh, frame])

        #mesh = trimesh.load(file, process=False)
        #print(mesh.bounds)
        #mesh.show()
    elif sdf:
        # for file in glob.glob("/home/matthias/Data2/datasets/shapenet/matthias/02876657/**/*.dist"):
        file = "/home/matthias/Data/Ubuntu/git/DISN/02876657/1ffd7113492d375593202bf99dddc268/1ffd7113492d375593202bf99dddc268.dist"
        sdf = load_sdf(file, 256)

        sample = False
        if sample:
            x = np.linspace(sdf["param"][0], sdf["param"][3], num=257)
            y = np.linspace(sdf["param"][1], sdf["param"][4], num=257)
            z = np.linspace(sdf["param"][2], sdf["param"][5], num=257)
            my_interpolating_function = RegularGridInterpolator((z, y, x), sdf["value"])

            num_points = 100000
            samples = np.random.random(size=num_points * 3).reshape(num_points, 3)
            samples = 2 * 0.49 * samples - 0.49
            sdf = 256 * np.hstack([samples, np.expand_dims(my_interpolating_function(samples), axis=1)])
            points = sdf[sdf[:, 3] <= 0][:, :3] + 128
        else:
            sdf = sdf["value"]
            print(sdf.shape, sdf.max(), sdf.min())
            points = np.argwhere(sdf <= 0)
            print(points.shape, points.min(), points.max())

        pcd = o3d.geometry.PointCloud()
        # ((1 / 256) >= sdf) & (sdf >= -(1 / 256))
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.3, 0.3, 0.5])
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        # grid = o3d.geometry.VoxelGrid().create_from_point_cloud(input=pcd, voxel_size=1)
        size = 0.5 * (pcd.get_max_bound() - pcd.get_min_bound()).max()

        points = [[0, 0, 0],
                  [0, 0, 256],
                  [0, 256, 256],
                  [256, 256, 256],
                  [256, 0, 0],
                  [256, 256, 0],
                  [0, 256, 0],
                  [256, 0, 256]]
        lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                 [4, 5], [5, 6], [6, 7], [4, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[0, 0, 0] for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=size)
        o3d.visualization.draw_geometries([pcd, frame, line_set])
    elif sdfgen:
        with h5py.File(sdfgen_path, "r") as f:
            grid = np.array(f["voxelgrid"][()])
            truncation_threshold = 1.0
            grid = (grid.astype(np.float64) / grid.max() - 0.5) * truncation_threshold
            print(grid.shape, grid.max(), grid.min())

            pcd = o3d.geometry.PointCloud()
            points = np.argwhere(grid <= 0)
            print(points.shape, points.min(), points.max())
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0.3, 0.3, 0.5])
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

            size = 0.5 * (pcd.get_max_bound() - pcd.get_min_bound()).max()

            box = pcd.get_axis_aligned_bounding_box()

            points = [[0, 0, 0],
                      [0, 0, 256],
                      [0, 256, 256],
                      [256, 256, 256],
                      [256, 0, 0],
                      [256, 256, 0],
                      [0, 256, 0],
                      [256, 0, 256]]
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                     [4, 5], [5, 6], [6, 7], [4, 7],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            colors = [[0, 0, 0] for _ in range(len(lines))]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=size)
            o3d.visualization.draw_geometries([pcd, frame, line_set])


def eval_agile():
    mesh_path = "/home/matthias/Data/Ubuntu/git/occupancy_networks/data/ShapeNet.build/02876657/2_watertight/1071fa4cddb2da2fc8724d5673a063a6.off"
    point_path = "/home/matthias/Data/Ubuntu/git/occupancy_networks/data/ShapeNet/02876657/1071fa4cddb2da2fc8724d5673a063a6/pointcloud.npz"

    data = np.load("/home/matthias/Data/Ubuntu/data/agile_justin/scene3/scene_data.npy", allow_pickle=True).item()
    # for val in np.unique(data['voxel_model']):
    #     print(val, np.argwhere(data['voxel_model'] == val).shape)

    # point_cloud = o3d.io.read_point_cloud("/home/matthias/Data/Ubuntu/data/agile_justin/scene3/scene_points.ply")
    # point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) * data['voxel_size'])

    pcd_3 = o3d.geometry.PointCloud()
    pcd_3.points = o3d.utility.Vector3dVector(np.argwhere(data['voxel_model'] == 3))
    pcd_3.paint_uniform_color([0.8, 0.0, 0.0])
    pcd_3.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    pcd_5 = o3d.geometry.PointCloud()
    pcd_5.points = o3d.utility.Vector3dVector(np.argwhere(data['voxel_model'] == 5))
    pcd_5.paint_uniform_color([0.0, 0.8, 0.0])
    pcd_5.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    pcd_6 = o3d.geometry.PointCloud()
    pcd_6.points = o3d.utility.Vector3dVector(np.argwhere(data['voxel_model'] == 6))
    pcd_6.paint_uniform_color([0.0, 0.0, 0.8])
    pcd_6.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    o3d.visualization.draw_geometries([pcd_3, pcd_5, pcd_6])

    voxel_grid_5 = o3d.geometry.VoxelGrid().create_from_point_cloud(input=pcd_5, voxel_size=1)
    voxel_grid_6 = o3d.geometry.VoxelGrid().create_from_point_cloud(input=pcd_6, voxel_size=1)
    o3d.visualization.draw_geometries([voxel_grid_5, voxel_grid_6])

    # start = time.time()
    # for _ in range(10):
    #     trimesh.load(mesh_path).sample(100000)
    # print(time.time() - start)
    #
    # start = time.time()
    # for _ in range(10):
    #     o3d.io.read_triangle_mesh(filename=mesh_path).sample_points_uniformly(number_of_points=100000)
    # print(time.time() - start)
    #
    # start = time.time()
    # for _ in range(10):
    #     np.load(point_path)
    # print(time.time() - start)

    # mesh.compute_triangle_normals()
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1 / 32)
    # o3d.visualization.draw_geometries([voxel_grid])


if __name__ == "__main__":
    eval_sdf()

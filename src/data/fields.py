import os
from typing import Union

import numpy as np
import open3d as o3d
import trimesh
from easy_o3d.utils import get_camera_parameters_from_blenderproc_bopwriter, convert_depth_image_to_point_cloud
from scipy.spatial.transform import Rotation

from src.common import coord2index, normalize_coord, look_at, get_rotation_from_point, sample_point_on_upper_hemisphere
from src.data.core import Field
from src.utils import binvox_rw


class IndexField(Field):
    """ Basic index field."""

    def load(self, model_path, idx, category):
        """ Loads the index field.

        Args:
            model_path (str): mesh_path to model
            idx (int): ID of data point
            category (int): index of category
        """
        return idx

    def check_complete(self, files):
        """ Check if field is complete.
        
        Args:
            files: files
        """
        return True


# 3D Fields
class PatchPointsField(Field):
    """ Patch Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape and then split to patches.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    """

    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files

    def load(self, model_path, idx, vol):
        """ Loads the data point.

        Args:
            model_path (str): mesh_path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        """
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        # acquire the crop
        ind_list = []
        for i in range(3):
            ind_list.append((points[:, i] >= vol['query_vol'][0][i])
                            & (points[:, i] <= vol['query_vol'][1][i]))
        ind = ind_list[0] & ind_list[1] & ind_list[2]
        data = {None: points[ind], 'occ': occupancies[ind]}

        if self.transform is not None:
            data = self.transform(data)

        # calculate normalized coordinate w.r.t. defined query volume
        p_n = {}
        for key in vol['plane_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(data[None].copy(), vol['input_vol'], plane=key)
        data['normalized'] = p_n

        return data


class PointsField(Field):
    """ Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    """

    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None, occ_from_sdf=False):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files
        self.occ_from_sdf = occ_from_sdf

    def load(self, model_path, idx, category):
        """ Loads the data point.

        Args:
            model_path (str): mesh_path to model
            idx (int): ID of data point
            category (int): index of category
        """
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_data = np.load(file_path)
        if isinstance(points_data, np.lib.npyio.NpzFile):
            points = points_data["points"]
        else:
            points = points_data[:, :3]
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        if isinstance(points_data, np.lib.npyio.NpzFile):
            occupancies = points_data["occupancies"]
            if self.unpackbits:
                occupancies = np.unpackbits(occupancies)[:points.shape[0]]
            occupancies = occupancies.astype(np.float32)
        elif self.occ_from_sdf:
            occupancies = (points_data[:, 3] <= 0).astype(np.float32)
        else:
            occupancies = points_data[:, 3]
            if occupancies.dtype == np.float16:
                occupancies = occupancies.astype(np.float32)
                occupancies += 1e-4 * np.random.randn(*occupancies.shape)

        data = {None: points,
                "occ": occupancies}

        if self.transform is not None:
            data = self.transform(data)

        return data


class VoxelsField(Field):
    """ Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    """

    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category, use_trimesh: bool = False):
        """ Loads the data point.

        Args:
            model_path (str): mesh_path to model
            idx (int): ID of data point
            category (int): index of category
            use_trimesh (bool): Whether to use Trimesh to load the binvox file
        """
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            if use_trimesh:
                voxels = trimesh.exchange.binvox.load_binvox(f)
            else:
                voxels = binvox_rw.read_as_3d_array(f)

        if not use_trimesh:
            voxels = voxels.data.astype(np.float32)
            if self.transform is not None:
                voxels = self.transform(voxels)
        return voxels

    def check_complete(self, files):
        complete = (self.file_name in files)
        return complete


class PatchPointCloudField(Field):
    """ Patch point cloud field.

    It provides the field used for patched point cloud data. These are the points
    randomly sampled on the mesh and then partitioned.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    """

    def __init__(self, file_name, transform=None, transform_add_noise=None, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files

    def load(self, model_path, idx, vol):
        """ Loads the data point.

        Args:
            model_path (str): mesh_path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        """
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        # add noise globally
        if self.transform is not None:
            data = {None: points,
                    'normals': normals}
            data = self.transform(data)
            points = data[None]

        # acquire the crop index
        ind_list = []
        for i in range(3):
            ind_list.append((points[:, i] >= vol['input_vol'][0][i])
                            & (points[:, i] <= vol['input_vol'][1][i]))
        mask = ind_list[0] & ind_list[1] & ind_list[2]  # points inside the input volume
        mask = ~mask  # True means outside the boundary!!
        data['mask'] = mask
        points[mask] = 0.0

        # calculate index of each point w.r.t. defined resolution
        index = {}

        for key in vol['plane_type']:
            index[key] = coord2index(points.copy(), vol['input_vol'], reso=vol['reso'], plane=key)
            if key == 'grid':
                index[key][:, mask] = vol['reso'] ** 3
            else:
                index[key][:, mask] = vol['reso'] ** 2
        data['ind'] = index

        return data

    def check_complete(self, files):
        complete = (self.file_name in files)
        return complete


class PointCloudField(Field):
    """ Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    """

    def __init__(self, file_name, transform=None, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files

    def load(self, model_path, idx, category):
        """ Loads the data point.

        Args:
            model_path (str): mesh_path to model
            idx (int): ID of data point
            category (int): index of category
        """
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        if isinstance(pointcloud_dict, np.lib.npyio.NpzFile):
            points = pointcloud_dict["points"].astype(np.float32)
            normals = pointcloud_dict["normals"].astype(np.float32)
        else:
            points = pointcloud_dict.astype(np.float32)
            normals = None

        data = {None: points}
        if normals is not None:
            data["normals"] = normals

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        complete = (self.file_name in files)
        return complete


class PartialPointCloudField(Field):
    """ Partial Point cloud field.

    It provides the field used for partial point cloud data. These are the points
    randomly sampled on the mesh and a bounding box with random size is applied.

    Args:
        file_name (str): file name
        transform (torch.): list of transformations applied to data points
        multi_files (callable): number of files
        part_ratio (float): max ratio for the remaining part
    """

    def __init__(self, file_name, transform=None, multi_files=None, part_ratio=0.7, plane: bool = False):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files
        self.part_ratio = part_ratio
        self.plane = plane

    def load(self, model_path, idx, category):
        """ Loads the data point.

        Args:
            model_path (str): mesh_path to model
            idx (int): ID of data point
            category (int): index of category
        """
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        if isinstance(pointcloud_dict, np.lib.npyio.NpzFile):
            points = pointcloud_dict['points'].astype(np.float32)
            normals = pointcloud_dict['normals'].astype(np.float32)
        else:
            points = pointcloud_dict.astype(np.float32)
            normals = None

        if self.plane:
            axes = np.random.choice([0, 1, 2], size=np.random.randint(1, 3))
            if len(axes) == 1:
                pass
            indices = np.arange(len(points))  # Todo: Fix
        else:
            side = np.random.randint(3)
            xb = [points[:, side].min(), points[:, side].max()]
            length = np.random.uniform(self.part_ratio * (xb[1] - xb[0]), (xb[1] - xb[0]))
            indices = (points[:, side] - xb[0]) <= length

        data = {None: points[indices]}
        if normals is not None:
            data["normals"] = normals[indices]

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        complete = (self.file_name in files)
        return complete


class BlenderProcDepthPointCloudField(Field):
    def __init__(self,
                 transform=None,
                 unscale: bool = True,
                 path_prefix: str = ""):
        self.transform = transform
        self.unscale = unscale
        self.path_prefix = path_prefix

    def load(self, model_path, idx, category):
        if self.path_prefix:
            synthset = model_path.split('/')[-2]
            model = model_path.split('/')[-1]
            model_path = os.path.join(self.path_prefix, synthset, model)
        path_to_camera_json = os.path.join(model_path, "camera.json")

        shard = np.random.randint(10)
        file = np.random.randint(100)

        chunk_path = os.path.join(model_path, "train_pbr", str(shard).zfill(6))
        path_to_scene_camera_json = os.path.join(chunk_path, "scene_camera.json")

        camera_parameters = get_camera_parameters_from_blenderproc_bopwriter(path_to_scene_camera_json,
                                                                             path_to_camera_json,
                                                                             scene_id=file)[0]
        pcd = convert_depth_image_to_point_cloud(os.path.join(chunk_path, "depth", str(file).zfill(6) + ".png"),
                                                 camera_intrinsic=camera_parameters.intrinsic,
                                                 camera_extrinsic=camera_parameters.extrinsic,
                                                 depth_trunc=10)
        points = np.asarray(pcd.points)
        scale = np.load(os.path.join(model_path, "train_pbr", "scales.npy"))[shard]
        cam = camera_parameters.extrinsic

        swap_xy = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
        swap_yz = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]])
        rot = cam[:3, :3]  # Blender world to camera rotation (rotates points from world to camera frame)
        trans = cam[:3, 3]  # Camera location in camera coordinates

        points = (swap_yz @ points.T).T  # Change Blender to ShapeNet coordinate frame

        camera = -rot.T @ trans  # Camera location in Blender world coordinates
        camera = swap_yz @ camera  # Camera location in ShapeNet coordinates

        rot = swap_xy @ rot @ swap_yz.T  # Shapenet to camera rotation (rotates points from ShapeNet to camera frame)
        x_angle = np.arctan2(rot.T[2, 1], rot.T[2, 2])
        x_angle = -np.rad2deg(x_angle - np.pi if x_angle > 0 else x_angle)  # Todo: why?

        if self.unscale:
            points /= scale

        data = {None: points.astype(np.float32),
                "rot": rot,
                "cam": camera,
                "x_angle": x_angle}

        if self.transform is not None:
            data = self.transform(data)

        return data


class PyrenderDepthPointCloudField(Field):
    def __init__(self,
                 file_name: str,
                 size: int = 224,
                 num_points: int = 3000,
                 upper_hemisphere: bool = False,
                 transform: Union[None, object] = None):
        self.file_name = file_name
        self.num_points = num_points
        self.upper_hemisphere = upper_hemisphere
        self.transform = transform

        res_x, res_y = size, size
        self.cam_intr = np.array([res_x, res_y, res_x / 2, res_y / 2], dtype=float)
        self.img_size = np.array([res_x, res_y], dtype=np.int32)
        self.znf = np.array([1 - 0.75, 1 + 0.75], dtype=float)

        Y, X = np.mgrid[0:res_y, 0:res_x]
        coordinates = np.vstack([X.ravel(), Y.ravel(), np.ones(res_x * res_y)])
        inv_camk = np.linalg.inv([[res_x, 0, res_x / 2], [0, res_y, res_y / 2], [0, 0, 1]])
        self.projection = (inv_camk @ coordinates).T

        import logging
        logger = logging.getLogger("trimesh")
        logger.setLevel(logging.ERROR)

    def load(self, model_path, idx, category):
        file_path = os.path.join(model_path, self.file_name)

        # Load & Normalize
        mesh = trimesh.load(file_path, process=False)
        scale = (mesh.bounds[1] - mesh.bounds[0]).max()
        loc = (mesh.bounds[1] + mesh.bounds[0]) / 2
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        # Transform
        if self.upper_hemisphere:
            point = sample_point_on_upper_hemisphere(direction=(0, 1, 0))
            R = get_rotation_from_point(point)  # Todo: Not working as expected
        else:
            R = Rotation.random().as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [0, 0, 1]
        mesh.apply_transform(T)

        # Render
        vertices = mesh.vertices.T.astype(np.float64).copy()
        faces = (mesh.faces + 1).T.astype(np.float64).copy()  # Todo: Why +1?

        depth, mask, img = pyrender.render(vertices,
                                           faces,
                                           self.cam_intr,
                                           self.znf,
                                           self.img_size)

        # Reproject
        points = depth.repeat(3).reshape(-1, 3) * self.projection
        mask = (depth.ravel() > self.znf[0]) & (depth.ravel() < self.znf[1])

        # Not enough points
        if np.sum(mask) < self.num_points:
            points = mesh.sample(self.num_points).astype(np.float32)
        else:
            points = points[mask]
            points[:, 2] -= 1
            points = points @ R

        data = {None: points.astype(np.float32)}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        complete = (self.file_name in files)
        return complete


class DepthLikePointCloudField(Field):
    def __init__(self,
                 file_name: str,
                 upper_hemisphere: bool = False,
                 rotate_object: str = '',
                 sample_camera_position: str = '',
                 transform: Union[None, object] = None):
        assert not (rotate_object and sample_camera_position)
        self.file_name = file_name
        self.upper_hemisphere = upper_hemisphere
        self.rotate_object = rotate_object
        self.sample_camera_position = sample_camera_position
        self.transform = transform

    def load(self, model_path, idx, category):
        file_path = os.path.join(model_path, self.file_name)

        if file_path.endswith(".npy") or file_path.endswith(".npz"):
            pointcloud_dict = np.load(file_path)
            if isinstance(pointcloud_dict, np.lib.npyio.NpzFile):
                points = pointcloud_dict['points']
            else:
                points = pointcloud_dict
            indices = np.random.randint(len(points), size=50000)
            points = points[indices]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
        else:
            mesh = o3d.io.read_triangle_mesh(file_path, enable_post_processing=False)
            scale = (mesh.get_max_bound() - mesh.get_min_bound()).max()
            loc = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
            mesh.translate(-loc)
            mesh.scale(1 / scale, center=(0, 0, 0))
            pcd = mesh.sample_points_uniformly(50000)

        rot = np.eye(3)
        x_angle = 0
        if self.rotate_object:
            angles = np.random.uniform(360, size=len(self.rotate_object) if len(self.rotate_object) > 1 else None)
            if self.upper_hemisphere and 'x' in self.rotate_object:
                x_angle = np.random.uniform(0, 80)
                angles[list(self.rotate_object).index('x')] = x_angle
            rot = Rotation.from_euler(self.rotate_object, angles, degrees=True).as_matrix()
            trafo = np.eye(4)
            trafo[:3, :3] = rot
            pcd.transform(trafo)

        camera = [0, 0, 1]
        if 'x' in self.sample_camera_position:
            camera[0] = np.random.uniform(low=-1, high=1)
        if 'y' in self.sample_camera_position:
            camera[1] = np.random.uniform(low=0 if self.upper_hemisphere else -1,
                                          high=0.9 if self.upper_hemisphere else 1)
        if 'z' in self.sample_camera_position:
            camera[2] = np.random.uniform(low=-1, high=1)

        camera /= np.linalg.norm(camera)
        _, indices = pcd.hidden_point_removal(camera, 100)

        pcd = pcd.select_by_index(indices)
        points = np.asarray(pcd.points, dtype=np.float32)

        if self.sample_camera_position:
            rot = look_at(camera)[:3, :3]
            x_angle = np.arctan2(rot[2, 1], rot[2, 2])
            x_angle = -np.rad2deg(x_angle - np.pi if x_angle > 0 else x_angle)
            rot = rot.T
        elif self.rotate_object:
            points = (rot.T @ points.T).T

        data = {None: points.astype(np.float32),
                "rot": rot,
                "cam": camera,
                "x_angle": x_angle}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        complete = (self.file_name in files)
        return complete

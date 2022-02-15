from typing import Tuple, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import open3d as o3d


class Rotate(object):
    """ Data rotation transformation class.

    It (randomly) rotates the point cloud.
    """
    def __init__(self,
                 to_cam_frame: bool = False,
                 to_world_frame: bool = False,
                 axis: str = "",
                 visualize: bool = False):
        assert not (to_cam_frame and to_world_frame)
        print(f"Rotating data (to world: {to_world_frame}, to cam: {to_cam_frame})")
        self.to_cam_frame = to_cam_frame
        self.to_world_frame = to_world_frame
        self.axis = axis
        self.visualize = visualize

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        if data.get('voxels') is not None or data.get('input_type') == 'voxels':
            raise TypeError("Voxels in data which cannot be rotated.")

        data_out = data.copy()

        points = data.get('points')
        points_iou = data.get('points_iou')

        inputs = data.get('inputs')
        normals = data.get('inputs.normals')

        pcd = data.get('pointcloud')
        pcd_normals = data.get('pointcloud.normals')

        pcd_chamfer = data.get('pointcloud_chamfer')
        pcd_chamfer_normals = data.get('pointcloud_chamfer.normals')

        if self.to_cam_frame:
            rot = data.get('inputs.rot')
        elif self.to_world_frame:
            rot = data.get('inputs.rot')
            x_angle = data.get('inputs.x_angle')
            rot_x = R.from_euler('x', x_angle, degrees=True).as_matrix()
            rot = rot_x.T @ rot
        elif self.axis:
            rot = R.from_euler(self.axis, np.random.uniform(360), degrees=True).as_matrix()
        else:
            rot = R.random().as_matrix()

        for k, v in zip(['points',
                         'points_iou',
                         'pointcloud',
                         'pointcloud.normals',
                         'pointcloud_chamfer',
                         'pointcloud_chamfer.normals',
                         'inputs',
                         'inputs.normals'],
                        [points,
                         points_iou,
                         pcd,
                         pcd_normals,
                         pcd_chamfer,
                         pcd_chamfer_normals,
                         inputs,
                         normals]):
            if v is not None:
                data_out[k] = (rot @ v.T).T.astype(np.float32)

                if self.visualize and k in ["points", "points_iou", "inputs", "pointcloud"]:
                    trimesh.PointCloud(data_out[k]).show()
        return data_out


# Transforms
class PointcloudNoise(object):
    """ Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (float): standard deviation
    """

    def __init__(self, stddev: float):
        if stddev > 0:
            print(f"Adding noise to pointcloud (STD={stddev})")
        self.stddev = stddev

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        if self.stddev <= 0:
            return data
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    """ Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        if N:
            print(f"Subsampling pointcloud (N={N})")
        self.N = N

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dict): data dictionary
        """
        if not self.N:
            return data
        data_out = data.copy()
        points = data[None]
        normals = data.get("normals")

        if points.shape[0] < self.N:
            indices = np.random.choice(points.shape[0], size=self.N)
        else:
            indices = np.random.randint(points.shape[0], size=self.N)

        data_out[None] = points[indices]
        if normals is not None:
            data_out['normals'] = normals[indices]

        return data_out


class VoxelizeInputs(object):
    def __init__(self, voxel_size: Union[float, Tuple[float, float]] = 0.002):
        print(f"Voxelizing inputs (voxel_size={voxel_size})")
        self.voxel_size = voxel_size

    def __call__(self, data):
        data_out = data.copy()
        points = data.get("inputs")
        normals = data.get("inputs.normals")
        scale = data.get("inputs.scale")

        voxel_size = self.voxel_size
        if isinstance(self.voxel_size, (tuple, list)):
            voxel_size = np.random.uniform(self.voxel_size[0], self.voxel_size[1])

        if scale is not None:
            points *= scale

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd = pcd.voxel_down_sample(voxel_size)
        points = np.asarray(pcd.points, dtype=np.float32)

        if scale is not None:
            points /= scale

        data_out["inputs"] = points
        if normals is not None:
            data_out["inputs.normals"] = np.asarray(pcd.normals, dtype=np.float32)

        return data_out


class VoxelizePointcloud(object):
    def __init__(self, voxel_size: Union[float, Tuple[float, float]] = 0.002):
        print(f"Voxelizing pointcloud (voxel_size={voxel_size})")
        self.voxel_size = voxel_size

    def __call__(self, data):
        data_out = data.copy()
        points = data[None]
        normals = data.get("normals")
        scale = data.get("scale")

        voxel_size = self.voxel_size
        if isinstance(self.voxel_size, (tuple, list)):
            voxel_size = np.random.uniform(self.voxel_size[0], self.voxel_size[1])

        if scale is not None:
            points *= scale

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd = pcd.voxel_down_sample(voxel_size)
        points = np.asarray(pcd.points, dtype=np.float32)

        if scale is not None:
            points /= scale

        data_out[None] = points
        if normals is not None:
            data_out['normals'] = np.asarray(pcd.normals, dtype=np.float32)

        return data_out


class Normalize(object):
    def __init__(self, center: str = "xyz", scale: bool = True, to_min_val: str = "", to_max_val: str = ""):
        print(f"Normalizing data (centering: {center}, scaling: {scale})")
        self.center = center
        self.scale = scale
        self.to_min_val = to_min_val
        self.to_max_val = to_max_val

    def __call__(self, data):
        data_out = data.copy()

        points = data.get("points")
        points_iou = data.get("points_iou")
        pointcloud = data.get("pointcloud")
        pointcloud_chamfer = data.get("pointcloud_chamfer")
        inputs = data.get("inputs")

        scale = (inputs.max(axis=0) - inputs.min(axis=0)).max()
        loc = (inputs.max(axis=0) + inputs.min(axis=0)) / 2
        min_vals = inputs.min(axis=0)
        max_vals = inputs.max(axis=0)
        loc_x = loc_y = loc_z = 0

        if self.center:
            loc_x = loc[0] if 'x' in self.center else loc_x
            loc_y = loc[1] if 'y' in self.center else loc_y
            loc_z = loc[2] if 'z' in self.center else loc_z

        if self.to_min_val:
            loc_x = min_vals[0] if 'x' in self.to_min_val else loc_x
            loc_y = min_vals[1] if 'y' in self.to_min_val else loc_y
            loc_z = min_vals[2] if 'z' in self.to_min_val else loc_z

        if self.to_max_val:
            loc_x = max_vals[0] if 'x' in self.to_max_val else loc_x
            loc_y = max_vals[1] if 'y' in self.to_max_val else loc_y
            loc_z = max_vals[2] if 'z' in self.to_max_val else loc_z

        loc = np.array([loc_x, loc_y, loc_z])

        for k, v in zip(["points", "points_iou", "pointcloud", "pointcloud_chamfer", "inputs"],
                        [points, points_iou, pointcloud, pointcloud_chamfer, inputs]):
            if v is not None:
                v = (v - loc).astype(np.float32)
                if self.scale:
                    v = (v / scale).astype(np.float32)
                data_out[k] = v

        return data_out


class Scale(object):
    def __init__(self,
                 axes: str = "xyz",
                 amount: Optional[Union[float, Tuple[float, float], Tuple[float, float, float], None]] = None,
                 random: Optional[bool] = False,
                 from_input: Optional[bool] = False):
        if from_input:
            print("Scaling data from input")
        else:
            print(f"Scaling {axes} by min/max (+-){amount}")
        self.axes = axes
        self.amount = amount
        self.random = random
        self.from_input = from_input

    def __call__(self, data):
        if not self.axes:
            print("Warning: Didn't provide any axes. Returning.")
            return data

        data_out = data.copy()

        points = data.get('points')
        points_iou = data.get('points_iou')

        inputs = data.get('inputs')
        normals = data.get('inputs.normals')

        pcd = data.get('pointcloud')
        pcd_normals = data.get('pointcloud.normals')

        pcd_chamfer = data.get('pointcloud_chamfer')
        pcd_chamfer_normals = data.get('pointcloud_chamfer.normals')

        if self.from_input:
            scale = data.get("scale")
            if scale is None:
                scale = data.get("inputs.scale")
            if scale is None:
                print("Warning: Didn't find 'scale' in data. Returning.")
                return data
            else:
                if isinstance(scale, float):
                    scale_x = scale if 'x' in self.axes else 1
                    scale_y = scale if 'y' in self.axes else 1
                    scale_z = scale if 'z' in self.axes else 1
                elif isinstance(scale, (tuple, list, np.ndarray)) and len(scale) == 3 and len(self.axes) == 3:
                    scale_x = scale[0] / scale[1]
                    scale_y = 1
                    scale_z = scale[2] / scale[1]
                else:
                    raise ValueError
        elif self.amount is not None:
            if isinstance(self.amount, (float, int)):
                if self.random:
                    scale_x = np.random.uniform(1 - self.amount, 1 + self.amount) if 'x' in self.axes else 1
                    scale_y = np.random.uniform(1 - self.amount, 1 + self.amount) if 'y' in self.axes else 1
                    scale_z = np.random.uniform(1 - self.amount, 1 + self.amount) if 'z' in self.axes else 1
                else:
                    scale_x = self.amount if 'x' in self.axes else 1
                    scale_y = self.amount if 'y' in self.axes else 1
                    scale_z = self.amount if 'z' in self.axes else 1
            elif isinstance(self.amount, (tuple, list)):
                if len(self.amount) == 2:
                    if self.random:
                        scale_x = np.random.uniform(1 - self.amount[0], 1 + self.amount[1]) if 'x' in self.axes else 1
                        scale_y = np.random.uniform(1 - self.amount[0], 1 + self.amount[1]) if 'y' in self.axes else 1
                        scale_z = np.random.uniform(1 - self.amount[0], 1 + self.amount[1]) if 'z' in self.axes else 1
                    else:
                        print(f"Warning: Scaling {self.axes} with {self.amount} is ambiguous. Returning.")
                        return data
                elif len(self.amount) == 3 and len(self.axes) == 3:
                    if self.random:
                        scale_x = np.random.uniform(1 - self.amount[0], 1 + self.amount[0])
                        scale_y = np.random.uniform(1 - self.amount[1], 1 + self.amount[1])
                        scale_z = np.random.uniform(1 - self.amount[2], 1 + self.amount[2])
                    else:
                        scale_x, scale_y, scale_z = self.amount
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            print("Warning: No scaling amount provided and not taken from input. Returning.")
            return data

        scale = np.array([scale_x, scale_y, scale_z])

        for k, v in zip(['points',
                         'points_iou',
                         'pointcloud',
                         'pointcloud.normals',
                         'pointcloud_chamfer',
                         'pointcloud_chamfer.normals',
                         'inputs',
                         'inputs.normals'],
                        [points,
                         points_iou,
                         pcd,
                         pcd_normals,
                         pcd_chamfer,
                         pcd_chamfer_normals,
                         inputs,
                         normals]):
            if v is not None:
                data_out[k] = (v * scale).astype(np.float32)

        return data_out


class SubsamplePoints(object):
    """ Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        print(f"Subsampling points (N={N})")
        self.N = N

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        points = data[None]
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({None: points[idx, :], 'occ': occ[idx]})
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({None: points,
                             'occ': occ,
                             'volume': volume})
        return data_out

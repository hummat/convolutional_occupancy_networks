from typing import Tuple, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh


class Rotate(object):
    """ Data rotation transformation class.

    It (randomly) rotates the point cloud.
    """
    def __init__(self,
                 to_cam_frame: bool = False,
                 to_world_frame: bool = False,
                 visualize: bool = False):
        assert not (to_cam_frame and to_world_frame)
        print("Rotating data")
        self.to_cam_frame = to_cam_frame
        self.to_world_frame = to_world_frame
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

        if self.to_cam_frame:
            rot = data.get('inputs.rot')
        elif self.to_world_frame:
            rot = data.get('inputs.rot')
            x_angle = data.get('inputs.x_angle')
            rot_x = R.from_euler('x', x_angle, degrees=True).as_matrix()
            rot = rot_x.T @ rot
        else:
            rot = R.random().as_matrix()

        for k, v in zip(['points', 'points_iou', 'inputs', 'inputs.normals', 'pointcloud', 'pointcloud.normals'],
                        [points, points_iou, inputs, normals, pcd, pcd_normals]):
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
        print(f"Adding noise to pointcloud (STD={stddev})")
        self.stddev = stddev

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
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
        print(f"Subsampling pointcloud (N={N})")
        self.N = N

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dict): data dictionary
        """
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        if points.shape[0] < self.N:
            indices = np.random.choice(points.shape[0], size=self.N)
        else:
            indices = np.random.randint(points.shape[0], size=self.N)

        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]

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
        pointcloud = data.get("pointcloud")
        inputs = data.get("inputs")

        scale = (inputs.max(axis=0) - inputs.min(axis=0)).max()
        _loc = (inputs.max(axis=0) + inputs.min(axis=0)) / 2
        min_vals = inputs.min(axis=0)
        max_vals = inputs.max(axis=0)

        loc = np.zeros(3)
        if 'x' in self.center:
            loc[0] = _loc[0]
        if 'y' in self.center:
            loc[1] = _loc[1]
        if 'z' in self.center:
            loc[2] = _loc[2]

        if 'x' in self.to_min_val:
            loc[0] = min_vals[0]
        if 'y' in self.to_min_val:
            loc[1] = min_vals[1]
        if 'z' in self.to_min_val:
            loc[2] = min_vals[2]

        if 'x' in self.to_max_val:
            loc[0] = max_vals[0]
        if 'y' in self.to_max_val:
            loc[1] = max_vals[1]
        if 'z' in self.to_max_val:
            loc[2] = max_vals[2]

        for k, v in zip(["points", "pointcloud", "inputs"],
                        [points, pointcloud, inputs]):
            if v is not None:
                data_out[k] = (v - loc).astype(np.float32)
                if self.scale:
                    data_out[k] = (v / scale).astype(np.float32)

        return data_out


class Scale(object):
    def __init__(self,
                 scale_range: Optional[Tuple[float, float]] = None,
                 add_points: bool = False,
                 box_size: float = 1.1):
        if scale_range:
            print(f"Randomly scaling data (range={scale_range})")
        else:
            print(f"Scaling data")
        self.scale_range = scale_range
        self.add_points = add_points
        self.box_size = box_size

    def __call__(self, data):
        data_out = data.copy()

        points = data.get("points")
        pointcloud = data.get("pointcloud")
        inputs = data.get("inputs")
        scale = data.get("scale")

        if scale is None and self.scale_range:
            scale = np.random.uniform(*self.scale_range)

        for k, v in zip(["points", "pointcloud", "inputs"],
                        [points, pointcloud, inputs]):
            if v is not None:
                data_out[k] = (v * scale).astype(np.float32)

        if scale < 1 and self.add_points:
            additional_points = np.random.rand(len(points), 3)
            additional_points = self.box_size * (additional_points - 0.5)
            additional_points = additional_points[(additional_points[:, 0] > scale / 2) | (additional_points[:, 0] < -scale / 2)]
            additional_points = additional_points[(additional_points[:, 1] > scale / 2) | (additional_points[:, 1] < -scale / 2)]
            additional_points = additional_points[(additional_points[:, 2] > scale / 2) | (additional_points[:, 2] < -scale / 2)]
            data_out["points"] = np.concatenate([points, additional_points], axis=0).astype(np.float32)
            data_out["points.occ"] = np.concatenate([data.get("points.occ"),
                                                     np.zeros(len(additional_points))]).astype(np.float32)
            
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

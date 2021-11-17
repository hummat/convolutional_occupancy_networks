import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh


class Rotate(object):
    """ Data rotation transformation class.

    It randomly rotates the point cloud.
    """
    def __init__(self, random: bool = False, visualize: bool = False):
        self.random = random
        self.visualize = visualize

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        data_out = data.copy()

        points = data.get('points')
        points_iou = data.get('points_iou')

        inputs = data.get('inputs')
        normals = data.get('inputs.normals')

        pcd = data.get('pointcloud')
        pcd_normals = data.get('pointcloud.normals')

        voxels = data.get('voxels')
        voxel_input = data.get('input_type') == 'voxels'

        if self.random:
            rot = R.random().as_matrix()
        else:
            rot = np.random.permutation(np.eye(3)) * np.array([np.random.choice([-1, 1]),
                                                               np.random.choice([-1, 1]),
                                                               np.random.choice([-1, 1])])
        data_out['rot'] = rot
        for k, v in zip(['points', 'points_iou', 'inputs', 'inputs.normals', 'pointcloud', 'pointcloud.normals', 'voxels'],
                        [points, points_iou, inputs, normals, pcd, pcd_normals, voxels]):
            if v is not None:
                if k == 'voxels' or (k == 'inputs' and voxel_input):
                    trans = np.eye(4)
                    trans[:3, :3] = rot
                    if isinstance(v, np.ndarray):
                        v = trimesh.voxel.VoxelGrid(trimesh.voxel.encoding.DenseEncoding(v.data))
                    data_out[k] = v.apply_transform(trans).matrix.astype(np.float32)
                data_out[k] = v @ rot

                if self.visualize and k in ["points", "points_iou", "inputs", "pointcloud", "voxels"]:
                    if k == 'voxels' or (k == 'inputs' and voxel_input):
                        trimesh.voxel.VoxelGrid(trimesh.voxel.encoding.DenseEncoding(data_out[k])).show()
                    else:
                        trimesh.PointCloud(data_out[k]).show()
        return data_out


# Transforms
class PointcloudNoise(object):
    """ Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    """

    def __init__(self, stddev):
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
        self.N = N

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dict): data dictionary
        """
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]

        return data_out


class SubsamplePoints(object):
    """ Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
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
            data_out.update({
                None: points[idx, :],
                'occ': occ[idx],
            })
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

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out

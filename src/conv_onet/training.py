import os

import torch
from torch.nn import functional as F

from src.common import compute_iou, make_3d_grid, add_key
from src.training import BaseTrainer


class Trainer(BaseTrainer):
    """ Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    """

    def __init__(self,
                 model,
                 optimizer,
                 device=None,
                 input_type='pointcloud',
                 vis_dir=None,
                 threshold=0.5,
                 eval_sample=False,
                 loss=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.loss = loss

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        """ Performs a training step.

        Args:
            data (dict): data dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, data):
        """ Performs an evaluation step.

        Args:
            data (dict): data dictionary
        """
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device, non_blocking=True)
        # occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device, non_blocking=True)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device, non_blocking=True)
        occ_iou = data.get('points_iou.occ').to(device, non_blocking=True)

        batch_size = points.size(0)

        kwargs = {}

        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        # points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs, sample=self.eval_sample, **kwargs)

        if self.model.decoder.kwargs.get("sdf"):
            loss = F.l1_loss(p_out, occ_iou, reduction='none').sum(-1).mean()  # |x_n - y_n|
            eval_dict['loss'] = loss.cpu().numpy().item()
            if isinstance(self.loss, str):
                if self.loss.lower() in ["l2", "squared_l2", "mse", "mean_squared_error"]:
                    loss = F.mse_loss(p_out, occ_iou, reduction='none').sum(-1).mean()  # (x_n - y_n)^2
                    eval_dict['l2'] = loss.cpu().numpy().item()
                elif self.loss.lower() in ["smooth_l1", "huber"]:
                    loss = F.smooth_l1_loss(p_out, occ_iou, reduction='none').sum(-1).mean()
                    eval_dict['smooth_l1'] = loss.cpu().numpy().item()
                elif self.loss.lower() in ["disn"]:
                    m1 = 4
                    m2 = 1
                    delta = 0.01
                    loss1 = (m1 * F.l1_loss(p_out[occ_iou < delta], occ_iou[occ_iou < delta], reduction='none')).sum(-1).mean()
                    loss2 = (m2 * F.l1_loss(p_out[occ_iou >= delta], occ_iou[occ_iou >= delta], reduction='none')).sum(-1).mean()
                    loss = loss1 + loss2
                    eval_dict['disn'] = loss.cpu().numpy().item()
                else:
                    raise NotImplementedError

            occ_iou_np = (occ_iou <= 0.0).cpu().numpy()
            occ_iou_hat_np = (p_out <= threshold).cpu().numpy()
        else:
            if self.loss is None or self.loss.lower() in ["bce",
                                                          "ce",
                                                          "cross_entropy",
                                                          "binary_cross_entropy",
                                                          "cross entropy",
                                                          "binary cross entropy"]:
                occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
                occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
                loss = F.binary_cross_entropy_with_logits(p_out.logits, occ_iou, reduction='none').sum(-1).mean()
            else:
                raise NotImplementedError
            eval_dict['loss'] = loss.cpu().numpy().item()

        iou = compute_iou(occ_iou_np, occ_iou_hat_np)
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device, non_blocking=True)
            #if self.model.decoder.kwargs.get("sdf"):
            #    points_voxels = make_3d_grid((-0.5,) * 3, (0.5,) * 3, voxels_occ.shape[1:])
            #else:
            points_voxels = make_3d_grid((-0.5 + 1 / 64,) * 3, (0.5 - 1 / 64,) * 3, voxels_occ.shape[1:])  # Todo: ?
            points_voxels = points_voxels.expand(batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device, non_blocking=True)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs, sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            if self.model.decoder.kwargs.get("sdf"):
                occ_hat_np = (p_out <= threshold).cpu().numpy()
            else:
                occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np)

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def compute_loss(self, data):
        """ Computes the loss.

        Args:
            data (dict): data dictionary
        """
        device = self.device
        points = data.get("points").to(device, non_blocking=True)
        occ = data.get("points.occ").to(device, non_blocking=True)
        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device, non_blocking=True)
        if self.model.decoder.kwargs.get("sdf"):
            sdf_values = self.model(points, inputs)
            if self.loss is None or self.loss in ["l1", "l1_loss", "l1 loss"]:
                loss = F.l1_loss(sdf_values, occ, reduction='none').sum(-1).mean()
            elif self.loss.lower() in ["l2", "squared_l2", "mse", "mean_squared_error"]:
                loss = F.mse_loss(sdf_values, occ, reduction='none').sum(-1).mean()
            elif self.loss.lower() in ["smooth_l1", "huber"]:
                loss = F.smooth_l1_loss(sdf_values, occ, reduction='none').sum(-1).mean()
            elif self.loss.lower() in ["disn"]:
                m1 = 4
                m2 = 1
                delta = 0.01
                loss1 = (m1 * F.l1_loss(sdf_values[occ < delta], occ[occ < delta], reduction='none')).sum(-1).mean()
                loss2 = (m2 * F.l1_loss(sdf_values[occ >= delta], occ[occ >= delta], reduction='none')).sum(-1).mean()
                loss = loss1 + loss2
            else:
                raise NotImplementedError
        else:
            if 'pointcloud_crop' in data.keys():
                # add pre-computed index
                inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
                inputs['mask'] = data.get('inputs.mask').to(device, non_blocking=True)
                # add pre-computed normalized coordinates
                points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)

            c = self.model.encode_inputs(inputs)

            # General points
            logits = self.model.decode(points, c).logits
            if self.loss is None or self.loss in ["bce",
                                                  "ce",
                                                  "cross_entropy",
                                                  "binary_cross_entropy",
                                                  "cross entropy",
                                                  "binary cross entropy"]:
                loss = F.binary_cross_entropy_with_logits(logits, occ, reduction='none').sum(-1).mean()
            else:
                raise NotImplementedError

        return loss

import os

from torch import nn

from src import data
from src.common import decide_total_volume_range, update_reso
from src.conv_onet import generation
from src.conv_onet import models, training
from src.encoder import encoder_dict


def get_model(cfg, dataset=None, device=None):
    """ Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    """
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']

    # for pointcloud_crop
    try:
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except:
        pass
    # local positional encoding
    if 'local_coord' in cfg['model'].keys():
        encoder_kwargs['local_coord'] = cfg['model']['local_coord']
        decoder_kwargs['local_coord'] = cfg['model']['local_coord']
    if 'pos_encoding' in cfg['model']:
        encoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    # update the feature volume/plane resolution
    if cfg['data']['input_type'] == 'pointcloud_crop':
        fea_type = cfg['model']['encoder_kwargs']['plane_type']
        if (dataset.split == 'train') or (cfg['generation']['sliding_window']):
            recep_field = 2 ** (cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
            reso = cfg['data']['query_vol_size'] + recep_field - 1
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = update_reso(reso, dataset.depth)
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = update_reso(reso, dataset.d**kwargsepth)
        # if dataset.split == 'val': #TODO run validation in room level during training
        else:
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = dataset.total_reso
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = dataset.total_reso

    decoder = models.decoder_dict[decoder](dim=dim, c_dim=c_dim, padding=padding, **decoder_kwargs)

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](dim=dim, c_dim=c_dim, padding=padding, **encoder_kwargs)
    else:
        encoder = None

    model = models.ConvolutionalOccupancyNetwork(decoder, encoder, device=device)

    return model


def get_trainer(model, optimizer, cfg, device):
    """ Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    """
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    loss = cfg['training']['loss']
    assert loss is None or isinstance(loss, str)

    trainer = training.Trainer(
        model,
        optimizer,
        device=device,
        input_type=input_type,
        vis_dir=vis_dir,
        threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        loss=loss)

    return trainer


def get_generator(model, cfg, device):
    """ Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    """

    if cfg['data']['input_type'] == 'pointcloud_crop':
        # calculate the volume boundary
        query_vol_metric = cfg['data']['padding'] + 1
        unit_size = cfg['data']['unit_size']
        recep_field = 2 ** (cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
        if 'unet' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet_kwargs']['depth']
        elif 'unet3d' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels']

        vol_info = decide_total_volume_range(query_vol_metric, recep_field, unit_size, depth)

        grid_reso = cfg['data']['query_vol_size'] + recep_field - 1
        grid_reso = update_reso(grid_reso, depth)
        query_vol_size = cfg['data']['query_vol_size'] * unit_size
        input_vol_size = grid_reso * unit_size
        # only for the sliding window case
        vol_bound = None
        if cfg['generation']['sliding_window']:
            vol_bound = {'query_crop_size': query_vol_size,
                         'input_crop_size': input_vol_size,
                         'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                         'reso': grid_reso}

    else:
        vol_bound = None
        vol_info = None

    generator = generation.Generator3D(
        model,
        points_batch_size=cfg['generation']['batch_size'],
        threshold=cfg['test']['threshold'],
        refinement_step=cfg['generation']['refinement_step'],
        device=device,
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        with_normals=cfg['generation']['normals'],
        padding=cfg['data']['padding'],
        sample=cfg['generation']['use_sampling'],
        input_type=cfg['data']['input_type'],
        vol_info=vol_info,
        vol_bound=vol_bound,
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        use_skimage=cfg['generation']['use_skimage'])
    return generator


def get_data_fields(mode, cfg):
    """ Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    """
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])

    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        if input_type != 'pointcloud_crop':
            fields['points'] = data.PointsField(cfg['data']['points_file'],
                                                points_transform,
                                                unpackbits=cfg['data']['points_unpackbits'],
                                                multi_files=cfg['data']['multi_files'],
                                                occ_from_sdf=cfg['data']['occ_from_sdf'])
        else:
            fields['points'] = data.PatchPointsField(cfg['data']['points_file'],
                                                     transform=points_transform,
                                                     unpackbits=cfg['data']['points_unpackbits'],
                                                     multi_files=cfg['data']['multi_files'])

    if mode in ('val', 'test'):
        pointcloud_file = cfg['data']['pointcloud_file']
        pointcloud_chamfer_file = cfg['data']['pointcloud_chamfer_file']
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']

        fields['idx'] = data.IndexField()
        if points_iou_file is not None:
            if input_type == 'pointcloud_crop':
                fields['points_iou'] = data.PatchPointsField(points_iou_file,
                                                             unpackbits=cfg['data']['points_unpackbits'],
                                                             multi_files=cfg['data']['multi_files'])
            else:
                fields['points_iou'] = data.PointsField(points_iou_file,
                                                        unpackbits=cfg['data']['points_unpackbits'],
                                                        multi_files=cfg['data']['multi_files'],
                                                        occ_from_sdf=cfg['data']['occ_from_sdf'])
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)
        if pointcloud_file is not None:
            fields['pointcloud'] = data.PointCloudField(pointcloud_file,
                                                        multi_files=cfg['data']['multi_files'])
        if pointcloud_chamfer_file != pointcloud_file:
            fields['pointcloud_chamfer'] = data.PointCloudField(pointcloud_chamfer_file,
                                                                multi_files=cfg['data']['multi_files'])

    return fields

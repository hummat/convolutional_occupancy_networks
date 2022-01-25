import yaml
from torchvision import transforms

from src import conv_onet
from src import data

method_dict = {
    'conv_onet': conv_onet
}


# General config
def load_config(path, default_path=None):
    """ Loads config file.

    Args:
        path (str): mesh_path to config file
        default_path (str): use default mesh_path
    """
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """ Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, dataset=None, device=None):
    """ Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    """
    method = cfg['method']
    model = method_dict[method].config.get_model(cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    """ Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    """
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    """ Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    """
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False):
    """ Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    """
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]
    print(f"Loading {split} data")

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        visualize = cfg['data']['visualize']
        transform = list()
        if cfg['data']['input_type'] in ['depth', 'depth_like', 'blenderproc']:
            if cfg['training']['in_cam_coords']:
                transform.append(data.Rotate(to_cam_frame=True,
                                             visualize=visualize))
            elif cfg['training']['in_world_coords']:
                transform.append(data.Rotate(to_world_frame=True,
                                             visualize=visualize))
        if cfg['data']['rotate']:
            transform.append(data.Rotate(visualize=visualize))
        if cfg['data']['scale']:
            scale = cfg['data']['scale']
            if not isinstance(scale, (tuple, list)):
                scale = (0.05, 0.5)
            transform.append(data.Scale(scale))
        if cfg['data']['normalize']:
            norm = cfg['data']['normalize']
            if isinstance(norm, str):
                norm = norm.lower()
            else:
                norm = 'center_scale'
            transform.append(data.Normalize(center='xyz' if 'center' in norm else "", scale='scale' in norm))

        dataset = data.Shapes3dDataset(
            dataset_folder,
            fields,
            split=split,
            categories=categories,
            transform=transforms.Compose(transform) if transform else None,
            cfg=cfg)
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


def get_inputs_field(cfg):
    """ Returns the inputs fields.

    Args:
        cfg (dict): config dictionary
    """
    input_type = cfg['data']['input_type']

    transform = [data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
                 data.PointcloudNoise(cfg['data']['pointcloud_noise'])]
    transform = transforms.Compose(transform)

    if input_type is None:
        inputs_field = None
    elif input_type == 'pointcloud':
        inputs_field = data.PointCloudField(cfg['data']['pointcloud_file'],
                                            transform,
                                            cfg['data']['multi_files'])
    elif input_type == 'partial_pointcloud':
        inputs_field = data.PartialPointCloudField(cfg['data']['pointcloud_file'],
                                                   transform,
                                                   cfg['data']['multi_files'],
                                                   part_ratio=cfg['data']['part_ratio'])
    elif input_type == 'depth_like':
        inputs_field = data.DepthLikePointCloudField(cfg['data']['mesh_file'] if cfg['data']['mesh_file'] else cfg['data']['pointcloud_file'],
                                                     num_points=cfg['data']['pointcloud_n'],
                                                     rotate_object='yx',
                                                     upper_hemisphere=cfg['data']['sample_upper_hemisphere'],
                                                     transform=transform)
    elif input_type == 'depth':
        inputs_field = data.PyrenderDepthPointCloudField(cfg['data']['mesh_file'],
                                                         num_points=cfg['data']['pointcloud_n'],
                                                         upper_hemisphere=cfg['data']['sample_upper_hemisphere'],
                                                         transform=transform)
    elif input_type == 'blenderproc':
        inputs_field = data.BlenderProcDepthPointCloudField(transform=transform,
                                                            path_prefix=cfg['data']['path_prefix'])
    elif input_type == 'pointcloud_crop':
        inputs_field = data.PatchPointCloudField(cfg['data']['pointcloud_file'], transform, None, cfg['data']['multi_files'])
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(cfg['data']['voxels_file'])
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field

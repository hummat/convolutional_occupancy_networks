import argparse
import os

import pandas as pd
import torch
import trimesh
from joblib import Parallel, delayed
from tabulate import tabulate
from tqdm import tqdm

from src import config, data
from src.eval import MeshEvaluator
from src.utils.io import load_pointcloud


def single_eval(data, args, generation_dir, dataset, cfg):
    evaluator = MeshEvaluator()
    # Output folders
    if not args.eval_input:
        mesh_dir = os.path.join(generation_dir, 'meshes')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    else:
        mesh_dir = os.path.join(generation_dir, 'input')
        pointcloud_dir = os.path.join(generation_dir, 'input')

    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}

    modelname = model_dict['model']
    category_id = model_dict['category']

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
        # for room dataset
        if category_name == 'n/a':
            category_name = category_id
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, category_id)
        pointcloud_dir = os.path.join(pointcloud_dir, category_id)

    # Evaluate
    pointcloud_tgt = data.get('pointcloud_chamfer')
    if pointcloud_tgt is None:
        pointcloud_tgt = data.get('pointcloud')
    pointcloud_tgt = pointcloud_tgt.squeeze(0).numpy()

    normals_tgt = data.get('pointcloud_chamfer.normals')
    if normals_tgt is None:
        normals_tgt = data.get('pointcloud.normals')
    normals_tgt = normals_tgt.squeeze(0).numpy()
    normals_tgt = normals_tgt if normals_tgt.sum() != 0 else None

    points_tgt = data['points_iou'].squeeze(0).numpy()
    occ_tgt = data['points_iou.occ'].squeeze(0).numpy()

    # Evaluating mesh and pointcloud
    # Start row and put basic information inside
    eval_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    # eval_dicts.append(eval_dict)

    # Evaluate mesh
    if cfg['test']['eval_mesh']:
        mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)

        if os.path.exists(mesh_file):
            try:
                mesh = trimesh.load(mesh_file, process=False)
                eval_dict_mesh = evaluator.eval_mesh(mesh,
                                                     pointcloud_tgt,
                                                     normals_tgt,
                                                     points_tgt,
                                                     occ_tgt,
                                                     remove_wall=cfg['test']['remove_wall'])
                for k, v in eval_dict_mesh.items():
                    eval_dict[k + ' (mesh)'] = v
            except Exception as e:
                print("Error: Could not evaluate mesh: %s" % mesh_file)
        else:
            print('Warning: mesh does not exist: %s' % mesh_file)

    # Evaluate point cloud
    if cfg['test']['eval_pointcloud']:
        pointcloud_file = os.path.join(
            pointcloud_dir, '%s.ply' % modelname)

        if os.path.exists(pointcloud_file):
            pointcloud = load_pointcloud(pointcloud_file)
            eval_dict_pcl = evaluator.eval_pointcloud(
                pointcloud, pointcloud_tgt)
            for k, v in eval_dict_pcl.items():
                eval_dict[k + ' (pcl)'] = v
        else:
            print('Warning: pointcloud does not exist: %s' % pointcloud_file)
    return eval_dict


def main():
    data.seed_all_rng(11)

    parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--eval_input', action='store_true', help='Evaluate inputs instead.')

    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    # Shorthands
    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    if not args.eval_input:
        out_file = os.path.join(generation_dir, 'eval_meshes_full.pkl')
        out_file_class = os.path.join(generation_dir, 'eval_meshes.csv')
    else:
        out_file = os.path.join(generation_dir, 'eval_input_full.pkl')
        out_file_class = os.path.join(generation_dir, 'eval_input.csv')

    # Dataset
    dataset = config.get_dataset('test', cfg, return_idx=True)
    # points_field = data.PointsField(
    #     cfg['data']['points_iou_file'],
    #     unpackbits=cfg['data']['points_unpackbits'],
    #     multi_files=cfg['data']['multi_files']
    # )
    # pointcloud_field = data.PointCloudField(
    #     cfg['data']['pointcloud_chamfer_file'],
    #     multi_files=cfg['data']['multi_files']
    # )
    # fields = {
    #     'points_iou': points_field,
    #     'pointcloud_chamfer': pointcloud_field,
    #     'idx': data.IndexField(),
    # }
    #
    # print('Test split: ', cfg['data']['test_split'])
    #
    # dataset_folder = cfg['data']['path']
    # dataset = data.Shapes3dDataset(
    #     dataset_folder,
    #     fields,
    #     cfg['data']['test_split'],
    #     categories=cfg['data']['classes'],
    #     cfg=cfg)

    # Evaluator
    # evaluator = MeshEvaluator(n_points=100000)

    # Loader
    n_jobs = cfg['test']['n_workers']
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              num_workers=n_jobs,
                                              shuffle=False,
                                              collate_fn=data.collate_remove_none,
                                              worker_init_fn=data.worker_init_reset_seed)

    # Evaluate all classes
    eval_dicts = []
    eval_dicts.extend(Parallel(n_jobs=n_jobs)(delayed(single_eval)(d,
                                                                   args,
                                                                   generation_dir,
                                                                   dataset,
                                                                   cfg) for d in tqdm(test_loader,
                                                                                      desc="Evaluating meshes")))

    # Create pandas dataframe and save
    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(['idx'], inplace=True)
    eval_df.to_pickle(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.groupby(by=['class name']).mean()

    # Print results
    eval_df_class.loc['mean'] = eval_df_class.mean()
    eval_df_class.to_csv(out_file_class)
    print(eval_df_class)

    # Tabulate
    table = tabulate(eval_df_class, headers='keys', showindex=False)
    with open(out_file_class.replace(".csv", ".txt"), 'w') as f:
        f.write(table)


if __name__ == "__main__":
    main()

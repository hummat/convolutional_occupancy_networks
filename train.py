import argparse
import datetime
import os
import time

import matplotlib;
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from tabulate import tabulate

matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from src.eval import MeshEvaluator
from collections import defaultdict
import shutil

# Arguments
parser = argparse.ArgumentParser(description='Train a 3D reconstruction model.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds with exit code 3.')
parser.add_argument('--weights', type=str, help="Path to weights.")

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
weights = args.weights if args.weights else cfg['training']['pre_trained_weights']
vis_n_outputs = cfg['generation']['vis_n_outputs']
validate_first = cfg['training']['validate_first']
max_iter = cfg['training']['max_iter']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg, return_idx=True)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=cfg['training']['n_workers'],
                                           shuffle=True,
                                           collate_fn=data.collate_remove_none,
                                           worker_init_fn=data.worker_init_reset_seed)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=1,
                                         num_workers=cfg['training']['n_workers_val'],
                                         shuffle=False,
                                         collate_fn=data.collate_remove_none,
                                         worker_init_fn=data.worker_init_reset_seed)

# For visualizations
vis_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         collate_fn=data.collate_remove_none,
                                         worker_init_fn=data.worker_init_reset_seed)
model_counter = defaultdict(int)
data_vis_list = []

# Build a data dictionary for visualization
iterator = iter(vis_loader)
for i in range(len(vis_loader)):
    data_vis = next(iterator)
    idx = data_vis['idx'].item()
    model_dict = val_dataset.get_model_dict(idx)
    category_id = model_dict.get('category', 'n/a')
    category_name = val_dataset.metadata[category_id].get('name', 'n/a')
    category_name = category_name.split(',')[0]
    if category_name == 'n/a':
        category_name = category_id

    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        data_vis_list.append({'category': category_name, 'it': c_it, 'data': data_vis})

    model_counter[category_id] += 1

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Generator
generator = config.get_generator(model, cfg, device=device)

# Evaluator
evaluator = MeshEvaluator(n_points=100000)

# Initialize training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
if weights is not None and not load_dict:
    weights = os.path.abspath(weights)
    print("Loading pre-trained weights from ", weights)
    checkpoint_io.load(weights)
epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get('loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f' % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)
print('output mesh_path: ', cfg['training']['out_dir'])

while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1
        loss = trainer.train_step(batch)
        logger.add_scalar('train/loss', loss, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                  % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            for data_vis in tqdm(data_vis_list, desc="Visualizing"):
                if cfg['generation']['sliding_window']:
                    out = generator.generate_mesh_sliding(data_vis['data'])
                else:
                    out = generator.generate_mesh(data_vis['data'])
                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}

                mesh.export(
                    os.path.join(out_dir, 'vis', '{}_{}_{}.off'.format(it, data_vis['category'], data_vis['it'])))

        # Run validation
        if validate_every > 0 and (it % validate_every) == 0 or (epoch_it == 1 and it == 1) or validate_first:
            validate_first = False
            eval_dict = trainer.evaluate(val_loader)

            if "f-score" in model_selection_metric:
                result_list = list()
                values_list = list()
                for data in tqdm(val_loader, desc="Computing F1-Score"):
                    mesh = generator.generate_mesh(data, return_stats=False)
                    result = evaluator.eval_mesh(mesh=mesh,
                                                 pointcloud_tgt=data['pointcloud_chamfer'].squeeze(0).numpy(),
                                                 normals_tgt=data['pointcloud_chamfer.normals'].squeeze(0).numpy(),
                                                 points_iou=data['points_iou'].squeeze(0).numpy(),
                                                 occ_tgt=data['points_iou.occ'].squeeze(0).numpy())
                    result_list.append(result[model_selection_metric])
                    values_list.append([values for values in result.values()])
                    header = result.keys()
                eval_dict[model_selection_metric] = np.mean(result_list)
                table = np.mean(values_list, axis=0)
                print(tabulate([table], header))

            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f' % (model_selection_metric, metric_val))
            print("All metrics:", eval_dict)

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model: %s %.4f' % (model_selection_metric, metric_val_best))
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

        # Save checkpoint
        if checkpoint_every > 0 and (it % checkpoint_every) == 0:
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

        # Backup if necessary
        if backup_every > 0 and (it % backup_every) == 0:
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

        # Exit if necessary
        if 0 < exit_after <= (time.time() - t0):
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
            exit(0)
        if max_iter is not None and it >= max_iter:
            print("Max iterations reached. Exiting.")
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
            exit(0)

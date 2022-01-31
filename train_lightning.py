import os
import argparse
from collections import defaultdict
from multiprocessing import cpu_count
import glob

import numpy as np
import torch
import torch.nn.functional as F
from trimesh import load_mesh
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb

from src import config, data
from src.common import compute_iou


class LitConvOnet(pl.LightningModule):
    def __init__(self,
                 cfg,
                 train_dataset,
                 val_dataset,
                 visualize: bool = True,
                 vis_every_n_evals: int = 1):
        super().__init__()
        self.cfg = cfg
        self.model = config.get_model(cfg, train_dataset)
        self.val_dataset = val_dataset
        self.threshold = float(cfg["test"]["threshold"])
        self.n_evals = 0
        self.vis_every_n_evals = vis_every_n_evals
        self.visualize = visualize
        self.save_hyperparameters(cfg)

        if visualize:
            out_dir = os.path.abspath(cfg["training"]["out_dir"])
            self.vis_path = os.path.join(out_dir, "vis")
            os.makedirs(self.vis_path, exist_ok=True)
            self.generator = config.get_generator(self.model, self.cfg, device=None)
            self.vis_data = dict()
            self.model_counter = defaultdict(int)
            self.vis_this_eval = False

    def generate_mesh(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)
        if inputs.shape[0] != 1:
            inputs = torch.unsqueeze(inputs, dim=0)
        return self.generator.generate_mesh({"inputs": inputs}, return_stats=False)

    def visualize_data(self, batch, batch_idx):
        if self.n_evals == 0:
            for idx, inputs in zip(batch.get("idx"), batch.get("inputs")):
                model_dict = self.val_dataset.get_model_dict(idx)
                category_id = model_dict.get("category")

                instance = self.model_counter[category_id]
                if instance < self.cfg["generation"]["vis_n_outputs"]:
                    category_name = self.val_dataset.metadata[category_id].get("name")
                    category_name = category_name.split(',')[0]

                    if batch_idx in self.vis_data:
                        self.vis_data[batch_idx].append({"inputs": inputs,
                                                         "category_name": category_name,
                                                         "instance": instance})
                    else:
                        self.vis_data[batch_idx] = [{"inputs": inputs,
                                                     "category_name": category_name,
                                                     "instance": instance}]

                    mesh = self.generate_mesh(inputs)
                    if len(mesh.faces) > 0 and len(mesh.vertices) > 0:
                        mesh.export(os.path.join(self.vis_path, f"{self.global_step}_{category_name}_{instance}.obj"))
                self.model_counter[category_id] += 1
        elif batch_idx in self.vis_data:
            for batch_data in self.vis_data[batch_idx]:
                inputs = batch_data["inputs"]
                category_name = batch_data["category_name"]
                instance = batch_data["instance"]

                mesh = self.generate_mesh(inputs)
                if len(mesh.faces) > 0 and len(mesh.vertices) > 0:
                    mesh.export(os.path.join(self.vis_path, f"{self.global_step}_{category_name}_{instance}.obj"))

    def forward(self, x):
        return self.model(**x)

    def on_fit_start(self):
        if self.visualize:
            self.generator.device = self.device

    def on_train_start(self):
        self.n_evals = 0
        if self.visualize:
            self.vis_data = dict()
            self.model_counter = defaultdict(int)
            meshes = glob.glob(os.path.join(self.vis_path, f"{self.global_step}*.obj"))
            for mesh in meshes:
                os.remove(mesh)

    def training_step(self, batch, batch_idx):
        logits = self({"points": batch.get("points"), "inputs": batch.get("inputs")}).logits
        occ = batch.get("points.occ")
        loss = F.binary_cross_entropy_with_logits(logits, occ, reduction="none").sum(-1).mean()
        self.log("loss", loss, batch_size=self.cfg["training"]["batch_size"])
        return loss

    def on_validation_start(self):
        self.vis_this_eval = False
        if self.visualize and self.n_evals % self.vis_every_n_evals == 0:
            self.vis_this_eval = True

    def validation_step(self, batch, batch_idx):
        out = self({"points": batch.get("points_iou"), "inputs": batch.get("inputs")})
        occ_iou = batch.get("points_iou.occ")
        loss = F.binary_cross_entropy_with_logits(out.logits, occ_iou, reduction="none").sum(-1).mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=self.cfg["training"]["batch_size_val"])

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (out.probs >= self.threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        self.log("val_iou", iou, prog_bar=True, batch_size=self.cfg["training"]["batch_size_val"])

        if self.vis_this_eval:
            self.visualize_data(batch, batch_idx)
        return loss

    def on_validation_end(self):
        self.n_evals += 1

    def test_step(self, batch, batch_idx):
        out = self({"points": batch.get("points_iou"), "inputs": batch.get("inputs")})
        occ_iou = batch.get("points_iou.occ")
        loss = F.binary_cross_entropy_with_logits(out.logits, occ_iou, reduction="none").sum(-1).mean()
        self.log("test_loss", loss, prog_bar=True, batch_size=self.cfg["training"]["batch_size_test"])

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (out.probs >= self.threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        self.log("test_iou", iou, prog_bar=True, batch_size=self.cfg["training"]["batch_size_test"])
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class LitDataModule(pl.LightningDataModule):
    def __init__(self,
                 cfg,
                 num_workers: int = cpu_count(),
                 prefetch_factor: int = 2,
                 pin_memory: bool = False,
                 seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.train = None
        self.val = None
        self.test = None
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.collate_fn = default_collate if self.cfg["data"]["pointcloud_n"] else data.heterogeneous_batching

    def setup(self, stage=None):
        self.train = config.get_dataset("train", self.cfg)
        self.val = config.get_dataset("val", self.cfg, return_idx=True)
        self.test = config.get_dataset("test", self.cfg, return_idx=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train,
                                           batch_size=self.cfg["training"]["batch_size"],
                                           num_workers=self.num_workers,
                                           collate_fn=self.collate_fn,
                                           shuffle=True,
                                           pin_memory=self.pin_memory,
                                           worker_init_fn=data.worker_init_reset_seed,
                                           generator=self.generator,
                                           prefetch_factor=self.prefetch_factor,
                                           persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val,
                                           batch_size=self.cfg["training"]["batch_size_val"],
                                           num_workers=self.num_workers,
                                           collate_fn=self.collate_fn,
                                           prefetch_factor=self.prefetch_factor,
                                           pin_memory=self.pin_memory,
                                           worker_init_fn=data.worker_init_reset_seed,
                                           generator=self.generator,
                                           persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test,
                                           batch_size=self.cfg["training"]["batch_size_test"],
                                           num_workers=self.num_workers,
                                           prefetch_factor=self.prefetch_factor,
                                           pin_memory=self.pin_memory,
                                           worker_init_fn=data.worker_init_reset_seed,
                                           generator=self.generator)


class WandbVisualizeCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.vis_this_eval:
            vis_data = pl_module.vis_data
            vis_path = pl_module.vis_path
            paths = list()
            names = list()
            instances = list()
            for batch_idx in vis_data.keys():
                for batch_data in vis_data[batch_idx]:
                    category_name = batch_data["category_name"]
                    instance = batch_data["instance"]
                    mesh_path = os.path.join(vis_path, f"{trainer.global_step}_{category_name}_{instance}.obj")
                    if os.path.isfile(mesh_path):
                        paths.append(mesh_path)
                        names.append(category_name)
                        instances.append(instance)

            if paths:
                trainer.logger.experiment.log({
                    "vis/completion": [wandb.Object3D(p,
                                                      caption=f"{n}_{i}") for p, n, i in zip(paths, names, instances)],
                    "global_step": trainer.global_step
                })


def main():
    parser = argparse.ArgumentParser(description="Train a 3D reconstruction model.")
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--weights", type=str, help="Path to pre-trained weights.")
    parser.add_argument("--checkpoint", type=str, help="Path to PyTorch Lightning checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume training instead of starting from scratch.")
    parser.add_argument("--early_stopping", type=str, default="none", choices=["none", "val_iou", "val_loss"],
                        help="Terminate if validation loss or iou stops improving.")
    parser.add_argument("--auto_lr", action="store_true", help="Tune learning rate automatically.")
    parser.add_argument("--auto_batch_size", action="store_true", help="Tune batch size automatically.")
    parser.add_argument("--test", action="store_true", help="Runs evaluation on the test set after training.")
    parser.add_argument("--wandb", action="store_true", help="Use the weights & biases logger.")
    parser.add_argument("--name", type=str, help="The name of the train run.")
    parser.add_argument("--project", type=str, help="The name of project.")
    parser.add_argument("--id", type=str, help="The weights & biases run id.")
    parser.add_argument("--offline", action="store_true", help="Log offline.")
    parser.add_argument("--precision", type=int, default=32, help="Train with float16, 32 or 64 precision.")
    parser.add_argument("--profile", action="store_true", help="Profile code to find bottlenecks.")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bar.")
    parser.add_argument("--pin_memory", action="store_true", help="Enable GPU memory pinning in data loaders.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--visualize", action="store_true", help="Generate some validation meshes.")
    parser.add_argument("--seed", type=int, default=0, help="Set random seed.")
    parser.add_argument("--num_workers", type=int, default=cpu_count(),
                        help="Number of workers to spawn for data loading.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch per worker.")
    args = parser.parse_args()

    data.seed_all_rng(args.seed)

    cfg = config.load_config(args.config, "configs/default.yaml")
    batch_size = cfg["training"]["batch_size"]
    model_selection_metric = cfg["training"]["model_selection_metric"]
    model_selection_mode = cfg["training"]["model_selection_mode"]
    out_dir = os.path.abspath(cfg["training"]["out_dir"])
    max_iter = cfg["training"]["max_iter"]
    validate_every = cfg["training"]["validate_every"]
    visualize_every = cfg["training"]["visualize_every"]
    print_every = cfg["training"]["print_every"]

    os.makedirs(out_dir, exist_ok=True)
    split_out_dir = out_dir.split('/')
    save_dir = '/'.join(split_out_dir[:-1])
    name = args.name if args.name else split_out_dir[-1]
    project = args.project if args.project else split_out_dir[-2]

    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = cpu_count()
    dataset = LitDataModule(cfg, num_workers, args.prefetch_factor, args.pin_memory, args.seed)
    dataset.setup()

    n_train_batches = int(np.ceil(len(dataset.train) / batch_size))
    eval_every_n_epochs = np.round(validate_every / n_train_batches, 1)
    vis_every_n_evals = int(np.ceil(visualize_every / validate_every))
    max_steps = max_iter
    max_epochs = max_steps // n_train_batches
    print(f"Number of train batches:", n_train_batches)
    print(f"Evaluating every {validate_every} steps")
    print(f"Evaluating every {eval_every_n_epochs} epochs")
    if args.visualize:
        print(f"Visualizing every {vis_every_n_evals} evaluations")
    print(f"Training for max. {max_steps} steps")
    print(f"Training for max. {max_epochs} epochs")
    print(f"Logging every {print_every} steps")

    conv_onet = LitConvOnet(cfg, dataset.train, dataset.val, args.visualize, vis_every_n_evals)
    if not args.resume:
        if args.weights:
            state_dict = torch.load(os.path.abspath(args.weights))["model"]
            conv_onet.model.load_state_dict(state_dict)
        if args.checkpoint:
            conv_onet = LitConvOnet.load_from_checkpoint(os.path.abspath(args.checkpoint),
                                                         cfg=cfg,
                                                         train_dataset=dataset.train,
                                                         val_dataset=dataset.val,
                                                         vis_every_n_evals=vis_every_n_evals)
    callbacks = [ModelCheckpoint(filename="{epoch}-{step}-{val_loss:.2f}-{val_iou:.2f}",
                                 monitor=f"val_{model_selection_metric}",
                                 verbose=args.verbose,
                                 save_last=True,
                                 mode="max" if "max" in model_selection_mode else "min"),
                 RichModelSummary()]
    if not args.no_progress:
        callbacks.append(RichProgressBar())
    patience = max_epochs // int(np.ceil(eval_every_n_epochs)) // 10
    if args.early_stopping is not "none" and patience >= 3:
        print(f"Will terminate after {patience} evaluations without improvement")
        metric = args.early_stopping
        callbacks.append(EarlyStopping(monitor=metric,
                                       mode="max" if metric == "val_iou" else "min",
                                       patience=patience,
                                       verbose=args.verbose))

    if args.wandb:
        logger = WandbLogger(name=name,
                             save_dir='/'.join(split_out_dir[:-2]),
                             offline=args.offline,
                             id=args.id,
                             project=project)
        logger.watch(conv_onet)
        if args.visualize:
            callbacks.append(WandbVisualizeCallback())
    else:
        logger = TensorBoardLogger(save_dir=save_dir, name=name)
    logger.log_hyperparams(cfg)

    trainer = pl.Trainer(logger=logger,
                         default_root_dir=out_dir,
                         callbacks=callbacks,
                         gpus=1,
                         enable_progress_bar=not args.no_progress,
                         accumulate_grad_batches=32 // batch_size if batch_size < 32 else None,
                         max_epochs=max_epochs,
                         max_steps=max_steps,
                         check_val_every_n_epoch=int(np.ceil(eval_every_n_epochs)),
                         val_check_interval=validate_every if eval_every_n_epochs < 1.0 else 1.0,
                         log_every_n_steps=print_every,
                         precision=args.precision,
                         profiler="simple" if args.profile else None,
                         auto_lr_find=args.auto_lr,
                         auto_scale_batch_size=args.auto_batch_size)
    trainer.fit(conv_onet,
                dataset,
                ckpt_path=os.path.abspath(args.checkpoint) if args.checkpoint and args.resume else None)
    print(f"Best validation {model_selection_metric}: {trainer.checkpoint_callback.best_model_score:.2f}")

    state_dict = torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
    model_best = state_dict.copy()
    for key, value in state_dict.items():
        model_best[key.replace("model.", "")] = model_best.pop(key)
    torch.save({"model": model_best}, os.path.join(out_dir, "model_best.pt"))

    if args.test:
        trainer.test(datamodule=dataset)


if __name__ == "__main__":
    main()

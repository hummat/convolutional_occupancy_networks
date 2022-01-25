import argparse
import os.path

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src import config, data
from src.common import compute_iou


class LitConvOnet(pl.LightningModule):
    def __init__(self, cfg, train_dataset, val_dataset, visualize: bool = True, vis_every_n_evals: int = 1):
        super().__init__()
        self.cfg = cfg
        self.model = config.get_model(cfg, train_dataset)
        self.val_dataset = val_dataset
        self.threshold = float(cfg["test"]["threshold"])
        self.n_evals = 0
        self.vis_every_n_evals = vis_every_n_evals
        self.visualize = visualize
        self.save_hyperparameters(cfg)

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        logits = self({"points": batch.get("points"), "inputs": batch.get("inputs")}).logits
        occ = batch.get("points.occ")
        loss = F.binary_cross_entropy_with_logits(logits, occ, reduction="none").sum(-1).mean()
        self.log("loss", loss, batch_size=self.cfg["training"]["batch_size"])
        return loss

    def validation_step(self, batch, batch_idx):
        self.n_evals += 1
        out = self({"points": batch.get("points_iou"), "inputs": batch.get("inputs")})
        occ_iou = batch.get("points_iou.occ")
        loss = F.binary_cross_entropy_with_logits(out.logits, occ_iou, reduction="none").sum(-1).mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=1)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (out.probs >= self.threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np)
        self.log("val_iou", iou, prog_bar=True, batch_size=1)

        if self.visualize and self.n_evals % self.vis_every_n_evals == 0:
            out_dir = os.path.abspath(self.cfg["training"]["out_dir"])
            vis_path = os.path.join(out_dir, "vis")
            os.makedirs(vis_path, exist_ok=True)
            iteration = self.n_evals * self.cfg["training"]["validate_every"]
            generator = config.get_generator(self.model, self.cfg, out.logits.device)

            if batch_idx < self.cfg["generation"]["vis_n_outputs"]:
                mesh = generator.generate_mesh(batch, return_stats=False)

                idx = batch["idx"].item()
                model_dict = self.val_dataset.get_model_dict(idx)
                category_id = model_dict.get("category")
                category_name = self.val_dataset.metadata[category_id].get("name")
                category_name = category_name.split(',')[0]

                mesh.export(os.path.join(vis_path, f"{iteration}_{category_name}_{idx}.off"))
        return loss

    def test_step(self, batch, batch_idx):
        out = self({"points": batch.get("points_iou"), "inputs": batch.get("inputs")})
        occ_iou = batch.get("points_iou.occ")
        loss = F.binary_cross_entropy_with_logits(out.logits, occ_iou, reduction="none").sum(-1).mean()
        self.log("test_loss", loss, prog_bar=True, batch_size=1)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (out.probs >= self.threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np)
        self.log("test_iou", iou, prog_bar=True, batch_size=1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage=None):
        self.train = config.get_dataset("train", self.cfg)
        self.val = config.get_dataset("val", self.cfg, return_idx=True)
        self.test = config.get_dataset("test", self.cfg, return_idx=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train,
                                           batch_size=self.cfg["training"]["batch_size"],
                                           num_workers=self.cfg["training"]["n_workers"],
                                           shuffle=True,
                                           collate_fn=data.collate_remove_none,
                                           worker_init_fn=data.worker_init_reset_seed)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val,
                                           batch_size=1,
                                           num_workers=self.cfg["training"]["n_workers_val"],
                                           shuffle=False,
                                           collate_fn=data.collate_remove_none,
                                           worker_init_fn=data.worker_init_reset_seed)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test,
                                           batch_size=1,
                                           num_workers=self.cfg["test"]["n_workers"],
                                           shuffle=False,
                                           collate_fn=data.collate_remove_none,
                                           worker_init_fn=data.worker_init_reset_seed)


def main():
    data.seed_all_rng(11)

    parser = argparse.ArgumentParser(description="Train a 3D reconstruction model.")
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--weights", type=str, help="Path to pre-trained weights.")
    parser.add_argument("--checkpoint", type=str, help="Path to PyTorch Lightning checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume training instead of starting from scratch.")
    parser.add_argument("--early_stopping", action="store_true", help="Terminate if validation loss stops improving.")
    parser.add_argument("--auto_lr", action="store_true", help="Tune learning rate automatically.")
    parser.add_argument("--auto_batch_size", action="store_true", help="Tune batch size automatically.")
    parser.add_argument("--test", action="store_true", help="Runs evaluation on the test set after training.")
    parser.add_argument("--wandb", action="store_true", help="Use the weights & biases logger.")
    parser.add_argument("--id", type=str, help="The weights & biases run id.")
    parser.add_argument("--offline", action="store_true", help="Log offline.")
    parser.add_argument("--profile", action="store_true", help="Profile code to find bottlenecks.")
    args = parser.parse_args()

    cfg = config.load_config(args.config, "configs/default.yaml")
    batch_size = cfg["training"]["batch_size"]
    model_selection_metric = cfg["training"]["model_selection_metric"]
    model_selection_mode = cfg["training"]["model_selection_mode"]
    out_dir = os.path.abspath(cfg["training"]["out_dir"])
    max_iter = cfg["training"]["max_iter"]
    validate_every = cfg["training"]["validate_every"]
    visualize_every = cfg["training"]["visualize_every"]
    print_every = cfg["training"]["print_every"]

    save_dir = '/'.join(out_dir.split('/')[:-1])
    name = out_dir.split('/')[-1]

    dataset = LitDataModule(cfg)
    dataset.setup()

    n_train_batches = int(np.ceil(len(dataset.train) / batch_size))
    eval_every_n_epochs = validate_every // n_train_batches
    vis_every_n_evals = int(np.ceil(visualize_every / validate_every))
    max_steps = max_iter
    max_epochs = max_steps // n_train_batches
    print(f"Number of train batches:", n_train_batches)
    print(f"Evaluating every {validate_every} steps")
    print(f"Evaluating every {eval_every_n_epochs} epochs")
    print(f"Visualizing every {vis_every_n_evals} evaluations")
    print(f"Training for max. {max_steps} steps")
    print(f"Training for max. {max_epochs} epochs")
    print(f"Logging every {print_every} steps")

    conv_onet = LitConvOnet(cfg, dataset.train, dataset.val, vis_every_n_evals=vis_every_n_evals)
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
                                 mode="max" if "max" in model_selection_mode else "min"),
                 RichProgressBar(),
                 RichModelSummary()]
    if args.early_stopping:
        patience = max_epochs // eval_every_n_epochs // 10
        print(f"Will terminate after {patience} evaluations without improvement")
        print("(1/10th of all planned evaluations)")
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=patience))

    if args.wandb:
        logger = WandbLogger(name=name,
                             save_dir=save_dir,
                             offline=args.offline,
                             id=args.id,
                             project="Convolutional Occupancy Networks")
        logger.watch(conv_onet)
    else:
        logger = TensorBoardLogger(save_dir=save_dir, name=name)
    logger.log_hyperparams(cfg)

    trainer = pl.Trainer(logger=logger,
                         default_root_dir=out_dir,
                         callbacks=callbacks,
                         gpus=1,
                         enable_progress_bar=False,
                         max_epochs=max_epochs,
                         max_steps=max_steps,
                         check_val_every_n_epoch=eval_every_n_epochs,
                         log_every_n_steps=print_every,
                         profiler="simple" if args.profile else None,
                         auto_lr_find=args.auto_lr,
                         auto_scale_batch_size=args.auto_batch_size)
    trainer.fit(conv_onet, dataset, ckpt_path=os.path.abspath(args.checkpoint) if args.resume else None)
    print(f"Best validation {model_selection_metric}: {trainer.checkpoint_callback.best_model_score:.2f}.")

    state_dict = torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
    best_model = state_dict.copy()
    for key, value in state_dict.items():
        best_model[key.replace("model.", "")] = best_model.pop(key)
    torch.save(best_model, os.path.join(out_dir, "best_model.pt"))

    if args.test:
        trainer.test(datamodule=dataset)


if __name__ == "__main__":
    main()

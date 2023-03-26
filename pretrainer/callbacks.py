from pytorch_lightning.callbacks import Callback
import torch
import os
import boto3
import numpy as np
import pytorch_lightning as pl


class UploadCheckpointToS3(Callback):
    """Custom PyTorch callback for uploading model checkpoints to a s3_resource bucket using a boto3
    resource object."""
    def __init__(
        self,
        local_path: str,
        desc: str,
        s3_resource: boto3.resource,
        bucket: str,
        upload_prefix: int = "model_checkpoints",
        n_epochs: int = 10,
        n_steps: int = None,
        quiet: bool = False,
    ) -> None:
        """
        Callback for uploading model checkpoints to s3_resource bucket using a boto3 resource object.
        :param local_path: Local path to folder where model checkpoints are saved
        :param desc: Description of checkpoint that is appended to checkpoint file name on save
        :param s3_resource: boto3 resource object for s3_resource
        :param bucket: Name of bucket to upload model checkpoints to
        :param upload_prefix: Path in bucket/ to upload model checkpoints to, defaults to model_checkpoints
        :param n_epochs: Number of epochs between checkpoints, defaults to 10
        :param n_steps: Number of steps between checkpoints, defaults to None
        :param quiet: Whether to print messages when uploading checkpoints, defaults to False
        """
        super().__init__()
        self.local_path = local_path
        self.desc = desc

        self.s3_resource = s3_resource
        self.bucket = bucket
        self.upload_prefix = upload_prefix

        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.quiet = quiet

        os.makedirs(self.local_path, exist_ok=True)

    def _save_and_upload_checkpoint(self, pl_module: pl.LightningModule, epoch: int, step: int) -> None:
        """
        Uploads a checkpoint to s3_resource bucket.
        :param pl_module: PyTorch Lightning module
        :param epoch: Current epoch
        :param step: Current step
        """
        checkpoint = f"checkpoint-{epoch}-step-{step}-desc-{self.desc}.ckpt"
        checkpoint_local_path = os.path.join(self.local_path, checkpoint)

        model = pl_module.model
        torch.save(model.state_dict(), checkpoint_local_path)
        self.silentprint(f"Uploading checkpoint at epoch {epoch} and step {step}")

        try:
            self.s3_resource.Bucket(self.bucket).upload_file(
                Filename=checkpoint_local_path,
                Key=os.path.join(self.upload_prefix, checkpoint_local_path.split("/")[-1]),
            )
        except Exception as e:
            self.silentprint(f"Error when uploading on epoch {epoch}")
            self.silentprint(e)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *args, **kwargs) -> None:
        epoch = pl_module.current_epoch
        step = trainer.global_step

        if self.n_steps is not None and step % self.n_steps == 0:
            self._save_and_upload_checkpoint(pl_module, epoch, step)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = pl_module.current_epoch
        step = trainer.global_step

        if epoch % self.n_epochs == 0:
            self._save_and_upload_checkpoint(pl_module, epoch, step)

    def silentprint(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

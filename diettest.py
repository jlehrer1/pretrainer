import torch
import torchvision
import pandas as pd 
import numpy as np
import torch.nn as nn
from tqdm.notebook import tqdm
import torchvision.transforms as tt
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pretrainer import UploadCheckpointToS3
import boto3


class DietDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def __getitem__(self, idx):
        return self.data[idx][0], idx 
    
    def __len__(self):
        return len(self.data)

class ModelWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.8)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    
    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.loss(logits, labels)
        
        self.log("train_loss", loss)
        
        return loss 

if __name__ == '__main__':
    train = torchvision.datasets.CIFAR100(root='./', train=True, transform=tt.ToTensor(), download=True)
    traindiet = DietDataset(train)
    trainloader = DataLoader(traindiet, batch_size=256, num_workers=32, shuffle=True)

    model = torchvision.models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(train))
    
    model = ModelWrapper(model)
    with open("credentials") as f:
        key, access = [line.rstrip() for line in f.readlines()]

    s3 = boto3.resource(
        "s3",
        endpoint_url="https://s3-west.nrp-nautilus.io/",
        aws_access_key_id=key,
        aws_secret_access_key=access,
    )

    trainer = pl.Trainer(max_epochs=5000, logger=pl.loggers.WandbLogger(project='Pretraining-DIET'), callbacks=[
        UploadCheckpointToS3(
            local_path='./', 
            desc='diet-pretraining', 
            s3_resource=s3, 
            bucket="braingeneersdev",
            upload_prefix="jlehrer/model_checkpoints",
            n_epochs=10,
            n_steps=None,
        )
    ])
    trainer.fit(model, trainloader)


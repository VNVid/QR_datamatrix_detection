import pytorch_lightning as pl
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import numpy as np

CLASSES = [
    '__background__', 'qrcode', 'datamatrix'
]


class FasterRCNN(pl.LightningModule):
    def __init__(self, classes):
        super().__init__()
        self.save_hyperparameters()
        self.classes = classes

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights='DEFAULT')
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            self.in_features, len(classes))
        self.model.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        images, targets = batch

        self.model.train()
        loss_dict = self.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses  # .item()

        # Logging to TensorBoard
        self.log("train_loss", loss_value)

        return loss_value

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        loss_dict = self.model(images, targets)

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses  # .item()

        # Logging to TensorBoard
        self.log("val_loss", loss_value)

        return loss_value

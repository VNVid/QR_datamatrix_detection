import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


import glob as glob
import sys

from xml.etree import ElementTree as et

import dataset
import model

pl.seed_everything(123)

# arguments: save_dir[optional], model_checkpoint_path [optional]
save_dir = ''
if len(sys.argv) >= 2:
    save_dir = sys.argv[1]

model_checkpoint_path = ''
if len(sys.argv) >= 3:
    model_checkpoint_path = sys.argv[2]

barcode_dataset = dataset.BarcodeDataModule(model.CLASSES, batch_size=4,
                                            data_dir='merged')
train_model = model.FasterRCNN(model.CLASSES)

checkpoint_callback = ModelCheckpoint(
    save_last=True, monitor="val_loss", save_top_k=3)

logger = TensorBoardLogger(
    save_dir=save_dir, sub_dir='logs')
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=logger, callbacks=[checkpoint_callback],
                     accumulate_grad_batches=15)

if model_checkpoint_path == '':
    trainer.fit(model=train_model, datamodule=barcode_dataset)
else:
    trainer.fit(model=train_model, datamodule=barcode_dataset,
                ckpt_path=model_checkpoint_path)

import augmentations

import os
import cv2
import glob as glob
from xml.etree import ElementTree as et
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import ToTensorV2


class BarcodeDataset(Dataset):
    ''' 
          https://universe.roboflow.com/new-workspace-jmdju/barcode-w4agw/dataset/3
    '''

    def __init__(self, classes, data_dir, width: int = 1000, height: int = 600, transforms=None):
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.classes = classes
        self.transforms = transforms

        self.image_paths = glob.glob(f"{self.data_dir}/*.jpg")
        self.all_images = sorted([image_path.split(os.path.sep)[-1]
                                 for image_path in self.image_paths])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.data_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        # image_resized /= 255.0
        # normalizedImg = np.zeros(image_resized.shape)
        # image_resized = cv2.normalize(image_resized,  normalizedImg, 0, 1, cv2.NORM_MINMAX)

        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.data_dir, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            if xmax_final > self.width:
                xmax_final = self.width
            if ymax_final > self.height:
                ymax_final = self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        if self.transforms:
            # print(target['boxes'])
            # print(image_name)
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'])

        return image_resized, target


class BarcodeDataModule(pl.LightningDataModule):
    def __init__(self, classes, data_dir: str = "/content/drive/MyDrive/Detection/data/Barcode.v3i.voc.clean", batch_size: int = 128,
                 width: int = 1000, height: int = 600):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.classes = classes

    def setup(self, stage: str):
        self.barcodes_train = BarcodeDataset(
            self.classes, self.data_dir + '/train', self.width, self.height, transforms=augmentations.get_train_transform())
        self.barcodes_val = BarcodeDataset(
            self.classes, self.data_dir + '/valid', self.width, self.height, transforms=augmentations.get_valid_transform())
        self.barcodes_test = BarcodeDataset(
            self.classes, self.data_dir + '/test', self.width, self.height)

    def train_dataloader(self):
        return DataLoader(self.barcodes_train, batch_size=self.batch_size, shuffle=True, collate_fn=augmentations.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.barcodes_val, batch_size=self.batch_size, collate_fn=augmentations.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.barcodes_test, batch_size=self.batch_size, collate_fn=augmentations.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.barcodes_test, batch_size=self.batch_size, collate_fn=augmentations.collate_fn)

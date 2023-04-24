import cv2
import os
import sys
import glob as glob
import torch
import numpy as np

import my_model

if len(sys.argv) < 2:
    print("You should define data path and desired threshold")
    exit(0)

COLORS = [(0, 0, 0), (0, 255, 0), (255, 0, 0)]

model_checkpoint_path = 'epoch=93-step=5518.ckpt'
model = my_model.FasterRCNN.load_from_checkpoint(
    model_checkpoint_path).model
model.eval()

data_dir = sys.argv[1]
image_paths = glob.glob(f"{data_dir}/*.jpg")
all_images = sorted([image_path.split(os.path.sep)[-1]
                    for image_path in image_paths])

threshold = 0.6
if len(sys.argv) > 2:
    threshold = float(sys.argv[2])

if not os.path.isdir("result"):
    os.mkdir("result")


def draw_boxes(image, pred, thr):
    img_with_boxes = image.copy()
    for i in range(len(pred['labels'])):
        if pred['scores'][i] < thr:
            continue

        color = COLORS[int(pred['labels'][i])]
        box = pred['boxes'][i]
        cv2.rectangle(img_with_boxes, (int(box[0]), int(
            box[1])), (int(box[2]), int(box[3])), color)
        cv2.putText(img_with_boxes, my_model.CLASSES[int(pred['labels'][i])], (int(box[0]), int(
            box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA)
    return img_with_boxes


for index in range(len(all_images)):
    image_name = all_images[index]
    image_path = os.path.join(data_dir, image_name)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_resized = cv2.resize(image, (416, 416))
    image_resized /= 255.0

    pred = model(torch.from_numpy(
        np.transpose(image_resized[None], (0, 3, 1, 2))))

    img_with_boxes = draw_boxes(image_resized, pred[0], threshold)
    img_save = cv2.cvtColor(img_with_boxes * 255,
                            cv2.COLOR_BGR2RGB).astype(np.float32)
    cv2.imwrite(os.path.join(os.path.join(
        os.getcwd(), 'result'), image_name), img_save)

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

COLORS = [(0, 0, 0), (30, 200, 0), (200, 0, 50)]

version = sys.argv[1]

if version == '1':
    model_checkpoint_path = 'epoch=93-step=5518.ckpt'
    model = my_model.FasterRCNN_v1.load_from_checkpoint(
        model_checkpoint_path).model
else:
    model_checkpoint_path = 'epoch=20-step=1218.ckpt'
    model = my_model.FasterRCNN_v2.load_from_checkpoint(
        model_checkpoint_path).model
model.eval()

data_dir = sys.argv[2]
image_paths = glob.glob(f"{data_dir}/*.jpg")
all_images = sorted([image_path.split(os.path.sep)[-1]
                    for image_path in image_paths])

threshold = 0.6 if version == '1' else 0.8
if len(sys.argv) > 2:
    threshold = float(sys.argv[3])

if not os.path.isdir("result"):
    os.mkdir("result")


def draw_boxes(image, pred, thr, width, height):
    new_width = image.shape[1]
    new_height = image.shape[0]

    img_with_boxes = image.copy()
    for i in range(len(pred['labels'])):
        if pred['scores'][i] < thr:
            continue

        color = COLORS[int(pred['labels'][i])]
        box = pred['boxes'][i]

        cv2.rectangle(img_with_boxes, (int(int(box[0]) / width * new_width), int(int(box[1]) / height * new_height)),
                      (int(int(box[2]) / width * new_width), int(int(box[3]) / height * new_height)), color, thickness=max(int(new_width / 416), 1))

        if int(pred['labels'][i]) == 1:
            text_coord = (int(int(box[0]) / width * new_width),
                          int(int(box[1] - 5) / height * new_height))
        else:
            text_coord = (int(int(box[0]) / width * new_width),
                          int(int(box[3] + 20) / height * new_height))

        cv2.putText(img_with_boxes, f'{my_model.CLASSES[int(pred["labels"][i])]}-{pred["scores"][i]:.2f}', text_coord,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * new_width / 416, color, int(2 * new_width / 416), lineType=cv2.LINE_AA)

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

    img_with_boxes = draw_boxes(image / 255, pred[0], threshold, 416, 416)
    img_save = cv2.cvtColor(img_with_boxes * 255,
                            cv2.COLOR_BGR2RGB).astype(np.float32)
    cv2.imwrite(os.path.join(os.path.join(
        os.getcwd(), 'result'), image_name), img_save)
